"""A recursive neural network over expression trees.

Instead of hand-made features phi(x), learn an embedding h(node) in R^k by the
same structural recursion as the differentiator of Section 2, one learned
rule per node type:

    h(x)        = a learned vector
    h(c)        = A_const [log(1 + |c|), sign c] + b
    h(u^n)      = tanh(W_pow [h(u), n, 1[n < 0], 1[n not an integer]] + b)
    h(u1 + ...) = tanh(W_add (h(u1) + ... + h(um)) + b)       sum: order-free, like +
    h(u1 * ...) = tanh(W_mul (h(u1) + ... + h(um)) + b)
    h(u / v)    = tanh(W_div [h(u), h(v)] + b)                 concatenation: order matters
    h(g(u))     = tanh(W_g h(u) + b_g)                          one matrix per function g

and score the root: delta_hat(x) = v . h(root). Training uses the same
Bradley-Terry loss as every other model; PyTorch's reverse-mode autodiff
(backpropagation) supplies the gradients through the recursion.

Efficiency: a forest of trees is "compiled" once into a single DAG. Identical
subtrees are shared (the nodes are hashable), and nodes are ordered by
height and then by type, so each (height, type) group is one contiguous block
computed by one batched matrix multiply. A forward pass is a few dozen
tensor operations whatever the number of trees.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from calcdiff.evaluate import pairs_touching
from calcdiff.expr import FUNCTIONS, Add, Const, Div, Fn, Mul, Node, Pow, Var

GROUPS = ("x", "const", "pow", "add", "mul", "div") + tuple(f"fn:{g}" for g in FUNCTIONS)


def _group(node: Node) -> str:
    match node:
        case Var():
            return "x"
        case Const():
            return "const"
        case Pow():
            return "pow"
        case Add():
            return "add"
        case Mul():
            return "mul"
        case Div():
            return "div"
        case Fn(name, _):
            return f"fn:{name}"
    raise TypeError(node)


@dataclass
class Forest:
    """A compiled set of trees: unique nodes in (height, group) order."""
    n_nodes: int
    levels: list            # per height: list of (group, start, stop, child_index, parent_pos, extra)
    roots: torch.Tensor     # global index of each tree's root


def compile_forest(trees: list[Node]) -> Forest:
    index: dict[Node, int] = {}
    height: dict[Node, int] = {}

    def visit(n: Node) -> int:
        if n in height:
            return height[n]
        h = 1 + max((visit(c) for c in n.children()), default=-1)
        height[n] = h
        return h

    for t in trees:
        visit(t)
    order = sorted(height, key=lambda n: (height[n], GROUPS.index(_group(n))))
    for k, n in enumerate(order):
        index[n] = k

    levels = []
    k = 0
    while k < len(order):
        h = height[order[k]]
        groups = []
        while k < len(order) and height[order[k]] == h:
            g = _group(order[k])
            start = k
            while k < len(order) and height[order[k]] == h and _group(order[k]) == g:
                k += 1
            members = order[start:k]
            child, parent, extra = [], [], []
            for pos, n in enumerate(members):
                for c in n.children():
                    child.append(index[c])
                    parent.append(pos)
                if isinstance(n, Const):
                    v = float(n.value)
                    extra.append([np.log1p(abs(v)), np.sign(v)])
                elif isinstance(n, Pow):
                    e = n.exp
                    extra.append([float(e), float(e < 0), float(e.denominator != 1)])
            groups.append((g, start, k, torch.tensor(child, dtype=torch.long),
                           torch.tensor(parent, dtype=torch.long),
                           torch.tensor(extra, dtype=torch.float32) if extra else None))
        levels.append(groups)
    roots = torch.tensor([index[t] for t in trees], dtype=torch.long)
    return Forest(len(order), levels, roots)


class TreeEncoder(nn.Module):
    def __init__(self, dim: int = 16):
        super().__init__()
        self.dim = dim
        self.x = nn.Parameter(0.5 * torch.randn(dim))
        self.const = nn.Linear(2, dim)
        self.pow = nn.Linear(dim + 3, dim)
        self.add = nn.Linear(dim, dim)
        self.mul = nn.Linear(dim, dim)
        self.div = nn.Linear(2 * dim, dim)
        self.fn = nn.ModuleDict({g: nn.Linear(dim, dim) for g in FUNCTIONS})

    def forward(self, forest: Forest) -> torch.Tensor:
        """Embeddings of every root, shape (n_trees, dim)."""
        blocks: list[torch.Tensor] = []
        for groups in forest.levels:
            H = torch.cat(blocks) if blocks else None          # everything of smaller height
            for g, start, stop, child, parent, extra in groups:
                m = stop - start
                if g == "x":
                    out = self.x.expand(m, -1)
                elif g == "const":
                    out = self.const(extra)
                elif g == "pow":
                    out = torch.tanh(self.pow(torch.cat([H[child], extra], dim=1)))
                elif g in ("add", "mul"):
                    summed = torch.zeros(m, self.dim).index_add(0, parent, H[child])
                    out = torch.tanh((self.add if g == "add" else self.mul)(summed))
                elif g == "div":
                    # children were appended in order (num, den) for each node
                    out = torch.tanh(self.div(H[child].reshape(m, 2 * self.dim)))
                else:
                    out = torch.tanh(self.fn[g[3:]](H[child]))
                blocks.append(out)
        return torch.cat(blocks)[forest.roots]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def bt_loss(scores: torch.Tensor, pairs: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Mean Bradley-Terry cross-entropy, as in Section 4 (ties are y = 1/2)."""
    t = scores[pairs[:, 0]] - scores[pairs[:, 1]]
    return (nn.functional.softplus(t) - y * t).mean()


class TreeBT:
    """delta_hat(x) = v . h(x)  [+ theta . z(x) if features Z are given: a hybrid model].

    Scores every tree in `trees` (training problems and unseen ones alike), so
    it plugs into evaluate.cross_val_diffs: score() ignores its argument.
    """

    def __init__(self, trees: list[Node], dim: int = 16, Z: np.ndarray | None = None,
                 encoder: TreeEncoder | None = None, freeze_encoder: bool = False,
                 lr: float = 0.01, weight_decay: float = 1e-3, max_epochs: int = 400,
                 patience: int = 40, seed: int = 0):
        torch.manual_seed(seed)
        self.forest = compile_forest(trees)
        self.encoder = encoder if encoder is not None else TreeEncoder(dim)
        self.head = nn.Linear(self.encoder.dim, 1, bias=False)
        self.Z = None if Z is None else torch.tensor(Z, dtype=torch.float32)
        self.lin = None if Z is None else nn.Linear(Z.shape[1], 1, bias=False)
        self.freeze_encoder = freeze_encoder
        self.lr, self.weight_decay = lr, weight_decay
        self.max_epochs, self.patience, self.seed = max_epochs, patience, seed

    def _scores(self) -> torch.Tensor:
        s = self.head(self.encoder(self.forest)).squeeze(1)
        if self.lin is not None:
            s = s + self.lin(self.Z).squeeze(1)
        return s

    def _params(self):
        ps = list(self.head.parameters()) + (list(self.lin.parameters()) if self.lin is not None else [])
        if not self.freeze_encoder:
            ps += list(self.encoder.parameters())
        return ps

    def fit(self, pairs: np.ndarray, y: np.ndarray, rng: np.random.Generator | None = None) -> "TreeBT":
        """Full-batch Adam with early stopping on a held-out 20% of the training *problems*."""
        rng = rng or np.random.default_rng(self.seed)
        items = np.unique(pairs)
        val_items = rng.choice(items, size=max(1, len(items) // 5), replace=False)
        val = pairs_touching(pairs, val_items)
        P = torch.tensor(pairs, dtype=torch.long)
        Y = torch.tensor(y, dtype=torch.float32)
        tr, va = torch.tensor(~val), torch.tensor(val)
        opt = torch.optim.AdamW(self._params(), lr=self.lr, weight_decay=self.weight_decay)
        best, best_state, since = np.inf, None, 0
        self.history_ = []
        for epoch in range(self.max_epochs):
            opt.zero_grad()
            s = self._scores()
            loss = bt_loss(s, P[tr], Y[tr])
            loss.backward()                       # reverse-mode autodiff through the whole forest
            opt.step()
            with torch.no_grad():
                v = bt_loss(self._scores(), P[va], Y[va]).item() if va.any() else loss.item()
            self.history_.append((loss.item(), v))
            if v < best - 1e-5:
                best, since = v, 0
                best_state = [p.detach().clone() for p in self._params()]
            else:
                since += 1
                if since > self.patience:
                    break
        with torch.no_grad():
            for p, b in zip(self._params(), best_state):
                p.copy_(b)
        self.best_epoch_ = int(np.argmin([h[1] for h in self.history_])) + 1
        return self

    def fit_lp_ft(self, pairs: np.ndarray, y: np.ndarray, ft_lr: float = 1e-3,
                  rng: np.random.Generator | None = None) -> "TreeBT":
        """Linear probing, then fine-tuning (Kumar et al. 2022).

        Fine-tuning a pretrained encoder under a randomly initialized head sends
        large, meaningless gradients into the encoder before the head has learned
        anything, and damages the representation. So: first fit the head with the
        encoder frozen, then unfreeze everything at a small learning rate.
        """
        frozen, lr = self.freeze_encoder, self.lr
        self.freeze_encoder = True
        self.fit(pairs, y, rng=np.random.default_rng(self.seed) if rng is None else rng)
        self.freeze_encoder, self.lr = False, ft_lr
        self.fit(pairs, y, rng=np.random.default_rng(self.seed) if rng is None else rng)
        self.freeze_encoder, self.lr = frozen, lr
        return self

    def score(self, _=None) -> np.ndarray:
        with torch.no_grad():
            return self._scores().numpy().astype(float)


def pretrain_encoder(trees: list[Node], targets: np.ndarray, dim: int = 16, epochs: int = 300,
                     lr: float = 0.01, seed: int = 0) -> tuple[TreeEncoder, list[float]]:
    """Self-supervised pretraining: predict facts the differentiator computes for free.

    `targets` (n_trees x m) are standardized and regressed from the root
    embedding with a linear head (mean squared error). The generator supplies
    unlimited unlabeled problems and the step-tracing differentiator labels
    them, so the encoder can learn "what makes a derivative long" before it
    sees a single human judgment.
    """
    torch.manual_seed(seed)
    forest = compile_forest(trees)
    T = torch.tensor((targets - targets.mean(0)) / (targets.std(0) + 1e-9), dtype=torch.float32)
    enc = TreeEncoder(dim)
    head = nn.Linear(dim, targets.shape[1])
    opt = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()), lr=lr)
    losses = []
    for _ in range(epochs):
        opt.zero_grad()
        loss = ((head(enc(forest)) - T) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return enc, losses
