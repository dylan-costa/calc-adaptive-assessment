"""A probabilistic grammar (PCFG) over derivative problems.

Grammar (one nonterminal F, a "differentiable factor"):

    E    ->  T_1 + ... + T_k  [+ c]              k ~ term_count_probs
    T    ->  [c ·] F
    F    ->  LEAF        c · x^n                  (0 children)
          |  FN          g(F) or g(F + c)         (1 child)
          |  POWF        (F + c)^n                (1 child)
          |  PROD        F · F                    (2 children)
          |  QUOT        F / F                    (2 children)

Seen as a Galton-Watson branching process, each F has 0, 1 or 2 F-children
with probabilities p_leaf, p_fn + p_powf, p_prod + p_quot. The mean offspring

    m = (p_fn + p_powf) + 2 (p_prod + p_quot)

must be < 1 (subcritical) for trees to be finite with probability 1; see
Section 1 of notes/lectures.tex. `max_depth` additionally truncates the
process by forcing LEAF at that depth.

Every random choice is recorded in `log_prob`, so log p(problem) is known
exactly. That is what lets us reweight between two grammars (covariate shift).
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

from calcdiff.expr import Add, Const, Div, Fn, Mul, Node, Pow, Var, X, size, to_latex, walk

RULES = ("LEAF", "FN", "POWF", "PROD", "QUOT")


@dataclass(frozen=True)
class GrammarConfig:
    p_leaf: float = 0.50
    p_fn: float = 0.22
    p_powf: float = 0.08
    p_prod: float = 0.12
    p_quot: float = 0.08
    max_depth: int | None = 4
    term_count_probs: tuple[float, ...] = (0.5, 0.35, 0.15)  # k = 1, 2, 3
    p_const_term: float = 0.2      # trailing "+ c" on the whole expression
    p_coef: float = 0.35           # non-unit coefficient on a leaf / term
    p_shift: float = 0.35          # g(F + c) instead of g(F)
    fn_weights: tuple[tuple[str, float], ...] = (
        ("sin", 1.0), ("cos", 1.0), ("tan", 0.5),
        ("exp", 1.0), ("ln", 1.0), ("sqrt", 0.7),
    )
    exponent_weights: tuple[tuple[Fraction, float], ...] = (
        (Fraction(1), 3.0), (Fraction(2), 3.0), (Fraction(3), 2.0),
        (Fraction(4), 1.0), (Fraction(5), 0.5),
        (Fraction(-1), 1.0), (Fraction(-2), 0.7),
        (Fraction(1, 2), 0.7), (Fraction(3, 2), 0.4),
        (Fraction(-1, 2), 0.4), (Fraction(1, 3), 0.3),
    )
    powf_exponent_weights: tuple[tuple[Fraction, float], ...] = (
        (Fraction(2), 3.0), (Fraction(3), 2.0), (Fraction(4), 1.0),
        (Fraction(5), 1.0), (Fraction(10), 0.3),
        (Fraction(-1), 0.7), (Fraction(1, 2), 0.7), (Fraction(-1, 2), 0.3),
    )
    max_size: int = 30             # reject problems with more AST nodes

    def __post_init__(self) -> None:
        total = self.p_leaf + self.p_fn + self.p_powf + self.p_prod + self.p_quot
        if not math.isclose(total, 1.0):
            raise ValueError(f"rule probabilities sum to {total}, not 1")

    @property
    def rule_probs(self) -> dict[str, float]:
        return {"LEAF": self.p_leaf, "FN": self.p_fn, "POWF": self.p_powf,
                "PROD": self.p_prod, "QUOT": self.p_quot}

    @property
    def mean_offspring(self) -> float:
        return (self.p_fn + self.p_powf) + 2 * (self.p_prod + self.p_quot)

    @property
    def offspring_dist(self) -> np.ndarray:
        """P(0 children), P(1 child), P(2 children) for one F node."""
        return np.array([self.p_leaf, self.p_fn + self.p_powf, self.p_prod + self.p_quot])


@dataclass
class Problem:
    tree: Node
    log_prob: float                      # log p(derivation) under the grammar
    rule_counts: Counter = field(default_factory=Counter)
    n_expansions: int = 0                # number of F nodes in the derivation
    derivation_depth: int = 0            # deepest F level reached (root F = 0)

    @property
    def latex(self) -> str:
        return to_latex(self.tree)

    @property
    def id(self) -> str:
        return hashlib.sha1(self.latex.encode()).hexdigest()[:12]


class Sampler:
    """Draws problems from the PCFG. Not thread-safe (it keeps a running log-prob)."""

    def __init__(self, config: GrammarConfig | None = None, seed: int | None = None):
        self.cfg = config or GrammarConfig()
        self.rng = np.random.default_rng(seed)
        self._fn_names = [n for n, _ in self.cfg.fn_weights]
        self._fn_p = _normalize([w for _, w in self.cfg.fn_weights])
        self._exps = [e for e, _ in self.cfg.exponent_weights]
        self._exp_p = _normalize([w for _, w in self.cfg.exponent_weights])
        self._pexps = [e for e, _ in self.cfg.powf_exponent_weights]
        self._pexp_p = _normalize([w for _, w in self.cfg.powf_exponent_weights])
        self._rule_p = np.array([self.cfg.rule_probs[r] for r in RULES])

    # -- primitive random choices, each adding its log-probability -------------

    def _choice(self, options, probs):
        i = self.rng.choice(len(options), p=probs)
        self._lp += math.log(probs[i])
        return options[i]

    def _bernoulli(self, p: float) -> bool:
        b = bool(self.rng.random() < p)
        self._lp += math.log(p if b else 1 - p)
        return b

    def _coef(self) -> int:
        """1 with prob 1 - p_coef, else uniform on {±2, ..., ±9} (negative w.p. 1/4)."""
        if not self._bernoulli(self.cfg.p_coef):
            return 1
        mag = int(self.rng.integers(2, 10))
        self._lp += math.log(1 / 8)
        neg = self._bernoulli(0.25)
        return -mag if neg else mag

    def _shift_const(self) -> int:
        c = int(self.rng.integers(1, 10))
        self._lp += math.log(1 / 9)
        return -c if self._bernoulli(0.3) else c

    # -- the grammar ----------------------------------------------------------

    def _F(self, d: int) -> Node:
        self._n += 1
        self._maxd = max(self._maxd, d)
        if self.cfg.max_depth is not None and d >= self.cfg.max_depth:
            rule = "LEAF"  # forced: contributes log 1 = 0
            self._counts["FORCED"] += 1
        else:
            rule = self._choice(RULES, self._rule_p)
        self._counts[rule] += 1

        if rule == "LEAF":
            n = self._choice(self._exps, self._exp_p)
            base: Node = X if n == 1 else Pow(X, n)
            c = self._coef()
            return base if c == 1 else Mul((Const(Fraction(c)), base))
        if rule == "FN":
            g = self._choice(self._fn_names, self._fn_p)
            inner = self._F(d + 1)
            if self._bernoulli(self.cfg.p_shift):
                inner = Add((inner, Const(Fraction(self._shift_const()))))
            return Fn(g, inner)
        if rule == "POWF":
            inner = self._F(d + 1)
            inner = Add((inner, Const(Fraction(self._shift_const()))))
            return Pow(inner, self._choice(self._pexps, self._pexp_p))
        if rule == "PROD":
            return Mul((self._F(d + 1), self._F(d + 1)))
        if rule == "QUOT":
            return Div(self._F(d + 1), self._F(d + 1))
        raise AssertionError(rule)

    def _term(self) -> Node:
        f = self._F(0)
        if _is_leaf(f):
            return f  # leaves already carry their own coefficient
        c = self._coef()
        return f if c == 1 else Mul((Const(Fraction(c)), f))

    def _reset(self) -> None:
        self._lp = 0.0
        self._n = 0
        self._maxd = 0
        self._counts: Counter = Counter()

    def sample_factor_tree(self) -> Problem:
        """A single F-tree with no outer sum. Used to study the branching process."""
        self._reset()
        tree = self._F(0)
        return Problem(tree, self._lp, self._counts, self._n, self._maxd)

    def sample_raw(self) -> Problem:
        """One draw from the grammar, with no rejection."""
        self._reset()
        ks = list(range(1, len(self.cfg.term_count_probs) + 1))
        k = self._choice(ks, _normalize(self.cfg.term_count_probs))
        terms = [self._term() for _ in range(k)]
        if self._bernoulli(self.cfg.p_const_term):
            terms.append(Const(Fraction(self._shift_const())))
        tree = terms[0] if len(terms) == 1 else Add(tuple(terms))
        return Problem(tree, self._lp, self._counts, self._n, self._maxd)

    def sample(self) -> Problem:
        """Rejection sampling: draw until the problem is one an instructor would write.

        The accepted problems follow p(x | A) = p(x) 1[x in A] / p(A), where A is
        the acceptance set. `log_prob` stays the *unconditioned* log p(x); the
        missing constant -log p(A) cancels in any ratio of two accepted problems.
        """
        while True:
            p = self.sample_raw()
            if size(p.tree) <= self.cfg.max_size and is_natural(p.tree):
                return p

    def sample_unique(self, n: int, max_tries: int | None = None) -> list[Problem]:
        """n distinct problems (distinct as written, i.e. distinct LaTeX)."""
        seen: dict[str, Problem] = {}
        tries = 0
        limit = max_tries or 50 * n
        while len(seen) < n and tries < limit:
            p = self.sample()
            seen.setdefault(p.id, p)
            tries += 1
        return list(seen.values())


def log_likelihood_ratio(problem: Problem, p: GrammarConfig, q: GrammarConfig) -> float:
    """log q(x) - log p(x) for a problem drawn from grammar p.

    Valid when p and q differ only in their rule probabilities (and share
    max_depth). Every other random choice has the same probability under both
    and cancels, so the ratio depends on x only through its rule counts:

        log q(x)/p(x) = sum_r c_r(x) log(q_r / p_r),

    i.e. derivation probabilities form an exponential family with sufficient
    statistic c(x). Forced leaves at max_depth have probability 1 under both
    grammars and are excluded.
    """
    if p.max_depth != q.max_depth:
        raise ValueError("grammars must share max_depth")
    counts = dict(problem.rule_counts)
    counts["LEAF"] = counts.get("LEAF", 0) - counts.get("FORCED", 0)
    pr, qr = p.rule_probs, q.rule_probs
    return sum(counts.get(r, 0) * math.log(qr[r] / pr[r]) for r in RULES)


def is_natural(tree: Node) -> bool:
    """Reject forms a calculus instructor would never write.

    - a product or quotient of two monomials (x^2 * x^3, x^3 / x^2): it simplifies
      to a single monomial, so it is really a power-rule problem in disguise
    - a fraction directly inside a fraction
    - like terms in a sum (6x^2 + x^2)
    - ln(e^u) or e^(ln u), which cancel
    - stacked coefficients: once a product is flattened, it may contain at most
      one constant, which must come first (3 * 9x * u, or u * 7x, are out), and
      at most one monomial in x (9x * sin(x) * x is out). Found by the labels
      of Section 8, not by a test.
    """
    for node in walk(tree):
        match node:
            case Mul(fs):
                if not isinstance(fs[0], Const) and all(_is_leaf(f) for f in fs):
                    return False
                flat = _flatten_product(node)
                consts = [k for k, f in enumerate(flat) if isinstance(f, Const)]
                if consts not in ([], [0]):
                    return False
                if sum(_is_monomial(f) for f in flat) > 1:
                    return False
            case Div(n, d):
                if _is_leaf(n) and _is_leaf(d):
                    return False
                if isinstance(n, Div) or isinstance(d, Div):
                    return False
            case Add(ts):
                exps = [_leaf_exponent(t) for t in ts if _is_leaf(t)]
                if len(exps) != len(set(exps)):
                    return False
                consts = [t for t in ts if isinstance(t, Const)]
                if len(consts) > 1:
                    return False
            case Fn("ln", Fn("exp", _)) | Fn("exp", Fn("ln", _)):
                return False
    return True


def _flatten_product(node: Node) -> list[Node]:
    """Factors of a product with nested products spliced in: a(bc) -> [a, b, c]."""
    if isinstance(node, Mul):
        return [g for f in node.factors for g in _flatten_product(f)]
    return [node]


def _is_monomial(node: Node) -> bool:
    """x or x^n (without coefficient: coefficients are separate factors once flattened)."""
    return isinstance(node, Var) or (isinstance(node, Pow) and isinstance(node.base, Var))


def _leaf_exponent(node: Node) -> Fraction:
    if isinstance(node, Mul):
        node = node.factors[1]
    if isinstance(node, Pow):
        return node.exp
    return Fraction(1)


def _is_leaf(node: Node) -> bool:
    if isinstance(node, Var):
        return True
    if isinstance(node, Pow) and isinstance(node.base, Var):
        return True
    return isinstance(node, Mul) and len(node.factors) == 2 and isinstance(node.factors[0], Const) \
        and _is_leaf(node.factors[1])


def _normalize(ws) -> np.ndarray:
    a = np.asarray(ws, dtype=float)
    return a / a.sum()
