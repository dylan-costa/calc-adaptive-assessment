"""Bradley-Terry with difficulty as a sum of regression trees (gradient boosting).

    delta_hat(x) = nu * sum_t f_t(phi(x)),   each f_t a small regression tree.

Boosting is gradient descent in *function space*. Let F_i = delta_hat(x_i)
be the current scores of the training problems and

    L(F) = sum_k [softplus(F_i - F_j) - y_k (F_i - F_j)]       (comparison k = (i, j)).

Each round:
  1. per-problem gradient g_i = dL/dF_i and diagonal curvature h_i = d2L/dF_i^2;
  2. fit a regression tree to the Newton direction -g_i / h_i, weighted by h_i,
     using only the features: this projects the step onto "functions a tree
     can express", which is what lets it generalize to unseen problems;
  3. set each leaf's value by an exact one-dimensional Newton step. Shifting the
     scores of the problems in a leaf R by gamma changes only comparisons that
     cross R, so dL/dgamma = sum_{i in R} g_i and d2L/dgamma2 = cut_w(R), the
     weighted cut of R in the comparison graph (= 1_R^T L_w 1_R, Section 4);
  4. take a small step nu (shrinkage) in that direction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import expit
from sklearn.tree import DecisionTreeRegressor

from calcdiff.evaluate import pairs_touching


def problem_grad_hess(F: np.ndarray, pairs: np.ndarray, y: np.ndarray):
    """Per-problem gradient g and diagonal Hessian h of the BT loss at scores F."""
    n = len(F)
    p = expit(F[pairs[:, 0]] - F[pairs[:, 1]])
    r = p - y                      # d loss_k / d t_k, the "lambda" of LambdaRank
    w = p * (1 - p)                # d2 loss_k / d t_k^2
    g = np.bincount(pairs[:, 0], r, n) - np.bincount(pairs[:, 1], r, n)
    h = np.bincount(pairs[:, 0], w, n) + np.bincount(pairs[:, 1], w, n)
    return g, h, w


def leaf_cut(leaf_of: np.ndarray, pairs: np.ndarray, w: np.ndarray, n_leaves: int) -> np.ndarray:
    """cut_w(R) for every leaf R: total weight of comparisons with exactly one end in R."""
    li, lj = leaf_of[pairs[:, 0]], leaf_of[pairs[:, 1]]
    cross = li != lj
    return (np.bincount(li[cross], w[cross], n_leaves) + np.bincount(lj[cross], w[cross], n_leaves))


@dataclass
class BoostedBT:
    n_trees: int = 300
    learning_rate: float = 0.1
    max_depth: int = 3
    min_leaf: int = 5
    leaf_l2: float = 1.0          # ridge on each leaf value: gamma = -G / (cut + leaf_l2)
    seed: int = 0
    trees_: list = field(default_factory=list, repr=False)
    train_loss_: list = field(default_factory=list, repr=False)

    def fit(self, Phi: np.ndarray, pairs: np.ndarray, y: np.ndarray) -> "BoostedBT":
        items = np.unique(pairs)
        F = np.zeros(len(Phi))
        self.trees_, self.train_loss_ = [], []
        for _ in range(self.n_trees):
            g, h, w = problem_grad_hess(F, pairs, y)
            tree = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_leaf=self.min_leaf,
                                         random_state=self.seed)
            tree.fit(Phi[items], -g[items] / (h[items] + 1e-12), sample_weight=h[items] + 1e-12)
            n_nodes = tree.tree_.node_count
            leaf_of = tree.apply(Phi)                                   # leaf id for every problem
            G = np.bincount(leaf_of[items], g[items], n_nodes)          # sum of g over each leaf
            value = -G / (leaf_cut(leaf_of, pairs, w, n_nodes) + self.leaf_l2)
            F += self.learning_rate * value[leaf_of]
            self.trees_.append((tree, value))
            t = F[pairs[:, 0]] - F[pairs[:, 1]]
            self.train_loss_.append(float(np.mean(np.logaddexp(0.0, t) - y * t)))
        return self

    def staged_scores(self, Phi: np.ndarray):
        """Scores after 1, 2, ..., n_trees trees (for choosing the number of trees)."""
        F = np.zeros(len(Phi))
        for tree, value in self.trees_:
            F = F + self.learning_rate * value[tree.apply(Phi)]
            yield F

    def score(self, Phi: np.ndarray, n_trees: int | None = None) -> np.ndarray:
        F = np.zeros(len(Phi))
        for tree, value in self.trees_[:n_trees]:
            F += self.learning_rate * value[tree.apply(Phi)]
        return F

    def feature_gain(self, n_features: int) -> np.ndarray:
        """Total weighted impurity decrease attributed to each feature, normalized to sum 1."""
        imp = np.zeros(n_features)
        for tree, _ in self.trees_:
            imp += tree.tree_.compute_feature_importances(normalize=False)
        return imp / imp.sum() if imp.sum() > 0 else imp


def staged_held_out_loss(model: BoostedBT, Phi: np.ndarray, pairs: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Held-out log-loss after each tree."""
    out = []
    for F in model.staged_scores(Phi):
        t = F[pairs[:, 0]] - F[pairs[:, 1]]
        out.append(np.mean(np.logaddexp(0.0, t) - y * t))
    return np.array(out)


def cv_n_trees(Phi: np.ndarray, pairs: np.ndarray, y: np.ndarray, rng: np.random.Generator,
               folds: int = 3, **params) -> tuple[int, np.ndarray]:
    """Choose the number of trees (early stopping) by cross-validation over problems.

    The number of trees is boosting's main regularizer, playing the role
    lambda played for ridge. One fit per fold gives the held-out loss after
    every tree, so the whole curve costs no more than a single fit.
    Returns (best number of trees, pooled held-out log-loss per number of trees).
    """
    chunks = np.array_split(rng.permutation(np.unique(pairs)), folds)
    total, count = 0.0, 0
    for held in chunks:
        te = pairs_touching(pairs, held)
        if te.sum() == 0 or (~te).sum() == 0:
            continue
        model = BoostedBT(**params).fit(Phi, pairs[~te], y[~te])
        total = total + te.sum() * staged_held_out_loss(model, Phi, pairs[te], y[te])
        count += te.sum()
    curve = total / count
    return int(np.argmin(curve)) + 1, curve
