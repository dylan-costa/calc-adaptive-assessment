"""Bradley-Terry with difficulty as a linear function of features.

    delta_hat(x) = theta . z(x),   z = standardized phi,
    P(i harder than j) = sigma(theta . (z_i - z_j)).

This is the model that generalizes: it scores any problem, labeled or not.
Fitting is the same convex problem as item-level BT (Section 4), with design
rows z_i - z_j instead of e_i - e_j, so `fit_newton` is reused unchanged.
There is no intercept: it would cancel in every difference.
"""

from __future__ import annotations

import numpy as np

from calcdiff.evaluate import log_loss, pairs_touching
from calcdiff.features import FEATURE_NAMES, Standardizer
from calcdiff.models.bt import BTFit, fit_newton


class FeatureBT:
    def __init__(self, l2: float = 1.0, columns: list[int] | None = None):
        self.l2 = l2
        self.columns = columns  # restrict to a subset of features (for baselines)

    def _select(self, Phi: np.ndarray) -> np.ndarray:
        return Phi if self.columns is None else Phi[:, self.columns]

    def fit(self, Phi: np.ndarray, pairs: np.ndarray, y: np.ndarray) -> "FeatureBT":
        """Phi holds features for all problems referenced by `pairs` (rows indexed by item id).

        The standardizer is fit only on the problems that appear in training
        comparisons, so held-out problems cannot leak into the scaling.
        """
        X = self._select(Phi)
        seen = np.unique(pairs)
        self.scaler_ = Standardizer().fit(X[seen])
        Z = self.scaler_.transform(X)
        D = Z[pairs[:, 0]] - Z[pairs[:, 1]]
        self.fit_: BTFit = fit_newton(D, y, l2=self.l2)
        self.theta_ = self.fit_.theta
        return self

    def score(self, Phi: np.ndarray) -> np.ndarray:
        """Predicted difficulty (in logits) for every row of Phi."""
        return self.scaler_.transform(self._select(Phi)) @ self.theta_

    def raw_weights(self) -> np.ndarray:
        """Weights per unit of the *raw* feature: theta_j / s_j."""
        return self.theta_ / self.scaler_.std_

    def raw_weight_se(self) -> np.ndarray:
        """Laplace standard errors of raw_weights()."""
        return np.sqrt(np.diag(self.fit_.covariance())) / self.scaler_.std_

    def names(self) -> list[str]:
        cols = range(len(FEATURE_NAMES)) if self.columns is None else self.columns
        return [FEATURE_NAMES[c] for c in cols]


def cv_l2(Phi: np.ndarray, pairs: np.ndarray, y: np.ndarray, items: np.ndarray,
          grid: np.ndarray, rng: np.random.Generator, folds: int = 5) -> tuple[float, np.ndarray]:
    """Choose l2 by K-fold cross-validation over *problems*.

    Each fold holds out a set of problems. The model trains on comparisons
    among the remaining problems and is scored on every comparison that
    involves a held-out problem: one unseen problem is enough to test
    generalization, since a feature model has no per-problem parameters.
    (Requiring *both* problems to be held out would discard about a third of
    the comparisons and can leave a fold with none at all.)

    Returns (best l2, held-out log-loss per grid value, pooled over folds).
    """
    perm = rng.permutation(items)
    chunks = np.array_split(perm, folds)
    total = np.zeros(len(grid))
    count = 0
    for held in chunks:
        te = pairs_touching(pairs, held)
        tr = ~te
        if te.sum() == 0 or tr.sum() == 0:
            continue
        for g, l2 in enumerate(grid):
            m = FeatureBT(l2=l2).fit(Phi, pairs[tr], y[tr])
            total[g] += te.sum() * log_loss(m.score(Phi), pairs[te], y[te])
        count += te.sum()
    mean = total / count
    return float(grid[np.argmin(mean)]), mean


def problem_bootstrap(Phi: np.ndarray, pairs: np.ndarray, y: np.ndarray, l2: float,
                      rng: np.random.Generator, n_boot: int = 200) -> np.ndarray:
    """Raw weights refit on problem-resampled data, shape (n_boot, d).

    Laplace standard errors (raw_weight_se) assume comparisons are independent.
    They are not: every comparison involving a problem shares whatever about
    that problem the features miss. Resampling *problems* (comparison (i, j)
    repeated c_i c_j times, as in evaluate.bootstrap_ci) respects that.
    """
    items = np.unique(pairs)
    n = pairs.max() + 1
    out = []
    for _ in range(n_boot):
        c = np.bincount(rng.choice(items, size=len(items), replace=True), minlength=n)
        idx = np.repeat(np.arange(len(pairs)), c[pairs[:, 0]] * c[pairs[:, 1]])
        out.append(FeatureBT(l2=l2).fit(Phi, pairs[idx], y[idx]).raw_weights())
    return np.array(out)
