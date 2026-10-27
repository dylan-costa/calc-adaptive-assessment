"""Evaluating difficulty models on held-out problems.

The question we care about is "how well does the model rank problems it has
never seen?", so the unit we hold out is the *problem*, not the comparison.
Splitting comparisons at random would put the same problems on both sides
of the split and measure something easier (interpolation between known
items), which is a form of leakage.

Comparisons that share a problem are dependent, so naive binomial confidence
intervals for pairwise accuracy are too narrow. `bootstrap_ci` resamples
problems instead of pairs.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.special import expit
from scipy.stats import kendalltau, spearmanr


def split_problems(n_items: int, test_frac: float, rng: np.random.Generator):
    """Random partition of item indices into (train, test)."""
    perm = rng.permutation(n_items)
    n_test = int(round(test_frac * n_items))
    return np.sort(perm[n_test:]), np.sort(perm[:n_test])


def pairs_within(pairs: np.ndarray, items: np.ndarray) -> np.ndarray:
    """Boolean mask: comparisons whose *both* problems lie in `items`."""
    inside = np.zeros(pairs.max() + 1 if len(pairs) else 0, dtype=bool)
    inside[items[items < len(inside)]] = True
    return inside[pairs[:, 0]] & inside[pairs[:, 1]]


def pairs_touching(pairs: np.ndarray, items: np.ndarray) -> np.ndarray:
    """Boolean mask: comparisons with *at least one* problem in `items`."""
    inside = np.zeros(max(pairs.max(), items.max()) + 1, dtype=bool)
    inside[items] = True
    return inside[pairs[:, 0]] | inside[pairs[:, 1]]


def pairwise_accuracy(scores: np.ndarray, pairs: np.ndarray, y: np.ndarray,
                      weights: np.ndarray | None = None) -> float:
    """Fraction of decided comparisons (y in {0, 1}) whose order the scores get right.

    A model that scores both problems equally gets credit 1/2, so a constant
    model scores exactly 0.5. Ties in the labels (y = 1/2) are skipped.
    """
    return accuracy_from_diff(scores[pairs[:, 0]] - scores[pairs[:, 1]], y, weights)


def accuracy_from_diff(diff: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None) -> float:
    """pairwise_accuracy, given predicted differences delta_hat_i - delta_hat_j directly."""
    decided = y != 0.5
    tied = np.abs(diff) < 1e-9  # equal scores, up to floating-point dust
    credit = np.where(tied, 0.5, ((diff > 0) == (y == 1)).astype(float))
    w = np.ones(len(y)) if weights is None else weights
    w = w * decided
    return float((w * credit).sum() / w.sum())


def log_loss(scores: np.ndarray, pairs: np.ndarray, y: np.ndarray) -> float:
    """Mean cross-entropy of the BT predictions sigma(score_i - score_j)."""
    return log_loss_from_diff(scores[pairs[:, 0]] - scores[pairs[:, 1]], y)


def log_loss_from_diff(t: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None) -> float:
    w = np.ones(len(y)) if weights is None else weights
    return float((w * (np.logaddexp(0.0, t) - y * t)).sum() / w.sum())


def cross_val_diffs(Phi: np.ndarray, pairs: np.ndarray, y: np.ndarray, make_model: Callable,
                    rng: np.random.Generator, folds: int = 5) -> np.ndarray:
    """Out-of-fold predicted differences, holding out *problems*.

    The problems are split into K folds. The model for fold f is trained on the
    comparisons that avoid f entirely, and predicts every comparison touching f.
    A comparison whose two problems lie in different folds is predicted by two
    models (neither of which saw it); we average the two predictions.
    `make_model(train_pairs, train_y)` must return a fitted model with .score(Phi).
    """
    items = np.unique(pairs)
    chunks = np.array_split(rng.permutation(items), folds)
    total = np.zeros(len(pairs))
    count = np.zeros(len(pairs))
    for held in chunks:
        te = pairs_touching(pairs, held)
        model = make_model(pairs[~te], y[~te])
        s = model.score(Phi)
        total[te] += s[pairs[te, 0]] - s[pairs[te, 1]]
        count[te] += 1
    return total / count


def bayes_ceiling(delta: np.ndarray, pairs: np.ndarray) -> float:
    """Expected accuracy of an oracle that knows the true difficulties.

    Labels are random: the harder problem is chosen only with probability
    sigma(|delta_i - delta_j|). No model can beat E[max(p, 1 - p)].
    """
    p = expit(delta[pairs[:, 0]] - delta[pairs[:, 1]])
    return float(np.mean(np.maximum(p, 1 - p)))


def rank_agreement(scores: np.ndarray, delta: np.ndarray) -> dict[str, float]:
    """Spearman rho and Kendall tau between predicted and true difficulty."""
    return {"spearman": float(spearmanr(scores, delta)[0]),
            "kendall": float(kendalltau(scores, delta)[0])}


def bootstrap_ci(metric: Callable[[np.ndarray], float], pairs: np.ndarray, items: np.ndarray,
                 rng: np.random.Generator, n_boot: int = 1000, level: float = 0.95):
    """Percentile CI for a pairwise metric, resampling *problems* with replacement.

    If problem i is drawn c_i times, each comparison (i, j) gets weight c_i c_j:
    the comparisons that would exist among the resampled problems. `metric`
    receives these per-comparison weights.
    """
    n = pairs.max() + 1
    stats = []
    for _ in range(n_boot):
        draw = rng.choice(items, size=len(items), replace=True)
        c = np.bincount(draw, minlength=n).astype(float)
        w = c[pairs[:, 0]] * c[pairs[:, 1]]
        if w.sum() > 0:
            stats.append(metric(w))
    lo, hi = np.quantile(stats, [(1 - level) / 2, (1 + level) / 2])
    return float(lo), float(hi)
