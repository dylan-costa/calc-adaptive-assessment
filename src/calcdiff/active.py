"""Active learning: which comparison should the instructor answer next?

The feature model of Section 6 carries an approximate Gaussian posterior
theta ~ N(theta_hat, Sigma) (the Laplace approximation, Sigma = H^-1). For a
candidate pair with feature difference d = z_i - z_j:

    predicted probability   p   = sigma(theta_hat . d)
    Fisher weight           w   = p (1 - p)            (large for close calls)
    posterior variance      s^2 = d^T Sigma d          (large in uncertain directions)

Answering the pair adds w d d^T to the posterior precision H, so by the matrix
determinant lemma

    log det(H + w d d^T) - log det H = log(1 + w s^2).

Half of that is the expected information gain about theta under the Laplace
approximation, and maximizing it greedily is sequential D-optimal design.
Uncertainty sampling uses w alone: it asks about close calls without asking
whether the answer would teach the model anything new.

With the mixed model (models/mixed_bt.py) the parameters are (theta, u), the
design row is a = [d, e_i - e_j], and what we want is information about theta
*alone*, the part that generalizes to new problems. The joint covariance Sigma
takes a rank-one update, and the marginal block for theta becomes

    Sigma_tt' = Sigma_tt - c v v^T,   v = (Sigma a)_theta,   c = w / (1 + w a^T Sigma a),

so, by the determinant lemma again, the information gain about theta is

    -1/2 log(1 - c v^T Sigma_tt^-1 v).

Asking about a problem whose u_i is already pinned down by earlier comparisons
still helps; asking about it for the fifth time mostly refines u_i, and this
formula knows the difference.
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit

from calcdiff.models.feature_bt import FeatureBT
from calcdiff.models.mixed_bt import MixedBT

STRATEGIES = ("random", "uncertainty", "information", "variance")


def all_pairs(items: np.ndarray) -> np.ndarray:
    """Every unordered pair of distinct items, as rows (i, j) with i < j."""
    a, b = np.triu_indices(len(items), k=1)
    return np.column_stack([items[a], items[b]])


def acquisition(model: FeatureBT | MixedBT, Phi: np.ndarray, cand: np.ndarray, strategy: str) -> np.ndarray:
    """Score each candidate pair; higher means more worth asking."""
    if strategy == "random":
        return np.zeros(len(cand))
    if isinstance(model, MixedBT):
        return _acquisition_mixed(model, cand, strategy)
    Z = model.scaler_.transform(model._select(Phi))
    D = Z[cand[:, 0]] - Z[cand[:, 1]]
    p = expit(D @ model.theta_)
    w = p * (1 - p)
    if strategy == "uncertainty":
        return w
    Sigma = model.fit_.covariance()
    s2 = np.einsum("kj,jl,kl->k", D, Sigma, D)       # d_k^T Sigma d_k for every candidate
    if strategy == "information":
        return 0.5 * np.log1p(w * s2)
    if strategy == "variance":
        # V-optimal: expected drop in average predictive variance over *random* pairs,
        # tr(Sigma C) with C = E[d d^T] = 2 Cov(z) over the pool
        C = 2 * np.cov(Z[np.unique(cand)], rowvar=False)
        SD = D @ Sigma                                 # rows (Sigma d_k)^T
        return w / (1 + w * s2) * np.einsum("kj,jl,kl->k", SD, C, SD)
    raise ValueError(strategy)


def _acquisition_mixed(model: MixedBT, cand: np.ndarray, strategy: str) -> np.ndarray:
    i, j = cand[:, 0], cand[:, 1]
    s = model.pool_score()
    p = expit(s[i] - s[j])
    w = p * (1 - p)
    if strategy == "uncertainty":
        return w
    if strategy not in ("information", "variance"):
        raise ValueError(strategy)
    d = model.d_
    Sigma = model.fit_.covariance()
    G = np.hstack([model.Z_, np.eye(len(model.Z_))])      # rows g_i, so a = g_i - g_j
    GS = G @ Sigma
    K = GS @ G.T                                           # a^T Sigma a = K_ii + K_jj - 2 K_ij
    B = GS[:, :d]                                          # v = B_i - B_j
    M = B @ np.linalg.solve(Sigma[:d, :d], B.T)            # v^T Sigma_tt^-1 v = M_ii + M_jj - 2 M_ij
    s2 = K[i, i] + K[j, j] - 2 * K[i, j]
    c = w / (1 + w * s2)
    if strategy == "variance":
        # expected drop in tr(Sigma_tt C): predictive variance of theta . (z - z') over random pairs
        C = 2 * np.cov(model.Z_[np.unique(cand)], rowvar=False)
        BC = B @ C @ B.T
        return c * (BC[i, i] + BC[j, j] - 2 * BC[i, j])
    q = M[i, i] + M[j, j] - 2 * M[i, j]
    return -0.5 * np.log(np.clip(1 - c * q, 1e-12, None))


def choose(model: FeatureBT | MixedBT | None, Phi: np.ndarray, cand: np.ndarray, strategy: str,
           rng: np.random.Generator, explore: float = 0.1) -> int:
    """Index into `cand` of the next pair to ask.

    With probability `explore` (and always before a model exists) the pick is
    uniformly random. Pure greedy selection can lock onto whatever the current,
    possibly wrong, model finds interesting, and never see evidence against it.
    Ties in the score are broken at random.
    """
    if model is None or strategy == "random" or rng.random() < explore:
        return int(rng.integers(len(cand)))
    score = acquisition(model, Phi, cand, strategy)
    best = np.flatnonzero(score >= score.max() - 1e-12)
    return int(rng.choice(best))


def run_loop(Phi: np.ndarray, cand: np.ndarray, answer, strategy: str, rng: np.random.Generator,
             n_start: int = 20, n_total: int = 400, batch: int = 10, l2: float = 3.0,
             explore: float = 0.1, evaluate=None, make_model=None):
    """Simulate a labeling session.

    `answer(k)` returns the label for candidate pair k (a simulated instructor,
    or a stored real judgment). Start with `n_start` random pairs, then refit
    every `batch` labels and ask the best remaining candidate under `strategy`.
    `evaluate(model)` is called after each refit; returns (sizes, evaluations, chosen indices).
    `make_model()` builds the model that is refit (default: FeatureBT(l2)).
    """
    make_model = make_model or (lambda: FeatureBT(l2=l2))
    remaining = np.ones(len(cand), dtype=bool)
    chosen: list[int] = []
    labels: list[float] = []            # each pair is answered exactly once

    def ask(k: int) -> None:
        chosen.append(k)
        labels.append(answer(k))
        remaining[k] = False

    for k in rng.choice(len(cand), size=n_start, replace=False):
        ask(int(k))
    sizes, evals = [], []
    while True:
        model = make_model().fit(Phi, cand[chosen], np.array(labels))
        sizes.append(len(chosen))
        evals.append(evaluate(model) if evaluate else None)
        if len(chosen) >= n_total or not remaining.any():
            return sizes, evals, chosen
        # A batch is chosen from one model: the next refit happens after `batch` answers.
        for _ in range(min(batch, n_total - len(chosen), int(remaining.sum()))):
            left = np.flatnonzero(remaining)
            ask(int(left[choose(model, Phi, cand[left], strategy, rng, explore)]))
