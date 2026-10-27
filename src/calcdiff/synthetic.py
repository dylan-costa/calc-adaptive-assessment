"""A simulated instructor, for testing the learning pipeline before real labels exist.

We posit a hidden "true" difficulty

    delta(x) = w . phi(x) + tau * eps(x),   eps(x) ~ N(0, 1) i.i.d. per problem,

and let the simulated instructor answer "which is harder?" by Bradley-Terry:
P(i harder than j) = sigma(delta_i - delta_j). The idiosyncratic term eps is
difficulty the features cannot see, so even a perfect feature model cannot
recover delta exactly: it sets a ceiling, just as real labels will.

This is a sanity check, not evidence. Real instructors will not follow these
weights; the point is that if the code cannot recover a truth we planted, it
certainly will not recover one we did not.
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit

from calcdiff.features import FEATURE_NAMES

# The planted weights, in units of BT logits per unit of each (raw) feature.
SIMULATED_INSTRUCTOR: dict[str, float] = {
    "n_chain": 0.30,
    "max_chain_depth": 0.35,
    "n_product": 0.35,
    "n_quotient": 0.55,
    "chain_in_product": 0.30,
    "n_frac_exp": 0.20,
    "n_neg_exp": 0.15,
    "n_trig": 0.10,
    "n_ln": 0.10,
    "n_terms": 0.10,
    "deriv_size": 0.012,
}
IDIOSYNCRATIC_SD = 0.5


def hidden_difficulty(Phi: np.ndarray, rng: np.random.Generator,
                      weights: dict[str, float] = SIMULATED_INSTRUCTOR,
                      tau: float = IDIOSYNCRATIC_SD) -> np.ndarray:
    """delta = w . phi + tau * eps, centered to mean zero (BT only sees differences)."""
    w = np.array([weights.get(n, 0.0) for n in FEATURE_NAMES])
    delta = Phi @ w + tau * rng.standard_normal(len(Phi))
    return delta - delta.mean()


def random_pairs(n_items: int, n_pairs: int, rng: np.random.Generator) -> np.ndarray:
    """n_pairs uniformly random unordered pairs i != j (with replacement across pairs)."""
    i = rng.integers(0, n_items, n_pairs)
    j = (i + rng.integers(1, n_items, n_pairs)) % n_items  # uniform over j != i
    return np.column_stack([i, j])


def simulate_judgments(delta: np.ndarray, pairs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """y_k = 1 if the simulated instructor judges pairs[k, 0] harder."""
    p = expit(delta[pairs[:, 0]] - delta[pairs[:, 1]])
    return (rng.random(len(pairs)) < p).astype(float)
