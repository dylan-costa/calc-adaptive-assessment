"""Why is problem A harder than problem B? Shapley values for a pairwise prediction.

A model scores problems from features, delta_hat = f(phi). To explain the gap
f(a) - f(b) between two problems, treat groups of features as players in a
cooperative game. A coalition S "switches on" A's values for its groups and
leaves the rest at B's values:

    v(S) = f(a_S, b_rest) - f(b),        v(empty) = 0,  v(all) = f(a) - f(b).

The Shapley value of group j is its marginal contribution averaged over all
orders in which the groups could be switched on:

    phi_j = sum_{S not containing j} |S|! (k - |S| - 1)! / k! * [v(S + j) - v(S)].

It is the unique attribution satisfying efficiency (the phi_j add up to
f(a) - f(b)), symmetry, null player, and additivity (Section 12 of the notes).
With k groups, exact computation needs the 2^k hybrid problems, evaluated in
one vectorized call; `shapley_sampling` estimates it from random orders instead.
"""

from __future__ import annotations

from itertools import combinations
from math import factorial
from typing import Callable

import numpy as np

from calcdiff.features import FEATURE_NAMES

# Players: groups of features that mean one thing to an instructor. Grouping also
# sidesteps Section 6's problem of ridge spreading weight across near-duplicates.
GROUPS: dict[str, tuple[str, ...]] = {
    "size": ("size", "depth", "n_terms", "deriv_size"),
    "chain rule": ("n_chain", "max_chain_depth", "chain_in_product"),
    "quotient rule": ("n_quotient",),
    "product rule": ("n_product",),
    "function types": ("n_trig", "n_exp", "n_ln", "n_sqrt"),
    "tricky exponents": ("n_neg_exp", "n_frac_exp"),
    "powers and constants": ("n_power", "n_const_mult"),
}
GROUP_COLUMNS = [[FEATURE_NAMES.index(f) for f in fs] for fs in GROUPS.values()]


def _masks(k: int) -> np.ndarray:
    """All 2^k coalitions as a boolean (2^k, k) array; row r is the binary expansion of r."""
    r = np.arange(2 ** k)
    return ((r[:, None] >> np.arange(k)) & 1).astype(bool)


def shapley_exact(f: Callable[[np.ndarray], np.ndarray], a: np.ndarray, b: np.ndarray,
                  groups: list[list[int]] = GROUP_COLUMNS) -> np.ndarray:
    """Exact Shapley values of the groups for v(S) = f(a_S, b_rest) - f(b).

    `f` maps an (m, d) array of raw feature rows to m scores.
    """
    k = len(groups)
    masks = _masks(k)
    X = np.repeat(b[None, :], len(masks), axis=0)
    for j, cols in enumerate(groups):
        X[np.ix_(masks[:, j], cols)] = a[cols]        # coalition members take A's values
    v = f(X)
    v = v - v[0]                                       # row 0 is the empty coalition: v(empty) = 0
    size = masks.sum(1)
    weight = np.array([factorial(s) * factorial(k - s - 1) / factorial(k) for s in range(k)])
    phi = np.zeros(k)
    index = masks @ (1 << np.arange(k))                # row number of each coalition
    for j in range(k):
        without = ~masks[:, j]                         # every S not containing j
        with_j = index[without] | (1 << j)
        phi[j] = np.sum(weight[size[without]] * (v[with_j] - v[without]))
    return phi


def shapley_sampling(f: Callable[[np.ndarray], np.ndarray], a: np.ndarray, b: np.ndarray,
                     n_orders: int, rng: np.random.Generator,
                     groups: list[list[int]] = GROUP_COLUMNS) -> np.ndarray:
    """Unbiased Monte Carlo estimate: average marginal contributions over random orders."""
    k = len(groups)
    phi = np.zeros(k)
    for _ in range(n_orders):
        order = rng.permutation(k)
        X = np.repeat(b[None, :], k + 1, axis=0)       # row t: first t players of the order switched on
        for t, j in enumerate(order, start=1):
            X[t:, groups[j]] = a[groups[j]]
        v = f(X)
        phi[order] += np.diff(v)
    return phi / n_orders


def linear_shapley(theta_raw: np.ndarray, a: np.ndarray, b: np.ndarray,
                   groups: list[list[int]] = GROUP_COLUMNS) -> np.ndarray:
    """Closed form for a linear model f(x) = theta . x + c: phi_j = sum_{i in group j} theta_i (a_i - b_i)."""
    return np.array([theta_raw[cols] @ (a[cols] - b[cols]) for cols in groups])


def describe(phi: np.ndarray, names: list[str] | None = None, top: int = 2) -> str:
    """One sentence justifying the model's pick: what makes its choice harder, and what pushes back.

    phi explains f(a) - f(b). If that is negative the model picked B, so the
    sentence is written from B's side (phi is negated).
    """
    names = names or list(GROUPS)
    if phi.sum() < 0:
        phi = -phi
    order = np.argsort(-phi)
    up = [names[j] for j in order[:top] if phi[j] > 0.05]
    down = [names[j] for j in order[::-1][:1] if phi[j] < -0.05]
    if not up:
        return "No single aspect stands out; the model sees these as close."
    s = "Mostly the " + " and the ".join(up)
    if down:
        s += f", despite the {down[0]}"
    return s + "."
