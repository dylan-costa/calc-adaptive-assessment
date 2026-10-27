import numpy as np

from calcdiff.explain import (GROUP_COLUMNS, GROUPS, describe, linear_shapley, shapley_exact,
                              shapley_sampling)
from calcdiff.features import FEATURE_NAMES, feature_matrix
from calcdiff.generate import Sampler
from calcdiff.models.boosted_bt import BoostedBT
from calcdiff.synthetic import hidden_difficulty, random_pairs, simulate_judgments


def test_groups_cover_every_feature_exactly_once():
    cols = sorted(c for g in GROUP_COLUMNS for c in g)
    assert cols == list(range(len(FEATURE_NAMES)))


def setup():
    rng = np.random.default_rng(0)
    Phi = feature_matrix(p.tree for p in Sampler(seed=5).sample_unique(200))
    pairs = random_pairs(200, 2000, rng)
    y = simulate_judgments(hidden_difficulty(Phi, rng), pairs, rng)
    model = BoostedBT(n_trees=60, max_depth=3).fit(Phi, pairs, y)
    return Phi, model


def test_efficiency_on_a_nonlinear_model():
    Phi, model = setup()
    for i, j in [(0, 1), (5, 17), (40, 3)]:
        phi = shapley_exact(model.score, Phi[i], Phi[j])
        gap = model.score(Phi[[i]])[0] - model.score(Phi[[j]])[0]
        assert np.isclose(phi.sum(), gap)


def test_linear_model_matches_closed_form():
    Phi, _ = setup()
    theta = np.random.default_rng(1).normal(size=Phi.shape[1])
    f = lambda X: X @ theta + 3.0
    assert np.allclose(shapley_exact(f, Phi[0], Phi[1]), linear_shapley(theta, Phi[0], Phi[1]))


def test_null_player_and_symmetry():
    Phi, _ = setup()
    a, b = Phi[2].copy(), Phi[9].copy()
    quot = list(GROUPS).index("quotient rule")
    prod = list(GROUPS).index("product rule")
    # f ignores the quotient group entirely -> its Shapley value is 0
    f = lambda X: X[:, GROUP_COLUMNS[prod][0]] * 2.0 + X[:, 0]
    assert np.isclose(shapley_exact(f, a, b)[quot], 0.0)
    # two groups that enter symmetrically, with equal changes, get equal credit
    a[GROUP_COLUMNS[quot][0]], b[GROUP_COLUMNS[quot][0]] = 1, 0
    a[GROUP_COLUMNS[prod][0]], b[GROUP_COLUMNS[prod][0]] = 1, 0
    g = lambda X: X[:, GROUP_COLUMNS[quot][0]] * X[:, GROUP_COLUMNS[prod][0]]   # pure interaction
    phi = shapley_exact(g, a, b)
    assert np.isclose(phi[quot], phi[prod]) and np.isclose(phi[quot], 0.5)


def test_sampling_estimator_converges_to_exact():
    Phi, model = setup()
    exact = shapley_exact(model.score, Phi[3], Phi[30])
    rng = np.random.default_rng(2)
    est = shapley_sampling(model.score, Phi[3], Phi[30], 2000, rng)
    assert np.allclose(est, exact, atol=0.05)
    assert np.isclose(est.sum(), exact.sum())          # efficiency holds for every single order


def test_describe_reads_naturally():
    phi = np.zeros(len(GROUPS))
    phi[list(GROUPS).index("quotient rule")] = 1.2
    phi[list(GROUPS).index("chain rule")] = 0.6
    phi[list(GROUPS).index("size")] = -0.4
    assert describe(phi) == "Mostly the quotient rule and the chain rule, despite the size."
    # if the model prefers B, the sentence justifies B: same words for the mirrored game
    assert describe(-phi) == describe(phi)
