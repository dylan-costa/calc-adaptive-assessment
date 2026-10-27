import numpy as np

from calcdiff.evaluate import pairs_within, pairwise_accuracy, split_problems
from calcdiff.features import FEATURE_NAMES, feature_matrix
from calcdiff.generate import Sampler
from calcdiff.models.feature_bt import FeatureBT, cv_l2
from calcdiff.synthetic import SIMULATED_INSTRUCTOR, hidden_difficulty, random_pairs, simulate_judgments


def setup(n=300, m=6000, seed=0):
    rng = np.random.default_rng(seed)
    Phi = feature_matrix(p.tree for p in Sampler(seed=seed).sample_unique(n))
    delta = hidden_difficulty(Phi, rng)
    pairs = random_pairs(len(Phi), m, rng)
    y = simulate_judgments(delta, pairs, rng)
    return rng, Phi, delta, pairs, y


def test_generalizes_to_unseen_problems():
    rng, Phi, delta, pairs, y = setup()
    tr_items, te_items = split_problems(len(Phi), 0.3, rng)
    tr, te = pairs_within(pairs, tr_items), pairs_within(pairs, te_items)
    model = FeatureBT(l2=1.0).fit(Phi, pairs[tr], y[tr])
    acc = pairwise_accuracy(model.score(Phi), pairs[te], y[te])
    oracle = pairwise_accuracy(delta, pairs[te], y[te])
    assert acc > 0.7 and oracle - acc < 0.04


def test_recovers_planted_weights():
    _, Phi, _, pairs, y = setup(n=400, m=20000, seed=1)
    model = FeatureBT(l2=0.1).fit(Phi, pairs, y)
    w = model.raw_weights()
    planted = np.array([SIMULATED_INSTRUCTOR.get(n, 0.0) for n in FEATURE_NAMES])
    i = FEATURE_NAMES.index("n_quotient")
    assert abs(w[i] - planted[i]) < 3 * model.raw_weight_se()[i]


def test_scaler_ignores_unseen_problems():
    _, Phi, _, pairs, y = setup(n=100, m=500)
    sub = pairs[(pairs < 50).all(axis=1)]
    model = FeatureBT().fit(Phi, sub, y[(pairs < 50).all(axis=1)])
    assert np.allclose(model.scaler_.mean_, Phi[np.unique(sub)].mean(axis=0))


def test_cross_validation_prefers_some_regularization_with_little_data():
    rng, Phi, _, pairs, y = setup(n=150, m=300, seed=3)
    grid = np.array([1e-3, 1e-1, 1.0, 10.0, 100.0])
    best, losses = cv_l2(Phi, pairs, y, np.arange(len(Phi)), grid, rng)
    assert best > 1e-3 and losses.shape == (5,)


def test_problem_bootstrap_widens_intervals_when_problems_have_shared_noise():
    from calcdiff.models.feature_bt import problem_bootstrap
    rng, Phi, _, pairs, y = setup(n=200, m=3000, seed=4)
    model = FeatureBT(l2=1.0).fit(Phi, pairs, y)
    boot = problem_bootstrap(Phi, pairs, y, 1.0, rng, n_boot=60)
    ratio = np.median(boot.std(axis=0) / model.raw_weight_se())
    assert ratio > 1.3  # the planted per-problem noise makes Laplace SEs too small
