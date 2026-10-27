import numpy as np

from calcdiff.evaluate import log_loss, pairs_touching, split_problems
from calcdiff.features import feature_matrix
from calcdiff.generate import Sampler
from calcdiff.models.boosted_bt import BoostedBT, leaf_cut, problem_grad_hess
from calcdiff.models.bt import grad_hess, nll, one_hot_design
from calcdiff.models.feature_bt import FeatureBT
from calcdiff.synthetic import random_pairs, simulate_judgments


def toy(n=40, m=400, seed=0):
    rng = np.random.default_rng(seed)
    pairs = random_pairs(n, m, rng)
    F = rng.normal(0, 1, n)
    y = simulate_judgments(rng.normal(0, 1.5, n), pairs, rng)
    return rng, pairs, F, y


def test_problem_gradient_is_the_one_hot_gradient():
    _, pairs, F, y = toy()
    g, h, _ = problem_grad_hess(F, pairs, y)
    D = one_hot_design(pairs, len(F))
    g_ref, H_ref = grad_hess(F, D, y)
    assert np.allclose(g, g_ref) and np.allclose(h, np.diag(H_ref))


def test_leaf_curvature_is_the_laplacian_cut():
    rng, pairs, F, y = toy()
    _, _, w = problem_grad_hess(F, pairs, y)
    leaf_of = rng.integers(0, 4, len(F))
    cut = leaf_cut(leaf_of, pairs, w, 4)
    _, H = grad_hess(F, one_hot_design(pairs, len(F)), y)      # = weighted Laplacian
    for leaf in range(4):
        one = (leaf_of == leaf).astype(float)
        assert np.isclose(cut[leaf], one @ H @ one)


def test_leaf_value_is_exact_newton_step_for_one_leaf():
    rng, pairs, F, y = toy()
    g, _, w = problem_grad_hess(F, pairs, y)
    R = rng.random(len(F)) < 0.3
    gamma = -g[R].sum() / leaf_cut(R.astype(int), pairs, w, 2)[1]
    D = one_hot_design(pairs, len(F))
    phi = lambda s: nll(F + s * R, D, y)
    eps = 1e-5
    d1 = (phi(eps) - phi(-eps)) / (2 * eps)
    d2 = (phi(eps) - 2 * phi(0) + phi(-eps)) / eps**2
    assert np.isclose(gamma, -d1 / d2, rtol=1e-3)


def test_training_loss_decreases():
    rng = np.random.default_rng(1)
    Phi = feature_matrix(p.tree for p in Sampler(seed=1).sample_unique(150))
    pairs = random_pairs(150, 1500, rng)
    y = simulate_judgments(Phi[:, 3] * 0.1 - 1, pairs, rng)
    m = BoostedBT(n_trees=50, learning_rate=0.1).fit(Phi, pairs, y)
    assert np.all(np.diff(m.train_loss_) < 1e-9)


def test_invariant_to_monotone_feature_transforms():
    rng = np.random.default_rng(2)
    Phi = feature_matrix(p.tree for p in Sampler(seed=2).sample_unique(120))
    pairs = random_pairs(120, 1000, rng)
    y = simulate_judgments(0.05 * Phi[:, 0], pairs, rng)
    warped = Phi.copy()
    warped[:, 0] = np.exp(Phi[:, 0] / 5)          # strictly increasing transform of "size"
    a = BoostedBT(n_trees=20).fit(Phi, pairs, y).score(Phi)
    b = BoostedBT(n_trees=20).fit(warped, pairs, y).score(warped)
    assert np.allclose(a, b)


def test_beats_linear_model_on_an_interaction_it_was_not_given():
    rng = np.random.default_rng(3)
    Phi = feature_matrix(p.tree for p in Sampler(seed=3).sample_unique(500))
    chain, quot = Phi[:, 5], Phi[:, 8]
    # chain rules make a problem harder with a quotient present, easier without:
    # no linear function of (n_chain, n_quotient) can represent this
    delta = np.where(quot > 0, 1.0, -1.0) * chain
    tr, te = split_problems(500, 0.3, rng)
    pairs = random_pairs(500, 8000, rng)
    y = simulate_judgments(delta, pairs, rng)
    test = pairs_touching(pairs, te)
    lin = FeatureBT(l2=1.0, columns=[5, 8]).fit(Phi, pairs[~test], y[~test])
    boost = BoostedBT(n_trees=150, max_depth=2).fit(Phi[:, [5, 8]], pairs[~test], y[~test])
    ll_lin = log_loss(lin.score(Phi), pairs[test], y[test])
    ll_boost = log_loss(boost.score(Phi[:, [5, 8]]), pairs[test], y[test])
    ll_oracle = log_loss(delta, pairs[test], y[test])
    assert ll_boost < ll_lin - 0.02 and ll_boost < ll_oracle + 0.01
