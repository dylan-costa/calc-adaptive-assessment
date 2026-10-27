import numpy as np
import pytest

from calcdiff.models.bt import (fit_gradient_descent, fit_newton, grad_hess, mle_exists, nll,
                                one_hot_design)
from calcdiff.synthetic import random_pairs, simulate_judgments


def small_problem(n=30, m=600, seed=0, with_pairs=False):
    rng = np.random.default_rng(seed)
    delta = rng.normal(0, 1.5, n)
    delta -= delta.mean()
    pairs = random_pairs(n, m, rng)
    y = simulate_judgments(delta, pairs, rng)
    if with_pairs:
        return delta, pairs, y
    return delta, one_hot_design(pairs, n), y


def test_gradient_matches_finite_differences():
    _, D, y = small_problem()
    theta = np.random.default_rng(1).normal(size=D.shape[1])
    g, _ = grad_hess(theta, D, y, l2=0.3)
    eps = 1e-6
    fd = np.array([(nll(theta + eps * e, D, y, 0.3) - nll(theta - eps * e, D, y, 0.3)) / (2 * eps)
                   for e in np.eye(len(theta))])
    assert np.allclose(g, fd, atol=1e-5)


def test_hessian_is_weighted_graph_laplacian():
    _, D, y = small_problem()
    _, H = grad_hess(np.zeros(D.shape[1]), D, y)
    assert np.allclose(H, H.T)
    assert np.allclose(H @ np.ones(len(H)), 0)          # rows sum to zero
    assert np.all(np.diag(H) >= 0) and np.all(H - np.diag(np.diag(H)) <= 1e-12)  # off-diagonals <= 0
    eig = np.linalg.eigvalsh(H)
    assert eig[0] > -1e-10 and eig[1] > 1e-6             # PSD, kernel = span(1) (graph connected)


def test_mle_existence_condition():
    # seed 0 happens to contain an item that lost every comparison
    _, pairs, y = small_problem(with_pairs=True)
    assert not mle_exists(pairs, y, 30)
    _, D, y = small_problem()
    assert fit_newton(D, y).theta.min() < -20       # that item runs off to -infinity
    _, pairs, y = small_problem(m=3000, with_pairs=True)
    assert mle_exists(pairs, y, 30)


def test_newton_converges_fast_and_stays_in_gauge():
    _, D, y = small_problem()
    fit = fit_newton(D, y, l2=0.01)  # a little ridge makes the optimum exist
    assert fit.converged and fit.n_iter <= 10
    assert abs(fit.theta.sum()) < 1e-8


def test_recovers_planted_difficulties():
    delta, pairs, y = small_problem(n=40, m=8000, seed=2, with_pairs=True)
    assert mle_exists(pairs, y, 40)
    fit = fit_newton(one_hot_design(pairs, 40), y)
    assert np.sqrt(np.mean((fit.theta - delta) ** 2)) < 0.25
    # Laplace intervals should cover the truth about 95% of the time
    se = np.sqrt(np.diag(fit.covariance()))
    coverage = np.mean(np.abs(fit.theta - delta) <= 1.96 * se)
    assert coverage >= 0.8


def test_gradient_descent_reaches_newton_optimum():
    _, D, y = small_problem()
    a = fit_newton(D, y, l2=0.1)
    b = fit_gradient_descent(D, y, l2=0.1, n_iter=5000)
    assert np.allclose(a.theta, b.theta, atol=1e-4)


def test_perfect_separation_diverges_without_ridge():
    # item 0 beats items 1 and 2 every time; 1 vs 2 is mixed
    pairs = np.array([[0, 1]] * 5 + [[0, 2]] * 5 + [[1, 2]] * 4)
    y = np.array([1.0] * 10 + [1, 0, 1, 0])
    D = one_hot_design(pairs, 3)
    assert not mle_exists(pairs, y, 3)
    # The loss decreases toward its infimum forever, gradients vanish along the
    # way, and the stopping rule declares convergence at a meaningless theta.
    free = fit_newton(D, y, max_iter=60)
    assert free.theta[0] - free.theta[1] > 10
    ridge = fit_newton(D, y, l2=1.0)
    assert ridge.converged and ridge.theta[0] - ridge.theta[1] < 5


def test_centered_covariance_removes_prior_only_direction():
    _, D, y = small_problem()
    fit = fit_newton(D, y, l2=0.1)
    naive, centered = fit.covariance(), fit.covariance(centered=True)
    ones = np.ones(len(naive))
    assert np.isclose(ones @ naive @ ones / len(ones), 1 / 0.1)  # prior variance 1/l2 along 1/sqrt(n)
    assert np.allclose(centered @ ones, 0)
    assert np.trace(centered) < np.trace(naive)


def test_tie_label_is_half_a_win_each_way():
    # one tie between two items: the optimum is theta_0 = theta_1
    D = one_hot_design(np.array([[0, 1]]), 2)
    fit = fit_newton(D, np.array([0.5]), l2=0.1)
    assert np.allclose(fit.theta, 0)
    # a tie counts the same as one win plus one loss
    D2 = one_hot_design(np.array([[0, 1], [0, 1]]), 2)
    theta = np.array([0.7, -0.2])
    assert np.isclose(2 * nll(theta, D, np.array([0.5])), nll(theta, D2, np.array([1.0, 0.0])))
