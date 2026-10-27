import numpy as np

from calcdiff.evaluate import (bayes_ceiling, bootstrap_ci, log_loss, pairs_within,
                               pairwise_accuracy, split_problems)
from calcdiff.synthetic import random_pairs, simulate_judgments


def test_split_is_a_partition():
    tr, te = split_problems(100, 0.3, np.random.default_rng(0))
    assert len(te) == 30 and len(np.intersect1d(tr, te)) == 0
    assert np.array_equal(np.sort(np.concatenate([tr, te])), np.arange(100))


def test_pairs_within_requires_both_endpoints():
    pairs = np.array([[0, 1], [1, 2], [2, 3]])
    assert pairs_within(pairs, np.array([0, 1, 2])).tolist() == [True, True, False]


def test_accuracy_conventions():
    pairs = np.array([[0, 1], [0, 1], [1, 2], [0, 2]])
    y = np.array([1.0, 0.0, 1.0, 0.5])               # the tie is skipped
    scores = np.array([2.0, 1.0, 0.0])
    assert pairwise_accuracy(scores, pairs, y) == 2 / 3
    assert pairwise_accuracy(np.zeros(3), pairs, y) == 0.5  # a constant model gets 1/2


def test_oracle_matches_ceiling_on_average():
    rng = np.random.default_rng(1)
    delta = rng.normal(0, 1.5, 200)
    pairs = random_pairs(200, 50_000, rng)
    y = simulate_judgments(delta, pairs, rng)
    assert abs(pairwise_accuracy(delta, pairs, y) - bayes_ceiling(delta, pairs)) < 0.01
    assert log_loss(delta, pairs, y) < log_loss(np.zeros(200), pairs, y)


def test_problem_bootstrap_is_wider_than_naive_interval():
    rng = np.random.default_rng(2)
    n = 60
    delta = rng.normal(0, 1.5, n)
    pairs = random_pairs(n, 3000, rng)
    y = simulate_judgments(delta, pairs, rng)
    noisy = delta + rng.normal(0, 1.0, n)            # a model with per-problem errors
    acc = pairwise_accuracy(noisy, pairs, y)
    lo, hi = bootstrap_ci(lambda w: pairwise_accuracy(noisy, pairs, y, w), pairs, np.arange(n), rng, 400)
    naive_half = 1.96 * np.sqrt(acc * (1 - acc) / len(pairs))
    assert lo < acc < hi
    assert (hi - lo) / 2 > 1.5 * naive_half
