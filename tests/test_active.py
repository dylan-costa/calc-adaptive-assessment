import numpy as np
from scipy.special import expit

from calcdiff.active import acquisition, all_pairs, choose
from calcdiff.features import feature_matrix
from calcdiff.generate import Sampler
from calcdiff.models.bt import grad_hess
from calcdiff.models.feature_bt import FeatureBT
from calcdiff.synthetic import hidden_difficulty, random_pairs, simulate_judgments


def fitted(n=80, m=200, seed=0):
    rng = np.random.default_rng(seed)
    Phi = feature_matrix(p.tree for p in Sampler(seed=seed).sample_unique(n))
    delta = hidden_difficulty(Phi, rng)
    pairs = random_pairs(n, m, rng)
    y = simulate_judgments(delta, pairs, rng)
    return rng, Phi, pairs, y, FeatureBT(l2=1.0).fit(Phi, pairs, y)


def test_all_pairs_counts():
    c = all_pairs(np.arange(5))
    assert len(c) == 10 and np.all(c[:, 0] < c[:, 1])


def test_information_gain_is_half_log_det_ratio():
    """Matrix determinant lemma: adding one pair's Fisher information to H."""
    _, Phi, pairs, y, model = fitted()
    cand = all_pairs(np.arange(20))
    score = acquisition(model, Phi, cand, "information")
    Z = model.scaler_.transform(Phi)
    _, H = grad_hess(model.theta_, Z[pairs[:, 0]] - Z[pairs[:, 1]], y, model.l2)
    for k in (0, 7, 33):
        d = Z[cand[k, 0]] - Z[cand[k, 1]]
        p = expit(d @ model.theta_)
        H_new = H + p * (1 - p) * np.outer(d, d)
        ratio = np.linalg.slogdet(H_new)[1] - np.linalg.slogdet(H)[1]
        assert np.isclose(score[k], 0.5 * ratio)


def test_uncertainty_prefers_close_calls():
    _, Phi, _, _, model = fitted()
    cand = all_pairs(np.arange(40))
    s = model.score(Phi)
    k = np.argmax(acquisition(model, Phi, cand, "uncertainty"))
    gaps = np.abs(s[cand[:, 0]] - s[cand[:, 1]])
    assert np.isclose(gaps[k], gaps.min())


def test_choose_explores_and_breaks_ties():
    rng, Phi, _, _, model = fitted()
    cand = all_pairs(np.arange(30))
    picks = {choose(None, Phi, cand, "information", rng) for _ in range(50)}
    assert len(picks) > 30                      # no model yet: uniform
    greedy = [choose(model, Phi, cand, "information", rng, explore=0.0) for _ in range(20)]
    assert len(set(greedy)) == 1                # deterministic argmax


def test_loop_asks_each_pair_once_and_never_relabels():
    from calcdiff.active import run_loop
    rng, Phi, _, _, _ = fitted(n=40)
    cand = all_pairs(np.arange(40))
    calls = []

    def answer(k):
        calls.append(k)
        return float(rng.random() < 0.5)

    sizes, _, chosen = run_loop(Phi, cand, answer, "information", rng, n_start=10, n_total=50, batch=10)
    assert sizes == [10, 20, 30, 40, 50]
    assert len(set(chosen)) == 50 and sorted(calls) == sorted(chosen)


def test_mixed_information_is_log_det_ratio_of_theta_block():
    from calcdiff.models.mixed_bt import MixedBT
    _, Phi, pairs, y, _ = fitted(n=40, m=150)
    model = MixedBT(l2=1.0, tau=0.5).fit(Phi, pairs, y)
    cand = all_pairs(np.arange(15))
    score = acquisition(model, Phi, cand, "information")
    d, n = model.d_, len(Phi)
    Sigma = model.fit_.covariance()
    s = model.pool_score()
    for k in (0, 9, 40):
        i, j = cand[k]
        a = np.concatenate([model.Z_[i] - model.Z_[j], np.eye(n)[i] - np.eye(n)[j]])
        p = expit(s[i] - s[j])
        H_new = np.linalg.inv(Sigma) + p * (1 - p) * np.outer(a, a)
        Sigma_new = np.linalg.inv(H_new)
        gain = 0.5 * (np.linalg.slogdet(Sigma[:d, :d])[1] - np.linalg.slogdet(Sigma_new[:d, :d])[1])
        assert np.isclose(score[k], gain, rtol=1e-5)


def test_mixed_reduces_to_feature_model_when_tau_is_tiny():
    from calcdiff.models.mixed_bt import MixedBT
    _, Phi, pairs, y, _ = fitted(n=40, m=150)
    mixed = MixedBT(l2=1.0, tau=1e-4).fit(Phi, pairs, y)
    plain = FeatureBT(l2=1.0).fit(Phi, pairs, y)
    cand = all_pairs(np.arange(15))
    assert np.allclose(mixed.theta_, plain.theta_, atol=1e-4)
    assert np.allclose(acquisition(mixed, Phi, cand, "information"),
                       acquisition(plain, Phi, cand, "information"), rtol=1e-3, atol=1e-6)
