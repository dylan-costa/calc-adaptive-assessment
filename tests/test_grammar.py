import math
from fractions import Fraction as Q

import numpy as np
import pytest
import sympy as sp

from calcdiff.expr import X_SYM, size, to_sympy
from calcdiff.generate import GrammarConfig, Sampler
from calcdiff.generate.grammar import is_natural


def test_reproducible_with_seed():
    a = [p.latex for p in Sampler(seed=7).sample_unique(30)]
    b = [p.latex for p in Sampler(seed=7).sample_unique(30)]
    assert a == b


def test_probabilities_must_sum_to_one():
    with pytest.raises(ValueError):
        GrammarConfig(p_leaf=0.9)


def test_samples_are_valid_problems():
    for p in Sampler(seed=1).sample_unique(300):
        assert is_natural(p.tree)
        assert size(p.tree) <= 30
        assert sp.diff(to_sympy(p.tree), X_SYM) != 0


def test_log_prob_of_simplest_problem_is_exact():
    """p("x^2") = P(k=1) P(LEAF) P(n=2) P(coef=1) P(no const term)."""
    cfg = GrammarConfig()
    exp_w = dict(cfg.exponent_weights)
    expected = (
        math.log(cfg.term_count_probs[0] / sum(cfg.term_count_probs))
        + math.log(cfg.p_leaf)
        + math.log(exp_w[Q(2)] / sum(exp_w.values()))
        + math.log(1 - cfg.p_coef)
        + math.log(1 - cfg.p_const_term)
    )
    s = Sampler(cfg, seed=0)
    hits = [p for p in (s.sample_raw() for _ in range(5000)) if p.latex == "x^{2}"]
    assert hits
    assert all(math.isclose(p.log_prob, expected) for p in hits)
    # and the empirical frequency agrees with exp(log p) (binomial 5-sigma band)
    n, p_true = 5000, math.exp(expected)
    assert abs(len(hits) - n * p_true) < 5 * math.sqrt(n * p_true * (1 - p_true))


def test_branching_process_mean_progeny():
    """Untruncated subcritical Galton-Watson tree: E[N] = 1 / (1 - m)."""
    cfg = GrammarConfig(max_depth=None)
    s = Sampler(cfg, seed=3)
    ns = np.array([s.sample_factor_tree().n_expansions for _ in range(20000)])
    m = cfg.mean_offspring
    var_offspring = cfg.offspring_dist @ np.array([0, 1, 4]) - m**2
    var_n = var_offspring / (1 - m) ** 3
    se = math.sqrt(var_n / len(ns))
    assert abs(ns.mean() - 1 / (1 - m)) < 5 * se


def test_likelihood_ratio_matches_direct_log_probs():
    """Self-normalized importance sampling: reweighting samples from p by q/p
    recovers expectations under q. Checked on E_q[# PROD nodes]."""
    from calcdiff.generate.grammar import log_likelihood_ratio

    p = GrammarConfig()
    q = GrammarConfig(p_leaf=0.40, p_fn=0.28, p_powf=0.10, p_prod=0.14, p_quot=0.08)
    sp_, sq = Sampler(p, seed=11), Sampler(q, seed=12)
    from_p = [sp_.sample_raw() for _ in range(20000)]
    from_q = [sq.sample_raw() for _ in range(20000)]
    logw = np.array([log_likelihood_ratio(x, p, q) for x in from_p])
    w = np.exp(logw - logw.max())
    w /= w.sum()
    f_p = np.array([x.rule_counts["PROD"] for x in from_p])
    f_q = np.array([x.rule_counts["PROD"] for x in from_q])
    est = (w * f_p).sum()
    assert abs(est - f_q.mean()) < 0.1 * f_q.mean()


def test_stacked_coefficients_are_unnatural():
    from fractions import Fraction as F

    from calcdiff.expr import X, Add, Const, Fn, Mul, Pow
    c = lambda v: Const(F(v))
    powf = Pow(Add((Pow(X, F(2)), c(9))), F(5))                    # (x^2 + 9)^5
    # 4 (x^2+9)^5 * 7x  -- the example from the label pool
    assert not is_natural(Mul((c(4), Mul((powf, Mul((c(7), X)))))))
    # 3 * 9x * u
    assert not is_natural(Mul((c(3), Mul((Mul((c(9), X)), powf)))))
    # 9x sin(x^-2) x  -- two monomials in one product
    assert not is_natural(Mul((Mul((c(9), X)), Fn("sin", Pow(X, F(-2))), X)))
    # still fine: a single leading coefficient, one monomial
    assert is_natural(Mul((c(4), Mul((Pow(X, F(2)), Fn("sin", X))))))
    assert is_natural(Mul((Mul((c(4), Pow(X, F(2)))), powf)))       # 4x^2 (x^2+9)^5
    assert is_natural(Fn("sin", Mul((c(-3), Pow(X, F(4))))))       # sin(-3x^4)


def test_generated_problems_never_stack_coefficients():
    for p in Sampler(seed=9).sample_unique(2000):
        assert r"\cdot" not in p.latex, p.latex
