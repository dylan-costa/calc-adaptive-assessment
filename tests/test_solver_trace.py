from fractions import Fraction as Q

import sympy as sp

from calcdiff.check import numerically_equal
from calcdiff.expr import X, X_SYM, Add, Const, Div, Fn, Mul, Pow, to_latex, to_sympy
from calcdiff.generate import Sampler
from calcdiff.solver_trace import derive


def rules(tree):
    return [s.rule for s in derive(tree)[1].steps]


def test_power_rule():
    d, _ = derive(Pow(X, Q(3)))
    assert to_latex(d) == "3 x^{2}"
    assert rules(Pow(X, Q(3))) == ["POWER"]


def test_chain_rule_trace_and_nesting():
    # sin(e^(x^2)): FN_sin, CHAIN, FN_exp, CHAIN, POWER
    t = Fn("sin", Fn("exp", Pow(X, Q(2))))
    d, tr = derive(t)
    assert rules(t) == ["FN_sin", "CHAIN", "FN_exp", "CHAIN", "POWER"]
    assert tr.max_chain_depth == 2
    assert to_latex(d) == r"2 \cos\left(e^{x^{2}}\right) e^{x^{2}} x"


def test_product_and_quotient():
    prod = Mul((Pow(X, Q(2)), Fn("sin", X)))
    assert rules(prod) == ["PRODUCT", "POWER", "FN_sin"]
    quot = Div(X, Add((X, Const(Q(1)))))
    assert rules(quot) == ["QUOTIENT", "VAR", "SUM", "VAR", "CONST"]
    d, _ = derive(quot)
    assert sp.simplify(to_sympy(d) - sp.diff(to_sympy(quot), X_SYM)) == 0


def test_constant_multiple_is_pulled_out():
    t = Mul((Const(Q(5)), Fn("cos", X)))
    d, _ = derive(t)
    assert rules(t) == ["CONST_MULT", "FN_cos"]
    assert to_latex(d) == r"-5 \sin x"


def test_numeric_check_rejects_wrong_answers():
    assert numerically_equal(sp.sin(X_SYM), sp.cos(X_SYM)) is False
    assert numerically_equal(sp.exp((X_SYM + 9) ** 3), 0 * X_SYM) is None
    assert numerically_equal(sp.sin(2 * X_SYM), 2 * sp.sin(X_SYM) * sp.cos(X_SYM))


def test_matches_sympy_on_generated_problems():
    inconclusive = 0
    for p in Sampler(seed=5).sample_unique(500):
        d, _ = derive(p.tree)
        verdict = numerically_equal(to_sympy(d), sp.diff(to_sympy(p.tree), X_SYM))
        assert verdict is not False, p.latex
        inconclusive += verdict is None
    assert inconclusive <= 10  # only the e^(huge) problems
