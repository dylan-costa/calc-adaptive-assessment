from fractions import Fraction as Q

import sympy as sp

from calcdiff.expr import X, X_SYM, Add, Const, Div, Fn, Mul, Pow, depth, size, to_latex, to_sympy


def test_to_sympy_matches_math():
    # 3 sin(x^2 + 1) / x
    t = Div(Mul((Const(Q(3)), Fn("sin", Add((Pow(X, Q(2)), Const(Q(1))))))), X)
    assert sp.simplify(to_sympy(t) - 3 * sp.sin(X_SYM**2 + 1) / X_SYM) == 0


def test_latex_preserves_written_form():
    assert to_latex(Div(X, Add((X, Const(Q(1)))))) == r"\frac{x}{x + 1}"
    assert to_latex(Pow(Fn("sin", X), Q(2))) == r"\sin^{2} x"
    assert to_latex(Fn("exp", Pow(X, Q(2)))) == "e^{x^{2}}"
    assert to_latex(Add((X, Mul((Const(Q(-3)), Pow(X, Q(2))))))) == "x - 3 x^{2}"
    assert to_latex(Mul((Const(Q(-1)), Fn("cos", X)))) == r"-\cos x"
    assert to_latex(Pow(Add((X, Const(Q(1)))), Q(1, 2))) == r"\left(x + 1\right)^{1/2}"


def test_size_and_depth():
    t = Mul((Pow(X, Q(2)), Fn("sin", X)))
    assert size(t) == 5  # Mul, Pow, x, sin, x
    assert depth(t) == 3
