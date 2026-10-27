from fractions import Fraction as Q

import numpy as np

from calcdiff.expr import X, Add, Const, Div, Fn, Mul, Pow
from calcdiff.features import FEATURE_NAMES, Standardizer, feature_matrix, featurize
from calcdiff.generate import Sampler


def feats(tree):
    return dict(zip(FEATURE_NAMES, featurize(tree)))


def test_simple_power():
    f = feats(Pow(X, Q(3)))
    assert f["n_power"] == 1 and f["n_chain"] == 0 and f["n_terms"] == 1


def test_chain_inside_product_is_counted():
    # x^2 sin(x^3): the chain rule on sin(x^3) happens inside a product rule
    t = Mul((Pow(X, Q(2)), Fn("sin", Pow(X, Q(3)))))
    f = feats(t)
    assert f["n_product"] == 1 and f["n_chain"] == 1 and f["chain_in_product"] == 1
    # 5 sin(x^3): a constant multiple is not a product rule
    f = feats(Mul((Const(Q(5)), Fn("sin", Pow(X, Q(3))))))
    assert f["chain_in_product"] == 0


def test_exponent_features():
    t = Add((Pow(X, Q(-2)), Pow(X, Q(3, 2)), Div(X, Fn("ln", X))))
    f = feats(t)
    assert f["n_neg_exp"] == 1 and f["n_frac_exp"] == 1 and f["n_quotient"] == 1
    assert f["n_terms"] == 3 and f["n_ln"] == 1


def test_deterministic_and_shaped():
    trees = [p.tree for p in Sampler(seed=0).sample_unique(50)]
    A, B = feature_matrix(trees), feature_matrix(trees)
    assert A.shape == (50, len(FEATURE_NAMES))
    assert np.array_equal(A, B)


def test_standardizer():
    X_ = np.array([[1.0, 5.0], [3.0, 5.0], [5.0, 5.0]])
    Z = Standardizer().fit_transform(X_)
    assert np.allclose(Z[:, 0].mean(), 0) and np.allclose(Z[:, 0].std(), 1)
    assert np.all(Z[:, 1] == 0)  # constant column does not blow up
