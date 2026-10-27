"""Feature map phi: problems -> R^d.

Every model before Section 11 sees a problem *only* through this vector, so
this file decides what the models can possibly learn. A linear model
f(x) = theta . phi(x) can express exactly the span of these coordinates:
if difficulty depends on something not computable from phi, no amount of data
will recover it.

Features come from three places:
  - the expression tree itself (size, depth, exponents),
  - the derivative trace from Step 2 (which rules, how nested),
  - the derivative a student has to write down (its size).
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from calcdiff.expr import Add, Const, Div, Fn, Mul, Node, Pow, Var, depth, size, walk
from calcdiff.solver_trace import derive

FEATURE_NAMES: tuple[str, ...] = (
    "size",              # nodes in the problem's tree
    "depth",             # height of the tree
    "n_terms",           # top-level summands
    "deriv_size",        # nodes in the (unsimplified) derivative: expression swell
    "n_power",           # POWER rule applications
    "n_chain",           # CHAIN rule applications
    "max_chain_depth",   # longest stack of nested compositions
    "n_product",         # PRODUCT rule applications
    "n_quotient",        # QUOTIENT rule applications
    "n_const_mult",      # constant-multiple pulls
    "n_trig",            # derivatives of sin / cos / tan
    "n_exp",             # derivatives of e^u
    "n_ln",              # derivatives of ln u
    "n_sqrt",            # derivatives of sqrt u
    "n_neg_exp",         # powers with negative exponent (x^-2: students drop the sign)
    "n_frac_exp",        # powers with fractional exponent (x^(3/2))
    "chain_in_product",  # chain rules nested inside a product/quotient (interaction)
)


def featurize(tree: Node) -> np.ndarray:
    """phi(tree) as a float vector, ordered as FEATURE_NAMES."""
    d, tr = derive(tree)
    c = tr.counts()
    pows = [n for n in walk(tree) if isinstance(n, Pow)]
    f = {
        "size": size(tree),
        "depth": depth(tree),
        "n_terms": len(tree.terms) if isinstance(tree, Add) else 1,
        "deriv_size": size(d),
        "n_power": c["POWER"],
        "n_chain": c["CHAIN"],
        "max_chain_depth": tr.max_chain_depth,
        "n_product": c["PRODUCT"],
        "n_quotient": c["QUOTIENT"],
        "n_const_mult": c["CONST_MULT"],
        "n_trig": c["FN_sin"] + c["FN_cos"] + c["FN_tan"],
        "n_exp": c["FN_exp"],
        "n_ln": c["FN_ln"],
        "n_sqrt": c["FN_sqrt"],
        "n_neg_exp": sum(p.exp < 0 for p in pows),
        "n_frac_exp": sum(p.exp.denominator != 1 for p in pows),
        "chain_in_product": _chains_under_product(tree, False),
    }
    return np.array([f[name] for name in FEATURE_NAMES], dtype=float)


def feature_matrix(trees: Iterable[Node]) -> np.ndarray:
    """Stack phi(x) for each tree into an (n, d) matrix."""
    return np.vstack([featurize(t) for t in trees])


def _is_chain(node: Node) -> bool:
    """Does differentiating this node invoke the chain rule?"""
    return (isinstance(node, Fn) and not isinstance(node.arg, Var)) or \
        (isinstance(node, Pow) and not isinstance(node.base, Var))


def _chains_under_product(node: Node, under: bool) -> int:
    """Count chain-rule nodes that have a product or quotient above them.

    A linear model in n_chain and n_product can only add their effects. If a
    chain rule *inside* a product rule is harder than the two separately, the
    model needs this interaction term to say so.
    """
    here = int(under and _is_chain(node))
    under = under or (isinstance(node, (Mul, Div)) and _has_two_nonconst(node))
    return here + sum(_chains_under_product(ch, under) for ch in node.children())


def _has_two_nonconst(node: Node) -> bool:
    if isinstance(node, Div):
        return True
    return sum(not isinstance(f, Const) for f in node.factors) >= 2  # type: ignore[union-attr]


class Standardizer:
    """z = (phi - mean) / std, with mean and std fit on training data only.

    Ridge penalties ||theta||^2 are not scale invariant: a feature measured in
    "nodes" (size ~ 15) and one in "counts" (n_quotient ~ 0.1) would be
    penalized very differently. Standardizing puts them on equal footing.
    """

    def fit(self, X: np.ndarray) -> "Standardizer":
        self.mean_ = X.mean(axis=0)
        sd = X.std(axis=0)
        self.std_ = np.where(sd > 0, sd, 1.0)  # constant columns stay constant (zero)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
