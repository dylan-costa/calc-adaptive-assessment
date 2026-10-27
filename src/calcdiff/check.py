"""Checking that two expressions denote the same function.

`sympy.simplify(a - b) == 0` is slow and can fail to prove a true identity.
Instead we evaluate both sides at random *complex* points. Both sides are
analytic away from a measure-zero set (poles and the branch cuts of ln, sqrt,
fractional powers), so by the identity theorem, if they agree on a set with an
accumulation point in a connected region, they agree on all of it. A random
point lands on a coincidental agreement with probability zero.
"""

from __future__ import annotations

import numpy as np
import sympy as sp

from calcdiff.expr import X_SYM


def numerically_equal(a: sp.Expr, b: sp.Expr, *, trials: int = 8, seed: int = 0,
                      rtol: float = 1e-8) -> bool | None:
    """Do a and b agree at `trials` random points near the positive real axis?

    Returns True or False, or None when the test is inconclusive: too few
    points where both sides are finite in double precision. That happens for
    problems like e^((x+9)^3) ~ e^1000, which overflow everywhere near the
    real axis. (Arbitrary precision does not rescue these cheaply: cos of a
    number near 10^400 needs ~400 digits of pi for the argument reduction.)

    Points have a small imaginary part: enough to step off branch cuts, small
    enough that nested exp/sin usually stay in range. Real points are kept as a
    fallback, and extra candidates are drawn so an occasional pole is harmless.
    """
    rng = np.random.default_rng(seed)
    n = 8 * trials
    z = np.concatenate([rng.uniform(0.3, 2.5, n) + 1j * rng.uniform(-0.1, 0.1, n),
                        rng.uniform(0.3, 2.5, n) + 0j])
    fa = sp.lambdify(X_SYM, a, modules="numpy")
    fb = sp.lambdify(X_SYM, b, modules="numpy")
    with np.errstate(all="ignore"):
        va = np.broadcast_to(np.asarray(fa(z), dtype=complex), z.shape)
        vb = np.broadcast_to(np.asarray(fb(z), dtype=complex), z.shape)
    ok = np.flatnonzero(np.isfinite(va) & np.isfinite(vb))[:trials]
    if len(ok) < trials // 2:
        return None
    return bool(np.all(np.abs(va[ok] - vb[ok]) <= rtol * (1 + np.abs(vb[ok]))))
