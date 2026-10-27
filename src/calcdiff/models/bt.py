"""Bradley-Terry by maximum likelihood.

A comparison k between items i and j has label y_k = 1 if i was judged harder,
0 if j was, and 1/2 for "about the same".
Each comparison is encoded as a row D_k of a design matrix, and the model is

    P(y_k = 1) = sigma(D_k . theta).

Two encodings matter in this project:

  - one-hot:  D_k = e_i - e_j,          theta = per-item difficulties (Step 4)
  - features: D_k = phi(x_i) - phi(x_j), theta = feature weights     (Step 6)

so a single fitter covers both: Bradley-Terry is logistic regression on
differences. The negative log-likelihood is

    L(theta) = sum_k [softplus(t_k) - y_k t_k] + (l2 / 2) ||theta||^2,
    t = D theta,  softplus(u) = log(1 + e^u),

the cross-entropy between y_k and sigma(t_k). (For y_k in {0, 1} this is
softplus(-s_k t_k) with s_k = 2 y_k - 1.)

with gradient  D^T (p - y) + l2 theta  and Hessian  D^T W D + l2 I, where
p = sigma(t) and W = diag(p (1 - p)). For one-hot rows, D^T W D is the
Laplacian of the comparison graph weighted by W (Section 4 of the notes).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.special import expit


@dataclass
class BTFit:
    theta: np.ndarray
    n_iter: int
    converged: bool
    loss_history: list[float] = field(default_factory=list)
    hessian: np.ndarray | None = None

    def covariance(self, centered: bool = False) -> np.ndarray:
        """Laplace / asymptotic covariance: the (pseudo-)inverse of the Hessian.

        With l2 = 0 and one-hot rows the Hessian is singular along the all-ones
        direction (difficulty is defined only up to a constant), so we use the
        Moore-Penrose pseudoinverse: the covariance of theta in the gauge
        sum(theta) = 0.

        With l2 > 0 the Hessian is invertible, but its only curvature along the
        all-ones direction is l2 itself, so the inverse assigns that direction
        the pure prior variance 1/l2. One-hot estimates satisfy sum(theta) = 0
        exactly, so there is no error along it: pass centered=True to project
        it out, P H^-1 P with P = I - 11^T / n.
        """
        C = np.linalg.pinv(self.hessian)
        if centered:
            n = len(C)
            P = np.eye(n) - np.full((n, n), 1.0 / n)
            C = P @ C @ P
        return C


def nll(theta: np.ndarray, D: np.ndarray, y: np.ndarray, l2: float | np.ndarray = 0.0) -> float:
    """l2 may be a scalar or a per-coordinate vector (a diagonal Gaussian prior)."""
    t = D @ theta
    # cross-entropy: softplus(t) - y t, with softplus(t) = log(1 + e^t) computed stably
    return float((np.logaddexp(0.0, t) - y * t).sum() + 0.5 * (l2 * theta) @ theta)


def grad_hess(theta: np.ndarray, D: np.ndarray, y: np.ndarray, l2: float | np.ndarray = 0.0):
    p = expit(D @ theta)
    g = D.T @ (p - y) + l2 * theta
    H = (D * (p * (1 - p))[:, None]).T @ D + np.diag(np.broadcast_to(l2, theta.shape))
    return g, H


def fit_newton(D: np.ndarray, y: np.ndarray, l2: float | np.ndarray = 0.0, max_iter: int = 100,
               tol: float = 1e-10) -> BTFit:
    """Newton's method with a backtracking (Armijo) line search.

    The step solves H step = -g by least squares. When H is singular (l2 = 0,
    one-hot rows) this returns the minimum-norm solution, which lies in the
    range of H = the orthogonal complement of the all-ones vector. Starting from
    theta = 0, every iterate therefore stays in the gauge sum(theta) = 0.
    """
    theta = np.zeros(D.shape[1])
    hist = [nll(theta, D, y, l2)]
    for it in range(1, max_iter + 1):
        g, H = grad_hess(theta, D, y, l2)
        step = np.linalg.lstsq(H, -g, rcond=None)[0]
        # Newton decrement lambda^2 = -g . step: half of it estimates L - L*
        decrement = float(-g @ step)
        if decrement / 2 < tol:
            return BTFit(theta, it - 1, True, hist, H)
        t = 1.0
        while nll(theta + t * step, D, y, l2) > hist[-1] - 0.25 * t * decrement:
            t *= 0.5
            if t < 1e-12:
                break
        theta = theta + t * step
        hist.append(nll(theta, D, y, l2))
    _, H = grad_hess(theta, D, y, l2)
    return BTFit(theta, max_iter, False, hist, H)


def fit_gradient_descent(D: np.ndarray, y: np.ndarray, l2: float = 0.0, lr: float | None = None,
                         n_iter: int = 500) -> BTFit:
    """Plain gradient descent, for comparison with Newton.

    The default step size 1/beta uses beta = ||D||_2^2 / 4 + l2, an upper bound
    on the largest Hessian eigenvalue (since p(1-p) <= 1/4), which guarantees
    monotone decrease of the loss.
    """
    if lr is None:
        lr = 1.0 / (np.linalg.norm(D, 2) ** 2 / 4 + l2)
    theta = np.zeros(D.shape[1])
    hist = [nll(theta, D, y, l2)]
    for _ in range(n_iter):
        p = expit(D @ theta)
        theta = theta - lr * (D.T @ (p - y) + l2 * theta)
        hist.append(nll(theta, D, y, l2))
    return BTFit(theta, n_iter, False, hist, grad_hess(theta, D, y, l2)[1])


def mle_exists(pairs: np.ndarray, y: np.ndarray, n_items: int) -> bool:
    """Zermelo/Ford condition: a finite one-hot MLE exists iff the directed graph
    with an edge winner -> loser for every comparison is strongly connected.

    If not, some set S of items never lost to the complement; pushing every
    theta in S up by t lowers the loss for all t, so the infimum is not attained.
    """
    win = np.where(y[:, None] == 1, pairs, pairs[:, ::-1])  # rows (winner, loser)
    G = coo_matrix((np.ones(len(win)), (win[:, 0], win[:, 1])), shape=(n_items, n_items))
    n_comp, _ = connected_components(G, directed=True, connection="strong")
    return n_comp == 1


def one_hot_design(pairs: np.ndarray, n_items: int) -> np.ndarray:
    """Rows e_i - e_j for each comparison (i, j)."""
    D = np.zeros((len(pairs), n_items))
    rows = np.arange(len(pairs))
    D[rows, pairs[:, 0]] = 1.0
    D[rows, pairs[:, 1]] = -1.0
    return D
