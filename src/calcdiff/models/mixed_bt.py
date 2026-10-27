"""Bradley-Terry with features *and* a random effect per problem (a mixed model).

    delta(x_i) = theta . z_i + u_i,     u_i ~ N(0, tau^2) independently,
    P(i harder than j) = sigma(theta . (z_i - z_j) + u_i - u_j).

theta is shared by every problem; u_i is what is particular to problem i and
invisible to the features (the tau * eps of the simulated instructor, or
whatever your judgments see that phi does not). The design row of a
comparison is [z_i - z_j, e_i - e_j], and the Gaussian prior on u is a ridge
penalty 1 / tau^2 on those coordinates, so `fit_newton` fits it unchanged.

Why bother: comparisons that share a problem share its u_i, so they are not
independent evidence about theta. The pure feature model treats them as if
they were (Sections 6 and 8), and active learning built on it over-values
asking about the same problem again (Section 10).

Prediction for an unseen problem uses u = 0: score(x) = theta . z(x).
"""

from __future__ import annotations

import numpy as np

from calcdiff.features import Standardizer
from calcdiff.models.bt import BTFit, fit_newton


class MixedBT:
    def __init__(self, l2: float = 3.0, tau: float = 0.5):
        self.l2 = l2
        self.tau = tau

    def fit(self, Phi: np.ndarray, pairs: np.ndarray, y: np.ndarray) -> "MixedBT":
        n, d = Phi.shape
        self.scaler_ = Standardizer().fit(Phi[np.unique(pairs)])
        self.Z_ = self.scaler_.transform(Phi)
        G = np.hstack([self.Z_, np.eye(n)])                  # row i: g_i = [z_i, e_i]
        D = G[pairs[:, 0]] - G[pairs[:, 1]]
        prior = np.concatenate([np.full(d, self.l2), np.full(n, 1.0 / max(self.tau, 1e-6) ** 2)])
        self.fit_: BTFit = fit_newton(D, y, l2=prior)
        self.theta_ = self.fit_.theta[:d]
        self.u_ = self.fit_.theta[d:]
        self.d_ = d
        return self

    def score(self, Phi: np.ndarray) -> np.ndarray:
        """Predicted difficulty from features alone (the u's of training problems are not used)."""
        return self.scaler_.transform(Phi) @ self.theta_

    def pool_score(self) -> np.ndarray:
        """theta . z_i + u_i for every row of the training Phi."""
        return self.Z_ @ self.theta_ + self.u_
