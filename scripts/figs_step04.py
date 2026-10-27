"""Figures and numbers for Section 4 of the lecture notes (Bradley-Terry on synthetic data).

Run:  .venv/Scripts/python scripts/figs_step04.py
Writes notes/figures/step04_*.{pdf,tex}.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit
from scipy.stats import spearmanr

from calcdiff.features import feature_matrix
from calcdiff.generate import Sampler
from calcdiff.models.bt import fit_gradient_descent, fit_newton, mle_exists, one_hot_design
from calcdiff.synthetic import hidden_difficulty, random_pairs, simulate_judgments

OUT = Path(__file__).resolve().parents[1] / "notes" / "figures"
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False})
BLUE, ORANGE, GREY = "#2f6db5", "#d9822b", "#6b6b6b"
numbers: dict[str, str] = {}

# --- The item pool and its planted difficulties ----------------------------------
N_ITEMS = 200
rng = np.random.default_rng(4)
probs = Sampler(seed=7).sample_unique(N_ITEMS)
Phi = feature_matrix(p.tree for p in probs)
delta = hidden_difficulty(Phi, rng)
numbers["NItems"] = str(N_ITEMS)
numbers["DeltaSD"] = f"{delta.std():.2f}"
numbers["DeltaMin"] = f"{delta.min():.1f}"
numbers["DeltaMax"] = f"{delta.max():.1f}"

# --- Experiment A: recovery with ~40 comparisons per item --------------------
M = 4000
pairs = random_pairs(N_ITEMS, M, rng)
y = simulate_judgments(delta, pairs, rng)
D = one_hot_design(pairs, N_ITEMS)
L2 = 0.1  # weak Gaussian prior theta_i ~ N(0, 1/L2): the MAP exists even when the MLE does not
fit = fit_newton(D, y, l2=L2)
games = np.abs(D).sum(0)
wins = (D * (2 * y - 1)[:, None] > 0).sum(0)
numbers["MLEExists"] = "exists" if mle_exists(pairs, y, N_ITEMS) else "does not exist"
numbers["NPerfect"] = str(int(np.sum((wins == games) | (wins == 0))))
numbers["LTwo"] = f"{L2:g}"
numbers["PriorSD"] = f"{1 / np.sqrt(L2):.1f}"
se = np.sqrt(np.diag(fit.covariance(centered=True)))
se_naive = np.sqrt(np.diag(fit.covariance()))
numbers["M"] = f"{M:,}".replace(",", "{,}")
numbers["PerItem"] = f"{2 * M / N_ITEMS:.0f}"
numbers["NewtonIters"] = str(fit.n_iter)
numbers["RMSE"] = f"{np.sqrt(np.mean((fit.theta - delta) ** 2)):.2f}"
numbers["MeanSE"] = f"{se.mean():.2f}"
numbers["MeanSENaive"] = f"{se_naive.mean():.2f}"
numbers["Coverage"] = f"{100 * np.mean(np.abs(fit.theta - delta) <= 1.96 * se):.0f}"
numbers["Spearman"] = f"{spearmanr(fit.theta, delta)[0]:.3f}"
numbers["SumTheta"] = f"{abs(fit.theta.sum()):.0e}"

# --- Experiment B: Newton vs gradient descent ---------------------------------
newton = fit_newton(D, y, l2=L2)
gd = fit_gradient_descent(D, y, l2=L2, n_iter=300)
best = min(newton.loss_history[-1], gd.loss_history[-1])
gap_newton = np.maximum(np.array(newton.loss_history) - best, 1e-16)
gap_gd = np.maximum(np.array(gd.loss_history) - best, 1e-16)
numbers["GDIters"] = "300"
numbers["GDToTen"] = str(int(np.argmax(gap_gd < 1e-10)))
H = newton.hessian
ev = np.linalg.eigvalsh(H)
numbers["HessCond"] = f"{ev[-1] / ev[0]:.0f}"

# --- Experiment C: error vs comparisons per item, against Fisher information -----
ks = [2, 4, 8, 16, 32, 64, 128]
SEEDS = 5
rmse = np.zeros((len(ks), SEEDS))
exists = np.zeros((len(ks), SEEDS), dtype=bool)
theory = []
W = expit(delta[:, None] - delta[None, :])
W = W * (1 - W)  # Fisher weight sigma'(delta_i - delta_j) of comparing i with j
L_all = np.diag(W.sum(1)) - W  # Laplacian of the complete graph with weights W
for a, k in enumerate(ks):
    m = k * N_ITEMS // 2
    for s in range(SEEDS):
        r = np.random.default_rng(100 * k + s)
        pr = random_pairs(N_ITEMS, m, r)
        yy = simulate_judgments(delta, pr, r)
        exists[a, s] = mle_exists(pr, yy, N_ITEMS)
        f = fit_newton(one_hot_design(pr, N_ITEMS), yy, l2=L2)
        rmse[a, s] = np.sqrt(np.mean((f.theta - delta) ** 2))
    # expected Fisher information: each comparison is a uniformly random pair
    n_pairs = N_ITEMS * (N_ITEMS - 1) / 2
    # expected Hessian at the truth; the all-ones direction is projected out (no error there)
    P = np.eye(N_ITEMS) - 1.0 / N_ITEMS
    info = (m / n_pairs) * L_all + L2 * P
    theory.append(np.sqrt(np.mean(np.diag(np.linalg.pinv(info)))))
slope = np.polyfit(np.log(ks[2:]), np.log(rmse.mean(1)[2:]), 1)[0]
numbers["RateSlope"] = f"{slope:.2f}"
numbers["ExistsTwo"] = f"{exists[0].mean():.1f}"
numbers["ExistsEight"] = f"{exists[2].mean():.1f}"
numbers["ExistsThirtyTwo"] = f"{exists[4].mean():.1f}"
numbers["SmallKRMSE"] = f"{rmse[0].mean():.2f}"
numbers["LargeKRMSE"] = f"{rmse[-1].mean():.2f}"
numbers["LargeKTheory"] = f"{theory[-1]:.2f}"

fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.6))
ax = axes[0]
order = np.argsort(delta)
ax.errorbar(delta[order], fit.theta[order], yerr=1.96 * se[order], fmt="o", ms=2, color=BLUE,
            ecolor=BLUE, elinewidth=0.5, alpha=0.6)
lim = [delta.min() - 0.5, delta.max() + 0.5]
ax.plot(lim, lim, "--", color=GREY, lw=1)
ax.set_xlabel(r"planted $\delta_i$")
ax.set_ylabel(r"estimate $\hat\theta_i$ ($\pm$1.96 se)")
ax.set_title("(a) recovery", loc="left")

ax = axes[1]
ax.semilogy(gap_newton, "o-", color=ORANGE, ms=3, label="Newton")
ax.semilogy(gap_gd, "-", color=BLUE, label="gradient descent")
ax.set_xlabel("iteration")
ax.set_ylabel(r"$L(\theta_t)-L^\ast$")
ax.set_xlim(0, 150)
ax.legend(frameon=False, fontsize=7)
ax.set_title("(b) optimization", loc="left")

ax = axes[2]
ax.loglog(ks, rmse.mean(1), "o-", color=BLUE, ms=3, label="ridge MAP, simulated")
ax.loglog(ks, theory, "--", color=GREY, label="Fisher information")
ax.set_xlabel("comparisons per item")
ax.set_ylabel("RMSE of $\\hat\\theta$")
ax.legend(frameon=False, fontsize=7)
ax.set_title("(c) rate", loc="left")
fig.tight_layout()
fig.savefig(OUT / "step04_bt.pdf")

with open(OUT / "step04_numbers.tex", "w") as f:
    for k_, v in numbers.items():
        f.write(f"\\newcommand{{\\B{k_}}}{{{v}}}\n")
print("\n".join(f"{k_} = {v}" for k_, v in numbers.items()))
