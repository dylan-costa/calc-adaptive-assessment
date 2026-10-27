"""Figures and numbers for Section 6 of the lecture notes (linear Bradley-Terry on features).

Run:  .venv/Scripts/python scripts/figs_step06.py
Writes notes/figures/step06_*.{pdf,tex}.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from calcdiff.evaluate import bayes_ceiling, pairwise_accuracy, split_problems
from calcdiff.features import FEATURE_NAMES, feature_matrix
from calcdiff.generate import Sampler
from calcdiff.models.feature_bt import FeatureBT, cv_l2, problem_bootstrap
from calcdiff.synthetic import SIMULATED_INSTRUCTOR, hidden_difficulty, random_pairs, simulate_judgments

OUT = Path(__file__).resolve().parents[1] / "notes" / "figures"
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False})
BLUE, ORANGE, GREEN, GREY = "#2f6db5", "#d9822b", "#3a9a5b", "#6b6b6b"
numbers: dict[str, str] = {}
rng = np.random.default_rng(6)

N = 500
Phi = feature_matrix(p.tree for p in Sampler(seed=13).sample_unique(N))
delta = hidden_difficulty(Phi, rng)
train_items, test_items = split_problems(N, 0.3, rng)
test_pairs = test_items[random_pairs(len(test_items), 5000, rng)]
test_y = simulate_judgments(delta, test_pairs, rng)
ceiling = bayes_ceiling(delta, test_pairs)
# The best any *feature* model can do: score by the planted feature part w . phi, no idiosyncratic term
w_planted = np.array([SIMULATED_INSTRUCTOR.get(n, 0.0) for n in FEATURE_NAMES])
feature_ceiling = pairwise_accuracy(Phi @ w_planted, test_pairs, test_y)
GRID = np.logspace(-2, 3, 11)

# --- Learning curve: held-out accuracy vs number of training comparisons ---------
sizes = [25, 50, 100, 200, 400, 800, 1600, 3200]
SEEDS = 8
size_col = [FEATURE_NAMES.index("size")]
deriv_col = [FEATURE_NAMES.index("deriv_size")]
curves = {"all features, CV ridge": [], "size only": [], "derivative size only": []}
chosen_l2 = []
for m in sizes:
    accs = {k: [] for k in curves}
    lams = []
    for s in range(SEEDS):
        r = np.random.default_rng(1000 * m + s)
        tr_pairs = train_items[random_pairs(len(train_items), m, r)]
        tr_y = simulate_judgments(delta, tr_pairs, r)
        lam, _ = cv_l2(Phi, tr_pairs, tr_y, np.unique(tr_pairs), GRID, r, folds=5)
        lams.append(lam)
        models = {"all features, CV ridge": FeatureBT(l2=lam),
                  "size only": FeatureBT(l2=1e-3, columns=size_col),
                  "derivative size only": FeatureBT(l2=1e-3, columns=deriv_col)}
        for k, mdl in models.items():
            mdl.fit(Phi, tr_pairs, tr_y)
            accs[k].append(pairwise_accuracy(mdl.score(Phi), test_pairs, test_y))
    for k in curves:
        curves[k].append((np.mean(accs[k]), np.std(accs[k])))
    chosen_l2.append(np.median(lams))

full = np.array([c[0] for c in curves["all features, CV ridge"]])
numbers["Ceiling"] = f"{ceiling:.3f}"
numbers["FeatCeiling"] = f"{feature_ceiling:.3f}"
numbers["AccSmall"] = f"{full[0]:.3f}"
numbers["AccTwoHundred"] = f"{full[sizes.index(200)]:.3f}"
numbers["AccLarge"] = f"{full[-1]:.3f}"
numbers["SizeOnlyLarge"] = f"{curves['size only'][-1][0]:.3f}"
numbers["DerivOnlyLarge"] = f"{curves['derivative size only'][-1][0]:.3f}"
numbers["LamSmall"] = f"{chosen_l2[0]:.3g}"
numbers["LamLarge"] = f"{chosen_l2[-1]:.3g}"
numbers["LamRatioSmall"] = f"{chosen_l2[0] / sizes[0]:.2f}"
numbers["LamRatioLarge"] = f"{chosen_l2[-1] / sizes[-1]:.3f}"
numbers["NTrainProblems"] = str(len(train_items))
numbers["NTestProblems"] = str(len(test_items))
# How many comparisons until within 2 points of the ceiling?
close = [m for m, a in zip(sizes, full) if feature_ceiling - a < 0.01]
numbers["WithinOne"] = f"{close[0]:,}".replace(",", "{,}") if close else "more than 3{,}200"
numbers["PerProblemAtTwoHundred"] = f"{2 * 200 / len(train_items):.1f}"

# --- Weight recovery with a large training set -----------------------------------
r = np.random.default_rng(77)
big_pairs = train_items[random_pairs(len(train_items), 3200, r)]
big_y = simulate_judgments(delta, big_pairs, r)
lam_big, _ = cv_l2(Phi, big_pairs, big_y, np.unique(big_pairs), GRID, r)
model = FeatureBT(l2=lam_big).fit(Phi, big_pairs, big_y)
w, se = model.raw_weights(), model.raw_weight_se()
se_boot = problem_bootstrap(Phi, big_pairs, big_y, lam_big, np.random.default_rng(8), n_boot=300).std(axis=0)
planted = w_planted
numbers["Covered"] = str(int(np.sum(np.abs(w - planted) <= 1.96 * se)))
numbers["CoveredBoot"] = str(int(np.sum(np.abs(w - planted) <= 1.96 * se_boot)))
numbers["SERatio"] = f"{np.median(se_boot / se):.1f}"
missed = [FEATURE_NAMES[i].replace("_", r"\_") for i in range(len(w)) if abs(w[i] - planted[i]) > 1.96 * se[i]]
numbers["Missed"] = ", ".join(rf"\code{{{m}}}" for m in missed)
numbers["NFeat"] = str(len(FEATURE_NAMES))

# Coverage of nominal 95% intervals over independent replications (light ridge, so bias is small):
# Laplace assumes independent comparisons; the problem bootstrap does not.
cov_lap, cov_boot, ratios = [], [], []
for rep in range(6):
    r = np.random.default_rng(300 + rep)
    P_ = feature_matrix(p.tree for p in Sampler(seed=400 + rep).sample_unique(350))
    d_ = hidden_difficulty(P_, r)
    pr = random_pairs(350, 3200, r)
    yy = simulate_judgments(d_, pr, r)
    mdl = FeatureBT(l2=1.0).fit(P_, pr, yy)
    sb = problem_bootstrap(P_, pr, yy, 1.0, r, n_boot=100).std(axis=0)
    cov_lap.append(np.mean(np.abs(mdl.raw_weights() - w_planted) <= 1.96 * mdl.raw_weight_se()))
    cov_boot.append(np.mean(np.abs(mdl.raw_weights() - w_planted) <= 1.96 * sb))
    ratios.append(np.median(sb / mdl.raw_weight_se()))
numbers["RepCovLaplace"] = f"{100 * np.mean(cov_lap):.0f}"
numbers["RepCovBoot"] = f"{100 * np.mean(cov_boot):.0f}"
numbers["RepSERatio"] = f"{np.mean(ratios):.1f}"
iq = FEATURE_NAMES.index("n_quotient")
numbers["QuotW"] = f"{w[iq]:.2f}"
numbers["QuotSE"] = f"{se_boot[iq]:.2f}"
numbers["QuotPlanted"] = f"{planted[iq]:.2f}"
isz = FEATURE_NAMES.index("size")
numbers["SizeW"] = f"{w[isz]:.3f}"
numbers["SizeSE"] = f"{se_boot[isz]:.3f}"

# --- CV curve at a small training size --------------------------------------------
r = np.random.default_rng(55)
small_pairs = train_items[random_pairs(len(train_items), 100, r)]
small_y = simulate_judgments(delta, small_pairs, r)
best_small, cv_curve = cv_l2(Phi, small_pairs, small_y, np.unique(small_pairs), GRID, r)
numbers["CVBestSmall"] = f"{best_small:g}"

fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.8), gridspec_kw={"width_ratios": [1.15, 1.25, 0.9]})
ax = axes[0]
for (k, c), col in zip(curves.items(), (BLUE, ORANGE, GREEN)):
    mu = np.array([a for a, _ in c])
    sd = np.array([b for _, b in c])
    ax.semilogx(sizes, mu, "o-", color=col, ms=3, label=k)
    ax.fill_between(sizes, mu - sd, mu + sd, color=col, alpha=0.15, lw=0)
ax.axhline(ceiling, color="k", ls="--", lw=0.8, label="oracle ceiling")
ax.axhline(feature_ceiling, color="k", ls="-.", lw=0.8, label=r"planted $w\cdot\varphi$")
ax.set_ylim(0.6, 0.78)
ax.set_xlabel("training comparisons")
ax.set_ylabel("accuracy on unseen problems")
ax.legend(frameon=False, fontsize=6.5, loc="lower right")
ax.set_title("(a) learning curve", loc="left")

ax = axes[1]
order = np.argsort(planted)
ypos = np.arange(len(FEATURE_NAMES))
ax.errorbar(w[order], ypos, xerr=1.96 * se_boot[order], fmt="none", ecolor=BLUE, elinewidth=0.6,
            alpha=0.5, label="95%, problem bootstrap")
ax.errorbar(w[order], ypos, xerr=1.96 * se[order], fmt="o", color=BLUE, ms=3, elinewidth=1.8,
            label="95%, Laplace")
ax.plot(planted[order], ypos, "x", color=ORANGE, ms=5, label="planted")
ax.axvline(0, color=GREY, lw=0.6)
ax.set_yticks(ypos, [FEATURE_NAMES[i] for i in order], fontsize=6)
ax.set_xlabel("logits per unit of feature")
ax.legend(frameon=False, fontsize=6.5, loc="lower right")
ax.set_title("(b) weights", loc="left")

ax = axes[2]
ax.semilogx(GRID, cv_curve, "o-", color=BLUE, ms=3)
ax.axvline(best_small, color=ORANGE, lw=0.8, ls="--")
ax.set_xlabel(r"ridge strength $\lambda$")
ax.set_ylabel("held-out log-loss (5-fold)")
ax.set_title("(c) choosing $\\lambda$, 100 labels", loc="left", fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "step06_feature_bt.pdf")

with open(OUT / "step06_numbers.tex", "w") as f:
    for k, v in numbers.items():
        f.write(f"\\newcommand{{\\G{k}}}{{{v}}}\n")
print("\n".join(f"{k} = {v}" for k, v in numbers.items()))
