"""Figures and numbers for Section 12 of the lecture notes (Shapley explanations).

Run:  .venv/Scripts/python scripts/figs_step12.py
Reads data/labels.sqlite; writes notes/figures/step12_*.{pdf,tex}.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit
from scipy.stats import spearmanr

from calcdiff.evaluate import cross_val_diffs
from calcdiff.explain import (GROUP_COLUMNS, GROUPS, describe, linear_shapley, shapley_exact,
                              shapley_sampling)
from calcdiff.features import feature_matrix
from calcdiff.labels.store import LabelStore
from calcdiff.models.boosted_bt import BoostedBT, cv_n_trees
from calcdiff.models.feature_bt import FeatureBT, cv_l2

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notes" / "figures"
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False})
BLUE, ORANGE, GREY, RED = "#2f6db5", "#d9822b", "#6b6b6b", "#b5432f"
numbers: dict[str, str] = {}
NAMES = list(GROUPS)

store = LabelStore(ROOT / "data" / "labels.sqlite")
probs, pairs, y = store.to_arrays("dylan")
Phi = feature_matrix(p.tree for p in probs)
items = np.unique(pairs)
numbers["NJudg"] = str(len(y))
numbers["NGroups"] = str(len(GROUPS))
numbers["NCoalitions"] = str(2 ** len(GROUPS))

lam, _ = cv_l2(Phi, pairs, y, items, np.logspace(-2, 3, 11), np.random.default_rng(0))
linear = FeatureBT(l2=lam).fit(Phi, pairs, y)
BOOST = dict(learning_rate=0.1, max_depth=2, min_leaf=5, leaf_l2=1.0)
n_trees, _ = cv_n_trees(Phi, pairs, y, np.random.default_rng(0), n_trees=250, **BOOST)
boosted = BoostedBT(n_trees=n_trees, **BOOST).fit(Phi, pairs, y)

# --- Global importance: mean |phi| over random pairs of pool problems --------------
rng = np.random.default_rng(1)
sample = rng.choice(len(Phi), size=(400, 2))
sample = sample[sample[:, 0] != sample[:, 1]]
phi_lin = np.array([linear_shapley(linear.raw_weights(), Phi[i], Phi[j]) for i, j in sample])
phi_bst = np.array([shapley_exact(boosted.score, Phi[i], Phi[j]) for i, j in sample])
imp_lin, imp_bst = np.abs(phi_lin).mean(0), np.abs(phi_bst).mean(0)
gain = boosted.feature_gain(Phi.shape[1])
gain_grp = np.array([gain[c].sum() for c in GROUP_COLUMNS])
for tag, imp in (("Lin", imp_lin), ("Bst", imp_bst)):
    order = np.argsort(-imp)
    numbers[f"Top{tag}"] = NAMES[order[0]]
    numbers[f"Second{tag}"] = NAMES[order[1]]
    numbers[f"TopShare{tag}"] = f"{100 * imp[order[0]] / imp.sum():.0f}"
numbers["RhoLinBst"] = f"{spearmanr(imp_lin, imp_bst)[0]:.2f}"
numbers["GainTopGroup"] = NAMES[int(np.argmax(gain_grp))]
numbers["GainTopShare"] = f"{100 * gain_grp.max():.0f}"
iq = NAMES.index("quotient rule")
numbers["QuotRankBst"] = str(int(np.argsort(-imp_bst).tolist().index(iq)) + 1)
numbers["QuotRankGain"] = str(int(np.argsort(-gain_grp).tolist().index(iq)) + 1)
# Efficiency check over the whole sample
gaps = boosted.score(Phi[sample[:, 0]]) - boosted.score(Phi[sample[:, 1]])
numbers["EffErr"] = f"{np.abs(phi_bst.sum(1) - gaps).max():.0e}"

# --- Sampling estimator: error vs number of random orders --------------------------
ns = [5, 20, 80, 320]
errs = []
for n in ns:
    e = []
    for (i, j), exact in zip(sample[:40], phi_bst[:40]):
        est = shapley_sampling(boosted.score, Phi[i], Phi[j], n, np.random.default_rng(n + i))
        e.append(np.sqrt(np.mean((est - exact) ** 2)))
    errs.append(np.mean(e))
slope = np.polyfit(np.log(ns), np.log(errs), 1)[0]
numbers["SampSlope"] = f"{slope:.2f}"
numbers["SampErrTwenty"] = f"{errs[1]:.3f}"

# --- Explaining your own comparisons ---------------------------------------------------
d_oof = np.mean([cross_val_diffs(Phi, pairs, y, lambda p_, y_: FeatureBT(l2=lam).fit(Phi, p_, y_),
                                 np.random.default_rng(r)) for r in range(3)], axis=0)
decided = y != 0.5
wrong = decided & ((d_oof > 0) != (y == 1))
right = decided & ~wrong


def row(k):
    i, j = pairs[k]
    hard, easy = (i, j) if y[k] == 1 else (j, i)
    phi = linear_shapley(linear.raw_weights(), Phi[hard], Phi[easy])
    p = expit(phi.sum())
    top = np.argsort(-np.abs(phi))[:3]
    contrib = ", ".join(f"{NAMES[t]} ${phi[t]:+.1f}$" for t in top)
    return (f"$\\displaystyle {probs[hard].latex}$ & $\\displaystyle {probs[easy].latex}$ & {p:.2f} & "
            f"\\parbox[t]{{4.2cm}}{{\\raggedright {contrib}\\\\ \\emph{{{describe(phi)}}}}} \\\\[6pt]\n")


has_quot = (Phi[pairs[:, 0], 8] > 0) != (Phi[pairs[:, 1], 8] > 0)
moderate = right & (np.abs(d_oof) > 1.1) & (np.abs(d_oof) < 2.9) & has_quot   # P between 0.75 and 0.95
pick_right = np.flatnonzero(moderate)[0]
pick_close = np.flatnonzero(right)[np.argsort(np.abs(d_oof[right]))][3]
pick_wrong = np.flatnonzero(wrong)[np.argmax(np.abs(d_oof[wrong]))]
with open(OUT / "step12_examples.tex", "w") as f:
    f.write("{\\renewcommand{\\arraystretch}{1.3}\\begin{tabular}{llcp{4.3cm}}\n\\toprule\n"
            "you said harder & than & $P$ & why, according to the model \\\\\n\\midrule\n")
    for k in (pick_right, pick_close, pick_wrong):
        f.write(row(k))
    f.write("\\bottomrule\n\\end{tabular}}\n")
wf_k = pick_right
i, j = pairs[wf_k]
hard, easy = (i, j) if y[wf_k] == 1 else (j, i)
phi_wf = linear_shapley(linear.raw_weights(), Phi[hard], Phi[easy])
numbers["WFHard"] = probs[hard].latex
numbers["WFEasy"] = probs[easy].latex
numbers["WFGap"] = f"{phi_wf.sum():.2f}"
numbers["WFProb"] = f"{expit(phi_wf.sum()):.2f}"

# --- Figure ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.8), gridspec_kw={"width_ratios": [1.25, 1.2, 0.85]})
ax = axes[0]
yy = np.arange(len(NAMES))
order = np.argsort(imp_lin)
ax.barh(yy - 0.2, imp_lin[order], 0.4, color=BLUE, label="linear BT")
ax.barh(yy + 0.2, imp_bst[order], 0.4, color=ORANGE, label="boosted BT")
ax.set_yticks(yy, [NAMES[k] for k in order], fontsize=7)
ax.set_xlabel(r"mean $|\phi|$ over random pairs (logits)")
ax.legend(frameon=False, fontsize=6.5)
ax.set_title("(a) what drives the gap", loc="left", fontsize=8)
ax = axes[1]
order = np.argsort(-np.abs(phi_wf))
cum = 0.0
for n_, k in enumerate(order):
    ax.barh(n_, phi_wf[k], left=cum, color=BLUE if phi_wf[k] > 0 else RED)
    cum += phi_wf[k]
ax.axvline(0, color=GREY, lw=0.6)
ax.axvline(cum, color="k", lw=0.8, ls="--")
ax.set_yticks(range(len(order)), [NAMES[k] for k in order], fontsize=7)
ax.invert_yaxis()
ax.set_xlabel(r"cumulative $\hat\delta(A)-\hat\delta(B)$ (logits)")
ax.set_title("(b) one comparison", loc="left", fontsize=8)
ax = axes[2]
ax.loglog(ns, errs, "o-", color=BLUE, ms=3, label="sampling")
ax.loglog(ns, errs[0] * np.sqrt(ns[0] / np.array(ns)), "--", color=GREY, label=r"$\propto n^{-1/2}$")
ax.set_xlabel("random orders $n$")
ax.set_ylabel("RMS error vs. exact")
ax.legend(frameon=False, fontsize=6.5, loc="lower left")
ax.set_title("(c) sampling", loc="left", fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "step12_shapley.pdf")

with open(OUT / "step12_numbers.tex", "w") as f:
    for k, v in numbers.items():
        f.write(f"\\newcommand{{\\X{k}}}{{{v}}}\n")
print("\n".join(f"{k} = {v}" for k, v in numbers.items()))
print(open(OUT / "step12_examples.tex").read())
