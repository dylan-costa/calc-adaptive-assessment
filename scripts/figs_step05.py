"""Figures and numbers for Section 5 of the lecture notes (evaluation).

Run:  .venv/Scripts/python scripts/figs_step05.py
Writes notes/figures/step05_*.{pdf,tex}.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit

from calcdiff.evaluate import (bayes_ceiling, bootstrap_ci, pairs_within, pairwise_accuracy,
                               rank_agreement, split_problems)
from calcdiff.features import feature_matrix
from calcdiff.generate import Sampler
from calcdiff.models.bt import fit_newton, one_hot_design
from calcdiff.models.feature_bt import FeatureBT
from calcdiff.synthetic import hidden_difficulty, random_pairs, simulate_judgments

OUT = Path(__file__).resolve().parents[1] / "notes" / "figures"
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False})
BLUE, GREY = "#2f6db5", "#6b6b6b"
numbers: dict[str, str] = {}
rng = np.random.default_rng(5)

N = 400
Phi = feature_matrix(p.tree for p in Sampler(seed=11).sample_unique(N))
delta = hidden_difficulty(Phi, rng)
train_items, test_items = split_problems(N, 0.5, rng)

# Comparisons are drawn separately within each half, so no comparison straddles the split.
def pairs_among(items, m):
    local = random_pairs(len(items), m, rng)
    return items[local]

tr_pairs = pairs_among(train_items, 4000)
tr_y = simulate_judgments(delta, tr_pairs, rng)
seen_pairs = pairs_among(train_items, 4000)          # NEW comparisons among TRAINING problems
seen_y = simulate_judgments(delta, seen_pairs, rng)
new_pairs = pairs_among(test_items, 4000)            # comparisons among UNSEEN problems
new_y = simulate_judgments(delta, new_pairs, rng)

item = fit_newton(one_hot_design(tr_pairs, N), tr_y, l2=0.1)
item_scores = item.theta                              # unseen problems keep theta = 0
feat = FeatureBT(l2=1.0).fit(Phi, tr_pairs, tr_y)
feat_scores = feat.score(Phi)

rows = [
    ("item-level BT", "new pairs, seen problems", pairwise_accuracy(item_scores, seen_pairs, seen_y),
     bayes_ceiling(delta, seen_pairs)),
    ("item-level BT", "unseen problems", pairwise_accuracy(item_scores, new_pairs, new_y),
     bayes_ceiling(delta, new_pairs)),
    ("feature BT", "new pairs, seen problems", pairwise_accuracy(feat_scores, seen_pairs, seen_y),
     bayes_ceiling(delta, seen_pairs)),
    ("feature BT", "unseen problems", pairwise_accuracy(feat_scores, new_pairs, new_y),
     bayes_ceiling(delta, new_pairs)),
]
with open(OUT / "step05_table.tex", "w") as f:
    f.write("\\begin{tabular}{llcc}\n\\toprule\nmodel & evaluated on & accuracy & oracle ceiling \\\\\n\\midrule\n")
    for model, split, acc, ceil in rows:
        f.write(f"{model} & {split} & {acc:.3f} & {ceil:.3f} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

acc = rows[3][2]
lo, hi = bootstrap_ci(lambda w: pairwise_accuracy(feat_scores, new_pairs, new_y, w),
                      new_pairs, test_items, rng, n_boot=1000)
naive = 1.96 * np.sqrt(acc * (1 - acc) / len(new_pairs))
numbers["FeatAcc"] = f"{acc:.3f}"
numbers["BootLo"] = f"{lo:.3f}"
numbers["BootHi"] = f"{hi:.3f}"
numbers["BootHalf"] = f"{(hi - lo) / 2:.3f}"
numbers["NaiveHalf"] = f"{naive:.3f}"
numbers["WidthRatio"] = f"{(hi - lo) / 2 / naive:.1f}"
numbers["NTest"] = str(len(test_items))
numbers["NTestPairs"] = f"{len(new_pairs):,}".replace(",", "{,}")
ra = rank_agreement(feat_scores[test_items], delta[test_items])
numbers["Spearman"] = f"{ra['spearman']:.3f}"
numbers["Kendall"] = f"{ra['kendall']:.3f}"
numbers["DeltaSD"] = f"{delta.std():.2f}"

# --- The ceiling as a function of how spread out difficulties are ----------------
s_grid = np.linspace(0.05, 4, 80)
z = np.random.default_rng(0).standard_normal(200_000)
ceil = [np.mean(expit(np.abs(np.sqrt(2) * s * z))) for s in s_grid]
here = np.mean(expit(np.abs(np.sqrt(2) * delta.std() * z)))
numbers["CeilHere"] = f"{here:.3f}"

fig, ax = plt.subplots(figsize=(3.6, 2.6))
ax.plot(s_grid, ceil, color=BLUE)
ax.plot([delta.std()], [here], "o", color="k", ms=4)
ax.annotate("our simulated\ninstructor", (delta.std(), here), xytext=(1.9, 0.62), fontsize=7,
            arrowprops=dict(arrowstyle="->", lw=0.6))
ax.axhline(0.5, color=GREY, lw=0.6, ls=":")
ax.set_xlabel(r"spread of difficulty, sd$(\delta)$ in logits")
ax.set_ylabel("best possible accuracy")
fig.tight_layout()
fig.savefig(OUT / "step05_ceiling.pdf")

with open(OUT / "step05_numbers.tex", "w") as f:
    for k, v in numbers.items():
        f.write(f"\\newcommand{{\\E{k}}}{{{v}}}\n")
print("\n".join(f"{k} = {v}" for k, v in numbers.items()))
