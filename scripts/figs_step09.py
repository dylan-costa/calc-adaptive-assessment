"""Figures and numbers for Section 9 of the lecture notes (gradient boosting).

Run:  .venv/Scripts/python scripts/figs_step09.py
Reads data/labels.sqlite; writes notes/figures/step09_*.{pdf,tex}.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from calcdiff.evaluate import (accuracy_from_diff, bootstrap_ci, cross_val_diffs, log_loss,
                               log_loss_from_diff, pairs_touching, split_problems)
from calcdiff.features import FEATURE_NAMES, feature_matrix
from calcdiff.generate import Sampler
from calcdiff.labels.store import LabelStore
from calcdiff.models.boosted_bt import BoostedBT, cv_n_trees, staged_held_out_loss
from calcdiff.models.feature_bt import FeatureBT, cv_l2
from calcdiff.synthetic import hidden_difficulty, random_pairs, simulate_judgments

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notes" / "figures"
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False})
BLUE, ORANGE, GREEN, GREY = "#2f6db5", "#d9822b", "#3a9a5b", "#6b6b6b"
numbers: dict[str, str] = {}
GRID = np.logspace(-2, 3, 11)
BOOST = dict(n_trees=250, learning_rate=0.1, max_depth=2, min_leaf=5, leaf_l2=1.0)


def fit_linear(Phi, pairs, y, rng):
    lam, _ = cv_l2(Phi, pairs, y, np.unique(pairs), GRID, rng)
    return FeatureBT(l2=lam).fit(Phi, pairs, y)


def fit_boosted(Phi, pairs, y, rng):
    n, _ = cv_n_trees(Phi, pairs, y, rng, **BOOST)
    return BoostedBT(**{**BOOST, "n_trees": n}).fit(Phi, pairs, y)


# --- A. Synthetic: which inductive bias wins depends on the truth ------------------
N = 500
Phi_s = feature_matrix(p.tree for p in Sampler(seed=21).sample_unique(N))
rng = np.random.default_rng(9)
truths = {
    "linear truth": hidden_difficulty(Phi_s, rng),
    # chain rules make a problem harder with a quotient present, easier without
    "interaction truth": np.where(Phi_s[:, FEATURE_NAMES.index("n_quotient")] > 0, 1.0, -1.0)
    * Phi_s[:, FEATURE_NAMES.index("n_chain")] + 0.03 * Phi_s[:, FEATURE_NAMES.index("deriv_size")],
}
train_items, test_items = split_problems(N, 0.3, rng)
sizes = [100, 200, 400, 800, 1600, 3200]
SEEDS = 3
curves = {}
for tname, delta in truths.items():
    test_pairs = test_items[random_pairs(len(test_items), 4000, rng)]
    test_y = simulate_judgments(delta, test_pairs, rng)
    oracle = log_loss(delta, test_pairs, test_y)
    res = {"linear": [], "boosted": []}
    for m in sizes:
        acc = {k: [] for k in res}
        for s in range(SEEDS):
            r = np.random.default_rng(1000 * m + s)
            tp = train_items[random_pairs(len(train_items), m, r)]
            ty = simulate_judgments(delta, tp, r)
            acc["linear"].append(log_loss(fit_linear(Phi_s, tp, ty, r).score(Phi_s), test_pairs, test_y))
            acc["boosted"].append(log_loss(fit_boosted(Phi_s, tp, ty, r).score(Phi_s), test_pairs, test_y))
        for k in res:
            res[k].append(np.mean(acc[k]))
    curves[tname] = (res, oracle)
lin_t, oracle_lin = curves["linear truth"]
int_t, oracle_int = curves["interaction truth"]
numbers["LinTruthLinSmall"] = f"{lin_t['linear'][0]:.3f}"
numbers["LinTruthBoostSmall"] = f"{lin_t['boosted'][0]:.3f}"
numbers["LinTruthLinLarge"] = f"{lin_t['linear'][-1]:.3f}"
numbers["LinTruthBoostLarge"] = f"{lin_t['boosted'][-1]:.3f}"
numbers["IntTruthLinLarge"] = f"{int_t['linear'][-1]:.3f}"
numbers["IntTruthBoostLarge"] = f"{int_t['boosted'][-1]:.3f}"
numbers["IntTruthOracle"] = f"{oracle_int:.3f}"
numbers["LinTruthOracle"] = f"{oracle_lin:.3f}"
cross = [m for m, a, b in zip(sizes, int_t["linear"], int_t["boosted"]) if b < a]
numbers["IntCross"] = f"{cross[0]:,}".replace(",", "{,}") if cross else "never"

# --- B. Real labels -------------------------------------------------------------------
store = LabelStore(ROOT / "data" / "labels.sqlite")
probs, pairs, y = store.to_arrays("dylan")
Phi = feature_matrix(p.tree for p in probs)
items = np.unique(pairs)
numbers["NJudg"] = str(len(y))
REPS = 5
d_lin = np.mean([cross_val_diffs(Phi, pairs, y, lambda p_, y_, r=r: fit_linear(Phi, p_, y_, np.random.default_rng(r)),
                                 np.random.default_rng(r)) for r in range(REPS)], axis=0)
d_bst = np.mean([cross_val_diffs(Phi, pairs, y, lambda p_, y_, r=r: fit_boosted(Phi, p_, y_, np.random.default_rng(r)),
                                 np.random.default_rng(r)) for r in range(REPS)], axis=0)
for tag, d in (("Lin", d_lin), ("Boost", d_bst)):
    numbers[f"RealAcc{tag}"] = f"{accuracy_from_diff(d, y):.3f}"
    numbers[f"RealLL{tag}"] = f"{log_loss_from_diff(d, y):.3f}"
lo, hi = bootstrap_ci(lambda w: log_loss_from_diff(d_bst, y, w) - log_loss_from_diff(d_lin, y, w),
                      pairs, items, np.random.default_rng(2), 1000)
numbers["RealDiffLo"] = f"{lo:.3f}"
numbers["RealDiffHi"] = f"{hi:.3f}"
n_real, _ = cv_n_trees(Phi, pairs, y, np.random.default_rng(0), **BOOST)
numbers["RealNTrees"] = str(n_real)

# Overfitting curves on real labels: held-out loss vs number of trees, for several depths
depth_curves = {}
for depth in (1, 2, 4):
    tot, cnt = 0.0, 0
    tr_curve = 0.0
    for r in range(3):
        chunks = np.array_split(np.random.default_rng(50 + r).permutation(items), 5)
        for held in chunks:
            te = pairs_touching(pairs, held)
            mdl = BoostedBT(**{**BOOST, "max_depth": depth, "n_trees": 400}).fit(Phi, pairs[~te], y[~te])
            tot = tot + te.sum() * staged_held_out_loss(mdl, Phi, pairs[te], y[te])
            tr_curve = tr_curve + np.array(mdl.train_loss_)
            cnt += te.sum()
    depth_curves[depth] = (tot / cnt, tr_curve / 15)
for depth, (held, train) in depth_curves.items():
    numbers[f"BestTreesDepth{['', 'One', 'Two', '', 'Four'][depth]}"] = str(int(np.argmin(held)) + 1)
    numbers[f"BestLLDepth{['', 'One', 'Two', '', 'Four'][depth]}"] = f"{held.min():.3f}"
numbers["FinalLLDepthFour"] = f"{depth_curves[4][0][-1]:.3f}"
numbers["FinalTrainDepthFour"] = f"{depth_curves[4][1][-1]:.3f}"

# Which features does the boosted model split on?
final = BoostedBT(**{**BOOST, "n_trees": n_real}).fit(Phi, pairs, y)
gain = final.feature_gain(len(FEATURE_NAMES))
top = np.argsort(-gain)[:3]
numbers["GainTop"] = ", ".join(f"\\code{{{FEATURE_NAMES[i].replace('_', chr(92) + '_')}}} ({100 * gain[i]:.0f}\\%)"
                               for i in top)

# --- Figure -------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7))
for ax, (tname, (res, oracle)), tag in zip(axes[:2], curves.items(), ("(a)", "(b)")):
    ax.semilogx(sizes, res["linear"], "o-", color=BLUE, ms=3, label="linear BT")
    ax.semilogx(sizes, res["boosted"], "s-", color=ORANGE, ms=3, label="boosted trees")
    ax.axhline(oracle, color="k", ls="--", lw=0.8, label="oracle")
    ax.set_xlabel("training comparisons")
    ax.set_ylabel("held-out log-loss")
    ax.set_title(f"{tag} {tname}", loc="left")
    ax.legend(frameon=False, fontsize=6.5)
ax = axes[2]
for depth, col in zip((1, 2, 4), (GREEN, ORANGE, BLUE)):
    held, train = depth_curves[depth]
    ax.plot(np.arange(1, len(held) + 1), held, color=col, label=f"depth {depth}: held-out")
    ax.plot(np.arange(1, len(train) + 1), train, color=col, ls=":", lw=0.9)
ax.set_xlabel("number of trees")
ax.set_ylabel("log-loss (your labels)")
ax.set_ylim(0.1, 0.7)
ax.legend(frameon=False, fontsize=6.5)
ax.set_title("(c) overfitting, dotted = train", loc="left", fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "step09_boosting.pdf")

with open(OUT / "step09_numbers.tex", "w") as f:
    for k, v in numbers.items():
        f.write(f"\\newcommand{{\\Q{k}}}{{{v}}}\n")
print("\n".join(f"{k} = {v}" for k, v in numbers.items()))
