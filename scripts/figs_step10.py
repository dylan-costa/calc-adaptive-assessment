"""Figures and numbers for Section 10 of the lecture notes (active learning).

Run:  .venv/Scripts/python scripts/figs_step10.py
Reads data/labels.sqlite; writes notes/figures/step10_*.{pdf,tex}.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit

from calcdiff.active import all_pairs, run_loop
from calcdiff.evaluate import log_loss, pairs_touching, pairwise_accuracy
from calcdiff.features import feature_matrix
from calcdiff.generate import Sampler
from calcdiff.labels.store import LabelStore
from calcdiff.models.feature_bt import FeatureBT
from calcdiff.models.mixed_bt import MixedBT
from calcdiff.synthetic import hidden_difficulty, random_pairs, simulate_judgments

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notes" / "figures"
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False})
COLORS = {"random": "#6b6b6b", "uncertainty": "#d9822b", "information": "#b5432f", "variance": "#2f6db5"}
LABELS = {"random": "random pairs", "uncertainty": "uncertainty (close calls)",
          "information": "information gain (D-opt.)", "variance": "pred. variance + mixed (V-opt.)"}
STRATS = ("random", "uncertainty", "information", "variance")
FEATURE = lambda: FeatureBT(l2=3.0)
MIXED = lambda: MixedBT(l2=3.0, tau=0.5)
MODEL = {"random": FEATURE, "uncertainty": FEATURE, "information": FEATURE, "variance": MIXED}
BATCH = 5
numbers: dict[str, str] = {}

# --- A. Simulation: a 300-problem pool like the app's, tested on unseen problems ----
POOL, TEST = 300, 150
Phi_all = feature_matrix(p.tree for p in Sampler(seed=31).sample_unique(POOL + TEST))
SEEDS, N_TOTAL = 8, 300
curves = {s: [] for s in STRATS}
acc_curves = {s: [] for s in STRATS}
gaps = {s: [] for s in STRATS}
max_use = {s: [] for s in STRATS}
for seed in range(SEEDS):
    rng = np.random.default_rng(seed)
    delta = hidden_difficulty(Phi_all, rng)
    test_items = np.arange(POOL, POOL + TEST)
    test_pairs = test_items[random_pairs(TEST, 4000, rng)]
    test_y = simulate_judgments(delta, test_pairs, rng)
    cand = all_pairs(np.arange(POOL))
    for strat in STRATS:
        r = np.random.default_rng(100 + seed)            # same starting pairs for every strategy
        coin = np.random.default_rng(200 + seed)
        answer = lambda k: float(coin.random() < expit(delta[cand[k, 0]] - delta[cand[k, 1]]))
        sizes, ev, chosen = run_loop(
            Phi_all[:POOL], cand, answer, strat, r, n_start=20, n_total=N_TOTAL, batch=BATCH,
            make_model=MODEL[strat],
            evaluate=lambda m: (log_loss(m.score(Phi_all), test_pairs, test_y),
                                pairwise_accuracy(m.score(Phi_all), test_pairs, test_y)))
        curves[strat].append([e[0] for e in ev])
        acc_curves[strat].append([e[1] for e in ev])
        picked = cand[chosen[20:]]
        gaps[strat].append(np.mean(np.abs(delta[picked[:, 0]] - delta[picked[:, 1]])))
        max_use[strat].append(np.bincount(cand[chosen].ravel(), minlength=POOL).max())
sizes = np.array(sizes)
mean = {s: np.mean(curves[s], axis=0) for s in STRATS}
se = {s: np.std(curves[s], axis=0) / np.sqrt(SEEDS) for s in STRATS}
acc = {s: np.mean(acc_curves[s], axis=0) for s in STRATS}


def labels_to_reach(curve, target):
    """First size at which a (decreasing) loss curve reaches `target`, linearly interpolated."""
    below = np.flatnonzero(curve <= target)
    if len(below) == 0:
        return np.nan
    k = below[0]
    if k == 0:
        return sizes[0]
    x0, x1, y0, y1 = sizes[k - 1], sizes[k], curve[k - 1], curve[k]
    return x0 + (target - y0) * (x1 - x0) / (y1 - y0)


for at in (100, 200):
    target = mean["random"][np.searchsorted(sizes, at)]
    need = labels_to_reach(mean["variance"], target)
    numbers[f"VarNeedsFor{ {100: 'Hundred', 200: 'TwoHundred'}[at] }"] = f"{need:.0f}"
    numbers[f"Saving{ {100: 'Hundred', 200: 'TwoHundred'}[at] }"] = f"{100 * (1 - need / at):.0f}"
for s, tag in zip(STRATS, ("Rand", "Unc", "Info", "Var")):
    k100 = np.searchsorted(sizes, 100)
    numbers[f"LL{tag}Hundred"] = f"{mean[s][k100]:.3f}"
    numbers[f"LL{tag}Final"] = f"{mean[s][-1]:.3f}"
    numbers[f"Acc{tag}Hundred"] = f"{acc[s][k100]:.3f}"
    numbers[f"Gap{tag}"] = f"{np.mean(gaps[s]):.2f}"
    numbers[f"MaxUse{tag}"] = f"{np.mean(max_use[s]):.0f}"
numbers["Seeds"] = str(SEEDS)
for s, tag in zip(STRATS, ("Rand", "Unc", "Info", "Var")):
    numbers[f"LL{tag}Sixty"] = f"{mean[s][np.searchsorted(sizes, 60)]:.3f}"
    numbers[f"LL{tag}TwoHundred"] = f"{mean[s][np.searchsorted(sizes, 200)]:.3f}"
# Paired comparisons across seeds (same starting pairs, same test set): mean difference +- standard error
for other, tag in (("variance", "Var"), ("information", "Info")):
    for at, word in ((60, "Sixty"), (200, "TwoHundred"), (300, "ThreeHundred")):
        k = np.searchsorted(sizes, at)
        diff = np.array(curves[other])[:, k] - np.array(curves["random"])[:, k]
        numbers[f"Diff{tag}{word}"] = f"{diff.mean():+.3f}"
        numbers[f"DiffSE{tag}{word}"] = f"{diff.std(ddof=1) / np.sqrt(len(diff)):.3f}"
        numbers[f"Wins{tag}{word}"] = str(int((diff < 0).sum()))
numbers["NTotal"] = str(N_TOTAL)
numbers["Batch"] = str(BATCH)

# --- B. Replay on the real labels ---------------------------------------------------
store = LabelStore(ROOT / "data" / "labels.sqlite")
probs, pairs, y = store.to_arrays("dylan", include_repeats=False)
Phi = feature_matrix(p.tree for p in probs)
items = np.unique(pairs)
numbers["NReal"] = str(len(y))
REPLAY = ("random", "information", "variance")
replay = {s: [] for s in REPLAY}
grid = None
for rep in range(4):
    chunks = np.array_split(np.random.default_rng(rep).permutation(items), 5)
    for f, held in enumerate(chunks):
        te = pairs_touching(pairs, held)
        cand_idx = np.flatnonzero(~te)
        cand = pairs[cand_idx]
        for strat in REPLAY:
            r = np.random.default_rng(1000 * rep + f)
            sz, ev, _ = run_loop(Phi, cand, lambda k: y[cand_idx[k]], strat, r, n_start=20,
                                 n_total=len(cand), batch=BATCH, make_model=MODEL[strat],
                                 evaluate=lambda m: log_loss(m.score(Phi), pairs[te], y[te]))
            sz = np.array(sz)
            grid = np.arange(20, 160, 10) if grid is None else grid
            replay[strat].append(np.interp(grid, sz, ev))
rmean = {s: np.mean(replay[s], axis=0) for s in REPLAY}
for s, tag in zip(REPLAY, ("Rand", "Info", "Var")):
    k = np.searchsorted(grid, 60)
    numbers[f"Replay{tag}Sixty"] = f"{rmean[s][k]:.3f}"
    numbers[f"Replay{tag}End"] = f"{rmean[s][-1]:.3f}"
target = rmean["random"][np.searchsorted(grid, 100)]
below = np.flatnonzero(rmean["variance"] <= target)
numbers["ReplayVarReach"] = str(int(grid[below[0]])) if len(below) else "more than 150"

# --- Figure ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7))
ax = axes[0]
for s in STRATS:
    ax.plot(sizes, mean[s], color=COLORS[s], label=LABELS[s])
    ax.fill_between(sizes, mean[s] - se[s], mean[s] + se[s], color=COLORS[s], alpha=0.2, lw=0)
ax.set_xlabel("labels")
ax.set_ylabel("held-out log-loss")
ax.set_title("(a) simulated instructor", loc="left")
ax.set_ylim(0.53, 0.70)
ax.legend(frameon=False, fontsize=5.8, loc="upper right", bbox_to_anchor=(1.0, 0.93))
ax = axes[1]
xs = np.arange(len(STRATS))
ax.bar(xs, [np.mean(max_use[s]) for s in STRATS], color=[COLORS[s] for s in STRATS])
ax.set_xticks(xs, ["rand.", "unc.", "D", "V"])
ax.set_ylabel(f"most uses of one problem ({N_TOTAL} labels)")
ax.set_title("(b) who gets asked", loc="left")
ax = axes[2]
for s in REPLAY:
    ax.plot(grid, rmean[s], color=COLORS[s], label=LABELS[s])
ax.set_xlabel("labels (replayed from yours)")
ax.set_ylabel("held-out log-loss")
ax.set_title("(c) replay on your labels", loc="left")
fig.tight_layout()
fig.savefig(OUT / "step10_active.pdf")

with open(OUT / "step10_numbers.tex", "w") as f:
    for k, v in numbers.items():
        f.write(f"\\newcommand{{\\A{k}}}{{{v}}}\n")
print("\n".join(f"{k} = {v}" for k, v in numbers.items()))
