"""Figures and numbers for Section 11 of the lecture notes (a neural network on expression trees).

Run:  .venv/Scripts/python scripts/figs_step11.py
Reads data/labels.sqlite; writes notes/figures/step11_*.{pdf,tex}.
"""

from __future__ import annotations

import copy
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from calcdiff.evaluate import (accuracy_from_diff, bootstrap_ci, cross_val_diffs, log_loss,
                               log_loss_from_diff, pairs_touching, split_problems)
from calcdiff.features import FEATURE_NAMES, feature_matrix
from calcdiff.generate import Sampler
from calcdiff.labels.store import LabelStore
from calcdiff.models.feature_bt import FeatureBT, cv_l2
from calcdiff.models.tree_net import TreeBT, compile_forest, pretrain_encoder
from calcdiff.synthetic import hidden_difficulty, random_pairs, simulate_judgments

warnings.filterwarnings("ignore")
torch.set_num_threads(4)
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notes" / "figures"
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False})
BLUE, ORANGE, GREEN, GREY, RED = "#2f6db5", "#d9822b", "#3a9a5b", "#6b6b6b", "#b5432f"
numbers: dict[str, str] = {}
GRID = np.logspace(-2, 3, 11)
fmt = lambda n: f"{n:,}".replace(",", "{,}")

# --- Pretraining on generated problems (no human labels involved) -----------------
N_PRE = 5000
gen = [p.tree for p in Sampler(seed=777).sample_unique(N_PRE + 1000)]
pre_train, pre_test = gen[:N_PRE], gen[N_PRE:]
T_train, T_test = feature_matrix(pre_train), feature_matrix(pre_test)
t0 = time.time()
encoder, pre_losses = pretrain_encoder(pre_train, T_train, dim=16, epochs=400)
numbers["PreTime"] = f"{time.time() - t0:.0f}"
numbers["NPre"] = fmt(N_PRE)
numbers["PreNodes"] = fmt(compile_forest(pre_train).n_nodes)
# How well does the pretrained embedding predict each target on NEW generated problems? (R^2 of a linear readout)
with torch.no_grad():
    E_tr = encoder(compile_forest(pre_train)).numpy()
    E_te = encoder(compile_forest(pre_test)).numpy()
A = np.hstack([E_tr, np.ones((len(E_tr), 1))])
coef, *_ = np.linalg.lstsq(A, T_train, rcond=None)
pred = np.hstack([E_te, np.ones((len(E_te), 1))]) @ coef
r2 = 1 - ((pred - T_test) ** 2).sum(0) / ((T_test - T_test.mean(0)) ** 2).sum(0)
for name in ("deriv_size", "n_chain", "max_chain_depth", "n_quotient", "chain_in_product"):
    numbers["Rsq" + "".join(w.title() for w in name.split("_"))] = f"{r2[FEATURE_NAMES.index(name)]:.2f}"
numbers["RsqMin"] = f"{r2.min():.2f}"
numbers["RsqMinName"] = FEATURE_NAMES[int(np.argmin(r2))].replace("_", "\\_")
numbers["NParams"] = fmt(sum(p.numel() for p in encoder.parameters()) + 16)


# --- A. Synthetic: how much data does each approach need? ---------------------------
def lin_fit(Phi, p_, y_):
    lam, _ = cv_l2(Phi, p_, y_, np.unique(p_), GRID, np.random.default_rng(0))
    return FeatureBT(l2=lam).fit(Phi, p_, y_)


probs_s = Sampler(seed=41).sample_unique(500)
trees_s = [p.tree for p in probs_s]
Phi_s = feature_matrix(trees_s)
rng = np.random.default_rng(11)
delta_s = hidden_difficulty(Phi_s, rng)
tr_items, te_items = split_problems(500, 0.3, rng)
test_pairs = te_items[random_pairs(len(te_items), 4000, rng)]
test_y = simulate_judgments(delta_s, test_pairs, rng)
sizes = [100, 300, 1000, 3000, 10000]
syn = {"linear": [], "scratch": [], "lpft": []}
for m in sizes:
    vals = {k: [] for k in syn}
    for seed in range(2):
        r = np.random.default_rng(1000 * m + seed)
        tp = tr_items[random_pairs(len(tr_items), m, r)]
        ty = simulate_judgments(delta_s, tp, r)
        vals["linear"].append(log_loss(lin_fit(Phi_s, tp, ty).score(Phi_s), test_pairs, test_y))
        vals["scratch"].append(log_loss(TreeBT(trees_s, seed=seed).fit(tp, ty).score(), test_pairs, test_y))
        lp = TreeBT(trees_s, encoder=copy.deepcopy(encoder), seed=seed).fit_lp_ft(tp, ty, ft_lr=3e-4)
        vals["lpft"].append(log_loss(lp.score(), test_pairs, test_y))
    for k in syn:
        syn[k].append(np.mean(vals[k]))
oracle = log_loss(delta_s, test_pairs, test_y)
numbers["SynOracle"] = f"{oracle:.3f}"
for k, tag in (("linear", "Lin"), ("scratch", "Scratch"), ("lpft", "LPFT")):
    numbers[f"Syn{tag}Small"] = f"{syn[k][0]:.3f}"
    numbers[f"Syn{tag}Large"] = f"{syn[k][-1]:.3f}"
cross = [m for m, a, b in zip(sizes, syn["linear"], syn["scratch"]) if b <= a]
numbers["ScratchCatchUp"] = fmt(cross[0]) if cross else "more than 10{,}000"

# --- B. Your labels -------------------------------------------------------------------
store = LabelStore(ROOT / "data" / "labels.sqlite")
probs, pairs, y = store.to_arrays("dylan")
trees = [p.tree for p in probs]
Phi = feature_matrix(trees)
items = np.unique(pairs)
numbers["NJudg"] = str(len(y))
arms = {
    "linear, 17 features": lambda p_, y_: lin_fit(Phi, p_, y_),
    "tree net, from scratch": lambda p_, y_: TreeBT(trees).fit(p_, y_),
    "pretrained, frozen": lambda p_, y_: TreeBT(trees, encoder=copy.deepcopy(encoder), freeze_encoder=True).fit(p_, y_),
    "pretrained, naive fine-tune": lambda p_, y_: TreeBT(trees, encoder=copy.deepcopy(encoder), lr=3e-4,
                                                         patience=120, max_epochs=800).fit(p_, y_),
    "pretrained, LP-FT": lambda p_, y_: TreeBT(trees, encoder=copy.deepcopy(encoder)).fit_lp_ft(p_, y_, ft_lr=3e-4),
}
REPS = 3
diffs = {}
for name, make in arms.items():
    diffs[name] = np.mean([cross_val_diffs(Phi, pairs, y, make, np.random.default_rng(r)) for r in range(REPS)], axis=0)
with open(OUT / "step11_models.tex", "w") as f:
    f.write("\\begin{tabular}{lcc}\n\\toprule\nmodel & accuracy & log-loss \\\\\n\\midrule\n")
    for name, d in diffs.items():
        f.write(f"{name} & {accuracy_from_diff(d, y):.3f} & {log_loss_from_diff(d, y):.3f} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
dL, dT = diffs["linear, 17 features"], diffs["pretrained, LP-FT"]
lo, hi = bootstrap_ci(lambda w: log_loss_from_diff(dT, y, w) - log_loss_from_diff(dL, y, w),
                      pairs, items, np.random.default_rng(2), 1000)
numbers["LPFTvsLinLo"] = f"{lo:.3f}"
numbers["LPFTvsLinHi"] = f"{hi:.3f}"
numbers["RealLLLin"] = f"{log_loss_from_diff(dL, y):.3f}"
numbers["RealLLLPFT"] = f"{log_loss_from_diff(dT, y):.3f}"
numbers["RealLLScratch"] = f"{log_loss_from_diff(diffs['tree net, from scratch'], y):.3f}"
numbers["RealLLFrozen"] = f"{log_loss_from_diff(diffs['pretrained, frozen'], y):.3f}"
numbers["RealLLNaive"] = f"{log_loss_from_diff(diffs['pretrained, naive fine-tune'], y):.3f}"
numbers["RealAccScratch"] = f"{accuracy_from_diff(diffs['tree net, from scratch'], y):.3f}"

# --- C. Training curves on one fold of your labels ------------------------------------
held = np.random.default_rng(0).permutation(items)[: len(items) // 5]
te = pairs_touching(pairs, held)
scratch = TreeBT(trees, patience=150).fit(pairs[~te], y[~te])
hs = np.array(scratch.history_)
numbers["ScratchBestEpoch"] = str(scratch.best_epoch_)
numbers["ScratchTrainEnd"] = f"{hs[-1, 0]:.3f}"
numbers["ScratchValEnd"] = f"{hs[-1, 1]:.3f}"
numbers["ScratchValBest"] = f"{hs[:, 1].min():.3f}"
n_tr_pairs = int((~te).sum())
numbers["NTrPairsFold"] = str(n_tr_pairs)

# --- Figure ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7))
ax = axes[0]
ax.semilogy(np.arange(1, len(pre_losses) + 1), pre_losses, color=BLUE)
ax.set_xlabel("pretraining epoch")
ax.set_ylabel("MSE on standardized targets")
ax.set_title(f"(a) pretraining, {N_PRE} problems", loc="left", fontsize=8)
ax = axes[1]
ax.semilogx(sizes, syn["linear"], "o-", color=GREY, ms=3, label="linear, 17 features")
ax.semilogx(sizes, syn["scratch"], "s-", color=RED, ms=3, label="tree net, scratch")
ax.semilogx(sizes, syn["lpft"], "^-", color=BLUE, ms=3, label="tree net, pretrained (LP-FT)")
ax.axhline(oracle, color="k", ls="--", lw=0.8, label="oracle")
ax.set_xlabel("training comparisons")
ax.set_ylabel("held-out log-loss")
ax.legend(frameon=False, fontsize=6)
ax.set_title("(b) data hunger (simulated)", loc="left", fontsize=8)
ax = axes[2]
ep = np.arange(1, len(hs) + 1)
ax.plot(ep, hs[:, 0], color=RED, ls=":", label="scratch: training")
ax.plot(ep, hs[:, 1], color=RED, label="scratch: validation")
ax.axvline(scratch.best_epoch_, color=GREY, lw=0.6)
ax.set_xlabel("epoch")
ax.set_ylabel("log-loss (your labels)")
ax.legend(frameon=False, fontsize=6)
ax.set_title("(c) memorizing your labels", loc="left", fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "step11_treenet.pdf")

with open(OUT / "step11_numbers.tex", "w") as f:
    for k, v in numbers.items():
        f.write(f"\\newcommand{{\\N{k}}}{{{v}}}\n")
print("\n".join(f"{k} = {v}" for k, v in numbers.items()))
print(open(OUT / "step11_models.tex").read())
