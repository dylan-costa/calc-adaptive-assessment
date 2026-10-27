"""Figures and numbers for Section 1 of the lecture notes (problem generation).

Run:  .venv/Scripts/python scripts/figs_step01.py
Writes notes/figures/step01_*.{pdf,tex}.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from calcdiff.expr import size
from calcdiff.generate import GrammarConfig, Sampler
from calcdiff.generate.grammar import is_natural, log_likelihood_ratio

OUT = Path(__file__).resolve().parents[1] / "notes" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False})
BLUE, ORANGE, GREY = "#2f6db5", "#d9822b", "#6b6b6b"

numbers: dict[str, str] = {}


def dwass_pmf(offspring: np.ndarray, n_max: int) -> np.ndarray:
    """P(N = n) = (1/n) P(S_n = n - 1), S_n a sum of n i.i.d. offspring counts."""
    pmf = np.zeros(n_max + 1)
    conv = np.array([1.0])
    for n in range(1, n_max + 1):
        conv = np.convolve(conv, offspring)  # law of S_n
        pmf[n] = conv[n - 1] / n
    return pmf


# --- Figure 1: total progeny of the branching process -----------------------
N_SAMPLES = 50_000
cfg_untr = GrammarConfig(max_depth=None)
cfg_tr = GrammarConfig()  # max_depth = 4
s_untr, s_tr = Sampler(cfg_untr, seed=0), Sampler(cfg_tr, seed=0)
n_untr = np.array([s_untr.sample_factor_tree().n_expansions for _ in range(N_SAMPLES)])
n_tr = np.array([s_tr.sample_factor_tree().n_expansions for _ in range(N_SAMPLES)])

m = cfg_untr.mean_offspring
n_max = 30
theory = dwass_pmf(cfg_untr.offspring_dist, n_max)
ns = np.arange(1, n_max + 1)
emp_untr = np.array([(n_untr == n).mean() for n in ns])
emp_tr = np.array([(n_tr == n).mean() for n in ns])

fig, ax = plt.subplots(figsize=(3.6, 2.9))
ax.semilogy(ns, theory[1:], "-", color=GREY, lw=1.5, label=r"Dwass $\frac{1}{n}P(S_n=n-1)$")
ax.semilogy(ns, emp_untr, "o", color=BLUE, ms=4, label="sampled, untruncated")
ax.semilogy(ns[emp_tr > 0], emp_tr[emp_tr > 0], "s", color=ORANGE, ms=4, mfc="none",
            label=r"sampled, depth $\leq 4$")
ax.set_xlabel(r"total progeny $N$ (number of $F$-nodes)")
ax.set_ylabel(r"$P(N=n)$")
ax.legend(frameon=False, fontsize=7)
fig.tight_layout()
fig.savefig(OUT / "step01_progeny.pdf")

numbers["MeanOffspring"] = f"{m:.2f}"
numbers["TheoryMeanN"] = f"{1 / (1 - m):.3f}"
numbers["EmpMeanN"] = f"{n_untr.mean():.3f}"
numbers["EmpMeanNTrunc"] = f"{n_tr.mean():.3f}"
numbers["EmpMaxN"] = str(int(n_untr.max()))
numbers["NSamples"] = f"{N_SAMPLES:,}".replace(",", "{,}")

# --- Acceptance rate of the rejection step ------------------------------------
s_raw = Sampler(cfg_tr, seed=1)
raw = [s_raw.sample_raw() for _ in range(20_000)]
nat = np.array([is_natural(p.tree) for p in raw])
small = np.array([size(p.tree) <= cfg_tr.max_size for p in raw])
numbers["AcceptRate"] = f"{(nat & small).mean():.3f}"
numbers["RejectUnnatural"] = f"{(~nat).mean():.3f}"
numbers["RejectSize"] = f"{(nat & ~small).mean():.3f}"

# --- Figure 2: covariate shift and importance weighting ----------------------
p_cfg = GrammarConfig()
q_cfg = GrammarConfig(p_leaf=0.38, p_fn=0.30, p_powf=0.12, p_prod=0.12, p_quot=0.08)
M = 20_000
sp_, sq_ = Sampler(p_cfg, seed=2), Sampler(q_cfg, seed=3)
from_p = [sp_.sample() for _ in range(M)]
from_q = [sq_.sample() for _ in range(M)]


def chain_count(p) -> int:
    return p.rule_counts["FN"] + p.rule_counts["POWF"]


cp = np.array([chain_count(x) for x in from_p])
cq = np.array([chain_count(x) for x in from_q])
logw = np.array([log_likelihood_ratio(x, p_cfg, q_cfg) for x in from_p])
w = np.exp(logw - logw.max())
w /= w.sum()
ess = 1.0 / np.sum(w**2)

ks = np.arange(0, 7)
hist_p = np.array([(cp == k).mean() for k in ks])
hist_q = np.array([(cq == k).mean() for k in ks])
hist_w = np.array([w[cp == k].sum() for k in ks])

fig, ax = plt.subplots(figsize=(3.6, 2.9))
wd = 0.27
ax.bar(ks - wd, hist_p, wd, color=BLUE, label=r"train grammar $p$")
ax.bar(ks, hist_q, wd, color=ORANGE, label=r"target grammar $q$")
ax.bar(ks + wd, hist_w, wd, color="none", edgecolor="k", hatch="///", label=r"$p$ reweighted by $q/p$")
ax.set_xlabel(r"compositions (FN + POWF nodes)")
ax.set_ylabel("probability")
ax.legend(frameon=False, fontsize=7)
fig.tight_layout()
fig.savefig(OUT / "step01_shift.pdf")

numbers["QMeanOffspring"] = f"{q_cfg.mean_offspring:.2f}"
numbers["ChainMeanP"] = f"{cp.mean():.3f}"
numbers["ChainMeanQ"] = f"{cq.mean():.3f}"
numbers["ChainMeanIS"] = f"{(w * cp).sum():.3f}"
numbers["TailP"] = f"{(cp >= 5).mean():.3f}"
numbers["TailQ"] = f"{(cq >= 5).mean():.3f}"
numbers["TailQ"] = f"{(cq >= 5).mean():.3f}"
numbers["ESS"] = f"{ess:,.0f}".replace(",", "{,}")
numbers["ISM"] = f"{M:,}".replace(",", "{,}")

# --- Sample problems, sorted by derivation size -------------------------------
probs = Sampler(cfg_tr, seed=2026).sample_unique(400)
by_n: dict[int, list] = {}
for p in probs:
    by_n.setdefault(p.n_expansions, []).append(p)
chosen = []
for n in sorted(by_n):
    chosen.extend(by_n[n][:1])
chosen = chosen[:8]
with open(OUT / "step01_samples.tex", "w") as f:
    f.write("{\\renewcommand{\\arraystretch}{1.6}\\begin{tabular}{rrl}\n\\toprule\n"
            "$N$ & $\\log p$ & problem: find $\\frac{d}{dx}$ of \\\\\n\\midrule\n")
    for p in chosen:
        f.write(f"{p.n_expansions} & ${p.log_prob:.1f}$ & $\\displaystyle {p.latex}$ \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}}\n")

numbers["UniqueOf"] = str(len(probs))

with open(OUT / "step01_numbers.tex", "w") as f:
    for k, v in numbers.items():
        f.write(f"\\newcommand{{\\S{k}}}{{{v}}}\n")

print("\n".join(f"{k} = {v}" for k, v in numbers.items()))
