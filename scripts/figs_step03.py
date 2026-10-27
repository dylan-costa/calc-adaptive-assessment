"""Figures and numbers for Section 3 of the lecture notes (features).

Run:  .venv/Scripts/python scripts/figs_step03.py
Writes notes/figures/step03_*.{pdf,tex}.
"""

from __future__ import annotations

from fractions import Fraction as Q
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from calcdiff.expr import X, Const, Fn, Mul, Pow
from calcdiff.features import FEATURE_NAMES, Standardizer, feature_matrix, featurize
from calcdiff.generate import Sampler

OUT = Path(__file__).resolve().parents[1] / "notes" / "figures"
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False})
BLUE = "#2f6db5"
numbers: dict[str, str] = {}

# --- The worked example's feature vector --------------------------------------
ex = Mul((Pow(X, Q(2)), Fn("sin", Fn("exp", Mul((Const(Q(3)), X))))))
phi = featurize(ex)
with open(OUT / "step03_example.tex", "w") as f:
    cells = [f"\\code{{{n.replace('_', chr(92) + '_')}}} & {int(v)}" for n, v in zip(FEATURE_NAMES, phi)]
    half = (len(cells) + 1) // 2
    left, right = cells[:half], cells[half:] + [" & "]
    f.write("\\begin{tabular}{lr@{\\qquad}lr}\n\\toprule\n")
    for a, b in zip(left, right):
        f.write(f"{a} & {b} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# --- Geometry of the feature matrix -------------------------------------------
probs = Sampler(seed=42).sample_unique(3000)
Phi = feature_matrix(p.tree for p in probs)
Z = Standardizer().fit_transform(Phi)
C = np.corrcoef(Phi, rowvar=False)
eig = np.sort(np.linalg.eigvalsh(C))[::-1]
explained = np.cumsum(eig) / eig.sum()

numbers["NFeatures"] = str(len(FEATURE_NAMES))
numbers["NProblems"] = f"{len(probs):,}".replace(",", "{,}")
numbers["Cond"] = f"{eig[0] / eig[-1]:.0f}"
numbers["EigMax"] = f"{eig[0]:.2f}"
numbers["EigMin"] = f"{eig[-1]:.3f}"
numbers["NinetyPct"] = str(int(np.searchsorted(explained, 0.90) + 1))
iu = np.triu_indices_from(C, k=1)
k = np.argmax(np.abs(C[iu]))
a, b = iu[0][k], iu[1][k]
numbers["TopPairA"] = FEATURE_NAMES[a].replace("_", "\\_")
numbers["TopPairB"] = FEATURE_NAMES[b].replace("_", "\\_")
numbers["TopCorr"] = f"{C[a, b]:.2f}"
numbers["CorrSizePower"] = f"{C[FEATURE_NAMES.index('size'), FEATURE_NAMES.index('n_power')]:.2f}"
numbers["CorrChainProd"] = f"{C[FEATURE_NAMES.index('n_chain'), FEATURE_NAMES.index('n_product')]:.2f}"
uniq, inv, cnt = np.unique(Phi, axis=0, return_inverse=True, return_counts=True)
numbers["NDistinct"] = f"{len(uniq):,}".replace(",", "{,}")
biggest = np.argmax(cnt)
numbers["BiggestClass"] = str(int(cnt[biggest]))
twins = [probs[i] for i in np.flatnonzero(inv.ravel() == biggest)[:2]]
numbers["TwinA"] = twins[0].latex
numbers["TwinB"] = twins[1].latex
# a pair of genuinely different-looking problems with identical features
for c in np.flatnonzero(cnt >= 2):
    members = [probs[i] for i in np.flatnonzero(inv.ravel() == c)]
    if Phi[np.flatnonzero(inv.ravel() == c)[0], FEATURE_NAMES.index("n_chain")] >= 1 and len(members[0].latex) > 12:
        numbers["CollideA"], numbers["CollideB"] = members[0].latex, members[1].latex
        break
numbers["FracCIPZero"] = f"{np.mean(Phi[:, FEATURE_NAMES.index('chain_in_product')] == 0):.2f}"

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), gridspec_kw={"width_ratios": [1.35, 1]})
ax = axes[0]
im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(FEATURE_NAMES)), FEATURE_NAMES, rotation=90, fontsize=6)
ax.set_yticks(range(len(FEATURE_NAMES)), FEATURE_NAMES, fontsize=6)
ax.set_title("(a) feature correlations", loc="left")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax = axes[1]
ax.semilogy(range(1, len(eig) + 1), eig, "o-", color=BLUE, ms=3)
ax.set_xlabel("index $i$")
ax.set_ylabel(r"eigenvalue $\lambda_i$ of correlation matrix")
ax.set_title("(b) spectrum", loc="left")
fig.tight_layout()
fig.savefig(OUT / "step03_features.pdf")

with open(OUT / "step03_numbers.tex", "w") as f:
    for k_, v in numbers.items():
        f.write(f"\\newcommand{{\\F{k_}}}{{{v}}}\n")
print("\n".join(f"{k_} = {v}" for k_, v in numbers.items()))
