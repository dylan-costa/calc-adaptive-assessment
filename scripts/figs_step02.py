"""Figures and numbers for Section 2 of the lecture notes (the step-tracing differentiator).

Run:  .venv/Scripts/python scripts/figs_step02.py
Writes notes/figures/step02_*.{pdf,tex}.
"""

from __future__ import annotations

import warnings
from fractions import Fraction as Q
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from scipy.stats import spearmanr

from calcdiff.check import numerically_equal
from calcdiff.expr import X, X_SYM, Const, Fn, Mul, Pow, size, to_latex, to_sympy
from calcdiff.generate import Sampler
from calcdiff.solver_trace import derive

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parents[1] / "notes" / "figures"
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False})
BLUE, ORANGE = "#2f6db5", "#d9822b"
numbers: dict[str, str] = {}

# --- Worked example: a trace, printed as a table -----------------------------
ex = Mul((Pow(X, Q(2)), Fn("sin", Fn("exp", Mul((Const(Q(3)), X))))))
d, tr = derive(ex)
with open(OUT / "step02_trace.tex", "w") as f:
    f.write("\\begin{tabular}{rccll}\n\\toprule\n"
            "\\# & depth & chain & rule & applied to \\\\\n\\midrule\n")
    for i, s in enumerate(tr.steps, 1):
        rule = s.rule.replace("_", "\\_")
        f.write(f"{i} & {s.depth} & {s.chain_depth} & \\code{{{rule}}} & ${to_latex(s.node)}$ \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
numbers["ExProblem"] = to_latex(ex)
numbers["ExDerivative"] = to_latex(d)

# --- Statistics over many generated problems ----------------------------------
probs = Sampler(seed=42).sample_unique(3000)
rows = []
verdicts = []
for p in probs:
    dp, t = derive(p.tree)
    verdicts.append(numerically_equal(to_sympy(dp), sp.diff(to_sympy(p.tree), X_SYM)))
    rows.append((p.n_expansions, size(p.tree), size(dp), t.n_steps, t.max_chain_depth, t.counts()))
N, s_in, s_out, steps, chain = (np.array([r[i] for r in rows]) for i in range(5))
counts = [r[5] for r in rows]

numbers["NProblems"] = f"{len(probs):,}".replace(",", "{,}")
assert False not in verdicts
numbers["NVerified"] = f"{verdicts.count(True):,}".replace(",", "{,}")
numbers["NInconclusive"] = str(verdicts.count(None))
numbers["RhoNSteps"] = f"{spearmanr(N, steps)[0]:.2f}"
numbers["RhoInOut"] = f"{spearmanr(s_in, s_out)[0]:.2f}"
numbers["MeanSteps"] = f"{steps.mean():.1f}"
numbers["MaxSwell"] = f"{(s_out / s_in).max():.1f}"
numbers["MedSwell"] = f"{np.median(s_out / s_in):.2f}"

# Log-log slope of derivative size against input size, among problems with a product/quotient
has_prod = np.array([c["PRODUCT"] + c["QUOTIENT"] > 0 for c in counts])
slope = np.polyfit(np.log(s_in), np.log(s_out), 1)[0]
numbers["SwellSlope"] = f"{slope:.2f}"

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
ax = axes[0]
jit = np.random.default_rng(0).uniform(-0.25, 0.25, len(s_in))
ax.scatter(s_in[~has_prod] + jit[~has_prod], s_out[~has_prod], s=5, alpha=0.35, color=BLUE,
           label="no product/quotient")
ax.scatter(s_in[has_prod] + jit[has_prod], s_out[has_prod], s=5, alpha=0.35, color=ORANGE,
           label="product or quotient")
ax.set_xlabel("size of problem (nodes)")
ax.set_ylabel("size of derivative (nodes)")
ax.legend(frameon=False, markerscale=3, fontsize=8)
ax.set_title("(a)", loc="left")

ax = axes[1]
show = ["POWER", "CHAIN", "SUM", "PRODUCT", "QUOTIENT", "CONST_MULT"]
frac = [np.mean([c[r] > 0 for c in counts]) for r in show]
ax.barh(range(len(show)), frac, color=BLUE)
ax.set_yticks(range(len(show)), [r.replace("_", " ").lower() for r in show])
ax.invert_yaxis()
ax.set_xlabel("fraction of problems using rule")
ax.set_title("(b)", loc="left")
fig.tight_layout()
fig.savefig(OUT / "step02_trace_stats.pdf")

for r in ("CHAIN", "PRODUCT", "QUOTIENT"):
    numbers[f"Frac{r.title()}"] = f"{np.mean([c[r] > 0 for c in counts]):.2f}"
numbers["FracChainTwo"] = f"{(chain >= 2).mean():.2f}"

with open(OUT / "step02_numbers.tex", "w") as f:
    for k, v in numbers.items():
        f.write(f"\\newcommand{{\\T{k}}}{{{v}}}\n")
print("\n".join(f"{k} = {v}" for k, v in numbers.items()))
