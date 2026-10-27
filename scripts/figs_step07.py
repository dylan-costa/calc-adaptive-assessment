"""Figures and numbers for Section 7 of the lecture notes (labels and label noise).

Run:  .venv/Scripts/python scripts/figs_step07.py
Writes notes/figures/step07_*.{pdf,tex}.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq
from scipy.special import expit
from scipy.stats import norm

OUT = Path(__file__).resolve().parents[1] / "notes" / "figures"
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False})
BLUE, ORANGE, GREY = "#2f6db5", "#d9822b", "#6b6b6b"
numbers: dict[str, str] = {}

# --- (a) logistic vs probit link -------------------------------------------------
t = np.linspace(-6, 6, 400)
# best scale c so that Phi(t / c) ~ sigma(t), in sup norm
grid = np.linspace(1.4, 2.0, 601)
c_best = grid[np.argmin([np.max(np.abs(norm.cdf(t / c) - expit(t))) for c in grid])]
gap = np.max(np.abs(norm.cdf(t / c_best) - expit(t)))
numbers["ProbitScale"] = f"{c_best:.3f}"
numbers["ProbitGap"] = f"{gap:.4f}"

# --- (b) self-agreement on repeats vs best achievable accuracy -----------------------
# A rater with difficulty spread s judges a random pair (gap D ~ N(0, 2 s^2)) correctly
# with prob p = sigma(|D|), independently on each showing. Then
#   agreement on a repeat  A(s) = E[p^2 + (1 - p)^2]
#   ceiling on accuracy    C(s) = E[p]          (= E[max(p, 1 - p)])
z = np.random.default_rng(0).standard_normal(400_000)
s_grid = np.linspace(0.01, 5, 200)


def agree_and_ceiling(s):
    p = expit(np.abs(np.sqrt(2) * s * z))
    return np.mean(p**2 + (1 - p) ** 2), np.mean(p)


AC = np.array([agree_and_ceiling(s) for s in s_grid])
WORDS = {0.70: "Seventy", 0.80: "Eighty", 0.90: "Ninety"}  # TeX macro names cannot contain digits
for A_obs, word in WORDS.items():
    s_hat = brentq(lambda s: agree_and_ceiling(s)[0] - A_obs, 0.01, 20)
    numbers[f"CeilAt{word}"] = f"{agree_and_ceiling(s_hat)[1]:.3f}"
    numbers[f"SAt{word}"] = f"{s_hat:.2f}"

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
ax = axes[0]
ax.plot(t, expit(t), color=BLUE, label=r"logistic $\sigma(t)$")
ax.plot(t, norm.cdf(t / c_best), "--", color=ORANGE, label=rf"probit $\Phi(t/{c_best:.2f})$")
ax.set_xlabel("difficulty gap $t$ (logits)")
ax.set_ylabel("P(judged harder)")
ax.legend(frameon=False, fontsize=7)
ax.set_title("(a) two links, nearly the same", loc="left")
ax = axes[1]
ax.plot(AC[:, 0], AC[:, 1], color=BLUE)
ax.plot([0.5, 1], [0.5, 1], ":", color=GREY, lw=0.8)
for A_obs, word in WORDS.items():
    c = float(numbers[f"CeilAt{word}"])
    ax.plot([A_obs], [c], "o", color=ORANGE, ms=3)
ax.set_xlabel("self-agreement on repeated pairs")
ax.set_ylabel("best achievable accuracy")
ax.set_title("(b) reading your own consistency", loc="left")
fig.tight_layout()
fig.savefig(OUT / "step07_noise.pdf")

with open(OUT / "step07_numbers.tex", "w") as f:
    for k, v in numbers.items():
        f.write(f"\\newcommand{{\\H{k}}}{{{v}}}\n")
print("\n".join(f"{k} = {v}" for k, v in numbers.items()))
