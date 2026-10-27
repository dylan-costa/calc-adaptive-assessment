"""Figures and numbers for Section 8 of the lecture notes (the first fit on real labels).

Run:  .venv/Scripts/python scripts/figs_step08.py
Reads data/labels.sqlite; writes notes/figures/step08_*.{pdf,tex}.
"""

from __future__ import annotations

import warnings
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit
from scipy.stats import beta as beta_dist
from scipy.stats import spearmanr

from calcdiff.evaluate import (accuracy_from_diff, bootstrap_ci, cross_val_diffs, log_loss_from_diff,
                               pairs_touching)
from calcdiff.features import FEATURE_NAMES, feature_matrix
from calcdiff.models.bt import grad_hess
from calcdiff.models.feature_bt import FeatureBT, cv_l2, problem_bootstrap

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notes" / "figures"
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False})
BLUE, ORANGE, GREEN, GREY = "#2f6db5", "#d9822b", "#3a9a5b", "#6b6b6b"
numbers: dict[str, str] = {}
fmt = lambda n: f"{n:,}".replace(",", "{,}")

from calcdiff.labels.store import LabelStore  # noqa: E402

store = LabelStore(ROOT / "data" / "labels.sqlite")
LABELER = "dylan"
probs, pairs, y = store.to_arrays(LABELER)
Phi = feature_matrix(p.tree for p in probs)
GRID = np.logspace(-2, 3, 11)
DERIV = [FEATURE_NAMES.index("deriv_size")]

# --- The data ----------------------------------------------------------------------
js = store.judgments(LABELER)
rt = np.array([j.response_ms for j in js if j.response_ms]) / 1000
decided = y != 0.5
numbers["NJudg"] = fmt(len(js))
numbers["NRepeats"] = str(sum(j.is_repeat for j in js))
numbers["NTies"] = str(int((y == 0.5).sum()))
numbers["TiePct"] = f"{100 * (y == 0.5).mean():.0f}"
numbers["NProblemsSeen"] = str(len(np.unique(pairs)))
numbers["NPool"] = str(len(probs))
numbers["LeftRate"] = f"{np.mean(y[decided] == 1):.3f}"
numbers["MedianRT"] = f"{np.median(rt):.1f}"
numbers["NFlags"] = str(len(store.flagged(LABELER)))
numbers["NAutoFlags"] = str(len(store.flagged("auto:is_natural")))

# Repeats, split by whether ties were involved
first, kinds = {}, []
for j in js:
    key = frozenset((j.left_id, j.right_id))
    if j.is_repeat and key in first:
        a = first[key]
        kind = ("tie" if a.winner is None else "dec") + "/" + ("tie" if j.winner is None else "dec")
        kinds.append((kind, a.winner == j.winner))
    elif not j.is_repeat:
        first.setdefault(key, j)
agree_all = sum(ok for _, ok in kinds)
dd = [ok for k, ok in kinds if k == "dec/dec"]
with_tie = sum(1 for k, ok in kinds if "tie" in k and not ok)
numbers["AgreeAll"] = f"{agree_all}/{len(kinds)}"
numbers["AgreeDD"] = f"{sum(dd)}/{len(dd)}"
numbers["DisagreeTie"] = str(with_tie)
numbers["DisagreeTotal"] = str(len(kinds) - agree_all)
# Jeffreys interval for the decided-decided agreement rate
a_hat = sum(dd) / len(dd)
lo, hi = beta_dist.ppf([0.025, 0.975], sum(dd) + 0.5, len(dd) - sum(dd) + 0.5)
numbers["AgreeDDRate"] = f"{a_hat:.2f}"
numbers["AgreeDDLo"] = f"{lo:.2f}"
numbers["AgreeDDHi"] = f"{hi:.2f}"

# Map agreement -> accuracy ceiling (same model as Section 7)
z = np.random.default_rng(0).standard_normal(200_000)


def ceiling_from_agreement(A: float) -> float:
    s_lo, s_hi = 0.01, 30.0
    for _ in range(60):  # bisection on the monotone map s -> agreement
        s = 0.5 * (s_lo + s_hi)
        p = expit(np.abs(np.sqrt(2) * s * z))
        if np.mean(p**2 + (1 - p) ** 2) < A:
            s_lo = s
        else:
            s_hi = s
    return float(np.mean(expit(np.abs(np.sqrt(2) * s * z))))


numbers["CeilDD"] = f"{ceiling_from_agreement(a_hat):.2f}"
numbers["CeilDDLo"] = f"{ceiling_from_agreement(lo):.2f}"
numbers["CeilDDHi"] = f"{ceiling_from_agreement(min(hi, 0.995)):.2f}"


# --- Model comparison by cross-validation over problems ------------------------
def full_model(tr_pairs, tr_y):
    lam, _ = cv_l2(Phi, tr_pairs, tr_y, np.unique(tr_pairs), GRID, np.random.default_rng(0))
    return FeatureBT(l2=lam).fit(Phi, tr_pairs, tr_y)


def single(col):
    return lambda tr_pairs, tr_y: FeatureBT(l2=1e-3, columns=[FEATURE_NAMES.index(col)]).fit(Phi, tr_pairs, tr_y)


class Constant:
    def score(self, P):
        return np.zeros(len(P))


models = {
    "constant (coin flip)": lambda p_, y_: Constant(),
    "\\code{size} only": single("size"),
    "\\code{n\\_chain} only": single("n_chain"),
    "\\code{depth} only": single("depth"),
    "\\code{deriv\\_size} only": single("deriv_size"),
    "all 17 features, CV ridge": full_model,
}
REPS = 5
results = {}
for name, make in models.items():
    diffs = [cross_val_diffs(Phi, pairs, y, make, np.random.default_rng(r)) for r in range(REPS)]
    accs = [accuracy_from_diff(d, y) for d in diffs]
    lls = [log_loss_from_diff(d, y) for d in diffs]
    results[name] = (np.mean(accs), np.mean(lls), np.mean(diffs, axis=0))

# Problem-bootstrap intervals for the two serious models
items = np.unique(pairs)
for key, tag in (("all 17 features, CV ridge", "Full"), ("\\code{deriv\\_size} only", "Deriv")):
    d = results[key][2]
    lo_, hi_ = bootstrap_ci(lambda w: accuracy_from_diff(d, y, w), pairs, items, np.random.default_rng(1), 1000)
    numbers[f"Acc{tag}"] = f"{results[key][0]:.3f}"
    numbers[f"LL{tag}"] = f"{results[key][1]:.3f}"
    numbers[f"Acc{tag}Lo"] = f"{lo_:.3f}"
    numbers[f"Acc{tag}Hi"] = f"{hi_:.3f}"
    lo_, hi_ = bootstrap_ci(lambda w: log_loss_from_diff(d, y, w), pairs, items, np.random.default_rng(1), 1000)
    numbers[f"LL{tag}Lo"] = f"{lo_:.3f}"
    numbers[f"LL{tag}Hi"] = f"{hi_:.3f}"
# Paired comparison of log-loss: full minus deriv, bootstrapped over problems
dF, dD = results["all 17 features, CV ridge"][2], results["\\code{deriv\\_size} only"][2]
lo_, hi_ = bootstrap_ci(lambda w: log_loss_from_diff(dF, y, w) - log_loss_from_diff(dD, y, w),
                        pairs, items, np.random.default_rng(2), 1000)
numbers["LLDiffLo"] = f"{lo_:.3f}"
numbers["LLDiffHi"] = f"{hi_:.3f}"

with open(OUT / "step08_models.tex", "w") as f:
    f.write("\\begin{tabular}{lcc}\n\\toprule\nmodel & accuracy & log-loss \\\\\n\\midrule\n")
    for name, (acc, ll, _) in results.items():
        f.write(f"{name} & {acc:.3f} & {ll:.3f} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# --- Learning curve: train vs held-out log-loss, against the Akaike/TIC prediction -----
sizes = [25, 50, 100, 150, 200]
curve = {"full": [], "deriv": []}
for m in sizes:
    for tag in curve:
        tr_ll, te_ll, te_acc, gaps_pred = [], [], [], []
        for rep in range(20):
            r = np.random.default_rng(100 * m + rep)
            chunks = np.array_split(r.permutation(items), 5)
            held = chunks[0]
            te = pairs_touching(pairs, held)
            tr_idx = np.flatnonzero(~te)
            sub = r.choice(tr_idx, size=min(m, len(tr_idx)), replace=False)
            if tag == "full":
                lam, _ = cv_l2(Phi, pairs[sub], y[sub], np.unique(pairs[sub]), GRID, r)
                mdl = FeatureBT(l2=lam).fit(Phi, pairs[sub], y[sub])
            else:
                mdl = FeatureBT(l2=1e-3, columns=DERIV).fit(Phi, pairs[sub], y[sub])
            s = mdl.score(Phi)
            dtr = s[pairs[sub, 0]] - s[pairs[sub, 1]]
            dte = s[pairs[te, 0]] - s[pairs[te, 1]]
            tr_ll.append(log_loss_from_diff(dtr, y[sub]))
            te_ll.append(log_loss_from_diff(dte, y[te]))
            te_acc.append(accuracy_from_diff(dte, y[te]))
            # effective degrees of freedom of a ridge fit: tr(H_data (H_data + lambda I)^-1)
            Z = mdl.scaler_.transform(mdl._select(Phi))
            D = Z[pairs[sub, 0]] - Z[pairs[sub, 1]]
            _, H_data = grad_hess(mdl.theta_, D, y[sub], 0.0)
            df = np.trace(H_data @ np.linalg.inv(H_data + mdl.l2 * np.eye(len(H_data))))
            gaps_pred.append(df / len(sub))
        curve[tag].append((np.mean(tr_ll), np.mean(te_ll), np.mean(te_acc), np.mean(gaps_pred)))
c_full = np.array(curve["full"])
c_der = np.array(curve["deriv"])
numbers["DFAtTwoHundred"] = f"{c_full[-1, 3] * 200:.1f}"
numbers["GapPredTwoHundred"] = f"{c_full[-1, 3]:.3f}"
numbers["GapObsTwoHundred"] = f"{c_full[-1, 1] - c_full[-1, 0]:.3f}"
numbers["GapPredTwentyFive"] = f"{c_full[0, 3]:.3f}"
numbers["GapObsTwentyFive"] = f"{c_full[0, 1] - c_full[0, 0]:.3f}"
numbers["AccFullTwentyFive"] = f"{c_full[0, 2]:.3f}"
numbers["AccDerivTwentyFive"] = f"{c_der[0, 2]:.3f}"
numbers["LLFullTwentyFive"] = f"{c_full[0, 1]:.3f}"
numbers["LLDerivTwentyFive"] = f"{c_der[0, 1]:.3f}"

# --- Final model on all labels: weights --------------------------------------------
lam_all, _ = cv_l2(Phi, pairs, y, items, GRID, np.random.default_rng(0))
final = FeatureBT(l2=lam_all).fit(Phi, pairs, y)
w, se = final.raw_weights(), final.raw_weight_se()
# Laplace SEs assume independent comparisons; resample problems instead (Section 6)
boot = problem_bootstrap(Phi, pairs, y, lam_all, np.random.default_rng(9), n_boot=500)
se_boot = boot.std(axis=0)
zs = w / se_boot
numbers["LamAll"] = f"{lam_all:g}"
iq = FEATURE_NAMES.index("n_quotient")
numbers["QuotW"] = f"{w[iq]:.1f}"
numbers["QuotLapLo"] = f"{w[iq] - 1.96 * se[iq]:.1f}"
numbers["QuotLapHi"] = f"{w[iq] + 1.96 * se[iq]:.1f}"
qlo, qhi = np.quantile(boot[:, iq], [0.025, 0.975])
numbers["QuotLo"] = f"{qlo:.1f}"
numbers["QuotHi"] = f"{qhi:.1f}"
numbers["QuotFracPos"] = f"{100 * np.mean(boot[:, iq] > 0):.1f}"
numbers["RealSERatio"] = f"{np.median(se_boot / se):.1f}"
numbers["NSig"] = str(int(np.sum(np.abs(zs) > 1.96)))
# How much does the quotient weight depend on the regularization?
for lam_, tag in ((0.316, "SmallLam"), (31.6, "LargeLam")):
    numbers[f"Quot{tag}"] = f"{FeatureBT(l2=lam_).fit(Phi, pairs, y).raw_weights()[iq]:.1f}"
# standardized weights tell "which features carry the prediction"
std_w = final.theta_
top = np.argsort(-np.abs(std_w))[:4]
numbers["TopStd"] = ", ".join(f"\\code{{{FEATURE_NAMES[i].replace('_', chr(92) + '_')}}}" for i in top)
scores = final.score(Phi)
seen = np.unique(pairs)
numbers["RhoScoreDeriv"] = f"{spearmanr(scores[seen], Phi[seen, DERIV[0]])[0]:.2f}"

# --- Robustness: drop judgments involving problems the fixed generator rejects ------
from calcdiff.generate.grammar import is_natural  # noqa: E402

bad = np.array([not is_natural(p.tree) for p in probs])
keep = ~(bad[pairs[:, 0]] | bad[pairs[:, 1]])
numbers["NDropped"] = str(int((~keep).sum()))
d_keep = [cross_val_diffs(Phi, pairs[keep], y[keep], full_model, np.random.default_rng(r)) for r in range(REPS)]
numbers["AccKeep"] = f"{np.mean([accuracy_from_diff(d, y[keep]) for d in d_keep]):.3f}"
numbers["LLKeep"] = f"{np.mean([log_loss_from_diff(d, y[keep]) for d in d_keep]):.3f}"
lam_keep, _ = cv_l2(Phi, pairs[keep], y[keep], np.unique(pairs[keep]), GRID, np.random.default_rng(0))
m_keep = FeatureBT(l2=lam_keep).fit(Phi, pairs[keep], y[keep])
wq, sq = m_keep.raw_weights()[iq], m_keep.raw_weight_se()[iq]
numbers["QuotKeep"] = f"{wq:.1f}"
numbers["QuotKeepLo"] = f"{wq - 1.96 * sq:.1f}"
numbers["QuotKeepHi"] = f"{wq + 1.96 * sq:.1f}"
numbers["FracQuot"] = f"{100 * np.mean(Phi[np.unique(pairs), iq] > 0):.0f}"

# --- Ties and response times vs predicted gap (out-of-fold) ----------------------
d_oof = results["all 17 features, CV ridge"][2]
gap = np.abs(d_oof)
bins = np.quantile(gap, [0, 0.25, 0.5, 0.75, 1.0])
which = np.clip(np.digitize(gap, bins[1:-1]), 0, 3)
tie_rate = [np.mean(y[which == b] == 0.5) for b in range(4)]
# align response times with pairs (to_arrays keeps judgment order, repeats included)
rts = np.array([j.response_ms / 1000 if j.response_ms else np.nan for j in js])
rts = np.where(rts > 120, np.nan, rts)  # a break, not a judgment
med_rt = [np.nanmedian(rts[which == b]) for b in range(4)]
numbers["TieLowGap"] = f"{100 * tie_rate[0]:.0f}"
numbers["TieHighGap"] = f"{100 * tie_rate[-1]:.0f}"
numbers["RTLowGap"] = f"{med_rt[0]:.1f}"
numbers["RTHighGap"] = f"{med_rt[-1]:.1f}"

# --- Most confident out-of-fold mistakes -------------------------------------------
wrong = decided & ((d_oof > 0) != (y == 1))
order = np.argsort(-np.abs(d_oof) * wrong)[:5]
with open(OUT / "step08_errors.tex", "w") as f:
    f.write("{\\renewcommand{\\arraystretch}{1.7}\\begin{tabular}{llc}\n\\toprule\n"
            "you said harder & model said harder & $P_{\\text{model}}$ \\\\\n\\midrule\n")
    for k in order:
        i, j = pairs[k]
        hard, easy = (i, j) if y[k] == 1 else (j, i)
        p_model = expit(abs(d_oof[k]))
        f.write(f"$\\displaystyle {probs[hard].latex}$ & $\\displaystyle {probs[easy].latex}$ & {p_model:.2f} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}}\n")
numbers["NWrong"] = str(int(wrong.sum()))

# --- Figure ------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7), gridspec_kw={"width_ratios": [1.1, 1.2, 0.9]})
ax = axes[0]
ax.plot(sizes, c_full[:, 1], "o-", color=BLUE, ms=3, label="all features: held-out")
ax.plot(sizes, c_full[:, 0], "o--", color=BLUE, ms=3, mfc="none", label="all features: training")
ax.plot(sizes, c_full[:, 0] + c_full[:, 3], ":", color="k", lw=1, label="training + $\\mathrm{df}/m$")
ax.plot(sizes, c_der[:, 1], "s-", color=ORANGE, ms=3, label="deriv\\_size only: held-out")
ax.set_xlabel("training comparisons $m$")
ax.set_ylabel("log-loss")
ax.legend(frameon=False, fontsize=6)
ax.set_title("(a) learning curves", loc="left")

ax = axes[1]
order_w = np.argsort(std_w)
ax.barh(range(len(std_w)), std_w[order_w],
        color=[BLUE if abs(zs[i]) > 1.96 else "#b8c9e3" for i in order_w])
ax.set_yticks(range(len(std_w)), [FEATURE_NAMES[i] for i in order_w], fontsize=6)
ax.axvline(0, color=GREY, lw=0.6)
ax.set_xlabel("weight per sd of feature (logits)")
ax.set_title("(b) your weights", loc="left")

ax = axes[2]
centers = np.arange(4)
ax.bar(centers, 100 * np.array(tie_rate), color=BLUE)
ax.set_xticks(centers, ["closest", "", "", "farthest"])
ax.set_xlabel("predicted gap (quartile)")
ax.set_ylabel("% judged \u201cabout the same\u201d")
ax.set_title("(c) ties = close calls", loc="left", fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "step08_real.pdf")

with open(OUT / "step08_numbers.tex", "w") as f:
    for k, v in numbers.items():
        f.write(f"\\newcommand{{\\R{k}}}{{{v}}}\n")
print("\n".join(f"{k} = {v}" for k, v in numbers.items()))
print(open(OUT / "step08_models.tex").read())
