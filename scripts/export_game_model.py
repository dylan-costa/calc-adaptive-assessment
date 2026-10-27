"""Export everything the guessing game needs into game/model.json.

Run:  .venv/Scripts/python scripts/export_game_model.py

The game runs on Streamlit Community Cloud with only streamlit + numpy, so the
model is shipped as data: the pool's raw features, the linear model's raw
weights (delta_hat = theta_raw . phi + const), the Shapley groups, and the
instructor's own answers on the pairs they judged (so players can compare).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from calcdiff.explain import GROUPS
from calcdiff.features import FEATURE_NAMES, feature_matrix
from calcdiff.labels.store import LabelStore
from calcdiff.models.feature_bt import FeatureBT, cv_l2

ROOT = Path(__file__).resolve().parents[1]
store = LabelStore(ROOT / "data" / "labels.sqlite")
probs, pairs, y = store.to_arrays("dylan", include_repeats=False)
Phi = feature_matrix(p.tree for p in probs)
lam, _ = cv_l2(Phi, pairs, y, np.unique(pairs), np.logspace(-2, 3, 11), np.random.default_rng(0))
model = FeatureBT(l2=lam).fit(Phi, pairs, y)

flagged = store.flagged()
keep = [k for k, p in enumerate(probs) if p.id not in flagged]
instructor = [
    {"a": probs[i].id, "b": probs[j].id, "y": float(t)}
    for (i, j), t in zip(pairs, y)
    if probs[i].id not in flagged and probs[j].id not in flagged
]
out = {
    "version": 1,
    "feature_names": list(FEATURE_NAMES),
    "groups": {g: [FEATURE_NAMES.index(f) for f in fs] for g, fs in GROUPS.items()},
    "theta_raw": model.raw_weights().tolist(),
    "lambda": lam,
    "n_training_judgments": int(len(y)),
    "problems": [{"id": probs[k].id, "latex": probs[k].latex, "phi": Phi[k].tolist()} for k in keep],
    "instructor": instructor,
}
path = ROOT / "game" / "model.json"
path.parent.mkdir(exist_ok=True)
path.write_text(json.dumps(out, separators=(",", ":")))
print(f"wrote {path.relative_to(ROOT)}: {len(keep)} problems, {len(instructor)} instructor judgments, "
      f"{path.stat().st_size / 1024:.0f} KB, lambda={lam:g}")
