"""The guessing game's model: a linear Bradley-Terry model shipped as data.

Mirrors calcdiff (models/feature_bt.py and explain.py) with numpy only, so the
deployed app needs no PyTorch, SymPy or scikit-learn. tests/test_game.py checks
that it agrees with calcdiff exactly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MODEL_PATH = Path(__file__).with_name("model.json")


@dataclass(frozen=True)
class Problem:
    id: str
    latex: str
    phi: np.ndarray


class GameModel:
    def __init__(self, path: Path = MODEL_PATH):
        d = json.loads(Path(path).read_text())
        self.theta = np.array(d["theta_raw"])
        self.groups: dict[str, list[int]] = d["groups"]
        self.problems = [Problem(p["id"], p["latex"], np.array(p["phi"])) for p in d["problems"]]
        self.by_id = {p.id: p for p in self.problems}
        # instructor judgments: (a, b) -> P(a harder) label, 1 / 0 / 0.5
        self.instructor = {(r["a"], r["b"]): r["y"] for r in d["instructor"]
                           if r["a"] in self.by_id and r["b"] in self.by_id}
        self.n_training = d["n_training_judgments"]

    def prob_harder(self, a: Problem, b: Problem) -> float:
        """P(a judged harder than b) = sigma(theta . (phi_a - phi_b))."""
        return float(1 / (1 + np.exp(-self.theta @ (a.phi - b.phi))))

    def shapley(self, a: Problem, b: Problem) -> dict[str, float]:
        """Exact Shapley values of the feature groups for the gap delta(a) - delta(b) (linear closed form)."""
        return {g: float(self.theta[cols] @ (a.phi[cols] - b.phi[cols])) for g, cols in self.groups.items()}

    def instructor_says(self, a: Problem, b: Problem) -> float | None:
        """The instructor's label for P(a harder), if they judged this pair (in either order)."""
        if (a.id, b.id) in self.instructor:
            return self.instructor[(a.id, b.id)]
        if (b.id, a.id) in self.instructor:
            return 1 - self.instructor[(b.id, a.id)]
        return None

    def next_pair(self, rng: np.random.Generator, instructor_share: float = 0.5) -> tuple[Problem, Problem]:
        """Half the time a pair the instructor judged (so people can be compared on the
        same questions), otherwise a fresh random pair. Sides are randomized."""
        if self.instructor and rng.random() < instructor_share:
            keys = list(self.instructor)
            a_id, b_id = keys[rng.integers(len(keys))]
            a, b = self.by_id[a_id], self.by_id[b_id]
        else:
            i, j = rng.choice(len(self.problems), size=2, replace=False)
            a, b = self.problems[i], self.problems[j]
        return (a, b) if rng.random() < 0.5 else (b, a)


def describe(phi: dict[str, float], top: int = 2) -> str:
    """Same sentence as calcdiff.explain.describe: justifies the side the model picks."""
    names = list(phi)
    v = np.array([phi[n] for n in names])
    if v.sum() < 0:
        v = -v
    order = np.argsort(-v)
    up = [names[j] for j in order[:top] if v[j] > 0.05]
    down = [names[j] for j in order[::-1][:1] if v[j] < -0.05]
    if not up:
        return "No single aspect stands out; the model sees these as close."
    s = "Mostly the " + " and the ".join(up)
    if down:
        s += f", despite the {down[0]}"
    return s + "."
