"""Which pair to show next.

Random scheduling (Section 7): a uniformly random pair of unflagged problems.
Active scheduling (Section 10): refit the mixed model (features + a random
effect per problem) on everything labeled so far and ask the pair that most
reduces the expected predictive variance over typical pairs ("variance",
V-optimal design), exploring at random a fraction of the time.

Either way, an occasional *repeat* of an earlier pair (with sides swapped)
measures how consistent the labeler is with themself, and left/right order
is randomized.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from calcdiff.active import all_pairs, choose
from calcdiff.labels.store import LabelStore
from calcdiff.models.mixed_bt import MixedBT

P_REPEAT = 0.1
MIN_BEFORE_REPEATS = 10
MIN_FOR_MODEL = 20       # below this, active scheduling falls back to random
ACTIVE_L2 = 3.0          # ridge strength for the model that picks pairs (CV chose ~3 in Section 8)
ACTIVE_TAU = 0.5         # sd of the per-problem effect; CV on held-out problems is flat in tau
EXPLORE = 0.1


@dataclass(frozen=True)
class Pair:
    left_id: str
    right_id: str
    is_repeat: bool
    strategy: str = "random"


def next_pair(store: LabelStore, labeler: str, rng: np.random.Generator,
              strategy: str = "random", Phi: np.ndarray | None = None) -> Pair:
    """`Phi` must hold the features of store.problems(), row for row, when strategy != "random"."""
    flagged = store.flagged()
    problems = store.problems()
    ids = [p.id for p in problems if p.id not in flagged]
    if len(ids) < 2:
        raise ValueError("need at least two problems in the pool")
    past = [j for j in store.judgments(labeler)
            if not j.is_repeat and j.left_id not in flagged and j.right_id not in flagged]
    if len(past) >= MIN_BEFORE_REPEATS and rng.random() < P_REPEAT:
        j = past[rng.integers(len(past))]
        return Pair(j.right_id, j.left_id, True, j.strategy)  # swapped, so position cannot be remembered
    if strategy == "random" or Phi is None:
        i, k = rng.choice(len(ids), size=2, replace=False)
        return Pair(ids[i], ids[k], False, "random")

    index = {p.id: n for n, p in enumerate(problems)}
    _, pairs, y = store.to_arrays(labeler)
    model = MixedBT(l2=ACTIVE_L2, tau=ACTIVE_TAU).fit(Phi, pairs, y) if len(y) >= MIN_FOR_MODEL else None
    cand = all_pairs(np.array([index[i] for i in ids]))
    asked = {frozenset((index[j.left_id], index[j.right_id])) for j in past}
    cand = cand[[frozenset(c) not in asked for c in cand.tolist()]]
    a, b = cand[choose(model, Phi, cand, strategy, rng, explore=EXPLORE)]
    if rng.random() < 0.5:
        a, b = b, a
    return Pair(problems[a].id, problems[b].id, False, strategy)
