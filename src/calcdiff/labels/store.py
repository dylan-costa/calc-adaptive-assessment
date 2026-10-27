"""SQLite storage for the problem pool and the instructor's judgments.

Three tables:

  problems(id, latex, tree_json, created_at)
  judgments(id, left_id, right_id, outcome, labeler, response_ms, is_repeat, created_at)
  flags(problem_id, labeler, created_at)    -- "no instructor would write this"

`outcome` is "left", "right" or "tie" -- what was clicked, relative to what was
shown on screen. Storing the screen position (rather than only the winner)
lets us measure position bias later. A judgment is converted to a training
label by `to_arrays`: pair = (left, right), y = 1 / 0 / 0.5.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from calcdiff.expr import Node, from_dict, to_dict
from calcdiff.generate import Problem

OUTCOME_Y = {"left": 1.0, "right": 0.0, "tie": 0.5}

SCHEMA = """
CREATE TABLE IF NOT EXISTS problems (
    id         TEXT PRIMARY KEY,
    latex      TEXT NOT NULL,
    tree_json  TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS judgments (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    left_id     TEXT NOT NULL REFERENCES problems(id),
    right_id    TEXT NOT NULL REFERENCES problems(id),
    outcome     TEXT NOT NULL CHECK (outcome IN ('left', 'right', 'tie')),
    labeler     TEXT NOT NULL,
    response_ms INTEGER,
    is_repeat   INTEGER NOT NULL DEFAULT 0,
    created_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS flags (
    problem_id TEXT NOT NULL REFERENCES problems(id),
    labeler    TEXT NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY (problem_id, labeler)
);
"""


@dataclass(frozen=True)
class StoredProblem:
    id: str
    latex: str
    tree: Node


@dataclass(frozen=True)
class Judgment:
    id: int
    left_id: str
    right_id: str
    outcome: str
    labeler: str
    response_ms: int | None
    is_repeat: bool
    strategy: str = "random"   # how the pair was chosen (Section 10)

    @property
    def winner(self) -> str | None:
        """Id of the problem judged harder (None for a tie)."""
        return {"left": self.left_id, "right": self.right_id, "tie": None}[self.outcome]


class LabelStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.executescript(SCHEMA)
        # Migration (Section 10): record how each pair was chosen. Every judgment
        # made before this column existed came from the random scheduler.
        cols = {r[1] for r in self.db.execute("PRAGMA table_info(judgments)")}
        if "strategy" not in cols:
            with self.db:
                self.db.execute("ALTER TABLE judgments ADD COLUMN strategy TEXT NOT NULL DEFAULT 'random'")

    # -- problems ---------------------------------------------------------------

    def add_problems(self, problems: list[Problem]) -> int:
        now = time.time()
        rows = [(p.id, p.latex, json.dumps(to_dict(p.tree)), now) for p in problems]
        with self.db:
            cur = self.db.executemany("INSERT OR IGNORE INTO problems VALUES (?, ?, ?, ?)", rows)
        return cur.rowcount

    def problems(self) -> list[StoredProblem]:
        rows = self.db.execute("SELECT id, latex, tree_json FROM problems ORDER BY created_at, id")
        return [StoredProblem(i, tex, from_dict(json.loads(js))) for i, tex, js in rows]

    def problem(self, pid: str) -> StoredProblem:
        i, tex, js = self.db.execute("SELECT id, latex, tree_json FROM problems WHERE id = ?", (pid,)).fetchone()
        return StoredProblem(i, tex, from_dict(json.loads(js)))

    def flag(self, problem_id: str, labeler: str) -> None:
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO flags VALUES (?, ?, ?)", (problem_id, labeler, time.time()))

    def flagged(self, labeler: str | None = None) -> set[str]:
        """Flagged problem ids: by anyone, or by one labeler (e.g. "auto:is_natural")."""
        if labeler is None:
            return {r[0] for r in self.db.execute("SELECT problem_id FROM flags")}
        return {r[0] for r in self.db.execute("SELECT problem_id FROM flags WHERE labeler = ?", (labeler,))}

    # -- judgments --------------------------------------------------------------

    def record(self, left_id: str, right_id: str, outcome: str, labeler: str,
               response_ms: int | None = None, is_repeat: bool = False, strategy: str = "random") -> int:
        with self.db:
            cur = self.db.execute(
                "INSERT INTO judgments (left_id, right_id, outcome, labeler, response_ms, is_repeat, created_at,"
                " strategy) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (left_id, right_id, outcome, labeler, response_ms, int(is_repeat), time.time(), strategy))
        return int(cur.lastrowid)

    def undo_last(self, labeler: str) -> Judgment | None:
        last = self.judgments(labeler)[-1:] or [None]
        if last[0] is not None:
            with self.db:
                self.db.execute("DELETE FROM judgments WHERE id = ?", (last[0].id,))
        return last[0]

    def judgments(self, labeler: str | None = None) -> list[Judgment]:
        q = "SELECT id, left_id, right_id, outcome, labeler, response_ms, is_repeat, strategy FROM judgments"
        args: tuple = ()
        if labeler is not None:
            q += " WHERE labeler = ?"
            args = (labeler,)
        rows = self.db.execute(q + " ORDER BY id", args)
        return [Judgment(r[0], r[1], r[2], r[3], r[4], r[5], bool(r[6]), r[7]) for r in rows]

    # -- conversion to training data ---------------------------------------------

    def to_arrays(self, labeler: str | None = None, include_repeats: bool = True):
        """(problems, pairs, y): pairs index into `problems`, y = P(left harder) label."""
        probs = self.problems()
        index = {p.id: k for k, p in enumerate(probs)}
        js = [j for j in self.judgments(labeler) if include_repeats or not j.is_repeat]
        pairs = np.array([(index[j.left_id], index[j.right_id]) for j in js], dtype=int).reshape(-1, 2)
        y = np.array([OUTCOME_Y[j.outcome] for j in js])
        return probs, pairs, y

    def repeat_agreement(self, labeler: str) -> tuple[int, int]:
        """(agreements, repeats): did each repeated pair get the same winner as the first time?"""
        js = self.judgments(labeler)
        first: dict[frozenset, Judgment] = {}
        agree = total = 0
        for j in js:
            key = frozenset((j.left_id, j.right_id))
            if j.is_repeat and key in first:
                total += 1
                agree += first[key].winner == j.winner
            elif not j.is_repeat:
                first.setdefault(key, j)
        return agree, total
