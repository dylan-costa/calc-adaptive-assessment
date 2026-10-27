"""The deployed game re-implements a few lines of calcdiff with numpy only.
These tests hold it to exactly the same answers."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "game"))

from core import GameModel, describe  # noqa: E402
from storage import GuessStore  # noqa: E402

from calcdiff.explain import GROUP_COLUMNS, linear_shapley  # noqa: E402
from calcdiff.explain import describe as describe_ref  # noqa: E402


def test_groups_match_calcdiff():
    m = GameModel()
    assert list(m.groups.values()) == GROUP_COLUMNS


def test_shapley_and_description_match_calcdiff():
    m = GameModel()
    rng = np.random.default_rng(0)
    for _ in range(50):
        a, b = m.next_pair(rng)
        phi = m.shapley(a, b)
        ref = linear_shapley(m.theta, a.phi, b.phi)
        assert np.allclose(list(phi.values()), ref)
        assert describe(phi) == describe_ref(ref)
        # efficiency: contributions add up to the log-odds of the model's prediction
        p = m.prob_harder(a, b)
        assert np.isclose(sum(phi.values()), np.log(p / (1 - p)))


def test_instructor_answer_is_symmetric():
    m = GameModel()
    (a_id, b_id), yv = next(iter(m.instructor.items()))
    a, b = m.by_id[a_id], m.by_id[b_id]
    assert m.instructor_says(a, b) == yv and m.instructor_says(b, a) == 1 - yv


def test_pairs_mix_instructor_and_fresh():
    m = GameModel()
    rng = np.random.default_rng(1)
    hits = sum(m.instructor_says(*m.next_pair(rng)) is not None for _ in range(400))
    assert 150 < hits < 260          # about half


def test_sqlite_store_round_trip(tmp_path):
    s = GuessStore(sqlite_path=tmp_path / "g.sqlite")
    s.add("r1", "student", "a", "b", "left", 1200, 0.7)
    s.add("r1", "student", "c", "d", "tie", 900, 0.5)
    assert s.count() == 2 and s.all()[1][5] == "tie"
    try:
        s.add("r1", "student", "a", "b", "maybe", 1, 0.5)
        assert False
    except ValueError:
        pass


def test_postgres_store_reconnects_after_a_dropped_connection(monkeypatch):
    """Simulate Neon closing an idle connection: the next insert must reconnect and succeed."""
    import storage

    class FakeConn:
        def __init__(self, log):
            self.log, self.closed = log, False

        def execute(self, sql, values=()):
            if self.closed:
                raise RuntimeError("connection is closed")
            self.log.append(sql.split()[0])
            return self

    log, conns = [], []

    def fake_connect(self):
        conns.append(FakeConn(log))
        self.conn = conns[-1]

    monkeypatch.setattr(storage.GuessStore, "_connect_postgres", fake_connect)
    s = storage.GuessStore("postgresql://example")
    s.add("r", "ta", "a", "b", "left", 10, 0.5)
    conns[-1].closed = True                       # the server dropped us while idle
    s.add("r", "ta", "c", "d", "right", 10, 0.5)
    assert len(conns) == 2 and log.count("INSERT") == 2
