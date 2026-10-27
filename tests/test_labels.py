import numpy as np

from calcdiff.expr import from_dict, to_dict
from calcdiff.generate import Sampler
from calcdiff.labels.schedule import next_pair
from calcdiff.labels.store import LabelStore


def test_tree_serialization_round_trips():
    for p in Sampler(seed=0).sample_unique(200):
        assert from_dict(to_dict(p.tree)) == p.tree


def make_store(tmp_path, n=20):
    store = LabelStore(tmp_path / "labels.sqlite")
    store.add_problems(Sampler(seed=1).sample_unique(n))
    return store


def test_problems_persist_and_dedupe(tmp_path):
    store = make_store(tmp_path)
    assert store.add_problems(Sampler(seed=1).sample_unique(20)) == 0   # already there
    again = LabelStore(tmp_path / "labels.sqlite")
    assert len(again.problems()) == 20


def test_judgments_become_training_labels(tmp_path):
    store = make_store(tmp_path)
    a, b, c = (p.id for p in store.problems()[:3])
    store.record(a, b, "left", "dylan")
    store.record(c, a, "right", "dylan")
    store.record(b, c, "tie", "dylan")
    probs, pairs, y = store.to_arrays("dylan")
    ids = [p.id for p in probs]
    assert [(ids[i], ids[j]) for i, j in pairs] == [(a, b), (c, a), (b, c)]
    assert y.tolist() == [1.0, 0.0, 0.5]


def test_undo_removes_only_the_last(tmp_path):
    store = make_store(tmp_path)
    a, b = (p.id for p in store.problems()[:2])
    store.record(a, b, "left", "dylan")
    store.record(a, b, "right", "dylan")
    assert store.undo_last("dylan").outcome == "right"
    assert [j.outcome for j in store.judgments("dylan")] == ["left"]


def test_repeat_agreement_accounts_for_swapped_sides(tmp_path):
    store = make_store(tmp_path)
    a, b = (p.id for p in store.problems()[:2])
    store.record(a, b, "left", "dylan")                    # a harder
    store.record(b, a, "right", "dylan", is_repeat=True)   # shown swapped, a harder again
    store.record(b, a, "left", "dylan", is_repeat=True)    # changed mind
    assert store.repeat_agreement("dylan") == (1, 2)


def test_scheduler_repeats_with_swapped_sides(tmp_path):
    store = make_store(tmp_path)
    rng = np.random.default_rng(0)
    ids = [p.id for p in store.problems()]
    for k in range(10):
        store.record(ids[k], ids[k + 1], "left", "dylan")
    pairs = [next_pair(store, "dylan", rng) for _ in range(400)]
    repeats = [p for p in pairs if p.is_repeat]
    assert 20 < len(repeats) < 70                           # about 10%
    seen = {(j.left_id, j.right_id) for j in store.judgments("dylan")}
    assert all((p.right_id, p.left_id) in seen for p in repeats)
    assert all(p.left_id != p.right_id for p in pairs)


def test_flagged_problems_are_never_shown(tmp_path):
    store = make_store(tmp_path, n=5)
    ids = [p.id for p in store.problems()]
    for pid in ids[:3]:
        store.flag(pid, "dylan")
    rng = np.random.default_rng(1)
    for _ in range(50):
        p = next_pair(store, "dylan", rng)
        assert {p.left_id, p.right_id} == set(ids[3:])


def test_active_scheduler_avoids_asked_pairs_and_prefers_close_calls(tmp_path):
    from calcdiff.features import feature_matrix
    from calcdiff.models.feature_bt import FeatureBT

    store = make_store(tmp_path, n=40)
    probs = store.problems()
    Phi = feature_matrix(p.tree for p in probs)
    rng = np.random.default_rng(0)
    size = Phi[:, 3]                                # a simulated labeler: bigger derivative = harder
    for _ in range(60):
        i, k = rng.choice(40, 2, replace=False)
        store.record(probs[i].id, probs[k].id, "left" if size[i] > size[k] else "right", "dylan")
    asked = {frozenset((j.left_id, j.right_id)) for j in store.judgments("dylan")}
    model = FeatureBT(l2=3.0).fit(Phi, *store.to_arrays("dylan")[1:])
    s = model.score(Phi)
    index = {p.id: n for n, p in enumerate(probs)}
    gaps_active, gaps_random = [], []
    for _ in range(40):
        p = next_pair(store, "dylan", rng, strategy="variance", Phi=Phi)
        if p.is_repeat:
            continue
        assert frozenset((p.left_id, p.right_id)) not in asked and p.strategy == "variance"
        gaps_active.append(abs(s[index[p.left_id]] - s[index[p.right_id]]))
        r = next_pair(store, "dylan", rng)
        if not r.is_repeat:
            gaps_random.append(abs(s[index[r.left_id]] - s[index[r.right_id]]))
    assert np.median(gaps_active) < np.median(gaps_random)


def test_strategy_is_recorded(tmp_path):
    store = make_store(tmp_path)
    a, b = (p.id for p in store.problems()[:2])
    store.record(a, b, "left", "dylan", strategy="information")
    assert store.judgments("dylan")[0].strategy == "information"
