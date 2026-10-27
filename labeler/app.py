"""Which derivative is harder? -- the labeling tool.

Run:  .venv/Scripts/python -m streamlit run labeler/app.py

Every click is one comparison stored in data/labels.sqlite. Streamlit reruns
this whole script top to bottom on every interaction; anything that must
survive a rerun lives in st.session_state (per browser tab) or in the
database (permanent).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import streamlit as st

from calcdiff.features import feature_matrix
from calcdiff.generate import Sampler
from calcdiff.labels.schedule import next_pair
from calcdiff.labels.store import LabelStore

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "labels.sqlite"
POOL_SIZE, POOL_SEED = 300, 2027

st.set_page_config(page_title="Which is harder?", page_icon="∂", layout="centered")
# Long formulas scroll sideways inside their card instead of being cut off.
st.markdown("<style>.katex-display { overflow-x: auto; overflow-y: hidden; padding-bottom: 4px; }</style>",
            unsafe_allow_html=True)


@st.cache_resource
def get_store() -> LabelStore:
    """One connection per server process; the pool is generated on first launch."""
    store = LabelStore(DB_PATH)
    if not store.problems():
        store.add_problems(Sampler(seed=POOL_SEED).sample_unique(POOL_SIZE))
    return store


@st.cache_resource
def pool_features(n_problems: int) -> np.ndarray:
    """phi(x) for every pool problem, row for row with store.problems() (cached; ~1 s to compute)."""
    return feature_matrix(p.tree for p in get_store().problems())


store = get_store()
Phi = pool_features(len(store.problems()))
state = st.session_state
if "rng" not in state:
    state.rng = np.random.default_rng()

# --- sidebar: who is labeling, progress, undo -----------------------------------
with st.sidebar:
    labeler = st.text_input("Your name", value="dylan").strip() or "anonymous"
    mode = st.radio("Pair selection", ["active", "random"], horizontal=True,
                    help="Active: ask the pair whose answer would teach the model the most (Section 10).")
    strategy = "variance" if mode == "active" else "random"
    mine = store.judgments(labeler)
    agree, repeats = store.repeat_agreement(labeler)
    st.metric("Comparisons", len(mine))
    st.metric("Self-agreement on repeats", f"{agree}/{repeats}" if repeats else "—")
    st.caption(f"Pool: {len(store.problems())} problems · {len(store.flagged())} flagged")
    if st.button("Undo last", use_container_width=True, disabled=not mine):
        store.undo_last(labeler)
        state.pop("pair", None)
        st.rerun()


def new_pair() -> None:
    state.pair = next_pair(store, labeler, state.rng, strategy=strategy, Phi=Phi)
    state.shown_at = time.time()


if "pair" not in state:
    new_pair()
pair = state.pair


def submit(outcome: str) -> None:
    ms = int(1000 * (time.time() - state.shown_at))
    store.record(pair.left_id, pair.right_id, outcome, labeler, ms, pair.is_repeat, pair.strategy)
    new_pair()


def flag(problem_id: str) -> None:
    store.flag(problem_id, labeler)
    new_pair()


# --- the comparison ----------------------------------------------------------------
st.markdown("#### Which derivative is harder for a typical Calculus I student?")
st.caption("Judge the work needed to find and write down the derivative, not how long the answer is to simplify.")

left, right = store.problem(pair.left_id), store.problem(pair.right_id)
c1, c2 = st.columns(2, border=True)
for col, prob in ((c1, left), (c2, right)):
    with col:
        st.latex(r"\frac{d}{dx}\left[" + prob.latex + r"\right]")
        st.button("Flag as unnatural", key=f"flag-{prob.id}", on_click=flag, args=(prob.id,),
                  type="tertiary", help="No instructor would write this. It will not be shown again.")

b1, b2, b3 = st.columns(3)
b1.button("Left is harder", on_click=submit, args=("left",), use_container_width=True, type="primary")
b2.button("About the same", on_click=submit, args=("tie",), use_container_width=True)
b3.button("Right is harder", on_click=submit, args=("right",), use_container_width=True, type="primary")
