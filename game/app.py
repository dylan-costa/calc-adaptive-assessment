"""Which derivative is harder? -- the public guessing game (Section 13).

Run locally:  .venv/Scripts/python -m streamlit run game/app.py --server.address localhost
Deployed on Streamlit Community Cloud with DATABASE_URL set as a secret.

Players judge pairs of derivative problems, then see what the model (trained
on one instructor's judgments) predicts and why. Every guess is stored with
the player's role and an anonymous session id: that is the multi-rater data
for Section 13.
"""

from __future__ import annotations

import os
import time
import uuid

import numpy as np
import pandas as pd
import streamlit as st

from core import GameModel, describe
from storage import GuessStore

APP_VERSION = 1
ROLES = {
    "Calculus instructor or professor": "instructor",
    "TA, tutor, or grader": "ta",
    "Student (taking or took calculus)": "student",
    "Something else": "other",
}

st.set_page_config(page_title="Which derivative is harder?", page_icon="∂", layout="centered")
st.markdown("<style>.katex-display { overflow-x: auto; overflow-y: hidden; padding-bottom: 4px; }</style>",
            unsafe_allow_html=True)


def database_url() -> str | None:
    try:
        url = st.secrets.get("DATABASE_URL")
    except Exception:          # no secrets file at all (local runs)
        url = None
    return url or os.environ.get("DATABASE_URL")


@st.cache_resource
def get_model() -> GameModel:
    return GameModel()


@st.cache_resource
def get_store() -> GuessStore:
    return GuessStore(database_url())


model, store = get_model(), get_store()
host = st.context.headers.get("Host", "localhost").split(":")[0]
if store.backend == "sqlite" and host not in ("localhost", "127.0.0.1"):
    # Hosted without a database: a local file there is wiped on every restart, so
    # collecting guesses would silently lose them. Refuse instead.
    st.error("This app is not connected to its database, so it is not collecting answers right now. "
             "(Owner: set DATABASE_URL in the app's Secrets.)")
    st.stop()
state = st.session_state
if "rater_id" not in state:
    state.rater_id = uuid.uuid4().hex[:16]
    state.rng = np.random.default_rng()
    state.n = state.agree_model = state.n_instr = state.agree_instr = 0


def new_pair() -> None:
    state.pair = model.next_pair(state.rng)
    state.shown_at = time.time()
    state.revealed = False


def submit(choice: str) -> None:
    left, right = state.pair
    p = model.prob_harder(left, right)
    store.add(state.rater_id, state.role, left.id, right.id, choice,
              int(1000 * (time.time() - state.shown_at)), p, APP_VERSION)
    state.choice, state.revealed = choice, True
    if choice != "tie":
        state.n += 1
        state.agree_model += (choice == "left") == (p >= 0.5)
        y = model.instructor_says(left, right)
        if y is not None and y != 0.5:
            state.n_instr += 1
            state.agree_instr += (choice == "left") == (y == 1)


# --- welcome screen ---------------------------------------------------------------
if "role" not in state:
    st.title("Which derivative is harder?")
    st.write(
        "You'll see two derivative problems. Pick the one that is **harder for a typical Calculus I student** "
        "to differentiate. Then see what a machine-learning model thinks, and why. The model learned from "
        f"{model.n_training} judgments by one calculus instructor."
    )
    choice = st.radio("Which describes you best?", list(ROLES), index=None)
    st.caption("Your answers are saved anonymously together with the role you pick, to study how different people "
               "judge difficulty. No names, emails, or other personal information are collected.")
    if st.button("Start", type="primary", disabled=choice is None):
        state.role = ROLES[choice]
        new_pair()
        st.rerun()
    st.stop()

# --- a round ------------------------------------------------------------------------
left, right = state.pair
top = st.columns([3, 2])
top[0].markdown("#### Which is harder to differentiate?")
if state.n:
    top[1].markdown(f"<div style='text-align:right'>You agreed with the model on "
                    f"<b>{state.agree_model}/{state.n}</b></div>", unsafe_allow_html=True)

c1, c2 = st.columns(2, border=True)
for col, prob in ((c1, left), (c2, right)):
    col.latex(r"\frac{d}{dx}\left[" + prob.latex + r"\right]")

if not state.revealed:
    b1, b2, b3 = st.columns(3)
    b1.button("Left is harder", on_click=submit, args=("left",), use_container_width=True, type="primary")
    b2.button("About the same", on_click=submit, args=("tie",), use_container_width=True)
    b3.button("Right is harder", on_click=submit, args=("right",), use_container_width=True, type="primary")
    st.stop()

# --- the reveal -----------------------------------------------------------------------
p = model.prob_harder(left, right)
model_side = "left" if p >= 0.5 else "right"
conf = max(p, 1 - p)
yours = {"left": "the left one", "right": "the right one", "tie": "about the same"}[state.choice]
if conf < 0.6:
    verdict = f"The model thinks it's a **close call** (leans {model_side}, {conf:.0%})."
else:
    verdict = f"The model says the **{model_side}** one is harder ({conf:.0%} sure)."
match = "" if state.choice == "tie" else (" You agree." if state.choice == model_side else " You disagree.")
st.markdown(f"You said **{yours}**. {verdict}{match}")

phi = model.shapley(left, right)
st.markdown(f"**Why:** {describe(phi)}")
sign = 1 if model_side == "left" else -1
chart = pd.DataFrame({"aspect": list(phi), "push toward the model's pick (logits)": [sign * v for v in phi.values()]})
chart = chart[chart.iloc[:, 1].abs() > 0.02].sort_values(chart.columns[1], ascending=False)
if len(chart):
    st.bar_chart(chart, x="aspect", y=chart.columns[1], horizontal=True, height=200)

y = model.instructor_says(left, right)
if y is not None:
    said = {1.0: "the left one", 0.0: "the right one", 0.5: "about the same"}[y]
    st.info(f"The instructor who trained the model judged this exact pair: **{said}**.")

st.button("Next pair", on_click=new_pair, type="primary")
if state.n_instr:
    st.caption(f"On pairs the instructor also judged, you agreed with them {state.agree_instr}/{state.n_instr} times.")
