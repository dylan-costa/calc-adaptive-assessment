# calc-adaptive-assessment

By Dylan Costa.

Learning how hard a calculus derivative problem is, from an instructor's pairwise
"which is harder?" judgments. This is the foundation for an adaptive placement exam.

The project is built in small steps, each paired with a section of lecture notes
([notes/lectures.pdf](notes/lectures.pdf)) explaining the mathematics behind it.

| Step | Status | Topic |
|------|--------|-------|
| 0 | done | Problem setup: difficulty as a latent variable |
| 1 | done | Problem generator: a probabilistic grammar over expression trees |
| 2 | done | Step-tracing differentiator: structural recursion, dual numbers, verification |
| 3 | done | Feature map: 17 features, interactions, collinearity, standardization |
| 4 | done | Bradley–Terry: Newton vs gradient descent, Zermelo condition, ridge MAP, Fisher information |
| 5 | done | Evaluation: hold out problems, noise ceilings, problem-level bootstrap |
| 6 | done | Linear Bradley–Terry on features, cross-validated ridge, bootstrap intervals for weights |
| 7 | done | Labeling app (Streamlit) and label-noise theory |
| 8 | done | First fit on real labels: nested CV, learning curves, error analysis |
| 9 | done | Gradient-boosted Bradley–Terry (own implementation): leaf Newton steps as graph cuts |
| 10 | done | Active learning: D- vs V-optimal design, a mixed model with per-problem effects; active mode in the app |
| 11 | done | Recursive neural network on expression trees (PyTorch): pretraining on generated problems, LP-FT |
| 12 | done | Shapley explanations of single comparisons: axioms, exact group Shapley, sampling estimator |
| 13 | planned | Guessing game: other people judge pairs, see the model's answer and why; multi-rater Bradley–Terry |
| 14 | optional | Adaptive quiz (item response theory), validated on real student answers |

## Setup

```bash
py -3.11 -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"
.venv/Scripts/python -m pytest
```

## Label problems

```bash
.venv/Scripts/python -m streamlit run labeler/app.py --server.address localhost
```

Then open http://localhost:8501. Judgments are saved to `data/labels.sqlite`.

## The guessing game (Step 13)

`game/` is a small, self-contained Streamlit app: players judge pairs, then see the model's prediction
and a Shapley explanation. It needs only `game/requirements.txt` (no PyTorch), and reads the model from
`game/model.json`, which is regenerated from your labels with:

```bash
.venv/Scripts/python scripts/export_game_model.py
```

Run it locally (guesses go to `game/guesses.sqlite`):

```bash
.venv/Scripts/python -m streamlit run game/app.py --server.address localhost
```

Deploy on Streamlit Community Cloud:

1. Create a free Postgres database (e.g. [Neon](https://neon.tech)) and copy its connection string.
2. Push this repository to GitHub.
3. At [share.streamlit.io](https://share.streamlit.io), create an app from the repo with main file
   `game/app.py`, Python 3.11, and under *Advanced settings → Secrets*:
   `DATABASE_URL = "postgresql://..."`.

Pull the collected guesses for analysis:

```bash
DATABASE_URL="postgresql://..." .venv/Scripts/python scripts/pull_game_guesses.py
```

## Rebuild the notes

Regenerate the figures and numbers for the notes, then build the PDF:

```bash
.venv/Scripts/python scripts/figs_step01.py
.venv/Scripts/python scripts/figs_step02.py
.venv/Scripts/python scripts/figs_step03.py
.venv/Scripts/python scripts/figs_step04.py
.venv/Scripts/python scripts/figs_step05.py
.venv/Scripts/python scripts/figs_step06.py
.venv/Scripts/python scripts/figs_step07.py
.venv/Scripts/python scripts/figs_step08.py   # needs data/labels.sqlite
.venv/Scripts/python scripts/figs_step09.py   # needs data/labels.sqlite
.venv/Scripts/python scripts/figs_step10.py   # needs data/labels.sqlite (about 6 minutes)
.venv/Scripts/python scripts/figs_step11.py   # needs data/labels.sqlite (several minutes)
.venv/Scripts/python scripts/figs_step12.py   # needs data/labels.sqlite
cd notes && latexmk -pdf lectures.tex
```

## License

© 2026 Dylan Costa. All rights reserved.
