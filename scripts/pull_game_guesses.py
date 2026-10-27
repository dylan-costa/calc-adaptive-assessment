"""Download every guess from the deployed game's Postgres database.

Run:  DATABASE_URL="postgresql://..." .venv/Scripts/python scripts/pull_game_guesses.py
Writes data/game_guesses.csv (one row per guess) for the multi-rater analysis.
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "game"))
from storage import COLUMNS, GuessStore  # noqa: E402

url = os.environ.get("DATABASE_URL")
if not url:
    sys.exit("Set DATABASE_URL to the game's Postgres connection string first.")
rows = GuessStore(url).all()
out = ROOT / "data" / "game_guesses.csv"
out.parent.mkdir(exist_ok=True)
with out.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(COLUMNS)
    w.writerows(rows)
print(f"wrote {len(rows)} guesses to {out.relative_to(ROOT)}")
