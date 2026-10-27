"""Where guesses go: Postgres in the cloud, SQLite on your machine.

If Streamlit secrets (or the environment) define DATABASE_URL, guesses go to
that Postgres database (e.g. a free Neon instance). Otherwise they go to a
local SQLite file, which is what you want when running the game locally.
Streamlit Community Cloud's disk is wiped on every restart, so the cloud
deployment must use Postgres.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS guesses (
    id           {pk},
    created_at   DOUBLE PRECISION NOT NULL,
    rater_id     TEXT NOT NULL,         -- random per browser session; no names or emails
    role         TEXT NOT NULL,         -- instructor / ta / student / other
    left_id      TEXT NOT NULL,
    right_id     TEXT NOT NULL,
    choice       TEXT NOT NULL CHECK (choice IN ('left', 'right', 'tie')),
    response_ms  INTEGER,
    model_p_left DOUBLE PRECISION,      -- the model's P(left harder), for later analysis
    app_version  INTEGER NOT NULL
)
"""
COLUMNS = ("created_at", "rater_id", "role", "left_id", "right_id", "choice", "response_ms",
           "model_p_left", "app_version")


class GuessStore:
    def __init__(self, database_url: str | None = None, sqlite_path: Path | None = None):
        self.database_url = database_url
        if database_url:
            self._connect_postgres()
            self.param = "%s"
        else:
            path = sqlite_path or Path(__file__).with_name("guesses.sqlite")
            self.conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
            self.param = "?"
            self.conn.execute(SCHEMA.format(pk="INTEGER PRIMARY KEY AUTOINCREMENT"))
        self.backend = "postgres" if database_url else "sqlite"

    def _connect_postgres(self) -> None:
        import psycopg  # only needed in the cloud
        self.conn = psycopg.connect(self.database_url, autocommit=True)
        self.conn.execute(SCHEMA.format(pk="BIGSERIAL PRIMARY KEY"))

    def _execute(self, sql: str, values: tuple = ()):
        """Run a statement; on Postgres, reconnect once if the connection was dropped.

        Free Postgres tiers (e.g. Neon) suspend after a few idle minutes and close
        open connections, so the first query after a quiet spell can fail.
        """
        try:
            return self.conn.execute(sql, values)
        except Exception:
            if self.backend != "postgres":
                raise
            self._connect_postgres()
            return self.conn.execute(sql, values)

    def add(self, rater_id: str, role: str, left_id: str, right_id: str, choice: str,
            response_ms: int | None, model_p_left: float, app_version: int = 1) -> None:
        if choice not in ("left", "right", "tie"):
            raise ValueError(choice)
        values = (time.time(), rater_id, role, left_id, right_id, choice, response_ms, model_p_left, app_version)
        marks = ", ".join([self.param] * len(values))
        self._execute(f"INSERT INTO guesses ({', '.join(COLUMNS)}) VALUES ({marks})", values)

    def count(self) -> int:
        return int(self._execute("SELECT COUNT(*) FROM guesses").fetchone()[0])

    def all(self) -> list[tuple]:
        return list(self._execute(f"SELECT {', '.join(COLUMNS)} FROM guesses ORDER BY id").fetchall())
