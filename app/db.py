"""SQLite schema and connection helper."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

from .config import db_path

SCHEMA = """
CREATE TABLE IF NOT EXISTS tickers (
    symbol     TEXT PRIMARY KEY,
    name       TEXT,
    exchange   TEXT NOT NULL,          -- market calendar code, e.g. XNYS, XBRU
    sector     TEXT,
    active     INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS ticker_universes (
    symbol   TEXT NOT NULL REFERENCES tickers(symbol),
    universe TEXT NOT NULL,            -- sp500 | europe | bel20
    PRIMARY KEY (symbol, universe)
);

CREATE TABLE IF NOT EXISTS prices (
    symbol TEXT NOT NULL,
    date   TEXT NOT NULL,              -- YYYY-MM-DD (exchange-local trading day)
    open REAL, high REAL, low REAL, close REAL NOT NULL, volume REAL,
    PRIMARY KEY (symbol, date)
);

-- One row per morning / evening job execution.
CREATE TABLE IF NOT EXISTS runs (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    run_type       TEXT NOT NULL,      -- predict | evaluate
    run_date       TEXT NOT NULL,
    started_at     TEXT NOT NULL,
    finished_at    TEXT,
    status         TEXT NOT NULL,      -- running | ok | partial | failed
    model          TEXT,
    input_tokens   INTEGER DEFAULT 0,
    output_tokens  INTEGER DEFAULT 0,
    n_items        INTEGER DEFAULT 0,
    market_view    TEXT,               -- Claude's overall market commentary
    playbook_version TEXT,             -- which playbook the morning forecast followed
    error          TEXT
);

-- Claude's morning predictions. Never updated after the morning run.
CREATE TABLE IF NOT EXISTS predictions (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id             INTEGER REFERENCES runs(id),
    symbol             TEXT NOT NULL REFERENCES tickers(symbol),
    horizon            TEXT NOT NULL CHECK (horizon IN ('daily','weekly')),
    made_on            TEXT NOT NULL,  -- date the prediction was made
    period_start       TEXT NOT NULL,  -- first trading day covered
    target_date        TEXT NOT NULL,  -- trading day whose close decides the outcome
    base_date          TEXT NOT NULL,  -- last close known when predicting
    base_close         REAL NOT NULL,
    direction          TEXT NOT NULL CHECK (direction IN ('UP','DOWN')),
    confidence         REAL NOT NULL,  -- 0.5 .. 1.0, probability direction is right
    expected_move_pct  REAL,           -- signed % move base_close -> target close
    rationale          TEXT,
    model              TEXT NOT NULL,
    created_at         TEXT NOT NULL,
    UNIQUE (symbol, horizon, period_start)
);

-- Filled by the evening job once the target day's close is known.
CREATE TABLE IF NOT EXISTS evaluations (
    prediction_id     INTEGER PRIMARY KEY REFERENCES predictions(id),
    actual_close      REAL NOT NULL,
    actual_change_pct REAL NOT NULL,
    actual_direction  TEXT NOT NULL,
    correct           INTEGER NOT NULL,
    evaluated_at      TEXT NOT NULL
);

-- Intraday price bars, one row per symbol per trading day (local date), for the 1D / 1W charts.
-- Refreshed by both daily jobs; only the last couple of weeks are kept.
CREATE TABLE IF NOT EXISTS intraday (
    symbol   TEXT NOT NULL,
    date     TEXT NOT NULL,              -- YYYY-MM-DD in the configured timezone
    interval TEXT NOT NULL DEFAULT '15m',
    bars     TEXT NOT NULL,              -- JSON [[unix_seconds, close], ...]
    PRIMARY KEY (symbol, date)
);

-- Self-improvement memory. Stored in the same state files as everything else, so every cloud run
-- starts with it. Written only by app/reflect.py and app/research.py.

-- Rules the morning forecaster follows. Added as 'trial' by the evening reflection, promoted to
-- 'active' once they have survived a few days, 'retired' (never deleted) when they stop helping.
CREATE TABLE IF NOT EXISTS playbook_rules (
    id          TEXT PRIMARY KEY,      -- short slug, e.g. earnings-day-news-first
    text        TEXT NOT NULL,
    status      TEXT NOT NULL,         -- trial | active | retired
    created_on  TEXT NOT NULL,
    updated_on  TEXT NOT NULL,
    evidence    TEXT,                  -- what in the results supports the rule
    history     TEXT                   -- dated notes: why it was added, promoted, edited, retired
);

-- One row per evening: what happened, why, and what was changed.
CREATE TABLE IF NOT EXISTS reflections (
    date           TEXT PRIMARY KEY,
    summary        TEXT NOT NULL,
    went_well      TEXT,
    went_wrong     TEXT,
    changes        TEXT,               -- plain-language list of playbook changes made
    research_focus TEXT,               -- what the next morning's news research should look for
    stats          TEXT,               -- JSON snapshot of the diagnostics the reflection saw
    playbook_version TEXT,             -- playbook after this reflection
    created_at     TEXT NOT NULL
);

-- Pre-market news research, one row per researched stock per morning.
CREATE TABLE IF NOT EXISTS research (
    date      TEXT NOT NULL,
    symbol    TEXT NOT NULL,
    sentiment INTEGER NOT NULL,        -- -2 very negative .. +2 very positive for today's price
    note      TEXT NOT NULL,           -- the catalyst, one short sentence
    source    TEXT,
    PRIMARY KEY (date, symbol)
);

-- The market-wide part of that research.
CREATE TABLE IF NOT EXISTS research_market (
    date       TEXT PRIMARY KEY,
    summary    TEXT NOT NULL,
    events     TEXT,                   -- scheduled events today (earnings, data releases, central banks)
    sectors    TEXT,                   -- JSON {sector: "lean and reason"}
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_pred_target ON predictions(target_date);
CREATE INDEX IF NOT EXISTS idx_pred_made   ON predictions(made_on, horizon);
CREATE INDEX IF NOT EXISTS idx_prices_date ON prices(date);
"""


def connect(path: Path | None = None) -> sqlite3.Connection:
    path = Path(path or db_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.executescript(SCHEMA)
    # Databases created before the playbook existed lack this column.
    if "playbook_version" not in {r[1] for r in conn.execute("PRAGMA table_info(runs)")}:
        conn.execute("ALTER TABLE runs ADD COLUMN playbook_version TEXT")
    return conn


@contextmanager
def session(path: Path | None = None):
    conn = connect(path)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()
