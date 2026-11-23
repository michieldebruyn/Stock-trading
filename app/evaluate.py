"""Evening job: fetch closing prices and score every prediction whose target day has closed.

    python -m app.evaluate             # normal evening run
    python -m app.evaluate --no-fetch  # score against prices already in the DB
"""
from __future__ import annotations

import argparse
import logging
from datetime import date, datetime, timezone

from . import prices
from .config import local_tz
from .db import session

log = logging.getLogger("evaluate")


def score(conn, today: date) -> dict:
    pending = conn.execute(
        """SELECT p.id, p.symbol, p.target_date, p.base_close, p.direction
           FROM predictions p LEFT JOIN evaluations e ON e.prediction_id = p.id
           WHERE e.prediction_id IS NULL AND p.target_date <= ?""",
        (today.isoformat(),)).fetchall()
    now = datetime.now(timezone.utc).isoformat()
    scored = waiting = 0
    for p in pending:
        row = conn.execute("SELECT close FROM prices WHERE symbol=? AND date=?",
                           (p["symbol"], p["target_date"])).fetchone()
        if row is None:
            waiting += 1
            continue
        change = (row["close"] / p["base_close"] - 1) * 100
        actual = "UP" if change > 0 else "DOWN"
        conn.execute(
            """INSERT INTO evaluations(prediction_id, actual_close, actual_change_pct,
                 actual_direction, correct, evaluated_at) VALUES (?,?,?,?,?,?)""",
            (p["id"], row["close"], change, actual, int(actual == p["direction"]), now))
        scored += 1
    return {"scored": scored, "waiting": waiting}


def run(today: date | None = None, fetch: bool = True) -> dict:
    today = today or datetime.now(local_tz()).date()
    started = datetime.now(timezone.utc).isoformat()
    error = None
    if fetch:
        try:
            prices.refresh()
        except Exception as exc:  # noqa: BLE001 - still score what we can
            error = f"price update failed: {exc}"
            log.error(error)
    with session() as conn:
        result = score(conn, today)
        conn.execute(
            """INSERT INTO runs(run_type, run_date, started_at, finished_at, status, n_items, error)
               VALUES ('evaluate',?,?,?,?,?,?)""",
            (today.isoformat(), started, datetime.now(timezone.utc).isoformat(),
             "ok" if not error else "partial", result["scored"], error))
    if result["waiting"]:
        log.warning("%d predictions still have no closing price; they will be retried next run",
                    result["waiting"])
    log.info("Scored %d predictions", result["scored"])
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--no-fetch", action="store_true")
    args = p.parse_args()
    run(fetch=not args.no_fetch)
