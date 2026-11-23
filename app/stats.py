"""Accuracy queries shared by the website and the prompt's track-record section."""
from __future__ import annotations

import sqlite3
from datetime import date, timedelta

SCORED = """
    FROM predictions p
    JOIN evaluations e ON e.prediction_id = p.id
"""


def accuracy(conn: sqlite3.Connection, horizon: str, since: date | None = None,
             universe: str | None = None) -> dict:
    sql = f"SELECT COUNT(*) n, SUM(e.correct) hits, AVG(p.confidence) conf, " \
          f"AVG(CASE WHEN p.direction='UP' THEN 1.0 ELSE 0 END) up_share, " \
          f"AVG(CASE WHEN e.actual_direction='UP' THEN 1.0 ELSE 0 END) actual_up " \
          f"{SCORED} WHERE p.horizon = ?"
    params: list = [horizon]
    if since:
        sql += " AND p.target_date >= ?"
        params.append(since.isoformat())
    if universe:
        sql += " AND p.symbol IN (SELECT symbol FROM ticker_universes WHERE universe = ?)"
        params.append(universe)
    r = conn.execute(sql, params).fetchone()
    n = r["n"] or 0
    return {
        "n": n,
        "hits": r["hits"] or 0,
        "rate": (r["hits"] or 0) / n if n else None,
        "avg_conf": r["conf"],
        "up_share": r["up_share"],
        "actual_up": r["actual_up"],
    }


def calibration(conn: sqlite3.Connection, horizon: str, since: date | None = None) -> list[dict]:
    sql = f"""SELECT CAST(MIN(p.confidence, 0.949) * 20 AS INT) / 20.0 AS bucket,
                     COUNT(*) n, AVG(e.correct) rate, AVG(p.confidence) conf
              {SCORED} WHERE p.horizon = ? AND p.target_date >= ?
              GROUP BY bucket ORDER BY bucket"""
    since = since or date(2000, 1, 1)
    return [dict(r) for r in conn.execute(sql, (horizon, since.isoformat()))]


def daily_series(conn: sqlite3.Connection, horizon: str, universe: str | None = None,
                 since: date | None = None) -> list[dict]:
    """Hit rate per target date (optionally per universe)."""
    sql = f"SELECT p.target_date d, COUNT(*) n, AVG(e.correct) rate {SCORED} WHERE p.horizon = ?"
    params: list = [horizon]
    if universe:
        sql += " AND p.symbol IN (SELECT symbol FROM ticker_universes WHERE universe = ?)"
        params.append(universe)
    if since:
        sql += " AND p.target_date >= ?"
        params.append(since.isoformat())
    sql += " GROUP BY p.target_date ORDER BY p.target_date"
    return [dict(r) for r in conn.execute(sql, params)]


def ticker_ranking(conn: sqlite3.Connection, horizon: str, since: date, min_n: int = 5) -> list[dict]:
    sql = f"""SELECT p.symbol, t.name, COUNT(*) n, AVG(e.correct) rate
              {SCORED} JOIN tickers t ON t.symbol = p.symbol
              WHERE p.horizon = ? AND p.target_date >= ?
              GROUP BY p.symbol HAVING n >= ? ORDER BY rate DESC, n DESC"""
    return [dict(r) for r in conn.execute(sql, (horizon, since.isoformat(), min_n))]


def recent_record(conn: sqlite3.Connection, symbols: list[str], n: int = 10) -> dict[str, str]:
    """Per symbol, the last n daily outcomes as a '+/-' string (oldest first)."""
    if not symbols:
        return {}
    sql = f"""SELECT symbol, correct FROM (
                SELECT p.symbol, e.correct, ROW_NUMBER() OVER
                  (PARTITION BY p.symbol ORDER BY p.target_date DESC) rn, p.target_date
                {SCORED} WHERE p.horizon = 'daily'
                  AND p.symbol IN ({','.join('?' * len(symbols))}))
              WHERE rn <= ? ORDER BY symbol, target_date"""
    out: dict[str, str] = {}
    for r in conn.execute(sql, (*symbols, n)):
        out[r["symbol"]] = out.get(r["symbol"], "") + ("+" if r["correct"] else "-")
    return out


def track_record_text(conn: sqlite3.Connection, today: date) -> str:
    """Summary of Claude's own past performance, included in the prompt."""
    lines = []
    for horizon, days in (("daily", 30), ("weekly", 90)):
        since = today - timedelta(days=days)
        overall = accuracy(conn, horizon, since)
        if not overall["n"]:
            lines.append(f"{horizon}: no scored predictions yet.")
            continue
        parts = [f"{horizon}, last {days} days: {overall['rate']:.1%} right on {overall['n']} calls; "
                 f"avg confidence {overall['avg_conf']:.2f}; you called UP {overall['up_share']:.0%} "
                 f"of the time, actual UP share was {overall['actual_up']:.0%}."]
        for uni in ("sp500", "europe", "bel20"):
            a = accuracy(conn, horizon, since, uni)
            if a["n"]:
                parts.append(f"  {uni}: {a['rate']:.1%} on {a['n']}")
        cal = [c for c in calibration(conn, horizon, since) if c["n"] >= 10]
        if cal:
            parts.append("  calibration (stated -> actual): " + ", ".join(
                f"{c['conf']:.2f}->{c['rate']:.2f} (n={c['n']})" for c in cal))
        lines.append("\n".join(parts))
    return "\n".join(lines)
