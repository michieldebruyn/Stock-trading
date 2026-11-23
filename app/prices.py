"""Daily price download from Yahoo Finance into the `prices` table.

    python -m app.prices            # update all active tickers + context symbols
    python -m app.prices --years 2  # backfill two years of history
"""
from __future__ import annotations

import argparse
import logging
import sqlite3
from datetime import date, timedelta

import pandas as pd
import yfinance as yf

from .config import get_config
from .db import session

log = logging.getLogger(__name__)


def _upsert(conn: sqlite3.Connection, symbol: str, df: pd.DataFrame) -> int:
    rows = []
    for ts, r in df.iterrows():
        if pd.isna(r.get("Close")):
            continue
        rows.append((
            symbol, ts.strftime("%Y-%m-%d"),
            _f(r.get("Open")), _f(r.get("High")), _f(r.get("Low")),
            float(r["Close"]), _f(r.get("Volume")),
        ))
    conn.executemany(
        """INSERT INTO prices(symbol,date,open,high,low,close,volume) VALUES (?,?,?,?,?,?,?)
           ON CONFLICT(symbol,date) DO UPDATE SET open=excluded.open, high=excluded.high,
             low=excluded.low, close=excluded.close, volume=excluded.volume""",
        rows,
    )
    return len(rows)


def _f(v):
    return None if v is None or pd.isna(v) else float(v)


def download(symbols: list[str], start: date, end: date | None = None, chunk: int = 100) -> dict[str, pd.DataFrame]:
    """Download daily bars. Returns {symbol: DataFrame[Open,High,Low,Close,Volume]}.

    Uses split/dividend-adjusted prices so returns are comparable across days.
    """
    out: dict[str, pd.DataFrame] = {}
    end = end or date.today() + timedelta(days=1)
    for i in range(0, len(symbols), chunk):
        batch = symbols[i:i + chunk]
        data = yf.download(
            batch, start=start.isoformat(), end=end.isoformat(), interval="1d",
            auto_adjust=True, group_by="ticker", progress=False, threads=True,
        )
        if data is None or data.empty:
            continue
        for sym in batch:
            try:
                df = data[sym] if isinstance(data.columns, pd.MultiIndex) else data
            except KeyError:
                continue
            df = df.dropna(how="all")
            if not df.empty:
                out[sym] = df
    return out


def update(symbols: list[str] | None = None, years: float | None = None) -> int:
    """Fetch recent bars for all active tickers (and context symbols). Returns rows written."""
    with session() as conn:
        if symbols is None:
            symbols = [r[0] for r in conn.execute("SELECT symbol FROM tickers WHERE active=1")]
            symbols += list(get_config().get("context_symbols", {}).keys())
        last = conn.execute("SELECT MAX(date) FROM prices").fetchone()[0]

    if years:
        start = date.today() - timedelta(days=int(365 * years))
    elif last:
        # Re-fetch a few days so late corrections from Yahoo are picked up.
        start = date.fromisoformat(last) - timedelta(days=7)
    else:
        start = date.today() - timedelta(days=400)  # enough for 200-day averages

    frames = download(symbols, start)
    written = 0
    with session() as conn:
        for sym, df in frames.items():
            written += _upsert(conn, sym, df)
    missing = sorted(set(symbols) - set(frames))
    if missing:
        log.warning("No data for %d symbols: %s", len(missing), ", ".join(missing[:20]))
    log.info("Wrote %d price rows for %d symbols", written, len(frames))
    return written


_intraday_cache: dict[str, tuple[float, list[dict]]] = {}


def intraday(symbol: str, ttl: int = 300) -> list[dict]:
    """Today's 5-minute bars for the live view on the website (cached for `ttl` seconds)."""
    import time
    now = time.time()
    hit = _intraday_cache.get(symbol)
    if hit and now - hit[0] < ttl:
        return hit[1]
    df = yf.Ticker(symbol).history(period="1d", interval="5m", auto_adjust=True)
    bars = [{"t": ts.isoformat(), "close": float(r["Close"])} for ts, r in df.iterrows()]
    _intraday_cache[symbol] = (now, bars)
    return bars


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--years", type=float, help="backfill this many years of history")
    args = p.parse_args()
    update(years=args.years)
