"""Fill a separate demo database with synthetic prices and fake predictions so you can see the
website before real data exists. Never touches data/stocks.db.

    python scripts/seed_demo.py
    STOCKS_DB=data/demo.db python -m app.serve --no-scheduler
"""
from __future__ import annotations

import os
import random
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("STOCKS_DB", str(ROOT / "data" / "demo.db"))

from app import evaluate, predict, universe  # noqa: E402
from app.claude_predictor import BatchResult, Call, Usage  # noqa: E402
from app.config import db_path  # noqa: E402
from app.db import session  # noqa: E402
from app.market import trading_days  # noqa: E402

US_DEMO = [("AAPL", "Apple", "Information Technology"), ("MSFT", "Microsoft", "Information Technology"),
           ("NVDA", "NVIDIA", "Information Technology"), ("AMZN", "Amazon", "Consumer Discretionary"),
           ("JPM", "JPMorgan Chase", "Financials"), ("XOM", "Exxon Mobil", "Energy"),
           ("JNJ", "Johnson & Johnson", "Health Care"), ("KO", "Coca-Cola", "Consumer Staples")]


class FakePredictor:
    """Stands in for Claude: leans on 5-day momentum with noise, so accuracy hovers near 50%."""

    def __init__(self, seed=1):
        self.rng = random.Random(seed)
        self.usage = Usage()

    def predict_batch(self, run_context, rows, wanted):
        r5 = {}
        for line in rows.split("\n")[1:]:
            parts = line.split("|")
            r5[parts[0]] = float(parts[5] or 0)
        calls = []
        for sym, horizon in wanted:
            up = (r5.get(sym, 0) + self.rng.gauss(0, 3)) > 0
            conf = round(self.rng.uniform(0.5, 0.75), 2)
            move = self.rng.uniform(0.2, 1.5) * (3 if horizon == "weekly" else 1)
            calls.append(Call(symbol=sym, horizon=horizon, direction="UP" if up else "DOWN",
                              confidence=conf, expected_move_pct=move if up else -move,
                              rationale="Demo call: momentum " + ("positive" if up else "negative") + "."))
        self.usage.calls += 1
        self.usage.input_tokens += 4000
        self.usage.output_tokens += 60 * len(wanted)
        return BatchResult(market_view="Demo data. Markets mixed; futures flat overnight, "
                                       "volatility subdued. No real forecast.", predictions=calls)


def main():
    path = db_path()
    if path.name == "stocks.db":
        sys.exit("Refusing to seed the real database; set STOCKS_DB to another file.")
    for suffix in ("", "-wal", "-shm"):
        Path(str(path) + suffix).unlink(missing_ok=True)

    universe.sync()
    with session() as conn:
        for sym, name, sector in US_DEMO:
            conn.execute("INSERT OR IGNORE INTO tickers(symbol,name,exchange,sector,active) VALUES (?,?,?,?,1)",
                         (sym, name, "XNYS", sector))
            conn.execute("INSERT OR IGNORE INTO ticker_universes VALUES (?, 'sp500')", (sym,))
        tickers = [dict(r) for r in conn.execute("SELECT symbol, exchange FROM tickers WHERE active=1")]

    rng = random.Random(7)
    end = date.today() - timedelta(days=1)
    hist_start, pred_start = end - timedelta(days=420), end - timedelta(days=70)
    # Shared market factor per exchange-day makes stocks move together, like real markets.
    market = {}
    series = {}
    for t in tickers:
        days = trading_days(t["exchange"], hist_start, end)
        price, vol = rng.uniform(20, 400), rng.uniform(0.008, 0.025)
        rows = []
        for d in days:
            m = market.setdefault(d, rng.gauss(0.0004, 0.009))
            price *= 1 + m + rng.gauss(0, vol)
            rows.append((d, price))
        series[t["symbol"]] = rows

    def insert_until(cutoff_exclusive: date, cutoff_inclusive: date | None = None):
        with session() as conn:
            for sym, rows in series.items():
                conn.executemany(
                    "INSERT OR IGNORE INTO prices(symbol,date,open,high,low,close,volume) VALUES (?,?,?,?,?,?,?)",
                    [(sym, d.isoformat(), p, p * 1.01, p * 0.99, p, rng.uniform(1e5, 5e6))
                     for d, p in rows if d < cutoff_exclusive or (cutoff_inclusive and d <= cutoff_inclusive)])

    insert_until(pred_start)
    fake = FakePredictor()
    d = pred_start
    while d <= end:
        if d.weekday() < 5:
            predict.run(today=d, fetch=False, force=True, predictor=fake)
            insert_until(d, d)
            with session() as conn:
                evaluate.score(conn, d)
        d += timedelta(days=1)
    with session() as conn:
        n = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        e = conn.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0]
    print(f"Demo database {path}: {len(tickers)} tickers, {n} predictions, {e} scored.")


if __name__ == "__main__":
    main()
