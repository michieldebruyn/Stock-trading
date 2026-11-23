"""Ticker universes: S&P 500, large European stocks, BEL20.

    python -m app.universe refresh   # download S&P 500 list + sync all lists into the DB
    python -m app.universe sync      # sync CSV lists into the DB only
"""
from __future__ import annotations

import io
import sys

import pandas as pd
import requests

from .config import ROOT, get_config
from .db import session
from .market import exchange_for

UNIVERSE_DIR = ROOT / "data" / "universe"
SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def download_sp500() -> pd.DataFrame:
    resp = requests.get(SP500_URL, headers={"User-Agent": "stock-prediction-app/0.1"}, timeout=30)
    resp.raise_for_status()
    table = pd.read_html(io.StringIO(resp.text), attrs={"id": "constituents"})[0]
    df = pd.DataFrame({
        # Yahoo uses '-' where the index uses '.', e.g. BRK.B -> BRK-B
        "symbol": table["Symbol"].str.replace(".", "-", regex=False),
        "name": table["Security"],
        "sector": table["GICS Sector"],
    })
    df.to_csv(UNIVERSE_DIR / "sp500.csv", index=False)
    return df


def load_universe(name: str) -> pd.DataFrame:
    path = UNIVERSE_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing. Run: python -m app.universe refresh")
    return pd.read_csv(path)


def sync() -> int:
    """Write the configured universes into tickers / ticker_universes. Returns ticker count."""
    names = get_config()["universes"]
    with session() as conn:
        conn.execute("DELETE FROM ticker_universes")
        conn.execute("UPDATE tickers SET active = 0")
        for uni in names:
            try:
                df = load_universe(uni)
            except FileNotFoundError as exc:
                print(f"warning: {exc}", file=sys.stderr)
                continue
            for row in df.itertuples(index=False):
                conn.execute(
                    """INSERT INTO tickers(symbol, name, exchange, sector, active)
                       VALUES (?,?,?,?,1)
                       ON CONFLICT(symbol) DO UPDATE SET name=excluded.name,
                         exchange=excluded.exchange, sector=excluded.sector, active=1""",
                    (row.symbol, row.name, exchange_for(row.symbol), row.sector),
                )
                conn.execute(
                    "INSERT OR IGNORE INTO ticker_universes(symbol, universe) VALUES (?,?)",
                    (row.symbol, uni),
                )
        return conn.execute("SELECT COUNT(*) FROM tickers WHERE active=1").fetchone()[0]


def main(argv: list[str]) -> None:
    cmd = argv[0] if argv else "sync"
    if cmd == "refresh":
        try:
            df = download_sp500()
            print(f"Downloaded {len(df)} S&P 500 constituents")
        except Exception as exc:  # noqa: BLE001 - fall back to the list in the repo
            if not (UNIVERSE_DIR / "sp500.csv").exists():
                raise
            print(f"warning: S&P 500 download failed ({type(exc).__name__}); "
                  f"using the saved data/universe/sp500.csv", file=sys.stderr)
    print(f"{sync()} active tickers in database")


if __name__ == "__main__":
    main(sys.argv[1:])
