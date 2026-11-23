"""Technical features computed from the prices table, fed to Claude as compact text."""
from __future__ import annotations

import sqlite3
from datetime import date, timedelta

import numpy as np
import pandas as pd


def load_closes(conn: sqlite3.Connection, symbols: list[str], before: date, days: int = 400) -> pd.DataFrame:
    """Long-format price history strictly before `before` (no look-ahead)."""
    start = (before - timedelta(days=days)).isoformat()
    q = f"""SELECT symbol, date, high, low, close, volume FROM prices
            WHERE date >= ? AND date < ? AND symbol IN ({','.join('?' * len(symbols))})
            ORDER BY symbol, date"""
    return pd.read_sql_query(q, conn, params=[start, before.isoformat(), *symbols])


def _rsi(close: pd.Series, n: int = 14) -> float:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = gain.iloc[-1] / loss.iloc[-1] if loss.iloc[-1] else np.inf
    return float(100 - 100 / (1 + rs))


def _pct(a: float, b: float) -> float:
    return float((a / b - 1) * 100) if b else float("nan")


def compute(df: pd.DataFrame) -> dict | None:
    """Features for one symbol's history (sorted by date). None if too little data."""
    if len(df) < 30:
        return None
    c = df["close"].astype(float).reset_index(drop=True)
    last = c.iloc[-1]
    ema12, ema26 = c.ewm(span=12).mean(), c.ewm(span=26).mean()
    macd = ema12 - ema26
    hist = macd - macd.ewm(span=9).mean()
    tr = pd.concat([
        df["high"].astype(float) - df["low"].astype(float),
        (df["high"].astype(float) - c.shift()).abs(),
        (df["low"].astype(float) - c.shift()).abs(),
    ], axis=1).max(axis=1)
    vol = df["volume"].astype(float).fillna(0)
    rets = c.pct_change() * 100
    year = c.iloc[-252:]

    def sma_gap(n):
        return _pct(last, c.iloc[-n:].mean()) if len(c) >= n else float("nan")

    return {
        "last_date": df["date"].iloc[-1],
        "close": float(last),
        "r1d": float(rets.iloc[-1]),
        "r5d": _pct(last, c.iloc[-6]),
        "r20d": _pct(last, c.iloc[-21]) if len(c) > 21 else float("nan"),
        "r60d": _pct(last, c.iloc[-61]) if len(c) > 61 else float("nan"),
        "rsi14": _rsi(c),
        "macd_h": float(hist.iloc[-1] / last * 100),
        "sma20": sma_gap(20),
        "sma50": sma_gap(50),
        "sma200": sma_gap(200),
        "atr14": float(tr.iloc[-14:].mean() / last * 100),
        "vol_ratio": float(vol.iloc[-5:].mean() / vol.iloc[-20:].mean()) if vol.iloc[-20:].mean() else float("nan"),
        "from_52w_hi": _pct(last, year.max()),
        "last5": " ".join(f"{x:+.1f}" for x in rets.iloc[-5:]),
    }


COLUMNS = ["r1d", "r5d", "r20d", "r60d", "rsi14", "macd_h", "sma20", "sma50", "sma200",
           "atr14", "vol_ratio", "from_52w_hi"]


def feature_table(conn: sqlite3.Connection, symbols: list[str], before: date) -> dict[str, dict]:
    hist = load_closes(conn, symbols, before)
    out = {}
    for sym, df in hist.groupby("symbol", sort=False):
        f = compute(df.reset_index(drop=True))
        if f:
            out[sym] = f
    return out


def to_rows(feats: dict[str, dict], meta: dict[str, dict]) -> str:
    """Pipe-separated table: compact for the prompt and easy for the model to read."""
    header = "symbol|name|sector|last_close|" + "|".join(COLUMNS) + "|last5_daily_%"
    lines = [header]
    for sym, f in feats.items():
        m = meta.get(sym, {})
        vals = "|".join("" if np.isnan(f[k]) else f"{f[k]:.2f}" for k in COLUMNS)
        lines.append(f"{sym}|{m.get('name','')}|{m.get('sector','')}|{f['close']:.2f}|{vals}|{f['last5']}")
    return "\n".join(lines)
