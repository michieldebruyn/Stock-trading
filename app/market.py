"""Exchange calendars: which days are trading days, and when sessions open."""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from functools import lru_cache

import pandas as pd
import pandas_market_calendars as mcal

# Yahoo ticker suffix -> market calendar code
SUFFIX_EXCHANGE = {
    "BR": "XBRU", "AS": "XAMS", "PA": "XPAR", "DE": "XETR", "F": "XETR",
    "L": "XLON", "SW": "XSWX", "MI": "XMIL", "MC": "XMAD", "CO": "XCSE",
    "ST": "XSTO", "HE": "XHEL", "OL": "XOSL", "LS": "XLIS", "VI": "XWBO",
    "IR": "XDUB",
}


def exchange_for(symbol: str) -> str:
    if "." in symbol:
        suffix = symbol.rsplit(".", 1)[1].upper()
        return SUFFIX_EXCHANGE.get(suffix, "XNYS")
    return "XNYS"


@lru_cache(maxsize=None)
def _calendar(exchange: str):
    return mcal.get_calendar(exchange)


@lru_cache(maxsize=4096)
def _schedule(exchange: str, start: date, end: date) -> pd.DataFrame:
    return _calendar(exchange).schedule(start_date=start, end_date=end)


def trading_days(exchange: str, start: date, end: date) -> list[date]:
    sched = _schedule(exchange, start, end)
    return [d.date() for d in sched.index]


def is_trading_day(exchange: str, d: date) -> bool:
    return bool(trading_days(exchange, d, d))


def session_open_utc(exchange: str, d: date) -> datetime | None:
    sched = _schedule(exchange, d, d)
    if sched.empty:
        return None
    return sched.iloc[0]["market_open"].to_pydatetime().astimezone(timezone.utc)


def has_opened(exchange: str, d: date, now_utc: datetime) -> bool:
    opens = session_open_utc(exchange, d)
    return opens is not None and now_utc >= opens


def week_bounds(d: date) -> tuple[date, date]:
    monday = d - timedelta(days=d.weekday())
    return monday, monday + timedelta(days=6)


def week_trading_days(exchange: str, d: date) -> list[date]:
    monday, sunday = week_bounds(d)
    return trading_days(exchange, monday, sunday)
