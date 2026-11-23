"""Loads config.yaml once and exposes it as a dict."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

ROOT = Path(__file__).resolve().parent.parent


@lru_cache(maxsize=1)
def get_config() -> dict:
    path = Path(os.environ.get("STOCKS_CONFIG", ROOT / "config.yaml"))
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    # STOCKS_DB lets tests / the demo use a separate database file.
    cfg["database"] = os.environ.get("STOCKS_DB", cfg.get("database", "data/stocks.db"))
    return cfg


def db_path() -> Path:
    p = Path(get_config()["database"])
    return p if p.is_absolute() else ROOT / p


def local_tz() -> ZoneInfo:
    return ZoneInfo(get_config().get("timezone", "Europe/Brussels"))
