"""Render the website as static files for GitHub Pages.

    python scripts/build_site.py --out _site --root /Stock-trading

Every page is rendered by the real app (through FastAPI's test client, so nothing listens on a
port) with links rewritten for a static site: pages become folders (/ticker/KBC.BR/index.html)
and every JSON answer the charts need becomes a file (api/ticker/KBC.BR/chart-1d.json). It reads
the local database only: run `python scripts/state.py pull` first to load the latest records.
Live Yahoo Finance features are left out, since a static site cannot fetch them.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from app.db import connect  # noqa: E402
from app.web import main  # noqa: E402

CHART_RANGES = list(main.CHART_RANGES)


def build(out: Path, root: str = "") -> dict:
    t0 = time.time()
    main.configure_site(root=root, static=True)
    main.templates.env.globals["built_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    client = TestClient(main.app)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    counts = {"pages": 0, "json": 0}

    def page(path: str, **params) -> None:
        res = client.get(path, params=params)
        res.raise_for_status()
        folder = main.static_page_path(path, params).split("#")[0]
        target = out / folder.strip("/") / "index.html"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(res.content)
        counts["pages"] += 1

    def data(path: str, **params) -> None:
        res = client.get(path, params=params)
        res.raise_for_status()
        name = path.strip("/") + ("-" + "-".join(str(v) for v in params.values()) if params else "") + ".json"
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(res.content)
        counts["json"] += 1

    with connect() as conn:
        symbols = [r[0] for r in conn.execute("SELECT symbol FROM tickers WHERE active=1 ORDER BY symbol")]
        dates = {h: [r[0] for r in conn.execute(
            "SELECT DISTINCT made_on FROM predictions WHERE horizon=? ORDER BY made_on DESC LIMIT 60", (h,))]
            for h in ("daily", "weekly")}

    page("/")
    page("/predictions")
    for horizon, ds in dates.items():
        page("/predictions", horizon=horizon)
        for d in ds:
            page("/predictions", horizon=horizon, made_on=d)
    page("/accuracy")
    for days in main.ACCURACY_DAYS:
        page("/accuracy", days=days)
        for horizon in ("daily", "weekly"):
            data("/api/accuracy/series", horizon=horizon, days=days)
            data("/api/accuracy/moves", horizon=horizon, days=days)
    data("/api/tickers")
    for sym in symbols:
        page(f"/ticker/{sym}")
        data(f"/api/ticker/{sym}/history", days=180)
        for rng in CHART_RANGES:
            data(f"/api/ticker/{sym}/chart", range=rng)

    shutil.copytree(main.HERE / "static", out / "static")
    (out / ".nojekyll").write_text("")          # serve folders as-is, no Jekyll processing
    (out / "404.html").write_text(
        f'<!doctype html><meta charset="utf-8"><title>Not found</title>'
        f'<p style="font-family:system-ui;padding:2rem">Page not found. '
        f'<a href="{main.url("/")}">Back to the dashboard</a></p>', encoding="utf-8")
    counts["seconds"] = round(time.time() - t0, 1)
    return counts


def main_cli() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", type=Path, default=ROOT / "_site")
    p.add_argument("--root", default="", help="base path the site is served under, e.g. /Stock-trading")
    args = p.parse_args()
    counts = build(args.out, args.root)
    print(f"Built {counts['pages']} pages and {counts['json']} data files in {counts['seconds']}s -> {args.out}")


if __name__ == "__main__":
    main_cli()
