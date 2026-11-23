"""Morning job: refresh prices, ask Claude for daily + weekly direction calls, store them.

    python -m app.predict                 # normal morning run
    python -m app.predict --no-fetch      # skip the price download
    python -m app.predict --dry-run       # show what would be requested, no API call
    python -m app.predict --force         # also predict markets that have already opened
"""
from __future__ import annotations

import argparse
import logging
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone

from . import features, prices, stats, universe
from .claude_predictor import ClaudePredictor
from .config import get_config, local_tz
from .db import session
from .market import has_opened, is_trading_day, trading_days, week_trading_days

log = logging.getLogger("predict")


def plan(conn: sqlite3.Connection, today: date, now_utc: datetime, force: bool = False):
    """Decide which (symbol, horizon) predictions to request today.

    Returns (jobs, feats, meta, skipped) where jobs maps symbol -> list of horizon dicts.
    """
    tickers = {r["symbol"]: dict(r) for r in conn.execute(
        "SELECT symbol, name, exchange, sector FROM tickers WHERE active=1")}
    existing = {(r[0], r[1], r[2]) for r in conn.execute(
        "SELECT symbol, horizon, period_start FROM predictions WHERE period_start = ?",
        (today.isoformat(),))}
    feats = features.feature_table(conn, list(tickers), before=today)

    jobs: dict[str, list[dict]] = {}
    skipped: dict[str, list[str]] = {}

    def skip(reason, sym):
        skipped.setdefault(reason, []).append(sym)

    for sym, t in tickers.items():
        ex = t["exchange"]
        if not is_trading_day(ex, today):
            skip("market closed today", sym)
            continue
        if not force and has_opened(ex, today, now_utc):
            skip("market already open", sym)
            continue
        f = feats.get(sym)
        if not f:
            skip("not enough price history", sym)
            continue
        prev = trading_days(ex, date.fromordinal(today.toordinal() - 10), date.fromordinal(today.toordinal() - 1))
        if not prev or f["last_date"] != prev[-1].isoformat():
            skip("price data not up to date", sym)
            continue

        items = []
        if (sym, "daily", today.isoformat()) not in existing:
            items.append({"horizon": "daily", "period_start": today, "target_date": today})
        week = week_trading_days(ex, today)
        # Weekly calls are made on the first trading day of the week only.
        if week and week[0] == today and (sym, "weekly", today.isoformat()) not in existing:
            items.append({"horizon": "weekly", "period_start": today, "target_date": week[-1]})
        if items:
            for it in items:
                it.update(base_date=f["last_date"], base_close=f["close"])
            jobs[sym] = items
    return jobs, feats, tickers, skipped


def market_context(conn: sqlite3.Connection, today: date) -> str:
    syms = get_config().get("context_symbols", {})
    if not syms:
        return ""
    lines = ["symbol|name|last_date|last_close|r1d|r5d|r20d"]
    rows = conn.execute(
        f"""SELECT symbol, date, close FROM prices WHERE symbol IN ({','.join('?' * len(syms))})
            AND date <= ? AND date >= date(?, '-45 days') ORDER BY symbol, date""",
        (*syms, today.isoformat(), today.isoformat())).fetchall()
    by_sym: dict[str, list] = {}
    for r in rows:
        by_sym.setdefault(r["symbol"], []).append((r["date"], r["close"]))
    for sym, name in syms.items():
        h = by_sym.get(sym)
        if not h or len(h) < 2:
            continue
        c = [x[1] for x in h]

        def ret(n):
            return f"{(c[-1] / c[-1 - n] - 1) * 100:+.2f}" if len(c) > n else ""
        lines.append(f"{sym}|{name}|{h[-1][0]}|{c[-1]:.2f}|{ret(1)}|{ret(5)}|{ret(20)}")
    return "\n".join(lines)


def build_run_context(conn: sqlite3.Connection, today: date, now_local: datetime) -> str:
    return (
        f"Today is {today:%A %d %B %Y}; local time {now_local:%H:%M %Z}. "
        f"Markets have not opened yet for the stocks below.\n\n"
        f"Market backdrop (futures/Asia may already include today's session):\n"
        f"{market_context(conn, today)}\n\n"
        f"Your track record:\n{stats.track_record_text(conn, today)}"
    )


def run(today: date | None = None, fetch: bool = True, dry_run: bool = False,
        force: bool = False, predictor: ClaudePredictor | None = None) -> dict:
    cfg = get_config()
    ccfg = cfg["claude"]
    tz = local_tz()
    now_local = datetime.now(tz)
    now_utc = now_local.astimezone(timezone.utc)
    today = today or now_local.date()

    if fetch:
        universe.sync()
        prices.update()

    with session() as conn:
        jobs, feats, meta, skipped = plan(conn, today, now_utc, force)
        for reason, syms in skipped.items():
            log.info("Skipped %d tickers (%s)", len(syms), reason)
        n_pairs = sum(len(v) for v in jobs.values())
        log.info("Requesting %d predictions for %d tickers", n_pairs, len(jobs))
        if not jobs:
            return {"requested": 0, "stored": 0, "skipped": {k: len(v) for k, v in skipped.items()}}

        run_context = build_run_context(conn, today, now_local)
        records = stats.recent_record(conn, list(jobs))
        if dry_run:
            print(run_context)
            print(f"\n{n_pairs} predictions in {len(jobs)} tickers")
            return {"requested": n_pairs, "stored": 0}

        run_id = conn.execute(
            "INSERT INTO runs(run_type, run_date, started_at, status, model) VALUES ('predict',?,?,?,?)",
            (today.isoformat(), now_utc.isoformat(), "running", ccfg["model"])).lastrowid
        conn.commit()

    predictor = predictor or ClaudePredictor(ccfg["model"], ccfg.get("effort", "medium"),
                                             ccfg.get("server_fallback", True))
    symbols = list(jobs)
    size = int(ccfg.get("batch_size", 40))
    batches = [symbols[i:i + size] for i in range(0, len(symbols), size)]

    def do_batch(batch):
        rows = features.to_rows({s: feats[s] for s in batch}, meta)
        rows = _add_record_column(rows, records)
        wanted = [(s, it["horizon"]) for s in batch for it in jobs[s]]
        return predictor.predict_batch(run_context, rows, wanted)

    stored, market_view, errors = 0, None, []
    created_at = datetime.now(timezone.utc).isoformat()
    with ThreadPoolExecutor(max_workers=int(ccfg.get("max_parallel", 4))) as pool:
        futures = {pool.submit(do_batch, b): b for b in batches}
        for fut in as_completed(futures):
            batch = futures[fut]
            try:
                result = fut.result()
            except Exception as exc:  # noqa: BLE001 - one bad batch must not kill the run
                log.error("Batch starting %s failed: %s", batch[0], exc)
                errors.append(f"{batch[0]}..: {exc}")
                continue
            market_view = market_view or result.market_view
            with session() as conn:
                for c in result.predictions:
                    it = next(i for i in jobs[c.symbol] if i["horizon"] == c.horizon)
                    conn.execute(
                        """INSERT OR IGNORE INTO predictions(run_id, symbol, horizon, made_on,
                             period_start, target_date, base_date, base_close, direction,
                             confidence, expected_move_pct, rationale, model, created_at)
                           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (run_id, c.symbol, c.horizon, today.isoformat(),
                         it["period_start"].isoformat(), it["target_date"].isoformat(),
                         it["base_date"], it["base_close"], c.direction, c.confidence,
                         c.expected_move_pct, c.rationale, ccfg["model"], created_at))
                    stored += 1
    missing = n_pairs - stored
    if missing:
        log.warning("%d requested predictions were not returned", missing)

    status = "ok" if not errors and not missing else ("partial" if stored else "failed")
    with session() as conn:
        conn.execute(
            """UPDATE runs SET finished_at=?, status=?, input_tokens=?, output_tokens=?,
                 n_items=?, market_view=?, error=? WHERE id=?""",
            (datetime.now(timezone.utc).isoformat(), status, predictor.usage.input_tokens,
             predictor.usage.output_tokens, stored, market_view,
             "\n".join(errors) or None, run_id))
    log.info("Stored %d predictions (status %s, %d in / %d out tokens)", stored, status,
             predictor.usage.input_tokens, predictor.usage.output_tokens)
    return {"requested": n_pairs, "stored": stored, "status": status, "run_id": run_id}


def _add_record_column(rows: str, records: dict[str, str]) -> str:
    lines = rows.split("\n")
    out = [lines[0] + "|record"]
    for line in lines[1:]:
        out.append(line + "|" + records.get(line.split("|", 1)[0], ""))
    return "\n".join(out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--no-fetch", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    run(fetch=not args.no_fetch, dry_run=args.dry_run, force=args.force)
