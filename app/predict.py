"""Morning job: refresh prices, ask Claude for daily + weekly direction calls, store them.

    python -m app.predict                 # normal morning run
    python -m app.predict --no-fetch      # skip the price download
    python -m app.predict --dry-run       # show what would be requested, no API call
    python -m app.predict --force         # also predict markets that have already opened

Without an API key, a Claude Code session can make the calls instead: see export_batches below.
"""
from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from . import features, prices, stats, universe
from .claude_predictor import (OUTPUT_SCHEMA, SYSTEM_PROMPT, ClaudePredictor, parse_result,
                               user_message)
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


def _prepare(today: date | None, fetch: bool, force: bool):
    """Shared start of a morning run: refresh data, plan today's calls, build the prompt parts."""
    tz = local_tz()
    now_local = datetime.now(tz)
    now_utc = now_local.astimezone(timezone.utc)
    today = today or now_local.date()

    if fetch:
        universe.sync()
        prices.refresh()

    with session() as conn:
        jobs, feats, meta, skipped = plan(conn, today, now_utc, force)
        for reason, syms in skipped.items():
            log.info("Skipped %d tickers (%s)", len(syms), reason)
        n_pairs = sum(len(v) for v in jobs.values())
        log.info("Requesting %d predictions for %d tickers", n_pairs, len(jobs))
        run_context = build_run_context(conn, today, now_local) if jobs else ""
        records = stats.recent_record(conn, list(jobs)) if jobs else {}
    return SimpleNamespace(today=today, now_utc=now_utc, jobs=jobs, feats=feats, meta=meta,
                           skipped=skipped, n_pairs=n_pairs, run_context=run_context,
                           records=records)


def _batches(jobs: dict) -> list[list[str]]:
    symbols = list(jobs)
    size = int(get_config()["claude"].get("batch_size", 40))
    return [symbols[i:i + size] for i in range(0, len(symbols), size)]


def _batch_rows(p, batch: list[str]) -> str:
    rows = features.to_rows({s: p.feats[s] for s in batch}, p.meta)
    return _add_record_column(rows, p.records)


def _store(conn: sqlite3.Connection, run_id: int, today: date, jobs: dict, calls, model: str,
           created_at: str) -> int:
    """Insert calls (never overwriting an existing one). Returns how many were new."""
    stored = 0
    for c in calls:
        it = next(i for i in jobs[c.symbol] if i["horizon"] == c.horizon)
        cur = conn.execute(
            """INSERT OR IGNORE INTO predictions(run_id, symbol, horizon, made_on,
                 period_start, target_date, base_date, base_close, direction,
                 confidence, expected_move_pct, rationale, model, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (run_id, c.symbol, c.horizon, today.isoformat(),
             str(it["period_start"]), str(it["target_date"]),
             it["base_date"], it["base_close"], c.direction, c.confidence,
             c.expected_move_pct, c.rationale, model, created_at))
        stored += cur.rowcount
    return stored


def run(today: date | None = None, fetch: bool = True, dry_run: bool = False,
        force: bool = False, predictor: ClaudePredictor | None = None) -> dict:
    ccfg = get_config()["claude"]
    p = _prepare(today, fetch, force)
    today, now_utc, jobs, n_pairs = p.today, p.now_utc, p.jobs, p.n_pairs
    run_context = p.run_context
    if not jobs:
        return {"requested": 0, "stored": 0, "skipped": {k: len(v) for k, v in p.skipped.items()}}

    with session() as conn:
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
    batches = _batches(jobs)

    def do_batch(batch):
        wanted = [(s, it["horizon"]) for s in batch for it in jobs[s]]
        return predictor.predict_batch(run_context, _batch_rows(p, batch), wanted)

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
                stored += _store(conn, run_id, today, jobs, result.predictions, ccfg["model"],
                                 created_at)
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


# --- Hand-off mode: a Claude Code session makes the calls instead of the API --------------------
#
#   python -m app.predict --export-batches work/      # writes instructions + one file per batch
#   (Claude reads work/instructions.md and writes work/answer_NN.json for every batch_NN.md)
#   python -m app.predict --check-answers work/       # validate, store nothing
#   python -m app.predict --import-answers work/      # store the calls

def _answer_name(i: int) -> str:
    return f"answer_{i:02d}.json"


def export_batches(out_dir: Path, today: date | None = None, fetch: bool = True,
                   force: bool = False) -> dict:
    """Write today's prompt, split into batches, for a Claude Code session to answer."""
    p = _prepare(today, fetch, force)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in [*out_dir.glob("batch_*.md"), *out_dir.glob("answer_*.json")]:
        old.unlink()
    # A session answers the batches one after another, so put the markets that open first
    # (Europe, 08:00 UTC) ahead of New York (14:30 UTC).
    p.jobs = dict(sorted(p.jobs.items(), key=lambda kv: p.meta[kv[0]]["exchange"] == "XNYS"))
    batches = _batches(p.jobs)
    plan_doc = {
        "today": p.today.isoformat(),
        "exported_at": p.now_utc.isoformat(),
        "batches": batches,
        "jobs": {s: [{k: str(v) if isinstance(v, date) else v for k, v in it.items()}
                     | {"exchange": p.meta[s]["exchange"]} for it in items]
                 for s, items in p.jobs.items()},
    }
    (out_dir / "plan.json").write_text(json.dumps(plan_doc, indent=1), encoding="utf-8")
    if not batches:
        (out_dir / "instructions.md").write_text(
            "Nothing to predict today (markets closed, already open, or calls already made).\n",
            encoding="utf-8")
        return {"requested": 0, "batches": 0}

    example = {"market_view": "Two or three sentences on today's US and European outlook.",
               "predictions": [{"symbol": "AAPL", "horizon": "daily", "direction": "UP",
                                "confidence": 0.56, "expected_move_pct": 0.4,
                                "rationale": "One short sentence."}]}
    (out_dir / "instructions.md").write_text(
        f"# Morning predictions\n\n{SYSTEM_PROMPT}\n\n"
        f"## Today's context\n\n{p.run_context}\n\n"
        f"## What to do\n\n"
        f"There are {len(batches)} batch files, batch_01.md .. batch_{len(batches):02d}.md, "
        f"{p.n_pairs} predictions in total. For EACH batch_NN.md, read the stock table and the "
        f"requested (symbol, horizon) pairs, reason about them as described above, and write "
        f"your answer to answer_NN.json in this folder (same NN). Each answer is one JSON "
        f"object matching this schema, with exactly one prediction per requested pair. Work "
        f"through the batches in order and write each answer as soon as it is done: the first "
        f"batches hold European stocks, whose markets open first, and calls for a market that "
        f"has opened by the time they are imported are discarded.\n\n"
        f"```json\n{json.dumps(OUTPUT_SCHEMA, indent=1)}\n```\n\nExample:\n\n"
        f"```json\n{json.dumps(example, indent=1)}\n```\n\n"
        f"When all answers are written, run `python -m app.predict --check-answers "
        f"{out_dir.as_posix()}` and fix any batch it reports before importing.\n",
        encoding="utf-8")
    for i, batch in enumerate(batches, 1):
        wanted = [(s, it["horizon"]) for s in batch for it in p.jobs[s]]
        (out_dir / f"batch_{i:02d}.md").write_text(
            f"# Batch {i} of {len(batches)} -> write {_answer_name(i)}\n\n"
            f"{user_message(_batch_rows(p, batch), wanted)}\n", encoding="utf-8")
    log.info("Wrote %d batches (%d predictions) to %s", len(batches), p.n_pairs, out_dir)
    return {"requested": p.n_pairs, "batches": len(batches)}


def import_answers(in_dir: Path, model: str, force: bool = False, check: bool = False) -> dict:
    """Validate the answer files and (unless `check`) store the calls.

    Calls for a market that has opened since the export are dropped unless `force`, so no call
    is ever stored with knowledge of the session it predicts."""
    doc = json.loads((in_dir / "plan.json").read_text(encoding="utf-8"))
    today = date.fromisoformat(doc["today"])
    jobs = doc["jobs"]
    now_utc = datetime.now(timezone.utc)
    calls, market_view, problems, late = [], None, [], 0
    n_pairs = sum(len(v) for v in jobs.values())
    for i, batch in enumerate(doc["batches"], 1):
        wanted = [(s, it["horizon"]) for s in batch for it in jobs[s]]
        f = in_dir / _answer_name(i)
        if not f.exists():
            problems.append(f"{f.name}: missing")
            continue
        try:
            result = parse_result(f.read_text(encoding="utf-8"), wanted)
        except RuntimeError as exc:
            problems.append(f"{f.name}: {exc}")
            continue
        got = {(c.symbol, c.horizon) for c in result.predictions}
        if len(got) < len(wanted):
            lost = [f"{s} {h}" for s, h in wanted if (s, h) not in got]
            problems.append(f"{f.name}: {len(lost)} of {len(wanted)} requested pairs missing: "
                            + ", ".join(lost[:10]) + (" ..." if len(lost) > 10 else ""))
        market_view = market_view or result.market_view
        for c in result.predictions:
            if not force and has_opened(jobs[c.symbol][0]["exchange"], today, now_utc):
                late += 1
                continue
            calls.append(c)
    for msg in problems:
        log.warning(msg)
    if late:
        log.warning("Dropped %d calls: their market opened before the import", late)
    if check:
        print(f"{len(calls)} of {n_pairs} predictions valid; "
              + ("no problems" if not problems else f"{len(problems)} problem(s) listed above")
              + (f"; {late} more would be dropped because their market has opened." if late
                 else "."))
        return {"requested": n_pairs, "valid": len(calls), "problems": problems}

    started = now_utc.isoformat()
    with session() as conn:
        run_id = conn.execute(
            "INSERT INTO runs(run_type, run_date, started_at, status, model) VALUES ('predict',?,?,?,?)",
            (today.isoformat(), doc.get("exported_at", started), "running", model)).lastrowid
        stored = _store(conn, run_id, today, jobs, calls, model, started)
        if calls and not stored:
            conn.execute("DELETE FROM runs WHERE id=?", (run_id,))
            log.info("These answers were already imported; nothing new stored")
            return {"requested": n_pairs, "stored": 0, "status": "already imported"}
        missing = n_pairs - len(calls)
        status = "ok" if not problems and not missing else ("partial" if stored else "failed")
        conn.execute(
            """UPDATE runs SET finished_at=?, status=?, n_items=?, market_view=?, error=?
               WHERE id=?""",
            (datetime.now(timezone.utc).isoformat(), status, stored, market_view,
             "\n".join(problems + ([f"{late} calls dropped: market opened before import"]
                                   if late else [])) or None, run_id))
    log.info("Stored %d of %d predictions (status %s)", stored, n_pairs, status)
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
    hand_off = p.add_mutually_exclusive_group()
    hand_off.add_argument("--export-batches", type=Path, metavar="DIR",
                          help="write the prompt for a Claude Code session instead of calling the API")
    hand_off.add_argument("--check-answers", type=Path, metavar="DIR",
                          help="validate the session's answer files, store nothing")
    hand_off.add_argument("--import-answers", type=Path, metavar="DIR",
                          help="store the session's answer files")
    p.add_argument("--model", default="claude-sonnet-5-5 (Claude Code)",
                   help="model name recorded with imported answers")
    args = p.parse_args()
    if args.export_batches:
        export_batches(args.export_batches, fetch=not args.no_fetch, force=args.force)
    elif args.check_answers or args.import_answers:
        res = import_answers(args.check_answers or args.import_answers, args.model,
                             force=args.force, check=bool(args.check_answers))
        if args.check_answers and res["problems"]:
            raise SystemExit(1)
    else:
        run(fetch=not args.no_fetch, dry_run=args.dry_run, force=args.force)
