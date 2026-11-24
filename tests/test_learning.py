"""The self-improvement loop: diagnostics, playbook, reflection import, research, the brief."""
from __future__ import annotations

import importlib.util
import json
import os
from datetime import date, datetime, timedelta, timezone

import pytest

def approx(x):
    return pytest.approx(x, abs=1e-3)       # the diagnostics are rounded to 4 decimals


DAYS = [date(2026, 11, d) for d in (16, 17, 18, 19, 20)]   # Monday to Friday
UP, DOWN = "UP", "DOWN"


@pytest.fixture()
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("STOCKS_DB", str(tmp_path / "test.db"))
    from app import config
    config.get_config.cache_clear()
    from app.db import connect
    conn = connect()
    stocks = [("AAPL", "XNYS", "Information Technology", "sp500"), ("XOM", "XNYS", "Energy", "sp500"),
              ("JPM", "XNYS", "Financials", "sp500"), ("KBC.BR", "XBRU", "Financials", "bel20"),
              ("SAP.DE", "XETR", "Information Technology", "europe"), ("NESN.SW", "XSWX", "Consumer Staples", "europe")]
    for sym, ex, sector, uni in stocks:
        conn.execute("INSERT INTO tickers(symbol,name,exchange,sector) VALUES (?,?,?,?)", (sym, sym, ex, sector))
        conn.execute("INSERT INTO ticker_universes VALUES (?,?)", (sym, uni))
    conn.commit()
    yield conn
    conn.close()
    config.get_config.cache_clear()


def add_scored(conn, day, symbol, direction, actual, conf=0.6, exp=None, horizon="daily", run_id=None):
    """A prediction made on `day` that was scored with the given actual % move."""
    exp = exp if exp is not None else (1.0 if direction == UP else -1.0)
    cur = conn.execute(
        """INSERT INTO predictions(run_id, symbol, horizon, made_on, period_start, target_date, base_date,
             base_close, direction, confidence, expected_move_pct, rationale, model, created_at)
           VALUES (?,?,?,?,?,?,?,100,?,?,?,'why','m','x')""",
        (run_id, symbol, horizon, day.isoformat(), day.isoformat(), day.isoformat(),
         (day - timedelta(days=1)).isoformat(), direction, conf, exp))
    conn.execute("INSERT INTO evaluations VALUES (?,?,?,?,?,'x')",
                 (cur.lastrowid, 100 + actual, actual, "UP" if actual > 0 else "DOWN",
                  int((actual > 0) == (direction == UP))))


def seed_week(conn, with_runs=False):
    """Five days, six stocks: Claude calls everything UP; the market falls on day 3."""
    for i, day in enumerate(DAYS):
        run_id = None
        if with_runs:
            run_id = conn.execute("INSERT INTO runs(run_type, run_date, started_at, status, playbook_version) "
                                  "VALUES ('predict',?,'x','ok',?)", (day.isoformat(), "pb0" if i < 3 else "pb1-abc")).lastrowid
        market = -1.0 if i == 2 else 0.5
        for sym, tilt in [("AAPL", 2), ("XOM", 1), ("JPM", 0), ("KBC.BR", -1), ("SAP.DE", 0.5), ("NESN.SW", -2)]:
            add_scored(conn, day, sym, UP, market + tilt, run_id=run_id)
    conn.commit()


def test_diagnose_numbers_and_baselines(db):
    from app import reflect
    seed_week(db)
    d = reflect.diagnose(db, DAYS[0])
    daily = d["daily"]
    assert daily["n"] == 6 and d["n_calls"] == 6
    assert daily["hit"] == approx(4 / 6)                       # moves 2.5 1.5 .5 -.5 1 -1.5
    assert daily["up_share"] == 1.0 and daily["actual_up"] == approx(4 / 6)
    assert daily["always_up"] == approx(4 / 6)                 # everything called UP == always-UP
    assert daily["spread"] is None                                    # no DOWN calls to compare with
    assert daily["market_avg"] == approx(3.5 / 6)
    # Called UP: beats the market average (0.583) for moves above it: 2.5, 1.5, 1.0 -> 3 of 6
    assert daily["rel_hit"] == approx(3 / 6)
    assert d["by_universe"]["sp500"]["n"] == 3 and d["by_universe"]["bel20"]["n"] == 1
    assert d["misses"][0]["symbol"] == "NESN.SW"                      # wrong direction, biggest move
    assert json.dumps(d)                                              # serialisable for the stats column


def test_daily_metrics_and_market_call(db):
    from app import reflect
    seed_week(db)
    rows = reflect.daily_metrics(db, "daily", 30, DAYS[-1])
    assert [r["date"] for r in rows] == [d.isoformat() for d in DAYS]
    assert rows[2]["market_call_hit"] is False and rows[0]["market_call_hit"] is True
    assert rows[2]["always_up"] < 0.5 and rows[0]["always_up"] > 0.5


def test_playbook_rules_and_safeguards(db, tmp_path):
    from app import reflect
    seed_week(db)

    def submit(changes, day, summary="A summary that is long enough to count as a real reflection."):
        f = tmp_path / "r.json"
        f.write_text(json.dumps({"summary": summary, "rule_changes": changes}))
        return reflect.import_reflection(f, day)

    good = {"action": "add", "id": "size-moves-to-atr", "text": "Size expected_move_pct from atr14, not from habit.",
            "evidence": "Expected moves were 4x too small on 3 of the last 4 days."}
    res = submit([good, {**good, "id": "bad id!"}, {**good, "id": "short-evidence", "evidence": "meh"},
                  {"action": "promote", "id": "size-moves-to-atr", "why": "It held up well in the data we saw."},
                  {"action": "retire", "id": "missing-rule", "why": "Does not exist, so cannot be retired."}], DAYS[0])
    assert res["applied"] == ["added trial rule size-moves-to-atr"]
    assert len(res["rejected"]) == 4 and any("too early" in r or "needs" in r for r in res["rejected"])
    v1 = res["playbook_version"]
    assert v1.startswith("pb1-")

    # Importing the same evening again must not duplicate or fail
    again = submit([good], DAYS[0])
    assert again["applied"] == ["added trial rule size-moves-to-atr"] and db.execute(
        "SELECT COUNT(*) FROM playbook_rules").fetchone()[0] == 1

    # Promotion needs three evenings of evidence
    submit([], DAYS[1])
    submit([], DAYS[2])
    early = submit([{"action": "promote", "id": "size-moves-to-atr", "why": "Held up on the last three evenings."}], DAYS[2])
    assert early["rejected"] and "too early" in early["rejected"][0]
    ok = submit([{"action": "promote", "id": "size-moves-to-atr", "why": "Held up on the last three evenings."}], DAYS[3])
    assert ok["applied"] == ["promoted rule size-moves-to-atr to active"]

    # Editing sends an active rule back to trial; retiring keeps the row
    edit = submit([{"action": "edit", "id": "size-moves-to-atr", "text": "Use 0.8 times atr14 as the typical move.",
                    "why": "The ratio of actual to predicted size was 0.8 over 20 days."}], DAYS[4])
    assert db.execute("SELECT status FROM playbook_rules").fetchone()[0] == "trial"
    assert "back to trial" in edit["applied"][0]
    submit([{"action": "retire", "id": "size-moves-to-atr", "why": "No longer supported by the last 20 days."}], DAYS[4])
    row = db.execute("SELECT status, history FROM playbook_rules").fetchone()
    assert row["status"] == "retired" and row["history"].count("\n") >= 3
    assert reflect.playbook_version(db) == "pb0"                      # retired rules leave the playbook

    # Short summaries and malformed files are refused
    with pytest.raises(SystemExit):
        submit([], DAYS[4], summary="too short")


def test_rule_limits(db, tmp_path):
    from app import reflect
    adds = [{"action": "add", "id": f"rule-number-{i}", "text": f"Rule number {i} says do something useful.",
             "evidence": "Seen on many calls across several days."} for i in range(5)]
    f = tmp_path / "r.json"
    f.write_text(json.dumps({"summary": "x" * 60, "rule_changes": adds}))
    res = reflect.import_reflection(f, DAYS[0])
    assert len(res["applied"]) == reflect.MAX_ADDS_PER_DAY and len(res["rejected"]) == 2
    for day, k in zip(DAYS[1:], range(3)):    # fill up to MAX_RULES over the next evenings
        f.write_text(json.dumps({"summary": "x" * 60, "rule_changes": [
            {**a, "id": f"later-{k}-{j}"} for j, a in enumerate(adds[:3])]}))
        reflect.import_reflection(f, day)
    assert len(reflect.live_rules(db)) <= reflect.MAX_RULES


def test_learning_text_and_brief(db, tmp_path):
    from app import reflect
    seed_week(db, with_runs=True)
    f = tmp_path / "r.json"
    f.write_text(json.dumps({"summary": "The market fell on Wednesday and every call was UP; lean was the problem.",
                             "research_focus": ["Check overnight futures before leaning UP."],
                             "rule_changes": [{"action": "add", "id": "dont-lean-one-way", "text":
                                               "Do not call every stock UP: call DOWN where the news or sector is negative.",
                                               "evidence": "All 30 calls this week were UP and 40% of stocks fell."}]}))
    reflect.import_reflection(f, DAYS[-1])
    text = reflect.learning_text(db, DAYS[-1] + timedelta(days=3))
    assert "[trial] dont-lean-one-way" in text and "lean was the problem" in text
    assert "You called UP 100%" in text.replace("you called", "You called") or "called UP 100%" in text
    assert "Check overnight futures" in text
    assert "CAUTION" not in text                                      # five days of history is enough
    assert "CAUTION: only 4 days" in reflect.learning_text(db, DAYS[3])

    out = tmp_path / "work"
    res = reflect.export_brief(out, DAYS[-1])
    brief = (out / "reflection_brief.md").read_text()
    assert res["calls"] == 6 and "Costliest misses" in brief and "Playbook versions" in brief
    assert "Ground rules" in brief and '"rule_changes"' in brief and "dont-lean-one-way" in brief
    # A day with no scored calls produces a "do not write a reflection" note
    assert reflect.export_brief(out, date(2026, 12, 25))["calls"] == 0
    assert "Do not write a reflection" in (out / "reflection_brief.md").read_text()

    versions = reflect.by_version(db)
    assert [v["version"] for v in versions] == ["pb0", "pb1-abc"] and versions[0]["days"] == 3


def test_run_sql_is_read_only(db):
    from app import reflect
    seed_week(db)
    db.commit()
    assert "n" in reflect.run_sql("SELECT COUNT(*) AS n FROM predictions").splitlines()[0]
    for bad in ("DELETE FROM predictions", "SELECT 1; DROP TABLE predictions", "PRAGMA writable_schema=1"):
        with pytest.raises(SystemExit):
            reflect.run_sql(bad)


def test_research_import_and_news_column(db, tmp_path):
    from app import predict, research
    out = tmp_path / "res"
    day = date(2026, 11, 23)
    f = out / "research.json"
    out.mkdir()
    f.write_text(json.dumps({
        "market": {"summary": "Futures are lower after chip export news.", "events": ["14:30 UTC CPI"],
                   "sectors": {"Information Technology": "lean down: export rules"}},
        "stocks": [{"symbol": "AAPL", "sentiment": -1, "note": "Supplier warned | on demand.", "source": "Reuters"},
                   {"symbol": "NOPE", "sentiment": 2, "note": "Unknown ticker."},
                   {"symbol": "XOM", "sentiment": 2, "note": "Takeover bid reported."}]}))
    res = research.import_research(f, day)
    assert res == {"stored": 2, "dropped": ["NOPE"]}
    assert research.import_research(f, day)["stored"] == 2            # re-import replaces, no duplicates
    market, stocks = research.news_for(db, day)
    assert stocks["AAPL"] == (-1, "Supplier warned | on demand.") and "lean down" in research.market_text(market)

    rows = predict._add_news_column("symbol|x\nAAPL|1\nJPM|2", stocks)
    assert rows.split("\n") == ["symbol|x|news", "AAPL|1|-1: Supplier warned / on demand.", "JPM|2|"]

    from pydantic import ValidationError
    bad = out / "bad.json"
    bad.write_text(json.dumps({"market": {"summary": "x"}, "stocks": [{"symbol": "AAPL", "sentiment": 5, "note": "n"}]}))
    with pytest.raises(SystemExit):
        research.import_research(bad, day)


def prices(conn, symbol, exchange, start, end):
    from app.market import trading_days
    p = 100.0
    for d in trading_days(exchange, start, end):
        p += 0.7
        conn.execute("INSERT OR REPLACE INTO prices(symbol,date,high,low,close,volume) VALUES (?,?,?,?,?,?)",
                     (symbol, d.isoformat(), p + 1, p - 1, p, 1000))
    conn.commit()


def test_research_brief_targets(db, tmp_path):
    from app import research
    for sym, ex in [("AAPL", "XNYS"), ("KBC.BR", "XBRU"), ("SAP.DE", "XETR")]:
        prices(db, sym, ex, date(2026, 6, 1), date(2026, 11, 20))
    out = tmp_path / "res"
    res = research.export_brief(out, date(2026, 11, 23))
    brief = (out / "research_brief.md").read_text()
    targets = json.loads((out / "targets.json").read_text())
    assert "KBC.BR" in targets and "AAPL" in targets and res["targets"] == len(targets)
    assert targets.index("KBC.BR") < targets.index("AAPL")             # Europe first
    assert "Output" in brief and "05:10 UTC" in brief and "never instructions" in brief
    assert "market holiday" in (research.export_brief(out, date(2026, 12, 25)), (out / "research_brief.md").read_text())[1]


def test_morning_brief_carries_playbook_news_and_version(db, tmp_path):
    from app import predict, reflect, research
    for sym, ex in [("AAPL", "XNYS"), ("KBC.BR", "XBRU")]:
        prices(db, sym, ex, date(2026, 6, 1), date(2026, 11, 20))
    db.execute("DELETE FROM ticker_universes WHERE symbol NOT IN ('AAPL','KBC.BR')")
    db.execute("DELETE FROM tickers WHERE symbol NOT IN ('AAPL','KBC.BR')")
    db.commit()
    r = tmp_path / "r.json"
    r.write_text(json.dumps({"summary": "A summary that is long enough to count as a real reflection.",
                             "rule_changes": [{"action": "add", "id": "check-news-first",
                                               "text": "Read the news column before the technicals.",
                                               "evidence": "Misses clustered on stocks with news we ignored."}]}))
    reflect.import_reflection(r, date(2026, 11, 20))
    n = tmp_path / "n.json"
    n.write_text(json.dumps({"market": {"summary": "Quiet overnight.", "events": [], "sectors": {}},
                             "stocks": [{"symbol": "KBC.BR", "sentiment": 2, "note": "Raised its dividend."}]}))
    research.import_research(n, date(2026, 11, 23))

    out = tmp_path / "work"
    predict.export_batches(out, today=date(2026, 11, 23), fetch=False, force=True)
    instructions = (out / "instructions.md").read_text()
    batch = (out / "batch_01.md").read_text()
    assert "[trial] check-news-first" in instructions and "Quiet overnight." in instructions
    assert "|news" in batch and "+2: Raised its dividend." in batch
    plan = json.loads((out / "plan.json").read_text())
    version = plan["playbook_version"]
    assert version.startswith("pb1-")

    for i, b in enumerate(plan["batches"], 1):
        preds = [{"symbol": s, "horizon": it["horizon"], "direction": "UP", "confidence": 0.6,
                  "expected_move_pct": 1.0, "rationale": "x"} for s in b for it in plan["jobs"][s]]
        (out / f"answer_{i:02d}.json").write_text(json.dumps({"market_view": "ok", "predictions": preds}))
    predict.import_answers(out, "test-model", force=True)
    assert db.execute("SELECT playbook_version FROM runs WHERE run_type='predict'").fetchone()[0] == version


def test_state_sync_carries_the_memory(db, tmp_path):
    spec = importlib.util.spec_from_file_location(
        "state", os.path.join(os.path.dirname(__file__), "..", "scripts", "state.py"))
    state = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(state)
    from app import reflect
    db.execute("INSERT INTO playbook_rules VALUES ('r-one','Some rule text here.','trial','2026-11-20','2026-11-20','ev','h')")
    db.execute("INSERT INTO reflections(date, summary, created_at) VALUES ('2026-11-20','s','x')")
    db.execute("INSERT INTO research VALUES ('2026-11-23','AAPL',1,'n','s')")
    db.execute("INSERT INTO research_market VALUES ('2026-11-23','m','e','{}','x')")
    db.execute("INSERT INTO runs(run_type, run_date, started_at, status, playbook_version) VALUES ('predict','2026-11-23','x','ok','pb1-abc')")
    db.commit()
    files = state.export_files()
    assert {"playbook_rules.jsonl", "reflections.jsonl", "research_market.jsonl", "research/2026-11.jsonl"} <= set(files)
    for t in ("playbook_rules", "reflections", "research_market", "research"):
        assert t in state.APPEND_ONLY
    for t in ("playbook_rules", "reflections", "research_market", "research"):
        db.execute(f"DELETE FROM {t}")
    db.execute("DELETE FROM runs")
    db.commit()
    state.import_files(files)
    assert db.execute("SELECT COUNT(*) FROM playbook_rules").fetchone()[0] == 1
    assert db.execute("SELECT playbook_version FROM runs").fetchone()[0] == "pb1-abc"
    assert reflect.playbook_version(db).startswith("pb1-")


def test_old_database_gets_the_playbook_column(tmp_path, monkeypatch):
    import sqlite3
    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.execute("CREATE TABLE runs (id INTEGER PRIMARY KEY AUTOINCREMENT, run_type TEXT NOT NULL, run_date TEXT NOT NULL, "
                "started_at TEXT NOT NULL, finished_at TEXT, status TEXT NOT NULL, model TEXT, input_tokens INTEGER DEFAULT 0, "
                "output_tokens INTEGER DEFAULT 0, n_items INTEGER DEFAULT 0, market_view TEXT, error TEXT)")
    old.commit()
    old.close()
    from app.db import connect
    conn = connect(path)
    assert "playbook_version" in {r[1] for r in conn.execute("PRAGMA table_info(runs)")}
    conn.close()


def test_learning_page_and_api(db, tmp_path):
    from fastapi.testclient import TestClient
    from app import reflect, research
    from app.web.main import app
    client = TestClient(app)
    empty = client.get("/learning")
    assert empty.status_code == 200 and "The playbook is empty" in empty.text and "No reflections yet" in empty.text

    seed_week(db, with_runs=True)
    f = tmp_path / "r.json"
    f.write_text(json.dumps({"summary": "The market fell on Wednesday and every call was UP; lean was the problem.",
                             "went_wrong": ["Called everything UP into a falling market."],
                             "rule_changes": [{"action": "add", "id": "dont-lean-one-way", "text":
                                               "Do not call every stock UP: call DOWN where the news is negative.",
                                               "evidence": "All 30 calls this week were UP and 40% of stocks fell."}]}))
    reflect.import_reflection(f, DAYS[-1])
    n = tmp_path / "n.json"
    n.write_text(json.dumps({"market": {"summary": "Quiet overnight.", "events": ["CPI"], "sectors": {"Energy": "up"}},
                             "stocks": [{"symbol": "XOM", "sentiment": 1, "note": "Oil up."}]}))
    research.import_research(n, DAYS[-1])
    page = client.get("/learning").text
    assert "dont-lean-one-way" in page and "Called everything UP" in page and "pb1-" in page
    assert "Quiet overnight." in page and "playbook" in page.lower() and "pb1-abc" in page
    rows = client.get("/api/learning/daily").json()
    assert len(rows) == 5 and rows[0]["version"] == "pb0" and rows[-1]["version"] == "pb1-abc"
    assert "Learning" in client.get("/").text
