"""End-to-end tests with synthetic prices and a fake Claude client (no network)."""
from __future__ import annotations

import json
import os
from datetime import date, datetime, timezone
from types import SimpleNamespace

import pytest


@pytest.fixture()
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("STOCKS_DB", str(tmp_path / "test.db"))
    from app import config
    config.get_config.cache_clear()
    from app.db import connect
    conn = connect()
    for sym, ex in [("AAPL", "XNYS"), ("KBC.BR", "XBRU")]:
        conn.execute("INSERT INTO tickers(symbol,name,exchange,sector) VALUES (?,?,?,?)", (sym, sym, ex, "x"))
        conn.execute("INSERT INTO ticker_universes VALUES (?, ?)", (sym, "sp500" if ex == "XNYS" else "bel20"))
    conn.commit()
    yield conn
    conn.close()
    config.get_config.cache_clear()


def add_prices(conn, symbol, exchange, start, end, base=100.0, step=0.5):
    from app.market import trading_days
    price = base
    for d in trading_days(exchange, start, end):
        price += step
        conn.execute("INSERT OR REPLACE INTO prices(symbol,date,high,low,close,volume) VALUES (?,?,?,?,?,?)",
                     (symbol, d.isoformat(), price + 1, price - 1, price, 1000))
    conn.commit()


def test_plan_weekly_only_on_first_trading_day(db):
    from app.predict import plan
    add_prices(db, "AAPL", "XNYS", date(2026, 6, 1), date(2026, 11, 20))
    add_prices(db, "KBC.BR", "XBRU", date(2026, 6, 1), date(2026, 11, 20))
    monday, before_open = date(2026, 11, 23), datetime(2026, 11, 23, 6, 30, tzinfo=timezone.utc)
    jobs, *_ = plan(db, monday, before_open)
    assert {h["horizon"] for h in jobs["AAPL"]} == {"daily", "weekly"}
    weekly = next(h for h in jobs["AAPL"] if h["horizon"] == "weekly")
    # Thanksgiving week: NYSE closes Thursday, Friday is a (half) trading day
    assert weekly["target_date"] == date(2026, 11, 27)
    assert weekly["base_date"] == "2026-11-20"

    add_prices(db, "AAPL", "XNYS", date(2026, 11, 23), date(2026, 11, 23))
    add_prices(db, "KBC.BR", "XBRU", date(2026, 11, 23), date(2026, 11, 23))
    jobs, *_ = plan(db, date(2026, 11, 24), datetime(2026, 11, 24, 6, 30, tzinfo=timezone.utc))
    assert [h["horizon"] for h in jobs["AAPL"]] == ["daily"]


def test_plan_skips_open_markets_and_stale_data(db):
    from app.predict import plan
    add_prices(db, "AAPL", "XNYS", date(2026, 6, 1), date(2026, 11, 20))
    add_prices(db, "KBC.BR", "XBRU", date(2026, 6, 1), date(2026, 11, 18))  # stale
    # 10:00 UTC: Brussels open, New York not
    jobs, _, _, skipped = plan(db, date(2026, 11, 23), datetime(2026, 11, 23, 10, 0, tzinfo=timezone.utc))
    assert "AAPL" in jobs and "KBC.BR" not in jobs
    assert "KBC.BR" in skipped["market already open"]
    jobs, _, _, skipped = plan(db, date(2026, 11, 23), datetime(2026, 11, 23, 6, 0, tzinfo=timezone.utc))
    assert "KBC.BR" in skipped["price data not up to date"]


class FakeStream:
    def __init__(self, message):
        self.message = message

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get_final_message(self):
        return self.message


class FakeClient:
    """Mimics client.beta.messages.stream(...) and records the request."""

    def __init__(self):
        self.requests = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=self._stream))

    def _stream(self, **kwargs):
        self.requests.append(kwargs)
        wanted = [line.split() for line in kwargs["messages"][0]["content"].split("one per line as 'SYMBOL horizon':\n")[1].splitlines()]
        preds = [{"symbol": s, "horizon": h, "direction": "UP", "confidence": 0.99,
                  "expected_move_pct": -0.5, "rationale": "test"} for s, h in wanted]
        preds.append({"symbol": "BOGUS", "horizon": "daily", "direction": "UP", "confidence": 0.6,
                      "expected_move_pct": 1, "rationale": "not requested"})
        text = json.dumps({"market_view": "flat", "predictions": preds})
        msg = SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text=text)],
                              usage=SimpleNamespace(input_tokens=100, output_tokens=50,
                                                    cache_read_input_tokens=0, cache_creation_input_tokens=0))
        return FakeStream(msg)


def test_predict_then_evaluate(db, monkeypatch):
    from app import evaluate, predict
    from app.claude_predictor import ClaudePredictor
    add_prices(db, "AAPL", "XNYS", date(2026, 6, 1), date(2026, 11, 20))
    add_prices(db, "KBC.BR", "XBRU", date(2026, 6, 1), date(2026, 11, 20))

    client = FakeClient()
    predictor = ClaudePredictor("claude-opus-5-5", client=client)
    out = predict.run(today=date(2026, 11, 23), fetch=False, force=True, predictor=predictor)
    assert out["stored"] == 4 and out["status"] == "ok"
    req = client.requests[0]
    assert req["model"] == "claude-opus-5-5"
    assert req["output_config"]["format"]["type"] == "json_schema"
    assert req["fallbacks"] == "default"

    rows = db.execute("SELECT * FROM predictions ORDER BY symbol, horizon").fetchall()
    assert all(r["confidence"] == 0.95 for r in rows)           # clamped
    assert all(r["expected_move_pct"] == 0.5 for r in rows)     # sign follows direction
    assert db.execute("SELECT COUNT(*) FROM predictions WHERE symbol='BOGUS'").fetchone()[0] == 0

    # Re-running the same morning does not duplicate predictions.
    again = predict.run(today=date(2026, 11, 23), fetch=False, force=True, predictor=predictor)
    assert again["requested"] == 0

    # AAPL rises, KBC falls on the target day.
    db.execute("INSERT INTO prices(symbol,date,close) VALUES ('AAPL','2026-11-23', 1000)")
    db.execute("INSERT INTO prices(symbol,date,close) VALUES ('KBC.BR','2026-11-23', 1)")
    db.commit()
    res = evaluate.run(today=date(2026, 11, 23), fetch=False)
    assert res == {"scored": 2, "waiting": 0}   # weekly calls wait for Friday
    got = {r["symbol"]: r["correct"] for r in db.execute(
        "SELECT p.symbol, e.correct FROM evaluations e JOIN predictions p ON p.id = e.prediction_id")}
    assert got == {"AAPL": 1, "KBC.BR": 0}


def test_hand_off_export_check_import(db, tmp_path, monkeypatch):
    """A Claude Code session answers exported batch files instead of the API being called."""
    from app import predict
    add_prices(db, "AAPL", "XNYS", date(2026, 6, 1), date(2026, 11, 20))
    add_prices(db, "KBC.BR", "XBRU", date(2026, 6, 1), date(2026, 11, 20))
    work = tmp_path / "work"
    monkeypatch.setattr(predict, "has_opened", lambda ex, d, now: False)

    out = predict.export_batches(work, today=date(2026, 11, 23), fetch=False, force=True)
    assert out == {"requested": 4, "batches": 1}
    batch = (work / "batch_01.md").read_text(encoding="utf-8")
    assert "answer_01.json" in batch and "symbol|name|sector" in batch
    assert "Return exactly one prediction" in (work / "instructions.md").read_text(encoding="utf-8")

    # Nothing answered yet.
    res = predict.import_answers(work, "sonnet-test", check=True)
    assert res["valid"] == 0 and "missing" in res["problems"][0]

    # The session answers only three of the four pairs (plus junk).
    wanted = [line.split() for line in batch.split("'SYMBOL horizon':\n")[1].strip().splitlines()]
    # European stocks come first: their market opens before New York.
    assert wanted == [["KBC.BR", "daily"], ["KBC.BR", "weekly"], ["AAPL", "daily"], ["AAPL", "weekly"]]
    preds = [{"symbol": s, "horizon": h, "direction": "DOWN", "confidence": 0.3,
              "expected_move_pct": 0.7, "rationale": "test"} for s, h in wanted[:3]]
    preds.append({"symbol": "BOGUS", "horizon": "daily", "direction": "UP", "confidence": 0.6,
                  "expected_move_pct": 1, "rationale": "x"})
    (work / "answer_01.json").write_text(json.dumps({"market_view": "soft", "predictions": preds}))
    res = predict.import_answers(work, "sonnet-test", check=True)
    assert res["valid"] == 3 and "1 of 4 requested pairs missing: AAPL weekly" in res["problems"][0]
    assert db.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] == 0  # check stores nothing

    # A market that opened since the export gets no calls.
    monkeypatch.setattr(predict, "has_opened", lambda ex, d, now: ex == "XBRU")
    res = predict.import_answers(work, "sonnet-test")
    assert res["stored"] == 1 and res["status"] == "partial"   # only AAPL daily
    assert {r[0] for r in db.execute("SELECT symbol FROM predictions")} == {"AAPL"}

    # Importing again adds only what is new and never duplicates.
    monkeypatch.setattr(predict, "has_opened", lambda ex, d, now: False)
    res = predict.import_answers(work, "sonnet-test")
    rows = db.execute("SELECT * FROM predictions").fetchall()
    assert len(rows) == 3 and res["stored"] == 2
    assert all(r["confidence"] == 0.5 and r["expected_move_pct"] == -0.7 for r in rows)
    assert {r["model"] for r in rows} == {"sonnet-test"}
    view = db.execute("SELECT market_view FROM runs ORDER BY id DESC").fetchone()[0]
    assert view == "soft"

    # A third import has nothing new and leaves no empty run behind.
    res = predict.import_answers(work, "sonnet-test")
    assert res["status"] == "already imported"
    assert db.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 2


def test_website_renders(db):
    from fastapi.testclient import TestClient
    from app.web.main import app
    client = TestClient(app)
    for url in ["/", "/predictions", "/predictions?horizon=weekly", "/accuracy",
                "/ticker/AAPL", "/api/accuracy/series", "/api/ticker/AAPL/history"]:
        assert client.get(url).status_code == 200, url
    assert client.get("/ticker/NOPE").status_code == 404
