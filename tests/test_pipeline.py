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


def _fake_intraday_download(symbols):
    """A yf.download stand-in: 15-minute bars on two days, the way Yahoo returns them."""
    import pandas as pd
    idx = pd.DatetimeIndex(
        list(pd.date_range("2026-11-19 14:30", periods=4, freq="15min", tz="UTC"))
        + list(pd.date_range("2026-11-20 14:30", periods=4, freq="15min", tz="UTC")))
    frames = {s: pd.DataFrame({"Close": [100.0 + i for i in range(8)]}, index=idx) for s in symbols}
    return lambda batch, **kw: pd.concat({s: frames[s] for s in batch}, axis=1)


def test_update_intraday_groups_by_day_and_prunes(db, monkeypatch):
    from app import prices
    db.execute("INSERT INTO intraday(symbol,date,bars) VALUES ('AAPL','2020-01-02','[]')")  # too old
    db.commit()
    monkeypatch.setattr(prices.yf, "download", _fake_intraday_download(["AAPL", "KBC.BR"]))
    monkeypatch.setattr(prices, "INTRADAY_KEEP_DAYS", 100000)  # keep the 2026 test days whatever today is
    assert prices.update_intraday(["AAPL", "KBC.BR"]) == 4      # 2 symbols x 2 days
    rows = db.execute("SELECT symbol, date, bars FROM intraday WHERE date >= '2026-01-01' ORDER BY symbol, date").fetchall()
    assert [(r[0], r[1]) for r in rows] == [("AAPL", "2026-11-19"), ("AAPL", "2026-11-20"),
                                            ("KBC.BR", "2026-11-19"), ("KBC.BR", "2026-11-20")]
    bars = json.loads(rows[1]["bars"])
    assert bars[0] == [int(datetime(2026, 11, 20, 14, 30, tzinfo=timezone.utc).timestamp()), 104.0]
    assert len(bars) == 4
    # Running again replaces rows instead of duplicating them; the 2020 row was pruned.
    monkeypatch.setattr(prices, "INTRADAY_KEEP_DAYS", 14)
    prices.update_intraday(["AAPL"])
    assert db.execute("SELECT COUNT(*) FROM intraday WHERE date < '2021-01-01'").fetchone()[0] == 0


def test_chart_api_ranges(db):
    from fastapi.testclient import TestClient
    from app.web.main import app
    add_prices(db, "AAPL", "XNYS", date(2026, 10, 1), date(2026, 11, 20))
    add_prices(db, "KBC.BR", "XBRU", date(2026, 10, 1), date(2026, 11, 20))
    t0 = int(datetime(2026, 11, 19, 14, 30, tzinfo=timezone.utc).timestamp())
    for i, day in enumerate(["2026-11-17", "2026-11-18", "2026-11-19", "2026-11-20"]):
        bars = [[t0 + i * 86400 + k * 900, 200.0 + i + k] for k in range(3)]
        db.execute("INSERT INTO intraday(symbol,date,bars) VALUES ('AAPL',?,?)", (day, json.dumps(bars)))
    db.commit()
    client = TestClient(app)
    one = client.get("/api/ticker/AAPL/chart?range=1d").json()
    assert one["kind"] == "intraday" and len(one["points"]) == 3     # only the latest day
    prev = db.execute("SELECT close FROM prices WHERE symbol='AAPL' AND date<'2026-11-20' "
                      "ORDER BY date DESC LIMIT 1").fetchone()[0]
    assert one["base"] == prev and one["last"] == 205.0
    assert one["change_pct"] == pytest.approx((205.0 / prev - 1) * 100)
    week = client.get("/api/ticker/AAPL/chart?range=1w").json()
    assert len(week["points"]) == 12                                  # all 4 stored days
    assert [p["t"] for p in week["points"]] == sorted(p["t"] for p in week["points"])
    month = client.get("/api/ticker/AAPL/chart?range=1m").json()
    assert month["kind"] == "daily" and all("d" in p for p in month["points"])
    # No intraday bars stored: daily closes instead.
    kbc = client.get("/api/ticker/KBC.BR/chart?range=1d").json()
    assert kbc["kind"] == "daily"
    assert client.get("/api/ticker/AAPL/chart?range=5y").status_code == 400
    assert client.get("/api/ticker/NOPE/chart?range=1d").json()["points"] == []


def test_state_files_include_prices_and_intraday(db, tmp_path, monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "state", os.path.join(os.path.dirname(__file__), "..", "scripts", "state.py"))
    state = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(state)
    add_prices(db, "AAPL", "XNYS", date(2026, 10, 28), date(2026, 11, 20))
    db.execute("INSERT INTO intraday(symbol,date,bars) VALUES ('AAPL','2026-11-20','[[1,2.5]]')")
    db.commit()
    files = state.export_files()
    assert {"prices/2026-10.jsonl", "prices/2026-11.jsonl", "intraday/2026-11-20.jsonl"} <= set(files)
    assert "prices" in state.APPEND_ONLY and "intraday" not in state.APPEND_ONLY

    monkeypatch.setenv("STOCKS_DB", str(tmp_path / "copy.db"))
    from app import config
    config.get_config.cache_clear()
    counts = state.import_files(files)
    assert counts["prices"] == db.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
    assert counts["intraday"] == 1
    from app.db import connect
    copy = connect()
    assert copy.execute("SELECT bars FROM intraday").fetchone()[0] == "[[1,2.5]]"
    copy.close()


def test_tickers_api_for_search(db):
    from fastapi.testclient import TestClient
    from app.web.main import app
    add_prices(db, "AAPL", "XNYS", date(2026, 11, 2), date(2026, 11, 20))
    db.execute("INSERT INTO tickers(symbol,name,exchange,sector,active) VALUES ('OLD','Gone','XNYS','x',0)")
    db.commit()
    rows = TestClient(app).get("/api/tickers").json()
    assert [r["symbol"] for r in rows] == ["AAPL", "KBC.BR"]          # active only, sorted by name
    aapl, kbc = rows
    assert aapl["universes"] == ["S&P 500"] and kbc["universes"] == ["BEL20"]
    assert aapl["close"] == db.execute("SELECT close FROM prices WHERE symbol='AAPL' "
                                       "ORDER BY date DESC LIMIT 1").fetchone()[0]
    assert kbc["close"] is None


def test_static_site_build(db, tmp_path):
    import importlib.util
    from app import predict
    from app.claude_predictor import ClaudePredictor
    from app.web import main
    add_prices(db, "AAPL", "XNYS", date(2026, 6, 1), date(2026, 11, 20))
    add_prices(db, "KBC.BR", "XBRU", date(2026, 6, 1), date(2026, 11, 20))
    predict.run(today=date(2026, 11, 23), fetch=False, force=True,
                predictor=ClaudePredictor("claude-opus-5-5", client=FakeClient()))
    spec = importlib.util.spec_from_file_location(
        "build_site", os.path.join(os.path.dirname(__file__), "..", "scripts", "build_site.py"))
    build_site = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build_site)
    try:
        counts = build_site.build(tmp_path / "site", root="/Stock-trading")
        site = tmp_path / "site"
        for f in ["index.html", "predictions/index.html", "predictions/horizon-daily/made_on-2026-11-23/index.html",
                  "predictions/horizon-weekly/index.html", "accuracy/days-30/index.html",
                  "ticker/KBC.BR/index.html", "api/tickers.json", "api/ticker/KBC.BR/chart-1w.json",
                  "api/ticker/AAPL/history-180.json", "api/accuracy/series-daily-90.json",
                  "static/charts.js", ".nojekyll", "404.html"]:
            assert (site / f).exists(), f
        home = (site / "index.html").read_text(encoding="utf-8")
        assert 'href="/Stock-trading/predictions/"' in home
        assert 'href="/Stock-trading/predictions/#universe=bel20"' in home
        assert '/Stock-trading/static/style.css?v=' in home
        assert '"static": true' in home and 'href="/predictions' not in home
        ticker = (site / "ticker" / "AAPL" / "index.html").read_text(encoding="utf-8")
        assert "Today, live" not in ticker                      # live Yahoo chart left out
        preds = (site / "predictions" / "index.html").read_text(encoding="utf-8")
        assert "Load live prices" not in preds and 'data-unis="sp500"' in preds
        assert counts["pages"] >= 10
    finally:
        main.configure_site("", static=False)
        main._static_versions.clear()
    # The local server is back to plain links.
    from fastapi.testclient import TestClient
    assert 'href="/predictions"' in TestClient(main.app).get("/").text


def test_website_renders(db):
    from fastapi.testclient import TestClient
    from app.web.main import app
    client = TestClient(app)
    for url in ["/", "/predictions", "/predictions?horizon=weekly", "/accuracy",
                "/ticker/AAPL", "/api/accuracy/series", "/api/ticker/AAPL/history",
                "/api/ticker/AAPL/chart?range=1w"]:
        assert client.get(url).status_code == 200, url
    assert client.get("/ticker/NOPE").status_code == 404


def test_move_accuracy_formula():
    from app.stats import move_accuracy
    assert move_accuracy(7, 10) == pytest.approx(0.70)     # predicted +7%, went +10%
    assert move_accuracy(10, 7) == pytest.approx(0.70)     # symmetric
    assert move_accuracy(-2, -2) == pytest.approx(1.0)
    assert move_accuracy(2, -1) == 0.0                     # wrong direction
    assert move_accuracy(None, 3) is None


def test_move_accuracy_in_stats_and_pages(db):
    from fastapi.testclient import TestClient
    from app import stats
    from app.web.main import app
    calls = [("AAPL", 7.0, 10.0, 1), ("KBC.BR", -1.0, 2.0, 0)]
    for i, (sym, exp, act, ok) in enumerate(calls, start=1):
        db.execute("""INSERT INTO predictions(id, symbol, horizon, made_on, period_start, target_date,
                        base_date, base_close, direction, confidence, expected_move_pct, model, created_at)
                      VALUES (?,?,'daily','2026-11-23','2026-11-23','2026-11-23','2026-11-20',100,?,0.6,?,'m','x')""",
                   (i, sym, "UP" if exp > 0 else "DOWN", exp))
        db.execute("""INSERT INTO evaluations VALUES (?, ?, ?, ?, ?, 'x')""",
                   (i, 100 + act, act, "UP" if act > 0 else "DOWN", ok))
    db.commit()
    a = stats.accuracy(db, "daily")
    assert a["move_n"] == 2
    assert a["move_acc"] == pytest.approx((0.70 + 0.0) / 2)
    assert a["move_miss"] == pytest.approx((3 + 3) / 2)
    assert len(stats.moves(db, "daily", date(2026, 1, 1))) == 2

    client = TestClient(app)
    page = client.get("/predictions?made_on=2026-11-23").text
    assert "Move acc." in page and "35.0% move accuracy" in page
    assert "Size of the move" in client.get("/accuracy?days=3650").text
    assert client.get("/api/accuracy/moves?days=3650").json()["points"]
