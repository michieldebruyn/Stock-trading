"""Local website: dashboard, predictions, per-ticker charts, accuracy."""
from __future__ import annotations

import logging
import time
from datetime import date, datetime, timedelta
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .. import stats
from ..config import local_tz
from ..db import connect

HERE = Path(__file__).parent
app = FastAPI(title="Stock predictions")
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
templates = Jinja2Templates(directory=HERE / "templates")
log = logging.getLogger(__name__)

UNIVERSES = {"sp500": "S&P 500", "europe": "Europe large caps", "bel20": "BEL20"}
ACCURACY_DAYS = (30, 90, 180, 365, 3650)


def _pct(v):
    return "–" if v is None else f"{v * 100:.1f}%"


# How links are written. The local server uses plain paths with query strings. The static
# GitHub Pages build (scripts/build_site.py) calls configure_site(): every link then gets the
# site's base path, query parameters become folders (/accuracy?days=30 -> /accuracy/days-30/),
# and the predictions page's market/result filters move to the URL hash, where the page filters
# rows in the browser.
SITE = {"root": "", "static": False}
HASH_PARAMS = ("universe", "result")


def configure_site(root: str = "", static: bool = False) -> None:
    SITE.update(root=root.rstrip("/"), static=static)
    templates.env.globals.update(root=SITE["root"], static_site=static)


def static_page_path(path: str, params: dict | None = None) -> str:
    """Folder (without the base path) that holds a page in the static build, plus any #hash."""
    params = {k: v for k, v in (params or {}).items() if v not in (None, "")}
    segs = [f"{k}-{v}" for k, v in params.items() if k not in HASH_PARAMS]
    folder = "/".join([p for p in path.strip("/").split("/") if p] + segs)
    out = "/" + folder + ("/" if folder else "")
    frag = "&".join(f"{k}={params[k]}" for k in HASH_PARAMS if k in params)
    return out + (f"#{frag}" if frag else "")


def url(path: str, **params) -> str:
    """Link to a page of this site, for templates."""
    params = {k: v for k, v in params.items() if v not in (None, "")}
    if not SITE["static"]:
        from urllib.parse import urlencode
        return path + (f"?{urlencode(params)}" if params else "")
    return SITE["root"] + static_page_path(path, params)


_static_versions: dict[str, str] = {}


def static_url(name: str) -> str:
    """Static file URL with a hash of the file's contents, so browsers fetch it again after an
    update instead of running a cached old copy."""
    version = _static_versions.get(name)
    if version is None:
        import hashlib
        path = HERE / "static" / name
        version = hashlib.md5(path.read_bytes()).hexdigest()[:10] if path.exists() else "0"
        if SITE["static"]:              # a static build renders 600 pages: hash each file once
            _static_versions[name] = version
    return f"{SITE['root']}/static/{name}?v={version}"


templates.env.globals.update(static_url=static_url, url=url, root="", static_site=False)
templates.env.filters["pct"] = _pct
templates.env.filters["signed"] = lambda v: "–" if v is None else f"{v:+.2f}%"


def today() -> date:
    return datetime.now(local_tz()).date()


def _latest_made_on(conn, horizon: str) -> str | None:
    return conn.execute("SELECT MAX(made_on) FROM predictions WHERE horizon=?", (horizon,)).fetchone()[0]


def _prediction_rows(conn, horizon: str, made_on: str | None, universe: str | None = None,
                     symbol: str | None = None, limit: int | None = None):
    sql = """SELECT p.*, t.name, t.sector, e.actual_close, e.actual_change_pct, e.correct,
                    (SELECT GROUP_CONCAT(universe) FROM ticker_universes u WHERE u.symbol = p.symbol) AS unis,
                    (SELECT close FROM prices pr WHERE pr.symbol = p.symbol
                       ORDER BY date DESC LIMIT 1) AS latest_close,
                    (SELECT date FROM prices pr WHERE pr.symbol = p.symbol
                       ORDER BY date DESC LIMIT 1) AS latest_date
             FROM predictions p JOIN tickers t ON t.symbol = p.symbol
             LEFT JOIN evaluations e ON e.prediction_id = p.id
             WHERE p.horizon = ?"""
    params: list = [horizon]
    if made_on:
        sql += " AND p.made_on = ?"
        params.append(made_on)
    if universe:
        sql += " AND p.symbol IN (SELECT symbol FROM ticker_universes WHERE universe = ?)"
        params.append(universe)
    if symbol:
        sql += " AND p.symbol = ?"
        params.append(symbol)
    sql += " ORDER BY p.made_on DESC, p.confidence DESC"
    if limit:
        sql += f" LIMIT {int(limit)}"
    rows = []
    for r in conn.execute(sql, params):
        d = dict(r)
        d["tracking_pct"] = None
        d["move_acc"] = stats.move_accuracy(d["expected_move_pct"], d["actual_change_pct"])
        d["move_error"] = (None if d["move_acc"] is None
                           else d["actual_change_pct"] - d["expected_move_pct"])
        if d["correct"] is None and d["latest_close"] and d["latest_date"] and d["latest_date"] > d["base_date"]:
            # Not scored yet, but newer prices exist: show how the call is tracking.
            d["tracking_pct"] = (d["latest_close"] / d["base_close"] - 1) * 100
        rows.append(d)
    return rows


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    t = today()
    with connect() as conn:
        cards = []
        for uni, label in UNIVERSES.items():
            cards.append({
                "key": uni, "label": label,
                "daily30": stats.accuracy(conn, "daily", t - timedelta(days=30), uni),
                "weekly90": stats.accuracy(conn, "weekly", t - timedelta(days=90), uni),
                "n_tickers": conn.execute(
                    "SELECT COUNT(*) FROM ticker_universes WHERE universe=?", (uni,)).fetchone()[0],
            })
        overall = {
            "daily30": stats.accuracy(conn, "daily", t - timedelta(days=30)),
            "daily_all": stats.accuracy(conn, "daily"),
            "weekly90": stats.accuracy(conn, "weekly", t - timedelta(days=90)),
            "weekly_all": stats.accuracy(conn, "weekly"),
        }
        latest_daily = _latest_made_on(conn, "daily")
        latest_weekly = _latest_made_on(conn, "weekly")
        top = _prediction_rows(conn, "daily", latest_daily, limit=12) if latest_daily else []
        breadth = [dict(r) for r in conn.execute(
            """SELECT u.universe, SUM(p.direction='UP') up, SUM(p.direction='DOWN') down,
                      SUM(e.correct) hits, COUNT(e.prediction_id) scored
               FROM predictions p JOIN ticker_universes u ON u.symbol = p.symbol
               LEFT JOIN evaluations e ON e.prediction_id = p.id
               WHERE p.horizon='daily' AND p.made_on = ? GROUP BY u.universe""",
            (latest_daily,))] if latest_daily else []
        run = conn.execute(
            "SELECT * FROM runs WHERE run_type='predict' ORDER BY id DESC LIMIT 1").fetchone()
        last_eval = conn.execute(
            "SELECT * FROM runs WHERE run_type='evaluate' ORDER BY id DESC LIMIT 1").fetchone()
        n_tickers = conn.execute("SELECT COUNT(*) FROM tickers WHERE active=1").fetchone()[0]
    return templates.TemplateResponse(request, "index.html", {
        "cards": cards, "overall": overall, "top": top, "latest_daily": latest_daily,
        "latest_weekly": latest_weekly, "breadth": {b["universe"]: b for b in breadth},
        "run": run, "last_eval": last_eval, "universes": UNIVERSES, "n_tickers": n_tickers,
        "page": "dashboard",
    })


def _move_summary(rows) -> dict:
    """Average move accuracy and miss over the scored rows that have an expected move."""
    m = [r for r in rows if r["move_acc"] is not None]
    if not m:
        return {"move_n": 0, "move_acc": None, "move_miss": None}
    return {"move_n": len(m), "move_acc": sum(r["move_acc"] for r in m) / len(m),
            "move_miss": sum(abs(r["move_error"]) for r in m) / len(m)}


@app.get("/predictions", response_class=HTMLResponse)
def predictions_page(request: Request, horizon: str = "daily", universe: str = "",
                     made_on: str = "", result: str = ""):
    if horizon not in ("daily", "weekly"):
        raise HTTPException(400, "horizon must be daily or weekly")
    with connect() as conn:
        dates = [r[0] for r in conn.execute(
            "SELECT DISTINCT made_on FROM predictions WHERE horizon=? ORDER BY made_on DESC LIMIT 60",
            (horizon,))]
        made_on = made_on or (dates[0] if dates else "")
        rows = _prediction_rows(conn, horizon, made_on or None, universe or None)
    if result == "correct":
        rows = [r for r in rows if r["correct"] == 1]
    elif result == "wrong":
        rows = [r for r in rows if r["correct"] == 0]
    elif result == "pending":
        rows = [r for r in rows if r["correct"] is None]
    scored = [r for r in rows if r["correct"] is not None]
    summary = {
        "n": len(rows), "up": sum(r["direction"] == "UP" for r in rows),
        "scored": len(scored), "hits": sum(r["correct"] for r in scored),
        **_move_summary(rows),
    }
    return templates.TemplateResponse(request, "predictions.html", {
        "rows": rows, "dates": dates, "horizon": horizon, "universe": universe,
        "made_on": made_on, "result": result, "universes": UNIVERSES, "summary": summary,
        "page": "predictions",
    })


@app.get("/ticker/{symbol}", response_class=HTMLResponse)
def ticker_page(request: Request, symbol: str):
    with connect() as conn:
        t = conn.execute("SELECT * FROM tickers WHERE symbol=?", (symbol,)).fetchone()
        if not t:
            raise HTTPException(404, f"Unknown ticker {symbol}")
        unis = [r[0] for r in conn.execute(
            "SELECT universe FROM ticker_universes WHERE symbol=?", (symbol,))]
        daily = _prediction_rows(conn, "daily", None, symbol=symbol, limit=60)
        weekly = _prediction_rows(conn, "weekly", None, symbol=symbol, limit=26)
        last = conn.execute("SELECT * FROM prices WHERE symbol=? ORDER BY date DESC LIMIT 2",
                            (symbol,)).fetchall()
    def acc(rows):
        s = [r for r in rows if r["correct"] is not None]
        return {"n": len(s), "rate": (sum(r["correct"] for r in s) / len(s)) if s else None,
                **_move_summary(rows)}
    return templates.TemplateResponse(request, "ticker.html", {
        "t": t, "unis": [UNIVERSES.get(u, u) for u in unis], "daily": daily, "weekly": weekly,
        "acc_daily": acc(daily), "acc_weekly": acc(weekly),
        "last": last[0] if last else None,
        "day_change": ((last[0]["close"] / last[1]["close"] - 1) * 100) if len(last) == 2 else None,
        "page": "ticker",
    })


@app.get("/accuracy", response_class=HTMLResponse)
def accuracy_page(request: Request, days: int = 90):
    since = today() - timedelta(days=days)
    with connect() as conn:
        table = []
        for horizon in ("daily", "weekly"):
            row = {"horizon": horizon, "all": stats.accuracy(conn, horizon, since)}
            for uni in UNIVERSES:
                row[uni] = stats.accuracy(conn, horizon, since, uni)
            table.append(row)
        best = stats.ticker_ranking(conn, "daily", since)
    return templates.TemplateResponse(request, "accuracy.html", {
        "table": table, "universes": UNIVERSES, "days": days, "accuracy_days": ACCURACY_DAYS,
        "best": best[:10], "worst": list(reversed(best[-10:])) if len(best) > 10 else [],
        "page": "accuracy",
    })


# ---------- JSON API used by the charts and the search box ----------

@app.get("/api/tickers")
def api_tickers():
    """All active tickers for the search box, with their universes and latest close."""
    with connect() as conn:
        rows = conn.execute(
            """SELECT t.symbol, t.name, t.sector, t.exchange,
                      (SELECT GROUP_CONCAT(universe) FROM ticker_universes u WHERE u.symbol = t.symbol) AS unis,
                      (SELECT close FROM prices p WHERE p.symbol = t.symbol ORDER BY date DESC LIMIT 1) AS close
               FROM tickers t WHERE t.active = 1 ORDER BY t.name COLLATE NOCASE""").fetchall()
    return [{"symbol": r["symbol"], "name": r["name"] or r["symbol"], "sector": r["sector"],
             "universes": [UNIVERSES.get(u, u) for u in (r["unis"] or "").split(",") if u],
             "close": r["close"]} for r in rows]

@app.get("/api/ticker/{symbol}/history")
def api_history(symbol: str, days: int = 180):
    since = (today() - timedelta(days=days)).isoformat()
    with connect() as conn:
        px = [dict(r) for r in conn.execute(
            "SELECT date, close FROM prices WHERE symbol=? AND date>=? ORDER BY date", (symbol, since))]
        preds = [dict(r) for r in conn.execute(
            """SELECT p.target_date, p.horizon, p.direction, p.confidence, p.base_close,
                      p.expected_move_pct, e.correct
               FROM predictions p LEFT JOIN evaluations e ON e.prediction_id = p.id
               WHERE p.symbol=? AND p.target_date>=? ORDER BY p.target_date""", (symbol, since))]
    return {"prices": px, "predictions": preds}


CHART_RANGES = {"1d": 1, "1w": 5, "1m": 31}   # 1d/1w: trading days of intraday bars; 1m: calendar days


@app.get("/api/ticker/{symbol}/chart")
def api_chart(symbol: str, range: str = "1d"):  # noqa: A002 - matches the query parameter
    """Stored prices for the price chart: 15-minute bars for 1d / 1w, daily closes for 1m.

    `base` is the close before the first point, so the change shown is over the whole range."""
    if range not in CHART_RANGES:
        raise HTTPException(400, "range must be 1d, 1w or 1m")
    import json
    with connect() as conn:
        points, kind, first_day = [], "daily", None
        if range in ("1d", "1w"):
            days = conn.execute(
                "SELECT date, bars FROM intraday WHERE symbol=? ORDER BY date DESC LIMIT ?",
                (symbol, CHART_RANGES[range])).fetchall()[::-1]
            if days:
                kind = "intraday"
                points = [{"t": ts * 1000, "close": c} for d in days for ts, c in json.loads(d["bars"])]
                first_day = days[0]["date"]
        if not points:
            # No intraday bars stored (yet): fall back to daily closes (1d/1w: the last week).
            since = (today() - timedelta(days=31 if range == "1m" else 8)).isoformat()
            rows = conn.execute("SELECT date, close FROM prices WHERE symbol=? AND date>=? ORDER BY date",
                                (symbol, since)).fetchall()
            points = [{"d": r["date"], "close": r["close"]} for r in rows]
            first_day = rows[0]["date"] if rows else None
        base = None
        if first_day:
            b = conn.execute("SELECT close FROM prices WHERE symbol=? AND date<? ORDER BY date DESC LIMIT 1",
                             (symbol, first_day)).fetchone()
            base = b["close"] if b else None
    last = points[-1]["close"] if points else None
    return {
        "range": range, "kind": kind, "points": points, "base": base, "last": last,
        "change_pct": ((last / base - 1) * 100) if last and base else None,
    }


_live_cache: dict[str, tuple[float, dict]] = {}


@app.get("/api/ticker/{symbol}/intraday")
def api_intraday(symbol: str):
    from .. import prices
    try:
        return {"bars": prices.intraday(symbol)}
    except Exception as exc:  # noqa: BLE001
        return {"bars": [], "error": str(exc)}


@app.get("/api/live")
def api_live(symbols: str = Query(..., description="comma separated")):
    """Latest intraday price for many symbols (cached 5 minutes)."""
    import yfinance as yf
    syms = [s for s in symbols.split(",") if s][:600]
    now = time.time()
    out, need = {}, []
    for s in syms:
        hit = _live_cache.get(s)
        if hit and now - hit[0] < 300:
            out[s] = hit[1]
        else:
            need.append(s)
    if need:
        try:
            data = yf.download(need, period="1d", interval="5m", auto_adjust=True,
                               group_by="ticker", progress=False, threads=True)
            for s in need:
                try:
                    col = data[s]["Close"].dropna() if len(need) > 1 else data["Close"].dropna()
                except KeyError:
                    continue
                if len(col):
                    v = {"price": float(col.iloc[-1]), "time": col.index[-1].isoformat()}
                    _live_cache[s] = (now, v)
                    out[s] = v
        except Exception as exc:  # noqa: BLE001
            log.warning("live price fetch failed: %s", exc)
    return out


@app.get("/api/accuracy/series")
def api_accuracy_series(horizon: str = "daily", days: int = 90):
    since = today() - timedelta(days=days)
    with connect() as conn:
        series = {uni: stats.daily_series(conn, horizon, uni, since) for uni in UNIVERSES}
        series["all"] = stats.daily_series(conn, horizon, None, since)
        calib = stats.calibration(conn, horizon, since)
    return {"series": series, "calibration": calib, "labels": UNIVERSES}


@app.get("/api/accuracy/moves")
def api_accuracy_moves(horizon: str = "daily", days: int = 90):
    """Predicted vs actual % move for each scored call (scatter chart on the accuracy page)."""
    with connect() as conn:
        return {"points": stats.moves(conn, horizon, today() - timedelta(days=days))}
