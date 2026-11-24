# Stock Predictions

Every trading morning Claude predicts whether each stock will close **up or down**, for today
(**daily**) and for the whole week (**weekly**, made on Monday). Every evening the app checks those
calls against the real closing prices. A local website shows the predictions, live prices and
how accurate Claude has been.

Markets covered: **S&P 500**, **~50 of Europe's largest stocks** and the **BEL20**.

> This is a learning and tracking tool, not financial advice. Short-term price direction is
> close to a coin flip; the accuracy page tells you honestly whether there is any edge.

## How it works

```
07:30  predict   ─ download prices (Yahoo Finance) → compute indicators → ask Claude
                   in batches of 40 tickers → store calls in `predictions`
22:45  evaluate  ─ download closing prices → score every call whose target day has closed
                   → store results in `evaluations`
any    website   ─ http://localhost:8000: dashboard, predictions, ticker charts, accuracy
```

- **Daily call**: will today's close be above yesterday's close?
- **Weekly call**: will the last close of this week be above last week's final close?
  Made on the first trading day of the week only.
- Calls are only made **before a market opens** (European markets open 09:00 CET, NYSE 15:30
  CET), so no call is made with knowledge of the session it predicts. Each exchange's holidays
  are respected (`pandas_market_calendars`).
- Claude gets per stock: trailing returns, RSI, MACD, moving-average gaps, ATR, volume, distance
  from the 52-week high and its own recent hit record on that stock. It also gets the overnight
  backdrop (US futures, Asia, VIX, rates, FX, oil) and a summary of its own past accuracy and
  calibration, so it can correct biases.
- Each call stores direction, confidence (0.50–0.95), expected % move and a one-line rationale.

## Setup

Requires Python 3.11+ and an Anthropic API key.

```bash
python -m venv .venv
```

```bash
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

```bash
pip install -r requirements.txt
```

```bash
export ANTHROPIC_API_KEY=sk-ant-...   # Windows PowerShell: $env:ANTHROPIC_API_KEY="sk-ant-..."
```

Download the S&P 500 list and load all universes into the database:

```bash
python -m app.universe refresh
```

Backfill price history (needed for the 200-day average):

```bash
python -m app.prices --years 2
```

## Running

The easiest way is one process that serves the website **and** runs both jobs on schedule:

```bash
python -m app.serve
```

Open <http://localhost:8000>. Leave it running; the morning and evening jobs fire at the times in
`config.yaml` (Mon–Fri). If your computer was asleep, a missed job runs when it wakes up, and the
morning job skips any market that has already opened.

Run the jobs by hand:

```bash
python -m app.predict --dry-run    # show what would be sent, no API call
```

```bash
python -m app.predict
```

```bash
python -m app.evaluate
```

Prefer cron instead of the built-in scheduler? Start the site with
`python -m app.serve --no-scheduler` and add (adjust the path):

```bash
30 7  * * 1-5  cd ~/Stock-trading && .venv/bin/python -m app.predict  >> logs/predict.log 2>&1
45 22 * * 1-5  cd ~/Stock-trading && .venv/bin/python -m app.evaluate >> logs/evaluate.log 2>&1
```

### Run the jobs in the cloud (Claude Code routines)

Two Claude Code cloud routines can run the jobs instead of your computer, so it doesn't have to
be on. A cloud run starts from a fresh checkout, so the app's records (predictions, scores, runs,
tickers and prices) are kept in Git on the `claude/stock-data` branch as JSON-lines files: one
file per month for predictions, scores and daily prices, one file per day for the 15-minute
intraday prices (the last two weeks). Each run only downloads the most recent prices.

In the cloud the **routine's own Claude (Sonnet 5.5) makes the predictions**, so no API key is
needed. The morning routine runs:

```bash
pip install -r requirements.txt
python scripts/state.py pull                          # load the saved records
python -m app.universe refresh
python -m app.predict --export-batches work/          # same prompt and data as the API version
#   ...Claude reads work/instructions.md and writes work/answer_NN.json for each batch...
python -m app.predict --check-answers work/           # validate
python -m app.predict --import-answers work/          # store the calls
python scripts/state.py push -m "predict 2026-11-24"
```

The evening routine runs `pull`, `python -m app.evaluate` and `push`. European stocks come first
in the batches because their markets open first; a call for a market that has opened by import
time is discarded, so no call is made with knowledge of the session it predicts. Calls made this
way are recorded with the model `claude-sonnet-5-5 (Claude Code)`.

The cloud environment needs network access to Yahoo Finance and PyPI. The S&P 500 list is kept
in the repo (`data/universe/sp500.csv`), so Wikipedia is optional: if it can't be reached, the
saved list is used. To update the saved list, run `python -m app.universe refresh` locally and
commit the CSV.

To see the cloud results in your local dashboard, pull them into your database and start the
site without the local scheduler (so it doesn't make its own, conflicting predictions):

```bash
python scripts/state.py pull
```

```bash
python -m app.serve --no-scheduler
```

### Publish the website on GitHub Pages

`.github/workflows/pages.yml` publishes a static snapshot of the website to GitHub Pages: after
the morning predictions (08:15 UTC), after the evening scoring (22:15 UTC), on every push to
`main`, and on demand from the Actions tab. It loads the latest records from `claude/stock-data`
and renders every page with `scripts/build_site.py`. Everything works from the stored data,
including the 1D / 1W / 1M charts, the search box and the predictions filters; the live Yahoo
Finance features (the **Load live prices** button and the **Today, live** chart) exist only on
the local site.

One-time setup on GitHub: **Settings > Pages > Source: GitHub Actions**, and make `main` the
default branch (scheduled workflows only run from the default branch). Build it locally with:

```bash
python scripts/build_site.py --out _site
```

### Try the website with fake data first

```bash
python scripts/seed_demo.py
```

```bash
STOCKS_DB=data/demo.db python -m app.serve --no-scheduler
```

This builds `data/demo.db` with synthetic prices and random calls. Your real `data/stocks.db` is
never touched.

## Website

| Page | Shows |
|---|---|
| **Dashboard** | Overall daily/weekly accuracy, per-market cards (up/down split of today's calls, hit rates), Claude's market view, today's highest-confidence calls |
| **Predictions** | Every call for a date, filterable by market and result, with a **Load live prices** button that shows whether each open call is on track right now |
| **Ticker** (search box) | Price chart with **1D / 1W / 1M** buttons (15-minute prices for 1D and 1W, daily closes for 1M, all saved by the morning and evening jobs), 6-month price chart with every call marked ▲/▼ in green (right) or red (wrong), today's live intraday price against the base close, full daily and weekly history with rationales |
| **Accuracy** | Hit rate by market and horizon, rolling hit rate vs the 50% coin flip, calibration chart (does "70% confident" mean right 70% of the time?), **move accuracy** (how close the predicted % move was to the actual one: predicted +7%, actual +10% scores 70%, wrong direction scores 0%) with a predicted-vs-actual scatter, most/least predictable stocks |

## Configuration (`config.yaml`)

- `universes`: which lists to predict. Edit `data/universe/bel20.csv` / `europe.csv` to change
  constituents (the BEL20 is reviewed every quarter; check it now and then).
- `schedule`: job times in `timezone`.
- `claude.model`, `claude.effort`: `claude-opus-5-5` at `medium` by default. `low` is cheaper,
  `high` reasons more.
- `claude.batch_size`, `claude.max_parallel`: tickers per request and concurrent requests.

## Cost

About 575 tickers. With Claude Opus 5.5 at `medium` effort, expect roughly **$1–3 per trading
day** (more on Mondays, when weekly calls are added): about $25–60 a month. The **Dashboard**
shows the token usage of the last run. Reduce cost with `effort: low`, or by dropping a universe.

## Database

SQLite at `data/stocks.db` (open it with any SQLite browser).

| Table | Contents |
|---|---|
| `tickers`, `ticker_universes` | Symbols, names, exchange, and which universe(s) each belongs to |
| `prices` | Daily OHLCV (split/dividend adjusted) for tickers and market-context symbols |
| `intraday` | 15-minute closes per stock per day (last two weeks), refreshed by both daily jobs, for the 1D / 1W charts |
| `predictions` | One row per call: horizon, target date, base close, direction, confidence, expected move, rationale. Written once in the morning, never changed |
| `evaluations` | Actual close, % change, actual direction, right/wrong |
| `runs` | Each job run: status, token usage, Claude's market view, errors |

## Project layout

```
app/
  config.py, db.py          config + schema
  universe.py               S&P 500 download, CSV lists → DB
  market.py                 exchange calendars, opening times
  prices.py                 Yahoo Finance download, intraday quotes
  features.py               technical indicators
  claude_predictor.py       prompt, structured-output schema, API call
  predict.py                morning job
  evaluate.py               evening job
  stats.py                  accuracy / calibration queries
  serve.py                  website + scheduler
  web/                      FastAPI app, templates, Chart.js
scripts/seed_demo.py        demo database
tests/                      offline tests (pytest)
```

## Tests

```bash
python -m pytest -q
```
