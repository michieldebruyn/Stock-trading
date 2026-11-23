"""Runs the website and the morning/evening jobs in one process.

    python -m app.serve                 # http://localhost:8000 + scheduled jobs
    python -m app.serve --no-scheduler  # website only (if you use cron instead)
"""
from __future__ import annotations

import argparse
import logging
from logging.handlers import RotatingFileHandler

import uvicorn
from apscheduler.schedulers.background import BackgroundScheduler

from . import evaluate, predict
from .config import ROOT, get_config, local_tz


def setup_logging():
    (ROOT / "logs").mkdir(exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    fh = RotatingFileHandler(ROOT / "logs" / "jobs.log", maxBytes=2_000_000, backupCount=5)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(fh)
    root.addHandler(sh)


def _safe(fn, name):
    def wrapper():
        try:
            fn()
        except Exception:  # noqa: BLE001
            logging.getLogger(name).exception("%s job failed", name)
    return wrapper


def start_scheduler() -> BackgroundScheduler:
    cfg = get_config()["schedule"]
    sched = BackgroundScheduler(timezone=local_tz())
    mh, mm = map(int, cfg["morning"].split(":"))
    eh, em = map(int, cfg["evening"].split(":"))
    # misfire_grace_time: if the computer was asleep, still run when it wakes up
    # (predict.py itself refuses to predict markets that have already opened).
    sched.add_job(_safe(predict.run, "predict"), "cron", day_of_week="mon-fri", hour=mh, minute=mm,
                  id="predict", misfire_grace_time=3 * 3600, coalesce=True)
    sched.add_job(_safe(evaluate.run, "evaluate"), "cron", day_of_week="mon-fri", hour=eh, minute=em,
                  id="evaluate", misfire_grace_time=10 * 3600, coalesce=True)
    sched.start()
    for job in sched.get_jobs():
        logging.info("Scheduled %s, next run %s", job.id, job.next_run_time)
    return sched


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--no-scheduler", action="store_true")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    setup_logging()
    if not args.no_scheduler:
        start_scheduler()
    uvicorn.run("app.web.main:app", host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
