"""Keep the app's own records (predictions, scores, runs, tickers) in Git so cloud runs, which start
from a fresh checkout every time, can carry them from one run to the next.

The records live as JSON-lines files under `state/` on their own branch (default
`claude/stock-data`), apart from the code. Prices are not stored: the jobs re-download them.

    python scripts/state.py pull                 # load saved records from origin into the database
    python scripts/state.py push -m "message"    # save the database's records to origin
    python scripts/state.py export --out DIR     # write the files to DIR only (no Git)

`pull` merges into the database (rows with the same key are overwritten), so it is also how you
bring the cloud results into your local dashboard.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.db import connect  # noqa: E402

BRANCH = "claude/stock-data"
STATE_DIR = "state"

# Parent tables first, so foreign keys are satisfied on import. Value = primary-key columns.
TABLES: dict[str, tuple[str, ...]] = {
    "tickers": ("symbol",),
    "ticker_universes": ("symbol", "universe"),
    "runs": ("id",),
    "predictions": ("id",),
    "evaluations": ("prediction_id",),
}


def git(*args: str, cwd: Path = ROOT, check: bool = True) -> subprocess.CompletedProcess:
    res = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, encoding="utf-8")
    if check and res.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed:\n{res.stderr.strip()}")
    return res


# --- database <-> text ---------------------------------------------------------------------------

APPEND_ONLY = ("runs", "predictions", "evaluations")

# Large, ever-growing tables are split into one file per month (of the prediction date) so no
# single file approaches GitHub's 100 MB limit.
MONTHLY_SQL = {
    "predictions": "SELECT substr(made_on, 1, 7) AS _month, * FROM predictions ORDER BY id",
    "evaluations": ("SELECT substr(p.made_on, 1, 7) AS _month, e.* FROM evaluations e "
                    "JOIN predictions p ON p.id = e.prediction_id ORDER BY e.prediction_id"),
}


def _line(row: dict) -> str:
    return json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"


def export_files() -> dict[str, str]:
    """{path under state/: JSON-lines text}, rows sorted by primary key so diffs stay small."""
    out: dict[str, str] = {}
    conn = connect()
    try:
        for table, pk in TABLES.items():
            if table in MONTHLY_SQL:
                parts: dict[str, list[str]] = {}
                for r in conn.execute(MONTHLY_SQL[table]):
                    row = dict(r)
                    parts.setdefault(row.pop("_month"), []).append(_line(row))
                for month, lines in parts.items():
                    out[f"{table}/{month}.jsonl"] = "".join(lines)
            else:
                rows = conn.execute(f"SELECT * FROM {table} ORDER BY {', '.join(pk)}")
                out[f"{table}.jsonl"] = "".join(_line(dict(r)) for r in rows)
    finally:
        conn.close()
    return out


def _table_of(path: str) -> str:
    return path.split("/")[0].removesuffix(".jsonl")


def import_files(files: dict[str, str]) -> dict[str, int]:
    """Upsert rows from {path under state/: text} into the database. Returns rows per table."""
    counts = {}
    conn = connect()
    try:
        for table, pk in TABLES.items():
            text = "".join(t for p, t in sorted(files.items()) if _table_of(p) == table)
            if not text:
                continue
            rows = [json.loads(line) for line in text.splitlines() if line.strip()]
            for row in rows:
                cols = list(row)
                updates = [c for c in cols if c not in pk]
                sql = (f"INSERT INTO {table} ({', '.join(cols)}) "
                       f"VALUES ({', '.join('?' * len(cols))}) "
                       f"ON CONFLICT ({', '.join(pk)}) DO ")
                sql += ("UPDATE SET " + ", ".join(f"{c}=excluded.{c}" for c in updates)
                        if updates else "NOTHING")
                conn.execute(sql, [row[c] for c in cols])
            counts[table] = len(rows)
        conn.commit()
    finally:
        conn.close()
    return counts


# --- Git -----------------------------------------------------------------------------------------

def remote_branch_exists(branch: str) -> bool:
    return git("ls-remote", "--exit-code", "--heads", "origin", branch, check=False).returncode == 0


def pull(branch: str) -> None:
    if not remote_branch_exists(branch):
        print(f"No saved state on origin/{branch} yet; starting from an empty record.")
        return
    git("fetch", "origin", f"+refs/heads/{branch}:refs/remotes/origin/{branch}")
    ref = f"origin/{branch}"
    paths = git("ls-tree", "-r", "--name-only", ref, f"{STATE_DIR}/").stdout.split()
    files = {p.removeprefix(f"{STATE_DIR}/"): git("show", f"{ref}:{p}").stdout
             for p in paths if p.endswith(".jsonl")}
    counts = import_files(files)
    print("Imported " + ", ".join(f"{n} {t}" for t, n in counts.items()) if counts
          else "Saved state is empty.")


def _row_counts(files: dict[str, str]) -> dict[str, int]:
    rows: dict[str, int] = {}
    for p, t in files.items():
        rows[_table_of(p)] = rows.get(_table_of(p), 0) + t.count("\n")
    return rows


def push(branch: str, message: str, force: bool = False) -> None:
    files = export_files()
    tmp = Path(tempfile.mkdtemp(prefix="stock-state-"))
    wt = tmp / "wt"
    orphan = None
    try:
        if remote_branch_exists(branch):
            git("fetch", "origin", f"+refs/heads/{branch}:refs/remotes/origin/{branch}")
            git("worktree", "add", "--detach", str(wt), f"origin/{branch}")
        else:
            git("worktree", "add", "--detach", str(wt))
            orphan = "stock-state-init"
            git("checkout", "--orphan", orphan, cwd=wt)
            git("rm", "-rf", "--quiet", ".", cwd=wt)
        if not force:
            # Runs, predictions and scores are only ever added, so fewer rows than stored means
            # the database was not pulled first: pushing would wipe history. (Ticker lists can
            # legitimately shrink when index constituents change.)
            old, new = _row_counts(read_dir(wt / STATE_DIR)), _row_counts(files)
            shrunk = [f"{t}: {n} stored, {new.get(t, 0)} now" for t, n in old.items()
                      if t in APPEND_ONLY and new.get(t, 0) < n]
            if shrunk:
                raise SystemExit("Refusing to push, it would delete saved records (run "
                                 "`state.py pull` first, or pass --force):\n  " + "\n  ".join(shrunk))
        write_dir(wt / STATE_DIR, files)
        (wt / "README.md").write_text(
            "# Stock app records\n\nWritten by `scripts/state.py push` after each cloud run. "
            "Load them into a local database with `python scripts/state.py pull`.\n",
            encoding="utf-8", newline="\n")
        git("add", "-A", cwd=wt)
        if git("diff", "--cached", "--quiet", cwd=wt, check=False).returncode == 0:
            print("No changes to save.")
            return
        git("commit", "-m", message, cwd=wt)
        res = git("push", "origin", f"HEAD:refs/heads/{branch}", cwd=wt, check=False)
        if res.returncode != 0:
            raise SystemExit(f"git push failed:\n{res.stderr}")
        print(f"Saved state to origin/{branch}: " +
              ", ".join(f"{n} {t}" for t, n in _row_counts(files).items()))
    finally:
        git("worktree", "remove", "--force", str(wt), check=False)
        if orphan:
            git("branch", "-D", orphan, check=False)
        shutil.rmtree(tmp, ignore_errors=True)


def write_dir(d: Path, files: dict[str, str]) -> None:
    """Replace the contents of `d` with `files` (so months that no longer exist are removed)."""
    shutil.rmtree(d, ignore_errors=True)
    for rel, text in files.items():
        (d / rel).parent.mkdir(parents=True, exist_ok=True)
        (d / rel).write_text(text, encoding="utf-8", newline="\n")


def read_dir(d: Path) -> dict[str, str]:
    return {p.relative_to(d).as_posix(): p.read_text(encoding="utf-8")
            for p in sorted(d.rglob("*.jsonl"))}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("pull", "push"):
        s = sub.add_parser(name)
        s.add_argument("--branch", default=BRANCH)
        if name == "push":
            s.add_argument("-m", "--message", default="Update stock app records")
            s.add_argument("--force", action="store_true",
                           help="push even if it removes saved records")
    e = sub.add_parser("export")
    e.add_argument("--out", type=Path, required=True)
    i = sub.add_parser("import")
    i.add_argument("--dir", type=Path, required=True)
    args = p.parse_args(argv)

    if args.cmd == "pull":
        pull(args.branch)
    elif args.cmd == "push":
        push(args.branch, args.message, args.force)
    elif args.cmd == "export":
        files = export_files()
        write_dir(args.out, files)
        print(f"Wrote {len(files)} files to {args.out}")
    elif args.cmd == "import":
        print(import_files(read_dir(args.dir)))


if __name__ == "__main__":
    main()
