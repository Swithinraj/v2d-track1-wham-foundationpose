#!/usr/bin/env python3
"""Upload one packed file to the five Track 1 boards through the official client, within the rules (submission venv).

The challenge allows five submissions per week and unlimited ones in the final three days, which start 72 hours before the
freeze. Kaggle cannot express a weekly limit, so nothing stops an early upload except this check. The token is read from
a file and passed only through the environment, so it never appears in a command line or a log.

Usage: submit_candidate.py FILE.parquet -m "what changed"  [--token FILE] [--dry-run]
Writes FILE.<...>.kaggle-receipt.json next to the file (the official client's receipt) and prints the five scores.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
FREEZE = dt.datetime(2026, 11, 4, 22, 0, tzinfo=dt.timezone.utc)        # November 4, 5:00 p.m. EST
UNLIMITED_FROM = FREEZE - dt.timedelta(days=3)


def window_is_open(now):
    return UNLIMITED_FROM <= now < FREEZE


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("file", type=Path)
    parser.add_argument("-m", "--message", required=True)
    parser.add_argument("--token", type=Path, default=ROOT.parent / "v2d_track1_handoff_2026-10-31/tokens/kaggle_api.txt")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    now = dt.datetime.now(dt.timezone.utc)
    if not args.dry_run and not window_is_open(now):
        sys.exit(f"refusing to upload at {now:%Y-%m-%d %H:%M} UTC: the unlimited window runs {UNLIMITED_FROM:%Y-%m-%d %H:%M} "
                 f"to {FREEZE:%Y-%m-%d %H:%M} UTC, and this week's five submissions are already used")
    env = dict(os.environ, KAGGLE_API_TOKEN=args.token.read_text().strip(),
               KAGGLE_CONFIG_DIR=str(ROOT.parent / ".cache/kaggle"), TMPDIR=str(ROOT.parent / ".tmp"))
    kit = ROOT / "official/v2d_submission_kit"
    command = [sys.executable, str(ROOT / "scripts/v2d_submit_team.py"), str(kit / "submit/v2d_submit.py"),
               str(args.file.resolve()), "--track", "track_1", "-m", args.message]
    command += ["--dry-run"] if args.dry_run else ["--wait"]
    sys.exit(subprocess.run(command, cwd=kit, env=env).returncode)


if __name__ == "__main__":
    main()
