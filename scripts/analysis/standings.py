#!/usr/bin/env python3
"""Track 1 standings as the rules compute them: the lowest sum of ranks over the five boards (FAQ).

Tied scores share a rank, so every team at exactly PEN 0 is first on that board. Downloads each board from Kaggle, prints the
table and, for each board, the score that would reach each rank.
Usage: standings.py [--team NAME] [--top 12] [--out DIR]    (run from project1; reads the Kaggle token from the handoff folder)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import zipfile

BOARDS = ("cd-h", "cd-o", "acc-h", "acc-o", "pen")


def ranks(scores):
    """Competition ranking: ties share the best rank, the next score skips ahead."""
    ordered = sorted(scores.values())
    return {team: ordered.index(score) + 1 for team, score in scores.items()}


def rank_sums(table):
    """{board: {team: score}} -> (per-board ranks, [(team, sum)] for teams on every board, best first)."""
    per_board = {board: ranks(scores) for board, scores in table.items()}
    teams = set.intersection(*(set(scores) for scores in table.values()))
    totals = sorted(((team, sum(per_board[board][team] for board in table)) for team in teams), key=lambda item: (item[1],item[0]))
    return per_board, totals


def overall_table(totals):
    """Computed rank by rank sum, retaining ties without inventing a tiebreaker."""
    positions=ranks(dict(totals))
    return [(positions[team],team,total) for team,total in totals]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--team", default="Izus_systems")
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--out", type=Path, default=Path(".tmp/lb"))
    args = parser.parse_args()
    os.environ["KAGGLE_API_TOKEN"] = Path("v2d_track1_handoff_2026-10-31/tokens/kaggle_api.txt").read_text().strip()
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    table = {}
    for board in BOARDS:
        folder = args.out / board
        folder.mkdir(parents=True, exist_ok=True)
        for old in list(folder.glob("*")):
            old.unlink()
        api.competition_leaderboard_download(f"v2d-challenge-track1-{board}", str(folder), quiet=True)
        for archive in folder.glob("*.zip"):
            with zipfile.ZipFile(archive) as z:
                z.extractall(folder)
        with next(folder.glob("*.csv")).open(encoding="utf-8") as handle:
            table[board] = {row["TeamName"]: float(row["Score"]) for row in csv.DictReader(handle)}
    (args.out / "all.json").write_text(json.dumps(table))
    per_board, totals = rank_sums(table)
    print("#  team                 sum  " + " ".join(f"{b:>5s}" for b in BOARDS) + " | scores")
    for position,team,total in overall_table(totals)[:args.top]:
        print(f"{position:2d} {team[:20]:20s} {total:3d}  " + " ".join(f"{per_board[b][team]:5d}" for b in BOARDS) + " | "
              + " ".join(f"{table[b][team]:.5f}" for b in BOARDS))
    mine = [k for k, (team, _) in enumerate(totals) if team == args.team]
    if mine:
        team, total = totals[mine[0]]
        position=ranks(dict(totals))[team]
        print(f"{team}: computed overall #{position} by rank sum (ties retained), sum {total}, ranks " + str({b: per_board[b][team] for b in BOARDS}))
    for board in BOARDS:
        ladder = sorted(set(table[board].values()))[:10]
        print(f"{board:6s} scores to reach each place: " + " ".join(f"{score:.5f}" for score in ladder))


if __name__ == "__main__":
    main()
