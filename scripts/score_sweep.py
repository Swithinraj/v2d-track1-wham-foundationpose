#!/usr/bin/env python3
"""Score a list of candidate parquets against a pseudo-GT reference (one GPU shard).

usage: score_sweep.py --ref DIR --tasks tasks.json --out DIR [--shard 0] [--nshards 4]
tasks.json: [{"name": "P2", "path": ".../track1_cand_P2.parquet"}, ...]
Writes out/<name>.json with {"mean": {...}, "per_episode": {...}}.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402
import localscore as L  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--tasks", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    a.out.mkdir(parents=True, exist_ok=True)
    tasks = json.loads(a.tasks.read_text())
    mine = tasks[a.shard::a.nshards]
    for t in mine:
        dst = a.out / f"{t['name']}.json"
        if dst.exists():
            print(f"[shard {a.shard}] skip {t['name']} (exists)", flush=True)
            continue
        t0 = time.time()
        try:
            agg, per_ep = L.score_candidate(Path(t["path"]), a.ref, device=dev)
            dst.write_text(json.dumps({"mean": agg, "per_episode": per_ep}, indent=1))
            print(f"[shard {a.shard}] {t['name']}: " +
                  " ".join(f"{k}={v:.5f}" for k, v in agg.items()) +
                  f"  ({time.time() - t0:.0f}s)", flush=True)
        except Exception as e:  # noqa: BLE001 — record and continue the sweep
            dst.write_text(json.dumps({"error": str(e)}))
            print(f"[shard {a.shard}] {t['name']}: ERROR {e}", flush=True)


if __name__ == "__main__":
    main()
