#!/usr/bin/env python3
"""Object-anchored hand shapes for the human refit (submission venv, CPU; scorer's penetration measure).

For each grasp window (grasp_windows.py) whose hand-object relative pose is stable (median spread < 5 cm and
< 20 deg; wiping or sliding hands are not rigid grasps and are skipped), the grasping hand's dense MHR
vertices are expressed in the object frame and averaged over the window (q_bar). If q_bar penetrates the
object's budgeted mesh it is translated out (greedy search) plus a 4 mm margin, at most 3 cm, and only if
that lowers its penetration (thin walls can make a push counter-productive). human_refit.py re-poses q_bar
with the object track each frame, so the hand moves rigidly with the smooth FoundationPose object.
Writes <out>/episode_XXXXXX.npz (one array per used window + JSON meta).
Usage: hand_anchors.py --windows FILE --contact DIR --pre DIR --hands DIR --out DIR
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

import human_refine_common as C

MAX_T_SPREAD_CM, MAX_R_SPREAD_DEG, MARGIN_M, MAX_PUSH_M = 5.0, 20.0, 0.004, 0.03


def anchors(ep, windows, contact, pre, hands_dir, out):
    diag = json.loads(windows.read_text())[str(ep)]
    e = None
    rows, arrays = [], {}
    for k, w in enumerate(diag["windows"]):
        if w["rel_trans_spread_cm"] > MAX_T_SPREAD_CM or w["rel_rot_spread_deg"] > MAX_R_SPREAD_DEG:
            continue
        e = e or C.load_contact(ep, contact, pre, hands_dir)
        a, b = w["start"], w["end"]
        sl = slice(0, C.HAND_SPLIT) if w["hand"] == 0 else slice(C.HAND_SPLIT, None)
        h = e["hands"][a:b, sl]
        q = np.einsum("tji,tvj->tvi", e["R"][a:b], h - e["t"][a:b, None])        # object frame (world units)
        qbar = q.mean(0)
        eye, zero = np.eye(3), np.zeros(3)
        pen0 = float(C.penetration(qbar, eye, zero, e["s"], e["v"], e["f"]).mean() * 100)
        off = C.clear_frame(qbar, eye, zero, e["s"], e["v"], e["f"])             # object offset that clears the points
        n = np.linalg.norm(off)
        if n > 0:
            off = off * min(n + MARGIN_M, MAX_PUSH_M) / n
        pen1 = float(C.penetration(qbar - off, eye, zero, e["s"], e["v"], e["f"]).mean() * 100)
        if pen1 >= pen0:
            off, pen1 = np.zeros(3), pen0
        qbar = qbar - off                                                       # = moving the hand the other way
        arrays[f"w{len(rows)}"] = qbar.astype(np.float32)
        rows.append({"window": k, "hand": w["hand"], "start": a, "end": b, "push_cm": float(np.linalg.norm(off) * 100),
                     "pen_cm": [pen0, pen1], "spread_cm": w["rel_trans_spread_cm"], "spread_deg": w["rel_rot_spread_deg"],
                     "jitter_cm": float(np.linalg.norm(q - q.mean(0), axis=-1).mean() * 100)})
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"episode_{ep:06d}.npz", meta=json.dumps(rows), **arrays)
    return ep, rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--windows", type=Path, required=True)
    ap.add_argument("--contact", type=Path, required=True)
    ap.add_argument("--pre", type=Path, required=True)
    ap.add_argument("--hands", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    with ProcessPoolExecutor(args.workers) as pool:
        res = dict(pool.map(partial(anchors, windows=args.windows, contact=args.contact, pre=args.pre,
                                    hands_dir=args.hands, out=args.out), range(30)))
    (args.out / "anchors.json").write_text(json.dumps({str(k): v for k, v in res.items()}, indent=2))
    n = sum(len(v) for v in res.values())
    frames = sum(r["end"] - r["start"] for v in res.values() for r in v)
    print(f"hand anchors: {n} stable grasp windows ({frames} frames) in episodes {sorted(k for k, v in res.items() if v)}")


if __name__ == "__main__":
    main()
