#!/usr/bin/env python3
"""Grasp windows and hand-object consistency per episode (submission venv, CPU; human refinement stage).

Per frame and hand: distance from 128 hand points to the posed (budgeted) object vertices, and object
speed. Grasp window = a hand within CONTACT_CM of the object while the object moves (> MOVE_CM per frame),
runs >= 10 frames after closing gaps <= 5 frames. Per window: the hand's rigid motion (Kabsch of its 2318
dense vertices against the window's first frame), the spread of the hand-object relative pose (rigid grasp
and consistent tracks give a small spread) and the acceleration of the object as the hand would carry it.
For every rigid object the hand-carried pose is 5-30x jerkier than the FoundationPose track, so windows
are only used to anchor the HAND to the object (hand_anchors.py), never the other way round.
Writes <out> (JSON keyed by episode).
Usage: grasp_windows.py --contact DIR --pre DIR --hands DIR --out FILE
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

import human_refine_common as C

CONTACT_CM, MOVE_CM = 5.0, 0.3


def object_name(ep):
    for line in (C.ROOT / "data/track_1/meta/episodes_metadata.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["episode_index"] == ep:
            return row["object"]
    return "?"


def acc_cm(x):
    return np.linalg.norm(np.diff(x, n=2, axis=0), axis=1) * 100 if len(x) >= 3 else np.zeros(0)


def analyse(ep, contact, pre, hands_dir):
    e = C.load_contact(ep, contact, pre, hands_dir)
    rng = np.random.default_rng(0)
    T, R, t, s = e["T"], e["R"], e["t"], e["s"]
    vsub = e["v"][rng.choice(len(e["v"]), min(3000, len(e["v"])), replace=False)]
    hands = [e["hands"][:, :C.HAND_SPLIT], e["hands"][:, C.HAND_SPLIT:]]
    hsub = [rng.choice(C.HAND_SPLIT, 128, replace=False) for _ in hands]
    dist = np.zeros((2, T))
    for x in range(T):
        ov = s * vsub @ R[x].T + t[x]
        for h in range(2):
            d = np.linalg.norm(hands[h][x, hsub[h]][:, None] - ov[None], axis=-1)
            dist[h, x] = d.min() * 100
    vel = np.zeros(T)
    vel[1:-1] = np.linalg.norm(t[2:] - t[:-2], axis=1) / 2 * 100
    moving = np.convolve(vel, np.ones(5) / 5, mode="same") > MOVE_CM
    scored = np.zeros(T, bool)
    scored[e["scored"]] = True
    windows = []
    for h in range(2):
        for a, b in C.runs((dist[h] < CONTACT_CM) & moving, 10, 5):
            H = hands[h][a:b]
            Rt = [C.kabsch(H[0], H[k]) for k in range(b - a)]
            Rh = np.stack([r for r, _ in Rt]); th = np.stack([tt for _, tt in Rt])
            Rrel = np.einsum("kji,kjl->kil", Rh, R[a:b])                       # Rh^T Ro
            trel = np.einsum("kji,kj->ki", Rh, t[a:b] - th)                     # Rh^T (to - th)
            Rm, tm = C.chordal_mean(Rrel), trel.mean(0)
            carried = np.einsum("kij,j->ki", Rh, tm) + th                       # object translation if the hand carried it
            windows.append({"hand": h, "start": int(a), "end": int(b), "scored_frames": int(scored[a:b].sum()),
                            "rel_trans_spread_cm": float(np.median(np.linalg.norm(trel - tm, axis=1)) * 100),
                            "rel_rot_spread_deg": float(np.median(C.angle_deg(Rrel, Rm[None]))),
                            "acc_object": float(acc_cm(t[a:b]).mean()), "acc_carried": float(acc_cm(carried).mean()),
                            "object_travel_cm": float(np.linalg.norm(t[b - 1] - t[a]) * 100)})
    grasp = np.zeros(T, bool)
    for w in windows:
        grasp[w["start"]:w["end"]] = True
    return ep, {"object": object_name(ep), "frames": T, "grasp_frames": int(grasp.sum()),
                "grasp_scored_frac": float(grasp[scored].mean()) if scored.any() else 0.0,
                "contact_frac": float(((dist < CONTACT_CM).any(0))[scored].mean()) if scored.any() else 0.0,
                "windows": windows}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--contact", type=Path, required=True, help="episode dir whose object track defines contact")
    ap.add_argument("--pre", type=Path, required=True, help="episode dir the contact hands were computed from")
    ap.add_argument("--hands", type=Path, required=True, help="mhr_bodies.py --hands-only output for --pre")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    with ProcessPoolExecutor(args.workers) as pool:
        rows = dict(pool.map(partial(analyse, contact=args.contact, pre=args.pre, hands_dir=args.hands), range(30)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({str(k): v for k, v in rows.items()}, indent=2))
    print(f"grasp windows: {sum(len(r['windows']) for r in rows.values())} in {sum(bool(r['windows']) for r in rows.values())} episodes -> {args.out}")


if __name__ == "__main__":
    main()
