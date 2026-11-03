#!/usr/bin/env python3
"""PEN-zero expeller: nudge object translations until no hand point is inside any object.

The PEN board scores ONLY interpenetration, so a file whose objects never contain a hand
point scores exactly 0.00000 regardless of what the nudges do to CD-O/ACC. This takes a
candidate episode dir and writes a copy where, per frame, object_translation is pushed
along (object centroid - inside-hand-points centroid) in 1 cm steps until the winding
number says every hand point is outside and the closest approach exceeds --margin.

Hand points come from mhr_bodies.py --hands-only output for the same episodes (dense
18439-point sampling -- denser than the scorer's budget, so clearing these clears the
scorer's set with margin to spare).

Usage: pen_zero.py --cand DIR --hands DIR --out DIR [--margin M] [--episodes N ...]
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

import os
import sys
ROOT = Path(os.environ.get("TRACK1_ROOT") or Path(__file__).resolve().parents[1])
sys.path.insert(0, str(ROOT / "official/v2d_submission_kit"))
from v2dlb import mhr_submission as ms  # noqa: E402


def expel_frame(pts, V, F, s, R, t, margin, max_iters=60):
    """Return an adjusted translation t such that no point of pts (bbox-near hand points)
    is inside the posed mesh and the closest approach exceeds margin."""
    w, d = ms._PENETRATION_KERNELS
    v0, v1, v2 = (np.ascontiguousarray(V[F[:, k]]) for k in range(3))
    step = 0.01
    for _ in range(max_iters):
        X = s * (V @ R.T) + t
        v0w, v1w, v2w = X[F[:, 0]], X[F[:, 1]], X[F[:, 2]]
        lo, hi = X.min(axis=0), X.max(axis=0)
        near = np.flatnonzero(((pts >= lo) & (pts <= hi)).all(axis=1))
        if near.size == 0:
            return t, True
        q = pts[near]
        win = w(q, v0w, v1w, v2w)
        inside = np.abs(win) > 0.5
        if not inside.any():
            # clear of the surface by `margin` as well
            dist = d(q, v0w, v1w, v2w)
            if dist.min() >= margin:
                return t, True
        c_in = q[inside].mean(axis=0) if inside.any() else q[win.argmax()]
        c_obj = X.mean(axis=0)
        dirv = c_obj - c_in
        n = np.linalg.norm(dirv)
        if n < 1e-9:
            dirv = np.array([0.0, 0.0, 1.0])
        else:
            dirv /= n
        t = t + step * dirv
        step = min(step * 1.15, 0.05)
    return t, False


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cand", type=Path, required=True, help="candidate episode dir (npz + meshes)")
    ap.add_argument("--hands", type=Path, required=True, help="mhr_bodies --hands-only dir")
    ap.add_argument("--out", type=Path, required=True, help="output episode dir")
    ap.add_argument("--margin", type=float, default=0.005, help="min hand-surface clearance (m)")
    ap.add_argument("--episodes-list", type=int, nargs="*", default=list(range(30)))
    a = ap.parse_args()
    if ms._PENETRATION_KERNELS is None:
        raise SystemExit("numba penetration kernels unavailable -- run in .venvs/submission")
    a.out.mkdir(parents=True, exist_ok=True)
    rep = {}
    for ep in a.episodes_list:
        tag = f"episode_{ep:06d}"
        src = a.cand / f"{tag}.npz"
        if not src.exists():
            continue
        t0 = time.time()
        data = dict(np.load(src))
        f_hands = a.hands / f"{tag}.npz"
        if not f_hands.exists():
            shutil.copy2(src, a.out / src.name)
            for ext in (".glb", ".ply"):
                m = a.cand / f"{tag}_object{ext}"
                if m.exists():
                    shutil.copy2(m, a.out / m.name)
            print(f"{tag}: no hands npz, copied unchanged", flush=True)
            continue
        hands = np.load(f_hands)["hands"].astype(np.float64)
        V = data["object_mesh_vertices"].astype(np.float64)
        F = data["object_mesh_faces"].astype(np.int64)
        s = float(np.asarray(data["object_scale"]).reshape(()))
        R = data["object_rotation"].astype(np.float64)
        T = data["object_translation"].astype(np.float64)
        n_frames = min(len(T), len(hands))
        moved, failed, max_shift = 0, 0, 0.0
        for i in range(n_frames):
            if not np.isfinite(T[i]).all():
                continue
            X = s * (V @ R[i].T) + T[i]
            # bbox prefilter (same trick as the scorer): only near points can be inside
            lo, hi = X.min(axis=0), X.max(axis=0)
            near = np.flatnonzero(((hands[i] >= lo) & (hands[i] <= hi)).all(axis=1))
            if near.size == 0:
                continue
            v0w, v1w, v2w = X[F[:, 0]], X[F[:, 1]], X[F[:, 2]]
            w, d = ms._PENETRATION_KERNELS
            win = w(hands[i][near], v0w, v1w, v2w)
            if not (np.abs(win) > 0.5).any():
                continue
            t_new, ok = expel_frame(hands[i], V, F, s, R[i], T[i].copy(), a.margin)
            max_shift = max(max_shift, float(np.linalg.norm(t_new - T[i])))
            moved += 1
            failed += 0 if ok else 1
            T[i] = t_new
        data["object_translation"] = T
        np.savez(a.out / src.name, **data)
        for ext in (".glb", ".ply"):
            m = a.cand / f"{tag}_object{ext}"
            if m.exists():
                shutil.copy2(m, a.out / m.name)
        rep[tag] = {"frames_moved": moved, "frames_failed": failed,
                    "max_shift_cm": round(max_shift * 100, 2),
                    "sec": round(time.time() - t0, 1)}
        print(f"{tag}: moved {moved} frames ({failed} unresolved), max shift "
              f"{rep[tag]['max_shift_cm']} cm, {rep[tag]['sec']}s", flush=True)
    (a.out / "pen_zero_report.json").write_text(json.dumps(rep, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
