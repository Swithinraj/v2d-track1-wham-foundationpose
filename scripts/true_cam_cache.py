#!/usr/bin/env python3
"""Re-express GRU training caches (gru_train.py / gru_cache_export.py format) in the measured camera.

The caches hold, per episode, joint trajectories in the old (MoGe-2) camera frame: ours [T,22,3],
raw [T,22,3], gt [T,22,3] (the evaluator-frame multi-view GT), plus conf / vis / valid / anchor,
which are image-derived and unchanged. For a point p the camera change is the same linear map as
scripts/true_cam_params.py (K_new A p = (f_n/f_o) K_old p: every pixel stays, only its 3D
interpretation moves), so the trajectories are mapped by A directly; the band features and losses
are recomputed from the trajectories at load time, nothing else in the cache depends on the camera.

gt is mapped with the same A. That is exact for ours / raw and an approximation for gt (the
evaluator fits its Sim(3) in the old camera); the anchor-frame Procrustes residual between gt and
ours is reported before and after so the approximation is visible. This script only prepares
training data; actually retraining the motion head needs the owner's approval.

Usage: true_cam_cache.py --cache DIR --out DIR --old DIR --k_new FILE [--prefix GROUP]
  --cache   dir with [<group>_]episode_XXXXXX.npz caches
  --prefix  only process cache files named <prefix>_episode_*.npz (a multi-group cache dir has
            per-group camera metadata, so re-express one group at a time; other files are copied)
  --old     outputs/camera-style dir with <episode>/intrinsics.json (the old camera, per episode)
  --k_new   JSON {episode: {fx, cx, cy}} (the measured camera, per episode)
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--old", type=Path, required=True)
    ap.add_argument("--k_new", type=Path, required=True)
    ap.add_argument("--prefix", default="")
    args = ap.parse_args()
    kn = json.loads(args.k_new.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    rep = {}
    files = sorted(args.cache.glob(f"{args.prefix + '_' if args.prefix else ''}*episode_??????.npz"))
    for f in files:
        tag = f.stem.split("episode_")[-1]
        ep = f"episode_{tag}"
        d = dict(np.load(f))
        if ep not in kn or not (args.old / ep / "intrinsics.json").exists():
            shutil.copy2(f, args.out / f.name)
            print(f"{f.name}: no new camera, copied unchanged", flush=True)
            continue
        o = json.loads((args.old / ep / "intrinsics.json").read_text())
        n = kn[ep]
        fo, cox, coy = float(o["fx"]), float(o["cx"]), float(o["cy"])
        fn, cnx, cny = float(n["fx"]), float(n["cx"]), float(n["cy"])
        A = np.array([[1, 0, (cox - cnx) / fo], [0, 1, (coy - cny) / fo], [0, 0, fn / fo]])
        a = int(d["anchor"])

        def procrustes(x, y):
            """anchor-frame similarity residual (cm) between two trajectories, as the evaluator fits it."""
            xa, ya = x[a][None] - x[a], y[a][None] - y[a]     # anchor-relative
            xa, ya = xa - xa.mean(0), ya - ya.mean(0)
            u, _, vt = np.linalg.svd(xa.T @ ya)
            s = np.sign(np.linalg.det(u @ vt))
            r = u @ np.diag([1, 1, s]) @ vt
            sc = (ya ** 2).sum() / max((xa ** 2).sum(), 1e-12)
            return float(np.sqrt(((sc * (xa @ r.T) - ya) ** 2).sum(-1).mean()) * 100)

        before = procrustes(np.asarray(d["ours"], float), np.asarray(d["gt"], float))
        for k in ("ours", "raw", "gt"):
            d[k] = (np.asarray(d[k], float) @ A.T).astype(np.float32)
        after = procrustes(np.asarray(d["ours"], float), np.asarray(d["gt"], float))
        np.savez(args.out / f.name, **d)
        shift = float(np.median(np.linalg.norm(np.asarray(d["ours"]) - np.load(f)["ours"], axis=-1)) * 100)
        rep[f.name] = {"anchor_resid_cm_before": before, "anchor_resid_cm_after": after,
                       "shift_cm_median": shift}
        print(f"{f.name}: shift {shift:.2f} cm, anchor resid {before:.2f} -> {after:.2f} cm", flush=True)
    (args.out / f"true_cam_cache_{args.prefix or 'all'}.json").write_text(json.dumps(rep, indent=1))
    b = np.median([r["anchor_resid_cm_before"] for r in rep.values()])
    a_ = np.median([r["anchor_resid_cm_after"] for r in rep.values()])
    print(f"PASS: {len(rep)} caches -> {args.out}; anchor residual median {b:.2f} -> {a_:.2f} cm")


if __name__ == "__main__":
    main()
