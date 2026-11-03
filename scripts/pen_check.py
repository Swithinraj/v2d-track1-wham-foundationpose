#!/usr/bin/env python3
"""PEN safety check for object-track corrections (submission venv: numpy + trimesh + numba).

Did the corrected object pose push the object INTO the human's hands? The scorer's PEN counts
the submission's hand-mesh points inside the object; this reproduces that with the kit's own
penetration kernels (v2dlb.mhr_submission._PENETRATION_KERNELS, the numba fast path) on the
BUDGETED object mesh — the same mesh the packer writes and the scorer tests.

Inputs: --base and --cand episode dirs (npz + *_object.glb; cand = depth-bias-corrected) and
--hands (mhr_bodies.py --hands-only output for the SAME input episodes: the human is identical
in base and cand, only the object pose moved). Per sampled scored frame: transform the mesh by
each pose (X = s (R V) + t), count hand points with winding number > 0.5 (inside) and the
closest hand-point distance to the surface.

Writes a report JSON: hand-inside counts and min distances before/after per episode. A safe
correction keeps 'inside' at (or below) the base count.
Usage: pen_check.py --base DIR --cand DIR --hands DIR [--out FILE] [--frames K]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(os.environ.get("TRACK1_ROOT") or Path(__file__).resolve().parents[1])
sys.path.insert(0, str(ROOT / "official/v2d_submission_kit"))
from v2dlb import mhr_submission as ms  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--cand", type=Path, required=True)
    ap.add_argument("--hands", type=Path, required=True, help="mhr_bodies.py --hands-only dir")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/depth_icp/pen_check.json")
    ap.add_argument("--episodes-list", type=int, nargs="*", default=list(range(30)))
    ap.add_argument("--frames", type=int, default=48)
    a = ap.parse_args()
    rep = {}
    for ep in a.episodes_list:
        tag = f"episode_{ep:06d}"
        fb, fc = a.base / f"{tag}.npz", a.cand / f"{tag}.npz"
        if not (fb.exists() and fc.exists()):
            continue
        db, dc = dict(np.load(fb)), dict(np.load(fc))
        if np.allclose(db["object_translation"], dc["object_translation"]):
            print(f"{tag}: candidate == base, skipped", flush=True)
            continue
        f_hands = a.hands / f"{tag}.npz"
        if not f_hands.exists():
            print(f"{tag}: no hands npz, skipped", flush=True)
            continue
        hands = np.load(f_hands)["hands"].astype(np.float64)          # [T, 4636, 3] metres
        mesh_p = a.base / f"{tag}_object.glb"
        if not mesh_p.exists():
            mesh_p = a.base / f"{tag}_object.ply"                     # unpacked-candidate layout
        mesh = trimesh.load(mesh_p, force="mesh")
        V = np.asarray(mesh.vertices, np.float64)
        F = np.asarray(mesh.faces)
        v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        s = float(np.asarray(db["object_scale"]).reshape(()))
        Rb = db["object_rotation"].astype(np.float64)
        tb = db["object_translation"].astype(np.float64)
        Rc = dc["object_rotation"].astype(np.float64)
        tc = dc["object_translation"].astype(np.float64)
        scored = np.where(np.isfinite(tb).all(axis=1) & np.isfinite(tc).all(axis=1))[0]
        sub = scored[np.linspace(0, len(scored) - 1, min(a.frames, len(scored))).astype(int)]
        pts = hands[sub][:, ::4]                                      # subsample hand points
        stats = {"b": [0, []], "c": [0, []]}                          # inside count, min distances
        for i in range(len(sub)):
            for key, (R, t) in (("b", (Rb[sub[i]], tb[sub[i]])), ("c", (Rc[sub[i]], tc[sub[i]]))):
                X = s * (V @ R.T) + t
                v0w, v1w, v2w = X[F[:, 0]], X[F[:, 1]], X[F[:, 2]]
                if ms._PENETRATION_KERNELS is not None:
                    w, d = ms._PENETRATION_KERNELS
                    win = w(pts[i], v0w, v1w, v2w)
                    dist = d(pts[i], v0w, v1w, v2w)
                else:
                    raise RuntimeError("numba penetration kernels unavailable — run in .venvs/submission")
                inside = int((win > 0.5).sum())
                stats[key][0] += inside
                stats[key][1].append(float(dist.min()))
        rep[tag] = {"frames": int(len(sub)),
                    "hand_inside_before": stats["b"][0], "hand_inside_after": stats["c"][0],
                    "min_dist_cm_before": round(min(stats["b"][1]) * 100, 3),
                    "min_dist_cm_after": round(min(stats["c"][1]) * 100, 3)}
        print(f"{tag}: inside {rep[tag]['hand_inside_before']} -> {rep[tag]['hand_inside_after']} | min dist "
              f"{rep[tag]['min_dist_cm_before']} -> {rep[tag]['min_dist_cm_after']} cm", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
