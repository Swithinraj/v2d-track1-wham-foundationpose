#!/usr/bin/env python3
"""Floor plane from the human's soles + resting-object gravity gap, on recovered episodes (CPU).

Companion to unpack_submission.py: with outputs/ and submissions/ lost, the packed Kaggle
parquet is the only surviving candidate state. It recovers the human (MHR pose/scales/shape)
and the object (mesh + per-frame Sim(3)) but NOT the MoGe floor planes (kappa_floor.json), so
this script re-estimates the floor from the body itself: the person's lowest vertices (soles)
trace the floor over a session, so a plane fitted to them per episode plays kappa_floor's role.

Per episode: MHR forward on K subsampled scored frames -> vertices (mm -> m, the pipeline frame;
the same convention kappa_gravity.py uses); robust plane fit to the per-frame lowest vertices
(seeded from coordinate/PCA axes, 3 refinement iterations); then, on every scored frame, the
object hull's signed height over the plane with physics_objects.py's FLOOR_NEAR/FAR weights and
camera-ray snap step (ray = translation/|translation|, camera at origin; MAX_SNAP_M cap).

--apply writes corrected episodes (object_translation += w * step * ray on floor frames) for
repacking: the post-hoc gravity A/B without a pipeline re-run.

Usage: floor_gap_soles.py --input DIR [--out FILE] [--frames K] [--apply DIR]
  --input  recovered episodes dir (episode_XXXXXX.npz from unpack_submission.py)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
T1 = Path("/track1")
sys_path = str(ROOT / "official/v2d_submission_kit/tools/track1")
import sys  # noqa: E402

sys.path.insert(0, sys_path)
from mesh_to_mhr_params import MHR  # noqa: E402

MODEL = ROOT / "weights/mhr/assets/mhr_model.pt"
FLOOR_NEAR_M, FLOOR_FAR_M, MAX_SNAP_M = 0.06, 0.12, 0.25   # physics_objects.py's constants
SOLE_BAND_M = 0.015                                        # a vertex this close to the frame's lowest counts as sole


def fit_plane(points):
    """Total-least-squares plane through points; returns (unit normal, c) with n.x = c."""
    c = points.mean(0)
    _, _, vt = np.linalg.svd(points - c, full_matrices=False)
    n = vt[-1]
    return n / np.linalg.norm(n), float(n @ c)


def floor_plane(verts_m, iters=3):
    """verts_m [K, V, 3] metres. Seed from candidate axes/PCA, refine on the soles."""
    V = verts_m.reshape(-1, 3)
    _, _, vt = np.linalg.svd(V - V.mean(0), full_matrices=False)
    cands = [np.eye(3)[i] for i in range(3)] + [vt[i] for i in range(3)]
    best = None
    for n in cands:
        h = verts_m @ n
        low = h.min(axis=1, keepdims=True)
        sole = verts_m[(h - low) < SOLE_BAND_M]
        n2, c2 = fit_plane(sole)
        res = float(np.abs(sole @ n2 - c2).std())
        if best is None or res < best[0]:
            best = (res, n2, c2)
    res, n, c = best
    for _ in range(iters):
        h = verts_m @ n
        low = h.min(axis=1, keepdims=True)
        sole = verts_m[(h - low) < SOLE_BAND_M]
        n, c = fit_plane(sole)
        res = float(np.abs(sole @ n - c).std())
    return n, c, res, len(sole)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/physics/floor_gap_soles.json")
    ap.add_argument("--frames", type=int, default=48, help="frames subsampled for the plane fit")
    ap.add_argument("--apply", type=Path, help="write gravity-corrected episodes here")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mhr = MHR(str(MODEL), dev, chunk=64, precision="float32")
    if a.apply:
        a.apply.mkdir(parents=True, exist_ok=True)
    rep = {}
    for f in sorted(a.input.glob("episode_??????.npz")):
        d = dict(np.load(f))
        scored = np.where(np.isfinite(d["object_rotation"]).all(axis=(1, 2)))[0]
        if len(scored) < 5:
            print(f"{f.stem}: <5 scored frames, skipped", flush=True)
            continue
        sub = scored[np.linspace(0, len(scored) - 1, min(a.frames, len(scored))).astype(int)]
        ident = torch.as_tensor(np.concatenate([d["mhr_scales"], d["mhr_shape"]])[None])
        with torch.no_grad():
            v_mm, _ = mhr.run(torch.as_tensor(d["pose"][sub], dtype=torch.float32),
                              ident.expand(len(sub), -1))
        verts_m = v_mm.reshape(len(sub), -1, 3).double().numpy() / 1000.0   # mm -> m (kappa_gravity's convention)
        n, c, res, n_sole = floor_plane(verts_m)
        V = np.asarray(d["object_mesh_vertices"], float)
        s = float(np.asarray(d["object_scale"]).reshape(()))
        t = d["object_translation"].astype(np.float64)
        R = d["object_rotation"].astype(np.float64)
        low = (s * np.einsum("tij,vj->tvi", R, V) @ n).min(axis=1)
        height = low + t @ n - c
        w = np.clip((FLOOR_FAR_M - np.abs(height)) / (FLOOR_FAR_M - FLOOR_NEAR_M), 0.0, 1.0)
        w = np.array([np.median(w[max(0, i - 7):i + 8]) for i in range(len(w))])   # physics' 15-frame median
        on = w > 0.5
        ray = t / np.linalg.norm(t, axis=1, keepdims=True)
        nr = ray @ n
        step = np.where(np.abs(nr) > 0.05, -height / np.where(np.abs(nr) > 0.05, nr, 1.0), 0.0)
        step = np.clip(step, -MAX_SNAP_M, MAX_SNAP_M)
        anchor = scored[0]
        r = {"episode": f.stem, "scored_frames": int(len(scored)),
             "plane_normal": [round(float(x), 4) for x in n], "plane_c_m": round(c, 4),
             "plane_residual_cm": round(res * 100, 2), "sole_points": int(n_sole),
             "floor_share": float(on.mean()),
             "height_cm_median": float(np.median(height[on]) * 100) if on.any() else None,
             "gap_cm_median": float(np.median(np.abs(height[on])) * 100) if on.any() else None,
             "gap_cm_p90": float(np.percentile(np.abs(height[on]), 90) * 100) if on.any() else None,
             "snap_cm_median": float(np.median(np.abs(step[on])) * 100) if on.any() else None,
             "snap_cm_max": float(np.abs(step[on]).max() * 100) if on.any() else None,
             "snap_capped_share": float((np.abs(step[on]) >= MAX_SNAP_M - 1e-9).mean()) if on.any() else None,
             "height_cm_anchor": float(height[anchor] * 100), "snap_cm_anchor": float(step[anchor] * 100)}
        rep[f.stem] = r
        print(f"{f.stem}: floor {r['floor_share']*100:3.0f} % | gap {r['gap_cm_median'] or -1:5.1f} cm"
              f" (signed {r['height_cm_median'] or -1:+5.1f}) | snap {r['snap_cm_median'] or -1:4.1f} cm"
              f" | anchor f{anchor}: {r['height_cm_anchor']:+5.1f} cm, snap {r['snap_cm_anchor']:+5.1f} cm"
              f" | plane n~{np.argmax(np.abs(n))} res {r['plane_residual_cm']:.2f} cm", flush=True)
        if a.apply:
            t2 = t.copy()
            t2[on] += (w[on] * step[on])[:, None] * ray[on]
            d2 = dict(d)
            d2["object_translation"] = t2.astype(np.float32 if d["object_translation"].dtype == np.float32 else t2.dtype)
            np.savez(a.apply / f.name, **d2)
            for extra in a.input.glob(f"{f.stem}_object*"):
                import shutil
                shutil.copy2(extra, a.apply / extra.name)
    on_all = [r for r in rep.values() if r["gap_cm_median"] is not None and r["floor_share"] > 0.05]
    if on_all:
        print(f"\nPASS: {len(on_all)}/{len(rep)} episodes with floor contact; "
              f"|gap| median {np.median([r['gap_cm_median'] for r in on_all]):.1f} cm "
              f"(signed {np.median([r['height_cm_median'] for r in on_all]):+.1f}); "
              f"anchor-frame height median {np.median([r['height_cm_anchor'] for r in on_all]):+.1f} cm")
        print("verdict: compare with the documented 8-14 cm limit; if |gap| < 3 cm the gravity A/B is dead")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=1))
    print(f"report -> {a.out}" + (f"; corrected episodes -> {a.apply}" if a.apply else ""))


if __name__ == "__main__":
    main()
