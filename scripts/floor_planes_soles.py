#!/usr/bin/env python3
"""Floor planes for human_refit's --feet contract: MoGe-depth RANSAC + sole diagnostics (mhr_converter container).

The floor target for feet placement, estimated WITHOUT the diverging sole-band fit
(floor_gap_soles.py's blind refinement stopped mid-flight on our fresh humans: the band is
selected along the previous normal, the refit normal rotates, and after a fixed 3 iterations
the returned plane was sometimes 1.6 m off the body). This script uses depth_cues.py's
moge_floor_normal recipe instead, which cannot diverge:

  1. MHR forward on K subsampled frames -> vertices (MILLIMETRES, sole diagnostics) and
     joints (METRES -- mhr.run returns V in mm but J in m; mind the units) for body-up
     (head SC[15]=113 minus mean ankles SC[7],SC[8]=[4,20]) and the scored 22 joints via SC.
  2. MoGe-2 depth (65535/v - 1 m, decoded / scale = our frame) on the same frames, grid-
     subsampled, unprojected, body pixels excluded (0.35 m of any scored joint).
  3. Plane RANSAC (600 iters, 3 cm inliers) within 25 deg of body-up, requiring the body's
     joints to sit above the plane; refit on the inliers; c = n . centroid.
  4. Convert to human_refit's contract: MoGe metres, c_moge = c_our * depth_scale
     (human_refit divides by the alignment depth_scale again; kappa defaults to 1 without a
     kappa report, correct for the fresh chain's objects).

plant_feet gates per foot (near the floor AND still for a run), so every episode with a
confident depth floor is emitted; episodes without (few inliers) are left out — no floor, no
planting. The report's sole gap (lowest vertex above the plane) previews the placement error
the feet term will correct.

Writes OUT (default outputs/physics/floor_planes_soles.json) + OUT.report.json.
Usage: floor_planes_soles.py --episodes DIR [--out FILE] [--frames K]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(os.environ.get("TRACK1_ROOT") or Path(__file__).resolve().parents[1])
sys.path.insert(0, str(ROOT / "official/v2d_submission_kit/tools/track1"))
from mesh_to_mhr_params import MHR  # noqa: E402

MODEL = ROOT / "weights/mhr/assets/mhr_model.pt"
# scored-joint index map (joint_deploy.py's SC) and depth_cues.py's roles within it
SC = [1, 2, 18, 35, 3, 19, 36, 4, 20, 37, 8, 24, 110, 74, 38, 113, 75, 39, 76, 40, 78, 42]
ANK, HEAD = (7, 8), 15                                   # scored indices
MIN_INLIERS, RANSAC_ITERS, INLIER_M = 2000, 600, 0.03    # moge_floor_normal's recipe


def decode_depth(png, scale):
    v = np.asarray(Image.open(png)).astype(np.float64)
    if v.ndim == 3:
        v = v[..., 0]
    return (65535.0 / np.maximum(v, 1.0) - 1.0) / scale   # MoGe metres -> our frame


def floor_points(depth_dir, frames, scale, K, J_sc):
    """Unprojected depth points of the given frames, body pixels excluded."""
    fx, fy, cx, cy = K["fx"], K["fy"], K["cx"], K["cy"]
    pts = []
    for k, f in enumerate(frames):
        dp = depth_dir / f"{f:06d}.png"
        if not dp.exists():
            continue
        D = decode_depth(dp, scale)
        step = 6
        z = D[::step, ::step]
        vv, uu = np.mgrid[0:z.shape[0], 0:z.shape[1]] * step   # true pixel coords of the subsampled grid
        ok = (z > 0.3) & (z < 12.0)
        X = np.stack([(uu - cx) / fx * z, (vv - cy) / fy * z, z], -1)[ok]
        if len(X) == 0:
            continue
        far = np.min(np.linalg.norm(X[:, None] - J_sc[k][None], axis=-1), axis=1) > 0.35
        pts.append(X[far])
    return np.concatenate(pts) if pts else np.empty((0, 3))


def ransac_floor(P, up, rng, J_all):
    """Largest plane within 25 deg of up, body above it; refit on inliers.

    J_all: all frames' scored joints (metres) for the body-above check — without it the
    largest plane in the scene can be a CEILING (episodes 25/27 fitted a plane ~2 m above
    the body before this check existed).
    """
    best, bn, bc = 0, None, 0.0
    for _ in range(RANSAC_ITERS):
        a, b, c3 = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c3 - a)
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn
        if n @ up < 0:
            n = -n
        if n @ up < np.cos(np.radians(25.0)):
            continue
        c = float(n @ a)
        if float(np.median(J_all @ n - c)) < 0.3:             # the body must stand ON the plane
            continue
        inl = np.abs(P @ n - c) < INLIER_M
        if inl.sum() > best:
            best, bn, bc = int(inl.sum()), n, c
    if bn is None or best < MIN_INLIERS:
        return None, 0
    Q = P[np.abs(P @ bn - bc) < INLIER_M]
    c0 = Q.mean(0)
    _, _, vt = np.linalg.svd(Q - c0, full_matrices=False)
    n = vt[-1]
    if n @ up < 0:
        n = -n
    return (n, float(n @ c0)), best


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--episodes", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/physics/floor_planes_soles.json")
    ap.add_argument("--frames", type=int, default=24, help="frames subsampled for MHR + depth")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mhr = MHR(str(MODEL), dev, chunk=64, precision="float32")
    floors, rep = {}, {}
    for f in sorted(a.episodes.glob("episode_??????.npz")):
        tag = f.stem
        d = dict(np.load(f))
        pose = d["pose"].astype(np.float32)
        finite = np.flatnonzero(np.isfinite(pose).all(axis=1))
        if len(finite) < 5:
            print(f"{tag}: <5 finite frames, skipped", flush=True)
            continue
        frames = finite[np.linspace(0, len(finite) - 1, min(a.frames, len(finite))).astype(int)]
        ident = torch.as_tensor(np.concatenate([d["scales"], d["shape"]])[None], dtype=torch.float32)
        with torch.no_grad():
            v_mm, j_m = mhr.run(torch.as_tensor(pose[frames]), ident.expand(len(frames), -1))
        verts = v_mm.reshape(len(frames), -1, 3).double().numpy() / 1000.0   # mm -> m
        J_sc = j_m[:, SC, :].double().numpy()                                # metres already
        up = (J_sc[:, HEAD] - J_sc[:, list(ANK)].mean(1)).mean(0)
        up /= np.linalg.norm(up)
        align_p = ROOT / "outputs/alignment_sam" / tag / "alignment.json"
        scale = float(json.loads(align_p.read_text())["depth_scale"]) if align_p.exists() else 1.0
        K = json.loads((ROOT / "outputs/camera" / tag / "intrinsics.json").read_text())
        P = floor_points(ROOT / "outputs/object" / tag / "depth", frames, scale, K, J_sc)
        rng = np.random.default_rng(int(tag.split("_")[-1]))
        plane, n_inl = ransac_floor(P, up, rng, J_sc.reshape(-1, 3))
        if plane is None:
            print(f"{tag}: no depth floor ({len(P)} pts, best < {MIN_INLIERS} inliers) — excluded", flush=True)
            rep[tag] = {"inliers": n_inl, "points": int(len(P)), "excluded": True}
            continue
        n, c = plane
        ang = float(np.degrees(np.arccos(np.clip(n @ up, -1, 1))))
        h = verts @ n - c                                    # sole heights above the plane (m)
        low = h.min(axis=1)
        if float(np.median(J_sc.reshape(-1, 3) @ n - c)) < 0.3 or float(np.median(low)) < -0.10:
            print(f"{tag}: refit plane above/too far below the body — excluded", flush=True)
            rep[tag] = {"inliers": int(n_inl), "points": int(len(P)), "excluded": True}
            continue
        floors[tag] = {"normal": [round(float(x), 6) for x in n], "c": round(c * scale, 6)}
        rep[tag] = {"inliers": int(n_inl), "points": int(len(P)), "excluded": False,
                    "angle_from_up_deg": round(ang, 1), "depth_scale": scale,
                    "sole_gap_cm_median": round(float(np.median(low) * 100), 2),
                    "sole_gap_cm_p90": round(float(np.percentile(low, 90) * 100), 2),
                    "body_above_m": round(float(np.median(J_sc.reshape(-1, 3) @ n - c)), 3)}
        print(f"{tag}: inl {n_inl:6d}/{len(P):7d} | up {ang:4.1f} deg | sole gap "
              f"{rep[tag]['sole_gap_cm_median']:+6.2f} cm (p90 {rep[tag]['sole_gap_cm_p90']:+6.2f}) "
              f"| body above {rep[tag]['body_above_m']:+.2f} m", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(floors, indent=1))
    a.out.with_suffix(".report.json").write_text(json.dumps(rep, indent=1))
    print(f"wrote {a.out} ({len(floors)} episodes with a depth floor) + report")


if __name__ == "__main__":
    main()
