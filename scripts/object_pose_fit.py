#!/usr/bin/env python3
"""Per-frame object pose refinement against the image evidence (v2d-host venv, CPU).

The audit shows tracking errors while objects are handled (pan lifted off the table, stool 25 carried and tilted,
hoop 0 picked up): the posed mesh leaves its SAM 2 mask. Per frame, a small 6-DoF correction (rotation about the
object's centre + translation) is fitted by robust least squares at half resolution:
  A  projected surface samples outside the mask: distance to the mask (0 inside; 0 where the person mask covers the
     image - the object may be hidden behind the person there - and outside the image),
  B  mask pixels not explained: distance from mask samples to the nearest projected surface sample,
  priors: depth of the object centre stays within ~2 cm, rotation small, translation weakly tied.
A correction is kept only where the rendered silhouette IoU improves by > ACCEPT; kept corrections are blended over
time (Gaussian, sigma SIGMA frames, normalised by the local share of kept frames, so isolated corrections shrink), and
the object track is rebuilt (scale unchanged). Writes <out>/episode_*.npz (+ meshes, other files copied) and
objfit_report.json (IoU on scored frames and object acceleration before/after, per episode).
Usage: object_pose_fit.py --input DIR --out DIR [--episodes ...] [--rh_masks] [--accept iou|chamfer] [--workers N]
  OBJFIT_SIGMA (default 4): temporal blend sigma in frames
"""
import argparse
import json
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import trimesh
from PIL import Image
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

import os  # noqa: E402
ROOT = Path(os.environ.get("TRACK1_ROOT") or Path(__file__).resolve().parents[1])
R0 = ROOT
sys.path.insert(0, str(ROOT / "scripts"))
import resolve_hand_penetration as rp  # noqa: E402
SCORED = {str(k): v for k, v in rp.scored_frames_and_budget()[0].items()}
ACCEPT, SIGMA, DOWN = 0.02, float(__import__("os").environ.get("OBJFIT_SIGMA", 4.0)), 2
ARGS = None


def mask_path(tag, f):
    if getattr(ARGS, "seg2", False) and (R0 / f"outputs/seg2/{tag}/objfused/1").is_dir():
        return R0 / f"outputs/seg2/{tag}/objfused/1/{f:06d}.png"
    if ARGS.rh_masks and (R0 / f"outputs/object_rh/{tag}/sam2_masks/1").is_dir():
        return R0 / f"outputs/object_rh/{tag}/sam2_masks/1/{f:06d}.png"
    return R0 / f"outputs/object/{tag}/sam2_masks/1/{f:06d}.png"


def load(p, down):
    return np.array(Image.open(p))[::down, ::down] > 127 if p.exists() else None


def project(X, K):
    return (X @ K.T)[:, :2] / X[:, 2:3]


def render_iou(V, F, K, mask):
    """Silhouette IoU with the mesh rasterised by splatting dense surface samples V (3x3 dilation, 1/4 resolution);
    cv2.fillPoly over 40k overlapping triangles takes ~16 s per frame, the splat a few ms."""
    H, W = mask.shape
    if (V[:, 2] <= 0.05).any():
        return 0.0
    uv = np.round(project(V, K)).astype(int)
    ok = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    m = np.zeros((H, W), np.uint8); m[uv[ok, 1], uv[ok, 0]] = 1
    m = cv2.dilate(m, np.ones((3, 3), np.uint8)) > 0; u = (m | mask).sum()
    return float((m & mask).sum() / u) if u else 0.0


def chamfer2d(V, K, mask, occl):
    """Symmetric 2-D distance (px) between projected surface samples and visible mask pixels (thin structures)."""
    H, W = mask.shape
    if (V[:, 2] <= 0.05).any():
        return np.inf
    uv = project(V, K); ok = (uv[:, 0] >= 0) & (uv[:, 0] < W - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < H - 1)
    ui = uv[ok]
    if len(ui) < 10:
        return np.inf
    vis = ~occl[np.round(ui[:, 1]).astype(int), np.round(ui[:, 0]).astype(int)]
    dt_out = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3)
    a = map_coordinates(dt_out, [ui[vis, 1], ui[vis, 0]], order=1).mean() if vis.any() else 0.0
    ys, xs = np.nonzero(mask)
    sel = np.random.default_rng(1).choice(len(xs), min(600, len(xs)), replace=False)
    b = cKDTree(ui).query(np.stack([xs[sel], ys[sel]], 1))[0].mean()
    return float(a + b)


def fit_frame(Pw, c, mask, occl, K):
    """Pw: posed surface samples (camera frame), c: object centre. Returns (rotvec, dt)."""
    H, W = mask.shape
    dt_out = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3).astype(np.float32)
    ys, xs = np.nonzero(mask)
    sel = np.random.default_rng(0).choice(len(xs), min(400, len(xs)), replace=False)
    msamp = np.stack([xs[sel], ys[sel]], 1).astype(float)
    z0 = c[2]

    def res(x):
        Rr = Rotation.from_rotvec(x[:3]).as_matrix()
        X = (Pw - c) @ Rr.T + c + x[3:]
        if (X[:, 2] <= 0.05).any():
            return np.full(len(Pw) + len(msamp) + 7, 1e3)
        u = project(X, K)
        inside = (u[:, 0] >= 0) & (u[:, 0] < W - 1) & (u[:, 1] >= 0) & (u[:, 1] < H - 1)
        a = np.zeros(len(u))
        ui = u[inside]
        da = map_coordinates(dt_out, [ui[:, 1], ui[:, 0]], order=1)
        hidden = occl[np.round(ui[:, 1]).astype(int), np.round(ui[:, 0]).astype(int)]
        a[inside] = np.where(hidden, 0.0, da)
        b = cKDTree(u).query(msamp)[0]
        prior = np.concatenate([x[:3] / 0.15, x[3:5] / 0.30, [(x[5]) / 0.02], [0.0]]) * 3.0
        return np.concatenate([a / np.sqrt(len(a)) * 4, b / np.sqrt(len(b)) * 4, prior])

    r = least_squares(res, np.zeros(6), loss="huber", f_scale=1.0, max_nfev=80, diff_step=1e-3)
    return r.x[:3], r.x[3:]


def episode(path):
    tag = path.stem; ep = int(tag.split("_")[1])
    d = dict(np.load(path)); mesh = trimesh.load(path.parent / f"{tag}_object.glb", force="mesh")
    V0 = np.asarray(trimesh.sample.sample_surface(mesh, 30000, seed=2)[0], float); F = None   # dense samples for the IoU splat
    pc = (np.asarray(mesh.vertices).min(0) + np.asarray(mesh.vertices).max(0)) / 2
    P = np.asarray(trimesh.sample.sample_surface(mesh, 1500, seed=0)[0], float)
    Kj = json.loads((R0 / "outputs/camera" / tag / "intrinsics.json").read_text())
    K2 = np.array([[Kj["fx"] / DOWN, 0, Kj["cx"] / DOWN], [0, Kj["fy"] / DOWN, Kj["cy"] / DOWN], [0, 0, 1]])
    K4 = np.array([[Kj["fx"] / 4, 0, Kj["cx"] / 4], [0, Kj["fy"] / 4, Kj["cy"] / 4], [0, 0, 1]])
    Rm, t, s = d["object_rotation"].astype(float), d["object_translation"].astype(float), float(d["object_scale"])
    T = len(t)
    rv, dtr, w = np.zeros((T, 3)), np.zeros((T, 3)), np.zeros(T)
    gains = []
    for f in range(T):
        mask = load(mask_path(tag, f), DOWN)
        if mask is None or mask.sum() < 30:
            continue
        pm = load((R0 / f"outputs/seg2/{tag}/person/0/{f:06d}.png") if getattr(ARGS, "seg2", False) and (R0 / f"outputs/seg2/{tag}/person/0").is_dir()
                  else (R0 / f"outputs/person/{tag}/sam2_masks/0/{f:06d}.png"), DOWN)
        occl = cv2.dilate(pm.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0 if pm is not None else np.zeros_like(mask)
        c = s * Rm[f] @ pc + t[f]
        Pw = s * P @ Rm[f].T + t[f]
        x_r, x_t = fit_frame(Pw, c, mask, occl, K2)
        m4 = mask[::2, ::2]
        Vw = s * V0 @ Rm[f].T + t[f]
        Rr = Rotation.from_rotvec(x_r).as_matrix()
        Vn = (Vw - c) @ Rr.T + c + x_t
        if ARGS.accept == "chamfer":       # thin structures (hoop): IoU is insensitive, use the 2-D chamfer
            cb = chamfer2d((P * s) @ Rm[f].T + t[f], K2, mask, occl)
            ca = chamfer2d((((P * s) @ Rm[f].T + t[f]) - c) @ Rr.T + c + x_t, K2, mask, occl)
            before, after = -cb, -ca
            good = ca < 0.9 * cb
        else:
            before, after = render_iou(Vw, F, K4, m4), render_iou(Vn, F, K4, m4)
            good = after - before > ACCEPT
        if good:
            rv[f], dtr[f], w[f] = x_r, x_t, 1.0
            gains.append(after - before)
    # blend kept corrections over time (isolated corrections shrink)
    r = int(3 * SIGMA); k = np.exp(-0.5 * (np.arange(-r, r + 1) / SIGMA) ** 2); k /= k.sum()
    conv = lambda x: np.convolve(np.pad(x, r, mode="edge"), k, mode="valid")  # noqa: E731
    den = np.maximum(conv(w), 1e-9); share = np.minimum(den, 1.0)
    rv_s = np.stack([conv(w * rv[:, i]) / den * share for i in range(3)], 1)
    dt_s = np.stack([conv(w * dtr[:, i]) / den * share for i in range(3)], 1)
    Rn, tn = Rm.copy(), t.copy()
    for f in range(T):
        Rr = Rotation.from_rotvec(rv_s[f]).as_matrix(); c = s * Rm[f] @ pc + t[f]
        Rn[f] = Rr @ Rm[f]; tn[f] = Rr @ (t[f] - c) + c + dt_s[f]
    d["object_rotation"], d["object_translation"] = Rn.astype(np.float32), tn.astype(np.float32)
    np.savez(ARGS.out / f"{tag}.npz", **d)
    for g in path.parent.glob(f"{tag}_object*"):
        shutil.copy2(g, ARGS.out / g.name)
    sf = [f for f in SCORED[str(ep)] if f < T]
    iou_b, iou_a = [], []
    for f in sf[::4]:
        m4 = load(mask_path(tag, f), 4)
        if m4 is None or m4.sum() < 5:
            continue
        iou_b.append(render_iou(s * V0 @ Rm[f].T + t[f], F, K4, m4)); iou_a.append(render_iou(s * V0 @ Rn[f].T + tn[f], F, K4, m4))
    sfa = np.array(sf); inner = sfa[(sfa >= 1) & (sfa < T - 1)]
    acc = lambda x: float(np.linalg.norm(x[inner - 1] - 2 * x[inner] + x[inner + 1], axis=1).mean() * 100)  # noqa: E731
    return {"episode": tag, "kept_frames": int(w.sum()), "frames": T, "mean_gain_kept": float(np.mean(gains)) if gains else 0.0,
            "iou_scored": [float(np.median(iou_b)) if iou_b else 0.0, float(np.median(iou_a)) if iou_a else 0.0],
            "iou_scored_p10": [float(np.percentile(iou_b, 10)) if iou_b else 0.0, float(np.percentile(iou_a, 10)) if iou_a else 0.0],
            "object_acc_cm": [acc(t), acc(tn)], "max_shift_cm": float(np.linalg.norm(tn - t, axis=1).max() * 100),
            "max_rot_deg": float(np.degrees(np.linalg.norm(rv_s, axis=1).max()))}


def main():
    global ARGS
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--episodes", type=int, nargs="*", default=None)
    ap.add_argument("--rh_masks", action="store_true", help="use outputs/object_rh masks where they exist (roller)")
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--accept", choices=("iou", "chamfer"), default="iou", help="chamfer for thin structures (hoop)")
    ap.add_argument("--seg2", action="store_true", help="joint exclusive person/object masks from seg2_track.py")
    ARGS = ap.parse_args()
    ARGS.out.mkdir(parents=True, exist_ok=True)
    for f in ARGS.input.iterdir():
        if f.is_file():
            shutil.copy2(f, ARGS.out / f.name)
    paths = [p for p in sorted(ARGS.input.glob("episode_??????.npz")) if ARGS.episodes is None or int(p.stem[-6:]) in ARGS.episodes]
    with ProcessPoolExecutor(ARGS.workers) as pool:
        rep = list(pool.map(episode, paths))
    for r in rep:
        print(f"{r['episode']}: kept {r['kept_frames']}/{r['frames']} frames, IoU(scored) {r['iou_scored'][0]:.3f} -> {r['iou_scored'][1]:.3f} "
              f"(p10 {r['iou_scored_p10'][0]:.3f} -> {r['iou_scored_p10'][1]:.3f}), object acc {r['object_acc_cm'][0]:.4f} -> "
              f"{r['object_acc_cm'][1]:.4f}, max shift {r['max_shift_cm']:.1f} cm, max rot {r['max_rot_deg']:.1f} deg", flush=True)
    mean = {k: [float(np.mean([r[k][0] for r in rep])), float(np.mean([r[k][1] for r in rep]))] for k in ("iou_scored", "iou_scored_p10", "object_acc_cm")}
    (ARGS.out / "objfit_report.json").write_text(json.dumps({"accept": ACCEPT, "sigma": SIGMA, "mean": mean, "episodes": rep}, indent=1))
    print(json.dumps(mean))


if __name__ == "__main__":
    main()
