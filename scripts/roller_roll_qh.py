#!/usr/bin/env python3
"""Paint roller (episodes 18-20): handle roll re-decoded with handle evidence and a bounded angular velocity
(v2d-host venv, CPU).

roller_compose.py keeps the cylinder (cover) track of candidate A and attaches the handle; the only unknown per frame is
the handle's roll phi about the cylinder axis (36 states, 10 deg). It decodes with a transition cost LINEAR in |dphi|:
one 160 deg jump is then exactly as cheap as the same rotation spread over frames, so the decoded roll is piecewise
constant with single-frame jumps (up to 160 deg in one frame on scored frames). A hand-held handle cannot do that.
Evidence per frame f and roll i (all 36 hypotheses, every frame with a roller mask):
  img    whole-object IoU: surface samples of the composite (cover + handle, roller_compose.composite) splatted at 1/4
         resolution vs the roller+handle SAM 2 mask (outputs/object_rh/<episode>/sam2_masks/1), person pixels excluded
  h_iou  handle-only IoU at 1/2 resolution on the scorer's budgeted mesh: cylinder faces = all three vertices within
         0.036 (local units) of the axis and |axial| <= 0.112, handle = the other faces; evidence
         E = mask & ~cylinder silhouette & ~person (dilated 5x5), opened 3x3; h_iou = IoU(handle silhouette & ~cylinder
         & ~person, E). Used only on frames whose E has >= --amin pixels (the handle is actually visible).
Decoding: Viterbi over the 36 states with cost -img - LH * h_iou and a quadratic offset-angle prior
WQ * (|dphi| / 10 deg)^2, transitions above --cap deg/frame forbidden. The DP keeps every hypothesis and resolves them
globally, so occluded stretches are decided by the evidence on both sides.
The new rotation is R_A[f] @ Rot(u, phi_f); rotating about the cylinder's own axis through its centre leaves the cover in
place, so the translation of --base is kept (its centre term is zero to float precision for --base built on the same
cylinder track; checked and reported).
Writes <out>/episode_*.npz (all files of --base, episodes 18-20 rotations replaced), <out>/roller_qh/ (tables, paths)
and <out>/roller_qh/report.json.
Usage: roller_roll_qh.py --cand_a DIR --base DIR --mesh GLB --meshes DIR --out DIR [--amin 20] [--lh 1] [--wq 0.05] [--cap 30]
  --cand_a  episodes with the cylinder track and cylinder mesh (candidate A)
  --base    episodes whose roller handle mesh / translation / everything else are kept
  --mesh    the T-shaped roller mesh (SAM 3D Objects on the handle masks, as for roller_compose.py --mesh)
  --meshes  budget_object_meshes.py output of --base (<episode>_budget.npz)
"""
import argparse
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import trimesh
from PIL import Image

from roller_compose import axis_rot, composite

ROOT = Path(os.environ.get("TRACK1_ROOT") or Path(__file__).resolve().parents[1])
PHI = np.radians(np.arange(0, 360, 10)); NS = 12000; DS = 2
ARGS = None


def intrinsics(tag, ds):
    Kj = json.loads((ROOT / "outputs/camera" / tag / "intrinsics.json").read_text())
    K = np.array([[Kj["fx"] / ds, 0, Kj["cx"] / ds], [0, Kj["fy"] / ds, Kj["cy"] / ds], [0, 0, 1.0]])
    return K, (int(Kj["height"]) // ds, int(Kj["width"]) // ds)


def img_table(ep):
    """Whole-object splat IoU [T,36] at 1/4 resolution (roller_compose.py's image term)."""
    tag = f"episode_{ep:06d}"
    d = np.load(ARGS.cand_a / f"{tag}.npz"); cyl = trimesh.load(ARGS.cand_a / f"{tag}_object.glb", force="mesh")
    V, F, u, cc, _, _ = composite(cyl, trimesh.load(ARGS.mesh, force="mesh"))
    P = np.asarray(trimesh.sample.sample_surface(trimesh.Trimesh(V, F, process=False), 6000, seed=0)[0], float)
    K4, (H4, W4) = intrinsics(tag, 4)
    Rm, t, s = d["object_rotation"].astype(float), d["object_translation"].astype(float), float(d["object_scale"])
    T = len(t); rots = [axis_rot(u, p) for p in PHI]; img = np.zeros((T, len(PHI)))
    for f in range(T):
        mp = ROOT / f"outputs/object_rh/{tag}/sam2_masks/1/{f:06d}.png"
        if not mp.exists():
            continue
        mask = np.array(Image.open(mp))[::4, ::4] > 127
        pp = ROOT / f"outputs/person/{tag}/sam2_masks/0/{f:06d}.png"
        pers = cv2.dilate((np.array(Image.open(pp))[::4, ::4] > 127).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0 if pp.exists() else None
        for i, Rr in enumerate(rots):
            Rf, tf = Rm[f] @ Rr, t[f] + s * Rm[f] @ (cc - Rr @ cc)
            X = s * P @ Rf.T + tf
            uv = np.round((X @ K4.T)[:, :2] / X[:, 2:3]).astype(int)
            ok = (uv[:, 0] >= 0) & (uv[:, 0] < W4) & (uv[:, 1] >= 0) & (uv[:, 1] < H4)
            sil = np.zeros((H4, W4), np.uint8); sil[uv[ok, 1], uv[ok, 0]] = 1
            sil = cv2.dilate(sil, np.ones((3, 3), np.uint8)) > 0
            care = ~pers if pers is not None else np.ones_like(sil)
            un = ((sil | mask) & care).sum()
            img[f, i] = ((sil & mask) & care).sum() / un if un else 0.0
    return img, u, cc


def sample(V, F, n, rng):
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]; area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    ch = rng.choice(len(F), n, p=area / area.sum()); r1, r2 = rng.random(n), rng.random(n); s = np.sqrt(r1)[:, None]
    return (1 - s) * a[ch] + s * (1 - r2[:, None]) * b[ch] + s * r2[:, None] * c[ch]


def splat(X, K, shape):
    uv = np.round((X @ K.T)[:, :2] / X[:, 2:3]).astype(int); m = np.zeros(shape, np.uint8)
    ok = (uv[:, 0] >= 0) & (uv[:, 0] < shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < shape[0]); m[uv[ok, 1], uv[ok, 0]] = 1
    return cv2.dilate(m, np.ones((3, 3), np.uint8)) > 0


def iou(a, b):
    un = (a | b).sum(); return float((a & b).sum() / un) if un else np.nan


def handle_table(ep, u, cc):
    """Handle-only IoU [T,36] and the handle evidence area per frame at 1/2 resolution."""
    tag = f"episode_{ep:06d}"; m = np.load(ARGS.meshes / f"{tag}_budget.npz"); V, F = m["vertices"], m["faces"]
    ax = (V - cc) @ u; rad = np.linalg.norm((V - cc) - ax[:, None] * u, axis=1)
    cylf = ((rad <= 0.036) & (np.abs(ax) <= 0.112))[F].all(1)
    rng = np.random.default_rng(ep); Pc = sample(V, F[cylf], NS, rng); Ph = sample(V, F[~cylf], NS, rng)
    RA = np.load(ARGS.cand_a / f"{tag}.npz")["object_rotation"].astype(float); d = np.load(ARGS.base / f"{tag}.npz")
    t, s = d["object_translation"].astype(float), float(d["object_scale"]); T = len(t)
    K, shape = intrinsics(tag, DS); rots = [axis_rot(u, p) for p in PHI]
    h_iou = np.full((T, 36), np.nan); e_area = np.zeros(T); has = np.zeros(T, bool); k3 = np.ones((3, 3), np.uint8)
    for f in range(T):
        mk = ROOT / f"outputs/object_rh/{tag}/sam2_masks/1/{f:06d}.png"
        if not mk.exists():
            continue
        mask = cv2.resize(cv2.imread(str(mk), 0), (shape[1], shape[0]), interpolation=cv2.INTER_AREA) > 127
        pk = ROOT / f"outputs/person/{tag}/sam2_masks/0/{f:06d}.png"
        pers = (cv2.dilate((cv2.resize(cv2.imread(str(pk), 0), (shape[1], shape[0]), interpolation=cv2.INTER_AREA) > 127).astype(np.uint8),
                           np.ones((5, 5), np.uint8)) > 0) if pk.exists() else np.zeros(shape, bool)
        care = ~pers; has[f] = True
        cyl = splat(s * Pc @ RA[f].T + t[f], K, shape)
        E = cv2.morphologyEx((mask & ~cyl & care).astype(np.uint8), cv2.MORPH_OPEN, k3) > 0; e_area[f] = E.sum()
        for i, Rr in enumerate(rots):
            Ho = splat(s * Ph @ (RA[f] @ Rr).T + t[f], K, shape) & ~cyl & care
            h_iou[f, i] = iou(Ho, E) if (Ho.any() or E.any()) else np.nan
    return h_iou, e_area, has


def tables(ep):
    img, u, cc = img_table(ep)
    h_iou, e_area, has = handle_table(ep, u, cc)
    np.savez(ARGS.out / "roller_qh" / f"tables_{ep}.npz", img=img, h_iou=h_iou, e_area=e_area, has=has, u=u, cc=cc)
    return ep


def viterbi(cost, trans):
    T, N = cost.shape; acc = cost[0].copy(); back = np.zeros((T, N), int)
    for f in range(1, T):
        tot = acc[:, None] + trans; back[f] = tot.argmin(0); acc = tot.min(0) + cost[f]
    p = np.zeros(T, int); p[-1] = int(acc.argmin())
    for f in range(T - 1, 0, -1):
        p[f - 1] = back[f, p[f]]
    return p


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand_a", type=Path, required=True); ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--mesh", type=Path, required=True); ap.add_argument("--meshes", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True); ap.add_argument("--episodes", type=int, nargs="*", default=[18, 19, 20])
    ap.add_argument("--amin", type=float, default=20.0, help="handle evidence pixels (1/2 resolution) for the handle term")
    ap.add_argument("--lh", type=float, default=1.0, help="weight of the handle term")
    ap.add_argument("--wq", type=float, default=0.05, help="quadratic offset-angle prior per (10 deg)^2")
    ap.add_argument("--cap", type=float, default=30.0, help="largest roll change per frame (deg)")
    ap.add_argument("--reuse", action="store_true", help="decode from existing tables")
    ARGS = ap.parse_args(); (ARGS.out / "roller_qh").mkdir(parents=True, exist_ok=True)
    for f in ARGS.base.iterdir():
        if f.is_file():
            shutil.copy2(f, ARGS.out / f.name)
    if not ARGS.reuse:
        with ProcessPoolExecutor(len(ARGS.episodes)) as pool:
            list(pool.map(tables, ARGS.episodes))
    dphi = np.abs(((PHI[:, None] - PHI[None]) + np.pi) % (2 * np.pi) - np.pi) / np.radians(10)
    prior = np.where(dphi <= ARGS.cap / 10 + 1e-9, ARGS.wq * dphi ** 2, 1e9)
    rep = {"args": {k: str(v) for k, v in vars(ARGS).items()}, "episodes": {}}
    for ep in ARGS.episodes:
        tag = f"episode_{ep:06d}"; tb = np.load(ARGS.out / "roller_qh" / f"tables_{ep}.npz")
        vis = (tb["e_area"] >= ARGS.amin) & tb["has"]; h = np.where(vis[:, None], np.nan_to_num(tb["h_iou"], nan=0.0), 0.0)
        p = viterbi(-tb["img"] - ARGS.lh * h, prior)
        d = dict(np.load(ARGS.base / f"{tag}.npz")); a = np.load(ARGS.cand_a / f"{tag}.npz"); u, cc = tb["u"], tb["cc"]
        RA, tA, s = a["object_rotation"].astype(float), a["object_translation"].astype(float), float(d["object_scale"])
        Rn = np.stack([RA[f] @ axis_rot(u, PHI[p[f]]) for f in range(len(RA))])
        tn = np.stack([tA[f] + s * RA[f] @ (cc - axis_rot(u, PHI[p[f]]) @ cc) for f in range(len(RA))])
        old = np.array([np.argmin([np.abs(RA[f] @ axis_rot(u, q) - d["object_rotation"][f]).max() for q in PHI]) for f in range(len(RA))])
        diff = np.degrees(np.abs(((PHI[p] - PHI[old]) + np.pi) % (2 * np.pi) - np.pi))
        step = np.degrees(np.abs(((PHI[p][1:] - PHI[p][:-1]) + np.pi) % (2 * np.pi) - np.pi))
        rep["episodes"][tag] = {"frames": int(len(p)), "frames_with_handle_term": int(vis.sum()), "frames_changed_gt45deg": int((diff > 45).sum()),
                                "max_roll_step_deg": float(step.max()),
                                "max_translation_centre_term_cm": float(np.linalg.norm(tn - d["object_translation"], axis=1).max() * 100)}
        d["object_rotation"] = Rn.astype(d["object_rotation"].dtype)
        np.savez(ARGS.out / f"{tag}.npz", **d)
        np.save(ARGS.out / "roller_qh" / f"path_{ep}.npy", p); print(tag, rep["episodes"][tag], flush=True)
    (ARGS.out / "roller_qh" / "report.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
