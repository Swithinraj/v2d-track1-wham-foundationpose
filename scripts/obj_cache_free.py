#!/usr/bin/env python3
"""GT-free human + object caches for episodes WITHOUT ground truth (the challenge), in the layout the learned correction
selector (scripts/corr_select3.py) was trained on, so it deploys with the very loader and features it was trained with
(mhr_converter.sif, GPU; trimesh from the repo's submission venv if the container lacks it).

Per episode: a candidate's pre-contact episode INPUT (human + object track + object mesh) and RAW (the human before
refinement); ROOT_DIR holds outputs/object/<tag> (FoundationPose raw, SAM 2 masks, MoGe depth, intrinsics) and
outputs/contact_sdf/<INPUT name> (optional), SCALE_ROOT outputs/sam3d_body/<tag>/prepare.npz (the depth scale the INPUT
objects are expressed in; default ROOT_DIR). Observation fields are computed exactly as for the training caches (MoGe mask
depth decoded as metres = 65535 / v - 1, over the episode's depth scale); the GT fields hold our own track (gt = ours,
similarity = identity), which the deployed features never read; valid = the scored frames with --valid_scored (as in
training), anchor = the first scored frame.
Writes OUT/human/<name>_<tag>.npz and OUT/object/<name>_<tag>_obj.npz (name = --name, default ROOT_DIR's name).
Usage: obj_cache_free.py --input DIR --raw DIR --root_dir DIR --scored JSON --out DIR [--scale_root DIR] [--name N] [--valid_scored]
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(os.environ.get("TRACK1_ROOT") or "/track1")
sys.path.insert(0, str(ROOT / "official/v2d_submission_kit/tools/track1"))
sys.path.insert(0, str(ROOT / "scripts"))
from mesh_to_mhr_params import MHR  # noqa: E402
from mhr_bodies import MODEL  # noqa: E402
from extract_mhr_hands import hand_vertex_indices  # noqa: E402

try:
    import trimesh
except ImportError:                                       # containers without trimesh: the repo's submission venv (pure python)
    sys.path.append(str(Path(__file__).resolve().parents[1] / ".venvs/submission/lib/python3.12/site-packages"))
    import trimesh  # noqa: E402

SC = [1, 2, 18, 35, 3, 19, 36, 4, 20, 37, 8, 24, 110, 74, 38, 113, 75, 39, 76, 40, 78, 42]
NS = 512


def quat_to_R(q):
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
                     [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
                     [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])


def surface(path):
    m = trimesh.load(path, force="mesh")
    p, fi = trimesh.sample.sample_surface(m, NS, seed=0)
    return np.asarray(p, np.float32), np.asarray(m.face_normals[fi], np.float32)


def mask_stats(args):
    """-> mask area fraction, centroid u, v (px), median raw depth pixel, median decoded metres, robust spread of the
    decoded metres (1.4826 MAD) inside the mask."""
    mpath, dpath = args
    nan = (0.0, np.nan, np.nan, np.nan, np.nan, np.nan)
    if not mpath.exists():
        return nan
    m = np.asarray(Image.open(mpath)) > 0
    if m.ndim == 3:
        m = m.any(-1)
    if not m.any():
        return nan
    v, u = np.nonzero(m)
    raw = met = spread = np.nan
    if dpath.exists():
        d = np.asarray(Image.open(dpath)).astype(float)
        if d.shape == m.shape:
            z = d[m]; z = z[z > 0]
            if len(z):
                raw = float(np.median(z))
                zm = 65535.0 / z - 1.0                              # v2d DepthImage: v = 65535 / (metres + 1)
                met = float(np.median(zm)); spread = float(1.4826 * np.median(np.abs(zm - met)))
    return float(m.mean()), float(u.mean()), float(v.mean()), raw, met, spread


def main():
    ap = argparse.ArgumentParser()
    for k in ("input", "raw", "root_dir", "scored", "out"):
        ap.add_argument(f"--{k}", type=Path, required=True)
    ap.add_argument("--scale_root", type=Path); ap.add_argument("--name")
    ap.add_argument("--valid_scored", action="store_true", help="valid = the scored frames (as the training caches)")
    a = ap.parse_args(); name = a.name or a.root_dir.name; scale_root = a.scale_root or a.root_dir
    (a.out / "human").mkdir(parents=True, exist_ok=True); (a.out / "object").mkdir(parents=True, exist_ok=True)
    scored = json.loads(a.scored.read_text())
    mhr = MHR(str(MODEL), "cuda", chunk=128, precision="float32")
    hands = np.concatenate([h[::4] for h in hand_vertex_indices()])
    f32 = lambda x: np.asarray(x, np.float32)  # noqa: E731
    with ProcessPoolExecutor(16) as pool:
        for f in sorted(a.input.glob("episode_*.npz")):
            tag = f.stem; ep = int(tag.split("_")[1]); d = np.load(f); r = np.load(a.raw / f"{tag}.npz"); T = len(d["pose"])
            z = torch.as_tensor(np.concatenate([d["scales"], d["shape"]])[None], dtype=torch.float64, device="cuda")
            V, J = mhr.run(torch.as_tensor(d["pose"], dtype=torch.float64, device="cuda"), z)
            Vr, Jr = mhr.run(torch.as_tensor(r["pose"][:T], dtype=torch.float64, device="cuda"), z)
            J, Jr = J[:, SC].cpu().numpy(), Jr[:, SC].cpu().numpy()
            H, Hr = V[:, hands].cpu().numpy() / 1000.0, Vr[:, hands].cpu().numpy() / 1000.0
            anchor = min(x for x in scored[str(ep)] if x < T)
            valid = np.ones(T, bool)
            if a.valid_scored:
                valid[:] = False; valid[[x for x in scored[str(ep)] if x < T]] = True
            np.savez(a.out / "human" / f"{name}_{tag}.npz", ours=f32(J), raw=f32(Jr), gt=f32(J), conf=np.zeros((T, 22), np.float32),
                     vis=np.zeros((T, 22), np.float32), hands_ours=f32(H), hands_raw=f32(Hr), hands_gt=f32(H), valid=valid,
                     anchor=anchor, sim3_s=1.0, sim3_R=np.eye(3), sim3_t=np.zeros(3))
            oR, ot, os_ = d["object_rotation"].astype(float), d["object_translation"].astype(float), float(d["object_scale"])
            obj = a.root_dir / "outputs/object" / tag
            raw = [json.loads((obj / f"foundationpose/raw/{k:06d}.json").read_text()) for k in range(T)]
            rR = np.stack([quat_to_R(q["rotation"]) for q in raw]); rt = np.array([q["translation"] for q in raw], float)
            fin = np.isfinite(rt).all(1) & np.isfinite(rR).reshape(T, -1).all(1) & (np.linalg.norm(np.nan_to_num(rt), axis=1) > 1e-6)
            if fin.sum() >= 10:
                rt *= np.median(np.linalg.norm(ot[fin], axis=1)) / np.median(np.linalg.norm(rt[fin], axis=1))
            else:
                fin[:] = False
            rt[~fin] = np.nan; rR[~fin] = np.nan
            ang = np.degrees(np.arccos(np.clip((np.einsum("tij,tij->t", np.nan_to_num(rR), oR) - 1) / 2, -1, 1)))
            raw_ok = np.full(T, float(fin.any() and np.median(ang[fin]) < 30.0))
            so, no = surface(a.input / f"{tag}_object.glb")
            K = json.loads((obj / "intrinsics/000000.json").read_text())
            st = np.array(list(pool.map(mask_stats, [(obj / f"sam2_masks/1/{k:06d}.png", obj / f"depth/{k:06d}.png") for k in range(T)], chunksize=16)))
            area = st[:, 0]; mu = np.nan_to_num((st[:, 1] - K["cx"]) / K["fx"]); mv = np.nan_to_num((st[:, 2] - K["cy"]) / K["fy"])
            old = st[:, 3]; ok_old = np.isfinite(old) & (area > 0)
            if ok_old.any():
                old = old * np.median(ot[ok_old, 2] / old[ok_old])
            old = np.where(ok_old, old, 0.0)
            pp = scale_root / f"outputs/sam3d_body/{tag}/prepare.npz"
            scale = float(np.load(pp)["depth_scale"]) if pp.exists() else np.nan
            met, spread = st[:, 4], st[:, 5]
            ok = np.isfinite(met) & (area > 0) & np.isfinite(scale) & (met > 0)
            dep = np.where(ok, met / scale, 0.0); rel = np.where(ok, spread / np.maximum(met, 1e-6), 0.0)
            sp = a.root_dir / "outputs/contact_sdf" / a.input.name / f"{tag}.npz"
            sd = np.load(sp) if sp.exists() else {"sdf": np.ones((2, 2, 2)), "lo": -np.ones(3), "hi": np.ones(3)}
            np.savez_compressed(a.out / "object" / f"{name}_{tag}_obj.npz", sdf=f32(sd["sdf"]), sdf_lo=f32(sd["lo"]), sdf_hi=f32(sd["hi"]),
                                sdf_ok=float(sp.exists()), obj_R=f32(oR), obj_t=f32(ot), obj_s=os_, raw_R=f32(rR), raw_t=f32(rt), raw_ok=f32(raw_ok),
                                gt_R=f32(oR), gt_t=f32(ot), gt_s=os_, surf_ours=so, norm_ours=no, surf_gt=so, norm_gt=no,
                                mask_area=f32(area), mask_u=f32(mu), mask_v=f32(mv), mask_depth=f32(dep), depth_ok=f32(ok),
                                mask_depth_old=f32(old), depth_ok_old=f32(ok_old), depth_rel=f32(rel), depth_scale=scale,
                                K=f32([K["fx"], K["fy"], K["cx"], K["cy"]]), gt_dist=np.ones((T, 2), np.float32), gt_cvec=np.zeros((T, 2, 3), np.float32))
            dr = np.median(dep[ok] / ot[ok, 2]) if ok.any() else np.nan
            print(name, tag, T, f"decoded depth / our track z median {dr:.2f} | mask empty {int((area == 0).sum())} | "
                  f"sdf {'ok' if sp.exists() else 'not used'}", flush=True)


if __name__ == "__main__":
    main()
