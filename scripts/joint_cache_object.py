#!/usr/bin/env python3
"""Joint human-object refiner cache, object part (mhr_converter.sif for PIL, trimesh from the repo's
submission venv if the container lacks it; CPU). Needs the human part (ext/joint_cache_human.py) for T, the QC mask, the hand points and the evaluator's
similarity, so GT objects land in our frame through the very transform the scorer applies.

Per episode, all frames:
  obj_R, obj_t, obj_s        our object track (episodes_q3_in = the final object; the contact stage leaves objects alone)
  raw_R, raw_t, raw_ok       per-frame FoundationPose (outputs/object/<tag>/foundationpose/raw, wxyz quaternion), translation
                             brought to our frame by the per-episode median ratio |ours| / |raw|; raw_ok = 0 where the raw
                             rotation is in another mesh frame (symmetric-object episodes: median angle to ours > 30 deg)
  gt_R, gt_t, gt_s           GT object pose in OUR frame: x_ours = R_sim (x_gt - t_sim) / s_sim, so gt_R = R_sim P_R,
                             gt_t = R_sim (P_t - t_sim) / s_sim and gt_s = 1 / s_sim (the GT mesh is in metres)
  surf_ours, norm_ours       512 surface samples (+ face normals) of our mesh, local coordinates (posed: s R p + t)
  surf_gt, norm_gt           the same for the GT mesh (object_mesh/output_aligned.glb)
  mask_area, mask_u, mask_v  SAM 2 object mask: area fraction, centroid in normalised image coordinates ((u - cx) / fx)
  mask_depth, depth_ok       median MoGe depth inside the mask, scaled to our frame by the per-episode median ratio to our
                             object depth (0 / not ok where the mask is empty)
  K                          fx, fy, cx, cy (first frame)
  gt_dist                    [T, 2] GT hand-to-GT-object distance per hand (metres, GT metric), for contact labels
  gt_cvec                    [T, 2, 3] nearest GT object point minus the GT hand centroid, our frame
  sdf, sdf_lo, sdf_hi, sdf_ok  the scorer-consistent signed-distance grid of our budgeted mesh (outputs/contact_sdf/
                             episodes_vpq3_in, else episodes_q3_in, contact_sdf.py: winding-number sign, local mesh units; local = R^T (x - t) / s)
Writes OUT/<root>_<episode>_obj.npz. Usage: joint_cache_object.py HUMAN_CACHE OUT ROOT_DIR [ROOT_DIR ...]
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import trimesh
except ImportError:                                       # containers without trimesh: the repo's submission venv (pure python)
    sys.path.append(str(Path(__file__).resolve().parents[1] / ".venvs/submission/lib/python3.12/site-packages"))
    import trimesh  # noqa: E402

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
    mpath, dpath = args
    if not mpath.exists():
        return 0.0, np.nan, np.nan, np.nan
    m = np.asarray(Image.open(mpath)) > 0
    if m.ndim == 3:
        m = m.any(-1)
    if not m.any():
        return 0.0, np.nan, np.nan, np.nan
    v, u = np.nonzero(m)
    dep = np.nan
    if dpath.exists():
        d = np.asarray(Image.open(dpath)).astype(float)
        if d.shape == m.shape:
            z = d[m]; z = z[z > 0]
            dep = float(np.median(z)) if len(z) else np.nan
    return float(m.mean()), float(u.mean()), float(v.mean()), dep


def episode(root, tag, hc, out, pool):
    h = np.load(hc)
    T = len(h["ours"]); s_sim, R_sim, t_sim = float(h["sim3_s"]), h["sim3_R"], h["sim3_t"]
    d = np.load(root / "submissions/episodes_q3_in" / f"{tag}.npz")
    oR, ot, os_ = d["object_rotation"][:T].astype(float), d["object_translation"][:T].astype(float), float(d["object_scale"])
    obj = root / "outputs/object" / tag
    raw = [json.loads((obj / f"foundationpose/raw/{f:06d}.json").read_text()) for f in range(T)]
    rR = np.stack([quat_to_R(r["rotation"]) for r in raw]); rt = np.array([r["translation"] for r in raw], float)
    fin = np.isfinite(rt).all(1) & np.isfinite(rR).reshape(T, -1).all(1) & (np.linalg.norm(np.nan_to_num(rt), axis=1) > 1e-6)
    if fin.sum() >= 10:                                   # scale from valid frames (FoundationPose writes zeros where lost)
        rt *= np.median(np.linalg.norm(ot[fin], axis=1)) / np.median(np.linalg.norm(rt[fin], axis=1))
    else:
        fin[:] = False
    rt[~fin] = np.nan; rR[~fin] = np.nan                  # invalid frames -> NaN (the loader replaces them and sets rt_ok = 0)
    ang = np.degrees(np.arccos(np.clip((np.einsum("tij,tij->t", np.nan_to_num(rR), oR) - 1) / 2, -1, 1)))
    raw_ok = np.full(T, float(fin.any() and np.median(ang[fin]) < 30.0))
    P = np.load(root / "gt" / tag / "poses.npy")[:T]
    gR = R_sim @ P[:, :3, :3]; gt_t = ((P[:, :3, 3] - t_sim) @ R_sim.T) / s_sim; gs = 1.0 / s_sim
    so, no = surface(root / "submissions/episodes_q3_in" / f"{tag}_object.glb")
    sg, ng = surface(root / "gt" / tag / "object_mesh/output_aligned.glb")
    K = json.loads((obj / "intrinsics/000000.json").read_text())
    st = np.array(list(pool.map(mask_stats, [(obj / f"sam2_masks/1/{f:06d}.png", obj / f"depth/{f:06d}.png") for f in range(T)], chunksize=16)))
    area = st[:, 0]; mu = np.nan_to_num((st[:, 1] - K["cx"]) / K["fx"]); mv = np.nan_to_num((st[:, 2] - K["cy"]) / K["fy"])
    dep = st[:, 3]; ok = np.isfinite(dep) & (area > 0)
    if ok.any():
        dep = dep * np.median(ot[ok, 2] / dep[ok])
    dep = np.where(ok, dep, 0.0)
    # GT contact: GT hands vs the GT object surface, both in our frame, distances back in GT metres (x s_sim)
    Og = gs * np.einsum("tij,nj->tni", gR, sg) + gt_t[:, None]                       # [T, NS, 3]
    hg = h["hands_gt"].astype(float)                                                # [T, 128, 3]
    gdist = np.zeros((T, 2)); cvec = np.zeros((T, 2, 3))
    for k, sl in enumerate((slice(0, 64), slice(64, 128))):
        D = np.linalg.norm(hg[:, sl, None] - Og[:, None, ::2], axis=-1)              # [T, 64, NS/2]
        flat = D.reshape(T, -1).argmin(1); i, j = np.unravel_index(flat, D.shape[1:])
        gdist[:, k] = D.reshape(T, -1).min(1) * s_sim
        cvec[:, k] = Og[np.arange(T), 2 * j] - hg[:, sl].mean(1)
    sp = root / "outputs/contact_sdf/episodes_vpq3_in" / f"{tag}.npz"
    if not sp.exists():                                    # roots without the contact tail: same object mesh and track
        sp = root / "outputs/contact_sdf/episodes_q3_in" / f"{tag}.npz"
    sd = np.load(sp) if sp.exists() else {"sdf": np.ones((2, 2, 2)), "lo": -np.ones(3), "hi": np.ones(3)}
    f32 = lambda x: np.asarray(x, np.float32)  # noqa: E731
    np.savez_compressed(out / f"{root.name}_{tag}_obj.npz", sdf=f32(sd["sdf"]), sdf_lo=f32(sd["lo"]), sdf_hi=f32(sd["hi"]), sdf_ok=float(sp.exists()), obj_R=f32(oR), obj_t=f32(ot), obj_s=os_, raw_R=f32(rR), raw_t=f32(rt), raw_ok=f32(raw_ok),
                        gt_R=f32(gR), gt_t=f32(gt_t), gt_s=gs, surf_ours=so, norm_ours=no, surf_gt=sg, norm_gt=ng,
                        mask_area=f32(area), mask_u=f32(mu), mask_v=f32(mv), mask_depth=f32(dep), depth_ok=f32(ok),
                        K=f32([K["fx"], K["fy"], K["cx"], K["cy"]]), gt_dist=f32(gdist), gt_cvec=f32(cvec))
    v = h["valid"]
    print(root.name, tag, T, f"raw rot median {np.median(ang):.1f} deg | mask empty {int((area == 0).sum())} | "
          f"GT contact {np.mean(gdist[v] < 0.015) * 100:.0f} % hand-frames | obj err {np.linalg.norm(ot[v] - gt_t[v], axis=1).mean() * 100 * s_sim:.1f} cm", flush=True)


def main():
    hc, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(16) as pool:
        for root in map(Path, sys.argv[3:]):
            for ep in range(30):
                tag = f"episode_{ep:06d}"; f = hc / f"{root.name}_{tag}.npz"
                if f.exists():
                    episode(root, tag, f, out, pool)


if __name__ == "__main__":
    main()
