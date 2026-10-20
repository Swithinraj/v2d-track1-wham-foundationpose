#!/usr/bin/env python3
"""Step 3 of the joint scene solve: held objects' DEPTH toward the hands, human fixed (host venv).

Kaggle JD showed the human's depth is the more accurate one during grasps, so the 2-10 cm hand-object depth gaps are
object errors. Variable: eps_t, a shift of the object along the camera ray through its centre (m), only inside grasp
intervals (grasp posterior >= 0.5 for either hand, dilated by 30 frames), smooth onset/offset (30-frame ramps) and
smooth by construction (Gaussian bumps every 15 frames, sigma 40 frames) to add little object acceleration.
Terms: contact (each hand's contact patch on the moved object surface, weighted by the grasp posterior, SIG_C) and a
hinge against hand vertices entering the object; a weak depth prior (SIG_D). The human, the object's rotation and
scale stay unchanged. Checks per episode: silhouette IoU against the SAM 2 masks (independent: a nearer object of the
same size projects larger), object acceleration on scored frames, contact-patch gap.
Usage: held_object_depth.py BASE_DIR BODIES_DIR SDF_DIR STATES_DIR OUT_DIR
  (BASE_DIR: episodes with the reference human and objects; BODIES_DIR: mhr_bodies.py of that human)
"""
import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh
from PIL import Image
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

ROOT = Path(os.environ.get("TRACK1_ROOT") or Path(__file__).resolve().parents[1]); R0 = ROOT
sys.path.insert(0, str(ROOT / "scripts"))
from interaction_states import sdf_fn  # noqa: E402

SIG_C, SIG_D, NPATCH = 0.005, 0.10, 40
S = json.loads((R0 / "outputs/hand_fix/episodes_repro_depth_refine_meshes/scored_frames.json").read_text())


def silhouette_iou(R, t, s, tag, frames):
    """median silhouette IoU of the scorer's budgeted mesh posed (R, t, s) against the SAM 2 object mask, 1/4 resolution"""
    K = json.loads((R0 / "outputs/camera" / tag / "intrinsics.json").read_text())
    W, H = int(K["width"]) // 4, int(K["height"]) // 4
    Km = np.array([[K["fx"] / 4, 0, K["cx"] / 4], [0, K["fy"] / 4, K["cy"] / 4], [0, 0, 1]])
    b = np.load(R0 / "outputs/hand_fix/episodes_repro_depth_refine_meshes" / f"{tag}_budget.npz")
    V, F = b["vertices"].astype(float), b["faces"]
    out = []
    for f in frames:
        X = s * V @ R[f].T + t[f]
        if (X[:, 2] <= 0.05).any():
            continue
        uv = (X @ Km.T)[:, :2] / X[:, 2:3]
        m = np.zeros((H, W), np.uint8); cv2.fillPoly(m, [np.round(uv[tri]).astype(np.int32) for tri in F], 1)
        gt = np.array(Image.open(R0 / "outputs/object" / tag / "sam2_masks/1" / f"{f:06d}.png"))[::4, ::4] > 127
        u = (m.astype(bool) | gt).sum(); out.append((m.astype(bool) & gt).sum() / u if u else 1.0)
    return float(np.median(out)) if out else float("nan")


def main():
    base, bdir, sdir, stdir, out = map(Path, sys.argv[1:6])
    out.mkdir(parents=True, exist_ok=True)
    for f in base.glob("episode_??????_object*"):
        shutil.copy2(f, out / f.name)
    rep = {}
    for ep in range(30):
        tag = f"episode_{ep:06d}"; d = dict(np.load(base / f"{tag}.npz"))
        H = np.load(bdir / f"{tag}.npz")["hands"].astype(float)
        R, t, s = d["object_rotation"].astype(float), d["object_translation"].astype(float), float(d["object_scale"])
        T = min(len(H), len(t)); half = H.shape[1] // 2
        pg = np.load(stdir / f"{tag}.npz")["posterior"][:T, :, 2]
        sdf = sdf_fn(np.load(sdir / f"{tag}.npz"))
        mesh = trimesh.load(base / f"{tag}_object.glb", force="mesh")
        pc = (np.asarray(mesh.vertices).min(0) + np.asarray(mesh.vertices).max(0)) / 2
        cen_obj = s * np.einsum("fij,j->fi", R[:T], pc) + t[:T]; ray = cen_obj / np.linalg.norm(cen_obj, axis=1, keepdims=True)
        grasp = (pg >= 0.5).any(1)
        seg = np.zeros(T, bool)
        for f in np.flatnonzero(grasp):
            seg[max(0, f - 30):f + 31] = True
        # taper: 0 outside the dilated grasp intervals, smooth 30-frame ramps inside
        dist_out = np.full(T, 1e9); idx_out = np.flatnonzero(~seg)
        if len(idx_out):
            dist_out = np.min(np.abs(np.arange(T)[:, None] - idx_out[None]), axis=1).astype(float)
        rr = np.clip(dist_out / 30.0, 0, 1); W = rr * rr * (3 - 2 * rr); W[~seg] = 0
        cframes = []
        for f in np.flatnonzero(grasp):
            for k in range(2):
                if pg[f, k] < 0.5:
                    continue
                P = H[f, k * half:(k + 1) * half]; dist = sdf((P - t[f]) @ R[f] / s) * s
                idx = np.flatnonzero(np.abs(dist) < 0.03)
                if len(idx) >= 20:
                    cframes.append((f, P[idx[np.linspace(0, len(idx) - 1, min(NPATCH, len(idx))).astype(int)]], float(pg[f, k])))
        sc = [f for f in S[str(ep)] if f < T]
        acc = lambda tt: float(np.mean([np.linalg.norm(tt[f + 1] - 2 * tt[f] + tt[f - 1]) for f in sc if f - 1 in sc and f + 1 in sc]) * 100)  # noqa: E731
        sil = lambda tt: silhouette_iou(R, tt, s, tag, [int(f) for f in np.array(sc)[::10] if seg[f]] or [int(sc[0])])  # noqa: E731
        if not cframes:
            np.savez(out / f"{tag}.npz", **d); rep[tag] = {"grasp_frames": 0}; continue
        cen = np.arange(0, T + 15, 15); Phi = np.exp(-(np.arange(T)[:, None] - cen[None]) ** 2 / (2 * 40.0 ** 2)); Phi /= Phi.sum(1, keepdims=True)

        def resid(c):
            e = W * (Phi @ c); r = []
            for f, P, p in cframes:
                dist = sdf((P - t[f] - e[f] * ray[f]) @ R[f] / s) * s
                r.append(np.sqrt(p) * dist / SIG_C); r.append(np.sqrt(p) * np.minimum(dist, 0) / (SIG_C / 2))
            r.append(e / SIG_D)
            return np.concatenate(r)

        near = lambda f: np.flatnonzero(np.abs(cen - f) <= 140)  # noqa: E731
        n_c = sum(2 * len(P) for _, P, _ in cframes)
        Sp = lil_matrix((n_c + T, len(cen)), dtype=np.int8); r0 = 0
        for f, P, _ in cframes:
            Sp[r0:r0 + 2 * len(P), near(f)] = 1; r0 += 2 * len(P)
        for f in range(T):
            Sp[r0 + f, near(f)] = 1
        sol = least_squares(resid, np.zeros(len(cen)), jac_sparsity=Sp.tocsr(), loss="cauchy", f_scale=1.0, x_scale=0.01, max_nfev=200)
        e = W * (Phi @ sol.x)
        tn = t.copy(); tn[:T] = t[:T] + e[:, None] * ray
        d["object_translation"] = tn.astype(np.float32)
        np.savez(out / f"{tag}.npz", **d)
        g0 = np.median([np.abs(sdf((P - t[f]) @ R[f] / s) * s).mean() for f, P, _ in cframes]) * 1000
        g1 = np.median([np.abs(sdf((P - tn[f]) @ R[f] / s) * s).mean() for f, P, _ in cframes]) * 1000
        rep[tag] = {"grasp_frames": int(grasp.sum()), "eps_cm_median_in_grasp": float(np.median(e[grasp]) * 100), "eps_cm_absmax": float(np.abs(e).max() * 100),
                    "patch_gap_mm": [float(g0), float(g1)], "iou_grasp_frames": [sil(t), sil(tn)], "obj_accel_cm": [acc(t), acc(tn)]}
        print(tag, json.dumps(rep[tag]), flush=True)
    (out / "object_depth_from_contact_report.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
