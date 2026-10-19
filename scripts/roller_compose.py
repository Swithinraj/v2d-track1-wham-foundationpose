#!/usr/bin/env python3
"""Roller handle experiment, step 4b: cylinder track + rigidly attached handle (v2d-host venv, CPU).

Tracking the new T-shaped roller mesh (SAM 3D Objects c05: cover with the frame and handle perpendicular from the
middle) failed (silhouette IoU 0.65 -> 0.40: flips and offsets), while the old cylinder track keeps the cover on its
mask. So the cover pose is kept from candidate A and the handle is attached to it: c05 is aligned into the cylinder's
local frame (its cover onto the cylinder, scaled to the cylinder's length). The cylinder is symmetric about its axis,
so the only unknown is the roll phi of the handle about that axis. Per frame and phi (10 deg steps):
  image   silhouette IoU (splatted surface samples) against the new cover+handle masks, pixels of the person mask
          excluded from both (a held handle is hidden by the hand),
  hand    distance of the handle's grip end from the nearest dense hand vertex, only in frames where SOME roll puts
          the grip within hold_m (8 cm) of a hand (the handle is held),
  table   otherwise (not held) the grip must lie at the cover's lowest level along the floor normal (+-2 cm): gravity,
          an unheld handle rests on the table and neither floats nor sinks,
and a Viterbi pass picks a smooth phi sequence (cost per 10 deg of change). Rotation about the cylinder's own axis
through its centre leaves the cover where it was; the translation changes only by that centre term.
Writes submissions/episodes_cand_<NAME>/ (= candidate A with episodes 18-20 replaced) and runs/roller/compose_report.json.
Usage: roller_compose.py [--name Arc] [--mesh PATH] [--w_img 1] [--w_hand 1] [--w_table 1] [--w_smooth 0.05]
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
from scipy.spatial import cKDTree

import os  # noqa: E402
ROOT = Path(os.environ.get("TRACK1_ROOT") or Path(__file__).resolve().parents[1])

R0 = ROOT
A = BASE_BODIES = FLOOR = None        # set from --base / --bodies / --floor in main()
PHI = np.radians(np.arange(0, 360, 10))
ARGS = None


def axis_rot(u, a):
    u = u / np.linalg.norm(u); K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def align_vectors(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else axis_rot(np.cross(a, [1, 0, 0]) if abs(a[0]) < 0.9 else np.cross(a, [0, 1, 0]), np.pi)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1 + c)


def composite(cyl, t_mesh):
    """c05 aligned into the cylinder's local frame. Returns (vertices, faces, axis u, centre c, grip point g)."""
    Vc = np.asarray(cyl.vertices, float); cc = (Vc.min(0) + Vc.max(0)) / 2
    _, _, ax = np.linalg.svd(Vc - Vc.mean(0), full_matrices=False); u = ax[0]
    L = np.ptp((Vc - cc) @ u)
    V5 = np.asarray(t_mesh.vertices, float); m5 = V5.mean(0)
    _, _, a5 = np.linalg.svd(V5 - m5, full_matrices=False)
    best = None
    for stem, bar in ((0, 1), (1, 0)):
        s = (V5 - m5) @ a5[stem]; b = (V5 - m5) @ a5[bar]
        for end in (s.max(), s.min()):
            sel = np.abs(s - end) < 0.07
            spread = np.ptp(b[sel]) if sel.sum() > 20 else 0.0
            score = -abs(spread - L)
            if best is None or score > best[0]:
                best = (score, stem, bar, end, sel)
    _, stem, bar, end, cover = best
    s = (V5 - m5) @ a5[stem]
    Vcov = V5[cover]; _, _, ac = np.linalg.svd(Vcov - Vcov.mean(0), full_matrices=False); u5 = ac[0]
    proj = (Vcov - Vcov.mean(0)) @ u5
    c5 = Vcov.mean(0) + u5 * (proj.max() + proj.min()) / 2
    k = L / np.ptp(proj)
    R = align_vectors(u5, u)
    V = k * (V5 - c5) @ R.T + cc
    far = s.min() if end == s.max() else s.max()
    g5 = V5[np.abs(s - far) < 0.02].mean(0)
    g = k * R @ (g5 - c5) + cc
    return V, np.asarray(t_mesh.faces), u, cc, g, {"cover_vertices": int(cover.sum()), "scale_to_cylinder": float(k),
                                                    "cylinder_length_local": float(L)}


def episode(ep):
    tag = f"episode_{ep:06d}"
    d = dict(np.load(A / f"{tag}.npz")); cyl = trimesh.load(A / f"{tag}_object.glb", force="mesh")
    V, F, u, cc, g, info = composite(cyl, trimesh.load(ARGS.mesh, force="mesh"))
    comp = trimesh.Trimesh(V, F, process=False)
    P = np.asarray(trimesh.sample.sample_surface(comp, 6000, seed=0)[0], float)
    Kj = json.loads((R0 / "outputs/camera" / tag / "intrinsics.json").read_text())
    K4 = np.array([[Kj["fx"] / 4, 0, Kj["cx"] / 4], [0, Kj["fy"] / 4, Kj["cy"] / 4], [0, 0, 1]]); W4, H4 = int(Kj["width"]) // 4, int(Kj["height"]) // 4
    Rm, t, s = d["object_rotation"].astype(float), d["object_translation"].astype(float), float(d["object_scale"])
    T = len(t)
    hands = np.load(BASE_BODIES / f"{tag}.npz")["hands"].astype(float)
    sd = json.loads((R0 / "outputs/alignment_sam" / tag / "alignment.json").read_text())["depth_scale"]
    n = np.asarray(FLOOR[tag]["normal"], float); n /= np.linalg.norm(n)
    rots = [axis_rot(u, p) for p in PHI]
    cost = np.zeros((T, len(PHI))); diag = {"img_frames": 0, "hand_frames": 0, "table_frames": 0}
    for f in range(T):
        mp = R0 / f"outputs/object_rh/{tag}/sam2_masks/1/{f:06d}.png"
        mask = np.array(Image.open(mp))[::4, ::4] > 127 if mp.exists() else None
        pp = R0 / f"outputs/person/{tag}/sam2_masks/0/{f:06d}.png"
        pers = cv2.dilate((np.array(Image.open(pp))[::4, ::4] > 127).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0 if pp.exists() else None
        hand_tree = cKDTree(hands[f]) if f < len(hands) else None
        poses = [(Rm[f] @ Rr, t[f] + s * Rm[f] @ (cc - Rr @ cc)) for Rr in rots]
        grips = np.array([s * Rf @ g + tf for Rf, tf in poses])
        dh = hand_tree.query(grips)[0] if hand_tree is not None else np.full(len(PHI), np.inf)
        held = dh.min() < ARGS.hold_m          # some roll puts the grip into a hand: the handle is held
        if not held:
            Xc = s * np.asarray(cyl.vertices, float)[::20] @ Rm[f].T + t[f]
            cover_low = float((Xc @ n).min())
        for i, (Rf, tf) in enumerate(poses):
            X = s * P @ Rf.T + tf
            c = 0.0
            if mask is not None and mask.sum() > 5 and (X[:, 2] > 0.05).all():
                uv = np.round((X @ K4.T)[:, :2] / X[:, 2:3]).astype(int)
                ok = (uv[:, 0] >= 0) & (uv[:, 0] < W4) & (uv[:, 1] >= 0) & (uv[:, 1] < H4)
                sil = np.zeros((H4, W4), np.uint8); sil[uv[ok, 1], uv[ok, 0]] = 1
                sil = cv2.dilate(sil, np.ones((3, 3), np.uint8)) > 0
                care = ~pers if pers is not None else np.ones_like(sil)
                un = ((sil | mask) & care).sum()
                c -= ARGS.w_img * (((sil & mask) & care).sum() / un if un else 0.0)
                diag["img_frames"] += i == 0
            if held:
                c += ARGS.w_hand * min(dh[i], 0.25) / 0.25
                diag["hand_frames"] += i == 0
            else:   # gravity: an unheld handle lies on the table, at the cover's lowest level (+ its own radius)
                lift = float(grips[i] @ n) - cover_low - 0.015
                c += ARGS.w_table * max(0.0, abs(lift) - 0.02) / 0.05
                diag["table_frames"] += i == 0
            cost[f, i] = c
    # Viterbi with a roll-change penalty
    dphi = np.abs(((PHI[:, None] - PHI[None]) + np.pi) % (2 * np.pi) - np.pi) / np.radians(10)
    trans = ARGS.w_smooth * dphi
    acc, back = cost[0].copy(), np.zeros((T, len(PHI)), int)
    for f in range(1, T):
        tot = acc[:, None] + trans
        back[f] = tot.argmin(0); acc = tot.min(0) + cost[f]
    path = np.zeros(T, int); path[-1] = int(acc.argmin())
    for f in range(T - 1, 0, -1):
        path[f - 1] = back[f, path[f]]
    Rn, tn = Rm.copy(), t.copy()
    for f in range(T):
        Rr = rots[path[f]]; Rn[f] = Rm[f] @ Rr; tn[f] = t[f] + s * Rm[f] @ (cc - Rr @ cc)
    d["object_rotation"], d["object_translation"] = Rn.astype(np.float32), tn.astype(np.float32)
    out = R0 / f"submissions/episodes_cand_{ARGS.name}"
    np.savez(out / f"{tag}.npz", **d)
    comp.export(out / f"{tag}_object.glb")
    steps = np.degrees(np.abs(((PHI[path][1:] - PHI[path][:-1]) + np.pi) % (2 * np.pi) - np.pi))
    return tag, {**info, **diag, "phi_changes": int((steps > 0).sum()), "max_phi_step_deg": float(steps.max()) if len(steps) else 0.0,
                 "phi_hist_deg": np.bincount(path, minlength=len(PHI)).tolist(),
                 "max_translation_change_cm": float(np.linalg.norm(tn - t, axis=1).max() * 100)}


def main():
    global ARGS, A, BASE_BODIES, FLOOR
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", default="Arc")
    ap.add_argument("--mesh", type=Path, required=True, help="SAM 3D Objects mesh of the roller built from the handle masks")
    ap.add_argument("--base", type=Path, default=ROOT / "submissions/episodes_cand_A", help="episodes whose cylinder track is kept")
    ap.add_argument("--bodies", type=Path, required=True, help="mhr_bodies.py output of the human (dense hands)")
    ap.add_argument("--floor", type=Path, default=ROOT / "outputs/physics/kappa_floor.json")
    ap.add_argument("--w_img", type=float, default=1.0)
    ap.add_argument("--w_hand", type=float, default=1.0)
    ap.add_argument("--w_table", type=float, default=1.0)
    ap.add_argument("--w_smooth", type=float, default=0.05)
    ap.add_argument("--hold_m", type=float, default=0.08, help="held when some roll puts the grip within this of a hand")
    ARGS = ap.parse_args()
    A, BASE_BODIES, FLOOR = ARGS.base, ARGS.bodies, json.loads(ARGS.floor.read_text())
    out = R0 / f"submissions/episodes_cand_{ARGS.name}"; out.mkdir(parents=True, exist_ok=True)
    for f in A.iterdir():
        if f.is_file():
            shutil.copy2(f, out / f.name)
    with ProcessPoolExecutor(3) as pool:
        rep = dict(pool.map(episode, (18, 19, 20)))
    (ROOT / "outputs/roller").mkdir(parents=True, exist_ok=True)
    (ROOT / f"outputs/roller/compose_report_{ARGS.name}.json").write_text(json.dumps({"args": {k: str(v) for k, v in vars(ARGS).items()}, "episodes": rep}, indent=1))
    for t_, r in rep.items():
        print(t_, json.dumps({k: v for k, v in r.items() if k != "phi_hist_deg"}))


if __name__ == "__main__":
    main()
