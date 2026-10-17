"""Shared helpers of the human refinement stage (pipeline/stages/human_refine.sh), submission venv.

Geometry, grasp-window loading, the scorer's penetration measure, camera projection and 2D keypoint
matching used by grasp_windows.py, hand_anchors.py, wrist_fusion.py and root_lateral.py.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("TRACK1_ROOT") or Path(__file__).resolve().parents[1])
sys.path.insert(0, str(ROOT / "scripts"))
import resolve_hand_penetration as rp  # noqa: E402
from v2dlb.mesh_budget import budget_mesh  # noqa: E402

FLIP = np.array([1.0, -1.0, -1.0])     # MHR root translation axes vs world (pose[:, :3] is in decimetres)
HAND_SPLIT = 2318                      # dense hand vertices per hand: 0 = right wrist subtree (joint 42), 1 = left (78)
# (MHR joint, MediaPipe pose landmark); MediaPipe left = the person's left
PAIRS = {"wrist": ((78, 15), (42, 16)), "elbow": ((76, 13), (40, 14)), "shoulder": ((75, 11), (39, 12)),
         "hip": ((2, 23), (18, 24)), "knee": ((3, 25), (19, 26)), "ankle": ((4, 27), (20, 28))}
MATCH = [p for g in ("wrist", "elbow", "shoulder", "hip", "knee") for p in PAIRS[g]]
_SCORED = None


def scored_and_budget():
    global _SCORED
    if _SCORED is None:
        _SCORED = rp.scored_frames_and_budget()
    return _SCORED


def alignment(tag: str) -> dict:
    """Camera alignment of an episode (R_cw, t_cw, depth_scale, intrinsics). TRACK1_ALIGNMENT_DIR overrides the
    default outputs/alignment, e.g. for a human branch whose world frame is the camera itself."""
    root = Path(os.environ.get("TRACK1_ALIGNMENT_DIR") or ROOT / "outputs/alignment")
    return json.loads((root / tag / "alignment.json").read_text())


def load_contact(ep: int, contact_dir: Path, pre_dir: Path, hands_dir: Path) -> dict:
    """Episode for contact analysis: episode data from contact_dir, the budgeted object mesh, and the dense
    hands of the pre-correction body (hands_dir, from pre_dir) moved by the root change between pre_dir
    and contact_dir (a pure translation of the body)."""
    tag = f"episode_{ep:06d}"
    frames, budget = scored_and_budget()
    data = dict(np.load(contact_dir / f"{tag}.npz"))
    pre_root = np.load(pre_dir / f"{tag}.npz")["pose"][:, :3].astype(np.float64)
    shift = (data["pose"][:, :3].astype(np.float64) - pre_root) / 10.0 * FLIP
    hands = np.load(hands_dir / f"{tag}.npz")["hands"].astype(np.float64) + shift[:, None]
    v, f = budget_mesh(contact_dir / f"{tag}_object.glb", *budget)
    T = len(data["pose"])
    return {"ep": ep, "tag": tag, "data": data, "v": np.asarray(v, float), "f": f, "hands": hands,
            "R": data["object_rotation"].astype(np.float64), "t": data["object_translation"].astype(np.float64),
            "s": float(data["object_scale"]), "T": T, "scored": np.array([x for x in frames[ep] if x < T])}


def kabsch(P, Q):
    """R, t minimising |R P + t - Q| (rows are points)."""
    cp, cq = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - cp).T @ (Q - cq))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, cq - R @ cp


def chordal_mean(Rs):
    U, _, Vt = np.linalg.svd(Rs.sum(0))
    M = U @ Vt
    return M if np.linalg.det(M) > 0 else U @ np.diag([1, 1, -1]) @ Vt


def angle_deg(Ra, Rb):
    M = np.swapaxes(Ra, -1, -2) @ Rb
    return np.degrees(np.arccos(np.clip((np.trace(M, axis1=-2, axis2=-1) - 1) / 2, -1, 1)))


def runs(mask, min_len, close_gap):
    """Contiguous True runs after closing gaps of <= close_gap frames, keeping runs >= min_len."""
    m = mask.copy()
    idx = np.flatnonzero(m)
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= close_gap + 1:
            m[a:b] = True
    out, start = [], None
    for i, x in enumerate(np.r_[m, False]):
        if x and start is None:
            start = i
        elif not x and start is not None:
            if i - start >= min_len:
                out.append((start, i))
            start = None
    return out


# --- penetration (the scorer's winding-number depth against the budgeted mesh) ---------------------------
MARGIN_M = 0.002


def penetration(points, R, t, s, v, f):
    return rp.depths(points, R, t, s, v, f)


def clear_frame(points, R, t, s, v, f):
    """Smallest object translation (found greedily) that removes the penetration of all points."""
    offset = np.zeros(3)
    total = penetration(points, R, t, s, v, f).sum()
    if total <= 0:
        return offset
    for _ in range(10):
        grad = np.zeros(3)
        for k in range(3):
            e = np.zeros(3); e[k] = 0.003
            grad[k] = (penetration(points, R, t + offset + e, s, v, f).sum()
                       - penetration(points, R, t + offset - e, s, v, f).sum()) / 0.006
        if not np.any(grad):
            break
        direction = -grad / np.linalg.norm(grad)
        best = (total, offset)
        for step in (0.002, 0.005, 0.01, 0.02, 0.04):
            trial = offset + step * direction
            value = penetration(points, R, t + trial, s, v, f).sum()
            if value < best[0]:
                best = (value, trial)
        if best[0] >= total:
            break
        total, offset = best
        if total <= 0:
            break
    if np.linalg.norm(offset) > 0:
        offset += MARGIN_M * offset / np.linalg.norm(offset)
    return offset


# --- camera and 2D keypoints ------------------------------------------------------------------------------
def camera(J, a):
    """Track 1 frame -> MoGe camera frame (inverse of the assembly mapping)."""
    return float(a["depth_scale"]) * (J @ np.asarray(a["R_cw"]).T + np.asarray(a["t_cw"]))


def project(J, a):
    R, t, k, ds = np.asarray(a["R_cw"]), np.asarray(a["t_cw"]), a["intrinsics"], float(a["depth_scale"])
    c = ds * (J @ R.T + t)
    return np.stack([k["fx"] * c[..., 0] / c[..., 2] + k["cx"], k["fy"] * c[..., 1] / c[..., 2] + k["cy"]], -1)


def actor_landmarks(mp, uv):
    """Per frame, the MediaPipe person (of up to 3) closest to our projected joints; [T, 33, 3] x, y, visibility."""
    T = min(len(mp), len(uv))
    out = np.full((T, 33, 3), np.nan)
    for f in range(T):
        d = [np.nanmedian([np.linalg.norm(mp[f, i, l, :2] - uv[f, j]) for j, l in MATCH]) for i in range(3)]
        if np.isfinite(d).any() and np.nanmin(d) < 60:
            i = int(np.nanargmin(d)); out[f, :, :2] = mp[f, i, :, :2]; out[f, :, 2] = mp[f, i, :, 3]
    return out
