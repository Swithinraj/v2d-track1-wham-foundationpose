#!/usr/bin/env python3
"""Grip shape from the image: fit each hand's finger (and wrist-rotation) parameters to MediaPipe's 21 hand landmarks
(mhr_converter.sif, GPU).

SAM 3D Body guesses the fingers, and in a grip they are hidden behind the object: the guess is often too tight, so the
fingers cut through handles. MediaPipe Hands (320 px crops around each wrist, pipeline/stages/keypoints2d.sh) sees them
in most frames but was never used. Per frame and hand with a confident MediaPipe hand (score > MIN_SCORE, the actor
chosen against our projected joints): Levenberg-Marquardt on the hand's free columns (they move this hand but none of
the 22 scored joints, so ACC-H cannot change) minimising
   sum_k |project(MHR joint_k) - landmark_k|^2 / SIG_PX^2  +  |p - p0|^2 / SIG_RAD^2
over the 21 landmarks (wrist, thumb CMC/MCP/IP/tip, MCP/PIP/DIP/tip of each finger; anatomical map below).
Run after the human refinement, before the hand-side contact fix (which then clears penetration from this grip).
Writes <out>/episode_*.npz (+ meshes, shared files) and finger_fit_report[_<episodes>].json.
Usage: finger_fit.py --input DIR --out DIR [--episodes ...] [--min_score 0.8]
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

T1 = Path("/track1")
sys.path.insert(0, str(T1 / "official/v2d_submission_kit/tools/track1"))
sys.path.insert(0, str(T1 / "scripts"))
from mesh_to_mhr_params import MHR  # noqa: E402
from mhr_bodies import MODEL, hand_vertex_sets  # noqa: E402

SCORED_J = [1, 2, 18, 35, 3, 19, 36, 4, 20, 37, 8, 24, 110, 74, 38, 113, 75, 39, 76, 40, 78, 42]
MATCH = [(78, 15), (42, 16), (76, 13), (40, 14), (75, 11), (39, 12), (2, 23), (18, 24), (3, 25), (19, 26)]
# MediaPipe landmark k -> MHR joint (left hand = person's left = MediaPipe crop side 0; right = 42-based, side 1)
LEFT = [78, 97, 98, 99, 100, 92, 93, 94, 95, 88, 89, 90, 91, 84, 85, 86, 87, 80, 81, 82, 83]
RIGHT = [42, 61, 62, 63, 64, 56, 57, 58, 59, 52, 53, 54, 55, 48, 49, 50, 51, 44, 45, 46, 47]
SIG_PX, SIG_RAD, ITERS, EPS = 4.0, 0.35, 6, 1e-3
ARGS = None


def joints(mhr, pp, ident, chunk=512):
    """MHR joints only (metres), in chunks (the vertices of thousands of poses would not fit)."""
    return torch.cat([mhr.run(pp[i:i + chunk], ident)[1] for i in range(0, len(pp), chunk)])


def free_columns(mhr, pose, ident, hand_idx, other_idx):
    p = pose[None].repeat(136, 1)
    p[torch.arange(136), torch.arange(136)] += 0.05
    v0, j0 = mhr.run(pose[None], ident)
    v, j = mhr.run(p, ident)
    dj = (j[:, SCORED_J] - j0[:, SCORED_J]).norm(dim=-1).amax(1)
    dv = (v[:, hand_idx] - v0[:, hand_idx]).norm(dim=-1).amax(1) / 1000.0
    do = (v[:, other_idx] - v0[:, other_idx]).norm(dim=-1).amax(1) / 1000.0
    return [c for c in range(136) if dj[c] < 1e-6 and dv[c] > 1e-4 and do[c] < 1e-6]


def episode(ep, mhr, sets):
    tag = f"episode_{ep:06d}"
    d = dict(np.load(ARGS.input / f"{tag}.npz"))
    a = json.loads((T1 / "outputs/alignment_sam" / tag / "alignment.json").read_text())
    k, ds = a["intrinsics"], float(a["depth_scale"])
    z = np.load(T1 / "outputs/mediapipe" / f"{tag}.npz")
    mp, hands, hs = z["pose"].astype(np.float64), z["hands"].astype(np.float64), z["hand_score"].astype(np.float64)
    pose = torch.as_tensor(d["pose"], dtype=torch.float64, device="cuda")
    ident = torch.as_tensor(np.concatenate([d["scales"], d["shape"]])[None], dtype=torch.float64, device="cuda")
    T = min(len(pose), len(mp))
    J0 = joints(mhr, pose, ident)

    def project(J):          # J [..., 3] torch, Track 1 frame = camera frame scaled by 1/ds
        c = ds * J
        return torch.stack([k["fx"] * c[..., 0] / c[..., 2] + k["cx"], k["fy"] * c[..., 1] / c[..., 2] + k["cy"]], -1)

    uv0 = project(J0).cpu().numpy()
    actor = np.full(T, -1)
    for f in range(T):
        dd = [np.nanmedian([np.linalg.norm(mp[f, i, l, :2] - uv0[f, j]) for j, l in MATCH]) for i in range(3)]
        if np.isfinite(dd).any() and np.nanmin(dd) < 60:
            actor[f] = int(np.nanargmin(dd))
    hidx = [torch.as_tensor(s, device="cuda") for s in sets]          # 0 = right (42), 1 = left (78)
    res = {}
    for name, side, jmap, own, other in (("left", 0, LEFT, 1, 0), ("right", 1, RIGHT, 0, 1)):
        frames = [f for f in range(T) if actor[f] >= 0 and hs[f, actor[f], side] > ARGS.min_score
                  and np.isfinite(hands[f, actor[f], side, :, :2]).all()]
        if not frames:
            res[name] = {"frames": 0}
            continue
        cols = free_columns(mhr, pose[frames[0]], ident, hidx[own], hidx[other])
        fr = torch.as_tensor(frames, device="cuda")
        L = torch.as_tensor(np.stack([hands[f, actor[f], side, :, :2] for f in frames]), device="cuda")    # [F,21,2]
        jm = torch.as_tensor(jmap, device="cuda")
        p = pose[fr].clone()
        p0 = p[:, cols].clone()
        nc = len(cols)
        err0 = (project(J0[fr][:, jm]) - L).norm(dim=-1).median().item()
        mu = torch.full((len(frames),), 1e-2, dtype=torch.float64, device="cuda")

        def residual(pp):
            J = joints(mhr, pp, ident)
            r_img = ((project(J[:, jm]) - L) / SIG_PX).reshape(len(pp), -1)
            return torch.cat([r_img, (pp[:, cols] - p0) / SIG_RAD], 1)

        for _ in range(ITERS):
            r = residual(p)
            batch = p[:, None].repeat(1, nc + 1, 1)
            batch[:, 1:][:, torch.arange(nc), torch.as_tensor(cols, device="cuda")] += EPS
            Jb = joints(mhr, batch.reshape(-1, 136), ident).reshape(len(frames), nc + 1, -1, 3)
            img = ((project(Jb[:, :, jm]) - L[:, None]) / SIG_PX).reshape(len(frames), nc + 1, -1)
            Jimg = (img[:, 1:] - img[:, :1]).transpose(1, 2) / EPS                               # [F, 42, nc]
            Jac = torch.cat([Jimg, torch.eye(nc, dtype=torch.float64, device="cuda")[None].expand(len(frames), -1, -1) / SIG_RAD], 1)
            JtJ = Jac.transpose(1, 2) @ Jac
            g = Jac.transpose(1, 2) @ r[..., None]
            c0 = (r * r).sum(1)
            for _ in range(6):
                step = -torch.linalg.solve(JtJ + mu[:, None, None] * torch.diag_embed(torch.diagonal(JtJ, dim1=1, dim2=2) + 1e-9), g)[..., 0]
                trial = p.clone()
                trial[:, cols] += step
                c1 = (residual(trial) ** 2).sum(1)
                ok = c1 < c0
                p = torch.where(ok[:, None], trial, p)
                mu = torch.where(ok, mu * 0.3, mu * 10)
                if ok.all():
                    break
        pose[fr] = p
        err1 = (project(joints(mhr, p, ident)[:, jm]) - L).norm(dim=-1).median().item()
        res[name] = {"frames": len(frames), "free_columns": nc, "landmark_px_median": [err0, err1]}
    d["pose"] = pose.cpu().numpy().astype(np.float32)
    J1 = joints(mhr, torch.as_tensor(d["pose"], dtype=torch.float64, device="cuda"), ident)
    dj = float((J1[:, SCORED_J] - J0[:, SCORED_J]).norm(dim=-1).max())
    if dj > 1e-4:
        raise SystemExit(f"{tag}: scored joints moved by {dj} m")
    np.savez(ARGS.out / f"{tag}.npz", **d)
    return {"episode": tag, **res, "scored_joint_change_m": dj}


def main():
    global ARGS
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--episodes", type=int, nargs="*", default=list(range(30)))
    ap.add_argument("--min_score", type=float, default=0.8)
    ARGS = ap.parse_args()
    ARGS.out.mkdir(parents=True, exist_ok=True)
    tags = {f"episode_{ep:06d}" for ep in ARGS.episodes}
    for src in ARGS.input.iterdir():
        if src.is_file() and (src.name[:14] in tags if src.name.startswith("episode_") else not (ARGS.out / src.name).exists()):
            shutil.copy2(src, ARGS.out / src.name)
    mhr = MHR(str(MODEL), "cuda", chunk=256, precision="float64")
    sets = hand_vertex_sets()
    report = []
    for ep in ARGS.episodes:
        r = episode(ep, mhr, sets)
        report.append(r)
        print(r["episode"], json.dumps({h: r[h] for h in ("left", "right")}), flush=True)
    name = "finger_fit_report" + ("_" + "_".join(map(str, ARGS.episodes)) if len(ARGS.episodes) < 30 else "") + ".json"
    (ARGS.out / name).write_text(json.dumps({"min_score": ARGS.min_score, "episodes": report}, indent=1))
    print("PASS: finger fit", len(report), "episodes")


if __name__ == "__main__":
    main()
