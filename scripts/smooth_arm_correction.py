#!/usr/bin/env python3
"""Temporal low-pass of the arm part of a contact-solve correction (mhr_converter.sif, GPU).

human_contact_solver.py corrects the pose with two kinds of columns: arm columns, which move the wrist/elbow (scored
joints), and hand columns, which move only hand vertices. Its per-frame arm correction adds scored-joint acceleration
(ACC-H). Here delta = pose_solved - pose_base is split per episode: columns that move a scored joint (finite
differences at the first scored frame) are arm columns; their delta is low-passed in time (Gaussian, --sigma frames,
edge-padded); hand columns keep the solved value; the first scored frame (where the scorer fits its Sim(3)) stays exactly
the solved pose. A hand-only re-solve (human_contact_solver.py --no_arm) afterwards restores non-penetration with
hand columns only.
Writes <out>/episode_*.npz (+ the other files of --solved) and smooth_arm_correction_report.json (scored-joint
self-acceleration before/after per episode).
Usage (container): smooth_arm_correction.py --solved DIR --base DIR --out DIR [--sigma 4]
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from human_contact_solver import SCORED_J, T1, Body


def selfacc(J, frames):
    fr = np.array(frames); v = []
    for b in np.split(fr, np.flatnonzero(np.diff(fr) != 1) + 1):
        if len(b) >= 3:
            v.append(np.linalg.norm(np.diff(J[b], n=2, axis=0), axis=-1).mean(1))
    return float(np.concatenate(v).mean() * 100) if v else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solved", type=Path, required=True, help="human_contact_solver.py output")
    ap.add_argument("--base", type=Path, required=True, help="its --human input")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sigma", type=float, default=4.0)
    ap.add_argument("--scored_frames", type=Path, default=T1 / "outputs/hand_fix/episodes_repro_depth_refine_meshes/scored_frames.json")
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    for f in args.solved.iterdir():
        if f.is_file() and not f.name.endswith(".npz"):
            shutil.copy2(f, args.out / f.name)
    S = json.loads(args.scored_frames.read_text())
    body = Body(str(T1 / "weights/mhr/assets/mhr_model.pt")); rep = {}
    for ep in range(30):
        tag = f"episode_{ep:06d}"; a = dict(np.load(args.solved / f"{tag}.npz")); b = np.load(args.base / f"{tag}.npz")
        delta = a["pose"].astype(np.float64) - b["pose"].astype(np.float64); changed = np.flatnonzero(np.abs(delta).max(0) > 0)
        sf = [f for f in S[str(ep)] if f < len(delta)]; anchor = min(sf)
        if not len(changed):
            np.savez(args.out / f"{tag}.npz", **a); rep[tag] = {"changed_columns": 0}; continue
        z = torch.tensor(np.concatenate([a["scales"], a["shape"]])[None], dtype=torch.float32, device="cuda")
        p0 = torch.tensor(a["pose"][anchor], dtype=torch.float32, device="cuda")
        P = p0[None].repeat(len(changed) + 1, 1); P[1 + torch.arange(len(changed)), torch.as_tensor(changed, device="cuda")] += 0.05
        with torch.no_grad():
            _, Jfd = body(P, z, chunk=len(P))
        moves = (Jfd[1:, SCORED_J] - Jfd[:1, SCORED_J]).norm(dim=-1).amax(1).cpu().numpy()
        arm = changed[moves > 1e-6]
        k = np.arange(-int(3 * args.sigma), int(3 * args.sigma) + 1); g = np.exp(-k ** 2 / (2 * args.sigma ** 2)); g /= g.sum()
        new = a["pose"].astype(np.float64).copy()
        for c in arm:
            sm = np.convolve(np.pad(delta[:, c], len(k) // 2, mode="edge"), g, mode="valid")
            new[:, c] = b["pose"][:, c] + sm
        new[anchor] = a["pose"][anchor]                                                        # anchor exactly solved
        with torch.no_grad():
            _, J0 = body(torch.tensor(a["pose"], dtype=torch.float32, device="cuda"), z)
            _, J1 = body(torch.tensor(new, dtype=torch.float32, device="cuda"), z)
        J0 = J0[:, SCORED_J].cpu().numpy().astype(float); J1 = J1[:, SCORED_J].cpu().numpy().astype(float)
        rep[tag] = {"changed_columns": int(len(changed)), "arm_columns": int(len(arm)), "selfacc_solved": selfacc(J0, sf),
                    "selfacc_new": selfacc(J1, sf), "scored_joint_change_cm_mean": float(np.linalg.norm(J1 - J0, axis=-1)[sf].mean() * 100)}
        a["pose"] = new.astype(a["pose"].dtype); np.savez(args.out / f"{tag}.npz", **a)
        print(tag, json.dumps(rep[tag]), flush=True)
    done = [r for r in rep.values() if "selfacc_new" in r]
    print("episodes changed", len(done), "| mean scored-joint self-acc over changed eps: solved %.5f -> %.5f"
          % (np.mean([r["selfacc_solved"] for r in done]), np.mean([r["selfacc_new"] for r in done])))
    (args.out / "smooth_arm_correction_report.json").write_text(json.dumps({"sigma": args.sigma, "episodes": rep}, indent=1))


if __name__ == "__main__":
    main()
