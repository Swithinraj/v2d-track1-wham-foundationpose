#!/usr/bin/env python3
"""FORM-HOI local scoring, step 1 (mhr_converter.sif, GPU): MHR arrays of predicted episodes and of the GT.

Per episode, on its scored frames (ROOT/outputs/hand_fix/episodes_repro_depth_refine_meshes/scored_frames.json):
  joints   the 22 scored joints (MHR_TABLE3 order), metres, Track 1 frame
  align    every 20th MHR vertex (stand-in for the scorer's hidden "alignment" role)
  body64   64 farthest-point-sampled non-hand vertices of the neutral body (stand-in for the hidden "body" role)
  dense    2000 farthest-point-sampled non-hand vertices (lower-variance CD-H for comparing variants)
  hands    the 512 hand points of extract_mhr_hands.py (our approximation of the scorer's PEN points; predictions only)
GT: mhr_params_mv.pt (mhr_model_params = 136 pose + 68 scales, shape_params), which the kit MHR reproduces exactly.
Usage: fh_export.py ROOT PRED_DIR|gt OUT_DIR
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/track1/official/v2d_submission_kit/tools/track1")
sys.path.insert(0, "/track1/scripts")
from mesh_to_mhr_params import MHR  # noqa: E402
from mhr_bodies import hand_vertex_sets  # noqa: E402
from extract_mhr_hands import hand_vertex_indices  # noqa: E402

SC = [1, 2, 18, 35, 3, 19, 36, 4, 20, 37, 8, 24, 110, 74, 38, 113, 75, 39, 76, 40, 78, 42]


def fps(P, k, seed=0):
    idx = [int(np.random.default_rng(seed).integers(len(P)))]
    d = np.linalg.norm(P - P[idx[0]], axis=1)
    for _ in range(k - 1):
        idx.append(int(d.argmax())); d = np.minimum(d, np.linalg.norm(P - P[idx[-1]], axis=1))
    return np.array(idx)


def main():
    root, src, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    mhr = MHR("/track1/weights/mhr/assets/mhr_model.pt", "cuda", chunk=128, precision="float32")
    roles = out.parent / "roles.npz"
    if roles.exists():
        r = np.load(roles); align, body64, dense = r["align"], r["body64"], r["dense"]
    else:
        V0, _ = mhr.run(torch.zeros(1, 136, dtype=torch.float64, device="cuda"), torch.zeros(1, 113, dtype=torch.float64, device="cuda"))
        V0 = V0[0].cpu().numpy() / 1000.0
        hands = np.concatenate(hand_vertex_sets())
        nonhand = np.setdiff1d(np.arange(len(V0)), hands)
        align = np.arange(0, len(V0), 20)
        body64 = nonhand[fps(V0[nonhand], 64)]; dense = nonhand[fps(V0[nonhand], 2000, seed=1)]
        np.savez(roles, align=align, body64=body64, dense=dense)
    hand512 = np.concatenate(hand_vertex_indices())
    scored = json.loads((root / "outputs/hand_fix/episodes_repro_depth_refine_meshes/scored_frames.json").read_text())
    for ep in range(30):
        tag = f"episode_{ep:06d}"
        if src == "gt":
            g = torch.load(root / "gt" / tag / "mhr_params_mv.pt", map_location="cpu", weights_only=False)
            pose = g["mhr_model_params"][:, :136].double().cuda()
            z = torch.cat([g["mhr_model_params"][:1, 136:], g["shape_params"][:1]], 1).double().cuda()
        else:
            p = Path(src) / f"{tag}.npz"
            if not p.exists():
                continue
            d = np.load(p)
            pose = torch.as_tensor(d["pose"], dtype=torch.float64, device="cuda")
            z = torch.as_tensor(np.concatenate([d["scales"], d["shape"]])[None], dtype=torch.float64, device="cuda")
        fr = np.array([f for f in scored[str(ep)] if f < len(pose)])
        V, J = mhr.run(pose[fr], z)
        V = V.cpu().numpy() / 1000.0; J = J.cpu().numpy()
        np.savez_compressed(out / f"{tag}.npz", frames=fr, joints=J[:, SC].astype(np.float32), align=V[:, align].astype(np.float32),
                            body64=V[:, body64].astype(np.float32), dense=V[:, dense].astype(np.float32),
                            hands=V[:, hand512].astype(np.float32))
        print(tag, len(fr), "frames", flush=True)


if __name__ == "__main__":
    main()
