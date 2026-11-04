#!/usr/bin/env python3
"""Hand points of the SAM 3D Body human and its camera alignment, for the object stages (mhr_converter.sif, GPU).

The symmetric-object tracker (track_symmetric_object.py) uses the hands as a contact prior. Here they come from the
identity-fixed refit (outputs/sam3d_body/<episode>/refit.npz) through the kit's MHR wrapper: 512 points (the
extract_mhr_hands.py roles) in the static camera's frame, SAM 3D Body metric -> outputs/hands/<episode>.npz.
outputs/alignment_sam/<episode>/alignment.json maps that frame to MoGe's camera: R_cw = I, t_cw = 0,
depth_scale = the episode's MoGe/SAM depth scale (sam3d_body_identity.py), intrinsics = outputs/camera.
Usage: sam3d_body_hands.py --episodes E [...]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

T1 = Path("/track1")
sys.path.insert(0, str(T1 / "official/v2d_submission_kit/tools/track1"))
sys.path.insert(0, str(T1 / "scripts"))
from mesh_to_mhr_params import MHR  # noqa: E402
from extract_mhr_hands import hand_vertex_indices  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, nargs="+", required=True)
    args = ap.parse_args()
    mhr = MHR(str(T1 / "weights/mhr/assets/mhr_model.pt"), "cuda", chunk=256, precision="float32")
    idx = torch.as_tensor(np.concatenate(hand_vertex_indices()), device="cuda")
    for ep in args.episodes:
        tag = f"episode_{ep:06d}"
        work = T1 / "outputs/sam3d_body" / tag
        r = np.load(work / "refit.npz")
        pose = torch.as_tensor(r["pose"], dtype=torch.float64, device="cuda")
        ident = torch.as_tensor(r["identity"][None], dtype=torch.float64, device="cuda")
        hands = (mhr.run(pose, ident)[0][:, idx] / 1000.0).cpu().numpy().astype(np.float32)
        out = T1 / "outputs/hands"
        out.mkdir(parents=True, exist_ok=True)
        np.savez(out / f"{tag}.npz", hands=hands)
        s = float(np.load(work / "prepare.npz")["depth_scale"])
        k = json.loads((T1 / "outputs/camera" / tag / "intrinsics.json").read_text())
        a = T1 / "outputs/alignment_sam" / tag
        a.mkdir(parents=True, exist_ok=True)
        (a / "alignment.json").write_text(json.dumps({"R_cw": np.eye(3).tolist(), "t_cw": [0.0, 0.0, 0.0], "depth_scale": s,
                                                      "intrinsics": k, "note": "SAM 3D Body camera world"}))
        print(f"{tag}: hands {hands.shape}, depth scale {s:.3f}", flush=True)


if __name__ == "__main__":
    main()
