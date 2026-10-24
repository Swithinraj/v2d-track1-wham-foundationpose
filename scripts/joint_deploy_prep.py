#!/usr/bin/env python3
"""Deployment inputs for ext/joint_cache_visual.py on episodes WITHOUT ground truth (the challenge; mhr_converter.sif,
GPU, plus the submission venv's trimesh). Per episode of INPUT (a candidate's episode directory): our 22 scored joints and
128 hand points (MHR forward; layout of joint_cache_human.py) and our object track + surface samples (layout of
joint_cache_object.py), written as OUT/<root>_<episode>.npz and OUT/<root>_<episode>_obj.npz. joint_cache_visual.py
then builds the crops with the same code as in training. Usage: joint_deploy_prep.py INPUT OUT ROOT_NAME
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(os.environ.get("TRACK1_ROOT") or "/track1")
sys.path.insert(0, "/track1/official/v2d_submission_kit/tools/track1")
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mesh_to_mhr_params import MHR  # noqa: E402
from mhr_bodies import MODEL  # noqa: E402
from extract_mhr_hands import hand_vertex_indices  # noqa: E402
from joint_cache_object import surface  # noqa: E402

SC = [1, 2, 18, 35, 3, 19, 36, 4, 20, 37, 8, 24, 110, 74, 38, 113, 75, 39, 76, 40, 78, 42]


def main():
    inp, out, name = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    out.mkdir(parents=True, exist_ok=True)
    mhr = MHR(str(MODEL), "cuda", chunk=128, precision="float32")
    hands = np.concatenate([h[::4] for h in hand_vertex_indices()])
    for f in sorted(inp.glob("episode_*.npz")):
        tag = f.stem; d = np.load(f)
        z = torch.as_tensor(np.concatenate([d["scales"], d["shape"]])[None], dtype=torch.float64, device="cuda")
        V, J = mhr.run(torch.as_tensor(d["pose"], dtype=torch.float64, device="cuda"), z)
        f32 = lambda x: np.asarray(x, np.float32)  # noqa: E731
        np.savez(out / f"{name}_{tag}.npz", ours=f32(J[:, SC].cpu().numpy()), hands_ours=f32(V[:, hands].cpu().numpy() / 1000.0))
        so, no = surface(inp / f"{tag}_object.glb")
        np.savez(out / f"{name}_{tag}_obj.npz", obj_R=f32(d["object_rotation"]), obj_t=f32(d["object_translation"]),
                 obj_s=float(d["object_scale"]), surf_ours=so, norm_ours=no)
        print(name, tag, len(d["pose"]), "frames", flush=True)


if __name__ == "__main__":
    main()
