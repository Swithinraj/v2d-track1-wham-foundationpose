#!/usr/bin/env python3
"""Depth-reliability caches for episodes WITHOUT ground truth (the challenge; mhr_converter.sif, GPU). Per episode of
INPUT (the episodes the module corrects, e.g. episodes_bmq3k_in), the fields scripts/rel_cache.py reads, in the layouts of
the FORM-HOI caches (joint_cache_human.py / joint_cache_object.py / joint_cache_visual.py), with the same code:
  OUT/human/<P>_<tag>.npz      ours [T, 22, 3] (scored joints, camera frame, m), hands_ours [T, 128, 3], anchor (first scored
                               frame, as gru_depth_refit.py), valid (ones: no GT)
  OUT/object/<P>_<tag>_obj.npz obj_R / obj_t / obj_s (INPUT's object track), surf_ours (512 samples), raw_t / raw_ok
                               (FoundationPose raw track brought to our frame by the median |ours| / |raw|, as
                               joint_cache_object.py)
  OUT/visual/<P>_<tag>_vis.npz K, kp, kpc (ViTPose landmarks of outputs/kp2d_vpx, normalised as joint_cache_visual.py)
P = PREFIX (default: the root directory's name, as joint_cache_visual.py / depth_cues.py name their outputs).
Usage: rel_deploy_prep.py INPUT OUT [PREFIX]   (TRACK1_ROOT = the challenge root)
"""
import json
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
from joint_cache_object import surface, quat_to_R  # noqa: E402
from joint_deploy_prep import SC  # noqa: E402


def main():
    inp, out = Path(sys.argv[1]), Path(sys.argv[2]); P = sys.argv[3] if len(sys.argv) > 3 else ROOT.name
    for d in ("human", "object", "visual"):
        (out / d).mkdir(parents=True, exist_ok=True)
    S = json.loads((ROOT / "outputs/hand_fix/episodes_repro_depth_refine_meshes/scored_frames.json").read_text())
    mhr = MHR(str(MODEL), "cuda", chunk=128, precision="float32")
    hands = np.concatenate([h[::4] for h in hand_vertex_indices()])
    f32 = lambda x: np.asarray(x, np.float32)  # noqa: E731
    for f in sorted(inp.glob("episode_??????.npz")):
        tag = f.stem; ep = int(tag[8:]); d = np.load(f); T = len(d["pose"])
        anchor = min(x for x in S[str(ep)] if x < T)
        z = torch.as_tensor(np.concatenate([d["scales"], d["shape"]])[None], dtype=torch.float64, device="cuda")
        V, J = mhr.run(torch.as_tensor(d["pose"], dtype=torch.float64, device="cuda"), z)
        np.savez(out / "human" / f"{P}_{tag}.npz", ours=f32(J[:, SC].cpu().numpy()), hands_ours=f32(V[:, hands].cpu().numpy() / 1000.0),
                 anchor=anchor, valid=np.ones(T, bool))
        oR, ot = d["object_rotation"][:T].astype(float), d["object_translation"][:T].astype(float)
        obj = ROOT / "outputs/object" / tag
        raw = [json.loads((obj / f"foundationpose/raw/{i:06d}.json").read_text()) for i in range(T)]
        rR = np.stack([quat_to_R(r["rotation"]) for r in raw]); rt = np.array([r["translation"] for r in raw], float)
        fin = np.isfinite(rt).all(1) & np.isfinite(rR).reshape(T, -1).all(1) & (np.linalg.norm(np.nan_to_num(rt), axis=1) > 1e-6)
        if fin.sum() >= 10:
            rt *= np.median(np.linalg.norm(ot[fin], axis=1)) / np.median(np.linalg.norm(rt[fin], axis=1))
        else:
            fin[:] = False
        rt[~fin] = np.nan; rR[~fin] = np.nan
        ang = np.degrees(np.arccos(np.clip((np.einsum("tij,tij->t", np.nan_to_num(rR), oR) - 1) / 2, -1, 1)))
        raw_ok = np.full(T, float(fin.any() and np.median(ang[fin]) < 30.0))
        so, no = surface(inp / f"{tag}_object.glb")
        np.savez(out / "object" / f"{P}_{tag}_obj.npz", obj_R=f32(oR), obj_t=f32(ot), obj_s=float(d["object_scale"]), surf_ours=so,
                 norm_ours=no, raw_t=f32(rt), raw_ok=f32(raw_ok))
        Kj = json.loads((ROOT / f"outputs/camera/{tag}/intrinsics.json").read_text()); K = np.array([Kj["fx"], Kj["fy"], Kj["cx"], Kj["cy"]])
        pose = np.nan_to_num(np.load(ROOT / f"outputs/kp2d_vpx/{tag}.npz")["pose"][:T, 0].astype(float))
        if len(pose) < T:
            pose = np.concatenate([pose, np.zeros((T - len(pose),) + pose.shape[1:])])
        kp = np.stack([(pose[:, :, 0] - K[2]) / K[0], (pose[:, :, 1] - K[3]) / K[1]], -1)
        np.savez(out / "visual" / f"{P}_{tag}_vis.npz", K=f32(K), kp=np.nan_to_num(f32(kp)), kpc=np.nan_to_num(f32(pose[:, :, 3])))
        print(P, tag, T, "frames, anchor", anchor, "raw_ok", int(raw_ok[0]), flush=True)


if __name__ == "__main__":
    main()
