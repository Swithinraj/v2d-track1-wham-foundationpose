#!/usr/bin/env python3
"""Joint human-object refiner cache, human part (mhr_converter.sif, GPU). Same as gru_cache_export.py (22 scored joints of
ours / raw / GT-in-our-frame, ViTPose confidence, mask visibility, QC mask, anchor) plus what the interaction model needs:
  hands_ours / hands_raw / hands_gt   64 points per hand (every 4th of extract_mhr_hands' 256), [T, 128, 3], our frame;
                                      block 0 = wrist joint 42 (scored joint 21), block 1 = wrist joint 78 (scored joint 20)
  sim3_s, sim3_R, sim3_t              the evaluator's similarity (ours -> GT, b = s a @ R + t), so the object part maps
                                      the GT objects into our frame with the very same transform
Units: metres. Writes OUT/<root>_<episode>.npz. Usage: joint_cache_human.py OUT ROOT_DIR [ROOT_DIR ...]
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/track1/official/v2d_submission_kit/tools/track1")
sys.path.insert(0, "/track1/official/v2d_submission_kit")
sys.path.insert(0, "/track1/scripts")
from mesh_to_mhr_params import MHR  # noqa: E402
from extract_mhr_hands import hand_vertex_indices  # noqa: E402

SC = [1, 2, 18, 35, 3, 19, 36, 4, 20, 37, 8, 24, 110, 74, 38, 113, 75, 39, 76, 40, 78, 42]
LMK = {2: 23, 18: 24, 3: 25, 19: 26, 4: 27, 20: 28, 75: 11, 39: 12, 76: 13, 40: 14, 78: 15, 42: 16}
OURS, RAW = "episodes_q3_in", "episodes_repro_depth"


def sim3(A, B):
    ma, mb = A.mean(0), B.mean(0); A0, B0 = A - ma, B - mb
    U, sv, Vt = np.linalg.svd(A0.T @ B0); d = np.sign(np.linalg.det(U @ Vt)); D = np.diag([1, 1, d])
    R = U @ D @ Vt; s = (sv * np.diag(D)).sum() / (A0 ** 2).sum()
    return s, R, mb - s * ma @ R


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    mhr = MHR("/track1/weights/mhr/assets/mhr_model.pt", "cuda", chunk=128, precision="float32")
    align_idx = np.arange(0, 18439, 20)
    hands = np.concatenate([h[::4] for h in hand_vertex_indices()])            # [128]: wrist 42's 64, then wrist 78's 64
    for root in map(Path, sys.argv[2:]):
        S = json.loads((root / "outputs/hand_fix/episodes_repro_depth_refine_meshes/scored_frames.json").read_text())
        for ep in range(30):
            tag = f"episode_{ep:06d}"
            po, pr = root / "submissions" / OURS / f"{tag}.npz", root / "submissions" / RAW / f"{tag}.npz"
            if not (po.exists() and pr.exists() and (root / "gt" / tag / "mhr_params_mv.pt").exists()):
                continue
            d, r = np.load(po), np.load(pr)
            g = torch.load(root / "gt" / tag / "mhr_params_mv.pt", map_location="cpu", weights_only=False)
            T = min(len(d["pose"]), len(r["pose"]), len(g["mhr_model_params"]))
            z = torch.as_tensor(np.concatenate([d["scales"], d["shape"]])[None], dtype=torch.float64, device="cuda")
            Vo, Jo = mhr.run(torch.as_tensor(d["pose"][:T], dtype=torch.float64, device="cuda"), z)
            Vr, Jr = mhr.run(torch.as_tensor(r["pose"][:T], dtype=torch.float64, device="cuda"), z)
            zg = torch.cat([g["mhr_model_params"][:1, 136:], g["shape_params"][:1]], 1).double().cuda()
            Vg, Jg = mhr.run(g["mhr_model_params"][:T, :136].double().cuda(), zg)
            fr = np.array([f for f in S[str(ep)] if f < T]); a = int(fr[0])
            s, R, t = sim3(Vo[a, align_idx].cpu().numpy() / 1000.0, Vg[a, align_idx].cpu().numpy() / 1000.0)
            to_ours = lambda X: ((X - t) @ R.T) / s  # noqa: E731
            valid = np.zeros(T, bool); valid[fr] = True
            kp = np.load(root / f"outputs/kp2d_vpx/{tag}.npz")["pose"][:, 0].astype(float)
            vis = np.load(root / f"outputs/mask_visibility_vp/{tag}.npz")["vis"].astype(float)
            conf = np.zeros((T, 22)); vv = np.zeros((T, 22))
            n = min(T, len(kp)); nv = min(T, len(vis))
            for k, j in enumerate(SC):
                if j in LMK:
                    conf[:n, k] = np.nan_to_num(kp[:n, LMK[j], 3])
                vv[:nv, k] = vis[:nv, j] if vis.shape[1] > j else 1.0
            f32 = lambda x: np.asarray(x, np.float32)  # noqa: E731
            np.savez_compressed(out / f"{root.name}_{tag}.npz", ours=f32(Jo[:, SC].cpu().numpy()), raw=f32(Jr[:, SC].cpu().numpy()),
                                gt=f32(to_ours(Jg[:, SC].cpu().numpy())), conf=f32(conf), vis=f32(vv), valid=valid, anchor=a, sim3_scale=s,
                                hands_ours=f32(Vo[:, hands].cpu().numpy() / 1000.0), hands_raw=f32(Vr[:, hands].cpu().numpy() / 1000.0),
                                hands_gt=f32(to_ours(Vg[:, hands].cpu().numpy() / 1000.0)), sim3_s=s, sim3_R=R, sim3_t=t)
            print(root.name, tag, T, "frames, valid", int(valid.sum()), "sim3 scale %.4f" % s, flush=True)


if __name__ == "__main__":
    main()
