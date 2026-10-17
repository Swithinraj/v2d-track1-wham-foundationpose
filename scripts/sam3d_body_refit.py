"""SAM 3D Body: refit every frame's 136 pose parameters with ONE shared identity (mhr_converter.sif, GPU).

Targets = SAM 3D Body's own per-frame mesh (pred_vertices + pred_cam_t, camera frame); initial pose = its
mhr_model_params[:136] with the root translation set once from pred_cam_t; identity = the medoid (scales,
shape) of sam3d_body_identity.py, held fixed; the kit's lm_pose solves the pose. The kit wrapper's output frame equals
SAM's camera frame (MHR native with y/z flipped), so the fitted pose is directly a Track 1 pose with the static
camera as world frame. Writes outputs/sam3d_body/<episode>/refit.npz (pose, fit error per frame).
Usage: sam3d_body_refit.py --episodes E [...] [--raw_dir DIR --run_dir DIR]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

T1 = Path("/track1")
sys.path.insert(0, str(T1 / "official/v2d_submission_kit/tools/track1"))
from mesh_to_mhr_params import MHR, lm_pose  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", type=Path, default=T1 / "outputs/sam3d_body")
    ap.add_argument("--run_dir", type=Path, default=T1 / "outputs/sam3d_body")
    ap.add_argument("--episodes", type=int, nargs="+", required=True)
    args = ap.parse_args()
    mhr = MHR(str(T1 / "weights/mhr/assets/mhr_model.pt"), "cuda", chunk=128, precision="float64")
    for ep in args.episodes:
        tag = f"episode_{ep:06d}"
        z = np.load(args.raw_dir / tag / f"raw_{tag}.npz")
        prep = np.load(args.run_dir / tag / "prepare.npz")
        cam_t = z["pred_cam_t"].astype(np.float64)
        native = cam_t.copy(); native[:, [1, 2]] *= -1
        pose0 = z["mhr_model_params"][:, :136].astype(np.float64).copy(); pose0[:, :3] = native * 10.0
        target = (z["pred_vertices"].astype(np.float64) + cam_t[:, None]) * 1000.0                      # mm
        T = len(pose0)
        ident = torch.as_tensor(prep["identity"][None], dtype=torch.float64, device="cuda")
        p0 = torch.as_tensor(pose0, dtype=torch.float64, device="cuda")
        tg = torch.as_tensor(target.reshape(T, -1), dtype=torch.float64, device="cuda")
        before = (mhr.run(p0, ident)[0] - tg.reshape(T, -1, 3)).norm(dim=-1).mean(1)
        pose1, err = lm_pose(mhr, tg, p0.clone(), ident, iters=8)
        np.savez(args.run_dir / tag / "refit.npz", pose=pose1.cpu().numpy().astype(np.float32), identity=prep["identity"].astype(np.float32),
                 fit_err_mm=err.cpu().numpy(), err_before_mm=before.cpu().numpy())
        rep = {"frames": T, "identity_swap_error_mm": float(before.mean()), "refit_error_mm_mean": float(err.mean()),
               "refit_error_mm_p95": float(torch.quantile(err, 0.95))}
        (args.run_dir / tag / "refit.json").write_text(json.dumps(rep, indent=2))
        print(tag, json.dumps(rep), flush=True)


if __name__ == "__main__":
    main()
