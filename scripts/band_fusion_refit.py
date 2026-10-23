#!/usr/bin/env python3
"""Band fusion of the refined human with the raw per-frame human, learned on labelled data (mhr_converter.sif, GPU;
before the contact stage). A linear temporal estimator (Wiener-type): per frequency band and axis group, a fixed mix of
our refined trajectory and the raw SAM 3D Body trajectory, with coefficients regressed on multi-view ground truth
(FORM-HOI, the Track 1 GT generator's own labels) and shrunk toward "ours unchanged". No per-episode constants.

Per vertex, in the camera frame (the Track 1 frame), positions over the episode are split into Gaussian difference
bands (sigma 1/2/4/8 frames: B0 > 4.8 Hz, B1 2.4-4.8, B2 1.2-2.4, B3 0.6-1.2, B4 < 0.6 Hz at 30 fps), each into its
lateral part (perpendicular to the camera z axis) and depth part:
  target = IN + taper * sum_b [ (k_b - 1) * B_b(IN) + m_b * B_b(RAW) ]      (lateral and depth coefficients separate)
B4 keeps k = 1, m = 0 (low-frequency position and registration untouched). Coefficients: --coef JSON
{"lat": [k_B0..k_B4, m_B0..m_B4], "dep": [...]}, shrunk by --strength toward k = 1, m = 0 (0.5 = halfway).
The first scored frame is pinned (the change tapers to 0 over +-RAMP frames; its pose row is copied exactly), so the
scorer's first-frame registration and the objects are unchanged. Then each frame's pose is re-fitted to the targets
(kit lm_pose, identity fixed). Objects unchanged.
Writes OUT/episode_*.npz (+ meshes) and band_fusion_report_<episodes>.json.
Usage (container): band_fusion_refit.py --input DIR --raw DIR --coef JSON --out DIR [--strength 0.5] [--episodes ...]
  --raw: the refinement's unsmoothed input human (per-frame SAM 3D Body after the depth trend)
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(os.environ.get("TRACK1_ROOT") or Path(__file__).resolve().parents[1])
sys.path.insert(0, str(ROOT / "official/v2d_submission_kit/tools/track1"))
sys.path.insert(0, str(ROOT / "scripts"))
from mesh_to_mhr_params import MHR, lm_pose  # noqa: E402
from mhr_bodies import MODEL  # noqa: E402

SIGMAS = (1, 2, 4, 8)
NB = len(SIGMAS) + 1
RAMP = 10


def gauss_t(x, s):
    """Gaussian smoothing along axis 0 (time) with edge padding; x [T, ...] float64 torch."""
    r = int(4 * s + 0.5)
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1, dtype=x.dtype, device=x.device) / s) ** 2)
    k = (k / k.sum()).view(1, 1, -1)
    flat = x.reshape(len(x), -1).T.unsqueeze(1)                       # [C, 1, T]
    flat = torch.nn.functional.pad(flat, (r, r), mode="replicate")
    return torch.nn.functional.conv1d(flat, k).squeeze(1).T.reshape(x.shape)


def bands(x):
    gs = [x] + [gauss_t(x, s) for s in SIGMAS]
    return [gs[i] - gs[i + 1] for i in range(len(SIGMAS))] + [gs[-1]]


def coefficients(path, strength):
    c = json.loads(Path(path).read_text())
    out = {}
    for m in ("lat", "dep"):
        k, r = list(c[m][:NB]), list(c[m][NB:])
        k[NB - 1], r[NB - 1] = 1.0, 0.0
        out[m] = ([1.0 + strength * (a - 1.0) for a in k], [strength * a for a in r])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True); ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--coef", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--strength", type=float, default=0.5)
    ap.add_argument("--episodes", type=int, nargs="*", default=list(range(30)))
    args = ap.parse_args(); dev = "cuda"
    args.out.mkdir(parents=True, exist_ok=True)
    for f in args.input.iterdir():
        if f.is_file() and not f.name.startswith("episode_"):
            shutil.copy2(f, args.out / f.name)
    S = json.loads((ROOT / "outputs/hand_fix/episodes_repro_depth_refine_meshes/scored_frames.json").read_text())
    C = coefficients(args.coef, args.strength)
    mhr = MHR(str(MODEL), dev, chunk=128, precision="float64"); rep = {}
    z = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64, device=dev)
    for ep in args.episodes:
        tag = f"episode_{ep:06d}"; d = dict(np.load(args.input / f"{tag}.npz")); r = np.load(args.raw / f"{tag}.npz")
        T = len(d["pose"]); anchor = min(f for f in S[str(ep)] if f < T)
        ident = torch.as_tensor(np.concatenate([d["scales"], d["shape"]])[None], dtype=torch.float64, device=dev)
        p_in = torch.as_tensor(d["pose"], dtype=torch.float64, device=dev)
        p_raw = torch.as_tensor(r["pose"][:T], dtype=torch.float64, device=dev)
        V_in, J_in = mhr.run(p_in, ident); V_raw, _ = mhr.run(p_raw, ident)                     # mm, m
        bo, br = bands(V_in), bands(V_raw)
        delta = torch.zeros_like(V_in)
        for b in range(NB):
            for m, sel in (("lat", lambda a: a - (a @ z)[..., None] * z), ("dep", lambda a: (a @ z)[..., None] * z)):
                k, w = C[m][0][b], C[m][1][b]
                if k != 1.0:
                    delta += (k - 1.0) * sel(bo[b])
                if w != 0.0:
                    delta += w * sel(br[b])
        t = torch.arange(T, dtype=torch.float64, device=dev)
        ramp = torch.clamp((t - anchor).abs() / RAMP, 0, 1); ramp = 0.5 - 0.5 * torch.cos(np.pi * ramp)
        X = V_in + ramp[:, None, None] * delta
        pose1, err = lm_pose(mhr, X.reshape(T, -1), p_in.clone(), ident, iters=6)
        pose1[anchor] = p_in[anchor]
        V1, J1 = mhr.run(pose1, ident)
        d["pose"] = pose1.cpu().numpy().astype(np.float32); np.savez(args.out / f"{tag}.npz", **d)
        for mesh in args.input.glob(f"{tag}_object*"):
            shutil.copy2(mesh, args.out / mesh.name)
        lat = ((J1[..., :2] - J_in[..., :2]).norm(dim=-1)).mean() * 100
        rep[tag] = {"fit_err_mm": float(err.mean()), "target_change_mm": float((ramp[:, None, None] * delta).norm(dim=-1).mean()),
                    "joint_lateral_change_cm": float(lat), "joint_depth_change_cm": float((J1[..., 2] - J_in[..., 2]).abs().mean() * 100),
                    "anchor_change_cm": float((J1[anchor] - J_in[anchor]).norm(dim=-1).max() * 100)}
        print(tag, json.dumps(rep[tag]), flush=True)
    (args.out / f"band_fusion_report_{'_'.join(map(str, args.episodes))}.json").write_text(
        json.dumps({"args": {k: str(v) for k, v in vars(args).items()}, "coefficients": C, "episodes": rep}, indent=1))


if __name__ == "__main__":
    main()
