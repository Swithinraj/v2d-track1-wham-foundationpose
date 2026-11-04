#!/usr/bin/env python3
"""Deploy the learned temporal refiner (motion head of ext/gru_train.py) on episodes (mhr_converter.sif, GPU; before the
contact stage): E2's band fusion with time-varying, per-region coefficients predicted by the GRU.

Per episode: 22 scored joints of IN and RAW (MHR forward), ViTPose confidence (outputs/kp2d_vpx) and mask visibility
(--visibility) -> the same features as in training -> per frame and region (torso, L/R arm, L/R leg) coefficient
corrections around E2's half-strength coefficients. Every vertex takes its region's coefficients (human_refit.vertex_parts:
shoulder/elbow/wrist -> arm, hip/knee/ankle -> leg, rest -> torso), then
  target = IN + ramp * sum_b [(k_b(t) - 1) B_b(IN) + m_b(t) B_b(RAW)]    (lateral / depth separately; B4 = ours)
with the anchor taper of band_fusion_refit.py, the anchor pose row copied, and the kit lm_pose refit. Objects unchanged.
Usage: gru_fusion_refit.py --input DIR --raw DIR --visibility DIR --model PT --norm PT --out DIR [--episodes ...]
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(os.environ.get("TRACK1_ROOT") or "/track1")
sys.path.insert(0, str(ROOT / "official/v2d_submission_kit/tools/track1"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mesh_to_mhr_params import MHR, lm_pose  # noqa: E402
from mhr_bodies import MODEL  # noqa: E402
from human_refit import vertex_parts  # noqa: E402
import gru_train as G  # noqa: E402

SC = [1, 2, 18, 35, 3, 19, 36, 4, 20, 37, 8, 24, 110, 74, 38, 113, 75, 39, 76, 40, 78, 42]
LMK = {2: 23, 18: 24, 3: 25, 19: 26, 4: 27, 20: 28, 75: 11, 39: 12, 76: 13, 40: 14, 78: 15, 42: 16}
PARTS = [75, 39, 76, 40, 78, 42, 2, 18, 3, 19, 4, 20]
PART_REGION = [1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4]          # shoulder/elbow/wrist L,R -> arm; hip/knee/ankle -> leg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True); ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--visibility", type=Path, required=True); ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--norm", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--episodes", type=int, nargs="*", default=list(range(30)))
    args = ap.parse_args(); dev = "cuda"
    args.out.mkdir(parents=True, exist_ok=True)
    for f in args.input.iterdir():
        if f.is_file() and not f.name.startswith("episode_"):
            shutil.copy2(f, args.out / f.name)
    S = json.loads((ROOT / "outputs/hand_fix/episodes_repro_depth_refine_meshes/scored_frames.json").read_text())
    nrm = torch.load(args.norm); G.MU, G.SD = nrm["mu"].to(dev), nrm["sd"].to(dev)
    base = G.e2_base()
    mhr = MHR(str(MODEL), dev, chunk=128, precision="float64")
    parts = vertex_parts(MODEL, np.array(PARTS))
    vreg = torch.as_tensor(np.where(parts < 0, 0, np.array(PART_REGION)[np.clip(parts, 0, None)]), device=dev)
    model = None; rep = {}
    z = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64, device=dev)
    for ep in args.episodes:
        tag = f"episode_{ep:06d}"; d = dict(np.load(args.input / f"{tag}.npz")); r = np.load(args.raw / f"{tag}.npz")
        T = len(d["pose"]); anchor = min(f for f in S[str(ep)] if f < T)
        ident = torch.as_tensor(np.concatenate([d["scales"], d["shape"]])[None], dtype=torch.float64, device=dev)
        p_in = torch.as_tensor(d["pose"], dtype=torch.float64, device=dev)
        V_in, J_in = mhr.run(p_in, ident); V_raw, J_raw = mhr.run(torch.as_tensor(r["pose"][:T], dtype=torch.float64, device=dev), ident)
        # features exactly as in training (joints in metres)
        jo, jr = J_in[:, SC].float(), J_raw[:, SC].float()
        bo, br = G.bands(jo.double()).float(), G.bands(jr.double()).float()
        kp = np.load(ROOT / f"outputs/kp2d_vpx/{tag}.npz")["pose"][:, 0].astype(float)
        vis = np.load(args.visibility / f"{tag}.npz")["vis"].astype(float)
        conf = np.zeros((T, 22)); vv = np.zeros((T, 22)); n = min(T, len(kp)); nv = min(T, len(vis))
        for k, j in enumerate(SC):
            if j in LMK:
                conf[:n, k] = np.nan_to_num(kp[:n, LMK[j], 3])
            vv[:nv, k] = vis[:nv, j]
        feat = G.features(bo, br, torch.as_tensor(conf, dtype=torch.float32, device=dev), torch.as_tensor(vv, dtype=torch.float32, device=dev))
        if model is None:
            model = G.Refiner(feat.shape[1]).to(dev); model.load_state_dict(torch.load(args.model)); model.eval()
        with torch.no_grad():
            corr = model(((feat - G.MU) / G.SD)[None])[0]                                    # [T, NREG, 2, 2, 4]
        coef = base.to(dev).view(1, 1, 2, 2, G.NB).expand(T, G.NREG, 2, 2, G.NB).clone()
        coef[..., :4] += corr
        coef = coef.double()
        # per-vertex fusion in mm with each vertex's region coefficients
        BO, BR = G.bands(V_in), G.bands(V_raw)                                                 # [NB, T, V, 3]
        cv = coef[:, vreg]                                                                    # [T, V, 2, 2, NB]
        delta = torch.zeros_like(V_in)
        for b in range(G.NB):
            for a in range(2):
                sel = (lambda x: x - (x @ z)[..., None] * z) if a == 0 else (lambda x: (x @ z)[..., None] * z)
                delta += (cv[..., a, 0, b] - 1.0)[..., None] * sel(BO[b]) + cv[..., a, 1, b][..., None] * sel(BR[b])
        t = torch.arange(T, dtype=torch.float64, device=dev)
        ramp = torch.clamp((t - anchor).abs() / G.RAMP, 0, 1); ramp = 0.5 - 0.5 * torch.cos(np.pi * ramp)
        X = V_in + ramp[:, None, None] * delta
        pose1, err = lm_pose(mhr, X.reshape(T, -1), p_in.clone(), ident, iters=6)
        pose1[anchor] = p_in[anchor]
        _, J1 = mhr.run(pose1, ident)
        d["pose"] = pose1.cpu().numpy().astype(np.float32); np.savez(args.out / f"{tag}.npz", **d)
        for mesh in args.input.glob(f"{tag}_object*"):
            shutil.copy2(mesh, args.out / mesh.name)
        rep[tag] = {"fit_err_mm": float(err.mean()), "corr_abs_mean": float(corr.abs().mean()),
                    "joint_change_cm": float((J1 - J_in).norm(dim=-1).mean() * 100), "anchor_change_cm": float((J1[anchor] - J_in[anchor]).norm(dim=-1).max() * 100)}
        print(tag, json.dumps(rep[tag]), flush=True)
    (args.out / f"gru_fusion_report_{'_'.join(map(str, args.episodes))}.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
