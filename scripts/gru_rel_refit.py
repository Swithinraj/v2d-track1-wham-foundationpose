#!/usr/bin/env python3
"""gru_depth_refit.py with the depth-reliability module in place of the depth head (pipeline/recipes/rel.md). With --rel_dir,
the per-frame along-ray corrections come from scripts/rel_train.py --dump (c_h, c_o = log distance ratios, ramped and
smoothed already): human shift = (exp(eps_scale * c_h) - 1) x our joint centroid (the depth head's mechanics), object
translation x exp(eps_scale * c_o) (rigid, size unchanged). Without --rel_dir it is gru_depth_refit.py.
Deploy the learned depth fusion (scripts/gru_depth_train.py: the motion head + GT-free depth cues + depth head) on episodes
(mhr_converter.sif, GPU; before the contact stage). As scripts/gru_fusion_refit.py (E2 band fusion with the predicted
time-varying coefficients, per vertex), plus the predicted depth correction: every vertex moves by ramp(t) eps(t) c(t)
(c = our input centroid, camera frame), i.e. our distance is scaled by (1 + ramp eps) without changing body size. The cues
are computed with scripts/depth_cues.py's own episode() on our input joints (training used the same function).

Per episode: 22 scored joints of IN and RAW (MHR forward), ViTPose confidence (outputs/kp2d_vpx) and mask visibility
(--visibility) -> the same features as in training -> per frame and region (torso, L/R arm, L/R leg) coefficient
corrections around E2's half-strength coefficients. Every vertex takes its region's coefficients (human_refit.vertex_parts:
shoulder/elbow/wrist -> arm, hip/knee/ankle -> leg, rest -> torso), then
  target = IN + ramp * sum_b [(k_b(t) - 1) B_b(IN) + m_b(t) B_b(RAW)]    (lateral / depth separately; B4 = ours)
with the anchor taper of band_fusion_refit.py, the anchor pose row copied, and the kit lm_pose refit. Objects unchanged.
Usage: gru_depth_refit.py --input DIR --raw DIR --visibility DIR --init DIR --depth_model PT --out DIR [--scale_root DIR]
       [--check_cache DIR --cache_prefix NAME] [--episodes ...]
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
import gru_depth_train as GD  # noqa: E402
import depth_cues as DC  # noqa: E402

SC = [1, 2, 18, 35, 3, 19, 36, 4, 20, 37, 8, 24, 110, 74, 38, 113, 75, 39, 76, 40, 78, 42]
LMK = {2: 23, 18: 24, 3: 25, 19: 26, 4: 27, 20: 28, 75: 11, 39: 12, 76: 13, 40: 14, 78: 15, 42: 16}
PARTS = [75, 39, 76, 40, 78, 42, 2, 18, 3, 19, 4, 20]
PART_REGION = [1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4]          # shoulder/elbow/wrist L,R -> arm; hip/knee/ankle -> leg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True); ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--visibility", type=Path, required=True); ap.add_argument("--init", type=Path, required=True)
    ap.add_argument("--depth_model", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scale_root", type=Path, help="root whose outputs/sam3d_body/<tag>/prepare.npz holds the depth scale of IN (default TRACK1_ROOT)")
    ap.add_argument("--check_cache", type=Path); ap.add_argument("--cache_prefix")
    ap.add_argument("--eps_scale", type=float, default=1.0, help="shrink factor on the predicted depth correction (chosen on validation)")
    ap.add_argument("--lock_dir", type=Path, help="background-locked MoGe results for this root (unshipped experiment; only for models trained on them)")
    ap.add_argument("--lock_prefix")
    ap.add_argument("--episodes", type=int, nargs="*", default=list(range(30)))
    ap.add_argument("--save_cues", type=Path, help="also write each episode's cues, predicted depth correction (scaled) and input joints here")
    ap.add_argument("--cues_only", action="store_true", help="with --save_cues: stop after the cues / correction (no refit, no episodes written)")
    ap.add_argument("--rel_dir", type=Path, help="rel_train.py --dump output: <rel_prefix>_<tag>.npz with c_h, c_o (replaces the depth head)")
    ap.add_argument("--rel_prefix")
    args = ap.parse_args(); dev = "cuda"
    args.out.mkdir(parents=True, exist_ok=True)
    for f in args.input.iterdir():
        if f.is_file() and not f.name.startswith("episode_"):
            shutil.copy2(f, args.out / f.name)
    S = json.loads((ROOT / "outputs/hand_fix/episodes_repro_depth_refine_meshes/scored_frames.json").read_text())
    nrm = torch.load(args.init / "refiner_motion_norm.pt"); G.MU, G.SD = nrm["mu"].to(dev), nrm["sd"].to(dev)
    ck = torch.load(args.depth_model, weights_only=False); MUc, SDc = ck["mu_c"].to(dev), ck["sd_c"].to(dev); GD.SMOOTH_D = float(ck["smooth_d"]); RAMP_D = float(ck["ramp_d"])
    GD.CUE_SET = ck.get("cue_set", "v1")
    assert GD.CUE_SET != "v4" or args.lock_dir, "a v4 (locked) depth model needs --lock_dir"
    scale_root = args.scale_root or ROOT
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
        t = torch.arange(T, dtype=torch.float64, device=dev)
        ramp = torch.clamp((t - anchor).abs() / G.RAMP, 0, 1); ramp = 0.5 - 0.5 * torch.cos(np.pi * ramp)
        jo_np = J_in[:, SC].cpu().numpy()
        assert 300 < float(V_in.abs().median() / J_in.abs().median()) < 3000, "unit check: vertices mm, joints m"
        if args.check_cache:                                                                   # same input joints as in training?
            cj = np.load(args.check_cache / f"{args.cache_prefix}_{tag}.npz")["ours"]
            rep_cache = float(np.abs(cj - jo_np[:len(cj)]).max()); assert rep_cache < 1e-3, f"{tag}: input joints differ from the training cache by {rep_cache} m"
        lk = dict(np.load(args.lock_dir / f"{args.lock_prefix}_{tag}.npz")) if args.lock_dir else None
        if lk is not None:
            assert len(lk["a"]) == T, f"{tag}: lock length {len(lk['a'])} != {T}"
        cz = DC.episode(jo_np, tag, ROOT, scale_root, anchor, lk)
        if GD.CUE_SET == "v4":                                                                 # plane-free foot / background cues
            Kc = json.loads((ROOT / "outputs/camera" / tag / "intrinsics.json").read_text())
            sc_ = float(np.load(scale_root / "outputs/sam3d_body" / tag / "prepare.npz")["depth_scale"])
            cz = dict(cz); cz.update(DC.foot_bg(jo_np, lk, Kc, sc_, anchor))
        rd = torch.clamp((t - anchor).abs() / RAMP_D, 0, 1); ramp_d = 0.5 - 0.5 * torch.cos(np.pi * rd)        # the depth taper (as training)
        cue = GD.cue_tensor(cz, ramp_d.float())
        if model is None:
            model = GD.DepthRefiner(feat.shape[1], GD.ncue()).to(dev); model.load_state_dict(ck["model"]); model.eval()
        with torch.no_grad():
            corr, eps = model(torch.cat([(feat - G.MU) / G.SD, (cue - MUc) / SDc], 1)[None]); corr, eps = corr[0], eps[0].double() * args.eps_scale
        if args.save_cues:                                             # the selector's cues (scripts/corr_select3.py)
            args.save_cues.mkdir(parents=True, exist_ok=True)
            np.savez(args.save_cues / f"{tag}.npz", **{k: np.asarray(v) for k, v in cz.items()}, eps=eps.cpu().numpy(), ramp_d=ramp_d.cpu().numpy(),
                     anchor=anchor, ours=jo_np.astype(np.float32))
            if args.cues_only:
                print(tag, f"|eps| {float(eps.abs().mean()) * 100:.3f} %", flush=True); continue
        coef = base.to(dev).view(1, 1, 2, 2, G.NB).expand(T, G.NREG, 2, 2, G.NB).clone()
        coef[..., :4] += corr
        coef = coef.double()
        # per-vertex fusion in mm with each vertex's region coefficients, in vertex chunks (the band filters run along time
        # per vertex, so chunking gives the same result; it keeps long episodes inside a 16 GB GPU)
        delta = torch.zeros_like(V_in); VC = int(os.environ.get("REFIT_VCHUNK", "4096")) or V_in.shape[1]
        for v0 in range(0, V_in.shape[1], VC):
            vs = slice(v0, v0 + VC)
            BO, BR = G.bands(V_in[:, vs]), G.bands(V_raw[:, vs])                                # [NB, T, v, 3]
            cv = coef[:, vreg[vs]]                                                            # [T, v, 2, 2, NB]
            for b in range(G.NB):
                for a in range(2):
                    sel = (lambda x: x - (x @ z)[..., None] * z) if a == 0 else (lambda x: (x @ z)[..., None] * z)
                    delta[:, vs] += (cv[..., a, 0, b] - 1.0)[..., None] * sel(BO[b]) + cv[..., a, 1, b][..., None] * sel(BR[b])
            del BO, BR, cv
        c_o = None
        if args.rel_dir:                                                                       # reliability module instead of the depth head
            rz = np.load(args.rel_dir / f"{args.rel_prefix}_{tag}.npz"); assert len(rz["c_h"]) == T, f"{tag}: c_h length {len(rz['c_h'])} != {T}"
            ramp_d = torch.ones(T, dtype=torch.float64, device=dev)                            # c_h is ramped / smoothed already
            eps = torch.as_tensor(np.expm1(args.eps_scale * rz["c_h"].astype(np.float64)), device=dev); eps[anchor] = 0.0
            c_o = args.eps_scale * rz["c_o"].astype(np.float64); c_o[anchor] = 0.0
        shift = (ramp_d * eps)[:, None] * (J_in[:, SC].mean(1) * 1000.0)                      # mm, along the ray through our centroid
        X = V_in + ramp[:, None, None] * delta + shift[:, None, :]
        pose1, err = lm_pose(mhr, X.reshape(T, -1), p_in.clone(), ident, iters=6)
        pose1[anchor] = p_in[anchor]
        _, J1 = mhr.run(pose1, ident)
        d["pose"] = pose1.cpu().numpy().astype(np.float32)
        if c_o is not None:                                                                    # object along its ray, size unchanged
            d["object_translation"] = (d["object_translation"].astype(np.float64) * np.exp(c_o)[:len(d["object_translation"]), None]).astype(np.float32)
        np.savez(args.out / f"{tag}.npz", **d)
        for mesh in args.input.glob(f"{tag}_object*"):
            shutil.copy2(mesh, args.out / mesh.name)
        rep[tag] = {"fit_err_mm": float(err.mean()), "corr_abs_mean": float(corr.abs().mean()),
                    "joint_change_cm": float((J1 - J_in).norm(dim=-1).mean() * 100), "anchor_change_cm": float((J1[anchor] - J_in[anchor]).norm(dim=-1).max() * 100),
                    "eps_abs_mean_pct": float(eps.abs().mean() * 100), "depth_shift_cm_mean": float(shift.norm(dim=-1).mean() / 10), "depth_shift_cm_max": float(shift.norm(dim=-1).max() / 10)}
        print(tag, json.dumps(rep[tag]), flush=True)
    (args.out / f"gru_depth_report_{'_'.join(map(str, args.episodes))}.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
