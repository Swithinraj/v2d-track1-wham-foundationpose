#!/usr/bin/env python3
"""Deploy the joint human-object refiner (ext/joint_train.py) on episodes (mhr_converter.sif, GPU; before the contact
stage). Every input is recomputed from pipeline outputs exactly as the training cache builds it, without GT.

Per episode: 22 scored joints and 128 hand points of IN and RAW (MHR forward), ViTPose confidence, mask visibility,
our object track, raw FoundationPose, SAM 2 mask statistics, MoGe depth in the mask, our mesh's surface samples and its
scorer-consistent SDF grid -> predict() at --level (branch / inter / final). Then:
  human   per-vertex band fusion with the predicted per-frame region coefficients (as gru_fusion_refit.py), plus the
          predicted hand offsets on each side's wrist part (hand vertices, full) and elbow part (forearm, half), with the
          anchor taper; the kit lm_pose refit; the anchor pose row copied
  object  the corrected rotation and translation (scale and mesh unchanged)
Usage: joint_deploy.py --input DIR --raw DIR --visibility DIR --sdf DIR --model PT --out DIR [--level final] [--episodes ...]
"""
import argparse
import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

ROOT = Path(os.environ.get("TRACK1_ROOT") or "/track1")
sys.path.insert(0, "/track1/official/v2d_submission_kit/tools/track1")
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mesh_to_mhr_params import MHR, lm_pose  # noqa: E402
from mhr_bodies import MODEL  # noqa: E402
from human_refit import vertex_parts  # noqa: E402
from extract_mhr_hands import hand_vertex_indices  # noqa: E402
import gru_train as G  # noqa: E402
import joint_train as JT  # noqa: E402
from joint_cache_object import mask_stats, quat_to_R, surface  # noqa: E402

SC = [1, 2, 18, 35, 3, 19, 36, 4, 20, 37, 8, 24, 110, 74, 38, 113, 75, 39, 76, 40, 78, 42]
LMK = {2: 23, 18: 24, 3: 25, 19: 26, 4: 27, 20: 28, 75: 11, 39: 12, 76: 13, 40: 14, 78: 15, 42: 16}
PARTS = [75, 39, 76, 40, 78, 42, 2, 18, 3, 19, 4, 20]
PART_REGION = [1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4]
HAND_PARTS = ((5, 3), (4, 2))          # hand block 0 = wrist 42: (wrist part, elbow part); block 1 = wrist 78


def main():
    ap = argparse.ArgumentParser()
    for k in ("input", "raw", "visibility", "sdf", "model", "out"):
        ap.add_argument(f"--{k}", type=Path, required=True)
    ap.add_argument("--level", default="final"); ap.add_argument("--episodes", type=int, nargs="*", default=list(range(30)))
    ap.add_argument("--obj_smooth", type=float); ap.add_argument("--opt_obj_only", action="store_true"); ap.add_argument("--no_hand_head", action="store_true")
    ap.add_argument("--visual_cache", type=Path, help="visual model: dir of <root>_<episode>_vis.npz (ext/joint_cache_visual.py)")
    ap.add_argument("--objects_only", action="store_true", help="write only the corrected objects; the input human is kept")
    args = ap.parse_args(); dev = "cuda"
    args.out.mkdir(parents=True, exist_ok=True)
    for f in args.input.iterdir():
        if f.is_file() and not f.name.startswith("episode_"):
            shutil.copy2(f, args.out / f.name)
    S = json.loads((ROOT / "outputs/hand_fix/episodes_repro_depth_refine_meshes/scored_frames.json").read_text())
    ck = torch.load(args.model); MU = ck["mu"]
    cfg = ck.get("cfg", {})                                   # training settings (checkpoints after 10-23 20:15 carry them)
    JT.OBJ_SMOOTH = args.obj_smooth or cfg.get("obj_smooth", JT.OBJ_SMOOTH)
    JT.OPT_OBJ_ONLY = args.opt_obj_only or cfg.get("opt_obj_only", False)
    JT.NO_HAND_HEAD = args.no_hand_head or cfg.get("no_hand_head", False)
    JT.GATE = cfg.get("gate", False)
    print(f"deploy settings: OBJ_SMOOTH {JT.OBJ_SMOOTH} OPT_OBJ_ONLY {JT.OPT_OBJ_ONLY} NO_HAND_HEAD {JT.NO_HAND_HEAD} GATE {JT.GATE} level {args.level}", flush=True)
    mhr = MHR(str(MODEL), dev, chunk=128, precision="float64")
    parts = vertex_parts(MODEL, np.array(PARTS))
    vreg = torch.as_tensor(np.where(parts < 0, 0, np.array(PART_REGION)[np.clip(parts, 0, None)]), device=dev)
    hw = torch.zeros(len(parts), 2, dtype=torch.float64, device=dev)                       # per-vertex hand-offset weights
    for b, (pw, pe) in enumerate(HAND_PARTS):
        hw[torch.as_tensor(parts == pw, device=dev), b] = 1.0; hw[torch.as_tensor(parts == pe, device=dev), b] = 0.5
    hands = np.concatenate([h[::4] for h in hand_vertex_indices()])
    model = None; rep = {}
    z = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64, device=dev)
    pool = ProcessPoolExecutor(8)
    for ep in args.episodes:
        tag = f"episode_{ep:06d}"
        if not (args.input / f"{tag}.npz").exists():
            continue
        d = dict(np.load(args.input / f"{tag}.npz")); r = np.load(args.raw / f"{tag}.npz")
        T = len(d["pose"]); anchor = min(f for f in S[str(ep)] if f < T)
        ident = torch.as_tensor(np.concatenate([d["scales"], d["shape"]])[None], dtype=torch.float64, device=dev)
        p_in = torch.as_tensor(d["pose"], dtype=torch.float64, device=dev)
        V_in, J_in = mhr.run(p_in, ident); V_raw, J_raw = mhr.run(torch.as_tensor(r["pose"][:T], dtype=torch.float64, device=dev), ident)
        f32 = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=dev)  # noqa: E731
        e = {"ours": J_in[:, SC].float(), "raw": J_raw[:, SC].float(),
             "hands_ours": (V_in[:, hands] / 1000).float(), "hands_raw": (V_raw[:, hands] / 1000).float()}
        kp = np.load(ROOT / f"outputs/kp2d_vpx/{tag}.npz")["pose"][:, 0].astype(float)
        vis = np.load(args.visibility / f"{tag}.npz")["vis"].astype(float)
        conf = np.zeros((T, 22)); vv = np.zeros((T, 22)); n = min(T, len(kp)); nv = min(T, len(vis))
        for k, j in enumerate(SC):
            if j in LMK:
                conf[:n, k] = np.nan_to_num(kp[:n, LMK[j], 3])
            vv[:nv, k] = vis[:nv, j] if vis.shape[1] > j else 1.0
        e["conf"], e["vis"] = f32(conf), f32(vv)
        # objects, as joint_cache_object.py builds them
        oR, ot, os_ = d["object_rotation"].astype(float), d["object_translation"].astype(float), float(d["object_scale"])
        obj = ROOT / "outputs/object" / tag
        raw = [json.loads((obj / f"foundationpose/raw/{f:06d}.json").read_text()) for f in range(T)]
        rR = np.stack([quat_to_R(q["rotation"]) for q in raw]); rt = np.array([q["translation"] for q in raw], float)
        fin = np.isfinite(rt).all(1) & np.isfinite(rR).reshape(T, -1).all(1)
        fin &= np.linalg.norm(np.nan_to_num(rt), axis=1) > 1e-6          # FoundationPose writes zeros where it lost the object
        if fin.sum() >= 10:
            rt *= np.median(np.linalg.norm(ot[fin], axis=1)) / np.median(np.linalg.norm(rt[fin], axis=1))
        else:
            fin[:] = False
        rt[~fin] = ot[~fin]; rR[~fin] = oR[~fin]
        ang = np.degrees(np.arccos(np.clip((np.einsum("tij,tij->t", rR, oR) - 1) / 2, -1, 1)))
        ang = ang[fin] if fin.any() else np.array([180.0])
        K = json.loads((obj / "intrinsics/000000.json").read_text())
        st = np.array(list(pool.map(mask_stats, [(obj / f"sam2_masks/1/{f:06d}.png", obj / f"depth/{f:06d}.png") for f in range(T)], chunksize=16)))
        area = st[:, 0]; mu = np.nan_to_num((st[:, 1] - K["cx"]) / K["fx"]); mv = np.nan_to_num((st[:, 2] - K["cy"]) / K["fy"])
        dep = st[:, 3]; ok = np.isfinite(dep) & (area > 0)
        if ok.any():
            dep = dep * np.median(ot[ok, 2] / dep[ok])
        so, _ = surface(args.input / f"{tag}_object.glb")
        sd = np.load(args.sdf / f"{tag}.npz")
        e.update(obj_R=f32(oR), obj_t=f32(ot), raw_R=f32(rR), raw_t=f32(rt), raw_ok=f32(np.full(T, float(np.median(ang) < 30.0))),
                 rt_ok=f32(fin.astype(float)), surf_ours=f32(so), mask_area=f32(area), mask_u=f32(mu), mask_v=f32(mv),
                 mask_depth=f32(np.where(ok, dep, 0.0)), depth_ok=f32(ok.astype(float)), sdf=f32(sd["sdf"]), lo=f32(sd["lo"]), hi=f32(sd["hi"]))
        e = {k: torch.nan_to_num(v) for k, v in e.items()}
        t = torch.arange(T, dtype=torch.float32, device=dev)
        ramp = torch.clamp((t - anchor).abs() / G.RAMP, 0, 1); e["ramp"] = 0.5 - 0.5 * torch.cos(np.pi * ramp)
        e["bo"], e["br"] = (G.bands(e[k].double()).float() for k in ("ours", "raw"))
        e["bho"], e["bhr"] = (G.bands(e[k].double()).float() for k in ("hands_ours", "hands_raw"))
        e["bto"], e["btr"] = G.bands(e["obj_t"].double()).float(), G.bands(e["raw_t"].double()).float()
        e["os"] = os_
        if ck.get("visual"):                                   # image evidence from the visual cache of this root
            vz = np.load(args.visual_cache / f"{ROOT.name}_{tag}_vis.npz")
            for k in ("kp", "kpc", "boxes", "box_src", "mask_m"):
                e[k] = torch.nan_to_num(torch.as_tensor(vz[k][:T], dtype=torch.float32, device=dev))
            h2 = torch.as_tensor(vz["hand2d"][:T], dtype=torch.float32, device=dev)
            e["hc2d"] = torch.nan_to_num(h2.mean(2)); e["h2s"] = torch.as_tensor(vz["hand2d_s"][:T], dtype=torch.float32, device=dev)
            e["crops"] = torch.as_tensor(vz["crops"][:T]); e["roll"] = torch.zeros(T, device=dev)
        if model is None:
            model = JT.JointRefiner(len(MU["h"]), len(MU["o"]), visual=bool(ck.get("visual")), pretrained=False).to(dev)
            model.load_state_dict(ck["model"], strict=False); model.eval()   # pre-gate checkpoints lack head_gate
        bad_in = [k for k, v in e.items() if torch.is_tensor(v) and not torch.isfinite(v).all()]
        if bad_in:
            print(f"{tag}: non-finite inputs {bad_in}", flush=True)
        out = JT.predict(model, e, MU, args.level)
        nf = ~(torch.isfinite(out["t"]).all(-1) & torch.isfinite(out["R"]).flatten(1).all(-1))
        if nf.any():                                          # safety: keep our object pose where the output is not finite
            print(f"{tag}: non-finite object output in {int(nf.sum())} frames -> input pose kept there", flush=True)
            out["t"] = torch.where(nf[:, None], e["obj_t"], out["t"]); out["R"] = torch.where(nf[:, None, None], e["obj_R"], out["R"])
        if not (torch.isfinite(out["c"]).all() and torch.isfinite(out["dh"]).all()):
            print(f"{tag}: non-finite human output -> human corrections dropped", flush=True)
            out["c"] = torch.nan_to_num(out["c"]) * 0; out["dh"] = torch.zeros_like(out["dh"])
        # human: per-vertex fusion with the predicted coefficients (mm) + hand offsets, then the MHR refit
        coef = G.e2_base().to(dev).view(1, 1, 2, 2, G.NB).expand(T, G.NREG, 2, 2, G.NB).clone()
        coef[..., :4] += out["c"]; coef = coef.double()
        BO, BR = G.bands(V_in), G.bands(V_raw); cv = coef[:, vreg]
        delta = torch.zeros_like(V_in)
        for b in range(G.NB):
            for a in range(2):
                sel = (lambda x: x - (x @ z)[..., None] * z) if a == 0 else (lambda x: (x @ z)[..., None] * z)
                delta += (cv[..., a, 0, b] - 1.0)[..., None] * sel(BO[b]) + cv[..., a, 1, b][..., None] * sel(BR[b])
        delta += torch.einsum("vb,tbk->tvk", hw, out["dh"].double() * 1000)
        rp = e["ramp"].double()
        X = V_in + rp[:, None, None] * delta
        if args.objects_only:
            pose1, err = p_in.clone(), torch.zeros(1, dtype=torch.float64, device=dev)
        else:
            pose1, err = lm_pose(mhr, X.reshape(T, -1), p_in.clone(), ident, iters=6)
        pose1[anchor] = p_in[anchor]
        _, J1 = mhr.run(pose1, ident)
        d["pose"] = pose1.cpu().numpy().astype(np.float32)
        d["object_rotation"] = out["R"].cpu().numpy().astype(np.float32); d["object_translation"] = out["t"].cpu().numpy().astype(np.float32)
        np.savez(args.out / f"{tag}.npz", **d)
        for mesh in args.input.glob(f"{tag}_object*"):
            shutil.copy2(mesh, args.out / mesh.name)
        rep[tag] = {"fit_err_mm": float(err.mean()), "joint_change_cm": float((J1 - J_in).norm(dim=-1).mean() * 100),
                    "hand_offset_cm": float(out["dh"].norm(dim=-1).mean() * 100),
                    "object_change_cm": float((out["t"] - e["obj_t"]).norm(dim=-1).mean() * 100),
                    "object_rot_change_deg": float(np.degrees(JT.rot_angle(out["R"], e["obj_R"]).mean().cpu())),
                    "anchor_change_cm": float((J1[anchor] - J_in[anchor]).norm(dim=-1).max() * 100)}
        print(tag, json.dumps(rep[tag]), flush=True)
    (args.out / f"joint_deploy_report_{args.level}.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
