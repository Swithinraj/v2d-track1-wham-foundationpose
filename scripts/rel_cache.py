#!/usr/bin/env python3
"""Depth-reliability training cache (10-27, pipeline/recipes/rel.md; v2d-host venv: numpy, PIL, cv2; CPU, parallel per episode).
Per episode of a FORM-HOI root, every frame, the evidence a small module needs to decide WHETHER and HOW MUCH to move the
human and the object along their camera rays, and the GT answer. GT is used for the targets only.

Human (from the joint cache's OUR joints, camera frame; 11 torso joints as scripts/depth_cues.py):
  h_moge, h_vda     log(source depth at our torso joints' pixels (5x5 median) / our joint depth), median over joints,
                    minus log a_t (the source's background lock: per-frame scale wobble removed), anchor-relative
  h_dis             h_vda - h_moge (disagreement: a warning, not a label)
  h_valid           share of torso joints with a valid sample (both sources, inside the image)
  h_occ             share of torso joint pixels inside the object mask
  h_border          min projected-joint distance to the image border (fraction of the shorter side)
  h_kp, h_kpc       ViTPose-vs-our-projection residual (normalised by our 2D extent), mean ViTPose confidence
  c_size, n_size, c_floor, st_floor   the existing GT-free cues (scripts/depth_cues.py), anchor-relative size
  h_vr, h_vlat, h_ar  our centroid's along-ray velocity / lateral angular velocity / along-ray acceleration
Object (joint object cache: our track, 512 surface samples):
  o_moge, o_vda     log(source depth median inside the SAM 2 object mask / our object centroid depth), locked, anchor-relative
  o_dis, o_spread_m, o_spread_v   disagreement; robust spread / median inside the mask, each source
  o_size            log(sqrt(mask area) / our projected extent), anchor-relative (apparent size)
  o_iou             mask vs our projected surface samples (coarse silhouette agreement), o_mask (mask present)
  o_occ             share of the object mask covered by the person mask
  o_fpjump          |FoundationPose raw - our track| / |our track| (0 when raw is in another mesh frame)
  o_vr, o_vlat, o_ar  object along-ray / lateral velocity, along-ray acceleration
  contact           min distance (m) between our hand points and our posed object surface
Targets (GT through the scorer's first-frame similarity, our frame):
  tgt_h = log(|c_gt| / |c_ours|) - (same at the anchor), c = joint centroid;  tgt_o likewise for the object centroid
  err_h = mean joint error (m); err_o = mean nearest-surface distance ours -> GT (m); valid_h / valid_o
Raw per-source log ratios (before anchoring) are stored too, so corruptions can be recomputed consistently: moving OUR
estimate by delta along its ray lowers every "source / ours" log ratio by delta and the target by delta.
Writes OUT/<prefix>_<tag>.npz. Usage: rel_cache.py --root ROOT_DIR --mode full|bm --prefix rootX --out DIR [--workers 16] [--no_vda]
  --no_vda: MoGe-only (h_vda, o_vda, o_spread_v and the disagreements NaN; trainer: zero + finite flag 0)
"""
import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

F = Path(os.environ.get("FORMHOI_DIR", "runs/formhoi"))   # training caches (FORM-HOI); deployment passes --dirs
DIRS = None        # --dirs: cache directories of a deployment (scripts/rel_deploy_prep.py, depth_cues.py, moge_lock.py)
DEPLOY = False     # --deploy: no GT (targets NaN, valid False; features unchanged)
TORSO = [0, 1, 2, 3, 6, 9, 12, 13, 14, 16, 17]
SC2MP = {16: 11, 17: 12, 18: 13, 19: 14, 20: 15, 21: 16, 1: 23, 2: 24, 4: 25, 5: 26, 7: 27, 8: 28}
HN = ["h_moge", "h_vda", "h_dis", "h_valid", "h_occ", "h_border", "h_kp", "h_kpc", "c_size", "n_size", "c_floor", "st_floor",
      "h_vr", "h_vlat", "h_ar"]
ON = ["o_moge", "o_vda", "o_dis", "o_spread_m", "o_spread_v", "o_size", "o_iou", "o_mask", "o_occ", "o_fpjump", "o_vr", "o_vlat",
      "o_ar", "contact"]


def depth_m(p):
    if not p.exists():
        return None
    v = np.asarray(Image.open(p)).astype(np.float64)
    return np.where(v > 0, 65535.0 / np.where(v > 0, v, 1) - 1.0, np.nan)


def mask(p, shape):
    return (np.asarray(Image.open(p)) > 127) if p.exists() else np.zeros(shape, bool)


def anchored(x, a, win=5):
    lo, hi = max(0, a - win), min(len(x), a + win + 1); ref = np.nanmedian(x[lo:hi])
    return x - (ref if np.isfinite(ref) else 0.0)


def kinematics(c):
    """c [T, 3] -> along-ray velocity (log-distance / frame), lateral angular velocity (rad / frame), along-ray accel."""
    d = np.linalg.norm(c, axis=1); u = c / d[:, None]; ld = np.log(d)
    vr = np.gradient(ld); vlat = np.r_[0.0, np.arccos(np.clip((u[1:] * u[:-1]).sum(1), -1, 1))]; ar = np.gradient(vr)
    return vr, vlat, ar


def episode(args):
    root, mode, pref, out, tag, no_vda = args
    dh = DIRS["human"] if DIRS else F / ("joint_cache/human" if mode == "full" else "joint_cache_bm/human")
    do_ = DIRS["object"] if DIRS else F / ("joint_cache/object" if mode == "full" else "joint_cache_bm/object")
    hc = np.load(dh / f"{pref}_{tag}.npz")
    oc = np.load(do_ / f"{pref}_{tag}_obj.npz")
    vc = np.load((DIRS["visual"] if DIRS else F / "joint_cache/visual") / f"{pref}_{tag}_vis.npz")
    cz = (DIRS["cues"] if DIRS else F / ("depth_cues" if mode == "full" else "depth_cues_bm")) / f"{pref}_{tag}.npz"
    cz = np.load(cz) if cz.exists() else None
    lm, lv = (DIRS["lock"] if DIRS else F / "moge_lock") / f"{pref}_{tag}.npz", F / "lock_vda" / f"{pref}_{tag}.npz"
    if not (lm.exists() and (no_vda or lv.exists())):
        return tag, "no lock"
    am = np.load(lm)["a"]; av = np.ones_like(am) if no_vda else np.load(lv)["a"]
    ours = hc["ours"].astype(np.float64); gt = ours if DEPLOY else hc["gt"].astype(np.float64)
    T = len(ours); a = int(hc["anchor"]); valid_h = hc["valid"][:T].astype(bool)
    fx, fy, cx, cy = [float(x) for x in vc["K"]]
    R, t, s = oc["obj_R"].astype(np.float64), oc["obj_t"].astype(np.float64), float(oc["obj_s"])
    surf = oc["surf_ours"].astype(np.float64)
    P = s * np.einsum("tij,nj->tni", R[:T], surf) + t[:T, None]                         # [T, 512, 3] our posed object
    Pg = P if DEPLOY else float(oc["gt_s"]) * np.einsum("tij,nj->tni", oc["gt_R"][:T].astype(np.float64), oc["surf_gt"].astype(np.float64)) + oc["gt_t"][:T, None]
    hands = hc["hands_ours"].astype(np.float64)[:T]
    obj_dir = root / "outputs/object" / tag; per_dir = root / "outputs/person" / tag
    raw = {k: np.full(T, np.nan) for k in ("h_moge", "h_vda", "o_moge", "o_vda", "o_sm", "o_sv", "o_size", "o_iou", "o_occ")}
    hv, ho, hb, okm = np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T)
    for f in range(T):
        Dm = depth_m(obj_dir / "depth" / f"{f:06d}.png")
        Dv = Dm if no_vda else depth_m(obj_dir / "depth_vda" / f"{f:06d}.png")   # --no_vda: MoGe's validity only, VDA set NaN below
        if Dm is None or Dv is None:
            continue
        H, W = Dm.shape
        mo = mask(obj_dir / "sam2_masks/1" / f"{f:06d}.png", (H, W)); mp = mask(per_dir / "sam2_masks/0" / f"{f:06d}.png", (H, W))
        J = ours[f, TORSO]; u = fx * J[:, 0] / J[:, 2] + cx; v = fy * J[:, 1] / J[:, 2] + cy
        inside = (u >= 2) & (u < W - 2) & (v >= 2) & (v < H - 2) & (J[:, 2] > 0.1)
        rm, rv, occ = [], [], 0
        for i in np.flatnonzero(inside):
            ui, vi = int(u[i]), int(v[i])
            pm, pv = np.nanmedian(Dm[vi - 2:vi + 3, ui - 2:ui + 3]), np.nanmedian(Dv[vi - 2:vi + 3, ui - 2:ui + 3])
            occ += int(mo[vi, ui])
            if np.isfinite(pm) and np.isfinite(pv) and pm > 0 and pv > 0:
                rm.append(pm / J[i, 2]); rv.append(pv / J[i, 2])
        if len(rm) >= 3:
            raw["h_moge"][f], raw["h_vda"][f] = np.log(np.median(rm)) - np.log(am[f]), np.log(np.median(rv)) - np.log(av[f])
        hv[f] = len(rm) / len(TORSO); ho[f] = occ / max(inside.sum(), 1)
        Ja = ours[f]; ua = fx * Ja[:, 0] / Ja[:, 2] + cx; va = fy * Ja[:, 1] / Ja[:, 2] + cy
        hb[f] = min(ua.min(), W - ua.max(), va.min(), H - va.max()) / min(H, W)
        if mo.sum() > 50:
            okm[f] = 1; zo = P[f, :, 2].mean()
            dm, dv = Dm[mo], Dv[mo]; dm, dv = dm[np.isfinite(dm) & (dm > 0)], dv[np.isfinite(dv) & (dv > 0)]
            if len(dm) > 30 and len(dv) > 30 and zo > 0.1:
                mdm, mdv = np.median(dm), np.median(dv)
                raw["o_moge"][f], raw["o_vda"][f] = np.log(mdm / zo) - np.log(am[f]), np.log(mdv / zo) - np.log(av[f])
                raw["o_sm"][f] = 1.4826 * np.median(np.abs(dm - mdm)) / mdm; raw["o_sv"][f] = 1.4826 * np.median(np.abs(dv - mdv)) / mdv
            pu = fx * P[f, :, 0] / P[f, :, 2] + cx; pv_ = fy * P[f, :, 1] / P[f, :, 2] + cy
            ext = np.hypot(pu.max() - pu.min(), pv_.max() - pv_.min())
            if ext > 2:
                raw["o_size"][f] = np.log(np.sqrt(mo.sum()) / ext)
            g = np.zeros((H // 8 + 1, W // 8 + 1), bool); ok = (pu >= 0) & (pu < W) & (pv_ >= 0) & (pv_ < H)
            g[(pv_[ok] / 8).astype(int), (pu[ok] / 8).astype(int)] = True; g = cv2.dilate(g.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            ms = mo[::8, ::8]; hh, ww = min(g.shape[0], ms.shape[0]), min(g.shape[1], ms.shape[1]); g, ms = g[:hh, :ww], ms[:hh, :ww]
            raw["o_iou"][f] = (g & ms).sum() / max((g | ms).sum(), 1); raw["o_occ"][f] = (mo & mp).sum() / mo.sum()
    if no_vda:
        for k in ("h_vda", "o_vda", "o_sv"):
            raw[k][:] = np.nan
    fh = {"h_moge": anchored(raw["h_moge"], a), "h_vda": anchored(raw["h_vda"], a)}
    fh["h_dis"] = fh["h_vda"] - fh["h_moge"]; fh["h_valid"], fh["h_occ"], fh["h_border"] = hv, ho, hb
    kp, kpc = vc["kp"][:T].astype(np.float64), vc["kpc"][:T].astype(np.float64); res, conf = np.full(T, np.nan), np.zeros(T)
    for f in range(T):
        pj = ours[f]; pu, pv_ = pj[:, 0] / pj[:, 2], pj[:, 1] / pj[:, 2]; d, c = [], []
        for sc, mpi in SC2MP.items():
            if kpc[f, mpi] > 0.5:
                d.append(np.hypot(pu[sc] - kp[f, mpi, 0], pv_[sc] - kp[f, mpi, 1])); c.append(kpc[f, mpi])
        ext = np.hypot(pu.max() - pu.min(), pv_.max() - pv_.min())
        if len(d) >= 4 and ext > 0:
            res[f] = np.median(d) / ext
        conf[f] = np.mean(kpc[f, list(SC2MP.values())])
    fh["h_kp"], fh["h_kpc"] = res, conf
    for k, src in (("c_size", "c_size_rel"), ("n_size", "n_size"), ("c_floor", "c_floor"), ("st_floor", "st_floor")):
        fh[k] = cz[src][:T].astype(np.float64) if cz is not None and src in cz.files else np.full(T, np.nan)
    c_ours, c_gt = ours.mean(1), gt.mean(1)
    fh["h_vr"], fh["h_vlat"], fh["h_ar"] = kinematics(c_ours)
    fo = {"o_moge": anchored(raw["o_moge"], a), "o_vda": anchored(raw["o_vda"], a)}
    fo["o_dis"] = fo["o_vda"] - fo["o_moge"]; fo["o_spread_m"], fo["o_spread_v"] = raw["o_sm"], raw["o_sv"]
    fo["o_size"] = anchored(raw["o_size"], a); fo["o_iou"], fo["o_mask"], fo["o_occ"] = raw["o_iou"], okm, raw["o_occ"]
    rt = oc["raw_t"][:T].astype(np.float64); rok = oc["raw_ok"][:T].astype(bool) if "raw_ok" in oc.files else np.ones(T, bool)
    fo["o_fpjump"] = np.where(rok, np.linalg.norm(rt - t[:T], axis=1) / np.linalg.norm(t[:T], axis=1), 0.0)
    o_ours, o_gt = P.mean(1), Pg.mean(1)
    fo["o_vr"], fo["o_vlat"], fo["o_ar"] = kinematics(o_ours)
    fo["contact"] = np.array([np.sqrt(((hands[f][:, None] - P[f][None]) ** 2).sum(-1)).min() for f in range(T)])
    tgt_h = np.log(np.linalg.norm(c_gt, axis=1) / np.linalg.norm(c_ours, axis=1)); tgt_h -= tgt_h[a]
    tgt_o = np.log(np.linalg.norm(o_gt, axis=1) / np.linalg.norm(o_ours, axis=1)); tgt_o -= tgt_o[a]
    err_h = np.linalg.norm(ours - gt, axis=-1).mean(1)
    err_o = np.array([np.sqrt(((P[f][:, None] - Pg[f][None]) ** 2).sum(-1)).min(1).mean() for f in range(T)])
    gv = next((oc[k][:T].astype(bool) for k in ("gt_valid", "gt_ok", "valid") if k in oc.files), np.ones(T, bool))
    valid_o = gv & np.isfinite(tgt_o) & (okm > 0)
    if DEPLOY:                                                                         # no GT: targets unknown
        tgt_h[:] = np.nan; tgt_o[:] = np.nan; err_h[:] = np.nan; err_o[:] = np.nan; valid_h[:] = False; valid_o[:] = False
    np.savez_compressed(out / f"{pref}_{tag}.npz", fh=np.stack([fh[k] for k in HN], 1).astype(np.float32),
                        fo=np.stack([fo[k] for k in ON], 1).astype(np.float32), h_names=np.array(HN), o_names=np.array(ON),
                        tgt_h=tgt_h.astype(np.float32), tgt_o=tgt_o.astype(np.float32), err_h=err_h.astype(np.float32),
                        err_o=err_o.astype(np.float32), valid_h=valid_h, valid_o=valid_o, anchor=a,
                        raw_h_moge=raw["h_moge"].astype(np.float32), raw_h_vda=raw["h_vda"].astype(np.float32),
                        raw_o_moge=raw["o_moge"].astype(np.float32), raw_o_vda=raw["o_vda"].astype(np.float32),
                        dist_h=np.linalg.norm(c_ours, axis=1).astype(np.float32), dist_o=np.linalg.norm(o_ours, axis=1).astype(np.float32))
    cov = np.isfinite(raw["h_moge"]).mean(); covo = np.isfinite(raw["o_moge"]).mean()
    return tag, (f"T {T} | human depth coverage {cov * 100:.0f} % | object {covo * 100:.0f} % | tgt_h rms {np.sqrt(np.nanmean(tgt_h[valid_h] ** 2)) * 100:.2f} % "
                 f"| tgt_o rms {np.sqrt(np.nanmean(tgt_o[valid_o] ** 2)) * 100 if valid_o.any() else float('nan'):.2f} % | object cache keys {sorted(oc.files)[:6]}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", type=Path, required=True); ap.add_argument("--mode", choices=("full", "bm"), required=True)
    ap.add_argument("--prefix", required=True); ap.add_argument("--out", type=Path, required=True); ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--no_vda", action="store_true", help="MoGe-only features (VDA columns NaN; roots without VDA maps)")
    ap.add_argument("--deploy", action="store_true", help="episodes without GT (targets NaN)")
    ap.add_argument("--dirs", help="human=DIR,object=DIR,visual=DIR,cues=DIR,lock=DIR (deployment caches instead of FORMHOI_DIR)")
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    global DIRS, DEPLOY
    DEPLOY = a.deploy
    if a.dirs:
        DIRS = {k: Path(v) for k, v in (x.split("=", 1) for x in a.dirs.split(","))}
    hdir = DIRS["human"] if DIRS else F / ("joint_cache/human" if a.mode == "full" else "joint_cache_bm/human")
    tags = sorted(p.stem[len(a.prefix) + 1:] for p in hdir.glob(f"{a.prefix}_episode_*.npz"))
    with Pool(a.workers) as pool:
        for tag, msg in pool.imap(episode, [(a.root, a.mode, a.prefix, a.out, t, a.no_vda) for t in tags]):
            print(a.prefix, a.mode, tag, msg, flush=True)
    print(f"PASS: rel_cache {a.prefix} {a.mode}: {len(list(a.out.glob(a.prefix + '_episode_*.npz')))} episodes")


if __name__ == "__main__":
    main()
