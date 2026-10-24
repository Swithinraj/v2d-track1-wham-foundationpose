#!/usr/bin/env python3
"""Joint human-object refiner (mhr_converter.sif, GPU): one integrated model trained end to end on real pipeline errors
(ext/joint_cache_human.py + ext/joint_cache_object.py) against FORM-HOI GT in the evaluator's frame.

  human observations  -> human encoder  (bands of ours/raw joints, ViTPose confidence, mask visibility; bi-GRU)  -+
  object observations -> object encoder (track vs raw FoundationPose bands, rotation disagreement, SAM 2 mask      +-> interaction -> heads -> unrolled joint optimiser
                                         render residual, MoGe depth in the mask, motion, validity flags; bi-GRU)  -+
Interaction: per frame, four tokens (body, hand 42, hand 78, object) carrying their geometric relation (SDF hand-object
distance, hand -> object vector and relative velocity in the object's ray basis, root -> object), a 2-layer transformer
over the tokens (hand <-> object surface, body <-> object centre), then a bi-GRU over time (neighbouring frames).
Heads (zero-initialised: the model starts at E2 for the human and at our track for the object):
  human     per-region E2 band-coefficient corrections (as the GRU motion head) + a bounded per-hand offset
  object    bounded lateral / camera-ray depth / rotation twist correction (about the surface centroid)
  uncertainty  log sigma per human region, per hand, object lateral / depth / rotation
  contact   per-hand contact logit + hand -> contact-surface vector
Branch heads on the encoders alone give intermediate outputs (supervised, so neither branch goes idle); the interaction
heads add residuals to them.
Unrolled optimiser (K Gauss-Newton-like steps, learned step sizes, differentiable): per-frame object translation and hand
offsets minimising  sum p_contact d_SDF(hand, object)^2 / tau^2 + one-sided penetration + uncertainty-weighted priors
(sigma from the heads: whichever estimate is less certain absorbs the correction) + second-difference smoothness
+ contact held in object coordinates across consecutive contact frames. The first scored human frame stays pinned.
Loss on every output level (branch 0.3, interaction 0.5, final 1.0): joint + hand position, CD-O-style surface Chamfer,
human and object acceleration error against GT, contact BCE + vector + distance in true contact, one-sided penetration
(scorer-consistent SDF grids), mask-centroid render residual, heteroscedastic uncertainty NLL on detached errors.
Augmentation: mirror, roll and scale about the camera, time reversal, raw-error amplitude, smooth object depth
perturbation, and source degradation with validity flags (object hidden, wrist confidence lowered, hand masked).
Schedule: warm-start branches -> interaction -> optimiser -> end-to-end fine-tuning at lower learning rate.
Validation (whole episodes, every 250 steps): CD-H / ACC-H on the 22 scored joints, CD-O (surface Chamfer), ACC-O
(translation, the scorer's formula), PEN proxy (mean SDF penetration of the hand points), all x the evaluator scale, for
ours, E2 and each level; paired deltas and better-counts.
Usage: joint_train.py CACHE_DIR OUT_DIR --train ROOT[,ROOT] --val ROOT[,ROOT] [--steps a,b,c,d] [--eval_only MODEL]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gru_train import MIRROR, NB, NREG, RAMP, REGION, SMOOTH, accel, bands, e2_base, features as human_features, gauss_t  # noqa: E402

HREG = np.r_[np.full(64, 2), np.full(64, 1)]          # hand block 0 = wrist 42 (right arm region 2), block 1 = wrist 78 (left, 1)
HJ = ((21, 19), (20, 18))                              # per hand block: (wrist, elbow) scored-joint index
OBJ_SMOOTH = 4.0
BOUND = torch.tensor([0.10, 0.10, 0.40, 0.30, 0.30, 0.30])   # lateral x/y, depth (m), rotation twist (rad)
HAND_BOUND = 0.05
TAU, BETA = 0.01, 300.0                                # contact scale (m), soft-min sharpness (1/m)
LAM_PEN, LAM_SM, LAM_OC = 1.0, 1e4, 1e3
W_ACC, W_TV, W_ACC_O = 50.0, 5.0, 50.0
LEVEL_W = {"branch": 0.3, "inter": 0.5, "final": 1.0}
CONTACT = 0.015
# visual evidence (--visual): ViTPose body landmarks (MediaPipe layout) of the 12 mapped scored joints, the mirror map of
# the 33 landmarks, crop / CNN sizes
SC = [1, 2, 18, 35, 3, 19, 36, 4, 20, 37, 8, 24, 110, 74, 38, 113, 75, 39, 76, 40, 78, 42]
LMK = {2: 23, 18: 24, 3: 25, 19: 26, 4: 27, 20: 28, 75: 11, 39: 12, 76: 13, 40: 14, 78: 15, 42: 16}
JI, MI = [SC.index(j) for j in LMK], list(LMK.values())
MPM = [0, 4, 5, 6, 1, 2, 3, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17, 20, 19, 22, 21, 24, 23, 26, 25, 28, 27, 30, 29, 32, 31]
HAND_MP = ((16, 14), (15, 13))   # hand block -> MediaPipe wrist, elbow
HV = 64                          # CNN embedding per crop
VIS_KEYS = ("kp", "kpc", "hc2d", "h2s", "boxes", "box_src", "mask_m")
GAP_OBJ_ONLY = False   # --gap_obj_only: the true-contact gap term moves only the object (the human is held by its GT losses)
NO_HAND_HEAD = False   # --no_hand_head: no direct hand offsets from the network (no image evidence of hand position)
OPT_OBJ_ONLY = False   # --opt_obj_only: the unrolled optimiser moves only the object
GATE = False           # --gate: a learned per-frame gate (smoothed) scales ALL object corrections; LAM_GATE x mean gate
LAM_GATE = 0.5         #   penalises using it, so well-tracked objects stay untouched (corrections only where they pay)
SIM_FAIL = 0.0         # --sim_fail p: windows with simulated human pipeline failure (raw error x2-6, no position loss)
W_PEN = 2.0            # --w_pen: one-sided penetration weight (cm)


# ---------------------------------------------------------------- geometry
def rodrigues(v):
    th = v.norm(dim=-1, keepdim=True).clamp_min(1e-8); k = v / th
    K = torch.zeros(*v.shape[:-1], 3, 3, device=v.device, dtype=v.dtype)
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 2] = -k[..., 2], k[..., 1], -k[..., 0]
    K = K - K.transpose(-1, -2)
    th = th[..., None]
    return torch.eye(3, device=v.device, dtype=v.dtype) + torch.sin(th) * K + (1 - torch.cos(th)) * K @ K


def sdf_query(grid, lo, hi, P):
    """P [..., 3] local mesh units -> signed distance (local units), trilinear inside the grid; outside it the border value
    plus the distance to the grid box (an upper bound that still points toward the object)."""
    sh = P.shape[:-1]; P = P.reshape(-1, 3)
    g = (P - lo) / (hi - lo) * 2 - 1
    gc = g.clamp(-1, 1)
    n = torch.tensor(grid.shape, device=P.device)                 # grid[ix, iy, iz] (contact_sdf.py layout)
    idx = torch.nan_to_num((gc + 1) / 2 * (n - 1))
    i0 = idx.detach().floor().clamp(min=torch.zeros_like(n), max=n - 2).long(); f = idx - i0
    v = 0.0                                                       # hand-written trilinear: twice differentiable (grid_sample 3D is not)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                wgt = (f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1]) * (f[:, 2] if dz else 1 - f[:, 2])
                v = v + wgt * grid[i0[:, 0] + dx, i0[:, 1] + dy, i0[:, 2] + dz]
    out = (((P - (lo + (gc + 1) / 2 * (hi - lo))) ** 2).sum(-1) + 1e-12).sqrt() - 1e-6   # smooth at 0 (inside the grid)
    return (v + out).reshape(sh)


def signed(w, X, R, t):
    """Points X [L, N, 3] (our frame) -> signed distance to our object posed (R, t, s) [L, N] (our-frame metres)."""
    loc = torch.einsum("lij,lni->lnj", R, X - t[:, None]) / w["os"]
    return sdf_query(w["sdf"], w["lo"], w["hi"], loc) * w["os"]


def centroid(R, t, s, surf):
    return s * torch.einsum("lij,j->li", R, surf.mean(0)) + t


def ray_basis(c):
    ray = c / c.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    x = torch.zeros_like(ray); x[:, 0] = 1
    ex = x - (x * ray).sum(-1, keepdim=True) * ray; ex = ex / ex.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return ex, torch.cross(ray, ex, dim=-1), ray


def rot_angle(A, B):
    return torch.arccos(((torch.einsum("lij,lij->l", A, B) - 1) / 2).clamp(-1 + 1e-6, 1 - 1e-6))


def softmin(x, dim):
    return -torch.logsumexp(-BETA * x, dim=dim) / BETA


def chamfer(A, B):
    d = torch.cdist(A, B)
    return 0.5 * (d.min(-1).values.mean(-1) + d.min(-2).values.mean(-1))


# ---------------------------------------------------------------- data
class Episode:
    def __init__(self, hpath, opath, dev, vpath=None):
        h, o = np.load(hpath), np.load(opath)
        T = lambda k, src: torch.as_tensor(src[k], dtype=torch.float32, device=dev)  # noqa: E731
        self.name = hpath.stem
        e = {k: T(k, h) for k in ("ours", "raw", "gt", "conf", "vis", "hands_ours", "hands_raw", "hands_gt")}
        e.update({k: T(k, o) for k in ("obj_R", "obj_t", "raw_R", "raw_t", "raw_ok", "gt_R", "gt_t", "surf_ours", "surf_gt",
                                         "mask_area", "mask_u", "mask_v", "mask_depth", "depth_ok", "gt_dist", "gt_cvec", "sdf", "sdf_lo", "sdf_hi")})
        L = len(e["ours"])
        for k in ("obj_R", "obj_t", "raw_R", "raw_t", "raw_ok", "gt_R", "gt_t", "mask_area", "mask_u", "mask_v", "mask_depth", "depth_ok", "gt_dist", "gt_cvec"):
            e[k] = e[k][:L]
        self.L = L
        bad = ~torch.isfinite(e["raw_t"]).all(-1) | ~torch.isfinite(e["raw_R"]).flatten(1).all(-1)   # raw FoundationPose gaps
        e["raw_t"][bad] = e["obj_t"][bad]; e["raw_R"][bad] = e["obj_R"][bad]
        for k in list(e):
            e[k] = torch.nan_to_num(e[k])
        e["rt_ok"] = (~bad).float()
        self.visual = vpath is not None
        if self.visual:                                               # image evidence (ext/joint_cache_visual.py)
            vz = np.load(vpath)
            for k in ("kp", "kpc", "boxes", "box_src", "mask_m"):
                e[k] = torch.nan_to_num(torch.as_tensor(vz[k][:L], dtype=torch.float32, device=dev))
            h2 = torch.as_tensor(vz["hand2d"][:L], dtype=torch.float32, device=dev)
            e["hc2d"] = torch.nan_to_num(h2.mean(2)); e["h2s"] = torch.as_tensor(vz["hand2d_s"][:L], dtype=torch.float32, device=dev)
            e["crops"] = torch.as_tensor(vz["crops"][:L])                # uint8 [L, 3, S, S, 3], stays on the CPU
            e["roll"] = torch.zeros(L, device=dev)
        self.valid = torch.as_tensor(h["valid"], device=dev)[:L]; self.anchor = int(h["anchor"]); self.s_sim = float(h["sim3_s"])
        self.os, self.gs, self.sdf_ok = float(o["obj_s"]), float(o["gt_s"]), float(o["sdf_ok"])
        t = torch.arange(L, dtype=torch.float32, device=dev)
        ramp = torch.clamp((t - self.anchor).abs() / RAMP, 0, 1); e["ramp"] = 0.5 - 0.5 * torch.cos(np.pi * ramp)
        e["bo"], e["br"], e["bg"] = (bands(e[k].double()).float() for k in ("ours", "raw", "gt"))
        e["bho"], e["bhr"], e["bhg"] = (bands(e[k].double()).float() for k in ("hands_ours", "hands_raw", "hands_gt"))
        e["bto"], e["btr"] = bands(e["obj_t"].double()).float(), bands(e["raw_t"].double()).float()
        self.e = e
        v = self.valid
        self.h_err = float((e["ours"] - e["gt"]).norm(dim=-1).mean(-1)[v].mean())
        self.obj_ok = False
        with torch.no_grad():
            w = self.window(slice(None))
            self.o_err = float(object_metrics(w, e["obj_R"], e["obj_t"])[0]) / 100 / self.s_sim if self.sdf_ok else 9.0
        self.obj_ok = self.sdf_ok > 0 and self.o_err < 0.5

    def window(self, sl):
        w = {k: (v[:, sl] if k in ("bo", "br", "bg", "bho", "bhr", "bhg", "bto", "btr") else
                 v if k in ("surf_ours", "surf_gt", "sdf", "sdf_lo", "sdf_hi") else v[sl]) for k, v in self.e.items()}
        w["valid"] = self.valid[sl]
        w.update(os=self.os, gs=self.gs, s_sim=self.s_sim, lo=self.e["sdf_lo"], hi=self.e["sdf_hi"], obj_ok=float(self.obj_ok))
        return w


def augment(w, rng):
    """Same physical transform on every input and target. Mirror (camera x; joints / hands / contact labels swap sides;
    rotations F R F, local surfaces and SDF grid mirrored), roll + scale about the camera, time reversal, raw-error
    amplitude, smooth object depth perturbation (our track only: the model must find the error from evidence), and source
    degradation with validity flags."""
    dev = w["ours"].device
    F = torch.tensor([-1.0, 1.0, 1.0], device=dev)
    pos = ("ours", "gt", "raw", "hands_ours", "hands_raw", "hands_gt", "obj_t", "raw_t", "gt_t", "gt_cvec")
    bnd = ("bo", "br", "bg", "bho", "bhr", "bhg", "bto", "btr")
    if rng.random() < 0.5:
        jm = torch.as_tensor(MIRROR, device=dev); hm = torch.cat([torch.arange(64, 128), torch.arange(0, 64)]).to(dev)
        for k in pos + bnd:
            w[k] = w[k] * F
        for k in ("ours", "gt", "raw", "conf", "vis"):
            w[k] = w[k][:, jm]
        for k in ("bo", "br", "bg"):
            w[k] = w[k][:, :, jm]
        for k in ("hands_ours", "hands_raw", "hands_gt"):
            w[k] = w[k][:, hm]
        for k in ("bho", "bhr", "bhg"):
            w[k] = w[k][:, :, hm]
        w["gt_dist"], w["gt_cvec"] = w["gt_dist"].flip(1), w["gt_cvec"].flip(1)
        Fm = torch.diag(F)
        for k in ("obj_R", "raw_R", "gt_R"):
            w[k] = Fm @ w[k] @ Fm
        w["surf_ours"], w["surf_gt"] = w["surf_ours"] * F, w["surf_gt"] * F
        w["sdf"] = w["sdf"].flip(0)
        w["lo"], w["hi"] = torch.stack([-w["hi"][0], w["lo"][1], w["lo"][2]]), torch.stack([-w["lo"][0], w["hi"][1], w["hi"][2]])
        w["mask_u"] = -w["mask_u"]
        if "crops" in w:
            mm = torch.as_tensor(MPM, device=dev)
            w["kp"] = (w["kp"] * torch.tensor([-1.0, 1.0], device=dev))[:, mm]; w["kpc"] = w["kpc"][:, mm]
            w["hc2d"] = (w["hc2d"] * torch.tensor([-1.0, 1.0], device=dev)).flip(1); w["h2s"] = w["h2s"].flip(1)
            b = w["boxes"] * torch.tensor([-1.0, 1.0, 1.0], device=dev); w["boxes"] = b[:, [1, 0, 2]]; w["box_src"] = w["box_src"][:, [1, 0, 2]]
            w["mask_m"] = w["mask_m"] * torch.tensor([1.0, -1.0, 1.0, 1.0, 1.0], device=dev)
            w["crops"] = w["crops"].flip(3)[:, [1, 0, 2]]
    th = float(np.radians(rng.uniform(-10, 10))); sc = float(rng.uniform(0.9, 1.1))
    Rz = torch.tensor([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]], dtype=torch.float32, device=dev)
    for k in pos + bnd:
        w[k] = sc * w[k] @ Rz.T
    for k in ("obj_R", "raw_R", "gt_R"):
        w[k] = Rz @ w[k]
    u, v = w["mask_u"], w["mask_v"]
    w["mask_u"], w["mask_v"] = np.cos(th) * u - np.sin(th) * v, np.sin(th) * u + np.cos(th) * v
    w["mask_depth"] = w["mask_depth"] * sc
    if "crops" in w:                                                 # roll the image evidence with the geometry
        R2 = torch.tensor([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]], dtype=torch.float32, device=dev)
        w["kp"] = w["kp"] @ R2.T; w["hc2d"] = w["hc2d"] @ R2.T
        w["boxes"] = torch.cat([w["boxes"][..., :2] @ R2.T, w["boxes"][..., 2:]], -1)
        w["mask_m"] = torch.cat([w["mask_m"][:, :1], w["mask_m"][:, 1:3] @ R2.T, w["mask_m"][:, 3:]], -1)
        w["roll"] = w["roll"] + th
    w["os"], w["gs"], w["s_sim"] = w["os"] * sc, w["gs"] * sc, w["s_sim"] / sc
    if rng.random() < 0.5:
        for k in bnd:
            w[k] = w[k].flip(1)
        for k, val in list(w.items()):
            if torch.is_tensor(val) and k not in bnd and k not in ("surf_ours", "surf_gt", "sdf", "lo", "hi", "sdf_lo", "sdf_hi") and val.dim() >= 1 and val.shape[0] == w["ours"].shape[0]:
                w[k] = val.flip(0)
    a = float(rng.uniform(0.7, 1.3))
    w["sim_failed"] = SIM_FAIL > 0 and rng.random() < SIM_FAIL
    if w["sim_failed"]:
        a = float(rng.uniform(2.0, 6.0))
    w["br"] = w["bg"] + a * (w["br"] - w["bg"]); w["bhr"] = w["bhg"] + a * (w["bhr"] - w["bhg"])
    L = w["ours"].shape[0]
    if rng.random() < 0.5:                                           # smooth object depth / lateral perturbation of OUR track
        nz = torch.as_tensor(rng.standard_normal((L, 3)), dtype=torch.float32, device=dev)
        nz = gauss_t(nz, float(rng.uniform(8, 30))); nz = nz / nz.std(0).clamp_min(1e-6)
        ex, ey, ray = ray_basis(w["obj_t"])
        d = float(rng.uniform(0, 0.08)) * nz[:, 2:3] * ray + float(rng.uniform(0, 0.02)) * (nz[:, :1] * ex + nz[:, 1:2] * ey)
        w["obj_t"] = w["obj_t"] + d; w["bto"] = w["bto"] + bands(d.double()).float()
    for _ in range(2):                                               # source degradation (anti-collapse)
        s0 = int(rng.integers(0, max(1, L - 10))); s1 = min(L, s0 + int(rng.integers(10, 61))); r = rng.random()
        if r < 0.25:                                                 # object hidden: no mask, depth or raw track
            for k in ("mask_area", "mask_u", "mask_v", "mask_depth", "depth_ok", "raw_ok", "rt_ok") + (("mask_m",) if "crops" in w else ()):
                w[k] = w[k].clone(); w[k][s0:s1] = 0.0
            if "crops" in w:
                w["box_src"] = w["box_src"].clone(); w["box_src"][s0:s1, 2] = 0; w["crops"] = w["crops"].clone(); w["crops"][s0:s1, 2] = 0
        elif r < 0.5:                                                # wrist / elbow confidence lowered
            fct = float(rng.uniform(0, 0.3))
            w["conf"] = w["conf"].clone(); w["conf"][s0:s1, 18:22] *= fct
            if "crops" in w:
                w["kpc"] = w["kpc"].clone(); w["kpc"][s0:s1, 13:17] *= fct
        elif r < 0.65:                                               # one hand masked
            hb = int(rng.integers(0, 2)); j = list(HJ[hb]); w["vis"] = w["vis"].clone(); w["vis"][s0:s1, j] = 0.0
            if "crops" in w:
                w["crops"] = w["crops"].clone(); w["crops"][s0:s1, hb] = 0
                for k in ("h2s", "box_src"):
                    w[k] = w[k].clone(); w[k][s0:s1, hb] = 0
                w["kpc"] = w["kpc"].clone(); w["kpc"][s0:s1, list(HAND_MP[hb])] = 0
    return w


# ---------------------------------------------------------------- features
def object_features(w):
    R, t, s, surf = w["obj_R"], w["obj_t"], w["os"], w["surf_ours"]
    c = centroid(R, t, s, surf); ex, ey, ray = ray_basis(c)
    d = (w["btr"] - w["bto"]) * w["rt_ok"][None, :, None]
    lat = lambda b: torch.stack([(b * ex).sum(-1), (b * ey).sum(-1)], -1).norm(dim=-1)  # noqa: E731
    dep = lambda b: (b * ray).sum(-1)  # noqa: E731
    f = [torch.stack([lat(d[i]) for i in range(4)], 1), torch.stack([dep(d[i]) for i in range(4)], 1),
         torch.stack([lat(w["bto"][i]) for i in range(4)], 1), torch.stack([dep(w["bto"][i]).abs() for i in range(4)], 1),
         torch.stack([lat(d[4]), dep(d[4])], 1)]
    rok = w["raw_ok"] * w["rt_ok"]
    f.append(torch.stack([rot_angle(w["raw_R"], R) * rok, rok], 1))
    vis = (w["mask_area"] > 0).float()
    pu, pv = c[:, 0] / c[:, 2], c[:, 1] / c[:, 2]
    f.append(torch.stack([torch.log(w["mask_area"] + 1e-4), vis, (w["mask_u"] - pu) * vis, (w["mask_v"] - pv) * vis,
                          (w["mask_depth"] - c[:, 2]) * w["depth_ok"], w["depth_ok"]], 1))
    vel = torch.cat([c[1:] - c[:-1], c[-1:] - c[-2:-1]])
    om = torch.cat([rot_angle(R[1:], R[:-1]), rot_angle(R[-1:], R[-2:-1])])
    f.append(torch.stack([c[:, 2], lat(vel), dep(vel), om], 1))
    return torch.cat(f, 1)


def interaction_features(w, J, H, R, t):
    """Per hand block [L, 2, 10] and body-object [L, 4] relation features from the current estimate."""
    c = centroid(R, t, w["os"], w["surf_ours"]); ex, ey, ray = ray_basis(c)
    B = lambda x: torch.stack([(x * ex[:, None]).sum(-1), (x * ey[:, None]).sum(-1), (x * ray[:, None]).sum(-1)], -1)  # noqa: E731
    sd = signed(w, H, R, t)                                                     # [L, 128]
    sdb = sd.view(-1, 2, 64)
    hc = H.view(-1, 2, 64, 3).mean(2)                                           # [L, 2, 3]
    vh = torch.cat([hc[1:] - hc[:-1], hc[-1:] - hc[-2:-1]]); vo = torch.cat([c[1:] - c[:-1], c[-1:] - c[-2:-1]])
    conf = torch.stack([w["conf"][:, [j for j, _ in HJ]], w["vis"][:, [j for j, _ in HJ]]], -1)   # [L, 2, 2]
    hand = torch.cat([sdb.min(-1).values[..., None], (sdb < 0).float().mean(-1)[..., None], B(c[:, None] - hc), B(vh - vo[:, None]), conf], -1)
    body = torch.cat([B((c - J[:, 0])[:, None])[:, 0], (c - J[:, 0]).norm(dim=-1, keepdim=True)], -1)
    return hand, body


def visual_features(w):
    """2D evidence vs our current estimate (normalised camera coordinates): ViTPose joints minus our projected joints
    [12 x 2, confidence-weighted] + confidence; detected hand centre minus our projected hand + score; crop box offset and
    size per hand -> [L, 48]. Object: mask present, mask centroid minus our projected centroid, log ratio of the mask's
    spread to our projected silhouette's (a depth / scale cue), object box offset -> [L, 7]."""
    P = lambda X: X[..., :2] / X[..., 2:3].clamp_min(1e-3)  # noqa: E731
    J, H = w["ours"], w["hands_ours"]; L = len(J)
    rj = (w["kp"][:, MI] - P(J[:, JI])) * w["kpc"][:, MI, None]
    hcp = P(H.view(L, 2, 64, 3).mean(2))
    rh = (w["hc2d"] - hcp) * w["h2s"][..., None]
    bo = w["boxes"][:, :2, :2] - hcp
    vh = torch.cat([rj.flatten(1), w["kpc"][:, MI], rh.flatten(1), w["h2s"], bo.flatten(1), torch.log(w["boxes"][:, :2, 2] + 1e-3)], 1)
    Xo = w["os"] * torch.einsum("lij,nj->lni", w["obj_R"], w["surf_ours"][::4]) + w["obj_t"][:, None]
    po = P(Xo); pc = po.mean(1); sp = po.std(1)
    m = w["mask_m"]; m0 = m[:, :1]
    vo = torch.cat([m0, (m[:, 1:3] - pc) * m0, torch.log((m[:, 3:5] + 1e-4) / (sp + 1e-4)) * m0, w["boxes"][:, 2, :2] - pc], 1)
    return torch.nan_to_num(vh), torch.nan_to_num(vo)


# ---------------------------------------------------------------- model
class JointRefiner(nn.Module):
    def __init__(self, nfh, nfo, nfi=10, nfb=4, H=128, K=4, visual=False, pretrained=True):
        super().__init__()
        self.K = K; self.visual = visual
        if visual:                                                    # trainable CNN on the hand / object crops
            import torchvision
            r = torchvision.models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)   # deployment: weights from the checkpoint
            self.cnn = nn.Sequential(*list(r.children())[:-1]); self.cnn_proj = nn.Linear(512, HV)
            self.crop_emb = nn.Parameter(torch.zeros(3, HV))
            self.register_buffer("img_mu", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("img_sd", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
            nfh, nfo, nfi, nfb = nfh + 48 + 2 * HV, nfo + 7 + HV, nfi + HV, nfb + HV
        self.h_inp = nn.Sequential(nn.Linear(nfh, H), nn.GELU()); self.h_gru = nn.GRU(H, H, 2, bidirectional=True, batch_first=True)
        self.o_inp = nn.Sequential(nn.Linear(nfo, H), nn.GELU()); self.o_gru = nn.GRU(H, H, 2, bidirectional=True, batch_first=True)
        self.h_branch = nn.Linear(2 * H, NREG * 16); self.o_branch = nn.Linear(2 * H, 6)
        self.tok_b = nn.Linear(2 * H, H); self.tok_h = nn.Linear(2 * H + nfi, H); self.tok_o = nn.Linear(2 * H + nfb, H)
        self.typ = nn.Parameter(torch.zeros(4, H))
        self.inter = nn.TransformerEncoder(nn.TransformerEncoderLayer(H, 4, 2 * H, dropout=0.0, batch_first=True, norm_first=True), 2)
        self.t_gru = nn.GRU(4 * H, H, 1, bidirectional=True, batch_first=True)
        self.head_h = nn.Linear(3 * H, NREG * 16); self.head_hand = nn.Linear(3 * H, 3); self.head_o = nn.Linear(3 * H, 6)
        self.head_unc = nn.Linear(6 * H, NREG + 2 + 3); self.head_contact = nn.Linear(3 * H, 4)
        self.head_gate = nn.Linear(3 * H, 1); nn.init.zeros_(self.head_gate.weight); nn.init.constant_(self.head_gate.bias, -2.0)
        for m in (self.h_branch, self.o_branch, self.head_h, self.head_hand, self.head_o, self.head_unc):
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)
        nn.init.constant_(self.head_unc.bias, -2.5)                              # sigma ~ 8 cm at start
        self.step = nn.Parameter(torch.full((K, 2), 0.3))                        # softplus(0.3) ~ 0.85

    def train(self, mode=True):
        super().train(mode)
        if self.visual:                                               # frozen BatchNorm statistics (pretrained CNN)
            for m in self.cnn.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        return self

    def crop_features(self, crops, roll, stride=2, chunk=1024):
        """crops uint8 [B, L, 3, S, S, 3] (GPU), roll [B] (radians, augmentation) -> [B, L, 3, HV]; the CNN runs on every
        stride-th frame and its features are linearly interpolated in time."""
        B, L = crops.shape[:2]
        idx = torch.arange(0, L, stride, device=crops.device)
        if idx[-1] != L - 1:
            idx = torch.cat([idx, torch.tensor([L - 1], device=crops.device)])
        x = crops[:, idx].permute(0, 1, 2, 5, 3, 4).float().div(255)                 # [B, Ls, 3, 3, S, S]
        Ls, S = len(idx), x.shape[-1]
        if roll is not None and bool((roll != 0).any()):
            c, s_ = torch.cos(roll), torch.sin(roll)
            A = torch.zeros(B, 2, 3, device=x.device); A[:, 0, 0], A[:, 0, 1], A[:, 1, 0], A[:, 1, 1] = c, s_, -s_, c
            A = A[:, None].expand(B, Ls * 3, 2, 3).reshape(-1, 2, 3)
            g = Fn.affine_grid(A, (B * Ls * 3, 3, S, S), align_corners=False)
            x = Fn.grid_sample(x.reshape(-1, 3, S, S), g, align_corners=False).view(x.shape)
        x = (x.reshape(-1, 3, S, S) - self.img_mu) / self.img_sd
        f = torch.cat([self.cnn(x[i:i + chunk]).flatten(1) for i in range(0, len(x), chunk)])
        f = self.cnn_proj(f).view(B, Ls, 3, HV) + self.crop_emb
        t = idx.float()
        full = torch.arange(L, device=crops.device, dtype=torch.float32)
        j = torch.searchsorted(t, full, right=True).clamp(1, Ls - 1)
        wgt = ((full - t[j - 1]) / (t[j] - t[j - 1]).clamp_min(1)).view(1, L, 1, 1)
        return f[:, j - 1] * (1 - wgt) + f[:, j] * wgt

    def encode(self, fh, fo, cf=None):
        if cf is not None:
            fh = torch.cat([fh, cf[:, :, 0], cf[:, :, 1]], -1); fo = torch.cat([fo, cf[:, :, 2]], -1)
        hh, _ = self.h_gru(self.h_inp(fh)); ho, _ = self.o_gru(self.o_inp(fo))
        return hh, ho

    def branch(self, hh, ho):
        ch = 0.5 * torch.tanh(self.h_branch(hh)); co = torch.tanh(self.o_branch(ho))
        return ch, co

    def interact(self, hh, ho, fhand, fbody, cf=None):
        B, L = hh.shape[:2]
        if cf is not None:
            fhand = torch.cat([fhand, cf[:, :, :2]], -1); fbody = torch.cat([fbody, cf[:, :, 2]], -1)
        tb = self.tok_b(hh); to = self.tok_o(torch.cat([ho, fbody], -1))
        th = self.tok_h(torch.cat([hh[:, :, None].expand(B, L, 2, hh.shape[-1]), fhand], -1))
        tok = torch.cat([tb[:, :, None], th, to[:, :, None]], 2) + self.typ                  # [B, L, 4, H]
        z = self.inter(tok.reshape(B * L, 4, -1)).view(B, L, 4, -1)
        ctx, _ = self.t_gru(z.reshape(B, L, -1))                                             # [B, L, 2H]
        zb, zh, zo = z[:, :, 0], z[:, :, 1:3], z[:, :, 3]
        ch = 0.5 * torch.tanh(self.head_h(torch.cat([ctx, zb], -1)))
        cxh = torch.cat([ctx[:, :, None].expand(B, L, 2, ctx.shape[-1]), zh], -1)
        dh = HAND_BOUND * torch.tanh(self.head_hand(cxh))                                    # [B, L, 2, 3]
        if NO_HAND_HEAD:
            dh = torch.zeros_like(dh)
        co = torch.tanh(self.head_o(torch.cat([ctx, zo], -1)))
        unc = self.head_unc(torch.cat([ctx, z.reshape(B, L, -1)], -1))
        con = self.head_contact(cxh)                                                         # [B, L, 2, 4]
        g = torch.sigmoid(self.head_gate(torch.cat([ctx, zo], -1))) if GATE else torch.ones_like(co[..., :1])
        return ch, dh, co, unc, con, g


def smooth_t(x, s):
    return torch.stack([gauss_t(xi, s) for xi in x])


def fuse(bo, br, ours, ramp, corr, base, region):
    """E2 band fusion with per-region coefficient corrections, for joints (region = REGION) or hand points (HREG)."""
    T = bo.shape[1]
    coef = base.to(bo.device).view(1, 1, 2, 2, NB).expand(T, NREG, 2, 2, NB).clone()
    if corr is not None:
        coef[..., :4] = coef[..., :4] + corr
    cj = coef[:, torch.as_tensor(region, device=bo.device)]
    k, m = cj[:, :, :, 0].permute(3, 0, 1, 2), cj[:, :, :, 1].permute(3, 0, 1, 2)
    axis = torch.tensor([0, 0, 1], device=bo.device)
    return ours + ramp[:, None, None] * ((k[..., axis] - 1) * bo + m[..., axis] * br).sum(0)


def apply_hand(J, H, dh, ramp):
    """Per-hand offsets [L, 2, 3] -> hand points, wrist (full) and elbow (half), with the anchor taper."""
    d = ramp[:, None, None] * dh
    H = H + d.repeat_interleave(64, 1)
    J = J.clone()
    for b, (wj, ej) in enumerate(HJ):
        J[:, wj] = J[:, wj] + d[:, b]; J[:, ej] = J[:, ej] + 0.5 * d[:, b]
    return J, H


def apply_obj(w, R, t, o):
    """o [L, 6] (bounded, metres / radians): lateral x / y and depth along the ray through the surface centroid, rotation
    twist about the centroid. Returns the corrected (R, t)."""
    c = centroid(R, t, w["os"], w["surf_ours"]); ex, ey, ray = ray_basis(c)
    dR = rodrigues(o[:, 3:])
    t2 = torch.einsum("lij,lj->li", dR, t - c) + c + o[:, :1] * ex + o[:, 1:2] * ey + o[:, 2:3] * ray
    return dR @ R, t2


def optimise(model, w, J, H, R, t, unc, con, train, return_offsets=False, gate=None):
    """Unrolled joint optimiser: object translation offsets do [L, 3] and hand offsets dh [L, 2, 3]."""
    L = len(t)
    pc = torch.sigmoid(con[..., 0])                                              # [L, 2]
    sig_h = torch.exp(unc[:, NREG:NREG + 2]).clamp(0.005, 0.5)                   # per hand
    sig_o = torch.exp(unc[:, NREG + 2:NREG + 4]).clamp(0.005, 1.0)               # lateral, depth
    ex, ey, ray = (b.detach() for b in ray_basis(centroid(R, t, w["os"], w["surf_ours"]).detach()))
    ramp = w["ramp"]
    do = torch.zeros(L, 3, device=t.device, requires_grad=True); dh = torch.zeros(L, 2, 3, device=t.device, requires_grad=True)
    Ho = (2 / sig_o.min(-1).values ** 2 + 2 * (pc.sum(-1) + 2 * LAM_PEN) / TAU ** 2 + 12 * LAM_SM)[:, None]
    Hh = (2 / sig_h ** 2 + 2 * (pc + LAM_PEN) / TAU ** 2 + 12 * LAM_SM)[..., None]
    with torch.enable_grad():
        for k in range(model.K):
            Jk, Hk = apply_hand(J, H, dh, ramp); tk = t + do
            sd = signed(w, Hk, R, tk).view(L, 2, 64)
            dmin = softmin(sd, -1)
            E = (pc * dmin ** 2).sum() / TAU ** 2 + LAM_PEN * (Fn.relu(-sd) ** 2).mean(-1).sum() / TAU ** 2
            ob = torch.stack([(do * ex).sum(-1), (do * ey).sum(-1), (do * ray).sum(-1)], -1)
            E = E + ((ob[:, :2] ** 2).sum(-1) / sig_o[:, 0] ** 2 + ob[:, 2] ** 2 / sig_o[:, 1] ** 2).sum() + ((dh ** 2).sum(-1) / sig_h ** 2).sum()
            E = E + LAM_SM * ((accel(do) ** 2).sum() + (accel(dh) ** 2).sum())
            hc = Hk.view(L, 2, 64, 3).mean(2)
            loc = torch.einsum("lij,lki->lkj", R, hc - tk[:, None]) / w["os"]                 # hand centroid in object coordinates
            E = E + LAM_OC * (pc[1:] * pc[:-1] * ((loc[1:] - loc[:-1]) ** 2).sum(-1)).sum() * w["os"] ** 2
            go, gh = torch.autograd.grad(E, [do, dh], create_graph=train)
            st = Fn.softplus(model.step[k])
            do = do - st[0] * go / Ho
            if not OPT_OBJ_ONLY:
                dh = dh - st[1] * gh / Hh
    # contact / penetration corrections are low-frequency: low-pass them as the heads' corrections are (no added jerk)
    do = gauss_t(do, OBJ_SMOOTH); dh = gauss_t(dh.reshape(L, 6), SMOOTH).view(L, 2, 3)
    if gate is not None:
        do = do * gate
    Jf, Hf = apply_hand(J, H, dh, ramp)
    if return_offsets:
        return Jf, Hf, R, t + do, dh
    return Jf, Hf, R, t + do


# ---------------------------------------------------------------- metrics and losses
def contiguous(v):
    return v[2:] & v[1:-1] & v[:-2]


def object_metrics(w, R, t, n=None):
    """CD-O-style surface Chamfer (cm, x evaluator scale) and ACC-O (translation, cm/frame^2), on valid frames."""
    v = w["valid"]
    so, sg = (w["surf_ours"], w["surf_gt"]) if n is None else (w["surf_ours"][::n], w["surf_gt"][::n])
    Xo = w["os"] * torch.einsum("lij,nj->lni", R[v], so) + t[v][:, None]
    Xg = w["gs"] * torch.einsum("lij,nj->lni", w["gt_R"][v], sg) + w["gt_t"][v][:, None]
    cd = chamfer(Xo, Xg).mean() * 100 * w["s_sim"]
    ok = contiguous(v)
    acc = (accel(t) - accel(w["gt_t"])).norm(dim=-1)[ok].mean() * 100 * w["s_sim"]
    return cd, acc


def human_metrics(w, J):
    v = w["valid"]; ok = contiguous(v)
    pos = (J - w["gt"]).norm(dim=-1).mean(-1)[v].mean() * 100 * w["s_sim"]
    acc = (accel(J) - accel(w["gt"])).norm(dim=-1).mean(-1)[ok].mean() * 100 * w["s_sim"]
    return pos, acc


def pen_metric(w, H, R, t):
    sd = signed(w, H[w["valid"]], R[w["valid"]], t[w["valid"]])
    return Fn.relu(-sd).mean() * 100 * w["s_sim"]


def level_loss(w, J, H, R, t, ch, co):
    v = w["valid"]; ok = contiguous(v)
    if int(ok.sum()) < 10:
        return None
    pos = (J - w["gt"]).norm(dim=-1).mean(-1)[v].mean() * 100 + 0.5 * (H - w["hands_gt"]).norm(dim=-1).mean(-1)[v].mean() * 100
    acc = (accel(J) - accel(w["gt"])).norm(dim=-1).mean(-1)[ok].mean() * 100
    L = (0.0 if w.get("sim_failed") else pos) + W_ACC * acc + W_TV * (ch[1:] - ch[:-1]).abs().mean()
    if w["obj_ok"]:
        cd, acco = object_metrics(w, R, t, n=2)
        c = centroid(R, t, w["os"], w["surf_ours"]); vis = (w["mask_area"] > 0) & v
        rend = ((w["mask_u"] - c[:, 0] / c[:, 2]).abs() + (w["mask_v"] - c[:, 1] / c[:, 2]).abs())[vis].mean() * 100 if vis.any() else 0.0
        L = L + cd / w["s_sim"] + W_ACC_O * acco / w["s_sim"] + (co[1:] - co[:-1]).abs().mean() * 100 + rend
        sd = signed(w, H[v], R[v], t[v])
        L = L + W_PEN * Fn.relu(-sd).mean() * 100
    return L


def encoder_inputs(model, xs, MU):
    """Normalised encoder inputs for a list of windows, plus the CNN crop features when the model is visual."""
    fh = torch.stack([(human_features(x["bo"], x["br"], x["conf"], x["vis"]) - MU["h"]) / MU["hs"] for x in xs])
    fo = torch.stack([(object_features(x) - MU["o"]) / MU["os"] for x in xs])
    cf = None
    if model.visual:
        vf = [visual_features(x) for x in xs]
        fh = torch.cat([fh, torch.stack([(v[0] - MU["vh"]) / MU["vhs"] for v in vf])], -1)
        fo = torch.cat([fo, torch.stack([(v[1] - MU["vo"]) / MU["vos"] for v in vf])], -1)
        crops = torch.stack([x["crops"] for x in xs]).to(fh.device, non_blocking=True)
        cf = model.crop_features(crops, torch.stack([x["roll"][0] for x in xs]))
    return fh, fo, cf


def window_loss(model, w, MU, stage):
    """All output levels for one window batch -> total loss."""
    fh, fo, cf = encoder_inputs(model, w, MU)
    hh, ho = model.encode(fh, fo, cf)
    chb, cob = model.branch(hh, ho)
    chb = smooth_t(chb, SMOOTH); cob = smooth_t(cob, OBJ_SMOOTH) * BOUND.to(cob.device)
    base = e2_base(); total = 0.0; n = 0
    outs = []
    for b, x in enumerate(w):
        cb = chb[b].view(-1, NREG, 2, 2, 4)
        J = fuse(x["bo"], x["br"], x["ours"], x["ramp"], cb, base, REGION); H = fuse(x["bho"], x["bhr"], x["hands_ours"], x["ramp"], cb, base, HREG)
        R, t = apply_obj(x, x["obj_R"], x["obj_t"], cob[b])
        outs.append((J, H, R, t))
        lb = level_loss(x, J, H, R, t, chb[b], cob[b])
        if lb is not None:
            total = total + (LEVEL_W["branch"] if stage > 1 else 1.0) * lb; n += 1
    if stage > 1:
        fi = [interaction_features(x, *o[:2], *o[2:]) for x, o in zip(w, outs)]
        fhand = torch.stack([(f[0] - MU["i"]) / MU["is"] for f in fi]); fbody = torch.stack([(f[1] - MU["b"]) / MU["bs"] for f in fi])
        chi, dhi, coi, unc, con, gate = model.interact(hh, ho, fhand, fbody, cf)
        chi = smooth_t(chi, SMOOTH); coi = smooth_t(coi, OBJ_SMOOTH) * BOUND.to(coi.device); gate = smooth_t(gate, OBJ_SMOOTH)
        dhi = smooth_t(dhi.flatten(2), SMOOTH).view(dhi.shape)
        for b, x in enumerate(w):
            c = (chb[b] + chi[b]).view(-1, NREG, 2, 2, 4)
            J = fuse(x["bo"], x["br"], x["ours"], x["ramp"], c, base, REGION); H = fuse(x["bho"], x["bhr"], x["hands_ours"], x["ramp"], c, base, HREG)
            J, H = apply_hand(J, H, dhi[b], x["ramp"])
            R, t = apply_obj(x, x["obj_R"], x["obj_t"], gate[b] * (cob[b] + coi[b]))
            li = level_loss(x, J, H, R, t, chb[b] + chi[b], gate[b] * (cob[b] + coi[b]))
            if li is None:
                continue
            total = total + LEVEL_W["inter"] * li + (LAM_GATE * gate[b][x["valid"]].mean() if GATE else 0.0)
            v = x["valid"]; lab = (x["gt_dist"] < CONTACT).float()
            total = total + Fn.binary_cross_entropy_with_logits(con[b][..., 0][v], lab[v])
            m = (lab > 0) & v[:, None]
            if m.any():
                total = total + 0.1 * (con[b][..., 1:] - x["gt_cvec"]).norm(dim=-1)[m].mean() * 100
            if stage > 2:
                Jf, Hf, Rf, tf = optimise(model, x, J, H, R, t, unc[b], con[b], train=True, gate=gate[b] if GATE else None)
                lf = level_loss(x, Jf, Hf, Rf, tf, chb[b] + chi[b], gate[b] * (cob[b] + coi[b]))
                total = total + LEVEL_W["final"] * lf
                J, H, R, t = Jf, Hf, Rf, tf
            if m.any() and x["obj_ok"]:                                          # close the gap in true contact
                sd = signed(x, H.detach() if GAP_OBJ_ONLY else H, R, t).view(-1, 2, 64)
                total = total + 0.5 * softmin(sd, -1)[m].abs().mean() * 100
            # heteroscedastic uncertainty on detached errors
            er = torch.stack([(J - x["gt"]).norm(dim=-1)[:, np.flatnonzero(REGION == r)].mean(-1) for r in range(NREG)], -1).detach()
            eh = (H - x["hands_gt"]).norm(dim=-1).view(-1, 2, 64).mean(-1).detach()
            ls = unc[b]
            nll = (0.5 * (er / ls[:, :NREG].exp()) ** 2 + ls[:, :NREG])[v].mean() + (0.5 * (eh / ls[:, NREG:NREG + 2].exp()) ** 2 + ls[:, NREG:NREG + 2])[v].mean()
            if x["obj_ok"]:
                cg = centroid(x["gt_R"], x["gt_t"], x["gs"], x["surf_gt"]); cc = centroid(R, t, x["os"], x["surf_ours"])
                ex, ey, ray = ray_basis(cc.detach()); d = (cc - cg).detach()
                eo = torch.stack([torch.stack([(d * ex).sum(-1), (d * ey).sum(-1)], -1).norm(dim=-1), (d * ray).sum(-1).abs()], -1)
                nll = nll + (0.5 * (eo / ls[:, NREG + 2:NREG + 4].exp()) ** 2 + ls[:, NREG + 2:NREG + 4])[v].mean()
            total = total + 0.1 * nll
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, eps, MU, stage):
    base = e2_base(); rows = {}
    for ep in eps:
        x = ep.window(slice(None)); r = {}
        R0, t0 = x["obj_R"], x["obj_t"]
        J0 = x["ours"]; H0 = x["hands_ours"]

        def met(J, H, R, t):
            h = human_metrics(x, J); o = object_metrics(x, R, t) if ep.sdf_ok else (torch.tensor(0.), torch.tensor(0.))
            p = pen_metric(x, H, R, t) if ep.sdf_ok else torch.tensor(0.)
            return [float(h[0]), float(h[1]), float(o[0]), float(o[1]), float(p)]
        r["ours"] = met(J0, H0, R0, t0)
        Je = fuse(x["bo"], x["br"], J0, x["ramp"], None, base, REGION); He = fuse(x["bho"], x["bhr"], H0, x["ramp"], None, base, HREG)
        r["e2"] = met(Je, He, R0, t0)
        if model is not None:
            fh, fo, cf = encoder_inputs(model, [x], MU)
            hh, ho = model.encode(fh, fo, cf); chb, cob = model.branch(hh, ho)
            chb = smooth_t(chb, SMOOTH); cob = smooth_t(cob, OBJ_SMOOTH) * BOUND.to(cob.device)
            cb = chb[0].view(-1, NREG, 2, 2, 4)
            J = fuse(x["bo"], x["br"], J0, x["ramp"], cb, base, REGION); H = fuse(x["bho"], x["bhr"], H0, x["ramp"], cb, base, HREG)
            R, t = apply_obj(x, R0, t0, cob[0])
            r["branch"] = met(J, H, R, t)
            if stage > 1:
                fi = interaction_features(x, J, H, R, t)
                chi, dhi, coi, unc, con, gate = model.interact(hh, ho, ((fi[0] - MU["i"]) / MU["is"])[None], ((fi[1] - MU["b"]) / MU["bs"])[None], cf)
                chi = smooth_t(chi, SMOOTH); coi = smooth_t(coi, OBJ_SMOOTH) * BOUND.to(coi.device); dhi = smooth_t(dhi.flatten(2), SMOOTH).view(dhi.shape)
                gate = smooth_t(gate, OBJ_SMOOTH)
                c = (chb[0] + chi[0]).view(-1, NREG, 2, 2, 4)
                J = fuse(x["bo"], x["br"], J0, x["ramp"], c, base, REGION); H = fuse(x["bho"], x["bhr"], H0, x["ramp"], c, base, HREG)
                J, H = apply_hand(J, H, dhi[0], x["ramp"])
                R, t = apply_obj(x, R0, t0, gate[0] * (cob[0] + coi[0]))
                r["inter"] = met(J, H, R, t)
                r["gate"] = [float(gate[0][x["valid"]].mean())]
                v = x["valid"]; lab = x["gt_dist"][v] < CONTACT; pred = con[0][..., 0][v] > 0
                r["contact"] = [float((pred & lab).sum()), float(lab.sum()), float(pred.sum())]
                if stage > 2 and ep.sdf_ok:
                    r["final"] = met(*optimise(model, x, J, H, R, t, unc[0], con[0], train=False, gate=gate[0] if GATE else None))
        r["flags"] = [float(ep.h_err < 0.30), float(ep.obj_ok)]
        rows[ep.name] = r
    return rows


@torch.no_grad()
def predict(model, x, MU, level="final"):
    """Deployment: whole-episode window x (no GT needed) -> per-frame region coefficient corrections [T, NREG, 2, 2, 4],
    total hand offsets [T, 2, 3] (head + optimiser, before the anchor taper), corrected object (R, t), joints and hands."""
    fh, fo, cf = encoder_inputs(model, [x], MU)
    hh, ho = model.encode(fh, fo, cf); chb, cob = model.branch(hh, ho)
    chb = smooth_t(chb, SMOOTH); cob = smooth_t(cob, OBJ_SMOOTH) * BOUND.to(cob.device)
    base = e2_base(); J0, H0 = x["ours"], x["hands_ours"]
    c, co, dh = chb[0], cob[0], torch.zeros(len(J0), 2, 3, device=J0.device)
    J = fuse(x["bo"], x["br"], J0, x["ramp"], c.view(-1, NREG, 2, 2, 4), base, REGION); H = fuse(x["bho"], x["bhr"], H0, x["ramp"], c.view(-1, NREG, 2, 2, 4), base, HREG)
    R, t = apply_obj(x, x["obj_R"], x["obj_t"], co)
    if level in ("inter", "final"):
        fi = interaction_features(x, J, H, R, t)
        chi, dhi, coi, unc, con, gate = model.interact(hh, ho, ((fi[0] - MU["i"]) / MU["is"])[None], ((fi[1] - MU["b"]) / MU["bs"])[None], cf)
        chi = smooth_t(chi, SMOOTH); coi = smooth_t(coi, OBJ_SMOOTH) * BOUND.to(coi.device); dhi = smooth_t(dhi.flatten(2), SMOOTH).view(dhi.shape)
        gate = smooth_t(gate, OBJ_SMOOTH)
        c, co, dh = chb[0] + chi[0], gate[0] * (cob[0] + coi[0]), dhi[0]
        J = fuse(x["bo"], x["br"], J0, x["ramp"], c.view(-1, NREG, 2, 2, 4), base, REGION); H = fuse(x["bho"], x["bhr"], H0, x["ramp"], c.view(-1, NREG, 2, 2, 4), base, HREG)
        J, H = apply_hand(J, H, dh, x["ramp"])
        R, t = apply_obj(x, x["obj_R"], x["obj_t"], co)
        if level == "final":
            J, H, R, t, d2 = optimise(model, x, J, H, R, t, unc[0], con[0], train=False, return_offsets=True, gate=gate[0] if GATE else None)
            dh = dh + d2
    return {"c": c.view(-1, NREG, 2, 2, 4), "dh": dh, "R": R, "t": t, "J": J, "H": H, "gate": gate[0] if level in ("inter", "final") else None}


METRICS = ("CD-H", "ACC-H", "CD-O", "ACC-O", "PEN")
SCALE = np.array([0.1, 0.001, 0.3, 0.0007, 0.01])       # rough leaderboard rank spacing per metric


def summarize(rows, label):
    keep = [r for r in rows.values() if r["flags"][0] and r["flags"][1]]
    levels = [k for k in ("e2", "branch", "inter", "final") if k in keep[0]]
    base = np.array([r["ours"] for r in keep])
    print(f"[{label}] {len(keep)}/{len(rows)} episodes (catastrophic human/object excluded) | ours: " +
          " ".join(f"{m} {v:.4f}" for m, v in zip(METRICS, base.mean(0))), flush=True)
    score = {}
    for lv in levels:
        d = np.array([r[lv] for r in keep]) - base
        print(f"   {lv:6s} - ours: " + " | ".join(f"{m} {d[:, i].mean():+.5f} ({int((d[:, i] < 0).sum())}/{len(d)})" for i, m in enumerate(METRICS)), flush=True)
        score[lv] = float((d.mean(0) / SCALE).sum())
    if "gate" in keep[0] and GATE:
        print(f"   object gate: mean {np.mean([r['gate'][0] for r in keep]):.3f}", flush=True)
    if "contact" in keep[0]:
        c = np.array([r["contact"] for r in keep]).sum(0)
        print(f"   contact head: recall {c[0] / max(c[1], 1) * 100:.0f} % precision {c[0] / max(c[2], 1) * 100:.0f} %", flush=True)
    return score


def norm_stats(eps):
    f = {"h": [], "o": [], "i": [], "b": [], "vh": [], "vo": []}
    with torch.no_grad():
        for ep in eps:
            x = ep.window(slice(None))
            f["h"].append(human_features(x["bo"], x["br"], x["conf"], x["vis"])); f["o"].append(object_features(x))
            hi, bi = interaction_features(x, x["ours"], x["hands_ours"], x["obj_R"], x["obj_t"])
            f["i"].append(hi.reshape(-1, hi.shape[-1])); f["b"].append(bi)
            if ep.visual:
                vh, vo = visual_features(x); f["vh"].append(vh); f["vo"].append(vo)
    MU = {}
    for k, v in f.items():
        if not v:
            continue
        a = torch.cat(v); MU[k] = a.mean(0); MU[k + "s"] = a.std(0) + 1e-6
    return MU


def main():
    global W_ACC_O, OBJ_SMOOTH, GAP_OBJ_ONLY, NO_HAND_HEAD, OPT_OBJ_ONLY, GATE, LAM_GATE, SIM_FAIL, W_PEN
    ap = argparse.ArgumentParser()
    ap.add_argument("cache", type=Path); ap.add_argument("out", type=Path)
    ap.add_argument("--train", required=True); ap.add_argument("--val", required=True)
    ap.add_argument("--steps", default="600,1200,1200,800"); ap.add_argument("--window", type=int, default=120)
    ap.add_argument("--batch", type=int, default=16); ap.add_argument("--eval_only", type=Path)
    ap.add_argument("--no_aug", action="store_true")
    ap.add_argument("--w_acc_o", type=float, default=50.0); ap.add_argument("--obj_smooth", type=float, default=4.0)
    ap.add_argument("--gap_obj_only", action="store_true"); ap.add_argument("--no_hand_head", action="store_true")
    ap.add_argument("--opt_obj_only", action="store_true")
    ap.add_argument("--gate", action="store_true"); ap.add_argument("--lam_gate", type=float, default=0.5)
    ap.add_argument("--sim_fail", type=float, default=0.0); ap.add_argument("--w_pen", type=float, default=2.0)
    ap.add_argument("--visual", action="store_true", help="image evidence: 2D keypoint / mask features + a trainable CNN on the crops")
    args = ap.parse_args(); dev = "cuda"; args.out.mkdir(parents=True, exist_ok=True)
    W_ACC_O, OBJ_SMOOTH, GAP_OBJ_ONLY, NO_HAND_HEAD, OPT_OBJ_ONLY = args.w_acc_o, args.obj_smooth, args.gap_obj_only, args.no_hand_head, args.opt_obj_only
    GATE, LAM_GATE, SIM_FAIL, W_PEN = args.gate, args.lam_gate, args.sim_fail, args.w_pen
    print(f"W_ACC_O {W_ACC_O} OBJ_SMOOTH {OBJ_SMOOTH} GAP_OBJ_ONLY {GAP_OBJ_ONLY} NO_HAND_HEAD {NO_HAND_HEAD} OPT_OBJ_ONLY {OPT_OBJ_ONLY} "
          f"GATE {GATE} LAM_GATE {LAM_GATE} SIM_FAIL {SIM_FAIL} W_PEN {W_PEN}", flush=True)
    torch.manual_seed(0); rng = np.random.default_rng(0)

    def load(roots):
        out = []
        for r in roots.split(","):
            for hp in sorted((args.cache / "human").glob(f"{r}_episode_*.npz")):
                op = args.cache / "object" / f"{hp.stem}_obj.npz"; vp = args.cache / "visual" / f"{hp.stem}_vis.npz"
                if op.exists() and (vp.exists() or not args.visual):
                    out.append(Episode(hp, op, dev, vp if args.visual else None))
        return out
    tr, va = load(args.train), load(args.val)
    tr = [e for e in tr if e.h_err < 0.30]
    MU = norm_stats(tr)
    print(f"train episodes {len(tr)} (objects usable {sum(e.obj_ok for e in tr)}), val {len(va)}", flush=True)
    model = JointRefiner(len(MU["h"]), len(MU["o"]), visual=args.visual).to(dev)
    if args.eval_only:
        ck = torch.load(args.eval_only); model.load_state_dict(ck["model"], strict=False); summarize(evaluate(model, va, ck["mu"], 3), "val"); return
    summarize(evaluate(None, va, MU, 0), "val before")
    steps = [int(s) for s in args.steps.split(",")]
    stage_of = np.repeat([1, 2, 3, 4], steps)
    cnn = [p for n, p in model.named_parameters() if n.startswith("cnn.")]
    rest = [p for n, p in model.named_parameters() if not n.startswith("cnn.")]
    opt = torch.optim.Adam([{"params": rest, "lr": 3e-4}] + ([{"params": cnn, "lr": 3e-5}] if cnn else []))
    best = {"score": 0.0}
    for step in range(1, len(stage_of) + 1):
        stage = int(stage_of[step - 1])
        if stage == 4 and stage_of[step - 2] == 3:
            for g in opt.param_groups:
                g["lr"] = g["lr"] / 5
        wins = []
        for i in rng.integers(0, len(tr), args.batch):
            ep = tr[i]; s0 = int(rng.integers(0, max(1, ep.L - args.window)))
            w = ep.window(slice(s0, s0 + min(args.window, ep.L - s0)))
            wins.append(w if args.no_aug else augment(w, rng))
        L = min(len(w["ours"]) for w in wins)
        wins = [{k: (v[:, :L] if k in ("bo", "br", "bg", "bho", "bhr", "bhg", "bto", "btr") else
                     v if (not torch.is_tensor(v) or k in ("surf_ours", "surf_gt", "sdf", "lo", "hi", "sdf_lo", "sdf_hi")) else v[:L])
                 for k, v in w.items()} for w in wins]
        loss = window_loss(model, wins, MU, min(stage, 3))
        if not torch.is_tensor(loss) or not torch.isfinite(loss):
            continue
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 250 == 0 or step == len(stage_of):
            print(f"step {step} stage {stage} loss {float(loss):.4f}", flush=True)
            sc = summarize(evaluate(model, va, MU, min(stage, 3)), f"val step {step}")
            lv = "final" if stage >= 3 else "inter" if stage == 2 else "branch"
            if lv in sc and sc[lv] < best["score"]:
                best = {"score": sc[lv], "step": step, "level": lv}
                torch.save({"model": model.state_dict(), "mu": MU, "visual": args.visual, "cfg": vars(args) | {"cache": str(args.cache), "out": str(args.out), "eval_only": None}, "level": lv, "step": step}, args.out / "joint_best.pt")
                print(f"  best at step {step} ({lv}, score {sc[lv]:+.3f})", flush=True)
        if step < len(stage_of) and stage_of[step] != stage:
            torch.save({"model": model.state_dict(), "mu": MU, "visual": args.visual, "cfg": vars(args) | {"cache": str(args.cache), "out": str(args.out), "eval_only": None}, "level": stage, "step": step}, args.out / f"joint_stage{stage}.pt")
    torch.save({"model": model.state_dict(), "mu": MU, "visual": args.visual, "cfg": vars(args) | {"cache": str(args.cache), "out": str(args.out), "eval_only": None}, "level": "final", "step": len(stage_of)}, args.out / "joint_final.pt")
    rows = evaluate(model, va, MU, 3); summarize(rows, "val final")
    (args.out / "val_rows.json").write_text(json.dumps(rows, indent=1))
    print("best", json.dumps(best), flush=True)


if __name__ == "__main__":
    main()
