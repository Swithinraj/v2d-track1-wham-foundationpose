#!/usr/bin/env python3
"""Contact-constrained human solver with the object held fixed (mhr_converter.sif, GPU, autograd through MHR).

Interaction model per frame and hand (from the current human vs the fixed object track):
  no contact   min signed distance of the hand > CONTACT_M        -> non-penetration only
  stable grasp in contact, the contacting region keeps its place in the object frame (< RIGID_MPF per frame) for
               >= MIN_RUN frames                                   -> non-penetration + surface contact + rigidity
  sliding      in contact but moving along the surface             -> non-penetration + surface contact
  uncertain    no reliable MediaPipe hand                          -> 2-D term off for that hand, contact halved
Non-penetration applies to every frame. Contact/rigidity apply only to the hand points that actually touch (initial
signed distance < 1 cm on most of the window's frames), so the hand is not flattened onto the surface.
Variables: per-frame corrections of the arm columns (move that wrist, not the pelvis, the other wrist or the legs) and
the hand columns (move the hand, not any scored joint), on frames within ACTIVE_M of the object (+- PAD frames).
Energy (Adam): non-penetration hinge on dense hand points (SDF grid of the object mesh, trilinear), contact d^2,
rigidity (variance of the contact points in the object frame per window), 2-D agreement with MediaPipe hand landmarks
and wrists/elbows, a prior to the starting pose (stronger on the arm columns, which move scored joints), and temporal
smoothness of the corrections and of the scored joints' change.
Object geometry: signed-distance grids from contact_sdf.py (--sdf, default outputs/contact_sdf).
Writes <out>/episode_*.npz (+ copies) and human_contact_report_<episodes>.json.
Usage (container): human_contact_solver.py --human DIR --objects DIR --out DIR [--sdf DIR] [--episodes ...] [--iters 200]
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

T1 = Path(os.environ.get("TRACK1_ROOT") or Path(__file__).resolve().parents[1])
sys.path.insert(0, str(T1 / "official/v2d_submission_kit/tools/track1"))
sys.path.insert(0, str(T1 / "scripts"))
from mesh_to_mhr_params import MHR, NS  # noqa: E402
from mhr_bodies import hand_vertex_sets  # noqa: E402

SCORED_J = [1, 2, 18, 35, 3, 19, 36, 4, 20, 37, 8, 24, 110, 74, 38, 113, 75, 39, 76, 40, 78, 42]
LEFT = [78, 97, 98, 99, 100, 92, 93, 94, 95, 88, 89, 90, 91, 84, 85, 86, 87, 80, 81, 82, 83]
RIGHT = [42, 61, 62, 63, 64, 56, 57, 58, 59, 52, 53, 54, 55, 48, 49, 50, 51, 44, 45, 46, 47]
PAIRS = [(78, 15), (42, 16), (76, 13), (40, 14), (75, 11), (39, 12), (2, 23), (18, 24), (3, 25), (19, 26)]
CONTACT_M, RIGID_MPF, MIN_RUN, ACTIVE_M, PAD, MARGIN = 0.015, 0.006, 10, 0.05, 15, 0.002
NSUB = 900                                          # hand points per hand used in the energy


class Body:
    def __init__(self, path):
        self.m = MHR(path, "cuda", chunk=96, precision="float32")

    def __call__(self, pose, z, chunk=96):
        """pose [N,136] float32 (grad ok) -> vertices [N,V,3] m, joints [N,127,3] m (Track 1 frame)."""
        vs, js = [], []
        for a in range(0, len(pose), chunk):
            p = pose[a:a + chunk]; zz = z.expand(len(p), -1)
            params = torch.cat([p, zz[:, :NS]], 1); expr = torch.zeros(len(p), 72, device="cuda")
            v, sk = self.m.model(zz[:, NS:].contiguous(), params, expr, True)
            vs.append(v * self.m.flip.float() / 100.0); js.append(sk[..., :3] * self.m.flip.float() / 100.0)
        return torch.cat(vs), torch.cat(js)


def columns(body, pose0, z, hidx):
    """arm columns per arm and hand columns per hand (finite differences at one frame)."""
    p = pose0[None].repeat(137, 1); p[1 + torch.arange(136), torch.arange(136)] += 0.05   # row 0: unperturbed, same batch
    with torch.no_grad():
        v, j = body(p, z, chunk=137)
    v0, j0, v, j = v[:1], j[:1], v[1:], j[1:]                    # one batch: no kernel-dependent float32 noise
    dj = (j - j0).norm(dim=-1)                                   # [136, 127]
    dsc = dj[:, SCORED_J].amax(1); dh = [(v[:, h] - v0[:, h]).norm(dim=-1).amax(1) for h in hidx]
    legs = dj[:, [2, 18, 3, 19, 4, 20, 8, 24]].amax(1)
    arm = []
    for w, o in ((42, 78), (78, 42)):                            # right then left: wrist moves, other wrist/pelvis/legs fixed
        arm.append([c for c in range(136) if dj[c, w] > 1e-4 and dj[c, o] < 1e-6 and dj[c, 1] < 1e-6 and legs[c] < 1e-6])
    hand = [[c for c in range(136) if dsc[c] < 1e-6 and dh[k][c] > 1e-4 and dh[1 - k][c] < 1e-6] for k in range(2)]
    return arm, hand


def sdf_lookup(grid, lo, hi, P):
    """P [N,3] local mesh units -> signed distance (local units), trilinear; outside the grid: large positive."""
    g = (P - lo) / (hi - lo) * 2 - 1                              # [-1, 1], order x,y,z
    out = Fn.grid_sample(grid[None, None], g[None, None, None, :, [2, 1, 0]], mode="bilinear", padding_mode="border",
                         align_corners=True)[0, 0, 0, 0]
    inside = (g.abs() <= 1).all(1)
    return torch.where(inside, out, torch.full_like(out, 1.0))


def episode(ep, args, body, hsets):
    tag = f"episode_{ep:06d}"
    dh = dict(np.load(args.human / f"{tag}.npz")); do = np.load(args.objects / f"{tag}.npz")
    pose0 = torch.tensor(dh["pose"], dtype=torch.float32, device="cuda"); T = len(pose0)
    z = torch.tensor(np.concatenate([dh["scales"], dh["shape"]])[None], dtype=torch.float32, device="cuda")
    R = torch.tensor(do["object_rotation"], dtype=torch.float32, device="cuda"); t = torch.tensor(do["object_translation"], dtype=torch.float32, device="cuda")
    s = float(do["object_scale"])
    g = np.load(args.sdf / f"{tag}.npz"); grid = torch.tensor(g["sdf"], device="cuda"); lo = torch.tensor(g["lo"], device="cuda"); hi = torch.tensor(g["hi"], device="cuda")
    rng = np.random.default_rng(0)
    nsub = args.nsub if args.nsub > 0 else 10 ** 9                 # --nsub 0: every hand vertex
    hsub = [torch.as_tensor(rng.choice(h, min(nsub, len(h)), replace=False), device="cuda") for h in hsets]   # 0 right, 1 left
    hidx = [torch.as_tensor(h, device="cuda") for h in hsets]
    arm, hand = columns(body, pose0[0], z, hidx)
    if args.no_arm:
        arm = [[], []]
    cols = sorted(set(sum(arm, []) + sum(hand, [])))
    ci = torch.as_tensor(cols, device="cuda")
    is_arm = torch.tensor([c in arm[0] or c in arm[1] for c in cols], device="cuda")

    def signed(V, idx):                                           # V [N,Nv,3] human frame -> signed distance (human m)
        P = V[:, idx]; loc = torch.einsum("nij,nvi->nvj", R[fr], P - t[fr][:, None]) / s
        return sdf_lookup(grid, lo, hi, loc.reshape(-1, 3)).reshape(P.shape[:2]) * s

    # initial state and active frames
    with torch.no_grad():
        V0, J0 = body(pose0, z)
        fr = torch.arange(T, device="cuda")
        d0 = [signed(V0, h) for h in hsub]                        # [T, NSUB]
    dmin = torch.stack([d.min(1).values for d in d0], 1).cpu().numpy()          # [T, 2]
    near = (dmin < 0).any(1) if args.pen_frames_only else (dmin < ACTIVE_M).any(1)
    act = np.zeros(T, bool)
    for f in np.flatnonzero(near):
        act[max(0, f - PAD):f + PAD + 1] = True
    if act.sum() < 5:
        np.savez(args.out / f"{tag}.npz", **dh)
        return tag, {"active_frames": 0}
    fr = torch.as_tensor(np.flatnonzero(act), device="cuda"); F = len(fr)
    frn = fr.cpu().numpy()
    # corrections fade to zero at both ends of every active segment (continuity with the untouched frames) and the
    # temporal terms only couple consecutive frames of one segment
    taper, seg = np.zeros(F), np.r_[0, np.cumsum(np.diff(frn) > 1)]
    for s_id in np.unique(seg):
        ix = np.flatnonzero(seg == s_id); n = len(ix); r = np.minimum(np.arange(n), np.arange(n)[::-1]) / max(PAD, 1)
        r = np.clip(r, 0, 1); taper[ix] = r * r * (3 - 2 * r)
    if args.freeze_anchor:                                         # keep the human of the first scored frame (the
        sc = json.loads(Path(args.scored_frames).read_text())       # frame the scorer's Sim(3) is fitted on) exactly
        a0 = min(sc[str(ep)])                                         # unchanged; corrections ramp in over 15 frames
        r = np.clip((np.abs(frn - a0) - 2) / 15.0, 0, 1); taper = taper * r * r * (3 - 2 * r)
    taper = torch.tensor(taper, dtype=torch.float32, device="cuda")[:, None]

    def basis(sigma):                                             # smooth corrections: Gaussian kernel within segments
        G = np.zeros((F, F))
        for s_id in np.unique(seg):
            ix = np.flatnonzero(seg == s_id); tt = frn[ix].astype(float)
            Kk = np.exp(-(tt[:, None] - tt[None]) ** 2 / (2 * max(sigma, 1e-3) ** 2)); G[np.ix_(ix, ix)] = Kk / Kk.sum(1, keepdims=True)
        return torch.tensor(G, dtype=torch.float32, device="cuda")
    trip = torch.tensor((frn[2:] - frn[:-2] == 2).astype(np.float32), device="cuda")[:, None] if F > 2 else None
    # interaction states (per hand): contact runs, rigidity of the contact region in the object frame
    with torch.no_grad():
        loc0 = [torch.einsum("nij,nvi->nvj", R[fr], V0[fr][:, h] - t[fr][:, None]) for h in hsub]   # object frame (human m)
    d0f = [d[fr] for d in d0]
    states, windows = np.zeros((F, 2), int), []
    for k in range(2):
        con = (d0f[k].min(1).values < CONTACT_M).cpu().numpy()
        f = 0
        while f < F:
            if not con[f]:
                f += 1; continue
            g2 = f
            while g2 < F and con[g2]:
                g2 += 1
            if g2 - f >= 5:
                touch = (d0f[k][f:g2] < 0.01).float().mean(0) > 0.5          # contacting region of this run
                if touch.sum() >= 5:
                    L = loc0[k][f:g2][:, touch]
                    step = (L[1:] - L[:-1]).norm(dim=-1).mean().item()
                    stable = step < RIGID_MPF and g2 - f >= MIN_RUN
                    states[f:g2, k] = 2 if stable else 1
                    windows.append((k, f, g2, touch, stable))
            f = g2
    # MediaPipe 2-D targets (actor by nearest body landmarks), projection
    adir = os.environ.get("CONTACT_ALIGNMENT_DIR") or "outputs/alignment_sam"          # truecam_final.sh: the measured camera
    a = json.loads((T1 / adir / tag / "alignment.json").read_text()); K, ds = a["intrinsics"], float(a["depth_scale"])
    mpz = np.load(T1 / args.kp2d_dir / f"{tag}.npz"); mp, mh, ms = mpz["pose"], mpz["hands"], mpz["hand_score"]

    def proj(J):
        c = ds * J
        return torch.stack([K["fx"] * c[..., 0] / c[..., 2] + K["cx"], K["fy"] * c[..., 1] / c[..., 2] + K["cy"]], -1)

    uv0 = proj(J0).detach().cpu().numpy()
    hand_t, hand_w, body_t, body_w = np.zeros((F, 2, 21, 2)), np.zeros((F, 2)), np.zeros((F, len(PAIRS), 2)), np.zeros((F, len(PAIRS)))
    for i, f in enumerate(frn):
        if f >= len(mp):
            continue
        dd = [np.nanmedian([np.linalg.norm(mp[f, p, l, :2] - uv0[f, j]) for j, l in PAIRS]) for p in range(mp.shape[1])]
        if not np.isfinite(dd).any() or np.nanmin(dd) > 60:
            continue
        p = int(np.nanargmin(dd))
        for q, (j, l) in enumerate(PAIRS):
            if np.isfinite(mp[f, p, l, :2]).all() and mp[f, p, l, 3] > 0.5:
                body_t[i, q], body_w[i, q] = mp[f, p, l, :2], 1.0
        for side, k in ((0, 1), (1, 0)):                          # MediaPipe side 0 = left (LEFT map, hand k=1)
            if ms[f, p, side] > 0.6 and np.isfinite(mh[f, p, side, :, :2]).all():
                hand_t[i, k], hand_w[i, k] = mh[f, p, side, :, :2], 1.0
    hand_t, hand_w = torch.tensor(hand_t, dtype=torch.float32, device="cuda"), torch.tensor(hand_w, dtype=torch.float32, device="cuda")
    body_t, body_w = torch.tensor(body_t, dtype=torch.float32, device="cuda"), torch.tensor(body_w, dtype=torch.float32, device="cuda")
    jmaps = [torch.as_tensor(RIGHT, device="cuda"), torch.as_tensor(LEFT, device="cuda")]
    # reliable image evidence: landmarks whose starting projection already agrees with MediaPipe (body <= 10 px,
    # hand <= 12 px); with --w_img their projection is held in place, so corrections prefer the camera-ray direction
    with torch.no_grad():
        uvb0 = proj(J0[fr][:, [j for j, _ in PAIRS]]); rel_b = body_w * ((uvb0 - body_t).norm(dim=-1) < 10.0).float()
        uvh0 = [proj(J0[fr][:, jmaps[k]]) for k in range(2)]
        rel_h = [hand_w[:, k, None] * ((uvh0[k] - hand_t[:, k]).norm(dim=-1) < 12.0).float() for k in range(2)]
    if args.free_fingers_in_grasp:                                 # fingers wrapped around the object are occluded:
        for k in range(2):                                          # in stable grasps contact + rigidity decide them
            rel_h[k] = rel_h[k] * torch.tensor(states[:, k] != 2, dtype=torch.float32, device="cuda")[:, None]
    cap_j = torch.as_tensor([42, 78, 40, 76], device="cuda")
    pj = torch.as_tensor([j for j, _ in PAIRS], device="cuda")
    sc = torch.as_tensor(SCORED_J, device="cuda")
    st = torch.tensor(states, device="cuda")
    # optimisation
    G_arm, G_hand = basis(args.sigma_arm), basis(args.sigma_hand)
    delta_c = torch.zeros(F, len(cols), device="cuda", requires_grad=True)

    def make_delta():
        return torch.where(is_arm[None], G_arm @ delta_c, G_hand @ delta_c)
    delta = make_delta()
    params = [delta_c]
    # joint whole-body depth shift (stage 5, step 2): smooth along the camera ray through the pelvis, only inside the
    # interaction segments (segment taper; first scored frame frozen), evaluated in the same solve as the hands
    ray = (J0[fr][:, 1] / J0[fr][:, 1].norm(dim=-1, keepdim=True)).detach()
    flipv = torch.tensor([1.0, -1.0, -1.0], device="cuda")
    d_c = torch.zeros(F, device="cuda", requires_grad=bool(args.depth_shift))
    G_d = basis(args.depth_sigma) if args.depth_shift else None
    if args.depth_shift:
        params.append(d_c)

    def depth():
        return (taper[:, 0] * (G_d @ d_c)) if args.depth_shift else torch.zeros(F, device="cuda")
    # grasp-weighted contact patches from the offline interaction model (posterior of 'grasp' per frame and hand)
    patch_m = None
    if args.states_dir:
        stz = np.load(Path(args.states_dir) / f"{tag}.npz"); pg = torch.tensor(stz["posterior"][frn, :, 2], dtype=torch.float32, device="cuda")
        patch_m = []
        for k in range(2):
            m = (d0f[k].abs() < 0.03).float() * (pg[:, k:k + 1] >= 0.5).float()
            patch_m.append(m * pg[:, k:k + 1] / m.sum(1, keepdim=True).clamp(min=1.0))
    opt = torch.optim.Adam(params, lr=args.lr)
    prior_w = torch.where(is_arm, torch.tensor(args.w_prior_arm, device="cuda"), torch.tensor(args.w_prior_hand, device="cuda"))
    base = pose0[fr]
    hist = []
    for it in range(args.iters):
        opt.zero_grad()
        delta = make_delta(); dd = depth()
        pose = base.clone(); pose[:, ci] = base[:, ci] + taper * delta
        pose[:, 0:3] = pose[:, 0:3] + 10.0 * flipv * (dd[:, None] * ray)
        V, J = body(pose, z)
        dist = [signed(V, h) for h in hsub]
        e_pen = sum((torch.relu(MARGIN - d) ** 2).sum(1).mean() for d in dist)
        e_patch = sum((patch_m[k] * dist[k] ** 2).sum() for k in range(2)) / F if patch_m is not None else torch.zeros((), device="cuda")
        e_dep = ((dd / args.sig_depth) ** 2).mean()
        e_con, e_rig = torch.zeros((), device="cuda"), torch.zeros((), device="cuda")
        for k, f0, f1, touch, stable in windows:
            dk = dist[k][f0:f1][:, touch]; gate = (d0f[k][f0:f1][:, touch].abs() < CONTACT_M).float()
            e_con = e_con + (gate * dk ** 2).sum() / gate.sum().clamp(min=1.0)
            if stable:
                L = torch.einsum("nij,nvi->nvj", R[fr[f0:f1]], V[f0:f1][:, hsub[k][touch]] - t[fr[f0:f1]][:, None])
                e_rig = e_rig + ((L - L.mean(0, keepdim=True)) ** 2).sum(-1).mean()
        e_2d = torch.zeros((), device="cuda")
        for k in range(2):
            r = (proj(J[:, jmaps[k]]) - hand_t[:, k]).norm(dim=-1) / 6.0
            e_2d = e_2d + args.w_2d_hand * (hand_w[:, k, None] * torch.nn.functional.huber_loss(r, torch.zeros_like(r), reduction="none", delta=2.0)).mean()
        e_hd = torch.zeros((), device="cuda")
        if args.w_hdisp > 0:                                        # metric prior: dense hand vertices stay where SAM put them
            for k in range(2):
                e_hd = e_hd + ((V[:, hidx[k]] - V0[fr][:, hidx[k]] - dd[:, None, None] * ray[:, None]) ** 2).sum(-1).mean() / 0.002 ** 2
        rb = (proj(J[:, pj]) - body_t).norm(dim=-1) / 6.0
        e_2d = e_2d + (body_w * torch.nn.functional.huber_loss(rb, torch.zeros_like(rb), reduction="none", delta=2.0)).mean()
        e_prior = (prior_w * delta ** 2).sum(1).mean()
        dl = taper * delta
        e_tmp = (trip * (dl[2:] - 2 * dl[1:-1] + dl[:-2]) ** 2).sum(1).mean() if F > 2 else torch.zeros((), device="cuda")
        dJ = J[:, sc] - J0[fr][:, sc]
        e_jacc = (trip[:, :, None] * (dJ[2:] - 2 * dJ[1:-1] + dJ[:-2]) ** 2).sum(-1).mean() * 1e4 if F > 2 else torch.zeros((), device="cuda")
        e_img, e_cap = torch.zeros((), device="cuda"), torch.zeros((), device="cuda")
        if args.w_img > 0:
            pj_now = proj(J[:, [j for j, _ in PAIRS]])
            e_img = (rel_b * ((pj_now - uvb0).norm(dim=-1) / args.sigma_img) ** 2).sum() / rel_b.sum().clamp(min=1.0)
            for k in range(2):
                e_img = e_img + (rel_h[k] * ((proj(J[:, jmaps[k]]) - uvh0[k]).norm(dim=-1) / args.sigma_img) ** 2).sum() / rel_h[k].sum().clamp(min=1.0)
        if args.w_cap > 0:
            dz = (J[:, cap_j, 2] - J0[fr][:, cap_j, 2]).abs()
            e_cap = (torch.relu(dz - args.depth_cap) ** 2).sum(1).mean() / args.depth_cap ** 2
        loss = (args.w_pen * e_pen + args.w_con * e_con + args.w_rig * e_rig + args.w_2d * e_2d + e_prior
                + args.w_tmp * e_tmp + args.w_jacc * e_jacc + args.w_img * e_img + args.w_cap * e_cap + args.w_hdisp * e_hd
                + args.w_patch * e_patch + e_dep)
        loss.backward(); opt.step()
        if it % 25 == 0 or it == args.iters - 1:
            hist.append({"it": it, "loss": float(loss), "pen": float(e_pen), "con": float(e_con), "rig": float(e_rig), "2d": float(e_2d)})
    with torch.no_grad():
        delta = make_delta(); dd = depth()
        pose = base.clone(); pose[:, ci] = base[:, ci] + taper * delta
        pose[:, 0:3] = pose[:, 0:3] + 10.0 * flipv * (dd[:, None] * ray)
        V, J = body(pose, z)
        depth_stats = [float(dd.abs().median() * 100), float(dd.abs().max() * 100)]
        dist = [signed(V, h) for h in hsub]
        pen_after = float(sum(torch.relu(-d).mean(1).mean() for d in dist) * 100)
        pen_before = float(sum(torch.relu(-d).mean(1).mean() for d in d0f) * 100)
        dsc = (J[:, sc] - J0[fr][:, sc]).norm(dim=-1)
        e2 = lambda JJ: float(np.nanmean([((proj(JJ[:, jmaps[k]]) - hand_t[:, k]).norm(dim=-1).mean(1))[hand_w[:, k] > 0].mean().item() for k in range(2) if (hand_w[:, k] > 0).any()]))  # noqa: E731
        hand2d = [e2(J0[fr]), e2(J)]
        gap, drift = [[], []], [[], []]
        for k, f0, f1, touch, stable in windows:
            gap[0].append(float(d0f[k][f0:f1][:, touch].abs().mean())); gap[1].append(float(dist[k][f0:f1][:, touch].abs().mean()))
            if stable:
                for i, VV in enumerate((V0[fr], V)):
                    L = torch.einsum("nij,nvi->nvj", R[fr[f0:f1]], VV[f0:f1][:, hsub[k][touch]] - t[fr[f0:f1]][:, None])
                    drift[i].append(float((L[1:] - L[:-1]).norm(dim=-1).mean()))
        uvr = proj(J[:, [j for j, _ in PAIRS]]); img_move = float(((uvr - uvb0).norm(dim=-1) * rel_b).sum() / rel_b.sum().clamp(min=1.0))
        dzr = (J[:, cap_j, 2] - J0[fr][:, cap_j, 2]).abs()
    full = dh["pose"].copy(); full[frn] = pose.detach().cpu().numpy()
    dh["pose"] = full.astype(np.float32)
    np.savez(args.out / f"{tag}.npz", **dh)
    rep = {"active_frames": int(F), "columns_arm": [len(c) for c in arm], "columns_hand": [len(c) for c in hand],
           "frames_stable": int((states == 2).any(1).sum()), "frames_sliding": int((states == 1).any(1).sum()),
           "windows": len(windows), "sdf_pen_cm": [pen_before, pen_after], "hand_2d_px": hand2d,
           "scored_joint_move_cm": [float(dsc.mean() * 100), float(dsc.max() * 100)],
           "scored_joint_move_frame_p50_p95_max_cm": [float(np.percentile(dsc.amax(1).cpu().numpy(), q) * 100) for q in (50, 95, 100)],
           "max_move_at": [int(frn[int(dsc.amax(1).argmax())]), SCORED_J[int(dsc.amax(0).argmax())]],
           "contact_gap_mm": [float(np.mean(g) * 1000) if g else None for g in gap],
           "stable_contact_drift_mm_per_frame": [float(np.mean(g) * 1000) if g else None for g in drift],
           "reliable_landmark_image_move_px": img_move, "depth_shift_cm_median_max": depth_stats, "wrist_elbow_depth_move_cm_mean_max": [float(dzr.mean() * 100), float(dzr.max() * 100)],
           "history": hist}
    return tag, rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--human", type=Path, required=True); ap.add_argument("--objects", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True); ap.add_argument("--sdf", type=Path, default=T1 / "outputs/contact_sdf")
    ap.add_argument("--episodes", type=int, nargs="*", default=list(range(30)))
    ap.add_argument("--kp2d_dir", default="outputs/mediapipe",
                    help="2D body/hand landmarks in MediaPipe's layout under TRACK1_ROOT (e.g. outputs/kp2d_vitpose: ViTPose body, no hands)")
    ap.add_argument("--iters", type=int, default=200); ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--w_pen", type=float, default=2e5); ap.add_argument("--w_con", type=float, default=2e3)
    ap.add_argument("--w_rig", type=float, default=2e3); ap.add_argument("--w_2d", type=float, default=1.0)
    ap.add_argument("--w_prior_arm", type=float, default=50.0); ap.add_argument("--w_prior_hand", type=float, default=2.0)
    ap.add_argument("--w_tmp", type=float, default=200.0); ap.add_argument("--w_jacc", type=float, default=50.0)
    ap.add_argument("--sigma_arm", type=float, default=6.0); ap.add_argument("--sigma_hand", type=float, default=2.0)
    ap.add_argument("--no_arm", action="store_true")
    ap.add_argument("--nsub", type=int, default=NSUB)
    ap.add_argument("--w_hdisp", type=float, default=0.0, help="weight of the dense hand-vertex displacement prior (2 mm scale)")
    ap.add_argument("--w_2d_hand", type=float, default=1.0, help="factor on the MediaPipe hand-landmark 2D term")
    ap.add_argument("--pen_frames_only", action="store_true", help="solve only frames where a hand penetrates (+ taper)")
    ap.add_argument("--freeze_anchor", action="store_true", help="no correction within 2 frames of the first scored frame (ramp 15)")
    ap.add_argument("--depth_shift", action="store_true", help="joint smooth whole-body depth shift along the pelvis ray")
    ap.add_argument("--depth_sigma", type=float, default=20.0); ap.add_argument("--sig_depth", type=float, default=0.05)
    ap.add_argument("--states_dir", default="", help="interaction posteriors (interaction_states.py) for grasp-weighted contact patches")
    ap.add_argument("--w_patch", type=float, default=0.0)
    ap.add_argument("--scored_frames", default=str(T1 / "outputs/hand_fix/episodes_repro_depth_refine_meshes/scored_frames.json"),
                    help="JSON {episode: [scored frame indices]} (budget_object_meshes.py output)")
    ap.add_argument("--free_fingers_in_grasp", action="store_true",
                    help="no image preservation of finger landmarks in stable-grasp windows (wrist and elbow stay held)")
    ap.add_argument("--w_img", type=float, default=0.0); ap.add_argument("--sigma_img", type=float, default=2.0)
    ap.add_argument("--w_cap", type=float, default=0.0); ap.add_argument("--depth_cap", type=float, default=0.05)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for f in args.objects.glob("episode_??????_object*"):         # object meshes from --objects (no stale reports)
        if f.is_file():
            shutil.copy2(f, args.out / f.name)
    body = Body(str(T1 / "weights/mhr/assets/mhr_model.pt")); hsets = hand_vertex_sets()
    report = {}
    for ep in args.episodes:
        tag = f"episode_{ep:06d}"
        dh = dict(np.load(args.human / f"{tag}.npz")); do = np.load(args.objects / f"{tag}.npz")
        for k in ("object_rotation", "object_translation", "object_scale"):
            dh[k] = do[k]
        np.savez(args.out / f"{tag}.npz", **dh)                   # objects fixed (from --objects)
        tag, r = episode(ep, argparse.Namespace(**{**vars(args), "human": args.out}), body, hsets)
        report[tag] = r
        print(tag, json.dumps({k: v for k, v in r.items() if k != "history"}), flush=True)
    (args.out / f"human_contact_report_{'_'.join(map(str, args.episodes))}.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
