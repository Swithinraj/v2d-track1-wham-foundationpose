#!/usr/bin/env python3
"""Clear hand-object penetration on the ARM side: move the penetrating hand by the smallest smooth offset, realised
with that arm's own joints (mhr_converter.sif, GPU).

Why the arm: CD-H scores the whole body surface, so moving one hand (plus forearm) by 1-2 cm changes it by a few % of
that; moving the object instead shifts its whole surface (CD-O) and adds object acceleration (ACC-O). Runs after the
hand-side contact fix (fingers, wrist rotations), on what is left (palms, grips). Per episode and hand, scored frames:
  1. the smallest translation of the hand's dense vertices that clears them from the scorer-budgeted mesh (inflated by
     MARGIN): candidates = the outward depth gradient, away from the object centre, the six axes (ladder up to CAP,
     bisection); frames needing more than CAP are left to the object push (pen_zero.py) after this stage;
  2. the offsets are spread over a window (signed max-abs) and Gaussian-smoothed (no wrist jerk: ACC-H);
  3. each frame is re-posed by Levenberg-Marquardt on that arm's pose columns only (columns that move this wrist but
     not the pelvis, the other wrist or the legs): dense hand vertices -> shifted targets (weight 1), every other
     vertex -> its current place (weight W_BODY); repeated ROUNDS times on the frames still penetrating.
Inside/outside is the scorer's winding-number test (|w| > 0.5), on the GPU.
Writes <out>/episode_*.npz (+ meshes, shared files) and arm_clear_report[_<episodes>].json.
Usage: arm_clear.py --input DIR --out DIR --meshes DIR [--episodes ...] [--margin_m 0.002] [--cap_m 0.05]
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

T1 = Path("/track1")
sys.path.insert(0, str(T1 / "official/v2d_submission_kit/tools/track1"))
sys.path.insert(0, str(T1 / "official/v2d_submission_kit"))
sys.path.insert(0, str(T1 / "scripts"))
from mesh_to_mhr_params import MHR, lm_pose  # noqa: E402
from v2dlb import mhr_submission as ms  # noqa: E402
from mhr_bodies import MODEL, WRISTS, hand_vertex_sets  # noqa: E402

SCORED_J = [1, 2, 18, 35, 3, 19, 36, 4, 20, 37, 8, 24, 110, 74, 38, 113, 75, 39, 76, 40, 78, 42]
ROUNDS, W_BODY = 3, 0.05
ARGS = None


def inflate(V, F, d):
    v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    sign = 1.0 if np.einsum("ij,ij->", v0, np.cross(v1, v2)) >= 0 else -1.0
    vn = np.zeros_like(V)
    for k in range(3):
        np.add.at(vn, F[:, k], fn)
    vn /= np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12
    return V + sign * d * vn


def spread(offsets, window, sigma):
    half = window // 2
    padded = np.pad(offsets, ((half, half), (0, 0)), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, window, axis=0)
    pick = np.abs(windows).argmax(axis=-1)
    dilated = np.take_along_axis(windows, pick[..., None], axis=-1)[..., 0]
    radius = int(3 * sigma)
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(dilated, ((radius, radius), (0, 0)), mode="edge")
    return np.stack([np.convolve(padded[:, c], kernel, mode="valid") for c in range(3)], axis=1)


class Mesh:
    """Zero-area faces dropped (as the scorer); inside = |winding number| > 0.5 on the GPU; depth on the CPU."""

    def __init__(self, V, F):
        v = [np.ascontiguousarray(V[F[:, k]], dtype=np.float64) for k in range(3)]
        keep = np.linalg.norm(np.cross(v[1] - v[0], v[2] - v[0]), axis=1) > 0.0
        self.tri = [x[keep] for x in v]
        self.gpu = [torch.as_tensor(x, device="cuda") for x in self.tri]
        self.lo, self.hi = V.min(axis=0), V.max(axis=0)

    def winding(self, q):
        a, b, c = (x[None] - q[:, None] for x in self.gpu)
        la, lb, lc = (x.norm(dim=-1) for x in (a, b, c))
        num = (a * torch.cross(b, c, dim=-1)).sum(-1)
        den = la * lb * lc + (a * b).sum(-1) * lc + (b * c).sum(-1) * la + (c * a).sum(-1) * lb
        return torch.atan2(num, den).sum(1) / (2.0 * np.pi)

    def inside(self, q):
        out = np.zeros(len(q), dtype=bool)
        near = np.flatnonzero(((q >= self.lo) & (q <= self.hi)).all(axis=1))
        if near.size:
            Q = torch.as_tensor(q[near], dtype=torch.float64, device="cuda")
            per = max(1, 4_000_000 // len(self.tri[0]))
            w = torch.cat([self.winding(Q[i:i + per]) for i in range(0, len(Q), per)])
            out[near] = (w.abs() > 0.5).cpu().numpy()
        return out

    def depth(self, q):
        d = np.zeros(len(q))
        ins = np.flatnonzero(self.inside(q))
        if ins.size:
            d[ins] = ms._point_triangle_distance(np.ascontiguousarray(q[ins]), *self.tri)
        return d


class Obj:
    def __init__(self, R, t, s, V, F, margin):
        self.R, self.t, self.s = R, t, s
        self.plain, self.inflated = Mesh(V, F), Mesh(inflate(V, F, margin / s), F)
        self.centre = (V.min(0) + V.max(0)) / 2

    def local(self, pts, f):
        return (pts - self.t[f]) @ self.R[f] / self.s

    def depth(self, pts, f, inflated=True):
        return (self.inflated if inflated else self.plain).depth(self.local(pts, f)) * self.s

    def clear(self, pts, f):
        return not self.inflated.inside(self.local(pts, f)).any()

    def hand_move(self, pts, f, cap):
        """Smallest translation of pts that clears them at frame f, or None if none <= cap."""
        if self.clear(pts, f):
            return np.zeros(3)
        dep = self.depth(pts, f)
        inside = pts[dep > 0]
        grad = np.zeros(3)
        for k in range(3):
            e = np.zeros(3); e[k] = 0.002
            grad[k] = (self.depth(inside + e, f).sum() - self.depth(inside - e, f).sum()) / 0.004
        cands = [-grad, pts.mean(0) - (self.t[f] + self.s * self.R[f] @ self.centre)]
        cands += [np.eye(3)[k] * sg for k in range(3) for sg in (1.0, -1.0)]
        cands = [c / np.linalg.norm(c) for c in cands if np.linalg.norm(c) > 1e-9]
        steps = [x for x in (0.0025, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05) if x < cap] + [cap]
        prev = 0.0
        for step in steps:
            hit = next((c for c in cands if self.clear(pts + step * c, f)), None)
            if hit is not None:
                lo, hi = prev, step
                for _ in range(6):
                    mid = (lo + hi) / 2
                    lo, hi = (lo, mid) if self.clear(pts + mid * hit, f) else (mid, hi)
                return hi * hit
            prev = step
        return None


def arm_columns(mhr, pose, ident):
    """Per arm: pose columns that move its wrist joint but not the pelvis (joint 1), the other wrist or the legs."""
    p = pose[None].repeat(136, 1)
    p[torch.arange(136), torch.arange(136)] += 0.05
    _, j0 = mhr.run(pose[None], ident)
    _, j = mhr.run(p, ident)
    dj = (j - j0).norm(dim=-1).cpu().numpy()          # [136, 127]
    legs = [2, 18, 35, 3, 19, 36, 4, 20, 37, 8, 24]
    out = []
    for h, w in enumerate(WRISTS):
        other = WRISTS[1 - h]
        out.append([c for c in range(136) if dj[c, w] > 1e-4 and dj[c, other] < 1e-6 and dj[c, 1] < 1e-6
                    and dj[c, legs].max() < 1e-6])
    return out


def episode(ep, mhr, sets):
    tag = f"episode_{ep:06d}"
    d = dict(np.load(ARGS.input / f"{tag}.npz"))
    mz = np.load(ARGS.meshes / f"{tag}_budget.npz")
    R, t, s = d["object_rotation"].astype(np.float64), d["object_translation"].astype(np.float64), float(d["object_scale"])
    obj = Obj(R, t, s, mz["vertices"].astype(np.float64), mz["faces"], ARGS.margin_m)
    pose = torch.as_tensor(d["pose"], dtype=torch.float64, device="cuda")
    ident = torch.as_tensor(np.concatenate([d["scales"], d["shape"]])[None], dtype=torch.float64, device="cuda")
    T = len(pose)
    sf = [f for f in ARGS.scored[str(ep)] if f < T]
    idx = [torch.as_tensor(x, device="cuda") for x in sets]
    verts0, joints0 = mhr.run(pose, ident)
    verts = verts0.clone()
    cols = arm_columns(mhr, pose[sf[0]], ident)
    hands = lambda h, f: (verts[f, idx[h]] / 1000.0).cpu().numpy()  # noqa: E731
    before = {f: obj.depth(np.concatenate([hands(0, f), hands(1, f)]), f, inflated=False) for f in sf}
    required = np.zeros((2, T, 3))
    offsets = np.zeros((2, T, 3))
    beyond = set()
    rounds = 0
    for rounds in range(1, ROUNDS + 1):
        todo = {}
        for h in (0, 1):
            for f in sf:
                if (h, f) in beyond:
                    continue
                m = obj.hand_move(hands(h, f), f, ARGS.cap_m - np.linalg.norm(offsets[h, f]))
                if m is None:
                    beyond.add((h, f))
                elif np.linalg.norm(m) > 0:
                    required[h, f] = offsets[h, f] + m
                    todo[(h, f)] = True
        if not todo:
            break
        new = np.stack([spread(required[h], ARGS.window, ARGS.sigma) for h in (0, 1)])
        delta = new - offsets
        offsets = new
        for h in (0, 1):
            frames = [f for f in range(T) if np.linalg.norm(delta[h, f]) > 1e-4]
            if not frames:
                continue
            fr = torch.as_tensor(frames, device="cuda")
            tgt = verts[fr].clone()
            tgt[:, idx[h]] += torch.as_tensor(delta[h, frames] * 1000.0, device="cuda")[:, None]
            w = torch.full((verts.shape[1],), W_BODY, dtype=torch.float64, device="cuda")
            w[idx[h]] = 1.0
            p1, _ = lm_pose(mhr, tgt.reshape(len(frames), -1), pose[fr], ident, iters=8, cols=cols[h], w=w)
            pose[fr] = p1
            verts[fr] = mhr.run(p1, ident)[0]
    d["pose"] = pose.cpu().numpy().astype(np.float32)
    verts1, joints1 = mhr.run(torch.as_tensor(d["pose"], dtype=torch.float64, device="cuda"), ident)
    after = {f: obj.depth((verts1[f, torch.cat(idx)] / 1000.0).cpu().numpy(), f, inflated=False) for f in sf}
    np.savez(ARGS.out / f"{tag}.npz", **d)
    sfa = np.array(sf)
    inner = sfa[(sfa >= 1) & (sfa < T - 1)]
    J0, J1 = joints0[:, SCORED_J].cpu().numpy(), joints1[:, SCORED_J].cpu().numpy()
    acc = lambda J: np.linalg.norm(J[inner - 1] - 2 * J[inner] + J[inner + 1], axis=-1)  # noqa: E731
    mean_cm = lambda dd: float(np.mean([dd[f].mean() for f in sf]) * 100)  # noqa: E731
    return {"episode": tag, "dense_pen_cm": [mean_cm(before), mean_cm(after)],
            "penetrating_frames": [int(sum(before[f].max() > 0 for f in sf)), int(sum(after[f].max() > 0 for f in sf))],
            "rounds": rounds, "beyond_cap_hand_frames": len(beyond),
            "max_hand_offset_cm": float(np.linalg.norm(offsets, axis=-1).max() * 100),
            "mean_hand_offset_cm_scored": float(np.linalg.norm(offsets[:, sf], axis=-1).mean() * 100),
            "max_scored_joint_move_cm": float(np.linalg.norm(J1 - J0, axis=-1).max() * 100),
            "mean_scored_joint_move_cm": float(np.linalg.norm(J1[sf] - J0[sf], axis=-1).mean() * 100),
            "scored_joint_acc_change_cm": float(np.abs(acc(J1) - acc(J0)).mean() * 100),
            "arm_columns": [len(c) for c in cols]}


def main():
    global ARGS
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--meshes", type=Path, required=True, help="budget_object_meshes.py output of --input")
    ap.add_argument("--episodes", type=int, nargs="*", default=list(range(30)))
    ap.add_argument("--margin_m", type=float, default=0.002)
    ap.add_argument("--cap_m", type=float, default=0.05)
    ap.add_argument("--window", type=int, default=13)
    ap.add_argument("--sigma", type=float, default=6.0)
    ARGS = ap.parse_args()
    ARGS.out.mkdir(parents=True, exist_ok=True)
    tags = {f"episode_{ep:06d}" for ep in ARGS.episodes}
    for src in ARGS.input.iterdir():
        if src.is_file() and (src.name[:14] in tags if src.name.startswith("episode_") else not (ARGS.out / src.name).exists()):
            shutil.copy2(src, ARGS.out / src.name)
    ARGS.scored = json.loads((ARGS.meshes / "scored_frames.json").read_text())
    mhr = MHR(str(MODEL), "cuda", chunk=64, precision="float64")
    sets = hand_vertex_sets()
    report = []
    for ep in ARGS.episodes:
        r = episode(ep, mhr, sets)
        report.append(r)
        print(f"{r['episode']}: dense PEN {r['dense_pen_cm'][0]:.5f} -> {r['dense_pen_cm'][1]:.5f} cm, frames "
              f"{r['penetrating_frames'][0]} -> {r['penetrating_frames'][1]} (beyond cap {r['beyond_cap_hand_frames']}), "
              f"hand offset max {r['max_hand_offset_cm']:.2f} / mean {r['mean_hand_offset_cm_scored']:.2f} cm, scored joints "
              f"max {r['max_scored_joint_move_cm']:.2f} / mean {r['mean_scored_joint_move_cm']:.3f} cm, acc change "
              f"{r['scored_joint_acc_change_cm']:.4f}, arm columns {r['arm_columns']}", flush=True)
    name = "arm_clear_report" + ("_" + "_".join(map(str, ARGS.episodes)) if len(ARGS.episodes) < 30 else "") + ".json"
    (ARGS.out / name).write_text(json.dumps({"margin_m": ARGS.margin_m, "cap_m": ARGS.cap_m, "episodes": report}, indent=1))
    print("PASS: arm clearance", len(report), "episodes")


if __name__ == "__main__":
    main()
