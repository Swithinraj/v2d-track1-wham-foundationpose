#!/usr/bin/env python3
"""Physics contact solver for objects: each object's path re-solved jointly from image evidence, hand contact and
smoothness (mhr_converter.sif, GPU, torch).

Pushing the object out of the hand after the fact (pen_zero.py) moves it off what the camera sees and jolts it.
Here every frame's object pose gets a small correction (translation dt, rotation dr about the object centre) that
minimises, over the whole episode at once:
  IMAGE    the projections of object surface points stay where the tracked pose put them (pixels, sigma SIG_PX):
           the tracker fitted the image, and a move along the camera ray (depth, the direction one camera cannot see)
           costs only through perspective, so depth is the cheap direction to correct;
  NO OVERLAP  hand vertices inside the object (winding number, the scorer's test) are pushed out: squared depth;
  TOUCH    in contact frames (a hand vertex within CONTACT_M at the start) the nearest hand vertex should reach the
           surface (gap -> GAP_M), so a corrected object does not float away from the hand;
  SMOOTH   second differences of the corrections (the object keeps the tracked motion, corrections are slow), ACC-O;
  PRIOR    small corrections where nothing pushes.
Distances use dense surface samples of the scorer-budgeted mesh; inside/outside is refreshed every REFRESH steps.
The human is untouched. Run after the hand fix; arm_clear.py / pen_zero.py clear any exact remainder.
Writes <out>/episode_*.npz (+ meshes, shared files) and contact_solver_report[_<episodes>].json.
Usage: contact_solver.py --input DIR --out DIR --hands DIR --meshes DIR [--episodes ...] [--iters 600]
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

T1 = Path("/track1")
sys.path.insert(0, str(T1 / "official/v2d_submission_kit"))
from v2dlb import mhr_submission as ms  # noqa: E402

SIG_PX, CONTACT_M, GAP_M = 3.0, 0.02, 0.002
W_PEN, W_TOUCH, W_SMOOTH_T, W_SMOOTH_R, W_PRIOR_T, W_PRIOR_R = 1.0 / 0.005 ** 2, 0.0, 1.0 / 0.0005 ** 2, \
    1.0 / 0.002 ** 2, 1.0 / 0.05 ** 2, 1.0 / 0.05 ** 2
SPACING = 8          # corrections are cubic B-splines with a control point every SPACING frames: smooth by construction
REFRESH, K_ACTIVE, N_SURF, N_IMG = 25, 192, 6000, 128
ARGS = None


class Mesh:
    """Zero-area faces dropped (as the scorer); inside = |winding number| > 0.5 on the GPU."""

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


def surface_samples(V, F, n, seed):
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    area = np.linalg.norm(np.cross(b - a, c - a), axis=1)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(F), n, p=area / area.sum())
    u, v = rng.random((2, n))
    m = u + v > 1
    u[m], v[m] = 1 - u[m], 1 - v[m]
    return a[idx] + u[:, None] * (b[idx] - a[idx]) + v[:, None] * (c[idx] - a[idx])


def bspline_basis(T, spacing):
    """[T, M] uniform cubic B-spline basis (control points every `spacing` frames, clamped at the ends)."""
    M = int(np.ceil((T - 1) / spacing)) + 3
    x = np.arange(T) / spacing + 1.0
    B = np.zeros((T, M))
    for m in range(M):
        u = np.abs(x - m)
        B[:, m] = np.where(u < 1, 2 / 3 - u ** 2 + u ** 3 / 2, np.where(u < 2, (2 - u) ** 3 / 6, 0.0))
    return B / B.sum(1, keepdims=True)


def rotation(r):
    """exp of the skew matrix of r (differentiable everywhere, also at r = 0)."""
    K = torch.zeros(r.shape[:-1] + (3, 3), dtype=r.dtype, device=r.device)
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 2] = -r[..., 2], r[..., 1], -r[..., 0]
    return torch.linalg.matrix_exp(K - K.transpose(-1, -2))


def episode(ep):
    tag = f"episode_{ep:06d}"
    d = dict(np.load(ARGS.input / f"{tag}.npz"))
    mz = np.load(ARGS.meshes / f"{tag}_budget.npz")
    V, F = mz["vertices"].astype(np.float64), mz["faces"]
    mesh = Mesh(V, F)
    K = json.loads((T1 / "outputs/camera" / tag / "intrinsics.json").read_text())
    fx, fy, cx, cy = K["fx"], K["fy"], K["cx"], K["cy"]
    R0 = torch.as_tensor(d["object_rotation"], dtype=torch.float64, device="cuda")
    t0 = torch.as_tensor(d["object_translation"], dtype=torch.float64, device="cuda")
    s = float(np.asarray(d["object_scale"]).reshape(()))
    T = len(t0)
    H = np.load(ARGS.hands / f"{tag}.npz")["hands"].astype(np.float64)[:T]
    Hg = torch.as_tensor(H, device="cuda")
    S = torch.as_tensor(surface_samples(V, F, N_SURF, 0), device="cuda")                 # local units
    P = torch.as_tensor(surface_samples(V, F, N_IMG, 1), device="cuda")
    c_loc = torch.as_tensor((V.min(0) + V.max(0)) / 2, device="cuda")
    centre0 = t0 + s * R0 @ c_loc

    def pose(dt, dr):
        R = rotation(dr) @ R0
        return R, centre0 + dt - s * R @ c_loc

    def project(X):
        return torch.stack([fx * X[..., 0] / X[..., 2] + cx, fy * X[..., 1] / X[..., 2] + cy], -1)

    u0 = project(s * P @ R0.transpose(1, 2) + t0[:, None])

    def local_np(R, t, f):
        return (H[f] - t[f]) @ R[f] / s

    def refresh(R, t):
        """Per frame: active hand points (inside, or within CONTACT_M), their inside flags, and contact frames."""
        Rn, tn = R.detach().cpu().numpy(), t.detach().cpu().numpy()
        idx = np.zeros((T, K_ACTIVE), np.int64)
        inside = np.zeros((T, K_ACTIVE), bool)
        valid = np.zeros((T, K_ACTIVE), bool)
        for f in range(T):
            q = local_np(Rn, tn, f)
            ins = mesh.inside(q)
            dist = torch.cdist(torch.as_tensor(q, device="cuda")[None], S[None])[0].min(1).values.cpu().numpy() * s
            cand = np.flatnonzero(ins | (dist < CONTACT_M))
            if cand.size == 0:
                continue
            order = cand[np.argsort(np.where(ins[cand], -1.0, dist[cand]))][:K_ACTIVE]   # inside first, then nearest
            idx[f, :len(order)], inside[f, :len(order)], valid[f, :len(order)] = order, ins[order], True
        return (torch.as_tensor(idx, device="cuda"), torch.as_tensor(inside, device="cuda"),
                torch.as_tensor(valid, device="cuda"))

    Bs = torch.as_tensor(bspline_basis(T, SPACING), device="cuda")
    ct = torch.zeros(Bs.shape[1], 3, dtype=torch.float64, device="cuda", requires_grad=True)
    cr = torch.zeros(Bs.shape[1], 3, dtype=torch.float64, device="cuda", requires_grad=True)
    dt, dr = Bs @ ct, Bs @ cr
    opt = torch.optim.Adam([ct, cr], lr=ARGS.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, ARGS.iters, eta_min=ARGS.lr / 10)
    R, t = pose(dt, dr)
    idx, inside, valid = refresh(R, t)
    contact = valid.any(1)                                  # frames in (or near) contact at the start: TOUCH applies
    frames_contact = int(contact.sum())
    hist = []
    for it in range(ARGS.iters):
        if it and it % REFRESH == 0:
            with torch.no_grad():
                R, t = pose(Bs @ ct, Bs @ cr)
            idx, inside, valid = refresh(R, t)
        dt, dr = Bs @ ct, Bs @ cr
        R, t = pose(dt, dr)
        q = (torch.gather(Hg, 1, idx[..., None].expand(-1, -1, 3)) - t[:, None]) @ R / s       # [T,K,3] local
        dist = torch.cdist(q.float(), S[None].float().expand(T, -1, -1)).min(-1).values.double() * s
        ninside = (inside & valid).sum(1).clamp_min(1)
        pen = ((dist * inside * valid) ** 2).sum(1) / ninside
        out_ok = valid & ~inside
        gap = torch.where(out_ok, dist, torch.full_like(dist, 1.0)).min(1).values
        touch = torch.where(contact & out_ok.any(1) & ~(inside & valid).any(1), torch.relu(gap - GAP_M) ** 2,
                            torch.zeros_like(gap))
        u = project(s * P @ R.transpose(1, 2) + t[:, None])
        img = ((u - u0) ** 2).sum(-1).mean(1) / SIG_PX ** 2
        d2t = dt[:-2] - 2 * dt[1:-1] + dt[2:]
        d2r = dr[:-2] - 2 * dr[1:-1] + dr[2:]
        loss = (img.sum() + W_PEN * pen.sum() + W_TOUCH * touch.sum() + W_SMOOTH_T * (d2t ** 2).sum()
                + W_SMOOTH_R * (d2r ** 2).sum() + W_PRIOR_T * (dt ** 2).sum() + W_PRIOR_R * (dr ** 2).sum()) / T
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if it % 50 == 0 or it == ARGS.iters - 1:
            hist.append([it, float(loss), float(img.mean()), float(pen.sum()), float(touch.sum())])

    with torch.no_grad():
        dt, dr = Bs @ ct, Bs @ cr
        R, t = pose(dt, dr)
    Rn, tn = R.cpu().numpy(), t.cpu().numpy()
    R0n, t0n = R0.cpu().numpy(), t0.cpu().numpy()
    sf = [f for f in ARGS.scored[str(ep)] if f < T]
    meshp = Mesh(V, F)
    before = [meshp.depth(local_np(R0n, t0n, f)) * s for f in sf]
    after = [meshp.depth(local_np(Rn, tn, f)) * s for f in sf]
    d["object_rotation"], d["object_translation"] = Rn.astype(np.float32), tn.astype(np.float32)
    np.savez(ARGS.out / f"{tag}.npz", **d)
    sfa = np.array(sf)
    inner = sfa[(sfa >= 1) & (sfa < T - 1)]
    acc = lambda x: float(np.linalg.norm(x[inner - 1] - 2 * x[inner] + x[inner + 1], axis=1).mean() * 100)  # noqa: E731
    with torch.no_grad():
        px = float(((project(s * P @ R.transpose(1, 2) + t[:, None]) - u0) ** 2).sum(-1).sqrt().mean())
    off = np.linalg.norm(tn - t0n, axis=1) * 100
    return {"episode": tag, "dense_pen_cm": [float(np.mean([b.mean() for b in before]) * 100), float(np.mean([a.mean() for a in after]) * 100)],
            "penetrating_frames": [int(sum(b.max() > 0 for b in before)), int(sum(a.max() > 0 for a in after))],
            "deep_frames_gt2cm": [int(sum(b.max() > 0.02 for b in before)), int(sum(a.max() > 0.02 for a in after))],
            "contact_frames": frames_contact, "image_shift_px": px, "offset_cm_mean": float(off.mean()), "offset_cm_max": float(off.max()),
            "rotation_deg_max": float(np.degrees(dr.detach().norm(dim=-1).max().item())),
            "object_acc_cm": [acc(t0n), acc(tn)], "history": hist}


def main():
    global ARGS
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--hands", type=Path, required=True, help="mhr_bodies.py --hands-only output of --input")
    ap.add_argument("--meshes", type=Path, required=True, help="budget_object_meshes.py output (+ scored_frames.json)")
    ap.add_argument("--episodes", type=int, nargs="*", default=list(range(30)))
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--lr", type=float, default=3e-3)
    ARGS = ap.parse_args()
    ARGS.out.mkdir(parents=True, exist_ok=True)
    tags = {f"episode_{ep:06d}" for ep in ARGS.episodes}
    for src in ARGS.input.iterdir():
        if src.is_file() and (src.name[:14] in tags if src.name.startswith("episode_") else not (ARGS.out / src.name).exists()):
            shutil.copy2(src, ARGS.out / src.name)
    ARGS.scored = json.loads((ARGS.meshes / "scored_frames.json").read_text())
    report = []
    for ep in ARGS.episodes:
        r = episode(ep)
        report.append(r)
        print(f"{r['episode']}: dense PEN {r['dense_pen_cm'][0]:.5f} -> {r['dense_pen_cm'][1]:.5f} cm, frames "
              f"{r['penetrating_frames'][0]} -> {r['penetrating_frames'][1]} (deep {r['deep_frames_gt2cm'][0]} -> "
              f"{r['deep_frames_gt2cm'][1]}), contact frames {r['contact_frames']}, image shift {r['image_shift_px']:.2f} px, "
              f"offset mean {r['offset_cm_mean']:.2f} / max {r['offset_cm_max']:.2f} cm, rot max {r['rotation_deg_max']:.1f} deg, "
              f"object acc {r['object_acc_cm'][0]:.4f} -> {r['object_acc_cm'][1]:.4f}", flush=True)
    name = "contact_solver_report" + ("_" + "_".join(map(str, ARGS.episodes)) if len(ARGS.episodes) < 30 else "") + ".json"
    (ARGS.out / name).write_text(json.dumps({"iters": ARGS.iters, "lr": ARGS.lr, "episodes": report}, indent=1))
    print("PASS: contact solver", len(report), "episodes")


if __name__ == "__main__":
    main()
