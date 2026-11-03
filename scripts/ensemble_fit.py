#!/usr/bin/env python3
"""Fit MHR pose to the multi-chain pseudo-GT vertex average (an ensemble human), warm-started from P2.

The pseudo-GT reference (localscore.py build-ref, H-BRS job 445021) holds per episode the vertex
average of three independent human systems (P2 SAM3D-ensemble, MF WHAM-based, C1 CARI4D), each
Sim(3)-aligned per episode to the base chain P2, plus the P4tab102 objects mapped into that frame.
Averaging independent estimates cancels their independent errors but keeps the motion they agree
on. Temporal smoothing does the opposite: it removes real motion, and Kaggle showed it hurts ACC-H.

This script turns the averaged vertices back into MHR parameters with the kit's own solver
(official/v2d_submission_kit/tools/track1/mesh_to_mhr_params.py: lm_pose, per-frame
Levenberg-Marquardt, finite-difference Jacobians through the reference model, float64):
  * warm start: the base chain's pose for every frame (no cold fits);
  * fitted: the non-finger pose columns (0..67 root + body, 122..135);
  * held: identity (scales, shape) and finger parameters (68..121) from the base chain;
  * weights: hand vertices weighted out, since the averaged fingers are not a pose.

Output, one npz per episode in the scored-frames layout (as make_variants/localscore): frames,
mhr_pose, mhr_scales, mhr_shape, the reference's object arrays, and per-frame body-vertex errors
before/after (mm) against the reference.

usage: ensemble_fit.py --ref DIR --base DIR --out DIR --model mhr_model.pt --kit KITDIR
                       [--episodes E ...] [--shard I --nshards N] [--iters 15] [--chunk 550]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

FIT_COLS = list(range(0, 68)) + list(range(122, 136))
OBJECT_KEYS = ("object_rotation", "object_translation", "object_scale",
               "object_mesh_vertices", "object_mesh_faces")


def shard_episodes(sizes: dict[int, int], shard: int, nshards: int) -> list[int]:
    """Longest-processing-time assignment of episodes to shards by frame count."""
    loads = [0] * nshards
    owner = {}
    for ep in sorted(sizes, key=lambda e: -sizes[e]):
        i = int(np.argmin(loads))
        owner[ep] = i
        loads[i] += sizes[ep]
    return sorted(e for e, i in owner.items() if i == shard)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--base", type=Path, required=True, help="base chain npz dir (frames, mhr_pose, ...)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--kit", type=Path, required=True)
    ap.add_argument("--episodes", type=int, nargs="*", default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--chunk", type=int, default=550, help="model forward batch")
    ap.add_argument("--precision", default="float64", choices=("float64", "float32"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    sys.path.insert(0, str(a.kit / "tools/track1"))
    import mesh_to_mhr_params as m2m  # the kit's solver and model wrapper

    eps = a.episodes
    if eps is None:
        eps = sorted(int(p.stem.split("_")[1]) for p in a.ref.glob("episode_*.npz"))
    sizes = {e: int(np.load(a.ref / f"episode_{e:06d}.npz", mmap_mode="r")["ref_vertices"].shape[0]) for e in eps}
    mine = shard_episodes(sizes, a.shard, a.nshards)
    print(f"shard {a.shard}/{a.nshards}: episodes {mine} ({sum(sizes[e] for e in mine)} frames) "
          f"device={a.device} precision={a.precision} iters={a.iters}", flush=True)

    mhr = m2m.MHR(str(a.model), a.device, chunk=a.chunk, precision=a.precision)
    w = m2m.hand_vertex_weights(mhr)
    print(f"body vertices weighted in: {int(w.sum())}/{len(w)}", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    report = {}
    for ep in mine:
        t0 = time.time()
        ref = np.load(a.ref / f"episode_{ep:06d}.npz")
        base = dict(np.load(a.base / f"episode_{ep:06d}.npz"))
        T = len(base["frames"])
        assert ref["ref_vertices"].shape == (T, m2m.NUM_VERTS, 3), (ep, ref["ref_vertices"].shape, T)
        dd = dict(dtype=torch.float64, device=mhr.device)
        tgt = torch.as_tensor(ref["ref_vertices"].astype(np.float64) * 1000.0, **dd).reshape(T, -1)  # m -> mm
        pose0 = torch.as_tensor(base["mhr_pose"], **dd)
        z = torch.as_tensor(np.concatenate([base["mhr_scales"], base["mhr_shape"]])[None], **dd)

        def body_err(p):
            e = (mhr.verts(p, z) - tgt).reshape(T, -1, 3).norm(dim=-1)
            return ((e * w).sum(1) / w.sum()).cpu().numpy()

        e0 = body_err(pose0)
        pose, _ = m2m.lm_pose(mhr, tgt, pose0, z, iters=a.iters, cols=FIT_COLS, w=w)
        e1 = body_err(pose)
        P = pose.cpu().numpy()
        assert np.array_equal(P[:, 68:122], base["mhr_pose"][:, 68:122]), "finger columns moved"
        out = {"frames": base["frames"], "mhr_pose": P, "mhr_scales": base["mhr_scales"],
               "mhr_shape": base["mhr_shape"], "err_before_mm": e0, "err_after_mm": e1}
        out.update({k: ref[k] for k in OBJECT_KEYS})
        np.savez(a.out / f"episode_{ep:06d}.npz", **out)
        dt = time.time() - t0
        report[ep] = {"T": T, "err_before_mm": float(e0.mean()), "err_after_mm": float(e1.mean()),
                      "err_after_p95_mm": float(np.percentile(e1, 95)), "seconds": dt}
        print(f"  ep{ep}: T={T} body err {e0.mean():.2f} -> {e1.mean():.2f} mm "
              f"(p95 {np.percentile(e1, 95):.2f})  {dt:.0f}s ({dt / T:.2f} s/frame)", flush=True)
    (a.out / f"report_shard{a.shard}.json").write_text(json.dumps(report, indent=1))
    print(f"DONE shard {a.shard}: {len(report)} episodes", flush=True)


if __name__ == "__main__":
    main()
