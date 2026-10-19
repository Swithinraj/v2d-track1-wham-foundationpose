#!/usr/bin/env python3
"""Signed-distance grids of each episode's object, consistent with the scorer's PEN (submission venv, numba).

On the BUDGETED object mesh (the mesh the packer writes and the scorer tests): sign = generalised winding number
|w| > 0.5 (inside), magnitude = exact point-to-triangle distance (the kit's own penetration kernels), local mesh units.
Robust to small holes (non-watertight meshes such as the iron and the foam roll). Grid: N voxels along the longest
axis, margin 15 % of the extent. Used by human_contact_solver.py (trilinear, differentiable lookups).
Writes <out>/<episode>.npz (sdf, lo, hi, n). Usage: contact_sdf.py --episodes DIR [--out DIR] [--n 112]
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("TRACK1_ROOT") or Path(__file__).resolve().parents[1])
sys.path.insert(0, str(ROOT / "scripts"))
import resolve_hand_penetration as rp  # noqa: E402
from v2dlb import mhr_submission as ms  # noqa: E402
from v2dlb.mesh_budget import budget_mesh  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/contact_sdf")
    ap.add_argument("--n", type=int, default=112)
    a = ap.parse_args()
    _, budget = rp.scored_frames_and_budget()
    winding, distance = ms._PENETRATION_KERNELS
    a.out.mkdir(parents=True, exist_ok=True)
    for ep in range(30):
        tag = f"episode_{ep:06d}"
        V, F = budget_mesh(a.episodes / f"{tag}_object.glb", *budget); V = np.asarray(V, np.float64); F = np.asarray(F)
        v0, v1, v2 = (np.ascontiguousarray(V[F[:, k]]) for k in range(3))
        lo, hi = V.min(0), V.max(0); ext = hi - lo; lo, hi = lo - 0.15 * ext.max(), hi + 0.15 * ext.max()
        vox = (hi - lo).max() / (a.n - 1); dims = np.maximum(np.ceil((hi - lo) / vox).astype(int) + 1, 8); hi = lo + (dims - 1) * vox
        G = np.ascontiguousarray(np.stack(np.meshgrid(*[np.linspace(lo[i], hi[i], dims[i]) for i in range(3)], indexing="ij"), -1).reshape(-1, 3))
        w = winding(G, v0, v1, v2); d = distance(G, v0, v1, v2)
        sdf = np.where(np.abs(w) > 0.5, -d, d).astype(np.float32).reshape(dims)
        np.savez_compressed(a.out / f"{tag}.npz", sdf=sdf, lo=lo.astype(np.float32), hi=hi.astype(np.float32), n=dims)
        print(f"{tag}: grid {dims.tolist()}, voxel {vox*100:.2f} local cm, inside share {float((sdf < 0).mean()):.3f}, faces {len(F)}", flush=True)


if __name__ == "__main__":
    main()
