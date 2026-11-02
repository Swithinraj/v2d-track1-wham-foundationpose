#!/usr/bin/env python3
"""Along-ray depth-bias correction of the object track against MoGe-2's metric depth (numpy + cv2).

A geometric correction term that exists nowhere in the chain: FoundationPose registers in
image space, the chain only smooths. The full per-frame point-to-plane ICP variant of this
script DIVERGED on our data (~98% of frames: inside a thin object mask -- the hula-hoop ring,
the rollers -- most pixels are background, and depth-edge normals are garbage, so the
least-squares rigid solve wants >15 cm / >15 deg corrections and is rejected). This variant
estimates what the data actually supports, per scored frame:

  bias(f) = a high percentile (default PCT=70) of the along-ray residual r = z_render - d_depth
  over the object mask's valid pixels. z_render is the mesh's own FRONTMOST visible surface at
  this pose (z-buffered from a dense surface sample, not individual mesh vertices -- see FIX
  below), so r isolates true placement error instead of mixing it with object thickness.

FIX (2026-11-03): the original version projected every sampled mesh point -- front AND back
surface -- and compared each one individually to the observed depth, before taking the
percentile. Verified by direct test: on a PERFECTLY aligned synthetic pose (zero true placement
error, by construction), the original method reported a P70 bias of +2.8 to +11.2 cm across 5
different real objects from this project, entirely from back-facing/occluded points being
compared against a depth map that only ever sees the front surface. The depth/mesh frame-ratio
self-consistency check does NOT catch this (it read 0.98-1.00, "clean", in every one of those
cases) -- it is a different error mode from frame/scale mismatch.
  Fix: z-buffer the projected points first (np.argsort by depth + first-occurrence-per-pixel),
keeping only the nearest (frontmost) projected point per pixel, before comparing to the observed
depth. Re-validated on the same 5 synthetic perfect-pose tests: bias dropped to +0.1 to +0.6 cm
in 4/5 cases; the 5th (a large, possibly-concave object) dropped from +7.0 to +2.7 cm -- a large
improvement, not perfectly zero, likely pixel-discretization/sampling effects on big objects.
  This needs denser sampling than the original's <=6000-point cap to z-buffer reliably (sparse
points can't resolve front-vs-back per pixel) -- DENSE_SAMPLE points, drawn once per episode
from the mesh surface (pose changes per frame, the mesh does not), z-buffered fresh per frame.

The per-frame biases are then robustly regularised in time (per-episode median gating, then a
Gaussian sigma SMOOTH_SIGMA, then an anchor taper to zero at the first scored frame -- the
scorer's alignment frame must not move) and applied as a translation along the object centre's
camera ray. No rotation: the mask evidence cannot support it robustly. (This regularisation is
UNCHANGED from the original -- only the per-frame bias measurement itself was wrong.)

Known remaining limitation, not fixed here: the temporal smoothing/taper operates over the
`scored` array by LIST POSITION, assuming consecutive entries are consecutive video frames. If
an episode's tracked object data has internal NaN gaps, this would blend across them -- checked
empirically for P3 (all 30 episodes' object tracks are fully contiguous, no gaps), so this does
not affect P3, but would need `accel_shrink.py`-style stretch-splitting before trusting this on
any episode/candidate where the object track itself has gaps.

Depth is decoded 65535/v - 1 metres and DIVIDED by the episode's alignment depth_scale (the
first run proved this mandatory: the raw ratio diagnostic read ~0.887 = ep0's depth_scale).

Writes <out>/<tag>.npz (same schema + object_translation corrected) and a report JSON.
Without --apply: report only.

Usage: depth_icp_objects.py --episodes DIR [--apply OUT_DIR] [--episodes-list 0 1 ...]
  [--pct 70] [--sigma 8] [--min-bias-cm 1.0] [--max-bias-cm 20.0] [--report FILE]
  [--dense_sample 60000]
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh

ROOT = Path(os.environ.get("TRACK1_ROOT") or Path(__file__).resolve().parents[1])


def decode_depth(png, scale):
    """MoGe metres (65535/v - 1, depth_cues.py) -> our frame (divided by the episode's depth scale)."""
    v = cv2.imread(str(png), cv2.IMREAD_UNCHANGED)
    if v is None:
        return None
    v = v.astype(np.float64)
    if v.ndim == 3:
        v = v[..., 0]
    return (65535.0 / np.maximum(v, 1.0) - 1.0) / scale


def frame_bias(V, R, t, s, d, valid, K, mask, pct):
    """Robust along-ray bias of the mesh's FRONTMOST surface against the depth surface, or (nan, npts).

    V must be a DENSE surface sample (not sparse vertices) for the z-buffer step to reliably
    resolve front vs. back per pixel. Z-buffers the projection (nearest point wins each pixel)
    before computing the residual -- this is the fix; see the module docstring.

    Also returns the mask-gated depth/mesh z ratio (frame-consistency diagnostic: if it is
    far from 1, the episode's frame does not line up with depth/scale and any bias is
    suspect). Note: a ratio near 1 does NOT by itself mean the bias is trustworthy -- the
    pre-fix version of this function could read ratio~1 and still be off by several cm from
    the backface artifact alone; the z-buffer step is what actually fixes that, not the ratio.
    """
    fx, fy, cx, cy = K
    X = s * (V @ R.T) + t
    z = X[:, 2]
    u = fx * X[:, 0] / np.maximum(z, 1e-9) + cx
    v = fy * X[:, 1] / np.maximum(z, 1e-9) + cy
    ui, vi = np.round(u).astype(int), np.round(v).astype(int)
    inb = (z > 1e-3) & (ui >= 0) & (ui < d.shape[1]) & (vi >= 0) & (vi < d.shape[0])
    if inb.sum() == 0:
        return np.nan, 0, None, np.nan
    ui_in, vi_in, z_in = ui[inb], vi[inb], z[inb]
    order = np.argsort(z_in, kind="stable")              # nearest (smallest z) first
    ui_s, vi_s, z_s = ui_in[order], vi_in[order], z_in[order]
    pix = vi_s.astype(np.int64) * d.shape[1] + ui_s.astype(np.int64)
    _, first = np.unique(pix, return_index=True)          # first occurrence per pixel = nearest
    ui_f, vi_f, z_f = ui_s[first], vi_s[first], z_s[first]
    dv = np.where(valid[vi_f, ui_f], d[vi_f, ui_f], np.nan)
    mv = mask[vi_f, ui_f] > 0
    ok = np.isfinite(dv) & mv
    npts = int(ok.sum())
    if npts < 30:
        return np.nan, npts, None, np.nan
    r = z_f[ok] - dv[ok]                                  # along-ray residual, frontmost surface only
    bias = float(np.percentile(r, pct))
    cover = float(mv.mean())
    res = float(np.sqrt(np.median(r ** 2)))
    ratio = float(np.median(dv[ok] / z_f[ok]))
    return bias, npts, (res, cover), ratio


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--episodes", type=Path, required=True, help="episode dir (npz + *_object.glb)")
    ap.add_argument("--apply", type=Path, default=None, help="write corrected episodes here")
    ap.add_argument("--episodes-list", type=int, nargs="*", default=list(range(30)))
    ap.add_argument("--pct", type=float, default=70.0, help="residual percentile for the object's bias")
    ap.add_argument("--const", action="store_true", help="apply the episode-median bias only (no per-frame tracking of MoGe's scale wobble)")
    ap.add_argument("--sigma", type=int, default=8, help="temporal Gaussian sigma (frames)")
    ap.add_argument("--min-bias-cm", type=float, default=1.0, help="episodes with smaller median |bias| stay unchanged")
    ap.add_argument("--max-bias-cm", type=float, default=20.0, help="per-frame biases beyond this are outliers (gated to the episode median)")
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--dense_sample", type=int, default=60_000, help="surface points sampled once per episode for the z-buffer render")
    a = ap.parse_args()
    rep = {}
    if a.apply:
        a.apply.mkdir(parents=True, exist_ok=True)
    for ep in a.episodes_list:
        tag = f"episode_{ep:06d}"
        f_npz = a.episodes / f"{tag}.npz"
        if not f_npz.exists():
            continue
        d0 = dict(np.load(f_npz))
        R, t = d0["object_rotation"].astype(np.float64), d0["object_translation"].astype(np.float64)
        s = float(np.asarray(d0["object_scale"]).reshape(()))
        scored = np.where(np.isfinite(R).all(axis=(1, 2)) & np.isfinite(t).all(axis=1))[0]
        if len(scored) < 10:
            print(f"{tag}: <10 scored frames, skipped"); continue
        mesh_p = a.episodes / f"{tag}_object.glb"
        if not mesh_p.exists():
            mesh_p = a.episodes / f"{tag}_object.ply"          # unpacked-candidate layout
        mesh = trimesh.load(mesh_p, force="mesh")
        V = mesh.sample(a.dense_sample)                        # dense surface sample, once per episode
        K = json.loads((ROOT / f"outputs/camera/{tag}/intrinsics.json").read_text()) \
            if (ROOT / f"outputs/camera/{tag}/intrinsics.json").exists() else None
        dep_dir = ROOT / f"outputs/object/{tag}/depth"
        msk_dir = ROOT / f"outputs/object/{tag}/sam2_masks/1"
        if K is None:
            K = json.loads((dep_dir.parent / "intrinsics" / f"{scored[0]:06d}.json").read_text())
        Kt = (K["fx"], K["fy"], K["cx"], K["cy"])
        align_p = ROOT / "outputs/alignment_sam" / tag / "alignment.json"
        scale = float(json.loads(align_p.read_text())["depth_scale"]) if align_p.exists() else 1.0

        biases, res0s, cov0s, res1s, cov1s = [], [], [], [], []
        ratios = []
        for f in scored:
            d = decode_depth(dep_dir / f"{f:06d}.png", scale)
            if d is None:
                biases.append(np.nan); res0s.append(np.nan); cov0s.append(np.nan); continue
            valid = np.isfinite(d) & (d > 1e-3)
            msk = cv2.imread(str(msk_dir / f"{f:06d}.png"), cv2.IMREAD_GRAYSCALE)
            if msk is None:
                biases.append(np.nan); res0s.append(np.nan); cov0s.append(np.nan); continue
            b, npts, diag, rat = frame_bias(V, R[f], t[f], s, d, valid, Kt, msk > 127, a.pct)
            ratios.append(rat)
            biases.append(b)
            if diag is None:
                res0s.append(np.nan); cov0s.append(np.nan); res1s.append(np.nan); cov1s.append(np.nan)
            else:
                res0s.append(diag[0]); cov0s.append(diag[1])
        biases = np.asarray(biases)
        med = np.nanmedian(biases)
        if not np.isfinite(med) or abs(med) * 100 < a.min_bias_cm:
            ratio_med = float(np.nanmedian(ratios)) if ratios else None
            rep[tag] = {"frames": int(len(scored)), "bias_cm_median": None if not np.isfinite(med) else round(med * 100, 2),
                        "depth_frame_ratio": round(ratio_med, 4) if np.isfinite(ratio_med) else None,
                        "applied": False, "reason": "bias below threshold" if np.isfinite(med) else "no depth evidence"}
            print(f"{tag}: bias {med*100 if np.isfinite(med) else float('nan'):5.2f} cm — below threshold, unchanged", flush=True)
            continue
        # outlier gating: per-frame biases beyond the cap collapse to the episode median
        b_g = np.where(np.isfinite(biases) & (np.abs(biases - med) <= a.max_bias_cm / 100.0), biases, med)
        if a.const:
            b_g = np.full_like(b_g, med)
        # temporal Gaussian smoothing + anchor taper
        k = int(3 * a.sigma)
        xs = np.arange(-k, k + 1)
        g = np.exp(-0.5 * (xs / a.sigma) ** 2); g /= g.sum()
        B = np.convolve(np.nan_to_num(b_g), g, mode="same")
        taper = 1.0 - np.exp(-0.5 * (np.arange(len(scored)) / 20.0) ** 2)  # ramp UP from the pinned
        B *= taper                                    # anchor frame; full correction afterwards
        # apply along the object centre's ray; recompute the same-frame residual after
        t2 = t.copy()
        for i, f in enumerate(scored):
            ray = t[f] / max(np.linalg.norm(t[f]), 1e-9)
            t2[f] = t[f] - B[i] * ray
        for j, f in enumerate(scored):
            d = decode_depth(dep_dir / f"{f:06d}.png", scale) if np.isfinite(B[j]) else None
            if d is None or not np.isfinite(B[j]):
                res1s.append(np.nan); cov1s.append(np.nan); continue
            valid = np.isfinite(d) & (d > 1e-3)
            msk = cv2.imread(str(msk_dir / f"{f:06d}.png"), cv2.IMREAD_GRAYSCALE)
            if msk is None:
                res1s.append(np.nan); cov1s.append(np.nan); continue
            _, _, diag, _ = frame_bias(V, R[f], t2[f], s, d, valid, Kt, msk > 127, a.pct)
            res1s.append(np.nan if diag is None else diag[0])
            cov1s.append(np.nan if diag is None else diag[1])
        rep[tag] = {"frames": int(len(scored)), "applied": True,
                    "bias_cm_median": round(med * 100, 2),
                    "bias_cm_after_smooth": round(float(np.nanmedian(np.abs(B))) * 100, 2),
                    "depth_frame_ratio": round(float(np.nanmedian(ratios)), 4) if ratios else None,
                    "same_frame_cm_before": round(float(np.nanmedian(res0s)) * 100, 2),
                    "same_frame_cm_after": round(float(np.nanmedian(res1s)) * 100, 2),
                    "mask_cover_before": round(float(np.nanmedian(cov0s)), 3),
                    "mask_cover_after": round(float(np.nanmedian(cov1s)), 3),
                    "depth_scale": scale}
        print(f"{tag}: bias {med*100:5.2f} cm | same-frame res {rep[tag]['same_frame_cm_before']:5.2f}->"
              f"{rep[tag]['same_frame_cm_after']:5.2f} cm | cover {rep[tag]['mask_cover_before']:.3f}->"
              f"{rep[tag]['mask_cover_after']:.3f}", flush=True)
        if a.apply:
            d2 = dict(d0)
            d2["object_translation"] = t2.astype(np.float32)
            np.savez(a.apply / f"{tag}.npz", **d2)
            src_mesh = a.episodes / f"{tag}_object.glb"
            if not src_mesh.exists():
                src_mesh = a.episodes / f"{tag}_object.ply"        # unpacked-candidate layout
            shutil.copy2(src_mesh, a.apply / src_mesh.name)
    if a.report:
        a.report.parent.mkdir(parents=True, exist_ok=True)
        a.report.write_text(json.dumps(rep, indent=1))
    ap_ = [r for r in rep.values() if r.get("applied") and r["same_frame_cm_after"] < r["same_frame_cm_before"]]
    print(f"\nSUMMARY: {len(ap_)}/{len(rep)} episodes applied and improved same-frame residual; "
          f"median bias {np.median([r['bias_cm_median'] for r in rep.values() if r.get('bias_cm_median') is not None]):.2f} cm")


if __name__ == "__main__":
    main()
