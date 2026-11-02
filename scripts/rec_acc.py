#!/usr/bin/env python3
"""Part B: exact ACC-O transfer of the along-ray object-bias correction on the RECOVERED candidate.
Computes, per episode: the bias series B(f) (same pipeline as depth_icp_objects --apply, frame mode),
the added acceleration ||d2(B*ray)|| in cm/frame^2, and the original translation's own 2nd-diff scale.
Run with the submission venv (needs cv2). Writes logs/rec_acc.json."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("/hpcwork/mdk96850/v2d-track1")
sys.path.insert(0, str(ROOT / "scripts"))
from depth_icp_objects import decode_depth, frame_bias  # noqa: E402

EPS = list(range(30))
PCT, SIGMA, MIN_B, MAX_B = 70.0, 8, 0.01, 0.20

out = {}
for ep in EPS:
    tag = f"episode_{ep:06d}"
    f_npz = ROOT / f"submissions/episodes_recovered_cand/{tag}.npz"
    if not f_npz.exists():
        continue
    d0 = dict(np.load(f_npz))
    R, t = d0["object_rotation"].astype(np.float64), d0["object_translation"].astype(np.float64)
    s = float(np.asarray(d0["object_scale"]).reshape(()))
    scored = np.where(np.isfinite(R).all(axis=(1, 2)) & np.isfinite(t).all(axis=1))[0]
    if len(scored) < 10:
        continue
    V = d0["object_mesh_vertices"].astype(np.float64)
    if V.shape[0] > 6000:
        V = V[np.random.default_rng(0).choice(V.shape[0], 6000, replace=False)]
    dep_dir = ROOT / f"outputs/object/{tag}/depth"
    msk_dir = ROOT / f"outputs/object/{tag}/sam2_masks/1"
    Kp = ROOT / f"outputs/camera/{tag}/intrinsics.json"
    K = json.loads(Kp.read_text()) if Kp.exists() else \
        json.loads((dep_dir.parent / "intrinsics" / f"{scored[0]:06d}.json").read_text())
    Kt = (K["fx"], K["fy"], K["cx"], K["cy"])
    ap = ROOT / f"outputs/alignment_sam/{tag}/alignment.json"
    scale = float(json.loads(ap.read_text())["depth_scale"]) if ap.exists() else 1.0

    biases, res0 = [], []
    for f in scored:
        dd = decode_depth(dep_dir / f"{f:06d}.png", scale)
        if dd is None:
            biases.append(np.nan); res0.append(np.nan); continue
        valid = np.isfinite(dd) & (dd > 1e-3)
        msk = cv2.imread(str(msk_dir / f"{f:06d}.png"), cv2.IMREAD_GRAYSCALE)
        if msk is None:
            biases.append(np.nan); res0.append(np.nan); continue
        b, npts, diag, _ = frame_bias(V, R[f], t[f], s, dd, valid, Kt, msk > 127, PCT)
        biases.append(b)
        res0.append(np.nan if diag is None else diag[0])
    biases = np.asarray(biases)
    med = np.nanmedian(biases)
    if not np.isfinite(med) or abs(med) < MIN_B:
        print(f"{tag}: no correction (bias {med * 100 if np.isfinite(med) else float('nan'):5.2f} cm)", flush=True)
        continue
    b_g = np.where(np.isfinite(biases) & (np.abs(biases - med) <= MAX_B), biases, med)
    k = int(3 * SIGMA)
    xs = np.arange(-k, k + 1)
    g = np.exp(-0.5 * (xs / SIGMA) ** 2)
    g /= g.sum()
    B = np.convolve(np.nan_to_num(b_g), g, mode="same")
    B *= 1.0 - np.exp(-0.5 * (np.arange(len(scored)) / 20.0) ** 2)      # anchor taper (as applied)
    rays = t[scored] / np.maximum(np.linalg.norm(t[scored], axis=1, keepdims=True), 1e-9)
    C = -B[:, None] * rays                                              # correction vectors c = -B*ray
    T0 = t[scored]
    breaks = np.flatnonzero(np.diff(scored) != 1) + 1
    edges = [0, *breaks.tolist(), len(scored)]
    da, do = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        if b - a < 3:
            continue
        ca, ta = C[a:b], T0[a:b]
        da.append(np.linalg.norm(ca[:-2] - 2 * ca[1:-1] + ca[2:], axis=1))
        do.append(np.linalg.norm(ta[:-2] - 2 * ta[1:-1] + ta[2:], axis=1))
    da = np.concatenate(da) * 100.0                                     # cm/frame^2
    do = np.concatenate(do) * 100.0
    out[tag] = {"n_frames": int(len(scored)), "bias_cm": round(float(med) * 100, 2),
                "res_before_cm": round(float(np.nanmedian(res0)) * 100, 2),
                "acc_added_cm": round(float(np.mean(da)), 3),
                "acc_orig_cm": round(float(np.mean(do)), 3)}
    r = out[tag]
    print(f"{tag}: bias {r['bias_cm']:6.2f} cm | res {r['res_before_cm']:6.2f} cm | "
          f"added |d2c| {r['acc_added_cm']:7.3f} cm/f2 | orig |d2t| {r['acc_orig_cm']:7.3f} cm/f2", flush=True)

add = np.array([v["acc_added_cm"] for v in out.values()])
org = np.array([v["acc_orig_cm"] for v in out.values()])
summary = {"n_eps": len(out),
           "mean_added_cm": round(float(add.mean()), 3), "p90_added_cm": round(float(np.percentile(add, 90)), 3),
           "max_added_cm": round(float(add.max()), 3),
           "mean_orig_cm": round(float(org.mean()), 3),
           "ratio_pct": round(float(add.mean() / org.mean() * 100), 1) if org.mean() > 0 else None}
print(f"\nB-SUMMARY: {summary}", flush=True)
(ROOT / "logs/rec_acc.json").write_text(json.dumps({"episodes": out, "summary": summary}, indent=1))
print("wrote logs/rec_acc.json")
