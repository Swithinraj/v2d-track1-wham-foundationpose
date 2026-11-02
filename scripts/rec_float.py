#!/usr/bin/env python3
"""Part A: sole float of the RECOVERED candidate's humans above the depth-RANSAC floor planes.
Run inside mhr_converter.sif (needs torch; no cv2). Writes logs/rec_float.json."""
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/track1")
EPS = list(range(30))

floors = json.loads((ROOT / "outputs/physics/floor_planes_soles.json").read_text())
floor_rep = json.loads((ROOT / "outputs/physics/floor_planes_soles.report.json").read_text())

sys_path = str(ROOT / "official/v2d_submission_kit/tools/track1")
import sys
sys.path.insert(0, sys_path)
from mesh_to_mhr_params import MHR  # noqa: E402

dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[device] {dev}", flush=True)
mhr = MHR(str(ROOT / "weights/mhr/assets/mhr_model.pt"), dev, chunk=64, precision="float32")

out = {}
for ep in EPS:
    tag = f"episode_{ep:06d}"
    if tag not in floors:
        continue
    d = dict(np.load(ROOT / f"submissions/episodes_recovered_cand/{tag}.npz"))
    pose = d["pose"].astype(np.float32)
    finite = np.flatnonzero(np.isfinite(pose).all(axis=1))
    if len(finite) < 5:
        print(f"{tag}: <5 finite frames, skipped", flush=True)
        continue
    frames = finite[np.linspace(0, len(finite) - 1, min(24, len(finite))).astype(int)]
    ident = np.concatenate([d["mhr_scales"], d["mhr_shape"]]).astype(np.float32)
    with torch.no_grad():
        v_mm, _ = mhr.run(torch.as_tensor(pose[frames]),
                          torch.as_tensor(np.broadcast_to(ident, (len(frames), len(ident)))))
    verts = v_mm.reshape(len(frames), -1, 3).double().numpy() / 1000.0   # mm -> m, our frame
    n = np.asarray(floors[tag]["normal"], np.float64)
    c_our = floors[tag]["c"] / floor_rep[tag]["depth_scale"]             # stored c is MoGe-frame
    low = (verts @ n - c_our).min(axis=1)                                # lowest vertex above plane
    out[tag] = {"sole_float_cm_median": round(float(np.median(low) * 100), 2),
                "sole_float_cm_p90": round(float(np.percentile(low, 90) * 100), 2)}
    print(f"{tag}: sole float median {out[tag]['sole_float_cm_median']:+6.2f} cm  "
          f"p90 {out[tag]['sole_float_cm_p90']:+6.2f} cm", flush=True)

meds = np.array([v["sole_float_cm_median"] for v in out.values()])
summary = {"n_eps": len(out), "mean_cm": round(float(meds.mean()), 2), "median_cm": round(float(np.median(meds)), 2),
           "eps_gt2cm": int((meds > 2).sum()), "eps_gt5cm": int((meds > 5).sum()), "eps_gt10cm": int((meds > 10).sum()),
           "eps_gt1cm": int((meds > 1).sum()), "mean_gt1cm": round(float(meds[meds > 1].mean()), 2) if (meds > 1).any() else 0.0}
print(f"\nA-SUMMARY: {summary}", flush=True)
(ROOT / "logs/rec_float.json").write_text(json.dumps({"episodes": out, "summary": summary}, indent=1))
print("wrote logs/rec_float.json")
