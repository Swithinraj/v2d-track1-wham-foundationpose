#!/usr/bin/env python3
"""Apply per-frame rigid displacements to an episodes directory (mhr_converter.sif, GPU): the human through its root
translation (pose[:, :3]), the object through object_translation. Displacements from scripts/corr_select3.py --dump_shifts
(d_h, d_o [T, 3], our camera frame, metres); a common translation (d_h = d_o) leaves every hand-object relation unchanged.
Checked per episode: the MHR joints of every frame move by exactly d_h (tolerance 0.5 mm). Frames beyond the shift file
(cache shorter than the episode) keep its last displacement (no jump); episodes without a shift file are copied unchanged.
Usage: apply_shift.py --input DIR --shifts DIR --prefix ROOTNAME --out DIR
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(os.environ.get("TRACK1_ROOT") or "/track1")
sys.path.insert(0, str(ROOT / "official/v2d_submission_kit/tools/track1"))
sys.path.insert(0, str(ROOT / "scripts"))
from mesh_to_mhr_params import MHR  # noqa: E402
from mhr_bodies import MODEL  # noqa: E402

# The root-translation units / axes are MEASURED per frame (one forward pass per axis, joints 1: ; joint 0 is the model's
# static world origin and does not follow the root translation) and inverted; measured: 0.1 m per unit with the axis flip.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True); ap.add_argument("--shifts", type=Path, required=True)
    ap.add_argument("--prefix", required=True); ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    for f in a.input.iterdir():
        if f.is_file() and not (f.name.startswith("episode_") and f.suffix == ".npz"):
            shutil.copy2(f, a.out / f.name)                                                     # object meshes, reports
    mhr = MHR(str(MODEL), "cuda", chunk=128, precision="float64"); rep = {}
    for src in sorted(a.input.glob("episode_*.npz")):
        tag = src.stem; d = dict(np.load(src)); sp = a.shifts / f"{a.prefix}_{tag}.npz"
        if not sp.exists():
            np.savez(a.out / src.name, **d); rep[tag] = "unshifted (no shift file)"; continue
        s = np.load(sp); T = len(d["pose"]); n = min(T, len(s["d_h"]))
        pad = lambda x: np.concatenate([x[:n], np.repeat(x[n - 1:n], T - n, 0)]) if T > n else x[:T]   # noqa: E731
        dh, do = pad(s["d_h"].astype(float)), pad(s["d_o"].astype(float))
        ident = torch.as_tensor(np.concatenate([d["scales"], d["shape"]])[None], dtype=torch.float64, device="cuda")
        p0 = torch.as_tensor(d["pose"], dtype=torch.float64, device="cuda"); _, J0 = mhr.run(p0, ident)
        M = []                                                     # measured joint shift (m, camera) per unit of each translation param
        for k in range(3):                                         # joint 0 = the model's static world origin: excluded
            pk = p0.clone(); pk[:, k] += 1.0; _, Jk = mhr.run(pk, ident); M.append((Jk - J0)[:, 1:].mean(1))
        M = torch.stack(M, -1)                                                                    # [T, 3, 3]
        p1 = p0.clone(); p1[:, :3] += torch.linalg.solve(M, torch.as_tensor(dh, device="cuda")[..., None])[..., 0]
        _, J1 = mhr.run(p1, ident)
        err = float((J1 - J0 - torch.as_tensor(dh, device="cuda")[:, None])[:, 1:].norm(dim=-1).max())
        assert err < 5e-4, f"{tag}: root-translation shift off by {err * 1000:.2f} mm"
        d["pose"] = p1.cpu().numpy().astype(d["pose"].dtype)
        ot = d["object_translation"].astype(float); assert ot.ndim == 2 and ot.shape[1] == 3, ot.shape
        ot[:T] += do[:len(ot[:T])]; d["object_translation"] = ot.astype(d["object_translation"].dtype)
        np.savez(a.out / src.name, **d)
        rep[tag] = {"human_shift_cm_mean": float(np.linalg.norm(dh, axis=1).mean() * 100), "human_shift_cm_max": float(np.linalg.norm(dh, axis=1).max() * 100),
                    "object_shift_cm_mean": float(np.linalg.norm(do, axis=1).mean() * 100), "relative_shift_cm_max": float(np.linalg.norm(dh - do, axis=1).max() * 100),
                    "check_err_mm": err * 1000, "frames_padded": T - n, "m_per_unit_diag": [round(float(x), 5) for x in M[0].diagonal()]}
        print(tag, json.dumps(rep[tag]), flush=True)
    (a.out / "apply_shift_report.json").write_text(json.dumps(rep, indent=1))
    print(f"shifted {sum(isinstance(v, dict) and v['human_shift_cm_max'] > 0 for v in rep.values())} of {len(rep)} episodes", flush=True)


if __name__ == "__main__":
    main()
