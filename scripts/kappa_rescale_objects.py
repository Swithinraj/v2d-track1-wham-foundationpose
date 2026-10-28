"""Re-express the scored objects for a different full-mode human depth trend (host venv: numpy).

The objects' scale comes from the gravity kappa (kappa_gravity.py) computed on the full-mode human after its depth
trend (ipf30.md). When the trend changes, kappa is recomputed on the new human (NEW) and on the human the objects were
built with (OLD), and every episode's objects are re-scaled about the camera by kappa_new / kappa_old, translation and
scale, which is what kappa_gravity.py itself applies.
Usage: kappa_rescale_objects.py --objects DIR --kappa_new DIR --kappa_old DIR --out DIR
  (--kappa_*: kappa_gravity.py output directories holding kappa_gravity_report.json)
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", type=Path, required=True)
    ap.add_argument("--kappa_new", type=Path, required=True)
    ap.add_argument("--kappa_old", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    k_new = json.loads((args.kappa_new / "kappa_gravity_report.json").read_text())["episodes"]
    k_old = json.loads((args.kappa_old / "kappa_gravity_report.json").read_text())["episodes"]
    if args.out.exists():
        shutil.rmtree(args.out)
    shutil.copytree(args.objects, args.out)
    report = {}
    for path in sorted(args.out.glob("episode_*.npz")):
        tag = path.stem
        r = k_new[tag]["kappa"] / k_old[tag]["kappa"]
        d = dict(np.load(path))
        d["object_translation"] = (d["object_translation"].astype(np.float64) * r).astype(np.float32)
        d["object_scale"] = np.float32(float(d["object_scale"]) * r)
        np.savez(path, **d)
        report[tag] = {"kappa_new": k_new[tag]["kappa"], "kappa_old": k_old[tag]["kappa"], "ratio": r}
    if len(report) != len(k_old):
        raise SystemExit(f"{len(report)} episodes re-scaled, {len(k_old)} in the kappa report")
    (args.out / "kappa_rescale.json").write_text(json.dumps(report, indent=1))
    rs = np.array([v["ratio"] for v in report.values()])
    print(f"PASS: {len(rs)} episodes' objects re-scaled by kappa_new / kappa_old: median x{np.median(rs):.4f}, "
          f"range {rs.min():.4f}-{rs.max():.4f} -> {args.out}")


if __name__ == "__main__":
    main()
