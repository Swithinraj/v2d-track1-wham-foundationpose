#!/usr/bin/env python3
"""Gate step-3 object depth corrections per episode by two checks: silhouette IoU on grasp frames does not drop
(independent evidence) and the added object acceleration on scored frames stays <= MAX_DACC cm/frame^2 (ACC-O guard).
Assembles BASE with the accepted episodes' object translations from the step-3 output.
Usage: held_object_depth_gate.py BASE_DIR STEP3_DIR OUT_DIR [MAX_DACC]"""
import json, shutil, sys
from pathlib import Path
import numpy as np
base, s3, out = map(Path, sys.argv[1:4]); MAX = float(sys.argv[4]) if len(sys.argv) > 4 else 0.003
rep = json.loads((s3 / "object_depth_from_contact_report.json").read_text())
if out.exists():
    shutil.rmtree(out)
shutil.copytree(base, out, ignore=shutil.ignore_patterns("*report*.json"))
acc = []
for tag, r in sorted(rep.items()):
    if not r.get("grasp_frames"):
        continue
    ok = r["iou_grasp_frames"][1] >= r["iou_grasp_frames"][0] and r["obj_accel_cm"][1] - r["obj_accel_cm"][0] <= MAX
    if ok:
        d = dict(np.load(out / f"{tag}.npz")); d["object_translation"] = np.load(s3 / f"{tag}.npz")["object_translation"]
        np.savez(out / f"{tag}.npz", **d); acc.append(tag[-2:])
print("accepted episodes:", acc, "->", out)
(out / "odc_gate.json").write_text(json.dumps({"max_dacc": MAX, "accepted": acc}, indent=1))
