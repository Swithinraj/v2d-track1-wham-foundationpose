#!/usr/bin/env python3
"""Camera check against the FORM-HOI calibration (10-29 formula audit). Our camera (outputs/camera/<tag>/intrinsics.json)
is the median of MoGe-2's per-frame guesses, and MoGe always puts the principal point at the image centre. FORM-HOI's
edex holds the real calibration of every recording camera (CARI4D prep/mhr_export_utils.py: left cameras
front / back / right / left = edex cameras 0 / 2 / 6 / 4; K = focal, principal). Per episode of each root: true fx, cx, cy
vs ours, and whether the true calibration is the same across recordings (one physical camera per name). Also the
challenge cameras (our MoGe medians) for comparison. Usage: camera_truth_check.py FORMHOI_DIR CHALLENGE_ROOT
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

IDX = {"front_stereo_camera_left": 0, "back_stereo_camera_left": 2, "right_stereo_camera_left": 6, "left_stereo_camera_left": 4}
F, C = Path(sys.argv[1]), Path(sys.argv[2])
rows, by_cam = [], defaultdict(set)
for root in sorted(F.glob("root*")):
    meta = root / "data/track_1/meta/episodes_metadata.jsonl"
    if root.is_symlink() or not meta.exists() or not (root / "gt").exists():
        continue
    for m in map(json.loads, filter(str.strip, meta.read_text().splitlines())):
        tag = f"episode_{m['episode_index']:06d}"; ed, cam = root / "gt" / tag / "edex", root / "outputs/camera" / tag / "intrinsics.json"
        if not ed.exists() or not cam.exists() or m.get("camera") not in IDX:
            continue
        e = json.loads(ed.read_text())[0]["cameras"][IDX[m["camera"]]]["intrinsics"]
        o = json.loads(cam.read_text()); f, p = e["focal"], e["principal"]
        rows.append((root.name, m["camera"], f[0], p[0], p[1], o["fx"], o["cx"], o["cy"], o["width"]))
        by_cam[m["camera"]].add((round(f[0], 1), round(p[0], 1), round(p[1], 1), m["sequence_id"][:10]))
a = np.array([r[2:] for r in rows], float)
print(f"FORM-HOI episodes with calibration: {len(rows)} ({len(set(r[0] for r in rows))} roots)")
print(f"  focal: true median {np.median(a[:, 0]):.1f} (range {a[:, 0].min():.1f}-{a[:, 0].max():.1f}), ours (MoGe median) "
      f"{np.median(a[:, 3]):.1f}; ours / true median {np.median(a[:, 3] / a[:, 0]):.4f} (5-95 % {np.percentile(a[:, 3] / a[:, 0], 5):.4f}-"
      f"{np.percentile(a[:, 3] / a[:, 0], 95):.4f})")
print(f"  principal x: true median {np.median(a[:, 1]):.1f} (range {a[:, 1].min():.1f}-{a[:, 1].max():.1f}), ours {np.median(a[:, 4]):.1f}; "
      f"offset ours - true median {np.median(a[:, 4] - a[:, 1]):+.1f} px")
print(f"  principal y: true median {np.median(a[:, 2]):.1f} (range {a[:, 2].min():.1f}-{a[:, 2].max():.1f}), ours {np.median(a[:, 5]):.1f}; "
      f"offset ours - true median {np.median(a[:, 5] - a[:, 2]):+.1f} px")
print(f"  angle of the principal-point error: median {np.degrees(np.median(np.hypot(a[:, 4] - a[:, 1], a[:, 5] - a[:, 2]) / a[:, 0])):.2f} deg")
for cam, s in sorted(by_cam.items()):
    ks = sorted(set(x[:3] for x in s)); days = sorted(set(x[3] for x in s))
    print(f"  {cam}: {len(ks)} distinct calibrations over {len(days)} recording days; focal {sorted(set(k[0] for k in ks))[:6]}"
          f" principal {[k[1:] for k in ks][:4]}")
ch = [json.loads(p.read_text()) for p in sorted((C / "outputs/camera").glob("episode_*/intrinsics.json"))]
if ch:
    fx = np.array([c["fx"] for c in ch])
    print(f"challenge: {len(ch)} episodes, resolution {ch[0]['width']:.0f}x{ch[0]['height']:.0f}, our fx median {np.median(fx):.1f} "
          f"(range {fx.min():.1f}-{fx.max():.1f}), principal ({ch[0]['cx']:.0f}, {ch[0]['cy']:.0f})")
