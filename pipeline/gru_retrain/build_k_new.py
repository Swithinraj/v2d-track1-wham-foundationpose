#!/usr/bin/env python3
"""Build the measured-camera k_new for a FORM-HOI training group (host, stdlib only).

Unlike the challenge episodes (whose camera must be inferred and passes an acceptance
threshold), the training groups' camera is KNOWN: episodes_metadata.jsonl labels every
recording with its rig camera. So every episode gets its true camera's principal point,
with the rig mean only as fallback for a missing label. Mode tcp (default, matching
TRUECAM_MODE in truecam_final.sh) keeps MoGe's focal and replaces the principal point;
tc replaces the focal too.

Usage: build_k_new.py --camera DIR --meta episodes_metadata.jsonl [--mode tcp|tc]
  --camera  the group root's outputs/camera (per-episode MoGe-2 intrinsics.json: fx/fy/cx/cy)
  --meta    the group's data/track_1/meta/episodes_metadata.jsonl (episode_index, camera label)
Writes k_new.json next to --camera (k_new_tcp.json / k_new_tc.json): {episode_XXXXXX: {fx, cx, cy}}.
"""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--camera", type=Path, required=True)
    ap.add_argument("--meta", type=Path, required=True)
    ap.add_argument("--rig", type=Path, default=ROOT / "config/camera/rig_cameras.json")
    ap.add_argument("--mode", choices=("tcp", "tc"), default="tcp")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    rig = {k: v for k, v in json.loads(a.rig.read_text()).items() if k != "note"}
    mean = {x: sum(c[x] for c in rig.values()) / len(rig) for x in ("fx", "cx", "cy")}
    label = {}
    for line in a.meta.read_text().splitlines():
        row = json.loads(line) if line.strip() else {}
        if "episode_index" in row and "camera" in row:
            label[row["episode_index"]] = row["camera"]
    kn, src = {}, {}
    for f in sorted(a.camera.glob("episode_*/intrinsics.json")):
        k = json.loads(f.read_text())
        cam = label.get(int(f.parent.name.split("_")[1]))
        m = rig.get(cam, mean)
        k["cx"], k["cy"] = m["cx"], m["cy"]
        if a.mode == "tc":
            k["fx"] = k["fy"] = m["fx"]
        kn[f.parent.name] = {"fx": k["fx"], "cx": k["cx"], "cy": k["cy"]}
        src[cam if cam in rig else "rig mean"] = src.get(cam if cam in rig else "rig mean", 0) + 1
    out = a.out or (a.camera.parent / f"k_new_{a.mode}.json")
    out.write_text(json.dumps(kn, indent=1))
    print(f"PASS: {len(kn)} episodes -> {out}; cameras {src}"
          + ("" if all(c in rig for c in label.values()) else " (rig-mean fallback used for some)"))


if __name__ == "__main__":
    main()
