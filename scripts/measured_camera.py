#!/usr/bin/env python3
"""The measured rig camera for every episode (host, stdlib only; see pipeline/recipes/truecam.md).

Every stage so far uses MoGe-2's camera (outputs/camera, scripts/camera_intrinsics.py): on FORM-HOI its focal is 3.4 %
short and its principal point is the image centre, 54 / 38 px from the real one (scripts/camera_truth_check.py). The four
rig cameras each have one calibration over all recording days (config/camera/rig_cameras.json). An episode whose
background matches one rig camera clearly (config/camera/episode_cameras.json from scripts/camera_identify.py:
corr >= 0.8 and >= 0.3 above the best other camera) takes that camera; any other takes the rig mean (every camera's
principal point lies within ~9 px of it). MODE tcp (default): the principal point only, MoGe's focal kept (the CD-H lever
on FORM-HOI; the depth trend already absorbs the focal); tc: the focal too.
Writes outputs/camera_measured/<episode>/intrinsics.json, outputs/alignment_measured/<episode>/alignment.json (a copy of
outputs/alignment_sam's with these intrinsics, for the contact solver: CONTACT_ALIGNMENT_DIR) and
outputs/camera_measured/k_new.json (scripts/true_cam_params.py --k_new).
Usage: measured_camera.py [--mode tcp|tc]
"""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(os.environ.get("TRACK1_ROOT") or Path(__file__).resolve().parents[1])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=("tcp", "tc"), default="tcp")
    args = ap.parse_args()
    rig = {k: v for k, v in json.loads((ROOT / "config/camera/rig_cameras.json").read_text()).items() if k != "note"}
    ids = json.loads((ROOT / "config/camera/episode_cameras.json").read_text())
    mean = {x: sum(c[x] for c in rig.values()) / len(rig) for x in ("fx", "cx", "cy")}
    kn, src = {}, {}
    for cam in sorted((ROOT / "outputs/camera").glob("episode_*")):
        tag = cam.name; k = json.loads((cam / "intrinsics.json").read_text()); r = ids.get(tag)
        ok = bool(r) and r["corr"] >= 0.8 and r["corr"] - r["corr_other"] >= 0.3
        m = rig[r["camera"]] if ok else mean
        k["cx"], k["cy"] = m["cx"], m["cy"]
        if args.mode == "tc":
            k["fx"] = k["fy"] = m["fx"]
        kn[tag] = k; s = r["camera"] if ok else "rig mean"; src[s] = src.get(s, 0) + 1
        (ROOT / "outputs/camera_measured" / tag).mkdir(parents=True, exist_ok=True)
        (ROOT / "outputs/camera_measured" / tag / "intrinsics.json").write_text(json.dumps(k, indent=2))
        al = json.loads((ROOT / "outputs/alignment_sam" / tag / "alignment.json").read_text()); al["intrinsics"] = k
        al["note"] = al.get("note", "") + f"; measured rig camera ({args.mode})"
        (ROOT / "outputs/alignment_measured" / tag).mkdir(parents=True, exist_ok=True)
        (ROOT / "outputs/alignment_measured" / tag / "alignment.json").write_text(json.dumps(al))
    (ROOT / "outputs/camera_measured/k_new.json").write_text(json.dumps(kn, indent=1))
    print(f"PASS: {len(kn)} episodes on the measured camera ({args.mode}); sources {src}")


if __name__ == "__main__":
    main()
