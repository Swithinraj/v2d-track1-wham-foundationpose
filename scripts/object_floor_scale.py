#!/usr/bin/env python3
"""Per-object depth from floor contact: scale the object about the camera onto the person's floor (host, CPU).

An object lying on the floor cannot be below it, yet objects sit several centimetres below the floor the person stands
on. Gravity kappa (the largest CD-O gain so far) fits ONE scale per episode from MoGe's floor. Here each object answers for
itself. Scaling translation and object scale about the camera by kappa leaves every pixel unchanged, so the masks stay
matched and this needs no image check. Rotation and the mesh are untouched, and ACC-O changes only by the factor kappa.

Evidence, strongest first:
  rest         the object is nearly still with its lowest point near the floor: that level should be the floor;
  below floor  it is never still there but its lowest point goes below the floor: its lowest level should be the floor.
An object that never comes near the floor (on a table, or held) abstains and keeps its scale.

The floor level is the person's own stance level at the first scored frame, in the same camera frame, because the scorer
places the object through a similarity fitted on the person there.

Usage: object_floor_scale.py --input DIR --planes JSON --levels FLOOR_ROOT_REPORT --out DIR
Writes OUT/episode_*.npz (+ meshes and the other files of --input) and object_floor_scale_report.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np

NEAR_FLOOR = 0.15      # m: how far from the floor an object can be at rest and still count as resting on it
STILL = 0.005          # m per frame
MIN_REST_FRAMES = 8
PLATEAU = 0.02         # m above the lowest rest level
MAX_CHANGE = 0.15      # a larger scale change is not a placement error this could plausibly explain


def floor_scale(vertices, rotation, translation, scale, normal, floor):
    """kappa moving the object's resting level onto floor = normal . X, and a report. kappa = 1 when rejected.

    vertices [V,3] mesh; rotation [T,3,3]; translation [T,3]; scale; normal [3] pointing up (camera at the origin, so the
    floor level is negative); floor the target level. Only scored frames should be passed.
    """
    vertices, rotation, translation, normal = (np.asarray(a, float) for a in (vertices, rotation, translation, normal))
    if not floor < 0:
        raise ValueError("the camera is above the floor, so the floor level must be negative")
    normal = normal / np.linalg.norm(normal)
    local = np.einsum("tij,pj->tpi", rotation, vertices) * scale + translation[:, None]
    level = (local @ normal).min(1)
    speed = np.zeros(len(translation))
    if len(translation) > 2:
        speed[1:-1] = np.linalg.norm(translation[2:] - translation[:-2], axis=1) / 2
        speed[0], speed[-1] = speed[1], speed[-2]
    report = dict(accepted=False, level_min_above_floor_m=float(level.min() - floor))
    rest = (speed < STILL) & (np.abs(level - floor) < NEAR_FLOOR)
    report["rest_frames"] = int(rest.sum())
    if rest.sum() >= MIN_REST_FRAMES:
        # A held object can also be still near the floor. The lowest plateau is the one standing on it.
        lowest = np.percentile(level[rest], 5)
        target, evidence = float(np.median(level[rest][level[rest] < lowest + PLATEAU])), "rest"
    elif level.min() < floor:
        target, evidence = float(np.percentile(level, 2)), "below floor"
    else:
        report["reason"] = "the object does not touch the floor in the scored frames"
        return 1., report
    kappa = floor / target
    report.update(kappa_proposed=kappa, evidence=evidence, level_above_floor_m=target - floor)
    if abs(kappa - 1) > MAX_CHANGE:
        report["reason"] = "implausible scale"
        return 1., report
    report.update(accepted=True, reason="floor contact")
    return kappa, report


def process_directory(base, out, normals, floors, scored, *, fixed_kappa=None, abstain_kappa=None):
    """fixed_kappa: the same scale for every object, ignoring floor evidence (a global dose experiment).
    abstain_kappa: the scale for objects without floor evidence (default: unchanged)."""
    import trimesh
    base, out = Path(base), Path(out)
    if out.exists():
        raise ValueError(f"{out} exists; refusing to overwrite an episode directory")
    shutil.copytree(base, out)
    report = {}
    for path in sorted(base.glob("episode_??????.npz")):
        tag = path.stem
        frames = np.asarray(scored[str(int(tag.split("_")[1]))])
        with np.load(path) as archive:
            data = dict(archive)
        mesh = trimesh.load(base / f"{tag}_object.glb", force="mesh", process=False)
        if tag not in floors:
            report[tag] = dict(accepted=False, kappa=1., rest_frames=0, level_min_above_floor_m=float("nan"),
                               reason="no accepted floor level for the person")
            continue
        kappa, entry = floor_scale(mesh.vertices, data["object_rotation"][frames], data["object_translation"][frames],
                                   float(data["object_scale"]), normals[tag], floors[tag])
        if fixed_kappa is not None:
            kappa, entry["evidence"] = fixed_kappa, "fixed"
        elif abstain_kappa is not None and not entry["accepted"]:
            kappa, entry["evidence"] = abstain_kappa, "abstain default"
        entry["kappa"] = kappa
        report[tag] = entry
        if kappa != 1.:
            data["object_translation"] = (data["object_translation"].astype(np.float64) * kappa).astype(data["object_translation"].dtype)
            data["object_scale"] = np.asarray(float(data["object_scale"]) * kappa, data["object_scale"].dtype)
            np.savez(out / path.name, **data)
    (out / "object_floor_scale_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--planes", type=Path, required=True)
    parser.add_argument("--levels", type=Path, required=True, help="floor_root_report.json (each episode's anchor_height_m)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scored", type=Path, default=Path("outputs/hand_fix/episodes_repro_depth_refine_meshes/scored_frames.json"))
    parser.add_argument("--fixed_kappa", type=float, help="one scale for every object (global experiment)")
    parser.add_argument("--abstain_kappa", type=float, help="scale for objects without floor evidence")
    parser.add_argument("--floor_offset", type=float, default=0., help="m the real floor lies below the person's lowest vertex")
    args = parser.parse_args()
    planes = json.loads(args.planes.read_text())
    levels = json.loads(args.levels.read_text())
    floors = {tag: entry["anchor_height_m"] - args.floor_offset for tag, entry in levels.items() if entry.get("accepted")}
    normals = {tag: entry["normal"] for tag, entry in planes.items()}
    report = process_directory(args.input, args.out, normals, floors, json.loads(args.scored.read_text()),
                               fixed_kappa=args.fixed_kappa, abstain_kappa=args.abstain_kappa)
    for tag, entry in report.items():
        print(tag, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in entry.items()}), flush=True)
    print(f"PASS: {sum(r['accepted'] for r in report.values())}/{len(report)} objects scaled -> {args.out}")


if __name__ == "__main__":
    main()
