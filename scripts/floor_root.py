#!/usr/bin/env python3
"""The human's root depth from floor contact (submission venv, CPU; before the contact stage). No learned parameters.

The scorer aligns once, at the first scored frame, so what counts is how the person moves relative to that frame. A foot
in stance is on the floor, and the floor is one plane in this static camera. If the stance foot's height above that plane
changes, the body has slid along the camera ray: sliding by d changes the height by d * (n . ray). So the height change
gives the slide, from evidence independent of the depth networks.

Earlier notes rejected a hard floor constraint because 1 cm of ABSOLUTE floor height is about 2.4 cm of depth. This uses
no absolute floor. It compares each frame's stance height with the anchor's, smoothed over time, so the first scored
frame is unchanged and the scorer's alignment, and therefore CD-O, stays exactly the same.

The slide runs along the ray through the stance foot, which keeps the foot's pixel. A foot is in stance when it is near the
local lower envelope of both feet and barely moving. Rejected episodes keep their input unchanged.

Usage: floor_root.py --input DIR --bodies DIR --planes JSON --out DIR [--lam 1e4] [--beta 1]
  --bodies: floor_bodies.py output for --input; --planes: {episode: {normal}} with the floor normal in --input's camera,
  pointing up. Writes OUT/episode_*.npz (+ the other files of --input) and floor_root_report.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.linalg import solveh_banded

FLIP = np.array([1.0, -1.0, -1.0])     # MHR root translation axes vs the Track 1 frame; pose[:, :3] is in decimetres
ENVELOPE_FRAMES = 15                   # half window of the lower envelope: every stride touches the floor within it
FOOT_NEAR = 0.03                       # m above the lower envelope
FOOT_STILL = 0.012                     # m per frame
TRUST_RAMP = 30.                       # frames over which a blended reference height ramps in around the anchor


def whittaker(values, weights, lam):
    """argmin sum w (x - y)^2 + lam |D2 x|^2 for [T] or [T, K] values; pentadiagonal, so long episodes stay cheap."""
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    T = len(values)
    if T < 3:
        return values.copy()
    band = np.zeros((3, T))
    band[2] = weights + lam * np.convolve(np.ones(T - 2), [1., 4., 1.])
    band[1, 1:] = lam * -2 * np.convolve(np.ones(T - 2), [1., 1.])
    band[0, 2:] = lam
    right = weights[:, None] * values if values.ndim == 2 else weights * values
    return solveh_banded(band, right)


def floor_root_shift(feet, ray, normal, confidence, anchor, *, lam=1e3, max_shift=.4, min_coverage=.2,
                     min_incidence=.1, anchor_window=0, anchor_trust=1.):
    """Per-frame body displacement [T,3] that keeps stance feet at the anchor's floor height, and a report.

    feet [T,K,3] lowest point per foot, ray [T,3] unit slide direction, normal [3] floor normal, confidence [T,K] stance
    evidence (0 = not on the floor), anchor the first scored frame. The anchor's displacement is exactly zero. An
    unsupported or implausible correction is rejected, never clipped, and returns zeros.

    anchor_trust below 1 blends the reference height toward the episode's median stance height, because noise in one
    frame's height otherwise shifts every other frame. The anchor still does not move: the difference ramps in over
    TRUST_RAMP frames on either side of it.
    """
    feet, ray, normal, confidence = (np.asarray(a, float) for a in (feet, ray, normal, confidence))
    if feet.ndim != 3 or feet.shape[2] != 3 or ray.shape != (len(feet), 3) or confidence.shape != feet.shape[:2]:
        raise ValueError("feet [T,K,3], ray [T,3] and confidence [T,K] must agree")
    if normal.shape != (3,) or not np.isfinite(normal).all() or np.linalg.norm(normal) < 1e-8:
        raise ValueError("the floor normal must be a finite nonzero vector")
    if not (np.isfinite(feet).all() and np.isfinite(ray).all() and np.isfinite(confidence).all()):
        raise ValueError("feet, ray and confidence must be finite")
    if not 0 <= anchor < len(feet):
        raise ValueError("anchor is outside the sequence")
    normal = normal / np.linalg.norm(normal)
    T = len(feet)
    zero = np.zeros((T, 3))
    heights = np.where(confidence > 0, feet @ normal, np.inf)
    measured = heights.min(1)
    weights = np.where(np.isfinite(measured), np.max(confidence, axis=1), 0.)
    report = dict(accepted=False, translation_max_m=0., coverage=float((weights > 0).mean()))
    if report["coverage"] < min_coverage or not weights.any():
        report["reason"] = "insufficient floor contact"
        return zero, report
    low, high = max(0, anchor - anchor_window), min(T, anchor + anchor_window + 1)
    if not weights[low:high].any():
        report["reason"] = "the anchor has no floor contact to compare with"
        return zero, report
    incidence = ray @ normal
    report["incidence_median"] = float(np.median(np.abs(incidence)))
    if np.abs(incidence).min() < min_incidence:
        report["reason"] = "the slide direction is not observable from floor height"
        return zero, report
    reference = measured[weights > 0].mean()
    smooth = whittaker(np.where(weights > 0, measured - reference, 0.), weights, lam)
    slide = (smooth[anchor] - smooth) / incidence
    if anchor_trust != 1.:
        stance_level = np.median(smooth[weights > 0])
        ramp = np.minimum(np.abs(np.arange(T) - anchor) / TRUST_RAMP, 1.)
        slide += ramp * (1. - anchor_trust) * (stance_level - smooth[anchor]) / incidence
    slide[anchor] = 0.
    report["translation_max_m"] = float(np.abs(slide).max())
    report["translation_rms_m"] = float(np.sqrt((slide ** 2).mean()))
    report["anchor_height_m"] = float(smooth[anchor] + reference)
    if report["translation_max_m"] > max_shift:
        report["reason"] = "implausible correction"
        return zero, report
    report.update(accepted=True, reason="floor contact")
    return slide[:, None] * ray, report


def lower_envelope(values, half):
    out = np.empty(len(values))
    for t in range(len(values)):
        out[t] = values[max(0, t - half):t + half + 1].min()
    return out


def stance_feet(left, right, normal, scored):
    """Lowest point per foot [T,2,3] and stance confidence [T,2], from foot vertices [T,F,3]; unscored frames carry 0."""
    T = len(left)
    points, confidence = np.zeros((T, 2, 3)), np.zeros((T, 2))
    heights = []
    for k, vertices in enumerate((left, right)):
        h = vertices @ normal
        low = h.argmin(1)
        points[:, k] = vertices[np.arange(T), low]
        heights.append(h[np.arange(T), low])
    heights = np.stack(heights, 1)
    mask = np.zeros(T, bool)
    mask[scored] = True
    envelope = lower_envelope(np.where(mask, heights.min(1), np.inf), ENVELOPE_FRAMES)
    for k, vertices in enumerate((left, right)):
        center = vertices.mean(1)
        speed = np.zeros(T)
        speed[1:-1] = np.linalg.norm(center[2:] - center[:-2], axis=1) / 2
        speed[0], speed[-1] = speed[1], speed[-2]
        confidence[:, k] = (heights[:, k] - envelope < FOOT_NEAR) & (speed < FOOT_STILL) & mask
    return points, confidence


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--bodies", type=Path, required=True)
    parser.add_argument("--planes", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scored", type=Path, default=Path("outputs/hand_fix/episodes_repro_depth_refine_meshes/scored_frames.json"))
    parser.add_argument("--lam", type=float, default=1e4)
    parser.add_argument("--beta", type=float, default=1.0, help="fraction of the correction applied")
    parser.add_argument("--anchor_trust", type=float, default=1.0, help="weight of the anchor's height against the median stance height")
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; refusing to overwrite an episode directory")
    planes, scored = json.loads(args.planes.read_text()), json.loads(args.scored.read_text())
    shutil.copytree(args.input, args.out)
    report = {}
    for path in sorted(args.input.glob("episode_??????.npz")):
        tag, episode = path.stem, int(path.stem.split("_")[1])
        frames = np.asarray(scored[str(episode)])
        with np.load(args.bodies / path.name) as body:
            left, right = body["left_foot"].astype(float), body["right_foot"].astype(float)
        normal = np.asarray(planes[tag]["normal"], float)
        points, confidence = stance_feet(left, right, normal, frames)
        # The ray through the stance foot: the foot's pixel stays, and the ray varies slowly.
        stance = np.argmin(np.where(confidence > 0, points @ normal, np.inf), axis=1)
        weights = (confidence.max(1) > 0).astype(float)
        foot = whittaker(np.where(weights[:, None] > 0, points[np.arange(len(points)), stance], 0.), weights + 1e-9, 100.)
        ray = foot / np.linalg.norm(foot, axis=1, keepdims=True)
        # Coverage counts scored frames: unscored ones carry no evidence by construction.
        delta, entry = floor_root_shift(points, ray, normal, confidence, int(frames[0]), lam=args.lam,
                                        min_coverage=.2 * len(frames) / len(points), anchor_window=ENVELOPE_FRAMES,
                                        anchor_trust=args.anchor_trust)
        report[tag] = entry
        if not entry["accepted"]:
            print(tag, json.dumps(entry), flush=True)
            continue
        with np.load(path) as archive:
            data = dict(archive)
        pose = data["pose"].astype(np.float64)
        pose[:, :3] += FLIP * args.beta * delta * 10.0
        data["pose"] = pose.astype(data["pose"].dtype)
        np.savez(args.out / path.name, **data)
        print(tag, json.dumps(entry), flush=True)
    (args.out / "floor_root_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"PASS: floor root, {sum(r['accepted'] for r in report.values())}/{len(report)} episodes -> {args.out}")


if __name__ == "__main__":
    main()
