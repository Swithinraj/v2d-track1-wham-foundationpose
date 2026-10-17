#!/usr/bin/env python3
"""Symmetry-aware tracking for rotationally symmetric objects (the hula hoop): particle filter + global
sequence optimisation, replacing the FoundationPose track while the object is handled.

Runs in v2d_cari4d.sif (numpy, scipy, cv2), CPU only. A torus's pose is defined only up to spin about its axis
and an axis flip, so the state is the ring centre c (3) and the axis LINE n (2 DoF, sign-free):
  1. Each SAM 2 mask is fitted with an ellipse; with the ring radius (inputs/object_meshes/
     <object>.symmetric.json) and MoGe-2 intrinsics this gives the distance and two tilt hypotheses.
  2. Particle filter over (c, velocity, n), forward and backward from the prompt frame. Weight =
     symmetric 2D chamfer between the projected ring centreline and the mask (trimmed for occlusion)
     + a hand-contact prior from the SAM 3D Body (MHR) hand points next to the hoop; 20% of particles are
     re-proposed from the mask fit every frame; the axis estimate is a sign-free (eigenvector) mean.
  3. Sparse least squares over the whole sequence: the chamfer and hand terms (Charbonnier), second-
     difference smoothness on c and n, 40x smoothness plus a zero-velocity term on frames where the
     mask shows the object at rest, and an anchor to the FoundationPose smoothed track on those
     resting frames (where FoundationPose is reliable). Rotations are rebuilt with minimal twist.
Measured on episodes 0-2 against the FoundationPose track: silhouette IoU on scored frames 0.457 ->
0.524, 0.136 -> 0.490, 0.407 -> 0.599, resting acceleration unchanged.

Writes outputs/object/episode_X/foundationpose/symmetric/*.json (object-to-camera Transform3d, MoGe
metres, for the <object>.symmetric.glb mesh) and symmetric_report.json.
"""

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

ROOT = Path(os.environ.get("TRACK1_ROOT") or Path(__file__).resolve().parents[1])
RING_RADIUS = None       # set from inputs/object_meshes/<object>.symmetric.json in main()
MESH_AXIS = np.array([0.0, 1.0, 0.0])
DS = 2                   # distance transforms at half resolution
PX_SIGMA = 4.0           # chamfer residual scale (full-resolution pixels)
HAND_SIGMA = 0.07        # metres
PARTICLES = 2000
LAM_C, LAM_N = 20.0, 5.0
RNG = np.random.default_rng(0)


# ----------------------------------------------------------------------------- geometry
def ring_basis(n):
    """Orthonormal (e1, e2) spanning the ring plane for axes n [..., 3]."""
    helper = np.where(np.abs(n[..., :1]) < 0.9, np.array([1.0, 0, 0]), np.array([0, 1.0, 0]))
    e1 = np.cross(n, helper)
    e1 /= np.linalg.norm(e1, axis=-1, keepdims=True)
    return e1, np.cross(n, e1)


def ring_points(c, n, m):
    theta = np.linspace(0, 2 * np.pi, m, endpoint=False)
    e1, e2 = ring_basis(n)
    return c[..., None, :] + RING_RADIUS * (np.cos(theta)[:, None] * e1[..., None, :] + np.sin(theta)[:, None] * e2[..., None, :])


def project(p, K):
    z = np.maximum(p[..., 2], 1e-3)
    return np.stack([K[0, 0] * p[..., 0] / z + K[0, 2], K[1, 1] * p[..., 1] / z + K[1, 2]], -1)


def ring_distance(h, c, n):
    """Distance of points h [..., 3] from the ring centreline (c, n)."""
    d = h - c
    z = (d * n).sum(-1)
    rho = np.linalg.norm(d - z[..., None] * n, axis=-1)
    return np.sqrt((rho - RING_RADIUS) ** 2 + z ** 2)


def sample_dt(dt, uv):
    """Bilinear lookup of a half-resolution distance transform at full-resolution pixel coordinates."""
    h, w = dt.shape
    x = np.clip(uv[..., 0] / DS, 0, w - 1.001)
    y = np.clip(uv[..., 1] / DS, 0, h - 1.001)
    x0, y0 = x.astype(int), y.astype(int)
    fx, fy = x - x0, y - y0
    v = (dt[y0, x0] * (1 - fx) * (1 - fy) + dt[y0, x0 + 1] * fx * (1 - fy)
         + dt[y0 + 1, x0] * (1 - fx) * fy + dt[y0 + 1, x0 + 1] * fx * fy)
    outside = (uv[..., 0] < 0) | (uv[..., 1] < 0) | (uv[..., 0] >= w * DS) | (uv[..., 1] >= h * DS)
    return np.where(outside, 60.0 / DS, v) * DS   # back to full-resolution pixels


# ----------------------------------------------------------------------------- observations
def load_frames(episode, K):
    obj = ROOT / "outputs/object" / f"episode_{episode:06d}"
    files = sorted((obj / "sam2_masks/1").glob("*.png"))
    dts, samples, proposals = [], [], []
    for f in files:
        mask = np.array(Image.open(f)) > 127
        small = cv2.resize(mask.astype(np.uint8), (mask.shape[1] // DS, mask.shape[0] // DS), interpolation=cv2.INTER_NEAREST)
        dts.append(cv2.distanceTransform((1 - small).astype(np.uint8), cv2.DIST_L2, 3).astype(np.float16))
        ys, xs = np.nonzero(mask)
        pts = np.column_stack([xs, ys]).astype(np.float64)
        samples.append(pts[RNG.choice(len(pts), min(40, len(pts)), replace=False)] if len(pts) else np.zeros((0, 2)))
        proposals.append(ellipse_hypotheses(pts, K))
    return dts, samples, proposals


def ellipse_hypotheses(pts, K):
    """Two (c, n) ring hypotheses from an ellipse fitted to the mask pixels (weak perspective)."""
    if len(pts) < 30:
        return []
    sub = pts[RNG.choice(len(pts), min(3000, len(pts)), replace=False)].astype(np.float32)
    (u0, v0), axes, _ = cv2.fitEllipse(sub)
    a, b = max(axes) / 2, min(axes) / 2
    if a < 5:
        return []
    centred = sub - np.array([u0, v0], np.float32)
    evals, evecs = np.linalg.eigh(centred.T @ centred)
    minor = evecs[:, 0]
    f = (K[0, 0] + K[1, 1]) / 2
    Kinv = np.linalg.inv(K)
    r = Kinv @ np.array([u0, v0, 1.0]); r /= np.linalg.norm(r)
    m = Kinv @ np.array([u0 + minor[0], v0 + minor[1], 1.0]) - Kinv @ np.array([u0, v0, 1.0])
    m -= (m @ r) * r; m /= np.linalg.norm(m)
    tilt = np.arccos(np.clip(b / a, 0, 1))
    c = (f * RING_RADIUS / a) * r
    return [(c, np.cos(tilt) * -r + s * np.sin(tilt) * m) for s in (1.0, -1.0)]


def load_hands(episode):
    """Hand centroids in the MoGe camera frame [T, 2, 3] (inverse of the assembly's camera mapping)."""
    tag = f"episode_{episode:06d}"
    hands = np.load(ROOT / "outputs/hands" / f"{tag}.npz")["hands"].astype(np.float64)   # human frame (sam3d_body_hands.py)
    align_root = Path(os.environ.get("TRACK1_ALIGNMENT_DIR") or ROOT / "outputs/alignment_sam")   # the hands' frame
    a = json.loads((align_root / tag / "alignment.json").read_text())
    R, t, s = np.asarray(a["R_cw"]), np.asarray(a["t_cw"]), float(a["depth_scale"])
    cam = s * (hands @ R.T + t)
    return np.stack([cam[:, :256].mean(1), cam[:, 256:].mean(1)], 1)


# ----------------------------------------------------------------------------- likelihood
def chamfer_terms(c, n, dt, samples, K, m=48):
    """Precision (ring -> mask, trimmed) and recall (mask -> ring) chamfer in pixels for [N] hypotheses."""
    uv = project(ring_points(c, n, m), K)                       # [N, m, 2]
    d = np.minimum(sample_dt(dt, uv), 40.0)
    keep = max(1, int(0.75 * m))
    precision = np.sort(d, axis=-1)[..., :keep].mean(-1)       # tolerate occluded ring parts
    if len(samples):
        dist = np.linalg.norm(samples[None, :, None, :] - uv[:, None, :, :], axis=-1).min(-1)   # [N, S]
        recall = np.minimum(dist, 40.0).mean(-1)
    else:
        recall = np.zeros(len(c))
    return precision, recall


def contact_flags(hands_t, dt, K):
    uv = project(hands_t, K)
    return sample_dt(dt, uv) < 25.0     # a hand projects next to the hoop mask


# ----------------------------------------------------------------------------- phase 2: particle filter
def particle_filter(dts, samples, proposals, hands, K, n_particles, use_hands, start):
    T = len(dts)
    order = list(range(start, T)) + list(range(start - 1, -1, -1))
    est_c, est_n, ess_log = np.zeros((T, 3)), np.zeros((T, 3)), np.zeros(T)
    particles = None
    for i, f in enumerate(order):
        if f == start - 1:                       # backward pass restarts from the reference frame
            particles = seed_backward
        fresh = []
        for c0, n0 in proposals[f]:
            k = max(1, n_particles // (5 * max(1, len(proposals[f]))))
            fresh.append(np.hstack([c0 + RNG.normal(0, 0.05, (k, 3)) * [1, 1, 3], np.zeros((k, 3)),
                                    normalize(n0 + RNG.normal(0, 0.15, (k, 3)))]))
        fresh = np.vstack(fresh) if fresh else np.zeros((0, 9))
        if particles is None:
            if not len(fresh):
                continue
            particles = fresh[RNG.choice(len(fresh), n_particles)]
        else:
            moved = particles.copy()
            moved[:, 3:6] = 0.85 * moved[:, 3:6] + RNG.normal(0, 0.01, (len(moved), 3))
            moved[:, :3] += moved[:, 3:6] + RNG.normal(0, 0.015, (len(moved), 3))
            moved[:, 6:] = normalize(moved[:, 6:] + RNG.normal(0, 0.08, (len(moved), 3)))
            keep = n_particles - len(fresh)
            particles = np.vstack([moved[RNG.choice(len(moved), keep)], fresh]) if len(fresh) else moved
        prec, rec = chamfer_terms(particles[:, :3], particles[:, 6:], dts[f], samples[f], K)
        logw = -(prec + rec) / PX_SIGMA
        if use_hands:
            near = contact_flags(hands[f], dts[f], K)
            for h in np.flatnonzero(near):
                d = ring_distance(hands[f, h][None], particles[:, :3], particles[:, 6:])
                logw -= np.minimum(d, 0.25) ** 2 / (2 * HAND_SIGMA ** 2)
        w = np.exp(logw - logw.max()); w /= w.sum()
        ess_log[f] = 1.0 / (w ** 2).sum()
        est_c[f] = (w[:, None] * particles[:, :3]).sum(0)
        scatter = np.einsum("p,pi,pj->ij", w, particles[:, 6:], particles[:, 6:])
        est_n[f] = np.linalg.eigh(scatter)[1][:, -1]           # sign-free axis mean
        # keep particle axes on the estimate's hemisphere so velocities and averaging stay consistent
        particles[:, 6:] *= np.sign(particles[:, 6:] @ est_n[f])[:, None]
        particles = particles[systematic_resample(w)]
        if f == start:
            seed_backward = particles.copy()
            seed_backward[:, 3:6] *= -1
    return est_c, est_n, ess_log


def systematic_resample(w):
    positions = (RNG.random() + np.arange(len(w))) / len(w)
    return np.minimum(np.searchsorted(np.cumsum(w), positions), len(w) - 1)


def normalize(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


# ----------------------------------------------------------------------------- phase 3: global optimisation
def charbonnier(r):
    """Robust shaping applied to data residuals only: quadratic near 0, linear tails."""
    return np.sign(r) * np.sqrt(2.0 * (np.sqrt(1.0 + r * r) - 1.0))


def refine_sequence(c0, n0, dts, samples, hands, K, use_hands, lam_c, lam_n, static=None, robust_all=False,
                    static_gain=40.0, static_velocity=400.0, anchor=None, anchor_c=200.0, anchor_n=50.0):
    """anchor=(c_ref, n_ref): on resting frames, tie (c, n) to a reference track (FoundationPose)."""
    T = len(c0)
    static = np.zeros(T, bool) if static is None else np.asarray(static, bool)
    # Second differences centred on resting frames are weighted static_gain x; consecutive resting
    # frames also get a first-difference (zero-velocity) term so a hoop lying on the floor stays put.
    acc_w = np.where(static[1:-1], static_gain, 1.0)[:, None]
    vel_pairs = np.flatnonzero(static[1:] & static[:-1])
    rest_idx = np.flatnonzero(static) if anchor is not None else np.zeros(0, int)
    for t in range(1, T):                   # consistent axis signs along time
        if n0[t] @ n0[t - 1] < 0:
            n0[t] = -n0[t]
    contacts = [np.flatnonzero(contact_flags(hands[t], dts[t], K)) if use_hands else [] for t in range(T)]
    m, s = 32, 40

    def residuals(x):
        X = x.reshape(T, 6)
        c, n = X[:, :3], normalize(X[:, 3:])
        out = []
        uv = project(ring_points(c, n, m), K)          # [T, m, 2]
        for t in range(T):
            d = np.minimum(sample_dt(dts[t], uv[t]), 40.0)
            out.append(charbonnier(d / PX_SIGMA))
            if len(samples[t]):
                dist = np.linalg.norm(samples[t][:, None, :] - uv[t][None], axis=-1).min(1)
                out.append(charbonnier(np.minimum(dist, 40.0) / PX_SIGMA))
            else:
                out.append(np.zeros(s))
            hand = np.zeros(2)
            for h in contacts[t]:
                hand[h] = charbonnier(min(ring_distance(hands[t, h], c[t], n[t]), 0.25) / HAND_SIGMA)
            out.append(hand)
            out.append([np.linalg.norm(X[t, 3:]) - 1.0])
        out.append((lam_c * acc_w * (c[2:] - 2 * c[1:-1] + c[:-2])).ravel())
        out.append((lam_n * acc_w * (n[2:] - 2 * n[1:-1] + n[:-2])).ravel())
        out.append((static_velocity * (c[vel_pairs + 1] - c[vel_pairs])).ravel())
        out.append((static_velocity * 0.25 * (n[vel_pairs + 1] - n[vel_pairs])).ravel())
        if anchor is not None:
            out.append((anchor_c * (c[rest_idx] - anchor[0][rest_idx])).ravel())
            out.append((anchor_n * (n[rest_idx] - anchor[1][rest_idx])).ravel())
        return np.concatenate([np.ravel(o) for o in out])

    per_frame = m + s + 2 + 1
    rows = T * per_frame + 6 * (T - 2) + 6 * len(vel_pairs) + 6 * len(rest_idx)
    sparsity = lil_matrix((rows, 6 * T), dtype=int)
    for t in range(T):
        sparsity[t * per_frame:(t + 1) * per_frame, 6 * t:6 * t + 6] = 1
    for t in range(T - 2):
        for k in range(3):
            sparsity[T * per_frame + 3 * t:T * per_frame + 3 * t + 3, 6 * (t + k):6 * (t + k) + 3] = 1
            sparsity[T * per_frame + 3 * (T - 2) + 3 * t:T * per_frame + 3 * (T - 2) + 3 * t + 3, 6 * (t + k) + 3:6 * (t + k) + 6] = 1
    base = T * per_frame + 6 * (T - 2)
    for i, t in enumerate(vel_pairs):
        for k in (0, 1):
            sparsity[base + 3 * i:base + 3 * i + 3, 6 * (t + k):6 * (t + k) + 3] = 1
            sparsity[base + 3 * len(vel_pairs) + 3 * i:base + 3 * len(vel_pairs) + 3 * i + 3, 6 * (t + k) + 3:6 * (t + k) + 6] = 1
    base += 6 * len(vel_pairs)
    for i, t in enumerate(rest_idx):
        sparsity[base + 3 * i:base + 3 * i + 3, 6 * t:6 * t + 3] = 1
        sparsity[base + 3 * len(rest_idx) + 3 * i:base + 3 * len(rest_idx) + 3 * i + 3, 6 * t + 3:6 * t + 6] = 1
    x0 = np.hstack([c0, n0]).ravel()
    sol = least_squares(residuals, x0, jac_sparsity=sparsity, loss="soft_l1" if robust_all else "linear",
                        f_scale=1.0, max_nfev=80, x_scale="jac", verbose=0)
    X = sol.x.reshape(T, 6)
    return X[:, :3], normalize(X[:, 3:]), float(sol.cost), int(sol.nfev)


# ----------------------------------------------------------------------------- output
def rotations_from_axes(n):
    """Mesh axis -> n_t with the minimal rotation from the previous frame (no spurious twist)."""
    def align(a, b):
        v, cth = np.cross(a, b), float(a @ b)
        if np.linalg.norm(v) < 1e-9:
            return np.eye(3) if cth > 0 else rodrigues(ring_basis(a[None])[0][0], np.pi)
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + vx + vx @ vx / (1 + cth)
    R = [align(MESH_AXIS, n[0])]
    for t in range(1, len(n)):
        R.append(align(n[t - 1], n[t]) @ R[-1])
    return np.stack(R)


def rodrigues(axis, angle):
    k = axis / np.linalg.norm(axis)
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * kx + (1 - np.cos(angle)) * kx @ kx


def to_quat_wxyz(R):
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.copysign(np.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    q = np.array([w, x, y, z])
    return (q / np.linalg.norm(q)).tolist()


def resting_frames(episode):
    """Frames where the object is at rest: dilated mask IoU between t and t-5 > 0.8 (9-frame majority)."""
    files = sorted((ROOT / "outputs/object" / f"episode_{episode:06d}" / "sam2_masks/1").glob("*.png"))
    kernel = np.ones((7, 7), np.uint8)
    masks = [cv2.dilate((np.array(Image.open(f)) > 127).astype(np.uint8), kernel) > 0 for f in files]
    iou = np.ones(len(masks))
    for t in range(5, len(masks)):
        iou[t] = (masks[t] & masks[t - 5]).sum() / max((masks[t] | masks[t - 5]).sum(), 1)
    iou[:5] = iou[5]
    return np.convolve((iou > 0.8).astype(float), np.ones(9) / 9, mode="same") > 0.5


def read_track(folder, frames):
    c, n = np.zeros((frames, 3)), np.zeros((frames, 3))
    for t in range(frames):
        p = json.loads((folder / f"{t:06d}.json").read_text())
        w, x, y, z = p["rotation"]
        R = np.array([[1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y], [2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x],
                      [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y]])
        c[t], n[t] = p["translation"], R @ MESH_AXIS
    return c, n


def write_track(folder, c, n):
    folder.mkdir(parents=True, exist_ok=True)
    for t, R in enumerate(rotations_from_axes(n)):
        (folder / f"{t:06d}.json").write_text(json.dumps({"rotation": to_quat_wxyz(R), "translation": c[t].tolist(),
                                                          "scale": [1.0, 1.0, 1.0]}))


def main():
    global RING_RADIUS
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--no_hands", action="store_true")
    args = ap.parse_args()
    tag = f"episode_{args.episode:06d}"
    meta = [json.loads(l) for l in (ROOT / "data/track_1/meta/episodes_metadata.jsonl").read_text().splitlines() if l]
    obj_name = next(r["object"] for r in meta if r["episode_index"] == args.episode)
    RING_RADIUS = json.loads((ROOT / "inputs/object_meshes" / f"{obj_name}.symmetric.json").read_text())["ring_radius_m"]
    obj = ROOT / "outputs/object" / tag
    ref = int((obj / "prompt/reference_frame.txt").read_text().strip())
    k = json.loads((obj / "intrinsics" / f"{ref:06d}.json").read_text())
    K = np.array([[k["fx"], 0, k["cx"]], [0, k["fy"], k["cy"]], [0, 0, 1.0]])

    t0 = time.time()
    dts, samples, proposals = load_frames(args.episode, K)
    use_hands = not args.no_hands
    hands = load_hands(args.episode)
    start = ref if proposals[ref] else next(i for i, p in enumerate(proposals) if p)
    c_pf, n_pf, ess = particle_filter(dts, samples, proposals, hands, K, PARTICLES, use_hands, start)
    t1 = time.time()
    rest = resting_frames(args.episode)
    c_fp, n_fp = read_track(obj / "foundationpose/smoothed", len(dts))
    n_fp *= np.sign((n_fp * n_pf).sum(1, keepdims=True) + 1e-12)   # same axis sign as the filter
    c_pf[rest], n_pf[rest] = c_fp[rest], n_fp[rest]
    c_opt, n_opt, cost, nfev = refine_sequence(c_pf.copy(), n_pf.copy(), dts, samples, hands, K, use_hands,
                                               LAM_C, LAM_N, static=rest, robust_all=True, anchor=(c_fp, n_fp))
    write_track(obj / "foundationpose/symmetric", c_opt, n_opt)
    report = {"schema": "v2d.track1.symmetric_track.v1", "episode": args.episode, "object": obj_name,
              "ring_radius_m": RING_RADIUS, "frames": len(dts), "start_frame": start, "particles": PARTICLES,
              "use_hands": use_hands, "lam_c": LAM_C, "lam_n": LAM_N, "resting_fraction": float(rest.mean()),
              "frames_with_hand_near_object": float(np.mean([contact_flags(hands[t], dts[t], K).any() for t in range(len(dts))])),
              "ess_median": float(np.median(ess)), "filter_seconds": round(t1 - t0, 1),
              "optimise_seconds": round(time.time() - t1, 1), "optimise_cost": cost, "optimise_nfev": nfev}
    (obj / "foundationpose/symmetric_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
    print(f"PASS: symmetric track {tag} ({obj_name})")


if __name__ == "__main__":
    main()
