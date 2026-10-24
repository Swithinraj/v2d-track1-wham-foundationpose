#!/usr/bin/env python3
"""Joint refiner cache, visual part (v2d_sam3d_body.sif for cv2; CPU): image evidence from the episode video.

Per episode, all frames, from pipeline outputs only (no GT; the same code serves deployment):
  crops      [T, 3, S, S, 3] uint8 RGB crops: hand of wrist 42 (block 0, the person's right), hand of wrist 78 (block 1,
             left), object. Hands are centred on the ViTPose hand landmarks when the hand is detected (score > 0.5),
             else beyond the confident pose wrist, else on our projected hand; the object on its SAM 2 mask box, else
             on our projected object. Side: 1.6 x the landmark extent / 2.5 x the projected extent / 1.25 x the mask box.
  boxes      [T, 3, 3] crop centre and side in normalised camera coordinates ((u - cx) / fx, (v - cy) / fy, side / fx)
  box_src    [T, 3] 0 = our projection, 1 = pose wrist, 2 = hand landmarks / mask (how much image evidence placed it)
  kp, kpc    [T, 33, 2] normalised ViTPose body landmarks (MediaPipe layout), [T, 33] confidence
  hand2d, hand2d_s   [T, 2, 21, 2] normalised hand landmarks assigned to blocks by the nearer pose wrist, [T, 2] score
  mask_m     [T, 5] object mask: present, centroid u / v, std u / v (normalised; silhouette size = depth evidence)
  K          fx, fy, cx, cy (outputs/camera/<tag>/intrinsics.json: the human's camera; the objects share it)
Needs the joint human/object caches for our hand points and object track (projections). Writes OUT/<root>_<tag>_vis.npz.
Usage: joint_cache_visual.py HUMAN_CACHE OBJECT_CACHE OUT ROOT_DIR [ROOT_DIR ...] [--size 96]
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

HAND_LM = ((16, 14), (15, 13))          # block 0 = wrist 42 = right: MediaPipe right wrist / elbow; block 1 = left


def crop(img, cx, cy, side, S):
    """Square crop centred (cx, cy) of side pixels (zero padded), resized to S x S."""
    h = side / 2.0
    M = np.array([[S / side, 0, -(cx - h) * S / side], [0, S / side, -(cy - h) * S / side]], np.float32)
    return cv2.warpAffine(img, M, (S, S), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT)


def project(X, K):
    return np.stack([K[0] * X[..., 0] / X[..., 2] + K[2], K[1] * X[..., 1] / X[..., 2] + K[3]], -1)


def episode(args):
    root, tag, hpath, opath, out, S = args
    h, o = np.load(hpath), np.load(opath)
    T = len(h["ours"])
    Kj = json.loads((root / f"outputs/camera/{tag}/intrinsics.json").read_text()); K = np.array([Kj["fx"], Kj["fy"], Kj["cx"], Kj["cy"]])
    W, H = int(Kj["width"]), int(Kj["height"])
    kpd = np.load(root / f"outputs/kp2d_vpx/{tag}.npz")
    pose = np.nan_to_num(kpd["pose"][:T, 0].astype(float)); hands = kpd["hands"][:T, 0].astype(float); hs = np.nan_to_num(kpd["hand_score"][:T, 0].astype(float))
    n = len(pose)
    if n < T:                                                        # pad short keypoint tracks
        pose = np.concatenate([pose, np.zeros((T - n,) + pose.shape[1:])]); hands = np.concatenate([hands, np.full((T - n,) + hands.shape[1:], np.nan)])
        hs = np.concatenate([hs, np.zeros((T - n, 2))])
    kpc = pose[:, :, 3]
    # hand landmarks -> blocks by the nearer pose wrist
    h2d = np.full((T, 2, 21, 2), np.nan); h2s = np.zeros((T, 2))
    for f in range(T):
        for k in range(2):
            if hs[f, k] <= 0 or not np.isfinite(hands[f, k, :, :2]).all():
                continue
            c = hands[f, k, :, :2].mean(0)
            d = [np.linalg.norm(c - pose[f, w, :2]) if kpc[f, w] > 0.3 else 1e9 for w, _ in HAND_LM]
            b = int(np.argmin(d))
            if hs[f, k] > h2s[f, b]:
                h2d[f, b] = hands[f, k, :, :2]; h2s[f, b] = hs[f, k]
    hp = project(h["hands_ours"].astype(float), K).reshape(T, 2, 64, 2)
    # object projection and mask boxes
    so = o["surf_ours"].astype(float)[::4]
    Xo = float(o["obj_s"]) * np.einsum("tij,nj->tni", o["obj_R"].astype(float), so) + o["obj_t"].astype(float)[:, None]
    op = project(Xo, K)
    boxes = np.zeros((T, 3, 3)); src = np.zeros((T, 3)); mm = np.zeros((T, 5))
    for f in range(T):
        for b, (wi, ei) in enumerate(HAND_LM):
            ext = np.ptp(hp[f, b], axis=0).max()
            if h2s[f, b] > 0.5:
                c = h2d[f, b].mean(0); side = 1.6 * np.ptp(h2d[f, b], axis=0).max(); src[f, b] = 2
            elif kpc[f, wi] > 0.5:
                w_, e_ = pose[f, wi, :2], pose[f, ei, :2]
                c = w_ + 0.4 * (w_ - e_) if kpc[f, ei] > 0.3 else w_; side = 2.5 * ext; src[f, b] = 1
            else:
                c = hp[f, b].mean(0); side = 2.5 * ext
            boxes[f, b] = (c[0], c[1], np.clip(side, 48, 400))
        m = cv2.imread(str(root / f"outputs/object/{tag}/sam2_masks/1/{f:06d}.png"), cv2.IMREAD_GRAYSCALE)
        if m is not None and (m > 0).any():
            v, u = np.nonzero(m > 0)
            mm[f] = (1, u.mean(), v.mean(), u.std(), v.std())
            boxes[f, 2] = ((u.min() + u.max()) / 2, (v.min() + v.max()) / 2, np.clip(1.25 * max(np.ptp(u), np.ptp(v)), 48, 900)); src[f, 2] = 2
        else:
            c = op[f].mean(0); boxes[f, 2] = (c[0], c[1], np.clip(1.25 * np.ptp(op[f], axis=0).max(), 48, 900))
    boxes = np.nan_to_num(boxes)
    crops = np.zeros((T, 3, S, S, 3), np.uint8)
    cap = cv2.VideoCapture(str(root / f"data/track_1/videos/chunk-000/observation.images.exo_camera/{tag}.mp4"))
    for f in range(T):
        ok, img = cap.read()
        if not ok:
            break
        if img.shape[1] != W:                                        # keypoints / masks are in the intrinsics' resolution
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        img = img[:, :, ::-1]
        for b in range(3):
            crops[f, b] = crop(np.ascontiguousarray(img), *boxes[f, b], S)
    cap.release()
    nrm = lambda u, v: np.stack([(u - K[2]) / K[0], (v - K[3]) / K[1]], -1)  # noqa: E731
    bn = np.concatenate([nrm(boxes[..., 0], boxes[..., 1]), boxes[..., 2:3] / K[0]], -1)
    mn = np.stack([mm[:, 0], (mm[:, 1] - K[2]) / K[0] * mm[:, 0], (mm[:, 2] - K[3]) / K[1] * mm[:, 0], mm[:, 3] / K[0], mm[:, 4] / K[1]], -1)
    f32 = lambda x: np.nan_to_num(np.asarray(x, np.float32))  # noqa: E731
    np.savez(out / f"{root.name}_{tag}_vis.npz", crops=crops, boxes=f32(bn), box_src=f32(src), kp=f32(nrm(pose[:, :, 0], pose[:, :, 1])), kpc=f32(kpc),
             hand2d=f32(nrm(h2d[..., 0], h2d[..., 1])), hand2d_s=f32(h2s), mask_m=f32(mn), K=f32(K))
    return f"{root.name} {tag} T={T} hands by landmarks {np.mean(src[:, :2] == 2) * 100:.0f} % / wrist {np.mean(src[:, :2] == 1) * 100:.0f} % | object by mask {np.mean(src[:, 2] == 2) * 100:.0f} %"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hc", type=Path); ap.add_argument("oc", type=Path); ap.add_argument("out", type=Path); ap.add_argument("roots", type=Path, nargs="+")
    ap.add_argument("--size", type=int, default=96); ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for root in a.roots:
        for ep in range(30):
            tag = f"episode_{ep:06d}"; hp = a.hc / f"{root.name}_{tag}.npz"; op = a.oc / f"{root.name}_{tag}_obj.npz"
            if hp.exists() and op.exists() and not (a.out / f"{root.name}_{tag}_vis.npz").exists():
                jobs.append((root, tag, hp, op, a.out, a.size))
    with ProcessPoolExecutor(a.workers) as pool:
        for line in pool.map(episode, jobs):
            print(line, flush=True)


if __name__ == "__main__":
    main()
