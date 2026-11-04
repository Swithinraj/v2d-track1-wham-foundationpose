#!/usr/bin/env python3
"""Tracked 2D body keypoints of the actor with ViTPose-H (COCO-17, top-down; v2d_sam3d_body.sif, GPU).

MediaPipe-free 2D evidence for the human refinement. Per frame the crop is the actor's SAM 2 person mask box
(outputs/person/<episode>/sam2_masks/0; temporally median-filtered over 5 frames, aspect 3:4, padding 1.25; frames
without a mask interpolated from the neighbouring boxes), so the keypoints belong to the tracked actor by construction,
also when other people are in view. ViTPose-H (vitpose-h-multi-coco checkpoint, fp16) with flip test; per joint the
heatmap argmax with quarter-pixel refinement, confidence = heatmap maximum.
Writes <out>/<episode>.npz: kps2d [T,17,3] (COCO order, px + confidence), size [W,H].
Usage (container): vitpose_track.py [--weights FILE] [--out DIR] [EPISODE ...]
"""
import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(os.environ.get("TRACK1_ROOT") or Path(__file__).resolve().parents[1])
FLIP_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
ARGS = None


# ViTPose-H: mmpose ViT backbone + simple heatmap head
class Attn(nn.Module):
    def __init__(self, d, h):
        super().__init__(); self.h = h; self.qkv = nn.Linear(d, 3 * d); self.proj = nn.Linear(d, d)

    def forward(self, x):
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.h, C // self.h).permute(2, 0, 3, 1, 4)
        return self.proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, C))


class Mlp(nn.Module):
    def __init__(self, d):
        super().__init__(); self.fc1 = nn.Linear(d, 4 * d); self.fc2 = nn.Linear(4 * d, d)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.norm1, self.attn = nn.LayerNorm(d, eps=1e-6), Attn(d, h)
        self.norm2, self.mlp = nn.LayerNorm(d, eps=1e-6), Mlp(d)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class PatchEmbed(nn.Module):
    def __init__(self, d):
        super().__init__(); self.proj = nn.Conv2d(3, d, 16, 16, padding=2)

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2), x.shape[2:]


class Backbone(nn.Module):
    def __init__(self, d=1280, depth=32, h=16):
        super().__init__()
        self.patch_embed = PatchEmbed(d); self.pos_embed = nn.Parameter(torch.zeros(1, 193, d))
        self.blocks = nn.ModuleList([Block(d, h) for _ in range(depth)]); self.last_norm = nn.LayerNorm(d, eps=1e-6)

    def forward(self, x):
        x, (hp, wp) = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:] + self.pos_embed[:, :1]
        for b in self.blocks:
            x = b(x)
        x = self.last_norm(x)
        return x.transpose(1, 2).reshape(len(x), -1, hp, wp)


class Head(nn.Module):
    def __init__(self, d=1280):
        super().__init__()
        self.deconv_layers = nn.Sequential(nn.ConvTranspose2d(d, 256, 4, 2, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(),
                                           nn.ConvTranspose2d(256, 256, 4, 2, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU())
        self.final_layer = nn.Conv2d(256, 17, 1)

    def forward(self, x):
        return self.final_layer(self.deconv_layers(x))


class ViTPose(nn.Module):
    def __init__(self):
        super().__init__(); self.backbone = Backbone(); self.keypoint_head = Head()

    def forward(self, x):
        return self.keypoint_head(self.backbone(x))


def load_vitpose():
    m = ViTPose()
    sd = torch.load(ARGS.weights, map_location="cpu")["state_dict"]
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not missing and all("num_batches" in k for k in unexpected), (missing, unexpected)
    return m.cuda().eval().half()


MEAN = np.array([0.485, 0.456, 0.406]) * 255; STD = np.array([0.229, 0.224, 0.225]) * 255


def decode(hm):
    """hm [N,17,64,48] -> coords [N,17,2] heatmap px, conf [N,17]."""
    N, K, H, W = hm.shape
    flat = hm.reshape(N, K, -1); idx = flat.argmax(-1); conf = flat.max(-1)
    x, y = (idx % W).astype(float), (idx // W).astype(float)
    for n in range(N):
        for k in range(K):
            px, py = int(x[n, k]), int(y[n, k])
            if 0 < px < W - 1:
                x[n, k] += 0.25 * np.sign(hm[n, k, py, px + 1] - hm[n, k, py, px - 1])
            if 0 < py < H - 1:
                y[n, k] += 0.25 * np.sign(hm[n, k, py + 1, px] - hm[n, k, py - 1, px])
    return np.stack([x, y], -1), conf


def person_boxes(tag, T):
    d = ROOT / f"outputs/person/{tag}/sam2_masks/0"
    b = np.full((T, 4), np.nan)
    for f in range(T):
        m = cv2.imread(str(d / f"{f:06d}.png"), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        ys, xs = np.nonzero(m > 127)
        if len(xs) > 50:
            b[f] = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]
    ok = np.isfinite(b[:, 0])
    for c in range(4):
        b[:, c] = np.interp(np.arange(T), np.flatnonzero(ok), b[ok, c])
    b = np.median(np.lib.stride_tricks.sliding_window_view(np.pad(b, ((2, 2), (0, 0)), mode="edge"), 5, axis=0), -1)
    cx, cy, w, h = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2, b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
    w, h = np.maximum(w, 0.75 * h), np.maximum(h, w / 0.75)
    return np.stack([cx, cy, w * 1.25, h * 1.25], 1)


@torch.no_grad()
def vitpose_episode(model, tag):
    cap = cv2.VideoCapture(str(ROOT / f"data/track_1/videos/chunk-000/observation.images.exo_camera/{tag}.mp4"))
    frames = []
    while True:
        ok, im = cap.read()
        if not ok:
            break
        frames.append(im)
    T = len(frames); boxes = person_boxes(tag, T)
    kps = np.zeros((T, 17, 3))
    for a in range(0, T, 32):
        crops, A = [], []
        for f in range(a, min(a + 32, T)):
            cx, cy, w, h = boxes[f]
            M = np.array([[192 / w, 0, 96 - cx * 192 / w], [0, 256 / h, 128 - cy * 256 / h]])
            crop = cv2.warpAffine(frames[f], M, (192, 256), flags=cv2.INTER_LINEAR)[:, :, ::-1]
            crops.append((crop - MEAN) / STD); A.append((cx - w / 2, cy - h / 2, w / 48, h / 64))
        x = torch.tensor(np.stack(crops), dtype=torch.float16, device="cuda").permute(0, 3, 1, 2).contiguous()
        hm = model(x).float()
        hf = model(x.flip(3)).float().flip(3)
        perm = list(range(17))
        for i, j in FLIP_PAIRS:
            perm[i], perm[j] = j, i
        hf = hf[:, perm]; hf[..., 1:] = hf[..., :-1].clone()
        c, conf = decode(((hm + hf) / 2).cpu().numpy())
        for i, (x0, y0, sx, sy) in enumerate(A):
            kps[a + i, :, 0] = x0 + c[i, :, 0] * sx; kps[a + i, :, 1] = y0 + c[i, :, 1] * sy; kps[a + i, :, 2] = conf[i]
    H, W = frames[0].shape[:2]
    return kps, (W, H)


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("episodes", type=int, nargs="*", default=list(range(30)))
    ap.add_argument("--weights", type=Path, default=ROOT / "weights/vitpose/vitpose-h-multi-coco.pth")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/vitpose")
    ARGS = ap.parse_args(); ARGS.out.mkdir(parents=True, exist_ok=True)
    model = load_vitpose()
    for ep in ARGS.episodes:
        tag = f"episode_{ep:06d}"
        kps, (W, H) = vitpose_episode(model, tag)
        np.savez(ARGS.out / f"{tag}.npz", kps2d=kps.astype(np.float32), size=np.array([W, H]))
        print(tag, "frames", len(kps), "mean conf %.3f" % kps[..., 2].mean(), "low-conf share %.3f" % (kps[..., 2] < 0.3).mean(), flush=True)


if __name__ == "__main__":
    main()
