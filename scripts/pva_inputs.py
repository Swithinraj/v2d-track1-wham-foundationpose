#!/usr/bin/env python3
"""Prepare PVA-Net inputs from released RGB, actor masks and camera intrinsics.

Use the model's 192x256 affine crop convention. Expand to 3:4 before cropping;
do not shrink an arbitrary rectangular mask box and cut off limbs. Save RGB uint8
crops and explicit frame IDs so predictor input order can be audited.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def box_from_mask(mask):
    y, x = np.nonzero(mask)
    if not len(x):
        return np.full(4, np.nan)
    return np.array([x.min(), y.min(), x.max()+1, y.max()+1], dtype=float)


def expand_boxes(boxes, padding=1.15, max_gap=10):
    boxes = np.asarray(boxes, dtype=float).copy()
    if boxes.ndim != 2 or boxes.shape[1] != 4 or padding < 1:
        raise ValueError('expected [T,4] boxes and padding >= 1')
    valid = np.isfinite(boxes).all(1) & (boxes[:,2]>boxes[:,0]) & (boxes[:,3]>boxes[:,1])
    if not valid.any():
        raise ValueError('actor track has no valid boxes')
    boundaries = np.diff(np.r_[False, ~valid, False].astype(int))
    if np.any(np.flatnonzero(boundaries == -1)-np.flatnonzero(boundaries == 1) > max_gap):
        raise ValueError('actor track has an unsupported missing span')
    t = np.arange(len(boxes))
    for col in range(4):
        boxes[:,col] = np.interp(t, t[valid], boxes[valid,col])
    center = (boxes[:,:2]+boxes[:,2:])/2
    height = np.maximum(boxes[:,3]-boxes[:,1], (boxes[:,2]-boxes[:,0])/.75)*padding
    size = np.stack([.75*height, height], axis=1)
    return np.concatenate([center-size/2, center+size/2], axis=1)


def crop_rgb(image, box):
    x0,y0,x1,y1 = np.asarray(box, dtype=float)
    source = np.array([[x0,y0],[x1,y0],[(x0+x1)/2,(y0+y1)/2]],np.float32)
    target = np.array([[0,0],[191,0],[95.5,127.5]],np.float32)
    return cv2.warpAffine(image, cv2.getAffineTransform(source,target), (192,256),flags=cv2.INTER_LINEAR)


def validate_sequence(frames, fps, expected):
    frames = np.asarray(frames)
    if not np.isclose(fps,30.,atol=.01):
        raise ValueError(f'PVA-Net requires 30 FPS, got {fps}')
    if len(frames) != expected:
        raise ValueError(f'frame count {len(frames)} != {expected}')
    if not np.array_equal(frames,np.arange(expected)):
        raise ValueError('frame IDs must be consecutive starting at zero')


def main():
    parser=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--episodes',type=int,nargs='+',required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--max-frames',type=int,default=0,help='pilot only; zero means full clip')
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    meta={r['episode_index']:r for r in map(json.loads,(args.root/'data/track_1/meta/episodes.jsonl').read_text().splitlines())}
    for ep in args.episodes:
        tag=f'episode_{ep:06d}'
        target=args.out/f'{tag}.npz'
        if target.exists():
            raise FileExistsError(target)
        n=meta[ep]['length']
        if args.max_frames: n=min(n,args.max_frames)
        masks=args.root/f'outputs/person/{tag}/sam2_masks/0'
        boxes=[]
        for i in range(n):
            mask=cv2.imread(str(masks/f'{i:06d}.png'),cv2.IMREAD_GRAYSCALE)
            if mask is None: raise FileNotFoundError(masks/f'{i:06d}.png')
            boxes.append(box_from_mask(mask))
        boxes=expand_boxes(boxes)
        camera=json.loads((args.root/f'outputs/camera_measured/{tag}/intrinsics.json').read_text())
        K=np.array([[camera['fx'],0,camera['cx']],[0,camera['fy'],camera['cy']],[0,0,1]],np.float32)
        video=args.root/f'data/track_1/videos/chunk-000/observation.images.exo_camera/{tag}.mp4'
        cap=cv2.VideoCapture(str(video)); fps=cap.get(cv2.CAP_PROP_FPS)
        crops=[]
        try:
            for box in boxes:
                ok,bgr=cap.read()
                if not ok: raise ValueError(f'{tag}: RGB ends before mask/metadata frames')
                crops.append(crop_rgb(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),box))
        finally:
            cap.release()
        frames=np.arange(len(crops))
        validate_sequence(frames,fps,n)
        np.savez(target, images=np.stack(crops), boxes=boxes.astype(np.float32), K=K,
                 frames=frames, fps=fps)
        print(f'PVA_INPUT {tag}: {n} RGB frames, {fps:g} FPS -> {target}',flush=True)


if __name__=='__main__': main()
