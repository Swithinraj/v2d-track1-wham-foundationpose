"""SAM 3D Body: shared identity and MoGe/SAM depth scale per episode (wham.sif: numpy, cv2, PIL).

Identity: reliable frames = >= 12 confident (>= 0.5) ViTPose keypoints and a prompt box clear of the image border.
The shared identity is the medoid frame's (scales[68], shape[45]) vector (per-dimension standardised over the
reliable frames): one physically consistent body, not a per-dimension average of different bodies.
Depth scale (no WHAM quantities reused): every third frame, median of MoGe-2 depth / SAM front-surface depth
over the body pixels (4 px z-buffer cells), divided by MoGe's per-frame wobble on static background pixels
(no body, no object), as the pipeline's depth-trend stage does. depth_scale = median over frames; the
normalised series qn feeds an optional depth-trend correction later.
Writes outputs/sam3d_body/<episode>/prepare.npz (identity, medoid frame, reliable mask, depth series) + prepare.json.
Usage: sam3d_body_identity.py --episodes E [...] [--raw_dir DIR --run_dir DIR]
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

T1 = Path("/track1")
STRIDE, CELL = 3, 4


def depth_m(path):
    raw = np.array(Image.open(path)).astype(np.float64)
    with np.errstate(divide="ignore"):
        return np.where(raw > 0, 65535.0 / raw - 1.0, np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", type=Path, default=T1 / "outputs/sam3d_body", help="raw SAM 3D Body outputs (read only)")
    ap.add_argument("--run_dir", type=Path, default=T1 / "outputs/sam3d_body")
    ap.add_argument("--episodes", type=int, nargs="+", required=True)
    ap.add_argument("--identity_only", action="store_true",
                    help="skip the MoGe depth scale (depth_scale = 1): for outputs whose human and objects already share a frame (CARI4D)")
    args = ap.parse_args()
    for ep in args.episodes:
        tag = f"episode_{ep:06d}"
        z = np.load(args.raw_dir / tag / f"raw_{tag}.npz")
        (args.run_dir / tag).mkdir(parents=True, exist_ok=True)
        K = z["K"].astype(np.float64)
        P = z["mhr_model_params"].astype(np.float64)
        ident = np.concatenate([P[:, 136:204], z["shape_params"].astype(np.float64)], 1)       # [T, 113]
        kps = np.load(T1 / "outputs/vitpose" / f"{tag}.npz")["kps"]
        boxes = z["prompt_boxes"]
        T = len(P)
        H, W = 1152, 1536
        conf_ok = (np.nan_to_num(kps[:T, :, 2]) >= 0.5).sum(1) >= 12
        clear = (boxes[:, 0] > 4) & (boxes[:, 1] > 4) & (boxes[:, 2] < W - 5) & (boxes[:, 3] < H - 5)
        reliable = conf_ok & clear
        if reliable.sum() < 10:
            reliable = conf_ok
        R = ident[reliable]
        zs = (R - R.mean(0)) / (R.std(0) + 1e-9)
        med = np.median(zs, 0)
        medoid = np.flatnonzero(reliable)[int(np.argmin(np.linalg.norm(zs - med, axis=1)))]
        if args.identity_only:
            np.savez(args.run_dir / tag / "prepare.npz", identity=ident[medoid], medoid=medoid, reliable=reliable, depth_scale=1.0)
            print(tag, json.dumps({"frames": T, "reliable_frames": int(reliable.sum()), "medoid_frame": int(medoid)}), flush=True)
            continue
        verts = z["pred_vertices"].astype(np.float64) + z["pred_cam_t"].astype(np.float64)[:, None]
        frames = np.arange(0, T, STRIDE)
        gy, gx = np.mgrid[8:H:16, 8:W:16]
        r, stack, bg = [], [], []
        for f in frames:
            v = verts[f]
            uv = (v @ K.T)[:, :2] / v[:, 2:3]
            ok = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H) & (v[:, 2] > 0.1)
            uvv, zz = uv[ok], v[ok, 2]
            cells = (uvv[:, 1] // CELL).astype(int) * (W // CELL + 1) + (uvv[:, 0] // CELL).astype(int)
            order = np.lexsort((zz, cells))
            first = order[np.r_[True, cells[order][1:] != cells[order][:-1]]]
            D = depth_m(T1 / "outputs/object" / tag / "depth" / f"{f:06d}.png")
            ratio = D[uvv[first, 1].astype(int), uvv[first, 0].astype(int)] / zz[first]
            ratio = ratio[np.isfinite(ratio) & (ratio > 0.3) & (ratio < 3)]
            r.append(np.median(ratio) if len(ratio) > 30 else np.nan)
            body = np.zeros((H, W), np.uint8)
            body[uvv[:, 1].astype(int), uvv[:, 0].astype(int)] = 1
            body = cv2.dilate(body, np.ones((61, 61), np.uint8)) > 0
            mask = np.array(Image.open(T1 / "outputs/object" / tag / "sam2_masks/1" / f"{f:06d}.png")) > 127
            objm = cv2.dilate(mask.astype(np.uint8), np.ones((61, 61), np.uint8)) > 0
            bg.append(~(body | objm)[gy, gx]); stack.append(D[gy, gx])
        Ds, Bm = np.stack(stack), np.stack(bg)
        ref = np.nanmedian(np.where(Bm, Ds, np.nan), axis=0)
        b = np.array([np.nanmedian((Ds[i] / ref)[Bm[i] & np.isfinite(ref)]) for i in range(len(frames))])
        r = np.array(r)
        q = r / b
        depth_scale = float(np.nanmedian(q))
        qn = q / depth_scale
        np.savez(args.run_dir / tag / "prepare.npz", identity=ident[medoid], medoid=medoid, reliable=reliable,
                 frames=frames, r=r, b=b, qn=qn, depth_scale=depth_scale)
        rep = {"frames": T, "reliable_frames": int(reliable.sum()), "medoid_frame": int(medoid), "depth_scale_moge_over_sam": depth_scale,
               "depth_ratio_p10_p90": [float(np.nanpercentile(q, 10)), float(np.nanpercentile(q, 90))],
               "identity_distance_medoid_to_median_std": float(np.linalg.norm(zs[np.flatnonzero(np.flatnonzero(reliable) == medoid)[0]] - med))}
        (args.run_dir / tag / "prepare.json").write_text(json.dumps(rep, indent=2))
        print(tag, json.dumps(rep), flush=True)


if __name__ == "__main__":
    main()
