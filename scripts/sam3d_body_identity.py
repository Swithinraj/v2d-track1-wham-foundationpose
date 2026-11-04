"""SAM 3D Body: shared identity and MoGe/SAM depth scale per episode (v2d_cari4d.sif: numpy, cv2, PIL).

Identity: reliable frames = >= 14 of SAM 3D Body's 17 body keypoints (COCO subset of its 2D keypoints) inside the
actor's SAM 2 person mask (dilated 15 px) and a prompt box clear of the image border.
The shared identity is the medoid frame's (scales[68], shape[45]) vector (per-dimension standardised over the
reliable frames): one physically consistent body, not a per-dimension average of different bodies.
Depth scale: every third frame, median of MoGe-2 depth / SAM front-surface depth
over the body pixels (4 px z-buffer cells), divided by MoGe's per-frame wobble on static background pixels
(no body, no object), as the pipeline's depth-trend stage does. depth_scale = median over frames; the
normalised series qn feeds an optional depth-trend correction later.
Writes outputs/sam3d_body/<episode>/prepare.npz (identity, medoid frame, reliable mask, depth series) + prepare.json,
and outputs/keypoints2d_sam/<episode>.npz: SAM 3D Body's raw per-frame 2D body keypoints in COCO-17 order
(kps [T, 17, 3]: x, y, 1 inside the image else 0), the held-out 2D check of the human refinement.
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
COCO_FROM_MHR70 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 62, 41, 9, 10, 11, 12, 13, 14]


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
        boxes = z["prompt_boxes"]
        T = len(P)
        H, W = 1152, 1536
        k2 = z["pred_keypoints_2d"][:, COCO_FROM_MHR70, :2].astype(np.float64)
        inside_img = (k2[..., 0] >= 0) & (k2[..., 0] < W) & (k2[..., 1] >= 0) & (k2[..., 1] < H)
        kp_out = T1 / "outputs/keypoints2d_sam"
        kp_out.mkdir(parents=True, exist_ok=True)
        np.savez(kp_out / f"{tag}.npz", kps=np.concatenate([k2, inside_img[..., None]], -1).astype(np.float32))
        conf_ok = np.zeros(T, bool)
        for f in range(T):
            m = cv2.imread(str(T1 / "outputs/person" / tag / "sam2_masks/0" / f"{f:06d}.png"), cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            m = cv2.dilate((m > 127).astype(np.uint8), np.ones((31, 31), np.uint8)) > 0
            xy = np.round(k2[f]).astype(int)
            ok = inside_img[f]
            conf_ok[f] = int(m[xy[ok, 1], xy[ok, 0]].sum()) >= 14
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
