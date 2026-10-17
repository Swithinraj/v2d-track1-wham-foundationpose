"""SAM 3D Body on the selected actor of one or more episodes, raw outputs only (GPU; image with the SAM 3D Body library,
v2d_cari4d.sif by default, see pipeline/stages/sam3d_body.sh).

Per episode: full-resolution frames of the exo video; per-frame person box from the selected actor's ViTPose
keypoints (outputs/vitpose, the exported WHAM actor fragments: the same person the reference pipeline uses);
one static intrinsics estimate per episode (MoGe-2 at the reference frame, outputs/alignment). Each frame goes
through SAM3DBodyEstimator.process_one_image with inference_type "full" (body + hand decoders). Every output
field is saved unchanged (raw_<episode>.npz) before any refinement, plus projection overlays every N frames.
Writes outputs/sam3d_body/<episode>/raw_<episode>.npz, summary.json and overlay/.
Usage (inside the container): sam3d_body_video.py --episodes E [E ...] [--out DIR] [--inference_type full]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, "/workspace/v2d_sam3d_body/lib")
from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body  # noqa: E402

T1 = Path("/track1")
W = T1 / "weights/sam3d_body"
VIDEOS = T1 / "data/track_1/videos/chunk-000/observation.images.exo_camera"
FIELDS = ("bbox", "focal_length", "pred_keypoints_3d", "pred_keypoints_2d", "pred_vertices", "pred_cam_t", "pred_pose_raw",
          "global_rot", "body_pose_params", "hand_pose_params", "scale_params", "shape_params", "expr_params",
          "pred_joint_coords", "pred_global_rots", "mhr_model_params", "lhand_bbox", "rhand_bbox")


def person_boxes(kps, W_img, H_img, min_conf=0.3):
    """[T, 4] xyxy boxes around the confident ViTPose keypoints (+10 % width, +20 % head, +10 % feet);
    frames without keypoints take the nearest valid frame's box."""
    T = len(kps)
    boxes = np.full((T, 4), np.nan, np.float32)
    for f in range(T):
        k = kps[f]
        ok = np.isfinite(k[:, 0]) & (k[:, 2] >= min_conf)
        if ok.sum() < 4:
            continue
        x0, y0 = k[ok, 0].min(), k[ok, 1].min()
        x1, y1 = k[ok, 0].max(), k[ok, 1].max()
        w, h = x1 - x0, y1 - y0
        boxes[f] = [max(0, x0 - 0.1 * w), max(0, y0 - 0.2 * h), min(W_img - 1, x1 + 0.1 * w), min(H_img - 1, y1 + 0.1 * h)]
    valid = np.flatnonzero(np.isfinite(boxes[:, 0]))
    filled = valid.size > 0
    for f in np.flatnonzero(~np.isfinite(boxes[:, 0])):
        if filled:
            boxes[f] = boxes[valid[np.argmin(np.abs(valid - f))]]
    return boxes, valid.size


def to_np(v):
    if v is None:
        return None
    if torch.is_tensor(v):
        return v.detach().float().cpu().numpy()
    return np.asarray(v, np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=T1 / "outputs/sam3d_body")
    ap.add_argument("--episodes", type=int, nargs="+", required=True)
    ap.add_argument("--inference_type", default="full", choices=("full", "body"))
    ap.add_argument("--render_every", type=int, default=30)
    args = ap.parse_args()
    dev = torch.device("cuda")
    model, cfg = load_sam_3d_body(checkpoint_path=str(W / "sam-3d-body-dinov3/model.ckpt"), device=dev,
                                  mhr_path=str(W / "sam-3d-body-dinov3/assets/mhr_model.pt"))
    est = SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=cfg)
    torch.cuda.empty_cache = lambda: None            # as the toolkit does: avoid a full sync per frame
    faces = np.asarray(est.faces)
    for ep in args.episodes:
        tag = f"episode_{ep:06d}"
        out_dir = args.out / tag
        (out_dir / "overlay").mkdir(parents=True, exist_ok=True)
        intr = json.loads((T1 / "outputs/alignment" / tag / "alignment.json").read_text())["intrinsics"]
        K = np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1]], np.float32)
        cap = cv2.VideoCapture(str(VIDEOS / f"{tag}.mp4"))
        T = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        Wd, Hd = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        kps = np.load(T1 / "outputs/vitpose" / f"{tag}.npz")["kps"].astype(np.float64)
        boxes, n_valid = person_boxes(kps[:T], Wd, Hd)
        cols = {k: [] for k in FIELDS}
        t0 = time.time()
        for f in range(T):
            ok, bgr = cap.read()
            if not ok:
                raise RuntimeError(f"{tag}: video ended at frame {f} of {T}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            res = est.process_one_image(img=rgb, bboxes=boxes[f], masks=None, cam_int=torch.from_numpy(K)[None],
                                        inference_type=args.inference_type)
            if len(res) != 1:
                raise RuntimeError(f"{tag} frame {f}: {len(res)} outputs for one box")
            r = res[0]
            for k in FIELDS:
                cols[k].append(to_np(r.get(k)))
            if f % args.render_every == 0:
                v = to_np(r["pred_vertices"]) + to_np(r["pred_cam_t"])[None]
                uv = (v @ K.T)[:, :2] / v[:, 2:3]
                img = bgr.copy()
                for x, y in uv[::7].astype(int):
                    if 0 <= x < Wd and 0 <= y < Hd:
                        img[y, x] = (80, 230, 120)
                b = boxes[f].astype(int)
                cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 200, 255), 2)
                cv2.imwrite(str(out_dir / "overlay" / f"{f:06d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if f % 100 == 0:
                print(f"{tag} frame {f}/{T} ({(time.time() - t0) / (f + 1):.2f} s/frame)", flush=True)
        arrays = {k: np.stack(v).astype(np.float32) for k, v in cols.items() if v and v[0] is not None}
        np.savez_compressed(out_dir / f"raw_{tag}.npz", faces=faces, K=K, prompt_boxes=boxes, **arrays)
        summary = {"frames": T, "frames_with_keypoint_box": int(n_valid), "seconds": round(time.time() - t0, 1),
                   "fields": {k: list(v.shape) for k, v in arrays.items()}}
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(tag, json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
