#!/usr/bin/env python3
"""Roller handle experiment, step 1: masks with the handle (v2d_sam2.sif, CoTracker3 on the path, GPU).

The current paint-roller masks hold only the white cover. In the rest frames the blue handle is plainly visible but
excluded, so SAM 3D Objects built a bare 23 x 7 cm cylinder. Per episode (18, 19, 20):
  1. rest frames: a blue blob (the handle) next to the cover mask, clear of the person mask; the best ones (up to 3,
     >= 60 frames apart) get a SAM 2 mask prompt = cover mask U handle blob U a bridge for the wire frame;
  2. SAM 2 propagation from those prompts (pass 1);
  3. CoTracker3 (offline, both directions) tracks points seeded on the handle blob and on the cover at the rest
     frames. Frames where >= 4 visible handle points fall mostly outside the pass-1 mask get point prompts (visible
     handle points + cover points, all positive) and SAM 2 is rerun (pass 2);
  4. checks for old / pass 1 / pass 2: share of visible tracked handle points inside the mask, share of the mask on
     the person, mask area, empty frames. The better of pass 1 / pass 2 (handle share, person share as tie-break) is
     written (both passes are kept on disk).
Episodes whose handle colour is not separable get a reviewed handle line in config/roller_manual.json.
Writes outputs/object_rh/<episode>/sam2_masks/1/*.png (+ sam2_masks_pass1/), outputs/object_rh/roller_masks.json,
outputs/object_rh/masks_ep<N>.jpg. Membership is never inferred from motion: handle/cover points are seeded only where the
handle is unambiguous (rest frames) and kept by CoTracker's own visibility.
Usage: roller_masks.py [EPISODE ...]   (default 18 19 20)
"""
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(os.environ.get("TRACK1_ROOT") or Path(__file__).resolve().parents[1])
R = ROOT
sys.path.insert(0, os.environ.get("COTRACKER_DIR") or str(ROOT / "third_party/co-tracker"))   # pipeline/setup/fetch_sources.sh
from cotracker.predictor import CoTrackerPredictor  # noqa: E402
from v2d.sam2.lib.sam2_utils import build_sam2_video_predictor_low_mem  # noqa: E402

VIDEOS = R / "data/track_1/videos/chunk-000/observation.images.exo_camera"
OUT = R / "outputs/object_rh"
MANUAL = json.loads((ROOT / "config/roller_manual.json").read_text())   # reviewed handle prompts where colour cannot find it
TW, TH = 512, 384   # CoTracker input size (its model resolution)
rng = np.random.default_rng(0)


def read_video(path):
    cap, frames = cv2.VideoCapture(str(path)), []
    while True:
        ok, im = cap.read()
        if not ok:
            return frames
        frames.append(im)


def read_mask(path):
    return np.array(Image.open(path)) > 127 if path.exists() else None


def cover_geometry(cover):
    ys, xs = np.nonzero(cover)
    (_, _), (w, h), _ = cv2.minAreaRect(np.stack([xs, ys], 1).astype(np.float32))
    return max(w, h), min(w, h)


def handle_blob(img, cover, person):
    """Blue connected blob next to the cover and off the person, or None."""
    if cover is None or cover.sum() < 200:
        return None
    L, S = cover_geometry(cover)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blue = (hsv[..., 0] >= 100) & (hsv[..., 0] <= 130) & (hsv[..., 1] >= 80) & (hsv[..., 2] >= 30)
    k = int(0.6 * L) | 1
    near = cv2.dilate(cover.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
    pers = cv2.dilate(person.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0 if person is not None else np.zeros_like(cover)
    n, lab, st, _ = cv2.connectedComponentsWithStats((blue & near & ~pers & ~cover).astype(np.uint8), 8)
    if n <= 1:
        return None
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    blob, area = lab == i, int(st[i, cv2.CC_STAT_AREA])
    if not 0.05 * cover.sum() <= area <= 0.8 * cover.sum():
        return None
    dist = cv2.distanceTransform((~cover).astype(np.uint8), cv2.DIST_L2, 3)
    if dist[blob].min() > 0.25 * L:
        return None
    return blob, area, S


def full_mask(cover, blob, S):
    """Cover U handle U a bridge (the wire frame) between their nearest points."""
    cy, cx = np.nonzero(cover); by, bx = np.nonzero(blob)
    sub_c = rng.choice(len(cx), min(len(cx), 400), replace=False); sub_b = rng.choice(len(bx), min(len(bx), 400), replace=False)
    pc, pb = np.stack([cx[sub_c], cy[sub_c]], 1), np.stack([bx[sub_b], by[sub_b]], 1)
    d = np.linalg.norm(pc[:, None] - pb[None], axis=2); i, j = np.unravel_index(d.argmin(), d.shape)
    m = (cover | blob).astype(np.uint8)
    cv2.line(m, tuple(int(v) for v in pc[i]), tuple(int(v) for v in pb[j]), 1, max(3, int(0.2 * S)))
    return m > 0


def sample_points(mask, n, erode=3):
    m = cv2.erode(mask.astype(np.uint8), np.ones((erode, erode), np.uint8)) > 0
    if m.sum() < n:
        m = mask
    ys, xs = np.nonzero(m)
    idx = rng.choice(len(xs), min(n, len(xs)), replace=False)
    return np.stack([xs[idx], ys[idx]], 1).astype(np.float32)


def lab_at(img, p):
    x, y = int(round(p[0])), int(round(p[1]))
    patch = img[max(y - 2, 0):y + 3, max(x - 2, 0):x + 3]
    if patch.size == 0:
        return None
    return cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(float).mean(0)


def sam2_run(pred, video, mask_prompts, point_prompts):
    out = {}
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        state = pred.init_state(str(video), offload_video_to_cpu=True)
        for f, m in mask_prompts:
            pred.add_new_mask(inference_state=state, frame_idx=int(f), obj_id=1, mask=m)
        for f, pts, lab in point_prompts:
            pred.add_new_points_or_box(inference_state=state, frame_idx=int(f), obj_id=1, points=pts, labels=lab)
        for reverse in (True, False):
            for f, _, masks in pred.propagate_in_video(state, reverse=reverse):
                out[f] = (masks[0, 0] > 0.0).cpu().numpy()
    return out


def handle_share(get_mask, tracks, vis, hidx, T, H, W):
    """Per frame: share of visible tracked handle points inside the (3 px dilated) mask (frames with >= 3 visible)."""
    shares = {}
    for f in range(T):
        sel = hidx[vis[f, hidx]]
        if len(sel) < 3:
            continue
        m = get_mask(f)
        if m is None:
            shares[f] = 0.0
            continue
        m = cv2.dilate(m.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        p = np.clip(np.round(tracks[f, sel]).astype(int), 0, [W - 1, H - 1])
        shares[f] = float(m[p[:, 1], p[:, 0]].mean())
    return shares


def stats(get_mask, person_mask, T):
    area, on_person, empty = [], [], 0
    for f in range(T):
        m = get_mask(f)
        if m is None or m.sum() == 0:
            empty += 1
            continue
        area.append(int(m.sum()))
        pm = person_mask(f)
        if pm is not None:
            on_person.append(float((m & pm).sum() / m.sum()))
    return {"area_median_px": float(np.median(area)) if area else 0.0, "on_person": float(np.mean(on_person)) if on_person else 0.0,
            "empty_frames": empty}


def episode(ep, pred, tracker):
    tag = f"episode_{ep:06d}"
    video = VIDEOS / f"{tag}.mp4"
    frames = read_video(video); T = len(frames); H, W = frames[0].shape[:2]
    old = lambda f: read_mask(R / f"outputs/object/{tag}/sam2_masks/1/{f:06d}.png")  # noqa: E731
    person = lambda f: read_mask(R / f"outputs/person/{tag}/sam2_masks/0/{f:06d}.png")  # noqa: E731
    # 1. rest frames with the handle visible
    cand = []
    areas = [m.sum() for m in (old(f) for f in range(0, T, 3)) if m is not None]
    med = float(np.median(areas))
    for f in range(0, T, 3):
        cov = old(f)
        hb = handle_blob(frames[f], cov, person(f))
        if hb is not None and cov.sum() >= 0.8 * med:
            cand.append((hb[1] * min(1.0, cov.sum() / med), f, hb))
    cand.sort(key=lambda c: -c[0])
    rest = []
    for score, f, hb in cand:
        if all(abs(f - g) >= 60 for g, _ in rest):
            rest.append((f, hb))
        if len(rest) == 3:
            break
    if str(ep) in MANUAL:   # reviewed handle line -> blob; first, automatic rest frames kept if far enough
        mspec = MANUAL[str(ep)]; f = int(mspec["frame"]); cov = old(f)
        blob = np.zeros(cov.shape, np.uint8)
        cv2.line(blob, tuple(mspec["handle_line"][0]), tuple(mspec["handle_line"][1]), 1, int(mspec["width"]))
        blob = (blob > 0) & ~cov
        rest = [(f, (blob, int(blob.sum()), cover_geometry(cov)[1]))] + [(g, hb) for g, hb in rest if abs(g - f) >= 60][:2]
    if not rest:
        return {"episode": tag, "error": "no rest frame with a visible handle"}
    mask_prompts = [(f, full_mask(old(f), hb[0], hb[2])) for f, hb in rest]
    # 2. SAM 2 pass 1
    p1 = sam2_run(pred, video, mask_prompts, [])
    # 3. CoTracker3: handle + cover points seeded at the rest frames
    q, kind = [], []
    for (f, hb), (_, fm) in zip(rest, mask_prompts):
        for p in sample_points(hb[0], 24):
            q.append([f, p[0] * TW / W, p[1] * TH / H]); kind.append(1)
        for p in sample_points(old(f), 16, erode=7):
            q.append([f, p[0] * TW / W, p[1] * TH / H]); kind.append(0)
    q, kind = torch.tensor(q, dtype=torch.float32), np.array(kind)
    vid = torch.from_numpy(np.stack([cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (TW, TH), interpolation=cv2.INTER_AREA)
                                     for im in frames])).permute(0, 3, 1, 2)[None].float().cuda()
    tracks, vis = tracker(vid, queries=q[None].cuda(), backward_tracking=True)
    del vid
    torch.cuda.empty_cache()
    tracks = tracks[0].cpu().numpy() * np.array([W / TW, H / TH]); vis = vis[0].cpu().numpy() > 0.5
    hidx, cidx = np.nonzero(kind == 1)[0], np.nonzero(kind == 0)[0]
    # consistency check (tracks can stick to the table when the hand lifts the handle while staying "visible"):
    # a handle point counts only where the image is still blue there; a cover point only inside the old cover mask
    vis_raw = vis.copy()
    qn = q.numpy()
    seed = {j: lab_at(frames[int(qn[j, 0])], (qn[j, 1] * W / TW, qn[j, 2] * H / TH)) for j in hidx}
    for f in range(T):
        for j in hidx:
            if vis[f, j]:
                lab = lab_at(frames[f], tracks[f, j])
                vis[f, j] = bool(lab is not None and np.linalg.norm(lab - seed[j]) < 35)
        cov = old(f)
        cov = cv2.dilate(cov.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0 if cov is not None else None
        for j in cidx:
            if vis[f, j]:
                x, y = np.clip(np.round(tracks[f, j]).astype(int), 0, [W - 1, H - 1])
                vis[f, j] = bool(cov is not None and cov[y, x])
    s_old = handle_share(old, tracks, vis, hidx, T, H, W)
    s_p1 = handle_share(lambda f: p1.get(f), tracks, vis, hidx, T, H, W)
    # pass 2: point prompts where pass 1 loses visible handle points
    flagged = sorted(f for f, s in s_p1.items() if s < 0.5 and vis[f, hidx].sum() >= 4 and all(abs(f - g) > 2 for g, _ in rest))
    chosen = []
    for f in flagged:
        if all(abs(f - g) >= 8 for g in chosen):
            chosen.append(f)
    chosen = chosen[:40]
    point_prompts = []
    for f in chosen:
        hs = hidx[vis[f, hidx]]; cs = cidx[vis[f, cidx]]
        hs = rng.choice(hs, min(5, len(hs)), replace=False); cs = rng.choice(cs, min(3, len(cs)), replace=False) if len(cs) else cs
        pts = np.concatenate([tracks[f, hs], tracks[f, cs]]).astype(np.float32)
        point_prompts.append((f, pts, np.ones(len(pts), np.int32)))
    p2 = sam2_run(pred, video, mask_prompts, point_prompts) if point_prompts else p1
    s_p2 = handle_share(lambda f: p2.get(f), tracks, vis, hidx, T, H, W)
    res = {}
    for name, get, sh in (("old", old, s_old), ("pass1", lambda f: p1.get(f), s_p1), ("pass2", lambda f: p2.get(f), s_p2)):
        res[name] = {"handle_in_mask": float(np.mean(list(sh.values()))) if sh else 0.0, "frames_with_handle_visible": len(sh),
                     **stats(get, person, T)}
    # the handle is held most of the time, so some overlap with the person (hand) mask is expected; reject only bleed
    best = "pass2" if (res["pass2"]["handle_in_mask"] > res["pass1"]["handle_in_mask"] + 0.05 and res["pass2"]["on_person"] <= 0.06
                       and res["pass2"]["area_median_px"] < 1.6 * max(res["old"]["area_median_px"], 1.0)) else "pass1"
    final = p2 if best == "pass2" else p1
    for sub, masks in (("sam2_masks/1", final), ("sam2_masks_pass1/1", p1), ("sam2_masks_pass2/1", p2)):
        d = OUT / tag / sub; d.mkdir(parents=True, exist_ok=True)
        for f in range(T):
            Image.fromarray((masks.get(f, np.zeros((H, W), bool)) * 255).astype(np.uint8)).save(d / f"{f:06d}.png")
    np.savez_compressed(OUT / tag / "cotracker.npz", tracks=tracks.astype(np.float32), vis=vis, vis_raw=vis_raw, kind=kind, queries=q.numpy())
    # overlays: rest frame + frames spread over the episode (old red, new green, visible handle tracks blue)
    shows = [rest[0][0]] + [int(x) for x in np.linspace(0, T - 1, 7)[1:-1]]
    tiles = []
    for f in shows:
        im = frames[f].copy()
        for m, col in ((old(f), (0, 0, 255)), (p1.get(f), (0, 255, 255)), (p2.get(f), (0, 255, 0))):
            if m is not None and m.any():
                cnt, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(im, cnt, -1, col, 2)
        for i in hidx[vis[f, hidx]]:
            cv2.circle(im, tuple(int(v) for v in tracks[f, i]), 3, (255, 128, 0), -1)
        ref = final.get(f) if final.get(f) is not None and final.get(f).any() else old(f)
        ys, xs = np.nonzero(ref) if ref is not None and ref.any() else (np.array([H // 2]), np.array([W // 2]))
        cx, cy = int(xs.mean()), int(ys.mean()); x0, y0 = min(max(cx - 240, 0), W - 480), min(max(cy - 160, 0), H - 320)
        crop = cv2.resize(im[y0:y0 + 320, x0:x0 + 480], (600, 400))
        cv2.putText(crop, f"ep{ep} f{f}" + (" (prompt)" if f == rest[0][0] else "") + f"  old red, p1 yellow, p2 green; kept {best}", (8, 26), 0, 0.6, (255, 255, 255), 2)
        tiles.append(crop)
    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % 3)
    cv2.imwrite(str(OUT / f"masks_ep{ep}.jpg"), np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]))
    return {"episode": tag, "rest_frames": [f for f, _ in rest], "pass2_prompt_frames": chosen, "written": best,
            "handle_point_frames_rejected_by_appearance": int((vis_raw[:, hidx] & ~vis[:, hidx]).sum()), **res}


def main():
    eps = [int(e) for e in sys.argv[1:]] or [18, 19, 20]
    os.environ["HYDRA_CONFIG_SEARCH_PATH"] = str(R / "weights/sam2")   # as v2d.sam2.lib.video_to_masks does
    pred = build_sam2_video_predictor_low_mem(config_file="configs/sam2.1/sam2.1_hiera_l.yaml",
                                              ckpt_path=str(R / "weights/sam2/sam2.1_hiera_large.pt"))
    tracker = CoTrackerPredictor(checkpoint=str(ROOT / "weights/cotracker3/scaled_offline.pth"), offline=True, window_len=60).cuda()
    report = {}
    for ep in eps:
        r = episode(ep, pred, tracker)
        report[r["episode"]] = r
        print(json.dumps(r), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "roller_masks.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
