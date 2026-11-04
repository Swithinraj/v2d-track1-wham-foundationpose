#!/usr/bin/env python3
"""True-camera replay (10-29 formula audit). Every stage uses MoGe-2's camera: its focal is 3.4 % short (median over
1488 FORM-HOI episodes) and its principal point is the image centre, 54 / 38 px from the real one (scripts/camera_truth_check.py;
each rig camera has ONE calibration over all recording days). Under the pipeline's own model the image evidence stays
and only its 3D interpretation changes: a pixel u = f_h x / z + c_h of the human, whose SAM 3D Body depth scales with
the focal (metric body), is at z' = z f_t / f_h on the true ray (u - c_t) / f_t, i.e. one LINEAR map per episode
  A_h = [[1, 0, (c_hx - c_tx) / f_h], [0, 1, (c_hy - c_ty) / f_h], [0, 0, f_t / f_h]]   (a shear + a depth stretch)
The object follows the human's depth scale (object = MoGe / s, s = MoGe / SAM at the body), seen through its own
FoundationPose camera (f_o, image-centre principal point):
  A_o = [[f_o / f_h, 0, (c_ox - c_tx) / f_h], [0, f_o / f_h, (c_oy - c_ty) / f_h], [0, 0, f_t / f_h]]
Human: per frame a rigid move, the body centroid by A_h, the body turned with its viewing ray (SAM's orientation is
relative to the ray; the metric body keeps its shape). Object: translation by A_o, turned with its viewing ray,
scale x f_o / f_h (same apparent size). MODE tc = true focal + principal point; tcf = focal only; tcp = principal
point only. Writes fh_eval/<EXP>_<MODE> and submissions/<EP>_<MODE> for fh_score.py.
Usage: true_cam_replay.py ROOT EXP EP MODE   (EXP under fh_eval/, EP under submissions/)
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np

IDX = {"front_stereo_camera_left": 0, "back_stereo_camera_left": 2, "right_stereo_camera_left": 6, "left_stereo_camera_left": 4}


def ray_rot(P, A):
    """Per-frame rotation taking the viewing ray of P [n, 3] to that of A P (Rodrigues): the body / object keeps its
    orientation relative to the ray it is seen along."""
    d = P / np.linalg.norm(P, axis=1, keepdims=True); d2 = P @ A.T; d2 /= np.linalg.norm(d2, axis=1, keepdims=True)
    v = np.cross(d, d2); s2 = np.maximum((v ** 2).sum(1), 1e-18); c = (d * d2).sum(1)
    Vx = np.zeros((len(P), 3, 3)); Vx[:, 0, 1], Vx[:, 0, 2], Vx[:, 1, 2] = -v[:, 2], v[:, 1], -v[:, 0]
    Vx[:, 1, 0], Vx[:, 2, 0], Vx[:, 2, 1] = v[:, 2], -v[:, 1], v[:, 0]
    return np.eye(3)[None] + Vx + Vx @ Vx * ((1 - c) / s2)[:, None, None]


def main():
    root, exp, ep, mode = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
    ein, eout = root / "fh_eval" / exp, root / "fh_eval" / f"{exp}_{mode}"
    sin, sout = root / "submissions" / ep, root / "submissions" / f"{ep}_{mode}"
    for d in (eout, sout):
        if d.exists():
            shutil.rmtree(d)
    shutil.copytree(ein, eout); shutil.copytree(sin, sout)
    meta = {m["episode_index"]: m for m in map(json.loads, filter(str.strip, (root / "data/track_1/meta/episodes_metadata.jsonl").read_text().splitlines()))}
    rep = {}
    for f in sorted(eout.glob("episode_??????.npz")):
        tag = f.stem; m = meta[int(tag[-6:])]
        e = json.loads((root / "gt" / tag / "edex").read_text())[0]["cameras"][IDX[m["camera"]]]["intrinsics"]
        h = json.loads((root / "outputs/camera" / tag / "intrinsics.json").read_text())
        ft, (ctx, cty) = float(e["focal"][0]), map(float, e["principal"])
        fh, chx, chy = float(h["fx"]), float(h["cx"]), float(h["cy"])
        if mode == "tcf":
            ctx, cty = chx, chy
        elif mode == "tcp":
            ft = fh
        Ah = np.array([[1, 0, (chx - ctx) / fh], [0, 1, (chy - cty) / fh], [0, 0, ft / fh]])
        p = dict(np.load(f)); c = p["joints"].astype(np.float64).mean(1); Qh = ray_rot(c, Ah)
        for k in ("align", "joints", "body64", "dense", "hands"):
            if k in p:
                X = p[k].astype(np.float64)
                p[k] = (np.einsum("tij,tpj->tpi", Qh, X - c[:, None]) + (c @ Ah.T)[:, None]).astype(p[k].dtype)
        np.savez(f, **p)
        rf = root / "outputs/object" / tag / "prompt/reference_frame.txt"; so = sout / f"{tag}.npz"
        if rf.exists() and so.exists():
            ref = int(rf.read_text().strip())
            o = json.loads((root / "outputs/object" / tag / "intrinsics" / f"{ref:06d}.json").read_text())
            fo, cox, coy = float(o["fx"]), float(o["cx"]), float(o["cy"])
            Ao = np.array([[fo / fh, 0, (cox - ctx) / fh], [0, fo / fh, (coy - cty) / fh], [0, 0, ft / fh]])
            z = dict(np.load(so)); t = z["object_translation"].astype(np.float64); Qo = ray_rot(t, Ao)
            z["object_translation"] = (t @ Ao.T).astype(np.float32)
            z["object_rotation"] = np.einsum("tij,tjk->tik", Qo, z["object_rotation"].astype(np.float64)).astype(np.float32)
            z["object_scale"] = np.float32(float(z["object_scale"]) * fo / fh)
            np.savez(so, **z)
        rep[tag] = {"f_true_over_ours": ft / fh, "principal_shift_px": [chx - ctx, chy - cty],
                    "turn_deg": float(np.degrees(np.median(np.arccos(np.clip((np.trace(Qh, axis1=1, axis2=2) - 1) / 2, -1, 1)))))}
    (eout / "true_cam.json").write_text(json.dumps(rep, indent=1))
    r = np.array([v["f_true_over_ours"] for v in rep.values()]); t = np.array([v["turn_deg"] for v in rep.values()])
    print(f"PASS: {len(rep)} episodes ({mode}): f_true / ours median {np.median(r):.4f}, body turn median {np.median(t):.2f} deg -> {eout.name}")


if __name__ == "__main__":
    main()
