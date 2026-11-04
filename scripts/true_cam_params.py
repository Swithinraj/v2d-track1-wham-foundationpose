#!/usr/bin/env python3
"""Episode files re-expressed in the measured camera (mhr_converter.sif, GPU or CPU; see pipeline/recipes/truecam.md).
The image evidence stays and only its 3D interpretation changes, applied to the MHR parameters so the result goes
through the contact stage and the packer. Per episode, old camera K_o (outputs/camera of --old_root: MoGe's focal, principal point at the image centre) and new
camera K_n (--k_new JSON {tag: {fx, fy, cx, cy}}):
  A_h = [[1, 0, (c_ox - c_nx) / f_o], [0, 1, (c_oy - c_ny) / f_o], [0, 0, f_n / f_o]]
Human: the MHR root pivot (the joint that the root rotation leaves in place, found numerically) moves to A_h p, the body
turns with its viewing ray Q_t (pose[:, :3] += FLIP (A_h p - p) x 10 as sam3d_body_depth_trend.py; root Euler
R = Rz Ry Rx with the camera rotation F R F). Object (FoundationPose camera f_b, image-centre
principal point = --old_root's outputs/object/<tag>/intrinsics/<ref>.json):
  A_b = [[f_b / f_o, 0, (c_bx - c_nx) / f_o], [0, f_b / f_o, (c_by - c_ny) / f_o], [0, 0, f_n / f_o]]
translation A_b t, rotation turned with its ray, scale x f_b / f_o. Checked by a second MHR pass: joints vs the target
rigid move (rms over the joints the root moves; the world joint stays at the origin), mm. Usage: true_cam_params.py --input DIR --out DIR --old_root ROOT --k_new JSON [--no_objects]
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(os.environ.get("TRACK1_ROOT") or "/track1")
sys.path.insert(0, str(Path("/track1/official/v2d_submission_kit/tools/track1")))
from mesh_to_mhr_params import MHR  # noqa: E402

FLIP = np.array([1.0, -1.0, -1.0]); F = np.diag(FLIP)


def ray_rot(P, A):
    d = P / np.linalg.norm(P, axis=1, keepdims=True); d2 = P @ A.T; d2 /= np.linalg.norm(d2, axis=1, keepdims=True)
    v = np.cross(d, d2); s2 = np.maximum((v ** 2).sum(1), 1e-18); c = (d * d2).sum(1)
    Vx = np.zeros((len(P), 3, 3)); Vx[:, 0, 1], Vx[:, 0, 2], Vx[:, 1, 2] = -v[:, 2], v[:, 1], -v[:, 0]
    Vx[:, 1, 0], Vx[:, 2, 0], Vx[:, 2, 1] = v[:, 2], -v[:, 1], v[:, 0]
    return np.eye(3)[None] + Vx + Vx @ Vx * ((1 - c) / s2)[:, None, None]


def eul2mat(e):
    a, b, c = e[:, 0], e[:, 1], e[:, 2]; ca, sa, cb, sb, cc, sc = np.cos(a), np.sin(a), np.cos(b), np.sin(b), np.cos(c), np.sin(c)
    R = np.zeros((len(e), 3, 3))
    R[:, 0, 0], R[:, 0, 1], R[:, 0, 2] = cc * cb, cc * sb * sa - sc * ca, cc * sb * ca + sc * sa
    R[:, 1, 0], R[:, 1, 1], R[:, 1, 2] = sc * cb, sc * sb * sa + cc * ca, sc * sb * ca - cc * sa
    R[:, 2, 0], R[:, 2, 1], R[:, 2, 2] = -sb, cb * sa, cb * ca
    return R                                                              # R = Rz(c) Ry(b) Rx(a)


def mat2eul(R, old):
    b = np.arcsin(np.clip(-R[:, 2, 0], -1, 1)); a = np.arctan2(R[:, 2, 1], R[:, 2, 2]); c = np.arctan2(R[:, 1, 0], R[:, 0, 0])
    e1 = np.stack([a, b, c], 1); e2 = np.stack([a + np.pi, np.pi - b, c + np.pi], 1); out = []
    for x in (e1, e2):
        out.append(x + 2 * np.pi * np.round((old - x) / (2 * np.pi)))
    d1, d2 = np.abs(out[0] - old).sum(1), np.abs(out[1] - old).sum(1)
    return np.where((d1 <= d2)[:, None], out[0], out[1])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--old_root", type=Path, required=True); ap.add_argument("--k_new", type=Path, required=True)
    ap.add_argument("--no_objects", action="store_true")
    args = ap.parse_args()
    kn = json.loads(args.k_new.read_text())
    if args.out.exists():
        shutil.rmtree(args.out)
    shutil.copytree(args.input, args.out)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mhr = MHR(str(ROOT / "weights/mhr/assets/mhr_model.pt"), dev, chunk=128, precision="float64"); piv = None; rep = {}
    for f in sorted(args.out.glob("episode_??????.npz")):
        tag = f.stem
        if tag not in kn:
            print(f"{tag}: no new camera, unchanged", flush=True); continue
        d = dict(np.load(f)); pose = d["pose"].astype(np.float64)
        ident = torch.as_tensor(np.concatenate([d["scales"], d["shape"]])[None], dtype=torch.float64, device=dev)
        run = lambda P: mhr.run(torch.as_tensor(P, dtype=torch.float64, device=dev), ident)[1].cpu().numpy().astype(np.float64)
        J = run(pose)
        if piv is None:          # the joint the root rotation leaves in place among those the root translation moves
            p2 = pose[:4].copy(); p2[:, 3:6] += np.array([0.2, -0.15, 0.1]); turn = np.abs(run(p2) - J[:4]).max((0, 2))
            p3 = pose[:4].copy(); p3[:, :3] += 1.0; moved = np.abs(run(p3) - J[:4]).min((0, 2)) > 1e-3
            turn[~moved] = np.inf; piv = int(np.argmin(turn))
            print(f"root pivot joint {piv} (moved {turn[piv] * 1000:.4f} mm under a root turn; {int((~moved).sum())} joints fixed "
                  f"under a root shift excluded)", flush=True)
        o = json.loads((args.old_root / "outputs/camera" / tag / "intrinsics.json").read_text()); n = kn[tag]
        fo, cox, coy = float(o["fx"]), float(o["cx"]), float(o["cy"]); fn, cnx, cny = float(n["fx"]), float(n["cx"]), float(n["cy"])
        Ah = np.array([[1, 0, (cox - cnx) / fo], [0, 1, (coy - cny) / fo], [0, 0, fn / fo]])
        p = J[:, piv]; pn = p @ Ah.T; Q = ray_rot(p, Ah)
        new = pose.copy(); new[:, :3] += FLIP * (pn - p) * 10.0
        Rc = np.einsum("ij,tjk,kl->til", F, eul2mat(pose[:, 3:6]), F)       # camera-frame root rotation
        new[:, 3:6] = mat2eul(np.einsum("ij,tjk,tkl,lm->tim", F, Q, Rc, F), pose[:, 3:6])
        assert np.isfinite(new).all() and np.isfinite(Q).all(), f"{tag}: non-finite camera move"
        tgt = np.einsum("tij,tpj->tpi", Q, J - p[:, None]) + pn[:, None]
        for _ in range(3):                    # close the residual: per-frame Kabsch turn about the pivot, then the pivot shift
            Jn = run(new); a_, b_ = (Jn - Jn[:, piv:piv + 1])[:, moved], (tgt - pn[:, None])[:, moved]   # the world joint stays at the origin
            U, _, Vt = np.linalg.svd(np.einsum("tpi,tpj->tij", b_, a_)); D = np.ones((len(U), 3)); D[:, 2] = np.sign(np.linalg.det(U @ Vt))
            Rr = np.einsum("tij,tj,tjk->tik", U, D, Vt)                      # achieved -> target, camera frame
            Rc_now = np.einsum("ij,tjk,kl->til", F, eul2mat(new[:, 3:6]), F)
            new[:, 3:6] = mat2eul(np.einsum("ij,tjk,tkl,lm->tim", F, Rr, Rc_now, F), new[:, 3:6])
            new[:, :3] += FLIP * (pn - run(new)[:, piv]) * 10.0
        Jn = run(new)
        err = float(np.sqrt(((Jn - tgt)[:, moved] ** 2).sum(-1).mean()) * 1000); d["pose"] = new.astype(np.float32)
        rf = args.old_root / "outputs/object" / tag / "prompt/reference_frame.txt"
        if not args.no_objects and rf.exists() and "object_translation" in d:
            ref = int(rf.read_text().strip()); b = json.loads((args.old_root / "outputs/object" / tag / "intrinsics" / f"{ref:06d}.json").read_text())
            fb, cbx, cby = float(b["fx"]), float(b["cx"]), float(b["cy"])
            Ab = np.array([[fb / fo, 0, (cbx - cnx) / fo], [0, fb / fo, (cby - cny) / fo], [0, 0, fn / fo]])
            t = d["object_translation"].astype(np.float64); Qb = ray_rot(t, Ab)
            d["object_translation"] = (t @ Ab.T).astype(np.float32)
            d["object_rotation"] = np.einsum("tij,tjk->tik", Qb, d["object_rotation"].astype(np.float64)).astype(np.float32)
            d["object_scale"] = np.float32(float(d["object_scale"]) * fb / fo)
        np.savez(f, **d)
        rep[tag] = {"rms_mm": err, "shift_cm_median": float(np.median(np.linalg.norm(pn - p, axis=1)) * 100),
                    "turn_deg_median": float(np.degrees(np.median(np.arccos(np.clip((np.trace(Q, axis1=1, axis2=2) - 1) / 2, -1, 1)))))}
        print(tag, json.dumps(rep[tag]), flush=True)
    (args.out / "true_cam_params.json").write_text(json.dumps(rep, indent=1))
    e = max(r["rms_mm"] for r in rep.values())
    print(f"{'PASS' if e < 1.0 else 'FAIL'}: {len(rep)} episodes, max joint rms vs the target move {e:.4f} mm -> {args.out}")


if __name__ == "__main__":
    main()
