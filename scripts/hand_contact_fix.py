"""Clear hand-object penetration on the HAND side, without touching any scored joint or the object (mhr_converter.sif, GPU).

PEN counts hand points inside our own object mesh; most of that depth sits in fingers/knuckles of grasps, which no
object translation can clear. ACC-H uses 22 joints (MHR_TABLE3_BODY_JOINT_INDICES) that finger and wrist-rotation
parameters do not move, so refitting only those parameters leaves ACC-H (and ACC-O, CD-O) exactly unchanged and moves
CD-H by the hand-vertex displacement only (mm).
Run through pipeline/stages/hand_fix.sh (pack.sh HAND_FIX=1); meshes from budget_object_meshes.py.
Free columns: every pose column whose perturbation moves the dense hand vertices (mhr_bodies.hand_vertex_sets) but
none of the 22 scored joints (< 1e-6 m), found by a finite-difference test on the episode's own poses.
Per frame where any dense hand vertex is inside the scorer-budgeted mesh (budget_object_meshes.py): penetrating vertices
get the target x + (depth + margin) * outward direction (minus the depth gradient, central differences, h = 2 mm),
other hand vertices keep their place (weight 0.2), and the kit's lm_pose solves the free columns; three rounds.
A frame's update is kept only if its dense-hand penetration drops. Self-PEN is reported on the dense hand and on
the 512 guessed scorer points (extract_mhr_hands.py), scored frames only.
--scored_only: only scored frames are checked and refitted (PEN and CD-H count only those; finger/wrist-rotation
parameters are in no acceleration metric).
Usage: hand_contact_fix.py --input DIR --out DIR --meshes DIR [--margin 0.003] [--rounds 3] [--scored_only] [--episodes ...]
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

T1 = Path("/track1")
sys.path.insert(0, str(T1 / "official/v2d_submission_kit/tools/track1"))
sys.path.insert(0, str(T1 / "official/v2d_submission_kit"))
sys.path.insert(0, str(T1 / "scripts"))
from mesh_to_mhr_params import MHR, lm_pose  # noqa: E402
from v2dlb import mhr_submission as ms  # noqa: E402
from mhr_bodies import hand_vertex_sets  # noqa: E402
from extract_mhr_hands import hand_vertex_indices  # noqa: E402

SCORED_J = [1, 2, 18, 35, 3, 19, 36, 4, 20, 37, 8, 24, 110, 74, 38, 113, 75, 39, 76, 40, 78, 42]
MODEL = T1 / "weights/mhr/assets/mhr_model.pt"


def depth_world(points, R, t, s, V, F):
    return ms.penetration_depth((points - t) @ R / s, V, F) * s


def free_columns(mhr, pose, ident, hand_idx):
    p = pose[None].repeat(136, 1)
    p[torch.arange(136), torch.arange(136)] += 0.05
    v0, j0 = mhr.run(pose[None], ident)
    v, j = mhr.run(p, ident)
    dj = (j[:, SCORED_J] - j0[:, SCORED_J]).norm(dim=-1).amax(1)
    dv = (v[:, hand_idx] - v0[:, hand_idx]).norm(dim=-1).amax(1) / 1000.0
    return [c for c in range(136) if dj[c] < 1e-6 and dv[c] > 1e-4]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--meshes", type=Path, required=True)
    ap.add_argument("--margin", type=float, default=0.003)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--episodes", type=int, nargs="*", default=list(range(30)))
    ap.add_argument("--scored_only", action="store_true")
    args = ap.parse_args()
    # copy only this run's episodes (+ shared top-level files), so runs on disjoint episode sets can share --out
    args.out.mkdir(parents=True, exist_ok=True)
    tags = {f"episode_{ep:06d}" for ep in args.episodes}
    for src in args.input.iterdir():
        if src.is_file() and (src.name[:14] in tags if src.name.startswith("episode_") else not (args.out / src.name).exists()):
            shutil.copy2(src, args.out / src.name)
    scored = json.loads((args.meshes / "scored_frames.json").read_text())
    mhr = MHR(str(MODEL), "cuda", chunk=64, precision="float64")
    sets = hand_vertex_sets()
    hand_idx = torch.as_tensor(np.concatenate(sets), device="cuda")
    guess = np.concatenate(hand_vertex_indices())
    pos_in_dense = {int(v): k for k, v in enumerate(np.concatenate(sets))}
    guess_dense = np.array([pos_in_dense[int(g)] for g in guess if int(g) in pos_in_dense])
    report = {}
    for ep in args.episodes:
        tag = f"episode_{ep:06d}"
        d = dict(np.load(args.input / f"{tag}.npz"))
        mz = np.load(args.meshes / f"{tag}_budget.npz")
        V, F = mz["vertices"].astype(np.float64), mz["faces"]
        R, t, s = d["object_rotation"].astype(np.float64), d["object_translation"].astype(np.float64), float(d["object_scale"])
        pose = torch.as_tensor(d["pose"], dtype=torch.float64, device="cuda")
        ident = torch.as_tensor(np.concatenate([d["scales"], d["shape"]])[None], dtype=torch.float64, device="cuda")
        T = len(pose)
        cols = free_columns(mhr, pose[0], ident, hand_idx)
        verts, joints0 = mhr.run(pose, ident)
        hands = (verts[:, hand_idx] / 1000.0).cpu().numpy()
        sf = [x for x in scored[str(ep)] if x < T]

        def pen_frame(h, f):
            return depth_world(h, R[f], t[f], s, V, F)

        work = sf if args.scored_only else list(range(T))
        before = {f: pen_frame(hands[f], f) for f in work}
        pen_before = [f for f in work if before[f].max() > 0]
        cur = dict(before)
        changed = 0
        for _ in range(args.rounds):
            todo = [f for f in work if cur[f].max() > 0]
            if not todo:
                break
            for f in todo:
                h = hands[f]
                dep = cur[f]
                inside = np.flatnonzero(dep > 0)
                grad = np.zeros((len(inside), 3))
                for k in range(3):
                    e = np.zeros(3); e[k] = 0.002
                    grad[:, k] = (pen_frame(h[inside] + e, f) - pen_frame(h[inside] - e, f)) / 0.004
                out = -grad
                n = np.linalg.norm(out, axis=1, keepdims=True)
                # where the gradient vanishes (deep inside), push away from the object's centre
                centre = t[f] + s * R[f] @ ((V.min(0) + V.max(0)) / 2)
                fallback = h[inside] - centre
                out = np.where(n > 1e-3, out / np.maximum(n, 1e-9), fallback / (np.linalg.norm(fallback, axis=1, keepdims=True) + 1e-9))
                tgt_h = h.copy()
                tgt_h[inside] += (dep[inside] + args.margin)[:, None] * out
                vf = verts[f].clone()
                vf[hand_idx] = torch.as_tensor(tgt_h * 1000.0, device="cuda")
                w = torch.zeros(verts.shape[1], dtype=torch.float64, device="cuda")
                w[hand_idx] = 0.2
                w[hand_idx[torch.as_tensor(inside, device="cuda")]] = 1.0
                p1, _ = lm_pose(mhr, vf.reshape(1, -1), pose[f:f + 1], ident, iters=6, cols=cols, w=w)
                v1, _ = mhr.run(p1, ident)
                h1 = (v1[0, hand_idx] / 1000.0).cpu().numpy()
                d1 = pen_frame(h1, f)
                if d1.mean() < dep.mean():
                    pose[f] = p1[0]
                    verts[f] = v1[0]
                    hands[f] = h1
                    cur[f] = d1
                    changed += 1
        _, joints1 = mhr.run(pose, ident)
        dj = float((joints1[:, SCORED_J] - joints0[:, SCORED_J]).norm(dim=-1).max())
        if dj > 1e-5:
            raise SystemExit(f"{tag}: scored joints moved by {dj} m")
        move = float(np.abs(hands - (mhr.run(torch.as_tensor(d["pose"], dtype=torch.float64, device="cuda"), ident)[0][:, hand_idx] / 1000.0).cpu().numpy()).max())
        dense_b = float(np.mean([before[f].mean() for f in sf]) * 100) if sf else 0.0
        dense_a = float(np.mean([cur[f].mean() for f in sf]) * 100) if sf else 0.0
        guess_b = float(np.mean([before[f][guess_dense].mean() for f in sf]) * 100) if sf else 0.0
        guess_a = float(np.mean([cur[f][guess_dense].mean() for f in sf]) * 100) if sf else 0.0
        d["pose"] = pose.cpu().numpy().astype(np.float32)
        np.savez(args.out / f"{tag}.npz", **d)
        report[tag] = {"free_columns": len(cols), "penetrating_frames_before": len(pen_before),
                       "penetrating_frames_after": int(sum(cur[f].max() > 0 for f in work)), "updates": changed,
                       "dense_pen_cm_before": dense_b, "dense_pen_cm_after": dense_a, "guess512_pen_cm_before": guess_b,
                       "guess512_pen_cm_after": guess_a, "scored_joint_change_m": dj, "max_hand_vertex_move_m": move}
        print(tag, json.dumps(report[tag]), flush=True)
    keys = ("dense_pen_cm_before", "dense_pen_cm_after", "guess512_pen_cm_before", "guess512_pen_cm_after")
    summary = {k: float(np.mean([r[k] for r in report.values()])) for k in keys}
    name = "hand_contact_fix_report_" + "_".join(str(e) for e in args.episodes) + ".json" if len(args.episodes) < 30 else "hand_contact_fix_report.json"
    (args.out / name).write_text(json.dumps({"summary": summary, "episodes": report}, indent=1))
    print("summary", json.dumps(summary))


if __name__ == "__main__":
    main()
