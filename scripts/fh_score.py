#!/usr/bin/env python3
"""FORM-HOI local scoring, step 2 (.venvs/submission): the kit's own episode metrics against the multi-view GT.

CD-H, CD-O, ACC-H, ACC-O exactly as v2dlb.mhr_submission._episode_metrics computes them (one similarity fitted on the
alignment vertices at the first scored frame, applied to the human AND the objects; Chamfer per frame; acceleration per
contiguous stretch of >= 3 frames), with fh_export.py's role stand-ins (body64 for CD-H; "*_dense" on the first 500 of the 2000 farthest-point-ordered points, itself a
farthest-point set). Episodes are scored in parallel processes.
Objects (when an episodes dir is given): our mesh budgeted as the packer budgets it (v2dlb.mesh_budget) and posed with
our track; the GT mesh (object_mesh/output_aligned.glb) posed with poses.npy; surface samples as the scorer draws them
(_sample_mesh_surface, fixed seed, same count as the body points).
Usage: fh_score.py ROOT_DIR NAME=EXPORT_DIR[:EPISODES_DIR] [...]   (GT export: ROOT_DIR/fh_eval/gt)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import trimesh

T1 = Path(os.environ.get("TRACK1_ROOT") or Path(__file__).resolve().parents[1])     # the repository root (official/, scripts/)
sys.path.insert(0, str(T1 / "official/v2d_submission_kit"))
sys.path.insert(0, str(T1 / "scripts"))
from v2dlb import mhr_submission as ms  # noqa: E402
from v2dlb.mesh_budget import budget_mesh  # noqa: E402

CD_O = None
_BUDGET = None


def sampler():
    global CD_O
    if CD_O is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location("cdo", T1 / "official/v2d_submission_kit/metric_code/track_1/CD-O.py")
        CD_O = importlib.util.module_from_spec(spec); spec.loader.exec_module(CD_O)
    return CD_O._sample_mesh_surface


def budget():
    global _BUDGET
    if _BUDGET is None:
        import resolve_hand_penetration as rp
        _BUDGET = rp.scored_frames_and_budget()[1]
    return _BUDGET


def stretch_frames(fr):
    keep = []
    for part in ms._stretches(fr):
        if part.stop - part.start >= 3:
            keep.extend(range(part.start, part.stop))
    return np.array(keep, int)


def objects(root, tag, ep_dir, frames, counts):
    g = Path(root) / "gt" / tag
    gm = trimesh.load(g / "object_mesh/output_aligned.glb", force="mesh")
    gV, gF = np.asarray(gm.vertices, float), np.asarray(gm.faces)
    P = np.load(g / "poses.npy")[frames]
    d = np.load(Path(ep_dir) / f"{tag}.npz")
    pV, pF = budget_mesh(Path(ep_dir) / f"{tag}_object.glb", *budget())
    pV, pF = np.asarray(pV, float), np.asarray(pF)
    R, t, s = d["object_rotation"].astype(float)[frames], d["object_translation"].astype(float)[frames], float(d["object_scale"])
    out = {}
    for n in counts:
        lg, lp = sampler()(gV, gF, n, 0), sampler()(pV, pF, n, 0)
        out[n] = (np.einsum("tij,pj->tpi", P[:, :3, :3], lg) + P[:, None, :3, 3],
                  s * np.einsum("tij,pj->tpi", R, lp) + t[:, None])
    scene = {"mesh_vertices": pV, "mesh_faces": pF, "rotation": R, "translation": t, "scale": s}
    return out, P[:, :3, 3], t, scene


DENSE = 500


def score_episode(args):
    root, gp, pp, ep_dir = args
    g, p = dict(np.load(gp)), dict(np.load(pp))
    g["dense"], p["dense"] = g["dense"][:, :DENSE], p["dense"][:, :DENSE]
    if True:
        fr = np.intersect1d(g["frames"], p["frames"])
        gi, pi = np.searchsorted(g["frames"], fr), np.searchsorted(p["frames"], fr)
        k = stretch_frames(fr)
        if len(k) < 10:
            return gp.stem, None
        gi, pi, fr = gi[k], pi[k], fr[k]
        obj = objects(root, gp.stem, ep_dir, fr, (p["body64"].shape[1], DENSE)) if ep_dir else None
        res = {}
        for pts, suf in (("body64", ""), ("dense", "_dense")):
            n = p[pts].shape[1]
            og, op = (obj[0][n] if obj else (g[pts][gi], g[pts][gi]))
            tg, tp = (obj[1], obj[2]) if obj else (np.zeros((len(fr), 3)), np.zeros((len(fr), 3)))
            pred = {"mhr_vertices": p["align"][pi].astype(float), "mhr_joints": p["joints"][pi].astype(float),
                    "human_surface_points": p[pts][pi].astype(float), "object_surface_points": op, "object_translation": tp}
            tgt = {"mhr_vertices": g["align"][gi].astype(float), "mhr_joints": g["joints"][gi].astype(float),
                   "human_surface_points": g[pts][gi].astype(float), "object_surface_points": og, "object_translation": tg}
            m = ms._episode_metrics(pred, tgt, fr, None)
            res["cd_h" + suf] = m["cd_h_cm"]
            if obj:
                res["cd_o" + suf] = m["cd_o_cm"]
            if suf == "":
                res["acc_h"] = m["acc_h_cm"]
                if obj:
                    res["acc_o"] = m["acc_o_cm"]
                    if "hands" in p:      # PEN: the kit's own function (our 512-point hand approximation)
                        sc = dict(obj[3], hands=p["hands"][pi].astype(float))
                        res["pen"] = ms.episode_penetration({"mhr_vertices": p["align"][pi].astype(float)}, sc,
                                                            {"mhr_vertices": g["align"][gi].astype(float)})
        return gp.stem, res


def score(root, exp_dir, ep_dir):
    from concurrent.futures import ProcessPoolExecutor
    gt_dir = Path(root) / "fh_eval/gt"
    jobs = [(root, gp, Path(exp_dir) / gp.name, ep_dir) for gp in sorted(gt_dir.glob("episode_*.npz"))
            if (Path(exp_dir) / gp.name).exists()]
    with ProcessPoolExecutor(min(16, len(jobs))) as ex:
        return {k: v for k, v in ex.map(score_episode, jobs) if v is not None}


def main():
    root = sys.argv[1]
    out = {}
    for spec in sys.argv[2:]:
        name, dirs = spec.split("=", 1)
        exp_dir, ep_dir = (dirs.split(":", 1) + [None])[:2]
        rows = score(root, exp_dir, ep_dir)
        keys = sorted({k for r in rows.values() for k in r})
        out[name] = {"episodes": len(rows), **{k: float(np.mean([r[k] for r in rows.values() if k in r])) for k in keys},
                     "per_episode": rows}
        print(f"{name:22s} eps {len(rows):2d} | " + " | ".join(f"{k} {out[name][k]:.5f}" for k in keys), flush=True)
    names = list(out)
    for n in names[1:]:
        common = sorted(set(out[n]["per_episode"]) & set(out[names[0]]["per_episode"]))
        for k in sorted(set(out[n]) & set(out[names[0]]) - {"episodes", "per_episode"}):
            d = np.array([out[n]["per_episode"][e][k] - out[names[0]]["per_episode"][e][k] for e in common
                          if k in out[n]["per_episode"][e] and k in out[names[0]]["per_episode"][e]])
            if len(d):
                print(f"  {n} - {names[0]} {k}: {d.mean():+.5f} (se {d.std() / np.sqrt(len(d)):.5f}, better {int((d < 0).sum())}/{len(d)})")
    (Path(root) / "fh_eval").mkdir(exist_ok=True)
    (Path(root) / "fh_eval" / ("scores_" + "_".join(names) + ".json")).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
