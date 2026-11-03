#!/usr/bin/env python3
"""Local Track-1 scorer: kit-exact metric math against a pseudo-GT reference.

The organizers ship the real scorers (metric_code/track_1/*.py) but not the reference
data. This module reproduces the scoring pipeline locally:

  * unpack   : submission parquet -> per-episode parameter arrays (kit layout)
  * forward  : MHR params -> vertices (m), joints (m) via the reference TorchScript model
  * build-ref: pseudo-GT = vertex-average of INDEPENDENT chains (SAM3D-ensemble P2,
               WHAM-based BMQ3KGMk, CARI4D-CoCoNet C1), each per-episode Sim(3)-aligned
               to the base chain on temporal-mean vertices; objects from the best CD-O
               parquet, mapped by the transform that takes its own human to the reference
  * score    : candidate vs reference through the kit's own mhr_table3_metric_series
               (frame-0 Sim(3) alignment, symmetric Chamfer, acceleration error)

Roles: the kit's role_* arrays are not shipped; we use all 18439 vertices for both the
alignment fit and the human surface. Consistent for pred and target, so candidate
DELTAS are comparable; absolute values differ slightly from Kaggle (calibration absorbs
the offset). PEN is not scored here (already rank 1 / separately calibrated).

CLI:
  python localscore.py unpack   --parquet F [--out DIR]
  python localscore.py build-ref --humans P2.parquet BMQ.parquet C1.parquet
                                 --objects tab102.parquet --out DIR [--base P2]
  python localscore.py score    --cand F.parquet --ref DIR [--episodes ...] [--out json]
  python localscore.py calibrate --ref DIR --pairs pairs.json [--episodes ...]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(os.environ.get("TRACK1_ROOT") or Path(__file__).resolve().parents[1])
sys.path.insert(0, str(ROOT / "official/v2d_submission_kit/tools/track1"))
sys.path.insert(0, str(ROOT / "official/v2d_submission_kit"))
from mesh_to_mhr_params import MHR, NS, NB  # noqa: E402

MODEL = ROOT / "weights/mhr/assets/mhr_model.pt"

#: array_name -> (number count, is_per_frame)
ARRAYS = {
    "mhr_pose": (136, True),
    "mhr_scales": (68, False),
    "mhr_shape": (45, False),
    "object_rotation": (9, True),
    "object_translation": (3, True),
    "object_scale": (1, False),
    "object_mesh_vertices": (None, False),   # count = 3 * n_verts
    "object_mesh_faces": (None, False),      # rows of 3 indices
}
EP_FRAME = 999999


def _load_kit_metric(path="official/v2d_submission_kit/metric_code/track_1/CD-H.py"):
    """Import a kit metric module by path (filename has a dash).

    The module imports pandas at module level but the metric math never uses it (it is
    only needed by the parquet-merge wrapper, which we bypass). A stub keeps the import
    happy in containers without pandas/pyarrow.
    """
    import types
    if "pandas" not in sys.modules:
        sys.modules["pandas"] = types.ModuleType("pandas")
    sys.modules["pandas"].DataFrame = getattr(sys.modules["pandas"], "DataFrame", type("DataFrame", (), {}))
    spec = importlib.util.spec_from_file_location("kit_metric", str(ROOT / path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


KIT = None  # lazy


def kit():
    global KIT
    if KIT is None:
        KIT = _load_kit_metric()
    return KIT


# --------------------------------------------------------------------------- unpack

ARRAY_ORDER_FALLBACK = ["mhr_pose", "mhr_scales", "mhr_shape", "object_rotation",
                        "object_translation", "object_scale", "object_mesh_vertices",
                        "object_mesh_faces"]


def load_episodes(source: Path) -> dict[int, dict]:
    """Dispatch: .parquet -> unpack_parquet; dir of episode_*.npz -> load directly."""
    if source.suffix == ".parquet":
        return unpack_parquet(source)
    out = {}
    for p in sorted(Path(source).glob("episode_*.npz")):
        out[int(p.stem.split("_")[1])] = dict(np.load(p))
    return out


def unpack_parquet(parquet: Path) -> dict[int, dict]:
    import pandas as pd
    """Submission parquet -> {episode: {array: np.ndarray}} in kit layout.

    row_id layout: t1_<episode>_<frame>_<array_index>_<point_index> (kit pack_reconstruction).
    """
    from v2dlb import mhr_submission as ms
    try:
        order = list(ms.array_order())
    except Exception:
        order = ARRAY_ORDER_FALLBACK
    df = pd.read_parquet(parquet, columns=["row_id", "x", "y", "z"])
    parts = df["row_id"].str.split("_", expand=True)
    df = df.assign(ep=parts[1].astype(int), frame=parts[2].astype(int),
                   arr=parts[3].astype(int), pt=parts[4].astype(int))
    out: dict[int, dict] = {}
    for ep, block in df.groupby("ep"):
        ep = int(ep)
        d: dict = {}
        for idx, name in enumerate(order):
            count, per_frame = ARRAYS[name]
            sub = block[block["arr"] == idx]
            if sub.empty:
                continue
            sub = sub.sort_values(["frame", "pt"])
            if name == "object_mesh_faces":
                d[name] = sub[["x", "y", "z"]].to_numpy(np.int64)
                continue
            if name == "object_mesh_vertices":
                d[name] = sub[["x", "y", "z"]].to_numpy(np.float64)
                continue
            if per_frame:
                rows = int(sub["pt"].max()) + 1
                sub = sub.sort_values(["frame", "pt"], kind="stable")
                frames = np.sort(sub["frame"].unique())
                T = len(frames)
                d["frames"] = frames
                vals = sub[["x", "y", "z"]].to_numpy(np.float64).reshape(T, rows, 3)
                flat = vals.reshape(T, -1)[:, :count]
                d[name] = flat.reshape(T, 3, 3) if name == "object_rotation" else flat
            else:
                rows = int(sub["pt"].max()) + 1
                vals = sub[["x", "y", "z"]].to_numpy(np.float64).reshape(rows, 3)
                if name == "object_scale":
                    d[name] = float(vals[0, 0])
                else:
                    d[name] = vals.reshape(-1)[:count]
        out[ep] = d
    return out


def pack_npz_dir(parquet: Path, out: Path) -> Path:
    """Unpack a parquet into an episode npz dir compatible with our other tooling."""
    out.mkdir(parents=True, exist_ok=True)
    for ep, d in unpack_parquet(parquet).items():
        np.savez(out / f"episode_{ep:06d}.npz", **d)
    return out


# -------------------------------------------------------------------------- forward

_MHR_CACHE: dict[str, object] = {}


def forward_chain(data: dict, device: str = "cpu", chunk: int = 128, precision: str = "float64"):
    """Episode param dict -> (verts_m [T,18439,3] f32, joints_m [T,127,3] f32).

    float64 model forward by default: the kit's own body runs float64, and float32
    skinning adds ~0.1 mm noise to the Chamfer floor."""
    pose = torch.as_tensor(data["mhr_pose"], dtype=torch.float64, device=device)
    z = torch.as_tensor(np.concatenate([data["mhr_scales"], data["mhr_shape"]])[None],
                        dtype=torch.float64, device=device)
    key = (device, precision)
    if key not in _MHR_CACHE:
        _MHR_CACHE[key] = MHR(str(MODEL), device, chunk=chunk, precision=precision)
    mhr = _MHR_CACHE[key]
    v, j = mhr.run(pose, z)
    verts = (v.reshape(len(pose), -1, 3) / 1000.0).float()      # mm -> m
    joints = j.reshape(len(pose), -1, 3).float()
    return verts, joints


def sim3(src: np.ndarray, dst: np.ndarray):
    """Umeyama similarity src->dst (no reflection). Returns (s, R [3,3], t [3])."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    cov = dc.T @ sc / len(src)
    U, S, Vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1.0, 1.0, d])
    R = U @ D @ Vt
    var = (sc ** 2).sum() / len(src)
    s = (S * np.diag(D)).sum() / var
    t = mu_d - s * (R @ mu_s)
    return s, R, t


def apply_sim3(x: np.ndarray, s: float, R: np.ndarray, t: np.ndarray):
    return s * (x @ R.T) + t


# ------------------------------------------------------------------------ reference

def build_ref(human_parquets: list[Path], object_parquet: Path, out: Path,
              base: str = "P2", device: str = "cpu"):
    """Vertex-average independent chains into a pseudo-GT reference dir.

    Objects are taken from `object_parquet` and mapped by the Sim(3) that takes that
    parquet's OWN human into the reference frame, preserving object-to-human placement.
    """
    out.mkdir(parents=True, exist_ok=True)
    chains = {p.stem.replace("track1_cand_", ""): load_episodes(p) for p in human_parquets}
    obj = load_episodes(object_parquet)
    base = base if base in chains else next(iter(chains))
    print(f"chains: {list(chains)} (base={base}); objects: {object_parquet.stem}", flush=True)
    for ep in sorted(chains[base]):
        data_b = chains[base][ep]
        if "mhr_pose" not in data_b:
            continue
        vb, jb = forward_chain(data_b, device)
        mu_base = vb.cpu().numpy().mean(0)
        acc_v, acc_j = [vb.cpu().numpy()], [jb.cpu().numpy()]
        for name, per_ep in chains.items():
            if name == base or ep not in per_ep or "mhr_pose" not in per_ep[ep]:
                continue
            v, j = forward_chain(per_ep[ep], device)
            s, R, t = sim3(v.cpu().numpy().mean(0), mu_base)
            acc_v.append(apply_sim3(v.cpu().numpy(), s, R, t))
            acc_j.append(apply_sim3(j.cpu().numpy(), s, R, t))
        ref_v = np.mean(acc_v, 0).astype(np.float32)
        ref_j = np.mean(acc_j, 0).astype(np.float32)
        # objects: map the object parquet's own human into the reference frame
        o = obj[ep]
        vo, _ = forward_chain(o, device)
        s, R, t = sim3(vo.cpu().numpy().mean(0), ref_v.mean(0))
        R_o = np.einsum("ij,ajk->aik", R, o["object_rotation"])
        t_o = apply_sim3(o["object_translation"], s, R, t)
        np.savez(out / f"episode_{ep:06d}.npz",
                 ref_vertices=ref_v, ref_joints=ref_j,
                 object_rotation=R_o.astype(np.float32),
                 object_translation=t_o.astype(np.float32),
                 object_scale=float(o["object_scale"]) * float(s),
                 object_mesh_vertices=o["object_mesh_vertices"].astype(np.float32),
                 object_mesh_faces=o["object_mesh_faces"])
        print(f"  ep{ep}: T={len(ref_v)} chains={len(acc_v)} obj_scale'={float(o['object_scale'])*float(s):.4f}",
              flush=True)
    print(f"wrote {out}")


# --------------------------------------------------------------------------- scoring

def object_surface_points(mesh_v, mesh_f, scale, rot, trans, n_samples, seed, device):
    """Kit's sampler (v2dlb.mhr_submission._sample_mesh_surface), posed."""
    from v2dlb import mhr_submission as ms
    local = ms._sample_mesh_surface(np.asarray(mesh_v, np.float64), np.asarray(mesh_f, np.int64),
                                    n_samples, seed)
    X = np.einsum("pi,tij->tpj", local, np.asarray(rot, np.float64)) * scale \
        + np.asarray(trans, np.float64)[:, None]
    return torch.as_tensor(X, dtype=torch.float32, device=device)


def score_candidate(cand: Path, ref_dir: Path, episodes=None, device="cpu",
                    n_obj_samples=18439, mesh_override: Path | None = None):
    """Returns ({metric: mean}, {ep: {metric: value}}) replicating the kit's
    `_episode_metrics` exactly: series CDs + stretch-wise acceleration means."""
    from v2dlb import mhr_submission as ms
    data = load_episodes(Path(cand))
    eps = episodes if episodes is not None else sorted(set(data) & {int(p.stem.split("_")[1])
                                                                    for p in ref_dir.glob("episode_*.npz")})
    per_ep, agg = {}, {}
    m = kit()
    for ep in eps:
        ref = np.load(ref_dir / f"episode_{ep:06d}.npz")
        d = data[ep]
        v, j = forward_chain(d, device)
        rot = d["object_rotation"]
        trans = d["object_translation"]
        scale = float(np.asarray(d["object_scale"]).reshape(()))
        mv = d.get("object_mesh_vertices")
        mf = d.get("object_mesh_faces")
        if mv is None and mesh_override is not None:
            import trimesh
            mm = trimesh.load(mesh_override / f"episode_{ep:06d}_object.glb", force="mesh")
            mv, mf = np.asarray(mm.vertices), np.asarray(mm.faces)
        pred = {
            "mhr_vertices": v, "mhr_joints": j,
            "human_surface_points": v,                       # all-vertex role substitute
            "object_surface_points": object_surface_points(mv, mf, scale, rot, trans,
                                                           n_obj_samples, ep, device),
            "object_translation": torch.as_tensor(np.ascontiguousarray(trans), dtype=torch.float32,
                                                  device=device),
        }
        tgt = {
            "mhr_vertices": torch.as_tensor(np.ascontiguousarray(ref["ref_vertices"]), device=device),
            "mhr_joints": torch.as_tensor(np.ascontiguousarray(ref["ref_joints"]), device=device),
            "human_surface_points": torch.as_tensor(np.ascontiguousarray(ref["ref_vertices"]), device=device),
            "object_surface_points": object_surface_points(
                ref["object_mesh_vertices"], ref["object_mesh_faces"],
                float(ref["object_scale"]), ref["object_rotation"],
                ref["object_translation"], n_obj_samples, ep, device),
            "object_translation": torch.as_tensor(np.ascontiguousarray(ref["object_translation"]),
                                                  dtype=torch.float32, device=device),
        }
        # kit-exact math without the (unscored, costly) combined Chamfer:
        transform = m.fit_similarity_transform(pred["mhr_vertices"][0], tgt["mhr_vertices"][0])
        pred_h = m.apply_similarity_transform(pred["human_surface_points"], transform)
        pred_os = m.apply_similarity_transform(pred["object_surface_points"], transform)
        cd_h = m.symmetric_chamfer_distance(pred_h, tgt["human_surface_points"], chunk_size=4096) * 100.0
        cd_o = m.symmetric_chamfer_distance(pred_os, tgt["object_surface_points"], chunk_size=4096) * 100.0
        pred_joints = m._select_body_joints(m.apply_similarity_transform(pred["mhr_joints"], transform), None)
        pred_trans = m.apply_similarity_transform(pred["object_translation"], transform)
        frames = d.get("frames")
        if frames is None:
            frames = np.arange(len(v))
        acc_h, acc_o = [], []
        for a, b in _stretches(np.asarray(frames)):
            if b - a < 3:
                continue
            acc_h.append(m.acceleration_error_per_frame(pred_joints[a:b], tgt["mhr_joints"][a:b]))
            acc_o.append(m.acceleration_error_per_frame(pred_trans[a:b], tgt["object_translation"][a:b]))
        acc_h = torch.cat(acc_h) * 100.0 if acc_h else torch.zeros(1, device=device)
        acc_o = torch.cat(acc_o) * 100.0 if acc_o else torch.zeros(1, device=device)
        per_ep[ep] = {"cd_h_cm": float(cd_h.mean()), "cd_o_cm": float(cd_o.mean()),
                      "acc_h_cm": float(acc_h.mean()), "acc_o_cm": float(acc_o.mean())}
    for k in ("cd_h_cm", "cd_o_cm", "acc_h_cm", "acc_o_cm"):
        if per_ep:
            agg[k] = float(np.mean([v[k] for v in per_ep.values()]))
    return agg, per_ep


def _stretches(frames):
    breaks = np.flatnonzero(np.diff(frames) != 1) + 1
    edges = [0, *breaks.tolist(), len(frames)]
    return [(a, b) for a, b in zip(edges[:-1], edges[1:])]


# ---------------------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("unpack")
    p.add_argument("--parquet", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)

    p = sub.add_parser("build-ref")
    p.add_argument("--humans", type=Path, nargs="+", required=True)
    p.add_argument("--objects", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--base", default="P2")
    p.add_argument("--device", default="cpu")

    p = sub.add_parser("score")
    p.add_argument("--cand", type=Path, required=True)
    p.add_argument("--ref", type=Path, required=True)
    p.add_argument("--episodes", type=int, nargs="*", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", type=Path, default=None)

    p = sub.add_parser("calibrate")
    p.add_argument("--ref", type=Path, required=True)
    p.add_argument("--pairs", type=Path, required=True, help="json: [{file, kaggle:{cd_h_cm,...}}]")
    p.add_argument("--episodes", type=int, nargs="*", default=None)
    p.add_argument("--device", default="cpu")

    a = ap.parse_args()
    if a.cmd == "unpack":
        pack_npz_dir(a.parquet, a.out)
        print(f"wrote {a.out}")
    elif a.cmd == "build-ref":
        build_ref(a.humans, a.objects, a.out, a.base, a.device)
    elif a.cmd == "score":
        agg, per_ep = score_candidate(a.cand, a.ref, a.episodes, a.device)
        print(json.dumps({"mean": agg, "per_episode": per_ep}, indent=1))
        if a.out:
            a.out.write_text(json.dumps({"mean": agg, "per_episode": per_ep}, indent=1))
    elif a.cmd == "calibrate":
        pairs = json.loads(Path(a.pairs).read_text())
        rows = []
        for p_ in pairs:
            agg, _ = score_candidate(Path(p_["file"]), a.ref, a.episodes, a.device)
            row = {"file": Path(p_["file"]).stem, "local": agg, "kaggle": p_["kaggle"]}
            rows.append(row)
            print(f"{row['file']:>28s} local " +
                  " ".join(f"{k}={agg.get(k, float('nan')):.5f}" for k in sorted(agg)), flush=True)
        Path("outputs/calibration_raw.json").write_text(json.dumps(rows, indent=1))
        # delta regressions: for each metric, kaggle_delta vs local_delta over all pairs
        print("\n== delta calibration (Kaggle_delta = c * local_delta + b) ==")
        for metric in ("cd_h_cm", "cd_o_cm", "acc_h_cm", "acc_o_cm"):
            pts = []
            for i, r1 in enumerate(rows):
                for r2 in rows[i + 1:]:
                    if metric in r1["kaggle"] and metric in r2["kaggle"] \
                       and metric in r1["local"] and metric in r2["local"]:
                        pts.append((r2["local"][metric] - r1["local"][metric],
                                    r2["kaggle"][metric] - r1["kaggle"][metric]))
            if len(pts) < 4:
                print(f"{metric}: too few pairs ({len(pts)})")
                continue
            x = np.array([p[0] for p in pts]); y = np.array([p[1] for p in pts])
            keep = np.isfinite(x) & np.isfinite(y)
            x, y = x[keep], y[keep]
            c, b = np.polyfit(x, y, 1)
            pred = c * x + b
            r = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 else float("nan")
            print(f"{metric}: c={c:.3f} b={b:.4f} corr={r:.3f} n={len(x)} "
                  f"resid_rms={float(np.sqrt(((y - pred) ** 2).mean())):.5f}")


if __name__ == "__main__":
    main()
