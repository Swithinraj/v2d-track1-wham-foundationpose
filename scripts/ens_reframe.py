#!/usr/bin/env python3
"""Re-frame objects onto a new human so the scorer places them where the old human had them.

The scorer fits ONE similarity from the submission's first-scored-frame human to the GT's and
applies it to the objects too. Replacing the human (old -> new) therefore moves the objects unless
they are carried along: with T = similarity(old frame 0 -> new frame 0) applied to the objects,
the scorer's placement fit(new_0 -> g0) o T equals fit(old_0 -> g0) up to the part of new_0 that
is not a similarity of old_0 (articulation). This script writes T per episode and measures that
residual: for several proxy GT first frames G (other chains' first frames), the mean displacement
of the posed object surface between fit(old_0->G) and fit(new_0->G) o T ("reframed"), and without
T ("not reframed"), in cm over all scored frames.

Object pose convention (kit metric_code/track_1/CD-O.py): X = scale * R_obj @ local + t.
Under a similarity (s, R, t0): R_obj' = R @ R_obj, t' = s R t + t0, scale' = s * scale.

usage: ens_reframe.py --old DIR --new DIR --proxies NAME=DIR ... --out T.json
                      [--model mhr_model.pt --kit KITDIR]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(dc.T @ sc / len(src))
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1.0, 1.0, d])
    R = U @ D @ Vt
    s = (S * np.diag(D)).sum() / ((sc ** 2).sum() / len(src))
    return s, R, mu_d - s * (R @ mu_s)


def apply(sim, X):
    s, R, t = sim
    return s * X @ R.T + t


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--old", type=Path, required=True, help="npz dir of the old human + objects")
    ap.add_argument("--new", type=Path, required=True, help="npz dir of the new human")
    ap.add_argument("--proxies", nargs="+", required=True, help="NAME=DIR, chains whose first frame stands in for the GT")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--kit", type=Path, required=True)
    ap.add_argument("--episodes", type=int, nargs="*", default=None)
    a = ap.parse_args()
    sys.path.insert(0, str(a.kit / "tools/track1"))
    import mesh_to_mhr_params as m2m
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mhr = m2m.MHR(str(a.model), dev, chunk=64, precision="float64")
    dd = dict(dtype=torch.float64, device=mhr.device)

    def frame0(d):
        v = mhr.verts(torch.as_tensor(d["mhr_pose"][:1], **dd),
                      torch.as_tensor(np.concatenate([d["mhr_scales"], d["mhr_shape"]])[None], **dd))
        return v.reshape(-1, 3).cpu().numpy() / 1000.0                       # mm -> m

    proxies = dict(p.split("=", 1) for p in a.proxies)
    eps = a.episodes or sorted(int(p.stem.split("_")[1]) for p in a.new.glob("episode_*.npz"))
    out, diag = {}, {}
    for ep in eps:
        old = dict(np.load(a.old / f"episode_{ep:06d}.npz"))
        new = dict(np.load(a.new / f"episode_{ep:06d}.npz"))
        assert np.array_equal(old["frames"], new["frames"])
        o0, n0 = frame0(old), frame0(new)
        T = umeyama(o0, n0)
        out[str(ep)] = {"s": float(T[0]), "R": T[1].tolist(), "t": T[2].tolist(),
                        "frame0_residual_cm": float(np.linalg.norm(apply(T, o0) - n0, axis=1).mean() * 100)}
        # posed object surface (kit convention), old frame
        rng = np.random.default_rng(ep)
        mv = np.asarray(old["object_mesh_vertices"], float)
        local = mv[rng.choice(len(mv), size=min(512, len(mv)), replace=False)]
        Ro, to, so = old["object_rotation"], old["object_translation"], float(old["object_scale"])
        X = so * np.einsum("tij,pj->tpi", Ro, local) + to[:, None]           # [T,P,3]
        Xr = apply(T, X.reshape(-1, 3)).reshape(X.shape)
        rec = {}
        for name, path in proxies.items():
            g0 = frame0(dict(np.load(Path(path) / f"episode_{ep:06d}.npz")))
            A1, A2 = umeyama(o0, g0), umeyama(n0, g0)
            P1 = apply(A1, X.reshape(-1, 3))
            rec[name] = {"reframed_cm": float(np.linalg.norm(apply(A2, Xr.reshape(-1, 3)) - P1, axis=1).mean() * 100),
                         "not_reframed_cm": float(np.linalg.norm(apply(A2, X.reshape(-1, 3)) - P1, axis=1).mean() * 100)}
        diag[str(ep)] = rec
        print(f"ep{ep}: s={T[0]:.4f} rot={np.degrees(np.arccos(np.clip((np.trace(T[1]) - 1) / 2, -1, 1))):.2f}deg "
              f"|t|={np.linalg.norm(T[2]) * 100:.2f}cm f0res={out[str(ep)]['frame0_residual_cm']:.2f}cm | "
              + " ".join(f"{k}: {v['reframed_cm']:.3f}/{v['not_reframed_cm']:.3f}" for k, v in rec.items()), flush=True)
    a.out.write_text(json.dumps({"sim3": out, "diagnostics": diag}, indent=1))
    allr = [v["reframed_cm"] for e in diag.values() for v in e.values()]
    alln = [v["not_reframed_cm"] for e in diag.values() for v in e.values()]
    print(f"object displacement under the scorer's alignment (cm): reframed mean {np.mean(allr):.3f} max {np.max(allr):.3f}"
          f" | not reframed mean {np.mean(alln):.3f} max {np.max(alln):.3f}", flush=True)


if __name__ == "__main__":
    main()
