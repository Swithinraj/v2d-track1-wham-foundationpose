#!/usr/bin/env python3
"""Learned temporal refiner, motion head (mhr_converter.sif, GPU): time-varying, per-region corrections to E2's band
coefficients, trained on real pipeline errors (ext/gru_cache_export.py) against multi-view GT in the evaluator's frame.

Per frame, bands of OUR refined and the RAW per-frame trajectory (Gaussian differences sigma 1/2/4/8, lateral = camera
x/y, depth = camera z) of the 22 scored joints. The model sees band disagreements (raw - ours), our band energies,
ViTPose confidence and mask visibility; a 2-layer bidirectional GRU (128/direction) outputs, per body region (torso,
L/R arm, L/R leg), band B0-B3 and axis group, bounded corrections (0.5 tanh) to E2's half-strength coefficients
(ours k, raw m); B4 stays ours. The last layer is zero-initialised: the model starts exactly at E2.
  fused = ours + ramp * sum_b [(k_b - 1) B_b(ours) + m_b B_b(raw)]      (ramp = the deployment's anchor taper)
Loss on QC-valid frames: |fused - gt| (cm) + W_ACC |acc(fused) - acc(gt)| (cm/frame^2, contiguous valid triples)
+ W_TV |d coef / dt|. Validation on whole episodes reports the scorer's ACC-H formula and mean joint error for
baseline (ours), E2 and the GRU, plus errors within 10 frames of the anchor separately.
Training data: FORM-HOI (nvidia/form-hoi, CC BY 4.0) sequences processed by this pipeline (no challenge data).
Usage: gru_train.py CACHE_DIR OUT_DIR --train ROOT[,ROOT] --val ROOT[,ROOT] [--steps N] [--eval_only MODEL]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

SIGMAS = (1, 2, 4, 8)
NB = 5
REGION = np.array([0, 3, 4, 0, 3, 4, 0, 3, 4, 0, 3, 4, 0, 1, 2, 0, 1, 2, 1, 2, 1, 2])   # per scored joint (SMPL order)
NREG = 5
RAMP = 10
W_ACC, W_TV = 50.0, 5.0
SIM_FAIL = 0.0                 # --sim_fail p: share of training windows with simulated failure (raw error x2-6, acc loss only)       # acceleration error weighted to match the position term (pos ~7 cm, acc ~0.13)
SMOOTH = 6.0                   # corrections are Gaussian-smoothed in time (frames): switching weights create acceleration
E2 = json.loads((Path(__file__).resolve().parents[1] / "config/band_fusion_coef.json").read_text())


def gauss_t(x, s):
    r = int(4 * s + 0.5)
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1, dtype=x.dtype, device=x.device) / s) ** 2); k = (k / k.sum()).view(1, 1, -1)
    flat = x.reshape(len(x), -1).T.unsqueeze(1)
    flat = torch.nn.functional.pad(flat, (r, r), mode="replicate")
    return torch.nn.functional.conv1d(flat, k).squeeze(1).T.reshape(x.shape)


def bands(x):
    gs = [x] + [gauss_t(x, s) for s in SIGMAS]
    return torch.stack([gs[i] - gs[i + 1] for i in range(len(SIGMAS))] + [gs[-1]])      # [NB, T, J, 3]


def e2_base():
    """[2 axis, 2 (k, m), NB] half-strength E2 coefficients (B4: 1, 0)."""
    c = np.zeros((2, 2, NB))
    for a, m in enumerate(("lat", "dep")):
        k, r = np.array(E2[m][:NB]), np.array(E2[m][NB:])
        k[-1], r[-1] = 1.0, 0.0
        c[a, 0], c[a, 1] = 1 + 0.5 * (k - 1), 0.5 * r
    return torch.as_tensor(c, dtype=torch.float32)


MIRROR = [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20]    # left <-> right (SMPL order)


def features(bo, br, conf, vis):
    T = bo.shape[1]
    diff = br[:4] - bo[:4]
    return torch.cat([diff[..., :2].norm(dim=-1).permute(1, 0, 2).reshape(T, -1), diff[..., 2].abs().permute(1, 0, 2).reshape(T, -1),
                      bo[:4, ..., :2].norm(dim=-1).permute(1, 0, 2).reshape(T, -1), bo[:4, ..., 2].abs().permute(1, 0, 2).reshape(T, -1),
                      conf, vis], 1)


def augment(w, rng):
    """Physically consistent augmentation of a training window (same transform on inputs and GT; bands are linear filters,
    so the transforms act on them directly): left/right mirror, global scale about the camera (+-10 %), roll about the
    camera axis (+-10 deg), time reversal, raw-error amplitude (raw = gt + alpha (raw - gt), alpha 0.7-1.3), and
    confidence/visibility dropout over random spans (simulated occlusion)."""
    dev = w["ours"].device
    if rng.random() < 0.5:                                           # mirror: flip camera x, swap left/right joints
        idx = torch.as_tensor(MIRROR, device=dev)
        flip = torch.tensor([-1.0, 1.0, 1.0], device=dev)
        for k in ("bo", "br", "bg"):
            w[k] = w[k][:, :, idx] * flip
        for k in ("ours", "gt"):
            w[k] = w[k][:, idx] * flip
        for k in ("conf", "vis"):
            w[k] = w[k][:, idx]
    th = np.radians(rng.uniform(-10, 10)); sc = rng.uniform(0.9, 1.1)  # roll + scale about the camera
    Rm = torch.tensor([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]], dtype=torch.float32, device=dev) * sc
    for k in ("bo", "br", "bg", "ours", "gt"):
        w[k] = w[k] @ Rm.T
    if rng.random() < 0.5:                                           # time reversal (Gaussian bands are symmetric)
        for k in ("bo", "br", "bg"):
            w[k] = w[k].flip(1)
        for k in ("ours", "gt", "conf", "vis", "valid", "ramp"):
            w[k] = w[k].flip(0)
    a = rng.uniform(0.7, 1.3)                                        # raw-error amplitude
    w["sim_failed"] = SIM_FAIL > 0 and rng.random() < SIM_FAIL       # simulated pipeline failure: wild raw input
    if w["sim_failed"]:
        a = rng.uniform(2.0, 6.0)
    w["br"] = w["bg"] + a * (w["br"] - w["bg"])
    if rng.random() < 0.3:                                           # occlusion: confidence/visibility dropout
        T = w["conf"].shape[0]; L = int(rng.integers(10, 61)); s0 = int(rng.integers(0, max(1, T - L)))
        js = torch.as_tensor(rng.choice(22, size=int(rng.integers(2, 7)), replace=False), device=dev)
        w["conf"] = w["conf"].clone(); w["vis"] = w["vis"].clone()
        w["conf"][s0:s0 + L, js] = 0.0; w["vis"][s0:s0 + L, js] = 0.0
    return w


class Episode:
    def __init__(self, path, dev):
        d = np.load(path)
        self.name = path.stem
        self.ours = torch.as_tensor(d["ours"], device=dev); self.raw = torch.as_tensor(d["raw"], device=dev)
        self.gt = torch.as_tensor(d["gt"], device=dev)
        self.valid = torch.as_tensor(d["valid"], device=dev); self.anchor = int(d["anchor"])
        T = len(self.ours)
        self.bo, self.br = bands(self.ours.double()).float(), bands(self.raw.double()).float()     # [NB, T, 22, 3]
        self.bg = bands(self.gt.double()).float()
        self.conf = torch.as_tensor(d["conf"], device=dev); self.vis = torch.as_tensor(d["vis"], device=dev)
        self.feat = features(self.bo, self.br, self.conf, self.vis)                                        # [T, F]
        t = torch.arange(T, dtype=torch.float32, device=dev)
        ramp = torch.clamp((t - self.anchor).abs() / RAMP, 0, 1); self.ramp = 0.5 - 0.5 * torch.cos(np.pi * ramp)


class Refiner(nn.Module):
    def __init__(self, nf, hidden=128):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(nf, hidden), nn.GELU())
        self.gru = nn.GRU(hidden, hidden, num_layers=2, bidirectional=True, batch_first=True)
        self.out = nn.Linear(2 * hidden, NREG * 2 * 2 * 4)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, f):                                   # f [B, T, F] -> corrections [B, T, NREG, 2 axis, 2 (k, m), 4 bands]
        h, _ = self.gru(self.inp(f))
        c = 0.5 * torch.tanh(self.out(h))                   # [B, T, C]
        c = torch.stack([gauss_t(ci, SMOOTH) for ci in c])  # smooth along time
        return c.view(f.shape[0], f.shape[1], NREG, 2, 2, 4)


def fuse(ep, corr, sl, base):
    """corr [T, NREG, 2, 2, 4] or None (E2) for the frame slice sl -> fused joints [T, 22, 3]."""
    return fuse_arrays(ep.bo[:, sl], ep.br[:, sl], ep.ours[sl], ep.ramp[sl], corr, base)


def fuse_arrays(bo, br, ours, ramp, corr, base):
    T = bo.shape[1]
    coef = base.to(bo.device).view(1, 1, 2, 2, NB).expand(T, NREG, 2, 2, NB).clone()
    if corr is not None:
        coef[..., :4] = coef[..., :4] + corr
    cj = coef[:, torch.as_tensor(REGION, device=bo.device)]                                  # [T, 22, 2, 2, NB]
    k, m = cj[:, :, :, 0].permute(3, 0, 1, 2), cj[:, :, :, 1].permute(3, 0, 1, 2)           # [NB, T, 22, 2axis]
    axis = torch.tensor([0, 0, 1], device=bo.device)
    delta = ((k[..., axis] - 1) * bo + m[..., axis] * br).sum(0)
    return ours + ramp[:, None, None] * delta


def accel(x):
    return x[2:] - 2 * x[1:-1] + x[:-2]


def metrics(ep, fused):
    """Scorer-style ACC-H (cm/frame^2, contiguous valid triples) and mean joint error (cm) on valid frames."""
    v = ep.valid
    ok = v[2:] & v[1:-1] & v[:-2]
    acc = (accel(fused) - accel(ep.gt)).norm(dim=-1).mean(-1)[ok].mean() * 100
    pos = (fused - ep.gt).norm(dim=-1).mean(-1)[v].mean() * 100
    near = (torch.arange(len(v), device=v.device) - ep.anchor).abs() <= 10
    okn = ok & near[1:-1]
    accn = (accel(fused) - accel(ep.gt)).norm(dim=-1).mean(-1)[okn].mean() * 100 if okn.any() else torch.tensor(float("nan"))
    return float(acc), float(pos), float(accn)


def evaluate(model, eps, base):
    rows = {}
    with torch.no_grad():
        for ep in eps:
            r = {}
            r["ours"] = metrics(ep, ep.ours)
            r["e2"] = metrics(ep, fuse(ep, None, slice(None), base))
            if model is not None:
                corr = model(((ep.feat - MU) / SD)[None])[0]
                r["gru"] = metrics(ep, fuse(ep, corr, slice(None), base))
            rows[ep.name] = r
    return rows


def summarize(rows, label):
    names = list(next(iter(rows.values())).keys())
    out = {}
    for n in names:
        acc = np.array([r[n][0] for r in rows.values()]); pos = np.array([r[n][1] for r in rows.values()])
        out[n] = (acc.mean(), pos.mean())
    line = " | ".join(f"{n}: ACC {a:.5f} pos {p:.3f}" for n, (a, p) in out.items())
    if "gru" in names:
        d = np.array([r["gru"][0] - r["e2"][0] for r in rows.values()]); dp = np.array([r["gru"][1] - r["e2"][1] for r in rows.values()])
        line += f" || GRU-E2: ACC {d.mean():+.5f} (better {int((d < 0).sum())}/{len(d)}) pos {dp.mean():+.4f} (better {int((dp < 0).sum())}/{len(dp)})"
        dn = np.array([r["gru"][2] - r["e2"][2] for r in rows.values() if np.isfinite(r["gru"][2])])
        line += f" | near-anchor ACC {dn.mean():+.5f}"
    print(f"[{label}] {line}", flush=True)
    return out


def main():
    global MU, SD, AUG, SIM_FAIL
    ap = argparse.ArgumentParser()
    ap.add_argument("cache", type=Path); ap.add_argument("out", type=Path)
    ap.add_argument("--train", required=True); ap.add_argument("--val", required=True)
    ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--window", type=int, default=120)
    ap.add_argument("--eval_only", type=Path)
    ap.add_argument("--no_aug", action="store_true")
    ap.add_argument("--sim_fail", type=float, default=0.0)
    ap.add_argument("--keep_failed", action="store_true", help="train on failed episodes too (acceleration loss only), so the "
                    "model learns to stay conservative on out-of-range inputs")
    args = ap.parse_args(); dev = "cuda"; args.out.mkdir(parents=True, exist_ok=True); AUG = not args.no_aug; SIM_FAIL = args.sim_fail
    torch.manual_seed(0); rng = np.random.default_rng(0)
    pick = lambda roots: sorted(p for r in roots.split(",") for p in args.cache.glob(f"{r}_episode_*.npz"))  # noqa: E731
    tr = [Episode(p, dev) for p in pick(args.train)]; va = [Episode(p, dev) for p in pick(args.val)]
    # exclude catastrophic episodes (pipeline failures: mean joint error > 30 cm) from training, report them in validation
    for e in tr:
        e.failed = float((e.ours - e.gt).norm(dim=-1).mean(-1)[e.valid].mean()) >= 0.30
    tr = [e for e in tr if args.keep_failed or not e.failed]
    print(f"failed episodes in training: {sum(e.failed for e in tr)}", flush=True)
    allf = torch.cat([e.feat for e in tr if not e.failed]); MU, SD = allf.mean(0), allf.std(0) + 1e-6
    base = e2_base()
    model = Refiner(tr[0].feat.shape[1]).to(dev)
    if args.eval_only:
        model.load_state_dict(torch.load(args.eval_only)); summarize(evaluate(model, va, base), "val"); return
    print(f"train episodes {len(tr)}, val {len(va)}, features {tr[0].feat.shape[1]}", flush=True)
    summarize(evaluate(None, va, base), "val before")
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    W = args.window
    BEST = [0.0]                     # validation-selected checkpoint: must beat E2 (score < 0) to be saved
    for step in range(1, args.steps + 1):
        batch = [tr[i] for i in rng.integers(0, len(tr), 16)]
        loss = 0.0
        wins = []
        failed = [ep.failed for ep in batch]
        for ep in batch:
            T = len(ep.ours); s0 = int(rng.integers(0, max(1, T - W))); L = min(W, T - s0)
            sl = slice(s0, s0 + L)
            w = {"bo": ep.bo[:, sl], "br": ep.br[:, sl], "bg": ep.bg[:, sl], "ours": ep.ours[sl], "gt": ep.gt[sl],
                 "conf": ep.conf[sl], "vis": ep.vis[sl], "valid": ep.valid[sl], "ramp": ep.ramp[sl]}
            wins.append(augment(w, rng) if AUG else w)
        L = min(w["ours"].shape[0] for w in wins)
        feats = torch.stack([((features(w["bo"][:, :L], w["br"][:, :L], w["conf"][:L], w["vis"][:L]) - MU) / SD) for w in wins])
        corr = model(feats)
        for b, w in enumerate(wins):
            fused = fuse_arrays(w["bo"][:, :L], w["br"][:, :L], w["ours"][:L], w["ramp"][:L], corr[b], base)
            gt = w["gt"][:L]; v = w["valid"][:L]
            ok = v[2:] & v[1:-1] & v[:-2]
            if int(ok.sum()) < 10:
                continue
            pos = (fused - gt).norm(dim=-1).mean(-1)[v].mean() * 100
            acc = (accel(fused) - accel(gt)).norm(dim=-1).mean(-1)[ok].mean() * 100
            tv = (corr[b][1:] - corr[b][:-1]).abs().mean()
            loss = loss + (0.0 if failed[b] or w.get("sim_failed") else pos) + W_ACC * acc + W_TV * tv
        if not torch.is_tensor(loss):
            continue
        loss = loss / len(batch)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 250 == 0:
            print(f"step {step} loss {float(loss):.4f}", flush=True)
            rows = evaluate(model, va, base); summarize(rows, f"val step {step}")
            score = np.mean([r["gru"][0] - r["e2"][0] for r in rows.values()]) + 0.01 * np.mean([r["gru"][1] - r["e2"][1] for r in rows.values()])
            if score < BEST[0]:
                BEST[0] = score; torch.save(model.state_dict(), args.out / "refiner_motion_best.pt"); print(f"  best at step {step}", flush=True)
    torch.save(model.state_dict(), args.out / "refiner_motion.pt")
    torch.save({"mu": MU.cpu(), "sd": SD.cpu()}, args.out / "refiner_motion_norm.pt")
    rows = evaluate(model, va, base); summarize(rows, "val final")
    (args.out / "val_rows.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
