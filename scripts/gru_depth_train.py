#!/usr/bin/env python3
"""Learned depth fusion (mhr_converter.sif, GPU): the GRU motion head (scripts/gru_train.py) fine-tuned from a trained model
with GT-free per-frame depth cues (scripts/depth_cues.py) and a depth head.

Inputs: the motion head's features (normalised with ITS training statistics) + cue features [anchor-relative MoGe and
apparent-size cues, floor-contact cue, their validity flags, the anchor ramp] (own statistics). The first layer is the
trained model's with zero columns for the cues and the depth head is zero-initialised, so training starts EXACTLY at
the trained model (checked at start: the script aborts if the outputs differ).
Depth head: eps(t) = 0.1 tanh(.), Gaussian-smoothed in time (SMOOTH_D frames); the whole body moves along the camera ray
through OUR input centroid c(t):  J'(t) = fused(t) + ramp_d(t) eps(t) |c(t)| c(t)/|c(t)|   (ramp_d = a cosine anchor
taper over RAMP_D frames, slower than the motion head's: the scorer aligns at the anchor and drift grows gradually)
i.e. our distance is scaled by (1 + ramp eps) without changing body size. Loss as the motion head (position + W_ACC x
acceleration error vs GT, coefficient TV) + W_TV x 100 |d eps / dt|. Augmentation as the motion head (mirror, roll / scale
about the camera, time reversal, raw-error amplitude, simulated failures, occlusion), cues reversed with time (they are
relative depth errors: invariant to mirror / roll / scale), plus cue dropout over random spans.
Selection on VAL: score = d ACC + 0.01 d pos vs the starting model, saved only if < 0 and d ACC <= +0.0001.
Usage: gru_depth_train.py CACHE_DIR CUE_DIR OUT_DIR --init DIR --train R,R --val R [--steps N] [--eval_only PT]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import gru_train as G

SMOOTH_D = 15.0
RAMP_D = 45.0                  # depth-correction anchor taper (frames): slower than the motion head's (a switch-on spike costs ACC)
EPS_SCALE = 1.0                # --eps_scale (eval only): shrink the predicted depth correction
CUE_SET = "v1"                 # --cue_set: v1 = per-frame MoGe + RANSAC floor (first models); v4 = locked MoGe + plane-free foot/background
# columns: (key, "value" | "valid"); the anchor ramp is always appended last. groups: columns dropped together (cue
# dropout); swap: column pairs exchanged by the left/right mirror augmentation.
CUE_SETS = {
    "v1": {"cols": [("c_moge_rel", "value"), ("c_size_rel", "value"), ("c_floor", "value"), ("c_moge_rel", "valid"),
                    ("c_size_rel", "valid"), ("st_floor", "value")],
           "groups": [(0, 3), (1, 4), (2, 5)], "swap": []},
    "v4": {"cols": [("c_moge_rel", "value"), ("c_size_rel", "value"), ("c_bg_l_rel", "value"), ("c_bg_r_rel", "value"),
                    ("c_bg_l", "value"), ("c_bg_r", "value"), ("c_moge_rel", "valid"), ("c_size_rel", "valid"),
                    ("vb_l", "value"), ("vb_r", "value"), ("sp_l", "value"), ("sp_r", "value")],
           "groups": [(0, 6), (1, 7), (2, 4, 8, 10), (3, 5, 9, 11)], "swap": [(2, 3), (4, 5), (8, 9), (10, 11)]},
}


def ncue():
    return len(CUE_SETS[CUE_SET]["cols"]) + 1


def cue_tensor(cz, ramp):
    """cz: dict of [T] arrays (scripts/depth_cues.py; set v4 = an unshipped background-locking experiment) -> [T, ncue()] for CUE_SET (NaN -> 0, validity
    flags explicit), the anchor ramp last."""
    cols = []
    for k, kind in CUE_SETS[CUE_SET]["cols"]:
        x = np.asarray(cz[k], np.float32)
        cols.append(torch.as_tensor(np.nan_to_num(x) if kind == "value" else np.isfinite(x).astype(np.float32), device=ramp.device))
    return torch.stack(cols + [ramp], 1)


def apply_depth(fused, ours, ramp, eps):
    """fused / ours [T, 22, 3] (our camera frame), ramp / eps [T] -> fused moved along the ray through ours' centroid."""
    c = ours.mean(1)
    return fused + ((ramp * eps)[:, None] * c)[:, None, :]                       # (ramp eps) |c| c / |c| = ramp eps c


class Episode(G.Episode):
    def __init__(self, path, cue_dir, dev):
        super().__init__(path, dev)
        cz = np.load(cue_dir / path.name)
        assert len(cz["c_size_rel"]) == len(self.ours), f"cue length mismatch {path.name}"
        t = torch.arange(len(self.ours), dtype=torch.float32, device=dev)
        rd = torch.clamp((t - self.anchor).abs() / RAMP_D, 0, 1); self.ramp_d = 0.5 - 0.5 * torch.cos(np.pi * rd)
        self.cue = cue_tensor(cz, self.ramp_d)


class DepthRefiner(nn.Module):
    def __init__(self, nf_base, n_cue, hidden=128):
        super().__init__()
        self.nf_base = nf_base
        self.inp = nn.Sequential(nn.Linear(nf_base + n_cue, hidden), nn.GELU())
        self.gru = nn.GRU(hidden, hidden, num_layers=2, bidirectional=True, batch_first=True)
        self.out = nn.Linear(2 * hidden, G.NREG * 2 * 2 * 4)
        self.dhead = nn.Linear(2 * hidden, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias); nn.init.zeros_(self.dhead.weight); nn.init.zeros_(self.dhead.bias)

    def load_base(self, sd):
        with torch.no_grad():
            self.inp[0].weight.zero_(); self.inp[0].weight[:, :self.nf_base] = sd["inp.0.weight"]; self.inp[0].bias.copy_(sd["inp.0.bias"])
            for k, v in sd.items():
                if k.startswith("gru.") or k.startswith("out."):
                    dict(self.named_parameters())[k].copy_(v)

    def forward(self, f):
        h, _ = self.gru(self.inp(f))
        c = 0.5 * torch.tanh(self.out(h)); c = torch.stack([G.gauss_t(ci, G.SMOOTH) for ci in c])
        e = 0.1 * torch.tanh(self.dhead(h)); e = torch.stack([G.gauss_t(ei, SMOOTH_D) for ei in e])[..., 0]
        return c.view(f.shape[0], f.shape[1], G.NREG, 2, 2, 4), e


def augment(w, rng):
    """G.augment's transforms with the cues: reversed with time, dropped over random spans (relative depth errors are
    invariant to mirror, roll and scale about the camera)."""
    dev = w["ours"].device
    if rng.random() < 0.5:
        idx = torch.as_tensor(G.MIRROR, device=dev); flip = torch.tensor([-1.0, 1.0, 1.0], device=dev)
        for k in ("bo", "br", "bg"):
            w[k] = w[k][:, :, idx] * flip
        for k in ("ours", "gt"):
            w[k] = w[k][:, idx] * flip
        for k in ("conf", "vis"):
            w[k] = w[k][:, idx]
        sw = CUE_SETS[CUE_SET]["swap"]                                 # left/right cue columns follow the mirrored body
        if sw:
            perm = list(range(w["cue"].shape[1]))
            for i, j in sw:
                perm[i], perm[j] = j, i
            w["cue"] = w["cue"][:, torch.as_tensor(perm, device=dev)]
    th = np.radians(rng.uniform(-10, 10)); sc = rng.uniform(0.9, 1.1)
    Rm = torch.tensor([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]], dtype=torch.float32, device=dev) * sc
    for k in ("bo", "br", "bg", "ours", "gt"):
        w[k] = w[k] @ Rm.T
    if rng.random() < 0.5:
        for k in ("bo", "br", "bg"):
            w[k] = w[k].flip(1)
        for k in ("ours", "gt", "conf", "vis", "valid", "ramp", "ramp_d", "cue"):
            w[k] = w[k].flip(0)
    a = rng.uniform(0.7, 1.3)
    w["sim_failed"] = G.SIM_FAIL > 0 and rng.random() < G.SIM_FAIL
    if w["sim_failed"]:
        a = rng.uniform(2.0, 6.0)
    w["br"] = w["bg"] + a * (w["br"] - w["bg"])
    if rng.random() < 0.3:
        T = w["conf"].shape[0]; L = int(rng.integers(10, 61)); s0 = int(rng.integers(0, max(1, T - L)))
        js = torch.as_tensor(rng.choice(22, size=int(rng.integers(2, 7)), replace=False), device=dev)
        w["conf"] = w["conf"].clone(); w["vis"] = w["vis"].clone()
        w["conf"][s0:s0 + L, js] = 0.0; w["vis"][s0:s0 + L, js] = 0.0
    if rng.random() < 0.3:                                            # cue dropout: one cue group (values and flags) over a span
        T = w["cue"].shape[0]; L = int(rng.integers(20, 121)); s0 = int(rng.integers(0, max(1, T - L)))
        grp = CUE_SETS[CUE_SET]["groups"]; g = grp[int(rng.integers(0, len(grp)))]
        w["cue"] = w["cue"].clone(); w["cue"][s0:s0 + L, list(g)] = 0.0
    w["cue"] = w["cue"].clone(); w["cue"][:, -1] = w["ramp_d"]          # the ramp feature follows the (possibly reversed) depth ramp
    return w


def run(model, ep, MUc, SDc, base):
    f = torch.cat([(ep.feat - G.MU) / G.SD, (ep.cue - MUc) / SDc], 1)
    corr, eps = model(f[None]); eps = eps * EPS_SCALE                       # EPS_SCALE: post-hoc shrinkage (1 = as trained)
    fused = G.fuse(ep, corr[0], slice(None), base)
    return apply_depth(fused, ep.ours, ep.ramp_d, eps[0]), eps[0]


def evaluate(model, ref, eps_list, MUc, SDc, base):
    rows = {}
    with torch.no_grad():
        for ep in eps_list:
            r = {"ours": G.metrics(ep, ep.ours), "e2": G.metrics(ep, G.fuse(ep, None, slice(None), base))}
            r["gru"] = G.metrics(ep, G.fuse(ep, ref(((ep.feat - G.MU) / G.SD)[None])[0], slice(None), base))
            if model is not None:
                out, e = run(model, ep, MUc, SDc, base); r["grud"] = G.metrics(ep, out); r["eps_abs"] = float(e.abs().mean())
            rows[ep.name] = r
    return rows


def summarize(rows, label):
    d = lambda a, b, i: np.array([r[a][i] - r[b][i] for r in rows.values()])  # noqa: E731
    line = f"[{label}] GRU-E2 ACC {d('gru', 'e2', 0).mean():+.5f} pos {d('gru', 'e2', 1).mean():+.4f}"
    if "grud" in next(iter(rows.values())):
        da, dp = d("grud", "gru", 0), d("grud", "gru", 1)
        dn = np.array([r["grud"][2] - r["gru"][2] for r in rows.values() if np.isfinite(r["grud"][2])])
        line += (f" || DEPTH-GRU: ACC {da.mean():+.5f} (better {int((da < 0).sum())}/{len(da)}) pos {dp.mean():+.4f} cm (better {int((dp < 0).sum())}/{len(dp)})"
                 f" | near-anchor ACC {dn.mean():+.5f} | |eps| {np.mean([r['eps_abs'] for r in rows.values()]) * 100:.2f} %")
    print(line, flush=True)
    return line


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cache", type=Path); ap.add_argument("cues", type=Path); ap.add_argument("out", type=Path)
    ap.add_argument("--init", type=Path, required=True, help="trained motion-head dir (refiner_motion_best.pt + norm)")
    ap.add_argument("--train", required=True); ap.add_argument("--val", required=True)
    ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--window", type=int, default=120)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--sim_fail", type=float, default=0.15)
    ap.add_argument("--smooth_d", type=float, default=15.0); ap.add_argument("--ramp_d", type=float, default=45.0); ap.add_argument("--eval_only", type=Path)
    ap.add_argument("--eps_scales", default="1", help="eval only: comma list of depth-correction shrink factors to report")
    ap.add_argument("--cue_set", default="v1", choices=sorted(CUE_SETS))
    args = ap.parse_args(); dev = "cuda"; args.out.mkdir(parents=True, exist_ok=True)
    global SMOOTH_D, RAMP_D, CUE_SET, EPS_SCALE
    SMOOTH_D, RAMP_D, CUE_SET = args.smooth_d, args.ramp_d, args.cue_set; G.SIM_FAIL = args.sim_fail
    ck = None
    if args.eval_only:                                                     # the checkpoint's settings BEFORE episodes are built
        ck = torch.load(args.eval_only)
        SMOOTH_D, RAMP_D, CUE_SET = float(ck["smooth_d"]), float(ck["ramp_d"]), ck.get("cue_set", "v1")
    torch.manual_seed(0); rng = np.random.default_rng(0)
    nrm = torch.load(args.init / "refiner_motion_norm.pt"); G.MU, G.SD = nrm["mu"].to(dev), nrm["sd"].to(dev)
    pick = lambda roots: sorted(p for r in roots.split(",") for p in args.cache.glob(f"{r}_episode_*.npz") if (args.cues / p.name).exists())  # noqa: E731
    tr = [Episode(p, args.cues, dev) for p in pick(args.train)]; va = [Episode(p, args.cues, dev) for p in pick(args.val)]
    for e in tr:
        e.failed = float((e.ours - e.gt).norm(dim=-1).mean(-1)[e.valid].mean()) >= 0.30
    tr = [e for e in tr if not e.failed]
    allc = torch.cat([e.cue for e in tr]); MUc, SDc = allc.mean(0), allc.std(0) + 1e-6
    base = G.e2_base()
    ref = G.Refiner(tr[0].feat.shape[1]).to(dev); ref.load_state_dict(torch.load(args.init / "refiner_motion_best.pt")); ref.eval()
    model = DepthRefiner(tr[0].feat.shape[1], ncue()).to(dev)
    if ck is not None:
        model.load_state_dict(ck["model"]); MUc, SDc = ck["mu_c"].to(dev), ck["sd_c"].to(dev); model.eval()
        print(f"eval only: cue set {CUE_SET}, smooth_d {SMOOTH_D}, ramp_d {RAMP_D}, checkpoint step {ck['step']}", flush=True)
        for k in map(float, args.eps_scales.split(",")):
            EPS_SCALE = k; rows = evaluate(model, ref, va, MUc, SDc, base); summarize(rows, f"val (eval only) eps x{k:g}")
            dp = np.array([r["grud"][1] - r["gru"][1] for r in rows.values()]); o = np.sort(dp)
            print(f"   position delta per episode: median {np.median(dp):+.3f} cm, worst {o[-1]:+.3f}, 2nd worst {o[-2]:+.3f}, best {o[0]:+.3f}, "
                  f"mean without best {np.mean(o[1:]):+.3f}, mean without best 3 {np.mean(o[3:]):+.3f}", flush=True)
        return
    model.load_base(torch.load(args.init / "refiner_motion_best.pt"))
    with torch.no_grad():                                                  # exact-start check
        for ep in va[:3]:
            out, e = run(model, ep, MUc, SDc, base); g = G.fuse(ep, ref(((ep.feat - G.MU) / G.SD)[None])[0], slice(None), base)
            dmax = float((out - g).abs().max()); assert dmax < 1e-5 and float(e.abs().max()) == 0.0, f"start differs from the trained model: {dmax}"
    print(f"train episodes {len(tr)}, val {len(va)}, base features {tr[0].feat.shape[1]} + cues {ncue()} (set {CUE_SET}); exact start verified", flush=True)
    summarize(evaluate(model, ref, va, MUc, SDc, base), "val start")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr); W = args.window; best = 0.0
    for step in range(1, args.steps + 1):
        batch = [tr[i] for i in rng.integers(0, len(tr), 16)]; wins = []
        for ep in batch:
            T = len(ep.ours); s0 = int(rng.integers(0, max(1, T - W))); L = min(W, T - s0); sl = slice(s0, s0 + L)
            w = {"bo": ep.bo[:, sl], "br": ep.br[:, sl], "bg": ep.bg[:, sl], "ours": ep.ours[sl], "gt": ep.gt[sl], "conf": ep.conf[sl],
                 "vis": ep.vis[sl], "valid": ep.valid[sl], "ramp": ep.ramp[sl], "ramp_d": ep.ramp_d[sl], "cue": ep.cue[sl]}
            wins.append(augment(w, rng))
        L = min(w["ours"].shape[0] for w in wins)
        f = torch.stack([torch.cat([(G.features(w["bo"][:, :L], w["br"][:, :L], w["conf"][:L], w["vis"][:L]) - G.MU) / G.SD, (w["cue"][:L] - MUc) / SDc], 1) for w in wins])
        corr, eps = model(f); loss = 0.0
        for b, w in enumerate(wins):
            fused = G.fuse_arrays(w["bo"][:, :L], w["br"][:, :L], w["ours"][:L], w["ramp"][:L], corr[b], base)
            fused = apply_depth(fused, w["ours"][:L], w["ramp_d"][:L], eps[b])
            gt = w["gt"][:L]; v = w["valid"][:L]; ok = v[2:] & v[1:-1] & v[:-2]
            if int(ok.sum()) < 10:
                continue
            pos = (fused - gt).norm(dim=-1).mean(-1)[v].mean() * 100
            acc = (G.accel(fused) - G.accel(gt)).norm(dim=-1).mean(-1)[ok].mean() * 100
            tv = (corr[b][1:] - corr[b][:-1]).abs().mean() + (eps[b][1:] - eps[b][:-1]).abs().mean() * 100
            loss = loss + (0.0 if w.get("sim_failed") else pos) + G.W_ACC * acc + G.W_TV * tv
        if not torch.is_tensor(loss):
            continue
        loss = loss / len(batch); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 250 == 0:
            print(f"step {step} loss {float(loss):.4f}", flush=True)
            rows = evaluate(model, ref, va, MUc, SDc, base); summarize(rows, f"val step {step}")
            da = np.mean([r["grud"][0] - r["gru"][0] for r in rows.values()]); dp = np.mean([r["grud"][1] - r["gru"][1] for r in rows.values()])
            score = da + 0.01 * dp
            if score < best and da <= 1e-4:
                best = score; torch.save({"model": model.state_dict(), "mu_c": MUc.cpu(), "sd_c": SDc.cpu(), "step": step, "smooth_d": SMOOTH_D, "ramp_d": RAMP_D, "cue_set": CUE_SET}, args.out / "depth_best.pt")
                print(f"  best at step {step} (score {score:+.5f})", flush=True)
    torch.save({"model": model.state_dict(), "mu_c": MUc.cpu(), "sd_c": SDc.cpu(), "step": args.steps, "smooth_d": SMOOTH_D, "ramp_d": RAMP_D, "cue_set": CUE_SET}, args.out / "depth_final.pt")
    rows = evaluate(model, ref, va, MUc, SDc, base); summarize(rows, "val final")
    (args.out / "val_rows.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
