#!/usr/bin/env python3
"""Depth-reliability module (10-27, pipeline/recipes/rel.md; mhr_converter.sif for torch, GPU): learn WHEN and HOW MUCH to move the
human and the object along their camera rays, from the per-frame evidence of scripts/rel_cache.py, so that specific
observation failures get corrected and good estimates are left alone.

Model (--arch gru | tf): per frame, human features and object features (standardised with TRAIN statistics; NaN -> 0
plus a finite flag per source), each through its own encoder; GRU = bidirectional 2-layer GRU per branch, TF = a 2-layer
temporal Transformer per branch; then an exchange layer (each branch sees the other's state) and separate heads per branch:
  delta = 0.1 tanh(.)     bounded correction (log distance, ±10 %)
  gate  = sigmoid(.)      how much of it to apply (0 = leave the estimate alone)
  logs                    log standard deviation of (target - delta): calibrated uncertainty
Applied correction c(t) = ramp(t) gate(t) delta(t), Gaussian-smoothed (sigma 10 frames); ramp = cosine taper from the
anchor over 45 frames (the scorer aligns at the anchor; the targets are anchor-relative).
Loss per branch on valid frames: Huber(target - c) + W_HARM relu(|target - c| - |target|) (harm: worse than no move) +
W_ACC |d2 c - d2 target| (motion consistency) + W_NLL [0.5 ((target - delta) / s)^2 + log s] (calibrated, NOT allowed to
call everything uncertain: log s is penalised).
Training corruptions, recomputed consistently (an along-ray shift d of OUR estimate lowers every "source / ours" log
ratio and the target by d and adds d' to the along-ray velocity):
  ours_h / ours_o  smooth wrong trajectory of OUR human / object (amplitude up to 5 % / 15 %), anchor-pinned
  src_bias         smooth or drifting bias of ONE source (VDA or MoGe) only, human and / or object (target unchanged)
  occ              occluded frames' source values contaminated
  none             untouched (accurate examples must stay unchanged: the harm term)
Selection on VAL (no corruption): mean of the human and object rms residual ratios (corrected / uncorrected), saved only
if both branches' harm shares stay below 50 %. Reports per branch: rms before / after, accurate / inaccurate halves (by
episode error), harm share (frames made worse), episodes improved.
Usage: rel_train.py --cache DIR [DIR ...] --train PREFIX,... --val PREFIX,... --out DIR [--arch gru|tf] [--steps N]
       rel_train.py ... --eval_only CKPT --val PREFIX,...   (report on any group)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

DEV = "cuda"
W_HARM, W_ACC, W_NLL = 2.0, 20.0, 0.05
SH = {"h_moge": 0, "h_vda": 1, "c_size": 8, "h_vr": 12, "h_ar": 14}      # columns of rel_cache HN that a shift moves
SO = {"o_moge": 0, "o_vda": 1, "o_size": 5, "o_vr": 10, "o_ar": 12}


def load(dirs, prefixes):
    eps = []
    for d in dirs:
        for p in prefixes:
            for f in sorted(Path(d).glob(f"{p}_episode_*.npz")):
                z = np.load(f); eps.append({k: z[k] for k in z.files}); eps[-1]["name"] = f"{Path(d).name}/{f.stem}"
    return eps


def stats(eps):
    H = np.concatenate([e["fh"] for e in eps]); O = np.concatenate([e["fo"] for e in eps])
    return (np.nanmean(H, 0), np.nanstd(H, 0) + 1e-6, np.nanmean(O, 0), np.nanstd(O, 0) + 1e-6)


def prep(x, mu, sd):
    fin = np.isfinite(x); z = np.where(fin, (x - mu) / sd, 0.0)
    return np.concatenate([np.clip(z, -6, 6), fin[:, :2].astype(np.float64)], 1)     # + finite flags of the two depth sources


def smooth_noise(T, amp, rng, sigma):
    k = int(4 * sigma); g = np.exp(-0.5 * (np.arange(-k, k + 1) / sigma) ** 2); g /= g.sum()
    n = np.convolve(rng.standard_normal(T + 2 * k), g, mode="same")[k:k + T]; n /= max(np.abs(n).max(), 1e-9)
    return amp * n


def corrupt(e, rng):
    """-> (fh, fo, tgt_h, tgt_o) with one of the targeted failure modes applied consistently."""
    fh, fo, th, to = e["fh"].astype(np.float64).copy(), e["fo"].astype(np.float64).copy(), e["tgt_h"].astype(np.float64).copy(), e["tgt_o"].astype(np.float64).copy()
    T, a = len(th), int(e["anchor"]); mode = rng.choice(["none", "ours_h", "ours_o", "both", "src_bias", "occ"], p=[0.3, 0.15, 0.15, 0.1, 0.2, 0.1])
    hn = [str(x) for x in e["h_names"]] if "h_names" in e else []; on = [str(x) for x in e["o_names"]] if "o_names" in e else []
    if mode in ("ours_h", "both"):
        d = smooth_noise(T, rng.uniform(0, 0.05), rng, rng.uniform(20, 120)); d -= d[a]
        # our distance up by d: every "source / ours" log ratio and every "our projected size / observed size" ratio goes down
        for c in ("h_moge", "h_vda", "c_size", "h_size_sm", "h_size_abs", "h_size_vis", "h_size_vis_sm"):
            if c in hn or (not hn and c in SH):
                fh[:, hn.index(c) if hn else SH[c]] -= d
        fh[:, SH["h_vr"]] += np.gradient(d); fh[:, SH["h_ar"]] += np.gradient(np.gradient(d)); th -= d
    if mode in ("ours_o", "both"):
        d = smooth_noise(T, rng.uniform(0, 0.15), rng, rng.uniform(20, 120)); d -= d[a]
        for c in ("o_moge", "o_vda"):                                  # "source depth / ours": down
            fo[:, on.index(c) if on else SO[c]] -= d
        for c in ("o_size", "o_size_sm", "o_sil_vis", "o_sil_vis_sm"):  # "observed size / our projected size": UP (v1 had the sign wrong)
            if c in on or (not on and c in SO):
                fo[:, on.index(c) if on else SO[c]] += d
        fo[:, SO["o_vr"]] += np.gradient(d); fo[:, SO["o_ar"]] += np.gradient(np.gradient(d)); to -= d
    if mode == "src_bias":
        src = rng.integers(2); b = smooth_noise(T, rng.uniform(0.01, 0.08), rng, rng.uniform(30, 200)) if rng.random() < 0.6 \
            else np.linspace(0, rng.uniform(-0.08, 0.08), T)
        b -= b[a]
        if rng.random() < 0.6:
            fh[:, src] += b
        if rng.random() < 0.6:
            fo[:, src] += b * 2
    if mode == "occ":
        occ = fh[:, 4] > 0.2
        fh[occ, :2] += rng.normal(0, 0.05, (occ.sum(), 2)); fo[fo[:, 8] > 0.3, :2] += rng.normal(0, 0.15, ((fo[:, 8] > 0.3).sum(), 2))
    fh[:, 2] = fh[:, 1] - fh[:, 0]; fo[:, 2] = fo[:, 1] - fo[:, 0]                      # disagreement recomputed
    return fh, fo, th, to


class Branch(nn.Module):
    def __init__(self, fin, arch, h=64):
        super().__init__(); self.inp = nn.Linear(fin + 1, h); self.arch = arch
        if arch == "gru":
            self.enc = nn.GRU(h, h, num_layers=2, batch_first=True, bidirectional=True); self.out_dim = 2 * h
        else:
            layer = nn.TransformerEncoderLayer(h, 4, 4 * h, 0.1, batch_first=True); self.enc = nn.TransformerEncoder(layer, 2); self.out_dim = h

    def forward(self, x, ramp):
        z = self.inp(torch.cat([x, ramp[..., None]], -1))
        if self.arch == "gru":
            return self.enc(z)[0]
        T = z.shape[1]; pos = torch.arange(T, device=z.device)[:, None] / torch.pow(10000, torch.arange(0, z.shape[2], 2, device=z.device) / z.shape[2])
        pe = torch.zeros_like(z[0]); pe[:, 0::2] = torch.sin(pos); pe[:, 1::2] = torch.cos(pos)
        return self.enc(z + pe)


class Rel(nn.Module):
    def __init__(self, fh, fo, arch):
        super().__init__(); self.bh, self.bo = Branch(fh, arch), Branch(fo, arch); d = self.bh.out_dim
        self.xh, self.xo = nn.Linear(2 * d, d), nn.Linear(2 * d, d)                            # information exchange
        self.hh, self.ho = nn.Linear(d, 3), nn.Linear(d, 3)
        for m in (self.hh, self.ho):
            nn.init.zeros_(m.weight); nn.init.constant_(m.bias, 0.0); m.bias.data[1] = -3.0     # start: no correction, gate closed

    def forward(self, xh, xo, ramp):
        sh, so = self.bh(xh, ramp), self.bo(xo, ramp)
        uh = torch.relu(self.xh(torch.cat([sh, so], -1))); uo = torch.relu(self.xo(torch.cat([so, sh], -1)))
        out = []
        for u, head in ((uh, self.hh), (uo, self.ho)):
            o = head(u); out.append((0.1 * torch.tanh(o[..., 0]), torch.sigmoid(o[..., 1]), o[..., 2].clamp(-6, -1.6)))
        return out


def gsmooth(x, sigma=10.0):
    k = int(3 * sigma); g = torch.exp(-0.5 * (torch.arange(-k, k + 1, device=x.device, dtype=x.dtype) / sigma) ** 2); g = g / g.sum()
    xp = torch.cat([x[:, :1].expand(-1, k), x, x[:, -1:].expand(-1, k)], 1)
    return torch.nn.functional.conv1d(xp[:, None], g[None, None])[:, 0]


def ramp_of(T, a):
    t = np.arange(T); r = np.clip(np.abs(t - a) / 45.0, 0, 1); return 0.5 - 0.5 * np.cos(np.pi * r)


def branch_loss(delta, gate, logs, tgt, valid, ramp):
    c = gsmooth(ramp * gate * delta); r = tgt - c; v = valid.float()
    hub = torch.where(r.abs() < 0.02, 0.5 * r ** 2 / 0.02, r.abs() - 0.01)
    harm = torch.relu(r.abs() - tgt.abs())
    acc = (c[:, 2:] - 2 * c[:, 1:-1] + c[:, :-2] - (tgt[:, 2:] - 2 * tgt[:, 1:-1] + tgt[:, :-2])).abs()
    nll = 0.5 * ((tgt - c) / logs.exp()) ** 2 + logs                     # uncertainty of the APPLIED correction (review 10-28)
    n = v.sum().clamp_min(1)
    return ((hub + W_HARM * harm + W_NLL * nll) * v).sum() / n + W_ACC * (acc * v[:, 1:-1]).sum() / n, c


def tensors(fh, fo, th, to, e, st):
    T = len(th); a = int(e["anchor"])
    xh = torch.tensor(prep(fh, st[0], st[1]), dtype=torch.float32, device=DEV)[None]; xo = torch.tensor(prep(fo, st[2], st[3]), dtype=torch.float32, device=DEV)[None]
    vh = torch.tensor(e["valid_h"][:T] & np.isfinite(th), device=DEV)[None]; vo = torch.tensor(e["valid_o"][:T] & np.isfinite(to), device=DEV)[None]
    tt = lambda x: torch.tensor(np.nan_to_num(x), dtype=torch.float32, device=DEV)[None]  # noqa: E731
    return xh, xo, tt(th), tt(to), vh, vo, torch.tensor(ramp_of(T, a), dtype=torch.float32, device=DEV)[None]


@torch.no_grad()
def evaluate(model, eps, st, dump=None):
    model.eval(); rows = []
    for e in eps:
        xh, xo, th, to, vh, vo, rp = tensors(e["fh"].astype(np.float64), e["fo"].astype(np.float64), e["tgt_h"].astype(np.float64), e["tgt_o"].astype(np.float64), e, st)
        (dh, gh, _), (do, go, _) = model(xh, xo, rp)
        ch, co = gsmooth(rp * gh * dh), gsmooth(rp * go * do)
        if dump is not None:                                                                 # per-frame corrections for gru_rel_refit.py
            dump.mkdir(parents=True, exist_ok=True); np.savez(dump / f"{e['name'].split('/')[-1]}.npz", c_h=ch[0].cpu().numpy(), c_o=co[0].cpu().numpy(),
                                                               g_h=gh[0].cpu().numpy(), g_o=go[0].cpu().numpy(), anchor=int(e["anchor"]))
        row = {"name": e["name"], "err_h": float(np.nanmean(e["err_h"])), "err_o": float(np.nanmean(e["err_o"]))}
        for k, t, c, v in (("h", th, ch, vh), ("o", to, co, vo)):
            v = v[0]; t, c = t[0][v], c[0][v]
            row[f"{k}_n"] = int(v.sum()); row[f"{k}_before"] = float((t ** 2).mean().sqrt()) if len(t) else np.nan
            row[f"{k}_after"] = float(((t - c) ** 2).mean().sqrt()) if len(t) else np.nan
            row[f"{k}_harm"] = float(((t - c).abs() > t.abs() + 1e-4).float().mean()) if len(t) else np.nan
            row[f"{k}_gate"] = float((gh if k == "h" else go)[0][v].mean()) if len(t) else np.nan
        rows.append(row)
    model.train(); return rows


def summary(rows, tag=""):
    out = {}
    for k in ("h", "o"):
        R = [r for r in rows if r[f"{k}_n"] > 10]; med = np.median([r[f"err_{k}"] for r in R])
        for lab, sel in (("all", R), ("accurate", [r for r in R if r[f"err_{k}"] <= med]), ("inaccurate", [r for r in R if r[f"err_{k}"] > med])):
            b = np.sqrt(np.mean([r[f"{k}_before"] ** 2 for r in sel])); af = np.sqrt(np.mean([r[f"{k}_after"] ** 2 for r in sel]))
            out[f"{k}_{lab}"] = {"before_pct": b * 100, "after_pct": af * 100, "ratio": af / b, "harm": float(np.mean([r[f"{k}_harm"] for r in sel])),
                                 "improved": int(sum(r[f"{k}_after"] < r[f"{k}_before"] for r in sel)), "n": len(sel),
                                 "gate": float(np.mean([r[f"{k}_gate"] for r in sel]))}
        line = " | ".join(f"{lab} {out[f'{k}_{lab}']['before_pct']:.2f} -> {out[f'{k}_{lab}']['after_pct']:.2f} % (harm {out[f'{k}_{lab}']['harm'] * 100:.0f} %, better {out[f'{k}_{lab}']['improved']}/{out[f'{k}_{lab}']['n']}, gate {out[f'{k}_{lab}']['gate']:.2f})"
                          for lab in ("all", "accurate", "inaccurate"))
        print(f"  {tag}[{'human' if k == 'h' else 'object'}] {line}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cache", nargs="+", required=True); ap.add_argument("--train", default="")
    ap.add_argument("--val", required=True); ap.add_argument("--out", type=Path, required=True); ap.add_argument("--arch", default="gru")
    ap.add_argument("--steps", type=int, default=6000); ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_only", type=Path); ap.add_argument("--no_corrupt", action="store_true")
    ap.add_argument("--dump", type=Path, help="with --eval_only: write each val episode's corrections c_h / c_o here")
    ap.add_argument("--deploy", action="store_true", help="with --eval_only --dump: episodes without GT (dump only, no summary)")
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    val = load(a.cache, a.val.split(","))
    if a.eval_only:
        ck = torch.load(a.eval_only, weights_only=False); st = ck["stats"]
        model = Rel(len(st[0]) + 2, len(st[2]) + 2, ck["arch"]).to(DEV); model.load_state_dict(ck["model"])
        if a.deploy:
            evaluate(model, val, st, a.dump); print(f"dumped {len(val)} episodes (deploy)"); return
        rows = evaluate(model, val, st, a.dump); s = summary(rows, "eval "); (a.out / f"eval_{a.val.replace(',', '_')}.json").write_text(json.dumps({"summary": s, "rows": rows}, indent=1)); return
    tr = load(a.cache, a.train.split(",")); st = stats(tr)
    print(f"train {len(tr)} episodes, val {len(val)}, arch {a.arch}", flush=True)
    model = Rel(len(st[0]) + 2, len(st[2]) + 2, a.arch).to(DEV); opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps); best = np.inf
    base = summary(evaluate(model, val, st), "start ")
    for step in range(1, a.steps + 1):
        e = tr[rng.integers(len(tr))]
        fh, fo, th, to = (e["fh"].astype(np.float64), e["fo"].astype(np.float64), e["tgt_h"].astype(np.float64), e["tgt_o"].astype(np.float64)) if a.no_corrupt else corrupt(e, rng)
        xh, xo, th_, to_, vh, vo, rp = tensors(fh, fo, th, to, e, st)
        (dh, gh, lh), (do, go, lo) = model(xh, xo, rp)
        l1, _ = branch_loss(dh, gh, lh, th_, vh, rp); l2, _ = branch_loss(do, go, lo, to_, vo, rp)
        loss = l1 + l2; opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        if step % 500 == 0:
            s = summary(evaluate(model, val, st), f"step {step} ")
            score = 0.5 * (s["h_all"]["ratio"] + s["o_all"]["ratio"])
            if score < best and s["h_all"]["harm"] < 0.5 and s["o_all"]["harm"] < 0.5:
                best = score; torch.save({"model": model.state_dict(), "stats": st, "arch": a.arch, "step": step}, a.out / "rel_best.pt")
                (a.out / "val_best.json").write_text(json.dumps({"step": step, "summary": s}, indent=1)); print(f"  saved (score {score:.4f})", flush=True)
    print(f"PASS: rel_train {a.arch} best score {best:.4f} (1 = no change)")


if __name__ == "__main__":
    main()
