#!/usr/bin/env python3
"""Learned correction selector (mhr_converter.sif, GPU): when to move the human and / or the object along the camera ray.
  actions   keep (exactly zero, not modelled) and, per 6-s block (non-overlapping), depth moves of e = +-0.5 / 1 / 1.5 %:
              H  human alone: displacement e c_h (c_h = our human centroid, i.e. along its camera ray)
              T  common translation: human AND object displaced by the SAME vector e c_h (hand-object geometry kept)
              S  shared relative depth: human e c_h, object e t_o (both pushed along their own rays; the hand-object
                 vector changes by e (t_o - c_h), so NOT contact-preserving; corr_select2's "C")
              O  object alone: e t_o
  targets   the deployed action: block i's choice enters the episode through K_i(t) = taper(t) * (G_30 * box_i)(t), the
            exact linear map deployment applies (Gaussian sigma 30, anchor taper 45). The blocks partition the episode, so
            inside a block the deployed shift is exactly its own e whatever the neighbours choose and single-block
            effects add up like the deployed composition except in the blended edges (with half-overlapping 3-s
            intervals a single choice reached only ~45 % of e, neighbours agreeing doubled it: learned moves overshot).
            Target = change of each EPISODE metric when option o is applied on block i alone (all others keep),
            effects outside the block included, per unit of correction mass m_i = mean over scored frames of K_i
  metrics   official definitions on the scored frames (the human cache's QC mask = the scorer's frames): human joint
            error (CD-H surrogate), object Chamfer = sum of both directed means (kit symmetric_chamfer_distance), ACC-H
            on joints, ACC-O on the object TRANSLATION, PEN = mean over hand points of the penetration depth into our
            object's SDF (kit sdf_interpenetration_distance_per_frame); object Chamfer and PEN along each action
            direction are tabulated on an amplitude grid and interpolated, the deployed result is evaluated exactly
  model     shared MLP; heads: the five metric changes (Huber, standardised); with --loss risk also an upper quantile
            (0.8) of J = human + object change (pinball) and P(J > 0) (harm, BCE, accurate tracks weighted x2).
            Rows weighted so that every episode counts equally (the challenge averages episodes)
  decision  per interval the feasible option with the lowest predicted J; feasible = predicted J < -margin, no predicted
            regression of the human OR the object term (CD-H and CD-O are separate boards: no trading one for the other)
            and no predicted episode-level regression of ACC-H / ACC-O (> 1e-5) or PEN (> 1e-6); RISK rule additionally
            q80(J) < 0 and P(harm) < p. Rule, margin, p chosen on DEV: largest mean J gain with harm (J, human or object
            change > +0.1 cm) on accurate tracks <= 10 % and on all <= 15 %, mean ACC-H / PEN not worse; else keep
  evaluation  exact deployed trajectories: the five metric changes per episode, harm on all and on accurate tracks
            (baseline J <= median), family shares
Usage: corr_select3.py OUT --init GRU_DIR --train H:O:C:R,R [...] --dev H:O:C:R [...] --test NAME=H:O:C:R [...]
       [--families HTSO] [--loss mean|risk] [--eval_only SELECTOR]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

import gru_train as G
import gru_depth_train as GD

ES = [0.005, 0.01, 0.015]
LW, SIG, RAMP = 180, 30.0, 45.0     # non-overlapping 6-s blocks: inside a block the deployed shift is exactly its own e, so
                                    # single-block targets add up like the deployed composition (overlapping blocks did not)
GRID = torch.linspace(-0.016, 0.016, 17)
MIN_MASS = 0.02
TOL_ACC, TOL_PEN = 1e-5, 1e-6
dev = "cuda"


def sdf_lookup(grid, lo, hi, P):
    """P [N, 3] local mesh units -> signed distance (local units), trilinear; outside the grid: large positive."""
    g = (P - lo) / (hi - lo) * 2 - 1
    out = Fn.grid_sample(grid[None, None], g[None, None, None, :, [2, 1, 0]], mode="bilinear", padding_mode="border",
                         align_corners=True)[0, 0, 0, 0]
    return torch.where((g.abs() <= 1).all(1), out, torch.full_like(out, 1.0))


def pose(s, R, p, t):
    return s * torch.einsum("tij,nj->tni", R, p) + t[:, None]


def chamfer(A, B, chunk=64):
    """official: per frame, mean nearest distance A->B plus B->A."""
    out = []
    for i in range(0, len(A), chunk):
        D = torch.cdist(A[i:i + chunk], B[i:i + chunk])
        out.append(D.min(2).values.mean(1) + D.min(1).values.mean(1))
    return torch.cat(out)


def accel(x):
    return x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2]


class Ep(GD.Episode):
    def __init__(self, hpath, opath, cue_dir):
        super().__init__(hpath, cue_dir, dev)
        o = np.load(opath); h = np.load(hpath); T = len(self.ours)
        tt = lambda k: torch.as_tensor(np.asarray(o[k][:T], np.float32), device=dev)  # noqa: E731
        self.ot, self.oR, gR, self.gt_t = tt("obj_t"), tt("obj_R"), tt("gt_R"), tt("gt_t")
        self.os = float(o["obj_s"]); so = torch.as_tensor(o["surf_ours"], device=dev); sg = torch.as_tensor(o["surf_gt"], device=dev)
        self.O = pose(self.os, self.oR, so, self.ot); self.Gs = pose(float(o["gt_s"]), gR, sg, self.gt_t)
        fin = torch.isfinite(self.O).all(-1).all(-1) & torch.isfinite(self.Gs).all(-1).all(-1) & torch.isfinite(self.gt_t).all(-1)
        self.ov = self.valid & fin                                                                # scored frames only
        for k in ("O", "Gs", "ot", "oR", "gt_t"):
            setattr(self, k, torch.nan_to_num(getattr(self, k)))
        self.sdf_ok = float(o["sdf_ok"]) > 0
        self.sdf = (torch.as_tensor(o["sdf"], device=dev), torch.as_tensor(o["sdf_lo"], device=dev), torch.as_tensor(o["sdf_hi"], device=dev))
        self.hands = torch.as_tensor(h["hands_ours"][:T], device=dev)                             # [T, 128, 3] metres, our frame
        # GT-free interval features (as an earlier selector)
        nrm = torch.einsum("tij,nj->tni", self.oR, torch.as_tensor(o["norm_ours"], device=dev))
        face = (nrm * self.O).sum(-1) < 0
        zv = torch.nanmedian(torch.where(face, self.O[..., 2], torch.full_like(self.O[..., 2], float("nan"))), 1).values.cpu().numpy()
        md, ok = np.asarray(o["mask_depth"][:T], float), np.asarray(o["depth_ok"][:T], float) > 0
        ok &= np.isfinite(zv) & (zv > 1e-3)
        co = np.where(ok, np.log(np.maximum(md, 1e-6) / np.where(ok, zv, 1.0)), np.nan)
        near = np.abs(np.arange(T) - self.anchor) <= 5
        ref = np.nanmedian(co[near]) if np.isfinite(co[near]).any() else (np.nanmedian(co) if np.isfinite(co).any() else 0.0)
        self.co, self.co_rel, self.co_ok = np.nan_to_num(co), np.nan_to_num(co - ref), ok.astype(float)
        self.drel = np.nan_to_num(np.asarray(o["depth_rel"][:T], float)); self.area = np.nan_to_num(np.asarray(o["mask_area"][:T], float))
        dm = torch.stack([torch.cat([torch.cdist(self.hands[i:i + 64, sl], self.O[i:i + 64]).flatten(1).min(1).values
                                     for i in range(0, T, 64)]) for sl in (slice(0, 64), slice(64, 128))], 1)
        k = dm.argmin(1); hc = torch.stack([self.hands[:, :64].mean(1), self.hands[:, 64:].mean(1)], 1)[torch.arange(T, device=dev), k]
        oc = self.ot; vo = torch.zeros(T, device=dev); vr = torch.zeros(T, device=dev)
        vo[1:] = (oc[1:] - oc[:-1]).norm(dim=-1); vr[1:] = ((oc - hc)[1:] - (oc - hc)[:-1]).norm(dim=-1)
        self.dmin, self.vo, self.vr = dm.min(1).values.cpu().numpy(), vo.cpu().numpy(), vr.cpu().numpy()
        c = self.ours.mean(1)
        self.odist = (oc.norm(dim=-1) / c.norm(dim=-1)).cpu().numpy()
        self.oproj = ((c * oc).sum(-1) / oc.norm(dim=-1).clamp_min(1e-3) ** 2).cpu().numpy()      # T-move's relative object depth change per e
        self.c_moge = np.nan_to_num(np.asarray(np.load(cue_dir / hpath.name)["c_moge"], float))

    def pen(self, rel):
        """rel [N, T, 3] hand displacement relative to the object -> per-frame PEN [N, T] (cm), kit definition."""
        if not self.sdf_ok:
            return torch.zeros(rel.shape[:2], device=dev)
        grid, lo, hi = self.sdf; out = []
        for r in rel:
            P = self.hands + r[:, None]
            loc = torch.einsum("tij,tpi->tpj", self.oR, P - self.ot[:, None]) / self.os
            sd = sdf_lookup(grid, lo, hi, loc.reshape(-1, 3)).reshape(P.shape[:2]) * self.os
            out.append(torch.relu(-sd).mean(1) * 100)
        return torch.stack(out)


def load_group(spec):
    hc, oc, cues, roots = spec.split(":"); hc, oc, cues = Path(hc), Path(oc), Path(cues)
    ps = sorted(p for r in roots.split(",") for p in hc.glob(f"{r}_episode_*.npz") if (cues / p.name).exists() and (oc / f"{p.stem}_obj.npz").exists())
    eps = [Ep(p, oc / f"{p.stem}_obj.npz", cues) for p in ps]
    return [e for e in eps if float((e.ours - e.gt).norm(dim=-1).mean(-1)[e.valid].mean()) < 0.30]   # failed human tracks out


def blocks(T):
    """partition of [0, T) into blocks of LW frames; a remainder shorter than LW / 2 joins the last block."""
    starts = list(range(0, T, LW))
    if len(starts) > 1 and T - starts[-1] < LW // 2:
        starts.pop()
    return starts, starts[1:] + [T]


def kernels(T, anchor, starts, ends):
    B = np.zeros((T, len(starts)), np.float32)
    for i, (s, e) in enumerate(zip(starts, ends)):
        B[s:e, i] = 1.0
    K = G.gauss_t(torch.as_tensor(B, device=dev), SIG)
    t = torch.arange(T, dtype=torch.float32, device=dev); rd = torch.clamp((t - anchor).abs() / RAMP, 0, 1); rd = 0.5 - 0.5 * torch.cos(np.pi * rd)
    return (K * rd[:, None]).T.contiguous()                                                       # [n, T]


def interp(Gv, A):
    """Gv [len(GRID), T] per-frame values on the amplitude grid, A [N, T] amplitudes -> [N, T]."""
    g0, step = float(GRID[0]), float(GRID[1] - GRID[0])
    x = ((A - g0) / step).clamp(0, len(GRID) - 1 - 1e-6); i0 = x.floor().long(); w = x - i0
    t = torch.arange(A.shape[1], device=dev)[None].expand_as(i0)
    return Gv[i0, t] * (1 - w) + Gv[i0 + 1, t] * w


def opts_for(fams):
    return [(f, s * e) for f in fams for e in ES for s in (-1, 1)]


def metrics_batch(ep, fr, c, Ah, Ac, At, chunk=48):
    """Episode metrics for displacement amplitudes [N, T]: human along c (Ah), object along c (Ac) / along t_o (At).
    -> pos (cm), cd (cm, interpolated), acc_h, acc_o (cm/frame^2), pen (cm) [N] each."""
    v, ov = ep.valid, ep.ov; ok = v[2:] & v[1:-1] & v[:-2]; ook = ov[2:] & ov[1:-1] & ov[:-2]
    pos, acch = [], []
    ea = accel((fr - ep.gt)[None])                                                                # [1, T-2, 22, 3]
    for i in range(0, len(Ah), chunk):
        a = Ah[i:i + chunk]; d = a[..., None] * c[None]                                           # [n, T, 3]
        pos.append(((fr[None] + d[:, :, None]) - ep.gt[None]).norm(dim=-1).mean(-1)[:, v].mean(1))
        acch.append((ea + accel(d)[:, :, None]).norm(dim=-1).mean(-1)[:, ok].mean(1))
    pos, acch = torch.cat(pos) * 100, torch.cat(acch) * 100
    Do = Ac[..., None] * c[None] + At[..., None] * ep.ot[None]                                   # [N, T, 3]
    acco = (accel((ep.ot - ep.gt_t)[None]) + accel(Do)).norm(dim=-1)[:, ook].mean(1) * 100
    cd = (interp(ep.CDc, Ac) + interp(ep.CDt, At) - ep.CDc[len(GRID) // 2][None])[:, ov].mean(1) * 100
    return pos, cd, acch, acco


def prepare(ep):
    """amplitude-grid tables of object Chamfer (displacement along c / along t_o) and PEN (hand-relative directions)."""
    c = ep.ours.mean(1)
    ep.CDc = torch.stack([chamfer(ep.O + float(g) * c[:, None], ep.Gs) for g in GRID])
    ep.CDt = torch.stack([chamfer(ep.O + float(g) * ep.ot[:, None], ep.Gs) for g in GRID])
    dirs = {"H": c, "S": c - ep.ot, "O": -ep.ot}
    ep.PEN = {f: ep.pen(torch.stack([float(g) * u for g in GRID])) for f, u in dirs.items()}       # [len(GRID), T]
    ep.PEN0 = ep.PEN["H"][len(GRID) // 2]


def pen_batch(ep, fam_amp):
    """fam_amp {family: [N, T] amplitude} -> episode PEN [N] (cm): hand displacement relative to the object."""
    N, T = next(iter(fam_amp.values())).shape; tot = torch.zeros(N, T, device=dev)
    for f, A in fam_amp.items():
        if f != "T":                                                                              # common translation: PEN unchanged
            tot = tot + interp(ep.PEN[f], A) - ep.PEN0[None]
    return (ep.PEN0[None] + tot)[:, ep.valid].mean(1)


def episode_table(ep, ref, base, fams):
    """-> dict: features [n, nopt, F], targets [n, nopt, 5] per unit mass (pos, cd, acc_h, acc_o, pen), mass [n],
    baseline metrics, kernels, GRU v2 joints."""
    with torch.no_grad():
        fr = G.fuse(ep, ref(((ep.feat - G.MU) / G.SD)[None])[0], slice(None), base)
    prepare(ep)
    T = len(fr); c = ep.ours.mean(1); dist = c.norm(dim=-1); ray = c / dist[:, None]
    starts, ends = blocks(T); K = kernels(T, ep.anchor, starts, ends); n = len(starts)
    mass = K[:, ep.valid].sum(1) / ep.valid.sum()
    opts = opts_for(fams); E = torch.tensor([e for _, e in opts], device=dev)
    msk = lambda fs: torch.tensor([f in fs for f, _ in opts], dtype=torch.float32, device=dev)  # noqa: E731
    A = E[None, :, None] * K[:, None, :]                                                          # [n, nopt, T]
    Ah, Ac, At = [(A * msk(fs)[None, :, None]).reshape(-1, T) for fs in ("HTS", "T", "SO")]
    z = torch.zeros(1, T, device=dev)
    with torch.no_grad():
        pos, cd, acch, acco = metrics_batch(ep, fr, c, Ah, Ac, At); b = metrics_batch(ep, fr, c, z, z, z)
        famA = {f: (A * msk(f)[None, :, None]).reshape(-1, T) for f in fams}
        pen = pen_batch(ep, famA); pen0 = ep.PEN0[ep.valid].mean()
    Y = torch.stack([pos - b[0], cd - b[1], acch - b[2], acco - b[3], pen - pen0], 1).reshape(n, len(opts), 5)
    Y = Y / mass.clamp_min(1e-3)[:, None, None]
    cue = ep.cue.cpu().numpy(); v = ep.valid.cpu().numpy()
    draw = (((ep.br[4] - ep.bo[4]) * ray[:, None]).sum(-1).mean(-1) / dist).cpu().numpy()
    F = []
    for i, (s, e_) in enumerate(zip(starts, ends)):
        sl = slice(s, e_); w = ep.co_ok[sl]
        cm, cs, cf, vm, vs, st, rd = (cue[sl, k] for k in range(7))
        m = lambda x, w_: float((x * w_).sum() / max(w_.sum(), 1e-6))                          # noqa: E731
        sd = lambda x, w_: float(np.sqrt(m((x - m(x, w_)) ** 2, w_)))                          # noqa: E731
        mm, ms, mf = m(cm, vm), m(cs, vs), m(cf, st); mo, mor = m(ep.co[sl], w), m(ep.co_rel[sl], w)
        base_f = [mm, ms, mf, vm.mean(), vs.mean(), st.mean(), sd(cm, vm), sd(cs, vs), sd(cf, st), float(np.sign(ms) == np.sign(mf)),
                  float(draw[sl].mean()), float(abs(s - ep.anchor) / 300.0), float(rd.mean()), float(v[sl].mean()),
                  mo, mor, w.mean(), sd(ep.co[sl], w), float(ep.drel[sl].mean()), float(ep.area[sl].mean() * 100),
                  float(np.minimum(ep.dmin[sl], 0.5).mean() * 100), float((ep.dmin[sl] < 0.03).mean()), float(ep.vo[sl].mean() * 100),
                  float(ep.vr[sl].mean() / (ep.vo[sl].mean() + 1e-3)), float(ep.c_moge[sl].mean()), float(np.log(np.maximum(ep.odist[sl], 1e-3)).mean()),
                  float(mass[i])]
        pj = float(ep.oproj[sl].mean()); fo = []
        for f, e in opts:
            eh = e if f in "HTS" else 0.0; eo = e * pj if f == "T" else (e if f in "SO" else 0.0); lh, lo = np.log1p(eh), np.log1p(eo)
            fo.append(base_f + [float(f == "H"), float(f == "T"), float(f == "S"), float(f == "O"), e * 100, abs(e) * 100, eh * 100, eo * 100,
                                mm - lh, ms - lh, mf - lh, abs(mm - lh), abs(ms - lh), abs(mf - lh), (ms - lh) * (mf - lh),
                                mo - lo, mor - lo, abs(mo - lo), abs(mor - lo), (mor - lo) * (ms - lh)])
        F.append(fo)
    return {"F": np.array(F, np.float32), "Y": Y.cpu().numpy(), "mass": mass.cpu().numpy(), "K": K, "fr": fr, "opts": opts,
            "base": [float(x) for x in b] + [float(pen0)]}


class Sel(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(nf, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU())
        self.mean = nn.Linear(128, 5); self.q = nn.Linear(128, 1); self.harm = nn.Linear(128, 1)

    def forward(self, x):
        h = self.trunk(x)
        return self.mean(h), self.q(h)[..., 0], self.harm(h)[..., 0]


def predict(model, ck, F):
    with torch.no_grad():
        mu, q, hl = model(torch.as_tensor((F - ck["mu"]) / ck["sd"], device=dev))
    ysd = torch.as_tensor(ck["ysd"], device=dev)
    return (mu * ysd).cpu().numpy(), (q * float(ck["jsd"])).cpu().numpy(), torch.sigmoid(hl).cpu().numpy()


def choose(tb, pred, rule):
    """per-interval option index or -1 (keep) under the rule {kind, margin, p}."""
    mu, q, ph = pred; mass = tb["mass"][:, None]
    J = mu[..., 0] + mu[..., 1]
    ok = (J < -rule["margin"]) & (mu[..., 2] * mass <= TOL_ACC) & (mu[..., 3] * mass <= TOL_ACC) & (mu[..., 4] * mass <= TOL_PEN) & (mass >= MIN_MASS)
    ok &= (mu[..., 0] <= 0) & (mu[..., 1] <= 0)                    # separate boards: no predicted CD-H or CD-O regression
    if rule["kind"] == "risk":
        ok &= (q < 0) & (ph < rule["p"])
    Jm = np.where(ok, J, np.inf); best = Jm.argmin(1)
    return np.where(np.isfinite(Jm.min(1)), best, -1)


def deploy_eval(ep, tb, choice):
    """exact metrics of the deployed trajectory vs keep: (pos, cd, acc_h, acc_o, pen) changes, baseline J, family shares."""
    T = len(tb["fr"]); c = ep.ours.mean(1); amp = {f: torch.zeros(T, device=dev) for f in "HTSO"}
    for i, j in enumerate(choice):
        if j >= 0:
            f, e = tb["opts"][j]; amp[f] = amp[f] + e * tb["K"][i]
    ah = amp["H"] + amp["T"] + amp["S"]; ac = amp["T"]; at = amp["S"] + amp["O"]
    with torch.no_grad():
        pos, _, acch, acco = metrics_batch(ep, tb["fr"], c, ah[None], ac[None], at[None])
        Od = ep.O + (ac[:, None] * c + at[:, None] * ep.ot)[:, None]
        cd = chamfer(Od, ep.Gs)[ep.ov].mean() * 100
        rel = ah[:, None] * c - (ac[:, None] * c + at[:, None] * ep.ot)
        pen = ep.pen(rel[None])[0][ep.valid].mean()
    b = tb["base"]; d = [float(pos[0]) - b[0], float(cd) - b[1], float(acch[0]) - b[2], float(acco[0]) - b[3], float(pen) - b[4]]
    ch = [tb["opts"][j][0] if j >= 0 else "keep" for j in choice]
    add = sum(float((tb["Y"][i, j, 0] + tb["Y"][i, j, 1]) * tb["mass"][i]) for i, j in enumerate(choice) if j >= 0)   # sum of true single-block J
    return d, b[0] + b[1], {f: ch.count(f) / max(len(ch), 1) for f in ("keep", "H", "T", "S", "O")}, add


def summarize(name, eps, tabs, model, ck, rule, quiet=False):
    rows = []
    for ep, tb in zip(eps, tabs):
        d, j0, sh, add = deploy_eval(ep, tb, choose(tb, predict(model, ck, tb["F"]), rule)); rows.append(d + [j0, sh["H"], sh["T"], sh["S"], sh["O"], add])
    r = np.array(rows); J = r[:, 0] + r[:, 1]; good = r[:, 5] <= np.median(r[:, 5])
    bad = (J > 0.1) | (r[:, 0] > 0.1) | (r[:, 1] > 0.1)                                          # harm on either board
    s = {"n": len(r), "J": float(J.mean()), "J_median": float(np.median(J)), "better": int((J < -1e-6).sum()), "worst": float(J.max()),
         "pos": float(r[:, 0].mean()), "cd_o": float(r[:, 1].mean()), "acc_h": float(r[:, 2].mean()), "acc_o": float(r[:, 3].mean()),
         "pen": float(r[:, 4].mean()), "harm": float(np.mean(bad)), "harm_accurate": float(np.mean(bad[good])),
         "harm_h_accurate": float(np.mean(r[good, 0] > 0.1)), "share": {f: float(r[:, 6 + k].mean()) for k, f in enumerate("HTSO")},
         "J_sum_single": float(r[:, 10].mean())}
    if not quiet:
        print(f"[{name}] {rule['kind']} m {rule['margin']} p {rule.get('p', '-')} | n {s['n']} | J {s['J']:+.3f} cm (median {s['J_median']:+.3f}, better {s['better']}, "
              f"worst {s['worst']:+.2f}) = human {s['pos']:+.3f} + object {s['cd_o']:+.3f} | ACC-H {s['acc_h']:+.6f} ACC-O {s['acc_o']:+.6f} PEN {s['pen']:+.6f} | "
              f"harm {s['harm'] * 100:.0f} %, accurate {s['harm_accurate'] * 100:.0f} % (human-only {s['harm_h_accurate'] * 100:.0f} %) | "
              f"sum of true single-block J {s['J_sum_single']:+.3f} | blocks "
              + " ".join(f"{f} {v * 100:.0f}%" for f, v in s["share"].items()), flush=True)
    return s


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out", type=Path); ap.add_argument("--init", type=Path, required=True)
    ap.add_argument("--train", action="append", default=[]); ap.add_argument("--dev", action="append", default=[])
    ap.add_argument("--test", action="append", default=[]); ap.add_argument("--families", default="HTSO")
    ap.add_argument("--loss", choices=("mean", "risk"), default="risk"); ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--eval_only", type=Path)
    ap.add_argument("--dump_shifts", type=Path, help="eval only: per test episode the chosen per-frame displacements (our camera "
                    "frame, metres): d_h [T, 3] for the human, d_o [T, 3] for the object (scripts/apply_shift.py applies them)")
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True); torch.manual_seed(0); rng = np.random.default_rng(0)
    nrm = torch.load(a.init / "refiner_motion_norm.pt"); G.MU, G.SD = nrm["mu"].to(dev), nrm["sd"].to(dev); base = G.e2_base()
    ip = a.init / "refiner_motion_best.pt"; ip = ip if ip.exists() else a.init / "refiner_motion.pt"; sdr = torch.load(ip); ref = G.Refiner(sdr["inp.0.weight"].shape[1]).to(dev); ref.load_state_dict(sdr); ref.eval()
    if a.eval_only:
        ck = torch.load(a.eval_only, weights_only=False); fams = ck["families"]
        model = Sel(len(ck["mu"])).to(dev); model.load_state_dict(ck["model"]); model.eval(); res = {}
        print(f"frozen selector {a.eval_only}: families {fams}, loss {ck['loss']}, rule {ck['rule']}", flush=True)
        for spec in a.test:
            name, rest = spec.split("=", 1); te = load_group(rest); tabs = [episode_table(e, ref, base, fams) for e in te]
            res[name] = summarize(f"TEST {name}", te, tabs, model, ck, ck["rule"])
            if a.dump_shifts:
                (a.dump_shifts / name).mkdir(parents=True, exist_ok=True)
                for ep, tb in zip(te, tabs):
                    choice = choose(tb, predict(model, ck, tb["F"]), ck["rule"]); T = len(tb["fr"]); c = ep.ours.mean(1)
                    amp = {f: torch.zeros(T, device=dev) for f in "HTSO"}
                    for i, j in enumerate(choice):
                        if j >= 0:
                            f, e = tb["opts"][j]; amp[f] = amp[f] + e * tb["K"][i]
                    d_h = (amp["H"] + amp["T"] + amp["S"])[:, None] * c; d_o = amp["T"][:, None] * c + (amp["S"] + amp["O"])[:, None] * ep.ot
                    np.savez(a.dump_shifts / name / f"{ep.name}.npz", d_h=d_h.cpu().numpy(), d_o=d_o.cpu().numpy(), anchor=ep.anchor,
                             blocks=np.array([tb["opts"][j][0] if j >= 0 else "keep" for j in choice]))
                print(f"shifts of {len(te)} episodes -> {a.dump_shifts / name} (episodes not dumped (failed tracks) keep unshifted)", flush=True)
        (a.out / f"frozen_eval_{'_'.join(s.split('=')[0] for s in a.test)}.json").write_text(json.dumps(res, indent=1))
        return
    fams = a.families
    tr = [e for s in a.train for e in load_group(s)]; dv = [e for s in a.dev for e in load_group(s)]
    ttr = [episode_table(e, ref, base, fams) for e in tr]; tdv = [episode_table(e, ref, base, fams) for e in dv]
    X, Y, W, H = [], [], [], []
    for e, tb in zip(tr, ttr):
        keep = tb["mass"] >= MIN_MASS; F, Yt, m = tb["F"][keep], tb["Y"][keep], tb["mass"][keep]
        acc_ep = float(tb["base"][0] + tb["base"][1])
        X.append(F.reshape(-1, F.shape[-1])); Y.append(Yt.reshape(-1, 5))
        W.append(np.repeat(m / m.sum(), F.shape[1]) / F.shape[1]); H.append(np.full(F.shape[0] * F.shape[1], acc_ep))
    X, Y, W, H = map(np.concatenate, (X, Y, W, H))
    accurate = (H <= np.median([tb["base"][0] + tb["base"][1] for tb in ttr])).astype(np.float32)
    mu, sd = X.mean(0), X.std(0) + 1e-6; ysd = Y.std(0) + 1e-9; Jt = Y[:, 0] + Y[:, 1]; jsd = float(Jt.std() + 1e-9)
    print(f"train episodes {len(tr)} ({X.shape[0]} interval-options, {X.shape[1]} features), dev {len(dv)}; families {fams}, loss {a.loss}; "
          f"target sd per unit mass: pos {ysd[0]:.3f} cd {ysd[1]:.3f} acc_h {ysd[2]:.2e} acc_o {ysd[3]:.2e} pen {ysd[4]:.2e}; "
          f"options improving J (episode-weighted) {np.sum(W * (Jt < 0)) / W.sum() * 100:.0f} %", flush=True)
    model = Sel(X.shape[1]).to(dev); opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt = torch.as_tensor((X - mu) / sd, device=dev); Yt = torch.as_tensor(Y / ysd, device=dev); Jn = torch.as_tensor(Jt / jsd, device=dev)
    Wt = torch.as_tensor(W, device=dev); At = torch.as_tensor(accurate, device=dev); Hb = torch.as_tensor((Jt > 0).astype(np.float32), device=dev)
    for step in range(1, a.steps + 1):
        i = torch.multinomial(Wt, 4096, replacement=True)
        m_, q_, h_ = model(Xt[i]); loss = nn.functional.huber_loss(m_, Yt[i], delta=1.0, reduction="none").sum(1).mean()
        if a.loss == "risk":
            u = Jn[i] - q_; loss = loss + torch.maximum(0.8 * u, -0.2 * u).mean()
            loss = loss + (nn.functional.binary_cross_entropy_with_logits(h_, Hb[i], reduction="none") * (1 + At[i])).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 1000 == 0:
            print(f"step {step} loss {float(loss):.4f}", flush=True)
    model.eval()
    ck = {"model": model.state_dict(), "mu": mu, "sd": sd, "ysd": ysd, "jsd": jsd, "families": fams, "loss": a.loss}
    rules = [{"kind": "mean", "margin": m} for m in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)]
    if a.loss == "risk":
        rules += [{"kind": "risk", "margin": m, "p": p} for m in (0.0, 0.5, 1.0, 2.0, 4.0) for p in (0.1, 0.2, 0.3, 0.5)]
    best = None; log = []
    for rule in rules:
        s = summarize("dev", dv, tdv, model, ck, rule); log.append({"rule": rule, **s})
        feas = s["harm_accurate"] <= 0.10 and s["harm"] <= 0.15 and s["acc_h"] <= 1e-6 and s["pen"] <= 1e-6 and s["pos"] <= 0 and s["cd_o"] <= 0
        if feas and s["J"] < 0 and (best is None or s["J"] < best[0]):
            best = (s["J"], rule)
    ck["rule"] = best[1] if best else {"kind": "mean", "margin": 1e9}                             # nothing safe on DEV: keep everywhere
    print(f"chosen on DEV: {ck['rule']}" + ("" if best else " (no rule met the harm constraints: keep)"), flush=True)
    torch.save(ck, a.out / "selector3.pt"); res = {"dev": log}
    for spec in a.test:
        name, rest = spec.split("=", 1); te = load_group(rest)
        res[name] = summarize(f"TEST {name}", te, [episode_table(e, ref, base, fams) for e in te], model, ck, ck["rule"])
    (a.out / "results.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
