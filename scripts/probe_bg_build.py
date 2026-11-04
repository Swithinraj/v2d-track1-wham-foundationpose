"""Paired B/G probe build (user-approved 2026-11-04): measure the divergence gate's CD-O effect like-for-like on ONE
FoundationPose lineage (the Oct-30 cluster run on Drive: its own raw/ and smoothed/), not against candidate A.

For every clean FP+RTS episode the gate touched (gated track != smoothed track on >= 1 shipped frame):
  L  = production's object smoothing for these episodes: physics_objects.solve, --no_rest --no_gravity, lam 10 then
       lam 100 (recipe_ipf30.md stage A, bmq_final.sh step 4), float32 between stages.
  S  = one per-episode similarity (Rr, sigma, c) mapping L(smoothed) into E3C30tab's frame. Rr: the candidate whose
       angle best matches the recorded object_transfer rotation (the only rotating step downstream): robust mean of
       R_E3 R_smooth^T, or the translation Umeyama rotation (symmetric objects make the first unreliable). sigma, c:
       robust least squares with Rr fixed.
  B  = S(L(smoothed)),  R_B = Rr R_smooth          G = S(L(gated)),  R_G = Rr R_gated
B and G use the identical S and L, so G - B is the gate's effect alone. Untouched episodes and every other array/mesh
stay byte-identical to E3C30tab in both.
Usage: probe_bg_build.py GATE_DIR OUT_ROOT
Writes OUT_ROOT/episodes_bg_B, OUT_ROOT/episodes_bg_G, OUT_ROOT/bg_build_report.json
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path('/work/smoses2s/thesis_abl')
sys.path.insert(0, str(ROOT / 'repo/scripts'))
import physics_objects as po  # noqa: E402

E3 = ROOT / 'data/incoming/episodes_cand_E3C30tab'
PROV = ROOT / 'data/incoming/object_provenance_E3C30tab'
CACHE = ROOT / 'repo/outputs/object'
GATE, OUT_ROOT = Path(sys.argv[1]), Path(sys.argv[2])
EPISODES = [3, 4, 5, 6, 7, 15, 16, 22, 23, 24, 26, 29]
LAMS = [10.0, 100.0]
transfer = json.loads((PROV / 'object_transfer_report.json').read_text())


def quat_R(q):
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def load_track(d, T):
    R, t = np.zeros((T, 3, 3)), np.zeros((T, 3))
    for f in range(T):
        j = json.loads((d / f'{f:06d}.json').read_text())
        R[f], t[f] = quat_R(j['rotation']), j['translation']
    return R, t


def L(t):
    x = t.astype(np.float32).astype(np.float64)
    labels = np.full(len(x), -1)
    for lam in LAMS:
        x = x + po.drift_correction(x, labels)
        x = po.solve(x, labels, lam).astype(np.float32).astype(np.float64)
    return x


def geo(Ra, Rb):
    return np.degrees(np.arccos(np.clip((np.einsum('...ij,...ij->...', Ra, Rb) - 1) / 2, -1, 1)))


def umeyama(X, Y):
    mx, my = X.mean(0), Y.mean(0)
    U, S, Vt = np.linalg.svd((Y - my).T @ (X - mx) / len(X))
    D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    return U @ D @ Vt


def fit_sigma_c(R, X, Y, sel):
    for _ in range(4):
        x = X[sel] @ R.T
        A = np.zeros((3 * sel.sum(), 4)); A[:, 0] = x.ravel()
        for k in range(3):
            A[k::3, 1 + k] = 1.0
        sol, *_ = np.linalg.lstsq(A, Y[sel].ravel(), rcond=None)
        res = np.linalg.norm(Y - (sol[0] * X @ R.T + sol[1:]), axis=1)
        sel = sel & (res <= np.percentile(res[sel], 90) + 1e-6)
    return sol[0], sol[1:], res


def stats_mm(v):
    return dict(median=round(float(np.median(v)) * 1000, 2), p90=round(float(np.percentile(v, 90)) * 1000, 2), max=round(float(v.max()) * 1000, 2))


B_DIR, G_DIR = OUT_ROOT / 'episodes_bg_B', OUT_ROOT / 'episodes_bg_G'
for d in (B_DIR, G_DIR):
    if d.exists():
        shutil.rmtree(d)
    shutil.copytree(E3, d)
report = dict(gate_dir=str(GATE), lams=LAMS, episodes={}, touched=[])
for ep in EPISODES:
    tag = f'episode_{ep:06d}'
    with np.load(E3 / f'{tag}.npz') as a:
        e3 = {k: a[k] for k in a.files}
    R_E, t_E = e3['object_rotation'].astype(np.float64), e3['object_translation'].astype(np.float64)
    T = len(t_E)
    R_c, t_c = load_track(CACHE / tag / 'foundationpose/smoothed', T)
    R_g, t_g = load_track(GATE / tag / 'gated_poses', T)
    # touched frames = the gate's own FALLBACK_TO_RAW events (a float-noise geodesic test flagged identical rotations)
    events = json.loads((GATE / tag / 'gated_smoothing_summary.json').read_text())['divergence_events']
    changed = np.zeros(T, bool)
    changed[[e['frame'] for e in events if e['frame'] < T]] = True
    keep = ~changed  # outside events the gated track IS the smoothed track: use it exactly
    R_g[keep], t_g[keep] = R_c[keep], t_c[keep]
    rep = dict(shipped_frames=T, gate_events=len(events), changed_frames=int(changed.sum()),
               gate_events_outside_shipped=int(sum(e['frame'] >= T for e in events)))
    if not changed.any():
        rep['status'] = 'untouched (byte-identical to E3C30tab in B and G)'
        report['episodes'][tag] = rep
        print(tag, json.dumps(rep), flush=True)
        continue
    fin = np.isfinite(t_E).all(1)
    Lc, Lg = L(t_c), L(t_g)
    # rotation candidates
    Rrel = R_E @ np.transpose(R_c, (0, 2, 1))
    sel = fin.copy()
    for _ in range(4):
        Rr_rot = Rotation.from_matrix(Rrel[sel]).mean().as_matrix()
        rr = geo(Rrel, Rr_rot[None])
        sel = fin & (rr <= np.percentile(rr[fin], 70) + 1e-6)
    Ru = umeyama(Lc[fin], t_E[fin])
    rec = float(transfer[tag]['rotation_deg'])
    cands = {'rotation_bridge': Rr_rot, 'translation_umeyama': Ru}
    angles = {k: float(geo(v, np.eye(3))) for k, v in cands.items()}
    pick = min(angles, key=lambda k: abs(angles[k] - rec))
    Rr = cands[pick]
    sigma, c, res = fit_sigma_c(Rr, Lc, t_E, fin.copy())
    tB, tG = sigma * Lc @ Rr.T + c, sigma * Lg @ Rr.T + c
    RB, RG = Rr @ R_c, Rr @ R_g
    dGB = np.linalg.norm(tG - tB, axis=1)
    rep.update(status='touched', recorded_transfer_rotation_deg=round(rec, 3),
               rotation_candidates_deg={k: round(v, 3) for k, v in angles.items()}, rotation_used=pick,
               rotation_used_minus_recorded_deg=round(angles[pick] - rec, 3), sigma=round(float(sigma), 5),
               e3c30tab_object_scale=round(float(np.asarray(e3['object_scale'])), 5),
               B_vs_E3C30tab_translation=stats_mm(np.linalg.norm(tB[fin] - t_E[fin], axis=1)),
               G_minus_B=dict(frames_translation_over_1mm=int((dGB > 1e-3).sum()), translation_cm_max=round(float(dGB.max() * 100), 2),
                              translation_cm_median_over_moved=round(float(np.median(dGB[dGB > 1e-3]) * 100), 2) if (dGB > 1e-3).any() else 0.0,
                              frames_rotation_changed=int((geo(RG, RB) > 1e-6).sum()),
                              rotation_deg_max=round(float(geo(RG, RB).max()), 2)))
    for d, R_, t_ in ((B_DIR, RB, tB), (G_DIR, RG, tG)):
        out = dict(e3)
        out['object_rotation'] = R_.astype(e3['object_rotation'].dtype)
        out['object_translation'] = t_.astype(e3['object_translation'].dtype)
        np.savez(d / f'{tag}.npz', **out)
    report['touched'].append(ep)
    report['episodes'][tag] = rep
    print(tag, json.dumps(rep), flush=True)
(OUT_ROOT / 'bg_build_report.json').write_text(json.dumps(report, indent=2))
print('touched episodes:', report['touched'])
print('saved', OUT_ROOT / 'bg_build_report.json')
