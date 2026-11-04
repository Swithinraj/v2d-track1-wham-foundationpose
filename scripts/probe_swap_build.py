"""Follow-up to the paired B/G CD-O probe (user, 2026-11-04 "one by one, both").
B (16.60027) = E3C30tab with the objects of episodes 5, 7, 15 re-derived from the Oct-30 FoundationPose+RTS run.
  S5, S7, S15 = E3C30tab with ONLY that one episode's objects taken from B (byte copies of B's npz)  -> which episode
                carries B's gain.
  B12         = B plus the other 9 clean FP+RTS episodes (3 4 6 16 22 23 24 26 29) re-derived exactly like B's
                episodes (L = production physics_objects.solve lam10 then lam100, float32 between stages; S = one
                per-episode similarity into E3C30tab's frame, rotation picked by agreement with the recorded
                object_transfer angle; R = Rr R_smooth)                                          -> B12 - B = the 9.
  B12x        = B12 with episodes 4 and 16 back to B's (= E3C30tab's) objects: the two weakest bridges (ep16 sigma 13%
                off object_scale, ep4 rotation candidates 91 deg apart)                         -> B12x - B12 = eps 4+16.
Usage: probe_swap_build.py OUT_ROOT. Writes OUT_ROOT/episodes_swap_{s5,s7,s15,b12,b12x} and OUT_ROOT/swap_build_report.json.
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
B_DIR = ROOT / 'reports/probe_bg/episodes_bg_B'
PROV = ROOT / 'data/incoming/object_provenance_E3C30tab'
CACHE = ROOT / 'repo/outputs/object'
OUT_ROOT = Path(sys.argv[1])
NEW9 = [3, 4, 6, 16, 22, 23, 24, 26, 29]
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


def stats(v, k=1000.0, nd=2):
    return dict(median=round(float(np.median(v)) * k, nd), p90=round(float(np.percentile(v, 90)) * k, nd), max=round(float(v.max()) * k, nd))


def fresh(name, src):
    d = OUT_ROOT / name
    if d.exists():
        shutil.rmtree(d)
    shutil.copytree(src, d)
    return d


report = dict(lams=LAMS, base_B=str(B_DIR), variants={}, episodes={})
# single-episode splits: byte copies of B's episode npz into a copy of E3C30tab
for ep in (5, 7, 15):
    tag = f'episode_{ep:06d}'
    d = fresh(f'episodes_swap_s{ep}', E3)
    shutil.copy2(B_DIR / f'{tag}.npz', d / f'{tag}.npz')
    report['variants'][f's{ep}'] = dict(base='E3C30tab', from_B=[ep])
# B12: B plus the 9 other clean FP+RTS episodes
d12 = fresh('episodes_swap_b12', B_DIR)
for ep in NEW9:
    tag = f'episode_{ep:06d}'
    with np.load(E3 / f'{tag}.npz') as a:
        e3 = {k: a[k] for k in a.files}
    R_E, t_E = e3['object_rotation'].astype(np.float64), e3['object_translation'].astype(np.float64)
    T = len(t_E)
    R_c, t_c = load_track(CACHE / tag / 'foundationpose/smoothed', T)
    fin = np.isfinite(t_E).all(1) & np.isfinite(R_E).all((1, 2))
    Lc = L(t_c)
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
    sigma, c, _ = fit_sigma_c(Rr, Lc, t_E, fin.copy())
    tN = sigma * Lc @ Rr.T + c
    RN = Rr @ R_c
    dt = np.linalg.norm(tN[fin] - t_E[fin], axis=1)
    dr = geo(RN[fin], R_E[fin])
    rep = dict(shipped_frames=T, e3_nonfinite_frames=int((~fin).sum()), cache_frames=len(list((CACHE / tag / 'foundationpose/smoothed').glob('*.json'))),
               recorded_transfer_rotation_deg=round(rec, 3), rotation_candidates_deg={k: round(v, 3) for k, v in angles.items()},
               rotation_used=pick, rotation_used_minus_recorded_deg=round(angles[pick] - rec, 3),
               candidates_apart_deg=round(float(geo(Rr_rot, Ru)), 3), sigma=round(float(sigma), 5),
               e3c30tab_object_scale=round(float(np.asarray(e3['object_scale'])), 5),
               new_vs_E3C30tab_translation_mm=stats(dt), new_vs_E3C30tab_rotation_deg=stats(dr, 1.0, 2),
               frames_rotation_over_90deg=int((dr > 90).sum()))
    out = dict(e3)
    out['object_rotation'] = RN.astype(e3['object_rotation'].dtype)
    out['object_translation'] = tN.astype(e3['object_translation'].dtype)
    np.savez(d12 / f'{tag}.npz', **out)
    report['episodes'][tag] = rep
    print(tag, json.dumps(rep), flush=True)
report['variants']['b12'] = dict(base='B', new=NEW9)
DROP = [4, 16]
d12x = fresh('episodes_swap_b12x', d12)
for ep in DROP:
    shutil.copy2(B_DIR / f'episode_{ep:06d}.npz', d12x / f'episode_{ep:06d}.npz')
report['variants']['b12x'] = dict(base='B', new=[e for e in NEW9 if e not in DROP], restored_from_B=DROP)
# integrity: every file not meant to change is byte-identical to its base
import filecmp  # noqa: E402
for name, base, changed in (('s5', E3, {5}), ('s7', E3, {7}), ('s15', E3, {15}), ('b12', B_DIR, set(NEW9)), ('b12x', B_DIR, set(NEW9) - set(DROP))):
    d = OUT_ROOT / f'episodes_swap_{name}'
    files = sorted(p.name for p in base.iterdir())
    assert files == sorted(p.name for p in d.iterdir()), name
    diff = [f for f in files if not filecmp.cmp(base / f, d / f, shallow=False)]
    want = sorted(f'episode_{e:06d}.npz' for e in changed)
    assert diff == want, (name, diff, want)
    report['variants'][name]['files_differing_from_base'] = diff
    print(name, 'files differing from base:', diff, flush=True)
for ep in (5, 7, 15):
    f = f'episode_{ep:06d}.npz'
    assert filecmp.cmp(B_DIR / f, OUT_ROOT / f'episodes_swap_s{ep}' / f, shallow=False)
(OUT_ROOT / 'swap_build_report.json').write_text(json.dumps(report, indent=2))
print('saved', OUT_ROOT / 'swap_build_report.json')
