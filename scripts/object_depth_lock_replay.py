#!/usr/bin/env python3
"""Background-locked object depth, replayed on a finished candidate (10-29, configuration audit of the CARI4D container).
The container ships v2d.foundation_pose.lib.run_align_depth_to_reference_depth: before tracking, every frame's MoGe depth
is aligned (scale + shift, static background) to the reference frame's depth. Our track_object.sh feeds the raw per-frame
MoGe maps instead, so the object's distance inherits MoGe's per-frame scale wobble (0.85 % std, LEDGER 10-25). With a
static camera the alignment is the per-frame scale a_t of scripts/moge_lock.py (LOCK_DIR/<prefix>_<tag>.npz).
FoundationPose's distance follows the depth it is given, so tracking on locked depth moves the object along its camera
ray by a_ref / a_t (reference-frame normalised, as the container's tool) or a_anchor / a_t (scored anchor untouched).
MODE med (wiring check, 10-29): the human's depth scale s (scripts/sam3d_body_identity.py) is the median of
MoGe / SAM depth AFTER dividing each frame by its background wobble b_t (normalised to the median background), but the
object (raw per-frame MoGe through FoundationPose) is divided by the same s without that step (build_track1_episodes_sam.py),
so the object carries b_t relative to the human. The consistent object is t / a_t (a_t normalised to median 1 like b_t);
its episode mean stays ~1, so the gravity kappa stays valid.
The ESKF smooths the raw track (z noise 0.1 m vs process 0.005 m/frame, ~20-frame window), so a_t is smoothed (sigma)
before the factor is applied. The human and the object's size are unchanged.
Usage: object_depth_lock_replay.py ROOT LOCK_DIR PREFIX MODE SIGMA [IN]   (anchor: REL_CACHE_DIR/<prefix>_<tag>.npz, default ROOT/../rel_cache_v2_bm)   MODE ref | anchor | med; writes submissions/<IN>_lk<MODE><SIGMA>
"""
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np


def gsmooth(x, sigma):
    if sigma <= 0:
        return x
    k = int(3 * sigma); g = np.exp(-0.5 * (np.arange(-k, k + 1) / sigma) ** 2); g /= g.sum()
    return np.convolve(np.pad(x, k, mode="edge"), g, mode="valid")


def main():
    root, lock, pref, mode, sig = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4], float(sys.argv[5])
    name = sys.argv[6] if len(sys.argv) > 6 else "episodes_cand_BMGM"
    src, dst = root / "submissions" / name, root / "submissions" / f"{name}_lk{mode}{int(sig)}"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst); st = {}
    for f in sorted(dst.glob("episode_??????.npz")):
        tag = f.stem; z = dict(np.load(f)); t = z["object_translation"].astype(np.float64); T = len(t)
        a = np.load(lock / f"{pref}_{tag}.npz")["a"].astype(np.float64)[:T]
        a = np.where(np.isfinite(a) & (a > 0), a, np.nan); ok = np.isfinite(a)
        a = np.interp(np.arange(len(a)), np.flatnonzero(ok), a[ok]); a = gsmooth(a, sig)
        if mode == "ref":
            i0 = int((root / "outputs/object" / tag / "prompt/reference_frame.txt").read_text().strip())
        elif mode == "anchor":
            i0 = int(np.load(Path(os.environ.get("REL_CACHE_DIR") or root.parent / "rel_cache_v2_bm") / f"{pref}_{tag}.npz")["anchor"])            # first scored frame
        a0 = 1.0 if mode == "med" else a[min(i0, len(a) - 1)]
        fac = np.ones(T); fac[:len(a)] = a0 / a
        z["object_translation"] = (t * fac[:, None]).astype(np.float32); np.savez(f, **z)
        st[tag] = [float(np.abs(np.log(fac)).mean()), float(np.log(fac).std())]
    (dst / "lock_replay.json").write_text(json.dumps(st, indent=1))
    v = np.array(list(st.values()))
    print(f"PASS: {len(v)} episodes, mean |log factor| {v[:, 0].mean() * 100:.2f} %, within-episode std {v[:, 1].mean() * 100:.2f} % -> {dst.name}")


if __name__ == "__main__":
    main()
