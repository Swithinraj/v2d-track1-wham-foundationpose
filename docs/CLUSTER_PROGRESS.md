# Cluster experiment ledger — 2026-11-01

Plan: `docs/superpowers/plans/2026-11-01-cluster-geometry.md`.

- Restored `vpq-vitpose` at `00c6e73` and the handoff's uncommitted changes.
- Read handoff, CURRENT_BEST, EXPERIMENT_REPORT, GRU/contact/camera/depth code.
- Slurm probe job 417829 completed on wr20: NVIDIA A100-SXM4-80GB.
- Drive credentials work; selected artifacts are restoring under repo/.
- Python 3.11 and submission dependencies installed within the workspace.
- Ruling: use a fresh clone as isolation; preserve branch and leave changes uncommitted.
- Ruling: first-frame human alignment accuracy is not absolute camera-space hand
  accuracy. Contact correction therefore requires explicit observability and image
  validation; the handoff's optimism alone is not sufficient evidence.
- Ruling: no new-code candidate may claim an old code commit reproduces it.
- Pre-flight: geometry consumes pre-contact MHR hands in metres and existing R/t/scale;
  downstream contact solver consumes the corrected NPZ/GLB and the matching camera.

## Evidence after the handoff (2026-11-01)

- Kaggle history: the GRU checkpoint swap was already scored (CD-H 14.34317, CD-O 19.35116): no gain. The handoff's top todo
  item is therefore closed. Removing the measured camera: CD-H 14.62436, CD-O 19.01432.
- Public board on 2026-11-01: best CD-H 12.127, best CD-O 16.224. No team is near 10 cm. Our bests are 14.310 (rank 17) and
  18.020 (rank 8).
- `scripts/moge_lock.py` now fails closed on empty or sparse background and on unsupported floor evidence (`tests/test_moge_lock.py`).
- `scripts/contact_translation.py`, a constant object translation from hand contact, abstains by default. Its first real
  pilot rejected its own proposal because contact did not improve. It is not part of any candidate.
- Submission rule: five per week, unlimited in the final three days (from 2026-11-01 22:00 UTC). The team made 13 uploads in
  the seven days before that, so nothing new is uploaded before the window opens (`scripts/submit_candidate.py` enforces it).
- Provenance: earlier Oct 31 files cite `2cd29f4` although some used scripts that commit does not contain. From here on, every
  file cites a pushed commit holding its code (`pipeline/recipes/pack_candidate.sh` checks), on branch
  `cluster-floor-2026-11`, with the owner's approval. `vpq-vitpose` is untouched.
- Analysis and first candidates: `pipeline/recipes/floor.md`.
- Compute: every job is pinned to the one extra node, `wr20`.

## Tried without a submission, and dropped (2026-11-01)

- **Depth stretch per episode.** Floor and apparent size both fit about 1.0 (0.99 and 1.01) and disagree per episode
  (correlation -0.04). No systematic compression to correct.
- **Body yaw from walking direction.** A yaw error at the alignment frame would swing the trajectory, but there is almost no
  walking in the scored frames and the offsets are inconsistent (`scripts/analysis/yaw.py`).
- **Object depth from the table's height.** On a table the viewing ray is nearly level: 5 cm of mesh error would mean a 10 %
  scale, so this is too sensitive to use.
- **Smoothing MHR pose parameters directly.** Euler parameters wrap, and joint acceleration got 16 times worse. Caught by the
  local probe, not uploaded. Smoothing the mesh and refitting (`scripts/human_refit.py`) works.

## Independent evidence for object depth (2026-11-01)

One physical object has one size, so size differences between episodes are depth differences: scaling about the camera keeps
every pixel. Within objects this agrees with floor contact (correlation 0.55, slope 0.62, 15 comparisons,
`scripts/analysis/size_constancy.py`). Floor-calibrated sizes also give table objects a depth scale for the first time
(`scripts/object_true_size.py`).

## The official pipeline was never scored

`docs/BASELINE.md` names NVIDIA's CARI4D MHR pipeline as the baseline to run first, yet no Kaggle submission used it, and the
earlier audit set it aside because CoCoNet needs SAM 3D Body decoder buffers. Its full inference pipeline produces those itself.
The checkpoint (`nvidia/cari4d_commercial`) is trained on 2,126 sequences with MoGe-2 and outputs MHR. In the CARI4D paper the
refinement network cuts object error by about 30 % (16.85 to 11.59 on BEHAVE). `pipeline/recipes/cari4d_episode.sh` runs it here.
