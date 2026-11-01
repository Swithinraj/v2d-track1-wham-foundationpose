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

## Kaggle round 1 (2026-11-01 22:00 UTC, commit 13ff13a)

| File | CD-H | CD-O | ACC-H | ACC-O | PEN |
|---|---|---|---|---|---|
| BMQ3KGM (reference, in MoGe's camera) | 14.35418 | 18.60437 | 0.13629 | 0.07067 | 0.00013 |
| MF: camera + person floor slide + object floor scale (17 objects) | **14.08779** | 18.49797 | 0.13637 | 0.07057 | 0.00019 |
| G97: camera + every object scaled 0.97 | 14.15256 | 22.65556 | 0.13636 | 0.07053 | 0.00016 |
| WF: camera + WHAM root mix 0.25 + floor objects + table objects 0.974 | 14.74125 | 20.91912 | 0.13632 | 0.07051 | 0.00014 |

- **Measured camera on the best file works:** CD-H -0.20 (G97 against BMQ3KGM, the person being the same).
- **The person's floor slide transfers:** CD-H -0.065 more (MF against G97). Its local cross-check with apparent size had
  not confirmed it, which is why only Kaggle decides.
- **WHAM root mix is rejected:** CD-H +0.59 at weight 0.25. Fitting a parabola gives an optimal weight of about 0.03 and a gain
  of about 0.01, so the independent-backbone idea does not pay with WHAM.
- **"Objects are too far" is false as a global claim.** Scaling all objects by 0.97 costs CD-O about +4. With nowham this puts the
  optimum of a global scale near 1.01. The sensitivity is about 1.25 CD-O per 1 % of scale.
- **The floor scale still carries per-episode information.** MF's floor objects have about the same mean scale as G97's 0.97
  (mean 0.974), yet are about 2.5 CD-O better per floor episode, so the differences between episodes are real.
- **Table objects should stay.** 0.974 on the 13 table objects costs +5.6 CD-O per such episode (WF against MF). So the
  size-constancy candidate SC, as built on floor calibration, is not uploaded as it stands.

## Kaggle round 2 (2026-11-01, commit 1280cc0)

| File | CD-H | CD-O | ACC-H | ACC-O | PEN |
|---|---|---|---|---|---|
| ML3: camera + person slide lam 3000, objects unchanged | 14.17711 | 19.39916 | 0.13644 | 0.07067 | 0.00013 |
| FB15: camera + person slide strength 1.5 + object floor scale, 2 cm floor offset | 15.20630 | **17.97125** | 0.13646 | 0.07060 | 0.00012 |

- ML3's objects are unchanged, so **19.40 is the camera-only CD-O**. The object floor scale is worth -0.90 (MF), and the 2 cm offset
  a further -0.53. The measured camera costs the objects +0.80 against BMQ3KGM, hence `scripts/object_transfer.py`.
- Person slide strength: 14.153 at 0, 14.088 at 1, 15.206 at 1.5. A parabola puts the best strength near 0.5 and predicts about
  13.74. Lighter smoothing (lam 3000) is worse than lam 10000.
