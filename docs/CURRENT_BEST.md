# Current best submission (2026-10-27)

**BMQ3KGM** is the best single file on every Track 1 board: built from commit `f60f8a6`. Its recipes are unchanged in
this commit.

| File | CD-H | CD-O | ACC-H | ACC-O | PEN | Rank sum (single file, 10-25 board) |
|---|---|---|---|---|---|---|
| **BMQ3KGM** | **14.35418** | 18.60437 | **0.13629** | **0.07067** | **0.00013** | **39** |
| BMQ3KGMT (+ common-translation selector, `a0d93f8`) | 14.41266 | 18.58305 | 0.13629 | 0.07067 | 0.00013 | (no rank change) |

What BMQ3KGM is: the BMQ3-K pre-contact human (SAM 3D Body in body mode, objects re-expressed in the body-mode depth
scale, VPQ3 lateral band), then the GRU v2 motion head fine-tuned with depth cues, with its depth correction switched
off, then the contact stage with penetration weight 4e6. The depth source of every stage is MoGe-2 (`object_depth.sh`).

Rebuild (after `run_all.sh`):

```
bash pipeline/recipes/bmq_final.sh            # body-mode SAM 3D Body run (BM_ROOT) and episodes_bmq_in
bash pipeline/recipes/bmqk_final.sh           # objects in the body-mode depth scale (episodes_bmqk_in)
bash pipeline/recipes/bmq3k_final.sh          # lateral band -> episodes_bmq3k_in
EPS_SCALE=0 bash pipeline/recipes/gru_depth_final.sh episodes_bmq3k_in episodes_bm_depth mask_visibility_bm BM_ROOT episodes_cand_BMQ3KGM
bash pipeline/stages/pack.sh                  # with the candidate directory as the episodes input
```

The common-translation selector (`gru_shift_final.sh`) is kept in the repository but does not beat BMQ3KGM: on Kaggle it
was CD-H +0.058 and CD-O −0.021, and on two untouched FORM-HOI groups its effect is near zero.

## Candidate submitted for scoring: BMQ3KGMa0 (depth trend off)

BMQ3KGMa0 is BMQ3KGM with one change: the human depth trend is switched off (`TREND_ALPHA=0`, see
`pipeline/recipes/trend.md`), and the objects are re-expressed for the changed full-mode human through the gravity
kappa. On four FORM-HOI groups, scored against multi-view ground truth at the scored output, the change gives
CD-H −0.73 and CD-O −0.96 on average. ACC-H, ACC-O and PEN are unchanged. The Kaggle values will be added once scored.

```
bash pipeline/recipes/bmq_final.sh                                   # body-mode SAM 3D Body (BM_ROOT)
TREND_ALPHA=0 bash pipeline/recipes/trend_final.sh episodes_cand_BMQ3KGMa0
bash pipeline/stages/pack.sh                                         # with the candidate directory as the episodes input
```

`scripts/gru_depth_refit.py` now builds its per-vertex band fusion in vertex chunks. The filters act along time per
vertex, so the output is unchanged: at most 0.0002 mm against the unchunked code on the same GPU. The change lets the
step run on 16 GB GPUs.

## Video Depth Anything: not adopted

Video Depth Anything (metric, Large) was tested in three ways:

1. **As the only depth source**, re-tuned. The human is slightly better, but the objects are worse by about 25 CD-O,
   because FoundationPose tracks worse on VDA depth.
2. **For the human depth trend only** (`trend.md`, BMQ3KGMh). Better on the validation group, worse on the held-out
   groups.
3. **Against sensor depth.** VDA follows the true change in the person's distance less well than MoGe-2.

None of these is part of a submission.

## Kaggle 2026-10-28: BMQ3KGMa0, and the candidates that keep BMQ3KGM's human

| File | CD-H | CD-O | ACC-H | ACC-O | PEN |
|---|---|---|---|---|---|
| BMQ3KGM | 14.35418 | 18.60437 | 0.13629 | 0.07067 | 0.00013 |
| BMQ3KGMa0 (`0bfe450`) | 14.70203 | 18.01952 | 0.13622 | 0.07066 | 0.00012 |

Switching the depth trend off improved the objects but hurt the human on the challenge data. On FORM-HOI it had improved
both, so the human result does not transfer.

The next candidates therefore keep BMQ3KGM's human and change only the objects:

- **BMQ3KGMR**: BMQ3KGM plus the depth-reliability module (`pipeline/recipes/rel_final.sh`, `rel.md`).
- **BMQ3KGMk**: BMQ3KGM with the α-0 object calibration (`pipeline/recipes/kcal_final.sh`).
- **BMQ3KGMkR**: both changes together (`kcal_final.sh OUTPUT rel`).

**Reproducibility.** Running `rel_final.sh` from this repository reproduces the module's object corrections exactly
(at most 2e-7) and its motion-head output exactly (scored joints within 0.0002 mm). After the contact stage, repeated
runs of the existing contact solver differ by 0.07 mm on average in the scored joints, with a few hand and arm frames
up to several centimetres. That variation comes from the solver itself; this change does not introduce it.

### Kaggle 2026-10-28, candidates that keep BMQ3KGM's human (commit 321e50d)

| File | CD-H | CD-O | ACC-H | ACC-O | PEN |
|---|---|---|---|---|---|
| BMQ3KGMR (+ module v1) | 14.36408 | 19.07004 | 0.13628 | 0.07056 | 0.00016 |
| BMQ3KGMk (α-0 object calibration) | 14.35430 | 18.68588 | 0.13629 | 0.07066 | 0.00013 |
| BMQ3KGMkR (both) | 14.37471 | 19.17272 | 0.13629 | 0.07055 | 0.00014 |

**Module v1 does not transfer on CD-O**, although it improves ACC-O.

**The α-0 object calibration alone does nothing.** So BMQ3KGMa0's CD-O gain came from its human: the scorer places
the objects through a similarity fitted on our human at the first scored frame.

The next candidate is **BMQ3KGMR2**: BMQ3KGM plus module version 2 (`REL_V2=1`, see `pipeline/recipes/rel.md`).
