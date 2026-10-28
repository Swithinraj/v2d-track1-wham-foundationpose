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
