# BMQ3KGM with a different human depth trend

`trend_final.sh` builds BMQ3KGM with one change: the strength and the source of the human depth trend
(`sam3d_body_depth_trend.py`). Every other stage and setting is BMQ3KGM's.

## What the trend does

The identity step (`sam3d_body_identity.py`) records, every third frame, the ratio of the depth map at the person to
SAM 3D Body's own depth, normalised by the map's drift on static background pixels (`qn`). The trend stage
median-filters and smooths this series, then moves the human along the camera ray by `alpha * (trend - 1)` of its
distance. It was inherited from the MoGe-2 pipeline with `alpha = 0.75`.

## Why change it

The check uses FORM-HOI, the labelled parent dataset of Track 1. Its multi-view ground truth gives the person's true
distance change in our frame.

- MoGe-2's `qn` follows the true change only weakly: rank correlation +0.25 to +0.45 depending on the group.
- Video Depth Anything's `qn` is smoother but follows it even less (+0.13 to +0.18).
- At `alpha = 0.75` the trend adds error on most groups.

| Candidate | `TREND_ALPHA` | `qn` source |
|---|---|---|
| BMQ3KGM (reference) | 0.75 | MoGe-2 |
| BMQ3KGMa0 | 0 | none (trend off) |
| BMQ3KGMh | 0.25 | Video Depth Anything (`TREND_PREP_FULL`, `TREND_PREP_BM`) |

Effect of α 0 against α 0.75 on the full chain's pre-contact human, FORM-HOI groups, scored as the challenge scores
(CD-H / CD-O deltas):

| Group | CD-H | CD-O |
|---|---|---|
| T3 (validation) | −1.61 | −1.05 |
| C (held out) | −0.84 | −0.98 |
| U1 (untouched) | −0.52 | −2.17 |
| U2 (untouched) | +0.09 | +0.36 |

ACC-H, ACC-O and PEN are unchanged within ±0.0001.

After the motion head and contact stage (the scored output):

| Group | CD-H | CD-O | ACC-H | ACC-O | PEN |
|---|---|---|---|---|---|
| T3 | −1.63 | −1.05 | −0.00005 | 0 | 0 |
| C | −0.82 | −0.98 | −0.00001 | +0.000005 | −0.00001 |
| U1 | −0.53 | −2.17 | −0.00002 | 0 | −0.00007 |
| U2 | +0.08 | +0.36 | 0 | 0 | 0 |
| Mean | −0.73 | −0.96 | | | |

The U2 changes are within one standard error.

## Objects

The full-mode human enters the objects only through the gravity kappa (`ipf30.md`). Kappa is recomputed on the
full-mode human with the new trend, and on the α-0.75 human the objects were built with. VPQ's objects are then
re-scaled about the camera by the ratio (`scripts/kappa_rescale_objects.py`), which is what `kappa_gravity.py` itself
applies.

## Video Depth Anything `qn` (BMQ3KGMh, not adopted)

BMQ3KGMh is slightly better than BMQ3KGMa0 on the validation group T3: CD-H −0.14 at the scored output. On the held-out
groups C and U1, however, it is worse at the body-mode pre-contact stage: CD-H +0.26 and +0.48, CD-O +0.32 and +0.23.
VDA's trend does not generalise, so BMQ3KGMa0 is the candidate. The VDA option is kept for reference.

Only the trend reads VDA. The object meshes, FoundationPose tracking, depth scales, floor and kappa all stay on
MoGe-2. On FORM-HOI, FoundationPose on VDA depth costs about +25 CD-O, which is why VDA is used for the trend only.

The steps:

1. `scripts/vda_depth.py` runs metric VDA ViT-L on each video: input size 518, fp16, a 1280 px pre-resize, then one
   constant per episode to bring it to MoGe-2's scale. The constant cancels in `qn`.
2. `pipeline/stages/vda_prep.sh RAW OUT` re-runs the unchanged identity step on the VDA maps. Run it once for full
   mode (`RAW = outputs/sam3d_body`) and once for body mode (`RAW = BM_ROOT/outputs/sam3d_body`).

On a FORM-HOI group, this reproduces exactly (difference 0) the depth series of a pipeline run whose only depth source
was VDA.

## Reproduce

```
bash pipeline/recipes/bmq_final.sh                       # body-mode SAM 3D Body (BM_ROOT), as for BMQ3KGM
TREND_ALPHA=0 bash pipeline/recipes/trend_final.sh episodes_cand_BMQ3KGMa0
# or, BMQ3KGMh:
VDA_PYTHON=... VDA_REPO=... VDA_WEIGHTS=... bash pipeline/stages/vda_prep.sh outputs/sam3d_body work/vda_prep_full
bash pipeline/stages/vda_prep.sh work/bm_root/outputs/sam3d_body work/vda_prep_bm
TREND_ALPHA=0.25 TREND_PREP_FULL=work/vda_prep_full TREND_PREP_BM=work/vda_prep_bm \
  bash pipeline/recipes/trend_final.sh episodes_cand_BMQ3KGMh
bash pipeline/stages/pack.sh                             # with the candidate directory as the episodes input
```
