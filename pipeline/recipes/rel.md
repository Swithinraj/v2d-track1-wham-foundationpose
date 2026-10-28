# Depth-reliability module: correct the object's distance where the evidence says so

`rel_final.sh` adds one step to a finished candidate, before its contact stage. A small learned module looks at
GT-free evidence for every frame and predicts how far the object (and the human) should move along its camera ray. It
also predicts how sure it is. It is trained to leave accurate tracks alone, which is where plain depth corrections did
harm in the past.

## Why

A decomposition against the multi-view FORM-HOI ground truth places the object error. For each component in turn, the
corresponding part of our object was replaced by the ground truth and the share of the object chamfer error removed was
measured, on validation group T3 and held-out groups C and U1.

| Component replaced by ground truth | T3 | C | U1 |
|---|---|---|---|
| Position | 89 % | 80 % | 87 % |
| Distance along the camera ray only | 45 % | 60 % | 51 % |
| Size, shape and orientation | about 0 % | about 0 % | about 0 % |

Most of the object error is therefore its distance along the camera ray. Feeding FoundationPose the recorded sensor
depth instead of MoGe-2 depth removes only part of it: 7 to 12 % of CD-O on T3.

## The module

**Features (`scripts/rel_cache.py`).** Computed per frame, for the human and the object:
- MoGe-2 depth at our joints, and inside the object mask, relative to our reconstructed distance. The per-frame scene
  scale wobble is removed with a background lock (`scripts/moge_lock.py`).
- Apparent size against our projection.
- Residual between ViTPose and our projected joints.
- Occlusion, closeness to the image border, and how much of the depth is valid.
- Motion along and across the ray.
- FoundationPose's raw-versus-smoothed jump.
- The distance from our hands to our object.

**Model (`scripts/rel_train.py`).** Two bidirectional GRU branches, one for the human and one for the object, exchange
information. For each frame they output a bounded correction (±10 %), a gate and a confidence. The correction is
tapered near the anchor frame and smoothed over time.

**Training.**
- Data: 14 FORM-HOI groups (480 episodes, full and body mode). The validation group is T3.
- Losses: Huber loss on the anchor-relative log distance ratio to the ground truth, plus a harm penalty, an
  acceleration term and a Gaussian negative log-likelihood.
- Targeted corruptions are recomputed consistently. They include smooth-but-wrong own trajectories, one-source drift
  and offsets, and occlusion noise.
- The deployed model is `config/rel_module/rel_best.pt`: the GRU variant, MoGe-only features, step 14000.

**Deployment.**
- `scripts/rel_deploy_prep.py` builds the same input fields from our own outputs, without ground truth.
- `rel_cache.py --deploy` computes the features.
- `rel_train.py --deploy --dump` writes the corrections.
- `scripts/rel_smooth.py` smooths the object correction again (σ 20 frames).
- `scripts/gru_rel_refit.py` applies the correction. It is `gru_depth_refit.py` with the depth head replaced by the
  module: the object's translation is multiplied by exp(c_o), so its size is unchanged, and the human moves along the
  ray through its centroid.

## Validation

All results are on the α-0 pipeline (`trend_final.sh`, `TREND_ALPHA=0`), at the final output after the contact stage.
Each row compares with the same pipeline without the module.

| FORM-HOI group | CD-O | ACC-O | CD-H, ACC-H, PEN |
|---|---|---|---|
| T3 (validation, σ chosen here) | −7.17 | −0.00016 | unchanged |
| U1 (untouched) | −3.31 | −0.00015 | unchanged |
| C (held out) | −2.30 | −0.00047 | unchanged |
| U2 (untouched) | −1.45 | −0.00023 | unchanged |

The human gate stays closed: without Video Depth Anything features, the module finds no human correction it can make
safely.

Without the second smoothing, CD-O is the same (−7.29, −3.41, −2.37, −1.44) but ACC-O gets worse on T3 and U1. The
earlier learned depth head at ×0.75 changes CD-H by +0.11 (T3) and +0.46 (U1), so it stays off.

## Reproduce

```
TREND_ALPHA=0 bash pipeline/recipes/trend_final.sh episodes_cand_BMQ3KGMa0     # or the BMQ3KGM recipes
bash pipeline/recipes/rel_final.sh episodes_bmq3k_in episodes_cand_BMQ3KGMa0R
bash pipeline/stages/pack.sh                                                    # with the candidate directory as the episodes input
```

## Version 2: overlay cues computed on visible parts (`REL_V2=1`, `config/rel_module_v2/rel_best.pt`)

Version 2 adds the overlay scale, meaning how large our reconstruction appears in the image compared with the image
evidence. It is computed only from parts that are visible:

- **Human.** Our projected bone lengths over ViTPose's. Only bones count whose two joints are both confidently detected
  by ViTPose and visible in the person mask. The gate also receives the number of such bones and the visible share of
  the body.
- **Object.** The observed mask area against our silhouette minus the part covered by the person mask. The gate also
  receives the object's visible share.

Each cue is given raw and smoothed over time (σ 30 frames).

Two training changes:

- The synthetic distance shifts now move every overlay feature consistently. This also fixes a sign error in version 1:
  moving an object farther away makes its "observed / projected size" larger, not smaller.
- The uncertainty term is computed on the correction that is actually applied.

Results on the production pipeline (BMQ3KGM, trend α 0.75), at the scored output, compared with the same pipeline
without the module:

| FORM-HOI group | CD-O | ACC-O | CD-H, ACC-H, PEN |
|---|---|---|---|
| T3 (validation) | −4.78 | −0.00008 | ≈0, 0, +0.00001 |
| U1 | −2.64 | −0.00016 | ≈0 |
| C | −1.57 | −0.00040 | ≈0 |
| U2 | −1.52 | −0.00015 | ≈0 |

Version 2 harms 4–5 % of frames on the depth proxy, against 16–17 % for version 1.

On Kaggle, version 1 (BMQ3KGMR) improved ACC-O but made CD-O worse, by +0.47.
