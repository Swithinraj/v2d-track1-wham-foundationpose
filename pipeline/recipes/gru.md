# Learned temporal refiner (GRU motion head)

`scripts/gru_fusion_refit.py` replaces E2's fixed band-fusion coefficients (`band_fusion.md`) with time-varying ones. A small bidirectional GRU predicts them per frame and per body region (torso, arms, legs).

## How it works
- **Inputs, per frame:**
  - the disagreement between the refined human and the raw per-frame human in four temporal bands (Gaussian differences σ 1/2/4/8, lateral and camera depth separately);
  - the refined human's own band energies;
  - ViTPose keypoint confidence;
  - mask visibility per joint.
- **Output:** bounded corrections (0.5·tanh, smoothed in time with σ 6 frames) around E2's half-strength coefficients. The lowest band stays the refined human's.
- **Deployment:** every vertex takes its region's coefficients, the anchor frame is tapered, and the kit's `lm_pose` refit follows. Objects are unchanged.

## Training (`scripts/gru_train.py`)
- **Data:** FORM-HOI (nvidia/form-hoi, CC BY 4.0) recordings run through this pipeline. Real pipeline errors are measured against the multi-view ground truth, mapped into our frame by the evaluator's own similarity at the first scored frame. No challenge data is used.
- **Loss:** joint position, 50 × acceleration error **against ground truth** (not plain smoothing), and coefficient total variation.
- **Augmentation:**
  - left/right mirror;
  - ±10 % scale and ±10° roll about the camera;
  - time reversal;
  - raw-error amplitude 0.7–1.3;
  - occlusion dropout of confidence and visibility;
  - simulated pipeline failures: 15 % of windows get the raw error ×2–6 and use the acceleration loss only, so the model stays conservative on out-of-range input.
- **Shipped model (`config/gru_motion/`):** trained on 117 episodes (4 recording-disjoint groups); the checkpoint was chosen on a fifth group of 30 episodes.

## Fresh-group confirmation
Measured on 30 more recordings, never used for any choice, with the kit metrics after this recipe's full contact stage. Changes are against the same base and contact stage without the GRU:

| Variant | ACC-H | CD-H | CD-O, ACC-O, PEN |
|---|---|---|---|
| GRU | **−0.0052 (better in 30/30)** | **−0.098 (27/30)** | unchanged |
| E2 | −0.0018 | −0.032 | unchanged |

## Recipe
`gru_final.sh IN RAW VISIBILITY OUTPUT`: the refiner on the pre-contact human, then the identical contact stage of `band_fusion_final.sh`.

| Candidate | Command |
|---|---|
| BMQ3-K + GRU | `gru_final.sh episodes_bmq3k_in episodes_bm_depth mask_visibility_bm episodes_cand_BMQ3KG` |

## Contact-stage penetration weight
`gru_pen_final.sh GRU_IN OUTPUT` reruns the identical contact stage on the GRU's pre-contact output, with the penetration weight in both contact solves raised from 2e5 to 2e6. The variables, priors and objects are unchanged; only the price of a hand point inside the object goes up.

Fresh-group results (30 recordings, kit metrics), against the same GRU output with the production contact stage:

| Penetration weight | PEN | ACC-H | CD-H | CD-O, ACC-O |
|---|---|---|---|---|
| 2e5 (production) | 0.00044 | — | — | — |
| **2e6** | **0.00017 (better in 27/30, worse in none)** | +0.00009 | +0.002 | unchanged |
| 2e7 | 0.00010 | +0.00027 | +0.011 | unchanged |
| 2e8 | 0.00009 | +0.00059 | +0.032 | unchanged |

2e6 is used: stronger weights buy little more PEN for a growing cost on the human. On the challenge episodes (no ground truth) the scorer-style self-penetration falls from 0.000319 to 0.000116, the anchor frame is unchanged and the human's own acceleration rises by ×1.004.

| Candidate | Command |
|---|---|
| BMQ3-K + GRU + PEN | `gru_pen_final.sh episodes_cand_BMQ3KG_gru_in episodes_cand_BMQ3KGP` |
