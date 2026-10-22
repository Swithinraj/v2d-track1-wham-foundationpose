# Candidate BMQ: the human from SAM 3D Body's body mode with a mask prompt

BMQ is VPQ2 (`vpq2_final.sh`) with one change: how the raw human is inferred.

## The change

| | VPQ2 (reference) | BMQ |
|---|---|---|
| `inference_type` | `full` | `body` |
| Prompt | box of the actor's SAM 2 mask | the SAM 2 actor mask itself (mask conditioning, `MASK_EMBED_TYPE v2`) with the same box |

- **What `full` mode adds:** the hand decoder (high-resolution hand crops whose pose replaces the body decoder's hands when four validity checks pass) and a wrist/elbow keypoint re-prompt that re-solves the arms in the body decoder.
- **Why test `body` mode:** it is the per-view path of the organizers' own MHR estimation code (`estimate_mhr_params`). Body mode drops both of those additions.

**Identical:**
- checkpoint, MHR asset and library code;
- frames at full resolution;
- person boxes and intrinsics;
- every downstream script and setting (identity medoid, refit, depth trend, refinement λ300 with ViTPose landmarks, VPQ's objects, hand fix, arm clearance, depth smoothing, object smoothing λ100, contact stage).

Re-derived from the new raw output with the same code:
- identity;
- MoGe/SAM depth scale (median +0.5 %);
- depth trend;
- SAM 2D keypoints;
- hand points.

The ViTPose hip calibration (`outputs/kp2d_vpx`) is unchanged.

## Reproduce

```
pipeline/recipes/bmq_final.sh episodes_cand_BMQ
```

Inputs are pinned in `bmq_inputs.sha256`: `submissions/episodes_objects_vpq` and `outputs/kp2d_vpx`, identical to VPQ's pins.

**Container note.** The scored raw inference ran in `v2d_sam3d_body.sif`, the recipe's default. As a control, full mode + box in that image vs the reference image changed the scored joints by 0.09 cm (noise-like). The body-mode change is 1.8-2.0 cm.

## Local validation vs VPQ2 (critic gates, `experiments/hmr_council_20261022`)

| Gate | Result |
|---|---|
| (a) objects | byte-identical |
| (b) systematic change after the first-frame Sim(3) | 1.20 cm (raw 1.24), arm \|mean\|/rms 0.61 (needed ≥ 0.5 cm and ≥ 0.3): a real, arm-weighted change |
| (c) episodes above 3× median | ep9 only |
| (d) Sim(3) hand shift at the anchor | mean 2.06 cm, max 4.53 cm |
| (e) self-PEN | 0.000294 vs 0.000342; ep27 0.00082 vs about 0.004 |
| (f) articulation self-acc | ×0.992 |

| Other checks | BMQ | VPQ2 |
|---|---|---|
| ViTPose 2D error | 5.52 px | 5.49 px |
| Wrist error | 5.52 px | 6.25 px |
| Flips | 0 | 0 |

- **Noise-cancelling 2D acceleration proxy:** MediaPipe −1.38 (se 3.30), better in 19/30 episodes. Self-acceleration on excluded frames +25 %. Estimated ACC-H change −0.00015.

**Pre-registered Kaggle decision (CD-H):**
- success: CD-H ≤ 14.42 with CD-O ≤ 18.498;
- failure: CD-H ≥ 14.52.
