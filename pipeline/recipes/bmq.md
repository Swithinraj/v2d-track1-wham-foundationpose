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

## Version BMQ-K: objects in the body-mode run's depth scale

BMQ re-derived the MoGe/SAM depth scale s from the body-mode raw output (median +0.53 %, range −0.12..+1.22 %) but kept
VPQ's objects, which the episode builder expresses with the full-mode s (`object_translation = t_co / s`,
`object_scale = s_co / s`). The human moved along the camera rays with s and the objects did not; the first-frame
Sim(3), fitted on the human, then moved the objects (BMQ's CD-O +0.26 with byte-identical objects).
`scripts/rescale_objects.py` multiplies object translation and scale by s_full / s_body (a scaling about the camera:
projection unchanged) on BMQ's pre-contact episodes; then the identical contact stage (`bmqk_final.sh`).
- Round trip old → new → old: max |Δt| 0.0002 mm; human and object rotation untouched before the contact stage.
- Local (challenge episodes): contact stage changes the human by 0.02 cm; self-PEN 0.000288 (BMQ 0.000294).
- **FORM-HOI validation** (60 held-out-style sequences with multi-view GT, `nvidia/form-hoi`, official metric code, two
  roots of 30): BMQ-K vs BMQ CD-O −0.77 (root A) / −0.83 (root B); BMQ vs VPQ2 CD-H −1.60 / −0.29, ACC-H −0.0004 / −0.0001.

## Version BMQ3-K: BMQ-K + VPQ3's restored image-plane motion

VPQ3's lateral band refit (`lateral_band_refit.py`, λ 100) on BMQ-K's pre-contact human, raw reference = the body-mode
human before refinement, visibility from the body-mode base bodies, then the identical contact stage (`bmq3k_final.sh`).
- Local vs BMQ-K: ViTPose 5.41 vs 5.51 px, wrist 5.26 vs 5.51 px; noise-cancelling 2D acceleration proxy MediaPipe −9.38
  (se 1.76) and ViTPose −9.86 (se 1.67), better in 28/30 episodes each; articulation self-acc ×1.040 (real motion added,
  as VPQ3 vs VPQ2 ×1.035, which Kaggle scored ACC-H −0.0004); objects identical; self-PEN 0.000291.
- FORM-HOI: the same refit on VPQ2's base improves CD-H in 29/29 and 29/30 episodes and ACC-H in 29/29 and 29/30
  (ACC-H −0.0006 / −0.0004), matching Kaggle's VPQ2 → VPQ3 (−0.0004).
