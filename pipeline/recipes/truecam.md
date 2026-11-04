# The measured rig camera instead of MoGe-2's camera

`truecam_final.sh` re-expresses a finished candidate's pre-contact output in the camera that actually recorded the
video, then runs the usual contact stage.

## Why

Every stage of the pipeline uses one camera per episode, `outputs/camera`, the median of MoGe-2's per-frame guesses
(`scripts/camera_intrinsics.py`). SAM 3D Body, FoundationPose, the depth cues and the contact solver all project with
it. FORM-HOI records its videos with four fixed rig cameras whose calibration is part of the dataset.
`scripts/camera_truth_check.py` compares the two over 1488 FORM-HOI episodes:

| | MoGe-2 (ours) | Measured |
|---|---|---|
| Focal length | 3.4 % short (median; 5-95 %: 1.6-5.5 %) | 840.6-856.6 px |
| Principal point | image centre (768, 576) | (713.5-727.8, 527.5-544.2): 54 / 38 px away, 4.3° |
| Calibration across recording days | per frame | one per camera over 34-46 days |

The principal-point error shears the scene: with the image evidence fixed, a point at depth z sits on a ray that is
about z × 54 / f too far to the side, roughly 6 cm per metre of depth. The scorer fits one similarity transform on the
human at the first scored frame, so a depth-dependent shear is not removed. The focal error stretches depth relative to
the sideways directions; the depth trend stage already absorbs most of it.

## The method

Under the pipeline's own model, a change of camera keeps every pixel and changes only its 3D interpretation. For one
episode this is a single linear map (`scripts/true_cam_params.py`):

- **Human.** The MHR root pivot moves by the map. The body turns with its viewing ray, because SAM 3D Body predicts its
  orientation relative to that ray. Its shape is unchanged. A second MHR pass checks the result against the target
  rigid move: 0.0000 mm on every episode.
- **Object.** The translation moves through the object's own FoundationPose camera. The rotation turns with its viewing
  ray, and the scale keeps the same apparent size.

Then the contact stage re-solves the hands against the moved objects, with its image term projecting through the
measured camera (`CONTACT_ALIGNMENT_DIR=outputs/alignment_measured`).

**Which camera an episode gets.** `scripts/camera_identify.py` matches each episode's static background against
FORM-HOI backgrounds of known camera (about 1500 videos).
- 18 of the 30 challenge episodes match one rig camera clearly: correlation 0.90-0.99, against 0.34-0.41 for any other
  camera.
- The other 12 come from scenes that are not in FORM-HOI. They take the rig mean
  (`config/camera/episode_cameras.json`, `scripts/measured_camera.py`). Every camera's principal point lies within
  about 9 px of the mean.

## Validation on FORM-HOI

Four held-out groups, production pipeline (BMGM = BMQ3KGM's recipe). The measured camera is applied to the motion
head's output, then the contact stage runs again. Each value is the change against production.

| Group | CD-H | CD-O | ACC-H | ACC-O | PEN |
|---|---|---|---|---|---|
| T3 | **−1.46** (better 24/30) | +0.62 | −0.00002 | 0 | −0.00003 |
| U1 | **−0.77** (18/30) | +0.48 | +0.00007 | +0.00006 | +0.00001 |
| C | **−0.91** (23/30) | −0.33 | +0.00004 | 0 | −0.00001 |
| U2 | **−1.00** (22/30) | −0.45 | +0.00002 | +0.00011 | −0.00004 |
| Mean | **−1.04** | +0.08 | +0.00003 | +0.00004 | ≈0 |

Other variants tested:

- **Measured camera before the motion head.** CD-H is the same (−1.04), but ACC-H is +0.00022 on every group. The
  motion head was trained in MoGe-2's camera.
- **Focal length only.** CD-H ≈0.
- **Principal point and focal length.** CD-H −0.94, CD-O +0.56.

## Reproduce

```
bash pipeline/recipes/truecam_final.sh episodes_cand_BMQ3KGM_in episodes_cand_BMQ3KGMtp   # after the BMQ3KGM recipes
bash pipeline/stages/pack.sh                                                               # PACK_EPISODES=episodes_cand_BMQ3KGMtp
```

Regenerate `config/camera/episode_cameras.json` with
`scripts/camera_identify.py FORMHOI_DIR TRACK1_ROOT OUT 100000`, using the FORM-HOI roots with their videos and metadata.
