# Track 1 experiment report (status 2026-10-29)

This report lists what has been tried for V2D Track 1 so far: the files scored on Kaggle, the experiments behind them,
and the configuration audit of 2026-10-29. That audit found that every stage used a wrong camera.

## Summary

- **Best scored file: BMQ3KGM** (commit `f60f8a6`). CD-H 14.354, CD-O 18.604, ACC-H 0.13629, ACC-O 0.07067,
  PEN 0.00013.
- **New finding: the camera.** Every stage used MoGe-2's camera guess, which puts the principal point at the image
  centre. The rig cameras' real principal point lies about 54 / 38 px away (4.3°), and their real focal length is
  3.4 % longer. All four rig calibrations are part of FORM-HOI and are constant over every recording day.
- **New candidate: BMQ3KGMtp.** BMQ3KGM re-expressed in the measured camera improves CD-H by 1.04 on average on four
  held-out FORM-HOI groups (better in 18-24 of 30 episodes per group). The other four metrics are unchanged.
  It is built and checked but not yet scored on Kaggle.
- **Human changes on FORM-HOI often did not transfer to the challenge.** Several gains measured against FORM-HOI's
  multi-view ground truth did not carry over to Kaggle: the learned depth correction, the depth trend switched off,
  and the object correction modules. What did transfer: the GRU motion head (ACC-H), the stronger penetration weight
  (PEN), the gravity scale (CD-O) and the object smoothing (ACC-O).

## Pipeline

A monocular video goes through these stages:

1. **Human.** SAM 3D Body gives the human, with one shared identity per episode.
2. **Depth.** MoGe-2 gives metric depth and the camera.
3. **Object shape and pose.** SAM 3D Objects builds one mesh per object. FoundationPose (v2d / CARI4D container)
   tracks it, and an ESKF/RTS smoother cleans the track.
4. **Placement and physics.** Objects are placed in the human's depth scale. Then come the depth trend, the gravity
   scale (kappa) and the physics smoothing.
5. **Body-mode human and lateral band** (BMQ3-K).
6. **GRU motion head.**
7. **Contact stage**, with penetration weight 4e6.

`docs/CURRENT_BEST.md` gives the recipes.

## How results are measured

**Kaggle.** One Parquet file is scored on five boards: CD-H, CD-O, ACC-H, ACC-O and PEN. The scorer fits one
similarity transform on our human at the first scored frame and applies it to the objects too.

**FORM-HOI.** FORM-HOI (`nvidia/form-hoi`) is the labelled parent dataset of Track 1. Each local group holds
30 recordings, and every result here is a paired per-episode difference against the same baseline.

| Group | Use |
|---|---|
| A, B, T1, T2, N1-N10 | training |
| T3 | validation; every choice is made here |
| C | held out |
| U1, U2 | untouched: recordings that informed nothing |

Scoring uses the kit's own metric code: `scripts/fh_export.py` and `scripts/fh_score.py`. "Better n/30" is the number
of episodes that improved.

## Kaggle submissions

| File (commit) | CD-H | CD-O | ACC-H | ACC-O | PEN | What changed |
|---|---|---|---|---|---|---|
| no-WHAM baseline (`70c79b2`, 10-17) | 14.670 | 24.257 | 0.13955 | 0.07461 | 0.00137 | SAM 3D Body + SAM 3D Objects |
| gravity-kappa probe (10-17) | — | 18.77 | — | — | — | objects scaled so the floor meets the feet |
| BMQ (`492627c`, 10-22) | 14.417 | — | — | — | — | body-mode SAM 3D Body |
| BMQ3KG (`4539102`, 10-24) | 14.36767 | 18.60437 | 0.13661 | 0.07067 | 0.00039 | + BMQ3-K + GRU motion head |
| BMQ3KGN (`67aac2a`) | 14.39310 | 21.48913 | 0.13661 | 0.07433 | 0.00084 | + joint model objects (non-visual) |
| BMQ3KGO (`67aac2a`) | 14.37751 | 18.57011 | 0.13661 | 0.07187 | 0.00042 | + joint model objects (visual) |
| BMQ3KG2P (`0e391f3`) | 14.35875 | 18.60437 | 0.13641 | 0.07067 | 0.00015 | GRU v2 (all data) + penetration weight 2e6 |
| BMQ3KGD (`eafb35f`, 10-25) | 14.56847 | 18.60437 | 0.13630 | 0.07067 | 0.00014 | + learned depth correction ×0.75 |
| **BMQ3KGM** (`f60f8a6`) | **14.35418** | **18.60437** | **0.13629** | **0.07067** | **0.00013** | depth correction off, penetration weight 4e6 |
| BMQ3KGMT (`a0d93f8`, 10-26) | 14.41266 | 18.58305 | 0.13629 | 0.07067 | 0.00013 | + common-translation selector |
| BMQ3KGMa0 (`0bfe450`, 10-28) | 14.70203 | 18.01952 | 0.13622 | 0.07066 | 0.00012 | depth trend off |
| BMQ3KGMR (`321e50d`) | 14.36408 | 19.07004 | 0.13628 | 0.07056 | 0.00016 | + reliability module v1 |
| BMQ3KGMk (`321e50d`) | 14.35430 | 18.68588 | 0.13629 | 0.07066 | 0.00013 | + α-0 object calibration |
| BMQ3KGMkR (`321e50d`) | 14.37471 | 19.17272 | 0.13629 | 0.07055 | 0.00014 | both |
| BMQ3KGMR2 (`8eaf249`, 10-29) | 14.36192 | 18.74770 | 0.13627 | 0.07056 | 0.00014 | + reliability module v2 |

On the leaderboard of 2026-10-29 07:14 UTC, BMQ3KGM ranks 14 / 5 / 7 / 8 / 18 (rank sum 52).

## Experiments by direction

Each row gives the key local result and the verdict. The detailed per-group numbers are in the recipe notes in
`pipeline/recipes/*.md`.

### Human

| Experiment | Result | Verdict |
|---|---|---|
| Body-mode SAM 3D Body (BMQ) | CD-H −1.60 locally vs VPQ2; Kaggle 14.417 | adopted |
| BMQ-K: objects re-expressed in the body-mode depth scale | removed BMQ's CD-O +0.26 coupling | adopted |
| GRU motion head (v1, then v2 on 412 episodes) | Kaggle ACC-H 0.13905 → 0.13661 → 0.13641 | adopted |
| Learned depth correction (depth GRU, ×0.75) | root C CD-H −0.52; Kaggle CD-H +0.21 | rejected (did not transfer) |
| Correction selector v3 (common translation) | root C CD-H −0.17; Kaggle CD-H +0.058 | rejected |
| Depth trend off (α 0) | four groups: CD-H −0.73, CD-O −0.96; Kaggle CD-H +0.35, CD-O −0.58 | rejected (human did not transfer) |
| Background-locked MoGe cues for the depth GRU | root C CD-H −0.25, one episode +6.9 | no gain |
| Per-frame overlay scale (projected bones vs ViTPose) | helps badly tracked humans, harms accurate ones | not adopted |
| Exact-zero PEN by moving hand / arm only (two designs) | arms jerked (self-acceleration ×1.5-3.9) | rejected |
| **Measured camera (2026-10-29)** | **four groups CD-H −1.04, other metrics unchanged** | **candidate BMQ3KGMtp** |

**Where the human error comes from** (root C):
- Depth drift along the camera ray: 37 % of CD-H. Lateral drift: 17 %. Rotation and scale drift: 8 %. Shape and
  articulation: 37 %.
- With the ground-truth distance per frame, CD-H drops by 4.71 (better on 30/30). With the ground-truth position,
  it drops by 7.26.
- The human is accurate at the first scored frame: 1.7-2.1 cm joint error, against 6.3-7.3 cm over the rest.

### Depth sources

| Experiment | Result | Verdict |
|---|---|---|
| Video Depth Anything as the only depth source | objects worse by about 25 CD-O (FoundationPose tracks worse) | rejected |
| VDA for the human depth trend only | better on T3, worse on C / U1 | rejected |
| Depth Anything 3, Depth Pro (evidence test) | carry information on badly tracked humans; on accurate tracks every source is at chance | diagnostic |
| MoGe depth at the human as evidence | right direction in only 35-49 % of cases | not used directly |
| Sensor depth fed to FoundationPose (upper bound) | raw tracks CD-O −13.5; after kappa only −5.3 | depth is not the main object error |

### Objects

| Experiment | Result | Verdict |
|---|---|---|
| Gravity kappa (objects scaled so the floor passes under the feet) | Kaggle CD-O 24.26 → 18.77 | adopted |
| Object smoothing (physics λ 10) | Kaggle ACC-O 0.0746 → 0.0719 | adopted |
| Joint human-object model, visual objects (v6) | root C CD-O −10.3; Kaggle −0.03 | did not transfer |
| Joint model, non-visual objects | Kaggle CD-O +2.88 | rejected |
| CoTracker3 point tracks + rigid solver | image alignment better, CD-O unchanged | rejected |
| Object refiner v7 (point tracks + BiGRU) | root C CD-O −2.37 | not uploaded |
| RGB-only tracking (zero-depth FoundationPose) | along-ray drift 11.0 → 20.1 % | rejected |
| Reliability module v1 (MoGe-only GRU, per-frame object depth) | four groups CD-O −1.5 to −7.2; Kaggle +0.47, ACC-O −0.0001 | CD-O did not transfer |
| Reliability module v2 (visibility-gated overlay cues) | four groups CD-O −2.63; Kaggle +0.14, ACC-O −0.0001 | CD-O did not transfer |
| α-0 object calibration alone | Kaggle ±0 | no effect |

**Where the object error comes from** (oracle decomposition on T3 / C / U1):
- **Position is almost all of it.** Replacing our position with the ground truth removes 89 / 80 / 87 % of the
  object error, and the distance along the camera ray alone removes 45 / 60 / 51 %.
- **Size, shape and orientation contribute ≈0.**
- **The error is already there at the first scored frame**, where the object sits about 10 cm off.

### Contact and penetration

| Experiment | Result | Verdict |
|---|---|---|
| Contact solver penetration weight 2e6 → 4e6 | root C PEN 0.00019 → 0.00014; Kaggle PEN 0.00039 → 0.00013 | adopted |
| PEN feasibility layer (hard constraints) | penalty swamped the movement guards | rejected |

## Configuration audit of 2026-10-29

**Question.** The owner asked whether the CARI4D container and the pipeline are configured properly for this use case.
The container source was read in full and every wiring point checked. Tools in the container that the pipeline does
not use were tested where they apply.

### Settings and wiring checked as correct

- **FoundationPose settings.** They equal the wrapper's defaults: register 10 and track 5 iterations, ESKF defaults,
  no particle filter, no depth masking. Re-registration below mask IoU 0.25 is a deliberate addition.
- **Depth images.** Every script decodes depth as v2d encodes it: metres = 65535 / v − 1. Quantisation is below
  0.3 mm at 3 m.
- **Frame pairing.** FoundationPose reads RGB frames by seeking, while depth and masks come from sequential reads.
  The two match exactly: 0 mismatches in 853 frames on four challenge videos (`scripts/video_seek_check.py`).
- **Mesh frame.** FoundationPose returns poses for the original mesh frame, and the build uses the same GLB.
- **Camera input to SAM 3D Body.** It receives the episode camera.
- **Mesh grounding.** It pairs each frame's depth with that frame's camera.
- **Frame indexing.** The kit indexes our arrays by video frame.

### Wiring gaps found

Each gap was replayed on the production output of the four groups. In these replays only the object moves, after the
contact stage, so the rise in PEN is a replay artefact.

| Gap | Test | Result | Verdict |
|---|---|---|---|
| Objects tracked with the reference frame's MoGe camera, the human with the median camera | object x, y × f_obj / f_cam (`scripts/object_camera_replay.py`) | CD-O −0.26 / −0.18 / +0.01 / −0.11 | small; covered by the measured camera |
| The container's per-frame depth alignment to the reference frame was not used | replay through the per-frame background scale (`scripts/object_depth_lock_replay.py`) | CD-O T3 −0.20 / −0.67, U1 +0.26 / −0.28; ACC-O +0.0001 to +0.0002 | dropped |
| Human depth scale from background-normalised MoGe, objects from raw per-frame MoGe | consistent per-frame correction | CD-O +0.18 / −0.14 / −0.03 / −0.30 (σ 10) | dropped |
| MoGe never given a known camera (`video_to_depth --input_intrinsics_path`) | MoGe with our median camera + FoundationPose re-tracked, T3, objects only | CD-O +0.39 raw, +0.16 after kappa | dropped |

Tools in the container that do not apply:

- `correct_depth_scale`, `estimate_mesh_scale` and `align_depth_to_object` need a mesh of known metric size. Ours is
  built from MoGe.
- CARI4D's CoCoNet in-the-wild inference needs SAM 3D Body decoder buffers.

### The camera

FORM-HOI's calibration file (`edex`) holds the measured intrinsics of the four rig cameras
(`config/camera/rig_cameras.json`). `scripts/camera_truth_check.py` compares them with our camera over 1488 FORM-HOI
episodes:

| | Ours (median of MoGe-2's guesses) | Measured |
|---|---|---|
| Focal length | 3.4 % short (5-95 %: 1.6-5.5 %) | 840.6-856.6 px |
| Principal point | (768, 576) = image centre | (713.5-727.8, 527.5-544.2) |
| Change over recording days | — | none: one calibration per camera over 34-46 days |

**The challenge uses the same rig.** The videos have the same resolution, and our focal guesses there are lower still
(median 807).

**Matching challenge episodes to rig cameras.** Each challenge episode's background was matched against about 1500
FORM-HOI backgrounds of known camera (`scripts/camera_identify.py`).
- 18 of 30 episodes match one rig camera clearly: correlation 0.90-0.99, against 0.34-0.41 for any other camera.
- The other 12 are scenes not in FORM-HOI. They take the rig mean, which is within about 9 px of every camera's
  principal point.

**The test.** Under the pipeline's own model, a change of camera keeps every pixel and changes only its 3D
interpretation. For one episode this is a single linear map (`scripts/true_cam_params.py` on the MHR parameters,
`scripts/true_cam_replay.py` on scorer exports). Variants, all on the four groups, change against production:

| Variant | CD-H (T3 / U1 / C / U2) | Mean CD-H | CD-O mean | ACC-H mean | PEN |
|---|---|---|---|---|---|
| Replay after the full pipeline: principal point only | −1.57 / −0.91 / −0.98 / −0.97 | −1.11 | +0.06 | ±0.00003 | replay artefact |
| Replay after the full pipeline: focal + principal point | −1.31 / −0.83 / −0.94 / −0.67 | −0.94 | +0.56 | ≤ +0.00004 | replay artefact |
| Replay after the full pipeline: focal only | +0.06 / −0.03 / −0.05 / −0.04 | ≈0 | +0.34 | ≤ +0.00005 | replay artefact |
| Before the GRU motion head, then GRU + contact stage | −1.46 / −0.77 / −0.91 / −1.00 | −1.04 | +0.08 | +0.00022 | ≈0 |
| **After the GRU motion head, then contact stage** | **−1.46 / −0.77 / −0.91 / −1.00** | **−1.04** | **+0.08** | **+0.00003** | **≈0** |

**Reading the variants.**
- The principal point is the lever. The focal error is mostly absorbed by the depth trend stage.
- Before the GRU costs ACC-H, because the motion head was trained in MoGe-2's camera. After the GRU does not.
- Detail of the adopted variant: ACC-O +0.00004 and PEN −0.00003 / +0.00001 / −0.00001 / −0.00004. CD-H is better
  in 24 / 18 / 23 / 22 of 30 episodes.

**The challenge candidate.**
- **Built by** `pipeline/recipes/truecam_final.sh episodes_cand_BMQ3KGM_in episodes_cand_BMQ3KGMtp`.
- **Local gates against BMQ3KGM.**
  - Self-penetration 0.000086 (BMQ3KGM 0.000100).
  - Articulation and whole-body jitter ×1.003 / ×1.001.
  - The 2D reprojection against MoGe-2's camera reads 65 px. That is the size of the correction itself; under the
    measured camera the pixels are unchanged by construction.
- **Reproduction.** The repository recipe reproduces the camera-mapped input exactly (every array identical). After
  the contact stage only the pose differs, from the contact solver's known run-to-run variation.
- **Leaderboard estimate** (board of 2026-10-29 07:14 UTC).
  - If half the FORM-HOI CD-H gain transfers: CD-H rank 14 → 10, rank sum 52 → 49.
  - If all of it transfers: CD-H rank 6, rank sum 45.

## Open items

1. **Score BMQ3KGMtp on Kaggle** (owner's decision).
2. **Retrain the GRU motion head in the measured camera.** Moving its input to the measured camera cost
   ACC-H +0.0002.
3. **Use the measured camera at the source.** SAM 3D Body and FoundationPose would run with the measured camera,
   and MoGe-2 would get its focal length as a prior. MoGe-2 cannot take an off-centre principal point, so its depth
   would be re-read through the measured camera.

## Reproduce

| What | How |
|---|---|
| Best scored file BMQ3KGM | `docs/CURRENT_BEST.md` |
| BMQ3KGMtp | `pipeline/recipes/truecam_final.sh` (`truecam.md`) after the BMQ3KGM recipes |
| Camera truth and identification | `scripts/camera_truth_check.py FORMHOI_DIR CHALLENGE_ROOT`, `scripts/camera_identify.py FORMHOI_DIR CHALLENGE_ROOT OUT 100000` |
| FORM-HOI scoring | `scripts/fh_export.py ROOT EPISODES_DIR EXPORT_DIR` (mhr_converter image), then `scripts/fh_score.py ROOT NAME=EXPORT_DIR:EPISODES_DIR ...` |
| Camera replays on scorer exports | `scripts/true_cam_replay.py ROOT EXP EP MODE` (tc / tcf / tcp) |
| Camera on episode files (any stage) | `scripts/true_cam_params.py --input DIR --out DIR --old_root ROOT --k_new JSON` |
| Wiring replays | `scripts/object_camera_replay.py`, `scripts/object_depth_lock_replay.py`, `scripts/video_seek_check.py`, `scripts/compare_episode_dirs.py` |
| Reliability module, depth trend, α-0 calibration | `pipeline/recipes/rel.md`, `trend.md`, `kcal_final.sh` |

FORM-HOI groups are built with the same stage scripts on FORM-HOI recordings laid out as Track 1 episodes, with the
ground truth exported for `fh_score.py`. Running every stage as a cluster job, and the variant roots that link a
finished group and replace one input (camera files, depth), are local job wrappers. They call the stage scripts and
recipes listed here.
