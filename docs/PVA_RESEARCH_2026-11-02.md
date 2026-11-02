# Image-conditioned dynamics experiment, November 2

**Uploads paused by the owner after these two tests completed.** Do not submit any
further candidate. Preserve the remaining shared quota for teammates; new research
and validation can continue locally.

## Scored outcome: rejected

Code was published to `vpq-vitpose` at
`3a576823d4803dcab820a306966f7c79f4d3f90d`. Both authorized candidates were packed
with that commit and accepted/scored on all five boards.

| Candidate | CD-H | CD-O | ACC-H | ACC-O | PEN |
|---|---:|---:|---:|---:|---:|
| PVA_POSE | 13.54747 | 18.04525 | 0.16712 | 0.07068 | 0.00256 |
| PVA_ACC | 13.55762 | 18.04525 | 0.15690 | 0.07068 | 0.00257 |

Neither beats any current best. ACC-H is worse than P2's 0.13616 by 0.03096
and 0.02074 respectively. Root-only also fails, so articulation fitting alone
does not explain the regression. The released visual dynamics signal did not
transfer through these adapters; improved fitting/2D agreement was misleading.
Stop this experiment and preserve all previous bests. No additional PVA GPU work.

Receipts: `../repo/submissions/track1_cand_PVA_POSE.parquet.ddb393ea091e.kaggle-receipt.json`
and `track1_cand_PVA_ACC.parquet.45697eabb442.kaggle-receipt.json`. Submission refs
are 57916970/973/975/978/979 and 57917037/039/046/048/052 (board order above).
Logs are `../logs/kaggle_pva_pose.log` and `kaggle_pva_acc.log`.

A read-only audit of 14 already refined CARI4D object tracks also completed on
wr20 (job444582). Refinement reduces their jitter but leaves much larger second
differences than the retained tracks; no new CARI4D reconstruction was launched.
Details: `../repo/work/analysis/refined_object_dynamics.json`.

## Decision and evidence

The next hypothesis is to correct motion derivatives using video evidence while
retaining the current reconstruction's positions and identity. This differs from
the previously unsuccessful generic smoothing and FORM-HOI-trained correction
heads. It is an experiment; no PVA Kaggle improvement has been established.

The current measured best scores remain CD-H 13.40710, CD-O 17.51622, ACC-H
0.13616, ACC-O 0.07051, PEN 0, obtained from different files. At 13:48 Berlin the
team's rank sum was 44; RobotPI's was 9. Reaching first requires several substantial
improvements, not merely a small fitting-loss decrease.

## Cached CARI4D result: reject the raw replacement

| Candidate | CD-H | CD-O | ACC-H | ACC-O | PEN |
|---|---:|---:|---:|---:|---:|
| N19H | 13.61196 | 17.58893 | 0.13669 | 0.07061 | 0.00282 |
| N19HO | 13.61196 | 19.32814 | 0.13669 | 1.37088 | 0.00295 |

Humans are byte-identical between this pair, isolating the object replacement.
Neither candidate improves a current best. The raw CoCoNet objects are unsuitable:
across the 19 replaced episodes their median per-episode mean translation second
difference is 2.051 cm/frame² versus 0.0283 for the old objects. Episode 12 reaches
5.394 mean and 23.856 p99 cm/frame². These are trajectory diagnostics, not errors
against ground truth. Smoothing could reduce jitter, but would not establish that
the worse object placement is corrected. Do not spend another full inference run
or upload on the unchanged hypothesis.

Receipts under `../repo/submissions/`:

- `track1_cand_N19H.parquet.118fdcdc825a.kaggle-receipt.json`
- `track1_cand_N19HO.parquet.390869daf67b.kaggle-receipt.json`

Per-episode diagnostics: `../repo/work/analysis/n19_object_diagnostics.json`.

## Research and dependencies

HTD-Refine's PVA-Net predicts COCO17 2D keypoints and metric 3D velocity/acceleration
from 30 FPS RGB. Its released optimizer uses SMPL/SMPL-X; this work supplies a new
MHR adapter. Published results do not automatically apply to that adapter.

- [Official source](https://github.com/ant-research/HTD-Refine), commit
  `2fcd6ddef3c4eb75a636062245f80a0136c09b7e`.
- [Paper](https://arxiv.org/abs/2605.26879),
  [project](https://zju3dv.github.io/htd-refine/).
- Upstream checkout: `../research/HTD-Refine`; its AGPL-3.0-only license is retained.
- Released `pvanet.pt`: 3,569,259,862 bytes; SHA256
  `a6e5f95a287a45dc1e4944fe7381e7d0c5fe02d4009c5794b09717077a4e1a26`.
- Predictor runtime: existing `v2d_cari4d.sif`, Torch 2.5.1/cu124, with project-local
  **timm 1.0.12** overlay at `../research/pva_site`. The container's timm 1.0.30
  cannot deserialize the model correctly (`Attention.gate` mismatch). Shared
  containers were not modified. Body fitting uses `mhr_converter.sif`.

No YOLO detector or SMPL model download is needed: existing Track 1 person masks
provide actor boxes and the official MHR model provides differentiable joints.

## Implementation

1. `pva_inputs.py` reads consecutive RGB frames and person masks, rejects long
   missing-mask gaps, expands boxes to 3:4 with padding 1.15, and uses the predictor's
   192×256 affine crop. Measured intrinsics and true video frame IDs travel with the
   input. Inputs are already cropped, so the upstream detector/crop path is bypassed.
2. `run_pva_track1.py` uses the released model with 120-frame windows and 30-frame
   overlap. It retains upstream **first-available** overlap selection. It checks
   finite outputs and T/T−1/T−2 time support, and records source/input provenance.
3. `pva_mhr_joints.py` exports float64 MHR joints. A real model test adds known
   translations and checks units and camera axes; all 30 tests agree to better than
   1e−8 metres. Cached joints are tied to exact pose/scales/shape SHA256 fingerprints.
4. `pva_root_dynamics.py` estimates a common torso translation from image-derived
   acceleration with a position prior. Velocity weight is zero in PVA_ACC;
   acceleration weight is 100. At least two visible torso joints are required.
   Derivative weights vanish across unscored gaps. The scored anchor is eliminated
   from the solve, and corrections above 5 cm are rejected rather than clipped.
5. `pva_pose_refine.py` optimizes MHR columns that move the twelve mapped body joints.
   Identity and objects are frozen. It fits derivatives in physical units, image
   motion after a constant anatomical-offset adjustment, and a position prior.
   Corrections are bounded and the anchor is fixed. After interpolating parameter
   deltas, **all output frames** are evaluated through MHR; reject the episode if
   any joint moves more than 5 cm or the objective/image gate fails. Only active
   columns are updated, preserving untouched float64 values as well as float32.
6. `validate_pva_candidate.py` independently checks all 30 episodes, preserved
   arrays/anchors, rotations, cache identities, actual full-trajectory displacement,
   and agreement with existing ViTPose observations. It selects the actor using
   the baseline and evaluates both candidates against identical observations.

COCO shoulders/elbows/wrists/hips/knees/ankles (indices 5–16) map to MHR
`[75,39,76,40,78,42,2,18,3,19,4,20]`. Prediction units are m/s and m/s², divided by
30 and 900 to compare finite differences. MHR root parameters are converted with
`10 * [1,-1,-1]` from Track 1 metres. This is verified against the model, not assumed.

## Actual runs and validation

All jobs requested one node, **wr20**, in partition `gpu4`.

| Job | Work | Result / elapsed |
|---|---|---|
| 443688 | Initial predictor pilot | Failed: timm mismatch; preserved diagnostics |
| 443689 | Corrected 150-frame pilot | Complete |
| 443690 | Full clips 9, 13, 24 | Complete |
| 443691 | All 30 predictions, joint caches, root candidate | Complete, 10m54s |
| 443702 | Initial body pilot, episode 9 | Complete, 19s |
| 443714 | Corrected body fit and float64 re-export, all 30 | Complete, 8m47s |

| Check | PVA_ACC | PVA_POSE |
|---|---:|---:|
| Complete episodes | 30 | 30 |
| Anchor/identity/object preservation | Pass | Pass |
| Maximum actual joint movement | 1.1757 cm | 4.2554 cm |
| Independent 2D mean before | 7.901975 px | 7.901975 px |
| Independent 2D mean after | 7.897285 px | 7.879898 px |
| Episodes with improved independent 2D mean | 17/30 | 22/30 |

All 30 body fits passed the acceptance gate; median final/initial objective ratio
is 0.2405. This is agreement with fitted targets, not held-out accuracy. Independent
2D changes are small and cannot establish depth or acceleration accuracy. PVA's
median acceleration magnitude is 0.642 m/s² versus baseline 0.263, so this method
is not simply suppressing all acceleration. Window-seam acceleration jumps have
median ratio 1.63 to ordinary transitions; retained as a limitation for evaluation.

Artifacts under `../repo/`:

- `submissions/episodes_cand_PVA_ACC/`, `submissions/episodes_cand_PVA_POSE/`
- `work/pva/inputs_all/`, `predictions_all/`, `joints_P2_all/`, `joints_PVA_POSE/`
- `work/pva/root_all_validation.json`, `pose_all_validation.json`
- `work/pva/dynamics_diagnostics.json`, `official_payload_validation.json`
- Logs under `../logs/pva_all_443691.log`, `pva_pose_all_443714.log`.

The unchanged official packing routine also builds each numeric payload locally
for exact row-ID and finite-value validation. No submission file or false commit
URL is produced by this check. A real Parquet must cite the newly published code.
Both payloads passed: **740,780 rows each**, exact sample IDs, all finite. Numeric
payload hashes are recorded in `official_payload_validation.json`.

Final verification: **138 tests passed**, shell syntax checks passed, and
`git diff --check` passed. Independent review found two export issues (gap-frame
displacement and float64 preservation); both were fixed with regressions and
re-reviewed with no remaining material correctness finding. Run the suite with
`TMPDIR=/work/mnidgu2s/project1/.tmp .venvs/submission/bin/python -m pytest -q
--basetemp=/work/mnidgu2s/project1/.tmp/pytest_pva_release_20261102` from this checkout.

## Reproduction and next decision

The batch scripts pin the single allowed node and use project-local cache/tmp
paths. From this development checkout, `sbatch pipeline/cluster/run_pva_all.sbatch`
runs input preparation, prediction, joint export and root fitting; then
`sbatch pipeline/cluster/run_pva_pose_all.sbatch` fits the body candidate. They
refuse existing output directories/files; preserve existing results and choose
new paths for a deliberate rerun. Baseline input is the preserved P2 candidate.
These are reproducible correction stages; they do not repair the separately
documented missing historical BMQ-from-video ancestry.

Next: publish the tested correction code with explicit authorization, pack with
its real commit, refresh shared Kaggle quota, and score one controlled PVA file
at a time. Prioritize PVA_POSE, then PVA_ACC as an ablation if quota permits. Stop
the hypothesis if both regress. Preserve the per-metric best files regardless.
Do not tune episode-specific weights from leaderboard feedback.

The handoff's §1 requires explicit approval and proven Kaggle improvement before
commits/pushes. On November 2 the user explicitly approved the exception for this
experiment: **"Approve push and both PVA tests"**. Publish these tested changes to
`vpq-vitpose`, then pack with that real commit and score PVA_POSE and PVA_ACC on all
five boards. N19 upload authorization has already been fulfilled; do not repeat it.
