# Correct current pipeline flaws and test a stronger approach

The user's priority is a better measured reconstruction, not rebuilding the historical
BMQ ancestry. Keep existing scored artifacts as controls. Reproduction packaging is
downstream of selecting the improved method.

## First repair: FoundationPose completion on occluded objects

The silhouette metric intentionally reports infinity when either visible centroid is
undefined. The exporter copies that into its diagnostics and the diagnostics validator
then refuses it at the end of an otherwise completed tracking pass. Episodes 12, 18
and 20 have failed here. Repair only the representation of an undefined diagnostic
(NaN, already supported by the schema). Preserve geometry validation, rejection masks,
reliability decisions and all finite measurements. Test the actual upstream function
after patching, then rerun the failed stages on wr20 using existing depth/human caches.

## Measured temporal-context hypothesis

CoCoNet currently uses 96-frame windows, stride 96 and first-occurrence output
selection. Even reducing stride does not blend overlaps: later estimates are discarded.
Measure boundary acceleration separately from interior acceleration on completed
episodes. If the boundary defect is confirmed, run an overlapping-window pilot and
combine redundant predictions in a representation-safe way. Compare seam error,
reprojection and actual Kaggle metrics against the existing C1/C2 controls. Do not
assume lower self-acceleration is better accuracy; the previous smoothing tests
explicitly disproved that assumption.

Measured on 14 available episodes: median per-episode boundary/interior acceleration
ratio is 1.047 for the predicted human and 1.046 for its initialization; this does
not support window boundaries as the main human-error source. The corresponding
object ratio is 1.214 (initialization 0.970), so object seams remain a narrower
possible issue. No overlap rewrite was promoted on this evidence.

## Confirmed temporal-axis flaw and controlled correction

`human_root_mix.py` smoothed the sequence of scored samples as though they were
consecutive video frames. Ten episodes contain internal gaps, up to 360 frames.
The corrected smoother uses the real video-time grid, zero observation weights in
gaps, and samples back to scored frames afterward. It preserves the first scored
pose. A constant-velocity synthetic gap case had 5.84276 cm maximum error under
the old implementation and numerical-zero error after correction. Independent
review caught a zero-penalty singular case; the fix now bypasses smoothing at
`lam=0`, with a regression test.

Candidate TIMEAXIS changes only this behavior against scored V5. Across all 30
episodes, identity, articulation, object arrays and anchor poses match V5 exactly.
The largest actual root difference is 2.04670 cm. This is an implementation
correction, not evidence of an accuracy gain until scored on Kaggle.

### TIMEAXIS measured result

| Candidate | CD-H | CD-O | ACC-H | ACC-O | PEN |
|---|---|---|---|---|---|
| V5 control | 13.53699 | 17.63890 | 0.13625 | 0.07062 | 0.00201 |
| TIMEAXIS | **13.52781** | 17.63890 | **0.13624** | 0.07062 | 0.00201 |

TIMEAXIS improves CD-H by 0.00918 cm: a small real improvement, not a replacement
for P3's best 13.40710. Submitted reconstruction commit: `5e0bc96`. Kaggle IDs:
CD-H 57889510, CD-O 57889513, ACC-H 57889516, ACC-O 57889519, PEN 57889523.
The user explicitly approved the push and test after automatic review initially
blocked the push under the older handoff condition.

## Shared-team uploader flaw

The official client used the most recent submitter's username plus team name as
an identity. When a teammate uploaded most recently on CD-H only, it refused the
same `Izus_systems` team. The wrapper now changes only the identity-set calculation:
prefer the actual team name, falling back to the user for solo entries. All other
official preflight, quota, validation, upload and receipt logic remains unchanged.
Unit/integration tests and independent review pass; live dry-run confirmed entry
and remaining quota on all five boards before the successful TIMEAXIS upload.
The stored official kit source is unchanged.

## Modeling ablation: inherited floor prior

CARI4D's learned spatial correction is currently followed by floor slide beta 0.55,
chosen on the old SAM pipeline. C5 tests beta 0 on exactly the same six episodes
(9, 13, 16, 24, 25, 28), model stage, smoothing and fallback as C1. C1 scored
13.49139 CD-H / 17.71056 CD-O. C5 scored 14.11530 CD-H / 17.71056 CD-O,
with ACC-H 0.13654 and ACC-O 0.07062. Removing the floor correction worsens
CD-H by 0.62391 cm, so beta 0 is rejected. This checks a model-specific assumption
instead of treating old tuning as universal. See `docs/TRACK1_COMPUTE_TRIAGE_2026-11-02.md`
for the subsequent cancellation of the full-stage GPU batch and preserved caches.

## Independent scene-geometry diagnostic

A throwaway SIFT/essential-matrix probe used only released RGB backgrounds and
dynamic-object/person exclusion masks. Three pairs passed the initial triangulation
gate, but estimated floor planes had weak support/spread and shallow parallax.
They were not applied. A feature-matching result alone is not a trustworthy floor
constraint. Results are in `work/spikes/background_geometry/report.json`.

## Validation and runtime evidence

- Full local suite after the diagnostic, time-axis and uploader corrections: 118 passed.
- Actual installed exporter source SHA256:
  `092c66dc41618ed3e82eeba641a6a4f067093e3dfb86e4956290f3dc48e2df67`.
- Patched exporter SHA256:
  `6b44008ac594808321f9ff002c69034f4459a63aa30a05a61399f65d3e85a0e4`.
- Episode 12 now passes tracking and reaches CoCoNet/post-optimization: all 405
  frames exported, 21 undefined centroid diagnostics as NaN, no infinite diagnostics.
  The complete repaired episode has not yet been declared finished.
- Independent code review completed; its zero-smoothing finding was fixed and tested.

## Execution

- Write regression tests before the runtime fix; run the full local suite afterward.
- Track runtime patches and source hashes in this repository. Bind only the intended
  file into newly started containers; preserve existing outputs and active processes.
- All GPU steps remain inside wr20, at most the existing one additional node.
- Promote a change only after comparison with a concrete control. Label unscored
  proposals clearly; do not promise a sub-10-cm result without measurement.
