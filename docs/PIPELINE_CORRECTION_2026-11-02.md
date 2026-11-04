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

## Independent scene-geometry diagnostic

A throwaway SIFT/essential-matrix probe used only released RGB backgrounds and
dynamic-object/person exclusion masks. Three pairs passed the initial triangulation
gate, but estimated floor planes had weak support/spread and shallow parallax.
They were not applied. A feature-matching result alone is not a trustworthy floor
constraint. Results are in `work/spikes/background_geometry/report.json`.

## Validation and runtime evidence

- Full local suite after both corrections: 114 passed.
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
