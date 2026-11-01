# Track 1 cluster experiments — 2026-11-01

## Objective and constraints

Target CD-H and CD-O below 10 cm while retaining acceleration and penetration quality.
This is a research target, not an established result. Only actual Kaggle measurements
establish challenge improvement. Work lives in `/work/mnidgu2s/project1`; computational
jobs are pinned to `wr20`, one additional node, leaving existing jobs untouched.
The isolated `vpq-vitpose` clone starts at `00c6e73`; the handoff's uncommitted patch
and new files were restored separately. No commits or pushes without the handoff's
score evidence and explicit user go-ahead. Preserve all baseline artifacts.

## Evidence and alternatives

1. Reproduce the recorded pipeline before testing new geometry. The prior Colab
   deployment used the plain GRU rather than the depth-finetuned motion head.
   `eps_scale=0` disables only its depth output; depth cues remain model inputs.
   Drive does not contain all BMQ3KGM ancestry, so a checkpoint substitution on a
   different human input must be named an ablation, not a reproduction of BMQ3KGM.
2. Estimate a constant object translation using independent pre-contact hands.
   A constant camera-frame translation preserves second differences exactly, unlike
   carrying the object with noisy hands. Use contact geometry, robust residuals,
   temporal agreement and independent silhouette validation; abstain on ambiguous
   constraints. Never use hands already fitted to that object's wrong position as
   independent evidence. Existing `held_object_depth.py` fits time-varying shifts
   and only considers patches already within 3 cm; it does not resolve the proposed
   first-frame-bias experiment.
3. Human floor anchoring is a separate experimental ablation. Bone lengths are
   already enforced by shared MHR identity; another bone-length calibration alone
   cannot resolve root-depth drift. Any floor correction needs contact evidence,
   a valid plane, preservation of the alignment frame, and a jitter guard.

Selected order: restore/audit, baseline/checkpoint ablation, constant object offset,
then floor ablation only with defensible input evidence. Do not retrain a new
FORM-HOI correction, repeat the disproven MoGe first-frame fit, or feed zero depth
to FoundationPose. RGBTrack additionally assumes true-scale CAD geometry, which
these generated meshes do not supply.

## Reliability and validation

- Add regression tests for empty/insufficient static depth evidence and failed floor
  fits. Missing evidence must not produce nonfinite scale or invented floor normals.
- Test geometric recovery against analytic synthetic scenes with known offsets,
  rotations, outliers, degenerate planes, and no-contact sequences. Synthetic success
  proves implementation behavior, not a competition improvement.
- Preserve episode parameters and meshes except the explicitly changed coordinates.
  Validate frame counts, finite arrays, units, anchor preservation and object
  acceleration invariance. Record exact input and model hashes with results.
- Use the official packer and scorer contracts, with per-episode diagnostics. Do not
  manufacture held-out ground truth or label self-consistency metrics as CD-H/CD-O.
- New-code submissions require truthful code provenance; record any conflict with
  the handoff's no-commit-before-score rule rather than attaching an unrelated commit.

## Research references

- Official challenge: https://nvidia-isaac.github.io/video_to_data/v2d_challenge/
- CHOIR: https://arxiv.org/abs/2605.20992 (contact can couple placement; its learned
  rectifier is not evidence that an ungated deterministic correction will transfer).
- RHINO: https://github.com/lxxue/RHINO (temporally gated contact spans).
- RGBTrack: https://arxiv.org/abs/2506.17119 (true-scale CAD assumption).

Deadline verified on official website: 2026-11-04 17:00 EST, 23:00 Europe/Berlin.
