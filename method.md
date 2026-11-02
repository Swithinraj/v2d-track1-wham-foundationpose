# Method: camera identification for novel-arrangement scenes via ceiling-only matching

New approach developed 2026-11-01 on the dev box (full analysis: `/data/formhoi/NOTES.md`,
scripts in `/data/formhoi/`, validation output `ceil_strength_apply.txt`).

## Problem

The pipeline needs each episode's rig camera (4 fixed cameras, different principal points,
up to ~9 px apart; MoGe-2's own estimate is 54/38 px off, so the assignment is worth real
CD-H). The shipped matcher (`scripts/camera_identify.py`) correlates an episode's background —
grey, 96x72, temporal median over 40 frames, zero-mean/unit-norm, cosine similarity — against
~1,500 FORM-HOI backgrounds and takes the top-5 count vote. It reliably identifies episodes
whose exact scene exists in FORM-HOI (corr >= 0.8): the challenge's 18 strong episodes. The
12 weak episodes (corr 0.36-0.53) are the same office with *changed setups* (furniture
rearranged, curtains added), fall back to the rig mean, and the question was whether their
mid-tier votes can be trusted.

## Why the baseline fails (measured, not assumed)

A leave-one-out calibration on 1,200 labelled FORM-HOI episodes (300 sessions x 4 cameras,
all 50 recording dates) with a novel-scene protocol — the query's own scene excluded from the
pool (correlation < 0.8), forcing the matcher into the weak-correlation regime — shows:

- Same-scene protocols (own session / date / week excluded): **100 % accuracy**, every
  feature variant, margins >= 0.40. The strong regime needs no work.
- Novel-scene regime, grey feature, corr 0.30-0.50: **0 % accuracy on 623 queries** — and not
  random: a stable anti-mapping (back votes right 300/300, stable across both halves of the
  recording period; left <-> front). The vote is an *anti-signal*.
- The mechanism: the full-frame background is dominated by furniture. When the setup changes,
  the image evidence for the camera is destroyed, and what remains correlates best with the
  *opposite* corner of the same office.
- The regime is bimodal (queries sit at 0.30-0.50 or 0.60-0.80, almost nothing between), so
  the challenge's mid-tier straddles an untested boundary — no threshold on the grey corr can
  fix this.
- A vanishing-point self-calibration (orthogonality of scene VPs per candidate K) was tested
  and cannot discriminate: it picks one camera for 17/18 known episodes at margins ~100x below
  the estimation noise. Principal-point differences of 5-14 px are below VP noise in a
  cluttered office.

## The proposed approach

Two changes, both arrangement-independent by construction:

### 1. Ceiling-only feature (replace the full-frame background)

Keep the exact feature recipe (grey, temporal median of 40 uniformly spaced frames, 96x72,
zero-mean/unit-norm, cosine similarity) but compute it on a **spatial crop: the top 35 % of
the frame** (ceil35; ceil50 = top 50 % as a robustness variant). The rig is fixed, so each
camera's ceiling view — light grid, beams, walls' top edge — is constant across all room
rearrangements. Furniture never enters the feature, so a changed setup cannot destroy the
camera evidence; it can only lower the correlation uniformly, which the decision rule below
handles.

No masking or person removal is involved: people average out in the temporal median as before.

### 2. Strength-based voting (replace the top-5 count vote)

The ceiling feature is so discriminative that the count vote becomes the bottleneck: 2
same-camera matches at corr 0.996 are outvoted by 3 cross-camera matches at 0.096. Instead,
per candidate camera take the **single best correlation against that camera's episodes** and
predict the camera with the highest one (argmax over cameras). One perfect match must not be
outvoted by several weak ones.

### 3. Calibrated accept/reject rule (instead of a fixed threshold)

Establish the cross-camera null distribution on the same 1,200-query novel-scene protocol:
the best per-camera correlation for every *non*-true camera. With ceil35: median 0.059,
**p99 = 0.116**, max 0.124 (ceil50: p99 0.165). Decisions then use evidence relative to the
null, not a magic correlation value:

- corr > null p99 with a clear gap to the second camera → accept;
- corr > 3x null p99 → certain;
- below → no signal, keep the current fallback (rig mean).

## Validation

Novel-scene protocol, strength-based vote, full 1,200-episode pool:

| Feature | accuracy | cross-camera null (median / p99 / max) |
|---|---|---|
| ceil35 | **100.0 %** (300/300 per camera) | 0.059 / 0.116 / 0.124 |
| ceil50 | 100.0 % | p99 0.165 |

For comparison, the baseline grey feature under the same protocol: 27 % (anti-mapped); with
the count vote even the ceiling feature only reaches 40 % — the vote change is half the fix.

## Application to the challenge (30 episodes)

Best-camera ceiling correlation vs the null (ceil35, second-best in parentheses):

| Group | Verdict |
|---|---|
| strong 18 | corr 0.94-1.00, second camera ~0.10 — **confirmed**, matches shipped |
| ep 18, 28 | back 0.346 / 0.385 (2nd 0.10) — 3x null: strong evidence for back |
| ep 12, 14, 16, 22, 23, 24 | front 0.136-0.168 — above the null but not 3x: probable |
| ep 15, 25, 26, 27 | 0.019-0.040 — below the null: no signal |

Outcome: the shipped `config/camera/episode_cameras.json` is **consistent with the evidence
30/30** — no assignment changes. The one actionable improvement: `scripts/measured_camera.py`
currently sends eps 18/28 to the rig mean although their back-camera evidence is 3x the null
(and visually corroborated); a second accept tier (accept when ceil35 corr > 0.35) would
promote them. Expected CD-H ~0.005-0.015 (9 px principal-point shift on 2/30 episodes);
geometric, so it should transfer to Kaggle, but it is below single-probe noise — bundle it
with another probe rather than spending one on it.

## Rules of thumb this establishes

1. Never trust a full-frame grey vote in the mid-tier regime (0.40-0.60): it is 0 % accurate
   and anti-mapped. Trust the ceiling strength-vote instead.
2. Accept when the best-camera corr clears the measured cross-camera null p99 (0.116 for
   ceil35) with a gap to the second camera; treat > 0.35 as certain.
3. Any per-camera "best-of" decision rule should use strength (best match per class), not
   counts, whenever class distributions in the top-k are imbalanced — the count vote failed
   here precisely because of a 2-vs-3 imbalance at a 10x score gap.

## Status

- Validated end-to-end on the dev box; feature extraction + validation + application scripts
  in `/data/formhoi/` (`build_pool_ceil.py`, `calibrate_ceil.py`, `ceil_strength_apply.txt`).
- No change shipped to the repo config (the shipped file was confirmed, not corrected); the
  optional eps-18/28 promotion is a one-line change to `scripts/measured_camera.py`, pending
  the owner's decision to bundle it with the next Kaggle probe.
- Porting into the pipeline (only if wanted): add the ceil35 feature and strength vote to
  `scripts/camera_identify.py`, and the calibrated thresholds to `scripts/measured_camera.py`.
