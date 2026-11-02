# Track 1 session, November 2 evening (Claude Code, continuing Codex's work)

Uploads use board-specific quota (`scripts/submit_candidate.py --only METRIC ...`). Owner rules for Nov 3-4: push tested
code to `vpq-vitpose` without asking; use every slot except 2 per board per day. `work/autopilot.sh` was stopped (it would
have uploaded C3/C4, close to the rejected N19H/N19HO).

## Scored results

| File | Board(s) | Score | Previous team best | Verdict |
|---|---|---|---|---|
| P4 (`6e03eac`): P3 person + slide lam 3e4 + root mix 0.3 toward a0 | all | CD-H **13.32564**, CD-O 17.51622, ACC-H 0.13889, ACC-O 0.07064, PEN 0.00257 | CD-H 13.40710 | new CD-H best |
| WFk60 (`eb572ff`): WF, object-translation acceleration x0.6 | ACC-O | **0.06971** | 0.07051 | new ACC-O best |
| P2r60 (`eb572ff`): P2, root-translation acceleration x0.6 | ACC-H | 0.13643 | 0.13616 | rejected |
| P4tab101 (`bc7a45f`): P4, 13 table objects x1.01 about the camera | CD-O | **17.08369** | 17.51622 | new CD-O best |

P5 (mix 0.5) was not uploaded: V5/V6 put the best mix weight near 0.33.

## Why the acceleration shrink works for objects

`scripts/accel_shrink.py` writes x_s + k (x - x_s) per contiguous scored stretch (x_s: Whittaker lam 1e4), first scored
frame pinned. Model: predicted acceleration = signal + independent noise, ground truth = signal + its own noise. Fitted to
the heavy-smoothing result (k = 0, +0.0009 ACC-O), it predicted the earlier table-object x0.974 result (-0.00006) without
being fitted to it. Three points (k = 1, 0.6, 0) put the optimum of this family at k ~ 0.59, where WFk60 already is.
For the human root the same shrink hurt: root acceleration is mostly real motion.

## Checked and dropped (no upload)

- Light low-pass (5-8 Hz) removes only 3-6 % of acceleration; the apparent 47 % above 8 Hz was spectral leakage.
- Depth-only shrink: along-ray acceleration is smaller than lateral (0.61 objects, 0.80 human root).
- Freezing objects at rest: only 10.5 % of scored frames, already low acceleration there.
- Lag: the motion head is a bidirectional GRU; smoothers are symmetric.
- Jumps across unscored gaps: none in P2 (full-video smoothing is safe).
- Codex's geometry_root pilot: failed its own gate on 2 of 3 clips (3D mode on all 3); not uploaded.

## In flight

- P2L1 / P2L03: P2 refit after light mesh smoothing (lam 1 / 0.3); ACC-H.
- P4C: CARI4D CoCoNet person added to E2's two members (fixed thirds) on its 19 episodes, then P4's recipe; CD-H/CD-O.
- P2F (feet_hold.py) built and REJECTED locally: holding feet while the pelvis moves gives targets the leg cannot meet;
  refit error 0.6 mm median (normally 0.01) and joint acceleration rose 0.0445 -> 0.0650. Not uploaded.
- Packed for the next quota: P4tab102 (table x1.02; a quadratic through x0.974, x1 and x1.01 puts the optimum near 1.02),
  P4tab101flo101 (floor objects x1.01 on top).

## Endgame (Nov 4, before 22:00 UTC)

Every counted file must cite ONE final commit. penzero_v1 (PEN 0, Oct 17, `70c79b2`) is not stored locally; regenerate a
PEN-0 file with `scripts/pen_zero.py` (needs dense hands from `mhr_bodies.py --hands-only`) on the final CD file, which
keeps its human (CD-H, ACC-H unchanged). Then re-pack each board's best with the final commit and upload with `--only`.
