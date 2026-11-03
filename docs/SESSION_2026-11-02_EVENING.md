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

## November 3 results (all uploads board-specific)

| File | Board | Score | Verdict |
|---|---|---|---|
| P4tab102: table objects x1.02 | CD-O | 16.89551 | better than x1.01 (17.08369) |
| P4Ctab101: CARI4D CoCoNet person as third ensemble member (19 eps, thirds), table x1.01 | CD-H / CD-O | **13.10770** / 17.07383 | CD-H -0.22 |
| P2S60: full-body acceleration x0.6 | ACC-H | 0.13692 | rejected (shrinking human motion hurts) |
| P4tab101flo101: floor objects x1.01 on top | CD-O | 16.92676 | floor scale helps (-0.157) |
| P4L300tab101: mesh smoothing lam 300 | CD-H | 13.39787 | rejected (lam 100 better) |
| WFk55final: object acceleration x0.55 | ACC-O | 0.06972 | k 0.6 stays (0.06971) |
| P4Cs23f2: table x1.023, floor x1.02 | CD-O | 17.04921 | floor x1.02 overshoots |
| P4Ctszf2: per-episode size constancy (table) | CD-O | 18.37516 | rejected (+1.33) |
| P4Cs23f08: table x1.023, floor x1.008 (fitted optima) | CD-O | **16.73331** | new CD-O best |
| E3W5tab: CoCoNet weight 0.5 (others 0.25) | CD-H | **13.08770** | new CD-H best (fitted optimum ~0.47) |
| P2E: P2 recipe on the E3W5 ensemble | ACC-H | 0.13664 | rejected (CoCoNet person is mesh-smoothed) |
| P4tab101Z: PEN-0 regeneration (pen_zero.py) | PEN | 0.00000 | endgame PEN file works |

Object scales for the final: `pipeline/cluster/object_factors_t1023_f1008.json`. ACC-H stays on P2 (0.13616): every
smoothing or shrink of the human scored worse. CARI4D's CoCoNet stage was run on the 11 missing episodes (one episode
per job: stopping the container after 06_coconet also ends its job), then `run_c4_c30.sbatch` and
`run_ensemble_variant.sbatch` (NAME=E3C30) build the all-30 ensemble, and `run_pen_zero.sbatch` its PEN-0 version.

## Final files (all cite commit 794e7726fabc6eab0fb43172638ea4baa8740430)

Late Nov 3: CARI4D's CoCoNet person on all 30 episodes (E3C30tab) gave CD-H 12.97765 / CD-O 16.71158; root mix 0.15
gave CD-H 12.96979 (0.3: 12.97765, 0.45: 13.01137); floor slide beta 0.45/0.65 were worse (13.01232/13.00200).

| Board | File | Score | Status |
|---|---|---|---|
| ACC-H | track1_cand_P2final.parquet (= P2) | 0.13616 | uploaded Nov 3 (57952955) |
| ACC-O | track1_cand_WFk60final.parquet (= WFk60) | 0.06971 | uploaded Nov 3 (57952977) |
| PEN | track1_cand_E3C30tabZ.parquet | 0.00000 | uploaded Nov 3 (57953874) |
| CD-H, CD-O | track1_cand_C30L15m15tab.parquet or track1_cand_C30m15tab.parquet | — | Nov 4, see below |

Nov 4 (fresh quota): upload C30L15m15tab (mesh smoothing lam 150, root mix 0.15) with `--only cd_h_cm cd_o_cm`. If its
CD-H beats 12.96979 it is the CD final; otherwise upload C30m15tab (12.96979 already measured under another commit) with
`--only cd_h_cm cd_o_cm`. Then the owner selects, per board on Kaggle, the files above as final submissions.
