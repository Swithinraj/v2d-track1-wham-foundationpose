# Track 1 compute triage — November 2, 2026

The owner authorized stopping low-value Track 1 compute and explicitly forbade
inspecting Track 2. Only the two already-known Track 1 jobs were queried:
`434328` (`v2d-p1-c4all`, running on wr20) and `438703` (`v2d-p1-c4f`, dependent
build pending on wr20). No Track 2 job or node was queried or changed.

## Action and verification

Cancelled dependent job **438703 first**, then running job **434328**. Cancelling
the dependency first prevents its `afterany` condition from starting another
build when the main job stops. A subsequent query of these exact job IDs showed
neither in the queue; accounting reported both **CANCELLED**. Main elapsed time
was 02:23:08. Its allocation was one node, three GPUs, 66 CPUs and 360 GB RAM.
No replacement GPU allocation has been launched as part of this triage.

All cached files were preserved. Before cancellation, 19 CoCoNet stage markers
were checked against their output files' existence, byte size and nanosecond
mtime. Valid episodes were:

`3 9 10 12 13 14 15 16 17 19 21 22 23 24 25 26 27 28 29`

These stage-complete episodes are distinct from the 14 full-pipeline `.done`
markers. Do not promote the other episodes to full-pipeline completion. Some
cancelled episodes have partial later-stage artifacts; validate markers before
reuse. Episode 8 had also failed in FoundationPose before cancellation.

## Why stop rather than finish the original batch

- The worker always ran all eight stages, including 300-step post-optimization
  and two visualization renders. The upstream runner has no stop-after flag.
- C1 (CoCoNet, mesh lambda 300) scored CD-H **13.49139** and ACC-H **0.13651**.
  C2 (refined, mesh lambda 30) scored **13.51259** and **0.13805**. Both replace
  the same six episodes in a V3 fallback. C2 improved CD-O and PEN versus C1,
  but neither beat P3's distance scores. This comparison changes stage AND
  smoothing; it does not prove refinement alone worsens human accuracy.
- C1 improved CD-H against its actual V3 control (13.56511), so CARI4D has
  evidence of value. It is not rejected as a method. A full expensive pass is
  premature while 19 useful stage outputs remain unevaluated.
- Visualization renders do not enter the scored reconstruction. Their outputs
  can be generated later if needed for inspection.

The already-built **C5** changes only floor beta from 0.55 to 0 on the same six
CoCoNet episodes as C1. It was evaluated without new GPU reconstruction:

| Candidate | CD-H | CD-O | ACC-H | ACC-O | PEN |
|---|---|---|---|---|---|
| C1, beta 0.55 | 13.49139 | 17.71056 | 0.13651 | 0.07062 | 0.00226 |
| C5, beta 0 | 14.11530 | 17.71056 | 0.13654 | 0.07062 | 0.00233 |

Removing the floor prior worsened human distance by **0.62391 cm**. Reject beta 0
for this pipeline; retain beta 0.55 when testing the cached 19-episode candidate.
All five metrics completed successfully. Exact Kaggle submission IDs are
57889868 / 57889872 / 57889874 / 57889877 / 57889880.
The full receipt is
`repo/submissions/track1_cand_C5.parquet.6b683ee43752.kaggle-receipt.json`.

Any resumed GPU run must remain pinned to wr20 and stop after the stage actually
being evaluated. Before selecting refinement, use a matched smoothing setting.
Do not restart the old all-stage batch or cancelled dependent build unchanged.

Runtime evidence is stored under
`/work/mnidgu2s/project1/.tmp/track1_triage/`; evaluation log:
`/work/mnidgu2s/project1/logs/kaggle_c5_triage.log`.
