# Track 1 progress audit — November 2, 2026, about 01:15 CET

## Where we stand

`Izus_systems` is **5th overall by the documented rank-sum method**, with sum **40**.
The leader, RobotPI Lab, has sum **13**. The sub-10-cm target has **not** been achieved.

| Board | Verified team best | Source | Rank | Board-leading score |
|---|---|---|---|---|
| CD-H (cm) | 13.40710 | P3, submission 57887961 | 10 | 11.86428 |
| CD-O (cm) | 17.51622 | P3, submission 57887975 | 8 | 16.19900 |
| ACC-H | 0.13616 | P2, submission 57887719 | 10 | 0.13197 |
| ACC-O | 0.07051 | WF, submission 57885116 | 11 | 0.06809 |
| PEN | 0.00000 | penzero_v1, submission 57326945 | 1, tied | 0.00000 |

These best scores come from different files. P3's complete score vector is
13.40710 / 17.51622 / 0.13887 / 0.07064 / 0.00260. It improves the distance
metrics while losing acceleration/penetration quality relative to specialist files.

Against the original BMQ3KGM reference (14.35418 / 18.60437), P3 improves CD-H
by **0.94708 cm (6.60%)** and CD-O by **1.08815 cm (5.85%)**. Reaching 10 cm
still requires more than 3.40710 cm of human improvement and 7.51622 cm of
object improvement. Current leaderboard scores do not prove that this is attainable.

## Evidence checked

- Original October 31 handoff, November 2 handoffs, historical experiment reports,
  current cluster ledger, floor/CARI4D recipes, current branch changes and relevant
  build/upload/analysis scripts and logs.
- Live Kaggle submission history across all five boards, and current leaderboard CSVs.
  History: `../logs/progress_audit_2026-11-02_kaggle.json`.
  Standings: `../.tmp/audit-20261102/leaderboards/all.json`.
- Fresh full local test suite: **98 passed in 9.63 s**. This covers implementation
  behavior; it is not a ground-truth metric or an offline container reproduction test.
- P3 Parquet: exactly **740,780** official sample row IDs in order, finite coordinates,
  one commit URL (`1717c3ae6b0b0e74d0dfe62e177d038a1e91b223`). SHA256:
  `52be0a747dab96d2f20808b401079724c3bef3e2ca836586b3fd145de7c551b4`.
- P3, P2 and WF each have all 30 stored NPZ reconstructions and 30 object GLBs.
- All 30 released videos and both mask sets are complete: 16,563 images per mask set.

## Findings that change the next decisions

1. **The checkpoint mismatch is closed.** It was already tested on Kaggle before the
   cluster experiments and did not improve the scores. The new GPU pilot reproduced
   the intended zero-depth-output behavior; another upload would repeat known work.
2. **Geometry and mesh processing delivered the distance gains.** Camera correction,
   a moderated floor slide, object floor scaling/transfer, and mesh ensembling/smoothing
   produced P3. WHAM root mixing, global object shrinkage and direct Euler smoothing
   were rejected. The contact-offset pilot rejected its own proposal.
3. **Acceleration needs separate selection.** P2 improved ACC-H. Extra smoothing and
   S30 sharpening worsened acceleration scores. P3 should not displace P2 on ACC-H.
4. **The old current-best documents were stale.** V3's 13.56511 / OTa's 17.63890 have
   been superseded by P3. The top of CURRENT_BEST and the cluster ledger are now updated.

## Running and pending

- Job **434328**, `v2d-p1-c4all`, is running on **wr20**, using three GPUs on that
  single additional node. The owner's wr23/wr24 work was left untouched.
- At the final queue inspection: **11 episodes done, two failed, six claimed but
  unfinished, 11 unclaimed**. Counts change as the run proceeds. Episodes 12 and 18
  failed because FoundationPose produced an infinite `fp_centroid_distance`.
- C1/C2 are being built from the first completed CARI4D episodes (9, 13, 16, 24,
  25, 28). Their build has not yet emitted `CANDIDATESBUILT`; no confirmed CARI4D
  leaderboard result exists at this snapshot.
- Job **438703** waits for 434328 and then builds C3/C4 from the full finished set.
  Failed episodes retain the existing fallback person.
- Round 6 completed. Round 7's V9 scored 13.57953 CD-H / 17.63890 CD-O, below P3's
  quality on both distance boards. V10 subsequently started; its remaining results
  were not part of the leaderboard snapshot above. `work/submit_round7.sh` and
  `work/autopilot.sh` are alive. The latter waits for the build markers, then uploads
  the fixed C1–C4 list. No new uploads or jobs were launched by this audit.
- P4/P5 Parquets exist but do not yet have submission receipts in this workspace.

## Unfinished work that matters for winning and final delivery

1. Finish and score the CARI4D comparison; this is the major unmeasured alternative.
   Diagnose episodes 12/18 before retrying them. Reassess from actual results rather than
   assuming that an official baseline must win.
2. Select the strongest file separately for each metric, using the actual scoring
   rules. If seeking one balanced file, apply and evaluate the contact stage on the
   chosen distance candidate; do not assume its distance gains survive unchanged.
3. Build and verify the required offline Docker submission container and final
   technical report. Existing historical day reports are not that final report.
4. Restore or rebuild the missing BMQ ancestry (`episodes_bmq3k_in`,
   `episodes_bm_depth`, `episodes_objects_vpq`, `mask_visibility_bm`, `work/bm_root`).
   Current refinements start from downloaded scored artifacts. The official container
   also requires reproduction from released videos, not just repacking those artifacts.
5. Consolidate final counted files under a reproducible final code commit and keep
   the exact submitted files, receipts, model weights and kit copies. Current best
   files cite different commits. Preserve original provenance; any replacement must
   be a genuinely new submission. Resolve the documented external-camera-data rule
   question before final eligibility is assumed.
6. Coordinate the observed team submission cap with the teammate. Round 5 hit the
   team's daily allowance; automation's `ROUND...DONE` marker alone does not establish
   successful scoring—check its receipts/API states.

Freeze remains **November 4, 2026, 23:00 CET (22:00 UTC)**. The audit changed only
status documentation and saved read-only evidence; it did not alter running experiments.
