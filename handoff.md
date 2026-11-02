# V2D Track 1: handoff

**Snapshot: 2026-11-02, 19:45 UTC (20:45 Berlin).** Built from the Kaggle receipts on disk, the Slurm queue, git, the
live leaderboard snapshot `.tmp/lb_20261102_1941`, and the transcripts of today's Claude Code and Codex sessions.

> **Another session is still active.** Claude Code session `f72fb8d8` was still working when this was written. It was
> monitoring GPU jobs and running analyses. Before uploading anything, check `repo/submissions/*.kaggle-receipt.json` and
> `logs/kaggle_*.log` for uploads made after this snapshot, so the same file is not uploaded twice.

Freeze: **2026-11-04 22:00 UTC (23:00 Berlin)**. Time left at this snapshot: about 50 hours.

---

## 1. Where we stand

**Overall: 6th of all teams, rank sum 41** (ranks CD-H 10, CD-O 10, ACC-H 13, ACC-O 7, PEN 1).
Lodestar is 7th with 42, one point behind. EviisKitson is 5th with 25. Rice RobotPI Lab leads with 9.

How the rank sum moved today:

| Time (UTC) | Sum | Overall | Ranks (CD-H / CD-O / ACC-H / ACC-O / PEN) | Cause |
|---|---|---|---|---|
| Nov 2, ~00:15 | 40 | 5th | 10 / 8 / 10 / 11 / 1 | Codex's P2/P3 round |
| Nov 2, 12:48 | 44 | 7th (tied 6th) | 10 / 9 / 12 / 12 / 1 | Other teams improved |
| Nov 2, 18:47 | 46 | 8th | 10 / 10 / 13 / 12 / 1 | P4 kept CD-H at 10th; others moved |
| Nov 2, 19:41 | **41** | **6th** | 10 / 10 / 13 / **7** / 1 | **WFk60 moved ACC-O from 12th to 7th** |

### Team best per board (these come from different files)

| Board | Best | File | Commit cited | What it needs to climb (19:41 board) |
|---|---|---|---|---|
| CD-H (cm) | **13.32564** | P4 (today) | 6e03eac | < 13.26540 → 9th; < 13.21103 → 8th; < 13.14870 → 7th |
| CD-O (cm) | **17.51622** | P3 (also P4, identical) | 1717c3a / 6e03eac | < 17.34818 → 9th; < 17.25432 → 8th; < 17.23590 → 7th |
| ACC-H | **0.13616** | P2 (V4 ties) | 1717c3a-era | < 0.13611 → 12th; < 0.13608 → 11th; < 0.13590 → 10th |
| ACC-O | **0.06971** | WFk60 (today) | eb572ff | 0.06970 → tie for 5th (−2 points); < 0.06894 → 4th |
| PEN | **0.00000** | penzero_v1 (Oct 17) | 70c79b2 | Already tied 1st. **The file is not on disk; see flaw F9** |

Board leaders: CD-H 11.83658, CD-O 15.89170 (both RobotPI), ACC-H 0.13197 (Hyletic AI), ACC-O 0.06809 (BarcelonaHoneyCod).

**Realistic target:** first place is out of reach. CD-H is 1.5 cm behind the leader and CD-O 1.6 cm, and nothing we have
tested moves distance by that much. The cheapest gains are rank points on the acceleration boards and one or two places on
CD-H. Climbing to 5th needs about 17 points.

---

## 2. Rules and constraints (owner decisions; follow exactly)

- **Pushing:** standing approval (given 2026-11-02) to commit and push tested code to branch `vpq-vitpose` of
  `Swithinraj/v2d-track1-wham-foundationpose` whenever a candidate needs it, until the freeze. Tell the owner about each push.
- **Upload slots on Nov 3 and Nov 4:** use every slot except **2 per board per day**, which are kept for teammate
  `swithinraj`. Kaggle's limit is 20 uploads per team per UTC day. Each board is a separate Kaggle competition, and the
  dry-run prints the live count per board. Read this way, the 20 applies per board: 14 overnight uploads to all five
  boards left exactly 6 on each.
- **Slots left today (Nov 2), worked out from today's uploads:** CD-H 0, CD-O 1, ACC-H 0, ACC-O 0, PEN 1. The earlier
  session left these for teammates. They reset at 00:00 UTC (01:00 Berlin). Confirm with a `--dry-run`, which uploads nothing.
- **Freeze:** 2026-11-04 22:00 UTC. Before it, **every board's final file must be re-uploaded citing ONE common commit**.
  The award-container rule requires this: `tools/docs/kit_latest/v2d_submission_kit/tools/verify_container.py` checks that
  each counted file's `code_commit_url` matches the commit inside the container. The owner picks the 2 final submissions
  per board on Kaggle; that choice is theirs.
- **Compute:** GPU jobs only on node **wr20** (`--partition=gpu4 --nodelist=wr20`), one node in total. **Never query or touch
  Track 2 jobs or nodes** (the owner's wr23/wr24 work).
- **Filesystem:** all work stays under `/work/mnidgu2s/project1`. Always set `TMPDIR=/work/mnidgu2s/project1/.tmp`.
- **`work/autopilot.sh` was stopped at 18:45 UTC at the owner's request. Do not restart it.** It would upload C3/C4 by itself.
- **Provenance:** every file must cite a pushed commit that contains the code that built it. Never relabel new code with an
  old commit. `pipeline/recipes/pack_candidate.sh` refuses commits that are missing from the remote or lack the named scripts.
- **No leaderboard overfitting:** do not tune per-episode weights from leaderboard feedback. The ground truth is held out,
  the technical report must describe the method, and per-episode fitting risks disqualification.
- `submit_candidate.py` only uploads inside the unlimited final window (open until the freeze).

---

## 3. How the scorer works (this drives every design choice)

Read from `repo/official/v2d_submission_kit/metric_code/track_1/*.py` and `v2dlb/mhr_metrics.py`:

1. **One similarity transform** (scale, rotation, translation) is fitted on *our human's* alignment vertices at the **first
   scored frame** of each episode. It is then applied to the human **and** the objects.
   - So if a change leaves the human fixed at the first scored frame (the "anchor"), **CD-O does not change at all**.
     This was confirmed many times: CD-O is identical to 5 decimals across all person-only variants (V1–V10, P3→P4).
   - If the human changes at the anchor (for example an ensemble mesh), CD-O changes too, even with the objects untouched.
2. **CD-H / CD-O:** Chamfer distance per scored frame, averaged.
3. **ACC-H / ACC-O:** the error of our second difference (acceleration) against ground truth. It is computed per
   **contiguous stretch of ≥ 3 scored frames**; unscored gaps break the stretches. It is an error, not a smoothness
   measure, so removing real motion hurts, and that is exactly what heavy smoothing did (flaw F6).
4. **PEN:** mean depth of the hand vertices inside the object mesh (winding number), over scored frames. Exactly 0 ties for
   1st place, and every top team has 0.
5. **Overall rank** = lowest sum of the five board ranks (FAQ). Ties share a rank. Different files per board are allowed.

Scored frames per episode: `repo/outputs/hand_fix/episodes_repro_depth_refine_meshes/scored_frames.json`. Ten episodes have
internal gaps of up to 360 frames. Episode 11 is scored on all 803 frames.

---

## 4. The current approach

**Strategy: specialist files per board, built by cheap post-processing of stored reconstructions.** Each change touches
only what its target board measures. The anchor frame stays pinned, so the other boards stay controlled. Each test uploads
with `--only <board>`, so it costs one slot on that board only. On Nov 4 every final file is re-packed citing one common
commit.

The lanes:

| Lane | Board | Status | Base file | Method |
|---|---|---|---|---|
| A | ACC-O | **Worked: 0.07051 → 0.06971** | WF | `accel_shrink.py --objects --k 0.6` |
| B | ACC-H | **Running** (P2L1, P2L03) | P2 | light mesh smoothing (λ 1 / 0.3) + MHR refit |
| B′ | ACC-H | Rejected (P2r60 0.13643) | P2 | `accel_shrink.py --root --k 0.6` |
| C | CD-H (+CD-O) | **Running** (P4C) | P4 recipe | three-member human ensemble (adds CARI4D CoCoNet) |
| D | CD-O | **No active lane** | P3/P4 | see §10 for ideas |
| E | PEN | Keep 0; **must rebuild** for the final commit | penzero_v1 | `scripts/pen_zero.py` |

### 4.1 Lane A: acceleration shrink (`vpq-vitpose/scripts/accel_shrink.py`, commit `eb572ff`)

**Idea.** Our predicted acceleration is part real motion and part estimation noise. If `a = signal + independent noise`,
the error is lowest when acceleration is scaled by `k = signal / (signal + noise)`. Scored files bracketed the effect:
- Heavy smoothing, which is roughly k = 0, made both ACC boards worse: ACC-O +0.0009 and ACC-H +0.0019.
- Shrinking the table-object tracks by 0.974 lowered ACC-O.

So the optimum lies between, and k ≈ 0.6 was the first guess.

**Method.** Within each contiguous scored stretch, write `x_s + k·(x − x_s)`, where `x_s` is a Whittaker smooth
(λ = 1e4, about 0.5 Hz) of that stretch. Position and velocity still follow the track, and acceleration becomes exactly
k times as large. Each stretch is processed separately, so a jump across an unscored gap never leaks in. One constant
offset then pins the first scored frame, so the scorer's alignment is unchanged; a constant changes no acceleration.
Unscored frames are left as they are.

- `--objects` modifies `object_translation` (for ACC-O). `--root` modifies the human root translation `pose[:, :3]`
  (for ACC-H), which moves every joint rigidly.
- The script refuses to overwrite an existing output directory. It writes `accel_shrink_report.json` and checks that the
  values are finite and the anchor did not move.
- Tests: `vpq-vitpose/tests/test_accel_shrink.py` (5 tests). They cover identity at k = 1, untouched linear motion, the
  exact k-scaled residual, the pinned anchor, independent stretches across gaps, that the CLI changes only the requested
  columns, and invalid inputs. The full suite passed **148 tests** at 19:01 UTC.

**Results (uploaded with `--only`):**

| File | Board | Score | Previous best | Verdict |
|---|---|---|---|---|
| **WFk60** (WF objects, k 0.6) | ACC-O | **0.06971** | 0.07051 (WF) | **New team best**, rank 12 → 7 |
| P2r60 (P2 human root, k 0.6) | ACC-H | 0.13643 | 0.13616 (P2) | Worse; human root acceleration is mostly real motion |

Local effect: WFk60 cut object self-acceleration from 0.0309 to 0.0229 cm/frame² (median ratio 0.761). The largest root
change in P2r60 was 5.9 cm (episode 17).

**Not done yet (the obvious next step):** a k sweep on WF objects (for example k 0.4 / 0.5 / 0.75) to locate the optimum.
Shrinking object **rotation** acceleration is another option. Today only translation is shrunk, but ACC-O is computed on
object vertices, so rotation jitter counts too. That needs new code (shrink in a rotation-vector space around a smoothed
rotation) plus tests.

### 4.2 Lane B: light refit of P2 (ACC-H), running

`vpq-vitpose/pipeline/cluster/run_p2_light_refit.sbatch` (commit `82b02db`) runs `scripts/human_refit.py` on P2 with mesh
smoothing **λ 1 (about 5 Hz)** and **λ 0.3 (about 7 Hz)**, writing `episodes_cand_P2L1` and `episodes_cand_P2L03`. Earlier
refits used λ 100 (about 1.5 Hz), which removed real motion (R100S: ACC-H 0.13829). Objects and identity stay
byte-identical to P2.

Why try it: in P2, 47% of the human's acceleration energy is above 8 Hz. A light low-pass on all joints lowers
self-acceleration from 0.0557 to 0.0519 cm/f² (λ 1) while moving points only 0.006 cm on average. A low-pass on the root
alone barely changes anything (0.0557 → 0.0554).
**Caution:** P2r60 showed that human root acceleration is mostly real. The refit can also add per-frame fitting error of
its own. Treat this lane as a test, not an expected win.

### 4.3 Lane C: P4C three-member human ensemble (CD-H, and CD-O through the anchor), running

`vpq-vitpose/pipeline/cluster/run_e3_ensemble.sbatch` (commit `82b02db`) works as follows:

1. `human_ensemble.py` averages, on the mesh, BMQ3KGM (`episodes_cand_Mtp_in`, weight 0.34), ktcp (`episodes_base_ktcp`,
   0.33) and CARI4D CoCoNet (`episodes_c4tp_N19H`, 0.33). This covers the **19 episodes CARI4D finished**:
   `3 9 10 12 13 14 15 16 17 19 21 22 23 24 25 26 27 28 29`.
2. It refits the result (λ 100) and rebuilds bodies and mesh caches.
3. The other 11 episodes reuse E2R exactly.
4. It then applies P4's recipe: floor slide (λ 3e4, β 0.55), root mix 0.3 toward a0, and OTa objects carried to the person.
5. Output: `submissions/episodes_cand_P4C`.

Why try it: E2 (two SAM 3D Body members) was worth CD-H −0.10. CoCoNet as a *replacement* helped on 6 episodes (C1) and
hurt on 19 (N19H). A fixed-weight mean is bounded by its members and gains when their errors differ. It is not tuned per
episode. The human changes at the anchor, so **CD-O will change as well**; upload it to both distance boards.

### 4.4 What P4 is (current CD-H best)

P4 = P3's person (mesh ensemble E2R, mesh smoothing λ 100) + floor slide β 0.55 with λ 3e4 + root mix 0.3 toward a0
(`human_root_mix.py`), with OTa objects. Build script: `vpq-vitpose/pipeline/cluster/build_p4.sh` (commit `6e03eac`).
It changes only the human path. Objects and the anchor frame match P3 exactly, so CD-O is identical to P3 (17.51622).
Scores: 13.32564 / 17.51622 / 0.13889 / 0.07064 / 0.00257.

---

## 5. Timeline of today's work (UTC, Nov 2)

| Time | Who | What | Result |
|---|---|---|---|
| 00:00–00:56 | Codex + autopilot | Rounds 6/7: P1, P2, P3, V5–V10, S30, C1, C2, TIMEAXIS, C5 | P3 best on CD-H/CD-O, P2 best on ACC-H |
| ~01:15 Berlin | Codex | Audit; overnight plan (Lane A: CARI4D cached outputs, Lane B: HTD-Refine PVA-Net) | `docs/OVERNIGHT_PLAN_2026-11-02.md` |
| ~02:14–13:11 Berlin | — | Codex stopped by usage limits | — |
| 13:43 / 13:45 | Claude | Uploaded N19H, N19HO (Lane A) | Both worse; N19HO objects broken |
| 13:54 | Codex job 443714 | PVA_POSE fitting on all 30 episodes | Passed local checks |
| 17:46 | Codex | Pushed PVA code `3a57682` | — |
| 17:51 / 17:53 | Codex | Uploaded PVA_POSE, PVA_ACC (Lane B) | ACC-H 0.16712 / 0.15690, far worse; rejected |
| 17:51 | Codex job 444582 | Read-only audit of refined CARI4D objects | Still much jitterier than ours |
| 17:56–18:02 | Codex jobs 444584/444585 | New "geometry_root" approach: cache + 3-clip pilot | Failed its own gate (flaw F1) |
| — | Owner | Paused uploads while Codex ran (later replaced by the new rules in §2) | — |
| 18:45 | Claude | Stopped `autopilot.sh` (owner OK); uploaded **P4** | **CD-H 13.32564, new best** |
| 18:49 | Owner | "Continue; goal is to improve the score and reach the top" | — |
| 18:50–19:00 | Claude | Read the metric code; acceleration spectrum analysis; wrote `accel_shrink.py` + tests | §9 |
| 19:30 | Owner | Standing push approval; slots = all but 2 per board on Nov 3–4 | Saved to memory |
| 19:30 | Claude | Pushed `eb572ff` (accel_shrink + `--only` in uploader); packed WFk60, P2r60 | — |
| 19:32 / 19:33 | Claude | WFk60 → ACC-O only; P2r60 → ACC-H only | **ACC-O 0.06971 best**; P2r60 rejected |
| 19:40 | Claude | Pushed `82b02db` (two batch jobs); started jobs 444621 and 444622 | Running |
| 19:41 | Claude | Leaderboard refresh | **6th, sum 41** |
| 19:44 | Claude | Started job 444623 (λ 0.3 split out; time limits can't be extended) | Running |

---

## 6. Flaws found

### Found today (open unless marked)

- **F1. The GPT/Codex "geometry_root" approach is unfinished and fails its own pilot.** Files:
  `vpq-vitpose/scripts/geometry_root.py`, `geometry_root_pilot.py`, `tests/test_geometry_root.py`, and
  `pipeline/cluster/run_geometry_root_{cache,pilot}.sbatch`. All are **uncommitted**.
  - It ran a perspective root fit on 3 clips only (episodes 9, 13, 24). There is no step that applies it to all 30
    episodes or packs a file.
  - Mode 2 (lateral) made held-out keypoint error worse on 2 of 3 clips: ep 9 8.25 → 8.73 px, ep 24 11.34 → 11.87 px.
    Only ep 13 improved (7.30 → 6.74). The pilot gate failed.
  - Mode 3 (with depth) failed on all 3 clips with "unsupported anchor".
  - Log: `logs/geometry_root_pilot_444585.log`. **Do not upload. Treat it as rejected unless someone fixes the anchor
    support and it passes all 30 episodes.**
- **F2. The HTD-Refine PVA-Net adapter made ACC-H much worse.** PVA_POSE 0.16712 and PVA_ACC 0.15690, against 0.13616.
  - Better 2D keypoint agreement (22 of 30 clips improved) was a misleading proxy.
  - Even the root-only version regressed, so articulation fitting alone does not explain it.
  - **License:** HTD-Refine is **AGPL-3.0-only**, and the adapter code went public in `3a57682`. The owner should decide
    whether that is acceptable for the team repository.
  - Stopped; no more PVA GPU work.
- **F3. Raw CARI4D CoCoNet output is unusable as-is.**
  - **Objects:** N19HO ACC-O was 1.37088, about 20 times worse. The raw objects' median second difference is
    2.051 cm/f², against 0.0283 for ours; episode 12 reaches a mean of 5.39 and a p99 of 23.9.
  - **Humans:** N19H made CD-H worse (13.61 against P3's 13.41) and PEN non-zero.
  - Diagnostics: `repo/work/analysis/n19_object_diagnostics.json` and `refined_object_dynamics.json`.
  - Only the ensemble use in P4C remains.
- **F4. `autopilot.sh` was still live.** It would have uploaded C3/C4 automatically if `repo/logs/c4final.log` ever
  appeared. C3/C4 are near-duplicates of the rejected N19H/N19HO. **Fixed:** stopped at 18:45 UTC and noted in
  `logs/autopilot.log`. Do not restart it.
- **F5. Codex's docs are stale and contradict each other.**
  - `OVERNIGHT_PLAN_2026-11-02.md` and `PVA_RESEARCH_2026-11-02.md` say "no further Kaggle submissions". The owner has
    since replaced that with the rules in §2.
  - `CURRENT_BEST.md` still says 5th with sum 40, and lists P3 as the CD-H best and WF as the ACC-O best.
  - No doc mentions P4, WFk60, P2r60, P4C or P2L*.
  - **Trust this file plus the receipts.** Update `vpq-vitpose/docs/CURRENT_BEST.md` when you next commit.
- **F6. The pipeline smooths far too heavily for 30 fps motion.** Whittaker λ 3000–10000 cuts off around 0.5–0.65 Hz, and
  mesh smoothing λ 100 around 1.5 Hz. That is why every smoothing test hurt the ACC boards: it removes real motion. It
  also explains why P3/P4 (mesh smoothing λ 100) score ACC-H 0.1389 against P2's 0.13616. Distance boards and acceleration
  boards need **different files**.
- **F7. Reporting bug in the first `accel_shrink.py` (fixed before any packing).**
  - `track = values[:, cols]` was a numpy *view*. Writing the result back overwrote the "before" copy, so the report showed
    no change (ratio 1.000 for `--root`). The written data itself was correct.
  - Fixed with `.copy()`, plus a CLI regression test. Both candidates were rebuilt after the fix.
- **F8. P5 is not a good next CD-H test.** P5 is P4 with a 0.5 mix instead of 0.3. The same comparison was already
  scored: V6 (0.5) got 13.56530 against V5 (0.3) at 13.53699, so 0.5 was worse. Earlier today P5 was suggested as the
  follow-up. **Don't spend a CD-H slot on it.**
- **F9. The PEN-0 file is not on disk.** `penzero_v1` is Kaggle submission 57326945, file `penzero_v1_70c79b2.parquet`,
  from Oct 17. Its description is "every dense hand vertex cleared from the object (object pushes, v1)". It cites
  `70c79b2` and cannot be found anywhere under `/work/mnidgu2s/project1`.
  - The final single-commit re-upload needs a PEN-0 file that cites the final commit, so **it has to be rebuilt** with
    `vpq-vitpose/scripts/pen_zero.py` (from commit `c23b2e8`).
    Usage: `pen_zero.py --input DIR --hands DIR --out DIR [--margin_m 0.001] [--sigma 3]`.
  - Pushing objects out of the hands can hurt CD-O and ACC-O, which doesn't matter for a PEN-only file.
  - **Score it with `--only interpenetration_cm` on Nov 3 and confirm 0.00000.** Losing PEN 1st place would cost
    roughly 35 rank points.
- **F10. Human root acceleration is mostly real motion; object translation acceleration is partly noise.** This follows
  from P2r60 being worse while WFk60 was better. Don't shrink the human root further.
- **F11. Users can't extend Slurm time limits** (`scontrol update` → permission denied). Split long work into separate
  jobs. That is why job 444623 exists.

### Inherited from Codex's work (for awareness)

- **F12. Time-axis bug in `human_root_mix.py`** (fixed in `5e0bc96`): it smoothed scored samples as if they were
  consecutive frames across gaps, with up to 5.84 cm error in a synthetic test. Candidate TIMEAXIS scored −0.009 CD-H against V5.
- **F13. Shared-team uploader identity bug** (fixed): the official client refused uploads after a teammate's upload. The
  wrapper `scripts/v2d_submit_team.py` now uses the team name.
- **F14. FoundationPose infinite-centroid diagnostic** killed CARI4D episodes 12/18/20. Patched to NaN; CARI4D work is stopped anyway.
- **F15. Provenance gap on Oct 31:** those uploads cite `2cd29f4`, which lacks some of the scripts used.
  `swithinraj`'s uploads also cite it.
- **F16. Upload-rule exposure:** 13 uploads were made in the 7 days before the final window, against the published
  limit of 5 per week.
- **F17. The award container is not reproducible yet.**
  - The BMQ ancestry intermediates are missing: `episodes_bmq3k_in`, `episodes_bm_depth`, `episodes_objects_vpq`,
    `mask_visibility_bm`, `work/bm_root`.
  - The Docker container (`run.sh pack`, `run.sh reproduce` offline from video) and the technical report are not built.
  - This matters for awards after the freeze, not for the leaderboard.
- **F18. Open rule question:** the measured rig camera comes from FORM-HOI. The FAQ says to estimate parameters "only
  from the data provided for Track 1". Not confirmed with the organizers.
- **F19. Local proxies predict Kaggle poorly.** Neither FORM-HOI validation nor 2D keypoint agreement has been reliable.
  Only Kaggle decides; promote nothing without a scored control.

---

## 7. Approaches tested and rejected (do not retry unchanged)

| Approach | Files | Evidence |
|---|---|---|
| CARI4D CoCoNet humans as replacement | N19H, C1, C2, C5 | N19H CD-H 13.61; C1 helped only on 6 eps; C5 (no floor slide) 14.12 |
| CARI4D CoCoNet objects | N19HO | ACC-O 1.37088; CD-O 19.33 |
| HTD-Refine PVA-Net dynamics | PVA_POSE, PVA_ACC | ACC-H 0.16712 / 0.15690 |
| Perspective "geometry_root" | pilot only | 2 of 3 clips worse; depth mode unsupported |
| Human root acceleration shrink | P2r60 | ACC-H 0.13643 |
| Heavy object smoothing | E2S, R100S | ACC-O 0.07089 / 0.07139 |
| Heavy mesh smoothing (for ACC) | R100S, P3, P4 | ACC-H 0.13829 / 0.13887 / 0.13889 (still good for CD-H) |
| Motion sharpening | S30 | ACC-H 0.13775, ACC-O 0.07105 |
| WHAM root mix | WF | CD-H +0.59 at 0.25 |
| Global object scale 0.97 | G97 | CD-O 22.66 |
| Scaling table objects 0.974 | WF vs MF | +5.6 CD-O per table episode |
| Mix 0.5 instead of 0.3 | V6, V4 | worse than V5 / V3 |
| Floor slide β ≥ 0.75 or 0 | V2, FB15, C5 | best β ≈ 0.55 |
| Depth trend off (a0) for the human | BMQ3KGMa0 | CD-H worse (objects better) |
| Video Depth Anything, learned depth v1/v2 | — | see `docs/CURRENT_BEST.md` |
| Smoothing MHR Euler parameters directly | local only | joint acceleration 16× worse; always smooth on the mesh, then refit |

---

## 8. All Kaggle-scored files (from receipts in `repo/submissions/`)

| When (UTC) | File | CD-H | CD-O | ACC-H | ACC-O | PEN |
|---|---|---|---|---|---|---|
| 11-01 22:02 | MF | 14.08779 | 18.49797 | 0.13637 | 0.07057 | 0.00019 |
| 11-01 22:04 | G97 | 14.15256 | 22.65556 | 0.13636 | 0.07053 | 0.00016 |
| 11-01 22:05 | WF | 14.74125 | 20.91912 | 0.13632 | 0.07051 | 0.00014 |
| 11-01 22:34 | ML3 | 14.17711 | 19.39916 | 0.13644 | 0.07067 | 0.00013 |
| 11-01 22:36 | FB15 | 15.20630 | 17.97125 | 0.13646 | 0.07060 | 0.00012 |
| 11-01 22:38 | E2S | 14.05375 | 19.32561 | 0.13638 | 0.07089 | 0.00017 |
| 11-01 22:40 | R100S | 14.03962 | 19.20512 | 0.13829 | 0.07139 | 0.00009 |
| 11-01 22:40 | OTa | 13.67080 | 17.63890 | 0.13623 | 0.07062 | 0.00220 |
| 11-01 22:42 | OTb | 13.72005 | 18.36695 | 0.13623 | 0.07063 | 0.00141 |
| 11-01 22:48 | V1 | 13.64329 | 17.63890 | 0.13624 | 0.07062 | 0.00214 |
| 11-01 22:50 | V2 | 13.72610 | 17.63890 | 0.13625 | 0.07062 | 0.00180 |
| 11-01 22:52 | V3 | 13.56511 | 17.63890 | 0.13624 | 0.07062 | 0.00206 |
| 11-01 22:54 | V4 | 13.76572 | 17.63890 | **0.13616** | 0.07062 | 0.00206 |
| 11-02 00:02 | P1 | 13.55091 | 18.08246 | 0.13812 | 0.07058 | 0.00241 |
| 11-02 00:05 | P2 | 13.56097 | 18.04525 | **0.13616** | 0.07068 | 0.00256 |
| 11-02 00:07 | P3 | 13.40710 | **17.51622** | 0.13887 | 0.07064 | 0.00260 |
| 11-02 00:08 | V5 | 13.53699 | 17.63890 | 0.13625 | 0.07062 | 0.00201 |
| 11-02 00:10 | V6 | 13.56530 | 17.63890 | 0.13627 | 0.07062 | 0.00197 |
| 11-02 00:12 | S30 | 13.83147 | 17.66263 | 0.13775 | 0.07105 | 0.00213 |
| 11-02 00:14 | V9 | 13.57953 | 17.63890 | 0.13625 | 0.07062 | 0.00215 |
| 11-02 00:16 | V10 | 13.79185 | 17.63890 | 0.13629 | 0.07062 | 0.00220 |
| 11-02 00:18 | V7 | 13.61675 | 17.63890 | 0.13625 | 0.07062 | 0.00203 |
| 11-02 00:20 | V8 | 13.53405 | 17.63890 | 0.13622 | 0.07062 | 0.00211 |
| 11-02 00:25 | C1 | 13.49139 | 17.71056 | 0.13651 | 0.07062 | 0.00226 |
| 11-02 00:27 | C2 | 13.51259 | 17.67464 | 0.13805 | 0.07062 | 0.00214 |
| 11-02 00:45 | TIMEAXIS | 13.52781 | 17.63890 | 0.13624 | 0.07062 | 0.00201 |
| 11-02 00:56 | C5 | 14.11530 | 17.71056 | 0.13654 | 0.07062 | 0.00233 |
| 11-02 13:43 | N19H | 13.61196 | 17.58893 | 0.13669 | 0.07061 | 0.00282 |
| 11-02 13:45 | N19HO | 13.61196 | 19.32814 | 0.13669 | 1.37088 | 0.00295 |
| 11-02 17:51 | PVA_POSE | 13.54747 | 18.04525 | 0.16712 | 0.07068 | 0.00256 |
| 11-02 17:53 | PVA_ACC | 13.55762 | 18.04525 | 0.15690 | 0.07068 | 0.00257 |
| 11-02 18:47 | **P4** | **13.32564** | **17.51622** | 0.13889 | 0.07064 | 0.00257 |
| 11-02 19:32 | **WFk60** (ACC-O only) | — | — | — | **0.06971** | — |
| 11-02 19:33 | P2r60 (ACC-H only) | — | — | 0.13643 | — | — |
| 10-17 | penzero_v1 (not on disk) | — | — | — | — | **0.00000** |

Older history: `docs/CLUSTER_PROGRESS.md`, `docs/EXPERIMENT_REPORT.md`, `logs/progress_audit_2026-11-02_kaggle.json`.

Built but never scored: P5 (see F8), C3/C4 (don't), PVA_ACC_PILOT, SC, EW30, the FB15_in / ML3_in variants, OT, OT_in.

---

## 9. Analyses run today and what they imply

All are measured on our own outputs, so "self-acceleration" means the size of our predicted acceleration, not its error
against ground truth. Scripts are in the other session's scratchpad
(`/tmp/claude-2711/-home-mnidgu2s/f72fb8d8-f6fc-400a-bb37-8d6ab34b82ab/scratchpad/`), which is temporary; the numbers are
copied here.

| Analysis | Result | Implication |
|---|---|---|
| Acceleration spectrum per scored stretch | Human (P2): 39% of energy at 0–3 Hz, 5.6% at 3–5, 8.8% at 5–8, **46.6% above 8 Hz**. Objects (WF): 51.9 / 3.7 / 6.7 / **37.8%** | Much of the acceleration is high-frequency; some of it is likely noise |
| Root vs articulation | Root 0.0386, joints relative to root 0.0453 cm/f² | Both matter for the human |
| Light Whittaker low-pass (λ 0.2–2, about 8–4 Hz) | Root alone: 0.0557 → 0.0553 (barely any change). All joints: 0.0557 → 0.0512 at λ 2, mean move 0.009 cm. Objects: 0.0309 → 0.0292 | Basis for lane B (P2L1/P2L03) |
| Blend toward a heavy smooth (λ 1e4) | Human w 0.2: self-acceleration 0.0476, mean move 0.34 cm; objects w 0.2: 0.0268, mean move 0.20 cm | Basis for accel_shrink (k = 1 − w) |
| Joint jumps at scored-stretch edges | Steps near edges are not larger than the 99th percentile inside the stretch | No edge artifact to fix |
| Acceleration along the camera ray vs across it | Ray/lateral mean-square ratio: objects 0.61, human 0.80; only 2–3 of 30 episodes above 2 | Noise is not mainly in depth, so depth-only smoothing is not indicated |
| Objects at rest (speed < 1 cm/s, no wrist within 25 cm) | 10.5% of scored frames; acceleration at rest is already small (0.0028 vs 0.0201 overall after shrink) | Zeroing acceleration at rest would gain little |
| Per-joint acceleration in P2 | Feet 0.081/0.078, wrists 0.081/0.078, ankles 0.077/0.075, elbows 0.066/0.064; trunk 0.039–0.052. Legs mean 0.063, arms 0.067, trunk 0.047 | Hands and feet dominate; an extremity-only shrink or smooth is a candidate for ACC-H |

---

## 10. Next steps, in priority order

### When the running jobs finish (tonight)

| Job | Name | Builds | Log | Note |
|---|---|---|---|---|
| 444621 | v2d-p1-p2-light | `episodes_cand_P2L1` | `logs/p2_light_refit_444621.log` | **Expected to end with "exists; refusing to overwrite" and status 2** after λ 1, because 444623 already owns P2L03. This is by design, not a failure |
| 444623 | v2d-p1-p2-light03 | `episodes_cand_P2L03` | `logs/p2_light_refit03_444623.log` | λ 0.3 |
| 444622 | v2d-p1-e3 | `episodes_cand_P4C` (+ `episodes_ens_E3`, `refit_E3`, `person_P4C`, `personm_P4C`, meshes `E3R`) | `logs/e3_ensemble_444622.log` | Done when the log prints `E3_BUILT` |
| 444624 | v2d-p1-p2-feet | unknown at snapshot | check `logs/` | Queued by the other session at about 19:48 UTC, after this file was drafted. Probably the hands-and-feet ACC-H variant from §9. Read its script and log before using its output |

For each finished candidate:
1. **Validate:** all 30 episodes present, values finite, anchor unchanged where expected. P2L* objects must be
   byte-identical to P2. P4C objects come from `object_transfer`, so they will differ.
2. **Pack** citing `82b02db`. That commit holds both sbatch files, and every script the jobs used is byte-identical to its
   committed version, which the other session verified for `human_ensemble`, `human_refit`, `floor_bodies`,
   `ensemble_meshes`, `floor_root`, `human_root_mix` and `object_transfer`. See §12 for the command.
3. **Dry-run, then upload after 00:00 UTC** (today's CD-H and ACC-H slots are used up):
   P2L1 and P2L03 with `--only acc_h_cm`; P4C with `--only cd_h_cm cd_o_cm`.
4. Promote a file only if it beats the per-board best (§1).

### Nov 3 (about 18 slots per board minus teammates' use)

1. **ACC-O k sweep** (CPU, minutes each): build WFk40, WFk50, WFk75 from WF with `accel_shrink.py --objects` and upload
   each with `--only acc_o_cm`. Fit a parabola over the k values that scored. A score of 0.06970 or better ties 5th, and
   anything below 0.06894 is 4th.
   - Then try a **rotation shrink** for objects. It needs new code and tests.
   - A different reference λ (for example 3e3 or 3e4) is a second knob.
2. **ACC-H:**
   - If P2L1 or P2L03 beats 0.13616, sweep λ around the better one.
   - If both are worse, try an **extremity-only** variant (hands and feet carry the most acceleration, §9) or drop the lane.
   - Do not retry root shrink (P2r60).
3. **CD-H:**
   - If P4C wins, it becomes the new base: check whether mesh smoothing or the slide λ needs retuning on it.
   - Otherwise try P4's untested neighbours, such as mix 0.2 or slide λ 1e5. **Not P5** (F8).
   - Scores to beat: 13.26540 for 9th, 13.21103 for 8th.
4. **CD-O** (no lane yet; rank 10, 17.51622; < 17.34818 is 9th). Options supported by the ledger:
   - (a) P4C's ensemble changes the human at the anchor, so its CD-O score is the first data point.
   - (b) Per-episode object depth from floor contact and size constancy: the floor scale is worth −0.90 and the 2 cm floor
     offset −0.53. Table objects stay unscaled.
   - (c) Improve the human at the anchor frame only, since that moves the objects through the similarity fit.
5. **PEN:** rebuild a PEN-0 file with `pen_zero.py` from a stored reconstruction and verify 0.00000 with
   `--only interpenetration_cm` (F9). This must be done **before Nov 4**.

### Nov 4: endgame (finish well before 22:00 UTC)

1. Freeze the code: one final commit on `vpq-vitpose` containing every script used by every final file (accel_shrink,
   human_refit, floor_root, human_root_mix, object_transfer, human_ensemble, pen_zero, …). Push it.
2. Re-pack each board's final file from its **stored episode directory**, citing that commit. The numbers come out
   identical; only `code_commit_url` changes. Keep the directories (Codex suggests `outputs/<label>/reconstructions`).
3. Re-upload each one with `--only <its board>`. Check that the re-uploaded score equals the original.
4. Ask the owner to pick the 2 final submissions per board on Kaggle.
5. Update `docs/CURRENT_BEST.md` and the ledger. Record which file and which receipt counts for each board.
6. After the freeze, for awards: the Docker container (`tools/docs/kit_latest/v2d_submission_kit/CONTAINER_SUBMISSION.md`),
   `verify_container.py`, and the technical report (F17).

---

## 11. Code and git state

**Dev checkout:** `/work/mnidgu2s/project1/vpq-vitpose`, branch `vpq-vitpose`, a git worktree of `repo/.git`.
HEAD = origin = `82b02db`.

| Commit | When (Berlin) | What |
|---|---|---|
| `82b02db` | 11-02 20:40 | Batch jobs: light refit of P2 (P2L1/P2L03) and the three-member ensemble (P4C) |
| `eb572ff` | 11-02 20:30 | `accel_shrink.py` + tests; `submit_candidate.py --only` |
| `3a57682` | 11-02 18:46 | PVA-Net MHR adapter (Codex; rejected; AGPL concern) |
| `5e0bc96` | 11-02 01:37 | Time-axis fix in root mixing; occlusion diagnostics |
| `6e03eac` | 11-02 01:10 | Stacked person candidates (`build_p4.sh`) |
| `1717c3a` | 11-02 00:12 | P1–P3 era |

**Uncommitted in `vpq-vitpose` (Codex's; left alone deliberately):**
- Modified: README, `config/*.def`, docs (HANDOFF, OVERNIGHT_PLAN, PIPELINE_CORRECTION, PVA_RESEARCH), and the
  `pipeline/setup/*` scripts.
- Untracked: `scripts/geometry_root*.py`, `tests/test_geometry_root.py`, `pipeline/cluster/run_geometry_root_*.sbatch`,
  `scripts/colab_mask_batch.py`, `scripts/compare_camera_source_episode.py`, several `pipeline/setup/*` helpers, and docs
  (`SETUP_STATUS`, `TRACK1_COMPUTE_TRIAGE`, `VPQ_BRANCH_REVIEW`).

Commit only the files a candidate needs, as done today.

**Runtime checkout:** `/work/mnidgu2s/project1/repo`, at `5e0bc96` on `cluster-floor-2026-11`. Data, containers, venvs,
submissions and batch jobs run from here (`--chdir=repo`). Its `scripts/*.py` are byte-identical to the dev checkout for
everything used today. `accel_shrink.py` exists **only** in `vpq-vitpose/scripts`; call it as
`../vpq-vitpose/scripts/accel_shrink.py` from inside `repo/`.

---

## 12. Command reference

```bash
P=/work/mnidgu2s/project1; export TMPDIR=$P/.tmp

# Leaderboard and rank sum (read-only Kaggle API)
cd $P && KAGGLE_CONFIG_DIR=$PWD/.cache/kaggle repo/.venvs/submission/bin/python repo/scripts/analysis/standings.py \
  --top 8 --out .tmp/lb_$(date -u +%Y%m%d_%H%M)

# Tests (148 passed at 19:01 UTC)
cd $P/vpq-vitpose && .venvs/submission/bin/python -m pytest -q --basetemp=$P/.tmp/pytest_$(date +%H%M)

# Build an ACC-O shrink candidate (CPU, seconds)
cd $P/repo && .venvs/submission/bin/python ../vpq-vitpose/scripts/accel_shrink.py \
  --input submissions/episodes_cand_WF --out submissions/episodes_cand_WFk50 --objects --k 0.5

# Pack, citing a pushed commit that contains the scripts
cd $P/repo && COMMIT=<40-char sha> CANDIDATE_SCRIPTS="scripts/accel_shrink.py scripts/floor_root.py" \
  bash pipeline/recipes/pack_candidate.sh WFk50          # -> submissions/track1_cand_WFk50.parquet
#   P2L*: CANDIDATE_SCRIPTS="scripts/human_refit.py"
#   P4C:  CANDIDATE_SCRIPTS="scripts/human_ensemble.py scripts/human_refit.py scripts/floor_bodies.py \
#         scripts/ensemble_meshes.py scripts/floor_root.py scripts/human_root_mix.py scripts/object_transfer.py"

# Dry-run (shows slots left per board), then upload to one board only
cd $P/vpq-vitpose && .venvs/submission/bin/python scripts/submit_candidate.py \
  ../repo/submissions/track1_cand_WFk50.parquet --only acc_o_cm -m dry --dry-run
.venvs/submission/bin/python scripts/submit_candidate.py ../repo/submissions/track1_cand_WFk50.parquet \
  --only acc_o_cm -m 'WFk50: <what changed>; code <sha>' > $P/logs/kaggle_wfk50.log 2>&1
# Board names: cd_h_cm cd_o_cm acc_h_cm acc_o_cm interpenetration_cm

# GPU job (wr20 only); the sbatch files chdir into repo/
cd $P/vpq-vitpose && sbatch pipeline/cluster/<job>.sbatch
squeue -u $USER -h -o '%i %j %T %M %R' | grep -v v2d-eval   # v2d-eval jobs belong to other work; leave them alone
```

### Data formats

- `submissions/episodes_cand_<NAME>/episode_000000.npz … episode_000029.npz`, plus `episode_*_object.glb`. Each npz holds:
  - `pose` (T, 136) float32. **`pose[:, :3]` is the root translation in decimetres in MHR axes**; to get Track 1 metres,
    use `pose[:, :3] * [1, -1, -1] / 10`.
  - `scales` (68,), `shape` (45,): identity; never change these.
  - `object_rotation` (T, 3, 3), `object_translation` (T, 3) in metres, `object_scale` (scalar).
- Mesh caches: `repo/work/analysis/meshes/<label>/episode_*.npy`. Bodies: `repo/work/analysis/bodies_<label>/`.
  PVA joint caches: `repo/work/pva/joints_P2_all/` (127 MHR joints).
- Containers: `repo/containers/*.sif`. `mhr_converter.sif` does GPU MHR refits; `v2d_cari4d.sif` runs CARI4D. Call them
  through `in_container` from `repo/pipeline/env.sh`.
- Kaggle token: `v2d_track1_handoff_2026-10-31/tokens/kaggle_api.txt`. The uploader reads it; never print it.

### Lineage of the files that matter

- **BMQ3KGM** (best file on Oct 27): SAM 3D Body in body mode, objects in the body-mode depth scale, VPQ3 lateral band,
  GRU v2 motion head with its depth correction off, then the contact stage.
- **WF:** BMQ3KGM + measured rig camera + WHAM root mix 0.25 + floor-scaled objects + table objects 0.974 + contact stage.
  → **WFk60** (ACC-O best).
- **OTa:** measured-camera person with floor slide 0.5; objects carried to the person (`object_transfer.py`); 2 cm floor
  offset; no contact stage. These are the objects every V*/P3/P4 file uses.
- **P2:** mesh ensemble with the Oct 31 human + floor slide 0.55; objects floor-scaled with relative size among table
  episodes. → ACC-H best.
- **P3:** mesh ensemble E2R + mesh smoothing λ 100 + floor slide 0.55 + OTa objects. → CD-O best.
- **P4:** P3 + slide λ 3e4 + root mix 0.3 toward a0. → CD-H best.
- **penzero_v1:** Oct 17, commit `70c79b2`, hand vertices cleared from objects. → PEN 0. Must be rebuilt (F9).

---

## 13. Other docs

- Up to date: **this file**, the receipts (`repo/submissions/*.kaggle-receipt.json`), and the logs (`logs/kaggle_*.log`).
- Codex's docs in `vpq-vitpose/docs/`:
  - `CLUSTER_PROGRESS.md` and `EXPERIMENT_REPORT.md`: full results ledger up to Nov 2 01:15 Berlin. Still useful for
    history.
  - `PIPELINE_CORRECTION_2026-11-02.md`, `PVA_RESEARCH_2026-11-02.md`, `TRACK1_COMPUTE_TRIAGE_2026-11-02.md`: accurate
    about what they cover. Ignore their "no uploads" lines (F5).
  - `OVERNIGHT_PLAN_2026-11-02.md`: Lane A and Lane B are finished and rejected.
  - `CURRENT_BEST.md` and `HANDOFF_2026-11-02.md` (also at the project root): **stale** scores, ranks and rules.
- Recipes: `repo/pipeline/recipes/floor.md`, `cari4d.md`, `pack_candidate.sh`.
- Official kit: `repo/official/v2d_submission_kit/` (metric code in `metric_code/track_1/`). The newest kit and the
  container docs are in `tools/docs/kit_latest/v2d_submission_kit/`.
- Owner rules for future sessions: memory note `v2d_track1_kaggle_rules`.
