# HANDOFF — V2D Track 1: depth-bias CD-O probe + floor-consistent placement

**Date:** 2026-11-02 · **Deadline:** Nov 4, 22:00 UTC (23:00 CET) · **Author:** agent session (owner: Ujjwal)
**Lane separation:** object-bias (this doc) is separate from Swithinraj's BMQ chain rebuild. Do not rebuild his chain.

---

## 1. Current standing (BMQ3KGMtp = best submission)

| Metric | Score | Kaggle rank |
|---|---|---|
| CD-H | 14.3103 cm | #11 |
| CD-O | 18.0195 cm | #10 |
| ACC-O | (read from submission history — not recorded here) | #12 |
| PEN | — | #5 |

- Track 1 = **5 separate Kaggle competitions** (cd-h, cd-o, acc-h, acc-o, pen); one parquet posts to all 5 via `official/v2d_submission_kit/submit/v2d_submit.py --track track_1`.
- **Quota: 3–4 probes per metric per day** (CD-H had 3 left on Nov 2). Uploads are owner-gated.
- Metric definitions (from `metric_code/track_1/`): CD = symmetric Chamfer on surface points; **ACC = mean ‖Δ²(traj)‖ error in cm/frame²** (second difference of 22 body joints / object translation); PEN = human–object interpenetration.

## 2. The two improvements (owner-approved, offline-tested)

### #1 Depth-ICP object bias → CD-O  (main prize)

**Design** (`scripts/depth_icp_objects.py`): per scored frame,
`bias(f) = P70 of along-ray residual r = z_mesh − d_depth` over the object mask's valid pixels
(the object is the nearest surface in its mask; P70 trims background/occluders). Depth decoded
`65535/v − 1` m and **divided by the episode's alignment `depth_scale`** (mandatory — the raw ratio
diagnostic read ≈0.887 = ep0's scale). Regularisation: episode-median gating (per-frame biases
>20 cm off the median collapse to it) → Gaussian σ=8 smoothing → taper to zero at the first scored
frame (the scorer's alignment frame must not move). Applied as a translation along the object
centre's camera ray. **No rotation** — the mask evidence can't support it.

**Why not real ICP:** per-frame point-to-plane ICP **diverged** on our data (~98% of frames rejected or
wanting >15 cm/>15° corrections) — thin-object masks (hula-hoop ring, rollers) are mostly background
pixels and depth-edge normals are garbage. The along-ray 1-DOF bias is what the data supports.

**Results, fresh chain** (`submissions/episodes_sam_objbias_{const,frame}`):
smoke 3/3 improve (ep7: 15.5→8.9 cm); full 30: frame-mode 19/30 improve, const 15/30; eps 1/2
(thin ring) give bad biases.

**Results, recovered candidate** (`outputs/depth_icp/report_recovered.json`) — the probe candidate:

| stat | value |
|---|---|
| Applied & improved | **23/30 episodes** (median bias 2.97 cm) |
| Biggest wins (same-frame res, cm) | ep0 34.9→4.2 · ep2 12.4→3.0 · ep5 11.7→5.0 · ep21 18.3→5.5 · ep16 9.7→4.8 · ep23 11.2→5.3 |
| `depth_frame_ratio` validity | 24 eps in 0.95–1.08 → same camera/depth frame as our data ⇒ biases are real geometry |
| **Excluded** | ep1 (ratio 1.33), ep9 (1.16) — invalid; ep28 (res worsened 17.4→23.8); ep29 (bias 0.21 cm no-op); eps 8/14/18/19/20 — no depth evidence |
| Mask covers 0.02–0.37 | explained: displaced objects don't sit on the depth region — low cover = less evidence, not wrong evidence (residual collapses where evidence exists) |

**Estimated gain: CD-O 18.0 → ~13–15** (mean same-frame residual 13.5→7.8 cm on touched eps,
diluted ~4.2 cm over all 30; optimistic ~11 if the big-bias episodes dominate the metric).

### #2 Floor-consistent human placement → CD-H

**Tool** (`scripts/floor_planes_soles.py`): MoGe-depth RANSAC floor (600 iters, 3 cm inliers,
within 25° of body-up, body-above-plane check, ≥2000 inliers; refit on inliers) — cannot diverge,
unlike the old sole-band fit. Body-up from MHR forward: head SC[15]=113 minus mean ankles
SC[7,8]=[4,20]. Output `outputs/physics/floor_planes_soles.json` (+ `.report.json`), `c` stored in
MoGe metres (= c_our × depth_scale) for human_refit's contract. **30/30 valid floors.**

**Findings — the two chains disagree vertically:**

| chain | sole position vs depth floor |
|---|---|
| Fresh chain (`episodes_sam_*`) | **floats 0–14 cm above** (median ~8) — systematic CD-H error |
| Recovered candidate (`rec_float.py` → `logs/rec_float.json`) | **sinks mean −6.8 cm below** (median −6.1, range −0.7…−16.6; 0 eps float >2 cm) |

Consequences:
- The float problem is **fresh-chain-only**; the old candidate's humans are already grounded ⇒
  **no floor gain available on the probe candidate; CD-H stays 14.31.**
- The pending `feet-refine` job (fresh chain, widened gates) is the Kaggle test of whether
  floor-consistent placement pays at all.
- Chains differ by up to ~16 cm vertically on the same episode (ep0: fresh +3.4 vs recovered −13.2).

## 3. ACC transfer analysis (exact, no GT needed) — `logs/rec_acc.json`

Because ACC-O = mean ‖Δ²t_pred − Δ²t_gt‖, the *change* from our correction is computable exactly:

| stat | value |
|---|---|
| Added acceleration ‖Δ²(B·ray)‖: mean / p90 / max | **0.021 / 0.036 / 0.047 cm/frame²** |
| Original translation's own 2nd-diff scale | 0.040 cm/frame² |
| Worst-case ACC-O increase (triangle ineq.) | **≤ +0.021 cm/frame²** |
| If uncorrelated, quadrature | E → √(E² + 0.021²) |

- Whether that is negligible depends on the current ACC-O value → **read BMQ3KGMtp's ACC-O from
  the Kaggle submission history (free, no probe)**. If ≥0.2 → worst case ~10% relative hit, go frame
  mode. If ≲0.05 → upload the **const-mode** parquet instead (constant per-episode bias ⇒ zero
  injected acceleration, slightly less CD-O gain).
- A **constant per-episode vertical shift of humans is ACC-H neutral** (Δ² ≡ 0) — relevant if the
  floor fix is ever applied to a candidate.

## 4. PEN safety gate (blocking for the object-bias probe)

`scripts/pen_check.py` (kit's numba penetration kernels; hand-inside counts before/after):
```
python scripts/pen_check.py --base submissions/episodes_sam_handfix \
  --cand <applied dir> --hands outputs/hands_handfix
```
Needs the hands npz from the `hands-hf` job (**5377177, PENDING**). Also note: moving humans up onto
the floor plane can change human–object penetrations → run pen_check after any human shift too.

## 5. Probe plan (quota 3–4/metric/day)

1. **Probe #1: recovered candidate + object bias (frame mode)** — CD-O 18.0 → ~13–15 expected;
   CD-H 14.31, ACC-H, PEN carried from the old submission; ACC-O bounded above.
   *Fallback:* const-mode parquet (zero ACC-O noise).
2. **Fresh-chain baseline probe** (`submissions/track1_sam3d_20261102_1654.parquet`, packed &
   verified, commit URL stamped; dev-box copy at `/tmp/opencode/`) — calibration: true CD-H/CD-O
   of the fresh chain (expected ~15–17 / ~18.5–19.5, PEN ~0–0.3).
3. **Feet-refine lands** → check `submissions/episodes_sam_depth_fit2/human_refit_report.json`
   (`planted_frames` per episode), foot heights before/after → decide probe #3.

**Transfer lesson (hard-won): geometric/physics corrections transfer to Kaggle; learned correctors don't.**

## 6. Files inventory

**Dev box** (`/home/upatil/nvidia-v2d/v2d-track1-wham-foundationpose`, branch `cluster-rebuild` @ `92c2800`):
- `scripts/depth_icp_objects.py` — along-ray bias corrector (glb/ply fallback, ratio diagnostic)
- `scripts/floor_planes_soles.py` — floor-plane emitter · `scripts/pen_check.py` — PEN gate
- `scripts/rec_float.py`, `scripts/rec_acc.py` — recovered-candidate float + ACC transfer measurements
- `scripts/human_refit.py`, `pipeline/stages/human_refine.sh` — foot-gate args added (defaults unchanged)
- `pipeline/fanout/feet_refine.slurm`, `pipeline/fanout/hands_handfix.slurm`, `prep_probe1.sh` (cluster root)
- `AGENTS.md` deliberately uncommitted; do not commit: containers/, weights/, data/, inputs/, outputs/, submissions/, .venvs/, logs/, third_party/, official/, .cache/

**Cluster** (`/hpcwork/mdk96850/v2d-track1/`, via `ssh claix` / `ssh claix-copy`):
- `submissions/episodes_recovered_cand/` — unpacked BMQ3KGMtp (30 npz + 30 ply, round-trip exact; scored-frame indices verified to match the parquet exactly; T=700 vs fresh 790, but indices align)
- `submissions/episodes_rec_objbias_{frame,const}/` — probe #1 candidates
- `submissions/track1_rec_objbias_<ts>.parquet` — uploadable probe #1 · `submissions/track1_sam3d_20261102_1654.parquet` — uploadable fresh baseline
- `outputs/depth_icp/report_recovered.json`, `report_rec_{frame,const}.json`, `logs/rec_float.json`, `logs/rec_acc.json`
- `outputs/physics/floor_planes_soles.json` (+ report) · `outputs/hands_handfix` (after hands-hf)
- Master runbook (dev box): `/data/formhoi/RUNBOOK.md`

## 7. Operational gotchas

- SLURM: always `-A rwth2243`, c23g partition only, **no `--wrap`**, job dependencies disabled, 24 h cap → stages must be resumable.
- **Venv python segfaults on CLAIX login nodes without `module load Python/3.12.3`.**
- `in_container` uses `--cleanenv` → env vars must pass as **CLI args through stage scripts**, not environment.
- Depth decode: `65535/v − 1` m **÷ alignment depth_scale**. `mhr.run` returns **V in mm but J in metres**.
- SC joint map: head=113, ankles=[4,20], balls=[8,24].
- HF gated downloads: `HF_TOKEN_FILE=$HOME/.hf_token_v2d` (never print/copy/commit tokens).
- Login node has 1 core → pure CPU work belongs on the dev box; heavy stages on c23g GPUs.
- GPU queue is slow (days) — the pending jobs (feet-refine 5376426, hands-hf 5377177) may not land before the deadline; the probe plan above does not depend on them except the PEN gate for *additional* object-bias variants.

## 8. If you only read one section — the decision tree

1. Read BMQ3KGMtp's ACC-O from submission history.
2. ACC-O ≥ 0.2 → dry-run + upload **frame-mode** probe. ACC-O ≲ 0.05 → upload **const-mode** probe.
3. Paste scores back; compare CD-O vs 18.0195 (and CD-H sanity = 14.31 ± tiny).
4. If CD-O improved → fresh baseline probe for calibration; if not → investigate `report_rec_frame.json` per-episode biases before spending more probes.

---

## 9. 2026-11-03 update: scale-artifact finding, rank math, and the curl+rotation plan

### Probe #1 verdict (submitted 00:29 UTC from ujjwalpatil7, all 5 boards)
`track1_rec_objbias_gated_20261103_0122.parquet`: CD-H 14.3168 (= old cand, mechanics ✓) ·
**CD-O 22.017 (+4.0 WORSE)** · ACC-H 0.13693 · ACC-O 0.07616 (= predicted frame-mode injection) ·
**PEN 0.00026 (10× better than team best)**.
**The "bias" was MoGe's per-episode depth-SCALE error, not object displacement**:
bias_cm ≈ (1 − depth_frame_ratio) × z_object on all 9 corrected eps (implied z 1.1–6.3 m, all
plausible). Depth-residual alignment is an INVALID objective under scale error; any future
corrector must subtract `(1−ratio)·z` first. Someone applied the raw depth-ICP to P3 on
H-BRS at 00:01 Nov 3 → CD-O 19.03 — same trap, independently confirmed. **Recovered-cand +
along-ray bias direction is CLOSED.**

### Single-file rule (owner, 2026-11-03)
One file for all 5 boards; overall = lowest SUM of ranks. NOTE: Zaid's handoff says different
files per board are allowed — needs owner confirmation; hedge = composed file works either way.

### Rank table (leaderboards 01:50 UTC Nov 3; lower is better everywhere)
| profile | CD-H | CD-O | ACC-H | ACC-O | PEN | SUM |
|---|---|---|---|---|---|---|
| P4 (Zaid best all-round) | 13.326/r11 | 17.516/r11 | 0.13889/r24 | 0.07064/r15 | 0.00257/r45 | 106 |
| V3 | 13.565/r15 | 17.639/r12 | 0.13624/r15 | 0.07062/r15 | 0.00206/r42 | 99 |
| **composed: P2 human + P4tab101 objects** | 13.561/r15 | 17.084/r7 | 0.13616/r14 | ~0.0713/r16 | ~0.0026/r45 | ~97 |
| **composed + PEN-0 expulsion (target)** | r15 | r11–15 | r14 | r16 | r5 | **~62–65** |
P4tab101 (P4 + table objects ×1.01) = CD-O 17.08369 = r7, team's best. P4tab101Z = PEN 0.0
(objects pushed out of every dense hand vertex; CD-O cost +3.4, ACC-O cost +0.086 — as a
single file: SUM 96, i.e. expulsion only pays if done ACC-neutrally).
PEN board: 21 teams at 0.00000; Izus already banks rank 5 there (chronological tie-break).

### The ACC-free expulsion theorem (verified in kit code)
- ACC-O = second differences of `object_translation` ONLY → **object rotation never enters**.
- ACC-H = 22 Table-3 BODY joints (1,2,18,35,3,19,36,4,20,37,8,24,110,74,38,113,75,39,76,40,78,42)
  → **finger pose params (68:122) move none of them** (verified: 0.0000 mm at ±0.05 rad;
  wrist param moves them 2.07 mm — sanity ✓).
- Therefore: per-frame finger curl (CD-H cost only) + per-frame object rotation (CD-O cost
  only) + translation bit-identical ⇒ ACC-H and ACC-O EXACTLY unchanged; PEN → 0 if all
  frames cleared. Physics: translation-only and rotation-only each fail on wrapped grasps
  (normals span the sphere); curl opens the wrap from the hand side, rotation slides the
  palm side. MHR identity (scales/shape) CANNOT shrink hands (probe: best −0.08 mm radial
  per step) — dead end, do not retry.

### State (04:00 UTC Nov 3)
- Composed dir built: `submissions/episodes_composed_P2hum_P4tab101obj` (P2 pose/scales/shape
  + P4tab101 object arrays; 30/30 T-compatible). Curl smoke tests on ep21/28 running.
- H-BRS: feet-refine 445011 on wr22 finishing; hands_P2 445014 queued behind it (P2-symlink
  bug fixed by real copy). ONE node discipline per owner.
- Cluster lesson: symlink paths inside apptainer `--bind` mounts are a failure class —
  always copy real dirs into the bind.
- Quota: 15 uploads left today (5 used 00:29 UTC), shared with swithinraj.

---

## 10. 2026-11-03 (overnight): local scorer + pseudo-GT built; pre-scored submission plan

### Why
Owner authorized up to 3 autonomous submissions overnight + one 4-GPU node. Building a local
version of Kaggle's scoring pipeline so every upload is pre-screened.

### What was built
- `scripts/localscore.py` — kit-exact local scorer:
  - unpacks submission parquets (row_id = t1_<ep>_<frame>_<array>_<point>) into arrays
  - MHR forward via the reference TorchScript model (float64, mm->m convention verified)
  - pseudo-GT reference = vertex-average of 3 INDEPENDENT human systems
    (P2 = SAM3D-ensemble, MF = WHAM/BMQ3KGM human, C1 = CARI4D-CoCoNet), each per-episode
    Sim(3)-aligned to base on temporal-mean vertices; objects from P4tab102 mapped by its
    own human's transform. Refs in `track1_objbias/outputs/pseudo_gt` (H-BRS).
  - scores via the kit's own `fit_similarity_transform` + `symmetric_chamfer_distance` +
    stretch-wise `acceleration_error_per_frame` (10/30 episodes have frame gaps — stretches
    replicated exactly). cd_c skipped (unscored). All-vertex role substitute for
    roles["alignment"]/["body"] (kit asset not shipped; deltas calibrated anyway).
- `scripts/score_sweep.py` + `locscore_*_hbrs.slurm` — 4-GPU data-parallel sweeps on wr14.
- `scripts/make_variants.py` — stretch-wise zero-phase smoothing / alpha / frame-0 variants.
- `scripts/kaggle_known_scores.json` — ~30 (file -> exact Kaggle scores) calibration pairs.
- Gotchas hit: apptainer `--cleanenv` strips CUDA_VISIBLE_DEVICES (use `--env`); container
  lacks pandas/pyarrow (stub pandas module — kit math never uses it); 15.77 GiB GPUs →
  1 process/GPU; unpack-empty-dir bug during conversion (assert added).

### Known absolute-score caveats
Local absolutes differ from Kaggle (reference built from our own chains, all-vertex roles).
Only calibrated DELTAS (Kaggle_delta ≈ c * local_delta + b, fitted per board on ~30 pairs)
are used for decisions.

### Submission candidates in flight
- 52-candidate base sweep (job 445021): all packed files incl. UNSCORED drawer
  (WFk55, P4tab103, P5, P4C*, P2A*/P2L*/P2S60, P4Ttab101, P4L300tab101, P4tab101flo101).
- Variant sweep (job queued next): P2sm2/5/10/20/5r (ACC-H), tab102sm2/5/10/20/40 (ACC-O),
  P2f0 + P4hum_tab102f0 (frame-0 alignment denoise for CD boards), P4hum_tab102 (CD-O file).
- After both: calibrate, pick top-3 by predicted Kaggle delta vs thresholds
  (acc-h r13 < 0.13571, acc-o r6 < 0.06970, cd-h r10 < 13.26540), pack, submit (max 3).

### 10.1 WFk60 recipe FOUND and reproduced bit-exactly (morning, 2026-11-03)

- Real recipe: `vpq-vitpose/scripts/accel_shrink.py` (H-BRS, NOT the hoop-only
  `smooth_symmetric_objects.py`): objects-only accel shrink, **all 30 episodes**, k=0.6,
  lam=1e4 Whittaker (2nd-difference penalized smooth), per **run of consecutive video-frame
  ids** (gaps respected), residual scaled x_smooth + k(x - x_smooth), first scored frame
  pinned (alignment anchor). Overnight probes were a DIFFERENT operator (Gaussian, all rows
  as one stretch) — that's why the Kaggle k-response disagreed.
- `scripts/accel_shrink_zaid.py` reproduces track1_cand_WFk60.parquet from
  track1_cand_WF.parquet with max |diff| 1.2e-7 m (float32 noise). VERIFIED.
- Kaggle-known points on HIS curve: k=1 -> 0.07051 (WF), k=0.6 -> 0.06971 (WFk60, counted,
  r7). r6 needs < 0.06970 (only -1e-5).
- Packed ladder (translation rows only patched, everything else bit-identical, commit
  6444236): `submissions/track1_WFz{0.5,0.65,0.7,0.75,0.8}.parquet` (H-BRS + /tmp/opencode).
  Linear interpolation predicts ~0.0693 at k=0.7; 0.65 is the minimal-change candidate.
- PROPOSAL for owner (needs approval, 0 autonomous uploads left): submit WFz0.65 first
  (safest flip of r7->r6), then WFz0.5 (bigger gain if curve keeps descending) or WFz0.8
  (if 0.65 disappoints). Board: v2d-challenge-track1-acc-o. ~2 uploads/day remain today.
