# V2D Challenge, Track 1: day-two report (day 0 → 2026-10-16, last probe 18:29 UTC)

Repository: https://github.com/Swithinraj/v2d-track1-wham-foundationpose (public; `main` = the WHAM pipeline,
last commit `cbb430a`). This report is the first commit on the branch `SAMDiNo`; the day-two
experiments ran in a separate local folder (not part of the repository, one directory and manifest per run). This report covers everything from the
first pipeline build: what each stage does, every technique tried, how each one moved each metric, what failed
and why. All scores are public Kaggle scores; lower is better on every board. Times are UTC.

---

## 0. Summary

- **Best complete file so far** (one Parquet on all five boards): the committed pipeline `cbb430a`,
  CD-H 20.95822 · CD-O 29.36947 · ACC-H 0.14237 · ACC-O 0.07700 · PEN 0.00822. Against the first submission
  that is −9.4 % CD-H, −6.0 % CD-O, −35.6 % ACC-H, −55.5 % ACC-O, −22.7 % PEN.
- **Day two's big result:** SAM 3D Body (Meta's gated model, access granted on day two) replaces WHAM as the
  human for all 30 episodes. Probe: **CD-O 25.22105 (−4.15)** at unchanged ACC-O (+0.0007). On six pilot
  episodes alone it gave CD-H −1.97 and CD-O −2.41. It fixes WHAM's wrong anchor-frame bodies (episodes 24, 8,
  6), halves the 2D error and removes all front/back flips.
- **Its cost is PEN** (0.00822 → 0.01290). The repository's penetration resolver on top brings PEN to
  **0.00575**, the best PEN so far; its ACC-O cost is still to be measured.
- **Object repairs failed on Kaggle.** A pan with a handle and a round foam roll fit the masks better but scored
  CD-O +0.68 and ACC-O +0.0014; not adopted.
- **Leaderboard** (25 teams, best per board): CD-H 21st, CD-O 11th, ACC-H 4th, ACC-O 7th, PEN 19th (§9).

---

## 1. The challenge

### 1.1 Task

NVIDIA's Video-to-Data (V2D) challenge, Track 1 ("4D human and object reconstruction"): from one static
third-person video of a person handling an object, reconstruct over time

- the **human** as Meta MHR body-model parameters: `pose [T,136]` per frame, `scales [68]` and `shape [45]`
  once per episode;
- the **object** as a triangle mesh with a per-frame rotation and translation and one positive scale.

### 1.2 Data

- 30 episodes (16,563 frames, 30 fps, 1536 × 1152 px), one static exo camera each.
- 10 objects, 3 episodes each: hula hoop (0–2), big red bowl (3–5), iron (6–8), white desk (9–11), black pan
  (12–14), foam grass block (15–17), paint roller (18–20), pink foam roll (21–23), short wood stool (24–26),
  white laptop cart (27–29).
- A text prompt per episode. **No object meshes, camera calibration, depth or ground truth are released**; the
  references are only scored on Kaggle. 9,877 of the 16,563 frames are scored (listed in the sample file).

### 1.3 Scoring

One Parquet (740,780 rows) carries all 30 episodes and is submitted to five separate Kaggle competitions,
5 submissions per board per UTC day, each with a `code_commit_url`. The scorer fits **one similarity transform
(Sim(3)) per episode, from our human to the reference human at the first scored frame**, applies it to the whole
episode, then measures:

| Metric (Kaggle board) | Measures | Unit |
|---|---|---|
| [CD-H](https://www.kaggle.com/competitions/v2d-challenge-track1-cd-h/leaderboard) | symmetric Chamfer distance of the human surface | cm |
| [CD-O](https://www.kaggle.com/competitions/v2d-challenge-track1-cd-o/leaderboard) | symmetric Chamfer distance of the posed object surface (sampled on the mesh) | cm |
| [ACC-H](https://www.kaggle.com/competitions/v2d-challenge-track1-acc-h/leaderboard) | error of the joint accelerations (22 joints) | cm/frame² |
| [ACC-O](https://www.kaggle.com/competitions/v2d-challenge-track1-acc-o/leaderboard) | error of the object-translation acceleration | cm/frame² |
| [PEN](https://www.kaggle.com/competitions/v2d-challenge-track1-pen/leaderboard) | mean depth of our hand points inside our own object mesh | cm |

Consequences (proven with the official metric code on day one): global per-episode errors are free; what costs
is error that changes over time and any error **in the first scored frame**, because that frame anchors the
human *and the object* for the whole episode. PEN uses the reference only for the alignment scale, so it can be
measured locally (§7.5).

### 1.4 Constraints

Until day two the two Meta checkpoints of the official CARI4D baseline (`facebook/sam-3d-body-dinov3`,
`facebook/sam-3d-objects`) were not approved, so days 0–1 used only ungated models ("SAM-free" pipeline).
Access was granted on day two. Every model runs in Apptainer images on A100 GPUs; the repository is portable and
contains no scheduler or site paths.

---

## 2. Timeline

| When | Milestone |
|---|---|
| Day 0 | SAM-free pipeline built: MoGe-2 depth, GroundingDINO + SAM 2 masks, WHAM → MHR, TRELLIS meshes, FoundationPose tracking, joint alignment, assembly. Hoop tracker `9230b27`; actor-track fix `2c1e8bd` |
| 10-15 13:12 | First submission (`2c1e8bd`) |
| 10-15 17:18–17:43 | Round 2: scorer analysis, human depth-trend correction → committed `9624185` |
| 10-15 18:42 | Round 3: hula-hoop centre smoothing → committed `5614cc4` |
| 10-15 22:32 – 10-16 00:01 | Round 4: human motion smoothing, object-anchored hand IK, MediaPipe + Kalman/RTS fusion |
| 10-16 | Round 4 merged as the human refinement stage → `cbb430a` (+ day-one report); repacked and uploaded 13:35 as the reference |
| 10-16 | SAM 3D Body / SAM 3D Objects access granted; branch `SAMDiNo`, round 5 (SAM 3D Body, CARI4D, audits, object repairs) |
| 10-16 14:37 | SAM 3D Body six-episode pilot probe |
| 10-16 18:25–18:29 | Full SAM 3D Body build, PEN resolver and object-repair probes |

---

## 3. The pipeline as of now

```text
                ┌─ main (cbb430a): WHAM ─ actor tracks ─ SMPL→MHR ─ PnP alignment + drift ─┐
video ──────────┤                                                                              ├─ assembly ─ depth trend ─ hoop smoothing ─ human refinement ─ pack
                └─ SAMDiNo: SAM 3D Body (per frame) ─ shared identity + refit ─ camera world ─┘
                   GroundingDINO + SAM 2 masks · MoGe-2 depth · TRELLIS meshes · FoundationPose tracks (shared by both)
```

| Stage | What it does | Expected / measured |
|---|---|---|
| Depth + intrinsics (MoGe-2) | metric depth and pinhole intrinsics per frame | MoGe scale wobble ≈ 1 % on static background |
| Object masks (GroundingDINO + SAM 2.1) | text detections seed SAM 2 propagation; reviewed overrides in `config/manual_object_prompts.json` | a mask per frame for all 30 episodes |
| Human, main (WHAM → MHR) | world SMPL motion, mapped to MHR with one identity per episode; reviewed actor tracks | fit error ≈ 4.5 mm |
| Human, SAMDiNo (SAM 3D Body) | full-resolution frames, person box from the actor's ViTPose keypoints, one intrinsics estimate per episode, body + hand decoders; raw outputs saved | export rebuilt by the kit's MHR wrapper to ≤ 0.0009 mm (§7.1) |
| Object meshes (TRELLIS + FoundationPose selection) | TRELLIS candidates from the cleanest views; FoundationPose fit in all three cameras within 0.75–1.33× of the MoGe size | one metric mesh per object (estimates) |
| Alignment, main | PnP of WHAM joints to ViTPose, low-frequency drift, depth scale MoGe/SMPL | 5.4–17.3 px |
| Camera world, SAMDiNo | the static camera is the world; objects = FoundationPose camera tracks ÷ MoGe/SAM depth ratio (0.84–0.96) | nothing taken from WHAM's alignment |
| Object tracking | FoundationPose + IoU-weighted ESKF/RTS; hula hoop: symmetric tracker + torus | |
| Depth trend | human root follows the trend of MoGe/body depth (α 0.75, σ 30) | median correction 0.5–2.5 cm |
| Hoop smoothing | λ 100 acceleration penalty on the ring centre | |
| Human refinement | mesh smoothing refit (λ), object-anchored hands, RTS wrist fusion, RTS lateral root | |
| Pack + upload | official packer → 740,780 rows; kit uploader with receipts | |

---

## 4. Every Kaggle submission

All probes are controlled: one named change against the stated base, the rest verified identical by diffing the
Parquet rows, sent only to the boards the change can affect. Probes carry their base commit's URL and are
labelled "probe" in the Kaggle message.

| # | UTC | File / change | CD-H | CD-O | ACC-H | ACC-O | PEN |
|---|---|---|---|---|---|---|---|
| 1 | 10-15 13:12 | **First submission** (`2c1e8bd`) | 23.13845 | 31.23021 | 0.22090 | 0.17288 | 0.01064 |
| 2 | 17:18 | H1 depth follows MoGe (σ 10, α 1) | 21.70005 | | | | |
| 3 | 17:19 | **H2t trend-only depth correction (σ 30, α 0.75)** | **21.48384** | **29.91374** | 0.22081 | 0.17288 | **0.00907** |
| 4 | 17:20 | H3s anchor-frame orientation fix (eps 10, 11, 24) | 23.02980 | | | | |
| 5 | 17:21 | H4 H2t gated by keypoint agreement + H3s | 21.95084 | | | | |
| 6 | 18:42 | Hoop smoothing λ 10 | | 29.90568 | | 0.08679 | |
| 7 | 18:42 | **Hoop smoothing λ 100** | | **29.89199** | | **0.07693** | |
| 8 | 22:32 | Human motion smoothing λ 10 | | | 0.15290 | | |
| 9 | 22:33 | Human motion smoothing λ 30 | | | 0.14722 | | |
| 10 | 22:34 | Human motion smoothing λ 100 (CD-H at 00:01) | 21.45583 | | 0.14327 | | |
| 11 | 22:35 | `fuse10cs_lat` (λ 10 + anchored IK + wrist RTS + lateral root) | 21.02400 | 29.23757 | 0.14897 | 0.07699 | 0.00831 |
| 12 | 10-16 00:00 | Human motion smoothing λ 300 | | | **0.14171** | | |
| 13 | 00:01 | **`fuse100cs_lat` (λ 100 variant), all five** | **20.95822** | 29.36947 | 0.14235 | 0.07700 | **0.00822** |
| 14 | 13:35 | **Committed `cbb430a`, repacked from the repository (reference)** | 20.95822 | 29.36947 | 0.14237 | 0.07700 | 0.00822 |
| 15 | 14:37 | SAM 3D Body pilot V2d on episodes 2, 8, 12, 21, 24, 27 | **18.99071** | 26.95491 | 0.14184 | 0.07719 | 0.01023 |
| 16 | 18:25 | **SAM 3D Body V3 λ 300, all 30 episodes** | | **25.22105** | | 0.07770 | 0.01290 |
| 17 | 18:27 | V3 + repository PEN resolver | | | | | **0.00575** |
| 18 | 18:29 | Object repairs on `cbb430a` (pan with handle 12–14, cylinder 21–23) | | 30.04936 | | 0.07839 | |

(The day-one report listed row 1 in local time, 15:12; Kaggle records 13:12 UTC.)

---

## 5. Effect of each technique on each metric

Kaggle deltas against the base of each probe. "Exact 0" means the scorer cannot see the change (proven).

| Technique (day) | CD-H | CD-O | ACC-H | ACC-O | PEN | Decision |
|---|---|---|---|---|---|---|
| Symmetric hoop tracker + torus (0) | – | – | – | – | – | kept `9230b27` (proxy: hoop IoU up to 3.5×) |
| Actor-track fix, episodes 2 and 15 (0) | – | – | – | – | – | kept `2c1e8bd` (ep 2 human 6.76 → 3.40 m) |
| H1 depth follows MoGe (1) | −1.438 | | | | | superseded by H2t |
| **H2t trend-only depth correction (1)** | **−1.655** | **−1.316** | −0.0001 | exact 0 | **−0.0016** | kept `9624185` |
| H3s anchor-frame orientation fix (1) | −0.109 | | | | | too small; superseded |
| H4 keypoint-gated depth correction (1) | −1.188 | | | | | rejected: gating hurts |
| Hoop smoothing λ 10 (1) | | −0.008 | | −0.0861 | | superseded by λ 100 |
| **Hoop smoothing λ 100 (1)** | exact 0 | −0.022 | exact 0 | **−0.0960 (−55 %)** | | kept `5614cc4` |
| Human motion smoothing λ 10 / 30 / 100 / 300 (1) | λ 100: −0.028 | | −0.068 / −0.074 / −0.078 / −0.079 | | | gains flatten above λ 100 |
| `fuse10cs_lat` (1) | −0.460 | −0.654 | −0.072 | +0.0001 | −0.0008 | |
| **`fuse100cs_lat` = human refinement (1)** | **−0.526** | **−0.523** | **−0.0785** | ≈ 0 | **−0.0009** | kept `cbb430a` |
| Repository rerun of `cbb430a` (2) | 0 | 0 | +0.00002 | 0 | 0 | reference (GPU noise only) |
| **SAM 3D Body human, 6 pilot episodes (2)** | **−1.968** | **−2.415** | −0.0005 | +0.0002 | +0.0020 | promising → full build |
| **SAM 3D Body V3, all 30 episodes (2)** | pending | **−4.148** | pending | +0.0007 | +0.0047 | candidate |
| **V3 + PEN resolver (2)** | | | | pending | **−0.0025** (−0.0072 vs V3) | candidate if ACC-O holds |
| Object repairs: parametric pan + cylinder (2) | | **+0.680** | | +0.0014 | (local +0.007) | **rejected** |

---

## 6. Days 0–1: techniques, results, failures (condensed from the day-one report)

- **Hula hoop.** FoundationPose cannot track a thin, textureless, symmetric ring (72–126 jumps > 10 cm per
  episode, 40–90° axis flips). A symmetry-aware tracker (centre + axis only, mask ellipses, particle filter,
  hand-contact prior) with an exact torus raised silhouette IoU from 0.14–0.46 to 0.49–0.60, but its centre was
  jerky (2.7 cm/frame² in episode 1, half of our ACC-O). The centre smoothing (λ 100) removed that: ACC-O −55 %.
- **Actor selection.** WHAM's longest track was a seated bystander in episode 2 and a split track in episode 15.
  DINOv3 wrist/object covisibility identified the right fragments; reviewed overrides are stitched.
- **Scorer analysis (round 2).** A global Sim(3) or constant offset changes every metric by exactly 0; a 0 → 20 cm
  depth drift costs 11.7 cm CD-H and 7.0 cm CD-O; a 5° error in the first scored frame alone costs 7.5 cm CD-H.
  Global grounding was dropped; time-varying error and the anchor frame became the focus.
- **Depth trend (round 2).** WHAM drifts along the camera ray; the trend of MoGe's background-normalised depth
  ratio is closer to the truth. Trend-only (σ 30, α 0.75) beat a per-frame version (σ 10, α 1). **Failure:**
  gating it by 2D keypoint agreement hurt, so 2D reprojection is not a CD-H proxy.
- **PEN push v1/v2 (round 2).** Pushing the object out of the hand every frame cut local PEN by 93 % but moved
  objects up to 34 cm (+26 % ACC-O worst case). An acceleration-penalised v2 halved PEN at ≤ +1 % ACC-O; not
  probed on day one.
- **Kinematic coupling (round 3).** **Failure:** locking the object to the hand would inject noise; every rigid
  object carried by the MHR hand is 5–30× jerkier than its FoundationPose track. Only the hoop benefited from
  smoothing, and plain smoothing beat hand coupling.
- **Rigid multi-view meshes (TSDF/Poisson of MoGe depth).** **Failure:** pan and iron came out as noisy blobs
  without a handle; monocular depth on small dark objects is too noisy.
- **DINOv3 finger refinement.** **Failure:** contact 1–2 mm better, PEN already ≈ 0 on the hoop, no metric change.
- **Human refinement (round 4).** WHAM's own joints are jittery (0.185 cm/frame² on the scored joints). Motion
  smoothing on the mesh + per-frame refit (MHR Euler angles wrap, so smoothing is done on vertices) took ACC-H
  from 0.2208 to 0.1417 (λ 300). The CD-H gain came from 2D evidence and contact: hands anchored to the
  FoundationPose object in 52 stable grasps, Kalman/RTS wrist fusion with MediaPipe (2D, low weight) and object
  anchors (3D), and an RTS lateral root fix where ViTPose agrees. **Failures fixed on the way:** 2D-only
  observations leaked into depth (30–117 cm moves), an unguarded wrist push went into the cart (episode 27).
- **Episode 24** was found to be a WHAM front/back flip (55 of 217 checked frames); no fix on day one.

---

## 7. Day 2: techniques, results, failures

### 7.1 SAM 3D Body as the human

- **Setup.** Model pinned at revision `11aaa346`; its MHR asset is byte-identical to the kit's MHR model
  (sha256 `352e271a…`), so topology and body model match the scorer. Apptainer recipe and download script are
  prepared for `SAMDiNo` (not committed yet); the pilot ran in the CARI4D image, which ships the same library.
- **Prompting.** One box per frame from the selected actor's ViTPose keypoints (the same person the reference
  uses), full-resolution frames, one static intrinsics estimate per episode, body + hand decoders.
  **No gain:** SAM 2 person masks as extra prompts (ViTPose 5.16 vs 5.18 px); boxes kept.
- **Export acceptance test** (no fitted transforms): pose = 136 MHR parameters with the root translation set
  once from the camera translation, scales = the 68 assembled MHR scales (not the 28 predicted coefficients),
  shape = 45. The kit's wrapper rebuilds SAM's own mesh to ≤ 0.0009 mm, hands included; projected 3D keypoints
  match SAM's 2D keypoints to 4e-5 px.
- **Shared identity.** SAM predicts a new body shape every frame (shape std 0.11–0.23). V1 takes the medoid
  identity over reliable frames and refits every frame with it fixed: swapping the identity moves the mesh by
  2.3–12.9 mm, the refit brings it back to 0.5–2.0 mm.
- **Camera world.** The static camera is the world; objects are the reference's FoundationPose camera tracks
  divided by a fresh MoGe/SAM depth ratio (0.84–0.96). Nothing is taken from WHAM's alignment; WHAM-tuned
  settings were only starting values.
- **Variants (six pilot episodes, scored frames, against held-out ViTPose and MediaPipe):**

| Variant | ViTPose px | MediaPipe wrist px | Front/back flips | Jitter cm/frame² |
|---|---|---|---|---|
| WHAM (committed, refined) | 13.7 | 15.2 | 47 (all ep 24) | 0.056 |
| SAM 3D Body raw | 5.2 | 6.5 | 0 | 2.87 |
| V1 shared identity + camera world | 5.3 | 6.6 | 0 | 2.87 |
| V2 V1 + mesh smoothing λ 300 | 5.6 | 6.7 | 0 | 0.070 |

  Raw SAM 3D Body is very jittery per frame (2.87 cm/frame²); λ 300 smoothing (higher than WHAM's λ 100)
  brings it to WHAM's level at little 2D cost. V2d adds the reference's depth-trend correction.
- **Pilot probe (row 15):** CD-H −1.97, CD-O −2.41, ACC-H −0.0005, ACC-O +0.0002, PEN +0.0020, from six
  episodes only. In episode 24 the flips disappear (ViTPose 28.9 → 6.3 px).
- **Full build, all 30 episodes** (local checks; self-PEN validated in §7.5):

| Build | self-PEN | ViTPose px | Wrist px | Jitter cm/frame² | Flips |
|---|---|---|---|---|---|
| Reference `cbb430a` | 0.00863 | 10.48 | 9.84 | 0.054 | 47 |
| V2d λ 300 | 0.01622 | 5.11 | 5.89 | 0.056 | 0 |
| V3 λ 300 (V2d + the committed refinement: anchored hands, wrist RTS, lateral root) | 0.01540 | 5.29 | 5.74 | 0.045 | 0 |

  V3 was probed (row 16): **CD-O −4.15**, ACC-O +0.0007, PEN +0.0047. The six-episode pilot gave −2.41 CD-O;
  the full build's larger gain fits the anchor analysis (§7.3): a correct body in the first scored frame puts
  the whole episode, object included, in place.

### 7.2 Full CARI4D baseline (episodes 24, 8, 27)

- Ran the official eight-stage pipeline (depth, export, SAM 3D Body init, depth alignment, FoundationPose,
  CoCoNet, contact refinement, render) with our meshes; outputs decoded with its own MHR head.
- **Failures fixed:** the video must be a real `<seq>.0.color.mp4` file (symlinks are resolved); two assets in
  its checkout and image were Git-LFS pointers (fetched at the pinned commit, hash-checked); SAM 2 person and
  object masks overlapped in 222–441 frames per episode (made exclusive); a model-download race in episode 27.
  **Unresolved:** episode 8, where CARI4D's FoundationPose loses the small iron (all poses render invisible).
- **Stage comparison** (episodes 24 / 27): init ViTPose 5.8 / 5.8 px; CoCoNet 6.6 / 7.0; contact refinement
  6.6 / 7.4 px; jitter 0.71–2.72 cm/frame² for every stage. **Result:** CoCoNet and the refinement cost 2D
  agreement and stay far too jittery; the init stage is no better than our own SAM 3D Body prompting. A
  combination file (CARI4D refined + λ 300 on 24/27) is packed but not probed.

### 7.3 First-scored-frame audit (all 30 episodes)

The anchor body sets the frame of the whole episode, so heading, walking direction and left/right labels were
compared at each episode's first scored frame.

- **WHAM is wrong at the anchor in three episodes; SAM 3D Body fixes all three:**
  - **24** (walking in profile): WHAM's legs cross towards the stool, 34 px (32 px if left/right swapped),
    44° off the walking direction; SAM 3D Body 4 px, 8° off; headings differ by 36°.
  - **8** (kneeling at the iron): WHAM throws a leg sideways, 40 px; SAM 3D Body 5 px; 31° apart.
  - **6:** WHAM 15 px, 25° from SAM 3D Body (4 px).
- The other 27 episodes agree within 15° (median 5°); SAM 3D Body is 2–8 px at the anchor, WHAM 5–19 px.

### 7.4 Objects: stage chain, repairs, Kaggle result

- **Visual stage chain for every object** (prompt → mask → condition crop → raw TRELLIS → cleaned → installed →
  posed render with the track, silhouette IoU against the mask). Median posed IoU: bowl 0.82–0.85, grass
  block 0.81–0.89, desk 0.71–0.77, foam roll 0.75–0.82, roller 0.75–0.81, stool 0.59–0.79, cart 0.59–0.65,
  pan 0.54–0.73, iron 0.49–0.64.
- **Black pan, where the handle was lost:** (1) GroundingDINO's box for "a black frying pan" covers the bowl
  only and SAM 2 propagates that box; fixed with reviewed point prompts (positives on bowl and handle, negatives
  on table and tripod leg), the masks now reach the handle tips while the pan rests. (2) The automatic mesh-view
  selection (a solidity score) then picks frames where the handle is hidden; fixed with manual views.
  (3) **Failure:** TRELLIS then draws the handle far too long, so at the measured size the bowl is ~9 cm instead
  of ~20 cm (selection IoU 0.10–0.16). A parametric pan (revolved 0.20 × 0.045 m bowl + 0.17 m handle) won the
  unchanged selection (IoU 0.833 vs 0.695 for the old disc) and tracked better (posed IoU 0.590 → 0.638).
- **Pink foam roll:** masks were clean; the selected TRELLIS mesh is a flat bar (0.587 × 0.108 × 0.038 m). A
  parametric cylinder (d 0.10 m) won the selection (0.922 vs 0.917) and tracked better (0.768 → 0.802).
- **Iron:** the winning TRELLIS mesh is slim (7 cm) and the IoU-driven scale search inflated it to 34.7 cm
  (two of three episodes at the 1.33× limit). **Failures:** the better-proportioned candidate is a wedge, not an
  iron; the uninflated mesh tracks worse (posed IoU 0.578 → 0.534, episode 8 breaks). Kept as installed.
- **Kaggle (row 18), object repairs on the reference: CD-O +0.68, ACC-O +0.0014 → rejected.** Better silhouette
  fits did not mean closer geometry. Likely causes: the parametric pan may be too long (35 cm); its new tracks
  are jumpier (object acceleration up to 2× in episode 13); the round roll sits inside the gripping hand in
  episode 22 (local self-PEN 0.0026 → 0.215 cm there). One probe cannot separate pan from roll.
- **Packing failure fixed:** the kit simplifies every mesh to 4,096 faces/vertices and stalled on the
  subdivided cylinder (145k thin faces); the same 96-section cylinder without subdivision (384 faces, identical
  surface to 5e-9 m) packs fine. CD-O samples the surface, so nothing changes.

### 7.5 Penetration (PEN)

- **Local PEN check validated.** The scorer's PEN with alignment scale 1 (512 hand points in the object frame,
  winding-number depth in the budgeted mesh, scored frames) reproduces Kaggle: reference 0.00863 vs 0.00822,
  pilot 0.01031 vs 0.01023, V3 0.01540 vs 0.01290 (right direction and order).
- **Where V3's PEN comes from:** hands resting on large objects, cart 29 / 28 / 27 (+0.083 / +0.058 /
  +0.025 cm), iron 8 (+0.026), desk 9 (+0.018). In the camera world the object is the track divided by the
  MoGe/SAM depth ratio; a small ratio error moves it against the hands.
- **Repository PEN resolver** (pushes the object out of the hands, only in episodes above 0.05 cm, dilated and
  smoothed offsets): local PEN 0.01540 → 0.00664, **Kaggle 0.00575** (row 17, best so far). Its cost: the
  object-acceleration proxy rises 27 % (0.0445 → 0.0573 cm/frame²), so ACC-O must be checked on Kaggle.

### 7.6 Mask agreement from colour (HSV, support signal only)

- A hue-saturation model of each object (mask vs surrounding ring, person pixels excluded) scores every SAM 2
  mask's precision and recall.
- **Failure of the first version:** 55 false flags on the grass block, because the unseen grass face read as
  background and skin matched the brown dirt. Fixed by learning from eight frames and excluding the person.
- **Result:** bowl, grass block, foam roll and stool masks are clean (precision 0.85–0.96, recall 0.95–1.00).
  Black, white and grey objects carry no colour evidence. The hoop gives false low recall (the curtain matches
  its colours). Conclusion: the remaining object problems are geometry and tracking, not masks.

### 7.7 Infrastructure fixes (day two)

Read-only torch hub caches (writable `trusted_list` bind), CARI4D's symlink and Git-LFS issues (§7.2), Track 1
row-id parsing of the scored frames (frame 999999 is a per-episode row), quaternion order of the FoundationPose
records (wxyz, as the assembler reads them), the kit's mesh budget (§7.4). Every run writes a manifest with input
hashes, model revisions, settings and upstream runs, so stale outputs are detectable.

---

## 8. What failed, and why (all days)

| Attempt | Why it failed |
|---|---|
| Keypoint-gated depth correction (H4) | 2D reprojection is not a CD-H proxy; excluding episodes removed real gains |
| Anchor orientation fix alone (H3s) | too small a share of the anchor error; the real error was WHAM's body (fixed by SAM 3D Body) |
| PEN push v1 | huge object offsets and acceleration |
| Hand–object kinematic coupling | the MHR hand is 5–30× jerkier than the FoundationPose object |
| TSDF/Poisson object meshes | monocular depth too noisy on small dark objects |
| DINOv3 finger refinement | no measurable effect |
| SAM 3D Body mask prompts | no 2D gain over keypoint boxes |
| CARI4D CoCoNet / contact refinement | worse 2D than its own init, jittery; episode 8 crashes |
| TRELLIS pan with handle | handle drawn too long, bowl too small at the measured size |
| Iron candidates (c03d wedge, uninflated c00d) | wrong shape / worse tracking |
| Parametric pan + cylinder on Kaggle | CD-O +0.68, ACC-O +0.0014 despite better silhouettes |
| HSV agreement, first version | object faces and skin colours confused the colour model |

---

## 9. Where we stand (live leaderboard, 2026-10-16 18:45 UTC, 25 teams)

Best score per board (these come from different files; the final entry must be one file):

| Board (leaderboard link) | Our best | From | Rank | Leader | 10th |
|---|---|---|---|---|---|
| [CD-H](https://www.kaggle.com/competitions/v2d-challenge-track1-cd-h/leaderboard) | 18.99071 | SAM 3D Body pilot (row 15) | 21 | 13.43306 | 15.12774 |
| [CD-O](https://www.kaggle.com/competitions/v2d-challenge-track1-cd-o/leaderboard) | 25.22105 | SAM 3D Body V3 (row 16) | 11 | 17.91571 | 25.03212 |
| [ACC-H](https://www.kaggle.com/competitions/v2d-challenge-track1-acc-h/leaderboard) | 0.14171 | smoothing λ 300 (row 12) | 4 | 0.13738 | 0.15089 |
| [ACC-O](https://www.kaggle.com/competitions/v2d-challenge-track1-acc-o/leaderboard) | 0.07693 | hoop smoothing λ 100 (row 7) | 7 | 0.07111 | 0.12236 |
| [PEN](https://www.kaggle.com/competitions/v2d-challenge-track1-pen/leaderboard) | 0.00575 | V3 + PEN resolver (row 17) | 19 | 0.00000 | 0.00149 |

| Complete file | CD-H | CD-O | ACC-H | ACC-O | PEN |
|---|---|---|---|---|---|
| First submission `2c1e8bd` | 23.138 | 31.230 | 0.2209 | 0.1729 | 0.01064 |
| `cbb430a` (reference, best complete file) | 20.958 | 29.369 | 0.1424 | 0.0770 | 0.00822 |
| Change | −9.4 % | −6.0 % | −35.6 % | −55.5 % | −22.7 % |

CD-H is the weakest board (the full SAM 3D Body build has not been scored there yet). PEN is a near-zero
board: half the teams are at or below 0.0021.

---

## 10. Lessons (updated)

1. **The anchor frame is everything.** The biggest single gain (CD-O −4.15) came from getting the human right in
   the first scored frame, which places the object too. WHAM was wrong there in episodes 24, 8 and 6.
2. **Proxies mislead; controlled probes do not.** 2D keypoints did not predict CD-H (day one); silhouette IoU
   did not predict CD-O for the repaired objects (day two). Local self-PEN is the one proxy that tracked Kaggle.
3. **Smooth what is noise, keep what is evidence.** λ 300 smoothing is needed for per-frame SAM 3D Body as much
   as λ 100 was for WHAM; 2D evidence and contacts carry the CD gains.
4. **Every fix needs a guard against its own failure mode:** depth leaks, pushes into objects, wedge-shaped
   "best" meshes, objects that fit the mask better but sit inside the hand.

## 11. Next steps

1. Next quota: CD-H and ACC-H for V3 and V3 + resolver; ACC-O for V3 + resolver. If the resolver's ACC-O cost
   is small, V3 + resolver becomes the single file for all five boards.
2. Commit the SAM 3D Body branch as pipeline stages (`pipeline/stages/sam3d_body.sh`, run and export scripts)
   and reproduce the probe file from the repository before the clean submission.
3. Lower PEN without object acceleration: the acceleration-penalised push (v2) on V3, and a better MoGe/SAM depth
   ratio on the contact episodes (cart, iron, desk).
4. Objects: probe pan and foam roll separately; fix the episode-22 roll/hand penetration; a pan with a shorter
   handle; the iron's size needs a direct CD-O comparison.
5. Hands: compare SAM 3D Body's hand decoder with the current fused hands; then the final temporal tuning.
