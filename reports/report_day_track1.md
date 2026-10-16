# V2D Challenge, Track 1: day report (2026-10-15, last probes 2026-10-16 00:00 UTC)

Repository: https://github.com/Swithinraj/v2d-track1-wham-foundationpose (public). This report is part of
the commit that adds the human refinement stage (§2.11). Experiments ran in separate local folders (not
part of the repository) against a read-only copy of the pipeline; only proven changes were merged.
All scores below are public Kaggle scores; lower is better on every board.

---

## 1. The challenge

### 1.1 Task

NVIDIA's Video-to-Data (V2D) challenge, Track 1 ("4D human and object reconstruction"): from a single,
static third-person video of a person manipulating an object, reconstruct over time

- the **human** as Meta MHR body-model parameters: `pose [T,136]` per frame, plus `scales [68]` and
  `shape [45]` once per episode;
- the **object** as a triangle mesh plus its per-frame rotation, translation and one positive scale.

### 1.2 Data

- 30 episodes (16,563 frames at 30 fps, 1536 × 1152 px), each filmed by one static exo camera.
- 10 objects, each in 3 episodes: hula hoop (0–2), big red bowl (3–5), iron (6–8), white desk (9–11),
  black pan (12–14), foam grass block (15–17), paint roller (18–20), pink foam roll (21–23), short wood
  stool (24–26), white laptop cart (27–29).
- Per episode a text prompt for the object (e.g. "A red plastic bowl."). **No object meshes, no camera
  calibration, no depth and no ground truth are released.** The reference poses and meshes are held
  out and only scored on Kaggle.
- Only a subset of frames is scored: 9,877 of the 16,563 (listed in the public sample submission).

### 1.3 How it is scored

One Parquet file (740,780 rows) carries the MHR parameters, object poses, scale and mesh for all 30
episodes. The same file is submitted to five separate Kaggle competitions, one per metric. Each board
allows 5 submissions per UTC day. Every submission must carry a `code_commit_url` pointing at the GitHub
commit that produced it.

The scorer fits **one similarity transform (Sim(3)) per episode, from our human's vertices to the
reference human's vertices in the first scored frame**, and applies it to everything in that episode
(human, joints, object surface, object translation). Then:

| Metric | Measures | Unit |
|---|---|---|
| CD-H | symmetric Chamfer distance, our human surface vs reference | cm |
| CD-O | symmetric Chamfer distance, our posed object surface vs reference | cm |
| ACC-H | error of the second frame difference (acceleration) of 22 body joints | cm/frame² |
| ACC-O | error of the second frame difference of the object translation | cm/frame² |
| PEN | mean depth of our hand points inside our own object mesh | cm |

Two consequences shape the whole project, both verified today (§4.5):

1. Any **global** per-episode error (wrong scale, constant offset, constant rotation) costs nothing:
   the Sim(3) removes it.
2. What costs is error that **changes over time** (drift, jitter) and any error in the **first scored
   frame**, because that frame anchors the whole episode, objects included.

### 1.4 Constraints

The official CARI4D baseline needs two gated Meta checkpoints (`facebook/sam-3d-body-dinov3`,
`facebook/sam-3d-objects`) that were never approved for us. The pipeline therefore uses only ungated
components ("SAM-free" baseline). Every model runs inside Apptainer images on A100-class GPUs (the CUDA
extensions need compute capability 8.0); the repository itself is portable and scheduler-independent.

---

## 2. The pipeline, stage by stage

```text
video ─┬─ WHAM ─ world SMPL ─ actor-track selection ─ SMPL→MHR fitter ───────────────┐
       │                                                                                ├─ joint alignment ─ assembly ─ depth-trend ─ hoop ─ official
       └─ GroundingDINO + SAM 2 masks, MoGe-2 depth ─ TRELLIS mesh ─ FoundationPose ───┘   (PnP, drift,     (episode     correction   smoothing   packer
                                                     (hoop: symmetric tracker + torus)      depth scale)      NPZ)
                                                                                         ... ─ human refinement (smoothing, anchored hands, RTS fusion) ─ packer
```

Each stage is a script in `pipeline/stages/`; `pipeline/run_all.sh` runs them in dependency order
and skips finished outputs.

### 2.1 Setup

- **Does:** fetches pinned upstream repos, builds 6 Apptainer images (WHAM, MHR converter, SAM 3D /
  TRELLIS, CARI4D/FoundationPose, GroundingDINO/SAM 2, MoGe), rebuilds WHAM's DPVO CUDA extensions for
  the local GPUs, downloads the videos and all public weights, creates the submission venv.
- **Expected:** validators pass (`scripts/validate_{workspace,wham_setup,mhr_setup,object_setup}.py`);
  all pins recorded in `config/upstream.env`. The licensed SMPL neutral model is supplied by hand and
  never committed.

### 2.2 Depth and intrinsics: `object_depth.sh` (MoGe-2)

- **Does:** MoGe-2 ViT-L predicts metric depth and pinhole intrinsics for every frame.
- **Expected:** one depth PNG and intrinsics JSON per frame. Depth is used for the object's metric size,
  for the human/object depth scale (§2.6) and later for the human depth-trend correction (§2.9).
  Measured: MoGe's own frame-to-frame scale wobble on static background ≈ 1 %.

### 2.3 Object masks: `object_masks.sh` (GroundingDINO + SAM 2.1)

- **Does:** GroundingDINO detects the object from the episode's prompt on sparse frames; the strongest
  detections seed SAM 2.1 propagation through the whole video. Reviewed prompt overrides live in
  `config/manual_object_prompts.json`.
- **Expected:** a mask per frame for all 30 episodes (complete). Episodes 4, 7, 18, 20 keep review flags:
  their small or occluded objects have empty masks on some frames.

### 2.4 Human: `wham.sh`, `wham_to_mhr.sh`

- **Does:** WHAM (YOLOv8 detection, ViTPose-H keypoints, DPVO, WHAM-BEDLAM) produces world-frame SMPL
  motion. A dense SMPL mesh is mapped to MHR topology with Meta's barycentric map and fitted with the
  kit's `mesh_to_mhr_params.py` (one shared identity per episode).
- **Actor selection:** WHAM's longest person track is exported by default. In episode 2 that was a
  seated background person, and in episode 15 the actor's track was split. DINOv3 wrist/object
  covisibility (experiment `dinov3_contact`) picked the right fragments, which are listed in
  `config/human_track_overrides.json` and stitched (commit `2c1e8bd`). Episode 2's human moved from
  6.76 m to its real 3.40 m from the camera, and its hands now meet the hoop in 65 % of frames (was 0 %).
- **Expected:** MHR parameters for every frame; topology fit error ≈ 4.5 mm (episode 0 smoke test).

### 2.5 Object meshes: `object_mesh.sh` (TRELLIS + FoundationPose selection)

- **Does:** picks the cleanest mask view in each of the object's three episodes and generates TRELLIS
  image-to-3D candidates from each view and from all three, with two seeds. Each candidate is
  registered with FoundationPose in all three cameras over a 0.75–1.33× scale range around the MoGe-2
  size. The mesh with the best mean (mask IoU + depth agreement) wins.
- **Expected:** one metric mesh per object (estimates, not ground truth; each has a provenance file).

| Object | Mean IoU (3 cameras) | Extents (m) | Known weakness |
|---|---|---|---|
| big_red_bowl | 0.93 | 0.28 × 0.13 × 0.38 | thin wire artifact |
| black_pan | 0.79 | 0.21 × 0.02 × 0.19 | flat disc, **no handle** |
| foam_grass_block | 0.96 | 0.19 × 0.20 × 0.19 | |
| hula_hoop | 0.60 | 0.68 × 0.03 × 0.70 | replaced by an exact torus (§2.7) |
| iron | 0.76 | 0.19 × 0.35 × 0.09 | scale at the 1.33× bound |
| paint_roller | 0.88 | 0.24 × 0.07 × 0.07 | |
| pink_foam_roll | 0.92 | 0.59 × 0.11 × 0.04 | flattened, not a cylinder |
| short_wood_stool | 0.80 | 0.43 × 0.48 × 0.27 | |
| white_desk | 0.81 | 0.94 × 0.82 × 0.55 | |
| white_laptop_cart | 0.78 | 0.54 × 0.77 × 0.42 | |

### 2.6 Joint alignment: `align.sh`

- **Does:** the cameras are static, so one rigid transform maps WHAM's world into the camera: RANSAC
  PnP + LM refinement of WHAM's COCO joints against its ViTPose keypoints with MoGe-2 intrinsics. A
  per-frame translation-only PnP measures WHAM's drift; only its low-frequency part (median 15 frames,
  Gaussian σ 20) is added to the MHR root. `depth_scale` = median MoGe depth / SMPL front-surface depth
  puts the object (MoGe metres) into the human's frame.
- **Expected / measured:** median reprojection 5.4–17.3 px after drift correction (6.2–150.6 px
  static), depth scale 0.83–0.95. Worst: episode 24 (17.1 px after a 1.35 m correction) and episode 8.

### 2.7 Object tracking: `track_object.sh`, `track_symmetric.sh`

- **Does:** FoundationPose registers and tracks the mesh; an IoU-weighted ESKF/RTS smoother produces
  the final trajectory.
- **Hula hoop (commit `9230b27`, experiment `hoop_symmetry`):** FoundationPose cannot track the thin,
  textureless, symmetric ring (72–126 jumps > 10 cm per episode, 40–90° axis flips), and RTS froze it
  (IoU 0.07–0.11 while handled). Replacement: an exactly round torus mesh, and a tracker whose state is
  only the ring centre and axis line. It uses mask-ellipse proposals, a particle filter on a 2D chamfer
  to the mask with a hand-contact prior, then global least squares with rest-aware smoothness anchored
  to FoundationPose while the hoop rests.
- **Expected / measured (hoop):** silhouette IoU on scored frames 0.46 → 0.52 (ep 0), 0.14 → 0.49
  (ep 1), 0.41 → 0.60 (ep 2). The trade-off, visible only later on Kaggle, was a jerky ring centre
  (≈ 2–3.6 cm/frame² while handled), fixed in §2.10.

### 2.8 Assembly: `build_track1_episodes.py`

- **Does:** writes one NPZ per episode in the Track 1 frame: MHR parameters (drift-corrected) and
  `object_rotation = R_cwᵀ R_co`, `object_translation = R_cwᵀ (t_co / depth_scale − t_cw)`,
  `object_scale = s_co / depth_scale`, plus the mesh (torus for the hoop).
- **Expected:** packs with the official kit to 740,780 rows with no warnings.

### 2.9 Human depth-trend correction: `depth_trend.sh` + `pack.sh` (new today, commit `9624185`)

- **Why:** WHAM's trajectory drifts along the camera ray. A constant depth error is free (Sim(3)), but
  a drifting one is not. Measured: the human's depth disagrees with MoGe by 5 cm median (p90 12 cm),
  with 10–20 cm first-to-last-third trends in 11 episodes.
- **Does:** every third frame, the ratio of MoGe depth to our SMPL front-surface depth over the body
  pixels, divided by MoGe's own wobble on static background pixels. Only the trend is kept (median 15,
  Gaussian σ 30 frames), and the MHR root moves along the camera ray by α = 0.75 of it. Body shape and
  objects are untouched.
- **Expected:** a smooth correction (median 2.3 cm, ACC-H cost bound 0.003 cm/frame²); the repo
  reproduces the Kaggle-tested file bit for bit.

### 2.10 Hula-hoop centre smoothing: `smooth_symmetric_objects.py` (new today, commit `5614cc4`)

- **Why:** the symmetric tracker's ring centre jitters while handled (episode 1: 2.7 cm/frame² mean on
  scored frames, about 24 m/s², against 0.01–0.08 for every FoundationPose object). That one episode
  was roughly half of our ACC-O.
- **Does:** weighted least squares on the world-frame centre: stay near the track (weight 40 on frames
  where the mask says the hoop rests, 1 elsewhere) + λ = 100 × squared second differences. Rotation
  and mesh unchanged.
- **Expected:** ≈ 85–95 % less ring-centre acceleration at unchanged silhouette IoU on scored frames.

### 2.11 Human refinement: `keypoints2d.sh` + `human_refine.sh` (new, this commit)

- **Why:** WHAM leaves frame-level jitter on the body (our 22 scored joints: 0.185 cm/frame² self-
  acceleration, wrists 0.39), and its hands and lateral position drift from the image.
- **Does:** per episode, MediaPipe pose and hand landmarks (hands on upscaled wrist crops) and WHAM's own
  ViTPose keypoints (held-out check). Then, on the assembled episodes:
  1. λ 10 mesh-smoothing refit as the process model;
  2. per-hand wrist fusion RTS (MediaPipe lateral, object anchors in 3D, contact guard);
  3. λ 100 refit with the fused wrists and hands anchored to the FoundationPose object in stable grasps;
  4. a second λ 100 smoothing pass;
  5. lateral root RTS from MediaPipe, applied where ViTPose agrees.
  Details: §4.7 and `docs/JOINT_BASELINE.md`.
- **Expected:** joint self-acceleration ≈ 0.06 cm/frame², local PEN ≈ 0.0082 cm, fit error < 0.3 mm,
  objects byte-identical. Needs a GPU (MHR refits); about 75 minutes for all 30 episodes on one A100.

### 2.12 Packing and upload: `pack.sh`, `submit/v2d_submit.py`

- **Does:** official `eval_reconstruction.py` → Parquet with the commit URL; the kit's uploader sends it
  to the five boards, waits for scores and writes a receipt.
- **Expected:** `submissions/track1_wham_fp_<date>.parquet`, 740,780 rows.

---

## 3. Scores

### 3.1 Every Kaggle submission today

The first upload was the full pipeline of §2.1–2.8 (commit `2c1e8bd`; hoop tracker and actor fix were
already in, so they have no separate Kaggle numbers). Every later submission is a **controlled A/B**:
one change (or, for the round-4 combined candidates, one named stack of changes) against the previous
best, everything else byte-identical (verified by diffing the Parquet rows), sent only to the boards that
change can affect. These probes carry the base commit's URL
and are labelled "probe" in the Kaggle message; winners were then committed and pushed.

| # | UTC | What changed | CD-H | CD-O | ACC-H | ACC-O | PEN |
|---|---|---|---|---|---|---|---|
| 1 | 15:12 | **First submission** (`2c1e8bd`) | 23.13845 | 31.23021 | 0.22090 | 0.17288 | 0.01064 |
| 2 | 17:18 | H1: human depth follows MoGe (σ 10, α 1) | 21.70005 | | | | |
| 3 | 17:19 | **H2t: trend-only depth correction (σ 30, α 0.75)** | **21.48384** | | | | |
| 4 | 17:20 | H3s: anchor-frame orientation fix (eps 10, 11, 24) | 23.02980 | | | | |
| 5 | 17:21 | H4: H2t gated to 8 episodes by keypoint agreement + H3s | 21.95084 | | | | |
| 6 | 17:43 | H2t on the other four boards | | **29.91374** | **0.22081** | 0.17288 | **0.00907** |
| 7 | 18:42 | Hoop smoothing λ 10 (on `9624185`) | | 29.90568 | | 0.08679 | |
| 8 | 18:42 | **Hoop smoothing λ 100** | | **29.89199** | | **0.07693** | |
| 9 | 22:32 | Human motion smoothing λ 10 (on `5614cc4`) | | | 0.15290 | | |
| 10 | 22:33 | Human motion smoothing λ 30 | | | 0.14722 | | |
| 11 | 22:34 | **Human motion smoothing λ 100** | | | **0.14327** | | |
| 12 | 22:37 | Combined `fuse10cs_lat` (λ 10 + object-anchored IK + RTS wrist fusion + RTS lateral root) | | **29.23757** | | 0.07699 | 0.00831 |
| 13 | 00:00* | Human motion smoothing λ 300 | | | **0.14171** | | |
| 14 | 00:01* | Human motion smoothing λ 100 on CD-H | 21.45583 | | | | |
| 15 | 00:01* | Combined `fuse10cs_lat` on CD-H and ACC-H | 21.02400 | | 0.14897 | | |
| 16 | 00:02* | **Combined `fuse100cs_lat` (λ 100), all five boards** | **20.95822** | 29.36947 | 0.14235 | 0.07700 | **0.00822** |

\* 2026-10-16 UTC (new daily quota).

### 3.2 Effect of each technique (Kaggle deltas)

| Technique | CD-H | CD-O | ACC-H | ACC-O | PEN | Decision |
|---|---|---|---|---|---|---|
| Depth-trend correction (H2t) | −1.655 (−7.2 %) | −1.316 (−4.2 %) | −0.0001 | 0 (exact) | −0.0016 (−15 %) | pushed `9624185` |
| Hoop centre smoothing (λ 100) | 0 (exact: human untouched) | −0.022 | 0 (exact) | −0.0960 (−55 %) | not probed (hoop PEN ≈ 0) | pushed `5614cc4` |
| Noisier depth correction (H1) | −1.438 | | | | | superseded by H2t |
| Anchor-frame orientation (H3s) | −0.109 | | | | | small; folded into later smoothing |
| Keypoint-gated correction (H4) | −1.188 | | | | | gating hurts; rejected |
| Human motion smoothing alone (λ 100) | −0.028 | | −0.0775 (−35 %) | | | part of the combined candidate |
| Human motion smoothing alone (λ 300) | | | −0.0791 (−36 %) | | | gains flatten above λ 100 |
| Smoothing λ 10 + IK + wrist fusion + lateral root (`fuse10cs_lat`) | −0.460 | −0.654 | −0.0718 | +0.00006 | −0.00076 | |
| **Smoothing λ 100 + IK + wrist fusion + lateral root (`fuse100cs_lat`)** | **−0.526 (−2.4 %)** | **−0.523 (−1.7 %)** | **−0.0785 (−35.5 %)** | +0.00007 (≈ 0) | **−0.00085 (−9.4 %)** | best complete file; merged as the human refinement stage |

ACC-O stayed exactly equal under the depth correction, which confirmed the analysis: a per-frame
translation of the human changes the anchor Sim(3) only by a translation, and accelerations ignore
translations.

### 3.3 Where we stand

| | CD-H | CD-O | ACC-H | ACC-O | PEN |
|---|---|---|---|---|---|
| First submission (`2c1e8bd`, 15:12) | 23.138 | 31.230 | 0.2209 | 0.1729 | 0.01064 |
| Pushed pipeline `5614cc4` | 21.484 | 29.892 | 0.2208 | 0.0769 | 0.00907 |
| **Best complete file `fuse100cs_lat`** (one Parquet, all five boards) | **20.958** | **29.369** | **0.1424** | **0.0770** | **0.00822** |
| Change vs first submission | **−9.4 %** | **−6.0 %** | **−35.6 %** | **−55.5 %** | **−22.7 %** |

Single-board bests from other probes: CD-O 29.238 (`fuse10cs_lat`), ACC-H 0.1417 (λ 300 smoothing).
The `fuse100cs_lat` probes were made with the experiment code on top of `5614cc4` (their Kaggle entries
carry that commit's URL and are labelled as probes). The same steps are now the pipeline's human refinement
stage (§2.11). Re-running them from the repository
reproduces that file to numerical precision: 99 % of the scored joints agree within 0.001 mm and the
local PEN is identical (0.008613 cm). The GPU solver is not bit-deterministic, so 62 of 16,563 frames
differ by more than 1 mm on some hand vertices (at most 2.5 mm on scored joints).

---

## 4. Experiments (what was tried, what it showed, what was kept)

Every experiment ran in its own local folder (scripts, outputs, logs and a README; not part of this
repository) against a read-only copy of the pipeline. Only proven winners were merged.

### 4.1 `hoop_symmetry`: kept (`9230b27`)

Symmetry-aware hoop tracker + torus (§2.7). Proxy wins on all three hoop episodes (silhouette IoU up to
3.5×, hand–ring distance 20 → 12 cm).

### 4.2 `dinov3_contact`: used for actor selection (`2c1e8bd`)

DINOv3 ViT-H+/16 patch features: wrist-to-object covisibility ranked the true actor tracks first in
episodes 2 and 15 (margins 0.006 and 0.47). The bounding-box-size baseline agreed, so DINOv3 was used to
pick and check the reviewed overrides rather than as an automatic selector.

### 4.3 `rigid_multiview_meshes`: rejected

TSDF/Poisson fusion of MoGe depth for the pan and iron. Held-out IoU pan 0.543 → 0.561, iron 0.547 →
0.530; visually a thick noisy puck without a handle and a ragged blob. Monocular depth on small dark
objects is too noisy for fusion.

### 4.4 `dinov3_hand_refine`: rejected

DINOv3 contact anchors moving only finger/wrist parameters on the hoop episodes: contact distance 1–2 mm
better, PEN already ≈ 0, joint acceleration unchanged. No benefit.

### 4.5 `round2_pen_cdh`: scorer analysis, depth correction (kept), PEN push (not kept)

- **Scorer invariance, proven with the official metric code** (episode 0 as pseudo ground truth):
  a global Sim(3) (scale 1.15, yaw 10°, 30 cm shift) or a constant 20 cm offset changes all five metrics
  by exactly 0. A 0 → 20 cm depth drift costs 11.7 cm CD-H and 7.0 cm CD-O, and a 5° rotation in the
  first scored frame only costs 7.5 cm CD-H. Per-video global grounding was therefore dropped.
- **Depth correction variants H1–H4** (§3.1). Two lessons: MoGe's background-normalised depth trend is
  closer to the truth than WHAM's trajectory, and **2D keypoint reprojection is not a CD-H proxy**. The
  gate that excluded episodes where keypoints disagreed with MoGe made it worse.
- **PEN push v1** (move the object out of the hand on every frame): local PEN −93 % (0.0126 → 0.0009),
  but offsets up to 34 cm and up to +26 % ACC-O worst case. **v2** (acceleration-penalised push, on top of
  H2t, λ 10⁴, 6 episodes): local PEN 0.0100 → 0.0045 (−56 %) for ≤ +1 % ACC-O and ≤ +0.73 cm CD-O worst
  case. Not yet probed on Kaggle.

### 4.6 `round3_kinematic_coupling`: hoop smoothing (kept)

- Proposal tested: lock the object to the wrist inside grasp windows (velocity matching), in a joint
  optimiser.
- **Diagnostic:** for every rigid object the object as carried by the MHR hand is **5–30× jerkier** than
  the FoundationPose track (0.26–0.83 vs 0.03–0.08 cm/frame²; hand rotation jitter × lever arm).
  Coupling would inject that noise. Only the hoop was tracked worse than the hand carries it.
- On the hoop, adding hand velocity coupling to smoothing lowered silhouette IoU (episode 1 scored IoU
  0.477 → 0.433) for no acceleration gain. Plain acceleration-penalised smoothing won locally and on
  Kaggle (§3.1 rows 7–8). Kaggle preferred more smoothing (λ 100) than the image check did (λ 10).

### 4.7 `round4_mediapipe_ik`: human smoothing, object-driven IK, MediaPipe, Kalman/RTS fusion (best results)

- **Where human acceleration comes from:** our 22 scored joints have a self-acceleration of 0.185
  cm/frame² (Kaggle ACC-H 0.221). Wrists (0.39) and elbows (0.24) are the worst. WHAM's own joints are
  just as jerky, so the jitter comes from WHAM, not from the MHR conversion.
- **Motion refit:** MHR Euler parameters wrap by multiples of 2π, so the motion is smoothed on the mesh
  (per-vertex λ smoother) and every frame's 136 parameters are refitted with the kit's LM solver (fit
  error 0.04–0.3 mm). Joint self-acceleration 0.185 → 0.090 (λ 10), 0.080 (λ 30), 0.068 (λ 100),
  0.058 (λ 300). 2D agreement with two detectors is unchanged at every λ, so λ can only be chosen on
  Kaggle.
- **Object-driven human IK:** in 52 stable grasp windows the hand rides rigidly on the smooth
  FoundationPose track, using its mean shape in the object frame pushed out of the surface (≤ 3 cm),
  blended over 15 frames. Local PEN 0.01000 → 0.00866 (−13 %, mostly episode 29). Wiping/sliding
  hands are excluded.
- **MediaPipe** (Tasks API; hands detected on upscaled wrist crops, ≈ 80 % of frames): agrees with our
  projected joints to 8–16 px median, but its 2D wrist track is about 9× jerkier than ours, so it is
  only a low-frequency reference. WHAM's ViTPose keypoints are held out as an independent check.
- **Kalman/RTS smoothers** (fixed-interval: forward Kalman filter + backward RTS pass, no lag):
  - **Root lateral drift** from MediaPipe body joints, applied only where ViTPose also improves:
    episodes 1, 8, 17, 20, 26. Episode 8 lateral error 7.5 → 6.4 cm (ViTPose).
  - **Wrist fusion** (MHR process model, MediaPipe 2D wrists as noisy observations, object anchors as
    tight 3D observations, contact guard): wrist 2D error 10.6 → 9.1 px against held-out ViTPose, best
    PEN 0.00864, joint self-acceleration 0.083.
  - Two failure modes found and fixed along the way: 2D-only observations leaked into depth (30–117 cm
    moves along the ray), and an unguarded wrist shift pushed a hand into the cart (episode 27).
- **Episode 24 is a WHAM front/back flip**, not drift. In 55 of 217 checked scored frames both
  MediaPipe and ViTPose prefer our body with left and right swapped (person spinning on a stool). The fix
  (rebuild those frames mirrored and refit) is not built yet.
- **Candidates** (packed, only human-pose rows differ from `5614cc4`): s10, s30, s100, ik10, ik10_lat,
  fuse10cs, and **fuse10cs_lat** = λ 10 smoothing + object-anchored IK + guarded wrist fusion + lateral
  root fix.
- **Kaggle (§3.1 rows 9–16):** ACC-H 0.2208 → 0.1529 (λ 10) → 0.1472 (λ 30) → 0.1433 (λ 100) → 0.1417
  (λ 300): the jitter WHAM leaves really is noise, with gains flattening above λ 100. Smoothing alone
  hardly moves CD-H (21.484 → 21.456). The CD-H gain comes from the 2D-evidence and contact parts
  (object-anchored hands, guarded wrist fusion, lateral root fix): 21.024 at λ 10, **20.958 at λ 100**.
  They also improve CD-O (better anchor frame and hands, so the object lands better: 29.24–29.37) and
  PEN (0.0082–0.0083), with ACC-O unchanged. **`fuse100cs_lat` is the best complete file.**

---

## 5. Lessons

1. **Read the scorer first.** The per-episode Sim(3) at the first scored frame makes global fixes
   worthless and makes time-varying error and the anchor frame everything.
2. **Proxies mislead; controlled A/B probes do not.** Keypoint reprojection disagreed with the CD-H
   winner, and silhouette IoU preferred less hoop smoothing than Kaggle did. Kaggle scoring is
   deterministic, so a one-change probe gives an exact delta.
3. **Trust the smoothest reliable source.** FoundationPose is the clean signal for rigid objects; MHR
   hands are noisy; MoGe depth trends beat WHAM depth; MediaPipe is a low-frequency reference only.
4. **Every fix needs a guard against its own failure mode**: depth leaks, pushes into objects, offsets
   at window edges that create acceleration spikes.

## 6. Open issues and next steps

- Repack with this commit's URL and submit that one file to all five boards as the clean final entry
  (the kit expects one file for all five boards; the probes so far carry the `5614cc4` URL).
- λ trade-off: λ 300 is best on ACC-H, λ 10 best on CD-O; a λ 100–300 combined candidate may edge
  further.
- Fix the episode-24 front/back flips (largest single human error found).
- Probe PEN v2 (acceleration-penalised object push) combined with the winners.
- Object meshes: pan without handle, iron at the scale bound, flattened foam roll, bowl wire artifact.
  The multiview fusion attempt failed; this needs better masks/depth or a different reconstructor.
