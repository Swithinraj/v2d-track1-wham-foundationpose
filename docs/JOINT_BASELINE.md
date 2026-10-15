# SAM-free joint baseline: WHAM human + TRELLIS/FoundationPose object

This is the end-to-end Track 1 path that avoids both gated Meta checkpoints
(`facebook/sam-3d-body-dinov3`, `facebook/sam-3d-objects`). It follows the decoupled baseline plan:
human track, object track, then a joint alignment layer, then the official packer.

```text
video ─┬─ WHAM ─ dense SMPL (WHAM world) ─ official MHR fitter ─ MHR params ─┐
       │                                                                      ├─ joint alignment ─ episode NPZ ─ eval_reconstruction.py
       └─ SAM 2 masks + MoGe-2 depth ─ TRELLIS mesh ─ FoundationPose ─────────┘
```

None of the generated object meshes is challenge ground truth. Each installed mesh has a
`<object>.provenance.json` recording how it was generated and selected.

## 1. Human track

`pipeline/stages/wham.sh EP` then `pipeline/stages/wham_to_mhr.sh EP`. Outputs:
`outputs/wham/episode_*/wham_smpl_world.npz` and `outputs/human/episode_*/mhr_human.npz`
(WHAM world metres). See `docs/WHAM_BASELINE.md`.

**Actor selection.** By default the export takes WHAM's longest person track. Two clips break that
assumption, so `config/human_track_overrides.json` lists reviewed track fragments for them (chosen
with DINOv3 wrist/object-mask covisibility in the separate `dinov3_contact` experiment, then checked
against the video):

- episode 2: tracks 1 → 2 (the longest track, 0, is a seated person in the background);
- episode 15: tracks 0 → 4 (the actor's track is split in two).

`scripts/export_wham_world_mesh.py --subjects` merges the fragments (later fragments win overlapping
frames), shifts each later fragment's world translation to continue the previous one (episode 2's
ID-switch jump 1.08 m → 3.9 cm, episode 15's 7.3 → 4.4 cm) and uses one median body shape across
stitched fragments. Alignment takes each frame's keypoints from the fragment that produced it.
Effect: episode 2's human moved from 6.76 m to 3.40 m from the camera (the actor's real position),
its drift correction fell from 1.57 m to 0.19 m and its hands now meet the hoop (65% of frames,
previously 0%); episode 15's human moved from 4.82 m to 3.42 m. Single-track episodes are unchanged.

## 2. Object meshes: TRELLIS candidates + FoundationPose selection

`pipeline/stages/object_mesh.sh OBJECT` (GPU with compute capability 8.0+), once per object:

1. `scripts/trellis_object_mesh.py` (SAM3D image): scores every third SAM 2 mask frame of each
   of the object's three episodes for size, compactness, border truncation and temporal stability,
   and keeps the best view per episode (`views/candidates.jpg` is the review sheet;
   `config/manual_mesh_views.json` can override views as `{"object": [{"episode": E, "frame": F}]}`).
   TRELLIS-image-large (MIT, `weights/trellis`) generates candidates from each view alone and from
   all three together, with seeds 1 and 2. Each is cleaned, decimated to 40k faces, made y-up, and
   given a coarse metric size from the masked MoGe-2 point cloud's principal extent.
2. `scripts/select_object_mesh.py` (CARI4D image): every candidate is registered with
   FoundationPose on each episode's selection frame over a log-scale grid of 0.75–1.33× the
   depth-derived size, scored as mask IoU + 0.5 · exp(−median |depth ratio − 1|). The best mean
   over the three cameras wins; its scale is refined (7 samples × 3 levels, same range) and the
   per-episode median applied. Result: `inputs/object_meshes/<object>.glb`.
   `SKIP_GENERATION=1` re-runs only this step on existing candidates.

Why the scale range is narrow: FoundationPose registration always places the mesh at the
observed depth, so the depth term is ≈1 at every scale, and silhouette IoU then rewards inflating
an imperfect mesh until it covers the mask. With a 0.5–2× range the (too slim) iron mesh went to
2× its depth-derived length (0.56 m); a fixed-scale diagnostic showed the IoU curve over scale is
noisy and nearly flat. MoGe-2's back-projected extent sets the metric size, FoundationPose only
refines it.

Why candidates: single-view TRELLIS output from these small, dim, top-down exo crops is
semantically unstable. Observed failure modes include a stool turned into a bench, a bowl into a
flat dish, and objects embedded in a full-width dark plate (`plate_suspect`, skipped when at least
two clean candidates exist). Three different cameras reject shapes that only explain one view.

TRELLIS imports `rembg`, `transformers` and `open3d` at import time; the SAM3D image lacks the first
two and its open3d wheel lacks `libusb`. Only the text-to-3D pipeline and background removal use
them, so `install_import_shims()` installs stand-ins that raise if ever called.

```bash
source pipeline/env.sh
for object in $(all_objects); do bash pipeline/stages/object_mesh.sh "$object"; done
python3 scripts/validate_object_setup.py
```

Selected meshes from the run that produced the submission (mean silhouette IoU over the object's
three cameras after FoundationPose registration; review sheet in
`outputs/object_meshes/review_sheet.png`):

| Object | Winner | Mean IoU | Extents (m) | Note |
|---|---|---|---|---|
| big_red_bowl | c05 | 0.93 | 0.28 × 0.13 × 0.38 | thin wire artifact |
| black_pan | c04 | 0.79 | 0.21 × 0.02 × 0.19 | flat disc, no handle |
| foam_grass_block | c04 | 0.96 | 0.19 × 0.20 × 0.19 | |
| hula_hoop | c03 | 0.60 | 0.68 × 0.03 × 0.70 | |
| iron | c00d | 0.76 | 0.19 × 0.35 × 0.09 | scale at the 1.33× bound |
| paint_roller | c05 | 0.88 | 0.24 × 0.07 × 0.07 | |
| pink_foam_roll | c02 | 0.92 | 0.59 × 0.11 × 0.04 | flattened, not a cylinder |
| short_wood_stool | c07d | 0.80 | 0.43 × 0.48 × 0.27 | multi-view, de-plated |
| white_desk | c02d | 0.81 | 0.94 × 0.82 × 0.55 | |
| white_laptop_cart | c05d | 0.78 | 0.54 × 0.77 × 0.42 | |

## 3. Object tracking

`pipeline/stages/track_object.sh EP` (FoundationPose + ESKF/RTS smoothing), as documented in
`docs/OBJECT_BASELINE.md`. Poses are object-to-camera in MoGe-2 metres.

Apptainer refuses `--env HOME=...` and keeps HOME on the unmounted, read-only host home. Then
`warp.init()` fails silently and FoundationPose's `Utils.erode_depth` is never defined (ImportError
at startup). `in_container` in `pipeline/env.sh` therefore sets `HOME=/tmp` inside the container.

## 3b. Symmetric objects: the hula hoop

FoundationPose cannot track the thin, textureless, rotationally symmetric hoop: its raw track has
72–126 single-frame jumps over 10 cm per episode (up to 6.7 m) and 40–90° axis flips, and the RTS
smoother removes them by nearly freezing the hoop (silhouette IoU 0.07–0.11 while it is handled).
The hoop's spin about its own axis is invisible to CD-O and ACC-O, so what matters is the ring centre
and axis. `config/symmetric_objects.json` lists such objects; for them:

1. `scripts/make_round_mesh.py` replaces the generated mesh (0.677 × 0.704 m, 4% elliptical) by an
   exactly round torus of the same size (`<object>.symmetric.glb`, ring radius 0.332 m, tube 0.0135 m).
2. `pipeline/stages/hands.sh` computes MHR hand points from the drift-corrected human.
3. `pipeline/stages/track_symmetric.sh` → `scripts/track_symmetric_object.py`: the state is the ring
   centre and the axis line (no spin, no sign). Mask ellipse fits with the known radius propose depth
   and both tilt hypotheses; a particle filter weights particles by a symmetric 2D chamfer between the
   projected ring and the SAM 2 mask (trimmed for occlusion) and a hand-contact prior when a hand is
   next to the hoop; a global sparse least-squares pass adds acceleration smoothness, 40× smoothness
   and zero velocity on frames where the mask shows the hoop at rest, and an anchor to the
   FoundationPose track on those resting frames.
4. `scripts/build_track1_episodes.py` uses the symmetric track and round mesh when present.

Measured on episodes 0–2 (silhouette IoU of the rendered hoop vs the SAM 2 mask; acceleration of the
ring centre in cm/frame², FoundationPose smoothed → symmetric):

| ep | IoU all frames | IoU scored frames | acc. at rest | acc. while handled | hand–ring distance |
|---|---|---|---|---|---|
| 0 | 0.31 → 0.52 | 0.46 → 0.52 | 0.01 → 0.02 | 0.04 → 2.33 | 16.6 → 9.1 cm |
| 1 | 0.22 → 0.61 | 0.14 → 0.49 | 0.01 → 0.03 | 0.04 → 2.09 | 20.1 → 12.2 cm |
| 2 | 0.29 → 0.62 | 0.41 → 0.60 | 0.01 → 0.03 | 0.06 → 3.63 | n/a |

Without ground truth these are proxies. The particle filter is stochastic: a re-run with a 0.1%
different radius changed a few edge-on frames by up to 0.38 m but left per-frame IoU unchanged on
average (≤ 0.008). With the corrected actor in episode 2, the hand prior became active there: same
IoU (0.617 / 0.598 scored), axis-change p99 83° → 23° per frame, no more implausible depths
(max 4.6 → 3.7 m), hand–ring distance 9.6 cm.

## 4. Joint alignment

`pipeline/stages/align.sh EP` runs `scripts/align_human_camera.py` (WHAM image, CPU) →
`outputs/alignment/episode_*/{alignment.json,drift.npz}`.

- **Static camera.** The exo cameras do not move (WHAM's DPVO returns identity), so one rigid
  transform maps WHAM world into the camera. It is solved by RANSAC PnP + LM refinement over all
  frames: WHAM world COCO joints (`J_regressor_coco`) against WHAM's ViTPose keypoints, using
  MoGe-2's intrinsics. WHAM itself ran with its default 1920 px focal length (MoGe-2 estimates
  ~811 px), so WHAM's own camera-frame translation is not used.
- **Drift.** WHAM's integrated world trajectory drifts against the static camera. A per-frame,
  translation-only PnP measures the offset; a 15-frame median and σ = 20-frame Gaussian keep only
  the low-frequency drift, which is added to the MHR root (`pose[:, :3] += diag(1,−1,−1)·d·10`,
  verified to move every MHR vertex by exactly `d` metres). WHAM's local motion is untouched.
- **Scale.** `depth_scale` = median MoGe depth / drift-corrected SMPL front-surface depth
  (4 px z-buffer cells). Object translations and object scale are divided by it.

Results on all 30 episodes: drift-corrected median reprojection 5.4–17.3 px (from 6.2–150.6 px
static), depth_scale 0.83–0.95. Review first: episode 24 (150.6 → 17.1 px, 1.35 m correction),
episode 8 (28.9 → 17.3 px, 1.27 m), episodes 2 and 11 (>1 m corrections).

## 5. Assembly and packing

`pipeline/stages/pack.sh`:

1. `scripts/build_track1_episodes.py` writes `submissions/episodes_wham_fp/episode_*.npz`
   (`pose`, `scales`, `shape`, `object_rotation`, `object_translation`, `object_scale`) and
   `episode_*_object.glb`, with
   `object_rotation = R_cwᵀ R_co`, `object_translation = R_cwᵀ (t_co / depth_scale − t_cw)`,
   `object_scale = s_co / depth_scale`.
2. Human depth-trend correction (`DEPTH_TREND=1`, on by default; `DEPTH_TREND_ALPHA`, default 0.75):
   the scorer re-anchors every episode at its first scored frame, so only depth error that changes
   over time costs CD-H, and WHAM's trajectory drifts in depth. `pipeline/stages/depth_trend.sh EP`
   (`scripts/depth_trend_correction.py series`, WHAM image, CPU) measures, every third frame, the
   ratio of MoGe-2 depth to the drift-corrected SMPL front-surface depth over the body's pixels,
   divided by MoGe's own per-frame scale wobble on static background pixels (no body, no object) and
   by its median → `outputs/depth_trend/<ep>.npz`. `depth_trend_correction.py apply` keeps only the
   trend (median filter 15 frames, Gaussian σ 30 frames) and moves the MHR root along the camera ray
   by α·(trend − 1)·centroid; body size and objects are unchanged →
   `submissions/episodes_wham_fp_depth/` + `depth_trend_report.json`. Kaggle A/B against the same
   pipeline without it: CD-H 23.14 → 21.48 cm, CD-O 31.23 → 29.91 cm, PEN 0.0106 → 0.0091,
   ACC-H 0.2209 → 0.2208, ACC-O unchanged (a per-frame translation of the human changes the anchor
   fit by a translation only).
   A version gated by 2D-keypoint agreement scored worse (21.95), so it is applied to every episode.
3. Symmetric-object smoothing (`SMOOTH_SYMMETRIC=1`, on by default; `SMOOTH_SYMMETRIC_LAM`, default 100):
   the symmetry-aware hoop tracker fits the ring to every frame's mask and leaves broadband jitter on
   moving frames (episode 1: 2.7 cm/frame² mean acceleration on scored frames, against 0.01–0.08 for
   every FoundationPose-tracked object). `scripts/smooth_symmetric_objects.py` (WHAM image) re-solves
   the world-frame centre of each symmetry-tracked episode as weighted least squares: stay near the
   track (weight 40 on frames where the tracker's mask-motion test says the object rests, 1 elsewhere)
   plus λ × squared second differences. Rotation and mesh are unchanged → `<previous dir>_smooth/` +
   `smooth_symmetric_report.json`. Kaggle A/B against the same pipeline without it: ACC-O 0.1729 →
   0.0769, CD-O 29.91 → 29.89 cm (λ 10: 0.0868 / 29.91). Coupling the centre's velocity to the
   grasping MHR hand instead lowered the silhouette IoU: the hands are noisier than the track, and for
   every rigid object the hand-carried pose is 5–30× jerkier than FoundationPose's.
4. Optional hand–object penetration step (`RESOLVE_PEN=1`, off by default):
   `scripts/extract_mhr_hands.py` (MHR image) computes 512 hand points per frame (256 per hand:
   vertices whose dominant skinning joint lies in the wrist subtrees 42/78, fixed seed; an
   approximation of the scorer's host-side hand roles). `scripts/resolve_hand_penetration.py`
   measures penetration like the scorer (winding number against the budgeted object mesh, inverse
   Sim(3) pose) and, on penetrating frames of episodes with self-measured PEN ≥ 0.05 cm, translates
   the object away from the hand by the smallest bisected distance, then dilates and smooths the
   offsets. Output: `<previous episode dir>_pen/` + `penetration_report.json`.
   Off by default because the baseline already has negligible penetration (self-measured PEN
   0.013 cm mean over 30 episodes). Only episodes 9, 27 and 29 crossed the threshold, and the push
   multiplied their object acceleration by 3–14× (episode 27 also got slightly worse PEN).
5. The official `eval_reconstruction.py` packs and validates the Track 1 Parquet.

`COMMIT_URL` must be the GitHub commit of the code that produced the run
(`https://github.com/<owner>/<repo>/commit/<sha>`, visible to Nvidia-V2D-Challenge). Without it,
the script writes `submissions/track1_LOCAL_CHECK_do_not_upload.parquet` for validation only.

```bash
bash pipeline/stages/pack.sh                                                          # local validation
COMMIT_URL=https://github.com/<owner>/<repo>/commit/<sha> bash pipeline/stages/pack.sh   # uploadable
```

Uploading (`submit/v2d_submit.py ... --track track_1`) needs a Kaggle account and accepted rules
for all five Track 1 competitions.

## Not yet done

- CoCoNet refinement: blocked on SAM 3D Body decoder buffers that the accessible
  `nvidia/cari4d_commercial` checkpoint does not contain.
