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
average (≤ 0.008). Episode 2 still shows large axis swings in its fastest motion and gets no hand help,
because its exported WHAM person is the seated background person (see "Known issues").

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
2. Optional hand–object penetration step (`RESOLVE_PEN=1`, off by default):
   `scripts/extract_mhr_hands.py` (MHR image) computes 512 hand points per frame (256 per hand:
   vertices whose dominant skinning joint lies in the wrist subtrees 42/78, fixed seed; an
   approximation of the scorer's host-side hand roles). `scripts/resolve_hand_penetration.py`
   measures penetration like the scorer (winding number against the budgeted object mesh, inverse
   Sim(3) pose) and, on penetrating frames of episodes with self-measured PEN ≥ 0.05 cm, translates
   the object away from the hand by the smallest bisected distance, then dilates and smooths the
   offsets. Output: `submissions/episodes_wham_fp_pen/` + `penetration_report.json`.
   Off by default because the baseline already has negligible penetration (self-measured PEN
   0.013 cm mean over 30 episodes). Only episodes 9, 27 and 29 crossed the threshold, and the push
   multiplied their object acceleration by 3–14× (episode 27 also got slightly worse PEN).
3. The official `eval_reconstruction.py` packs and validates the Track 1 Parquet.

`COMMIT_URL` must be the GitHub commit of the code that produced the run
(`https://github.com/<owner>/<repo>/commit/<sha>`, visible to Nvidia-V2D-Challenge). Without it,
the script writes `submissions/track1_LOCAL_CHECK_do_not_upload.parquet` for validation only.

```bash
bash pipeline/stages/pack.sh                                                          # local validation
COMMIT_URL=https://github.com/<owner>/<repo>/commit/<sha> bash pipeline/stages/pack.sh   # uploadable
```

Uploading (`submit/v2d_submit.py ... --track track_1`) needs a Kaggle account and accepted rules
for all five Track 1 competitions.

## Known issues

- WHAM person selection: `scripts/export_wham_world_mesh.py` exports the longest WHAM track. In
  episode 2 that is a seated person in the background (half the size, visible all clip) rather than
  the actor, and episode 15's actor track is split in two while another track is exported. Human
  parameters and every object placement relative to the human are wrong for those episodes.

## Not yet done

- CoCoNet refinement: blocked on SAM 3D Body decoder buffers that the accessible
  `nvidia/cari4d_commercial` checkpoint does not contain.
