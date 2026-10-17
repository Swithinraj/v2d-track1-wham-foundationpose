# Track 1 pipeline: SAM 3D Body + SAM 3D Objects + FoundationPose in the static camera

The exo cameras of Track 1 do not move, so the pipeline works in the camera's own frame from start to finish:

```text
video ─┬─ MoGe-2 depth + intrinsics ───────────────────────────────────────────────────────────┐
       ├─ GroundingDINO + SAM 2 object masks ─ SAM 3D Objects mesh ─ FoundationPose ─ ESKF/RTS ─┤
       └─ GroundingDINO person tracks → actor ─ SAM 2 actor masks ─ SAM 3D Body ─ identity/refit ┴─ assembly ─ refinement ─ packer
```

The scorer aligns every episode with one Sim(3) at its first scored frame (free scale), so a consistent frame and
consistent relative scale between human and object matter; absolute scale and the world origin do not.
None of the generated object meshes is challenge ground truth; each installed mesh has a `<object>.provenance.json`.

## 1. Object inputs

`pipeline/stages/object_depth.sh EP` (MoGe-2 ViT-L: metric depth and per-frame intrinsics) and
`pipeline/stages/object_masks.sh EP` (GroundingDINO prompts from each episode's `object_prompt`, reviewed overrides
in `config/manual_object_prompts.json`, SAM 2.1 propagation), as documented in `docs/OBJECT_BASELINE.md`.

## 2. Actor masks

`pipeline/stages/person.sh EP`:

1. `scripts/detect_person_prompts.py` (GroundingDINO image): "a person." on every 10th frame (boxes taller than 15 %
   of the frame), linked into tracks (greedy IoU > 0.3, short gaps bridged). The actor is the track with the lowest
   median distance to the object mask's box among tracks covering ≥ 20 % of the samples: in these clips the actor is
   the person handling the object, and background people (a seated person in episode 2, passers-by) never are.
   Box prompts every 30 frames where the actor was detected with confidence ≥ 0.35 (object id 0);
   `config/manual_person_prompts.json` can replace them.
2. SAM 2.1 propagates the prompts to every frame → `outputs/person/<ep>/sam2_masks/0/`.

Check against the masks of the previous architecture (actor chosen by a world-frame tracker plus reviewed track
overrides): per-episode median IoU 0.991–0.999 over all 30 episodes (10th percentile ≥ 0.955), and the same share of a
held-out 2D pose detector's keypoints inside the mask, so the automatic actor choice reproduces the reviewed one,
including the two clips that needed manual overrides before (episodes 2 and 15).

## 3. Human: SAM 3D Body

`pipeline/stages/sam3d_body.sh EP` (GPU):

0. `scripts/camera_intrinsics.py`: one static camera per episode, the median of MoGe-2's per-frame intrinsics →
   `outputs/camera/<ep>/intrinsics.json`.
1. `scripts/sam3d_body_video.py`: SAM 3D Body (DINOv3, body + hand decoders, `inference_type full`) on every frame,
   prompted with the box of the actor mask (nearest valid frame when a mask is empty) and the static intrinsics.
   Raw outputs are stored unchanged (`raw_<ep>.npz`).
2. `scripts/sam3d_body_identity.py`: one body per episode. Reliable frames have ≥ 14 of 17 body keypoints inside the
   (dilated) actor mask and a box clear of the image border; the identity (68 scales + 45 shape) is the medoid of the
   reliable frames' identities, i.e. one physically consistent body, not a per-dimension average. The MoGe/SAM depth
   scale s is the median of MoGe-2 depth / SAM front-surface depth over the body's pixels (every third frame, 4 px
   z-buffer cells, normalised by MoGe's per-frame wobble on static background). SAM's 2D body keypoints are exported
   to `outputs/keypoints2d_sam/<ep>.npz` (diagnostics).
3. `scripts/sam3d_body_refit.py` (MHR image): every frame's 136 pose parameters refitted with that identity (kit MHR
   wrapper, `lm_pose`).
4. `scripts/sam3d_body_hands.py` (MHR image): 512 hand points per frame (the PEN hand roles of
   `extract_mhr_hands.py`) → `outputs/hands/<ep>.npz`, and `outputs/alignment_sam/<ep>/alignment.json`
   (R = I, t = 0, depth_scale = s, static intrinsics), used by the hoop tracker and the refinement.

## 4. Object meshes: SAM 3D Objects candidates, one FoundationPose selection

`pipeline/stages/object_mesh.sh OBJECT` (GPU), once per object:

1. `scripts/sam3d_object_mesh.py` (SAM 3D Objects image): scores every third SAM 2 mask frame of each of the object's
   three episodes for size, compactness, border truncation and temporal stability and keeps the best view per episode
   (`views/candidates.jpg`; `config/manual_mesh_views.json` can override views as
   `{"object": [{"episode": E, "frame": F}]}`). SAM 3D Objects generates one candidate per view and seed (default
   seeds 1 and 2), conditioned on the RGB frame, the object mask and that frame's MoGe-2 depth as the scene point map.
   It is built for real, cluttered, occluded scenes and returns the mesh with its pose and scale in the camera, so the
   candidates are metric directly; each is cleaned (floaters dropped, decimated to 40k faces, centred) →
   `outputs/object_meshes/<object>/sam3d/candidates/cXX/coarse_metric.glb`.
2. `scripts/select_object_mesh.py` (CARI4D image): every candidate is registered with FoundationPose on each episode's
   selection frame over a log-scale grid of 0.75–1.33× its size, scored as mask IoU + 0.5 · exp(−median |depth ratio
   − 1|). The best mean over the three cameras wins; its scale is refined and the per-episode median applied →
   `inputs/object_meshes/<object>.glb`. A mesh already installed there is scored the same way first and recorded as
   `selection.reference` (comparison only). `SKIP_GENERATION=1` re-runs only this step.

Why the scale range is narrow: FoundationPose registration always places the mesh at the observed depth, so the
depth term is ≈ 1 at every scale, and silhouette IoU then rewards inflating an imperfect mesh until it covers the
mask. Why candidates from three cameras: single-view generation from small, dim, top-down exo crops is
semantically unstable, and three cameras reject shapes that only explain one view. Silhouettes from these top-down
cameras cannot see thickness, so a flattened mesh can score as well as a correct one (the old pink foam roll, 4 cm
thick, 0.921 vs 0.907 for SAM 3D Objects' 13 cm cylinder): the selection score is a 2D proxy, not a 3D check.

Generator choice (2026-10-17): on the bowl, pan and paint roller TRELLIS.2-4B candidates were pooled with SAM 3D
Objects' in the same selection. SAM 3D Objects won the pan (0.903 vs 0.886) and the roller (0.897 vs 0.840), TRELLIS.2
the bowl by 0.014; TRELLIS.2 was ~3× slower, returned empty structures for some views and mostly flat cut-outs (pan:
10 of 12 candidates 1 cm thick). It was dropped.

The SAM 3D Objects image needs `libusb` for open3d (`config/sam3d.def` installs it; for an image built before that,
see the note in `object_mesh.sh`).

## 5. Object tracking

`pipeline/stages/track_object.sh EP` (FoundationPose + IoU-weighted ESKF/RTS smoothing), as documented in
`docs/OBJECT_BASELINE.md`. Poses are object-to-camera in MoGe-2 metres.

Apptainer refuses `--env HOME=...` and keeps HOME on the unmounted, read-only host home; then `warp.init()` fails
silently and FoundationPose's `Utils.erode_depth` is never defined. `in_container` in `pipeline/env.sh` therefore
sets `HOME=/tmp` inside the container.

### Symmetric objects: the hula hoop

FoundationPose cannot track the thin, textureless, rotationally symmetric hoop: its raw track has 72–126
single-frame jumps over 10 cm per episode and 40–90° axis flips, and the RTS smoother removes them by nearly freezing
the hoop. The hoop's spin about its own axis is invisible to CD-O and ACC-O, so what matters is the ring centre and
axis. For objects in `config/symmetric_objects.json`:

1. `scripts/make_round_mesh.py` replaces the generated mesh by an exactly round torus of the same size
   (`<object>.symmetric.glb`).
2. `pipeline/stages/track_symmetric.sh` → `scripts/track_symmetric_object.py` (CPU): the state is the ring centre and
   the axis line. Mask ellipse fits with the known radius propose depth and both tilt hypotheses; a particle filter
   weights particles by a symmetric 2D chamfer between the projected ring and the SAM 2 mask (trimmed for occlusion)
   and a hand-contact prior from the SAM 3D Body hand points; a global sparse least-squares pass adds acceleration
   smoothness, 40× smoothness and zero velocity on frames where the mask shows the hoop at rest, and an anchor to the
   FoundationPose track on those resting frames.

## 6. Assembly, refinement and packing

`pipeline/stages/pack.sh`:

1. `scripts/build_track1_episodes_sam.py`: world = the static camera in SAM 3D Body's metric. Human = the refit;
   object_rotation = R_co, object_translation = t_co / s, object_scale = s_co / s; the hoop centre is smoothed
   (λ 100, weight 40 on resting frames) → `submissions/episodes_sam/`.
2. Depth trend (`DEPTH_TREND=1`, default; `DEPTH_TREND_ALPHA` 0.75), `scripts/sam3d_body_depth_trend.py`: the
   scorer re-anchors every episode at its first scored frame, so only depth error that changes over time costs
   CD-H. The background-normalised MoGe/SAM depth ratio of the body keeps only its trend (median 15 frames,
   Gaussian σ 30) and moves the MHR root along the camera ray by α·(trend − 1)·centroid → `_depth/`.
3. Human refinement (`HUMAN_REFINE=1`, default, GPU; `REFINE_LAM` 300; `pipeline/stages/human_refine.sh`):
   - `scripts/human_refit.py` (MHR image): MHR Euler parameters wrap by 2π, so the motion is smoothed on the mesh
     (per-vertex `argmin |X − V|² + λ|D²X|²`) and every frame's pose parameters are refitted to it (identity fixed).
   - `scripts/grasp_windows.py`, `scripts/hand_anchors.py`: where the hand-object relative pose is stable
     (< 5 cm, < 20°) the hand's mean shape in the object frame is re-posed with the object track and blended in/out
     over 15 frames. FoundationPose is 5–30× smoother than the hand-carried pose, so the object is never pulled by
     the hand.
   - `scripts/wrist_fusion.py`: per hand, a constant-velocity RTS smoother of a camera-frame wrist offset; MediaPipe
     wrists observe the lateral components, object anchors all three, no 2D update while the hand is within 5 cm of
     the object; residuals > 40 px dropped, offsets capped at 10 cm.
   - `scripts/root_lateral.py`: an RTS smoother of the body's lateral offset from 12 MediaPipe joints, applied only
     where the correction also moves those joints deeper into the actor's SAM 2 mask (`scripts/person_mask_sdf.py`:
     signed distance to the mask boundary; mean over scored frames must drop by > max(0.3 px, 5 %)). The body's own
     2D keypoints cannot gate this: they agree with the uncorrected body by construction. On SAM 3D Body the
     MediaPipe offsets are 0.3–1.6 cm (median per episode) and no episode passes, so the step is a safeguard: unlike
     a world-frame tracker, a per-frame camera-frame estimate does not drift sideways.
   Sequence: λ 10 refit (process model) → wrist fusion → λ 300 refit with fused wrists and anchored hands → λ 300
   second pass → lateral root fix → `_refine/`.
4. Hand-side contact fix (`HAND_FIX=1`, default, GPU; `pipeline/stages/hand_fix.sh`): PEN counts hand points inside
   the object, and most of that depth is in the fingers of grasps, which no object translation can clear.
   `scripts/hand_contact_fix.py` refits only the pose parameters that move hand vertices but none of the 22 joints
   scored by ACC-H (fingers, wrist rotations; 60 free columns), pushing hand vertices out of the scorer-budgeted
   object mesh (`scripts/budget_object_meshes.py`), two passes with 4 and 6 mm margins. Objects and scored joints are
   exactly unchanged → `_handfix/`.
5. Optional object-side push (`RESOLVE_PEN=1`, off): `scripts/extract_mhr_hands.py` + `scripts/resolve_hand_penetration.py`
   translate the object away from the hand on penetrating frames. Off because it costs object smoothness (ACC-O).
6. The official `eval_reconstruction.py` packs and validates the Track 1 Parquet. `COMMIT_URL` must be the GitHub
   commit of the code that produced the run; without it a local validation file is written.

## Status

Measured state of this architecture (2026-10-17, local checks only; the reference is held out, scores come from
Kaggle). Reference: the previous architecture's best Kaggle-scored file (SAM 3D Body with person boxes from a
world-frame tracker's 2D keypoints, the earlier generated meshes, the same refinement and hand fix).

| build | self-PEN (cm) | 2D vs ViTPose (px) | wrist 2D (px) | flips | jitter, 22 joints (cm/frame²) | object accel (cm/frame²) |
|---|---|---|---|---|---|---|
| previous architecture | 0.00453 | 5.29 | 5.74 | 0 | 0.0454 | 0.0445 |
| this pipeline, earlier meshes | 0.00457 | 5.32 | 5.65 | 0 | 0.0449 | 0.0415 |
| **this pipeline (SAM 3D Objects meshes)** | **0.00124** | 5.34 | 5.67 | 0 | 0.0456 | **0.0406** |

- self-PEN: the scorer's penetration with alignment scale 1 (`scripts/extract_mhr_hands.py` hand points, winding
  number in the budgeted mesh); it tracked Kaggle PEN at about 0.86x on earlier builds. 2D: the MHR joints projected
  with the static intrinsics against held-out 2D detections. Jitter: mean |second difference| of the 22 scored joints.
- Object meshes (nine non-symmetric objects): mean three-camera silhouette IoU after FoundationPose registration
  0.847 → 0.886; posed silhouette IoU of mesh + track over all tracked frames 0.717 → 0.761, frames below 0.5
  13.2 % → 8.1 %; the pan now has its handle (0.38 m; 0.841 vs 0.777 on handle-inclusive masks). The hoop torus
  from SAM 3D Objects has ring radius 0.335 m (earlier 0.332 m).
- Ablation (SAM 3D Objects build): raw assembly self-PEN 0.0030, 2D 4.76 px, jitter 1.88; the refinement brings
  jitter to 0.046 at +0.6 px 2D; without the depth trend the refined build is slightly better locally (2D 5.21 vs
  5.34 px, self-PEN 0.0026 vs 0.0027); whether the depth trend helps CD-H needs a Kaggle comparison.

Known limits: no local CD-H/CD-O; top-down silhouettes cannot see thickness; objects resting on the floor sit
8–14 cm below the floor the body stands on (the MoGe/SAM depth scale is measured on the body); the paint roller,
foam roll and stool track slightly worse with the new meshes (−0.005 to −0.024 IoU); the hand fix takes about three
hours on six GPUs.
