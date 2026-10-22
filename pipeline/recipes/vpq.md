# Candidate VPQ: MediaPipe-free human, smoothed arm correction, bounded roller roll

Starts from AFw (Kaggle-scored 2026-10-20) and changes three things:

1. **Human 2D evidence without MediaPipe.** The human refinement and the contact solve read tracked ViTPose-H body
   landmarks instead of MediaPipe. ViTPose runs top-down on the actor's SAM 2 track, so the landmarks belong to the
   actor by construction. No hand landmarks are used.
2. **Smoother arm correction.** The contact solve's arm correction is low-passed in time (Gaussian, sigma 4 frames).
   A hand-only re-solve then restores non-penetration without moving any scored joint. The first scored frame stays
   exactly the solved pose.
3. **Objects.**
   - The six episodes refitted in object pose fit pass 2 (8, 9, 12, 13, 14, 25) get the lambda-10 translation
     smoothing that pass 1 had. Pass 2 was not re-smoothed.
   - The paint roller (episodes 18-20) gets its handle roll re-decoded with handle-only evidence and a bounded angular
     velocity (QH).

Final steps, executable from this commit:

    pipeline/recipes/vpq_final.sh episodes_cand_VPQ
    (cd official/v2d_submission_kit && python3 eval_reconstruction.py --episodes ../../submissions/episodes_cand_VPQ \
        --sample data/track_1_sample_submission.parquet --commit <this commit's URL> --out <parquet>)

Inputs are pinned by SHA256 in `vpq_inputs.sha256`:

- `submissions/episodes_repro_depth` and `submissions/episodes_repro`: the human inputs of the scored 10-17 human
  (see `ipf30.md`).
- `outputs/kp2d_vpx`: the ViTPose landmarks (steps 1-3 below).
- `submissions/episodes_objects_vpq`: the objects (step 4 below).

## How the pinned inputs were built

Steps 1-4 use the scripts in this commit. Their outputs are pinned by hash.

1. **Tracked ViTPose-H keypoints.**
   - Script: `scripts/vitpose_track.py` (v2d_sam3d_body.sif, GPU), writing `outputs/vitpose`.
   - Checkpoint: `weights/vitpose/vitpose-h-multi-coco.pth`, the ViTPose-H multi-COCO checkpoint of the ViTPose model
     zoo (SHA256 `50e33f40...80bc`).
2. **Human refined without 2D landmarks.** This human is used only to calibrate the hip offset in step 3:

       cp -a submissions/episodes_repro_depth submissions/episodes_rdn
       HUMAN_KP2D_DIR=none REFINE_LAM=300 TRACK1_ALIGNMENT_DIR=outputs/alignment_sam HUMAN_REFINE_WORK=human_refine_nomp \
         pipeline/stages/human_refine.sh episodes_rdn episodes_repro_depth episodes_repro episodes_rdn_refine_nomp

   Then `scripts/mhr_bodies.py` is run on `episodes_rdn_refine_nomp`, giving the bodies.
3. **Landmark file.**

       scripts/kp2d_vitpose.py --vitpose outputs/vitpose --bodies <bodies of step 2> --out outputs/kp2d_vpx

   It maps COCO-17 to MediaPipe's 33-landmark layout. It shifts each hip by one constant per episode: the median
   offset between the projected MHR hip joint and the ViTPose hip.
4. **Objects** (`episodes_objects_vpq`). Start from AFw's objects (`episodes_cand_AFw`), then:
   - Episodes 8, 9, 12, 13, 14 and 25: translation from
     `scripts/physics_objects.py --no_rest --no_gravity --lam 10` on AFw. This solves (I + 10 D2ᵀD2) x = y on the
     translation only, and leaves the rotation untouched.
   - Episodes 18-20: rotation from

         scripts/roller_roll_qh.py --cand_a submissions/episodes_cand_A --base submissions/episodes_cand_AFw \
           --mesh <roller mesh of roller_compose.py> --meshes <budget_object_meshes.py output of AFw> --out <dir>

     It uses the default settings: handle term on frames with at least 20 handle pixels at half resolution, weight 1;
     quadratic prior 0.05 per (10 deg)²; at most 30 deg per frame. The translation is unchanged: the roll axis passes
     through the cylinder centre.
   - All other arrays and meshes: AFw's.

## Reproduction checks of the ports

These scripts first ran as experiment code. Each port was checked against the outputs this candidate used:

| Script | Check | Result |
|---|---|---|
| `kp2d_vitpose.py` | all 30 episodes | 0 arrays differ |
| `physics_objects.py --no_rest --no_gravity --lam 10` | same linear system as the experiment's smoother | same equation |
| `roller_roll_qh.py` | AFw base | bit-identical: rotation difference 0 and decoded roll path 0 frames different in episodes 18-20 |
| `vitpose_track.py` (v2d_sam3d_body.sif) | episodes 0 and 18 | mean 0.07 px, 99th percentile 2.9 / 3.8 px; not bit-identical (see note) |
| `smooth_arm_correction.py`, `combine_human_objects.py` | code identical to the experiment scripts | renamed arguments only |
| `human_contact_solver.py --kp2d_dir` | only change | path of the landmark file |
| `human_refine.sh` (`HUMAN_KP2D_DIR=none`) | same steps as the experiment's no-MediaPipe refinement | same steps |

Note on `vitpose_track.py`: the remaining differences are single heatmap-step changes caused by fp16 arithmetic in
a different torch build. One low-confidence joint in episode 0 lands elsewhere (81 px). The candidate uses the
original tracks, pinned through `outputs/kp2d_vpx`.

## Settings of the final steps

- Contact solve, both passes: AFw's settings with the ViTPose landmarks.

      --w_prior_arm 2000 --w_img 20 --w_cap 1000 --nsub 0 --w_2d_hand 0 --pen_frames_only --freeze_anchor
      --w_con 200000 --w_rig 20000 --w_hdisp 0.3 --kp2d_dir outputs/kp2d_vpx

  The second pass adds `--no_arm`.
- `smooth_arm_correction.py --sigma 4`.

As with IPf30 (see `ipf30.md`, Determinism), the GPU solve is not bit-reproducible. The submitted parquet is the one
produced by the recorded run.

## Local validation

All local measures are on the final MHR after all processing, over the scored frames of all 30 episodes.

| Measure | VPQ | AFw | SMSD4H (AFw + smoothing + QH + object smoothing) |
|---|---|---|---|
| scorer-style self-PEN | 0.000357 | 0.000379 | 0.000306 |
| object acceleration (ACC-O stand-in) | 0.034964 | 0.035134 | 0.034964 |
| scored-joint self-acceleration (ACC-H stand-in) | 0.05186 | 0.05266 | 0.05253 |
| dense-hand displacement from AFw | 1.16 cm | 0 | 0.08 cm |
| scored-joint change from AFw: all frames / first scored frame | 0.144 / 0.113 cm | 0 | 0.005 / 0 cm |
| ViTPose 2D | 5.479 px | 5.506 px | 5.507 px |

The ViTPose 2D row is not independent for VPQ: ViTPose is now one of the refinement's inputs.

Execution of the recorded run:

- Recipe steps 1-2 ran in one job.
- The hand fix ran split by episode, using the same commands (as `hand_fix.sh` allows).
- The remaining recipe lines ran verbatim from this commit's `vpq_final.sh` on a clean tree.

The packed parquet equals a fresh repack byte for byte. Packed file:
`candVPQ_d724234.parquet`, SHA256 `7cb003ee...7d3b`.

## Kaggle (2026-10-21)

| Board | VPQ | AFw | Change |
|---|---|---|---|
| CD-H | 14.51732 | 14.76807 | −0.251 |
| CD-O | 18.40107 | 18.73858 | −0.338 |
| ACC-H | 0.13942 | 0.13981 | −0.00039 |
| ACC-O | 0.07139 | 0.07147 | −0.00008 |
| PEN | 0.00051 | 0.00049 | +0.00002 |

The ACC-H change matched the local stand-in: Kaggle moves by about half the local scored-joint change, and the
predicted value was 0.1394.

CD-H improved even though the hands moved 1.16 cm from AFw's. The hand-displacement stand-in applies only to reshaping
the same human. A human rebuilt from better 2D evidence can move the hands toward the ground truth.

CD-O improved as well, through the scorer's single Sim(3) on the human at the first scored frame (0.113 cm change
there).

## Version VPQ2: depth-axis temporal refinement and stronger object smoothing (before the contact stage)

    pipeline/recipes/vpq2_final.sh episodes_cand_VPQ2      # inputs pinned in vpq2_inputs.sha256

**Human depth.** A monocular body estimate is much noisier along the camera ray than across it:
- Raw pelvis residuals in the camera frame have a median σz²/σxy² of 13.4, about 3.7 times the noise in std.
- The refit smooths all axes with LAM 300, and 2D evidence cannot see depth.

`depth_smooth_root.py` estimates that ratio per episode from the raw human. It adds a depth-axis Whittaker smoothing of
LAM_add = 300·(r² − 1), with a median of 3,704, keeps the first scored frame pinned, and moves the body along each frame's
camera ray through the pelvis. The pelvis projection is therefore unchanged. Mean depth change: 0.50 cm.

**Objects.** Kaggle ACC-O fell by 0.39-0.47 times the local object self-acceleration drop every time only object
smoothing changed. Our object acceleration therefore behaves like noise unrelated to the ground-truth motion, so
translations get λ 100 (`physics_objects.py --no_rest --no_gravity --lam 100`). Rigid objects move 0.02-0.20 cm.

**Contact.** Both steps come before VPQ's contact stage, which restores non-penetration: applied after it, the depth step
doubled PEN.

**Port check.** The ported `depth_smooth_root.py` reproduces the recorded run to float32 rounding (at most 0.0004 mm,
from the BLAS thread count).

Local results:

| Measure | VPQ2 | VPQ |
|---|---|---|
| self-PEN | 0.000342 | 0.000357 |
| ep27 self-PEN | 0.00387 | 0.00432 |
| object self-acceleration | 0.03172 | 0.03496 |
| scored-joint self-acceleration | 0.04899 | 0.05186 |
| ViTPose 2D | 5.487 px | 5.479 px |
| first scored frame | unchanged | unchanged |

Kaggle (2026-10-22):

| Board | VPQ2 | VPQ | Change |
|---|---|---|---|
| CD-H | 14.51720 | 14.51732 | −0.0001 |
| CD-O | 18.39830 | 18.40107 | −0.0028 |
| ACC-H | 0.13935 | 0.13942 | −0.00007 |
| ACC-O | 0.07073 | 0.07139 | −0.00066 |
| PEN | 0.00050 | 0.00051 | −0.00001 |

**Object smoothing worked.** ACC-O moved by 0.20 times the local object self-acceleration drop (λ100). This is below
the 0.39-0.47 seen at smaller λ steps, so the returns diminish.

**The depth step did not work for ACC-H.** It removed 5.5 % of the local scored-joint self-acceleration, which the
calibration predicted as −0.0012, but Kaggle moved by only −0.00007. The depth motion that remains after the LAM 300
refit is therefore mostly real motion that the ground truth shares; the noise ratio was measured on the raw human,
before the refit. The local ACC-H stand-in holds for articulation changes but not for root depth. The step is harmless:
the other four boards are unchanged or slightly better.

## Version VPQ3: restored image-plane motion (lateral band) before the contact stage

    pipeline/recipes/vpq3_final.sh episodes_cand_VPQ3      # inputs pinned in vpq3_inputs.sha256 (+ vpq_inputs.sha256)

**Why.**
- Triple collocation of 2D accelerations shows real visible motion at 1-2 Hz with raw SNR 4-30. The three sources are
  per-frame SAM 3D Body projected, tracked ViTPose and MediaPipe.
- The λ300 refit cuts image-plane motion at about 1 Hz.
- The ACC-H ground truth keeps that motion: it is a multi-view SAM 3D Body fusion with a first-order temporal prior, and
  its acceleration (0.116-0.134 cm/f² in the kit documentation) matches the visible motion.
- Root-depth smoothing did not transfer to Kaggle (VPQ2), whereas lateral content is mostly real.

**What.**
- Per vertex, in image coordinates: target = VPQ2 + w · [S_100 − S_300](raw u, v) · z.
- Depth is kept and the first scored frame is pinned (weight tapers to 0 over ±10 frames; anchor pose row copied).
- Evidence weight w per frame and part, upper body only:

      w = (clip((ViTPose conf − 0.5)/0.3) × exp(−½ (d / 3·1.4826·MAD)²) × mask visibility, Gaussian σ = 2)²

  with d = |raw 2D − ViTPose 2D| after removing its per-episode median. Mask visibility means inside the actor's SAM 2
  mask and outside the object's mask. No MediaPipe in the build.
- The pose is re-fitted with lm_pose, then VPQ's contact stage runs.

**Local validation on the finished output vs VPQ2.**
- Proxy: noise-cancelling 2D acceleration proxy, E|a_c − a_D|² − E|a_q − a_D|² = MSE_c − MSE_q. MediaPipe is the primary
  reference; it is not a build input.
- Upper body:
  - MediaPipe reference: ΔMSE −9.83e-4 (SE 1.74), 29/30 episodes better;
  - ViTPose reference: −9.24e-4, 29/30 episodes better.
- Self-acceleration rises +14 % on confident frames and +15 % on the others (gate: no more than confident + 5 points).
- self-PEN 0.000330 (VPQ2 0.000342), ep27 0.00376.
- Objects identical; anchor unchanged.
- MediaPipe wrist 2D 6.10 px (VPQ2 6.25).
- Planning value (council critic): ACC-H −0.0008 (range −0.0003 to −0.0013).

**Port checks.**
- `mask_visibility.py` reproduces the run's visibility exactly (0 of 30 episodes differ).
- `lateral_band_refit.py` reproduces the run's pre-contact poses to 7.3e-6 (float32 rounding).
