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
