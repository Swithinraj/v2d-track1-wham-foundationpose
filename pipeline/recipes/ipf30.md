# Candidate IPf30: contact-respecting human on Afr2b's objects

Final step, executable from this commit:

    pipeline/stages/contact_solve.sh episodes_repro_depth_refine episodes_cand_Afr2b episodes_cand_IPf30
    (cd official/v2d_submission_kit && python3 eval_reconstruction.py --episodes ../../submissions/episodes_cand_IPf30 \
        --sample data/track_1_sample_submission.parquet --commit <this commit's URL> --out <parquet>)

Inputs are pinned by SHA256 in `ipf30_inputs.sha256`:

- **Human**: `episodes_repro_depth_refine`. This is the scored 10-17 human: SAM 3D Body with a shared identity, the
  MoGe depth trend and the human refinement at lambda 300.

      REFINE_LAM=300 TRACK1_ALIGNMENT_DIR=outputs/alignment_sam \
        pipeline/stages/human_refine.sh episodes_repro_depth episodes_repro_depth episodes_repro episodes_repro_depth_refine

  It reproduces the scored human to 0.01 cm.
- **Objects**: object poses, scale and meshes of `episodes_cand_Afr2b`, Kaggle-scored on 2026-10-19 (rank sum 33).
  The solver copies them unchanged.

## How Afr2b's objects were built

Steps in order, with the scripts in this commit:

1. **Candidate A.** The scored 10-17 objects from SAM 3D Objects meshes and FoundationPose tracks.
   - `scripts/kappa_gravity.py`: gravity kappa. The objects are scaled so the person's soles meet the MoGe floor.
   - `scripts/physics_objects.py --no_rest --no_gravity --lam 10`: lambda-10 object smoothing.
2. **Object pose fit, pass 1.**
   - `scripts/object_pose_fit.py --input episodes_cand_A --episodes 8 10 11 12 13 14 17 21 25 27 28` with
     OBJFIT_SIGMA=10.
   - Then `scripts/physics_objects.py --no_rest --no_gravity --lam 10` on those episodes, giving A_of10.
3. **Paint roller (episodes 18-20).**
   - `scripts/roller_masks.py 18 19 20`: handle masks from SAM 2 and CoTracker3, with
     `config/roller_manual.json` for episode 19.
   - SAM 3D Objects run on those masks, giving the roller mesh.
   - `scripts/roller_compose.py --name Arc3 --w_smooth 0.3 --mesh <roller mesh> --bodies <mhr_bodies of the human>`:
     the cylinder track with the attached handle, and the handle's roll per frame.
4. **Assembly (Afr).** The objects of A_of10, plus the roller episodes from Arc3.
5. **Object pose fit, pass 2.** `scripts/object_pose_fit.py --input episodes_cand_Afr --episodes 0 8 9 12 13 14 25`
   with OBJFIT_SIGMA=10 and IoU acceptance, without re-smoothing.
6. **Afr2b.** Afr2 with the hoop (episode 0) taken from Afr.

Steps 1-6 are documented but were not re-executed from this commit. Their output is pinned by hash.

## Settings of the final step

`human_contact_solver.py`:

    --w_prior_arm 2000 --w_img 20 --w_cap 1000 --w_con 200000 --w_rig 20000 --nsub 0

All other weights are at their defaults: Adam lr 0.01, 200 iterations, sigma_arm 6, sigma_hand 2, sigma_img 2 px,
depth cap 5 cm.

Signed-distance grids come from `contact_sdf.py` (N = 112) on Afr2b's meshes.

Validation of the submitted file is done on the final MHR after all processing:

- scorer-style self-PEN;
- contact gaps;
- MediaPipe and ViTPose 2D;
- jitter;
- displacement of the scored joints and the first scored frame.

## Determinism

The solve is not bit-reproducible on GPU. Atomic additions in the MHR skinning backward pass let weakly
constrained finger parameters land differently between runs. Episode metrics stay within noise: two full runs gave
self-PEN 0.00060 and 0.00059, contact gap 18.0 and 18.1 mm, and identical held-out 2D. The submitted parquet is the
one produced by the recorded run (see its provenance JSON).

## Version IPfG30

Same as IPf30, with `CONTACT_ARGS` adding `--free_fingers_in_grasp`.

In stable-grasp windows the finger landmarks are not held to the image: fingers wrapped around an object are
occluded, so contact and rigidity decide them. Wrist and elbow stay held.

On all 30 episodes, two runs each:

| | IPf30 | IPfG30 |
|---|---|---|
| self-PEN | 0.00060 / 0.00059 | 0.00054 / 0.00054 |
| contact gap | 18.0 / 18.1 mm | 18.0 / 17.9 mm |
| stable-grasp drift | 2.6 mm/frame | 2.3 mm/frame |
| held-out ViTPose | 5.374 px | 5.380 px |

## Version Mw (minimal hand displacement)

Kaggle showed that reshaping the hands costs CD-H. Mean dense-hand displacement against the scored human tracked
CD-H on every scored file:

| File | hand displacement | Kaggle CD-H |
|---|---|---|
| no hand fix | 0 | 14.668 |
| 10-17 hand fix | 0.15 cm | 14.670 |
| Afr2b | 0.65 cm | 14.764 |
| IPf30 | 1.10 cm | 14.935 |

This version keeps the dense hand vertices where SAM 3D Body put them. It uses a metric prior at a 2 mm scale, drops
the MediaPipe hand-landmark term (11-27 px errors in grasps) and solves only frames with penetration.

    CONTACT_ARGS="--w_prior_arm 2000 --w_img 20 --w_cap 1000 --nsub 0 --w_2d_hand 0 --pen_frames_only \
                  --w_con 200000 --w_rig 20000 --w_hdisp 0.3"

Local result on all 30 episodes:

| Measure | Mw | Afr2b |
|---|---|---|
| self-PEN | 0.00069 | 0.00084 |
| hand displacement | 0.30 cm | 0.65 cm |
| contact gap after | 15.9 mm (base 19.9 mm) | — |

## Version AFw (Afr2b's human, first scored frame frozen)

Kaggle also showed that the human at each episode's first scored frame moves CD-O through the scorer's single
Sim(3). Three files with byte-identical objects scored CD-O 18.739 (Afr2b's human), 18.829 (Mw) and 18.871 (IPf30).

This version therefore starts from Afr2b's human (hand fix and arm clearance) and leaves the first scored frame
unchanged: no correction within ±2 frames, then a 15-frame ramp. The scorer's alignment is identical to Afr2b's, and
the minimal-displacement solve acts elsewhere.

    CONTACT_ARGS="--w_prior_arm 2000 --w_img 20 --w_cap 1000 --nsub 0 --w_2d_hand 0 --pen_frames_only --freeze_anchor \
                  --w_con 200000 --w_rig 20000 --w_hdisp 0.3"
    pipeline/stages/contact_solve.sh episodes_cand_Afr2b episodes_cand_Afr2b episodes_cand_AFw

Local result:

| Measure | AFw | Afr2b |
|---|---|---|
| self-PEN | 0.00038 | 0.00084 |
| contact gap | 20.3 → 16.7 mm | — |
| pose change at the first scored frame | 0 | — |
