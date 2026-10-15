# Track 1 baseline status

Verified on 2026-10-15 in `/work/smoses2s/nvidea_chal/TRACK1`.

## Ready

- Official Track 1 release: 30 episodes, 16,563 video frames.
- CARI4D commercial checkpoint: downloaded and inventoried.
- MoGe-2 depth and camera-intrinsics estimates: complete for all 30 episodes.
- GroundingDINO + SAM 2 object-mask pipeline: complete for all 30 episodes, with manually
  reviewed prompt overrides for detector failures.
- FoundationPose runtime, episode jobs, and trajectory smoothing: configured.
- WHAM and Meta MHR source trees: pinned locally.
- Meta MHR converter container: built and CUDA/import validated.
- WHAM inference, dense SMPL export, SMPL-to-MHR fitting, and 30-episode Slurm submission
  helpers: configured.
- Licensed neutral SMPL model and derived face array: installed locally and validated.
- DPVO CUDA extensions: rebuilt and import-checked for V100 (`sm_70`) and A100 (`sm_80`).
- End-to-end human smoke test: episode 0 passed with 790/790 observed WHAM frames and an MHR
  topology-fit mean error of 4.532 mm; no invalid frames or cold refits.
- Human track complete: WHAM and WHAM-to-MHR finished for all 30 episodes
  (`outputs/human/episode_*/mhr_human.npz`).
- Human/camera joint alignment complete for all 30 episodes (`outputs/alignment/`): static-camera
  PnP with MoGe-2 intrinsics, low-frequency WHAM drift correction, and MoGe/SMPL depth scale.
  Drift-corrected median reprojection 5.4–17.3 px; depth scale 0.83–0.95. Review episodes 24 and 8
  first (largest residuals and >1.2 m drift corrections).
- Official packer format test passed with real human parameters and a placeholder object:
  740,780 rows (`tmp/pack_test/format_test_do_not_upload.parquet`, never upload).
- FoundationPose startup bug fixed: Apptainer ignored `--env HOME=/tmp`, so warp could not create
  its cache and `Utils.erode_depth` was undefined. CARI4D-image calls now set HOME in-container.

## Generated object meshes (installed 2026-10-15)

The public release has no object meshes. `slurm/18_trellis_object_mesh.slurm` generated TRELLIS
candidates per object and installed the one FoundationPose fits best across the object's three
cameras (scale limited to 0.75–1.33× the MoGe-2 depth extent) as `inputs/object_meshes/<object>.glb`
with a `.provenance.json`. Generated estimates, not ground truth. Visual check:
`outputs/object_meshes/review_sheet.png`.

| Object | Winner | Mean IoU (3 cameras) | Extents (m) | Note |
|---|---|---|---|---|
| big_red_bowl | c05 | 0.93 | 0.28 × 0.13 × 0.38 | thin wire artifact |
| black_pan | c04 | 0.79 | 0.21 × 0.02 × 0.19 | flat disc, no handle |
| foam_grass_block | c04 | 0.96 | 0.19 × 0.20 × 0.19 | |
| hula_hoop | c03 | 0.60 | 0.68 × 0.03 × 0.70 | |
| iron | c00d | 0.76 | 0.19 × 0.35 × 0.09 | REVIEW: scale at 1.33× bound |
| paint_roller | c05 | 0.88 | 0.24 × 0.07 × 0.07 | |
| pink_foam_roll | c02 | 0.92 | 0.59 × 0.11 × 0.04 | flattened, not a cylinder |
| short_wood_stool | c07d | 0.80 | 0.43 × 0.48 × 0.27 | multi-view, de-plated |
| white_desk | c02d | 0.81 | 0.94 × 0.82 × 0.55 | |
| white_laptop_cart | c05d | 0.78 | 0.54 × 0.77 × 0.42 | |

FoundationPose tracking + RTS smoothing: complete for all 30 episodes
(`slurm/11_foundationpose_episode.slurm`).

## Packed baseline (local validation)

`slurm/20_pack_track1.slurm` assembled all 30 episodes (`submissions/episodes_wham_fp/`) and the
official packer wrote `submissions/track1_LOCAL_CHECK_do_not_upload.parquet` (740,780 rows, no
mesh-quality warnings). It carries a placeholder commit URL and must not be uploaded.

To produce the uploadable file: push this code to GitHub, then
`sbatch --export=ALL,COMMIT_URL=https://github.com/<owner>/<repo>/commit/<sha> TRACK1/slurm/20_pack_track1.slurm`,
then upload with `submit/v2d_submit.py <parquet> --track track_1` after accepting the rules of all
five Track 1 Kaggle competitions. Both steps are the user's decision. No local score exists: the
reference poses and meshes are held out and scored only on Kaggle.

Optional hand-penetration step (`RESOLVE_PEN=1`) is off: self-measured PEN is already 0.013 cm
mean, and the push raised object acceleration 3–14× on the three affected episodes.
See `docs/JOINT_BASELINE.md`.

## CARI4D integration boundary

WHAM to MHR produces the Track 1 submission fields (`pose`, `scales`, and `shape`) without the
gated SAM 3D Body estimator. It is not currently a drop-in `mhr-init` for CoCoNet. The accepted
CARI4D checkpoint contains CoCoNet delta heads, but not the SAM 3D Body decoder buffers required
by CARI4D's `MHRParametricBackend`. Therefore:

- a valid direct WHAM-to-MHR human submission path is configured;
- CoCoNet refinement still requires the missing decoder assets or a separately verified public
  MHR-backend adapter;
- no result should be described as a completed CARI4D/CoCoNet run until that boundary is solved.

The latest object report is `outputs/object/BASELINE_STATUS.json`. Its completeness check passes
all 30 episodes. Episodes 4, 7, 18, and 20 retain temporal-review flags because their small or
occluded objects have empty masks in some frames; they are not missing output files.

## Validation commands

```bash
cd /work/smoses2s/nvidea_chal
python3 TRACK1/scripts/summarize_object_outputs.py
python3 TRACK1/scripts/validate_object_setup.py
python3 TRACK1/scripts/validate_wham_setup.py
python3 TRACK1/scripts/validate_mhr_setup.py
```

See `docs/OBJECT_BASELINE.md`, `docs/WHAM_BASELINE.md`, and `docs/JOINT_BASELINE.md` for
execution details.
