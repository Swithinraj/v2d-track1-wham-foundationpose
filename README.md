# NVIDIA V2D Challenge — Track 1 workspace

This directory is the isolated workspace for **Track 1: monocular 4D human-object reconstruction**.
All data, environments, weights, container storage, logs, and generated artifacts stay below
`/work/smoses2s/nvidea_chal/TRACK1`.

The concise, evidence-backed readiness and blocker list is in [`STATUS.md`](STATUS.md).

## Baseline selected

The current NVIDIA toolkit contains the official `v2d_cari4d` implementation. Its released
pipeline is:

1. MoGe 2 depth initialization.
2. SAM 3D Body human initialization in MHR format.
3. FoundationPose object registration and tracking.
4. CoCoNet human/object refinement.
5. Full-clip contact-guided refinement.

The challenge leaderboard labels the Track 1 reference entry **CARI4D (commercial) Baseline**.
The current toolkit does not define HFL-4D or WHAM as the Track 1 baseline. Do not mix the two
descriptions when recording results.

An additional WHAM + Meta MHR human-motion branch is configured under `third_party/WHAM` and
`third_party/MHR`. It bypasses the SAM 3D Body Hugging Face gate and converts WHAM's temporal
SMPL sequence into the exact MHR parameters required by Track 1. It still needs the separately
licensed neutral SMPL model file required by WHAM and by the converter. See
[`docs/WHAM_BASELINE.md`](docs/WHAM_BASELINE.md) for the exact boundary and run commands.

## Workspace layout

```text
TRACK1/
├── config/                 path and runtime configuration
├── data/track_1/           official public Track 1 dataset (30 episodes)
├── docs/                   local operating notes
├── inputs/                 masks and metric object meshes prepared per episode
├── logs/                   Slurm stdout/stderr
├── official/               extracted official submission kit
├── outputs/                CARI4D reconstructions
├── scripts/                idempotent setup and validation helpers
├── slurm/                  WR-cluster jobs; heavy work runs here
├── submissions/            packed Track 1 parquet files and receipts
├── third_party/            pinned upstream source checkout
└── weights/                persistent model weights
```

## Current official contracts

- Input dataset: 30 third-person H.264 videos, 16,563 frames total, 1536x1152 at 30 fps.
- The public dataset contains videos and metadata only. It does **not** contain human pose,
  object pose, masks, or object meshes.
- CARI4D requires an RGB video, packed human/object masks, and a correctly scaled object mesh.
- A Track 1 submission contains per-episode MHR parameters, object pose/scale, and an object mesh.
- Metrics are CD-H, CD-O, ACC-H, ACC-O, and PEN; all are lower-is-better.

See [`docs/BASELINE.md`](docs/BASELINE.md) for the input gaps and run sequence.

The non-gated object branch (GroundingDINO, SAM 2, MoGe-2, FoundationPose, and trajectory
smoothing) is documented in [`docs/OBJECT_BASELINE.md`](docs/OBJECT_BASELINE.md). The released
Track 1 data has held-out ground truth and does not include the object CAD meshes required by
FoundationPose.

The non-gated human setup is submitted with `bash TRACK1/scripts/submit_human_setup.sh`.
After placing `SMPL_NEUTRAL.pkl` at the documented path and passing both validators, submit all
30 WHAM→MHR episodes with `bash TRACK1/scripts/submit_human_pipeline.sh`.

## Setup and run order

```bash
cd /work/smoses2s/nvidea_chal

# 1. Extract the official kit and create Python environments.
bash TRACK1/scripts/bootstrap_workspace.sh

# 2. Download only Track 1 of the public dataset.
bash TRACK1/scripts/download_track1_data.sh
python3 TRACK1/scripts/validate_workspace.py

# 3. Validate cluster/container support without launching a real baseline.
sbatch --test-only TRACK1/slurm/00_probe_runtime.slurm
sbatch TRACK1/slurm/00_probe_runtime.slurm

# 4. After the probe succeeds, build the CARI4D image through Slurm.
sbatch --test-only TRACK1/slurm/01_build_baseline.slurm
sbatch TRACK1/slurm/01_build_baseline.slurm

# Build SAM2/SAM3D input-preparation images after CARI4D succeeds.
sbatch --dependency=afterok:CARI4D_BUILD_JOB_ID TRACK1/slurm/01b_build_input_tools.slurm

# 5. Provide HF_TOKEN only at submission time; never store it in this repository.
sbatch --export=ALL,HF_TOKEN="${HF_TOKEN}" TRACK1/slurm/02_download_weights.slurm

# 6. Prepare masks and a metric mesh, then run one smoke-test episode.
sbatch --export=ALL,EPISODE=0 TRACK1/slurm/03_cari4d_episode.slurm
```

Do not launch steps 4–6 until the preceding job has succeeded. No jobs are submitted by the
setup scripts themselves.

## Upstream pins

The source checkout is recorded in `config/upstream.env`. Update deliberately and re-run all
contract checks after changing it.

Official references:

- Challenge: https://nvidia-isaac.github.io/video_to_data/v2d_challenge/
- Toolkit: https://github.com/nvidia-isaac/video_to_data
- Track 1 dataset: https://huggingface.co/datasets/nvidia/video_to_data_challenge/tree/main/track_1

