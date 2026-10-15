# WHAM + Meta MHR human baseline for Track 1

WHAM is the ungated temporal human-motion branch. It is not a drop-in replacement inside
released CARI4D: WHAM predicts SMPL, while the Track 1 schema uses Meta MHR. The pinned,
Apache-2.0 `facebookresearch/MHR` converter closes that topology/parameter boundary.

## What is reproducibly configured

- Official source: `yohanshin/WHAM`, commit `2b54f7797391c94876848b905ed875b154c4a295`.
- Runtime: the Docker image documented by WHAM, converted into `containers/wham.sif`.
- Public model assets: WHAM-BEDLAM/3DPW, HMR2a, DPVO, YOLOv8x, and ViTPose-H.
- Workspace-local DPVO CUDA extensions compiled for the cluster's V100 (`sm_70`) and A100
  (`sm_80`) GPUs. The binaries bundled in WHAM's public image target only `sm_86`.
- Meta MHR source: commit `d96fafa33bbf018647c70c3525e91f53e79d2a14`.
- Public auxiliary body-model data and the full Apache-2.0 MHR v1.0.1 assets.
- Per-episode Slurm inference, dense world-coordinate SMPL mesh export, and MHR fitting.
- Exact submission arrays: MHR pose `[T,136]`, scales `[68]`, and shape `[45]`.

The often-circulated `convert_smpl_to_mhr.py` command does not exist in the current Meta repo.
This workspace applies Meta's released `smpl2mhr_mapping.npz` barycentric topology map, then
calls the challenge kit's reference `mesh_to_mhr_params.py` fitter. This avoids the experimental
PyMomentum wheel while enforcing one identity shared across the episode.

## Required licensed file

WHAM's code and network checkpoints are public, but the official demo still requires a neutral
SMPL body model. Register at either the SMPLify/SMPL sites linked by WHAM, accept their terms,
download the neutral v1 model, rename it to `SMPL_NEUTRAL.pkl`, and place it at:

```text
/work/smoses2s/nvidea_chal/TRACK1/weights/wham/dataset/body_models/smpl/SMPL_NEUTRAL.pkl
```

Do not commit this file or put SMPL credentials in a Slurm script.
`submit_human_pipeline.sh` derives `smpl_faces.npy` locally from this licensed model; that file is
not present in WHAM's public auxiliary archive.

## Build and download

```bash
cd /work/smoses2s/nvidea_chal
bash TRACK1/scripts/submit_human_setup.sh
```

After supplying `SMPL_NEUTRAL.pkl`:

```bash
python3 TRACK1/scripts/validate_wham_setup.py
python3 TRACK1/scripts/validate_mhr_setup.py
sbatch TRACK1/slurm/16_build_dpvo_extensions.slurm
# Wait for the one-time DPVO build to pass.
bash TRACK1/scripts/submit_human_pipeline.sh
```

The submission helper creates six 5-task array chunks because this WR cluster has
`MaxArraySize=5`. Each MHR conversion chunk starts only after its corresponding WHAM chunk passes.
It validates the cluster-compatible DPVO extensions before creating any jobs, and completed WHAM
or MHR episode artifacts are shape/finite-value checked and reused.

## Output and submission boundary

WHAM writes `outputs/wham/episode_XXXXXX/wham_output.pkl`; the export helper writes a dense
`wham_smpl_world.npz` with `[T,6890,3]` world-coordinate SMPL vertices. Missed detections are
filled from the nearest observed frame. The Meta-map/reference-fitter adapter then writes:

```text
outputs/human/episode_XXXXXX/mhr_human.npz
  pose    [T,136]
  scales  [68]
  shape   [45]
```

The official fitter handles MHR's centimetres and y/z convention internally. The adapter records
fitting-error and coordinate provenance in the NPZ.
Object masks, a metric object mesh, and its pose trajectory remain separate requirements.
