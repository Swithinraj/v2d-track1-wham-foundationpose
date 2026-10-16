# V2D Challenge Track 1: WHAM + FoundationPose baseline

A reproducible, SAM-free baseline for **Track 1 (monocular 4D human–object reconstruction)** of the
[NVIDIA Video-to-Data challenge](https://nvidia-isaac.github.io/video_to_data/v2d_challenge/).
It turns each of the 30 released third-person videos into the Track 1 submission: per-frame MHR
human parameters, an object mesh, and the object's Sim(3) trajectory in a shared frame.

The official CARI4D baseline needs the gated `facebook/sam-3d-body-dinov3` and
`facebook/sam-3d-objects` checkpoints, and the public data has no object meshes. This pipeline
uses only ungated components:

```text
video ─┬─ WHAM ─ world-frame SMPL ─ Meta SMPL→MHR map + official MHR fitter ──────┐
       │                                                                          ├─ joint alignment ─ official packer
       └─ GroundingDINO + SAM 2 masks, MoGe-2 depth ─ TRELLIS mesh ─ FoundationPose ┘
```

| Stage | What it does |
|---|---|
| Human | WHAM world-frame motion (DPVO rebuilt for the local GPUs), DINOv3-reviewed actor-track overrides for ambiguous clips, fitted to MHR `pose [T,136]`, `scales [68]`, `shape [45]` with Meta's topology map and the kit's `mesh_to_mhr_params.py` |
| Object inputs | GroundingDINO prompts (with reviewed overrides in `config/manual_object_prompts.json`) → SAM 2 masks; MoGe-2 metric depth and intrinsics |
| Object mesh | TRELLIS candidates from the cleanest view in each of the object's three episodes (and all three together); de-plating; the candidate FoundationPose fits best across all three cameras wins; metric size from MoGe-2, refined within 0.75–1.33× |
| Object track | FoundationPose registration and tracking, ESKF/RTS smoothing; the rotationally symmetric hula hoop instead gets a symmetry-aware particle filter + global optimisation (centre + axis line only, hand-contact prior, FoundationPose anchor while at rest) and an exactly round torus mesh |
| Alignment | One static-camera PnP per episode (WHAM joints vs. ViTPose keypoints, MoGe-2 intrinsics), low-frequency WHAM drift correction, MoGe/SMPL depth scale |
| Refinement | Human depth drift removed with MoGe-2's depth trend; hula-hoop centre smoothed; human motion smoothed on the MHR mesh and refitted, hands anchored to the FoundationPose object in stable grasps, wrists corrected by an RTS smoother fusing MediaPipe 2D and those anchors, lateral root drift removed by an RTS smoother (checked against WHAM's ViTPose) |
| Packing | Episode NPZ files → the official `eval_reconstruction.py` |

Generated object meshes are estimates, never challenge ground truth; each installed mesh has a
`.provenance.json`. Details, design decisions and measured results: [`docs/JOINT_BASELINE.md`](docs/JOINT_BASELINE.md).

## Requirements

- Linux with an NVIDIA GPU of compute capability **8.0 or newer** (A100, A6000, L40, H100, …;
  the TRELLIS, FoundationPose and GroundingDINO extensions are built for 8.0+), ≥ 24 GB VRAM.
- [Apptainer](https://apptainer.org) ≥ 1.2 with `--fakeroot` builds (or root) and network access
  for the one-time setup. Every model runs inside the images defined in `config/*.def`.
- `git`, `curl`, `unzip`, Python ≥ 3.10 on the host (stdlib for helpers, a venv for packing).
- About 60 GB for images and 25 GB for weights, plus outputs.
- The **SMPL** neutral model, which is licensed separately: register at
  [smpl.is.tue.mpg.de](https://smpl.is.tue.mpg.de) and place it at
  `weights/wham/dataset/body_models/smpl/SMPL_NEUTRAL.pkl`. Never commit it.

## Setup (once)

```bash
bash pipeline/setup/fetch_sources.sh      # pinned upstream repos + submodules, Eigen, submission kit
bash pipeline/setup/build_containers.sh   # 6 Apptainer images in containers/ (slow: CUDA extensions)
bash pipeline/setup/build_dpvo.sh         # WHAM's DPVO for your GPUs (TORCH_CUDA_ARCH_LIST, default 7.0;8.0;8.6)
bash pipeline/setup/download_data.sh      # Track 1 videos + metadata (HF_TOKEN / HF_TOKEN_FILE if required)
bash pipeline/setup/download_weights.sh   # all public checkpoints; derives SMPL faces once SMPL is in place
bash pipeline/setup/setup_venv.sh         # host venv with the official kit's requirements (PYTHON=...)
bash pipeline/setup/setup_mediapipe.sh    # host venv with MediaPipe + its two pinned models (2D keypoints)
```

All pins (upstream commits, dataset revision, weight revisions, checksums) are in
[`config/upstream.env`](config/upstream.env).

## Run

```bash
bash pipeline/run_all.sh
COMMIT_URL=https://github.com/<owner>/<repo>/commit/<sha> bash pipeline/stages/pack.sh
```

`run_all.sh` runs every stage on the current machine in dependency order and skips completed
outputs, so it can be resumed. Without `COMMIT_URL` the packer writes
`submissions/track1_LOCAL_CHECK_do_not_upload.parquet`; with it, an uploadable
`submissions/track1_wham_fp_<date>.parquet`. Upload with the kit's
`submit/v2d_submit.py <parquet> --track track_1` after accepting the rules of all five Track 1
Kaggle competitions.

Each stage is also a standalone script, so a cluster can run one task per episode or object:

| Order | Script | Unit | GPU | Output |
|---|---|---|---|---|
| 1 | `pipeline/stages/object_depth.sh EP` | episode | yes | `outputs/object/<ep>/{depth,intrinsics}` |
| 1 | `pipeline/stages/object_masks.sh EP` | episode | yes | `outputs/object/<ep>/{prompt,sam2_masks}` |
| 1 | `pipeline/stages/wham.sh EP` | episode | yes | `outputs/wham/<ep>/wham_smpl_world.npz` |
| 2 | `pipeline/stages/wham_to_mhr.sh EP` | episode | yes | `outputs/human/<ep>/mhr_human.npz` |
| 2 | `pipeline/stages/align.sh EP` | episode | no | `outputs/alignment/<ep>/` |
| 2 | `pipeline/stages/object_mesh.sh OBJECT` | object (3 episodes) | yes | `inputs/object_meshes/<object>.glb` |
| 3 | `pipeline/stages/hands.sh EP` | episode | yes | `outputs/hands/<ep>.npz` |
| 3 | `pipeline/stages/depth_trend.sh EP` | episode | no | `outputs/depth_trend/<ep>.npz` |
| 3 | `pipeline/stages/keypoints2d.sh EP` | episode | no | `outputs/{mediapipe,vitpose}/<ep>.npz` |
| 3 | `pipeline/stages/track_object.sh EP` | episode | yes | `outputs/object/<ep>/foundationpose/` |
| 4 | `pipeline/stages/track_symmetric.sh EP` | episode (symmetric objects only) | no | `outputs/object/<ep>/foundationpose/symmetric/` |
| 5 | `pipeline/stages/pack.sh` (runs `human_refine.sh`) | all | yes (refinement) | `submissions/*.parquet` |

`EP` is 0–29. `object_mesh.sh` needs depth and masks of its object's three episodes; `align.sh`
needs WHAM and depth; `hands.sh` needs WHAM→MHR and alignment; `depth_trend.sh` needs WHAM,
alignment, depth and masks; `keypoints2d.sh` needs WHAM; `track_symmetric.sh` needs tracking
and hands (it does nothing for objects not listed in `config/symmetric_objects.json`); `pack.sh`
needs everything and, for the human refinement (on by default), a GPU. Optional knobs are documented
at the top of each script (e.g. `DEPTH_TREND=0` turns off the human depth-trend correction,
`SMOOTH_SYMMETRIC=0` the hoop-centre smoothing, `HUMAN_REFINE=0` the human refinement,
`RESOLVE_PEN=1` turns on the hand–object penetration push).

## Layout

```text
pipeline/      env.sh, run_all.sh, setup/ (one-time), stages/ (per episode/object)
scripts/       Python run inside the images (and the host venv for packing)
config/        Apptainer definitions, upstream pins, reviewed object prompts, symmetric objects
docs/          design notes and results
reports/       dated progress reports with every Kaggle score
```

Created at run time and ignored by git: `third_party/`, `official/`, `containers/`, `weights/`,
`data/`, `inputs/`, `outputs/`, `submissions/`, `.cache/`, `.venvs/`, `logs/`.

## Status and limits

- All 30 episodes run end to end and pack into a valid 740,780-row Track 1 file. The reference is
  held out, so there is no local score; scores come from Kaggle. Latest scores of this pipeline:
  CD-H 20.96 cm, CD-O 29.37 cm, ACC-H 0.142 cm/frame², ACC-O 0.077 cm/frame², PEN 0.0082 cm; every
  submission and what each change did: [`reports/report_day_track1.md`](reports/report_day_track1.md).
- Episodes 2 and 15 use reviewed actor-track fragments from `config/human_track_overrides.json`;
  other episodes retain the longest-track default.
- CoCoNet refinement from CARI4D is not used: it needs SAM 3D Body decoder buffers that the
  accessible `nvidia/cari4d_commercial` checkpoint does not contain.
- Generated meshes vary in quality (see `docs/JOINT_BASELINE.md`); `outputs/object_meshes/review_sheet.png`
  (`scripts/render_mesh_review.py`) shows them side by side with their source views.

References: [challenge](https://nvidia-isaac.github.io/video_to_data/v2d_challenge/) ·
[toolkit](https://github.com/nvidia-isaac/video_to_data) ·
[Track 1 data](https://huggingface.co/datasets/nvidia/video_to_data_challenge/tree/main/track_1) ·
[WHAM](https://github.com/yohanshin/WHAM) · [MHR](https://github.com/facebookresearch/MHR) ·
[TRELLIS](https://github.com/microsoft/TRELLIS) · [MediaPipe](https://github.com/google-ai-edge/mediapipe)
