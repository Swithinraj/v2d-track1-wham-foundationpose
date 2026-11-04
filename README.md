# V2D Challenge Track 1: SAM 3D Body + SAM 3D Objects + FoundationPose

A reproducible pipeline for **Track 1 (monocular 4D human–object reconstruction)** of the
[NVIDIA Video-to-Data challenge](https://nvidia-isaac.github.io/video_to_data/v2d_challenge/).
It turns each of the 30 released third-person videos into the Track 1 submission: per-frame MHR
human parameters, an object mesh, and the object's Sim(3) trajectory in a shared frame.

The exo cameras are static, so the world frame is the camera itself: the human comes from SAM 3D Body in camera
coordinates, the object from FoundationPose in MoGe-2's metric camera frame, and one per-episode depth scale ties the
two together. No world-frame human tracker, no SMPL and no camera alignment step are involved.

```text
video ─┬─ MoGe-2 depth + intrinsics (one static camera) ───────────────────────────────────────────────┐
       ├─ GroundingDINO + SAM 2 object masks ─ SAM 3D Objects mesh ─ FoundationPose + ESKF/RTS ─────────┤
       │                                       (hula hoop: symmetry-aware tracker, round torus)        ├─ camera-frame assembly
       └─ GroundingDINO person tracks → actor ─ SAM 2 actor masks ─ SAM 3D Body per frame ──────────────┘   ─ depth trend ─ refinement
                                                 ─ shared identity ─ MHR refit                              ─ hand fix ─ official packer
```

| Stage | What it does |
|---|---|
| Object inputs | GroundingDINO prompts (reviewed overrides in `config/manual_object_prompts.json`) → SAM 2.1 masks; MoGe-2 metric depth and intrinsics |
| Actor masks | GroundingDINO "a person." every 10th frame, linked into tracks; the actor is the track closest to the object mask; SAM 2.1 propagates its box prompts to every frame |
| Human | SAM 3D Body (DINOv3) on every frame, prompted with the actor-mask box and the median MoGe-2 intrinsics; one shared body identity per episode (medoid over reliable frames); every frame refitted with that identity (kit MHR wrapper, `lm_pose`) |
| Object mesh | SAM 3D Objects on each episode's cleanest view, grounded in its MoGe-2 depth (metric directly), several seeds; the candidate FoundationPose fits best across all three cameras wins, scale refined within 0.75–1.33× |
| Object track | FoundationPose registration and tracking, ESKF/RTS smoothing; the rotationally symmetric hula hoop instead gets a symmetry-aware particle filter + global optimisation (centre + axis line, hand-contact prior from the SAM 3D Body hands) and an exactly round torus mesh |
| Assembly | Static camera = world; objects mapped into SAM 3D Body's metric with the per-episode MoGe/SAM depth scale; hula-hoop centre smoothed |
| Refinement | Human depth trend from MoGe-2 (α 0.75); motion smoothed on the MHR mesh and refitted (λ 300), hands anchored to the object in stable grasps, wrists fused with MediaPipe 2D (RTS), lateral root drift removed where the actor's SAM 2 mask confirms it |
| Hand fix | Only finger and wrist-rotation parameters are refitted to push hand vertices out of the object (two passes): penetration drops while objects and the 22 scored joints stay exactly unchanged |
| Packing | Episode NPZ files → the official `eval_reconstruction.py` |

Generated object meshes are estimates, never challenge ground truth; each installed mesh has a
`.provenance.json`. Details, design decisions and measured results: [`docs/PIPELINE.md`](docs/PIPELINE.md).

## Requirements

- Linux with an NVIDIA GPU of compute capability **8.0 or newer** (A100, A6000, L40, H100, …), ≥ 24 GB VRAM.
- [Apptainer](https://apptainer.org) ≥ 1.2 with `--fakeroot` builds (or root) and network access
  for the one-time setup. Every model runs inside the images defined in `config/*.def`.
- `git`, `curl`, `unzip`, Python ≥ 3.10 on the host (stdlib for helpers, a venv for packing).
- About 60 GB for images and 30 GB for weights, plus outputs.
- A Hugging Face account with **approved access** to the gated
  [`facebook/sam-3d-body-dinov3`](https://huggingface.co/facebook/sam-3d-body-dinov3) and
  [`facebook/sam-3d-objects`](https://huggingface.co/facebook/sam-3d-objects) models and a read token
  (`HF_TOKEN` or `HF_TOKEN_FILE`; never commit it).

## Setup (once)

```bash
bash pipeline/setup/fetch_sources.sh          # pinned upstream repos + submodules, submission kit
bash pipeline/setup/build_containers.sh       # Apptainer images in containers/ (slow: CUDA extensions)
bash pipeline/setup/download_data.sh          # Track 1 videos + metadata (HF_TOKEN / HF_TOKEN_FILE if required)
bash pipeline/setup/download_weights.sh       # public checkpoints: MHR, SAM 2.1, MoGe-2, FoundationPose, GroundingDINO
bash pipeline/setup/download_sam3d_body.sh    # gated SAM 3D Body (HF_TOKEN / HF_TOKEN_FILE)
bash pipeline/setup/download_sam3d_objects.sh # gated SAM 3D Objects (HF_TOKEN / HF_TOKEN_FILE)
bash pipeline/setup/setup_venv.sh             # host venv with the official kit's requirements (PYTHON=...)
bash pipeline/setup/setup_mediapipe.sh        # host venv with MediaPipe + its two pinned models (2D keypoints)
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
`submissions/track1_sam3d_<date>.parquet`. Upload with the kit's
`submit/v2d_submit.py <parquet> --track track_1` after accepting the rules of all five Track 1
Kaggle competitions.

Each stage is also a standalone script, so a cluster can run one task per episode or object:

| Order | Script | Unit | GPU | Output |
|---|---|---|---|---|
| 1 | `pipeline/stages/object_depth.sh EP` | episode | yes | `outputs/object/<ep>/{depth,intrinsics}` |
| 1 | `pipeline/stages/object_masks.sh EP` | episode | yes | `outputs/object/<ep>/{prompt,sam2_masks}` |
| 1 | `pipeline/stages/keypoints2d.sh EP` | episode | no | `outputs/mediapipe/<ep>.npz` |
| 2 | `pipeline/stages/person.sh EP` | episode | yes | `outputs/person/<ep>/{prompt,sam2_masks/0}` |
| 2 | `pipeline/stages/object_mesh.sh OBJECT` | object (3 episodes) | yes | `inputs/object_meshes/<object>.glb` |
| 3 | `pipeline/stages/sam3d_body.sh EP` | episode | yes | `outputs/{sam3d_body,camera,hands,alignment_sam,keypoints2d_sam}/<ep>` |
| 3 | `pipeline/stages/track_object.sh EP` | episode | yes | `outputs/object/<ep>/foundationpose/` |
| 4 | `pipeline/stages/track_symmetric.sh EP` | episode (symmetric objects only) | no | `outputs/object/<ep>/foundationpose/symmetric/` |
| 5 | `pipeline/stages/pack.sh` (runs `human_refine.sh`, `hand_fix.sh`) | all | yes | `submissions/*.parquet` |

`EP` is 0–29. `person.sh` needs the episode's object masks; `object_mesh.sh` needs depth and masks of its object's
three episodes; `sam3d_body.sh` needs depth and actor masks; `track_symmetric.sh` needs tracking and the SAM 3D Body
hands (it does nothing for objects not listed in `config/symmetric_objects.json`); `pack.sh` needs everything.
Optional knobs are documented at the top of each script: `DEPTH_TREND=0` keeps SAM 3D Body's own depth,
`OBJECT_FOLLOW_HUMAN=1` (with `DEPTH_TREND=0`) lets the object depth follow the human's depth trend,
`HUMAN_REFINE=0` / `HAND_FIX=0` turn off the refinement / hand fix, `RESOLVE_PEN=1` adds the object-side
penetration push. On a cluster, run `pack.sh` with `HAND_FIX=0`, then `hand_fix.sh` once per disjoint
`HAND_FIX_EPISODES` subset (one GPU each, same input and output), then `PACK_EPISODES=<output> pack.sh`.

## Layout

```text
pipeline/      env.sh, run_all.sh, setup/ (one-time), stages/ (per episode/object)
scripts/       Python run inside the images (and the host venvs for packing and MediaPipe)
config/        Apptainer definitions, upstream pins, reviewed object prompts, symmetric objects
docs/          design notes and results
reports/       dated progress reports with every Kaggle score
```

Created at run time and ignored by git: `third_party/`, `official/`, `containers/`, `weights/`,
`data/`, `inputs/`, `outputs/`, `submissions/`, `.cache/`, `.venvs/`, `logs/`.

## Status and limits

See [`docs/PIPELINE.md`](docs/PIPELINE.md#status) for the measured state of this architecture. Kaggle history of
the earlier architectures: [`reports/`](reports/).

References: [challenge](https://nvidia-isaac.github.io/video_to_data/v2d_challenge/) ·
[toolkit](https://github.com/nvidia-isaac/video_to_data) ·
[Track 1 data](https://huggingface.co/datasets/nvidia/video_to_data_challenge/tree/main/track_1) ·
[SAM 3D Body](https://github.com/facebookresearch/sam-3d-body) ·
[SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects) · [MHR](https://github.com/facebookresearch/MHR) ·
[FoundationPose](https://github.com/NVlabs/FoundationPose) · [MediaPipe](https://github.com/google-ai-edge/mediapipe)
