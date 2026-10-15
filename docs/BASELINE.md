# Track 1 CARI4D baseline notes

## What is verified

The current upstream `v2d_cari4d` module is the concrete baseline to run first. It consumes:

- `<sequence>.0.color.mp4` (the Track 1 videos must be linked or renamed for this convention),
- one packed HDF5 file containing a human and manipulated-object mask for every frame,
- one textured triangle mesh in metric scale,
- persistent CARI4D/SAM 3D Body/MoGe/FoundationPose weights.

It writes staged outputs and identity markers so unchanged successful stages can be reused.

## Important gaps in the public Track 1 release

The dataset explicitly says that it contains no pose, visibility annotation, or object mesh.
Therefore downloading the dataset is not enough to execute CARI4D.

For every episode we still need:

1. SAM2 prompts and propagated masks for human ID `0` and object ID `1`.
2. A canonical object mesh. A SAM3D candidate can be generated from a clear frame, but it must
   be autoscaled/calibrated before CARI4D because the released pipeline preserves mesh scale.
3. The CARI4D and SAM 3D Body gated-repository approvals and a Hugging Face read token.

Expected prepared layout:

```text
inputs/episode_000000/
├── episode_000000.0.color.mp4 -> ../../../data/track_1/videos/.../episode_000000.mp4
├── prompts.json
├── sam2_masks/
│   ├── 0/000000.png ...       human
│   └── 1/000000.png ...       object
├── masks_k0.h5
└── object.glb                  metric scale
```

## Cluster/container constraint

Upstream orchestration calls `docker` directly. This WR login node has Podman/Buildah and
Apptainer but no Docker executable. A Slurm probe confirmed that rootless Podman cannot unpack
OCI layers here: the account has no subordinate UID/GID range and the NFS-backed workspace
rejects the required pivot operation. `00_probe_runtime.slurm` therefore tests the cluster-native
Apptainer definition-build and NVIDIA GPU passthrough path. The build and execution jobs use
Apptainer directly and never rely on a Docker compatibility shim.

The CARI4D image compiles large CUDA extensions and downloads many dependencies. It must be
built in Slurm, not on `wr0`.

## Submission format

Each `episode_XXXXXX.npz` passed to the official packer must contain:

| Key | Shape |
|---|---:|
| `pose` | `[T, 136]` |
| `scales` | `[68]` |
| `shape` | `[45]` |
| `object_rotation` | `[T, 3, 3]` |
| `object_translation` | `[T, 3]` |
| `object_scale` | scalar, positive |

The matching mesh is `episode_XXXXXX_object.{glb,obj,ply,stl,off}`. CARI4D's native refined
bundle still needs a checked conversion/export step into this challenge schema; do not claim a
valid submission until the official `eval_reconstruction.py` packer accepts all 30 episodes.

