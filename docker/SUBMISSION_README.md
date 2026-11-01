# Izus_systems: V2D Challenge Track 1 submission container

This container reproduces our counted Track 1 submission, `submitted/@FILE@`, made from commit `@SHA@`
(@URL@). The same file is selected on all five boards. The Kaggle submission IDs are in `submission.json`.

## Contents

| Path | Contents |
|---|---|
| `/submission/code` | Our repository at the submitted commit, with `.git` |
| `/submission/kit` | The kit copy that packed the file |
| `/submission/verifier_kit` | The official kit with `tools/verify_container.py` |
| `/submission/submitted` | The uploaded Parquet and its upload receipt |
| `/submission/outputs/track1/reconstructions` | The final episode files and object meshes packed into it |
| `/submission/containers`, `/opt/envs` | The pipeline's Apptainer images, and the same images unpacked |
| `/submission/weights`, `third_party`, `inputs` | Every model, source tree and input mesh the pipeline loads |

## Running

The dataset directory holding `track_1/` is mounted read-only at `/data`, and results are written to `/output`. No step
needs network access or extra privileges.

```bash
docker run --rm --network none IMAGE /submission/run.sh verify
docker run --rm --network none -v "$PWD/out:/output" IMAGE /submission/run.sh pack --label track1 --output /output
docker run --rm --gpus all --network none -v DATASET:/data:ro -v "$PWD/out:/output" IMAGE \
    /submission/run.sh reproduce --label track1 --dataset-root /data --output /output --episodes 3,12
```

`reproduce --episodes` writes to `/output/track1/reconstructions`, for comparison with the stored reconstructions. Without
`--episodes` it runs all 30 and packs the file.

Stages call `apptainer exec`. Real Apptainer cannot start under a plain `docker run`, so `/usr/local/bin/apptainer` is a
short shim (`docker/apptainer_shim.sh`) that runs the unpacked image with proot. It takes the same call, so the pipeline
code is unchanged. GPU work is unaffected. Python imports and file IO are slower than under Apptainer.

## Hardware and run time

- One NVIDIA GPU of compute capability 8.0 or newer (A100 class) with at least 24 GB of memory.
- About 32 GB of RAM and 8 CPU cores, plus free disk under `/output`.

## Expected variation

The models are not bitwise deterministic across machines, and SAM 3D Objects generation is seeded per mesh. Reconstructions
from `reproduce` differ a little from the stored ones. `pack` recreates the submitted file from the stored ones exactly.
