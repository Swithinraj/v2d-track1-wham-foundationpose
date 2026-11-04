# NVIDIA's CARI4D pipeline on the challenge videos

`docs/BASELINE.md` named NVIDIA's official pipeline (`v2d_cari4d` in `nvidia-isaac/video_to_data`) as the baseline to run first, but
no submission ever used it. It runs SAM 3D Body and FoundationPose, then CoCoNet, a network trained on 2,126 sequences of the same
family of data, then a contact-guided optimiser. It outputs the MHR human this challenge scores.

## Running it

```
bash pipeline/setup/fetch_cari4d_assets.sh                    # two Git LFS assets the image holds only as pointers
bash pipeline/cluster/cari4d_weights.sh                       # official checkpoints (needs the Hugging Face token)
sbatch pipeline/cluster/run_c4_all.sbatch                     # all episodes on the GPUs of one allocation
```

Inputs: the video, the SAM 2 masks (`outputs/person`, `outputs/object`) and one mesh per object, scaled to one size per physical
object (`scripts/object_true_size.py`, `scripts/cari4d_mesh.py`), because CARI4D keeps the mesh's scale.

About 30 minutes per episode with two workers per GPU. Stage times for a 240-frame episode: depth 43 s, export 21 s, SAM 3D Body
45 s, depth alignment 164 s, FoundationPose 64 s, CoCoNet 284 s, optimiser 810 s, render 244 s.

## From its output to a candidate

`pipeline/recipes/cari4d_candidate.sh NAME STAGE LAM BETA`:

1. `scripts/cari4d_export.py` decodes one stage (`init`, `coconet` or `refined`) to MHR vertices and object poses.
2. `scripts/mesh_smooth.py` smooths them. CARI4D's motion is far jerkier than the scored candidates' (0.9-2.2 against 0.06
   cm/frame^2), and CoCoNet's 96-frame windows join without overlap.
3. The kit's `mesh_to_mhr_params.py` converts vertices to the scored parameters (0.7 mm mean fit error).
4. `scripts/true_cam_params.py` moves it to the measured camera, and `scripts/floor_root.py` applies the floor slide.
5. `scripts/object_transfer.py` carries the chosen objects to this person.

Unfinished episodes keep the current best person, so a partial run already gives Kaggle's answer, scaled by the share of episodes.
