# Track 1 object branch: masks, depth and FoundationPose tracking

The object branch prepares the released monocular videos for object-pose estimation:

1. GroundingDINO uses each episode's released `object_prompt` on sparse frames.
2. The strongest detections seed SAM 2.1 propagation across the complete video.
3. MoGe-2 ViT-L Normal produces metric depth and per-frame pinhole intrinsics.
4. FoundationPose registers and tracks a supplied metric object mesh.
5. An IoU-weighted ESKF/RTS pass produces a smoother trajectory for ACC-O.

## Important data boundary

The public Track 1 release contains 30 videos, parquet indexes, and metadata. It contains no CAD,
OBJ, GLB, or other ground-truth object meshes; challenge ground truth is held out. FoundationPose
therefore cannot run until one metric mesh per object is placed at:

```text
TRACK1/inputs/object_meshes/<object>.glb
```

The required object names come from `meta/episodes_metadata.jsonl`. A generated or independently
obtained mesh can be used, but it must not be described as challenge ground truth.

Exactly ten metric meshes are needed; each is reused by three episodes:

```text
big_red_bowl.glb       black_pan.glb          foam_grass_block.glb
hula_hoop.glb          iron.glb               paint_roller.glb
pink_foam_roll.glb     short_wood_stool.glb   white_desk.glb
white_laptop_cart.glb
```

## Run sequence

```bash
for e in $(seq 0 29); do
  bash pipeline/stages/object_depth.sh "$e"
  bash pipeline/stages/object_masks.sh "$e"
done
```

Inspect every `outputs/object/episode_XXXXXX/prompt/primary_prompt.jpg` and
`mask_audit.json`. Automated text prompting is a proposal, not ground truth; incorrect prompts
must be replaced before FoundationPose. Reviewed box or positive/negative point prompts live in
`config/manual_object_prompts.json`; rerunning an episode preserves the override and records
`source=manual_visual_review` in its prompt metadata.

The GroundingDINO CUDA extension in the input-tools image targets compute capability 8.0+; on
older GPUs it reports "no kernel image is available" and silently returns no detections.

Metric meshes come from `pipeline/stages/object_mesh.sh` (see `docs/PIPELINE.md`). Then:

```bash
python3 scripts/validate_object_setup.py
for e in $(seq 0 29); do bash pipeline/stages/track_object.sh "$e"; done
```

MoGe-2 depth and monocular intrinsics are estimates. FoundationPose output remains in each
camera's coordinate system, which is also the submission's world frame (static camera);
`scripts/build_track1_episodes_sam.py` maps it into SAM 3D Body's metric with the episode's MoGe/SAM depth scale.
