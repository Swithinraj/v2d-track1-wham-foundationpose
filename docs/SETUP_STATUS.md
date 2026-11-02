# Track 1 setup status (2026-10-29)

## Selected result

The best **Kaggle-scored single file** recorded by this branch is BMQ3KGM:
CD-H 14.35418, CD-O 18.60437, ACC-H 0.13629, ACC-O 0.07067,
PEN 0.00013. See `docs/EXPERIMENT_REPORT.md` and `docs/CURRENT_BEST.md`.
The read-only Kaggle submission history checked on 2026-10-29 has no newer
scored file than BMQ3KGMR2, and it does not displace BMQ3KGM as the selected
single-file baseline.
The measured-camera BMQ3KGMtp variant has better local CD-H, but this branch
does not record a Kaggle score for it. Keep BMQ3KGM as the scored baseline.

## Installed in this workspace

- `vpq-vitpose` at commit `2cd29f4`.
- Pinned NVIDIA Track 1 release only, revision
  `5f68335f3acc802033d1e80728c1633197521de8`: 30 videos, 30 Parquet
  files, 16,563 frames. `scripts/validate_workspace.py --data-only` passes.
- Pinned `video_to_data`, MHR and CoTracker sources, plus the official
  submission kit and its sample submission.
- Linux `.venvs/submission` (Python 3.12), Windows
  `.venvs/submission-windows` (Python 3.13), and Linux `.venvs/mediapipe`.
  The kit's `eval_reconstruction.py --help` and `pip check` pass in the
  submission environments.
- Both pinned MediaPipe landmark models, and the SHA256-verified ViTPose-H
  checkpoint needed by the BMQ3KGM ancestry.
- The supplied Hugging Face token can list both gated SAM 3D repositories.
  Credentials remain in the parent directory, outside Git.

## Remaining before inference

This laptop has an RTX 3050 with 4 GB VRAM. The repository requires Linux,
Apptainer and at least 24 GB VRAM. WSL is present, but Apptainer and the six
pipeline images are not installed. The remaining SAM 3D Body, SAM 3D Objects,
MHR, SAM 2, MoGe-2, FoundationPose, GroundingDINO and CoTracker checkpoints
have not been downloaded. They should be installed on the supported GPU
runtime where inference will actually run, avoiding duplicate large downloads.

The branch also excludes generated inputs needed to reproduce BMQ3KGM:
`submissions/episodes_objects_vpq`, `outputs/kp2d_vpx`, and the earlier
`episodes_repro*` folders. `pipeline/run_all.sh` does not regenerate all of
these. Restore saved copies if available; otherwise the VPQ ancestry in
`pipeline/recipes/vpq.md` must be rebuilt before `bmq_final.sh`.

The workspace notebook `../Untitled-1.ipynb` now contains a staged Colab A100
mask and ViTPose-H workflow. It reuses Drive storage, creates separate virtual
environments, fetches pinned GroundingDINO and SAM 2 sources and only their
needed checkpoints, and times episode 0 before a resumable 29-episode mask
batch. It then checks all 30 actor-mask folders before downloading ViTPose-H,
times one pose episode, and resumes the remaining episodes. The notebook is
syntax validated but its native Colab mask installation and GPU stages have
not yet been run. The Colab A100 is disconnected while setup is prepared.

The installed `Notebook MCP for Claude Code & Codex` VS Code extension is now
registered as Codex's `notebook` MCP server. A read-only probe reached the
extension, listed the open notebook, and confirmed its kernel was stopped.
The installed `colab` CLI can start and stop paid runtimes; its OAuth/session
state has not yet been initialized in this workspace.

## Safe order on a supported GPU runtime

1. Confirm at least 24 GB VRAM, Apptainer with fakeroot, and persistent
   storage before installing images or starting inference.
2. Transfer or redownload only the pinned Track 1 release. Keep all model
   weights and caches on persistent storage.
3. Run the pinned source and checkpoint setup scripts in the README. Build
   `cari4d sam2 grounding_dino sam3d mhr_converter sam3d_body` images: the
   final one is required by `bmq_final.sh` but omitted from the default image
   list. `pipeline/setup/download_vitpose.py` fetches only the extra
   ViTPose-H model. `fetch_sources.sh` skips optional Git LFS payloads.
4. Run `pipeline/run_all.sh` for the basic stages. Restore the generated VPQ
   inputs, or rebuild them using `vpq.md`.
5. Reproduce BMQ3KGM as in `docs/CURRENT_BEST.md`:

   ```bash
   bash pipeline/recipes/bmq_final.sh
   bash pipeline/recipes/bmqk_final.sh
   bash pipeline/recipes/bmq3k_final.sh
   EPS_SCALE=0 bash pipeline/recipes/gru_depth_final.sh \
     episodes_bmq3k_in episodes_bm_depth mask_visibility_bm \
     "${BM_ROOT:-$PWD/work/bm_root}" episodes_cand_BMQ3KGM
   PACK_EPISODES=episodes_cand_BMQ3KGM bash pipeline/stages/pack.sh
   ```

`BM_ROOT` defaults to `work/bm_root` in `bmq_final.sh`. The packer writes a local validation file unless
`COMMIT_URL` is provided. Do not submit an unverified reconstruction.
