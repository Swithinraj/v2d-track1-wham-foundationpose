# Track 1 submission runbook

From a chosen candidate to the leaderboard and the award container. Rules are in the kit's `tracks/track_1/README.md` and
`CONTAINER_SUBMISSION.md`.

**Freeze: November 4, 22:00 UTC (5:00 p.m. EST).** Uploads are limited to 20 per team per UTC day, shared across accounts.

## Handoff: for Zaid or Swithinraj

Whoever makes the final submission owns these. Ujjwal prepared this branch and has no cluster or Kaggle access from the
machine it was written on.

1. **Agree who does it**, so two people do not spend the same daily uploads or pick different finals.
2. **Start from this branch.** Run `git fetch origin && git switch submission && git merge origin/cluster-floor-2026-11`,
   so the final commit holds both the candidate's scripts and this pipeline.
3. **Get the official kit.** Take `v2d_submission_kit_official.zip` (sha256 `667be286f664a5859bc631f7845df2576f1ed3bdcdad12e1e9f2de926a2d6c25`) from Ujjwal. `--kit` needs it,
   because `tools/verify_container.py` may be missing from the cluster's older `official/v2d_submission_kit`.
4. **Dry-run on the cluster once**, before the final choice, to catch problems early. Use any scored file with its real
   episode directory:

   ```bash
   .venvs/submission/bin/python docker/collect_container.py --dry-run --candidate NAME \
     --kit /path/to/v2d_submission_kit_official.zip --pack-kit /path/to/v2d_submission_kit_official_extracted_or_cluster_kit \
     --out work/container_dry
   ```

   If the cluster kit has no `tools/verify_container.py`, point `--pack-kit` at the extracted official kit. Its packing code
   matched the Oct 25 kit copy Ujjwal checked; confirm the cluster copy matches too.
5. **Find a Docker machine** (requirements below). The cluster's documented path has no Docker.
6. **Follow the two sections below.** The one step that needs real work is writing `final_recipe.sh`.

**Tested so far, on a laptop:** `verify`, `pack`, image unpacking and CUDA through the shim, all under a plain `docker run` with
no network, on a scored Oct 31 file.

**Not tested yet, so check these first:**

- Any real model stage under the proot shim, including how much slower it is.
- `requirements.lock` from the cluster's Python 3.11 venv installing on the image's Python 3.12. If a pin fails, keep the
  versions of `numpy`, `trimesh` and `fast_simplification`, because a changed mesh decimation breaks the exact re-pack.
- `reproduce --episodes` on a subset. The contact stage requires all 30 episodes, so `final_recipe.sh` has to handle this.

## The rule that shapes everything

The container holds **one commit**, and `verify_container.py` fails unless the counted file cites it. If nothing is selected,
each board counts its own best file, and those files cite different commits (`2cd29f4`, `13ff13a`, `1280cc0`, `af466e2`,
`07ff09b`, ...). No single container can pass that. So:

**Select one file as Final Submission on all five boards**, packed from one commit. It may be a re-pack of a scored candidate,
which scores the same. The other final slot on each board stays empty, or holds a file citing the same commit.

## Before the freeze (cluster)

1. **Choose the candidate** `NAME`. Compare by rank sum over the five boards, not one metric.
2. **Write its recipe.** Replace the marked lines in `pipeline/recipes/final_recipe.sh` with the steps that built it, starting
   from the videos. The container's `reproduce` runs this, and `collect_container.py` refuses to build until it is written.
3. **Freeze the commit.** Commit, push, and take the full sha. Every later file cites this commit.
4. **Pack, check, upload:**

   ```bash
   COMMIT=<full sha> CANDIDATE_SCRIPTS="pipeline/recipes/final_recipe.sh" bash pipeline/recipes/pack_candidate.sh NAME
   .venvs/submission/bin/python scripts/submit_candidate.py submissions/track1_cand_NAME.parquet -m "NAME final" --dry-run
   .venvs/submission/bin/python scripts/submit_candidate.py submissions/track1_cand_NAME.parquet -m "NAME final"
   ```

   Confirm all five scores match the scored candidate. Commit the receipt to `docker/receipts/`.
5. **Select it on Kaggle by hand** on all five boards (CD-H, CD-O, ACC-H, ACC-O, PEN). Links: kit `LEADERBOARDS.md`.
6. **Keep `submissions/episodes_cand_NAME` and the parquet read-only.** They must be the exact files behind the upload. Never
   substitute `parquet_episodes.py` output for them: it fills unscored frames by interpolation, so it is not what was packed.

## After the freeze: the container

1. On the cluster:

   ```bash
   .venvs/submission/bin/python docker/collect_container.py --candidate NAME \
     --receipt docker/receipts/RECEIPT.json --kit /path/to/v2d_submission_kit_official.zip
   ```

   This re-packs the stored episodes and compares them with the uploaded file before copying anything large.
2. On a machine with Docker, the NVIDIA toolkit, a GPU of at least 24 GB and **about 300 GB free**: images and weights are about
   90 GB, and they exist twice during the build (SIF and unpacked), plus the tarball. Docker is not on the cluster's
   documented path; the 8 GB laptop cannot run `reproduce`, and has only 82 GB free.

   ```bash
   REPRODUCE="3,12" DATASET=/path/to/dataset bash docker/build_container.sh outputs/container_build
   ```

   Every `verify` check must pass. `pack` must succeed. Compare the reproduced episodes with the stored ones.
3. Upload the tarball so the link downloads without an access request, and submit the link, sha256 and image id from `SEND.txt`
   with the technical report, which must describe the method the container runs.

## Checklist

- [ ] Five Track 1 competitions' rules accepted by every uploading account, same team
- [ ] Repository public or shared with `Nvidia-V2D-Challenge` (public as of 2026-11-02)
- [ ] One file selected on all five boards, citing one pushed commit
- [ ] `final_recipe.sh` runs from the videos on two or three episodes
- [ ] Receipt, parquet, episodes and the packing kit copy kept
- [ ] Redistribution of the gated SAM 3D weights inside the image checked with the organizers
- [ ] `run.sh verify` passes with no network
- [ ] Tarball link opens without an access request
