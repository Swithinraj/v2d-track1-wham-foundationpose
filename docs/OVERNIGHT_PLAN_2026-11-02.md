# Track 1 overnight research and execution plan

## Execution status: November 2, afternoon

The original 08:00 review target was missed. Automatic approval service usage
limits interrupted execution around 02:14; work resumed at 13:11 Europe/Berlin
after the user's continuation. The cached-19 job completed during the interruption.
This was not an uninterrupted overnight research run.

- Current verified rank sum: **44**, tied with one other team behind five teams
  (computed tied sixth by rank sum). RobotPI remains at **9**. Snapshot:
  `../.tmp/lb_20261102_1248/all.json`, downloaded 13:48 Berlin. Our five ranks are
  10 / 9 / 12 / 12 / 1. The earlier sum 40 below is a historical starting point.
- **Lane A completed and rejected for promotion.** The user approved both N19
  uploads and reported they were scored; receipts confirm all ten results.
  N19H: 13.61196 / 17.58893 / 0.13669 / 0.07061 / 0.00282.
  N19HO: 13.61196 / 19.32814 / 0.13669 / 1.37088 / 0.00295.
  The paired object comparison decisively rejects raw CoCoNet objects. No duplicate
  uploads or further full-stage CARI4D batch is planned.
- **Lane B built and locally validated.** Full PVA predictions and MHR joint caches
  exist for all 30 clips. PVA_ACC changes only root translation (maximum 1.176 cm).
  PVA_POSE fits body articulation and root motion (maximum actual joint movement
  4.256 cm). Both preserve the scored anchor, identity and object arrays exactly.
  Body refinement improves independent 2D mean error on 22/30 clips; this is not
  evidence of a Kaggle improvement. New reconstruction code must be published with
  authorization before packing a submission that truthfully cites it.
- All compute used **wr20 only**: cached19 job439078, predictor/root job443691,
  corrected body-pose job443714. All are complete; the old all-stage jobs remain
  cancelled. No Track 2 node/job has been queried.

See [PVA_RESEARCH_2026-11-02.md](PVA_RESEARCH_2026-11-02.md) for the actual method,
provenance, measured validation and next decision gates.

Target: improve the official five-board rank sum, with a morning review by
08:00 Europe/Berlin on November 2. Beating RobotPI Lab is the objective, not a
guaranteed outcome. Latest measured standings: Izus_systems sum 40, overall fifth;
RobotPI Lab sum 9. The deadline is November 4 at 23:00 Europe/Berlin.

## Boundaries and preserved controls

- All project work stays under `/work/mnidgu2s/project1`.
- All GPU jobs explicitly request wr20, one node. Never query Track 2 jobs or nodes.
- Development branch is `vpq-vitpose`; existing data and containers remain under
  `repo`. Runtime scripts record this explicitly instead of silently switching roots.
- Preserve P3 (CD-H 13.40710, CD-O 17.51622), P2 (ACC-H 0.13616), WF (ACC-O
  0.07051), and penzero_v1 (PEN 0). Promote per metric only after actual scoring.
- Keep the original cancelled all-stage CARI4D jobs stopped. No rendering or
  post-optimization is needed to evaluate the 19 completed CoCoNet outputs.
- Kaggle quota is shared across teammates. Preflight before uploads, limit the
  initial overnight comparison to two complete files, and reserve remaining quota
  for a validated new method. Never treat an upload marker as a scored result.

## Research conclusions

1. **Joint reconstruction is still substantially untested.** NVIDIA's CARI4D
   integrates human/object evidence and predicts joint spatial corrections. Our
   C1/C2 tests replaced only six humans; CARI4D's own objects have not been scored.
   Nineteen CoCoNet outputs are already validated by their stage markers. This is
   the first experiment because it needs conversion, not another video pass.
   Source: https://github.com/nvidia-isaac/video_to_data/blob/main/reconstruction/modules/v2d_cari4d/README.md
   Paper: https://arxiv.org/abs/2512.11988
2. **Acceleration needs image-derived dynamics.** Generic smoothing and sharpening
   have both worsened our scores. HTD-Refine's PVA-Net predicts 2D keypoints and 3D
   velocity/acceleration directly from video, then uses them as optimization targets.
   It publishes code and a checkpoint link and expects 30 FPS, matching Track 1.
   Its standard optimizer targets SMPL/SMPL-X; our submission uses MHR. Test the
   predictor first, then implement a tested joint/coordinate adapter and MHR
   optimization rather than pretending the standard CLI supports MHR.
   Source: https://github.com/ant-research/HTD-Refine
   Paper: https://arxiv.org/abs/2605.26879
3. **A new complete reconstruction stack has a larger setup burden.** GVHMR offers
   gravity-aware human motion but needs an SMPL-X-to-MHR transfer. RHINO combines
   object, human and scene reconstruction with substantial preprocessing and
   sequence optimization. Their geometry/contact ideas are useful references;
   prioritize the cached joint predictor and directly relevant dynamics model.
   Sources: https://github.com/zju3dv/GVHMR and https://github.com/lxxue/RHINO
4. **Retain evidence that already transferred.** C5 removed the floor correction
   and worsened CD-H by 0.62391 cm. Keep beta 0.55. Learned depth v1/v2, global
   object shrinkage, WHAM root blending, and repeated generic smoothing are not
   overnight candidates. C1 versus C2 changes both stage and smoothing, so it does
   not isolate the effect of post-optimization.

## Lane A: score the cached joint model

1. Revalidate the 19 stage outputs and snapshot the exact episode list.
2. On one wr20 GPU, decode and convert their CoCoNet human meshes to the official
   MHR representation, retaining the previous lambda 300 and floor beta 0.55.
3. Build **N19H**: those 19 humans, V3 fallback elsewhere, and the previous OTa
   objects. Compare CD-H against V3 and P3; compare ACC-H independently.
4. Build **N19HO**: identical humans/fallback, but CARI4D objects on those same
   19 episodes. N19HO versus N19H isolates the object choice. Confirm human
   parameters match exactly before uploading.
5. Validate all 30 episodes, finite arrays, rotations, frame coverage, anchor
   behavior and official sample row IDs. Pack with the published reconstruction
   commit that actually contains the unchanged builders. Save scores and receipts.
6. If the object prediction helps, transfer it to the strongest human while
   preserving its initial similarity relation. If human gains are credible,
   evaluate a fixed-weight mesh ensemble with P3 before spending on missing clips.
   Do not select per-episode weights from leaderboard responses.

## Lane B: image-conditioned motion dynamics

1. Download the public HTD-Refine code/checkpoint into project-local directories;
   pin the source commit, retain its license, and record model hashes.
2. Inspect dependency and tensor contracts. Reuse Track 1 person masks for boxes
   and the same camera intrinsics for prediction and fitting. Avoid a second
   detector and avoid unrelated body-model downloads for predictor-only inference.
3. Run a short representative pilot on wr20. Verify frame order, 30 FPS units,
   camera axes, joint names, finite output, overlap assembly and confidence. The
   released predictor uses first-available overlap selection, not blending.
4. Compare predicted 2D motion against visible keypoints and the current MHR
   motion; diagnose any lag. Use these as implementation/observability checks,
   not as a substitute for held-out 3D accuracy.
5. If the evidence is usable, adapt the predictor's joint dynamics to MHR. Preserve
   the anchor and identity, optimize using visual evidence and bounded movement
   from the baseline, and retain low-frequency position to prevent integration
   drift. Test known translations/rotations, constant velocity/acceleration,
   missing observations, and scored-frame gaps before applying to real outputs.
6. Run the full clips only after the pilot passes. Evaluate acceleration separately
   from position; a visually smoother result is not automatically better.

## Time and compute gates

- Start Lane A immediately. Its first deliverable is two scored comparisons using
  existing predictions, not a fresh long reconstruction run.
- During Lane A conversion, complete Lane B source/checkpoint setup and a pilot.
- No more than one wr20 node in total, even if separate tasks share it. Start with
  one GPU for conversion; request another GPU on wr20 only when its task is ready.
- Every batch has a wall-time limit, explicit output log and exit status. Failed
  experiments preserve diagnostics and do not trigger blind uploads or retries.
- By 08:00, produce a result table, exact artifacts/receipts, tests, retained/rejected
  hypotheses, current rank sum and remaining work. If no method improves, report
  that plainly; never label an unscored proposal a breakthrough.

## Validation and provenance

New behavior gets analytic regression tests plus an actual pilot. Run the full
local suite before publishing new implementation. Source commits cited by a
submission must contain its reconstruction code; never relabel new code with an
old commit. Existing best files remain immutable. The final offline container and
technical report must eventually cover the selected methods, but rebuilding the
historical BMQ ancestry is not the overnight optimization objective.
