# vpq-vitpose branch review — November 2, 2026

The owner requested that ongoing development move to `vpq-vitpose`. This supersedes
the older handoff's instruction to leave that branch untouched. The filesystem
boundary remains `/work/mnidgu2s/project1`; the one additional node remains wr20.

## Checkout and ancestry

- Freshly fetched `origin/vpq-vitpose`: `00c6e73776512b5a04c4d280f2d0e988b27dccc0`.
- That commit is an ancestor of the existing cluster work, not a separate newer
  implementation. The cluster branch contains 14 additional commits.
- Active development checkout: `/work/mnidgu2s/project1/vpq-vitpose`, branch
  `vpq-vitpose`, fast-forwarded locally to `5e0bc96`. No merge conflicts occurred.
- The remote `vpq-vitpose` branch has not been pushed or changed by this transition.
- Existing processes keep `/work/mnidgu2s/project1/repo` on
  `cluster-floor-2026-11`. Do not change that checkout underneath running jobs.
- Local setup and shared-team uploader changes were retained in the new checkout.
  The Python environment and official kit are linked to their existing copies
  inside the allowed project folder. Runtime datasets and outputs remain in the
  original checkout; this is not yet a second self-contained GPU run directory.

## Findings that affect the method

| Change in the upstream branch | Evidence and decision |
|---|---|
| GRU motion head and vertex-chunked refit | Retain. The motion head helped acceleration; chunking reduces memory without changing the intended fusion. The previously proposed checkpoint swap was already scored and did not improve the result. |
| Measured camera mapping | Retain the available correction and its experiments. Apply it after the motion head when using a head trained in the old camera convention. Earlier local results showed worse ACC-H when mapping before the head. |
| Visibility-aware reliability v2 | The implementation fixes visibility handling and the v1 corruption sign, but the recorded Kaggle result is CD-O 18.74770 versus BMQ3KGM 18.60437. Keep it experimental rather than enabling it as an accuracy improvement. |
| Human depth trend disabled | Recorded CD-H worsened from 14.35418 to 14.70203, while CD-O improved from 18.60437 to 18.01952. Keep human and object effects separate. |
| Later cluster corrections | Retained locally: floor/mesh candidates, CARI4D runner, occlusion-diagnostic repair, and root mixing on actual video frame indices. |

Sources are the branch's `docs/EXPERIMENT_REPORT.md`, `docs/CURRENT_BEST.md`,
`docs/CLUSTER_PROGRESS.md`, corresponding recipes and scripts, and the newer
`docs/PIPELINE_CORRECTION_2026-11-02.md`. Historical reports are dated snapshots;
their old best-score claims and pending-experiment lists are not current status.

The recorded best distance scores remain P3: **13.40710 cm CD-H / 17.51622 cm
CD-O**. TIMEAXIS scored **13.52781 / 17.63890**, a small improvement over its V5
control, but it did not beat P3. None of these results establishes a sub-10-cm
method. The full-stage CARI4D jobs were subsequently cancelled during compute
triage, preserving 19 validated CoCoNet outputs. See
`docs/TRACK1_COMPUTE_TRIAGE_2026-11-02.md`. C5's floor-prior ablation worsened
CD-H to 14.11530 and is rejected.

## Verification and execution caveat

Fresh verification in this checkout: **118 tests passed**, and `git diff --check`
passed. This covers local behavior, not a new full GPU reconstruction or a score
for every experimental module.

The cluster-specific wrappers under `pipeline/cluster` were written for the
original absolute runtime directory. Copying them into this worktree does not
retarget their jobs. Review each wrapper's `cd`, asset paths, `TRACK1_ROOT` and
container binds before launching a new experiment from this checkout. Existing
jobs and their outputs continue to belong to the original runtime directory.

No additional node, GPU reconstruction, Kaggle submission, or GitHub push was
launched as part of this branch transition and review.
