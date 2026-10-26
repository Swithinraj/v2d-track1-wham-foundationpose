# Current best submission (2026-10-27)

**BMQ3KGM** is the best single file on every Track 1 board: built from commit `f60f8a6`. Its recipes are unchanged in
this commit.

| File | CD-H | CD-O | ACC-H | ACC-O | PEN | Rank sum (single file, 10-25 board) |
|---|---|---|---|---|---|---|
| **BMQ3KGM** | **14.35418** | 18.60437 | **0.13629** | **0.07067** | **0.00013** | **39** |
| BMQ3KGMT (+ common-translation selector, `a0d93f8`) | 14.41266 | 18.58305 | 0.13629 | 0.07067 | 0.00013 | (no rank change) |

What BMQ3KGM is: the BMQ3-K pre-contact human (SAM 3D Body in body mode, objects re-expressed in the body-mode depth
scale, VPQ3 lateral band), then the GRU v2 motion head fine-tuned with depth cues, with its depth correction switched
off, then the contact stage with penetration weight 4e6. The depth source of every stage is MoGe-2 (`object_depth.sh`).

Rebuild (after `run_all.sh`):

```
bash pipeline/recipes/bmq_final.sh            # body-mode SAM 3D Body run (BM_ROOT) and episodes_bmq_in
bash pipeline/recipes/bmqk_final.sh           # objects in the body-mode depth scale (episodes_bmqk_in)
bash pipeline/recipes/bmq3k_final.sh          # lateral band -> episodes_bmq3k_in
EPS_SCALE=0 bash pipeline/recipes/gru_depth_final.sh episodes_bmq3k_in episodes_bm_depth mask_visibility_bm BM_ROOT episodes_cand_BMQ3KGM
bash pipeline/stages/pack.sh                  # with the candidate directory as the episodes input
```

The common-translation selector (`gru_shift_final.sh`) is kept in the repository but does not beat BMQ3KGM: on Kaggle it
was CD-H +0.058 and CD-O −0.021, and on two untouched FORM-HOI groups its effect is near zero.

## Under evaluation: Video Depth Anything instead of MoGe-2

Video Depth Anything (metric, Large) as the only depth source of every stage, with no re-tuning. On a held-out FORM-HOI
group, full-mode pre-contact chain against the multi-view ground truth:

| | CD-H | CD-O | ACC-H | ACC-O | PEN |
|---|---|---|---|---|---|
| MoGe-2 | 12.57 | 38.45 | 0.13312 | 0.07354 | 0.00420 |
| VDA | 14.11 | 48.52 | 0.13358 | 0.07286 | 0.00254 |

Several stages were tuned around MoGe-2: the depth-trend strength, the floor / kappa correction, the shared object mesh
scale, and the learned motion head's cues. VDA's absolute scale varies about ±12 % between videos, while MoGe-2's is
nearly constant, and VDA's per-frame depth at the person fluctuates more. The VDA path is being re-tuned on a validation
group. It is not part of any submission.
