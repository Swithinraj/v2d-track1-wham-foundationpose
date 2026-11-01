# Floor contact: depth from the room, not from a depth network

Candidates on this page start from a **scored submission file**, so each change is measured against a known score. They use
no learned parameters.

## Why

All challenge videos come from one room with a flat carpet, a static camera about 1.3 m above the floor, and the person's
feet in view. Many objects also stand on that floor. A point on the floor has only one possible depth for its pixel, and
this does not depend on any depth network.

Two things the scorer does make this useful:

- It fits one similarity on our person at the first scored frame and applies it to the whole episode. So what counts for
  the person is motion relative to that frame.
- It places the object through that same similarity. So what counts for the object is its depth **relative to the person**.

## The steps

| Step | Script | What it does |
|---|---|---|
| Baseline | `scripts/parquet_episodes.py` | Turns a submission file back into episode files. The round trip through the official packer is exact: pose, shape and translation are identical, rotation and mesh within 1e-7. |
| Camera | `scripts/true_cam_params.py` | The measured rig camera (`truecam.md`). |
| Floor normal | `scripts/floor_planes.py` | MoGe's normal, averaged over episodes of the same camera mounting, then mapped into the measured camera. |
| Person | `scripts/floor_root.py` | Keeps stance feet at the anchor's floor height by sliding the body along the ray through the stance foot. The first scored frame does not move, so CD-O cannot change through the alignment. |
| Object | `scripts/object_floor_scale.py` | Scales an object about the camera so its resting level meets the person's floor. Every pixel stays where it is. |
| Contact | `pipeline/recipes/contact_stage_split.sh` | The production two-pass contact stage, split across GPUs. |

Because the first scored frame stays fixed, **one file tests two things independently**: the person's change shows in CD-H
only and the object's change in CD-O only.

## What Kaggle measured

| File (commit 13ff13a) | CD-H | CD-O | ACC-H | ACC-O | PEN |
|---|---|---|---|---|---|
| BMQ3KGM (reference, MoGe's camera) | 14.35418 | 18.60437 | 0.13629 | 0.07067 | 0.00013 |
| MF: camera + person floor slide + object floor scale | **14.08779** | 18.49797 | 0.13637 | 0.07057 | 0.00019 |
| G97: camera + every object scaled 0.97 | 14.15256 | 22.65556 | 0.13636 | 0.07053 | 0.00016 |
| WF: camera + WHAM root mix 0.25 + floor objects + table objects 0.974 | 14.74125 | 20.91912 | 0.13632 | 0.07051 | 0.00014 |

- The measured camera gives CD-H -0.20 on the best file, and the person's floor slide a further -0.065.
- The object floor scale carries real differences between episodes, but moving objects closer **on average is wrong**: 0.97 on
  everything costs about +4 CD-O, which puts the best global scale near 1.01. A resting level below the person's lowest vertex
  is therefore not evidence of a mean depth error. The sensitivity is about 1.25 CD-O per 1 % of scale.
- Objects on tables should stay where they are: 0.974 costs +5.6 CD-O per such episode.
- Mixing in WHAM's trajectory is rejected: +0.59 CD-H at weight 0.25.

## What was measured without ground truth

On BMQ3KGM in the measured camera:

- 17 of 30 objects touch the floor, and their resting level sits a median 2-6 cm below the person's lowest vertex.
- Within objects, size constancy across episodes agrees with the floor scale (correlation 0.55, slope 0.62, 15 comparisons,
  `scripts/analysis/size_constancy.py`).
- The person's per-frame slide does not agree with apparent body size (pooled correlation 0.13), and a per-episode depth stretch
  fits about 1.0 from both cues. The slide helped on Kaggle anyway, so this cross-check was not a reliable predictor.
- Hand contact neither confirms nor refutes the object scale: for many objects a wide range of scales already touches the hand
  (`scripts/analysis/kappa_hand.py`).
- Candidates built on SAM 3D Body differ by about 2 cm per vertex and 1.6 cm in body position, against about 7 cm of real error
  implied by CD-H (`scripts/analysis/hat.py`), so averaging them can only gain a few tenths.

## Local gates

`scripts/local_gates.py` compares self-penetration and self-acceleration before a file costs a submission. These are not
CD-H or CD-O: only Kaggle measures those.

| Candidate | self-PEN | self-ACC-H | self-ACC-O |
|---|---|---|---|
| BMQ3KGM (scored) | 0.00008 | 0.06403 | 0.03154 |
| Mtp (measured camera) | 0.00008 | 0.06429 | 0.03150 |
| MF (+ person and object floor) | 0.00016 | 0.06432 | 0.03099 |

## Reproduce

```
srun --gres=gpu:4 ... bash -c 'FLOOR_HUMAN=1 FLOOR_OBJECTS=1 bash pipeline/recipes/floor_final.sh SUBMISSION.parquet MF'
bash pipeline/recipes/pack_candidate.sh MF
.venvs/submission/bin/python scripts/submit_candidate.py submissions/track1_cand_MF.parquet -m "what changed"
```

Kaggle results are recorded in `docs/CLUSTER_PROGRESS.md`.
