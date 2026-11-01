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

## What was measured without ground truth

On BMQ3KGM in the measured camera:

- **Objects are too far.** 17 of 30 objects touch the floor, and their resting level sits a median 2-6 cm below the person's
  floor (physically impossible). The scale that fixes it has median 0.962 and mean 0.974, about 6-15 cm at this distance.
- **Earlier scored files show the dose.** `nowham` has its objects 5.8 % farther than BMQ3KGM and scores CD-O 24.26 against
  18.60, about 1 point per 1 %.
- **The person's floor signal is not confirmed.** The per-frame slide does not agree with apparent body size, an independent
  image cue (pooled correlation 0.13). A per-episode depth stretch fits about 1.0 from both cues (floor 0.99, size 1.01) and
  the two do not agree per episode (correlation -0.04). This is why the person's correction is a Kaggle experiment and not
  an assumed gain. Earlier notes also rejected a hard floor constraint on the grounds that 1 cm of height is 2.4 cm of depth.
- **Hand contact neither confirms nor refutes the object scale.** For many objects a wide range of scales already touches the
  hand (`scripts/analysis/kappa_hand.py`).
- **Existing candidates share their error.** Candidates built on SAM 3D Body differ by about 2 cm per vertex and 1.6 cm in
  body position, against about 7 cm of real error implied by CD-H (`scripts/analysis/hat.py`). Averaging them can only gain a
  few tenths. WHAM-based estimates differ by 6-8 cm, hence the `scripts/human_root_mix.py` experiment.

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
