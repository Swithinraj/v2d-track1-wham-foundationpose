# Cluster geometry implementation plan

> Execute inline using executing-plans and test-driven-development. The user's
> existing autonomous implementation request supersedes extra design approval gates.

**Goal:** produce tested, reproducible candidates and measure whether they improve
the recorded challenge baseline toward CD-H/CD-O <10 cm.

**Architecture:** retain MHR episode NPZ/GLB interfaces and existing GRU/contact/pack
stages; add fail-closed depth validation and independent geometric calibration.

**Tech stack:** Python 3.11, NumPy/SciPy, pytest, PyTorch/MHR in Apptainer, Slurm.

**Spec:** `docs/CLUSTER_EXPERIMENT_DESIGN.md`.

## Global constraints

All project files/caches in `/work/mnidgu2s/project1`; GPU work on `wr20` only.
No changes to existing jobs. Preserve baselines and keep outputs separate.
No git commit/push absent the required score evidence and user authorization.
No success claim without measured evidence. No borrowed commit for new-code artifacts.

## Review focus

Invalid and empty depth maps; partial corrupted frames; unobservable contact offsets;
hand/body coordinate conventions and units; stale or missing model/episode artifacts.

## Task 1: Restore and verify evidence

- [ ] Restore cached artifacts and official kit, inspect current Kaggle history.
- [ ] Record ancestry mismatch and actual available inputs; lock dependency versions.
- [ ] Run one-node MHR smoke test before full inference.

## Task 2: Fail-closed depth locking

Files: `scripts/moge_lock.py`, `tests/test_moge_lock.py`.
Preserve `lock(depth_dir, T, step)` return tuple; insufficient frames use identity
scale and an invalid static mask; floor fitting rejects unsupported geometry.

- [ ] Write tests with encoded depth PNGs: recover known global scale, all-invalid
  and sparse maps return finite identity, single bad frame does not poison others,
  insufficient/collinear floor evidence is rejected.
- [ ] Run tests and observe failures, implement validity checks, rerun full suite.

## Task 3: Constant contact calibration

Files: `scripts/contact_translation.py`, `tests/test_contact_translation.py`.
Pure numerical solver receives local mesh points/normals, per-frame object R/t,
independent hand samples and evidence masks. Return translation and a diagnostic
accept/reject decision. A CLI writes a new episode directory and report.

- [ ] Test known offset recovery, robust outliers, already aligned input, no contact,
  planar ambiguity, array mismatch, finite output and exact acceleration invariance.
- [ ] Implement robust point-to-surface fitting with bounded displacement and
  observability/temporal-consistency gates; reject unsupported corrections.
- [ ] Test full NPZ/GLB round trip and preservation of unrelated data.

## Task 4: Cluster execution and validation

- [ ] Run existing-chain checkpoint ablation with depth cues and `eps_scale=0`.
- [ ] Run constant-offset proposal on independent pre-contact hands; measure image
  consistency and physical gates before contact solve and official packing.
- [ ] Test floor ablation only after recovering a valid floor and contact evidence.
- [ ] Record local checks, actual scores if legitimately submittable, rejected
  hypotheses, exact reproduction commands, and remaining blockers.
- [ ] Run full tests and obtain an independent final code review.
