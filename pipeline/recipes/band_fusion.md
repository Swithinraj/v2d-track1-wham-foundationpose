# E2: band fusion of the refined and the raw human (learned on multi-view ground truth)

`scripts/band_fusion_refit.py` mixes, per frequency band and axis group, our refined trajectory with the raw
per-frame SAM 3D Body trajectory. The weights are fixed numbers regressed on multi-view ground truth: a linear
temporal estimator of the Wiener type, with no per-episode constants.

- **Bands:** Gaussian differences at σ = 1/2/4/8 frames, giving B0 > 4.8 Hz, B1 2.4-4.8, B2 1.2-2.4, B3 0.6-1.2 and B4 < 0.6 Hz at 30 fps.
- **Axis groups:** lateral (perpendicular to the camera z axis) and depth.
- **Pinned:** B4 stays ours (k = 1, m = 0), so the low-frequency position and the registration are untouched. The first scored frame is pinned as well (the change tapers to zero over ±10 frames). The objects are unchanged.
- **Shrinkage:** strength 0.5, i.e. halfway toward "ours unchanged". This was pre-registered.

## Coefficients (`config/band_fusion_coef.json`)

GT acceleration was regressed on the bands of the refined (VPQ3-base) and the raw human. The data were 59 FORM-HOI sequences (`nvidia/form-hoi`, CC BY-4.0): two disjoint sets of 30, one catastrophic episode excluded, and the two fits averaged. External data is allowed by the challenge FAQ. No test sequence and no test object is in the set. The two independent fits agree closely:

| | lateral, ours B0 | lateral, raw B3 | depth, ours B0 | depth, raw B3 |
|---|---|---|---|---|
| fit on set A | 0.50 | 0.93 | 0.80 | 0.32 |
| fit on set B | 0.54 | 0.88 | 0.89 | 0.31 |

## Validation

FORM-HOI, cross-fitted (coefficients from the other set), after the MHR refit, official metric code. Healthy episodes only (one catastrophic episode excluded):

| Base | ACC-H, set A | ACC-H, set B | CD-H, set A | CD-H, set B |
|---|---|---|---|---|
| VPQ3 | −0.0020 (28/29) | −0.0015 (25/30) | −0.012 (20/29) | −0.017 (21/30) |
| BMQ3-K | −0.0022 (28/29) | −0.0017 (26/30) | −0.018 (25/29) | −0.018 (21/30) |

On the challenge episodes, BMQ3-K + E2 against BMQ3-K:

| Check | Result |
|---|---|
| ViTPose error | 5.24 px vs 5.41 px |
| Wrist error | 4.99 px vs 5.26 px |
| Noise-cancelling 2D acceleration proxy | MediaPipe −10.4 (better in 26/30), ViTPose −10.9 (better in 28/30) |
| Estimated ACC-H change | −0.0011 |
| Anchor and objects | unchanged |
| Self-PEN | 0.000292 vs 0.000291 |
| Survival through the contact stage | 0.998 |
| Articulation self-acceleration | ×1.15 |

The self-acceleration rise is real motion added. The ground truth confirms it lowers the acceleration error.

## Reproduce

```
pipeline/recipes/band_fusion_final.sh episodes_bmq3k_in episodes_bm_depth episodes_cand_BMQ3KE
```

The base comes from `bmq_final.sh`, then `bmqk_final.sh`, then `bmq3k_final.sh`.
