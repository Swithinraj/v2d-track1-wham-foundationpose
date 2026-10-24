# Jointly trained human-object refiner: object corrections

`scripts/joint_train.py` trains one model with specialised branches.

## Architecture
- **Human encoder:** the band features of `gru.md`, plus 2D-evidence features (ViTPose joints and hand-crop placement against our projection).
- **Object encoder:**
  - track vs raw FoundationPose band disagreements and rotation disagreement;
  - SAM 2 mask centroid and spread against our projected silhouette;
  - MoGe depth inside the mask;
  - motion;
  - validity flags.
- **Visual variant:** a trainable ResNet-18 on per-frame crops of both hands and the object, taken from the episode video.
- **Interaction:** per frame, a transformer over tokens for the body, both hands and the object (SDF hand-object distance and relative motion), followed by a bidirectional GRU over time.
- **Heads:**
  - bounded human and object corrections;
  - uncertainties;
  - per-hand contact;
  - a learned, time-smoothed **object gate**. Every object correction is scaled by the gate, and using the gate is penalised, so well-tracked objects stay untouched.
- **Unrolled optimiser:** differentiable. It moves the object toward the hands in predicted contact, keeps the contact point fixed in object coordinates, and applies one-sided penetration and second-difference smoothness. Its object step is also gated.

## Training
- **Data:** FORM-HOI (nvidia/form-hoi, CC BY 4.0) sequences processed by this pipeline. No challenge data and no Track 2 assets.
- **Losses**, on every output level:
  - joint and hand position;
  - CD-O-style surface Chamfer;
  - acceleration error **against ground truth**, for human and object (object weight 400, rank-balanced against Chamfer);
  - contact BCE and gap closing (object side only);
  - one-sided penetration on scorer-consistent SDF grids;
  - mask-centroid render residual;
  - uncertainty NLL;
  - gate sparsity.
- **Augmentation:**
  - mirror (crops flipped, sides swapped, SDF mirrored);
  - roll and scale about the camera (crops rotated);
  - time reversal;
  - raw-error amplitude;
  - smooth object-depth perturbation;
  - source degradation with validity flags (object hidden, wrist confidence lowered, hand masked);
  - simulated pipeline failures.
- **Schedule:** warm-start the branches → interaction → optimiser → fine-tuning at a lower learning rate.

## Deployment (`joint_objects_final.sh`)
- **Objects only:** the human of the input candidate, for example the GRU candidate, is kept. The object corrections come from the model's final output, then the identical contact stage runs.
- **Models in `config/joint_objects`** (both trained on 150 episodes, checkpoint chosen on a separate group of 30):

| Model | Variant |
|---|---|
| `v6.pt` | non-visual |
| `v6_visual.pt` | visual, ResNet-18 on crops |

## Fresh-group confirmation
Measured on 30 recordings never used in training or checkpoint selection, with the kit metrics after the full contact stage, against the GRU candidate (whose human is kept):

| Variant | CD-O | ACC-O | PEN |
|---|---|---|---|
| Non-visual | median −5.6, mean −11.3 (better in 26/30) | −0.0056 (26/30) | +0.00017 |
| Visual | median −3.3, mean −10.3 (20/30) | −0.0035 (17/30) | −0.0001 |

- **Human metrics:** unchanged in both variants.
- **Well-tracked objects** (baseline CD-O under 20 cm), CD-O: −0.8 cm non-visual (9/11), −0.5 cm visual (6/11).
- **Choosing between the variants on this group** makes it a selection set for that choice.
