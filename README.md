---
license: mit
pipeline_tag: image-classification
language: en
tags:
  - medical-imaging
  - medmnist
  - vesselmnist3d
  - 3d-cnn
  - intracranial-aneurysm
  - pytorch
  - onnx
  - safetensors
---

# VesselCNN — VesselMNIST3D vessel/aneurysm classifier

**A compact 3D CNN (104,722 parameters), designed and trained from scratch for binary classification of 28×28×28 voxelized intracranial vessel segments (vessel vs. aneurysm), released with something most model uploads don't have: a script that re-derives every headline metric from the shipped weights and data, so the numbers below are checkable, not just claimed.**

Read the weak result plainly, because it is the result: **on the held-out test split this model is not a usable aneurysm detector.** Its accuracy (0.7539) is *below* the 0.8874 majority-class baseline, it catches only 21 of 43 aneurysm segments (recall 0.4884), and 72 of its 93 aneurysm predictions are false alarms (precision 0.2258). Its AUC of 0.7219 trails the MedMNIST v2 paper's reference ResNet-18 (3D) result of 0.874 on the same split by about 15 points. What this release demonstrates is a complete, honest, re-derivable 3D training run — data card, training log, failure analysis and all — on a hard, tiny, imbalanced volumetric benchmark. It is the first volumetric model in a portfolio whose other medical models are 2D (SonoCNN, PathoCNN).

> **Research and benchmark use only. This is a 28×28×28 benchmark model trained on automatically generated vessel segments. It is not a medical device, it has never been evaluated on clinical MRA data, and it must not be used for diagnosis or screening.**

## Verified results — VesselMNIST3D test split (n = 382)

| Metric | Value | Source |
|---|---|---|
| Accuracy | **0.7539** (288/382) | reproduced 2026-10-29 |
| AUC | **0.7219** | reproduced 2026-10-29 |
| Balanced accuracy | 0.6380 | reproduced 2026-10-29 |
| Majority-class baseline | 0.8874 | reproduced 2026-10-29 |
| Best validation balanced accuracy | 0.7887 (epoch 6) | recorded at training time |
| Best validation AUC | 0.8179 (epoch 6) | recorded at training time |

Per-class results on the test split (all reproduced):

| Class | Recall | Precision | Support |
|---|---|---|---|
| vessel | 0.7876 (267/339) | 0.9239 | 339 |
| **aneurysm** | **0.4884 (21/43)** | **0.2258** | 43 |

Confusion matrix (rows = true, columns = predicted): vessel 267 correct / 72 read as aneurysm; aneurysm 22 read as vessel / 21 correct.

The failure analysis (full record in `assets/evaluation_results.json`) found one dominant pattern: **the model keys on segment bulk more than on aneurysm shape.** Aneurysm recall rises monotonically with the volume's non-zero voxel count — 0.0 in the second occupancy quartile (250–405 voxels, 10 aneurysms, all missed), 0.3333 in the third, 1.0 in the bulkiest quartile — while accuracy falls from 1.0 in the sparsest quartile (which contains no aneurysms at all) to 0.4632 in the bulkiest. Errors are concentrated in bulky volumes: mean occupancy 653 voxels for errors vs. 365 for correct predictions. Small aneurysm segments are missed; large healthy segments are flagged. One layout fact for anyone re-deriving the per-index records: in the released test array the 43 aneurysm volumes occupy indices 0–42 (the split is label-ordered); the metrics are order-invariant, but the error inventory reads accordingly.

Context, not a claim of parity: the MedMNIST v2 paper's reference results on this split are ResNet-18 (3D) at AUC 0.874 / ACC 0.877 and ResNet-18 with ACS convolutions at AUC 0.930 / ACC 0.928 — models roughly 100× this one's size, trained on the same data. VesselCNN does not approach them; the gap is the honest price of 104,722 parameters, 150 positive training volumes, and a deliberately simple recipe.

## Quickstart

```bash
pip install "git+https://github.com/habib-analyst/vesselcnn.git"
```

```python
from vesselcnn import load_model, predict

model = load_model("model.safetensors")
result = predict(model, "assets/samples/correct_aneurysm_most_confident_40.npy")
# {'label': 'aneurysm', 'confidence': 0.719,
#  'probabilities': {'vessel': 0.281, 'aneurysm': 0.719}}
```

Inputs are single volumes: a `.npy` path or a NumPy array of shape (28, 28, 28), uint8 as released. Preprocessing contract (identical to training): /255 → normalize with the VesselMNIST3D train-split mean/std (**0.01933 / 0.13768**, scalar — the volumes are single-channel). Labels are the MedMNIST v2 VesselMNIST3D order: 0 = vessel, 1 = aneurysm (see `config.json`).

ONNX: `vesselcnn.onnx` takes the same normalized `(N, 1, 28, 28, 28)` float tensor and returns logits.

## Verify it yourself

```bash
python examples/verify_metrics.py
```

This re-evaluates the shipped weights on the bundled test split (`assets/vesselmnist3d_test.npz`) and compares against the training record. On 2026-10-29 it printed `ALL HEADLINE METRICS REPRODUCED` — accuracy, AUC, balanced accuracy, the full confusion matrix, and both per-class recalls matched exactly.

```bash
python examples/predict_demo.py   # packaged samples + first misclassifications
python -m pytest -q               # 8 tests: formats agree, metrics files agree, parity
```

The demo is honest about the weak spots by construction: the four packaged samples were selected by a fixed rule (see `assets/samples/index.json`) that includes the first false negative (an aneurysm read as vessel at 0.690 confidence) and the first false positive. Of the four, two are classified correctly and two are errors, and the demo prints all four.

## Model details

- **Architecture:** 3 conv blocks (16→32→64 channels, 3×3×3 convolutions, BatchNorm, ReLU, max-pool, dropout) + global-average-pooling head with a single linear layer, designed for this release in the same compact family as SonoCNN / PathoCNN, re-derived for volumetric input. The GAP head is deliberate: a flattened MLP head would add ~1.4M parameters against only 1,335 training volumes. Max-pooling (not average) is also deliberate — the volumes are sparse (1.9% of voxels non-zero) and max-pooling preserves the bright thin vessel structures. 104,722 trainable parameters; 0.4 MB in safetensors.
- **Training:** from scratch on the VesselMNIST3D train split (1,335 volumes), AdamW (lr 2e-3, weight decay 1e-4, OneCycle cosine over the planned schedule), batch 32, seed 42. Loss: class-weighted cross-entropy with weights [0.5633, 4.4500] = N/(2·n_class) for the train split's 1,185 vessel / 150 aneurysm volumes — plain cross-entropy on an 88.8%-majority split would pay the model to ignore the aneurysm class. The weighting buys sensitivity at the cost of the 72 false alarms above; both sides of that trade are in the metrics, not just the flattering one.
- **Run record:** 30 epochs were planned; **training was stopped after epoch 18 on a measured plateau** — epochs 7–18 never beat the epoch-6 selection key, train loss stayed in 0.61–0.65 and validation AUC in 0.79–0.82 from epoch 5 on. The shipped weights are the epoch-6 checkpoint, selected by best validation balanced accuracy (ties by AUC). Summed epoch time was 3,227 s (~54 min) on a 2-core CPU. The exact script is `scripts/train.py` (resumable per epoch); the full per-epoch log is `training_log.txt`.
- **Augmentation:** a random element of the octahedral group per volume per epoch (random axis permutation composed with random axis flips; 48 transforms). Vessel segments are cut from vessel trees at arbitrary orientations and the label is a local shape property, so every element of the group is label-preserving. Training split only; no intensity augmentation.

## What VesselCNN is not

- **Not a clinical tool.** It classifies 28³ benchmark segments generated from 103 source vessel models, not patients or scans — and at 0.4884 aneurysm recall it would miss half the aneurysms in even this benchmark. Nothing about it transfers to a screening workflow.
- **Not state of the art, or close.** Reference ResNets score AUC 0.87–0.93 on this split (above); this model's 0.7219 is far off. The release exists because its numbers are verified and its failure modes are documented, not because it tops a leaderboard.
- **Not above its own majority baseline.** Accuracy 0.7539 < 0.8874 for always predicting "vessel". The class weighting deliberately trades raw accuracy for aneurysm sensitivity; whether that trade is worth it depends on the cost of a miss vs. a false alarm, and at precision 0.2258 this particular trade mostly is not.
- **Not trained on private or clinical data.** Only the public VesselMNIST3D benchmark (below).
- **Confidences are not calibrated.** Softmax outputs are modest in range here (the most confident correct aneurysm scores 0.719) but they are still rankings, not probabilities of being right — the false-negative sample is read as vessel with 0.690 confidence.

## Data and attribution

VesselMNIST3D is part of **MedMNIST v2** (Yang et al., *MedMNIST v2 — A large-scale lightweight benchmark for 2D and 3D biomedical image classification*, Scientific Data 2023). It derives from **IntrA** (Yang et al., 2020), an open-access dataset of 3D intracranial aneurysm models reconstructed from MRA images: 1,694 healthy and 215 aneurysm segments generated from 103 complete vessel models, meshes repaired with PyMeshFix and voxelized with trimesh. The dataset is licensed **CC BY 4.0**; `assets/vesselmnist3d_test.npz` (the official test split only) is redistributed here for metric reproduction, with attribution. The source file used for training is Zenodo record 10519652, md5 `b41fd4f7e7e2feedddb201585ecafa1b` (full data card: `DATA.md`). One caveat carried from the data card: the release carries no per-volume source-model identifier, so exact-duplicate volumes across splits were verified absent, but near-duplicates from adjacent segments of the same source vessel model cannot be excluded — read the test metrics with that in mind. The MIT license of this repository covers the code and the trained weights, not the dataset.

## Files

| File | Contents |
|---|---|
| `model.safetensors` | Release weights (safetensors) |
| `vesselcnn.onnx` | ONNX export (logits) |
| `config.json` | Architecture + preprocessing config |
| `DATA.md` | Dataset card: source, hashes, splits, class balance, caveats |
| `training_log.txt` | Per-epoch training log, including the early-stop record |
| `src/vesselcnn/` | `model.py` (architecture), `infer.py` (loading + prediction API) |
| `scripts/train.py` | The exact training script that produced the weights (resumable) |
| `scripts/evaluate.py` | Test evaluation + failure analysis + sample packaging |
| `scripts/export_onnx.py` | ONNX export + parity check |
| `scripts/download_data.sh` | Dataset fetch with MD5 verification |
| `examples/predict_demo.py` | Packaged samples + first misclassifications, printed honestly |
| `examples/verify_metrics.py` | Full metric reproduction from the artifacts |
| `assets/vesselcnn.pt` | Training checkpoint (state dict + selection record) |
| `assets/vesselcnn_metrics.json` | Training record + merged test metrics |
| `assets/evaluation_results.json` | Reproduced test metrics + full failure analysis |
| `assets/vesselmnist3d_test.npz` | VesselMNIST3D official test split (CC BY 4.0, see attribution) |
| `assets/samples/` | Four test volumes (.npy) + `index.json` with measured predictions |

## License

MIT for code and weights (see `LICENSE`). Dataset under CC BY 4.0 as above.
