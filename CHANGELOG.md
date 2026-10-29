# Changelog

## 0.1.0 - 2026-10-29

- Initial release of VesselCNN as a standalone model package — the
  portfolio's first volumetric (3D) model.
- From-scratch VesselMNIST3D model (104,722 parameters) trained for
  this release: 18 of a planned 30 epochs on the official 1,335-volume
  train split, seed 42, class-weighted cross-entropy, octahedral
  augmentation, checkpoint selected by validation balanced accuracy
  (epoch 6; val balanced accuracy 0.7887, val AUC 0.8179). Training
  was stopped after epoch 18 on a measured plateau (epochs 7–18 never
  beat the epoch-6 selection key). Training script included
  (`scripts/train.py`, resumable per epoch).
- Verified weights in three formats: `model.safetensors`, the
  training checkpoint (`assets/vesselcnn.pt`), and ONNX
  (`vesselcnn.onnx`, parity vs PyTorch measured at 1.19e-07 max logit
  difference over 32 test volumes, argmax agreement on all 32).
- Every headline metric reproduced from the shipped artifacts
  (`examples/verify_metrics.py`): test accuracy 0.7539, AUC 0.7219,
  balanced accuracy 0.6380, full 2×2 confusion matrix, per-class
  recalls — exact match with the training record. The weak spots are
  documented on the card, not hidden: accuracy is below the 0.8874
  majority-class baseline, aneurysm recall is 0.4884 (21 of 43), and
  aneurysm precision is 0.2258 (72 false alarms against 339 healthy
  segments).
- Failure analysis included (`assets/evaluation_results.json`):
  per-occupancy-quartile accuracy and aneurysm recall, and the full
  false-negative / false-positive inventory with predicted
  probabilities and voxel occupancies.
- Inference API (`load_model`, `predict`, `predict_proba`,
  `preprocess`) with the exact training preprocessing contract.
- Demo (`examples/predict_demo.py`): the four packaged sample volumes
  — selected by a fixed rule that includes a false negative and a
  false positive — plus the first misclassified test volumes.
