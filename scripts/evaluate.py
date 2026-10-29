"""Evaluate the packaged VesselCNN weights on the VesselMNIST3D test
split and produce the failure analysis + packaged sample volumes.

Outputs (defaults):
  assets/evaluation_results.json  — full test metrics, occupancy
                                    analysis, error inventory
  assets/samples/*.npy            — sample test volumes (uint8 28^3)
  assets/samples/index.json       — per-sample label + the prediction
                                    measured from the packaged .npy
                                    file itself (via vesselcnn.infer)

Sample selection rule (fixed, stated here so it cannot be mistaken
for cherry-picking): from the test split, in test order —
  * the first false negative (aneurysm read as vessel),
  * the first false positive (vessel read as aneurysm),
  * the correct aneurysm with the highest predicted aneurysm
    probability,
  * the correct vessel whose test index is the median of the correct
    vessels.
If a category does not exist in the evaluation (e.g. no false
negatives), it is simply absent and the index records the rule.

Usage:
  python scripts/evaluate.py [--weights model.safetensors]
      [--data /path/to/vesselmnist3d.npz] [--out assets/evaluation_results.json]
      [--samples-dir assets/samples]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,  # noqa: E402
                             confusion_matrix, precision_score,
                             recall_score, roc_auc_score)

from vesselcnn.infer import LABELS, TRAIN_MEAN, TRAIN_STD, load_model, predict  # noqa: E402
from vesselcnn.model import build_model  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
NUM_CLASSES = 2


@torch.no_grad()
def evaluate(model, images_u8, batch=64):
    model.eval()
    xt = torch.from_numpy(images_u8)[:, None]
    probs = []
    for i in range(0, len(xt), batch):
        xb = (xt[i:i + batch].float() / 255.0 - TRAIN_MEAN) / TRAIN_STD
        probs.append(torch.softmax(model(xb), dim=1))
    probs = torch.cat(probs).numpy()
    return probs, probs.argmax(1)


def full_metrics(y_true, probs, preds):
    cm = confusion_matrix(y_true, preds, labels=list(range(NUM_CLASSES)))
    rec = recall_score(y_true, preds, labels=list(range(NUM_CLASSES)),
                       average=None, zero_division=0)
    prec = precision_score(y_true, preds, labels=list(range(NUM_CLASSES)),
                           average=None, zero_division=0)
    return {
        "accuracy": round(float(accuracy_score(y_true, preds)), 4),
        "auc": round(float(roc_auc_score(y_true, probs[:, 1])), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, preds)), 4),
        "confusion_matrix": cm.tolist(),
        "per_class_recall": {LABELS[i]: round(float(rec[i]), 4) for i in range(NUM_CLASSES)},
        "per_class_precision": {LABELS[i]: round(float(prec[i]), 4) for i in range(NUM_CLASSES)},
        "class_counts": np.bincount(y_true, minlength=NUM_CLASSES).tolist(),
        "majority_baseline": round(float(np.bincount(y_true).max() / len(y_true)), 4),
        "n": int(len(y_true)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.path.join(PKG, "model.safetensors"))
    ap.add_argument("--data", required=True, help="path to vesselmnist3d.npz")
    ap.add_argument("--out", default=os.path.join(PKG, "assets", "evaluation_results.json"))
    ap.add_argument("--samples-dir", default=os.path.join(PKG, "assets", "samples"))
    args = ap.parse_args()

    d = np.load(args.data)
    x_test = d["test_images"]
    y_test = d["test_labels"].ravel().astype(np.int64)

    model = load_model(args.weights)
    probs, preds = evaluate(model, x_test)
    metrics = full_metrics(y_test, probs, preds)

    occupancy = (x_test > 0).sum(axis=(1, 2, 3))
    correct = preds == y_test
    quartiles = np.quantile(occupancy, [0.25, 0.5, 0.75])
    occ_rows = []
    edges = [-1, quartiles[0], quartiles[1], quartiles[2], occupancy.max()]
    for q in range(4):
        m = (occupancy > edges[q]) & (occupancy <= edges[q + 1])
        row = {"quartile": q + 1, "occupancy_range": [int(edges[q] + 1), int(edges[q + 1])],
               "n": int(m.sum()), "accuracy": round(float(correct[m].mean()), 4)}
        pos = m & (y_test == 1)
        row["aneurysm_n"] = int(pos.sum())
        row["aneurysm_recall"] = (round(float(correct[pos].mean()), 4)
                                  if pos.sum() else None)
        occ_rows.append(row)

    fn_idx = [int(i) for i in np.where((y_test == 1) & (preds == 0))[0]]
    fp_idx = [int(i) for i in np.where((y_test == 0) & (preds == 1))[0]]

    def err_rec(i):
        return {"test_index": int(i), "label": LABELS[int(y_test[i])],
                "predicted": LABELS[int(preds[i])],
                "prob_aneurysm": round(float(probs[i, 1]), 4),
                "occupancy_voxels": int(occupancy[i])}

    analysis = {
        "occupancy_quartiles": occ_rows,
        "false_negatives": {"n": len(fn_idx), "items": [err_rec(i) for i in fn_idx]},
        "false_positives": {"n": len(fp_idx), "items": [err_rec(i) for i in fp_idx]},
        "mean_prob_of_correct": round(float(probs[np.arange(len(y_test)), y_test][correct].mean()), 4),
        "occupancy_mean_correct": round(float(occupancy[correct].mean()), 1),
        "occupancy_mean_incorrect": round(float(occupancy[~correct].mean()), 1),
    }

    record = {"split": "test", **metrics, "failure_analysis": analysis}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(record, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"wrote {args.out}")

    # --- packaged samples, predictions re-measured from the .npy files ---
    os.makedirs(args.samples_dir, exist_ok=True)
    picks = {}
    if fn_idx:
        picks["false_negative"] = fn_idx[0]
    if fp_idx:
        picks["false_positive"] = fp_idx[0]
    correct_an = [int(i) for i in np.where((y_test == 1) & (preds == 1))[0]]
    if correct_an:
        picks["correct_aneurysm_most_confident"] = max(correct_an, key=lambda i: probs[i, 1])
    correct_v = [int(i) for i in np.where((y_test == 0) & (preds == 0))[0]]
    if correct_v:
        picks["correct_vessel_median"] = correct_v[len(correct_v) // 2]
    index = {"selection_rule": (
        "first false negative, first false positive, most-confident correct "
        "aneurysm, median correct vessel — all in test order, from the "
        "packaged-weights evaluation in assets/evaluation_results.json"),
        "samples": []}
    for role, i in picks.items():
        fname = f"{role}_{i}.npy"
        np.save(os.path.join(args.samples_dir, fname), x_test[i])
        r = predict(model, os.path.join(args.samples_dir, fname))
        index["samples"].append({
            "file": fname, "role": role, "test_index": int(i),
            "label": LABELS[int(y_test[i])],
            "predicted": r["label"],
            "probabilities": {k: round(v, 4) for k, v in r["probabilities"].items()},
            "occupancy_voxels": int(occupancy[i]),
        })
        print(f"sample {fname}: label={LABELS[int(y_test[i])]} predicted={r['label']}")
    with open(os.path.join(args.samples_dir, "index.json"), "w") as f:
        json.dump(index, f, indent=2)
    print(f"wrote {os.path.join(args.samples_dir, 'index.json')}")


if __name__ == "__main__":
    raise SystemExit(main())
