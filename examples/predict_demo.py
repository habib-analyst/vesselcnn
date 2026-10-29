"""Demo: run the packaged VesselCNN weights over the packaged sample
volumes (assets/samples/) and the first misclassified test volumes,
printing the predictions exactly as measured — errors included.

The sample volumes were selected from the test evaluation by a fixed
rule (see assets/samples/index.json): a false negative and a false
positive are in the set on purpose. A demo that only shows correct
predictions would misrepresent a model whose aneurysm recall is the
number that matters.

Usage:
  python examples/predict_demo.py [--weights model.safetensors]
      [--test-data assets/vesselmnist3d_test.npz]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vesselcnn.infer import LABELS, load_model, predict  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.path.join(PKG, "model.safetensors"))
    ap.add_argument("--test-data", default=os.path.join(PKG, "assets", "vesselmnist3d_test.npz"))
    args = ap.parse_args()

    model = load_model(args.weights)

    print("== packaged samples (assets/samples/) ==")
    index = json.load(open(os.path.join(PKG, "assets", "samples", "index.json")))
    truth = {s["file"]: s for s in index["samples"]}
    for path in sorted(glob.glob(os.path.join(PKG, "assets", "samples", "*.npy"))):
        name = os.path.basename(path)
        r = predict(model, path)
        t = truth[name]
        mark = "OK " if r["label"] == t["label"] else "ERR"
        print(f"[{mark}] {name}: true={t['label']} predicted={r['label']} "
              f"(confidence {r['confidence']:.3f}, role: {t['role']})")

    if os.path.exists(args.test_data):
        print("\n== first misclassified test volumes ==")
        import torch
        from vesselcnn.infer import TRAIN_MEAN, TRAIN_STD
        d = np.load(args.test_data)
        x, y = d["test_images"], d["test_labels"].ravel().astype(np.int64)
        shown = 0
        with torch.no_grad():
            for i in range(len(x)):
                r = predict(model, x[i])
                if r["label"] != LABELS[int(y[i])] and shown < 5:
                    print(f"test[{i}]: true={LABELS[int(y[i])]} predicted={r['label']} "
                          f"(confidence {r['confidence']:.3f}, "
                          f"occupancy {(x[i] > 0).sum()} voxels)")
                    shown += 1
        print(f"(first {shown} errors shown; the full error inventory is in "
              "assets/evaluation_results.json)")


if __name__ == "__main__":
    raise SystemExit(main())
