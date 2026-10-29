"""Tests for the VesselCNN release package.

The metrics test compares the two JSON records (training-time vs
independently reproduced) instead of re-running the full evaluation;
the sample test compares live predictions on the packaged sample
volumes against the predictions recorded in assets/samples/index.json
at packaging time.
"""
import json
import os
import sys

import numpy as np
import pytest
import torch

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(PKG, "assets")
sys.path.insert(0, os.path.join(PKG, "src"))
sys.path.insert(0, os.path.join(PKG, "scripts"))

from vesselcnn import load_model, predict, preprocess  # noqa: E402
from vesselcnn.infer import TRAIN_MEAN, TRAIN_STD  # noqa: E402
from vesselcnn.model import build_model  # noqa: E402
import train as train_mod  # noqa: E402  (scripts/train.py — augmentation + stats)


@pytest.fixture(scope="module")
def data():
    return np.load(os.path.join(ASSETS, "vesselmnist3d_test.npz"))


@pytest.fixture(scope="module")
def model_pt():
    return load_model(os.path.join(ASSETS, "vesselcnn.pt"))


@pytest.fixture(scope="module")
def model_st():
    return load_model(os.path.join(PKG, "model.safetensors"))


def _batch(data, n=8):
    x = data["test_images"][:n].astype(np.float32) / 255.0
    x = (x - TRAIN_MEAN) / TRAIN_STD
    return torch.from_numpy(x.astype(np.float32))[:, None]


def test_param_count():
    # 104,722 trainable parameters; the state dict holds 105,047 values
    # once BatchNorm running statistics (buffers) are included.
    model = build_model()
    assert sum(p.numel() for p in model.parameters()) == 104722
    assert sum(v.numel() for v in model.state_dict().values()) == 105047


def test_preprocess_shape_and_finiteness(data):
    t = preprocess(data["test_images"][0])
    assert t.shape == (1, 1, 28, 28, 28)
    assert torch.isfinite(t).all()


def test_preprocess_rejects_wrong_shape():
    with pytest.raises(ValueError):
        preprocess(np.zeros((10, 10, 10), dtype=np.uint8))


def test_safetensors_matches_pytorch_checkpoint(model_pt, model_st, data):
    x = _batch(data)
    with torch.no_grad():
        a = model_pt(x)
        b = model_st(x)
    assert torch.allclose(a, b, atol=1e-6)


def test_onnx_parity(model_pt, data):
    ort = pytest.importorskip("onnxruntime")
    sess = ort.InferenceSession(os.path.join(PKG, "vesselcnn.onnx"),
                                providers=["CPUExecutionProvider"])
    x = _batch(data, n=32).numpy()
    out = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    with torch.no_grad():
        ref = model_pt(torch.from_numpy(x)).numpy()
    assert np.abs(out - ref).max() < 1e-4


def test_augmentation_preserves_voxels_and_shape(data):
    # Octahedral transforms relabel voxel positions: the voxel sum and
    # the shape must be preserved exactly, or the augmentation would be
    # corrupting the vessel geometry it claims to preserve.
    x = torch.from_numpy(data["test_images"][:6])[:, None]
    gen = torch.Generator().manual_seed(123)
    out = train_mod.augment(x.float(), gen)
    assert out.shape == x.shape
    assert torch.equal(out.sum(dim=(1, 2, 3, 4)), x.float().sum(dim=(1, 2, 3, 4)))


def test_packaged_samples_match_index(model_st):
    index = json.load(open(os.path.join(ASSETS, "samples", "index.json")))
    assert len(index["samples"]) >= 3
    roles = {s["role"] for s in index["samples"]}
    # The set must include at least one error by design (see index rule).
    assert "false_negative" in roles or "false_positive" in roles
    for s in index["samples"]:
        r = predict(model_st, os.path.join(ASSETS, "samples", s["file"]))
        assert r["label"] == s["predicted"], s["file"]
        assert abs(r["probabilities"]["aneurysm"] - s["probabilities"]["aneurysm"]) < 1e-3


def test_metrics_files_agree():
    trained = json.load(open(os.path.join(ASSETS, "vesselcnn_metrics.json")))["test"]
    reproduced = json.load(open(os.path.join(ASSETS, "evaluation_results.json")))
    for k in ("accuracy", "auc", "balanced_accuracy", "confusion_matrix",
              "per_class_recall", "majority_baseline"):
        assert reproduced[k] == trained[k], k
