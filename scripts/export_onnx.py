"""Export the packaged VesselCNN weights to ONNX and check parity
against the PyTorch model on real test volumes.

The export is a single self-contained file (weights embedded — the
model is ~105k parameters, so no external .data sidecar is wanted).
Parity protocol: 32 test volumes through both runtimes under the
contract preprocessing; report the max absolute logit difference and
whether the argmax predictions agree on all of them.

Usage:
  python scripts/export_onnx.py [--weights model.safetensors]
      [--data /path/to/vesselmnist3d.npz] [--out vesselcnn.onnx]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402

from vesselcnn.infer import TRAIN_MEAN, TRAIN_STD, load_model  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.path.join(PKG, "model.safetensors"))
    ap.add_argument("--data", required=True, help="path to vesselmnist3d.npz (parity inputs)")
    ap.add_argument("--out", default=os.path.join(PKG, "vesselcnn.onnx"))
    args = ap.parse_args()

    model = load_model(args.weights)
    dummy = torch.zeros(1, 1, 28, 28, 28)
    torch.onnx.export(
        model, dummy, args.out,
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=14,
    )
    # torch.onnx.export may externalize tensors into a "<out>.data"
    # sidecar; embed them so the artifact is one file (the Awaaz-Nano
    # export hit the same behavior on this stack).
    import onnx

    model_proto = onnx.load(args.out)  # loads external data
    onnx.save_model(model_proto, args.out, save_as_external_data=False)
    sidecar = args.out + ".data"
    if os.path.exists(sidecar):
        os.remove(sidecar)
    print(f"wrote {args.out} ({os.path.getsize(args.out)} bytes), weights embedded")

    import onnxruntime as ort
    d = np.load(args.data)
    x = d["test_images"][:32].astype(np.float32) / 255.0
    x = ((x - TRAIN_MEAN) / TRAIN_STD).astype(np.float32)[:, None]
    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {"input": x})[0]
    with torch.no_grad():
        ref = model(torch.from_numpy(x)).numpy()
    max_diff = float(np.abs(onnx_out - ref).max())
    agree = bool((onnx_out.argmax(1) == ref.argmax(1)).all())
    print(f"parity over 32 test volumes: max |dlogit| = {max_diff:.3e}; "
          f"argmax agreement on all 32: {agree}")
    if not agree or max_diff > 1e-4:
        raise SystemExit("PARITY CHECK FAILED")
    print("ONNX PARITY OK")


if __name__ == "__main__":
    raise SystemExit(main())
