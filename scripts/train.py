"""Train VesselCNN from scratch on VesselMNIST3D — the exact script that
produced the released weights.

Protocol:
  - VesselMNIST3D official splits (train 1,335 / val 191 / test 382),
    downloaded from Zenodo (record 10519652, md5 b41fd4f7e7e2feedddb201585ecafa1b).
  - Inputs: /255, normalized with the train-split mean/std (the rounded
    contract values in vesselcnn.infer.TRAIN_MEAN/TRAIN_STD, re-derived
    here with chunked float64 accumulation and asserted to agree).
  - Augmentation (train only): a random element of the octahedral group
    per volume per epoch — a random axis permutation composed with
    random axis flips (48 transforms). Vessel segments are cut from
    vessel trees at arbitrary orientations and "aneurysm" is a local
    shape property, so every element of the group is label-preserving.
  - Loss: class-weighted cross-entropy, weights N/(2 * n_class) =
    [0.5633, 4.4500] for the train split's 1,185 vessel / 150 aneurysm
    volumes. Plain CE on an 88.8%-majority split would pay the model
    to ignore the aneurysm class; the weighting is stated here and in
    the metrics record rather than hidden.
  - AdamW (lr 2e-3, weight decay 1e-4, OneCycle cosine), batch 32,
    30 epochs, seed 42 throughout.
  - Checkpoint selected by best validation balanced accuracy (ties
    broken by validation AUC); the test split is evaluated exactly
    once, after selection, and recorded in the metrics JSON.
    examples/verify_metrics.py later reproduces those test numbers
    from the shipped weights.
  - Resumable: after every epoch the full training state is saved to
    <out>/last.pt; rerun with --resume to continue after an
    interruption with no work lost.

Usage:
  python scripts/train.py --data /path/to/vesselmnist3d.npz --out /path/to/workdir [--resume]
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,  # noqa: E402
                             confusion_matrix, precision_score,
                             recall_score, roc_auc_score)

from vesselcnn.infer import LABELS, TRAIN_MEAN, TRAIN_STD  # noqa: E402
from vesselcnn.model import build_model  # noqa: E402

NUM_CLASSES = 2


def compute_split_stats(images_u8):
    """Scalar mean/std with chunked float64 accumulation.

    A single float32 numpy reduction over a whole split can silently
    return wrong statistics at this size (the family's documented
    trap, measured on PathMNIST). Chunked float64 is the safe route.
    """
    s = 0.0
    ss = 0.0
    n = 0
    for i in range(0, len(images_u8), 200):
        xb = images_u8[i:i + 200].astype(np.float64) / 255.0
        s += float(xb.sum())
        ss += float((xb ** 2).sum())
        n += xb.size
    mean = s / n
    std = float(np.sqrt(ss / n - mean ** 2))
    return float(mean), std


def normalize(x_u8_ncdhw):
    """uint8 NCDHW tensor -> normalized float tensor (contract stats)."""
    x = x_u8_ncdhw.float() / 255.0
    return (x - TRAIN_MEAN) / TRAIN_STD


def augment(x_ncdhw, gen):
    """Random octahedral transform per volume (axis permutation + flips).

    The permutation is drawn per volume from the 6 spatial-axis orders;
    each spatial axis is then flipped with probability 1/2. Voxel sums
    are preserved exactly (it is a relabelling of voxel positions), and
    the vessel/aneurysm label is a property of the shape, not of its
    orientation in the cube.
    """
    n = x_ncdhw.shape[0]
    out = torch.empty_like(x_ncdhw)
    perms = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
    for i in range(n):
        v = x_ncdhw[i]
        p = perms[int(torch.randint(0, 6, (1,), generator=gen))]
        v = v.permute(0, p[0] + 1, p[1] + 1, p[2] + 1)
        flips = [d + 1 for d in range(3) if int(torch.randint(0, 2, (1,), generator=gen))]
        if flips:
            v = torch.flip(v, dims=flips)
        out[i] = v
    return out.contiguous()


@torch.no_grad()
def evaluate(model, images_u8, batch=64):
    """Return (probs np array, preds np array) for a split, contract preprocessing."""
    model.eval()
    xt = torch.from_numpy(images_u8)[:, None]
    probs = []
    for i in range(0, len(xt), batch):
        xb = normalize(xt[i:i + batch])
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
    ap.add_argument("--data", required=True, help="path to vesselmnist3d.npz")
    ap.add_argument("--out", required=True, help="work dir for checkpoints + metrics")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true",
                    help="resume from <out>/last.pt if it exists")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    gen = torch.Generator().manual_seed(args.seed)

    d = np.load(args.data)
    x_train_u8 = d["train_images"]
    y_train = d["train_labels"].ravel().astype(np.int64)
    x_val_u8 = d["val_images"]
    y_val = d["val_labels"].ravel().astype(np.int64)
    x_test_u8 = d["test_images"]
    y_test = d["test_labels"].ravel().astype(np.int64)

    exact_mean, exact_std = compute_split_stats(x_train_u8)
    print(f"train stats (chunked float64): mean={exact_mean:.5f} std={exact_std:.5f}",
          flush=True)
    print(f"contract stats used: mean={TRAIN_MEAN} std={TRAIN_STD}", flush=True)
    assert abs(exact_mean - TRAIN_MEAN) < 1e-4
    assert abs(exact_std - TRAIN_STD) < 1e-4

    counts = np.bincount(y_train, minlength=NUM_CLASSES)
    class_weights = len(y_train) / (NUM_CLASSES * counts)
    print(f"train class counts: {counts.tolist()}; "
          f"class weights: {class_weights.round(4).tolist()}", flush=True)

    xt = torch.from_numpy(x_train_u8)[:, None]  # uint8 NCDHW in RAM
    yt = torch.from_numpy(y_train)

    model = build_model()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model parameters: {n_params}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = (len(xt) + args.batch - 1) // args.batch
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=steps_per_epoch * args.epochs,
        pct_start=0.1, anneal_strategy="cos")
    lossf = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32))

    best_path = os.path.join(args.out, "vesselcnn.pt")
    last_path = os.path.join(args.out, "last.pt")
    log = []
    best_key = (-1.0, -1.0)  # (val balanced accuracy, val AUC)
    start_epoch = 0

    if args.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["state_dict"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        start_epoch = ck["epoch"]
        log = ck["log"]
        best_key = tuple(ck["best_key"])
        gen.set_state(ck["gen_state"])
        print(f"resumed at epoch {start_epoch}; best so far "
              f"(bal_acc, auc) = {best_key}", flush=True)

    t_start = time.time()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        perm = torch.randperm(len(xt), generator=gen)
        ep_loss, ep_correct, ep_n = 0.0, 0, 0
        t0 = time.time()
        for i in range(0, len(xt), args.batch):
            idx = perm[i:i + args.batch]
            xb = augment(normalize(xt[idx]), gen)
            yb = yt[idx]
            opt.zero_grad()
            out = model(xb)
            loss = lossf(out, yb)
            loss.backward()
            opt.step()
            sched.step()
            ep_loss += loss.item() * len(idx)
            ep_correct += (out.argmax(1) == yb).sum().item()
            ep_n += len(idx)
        val_probs, val_preds = evaluate(model, x_val_u8)
        val_acc = float(accuracy_score(y_val, val_preds))
        val_bal = float(balanced_accuracy_score(y_val, val_preds))
        val_auc = float(roc_auc_score(y_val, val_probs[:, 1]))
        row = {
            "epoch": epoch + 1,
            "train_loss": round(ep_loss / ep_n, 4),
            "train_acc": round(ep_correct / ep_n, 4),
            "val_acc": round(val_acc, 4),
            "val_balanced_acc": round(val_bal, 4),
            "val_auc": round(val_auc, 4),
            "epoch_seconds": round(time.time() - t0, 1),
        }
        log.append(row)
        print(json.dumps(row), flush=True)
        key = (round(val_bal, 4), round(val_auc, 4))
        if key > best_key:
            best_key = key
            torch.save({
                "state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "val_acc": val_acc,
                "val_balanced_acc": val_bal,
                "val_auc": val_auc,
                "seed": args.seed,
                "train_mean": TRAIN_MEAN,
                "train_std": TRAIN_STD,
                "class_weights": class_weights.tolist(),
            }, best_path)
            print(f"  saved best checkpoint (val bal_acc {val_bal:.4f}, "
                  f"auc {val_auc:.4f})", flush=True)
        torch.save({
            "state_dict": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "epoch": epoch + 1,
            "log": log,
            "best_key": list(best_key),
            "gen_state": gen.get_state(),
        }, last_path)

    total_seconds = round(time.time() - t_start, 1)

    # Final: load the best-val checkpoint and evaluate the test split once.
    ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    test_probs, test_preds = evaluate(model, x_test_u8)
    metrics = {
        "model": "VesselCNN",
        "version": "0.1.0",
        "dataset": "VesselMNIST3D (MedMNIST v2), official splits",
        "seed": args.seed,
        "epochs_run": args.epochs,
        "batch": args.batch,
        "lr": args.lr,
        "class_weights": class_weights.round(4).tolist(),
        "best_epoch": ckpt["epoch"],
        "best_val_acc": round(float(ckpt["val_acc"]), 4),
        "best_val_balanced_acc": round(float(ckpt["val_balanced_acc"]), 4),
        "best_val_auc": round(float(ckpt["val_auc"]), 4),
        "train_seconds_total": total_seconds,
        "train_mean_exact": round(exact_mean, 5),
        "train_std_exact": round(exact_std, 5),
        "test": full_metrics(y_test, test_probs, test_preds),
        "epoch_log": log,
    }
    with open(os.path.join(args.out, "vesselcnn_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics["test"], indent=2), flush=True)
    print(f"DONE total_seconds={total_seconds} best_epoch={ckpt['epoch']} "
          f"best_key={best_key}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
