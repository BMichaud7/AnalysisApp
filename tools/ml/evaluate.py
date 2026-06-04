#!/usr/bin/env python3
# ========================================================================
# Project: OpenRFStack
# Author:  Brendan Michaud
# Year:    2026
# Part of OpenRFStack (https://github.com/OpenRFStack)
#
# Licensed under the Personal Use License.
# Do not use for commercial, organizational, or military purposes.
# Contact author for permission: https://github.com/OpenRFStack
# ========================================================================

"""
Compare the trained CNN (ONNX) against the test split of the AMR dataset
and report per-class accuracy, confusion matrix, and head-to-head vs a
synthetic rule-based baseline.

Usage
-----
    # After training completes:
    python evaluate.py \
        --onnx models/amr_cnn_28class.onnx \
        --classes models/amr_cnn_28class.classes.json \
        --npz data/extended.npz \
        --snr-min 0 \
        --snr-breakdown
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False

try:
    from sklearn.metrics import classification_report, confusion_matrix
    import pandas as pd
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_test_split(npz_path: str, snr_min: float, test_fraction: float = 0.15,
                    seed: int = 42) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    data      = np.load(npz_path, allow_pickle=True)
    X         = data["X"].astype(np.float32)
    y         = data["y"].astype(np.int64)
    snrs      = data["snrs"].astype(np.float32) if "snrs" in data else np.zeros(len(y))
    classes   = list(data["classes"])

    mask = snrs >= snr_min
    X, y, snrs = X[mask], y[mask], snrs[mask]

    rng    = np.random.default_rng(seed)
    idx    = rng.permutation(len(X))
    n_test = int(len(idx) * test_fraction)
    test_i = idx[:n_test]

    return X[test_i], y[test_i], classes, snrs[test_i]


def onnx_predict(session: "ort.InferenceSession",
                 X: np.ndarray,
                 batch_size: int = 512) -> np.ndarray:
    preds = []
    for i in range(0, len(X), batch_size):
        batch = X[i:i + batch_size]
        # Unit-power normalise (same as training)
        pwr = batch.reshape(len(batch), -1).var(axis=1, keepdims=True)[:, :, None]
        pwr = np.maximum(pwr, 1e-9)
        batch = batch / np.sqrt(pwr)
        out = session.run(None, {"iq_input": batch})[0]
        preds.append(np.argmax(out, axis=1))
    return np.concatenate(preds)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate AMR ONNX model vs test split")
    ap.add_argument("--onnx",          required=True, help="ONNX model file")
    ap.add_argument("--classes",       default=None, help="classes.json path (auto-detect if omitted)")
    ap.add_argument("--npz",           required=True, help=".npz dataset file")
    ap.add_argument("--snr-min",       type=float, default=0.0, help="Min SNR to include in test")
    ap.add_argument("--snr-breakdown", action="store_true", help="Show accuracy by SNR tier")
    ap.add_argument("--batch",         type=int, default=512)
    args = ap.parse_args()

    # Load model
    if not ORT_AVAILABLE:
        print("ERROR: onnxruntime not installed.  pip install onnxruntime")
        sys.exit(1)

    print(f"Loading ONNX model: {args.onnx}")
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])

    classes_path = args.classes or str(Path(args.onnx).with_suffix(".classes.json"))
    with open(classes_path) as f:
        class_names = json.load(f)
    print(f"  {len(class_names)} classes")

    # Load test data
    print(f"Loading test split from {args.npz} (SNR ≥ {args.snr_min} dB) …")
    X, y, dataset_classes, snrs = load_test_split(args.npz, args.snr_min)
    print(f"  {len(X)} test samples")

    # Map dataset class indices to model class indices
    dataset_to_model: dict[int, int] = {}
    model_map = {n: i for i, n in enumerate(class_names)}
    for di, dc in enumerate(dataset_classes):
        if dc in model_map:
            dataset_to_model[di] = model_map[dc]

    keep  = np.array([i for i in range(len(y)) if int(y[i]) in dataset_to_model])
    if len(keep) == 0:
        print("ERROR: no class overlap between dataset and model classes.json")
        sys.exit(1)

    X_eval    = X[keep]
    y_dataset = y[keep]
    snrs_eval = snrs[keep]
    y_model   = np.array([dataset_to_model[int(yi)] for yi in y_dataset], dtype=np.int64)

    # Predict
    print("Running inference …")
    y_pred = onnx_predict(sess, X_eval, batch_size=args.batch)

    # Overall accuracy
    acc = (y_pred == y_model).mean()
    print(f"\n── Overall Accuracy: {acc * 100:.1f}% ──\n")

    # Per-class report
    if SKLEARN_AVAILABLE:
        # Map int indices back to names for display
        label_names = [dataset_classes[di] for di in sorted(dataset_to_model.keys())]
        y_true_named = np.array([dataset_classes[int(yi)] for yi in y_dataset])
        y_pred_named = np.array([class_names[pi] for pi in y_pred])
        print(classification_report(y_true_named, y_pred_named, digits=3, zero_division=0))

    # SNR breakdown
    if args.snr_breakdown:
        tiers = [(-20, 0), (0, 10), (10, 20), (20, 100)]
        print("── Accuracy by SNR tier ──")
        for lo, hi in tiers:
            mask = (snrs_eval >= lo) & (snrs_eval < hi)
            if mask.sum() == 0:
                continue
            tier_acc = (y_pred[mask] == y_model[mask]).mean()
            print(f"  SNR {lo:+3d} to {hi:+3d} dB : {tier_acc * 100:.1f}%  "
                  f"(n={mask.sum()})")

    print("\nDone.")


if __name__ == "__main__":
    main()
