#!/usr/bin/env python3
# ========================================================================
# Project: OpenRFStack
# Author:  Brendan Michaud
# Year:    2026
# Part of OpenRFStack (https://github.com/OpenRFStack)
#
# Licensed under the Personal Use License.
# Do not use for commercial, organizational, or military purposes.
# ========================================================================

"""
Ensemble ONNX AMR models by averaging their softmax outputs.

Usage:
    python ensemble_eval.py \
        --models models/amr_cnn_v9_47class.onnx \
                 models/amr_cnn_v10_47class.onnx \
                 models/amr_cnn_v11_47class.onnx \
        --classes models/amr_cnn_v11_47class.classes.json \
        --holdout data/v6_holdout_1024.npz \
        [--tta 8] [--hierarchical]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

BATCH = 512


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def load_session(path: str) -> ort.InferenceSession:
    return ort.InferenceSession(
        path,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )


def ensemble_probs(sessions: list[ort.InferenceSession],
                   X: np.ndarray,
                   tta: int = 1) -> np.ndarray:
    """Average softmax across all models and all TTA passes."""
    rng = np.random.default_rng(0)
    avg = np.zeros((len(X), sessions[0].get_outputs()[0].shape[1] or 1),
                   dtype=np.float32)
    n_passes = 0

    for sess in sessions:
        inp = sess.get_inputs()[0].name
        out_dim = None
        for t in range(max(1, tta)):
            if tta > 1:
                theta = rng.uniform(0, 2 * np.pi, (len(X), 1)).astype(np.float32)
                cos_t, sin_t = np.cos(theta), np.sin(theta)
                X_r = X.copy()
                X_r[:, 0, :] = X[:, 0, :] * cos_t - X[:, 1, :] * sin_t
                X_r[:, 1, :] = X[:, 0, :] * sin_t + X[:, 1, :] * cos_t
            else:
                X_r = X
            for i in range(0, len(X_r), BATCH):
                batch = X_r[i : i + BATCH]
                logits = sess.run(None, {inp: batch})[0]
                if out_dim is None:
                    out_dim = logits.shape[1]
                    if avg.shape[1] == 1:
                        avg = np.zeros((len(X), out_dim), dtype=np.float32)
                avg[i : i + BATCH] += softmax(logits)
            n_passes += 1

    return avg / n_passes


def hierarchical_argmax(probs: np.ndarray, model_classes: list[str]) -> np.ndarray:
    """Family-gated argmax on pre-averaged probabilities."""
    from family_map import CLASS_FAMILY, FAMILY_INDEX, NUM_FAMILIES
    class_family = np.array(
        [FAMILY_INDEX.get(CLASS_FAMILY.get(c, ""), -1) for c in model_classes],
        dtype=np.int32)
    family_mask = np.zeros((NUM_FAMILIES, len(model_classes)), dtype=np.float32)
    for ci, fi in enumerate(class_family):
        if fi >= 0:
            family_mask[fi, ci] = 1.0

    fam_probs = probs @ family_mask.T          # (B, F)
    best_fam  = fam_probs.argmax(axis=1)      # (B,)
    preds = np.zeros(len(probs), dtype=np.int64)
    for j, fi in enumerate(best_fam):
        masked   = probs[j] * family_mask[fi]
        preds[j] = masked.argmax()
    return preds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models",   nargs="+", required=True)
    ap.add_argument("--classes",  required=True)
    ap.add_argument("--holdout",  nargs="+",
                    default=["data/v6_holdout_1024.npz"])
    ap.add_argument("--tta",      type=int, default=1)
    ap.add_argument("--hierarchical", action="store_true")
    ap.add_argument("--out",      default=None)
    args = ap.parse_args()

    model_classes = json.load(open(args.classes))
    print(f"Ensemble: {len(args.models)} model(s)  TTA={args.tta}"
          f"  hierarchical={args.hierarchical}")
    sessions = [load_session(m) for m in args.models]
    for m in args.models:
        print(f"  {m}")

    for npz_path in args.holdout:
        if not Path(npz_path).exists():
            print(f"WARNING: {npz_path} not found", file=sys.stderr)
            continue

        d = np.load(npz_path, allow_pickle=True)
        holdout_classes = [str(c) for c in d["classes"]]
        X, y_raw, snrs = d["X"], d["y"], d["snrs"]

        hmap = {}
        for hi, cls in enumerate(holdout_classes):
            if cls in model_classes:
                hmap[hi] = model_classes.index(cls)

        mask  = np.isin(y_raw, list(hmap.keys()))
        X_f   = X[mask].astype(np.float32)
        y_m   = np.array([hmap[h] for h in y_raw[mask]], dtype=np.int64)
        snrs_f = snrs[mask]

        # Unit-power normalisation — matches train.py, evaluate.py, OnnxClassifier.cpp
        pwr = np.maximum((X_f ** 2).sum(axis=1, keepdims=True).mean(axis=2, keepdims=True), 1e-9)
        X_f = X_f / np.sqrt(pwr)

        probs = ensemble_probs(sessions, X_f, tta=args.tta)

        if args.hierarchical:
            preds = hierarchical_argmax(probs, model_classes)
        else:
            preds = probs.argmax(axis=1)

        correct = preds == y_m
        overall = float(correct.mean())

        per_class = {}
        for mi, cls in enumerate(model_classes):
            idx = y_m == mi
            if idx.sum() == 0:
                continue
            per_class[cls] = float(correct[idx].mean())

        per_snr = {}
        for snr in sorted(np.unique(snrs_f)):
            idx = snrs_f == snr
            per_snr[float(snr)] = float(correct[idx].mean())

        print(f"\n── {npz_path} ──")
        print(f"  overall_acc = {overall:.4f}  ({overall*100:.1f}%)")
        worst = sorted(per_class.items(), key=lambda x: x[1])[:10]
        print("  Worst 10 classes:")
        for cls, acc in worst:
            print(f"    {cls:15s} {acc*100:5.1f}%")
        print("  Per-SNR:")
        for snr, acc in sorted(per_snr.items()):
            print(f"    {snr:+5.0f} dB  {acc*100:.1f}%")

        if args.out:
            Path(args.out).write_text(json.dumps(
                {"overall_acc": overall, "per_class": per_class, "per_snr": per_snr},
                indent=2))
            print(f"  Saved → {args.out}")


if __name__ == "__main__":
    main()
