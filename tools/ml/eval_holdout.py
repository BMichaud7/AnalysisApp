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
Evaluate an ONNX AMR classifier against holdout NPZ files.

Usage:
    python eval_holdout.py [--model MODEL.onnx] [--classes classes.json]
                           [--holdout FILE.npz ...] [--out report.json]

Holdout NPZ files must have keys: X (N,2,L), y (N,), snrs (N,), classes (C,)
Classes in the holdout that are unknown to the model are skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort


BATCH = 512


def load_session(model_path: str) -> ort.InferenceSession:
    return ort.InferenceSession(
        model_path,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )


def run_inference(sess: ort.InferenceSession, X: np.ndarray) -> np.ndarray:
    input_name = sess.get_inputs()[0].name
    preds = []
    for i in range(0, len(X), BATCH):
        batch = X[i : i + BATCH]
        logits = sess.run(None, {input_name: batch})[0]
        preds.append(np.argmax(logits, axis=1))
    return np.concatenate(preds)


def evaluate(sess: ort.InferenceSession,
             model_classes: list[str],
             npz_path: str) -> dict:
    d = np.load(npz_path, allow_pickle=True)
    holdout_classes = [str(c) for c in d["classes"]]
    X, y_raw, snrs = d["X"], d["y"], d["snrs"]

    # Build index map: holdout label → model label (skip unknown)
    hmap = {}
    for hi, cls in enumerate(holdout_classes):
        if cls in model_classes:
            hmap[hi] = model_classes.index(cls)

    mask = np.isin(y_raw, list(hmap.keys()))
    X_f   = X[mask]
    y_h   = y_raw[mask]
    snrs_f = snrs[mask]
    y_m   = np.array([hmap[h] for h in y_h], dtype=np.int64)

    skipped = int((~mask).sum())

    preds = run_inference(sess, X_f)
    correct = (preds == y_m)

    # ── Overall ──────────────────────────────────────────────────────────────
    overall_acc = float(correct.mean())

    # ── Per-class ─────────────────────────────────────────────────────────────
    per_class = {}
    for mi, cls in enumerate(model_classes):
        idx = y_m == mi
        if idx.sum() == 0:
            continue
        per_class[cls] = float(correct[idx].mean())

    # ── Per-SNR ───────────────────────────────────────────────────────────────
    per_snr = {}
    for snr in sorted(np.unique(snrs_f)):
        idx = snrs_f == snr
        per_snr[float(snr)] = float(correct[idx].mean())

    return {
        "npz": npz_path,
        "total_samples": int(mask.sum()),
        "skipped_unknown": skipped,
        "overall_acc": overall_acc,
        "per_class": per_class,
        "per_snr": per_snr,
    }


def print_report(result: dict) -> None:
    print(f"\n{'='*62}")
    print(f"  {result['npz']}")
    print(f"  samples={result['total_samples']}  skipped={result['skipped_unknown']}")
    print(f"  Overall accuracy: {result['overall_acc']*100:.1f}%")
    print(f"{'='*62}")

    print("\nPer-class accuracy (worst → best):")
    for cls, acc in sorted(result["per_class"].items(), key=lambda x: x[1]):
        bar = "█" * int(acc * 30)
        print(f"  {cls:<14}  {acc*100:5.1f}%  {bar}")

    print("\nPer-SNR accuracy:")
    for snr, acc in sorted(result["per_snr"].items()):
        bar = "█" * int(acc * 30)
        print(f"  {snr:+5.0f} dB  {acc*100:5.1f}%  {bar}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate ONNX AMR model on holdout sets")
    ap.add_argument("--model",   default="models/amr_cnn_24class.onnx")
    ap.add_argument("--classes", default="models/amr_cnn_24class.classes.json")
    ap.add_argument("--holdout", nargs="+",
                    default=["data/v3_holdout.npz", "data/v3_holdout_impaired.npz"])
    ap.add_argument("--out", default=None, metavar="JSON",
                    help="Save JSON report to this path")
    args = ap.parse_args()

    model_classes = json.load(open(args.classes))
    print(f"Model: {args.model}  ({len(model_classes)} classes)")

    sess = load_session(args.model)

    results = []
    for npz in args.holdout:
        if not Path(npz).exists():
            print(f"WARNING: {npz} not found, skipping", file=sys.stderr)
            continue
        r = evaluate(sess, model_classes, npz)
        print_report(r)
        results.append(r)

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\nReport saved → {args.out}")


if __name__ == "__main__":
    main()

# ========================================================================
# End of file — OpenRFStack
# Subject to Personal Use License
# https://github.com/OpenRFStack
# ========================================================================
