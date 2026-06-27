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


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def run_inference(sess: ort.InferenceSession, X: np.ndarray,
                  tta: int = 1) -> np.ndarray:
    """Run inference with optional test-time augmentation.

    tta > 1: average softmax over `tta` passes, each with a random per-sample
    phase rotation (safe for all modulation classes — phase is arbitrary at
    the receiver). Typically tta=8 gives ~1-2pp gain with no training changes.
    """
    input_name = sess.get_inputs()[0].name
    if tta <= 1:
        preds = []
        for i in range(0, len(X), BATCH):
            batch = X[i : i + BATCH]
            logits = sess.run(None, {input_name: batch})[0]
            preds.append(np.argmax(logits, axis=1))
        return np.concatenate(preds)

    # TTA: accumulate averaged softmax probabilities across passes
    avg_probs = np.zeros((len(X), sess.get_outputs()[0].shape[1] or 1),
                         dtype=np.float32)
    rng = np.random.default_rng(0)
    for _ in range(tta):
        # Random phase rotation: multiply complex IQ by exp(j*theta)
        theta = rng.uniform(0, 2 * np.pi, size=(len(X), 1)).astype(np.float32)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        X_rot = X.copy()
        X_rot[:, 0, :] = X[:, 0, :] * cos_t - X[:, 1, :] * sin_t
        X_rot[:, 1, :] = X[:, 0, :] * sin_t + X[:, 1, :] * cos_t
        for i in range(0, len(X_rot), BATCH):
            batch = X_rot[i : i + BATCH]
            logits = sess.run(None, {input_name: batch})[0]
            if avg_probs.shape[1] == 1:
                avg_probs = np.zeros((len(X), logits.shape[1]), dtype=np.float32)
            avg_probs[i : i + BATCH] += _softmax(logits)
    return np.argmax(avg_probs, axis=1)


def _hierarchical_preds(sess: ort.InferenceSession,
                        X: np.ndarray,
                        model_classes: list[str]) -> np.ndarray:
    """Family-gated argmax: pick best class only within the predicted family.

    For each sample, softmax over all classes → sum within each family to get
    family probabilities → pick winning family → argmax within that family.
    This reduces the 47-class search space to ~5 per family, cutting inter-
    family confusion (e.g. FM_WB mistaken for GFSK instead of FM_NB).
    """
    from family_map import CLASS_FAMILY, FAMILY_INDEX, NUM_FAMILIES
    input_name = sess.get_inputs()[0].name

    # Precompute: class_idx → family_idx
    class_family = np.array(
        [FAMILY_INDEX.get(CLASS_FAMILY.get(c, ""), -1) for c in model_classes],
        dtype=np.int32)
    # family_mask[f, c] = 1 if class c belongs to family f
    family_mask = np.zeros((NUM_FAMILIES, len(model_classes)), dtype=np.float32)
    for ci, fi in enumerate(class_family):
        if fi >= 0:
            family_mask[fi, ci] = 1.0

    all_preds = []
    for i in range(0, len(X), BATCH):
        batch = X[i : i + BATCH]
        logits = sess.run(None, {input_name: batch})[0]
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)          # (B, C)
        fam_probs = probs @ family_mask.T                   # (B, F)
        best_fam  = fam_probs.argmax(axis=1)               # (B,)
        preds = np.zeros(len(batch), dtype=np.int64)
        for j, fi in enumerate(best_fam):
            mask_j = family_mask[fi]                        # (C,) 0/1
            masked  = probs[j] * mask_j
            preds[j] = masked.argmax()
        all_preds.append(preds)
    return np.concatenate(all_preds)


def evaluate(sess: ort.InferenceSession,
             model_classes: list[str],
             npz_path: str,
             tta: int = 1,
             hierarchical: bool = False,
             router: bool = False) -> dict:
    d = np.load(npz_path, allow_pickle=True)
    holdout_classes = [str(c) for c in d["classes"]]
    if router:
        from family_map import CLASS_FAMILY
        holdout_classes = [CLASS_FAMILY.get(c, c) for c in holdout_classes]
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

    if hierarchical:
        preds = _hierarchical_preds(sess, X_f, model_classes)
    else:
        preds = run_inference(sess, X_f, tta=tta)
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
    ap.add_argument("--tta", type=int, default=1, metavar="N",
                    help="Test-time augmentation passes (default 1 = off, 8 recommended)")
    ap.add_argument("--hierarchical", action="store_true",
                    help="Family-gated argmax: pick best class within predicted "
                         "modulation family (QAM/PSK/FSK/AM/FM/etc) rather than "
                         "flat 47-class argmax. Reduces inter-family confusion.")
    ap.add_argument("--router", action="store_true",
                    help="Evaluate a family router model (trained with "
                         "train.py --router): holdout per-class labels are "
                         "mapped to family names before scoring.")
    args = ap.parse_args()
    if args.hierarchical and args.router:
        ap.error("--hierarchical and --router are mutually exclusive")

    model_classes = json.load(open(args.classes))
    print(f"Model: {args.model}  ({len(model_classes)} classes)")

    sess = load_session(args.model)

    results = []
    for npz in args.holdout:
        if not Path(npz).exists():
            print(f"WARNING: {npz} not found, skipping", file=sys.stderr)
            continue
        r = evaluate(sess, model_classes, npz, tta=args.tta,
                     hierarchical=args.hierarchical, router=args.router)
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
