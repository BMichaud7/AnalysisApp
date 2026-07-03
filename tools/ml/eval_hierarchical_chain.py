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
Evaluate the hard-hierarchical pipeline: router model picks a family, then
that family's specialist model picks the fine-grained class. Unlike
eval_holdout.py --hierarchical (soft routing inside one 47-class model),
this chains two genuinely separate models end to end.

Usage:
    python eval_hierarchical_chain.py \
        --router models/amr_router_v1.onnx \
        --router-classes models/amr_router_v1.classes.json \
        --specialists-dir models \
        --holdout data/v6_holdout_v2_1024.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

from family_map import CLASS_FAMILY

BATCH = 512


def load_session(model_path: str) -> ort.InferenceSession:
    return ort.InferenceSession(
        model_path,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )


def run_argmax(sess: ort.InferenceSession, X: np.ndarray) -> np.ndarray:
    input_name = sess.get_inputs()[0].name
    preds = []
    for i in range(0, len(X), BATCH):
        batch = X[i:i + BATCH].astype(np.float32)
        pwr = np.maximum((batch ** 2).sum(axis=1, keepdims=True).mean(axis=2, keepdims=True), 1e-9)
        batch = batch / np.sqrt(pwr)
        logits = sess.run(None, {input_name: batch})[0]
        preds.append(np.argmax(logits, axis=1))
    return np.concatenate(preds)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate router -> specialist hierarchical pipeline")
    ap.add_argument("--router", required=True, metavar="ONNX")
    ap.add_argument("--router-classes", required=True, metavar="JSON")
    ap.add_argument("--specialists-dir", default="models", metavar="DIR",
                    help="Directory containing amr_family_<FAMILY>_v1.onnx / .classes.json")
    ap.add_argument("--specialist-tag", default="v1", metavar="TAG",
                    help="Suffix tag used in specialist filenames (default v1)")
    ap.add_argument("--holdout", nargs="+", required=True, metavar="FILE")
    ap.add_argument("--out", default=None, metavar="JSON")
    args = ap.parse_args()

    router_classes = json.load(open(args.router_classes))
    router_sess = load_session(args.router)
    print(f"Router: {args.router}  ({len(router_classes)} families)")

    spec_dir = Path(args.specialists_dir)
    specialist_sess: dict[str, ort.InferenceSession] = {}
    specialist_classes: dict[str, list[str]] = {}
    for fam in router_classes:
        onnx_path = spec_dir / f"amr_family_{fam}_{args.specialist_tag}.onnx"
        cls_path = spec_dir / f"amr_family_{fam}_{args.specialist_tag}.classes.json"
        if not onnx_path.exists():
            print(f"  [warn] missing specialist for family '{fam}': {onnx_path}", file=sys.stderr)
            continue
        specialist_sess[fam] = load_session(str(onnx_path))
        specialist_classes[fam] = json.load(open(cls_path))
        print(f"  Specialist[{fam}]: {len(specialist_classes[fam])} classes")

    for npz_path in args.holdout:
        if not Path(npz_path).exists():
            print(f"WARNING: {npz_path} not found, skipping", file=sys.stderr)
            continue

        d = np.load(npz_path, allow_pickle=True)
        holdout_classes = [str(c) for c in d["classes"]]
        X, y_raw, snrs = d["X"], d["y"], d["snrs"]

        # Drop samples whose class has no family mapping or no specialist.
        keep_mask = np.array([
            CLASS_FAMILY.get(holdout_classes[int(l)]) in specialist_sess for l in y_raw
        ])
        X_f, y_f, snrs_f = X[keep_mask], y_raw[keep_mask], snrs[keep_mask]
        true_class = np.array([holdout_classes[int(l)] for l in y_f])
        true_family = np.array([CLASS_FAMILY[c] for c in true_class])

        # ── Stage 1: router predicts family for every sample ────────────────
        fam_idx_pred = run_argmax(router_sess, X_f)
        pred_family = np.array([router_classes[i] for i in fam_idx_pred])
        router_acc = float((pred_family == true_family).mean())

        # ── Stage 2: route each sample to its predicted family's specialist ─
        final_pred = np.empty(len(X_f), dtype=object)
        for fam, sess in specialist_sess.items():
            idx = np.where(pred_family == fam)[0]
            if len(idx) == 0:
                continue
            cls_idx_pred = run_argmax(sess, X_f[idx])
            final_pred[idx] = [specialist_classes[fam][i] for i in cls_idx_pred]

        correct = (final_pred == true_class)
        overall_acc = float(correct.mean())

        per_class: dict[str, float] = {}
        for cls in sorted(set(true_class)):
            idx = true_class == cls
            per_class[cls] = float(correct[idx].mean())

        per_snr: dict[float, float] = {}
        for snr in sorted(np.unique(snrs_f)):
            idx = snrs_f == snr
            per_snr[float(snr)] = float(correct[idx].mean())

        result = {
            "npz": npz_path,
            "total_samples": int(keep_mask.sum()),
            "skipped_no_specialist": int((~keep_mask).sum()),
            "router_acc": router_acc,
            "overall_acc": overall_acc,
            "per_class": per_class,
            "per_snr": per_snr,
        }

        print(f"\n{'='*62}")
        print(f"  {npz_path}")
        print(f"  samples={result['total_samples']}  skipped={result['skipped_no_specialist']}")
        print(f"  Router family accuracy: {router_acc*100:.1f}%")
        print(f"  Chained overall accuracy: {overall_acc*100:.1f}%")
        print(f"{'='*62}")
        print("\nPer-class accuracy (worst -> best):")
        for cls, acc in sorted(per_class.items(), key=lambda x: x[1]):
            bar = "█" * int(acc * 30)
            print(f"  {cls:<14}  {acc*100:5.1f}%  {bar}")
        print("\nPer-SNR accuracy:")
        for snr, acc in sorted(per_snr.items()):
            bar = "█" * int(acc * 30)
            print(f"  {snr:+5.0f} dB  {acc*100:5.1f}%  {bar}")

        if args.out:
            Path(args.out).write_text(json.dumps(result, indent=2))
            print(f"\nReport saved -> {args.out}")


if __name__ == "__main__":
    main()

# ========================================================================
# End of file — OpenRFStack
# Subject to Personal Use License
# https://github.com/OpenRFStack
# ========================================================================
