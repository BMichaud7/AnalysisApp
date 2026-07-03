#!/usr/bin/env python3
"""
bug_hunt_v8.py — Repeatedly sample real-captured and holdout signals with
known ground-truth labels (real captures labeled via frequency/bandwidth
band-plan lookup in capture_real.py's TARGETS table; holdout labeled by
synthetic generation) and check whether the v8 AMR model classifies them
correctly. Runs for a fixed wall-clock duration, logging only mismatches
plus periodic confusion-matrix snapshots, to find systematic bugs.

Usage:
    .venv/bin/python3 bug_hunt_v8.py --hours 12
"""
from __future__ import annotations

import argparse
import collections
import json
import time

import numpy as np
import onnxruntime as ort


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/amr_cnn_v8_47class.onnx")
    ap.add_argument("--classes", default="models/amr_cnn_v8_47class.classes.json")
    ap.add_argument("--data", nargs="+",
                    default=["data/real_combined_1024.npz", "data/v6_holdout_1024.npz"])
    ap.add_argument("--batch-size", type=int, default=25)
    ap.add_argument("--hours", type=float, default=12.0)
    ap.add_argument("--report", default="/tmp/bug_hunt_v8_report.json")
    ap.add_argument("--snapshot-every", type=int, default=50,
                    help="batches between report snapshots")
    args = ap.parse_args()

    model_classes = json.load(open(args.classes))
    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    datasets = {}
    for path in args.data:
        d = np.load(path, allow_pickle=True)
        datasets[path] = {
            "X": d["X"], "y": d["y"], "snrs": d["snrs"],
            "classes": [str(c) for c in d["classes"]],
        }
        print(f"Loaded {path}: {d['X'].shape}, {len(datasets[path]['classes'])} classes")

    rng = np.random.default_rng()

    # per-dataset stats
    stats = {path: {
        "n_total": 0, "n_wrong": 0,
        "confusion": collections.Counter(),
        "snr_of_miss": collections.defaultdict(list),
    } for path in args.data}

    start = time.time()
    deadline = start + args.hours * 3600
    b = 0
    last_snapshot = 0

    print(f"Bug hunt starting — will run for {args.hours} hours "
          f"({len(args.data)} dataset(s), batch size {args.batch_size})", flush=True)

    while time.time() < deadline:
        path = args.data[b % len(args.data)]
        ds = datasets[path]
        X, y, snrs, data_classes = ds["X"], ds["y"], ds["snrs"], ds["classes"]
        st = stats[path]

        idx = rng.choice(len(X), size=args.batch_size, replace=False)
        batch_X = X[idx].astype(np.float32)
        pwr = np.maximum((batch_X ** 2).sum(axis=1, keepdims=True).mean(axis=2, keepdims=True), 1e-9)
        batch_X = batch_X / np.sqrt(pwr)
        logits = sess.run(None, {input_name: batch_X})[0]
        probs = softmax(logits)
        preds = np.argmax(logits, axis=1)

        for i, gi in enumerate(idx):
            true_name = data_classes[y[gi]]
            if true_name not in model_classes:
                continue  # holdout class not in v8's label set
            pred_name = model_classes[preds[i]]
            conf = float(probs[i, preds[i]])
            snr = float(snrs[gi])
            st["n_total"] += 1
            if true_name != pred_name:
                st["n_wrong"] += 1
                st["confusion"][(true_name, pred_name)] += 1
                st["snr_of_miss"][(true_name, pred_name)].append(snr)
                elapsed = time.time() - start
                print(f"[{elapsed/3600:5.2f}h] BUG  {path:35s} "
                      f"true={true_name:<10} pred={pred_name:<10} "
                      f"conf={conf*100:5.1f}% snr={snr:+5.1f}dB idx={gi}", flush=True)

        b += 1
        if b - last_snapshot >= args.snapshot_every:
            last_snapshot = b
            report = {"elapsed_hours": (time.time() - start) / 3600, "batches": b, "datasets": {}}
            for path, st in stats.items():
                top = st["confusion"].most_common(15)
                report["datasets"][path] = {
                    "n_total": st["n_total"],
                    "n_wrong": st["n_wrong"],
                    "acc": (st["n_total"] - st["n_wrong"]) / max(st["n_total"], 1),
                    "top_confusions": [
                        {"true": t, "pred": p, "count": c,
                         "avg_snr": float(np.mean(st["snr_of_miss"][(t, p)]))}
                        for (t, p), c in top
                    ],
                }
            with open(args.report, "w") as f:
                json.dump(report, f, indent=2)
            elapsed = time.time() - start
            print(f"\n[{elapsed/3600:5.2f}h] --- snapshot written to {args.report} "
                  f"(batch {b}) ---", flush=True)
            for path, st in stats.items():
                acc = (st["n_total"] - st["n_wrong"]) / max(st["n_total"], 1)
                print(f"  {path}: n={st['n_total']} acc={acc*100:.1f}%", flush=True)

    # final report
    report = {"elapsed_hours": (time.time() - start) / 3600, "batches": b, "datasets": {}}
    for path, st in stats.items():
        top = st["confusion"].most_common(30)
        report["datasets"][path] = {
            "n_total": st["n_total"],
            "n_wrong": st["n_wrong"],
            "acc": (st["n_total"] - st["n_wrong"]) / max(st["n_total"], 1),
            "top_confusions": [
                {"true": t, "pred": p, "count": c,
                 "avg_snr": float(np.mean(st["snr_of_miss"][(t, p)]))}
                for (t, p), c in top
            ],
        }
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nDone. Final report written to {args.report}")


if __name__ == "__main__":
    main()
