#!/usr/bin/env python3
"""Per-class x per-SNR accuracy breakdown for accuracy.html chart data."""
import json
import sys
import numpy as np
import onnxruntime as ort

model_path = sys.argv[1]
classes_path = sys.argv[2]
npz_path = sys.argv[3]

with open(classes_path) as f:
    model_classes = json.load(f)

d = np.load(npz_path, allow_pickle=True)
holdout_classes = [str(c) for c in d["classes"]]
X, y_raw, snrs = d["X"], d["y"], d["snrs"]

hmap = {hi: model_classes.index(cls) for hi, cls in enumerate(holdout_classes) if cls in model_classes}
mask = np.isin(y_raw, list(hmap.keys()))
X_f, y_h, snrs_f = X[mask], y_raw[mask], snrs[mask]
y_m = np.array([hmap[h] for h in y_h], dtype=np.int64)

sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
inp = sess.get_inputs()[0].name
preds = []
B = 1024
for i in range(0, len(X_f), B):
    out = sess.run(None, {inp: X_f[i:i+B].astype(np.float32)})[0]
    preds.append(out.argmax(axis=1))
preds = np.concatenate(preds)
correct = preds == y_m

snr_levels = sorted(np.unique(snrs_f))
result = {}
for mi, cls in enumerate(model_classes):
    cls_mask = y_m == mi
    if cls_mask.sum() == 0:
        continue
    row = {}
    for snr in snr_levels:
        idx = cls_mask & (snrs_f == snr)
        if idx.sum() == 0:
            continue
        row[f"{int(snr):+d}"] = round(float(correct[idx].mean()) * 100, 1)
    result[cls] = row

print(json.dumps(result))
