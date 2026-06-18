#!/bin/bash
# post_capture_merge.sh
# Merges real_missing7.npz into real_combined_1024.npz, then resumes sdr_acquisition.
# Run after capture_real.py --filter-classes finishes.

set -euo pipefail
cd /home/brendan/AnalysisApp/tools/ml

LOG=/tmp/post_capture_merge.log
exec >> "$LOG" 2>&1

ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo ""
echo "════════════════════════════════════════"
echo "post_capture_merge: $(ts)"
echo "════════════════════════════════════════"

if [[ ! -f data/real_missing7.npz ]]; then
    echo "ERROR: data/real_missing7.npz not found"
    exit 1
fi

echo "[$(ts)] Merging real_missing7.npz into real_combined_1024.npz ..."
.venv/bin/python3 - <<'EOF'
import numpy as np
from collections import Counter
from pathlib import Path

SRC_COMBINED = Path("data/real_combined_1024.npz")
SRC_NEW      = Path("data/real_missing7.npz")
OUT          = Path("data/real_combined_1024.npz")  # overwrite in-place

a = np.load(SRC_COMBINED)
b = np.load(SRC_NEW)

# Merge class lists — combined is the master
classes = a["classes"].tolist()
label_of = {c: i for i, c in enumerate(classes)}

# Remap b labels to combined label space, adding new classes if needed
b_classes = b["classes"].tolist()
for c in b_classes:
    if c not in label_of:
        label_of[c] = len(classes)
        classes.append(c)

y_b = b["y"]
b_local = b_classes
y_b_global = np.array([label_of[b_local[int(v)]] for v in y_b], dtype=y_b.dtype)

X_out    = np.concatenate([a["X"],    b["X"]],        axis=0)
y_out    = np.concatenate([a["y"],    y_b_global],    axis=0)
snrs_out = np.concatenate([a["snrs"], b["snrs"]],     axis=0)

perm = np.random.default_rng(42).permutation(len(X_out))
X_out = X_out[perm]; y_out = y_out[perm]; snrs_out = snrs_out[perm]

print(f"Merged: {len(a['X']):,} + {len(b['X']):,} = {len(X_out):,} total samples, {len(classes)} classes")
counts = Counter(y_out.tolist())
for i, c in enumerate(classes):
    if counts.get(i, 0) > 0:
        print(f"  {c}: {counts[i]:,}")

np.savez_compressed(OUT, X=X_out, y=y_out, snrs=snrs_out, classes=np.array(classes))
print(f"Saved → {OUT}")
EOF

echo "[$(ts)] Merge complete."

# Resume sdr_acquisition
SDR_ACQ_PID=2836919
if kill -0 "$SDR_ACQ_PID" 2>/dev/null; then
    echo "[$(ts)] Resuming sdr_acquisition (PID $SDR_ACQ_PID) ..."
    sudo kill -CONT "$SDR_ACQ_PID" && echo "[$(ts)] sdr_acquisition resumed." || \
        echo "[$(ts)] WARNING: could not resume sdr_acquisition — run: sudo kill -CONT $SDR_ACQ_PID"
else
    echo "[$(ts)] sdr_acquisition PID $SDR_ACQ_PID not found — already gone or restarted."
fi

echo "[$(ts)] Done."
