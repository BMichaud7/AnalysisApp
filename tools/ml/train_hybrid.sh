#!/usr/bin/env bash
# ========================================================================
# Project: OpenRFStack
# Author:  Brendan Michaud
# Year:    2026
# Part of OpenRFStack (https://github.com/OpenRFStack)
#
# Licensed under the Personal Use License.
# Do not use for commercial, organizational, or military purposes.
# ========================================================================

# Wait for capture + synthetic generation to finish, then train.
set -euo pipefail
cd "$(dirname "$0")"

echo "[train_hybrid] Waiting for data generation..."

# Wait for both background processes
CAPTURE_PID=$(cat /tmp/capture_pid.txt 2>/dev/null || echo "")
GEN_PID=$(cat /tmp/gen_pid.txt 2>/dev/null || echo "")
[[ -n "$CAPTURE_PID" ]] && wait "$CAPTURE_PID" 2>/dev/null || true
[[ -n "$GEN_PID"     ]] && wait "$GEN_PID"     2>/dev/null || true

echo "[train_hybrid] Data ready. Checking files..."
ls -lh data/real.npz data/synthetic_large.npz 2>/dev/null || {
    echo "ERROR: Missing data files"; exit 1
}

python3 -c "
import numpy as np
r = np.load('data/real.npz')
s = np.load('data/synthetic_large.npz')
print(f'Real:      {r[\"X\"].shape}  classes={list(r[\"classes\"])}')
print(f'Synthetic: {s[\"X\"].shape}  classes={len(s[\"classes\"])}')
"

echo ""
echo "[train_hybrid] Starting training..."
echo "  Synthetic: data/synthetic_large.npz"
echo "  Real:      data/real.npz"
echo "  Model:     RadioCNN  (28+ classes)"
echo "  Epochs:    60"
echo ""

python3 train.py \
    --npz data/synthetic_large.npz data/real.npz \
    --model cnn \
    --epochs 60 \
    --batch 512 \
    --lr 1e-3 \
    --max-per-class 12000 \
    --snr-min -4 \
    --out models/amr_cnn_28class.onnx

echo "[train_hybrid] Done. Model: models/amr_cnn_28class.onnx"

# Copy classes JSON so scan.sh can find it
ls -lh models/amr_cnn_28class.onnx models/amr_cnn_28class.classes.json 2>/dev/null
