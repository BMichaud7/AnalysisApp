#!/bin/bash
# ========================================================================
# Project: OpenRFStack
# Author:  Brendan Michaud
# Year:    2026
# Part of OpenRFStack (https://github.com/OpenRFStack)
#
# Licensed under the Personal Use License.
# Do not use for commercial, organizational, or military purposes.
# ========================================================================

# run_v5_train.sh — Wait for v5 data generation, then train with:
#   - Fixed PSK31/RTTY generator (sps from _gen_one, not hardcoded)
#   - No CFO augmentation (IqAugment fix)
#   - Holdout as actual val set (--val-npz)
#   - Real noise if noise_bank.npz exists (--noise-npz)
#   - Composite checkpoint (0.7*val_acc + 0.3*min_class)

set -euo pipefail
cd /home/brendan/AnalysisApp/tools/ml
LOG=/tmp/train_v5.log
exec >> "$LOG" 2>&1

ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo ""
echo "════════════════════════════════════════"
echo "v5 training pipeline started: $(ts)"
echo "════════════════════════════════════════"

# Wait for data generation
echo "[$(ts)] Waiting for v5 data generation..."
GEN_PID=1397414
while kill -0 "$GEN_PID" 2>/dev/null; do
    sleep 30
done
echo "[$(ts)] Generation complete."

# Verify files exist
if [[ ! -f data/synth_v5_47class_1024.npz ]] || [[ ! -f data/v5_holdout_1024.npz ]]; then
    echo "ERROR: Expected data files not found. Check /tmp/generate_v5.log"
    exit 1
fi

echo "[$(ts)] Training data: $(ls -lh data/synth_v5_47class_1024.npz | awk '{print $5}')"
echo "[$(ts)] Holdout:       $(ls -lh data/v5_holdout_1024.npz | awk '{print $5}')"

# Real noise bank — use if already captured
NOISE_ARG=""
if [[ -f data/noise_bank.npz ]]; then
    echo "[$(ts)] Real noise bank found — using for IqAugment"
    NOISE_ARG="--noise-npz data/noise_bank.npz"
else
    echo "[$(ts)] No noise_bank.npz — IqAugment will use synthetic AWGN"
fi

echo "[$(ts)] Starting training..."
.venv/bin/python3 -u train.py \
    --npz data/synth_v5_47class_1024.npz \
    --val-npz data/v5_holdout_1024.npz \
    $NOISE_ARG \
    --model resnet \
    --channels 128 \
    --n-blocks 8 \
    --epochs 80 \
    --batch 256 \
    --lr 1e-3 \
    --focal 2.0 \
    --label-smoothing 0.10 \
    --augment \
    --boost-hard 3.0 \
    --max-per-class 15000 \
    --snr-min -10 \
    --cuda \
    --out models/amr_cnn_v5_47class.onnx

echo ""
echo "[$(ts)] Training complete. Running holdout eval..."
.venv/bin/python3 eval_holdout.py \
    --model models/amr_cnn_v5_47class.onnx \
    --classes models/amr_cnn_v5_47class.classes.json \
    --holdout data/v5_holdout_1024.npz

echo ""
echo "════════════════════════════════════════"
echo "v5 pipeline DONE: $(ts)"
echo "════════════════════════════════════════"

# ========================================================================
# End of file — OpenRFStack
# Subject to Personal Use License
# https://github.com/OpenRFStack
# ========================================================================
