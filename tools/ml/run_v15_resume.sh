#!/bin/bash
# ========================================================================
# Project: OpenRFStack
# Author:  Brendan Michaud
# Year:    2026
# Part of OpenRFStack (https://github.com/OpenRFStack)
#
# Licensed under the Personal Use License.
# Do not use for commercial, unauthorized, or military purposes.
# ========================================================================

# run_v15_resume.sh — Resume v15 from best checkpoint after OOM-kill at epoch 46.
# Trains remaining 75 epochs (120-45 completed) from models/amr_cnn_v15_47class.best.pt.

set -euo pipefail
cd /home/brendan/AnalysisApp/tools/ml
LOG=/tmp/train_v15.log
exec >> "$LOG" 2>&1

ts() { date '+%Y-%m-%d %H:%M:%S'; }

RESUME_CKPT="models/amr_cnn_v15_47class.best.pt"

echo ""
echo "════════════════════════════════════════"
echo "v15 RESUME pipeline starting: $(ts)"
echo "════════════════════════════════════════"

[[ -f "$RESUME_CKPT" ]] || { echo "MISSING: $RESUME_CKPT"; exit 1; }

for f in data/synth_v9_nodstar_1024.npz data/synth_v12_47class_1024.npz \
          data/synth_v11_hardboost_1024.npz data/real_combined_1024.npz \
          data/v7_holdout_1024.npz; do
    [[ -f "$f" ]] || { echo "MISSING: $f"; exit 1; }
done

# ── Step 1: Resume training for remaining 75 epochs ──────────────────────────
echo ""
echo "[$(ts)] Resuming v15 training from epoch 42 checkpoint (75 epochs remaining)..."
.venv/bin/python3 -u train.py \
    --npz data/synth_v9_nodstar_1024.npz \
          data/synth_v12_47class_1024.npz \
          data/synth_v11_hardboost_1024.npz \
          data/real_combined_1024.npz \
    --val-npz data/v7_holdout_1024.npz \
    --resume "$RESUME_CKPT" \
    --model fusion \
    --epochs 75 \
    --batch 128 \
    --lr 7e-4 \
    --n-fft 512 \
    --focal 2.0 \
    --label-smoothing 0.05 \
    --augment \
    --boost-hard 2.5 \
    --max-per-class 20000 \
    --snr-min -10 \
    --family-aux 0.2 \
    --dann 0.1 \
    --dann-alpha 1.0 \
    --dann-real data/real_combined_1024.npz \
    --cuda \
    --out models/amr_cnn_v15_47class.onnx

# ── Step 2: Holdout eval ──────────────────────────────────────────────────────
echo ""
echo "[$(ts)] Holdout eval — flat..."
.venv/bin/python3 eval_holdout.py \
    --model models/amr_cnn_v15_47class.onnx \
    --classes models/amr_cnn_v15_47class.classes.json \
    --holdout data/v7_holdout_1024.npz

echo ""
echo "[$(ts)] Holdout eval — TTA=8..."
.venv/bin/python3 eval_holdout.py \
    --model models/amr_cnn_v15_47class.onnx \
    --classes models/amr_cnn_v15_47class.classes.json \
    --holdout data/v7_holdout_1024.npz \
    --tta 8

echo ""
echo "[$(ts)] Holdout eval — hierarchical + TTA=8 (best)..."
.venv/bin/python3 eval_holdout.py \
    --model models/amr_cnn_v15_47class.onnx \
    --classes models/amr_cnn_v15_47class.classes.json \
    --holdout data/v7_holdout_1024.npz \
    --tta 8 --hierarchical

echo ""
echo "════════════════════════════════════════"
echo "v15 RESUME pipeline DONE: $(ts)"
echo "════════════════════════════════════════"

# ========================================================================
# End of file — OpenRFStack
# Subject to Personal Use License
# https://github.com/OpenRFStack
# ========================================================================
