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

# run_v13_train.sh — Continues from v12 (full RadioFusion checkpoint).
#
# v12 trained from near-random init (all paths except iq_path were reset
# due to architecture mismatch with v10 resume) for only 50 epochs → 83.3%.
# Worst classes: QAM64 33%, 16PSK 33%, QAM32 41%.
#
# v13 goals:
#   1. Resume from v12 (all RadioFusion paths initialized → no re-init).
#   2. 100 epochs to allow cumulant/spectrogram paths to converge.
#   3. Stronger hard-class boost (2.5×) on QAM/PSK variants and NAVTEX/RTTY.
#   4. Lower LR (2e-5) for stable fine-tuning from the v12 baseline.

set -euo pipefail
cd /home/brendan/AnalysisApp/tools/ml
LOG=/tmp/train_v13.log
exec >> "$LOG" 2>&1

ts() { date '+%Y-%m-%d %H:%M:%S'; }

RESUME_CKPT="models/amr_cnn_v12_47class.best.pt"

echo ""
echo "════════════════════════════════════════"
echo "v13 pipeline starting: $(ts)"
echo "════════════════════════════════════════"

if [[ ! -f "$RESUME_CKPT" ]]; then
    echo "[$(ts)] $RESUME_CKPT missing — run v12 first"
    exit 1
fi

# ── Training ─────────────────────────────────────────────────────────────────
echo ""
echo "[$(ts)] Starting v13 training (RadioFusion fine-tune from v12)..."
.venv/bin/python3 -u train.py \
    --npz data/synth_v6_47class_1024.npz \
          data/synth_v10_47class_1024.npz \
          data/synth_v11_hardboost_1024.npz \
          data/real_combined_1024.npz \
    --val-npz data/v6_holdout_v2_1024.npz \
    --resume "$RESUME_CKPT" \
    --model fusion \
    --epochs 100 \
    --batch 128 \
    --lr 2e-5 \
    --focal 2.0 \
    --label-smoothing 0.05 \
    --augment \
    --no-mixup \
    --boost-hard 2.5 \
    --max-per-class 20000 \
    --snr-min -10 \
    --n-fft 256 \
    --family-aux 0.2 \
    --dann 0.1 \
    --dann-alpha 1.0 \
    --dann-real data/real_combined_1024.npz \
    --cuda \
    --out models/amr_cnn_v13_47class.onnx

# ── Holdout eval ─────────────────────────────────────────────────────────────
echo ""
echo "[$(ts)] Holdout eval — flat..."
.venv/bin/python3 eval_holdout.py \
    --model models/amr_cnn_v13_47class.onnx \
    --classes models/amr_cnn_v13_47class.classes.json \
    --holdout data/v6_holdout_v2_1024.npz

echo ""
echo "[$(ts)] Holdout eval — TTA=8..."
.venv/bin/python3 eval_holdout.py \
    --model models/amr_cnn_v13_47class.onnx \
    --classes models/amr_cnn_v13_47class.classes.json \
    --holdout data/v6_holdout_v2_1024.npz \
    --tta 8

echo ""
echo "[$(ts)] Holdout eval — hierarchical + TTA=8 (best)..."
.venv/bin/python3 eval_holdout.py \
    --model models/amr_cnn_v13_47class.onnx \
    --classes models/amr_cnn_v13_47class.classes.json \
    --holdout data/v6_holdout_v2_1024.npz \
    --tta 8 --hierarchical

echo ""
echo "[$(ts)] Ensemble eval (v10 + v12 + v13, hierarchical + TTA=8)..."
.venv/bin/python3 ensemble_eval.py \
    --models models/amr_cnn_v10_47class.onnx \
             models/amr_cnn_v12_47class.onnx \
             models/amr_cnn_v13_47class.onnx \
    --classes models/amr_cnn_v13_47class.classes.json \
    --holdout data/v6_holdout_v2_1024.npz \
    --tta 8 --hierarchical

echo ""
echo "════════════════════════════════════════"
echo "v13 pipeline DONE: $(ts)"
echo "════════════════════════════════════════"

# ========================================================================
# End of file — OpenRFStack
# Subject to Personal Use License
# https://github.com/OpenRFStack
# ========================================================================
