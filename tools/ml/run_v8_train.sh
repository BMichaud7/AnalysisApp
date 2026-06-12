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

# run_v8_train.sh — Fine-tunes from the best of v6/v7, adding real captured
# IQ (data/real_combined_1024.npz, produced by prep_real_v8.py — 223,914
# windows across FM_WB/AM_DSB/FSK/FM_NB/GMSK/OFDM/GFSK, FFT-resampled from
# 512 to 1024 samples to match the synthetic window length) alongside the
# existing v6 synthetic data/holdout.
#
# RESUME_CKPT below must be set to whichever of v6/amr_cnn_v6_47class.best.pt
# or amr_cnn_v7_47class.best.pt has the higher eval_holdout.py overall_acc —
# update before launching.

set -euo pipefail
cd /home/brendan/AnalysisApp/tools/ml
LOG=/tmp/train_v8.log
exec >> "$LOG" 2>&1

ts() { date '+%Y-%m-%d %H:%M:%S'; }

RESUME_CKPT="models/amr_cnn_v7_47class.best.pt"  # TODO: confirm vs v6 before launch

echo ""
echo "════════════════════════════════════════"
echo "v8 pipeline starting: $(ts)"
echo "════════════════════════════════════════"

if [[ ! -f data/real_combined_1024.npz ]]; then
    echo "[$(ts)] data/real_combined_1024.npz missing — run prep_real_v8.py first"
    exit 1
fi

NOISE_ARG=""
if [[ -f data/noise_bank.npz ]]; then
    echo "[$(ts)] Real noise bank found — using for IqAugment"
    NOISE_ARG="--noise-npz data/noise_bank.npz"
fi

echo ""
echo "[$(ts)] Starting v8 training (fine-tune from $RESUME_CKPT, + real captured IQ)..."
.venv/bin/python3 -u train.py \
    --npz data/synth_v6_47class_1024.npz data/real_combined_1024.npz \
    --val-npz data/v6_holdout_1024.npz \
    $NOISE_ARG \
    --resume "$RESUME_CKPT" \
    --model resnet \
    --channels 128 \
    --n-blocks 8 \
    --epochs 60 \
    --batch 256 \
    --lr 1e-4 \
    --focal 2.0 \
    --label-smoothing 0.10 \
    --augment \
    --no-mixup \
    --boost-hard 4.0 \
    --max-per-class 15000 \
    --snr-min -10 \
    --cuda \
    --out models/amr_cnn_v8_47class.onnx

# ── Holdout eval ──────────────────────────────────────────────────────────────
echo ""
echo "[$(ts)] Running holdout eval..."
.venv/bin/python3 eval_holdout.py \
    --model models/amr_cnn_v8_47class.onnx \
    --classes models/amr_cnn_v8_47class.classes.json \
    --holdout data/v6_holdout_1024.npz

echo ""
echo "════════════════════════════════════════"
echo "v8 pipeline DONE: $(ts)"
echo "════════════════════════════════════════"

# ========================================================================
# End of file — OpenRFStack
# Subject to Personal Use License
# https://github.com/OpenRFStack
# ========================================================================
