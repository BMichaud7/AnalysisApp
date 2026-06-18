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

# run_v11_train.sh — Fine-tunes from v10, with:
#   1. RadioFusion model (IQ + STFT spectrogram dual-path) — explicit frequency-
#      domain features for RTTY/DTMF/P25_C4FM/QAM-order disambiguation.
#      IQ path (channels=128, n_blocks=8) loads v10 weights; spectrogram path
#      and fusion head train from scratch.
#   2. Hard-class booster dataset (gen_v11_hardclass_boost.py):
#        Pass 1: 10k/SNR for 11 hardest classes, -10..+35 dB
#        Pass 2: 15k/SNR for same classes, +15..+35 dB only (high-SNR collapse)
#   3. Corrected AM_DSB_SC data from v10 (regen_am_dsb_sc.py applied)
#   4. TTA=8 at holdout eval (8× phase-rotation averaging, ~1-2pp free gain)
#
# Prerequisite: gen_v11_hardclass_boost.py run to produce
# data/synth_v11_hardboost_1024.npz (launched automatically below if missing).

set -euo pipefail
cd /home/brendan/AnalysisApp/tools/ml
LOG=/tmp/train_v11.log
exec >> "$LOG" 2>&1

ts() { date '+%Y-%m-%d %H:%M:%S'; }

RESUME_CKPT="models/amr_cnn_v10_47class.best.pt"

echo ""
echo "════════════════════════════════════════"
echo "v11 pipeline starting: $(ts)"
echo "════════════════════════════════════════"

if [[ ! -f "$RESUME_CKPT" ]]; then
    echo "[$(ts)] $RESUME_CKPT missing — v10 must finish before launching v11"
    exit 1
fi

if [[ ! -f data/synth_v11_hardboost_1024.npz ]]; then
    echo "[$(ts)] Generating hard-class booster dataset (n_per_snr=10000, SNR -10..+35 dB)..."
    .venv/bin/python3 -u gen_v11_hardclass_boost.py
fi

# ── Merge any new real captures into real_combined before training ─────────────
# Picks up data/real_v11_new.npz (and any other data/real_v11_*.npz) automatically.
echo "[$(ts)] Merging real captures into data/real_combined_1024.npz..."
.venv/bin/python3 prep_real_v8.py --out data/real_combined_1024.npz
echo "[$(ts)] real_combined updated."

NOISE_ARG=""
if [[ -f data/noise_bank.npz ]]; then
    echo "[$(ts)] Real noise bank found — using for IqAugment"
    NOISE_ARG="--noise-npz data/noise_bank.npz"
fi

echo ""
echo "[$(ts)] Starting v11 training (RadioFusion IQ+spectrogram, hard-class booster, SNR -10..+35)..."
.venv/bin/python3 -u train.py \
    --npz data/synth_v6_47class_1024.npz \
          data/synth_v10_47class_1024.npz \
          data/synth_v11_hardboost_1024.npz \
          data/real_combined_1024.npz \
    --val-npz data/v6_holdout_v2_1024.npz \
    $NOISE_ARG \
    --resume "$RESUME_CKPT" \
    --model fusion \
    --epochs 60 \
    --batch 128 \
    --lr 5e-5 \
    --focal 2.0 \
    --label-smoothing 0.08 \
    --augment \
    --no-mixup \
    --boost-hard 4.0 \
    --max-per-class 20000 \
    --snr-min -10 \
    --n-fft 256 \
    --family-aux 0.2 \
    --cuda \
    --out models/amr_cnn_v11_47class.onnx

# ── Holdout eval: flat, TTA=8, hierarchical, hierarchical+TTA ─────────────────
echo ""
echo "[$(ts)] Holdout eval — flat (baseline)..."
.venv/bin/python3 eval_holdout.py \
    --model models/amr_cnn_v11_47class.onnx \
    --classes models/amr_cnn_v11_47class.classes.json \
    --holdout data/v6_holdout_v2_1024.npz

echo ""
echo "[$(ts)] Holdout eval — TTA=8..."
.venv/bin/python3 eval_holdout.py \
    --model models/amr_cnn_v11_47class.onnx \
    --classes models/amr_cnn_v11_47class.classes.json \
    --holdout data/v6_holdout_v2_1024.npz \
    --tta 8

echo ""
echo "[$(ts)] Holdout eval — hierarchical (family-gated)..."
.venv/bin/python3 eval_holdout.py \
    --model models/amr_cnn_v11_47class.onnx \
    --classes models/amr_cnn_v11_47class.classes.json \
    --holdout data/v6_holdout_v2_1024.npz \
    --hierarchical

echo ""
echo "[$(ts)] Holdout eval — hierarchical + TTA=8 (best)..."
.venv/bin/python3 eval_holdout.py \
    --model models/amr_cnn_v11_47class.onnx \
    --classes models/amr_cnn_v11_47class.classes.json \
    --holdout data/v6_holdout_v2_1024.npz \
    --tta 8 --hierarchical

echo ""
echo "[$(ts)] Ensemble eval (v9 + v10 + v11, hierarchical + TTA=8)..."
.venv/bin/python3 ensemble_eval.py \
    --models models/amr_cnn_v9_47class.onnx \
             models/amr_cnn_v10_47class.onnx \
             models/amr_cnn_v11_47class.onnx \
    --classes models/amr_cnn_v11_47class.classes.json \
    --holdout data/v6_holdout_v2_1024.npz \
    --tta 8 --hierarchical

echo ""
echo "════════════════════════════════════════"
echo "v11 pipeline DONE: $(ts)"
echo "════════════════════════════════════════"

# ========================================================================
# End of file — OpenRFStack
# Subject to Personal Use License
# https://github.com/OpenRFStack
# ========================================================================
