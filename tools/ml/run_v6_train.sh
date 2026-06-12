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

# run_v6_train.sh — Generates v6 data (fixed generators) and fine-tunes
# from the v5 best checkpoint (models/amr_cnn_v5_47class.best.pt).
#
# Generator fixes vs v5:
#   - RRC filter singularity at T/(4α) (was T/(2α)) — P25_PHASE2 was 100% NaN
#   - P25_PHASE2: cumulative π/4-DQPSK
#   - AM_DSB: explicit carrier offset
#   - QAM32: correct 32-pt constellation
#   - MSK/GMSK: MI=0.5 (dev tied to baud rate)
#   - PSK/QAM: per-order rolloff spectral cue
#   - RTTY/NAVTEX/EAS_SAME: random window offset (both tones visible)
#   - DMR: burst_period=n//2 (TDMA envelope always visible)

set -euo pipefail
cd /home/brendan/AnalysisApp/tools/ml
LOG=/tmp/train_v6.log
exec >> "$LOG" 2>&1

ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo ""
echo "════════════════════════════════════════"
echo "v6 pipeline starting: $(ts)"
echo "════════════════════════════════════════"

# ── Generate v6 holdout ───────────────────────────────────────────────────────
echo ""
echo "[$(ts)] Generating v6 holdout (seed=3333)..."
.venv/bin/python3 generate_v3.py \
    --out data/v6_holdout_1024.npz \
    --n 34 --len 1024 \
    --snr-min -10 --snr-max 30 --snr-step 5 \
    --seed 3333
echo "[$(ts)] Holdout done: $(ls -lh data/v6_holdout_1024.npz | awk '{print $5}')"

# ── Generate v6 training data ─────────────────────────────────────────────────
echo ""
echo "[$(ts)] Generating v6 training data (seed=1111, 5000/SNR/class)..."
.venv/bin/python3 generate_v3.py \
    --out data/synth_v6_47class_1024.npz \
    --n 5000 --len 1024 \
    --snr-min -10 --snr-max 30 --snr-step 5 \
    --seed 1111
echo "[$(ts)] Training data done: $(ls -lh data/synth_v6_47class_1024.npz | awk '{print $5}')"

# ── Train v6 (fine-tune from v5 best checkpoint) ──────────────────────────────
NOISE_ARG=""
if [[ -f data/noise_bank.npz ]]; then
    echo "[$(ts)] Real noise bank found — using for IqAugment"
    NOISE_ARG="--noise-npz data/noise_bank.npz"
fi

echo ""
echo "[$(ts)] Starting v6 training (fine-tune from v6 epoch-2 checkpoint, lr=1e-4, no-mixup)..."
# lr=3e-4 destabilized at epoch 9 when mixup turned on (warmup_epochs=8) and never
# recovered through epoch 16 (val_acc stuck ~0.52-0.56 vs epoch 8's 0.607, min_class
# stayed 0.000). Restarted at lr=1e-4 + boost-hard=5.0 (HARD_CLASSES now also includes
# RTTY) — but the SAME pattern recurred at epoch 7 (loss 1.11->1.68, val_acc dropped
# and stayed ~0.49-0.55 with min_class~0 through epoch 26). Root cause: mixup_batch()
# linearly superimposes IQ samples from two different modulation classes with a
# blended soft label — for raw RF/IQ this produces ambiguous training examples that
# actively confuse the classifier rather than regularize it (especially the
# already-weak HARD_CLASSES). Added --no-mixup flag and restarted from the
# pre-mixup-collapse epoch-2 checkpoint (score=0.425, min_class=0.042, saved 00:18).
.venv/bin/python3 -u train.py \
    --npz data/synth_v6_47class_1024.npz \
    --val-npz data/v6_holdout_1024.npz \
    $NOISE_ARG \
    --resume models/amr_cnn_v6_47class.best.pt \
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
    --boost-hard 5.0 \
    --max-per-class 15000 \
    --snr-min -10 \
    --cuda \
    --out models/amr_cnn_v6_47class.onnx

# ── Holdout eval ──────────────────────────────────────────────────────────────
echo ""
echo "[$(ts)] Running holdout eval..."
.venv/bin/python3 eval_holdout.py \
    --model models/amr_cnn_v6_47class.onnx \
    --classes models/amr_cnn_v6_47class.classes.json \
    --holdout data/v6_holdout_1024.npz

echo ""
echo "════════════════════════════════════════"
echo "v6 pipeline DONE: $(ts)"
echo "════════════════════════════════════════"

# ========================================================================
# End of file — OpenRFStack
# Subject to Personal Use License
# https://github.com/OpenRFStack
# ========================================================================
