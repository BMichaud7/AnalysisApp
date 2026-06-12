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

# run_v7_train.sh — Fine-tunes from the v6 best checkpoint
# (models/amr_cnn_v6_47class.best.pt, holdout acc=0.694) to address two
# issues found in the v6 holdout breakdown:
#
#   1. Many classes (P25_C4FM, NXDN, RTTY, AM_DSB, DTMF, GFSK, FSK, TONE,
#      8PSK/16PSK/32PSK, QAM16/32/64) get WORSE at high SNR (+20..+30dB).
#      Root cause: IqAugment always adds extra AWGN at a random 5-30dB SNR
#      on top of the nominal sample, so the model rarely sees genuinely
#      clean high-SNR examples. Fixed: snr_hi_db 30 -> 40.
#   2. Those same classes are systematically confused with structurally
#      similar neighbors (P25_C4FM<->FM_NB, NXDN<->DSTAR/P25_PHASE2,
#      RTTY<->NAVTEX, AM_DSB<->TONE, QAM32/64<->QAM16/256,
#      16PSK/32PSK<->QPSK/8PSK). Added all of these to HARD_CLASSES for
#      oversampling.
#
# Attempt 1 (boost-hard=6.0) ran 19 epochs and plateaued at score~0.39-0.41
# (val_acc ~0.55-0.59), well below v6's 0.694 starting point, with no
# upward trend. Likely cause: boost-hard=6.0 shifted the per-batch class/SNR
# distribution enough that BatchNorm running stats diverged from the holdout
# distribution within 1 epoch. Attempt 2 reduces boost-hard to 4.0 (closer
# to v6's 5.0) while keeping the snr_hi=40 fix. Log/checkpoint from attempt 1
# archived as models/archive/amr_cnn_v7_attempt1_boost6.best.pt /
# /tmp/train_v7_attempt1.log.
#
# Reuses the existing v6 synthetic data/holdout (same generator, no need
# to regenerate).

set -euo pipefail
cd /home/brendan/AnalysisApp/tools/ml
LOG=/tmp/train_v7.log
exec >> "$LOG" 2>&1

ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo ""
echo "════════════════════════════════════════"
echo "v7 pipeline starting: $(ts)"
echo "════════════════════════════════════════"

NOISE_ARG=""
if [[ -f data/noise_bank.npz ]]; then
    echo "[$(ts)] Real noise bank found — using for IqAugment"
    NOISE_ARG="--noise-npz data/noise_bank.npz"
fi

echo ""
echo "[$(ts)] Starting v7 training attempt 2 (fine-tune from v6 best, lr=1e-4, no-mixup, snr_hi=40, boost-hard=4.0)..."
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
    --boost-hard 4.0 \
    --max-per-class 15000 \
    --snr-min -10 \
    --cuda \
    --out models/amr_cnn_v7_47class.onnx

# ── Holdout eval ──────────────────────────────────────────────────────────────
echo ""
echo "[$(ts)] Running holdout eval..."
.venv/bin/python3 eval_holdout.py \
    --model models/amr_cnn_v7_47class.onnx \
    --classes models/amr_cnn_v7_47class.classes.json \
    --holdout data/v6_holdout_1024.npz

echo ""
echo "════════════════════════════════════════"
echo "v7 pipeline DONE: $(ts)"
echo "════════════════════════════════════════"

# ========================================================================
# End of file — OpenRFStack
# Subject to Personal Use License
# https://github.com/OpenRFStack
# ========================================================================
