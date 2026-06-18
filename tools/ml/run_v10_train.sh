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

# run_v10_train.sh — Fine-tunes from v9, using the same datasets after
# fixing the AM_DSB_SC generator (was Hermitian-symmetric/DC-centered,
# confusable with 4ASK/16ASK/OOK — see regen_am_dsb_sc.py). AM_DSB_SC
# samples in synth_v6, synth_v9, and v6_holdout were regenerated in-place
# with the fixed generator (complex-carrier mixing, no Q=0).

set -euo pipefail
cd /home/brendan/AnalysisApp/tools/ml
LOG=/tmp/train_v10.log
exec >> "$LOG" 2>&1

ts() { date '+%Y-%m-%d %H:%M:%S'; }

RESUME_CKPT="models/amr_cnn_v10_47class.best.pt"

echo ""
echo "════════════════════════════════════════"
echo "v10 pipeline starting: $(ts)"
echo "════════════════════════════════════════"

NOISE_ARG=""
if [[ -f data/noise_bank.npz ]]; then
    echo "[$(ts)] Real noise bank found — using for IqAugment"
    NOISE_ARG="--noise-npz data/noise_bank.npz"
fi

echo ""
echo "[$(ts)] Starting v10 training (fine-tune from $RESUME_CKPT, AM_DSB_SC fix)..."
.venv/bin/python3 -u train.py \
    --npz data/synth_v6_47class_1024.npz data/synth_v9_47class_1024.npz data/real_combined_1024.npz \
    --val-npz data/v6_holdout_1024.npz \
    $NOISE_ARG \
    --resume "$RESUME_CKPT" \
    --model resnet \
    --channels 128 \
    --n-blocks 8 \
    --epochs 60 \
    --batch 128 \
    --lr 1e-4 \
    --focal 2.0 \
    --label-smoothing 0.10 \
    --augment \
    --no-mixup \
    --boost-hard 4.0 \
    --max-per-class 10000 \
    --snr-min -10 \
    --cuda \
    --out models/amr_cnn_v10_47class.onnx

# ── Holdout eval ──────────────────────────────────────────────────────────────
echo ""
echo "[$(ts)] Running holdout eval..."
.venv/bin/python3 eval_holdout.py \
    --model models/amr_cnn_v10_47class.onnx \
    --classes models/amr_cnn_v10_47class.classes.json \
    --holdout data/v6_holdout_v2_1024.npz

echo ""
echo "════════════════════════════════════════"
echo "v10 pipeline DONE: $(ts)"
echo "════════════════════════════════════════"

# ========================================================================
# End of file — OpenRFStack
# Subject to Personal Use License
# https://github.com/OpenRFStack
# ========================================================================
