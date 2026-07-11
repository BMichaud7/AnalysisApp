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

# run_router_train.sh — trains the coarse family router: a 9-way classifier
# (QAM/PSK/ASK/FSK/DIGI_VOICE/PROTOCOL/AM/FM/SPECIAL, see family_map.py) that
# picks which family specialist (run_family_train.sh) a signal should be
# routed to. Resumes from the v9 47-class checkpoint for a pretrained
# backbone; only the final 9-way head is reinitialised.

set -euo pipefail
cd /home/brendan/AnalysisApp/tools/ml
LOG=/tmp/train_router.log
exec >> "$LOG" 2>&1

ts() { date '+%Y-%m-%d %H:%M:%S'; }

RESUME_CKPT="models/amr_cnn_v9_47class.best.pt"

echo ""
echo "════════════════════════════════════════"
echo "Router training starting: $(ts)"
echo "════════════════════════════════════════"

if [[ ! -f "$RESUME_CKPT" ]]; then
    echo "[$(ts)] $RESUME_CKPT missing"
    exit 1
fi

.venv/bin/python3 -u train.py \
    --npz data/synth_v6_47class_1024.npz data/synth_v9_47class_1024.npz data/real_combined_1024.npz \
    --val-npz data/v6_holdout_v2_1024.npz \
    --router \
    --resume "$RESUME_CKPT" \
    --model resnet \
    --channels 128 \
    --n-blocks 8 \
    --epochs 40 \
    --batch 256 \
    --lr 3e-4 \
    --focal 2.0 \
    --label-smoothing 0.10 \
    --augment \
    --no-mixup \
    --max-per-class 15000 \
    --snr-min -10 \
    --cuda \
    --out models/amr_router_v1.onnx

echo ""
echo "[$(ts)] Holdout eval (router family accuracy)..."
.venv/bin/python3 eval_holdout.py \
    --model models/amr_router_v1.onnx \
    --classes models/amr_router_v1.classes.json \
    --holdout data/v6_holdout_v2_1024.npz \
    --router

echo ""
echo "════════════════════════════════════════"
echo "Router training DONE: $(ts)"
echo "════════════════════════════════════════"

# ========================================================================
# End of file — OpenRFStack
# Subject to Personal Use License
# https://github.com/OpenRFStack
# ========================================================================
