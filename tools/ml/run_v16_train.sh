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

# run_v16_train.sh — From-scratch training with fixed FLEX/QAM/PSK data.
#
# Fixes vs v15:
#   - synth_v13 instead of synth_v12:
#       FLEX: sync always-4-dibits (was 2/50%) — improves 8-symbol window cue
#       QAM64/QAM256/QAM32/16PSK/32PSK: non-overlapping rolloff ranges
#   - real_combined_v3 (416k samples, 2.8x more real data than v15)
#
# Same architecture choices as v15 (n_fft=512, from-scratch).

set -euo pipefail
cd /home/brendan/AnalysisApp/tools/ml
LOG=/tmp/train_v16.log
exec >> "$LOG" 2>&1

ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo ""
echo "════════════════════════════════════════"
echo "v16 pipeline starting: $(ts)"
echo "════════════════════════════════════════"

for f in data/synth_v9_nodstar_1024.npz data/synth_v13_47class_1024.npz \
          data/synth_v11_hardboost_1024.npz data/real_combined_v3_1024.npz \
          data/v7_holdout_1024.npz; do
    if [[ ! -f "$f" ]]; then
        echo "[$(ts)] ERROR: required file missing: $f"
        exit 1
    fi
done

echo "[$(ts)] Starting v16 training (from scratch, synth_v13 + real_v3)..."
.venv/bin/python3 -u train.py \
    --npz data/synth_v9_nodstar_1024.npz \
          data/synth_v13_47class_1024.npz \
          data/synth_v11_hardboost_1024.npz \
          data/real_combined_v3_1024.npz \
    --val-npz data/v7_holdout_1024.npz \
    --model fusion \
    --epochs 120 \
    --batch 128 \
    --lr 1e-3 \
    --focal 2.0 \
    --label-smoothing 0.05 \
    --augment \
    --boost-hard 2.5 \
    --max-per-class 20000 \
    --snr-min -10 \
    --n-fft 512 \
    --family-aux 0.2 \
    --dann 0.1 \
    --dann-alpha 1.0 \
    --dann-real data/real_combined_v3_1024.npz \
    --cuda \
    --out models/amr_cnn_v16_47class.onnx

echo ""
echo "[$(ts)] Holdout eval — flat..."
.venv/bin/python3 eval_holdout.py \
    --model models/amr_cnn_v16_47class.onnx \
    --classes models/amr_cnn_v16_47class.classes.json \
    --holdout data/v7_holdout_1024.npz

echo ""
echo "[$(ts)] Holdout eval — hierarchical + TTA=8..."
.venv/bin/python3 eval_holdout.py \
    --model models/amr_cnn_v16_47class.onnx \
    --classes models/amr_cnn_v16_47class.classes.json \
    --holdout data/v7_holdout_1024.npz \
    --tta 8 --hierarchical

echo ""
echo "[$(ts)] Ensemble eval (v15 + v16, hierarchical + TTA=8)..."
.venv/bin/python3 ensemble_eval.py \
    --models models/amr_cnn_v15_47class.onnx \
             models/amr_cnn_v16_47class.onnx \
    --classes models/amr_cnn_v16_47class.classes.json \
    --holdout data/v7_holdout_1024.npz \
    --tta 8 --hierarchical

echo ""
echo "════════════════════════════════════════"
echo "v16 pipeline DONE: $(ts)"
echo "════════════════════════════════════════"

# ========================================================================
# End of file — OpenRFStack
# Subject to Personal Use License
# https://github.com/OpenRFStack
# ========================================================================
