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

# run_v14_train.sh — DSTAR fix retrain from v13.
#
# Root cause of DSTAR 0% accuracy: generate_v3._dstar() was corrected from
# dev=4800Hz to dev=1600Hz but the training NPZ was never regenerated.
# regen_dstar.py patches synth_v10→synth_v12 and v6_holdout_v2→v7_holdout
# with the correct deviation.
#
# v14 goals:
#   1. Resume from v13 (best baseline so far).
#   2. Use synth_v12 (DSTAR at 1600Hz) instead of synth_v10 (DSTAR at 4800Hz).
#   3. Use v7_holdout (DSTAR at 1600Hz) for honest eval.
#   4. 60 epochs: correction should converge quickly from v13 weights.
#   5. Same LR (2e-5) and boost (2.5×) as v13.

set -euo pipefail
cd /home/brendan/AnalysisApp/tools/ml
LOG=/tmp/train_v14.log
exec >> "$LOG" 2>&1

ts() { date '+%Y-%m-%d %H:%M:%S'; }

RESUME_CKPT="models/amr_cnn_v13_47class.best.pt"

echo ""
echo "════════════════════════════════════════"
echo "v14 pipeline starting: $(ts)"
echo "════════════════════════════════════════"

# ── Step 0: Generate patched training/holdout data ───────────────────────────
if [[ ! -f data/synth_v12_47class_1024.npz || ! -f data/v7_holdout_1024.npz ]]; then
    echo "[$(ts)] Running regen_dstar.py ..."
    .venv/bin/python3 -u regen_dstar.py
else
    echo "[$(ts)] synth_v12 and v7_holdout already exist — skipping regen"
fi

if [[ ! -f "$RESUME_CKPT" ]]; then
    echo "[$(ts)] $RESUME_CKPT missing — run v13 first"
    exit 1
fi

# ── Step 1: Training ──────────────────────────────────────────────────────────
echo ""
echo "[$(ts)] Starting v14 training (DSTAR fix fine-tune from v13)..."
.venv/bin/python3 -u train.py \
    --npz data/synth_v6_47class_1024.npz \
          data/synth_v12_47class_1024.npz \
          data/synth_v11_hardboost_1024.npz \
          data/real_combined_1024.npz \
    --val-npz data/v7_holdout_1024.npz \
    --resume "$RESUME_CKPT" \
    --model fusion \
    --epochs 60 \
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
    --out models/amr_cnn_v14_47class.onnx

# ── Step 2: Holdout eval ──────────────────────────────────────────────────────
echo ""
echo "[$(ts)] Holdout eval — flat..."
.venv/bin/python3 eval_holdout.py \
    --model models/amr_cnn_v14_47class.onnx \
    --classes models/amr_cnn_v14_47class.classes.json \
    --holdout data/v7_holdout_1024.npz

echo ""
echo "[$(ts)] Holdout eval — TTA=8..."
.venv/bin/python3 eval_holdout.py \
    --model models/amr_cnn_v14_47class.onnx \
    --classes models/amr_cnn_v14_47class.classes.json \
    --holdout data/v7_holdout_1024.npz \
    --tta 8

echo ""
echo "[$(ts)] Holdout eval — hierarchical + TTA=8 (best)..."
.venv/bin/python3 eval_holdout.py \
    --model models/amr_cnn_v14_47class.onnx \
    --classes models/amr_cnn_v14_47class.classes.json \
    --holdout data/v7_holdout_1024.npz \
    --tta 8 --hierarchical

echo ""
echo "[$(ts)] Ensemble eval (v12 + v13 + v14, hierarchical + TTA=8)..."
.venv/bin/python3 ensemble_eval.py \
    --models models/amr_cnn_v12_47class.onnx \
             models/amr_cnn_v13_47class.onnx \
             models/amr_cnn_v14_47class.onnx \
    --classes models/amr_cnn_v14_47class.classes.json \
    --holdout data/v7_holdout_1024.npz \
    --tta 8 --hierarchical

echo ""
echo "════════════════════════════════════════"
echo "v14 pipeline DONE: $(ts)"
echo "════════════════════════════════════════"

# ========================================================================
# End of file — OpenRFStack
# Subject to Personal Use License
# https://github.com/OpenRFStack
# ========================================================================
