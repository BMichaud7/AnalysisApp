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

# run_v12_train.sh — Fine-tunes from v11, adding:
#   1. Instantaneous features path (IA/IP/IF) — 3rd RadioFusion stream that
#      directly encodes what a human demodulator computes: AM vs FM diverge
#      in IF, PSK orders diverge in IP trajectory.
#   2. Higher-order cumulant features (C20, C42, C40, C4P) — analytically
#      optimal discriminants for QAM-order and PSK constellation-size confusion.
#   3. DANN domain adaptation — gradient reversal on the shared hidden rep
#      forces the model to learn domain-invariant features, closing the ~5pp
#      synthetic-to-real gap.
#
# Prerequisite: v11 must have finished (models/amr_cnn_v11_47class.best.pt).
# v11 runs automatically after v10 via run_v11_train.sh.

set -euo pipefail
cd /home/brendan/AnalysisApp/tools/ml
LOG=/tmp/train_v12.log
exec >> "$LOG" 2>&1

ts() { date '+%Y-%m-%d %H:%M:%S'; }

RESUME_CKPT="models/amr_cnn_v11_47class.best.pt"

echo ""
echo "════════════════════════════════════════"
echo "v12 pipeline starting: $(ts)"
echo "════════════════════════════════════════"

if [[ ! -f "$RESUME_CKPT" ]]; then
    echo "[$(ts)] $RESUME_CKPT missing — v11 must finish before launching v12"
    exit 1
fi

# ── Training ─────────────────────────────────────────────────────────────────
echo ""
echo "[$(ts)] Starting v12 training (RadioFusion + IF/IA/IP path + cumulants + DANN)..."
.venv/bin/python3 -u train.py \
    --npz data/synth_v6_47class_1024.npz \
          data/synth_v10_47class_1024.npz \
          data/synth_v11_hardboost_1024.npz \
          data/real_combined_1024.npz \
    --val-npz data/v6_holdout_v2_1024.npz \
    --resume "$RESUME_CKPT" \
    --model fusion \
    --epochs 50 \
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
    --dann 0.1 \
    --dann-alpha 1.0 \
    --dann-real data/real_combined_1024.npz \
    --cuda \
    --out models/amr_cnn_v12_47class.onnx

# ── Holdout eval: flat, TTA=8, hierarchical, hierarchical+TTA ─────────────────
echo ""
echo "[$(ts)] Holdout eval — flat..."
.venv/bin/python3 eval_holdout.py \
    --model models/amr_cnn_v12_47class.onnx \
    --classes models/amr_cnn_v12_47class.classes.json \
    --holdout data/v6_holdout_v2_1024.npz

echo ""
echo "[$(ts)] Holdout eval — TTA=8..."
.venv/bin/python3 eval_holdout.py \
    --model models/amr_cnn_v12_47class.onnx \
    --classes models/amr_cnn_v12_47class.classes.json \
    --holdout data/v6_holdout_v2_1024.npz \
    --tta 8

echo ""
echo "[$(ts)] Holdout eval — hierarchical..."
.venv/bin/python3 eval_holdout.py \
    --model models/amr_cnn_v12_47class.onnx \
    --classes models/amr_cnn_v12_47class.classes.json \
    --holdout data/v6_holdout_v2_1024.npz \
    --hierarchical

echo ""
echo "[$(ts)] Holdout eval — hierarchical + TTA=8 (best)..."
.venv/bin/python3 eval_holdout.py \
    --model models/amr_cnn_v12_47class.onnx \
    --classes models/amr_cnn_v12_47class.classes.json \
    --holdout data/v6_holdout_v2_1024.npz \
    --tta 8 --hierarchical

echo ""
echo "[$(ts)] Ensemble eval (v9 + v10 + v11 + v12, hierarchical + TTA=8)..."
.venv/bin/python3 ensemble_eval.py \
    --models models/amr_cnn_v9_47class.onnx \
             models/amr_cnn_v10_47class.onnx \
             models/amr_cnn_v11_47class.onnx \
             models/amr_cnn_v12_47class.onnx \
    --classes models/amr_cnn_v12_47class.classes.json \
    --holdout data/v6_holdout_v2_1024.npz \
    --tta 8 --hierarchical

echo ""
echo "════════════════════════════════════════"
echo "v12 pipeline DONE: $(ts)"
echo "════════════════════════════════════════"

# ========================================================================
# End of file — OpenRFStack
# Subject to Personal Use License
# https://github.com/OpenRFStack
# ========================================================================
