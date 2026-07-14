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

# run_v15_train.sh — DSTAR fix: fresh training with n_fft=512.
#
# Root cause of DSTAR 0% (persisted through v12–v14):
#   n_fft=256 at SR=200kHz → 781Hz/bin. DSTAR deviation=1600Hz and P25
#   C4FM outer tone=1800Hz both land in bin 2 — indistinguishable in STFT.
#   n_fft=512 → 390Hz/bin: DSTAR→bin4, P25 outer→bin5. Separated.
#
#   Fine-tuning from v13 also failed because v13 was trained on DSTAR at
#   both 4800Hz (synth_v9, wrong) and 1600Hz (synth_v12, correct) —
#   contradictory labels baked into v13 weights. v15 trains from scratch.
#
# v15 goals:
#   1. Train from scratch (no --resume) — clean slate for DSTAR weights.
#   2. --n-fft 512 (the key fix; default is already 512, explicit here).
#   3. 120 epochs with default LR=1e-3 + warmup (not 2e-5 fine-tune LR).
#   4. Mixup enabled (good for from-scratch; v14 disabled it).
#   5. Same data as v14: synth_v9_nodstar + synth_v12 + v11_hardboost + real.

set -euo pipefail
cd /home/brendan/AnalysisApp/tools/ml
LOG=/tmp/train_v15.log
exec >> "$LOG" 2>&1

ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo ""
echo "════════════════════════════════════════"
echo "v15 pipeline starting: $(ts)"
echo "════════════════════════════════════════"

for f in data/synth_v9_nodstar_1024.npz data/synth_v12_47class_1024.npz \
          data/synth_v11_hardboost_1024.npz data/real_combined_1024.npz \
          data/v7_holdout_1024.npz; do
    [[ -f "$f" ]] || { echo "MISSING: $f"; exit 1; }
done

# ── Step 1: Training ──────────────────────────────────────────────────────────
echo ""
echo "[$(ts)] Starting v15 training (from scratch, n_fft=512)..."
.venv/bin/python3 -u train.py \
    --npz data/synth_v9_nodstar_1024.npz \
          data/synth_v12_47class_1024.npz \
          data/synth_v11_hardboost_1024.npz \
          data/real_combined_1024.npz \
    --val-npz data/v7_holdout_1024.npz \
    --model fusion \
    --epochs 120 \
    --batch 128 \
    --lr 1e-3 \
    --n-fft 512 \
    --focal 2.0 \
    --label-smoothing 0.05 \
    --augment \
    --boost-hard 2.5 \
    --max-per-class 20000 \
    --snr-min -10 \
    --family-aux 0.2 \
    --dann 0.1 \
    --dann-alpha 1.0 \
    --dann-real data/real_combined_1024.npz \
    --cuda \
    --out models/amr_cnn_v15_47class.onnx

# ── Step 2: Holdout eval ──────────────────────────────────────────────────────
echo ""
echo "[$(ts)] Holdout eval — flat..."
.venv/bin/python3 eval_holdout.py \
    --model models/amr_cnn_v15_47class.onnx \
    --classes models/amr_cnn_v15_47class.classes.json \
    --holdout data/v7_holdout_1024.npz

echo ""
echo "[$(ts)] Holdout eval — TTA=8..."
.venv/bin/python3 eval_holdout.py \
    --model models/amr_cnn_v15_47class.onnx \
    --classes models/amr_cnn_v15_47class.classes.json \
    --holdout data/v7_holdout_1024.npz \
    --tta 8

echo ""
echo "[$(ts)] Holdout eval — hierarchical + TTA=8 (best)..."
.venv/bin/python3 eval_holdout.py \
    --model models/amr_cnn_v15_47class.onnx \
    --classes models/amr_cnn_v15_47class.classes.json \
    --holdout data/v7_holdout_1024.npz \
    --tta 8 --hierarchical

echo ""
echo "[$(ts)] Ensemble eval (v13 + v14 + v15, hierarchical + TTA=8)..."
.venv/bin/python3 ensemble_eval.py \
    --models models/amr_cnn_v13_47class.onnx \
             models/amr_cnn_v14_47class.onnx \
             models/amr_cnn_v15_47class.onnx \
    --classes models/amr_cnn_v15_47class.classes.json \
    --holdout data/v7_holdout_1024.npz \
    --tta 8 --hierarchical

echo ""
echo "════════════════════════════════════════"
echo "v15 pipeline DONE: $(ts)"
echo "════════════════════════════════════════"

# ========================================================================
# End of file — OpenRFStack
# Subject to Personal Use License
# https://github.com/OpenRFStack
# ========================================================================
