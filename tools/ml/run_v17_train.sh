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

# run_v17_train.sh — Ensemble distillation: v15+v16 soft labels → smaller model.
#
# Approach:
#   1. Generate soft label NPZ from v15+v16 ensemble predictions on full
#      training set (temperature-scaled, T=2.0).
#   2. Train a half-width RadioFusion model (channels=64, n_blocks=6) using
#      KL-divergence loss against ensemble soft labels + 0.3× hard label CE.
#   3. Goal: ~4× faster inference with <2% accuracy drop vs v15+v16 ensemble.
#
# Prerequisites:
#   - models/amr_cnn_v15_47class.onnx  (from run_v15_train.sh)
#   - models/amr_cnn_v16_47class.onnx  (from run_v16_train.sh)
#   - data/synth_v13_47class_1024.npz  (from regen_flex_qam.py)
#   - data/real_combined_v3_1024.npz
#
# DO NOT LAUNCH until both v15 and v16 are complete.

set -euo pipefail
cd /home/brendan/AnalysisApp/tools/ml
LOG=/tmp/train_v17.log
exec >> "$LOG" 2>&1

ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo ""
echo "════════════════════════════════════════"
echo "v17 pipeline starting: $(ts)"
echo "════════════════════════════════════════"

for f in models/amr_cnn_v15_47class.onnx models/amr_cnn_v16_47class.onnx \
          data/synth_v13_47class_1024.npz data/real_combined_v3_1024.npz; do
    if [[ ! -f "$f" ]]; then
        echo "[$(ts)] ERROR: required file missing: $f — run v15+v16 first"
        exit 1
    fi
done

# Step 1: Generate soft labels from v15+v16 ensemble
echo ""
echo "[$(ts)] Generating ensemble soft labels (T=2.0)..."
.venv/bin/python3 -u distill_labels.py \
    --models models/amr_cnn_v15_47class.onnx \
             models/amr_cnn_v16_47class.onnx \
    --classes models/amr_cnn_v16_47class.classes.json \
    --npz data/synth_v9_nodstar_1024.npz \
          data/synth_v13_47class_1024.npz \
          data/synth_v11_hardboost_1024.npz \
          data/real_combined_v3_1024.npz \
    --temperature 2.0 \
    --max-per-class 20000 \
    --out data/soft_labels_v15v16_1024.npz

# Step 2: Train distilled student model
echo ""
echo "[$(ts)] Training v17 distilled student (channels=64, n_blocks=6)..."
.venv/bin/python3 -u train.py \
    --npz data/synth_v9_nodstar_1024.npz \
          data/synth_v13_47class_1024.npz \
          data/synth_v11_hardboost_1024.npz \
          data/real_combined_v3_1024.npz \
    --soft-labels data/soft_labels_v15v16_1024.npz \
    --soft-weight 0.7 \
    --val-npz data/v7_holdout_1024.npz \
    --model fusion \
    --channels 64 \
    --n-blocks 6 \
    --epochs 80 \
    --batch 256 \
    --lr 5e-4 \
    --focal 1.5 \
    --label-smoothing 0.03 \
    --augment \
    --boost-hard 2.0 \
    --max-per-class 20000 \
    --snr-min -10 \
    --n-fft 512 \
    --family-aux 0.15 \
    --cuda \
    --out models/amr_cnn_v17_distilled.onnx

echo ""
echo "[$(ts)] Holdout eval — v17 distilled, hierarchical + TTA=8..."
.venv/bin/python3 eval_holdout.py \
    --model models/amr_cnn_v17_distilled.onnx \
    --classes models/amr_cnn_v17_distilled.classes.json \
    --holdout data/v7_holdout_1024.npz \
    --tta 8 --hierarchical

echo ""
echo "════════════════════════════════════════"
echo "v17 pipeline DONE: $(ts)"
echo "════════════════════════════════════════"

# ========================================================================
# End of file — OpenRFStack
# Subject to Personal Use License
# https://github.com/OpenRFStack
# ========================================================================
