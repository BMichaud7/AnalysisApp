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

# run_full_retrain.sh — regenerate data with fixed freq offset, then retrain.
set -euo pipefail
cd "$(dirname "$0")"

LOG=/tmp/retrain_full.log
ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(ts)] Generating synth_v3_fixed.npz (fixed ±0.3% SR freq offset)..."
.venv/bin/python3 generate_v3.py \
    --out data/synth_v3_fixed.npz \
    --n 2500 --len 512 \
    --snr-min -10 --snr-max 30 --snr-step 5 \
    --seed 42 \
    2>&1 | tee /tmp/gen_v3_fixed.log

echo "[$(ts)] Generation done. Starting GPU training..."

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/python3 train.py \
    --npz data/synth_v3_fixed.npz data/qam_heavy.npz \
    --max-per-class 40000 \
    --model resnet \
    --epochs 60 \
    --batch 512 \
    --lr 1e-3 \
    --cuda \
    --out models/amr_cnn_24class.onnx \
    2>&1 | tee /tmp/retrain_full.log

echo "[$(ts)] Training complete. Rebuilding DAE chain..."

.venv/bin/python3 - << 'PYEOF'
import onnx
from onnx import compose, version_converter

dae = onnx.load("models/dae_iq.onnx")
clf = onnx.load("models/amr_cnn_24class.onnx")

clf_opset = clf.opset_import[0].version
if dae.opset_import[0].version != clf_opset:
    dae = version_converter.convert_version(dae, clf_opset)

clf_prefixed = compose.add_prefix(clf, prefix="clf_", rename_edges=True)
combined = compose.merge_models(dae, clf_prefixed,
                                io_map=[("iq_clean", "clf_iq_input")])

for inp in combined.graph.input:
    if inp.name == "iq_noisy": inp.name = "iq_input"
for node in combined.graph.node:
    node.input[:] = ["iq_input" if x == "iq_noisy" else x for x in node.input]
for out in combined.graph.output:
    if out.name == "clf_logits": out.name = "logits"
for node in combined.graph.node:
    node.output[:] = ["logits" if o == "clf_logits" else o for o in node.output]

onnx.checker.check_model(combined)
onnx.save(combined, "models/amr_low_snr_denoised.onnx")
m = onnx.load("models/amr_low_snr_denoised.onnx")
print(f"Chain OK: opset={m.opset_import[0].version} "
      f"inputs={[i.name for i in m.graph.input]} "
      f"outputs={[o.name for o in m.graph.output]}")
PYEOF

echo "[$(ts)] Done. Models: models/amr_cnn_24class.onnx + amr_low_snr_denoised.onnx"
