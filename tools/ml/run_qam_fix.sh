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

# run_qam_fix.sh — generate QAM-heavy data then fine-tune on GPU.
# Usage: ./run_qam_fix.sh  (runs generation + training end-to-end)
set -euo pipefail
cd "$(dirname "$0")"

LOG=/tmp/train_qam_fix.log
ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(ts)] Waiting for qam_heavy.npz..."
until [[ -f data/qam_heavy.npz ]]; do sleep 10; done
echo "[$(ts)] data/qam_heavy.npz ready — starting GPU training"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/python3 train.py \
    --npz data/synth_v3.npz data/qam_heavy.npz \
    --max-per-class 30000 \
    --model resnet \
    --epochs 40 \
    --batch 512 \
    --lr 2e-4 \
    --cuda \
    --focal 2.0 \
    --boost-qam 4 \
    --resume models/amr_cnn_24class.best.pt \
    --out models/amr_cnn_24class.onnx \
    2>&1 | tee "$LOG"

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

# ========================================================================
# End of file — OpenRFStack
# Subject to Personal Use License
# https://github.com/OpenRFStack
# ========================================================================
