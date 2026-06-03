#!/bin/bash
# run_protocol_retrain.sh — regenerate with 36 classes (28 original + 8 protocols)
# then train fresh on GPU.
set -euo pipefail
cd "$(dirname "$0")"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(ts)] Generating synth_v3_protocols.npz (36 classes, ±0.3% SR freq offset)..."
.venv/bin/python3 generate_v3.py \
    --out data/synth_v3_protocols.npz \
    --n 2500 --len 512 \
    --snr-min -10 --snr-max 30 --snr-step 5 \
    --seed 99 \
    2>&1 | tee /tmp/gen_protocols.log

echo "[$(ts)] Generation done. Starting GPU training (36 classes)..."

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/python3 train.py \
    --npz data/synth_v3_protocols.npz data/qam_heavy.npz \
    --max-per-class 30000 \
    --model resnet \
    --epochs 60 \
    --batch 512 \
    --lr 1e-3 \
    --cuda \
    --out models/amr_cnn_36class.onnx \
    2>&1 | tee /tmp/train_36class.log

echo "[$(ts)] Training complete. Rebuilding DAE chain..."

.venv/bin/python3 - << 'PYEOF'
import onnx
from onnx import compose, version_converter

dae = onnx.load("models/dae_iq.onnx")
clf = onnx.load("models/amr_cnn_36class.onnx")

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
onnx.save(combined, "models/amr_low_snr_denoised_36class.onnx")
m = onnx.load("models/amr_low_snr_denoised_36class.onnx")
print(f"Chain OK: opset={m.opset_import[0].version}")
PYEOF

echo "[$(ts)] Done. Models: amr_cnn_36class.onnx + amr_low_snr_denoised_36class.onnx"
