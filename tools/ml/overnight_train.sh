#!/bin/bash
# overnight_train.sh — Chained overnight training pipeline.
# Waits for any current train.py to finish, then runs:
#   1. Large synthetic dataset generation (v3 large, snr step=2)
#   2. ResNet 200 epochs (warmup + cosine, class-weighted, mixup)
#   3. DAE retrain on all real + synth data
#   4. Chain DAE + ResNet into amr_low_snr_denoised.onnx
#   5. Re-export amr_cnn_28class.onnx (opset 15, IR 8)
#   6. Restart scan pipeline with new models
set -euo pipefail

cd /home/brendan/AnalysisApp/tools/ml
LOG=/tmp/overnight_train.log
exec >> "$LOG" 2>&1

ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo ""; echo "════════════════════════════════════════════"
echo "Overnight training started: $(ts)"
echo "════════════════════════════════════════════"

# ── 0. Wait for current training run to finish ───────────────────────────────
echo "[$(ts)] Waiting for any running train.py to finish..."
while pgrep -f "train\.py" > /dev/null 2>&1; do
    sleep 30
done
echo "[$(ts)] No train.py running — proceeding."

# ── 1. Generate large synthetic dataset ──────────────────────────────────────
echo ""
echo "[$(ts)] Phase 1: Generating large synthetic dataset..."
echo "  28 classes × 4000 samples × SNR -10:+30 step 2 = ~2.35M samples"
.venv/bin/python3 generate_v3.py \
    --out data/synth_v3_large.npz \
    --n 4000 \
    --len 512 \
    --snr-min -10 \
    --snr-max 30 \
    --snr-step 2 \
    --seed 123
echo "[$(ts)] Generation done: $(ls -lh data/synth_v3_large.npz | awk '{print $5}')"

# ── 2. Train ResNet 200 epochs ───────────────────────────────────────────────
echo ""
echo "[$(ts)] Phase 2: Training ResNet (200 epochs, channels=128, n_blocks=8)..."
echo "  Data: synth_v3_large + extended + real_fresh + real_ils + ils_synthetic"
echo "  max-per-class=20000  lr=2e-4  batch=256  warmup+cosine"

# Temporarily patch ResNet size to avoid OOM (channels=128, n_blocks=8)
# We do this by passing --model resnet after patching model.py inline via env
ORIG_RESNET=$(grep "channels=160, n_blocks=12" train.py || true)

# Patch to safe ResNet size
sed -i 's/channels=160, n_blocks=12/channels=128, n_blocks=8/g' train.py
echo "[$(ts)] Patched ResNet: channels=128, n_blocks=8"

CKPT=models/amr_resnet_overnight.best.pt
ONNX_OUT=models/amr_cnn_28class.onnx   # output name kept for compatibility

.venv/bin/python3 -u train.py \
    --npz data/synth_v3_large.npz \
          data/extended.npz \
          data/real_fresh.npz \
          data/real_ils.npz \
          data/ils_synthetic.npz \
    --model resnet \
    --epochs 200 \
    --batch 256 \
    --lr 2e-4 \
    --max-per-class 20000 \
    --snr-min -10 \
    --cuda \
    --out "$ONNX_OUT"

# Restore original ResNet size in train.py
sed -i 's/channels=128, n_blocks=8/channels=160, n_blocks=12/g' train.py
echo "[$(ts)] Restored train.py ResNet size"

echo "[$(ts)] ResNet training done."

# ── 3. Re-export with correct ONNX settings (IR 8, opset 15, iq_input/logits) ──
echo ""
echo "[$(ts)] Phase 3: Re-exporting ONNX (opset 15, IR 8, iq_input/logits)..."
.venv/bin/python3 - << 'PYEOF'
import torch, json
from model import RadioCNN, RadioResNet

classes = json.load(open("models/amr_cnn_28class.classes.json"))
num_classes = len(classes)
input_len = 512

model = RadioResNet(num_classes=num_classes, input_len=input_len, channels=128, n_blocks=8)
ckpt = torch.load("models/amr_cnn_28class.onnx".replace(".onnx", ".best.pt"),
                  map_location="cpu", weights_only=True)
model.load_state_dict(ckpt)
model.eval()

dummy = torch.zeros(1, 2, input_len)
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    torch.onnx.export(
        model, dummy, "models/amr_cnn_28class.onnx",
        export_params=True, opset_version=15, dynamo=False,
        input_names=["iq_input"], output_names=["logits"],
        dynamic_axes={"iq_input": {0: "batch"}, "logits": {0: "batch"}},
    )

import onnx
m = onnx.load("models/amr_cnn_28class.onnx")
print(f"Exported: IR={m.ir_version}  inputs={[i.name for i in m.graph.input]}  outputs={[o.name for o in m.graph.output]}")
PYEOF
echo "[$(ts)] ONNX export done."

# ── 4. Retrain DAE ───────────────────────────────────────────────────────────
echo ""
echo "[$(ts)] Phase 4: Retraining DAE on all data..."
.venv/bin/python3 -u train_dae.py \
    --npz data/synth_v3_large.npz \
          data/extended.npz \
          data/real_fresh.npz \
          data/real_ils.npz \
    --epochs 60 \
    --batch 512 \
    --cuda \
    --out models/dae_iq.onnx
echo "[$(ts)] DAE training done."

# ── 5. Chain DAE + ResNet into low-SNR denoised model ───────────────────────
echo ""
echo "[$(ts)] Phase 5: Chaining DAE + ResNet into amr_low_snr_denoised.onnx..."
.venv/bin/python3 - << 'PYEOF'
import onnx
from onnx import compose

dae = onnx.load("models/dae_iq.onnx")
clf = onnx.load("models/amr_cnn_28class.onnx")

clf_prefixed = compose.add_prefix(clf, prefix="clf_", rename_edges=True)
combined = compose.merge_models(dae, clf_prefixed, io_map=[("iq_clean", "clf_iq_input")])

# Rename input iq_noisy -> iq_input and output clf_logits -> logits
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
print(f"Low-SNR chain: IR={m.ir_version}  inputs={[i.name for i in m.graph.input]}  outputs={[o.name for o in m.graph.output]}")
PYEOF
echo "[$(ts)] Chain done."

# ── 6. Update symlinks ───────────────────────────────────────────────────────
cd models
ln -sf amr_cnn_28class.onnx modulation_classifier.onnx
ln -sf amr_cnn_28class.classes.json classes.json
cd ..
echo "[$(ts)] Symlinks updated."

# ── 7. Restart scan pipeline ─────────────────────────────────────────────────
echo ""
echo "[$(ts)] Phase 6: Restarting scan pipeline with new models..."
cd /home/brendan/SdrScripts

# Restart just the analysis container (models are bind-mounted, no rebuild needed)
ML_MODEL_DIR=/home/brendan/AnalysisApp/tools/ml/models
LATEST_CFG=$(ls -t /tmp/analysis_config.*.xml 2>/dev/null | head -1)
if [[ -n "$LATEST_CFG" ]]; then
    podman stop sdr-analysis 2>/dev/null || true
    sleep 3
    podman run -d --rm --replace --name sdr-analysis --network=host \
        -v "${LATEST_CFG}:/etc/sdr-analysis/analysis.xml:ro,z" \
        -v "${ML_MODEL_DIR}:/models:ro,z" \
        sdr-analysis:hw-onnx
    echo "[$(ts)] AnalysisApp restarted with new models."
else
    echo "[$(ts)] WARNING: No analysis config found — restart scan.sh manually."
fi

echo ""
echo "════════════════════════════════════════════"
echo "Overnight training COMPLETE: $(ts)"
echo "════════════════════════════════════════════"
echo "Models deployed:"
ls -lh /home/brendan/AnalysisApp/tools/ml/models/*.onnx 2>/dev/null
