#!/bin/bash
# watch_finetune.sh — Wait for 24-class fine-tune to finish, rebuild DAE chain,
# restart sdr-analysis.
#
# Usage:
#   nohup ./watch_finetune.sh >> /tmp/watch_finetune.log 2>&1 &
set -euo pipefail

cd /home/brendan/AnalysisApp/tools/ml
LOG=/tmp/finetune_24class.log
ts() { date '+%Y-%m-%d %H:%M:%S'; }

CLF_ONNX=models/amr_cnn_24class.onnx
DAE_ONNX=models/dae_iq.onnx
CHAIN_OUT=models/amr_low_snr_denoised.onnx

echo "[$(ts)] Watcher started — polling ${LOG} for fine-tune completion"

# ── Wait for train.py to finish exporting ─────────────────────────────────────
until grep -q "Class names saved" "$LOG" 2>/dev/null; do
    sleep 300
done

# Confirm it didn't crash after that line
if grep -q "Traceback\|Error\|error" <(tail -5 "$LOG" 2>/dev/null); then
    echo "[$(ts)] ERROR: log shows failure after completion marker — aborting"
    exit 1
fi

echo "[$(ts)] Fine-tune complete — rebuilding DAE chain..."

# ── Rebuild amr_low_snr_denoised.onnx ────────────────────────────────────────
.venv/bin/python3 - << 'PYEOF'
import onnx
from onnx import compose, version_converter

dae = onnx.load("models/dae_iq.onnx")
clf = onnx.load("models/amr_cnn_24class.onnx")

# Align opset versions before merging (DAE may lag behind CNN on upgrades)
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
print(f"Chain OK: IR={m.ir_version} opset={m.opset_import[0].version} "
      f"inputs={[i.name for i in m.graph.input]} "
      f"outputs={[o.name for o in m.graph.output]}")
PYEOF

echo "[$(ts)] Chain written to ${CHAIN_OUT}"

# ── Restart sdr-analysis ──────────────────────────────────────────────────────
LATEST_CFG=$(ls -t /tmp/analysis_config.*.xml 2>/dev/null | head -1)
if [[ -z "$LATEST_CFG" ]]; then
    echo "[$(ts)] ERROR: no analysis config found — restart sdr-analysis manually"
    exit 1
fi

ML_MODEL_DIR=/home/brendan/AnalysisApp/tools/ml/models
podman stop sdr-analysis 2>/dev/null || true
sleep 3
podman run -d --rm --replace --name sdr-analysis --network=host \
    -v "${LATEST_CFG}:/etc/sdr-analysis/analysis.xml:ro,z" \
    -v "${ML_MODEL_DIR}:/models:ro,z" \
    sdr-analysis:hw-onnx

sleep 5
echo "[$(ts)] Container restarted. Startup log:"
podman logs sdr-analysis --tail 8 2>/dev/null

# ── Evaluate new model on holdout sets ───────────────────────────────────────
REPORT=/tmp/finetune_holdout_report.json
ML_DIR=/home/brendan/AnalysisApp/tools/ml
echo "[$(ts)] Running holdout evaluation..."
podman run --rm \
    -v "${ML_DIR}:/ml:z" \
    docker.io/library/python:3.12-slim \
    bash -c "pip install -q numpy onnxruntime && python /ml/eval_holdout.py \
        --model /ml/models/amr_cnn_24class.onnx \
        --classes /ml/models/amr_cnn_24class.classes.json \
        --holdout /ml/data/v3_holdout.npz /ml/data/v3_holdout_impaired.npz \
        --out /ml/data/finetune_report.json" \
    | tee "$REPORT"
echo "[$(ts)] Eval complete — report at /ml/data/finetune_report.json"

echo "[$(ts)] Done."
