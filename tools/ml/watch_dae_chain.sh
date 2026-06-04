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

# watch_dae_chain.sh — Wait for DAE retrain to finish, re-chain with 24-class
# classifier, update amr_low_snr_denoised.onnx, and restart sdr-analysis.
#
# Usage:
#   nohup ./watch_dae_chain.sh >> /tmp/watch_dae_chain.log 2>&1 &
set -euo pipefail

cd /home/brendan/AnalysisApp/tools/ml
LOG=/tmp/watch_dae_chain.log
ts() { date '+%Y-%m-%d %H:%M:%S'; }

NEW_ONNX=models/dae_iq_new.onnx
NEW_PT=models/dae_iq_new.pt
CLF_ONNX=models/amr_cnn_24class.onnx
CHAIN_OUT=models/amr_low_snr_denoised.onnx

echo "[$(ts)] Watcher started — polling for ${NEW_ONNX}"

# ── Wait for DAE export to land (poll every 5 min) ────────────────────────────
until [[ -f "$NEW_ONNX" ]]; do
    sleep 300
done

# Give the write a moment to flush fully
sleep 10

echo "[$(ts)] ${NEW_ONNX} detected — running ONNX chain..."

# ── Chain dae_iq_new.onnx + amr_cnn_24class.onnx ─────────────────────────────
.venv/bin/python3 - << 'PYEOF'
import onnx
from onnx import compose

dae = onnx.load("models/dae_iq_new.onnx")
clf = onnx.load("models/amr_cnn_24class.onnx")

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

# ── Promote new DAE as the canonical dae_iq.onnx/pt ─────────────────────────
cp -f "$NEW_ONNX" models/dae_iq.onnx
cp -f "$NEW_PT"   models/dae_iq.pt
echo "[$(ts)] Promoted dae_iq_new → dae_iq"

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

# Clean up new files so the watcher is idempotent if re-run
rm -f "$NEW_ONNX" "$NEW_PT"
echo "[$(ts)] Done."

# ========================================================================
# End of file — OpenRFStack
# Subject to Personal Use License
# https://github.com/OpenRFStack
# ========================================================================
