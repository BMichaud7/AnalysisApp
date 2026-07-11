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

# run_hierarchical_pipeline.sh — builds the hard-hierarchical AMR pipeline:
# trains the family router, then each of the 9 family specialists in turn
# (one GPU, sequential), then evaluates the chained router->specialist
# pipeline against the v6 holdout and prints a comparison line against the
# flat v9 model (52.7%) and v10 (38.0%).

set -euo pipefail
cd /home/brendan/AnalysisApp/tools/ml
LOG=/tmp/train_hierarchical_pipeline.log
exec >> "$LOG" 2>&1

ts() { date '+%Y-%m-%d %H:%M:%S'; }

FAMILIES=(AM ASK DIGI_VOICE FM FSK PROTOCOL PSK QAM SPECIAL)

echo ""
echo "════════════════════════════════════════"
echo "Hierarchical pipeline starting: $(ts)"
echo "Families: ${FAMILIES[*]}"
echo "════════════════════════════════════════"

echo ""
echo "[$(ts)] Stage 0/10: training router..."
./run_router_train.sh

for fam in "${FAMILIES[@]}"; do
    echo ""
    echo "[$(ts)] Training family specialist: $fam"
    ./run_family_train.sh "$fam"
done

echo ""
echo "[$(ts)] All models trained. Running chained router->specialist eval..."
.venv/bin/python3 -u eval_hierarchical_chain.py \
    --router models/amr_router_v1.onnx \
    --router-classes models/amr_router_v1.classes.json \
    --specialists-dir models \
    --specialist-tag v1 \
    --holdout data/v6_holdout_v2_1024.npz

echo ""
echo "════════════════════════════════════════"
echo "Hierarchical pipeline DONE: $(ts)"
echo "Compare overall_acc above against flat v9 (52.7%) and v10 (38.0%)."
echo "════════════════════════════════════════"

# ========================================================================
# End of file — OpenRFStack
# Subject to Personal Use License
# https://github.com/OpenRFStack
# ========================================================================
