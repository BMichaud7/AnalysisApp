#!/bin/bash
# Publishes the current contents of tools/ml/models/ as a GitHub Release on
# OpenRFStack/AnalysisApp, tagged models-<timestamp>. Deliberately separate
# from the latest-main release (that one's the CI-built RPM binaries the
# product containers pull at startup -- see SdrScripts/deploy/*/entrypoint.sh)
# so model drops never collide with or get mistaken for a binary build.
#
# Run by hand, or from the end of a training chain (e.g.
# run_router_family_chain.sh) once you're happy with what's in models/.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

REPO="OpenRFStack/AnalysisApp"
TAG="models-$(date +%Y%m%d-%H%M%S)"
MODELS_DIR="models"

shopt -s nullglob
assets=()
for f in "$MODELS_DIR"/*.onnx "$MODELS_DIR"/*.onnx.data "$MODELS_DIR"/*.classes.json; do
    [[ "$f" == *.bak ]] && continue
    assets+=("$f")
done
shopt -u nullglob

if [[ ${#assets[@]} -eq 0 ]]; then
    echo "publish_models_release: no .onnx/.classes.json files found in $MODELS_DIR, nothing to publish" >&2
    exit 1
fi

notes_file=$(mktemp)
trap 'rm -f "$notes_file"' EXIT
{
    echo "Model snapshot published $(date '+%Y-%m-%d %H:%M:%S %Z')."
    echo ""
    echo "| file | size |"
    echo "|---|---|"
    for f in "${assets[@]}"; do
        printf '| `%s` | %s |\n' "$(basename "$f")" "$(du -h "$f" | cut -f1)"
    done
} > "$notes_file"

echo "Publishing ${#assets[@]} file(s) to $REPO as $TAG ..."
if ! gh release create "$TAG" "${assets[@]}" \
    --repo "$REPO" \
    --title "Models $(date +%Y-%m-%d)" \
    --notes-file "$notes_file"; then
    echo "publish_models_release: gh release create failed" >&2
    exit 1
fi

echo "Published: https://github.com/$REPO/releases/tag/$TAG"
