#!/usr/bin/env bash
# ========================================================================
# Project: OpenRFStack
# Author:  Brendan Michaud
# Year:    2026
# Part of OpenRFStack (https://github.com/OpenRFStack)
#
# Licensed under the Personal Use License.
# Do not use for commercial, organizational, or military purposes.
# ========================================================================

set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_TYPE="Release"
RUN_TESTS=false
CLEAN=false
NO_CLONE=false

for arg in "$@"; do
    case $arg in
        --debug)    BUILD_TYPE="Debug" ;;
        --tests)    RUN_TESTS=true ;;
        --clean)    CLEAN=true ;;
        --no-clone) NO_CLONE=true ;;
        --help)
            echo "Usage: build.sh [--debug] [--tests] [--clean] [--no-clone]"
            echo ""
            echo "  --debug     Build in Debug mode (default: Release)"
            echo "  --tests     Run tests after build"
            echo "  --clean     Remove build directory before building"
            echo "  --no-clone  Do not attempt to clone SdrTaskApi"
            exit 0
            ;;
    esac
done

PARENT="$(dirname "$REPO_DIR")"
if [[ "$NO_CLONE" == "false" ]] && [[ ! -d "$PARENT/SdrTaskApi" ]]; then
    echo "[build.sh] Cloning SdrTaskApi sibling..."
    git clone https://github.com/BMichaud7/SdrTaskApi.git "$PARENT/SdrTaskApi"
fi

BUILD_DIR="$REPO_DIR/build"
if [[ "$CLEAN" == "true" ]]; then
    echo "[build.sh] Cleaning build directory..."
    rm -rf "$BUILD_DIR"
fi

echo "[build.sh] Configuring (BUILD_TYPE=$BUILD_TYPE)..."
cmake -B "$BUILD_DIR" \
      -S "$REPO_DIR" \
      -DCMAKE_BUILD_TYPE="$BUILD_TYPE"

echo "[build.sh] Building..."
cmake --build "$BUILD_DIR" --parallel "$(nproc)"

if [[ "$RUN_TESTS" == "true" ]]; then
    echo "[build.sh] Running tests..."
    ctest --test-dir "$BUILD_DIR" --output-on-failure
fi

echo "[build.sh] Done. Binary: $BUILD_DIR/sdr_analysis"
