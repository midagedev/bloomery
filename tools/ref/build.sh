#!/usr/bin/env bash
# Build the stage-0 reference harness on the box.
# Runs under tools/box.sh (toolchain env already sourced). IK, the data root and the ggml flags
# come from ref-build-common.sh; the binary goes to $BLOOMERY_DATA/bin unless Q3K_OUT says otherwise.
set -euo pipefail
# shellcheck source=tools/ref/ref-build-common.sh
source "$(dirname "${BASH_SOURCE[0]}")/ref-build-common.sh"
OUT=${Q3K_OUT:-$REF_BIN}
mkdir -p "$OUT"
ref_cxx -o "$OUT/q3k_ref" "$HERE/tools/ref/q3k_ref.cpp" "${REF_GGML_INC[@]}" "${REF_GGML_LINK[@]}"
echo "built $OUT/q3k_ref"
