#!/usr/bin/env bash
# Build the stage-1 oracle instrument on the box. Links libllama + libcommon (loads the model
# through llama.h); the flags come from ref-build-common.sh. Output: DUMP_OUT or $BLOOMERY_DATA/bin.
set -euo pipefail
# shellcheck source=tools/ref/ref-build-common.sh
source "$(dirname "${BASH_SOURCE[0]}")/ref-build-common.sh"
OUT=${DUMP_OUT:-$REF_BIN}
mkdir -p "$OUT"
ref_cxx -o "$OUT/dump_ref" "$HERE/tools/ref/dump_ref.cpp" "${REF_LLAMA_INC[@]}" "${REF_LLAMA_LINK[@]}"
echo "built $OUT/dump_ref"
