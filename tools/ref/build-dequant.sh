#!/usr/bin/env bash
# Build the stage-1 dequantization oracle on the box (round 1-1).
# Runs under tools/box.sh (toolchain env already sourced). IK, the data root and the ggml flags
# come from ref-build-common.sh; the binary goes to $BLOOMERY_DATA/bin unless DEQUANT_OUT says otherwise.
# `just dump-ref-dequant` (tools/ref/dump-dequant.sh) runs it before every dump: the dump records the
# binary's md5, which the refset dequant families pin, so a rebuilt binary of other bytes is a re-pin.
set -euo pipefail
# shellcheck source=tools/ref/ref-build-common.sh
source "$(dirname "${BASH_SOURCE[0]}")/ref-build-common.sh"
OUT=${DEQUANT_OUT:-$REF_BIN}
mkdir -p "$OUT"
ref_cxx -o "$OUT/dequant_ref" "$HERE/tools/ref/dequant_ref.cpp" "${REF_GGML_INC[@]}" "${REF_GGML_LINK[@]}"
echo "built $OUT/dequant_ref"
