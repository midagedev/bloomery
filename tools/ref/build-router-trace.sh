#!/usr/bin/env bash
# Build the router-trace harness (tools/ref/router_trace.cpp) on the box. Links libllama +
# libcommon like the oracle dumper; the flags come from ref-build-common.sh. router-trace.sh calls
# this before it takes the lease, so a trace always runs the source box.sh just synced.
#
# Output: ROUTER_OUT or $BLOOMERY_DATA/router/bin — never $BLOOMERY_DATA/bin, where the oracle
# instruments every gate depends on live.
#
#   bash tools/ref/build-router-trace.sh
set -euo pipefail
# shellcheck source=tools/ref/ref-build-common.sh
source "$(dirname "${BASH_SOURCE[0]}")/ref-build-common.sh"
OUT=${ROUTER_OUT:-$BLOOMERY_DATA/router/bin}
mkdir -p "$OUT"
ref_cxx -o "$OUT/router_trace" "$HERE/tools/ref/router_trace.cpp" "${REF_LLAMA_INC[@]}" "${REF_LLAMA_LINK[@]}"
echo "built $OUT/router_trace"
