#!/usr/bin/env bash
# Build the kv-cache-clear reproducer on the box. Links libllama + libcommon; the flags come from
# ref-build-common.sh. The default output is the scratch dir, not $BLOOMERY_DATA/bin: this is an
# investigation tool and the installed reference binaries are what the recorded numbers came from.
# IK points at the ik tree to link against; a patched build under the scratch dir is the
# way to test an intervention without touching the serving user's tree.
set -euo pipefail
# shellcheck source=tools/ref/ref-build-common.sh
source "$(dirname "${BASH_SOURCE[0]}")/ref-build-common.sh"
OUT=${KVCLEAR_OUT:-/root/bloomery-scratch/ikclear/bin}
mkdir -p "$OUT"
ref_cxx -o "$OUT/kvclear_probe" "$HERE/tools/ref/kvclear_probe.cpp" "${REF_LLAMA_INC[@]}" "${REF_LLAMA_LINK[@]}"
echo "built $OUT/kvclear_probe (IK=$IK)"
