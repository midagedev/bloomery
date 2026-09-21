#!/usr/bin/env bash
# Build the kv-cache-clear reproducer on the box. Mirrors tools/ref/build-argmax.sh.
# The runner rsyncs this tree with --delete, so the binary lives outside it. The default
# output is the scratch dir, not $BLOOMERY_DATA/bin: this is an investigation tool and the
# installed reference binaries are what the recorded numbers came from.
# IK points at the ik tree to link against; a patched build under the scratch dir is the
# way to test an intervention without touching the serving user's tree.
set -euo pipefail
: "${IK:=/home/user/ik_llama.cpp}"
OUT=${KVCLEAR_OUT:-/root/bloomery-scratch/ikclear/bin}
HERE=$(cd "$(dirname "$0")/../.." && pwd)
mkdir -p "$OUT"
g++ -std=c++17 -O2 -o "$OUT/kvclear_probe" "$HERE/tools/ref/kvclear_probe.cpp" \
  -I"$IK/ggml/include" -I"$IK/include" -I"$IK/common" -I"$IK/src" \
  -L"$IK/build/common" -L"$IK/build/src" -L"$IK/build/ggml/src" \
  -lcommon -lllama -lggml \
  -Wl,-rpath,"$IK/build/src" -Wl,-rpath,"$IK/build/ggml/src"
echo "built $OUT/kvclear_probe (IK=$IK)"
