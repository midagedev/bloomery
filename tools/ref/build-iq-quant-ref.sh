#!/usr/bin/env bash
# Build gate_iq_quant's oracle, iq_quant_ref (tools/ref/iq_quant_ref.cpp), on the box against the K3.1 quantization
# tree's libggml-base: the library llama-quantize links (tools/ref/build-k3-quant-tree.sh, K3_QUANT_TREE). Not the ik
# tree of ref-build-common.sh: the oracle is the quantizer the K3.1 pipeline runs. The binary goes to
# $BLOOMERY_DATA/bin unless IQ_QUANT_REF_OUT says otherwise; the tree's commit and libggml-base's md5 go beside it in
# iq_quant_ref.build.
#
#   tools/box.sh 'bash tools/ref/build-iq-quant-ref.sh'      (just build-iq-quant-ref)
#
# Exit: 0 built; 66 the tree or its libggml-base is missing (build it with just build-k3-quant-tree).
set -euo pipefail
# shellcheck source=tools/ref/ref-paths.sh
source "$(dirname "${BASH_SOURCE[0]}")/ref-paths.sh"
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TREE=${K3_QUANT_TREE:-/home/user/llama.cpp-kimik3-quant}
LIB=$TREE/build/bin
OUT=${IQ_QUANT_REF_OUT:-$BLOOMERY_DATA/bin}
[ -f "$LIB/libggml-base.so" ] || { echo "build-iq-quant-ref.sh: no $LIB/libggml-base.so (just build-k3-quant-tree)" >&2; exit 66; }
mkdir -p "$OUT"
g++ -std=c++17 -O2 -o "$OUT/iq_quant_ref" "$HERE/tools/ref/iq_quant_ref.cpp" -I"$TREE/ggml/include" \
  -L"$LIB" -lggml-base -Wl,-rpath,"$LIB"
{
  echo "tree $TREE"
  echo "commit $(git -c safe.directory="$TREE" -C "$TREE" rev-parse HEAD)"
  echo "libggml-base.so md5 $(md5sum < "$LIB/libggml-base.so" | cut -d' ' -f1)"
} > "$OUT/iq_quant_ref.build"
echo "built $OUT/iq_quant_ref"
cat "$OUT/iq_quant_ref.build"
