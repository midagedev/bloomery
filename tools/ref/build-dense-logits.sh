#!/usr/bin/env bash
# Build the dense logits oracle on the box: dense_logits_ref (tools/ref/dense_logits_ref.cpp) linked against llama.cpp
# mainline's libllama and libggml in the profile's LCPP build (tools/ref/models/qwen38d.sh, nanbeige43.sh: one tree, the one the
# qwen35 hidden-state oracle runs), under a profile that names logits sets (LOGITS_SETS). Either profile builds the one binary.
#
# The tree is proven, not trusted, and nothing is built in it (it is the build the depth runners time):
# tools/ref/dense-logits.py build --pin runs tools/ref/emb2/oracle-tree.sh on it (HEAD is LCPP_PIN, no tracked change,
# the libllama.so its llama-tokenize loads lies under the tree's build and is newer than every tracked source) and links.
# Output: $HERE/target/release/dense_logits_ref (DENSE_LOGITS_OUT moves it; tools/gpu-gate.sh runs a card binary from
# target/release of the box's directory) and dense_logits_ref.build.json beside it, naming the sha256 of
# dense_logits_ref.cpp, the commit and every library's sha256; dense-logits.sh refuses a binary whose record does not match
# its own tree. A rebuild replaces the old binary and record; a build that fails leaves neither.
set -euo pipefail
# shellcheck source=tools/ref/ref-paths.sh
source "$(dirname "${BASH_SOURCE[0]}")/ref-paths.sh"
declare -p LOGITS_SETS > /dev/null 2>&1 ||
  { echo "build-dense-logits: dense_logits_ref is built under a profile that names logits sets; the $MODEL_NAME profile names none" >&2; exit 2; }
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
OUT=${DENSE_LOGITS_OUT:-$HERE/target/release/dense_logits_ref}
rm -f "$OUT" "$OUT.build.json" "$OUT.partial"
python3 "$HERE/tools/ref/dense-logits.py" build --llama-tree "$LCPP" --llama-build "$LCPP/build" --pin "$LCPP_PIN" --out "$OUT"
echo "built $OUT from dense_logits_ref.cpp against $LCPP at ${LCPP_PIN:0:9}"
