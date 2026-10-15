#!/usr/bin/env bash
# Link nemotron_ref (tools/ref/nemotron_ref.cpp) against the nemotron_h profile's existing
# mainline build. This does not configure or rebuild that tree: a rebuild moves the md5s the
# tokenizer family pins. The .build record spells those md5s as crate::tokenizer::build_id does
# (llama-tokenize <md5> libllama.so <md5>; the file on disk is libllama.so.0).
# Output: $BLOOMERY_DATA/bin/nemotron_ref and nemotron_ref.build beside it.
set -euo pipefail
# shellcheck source=tools/ref/ref-paths.sh
source "$(dirname "${BASH_SOURCE[0]}")/ref-paths.sh"
[ "$MODEL_NAME" = nemotron_h ] || {
  echo "build-nemotron-ref: this binary is built under the nemotron_h profile; this command picked $MODEL_NAME" >&2
  exit 2
}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
OUT=${NEMOTRON_OUT:-$BLOOMERY_DATA/bin}
SRC=$HERE/tools/ref/nemotron_ref.cpp
git_lcpp() { git -c safe.directory="$LCPP" -C "$LCPP" "$@"; }
if [ -n "$(git_lcpp status --porcelain --untracked-files=no)" ]; then
  echo "build-nemotron-ref: $LCPP has tracked changes; the oracle is the commit's build, not an edited tree" >&2
  exit 2
fi
TOKENIZE=$LCPP/build/bin/llama-tokenize
[ -x "$TOKENIZE" ] || { echo "build-nemotron-ref: no llama-tokenize at $TOKENIZE" >&2; exit 66; }
LIBLLAMA=$(ldd "$TOKENIZE" | awk '$1 == "libllama.so" || $1 == "libllama.so.0" { print $3; exit }')
[ -f "$LIBLLAMA" ] || { echo "build-nemotron-ref: $TOKENIZE loads no libllama.so (ldd names '$LIBLLAMA')" >&2; exit 66; }
TOK_MD5=$(md5sum < "$TOKENIZE" | cut -d' ' -f1)
LIB_MD5=$(md5sum < "$LIBLLAMA" | cut -d' ' -f1)
BUILD_ID="llama-tokenize $TOK_MD5 libllama.so $LIB_MD5"
COMMIT=$(git_lcpp rev-parse --short=9 HEAD)
LIBDIR=$LCPP/build/bin
mkdir -p "$OUT"
TMP=$OUT/nemotron_ref.tmp.$$
trap 'rm -f "$TMP" "$TMP.build"' EXIT
SRC_SHA=$(sha256sum "$SRC" | cut -d' ' -f1)
g++ -std=c++17 -O2 -o "$TMP" "$SRC" -I"$LCPP/include" -I"$LCPP/ggml/include" \
  -L"$LIBDIR" -lllama -lggml -lggml-base -Wl,-rpath,"$LIBDIR"
printf 'source_sha256 %s\nlcpp_commit %s\nbuild %s\n' "$SRC_SHA" "$COMMIT" "$BUILD_ID" > "$TMP.build"
mv -f "$TMP" "$OUT/nemotron_ref"
mv -f "$TMP.build" "$OUT/nemotron_ref.build"
echo "built $OUT/nemotron_ref from nemotron_ref.cpp sha256 ${SRC_SHA:0:12} against mainline at $COMMIT"
echo "build $BUILD_ID"
