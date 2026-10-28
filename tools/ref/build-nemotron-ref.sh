#!/usr/bin/env bash
# Link nemotron_ref (tools/ref/nemotron_ref.cpp, with tools/ref/ref_io.h) against the existing mainline build of the
# nemotron_h, nemotron_h_moe or lfm3 profile (each names its own LCPP; the binary takes the architecture as --arch). This does
# not configure or rebuild that tree: a rebuild moves the md5s the logits families pin. The .build record spells those
# md5s as crate::tokenizer::build_id does (llama-tokenize <md5> libllama.so <md5>; the file on disk is libllama.so.0).
# Output: $BLOOMERY_DATA/bin/<name> and <name>.build beside it; <name> is nemotron_ref under nemotron_h, nemotron_ref_moe under
# nemotron_h_moe and nemotron_ref_lfm3 under lfm3, because the profiles name different mainline trees (dump-nemotron.sh names
# the same binary).
set -euo pipefail
# shellcheck source=tools/ref/ref-paths.sh
source "$(dirname "${BASH_SOURCE[0]}")/ref-paths.sh"
case $MODEL_NAME in
  nemotron_h) NAME=nemotron_ref ;;
  nemotron_h_moe) NAME=nemotron_ref_moe ;;
  lfm3) NAME=nemotron_ref_lfm3 ;;
  *)
    echo "build-nemotron-ref: this binary is built under the nemotron_h, nemotron_h_moe or lfm3 profile; this command picked $MODEL_NAME" >&2
    exit 2
    ;;
esac
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
OUT=${NEMOTRON_OUT:-$BLOOMERY_DATA/bin}
SRC=$HERE/tools/ref/nemotron_ref.cpp
IO=$HERE/tools/ref/ref_io.h
git_lcpp() { git -c safe.directory="$LCPP" -C "$LCPP" "$@"; }
dirty=$(git_lcpp status --porcelain --untracked-files=no) ||
  { echo "build-nemotron-ref: cannot inspect tracked changes in $LCPP" >&2; exit 2; }
if [ -n "$dirty" ]; then
  echo "build-nemotron-ref: $LCPP has tracked changes; the oracle is the commit's build, not an edited tree" >&2
  exit 2
fi
if [ "$MODEL_NAME" = lfm3 ]; then
  head=$(git_lcpp rev-parse HEAD) ||
    { echo "build-nemotron-ref: cannot read the commit of $LCPP" >&2; exit 2; }
  [ "$head" = "$LCPP_COMMIT" ] ||
    { echo "build-nemotron-ref: $LCPP is at $head, the lfm3 profile pins $LCPP_COMMIT" >&2; exit 2; }
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
TMP=$OUT/$NAME.tmp.$$
trap 'rm -f "$TMP" "$TMP.build"' EXIT
# cpp then the shared header, the same order dump-nemotron.sh hashes.
SRC_SHA=$(cat "$SRC" "$IO" | sha256sum | cut -d' ' -f1)
g++ -std=c++17 -O2 -o "$TMP" "$SRC" -I"$LCPP/include" -I"$LCPP/ggml/include" \
  -L"$LIBDIR" -lllama -lggml -lggml-base -Wl,-rpath,"$LIBDIR"
printf 'source_sha256 %s\nlcpp_commit %s\nbuild %s\n' "$SRC_SHA" "$COMMIT" "$BUILD_ID" > "$TMP.build"
mv -f "$TMP" "$OUT/$NAME"
mv -f "$TMP.build" "$OUT/$NAME.build"
echo "built $OUT/$NAME from nemotron_ref.cpp and ref_io.h sha256 ${SRC_SHA:0:12} against mainline at $COMMIT"
echo "build $BUILD_ID"
