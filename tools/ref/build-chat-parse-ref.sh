#!/usr/bin/env bash
# Link chat_parse_ref (tools/ref/chat_parse_ref.cpp) against mainline llama.cpp's existing build:
# its common library (the chat templates and parsers llama-server runs) and libllama. This does
# not configure or rebuild that tree. Not a model's profile: the binary opens no model.
#   CHAT_PARSE_LCPP   the mainline tree (default /home/user/llama.cpp-mainline)
#   CHAT_PARSE_OUT    where the binary and its .build record go (default $BLOOMERY_DATA/bin)
# The .build record names the source's sha256 and the mainline commit it was linked against.
set -euo pipefail
# shellcheck source=tools/ref/ref-paths.sh
source "$(dirname "${BASH_SOURCE[0]}")/ref-paths.sh"
LCPP=${CHAT_PARSE_LCPP:-/home/user/llama.cpp-mainline}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
OUT=${CHAT_PARSE_OUT:-$BLOOMERY_DATA/bin}
SRC=$HERE/tools/ref/chat_parse_ref.cpp
git_lcpp() { git -c safe.directory="$LCPP" -C "$LCPP" "$@"; }
dirty=$(git_lcpp status --porcelain --untracked-files=no)
if [ -n "$dirty" ]; then
  echo "build-chat-parse-ref: $LCPP has tracked changes; the oracle is the commit's build, not an edited tree" >&2
  exit 2
fi
LIBDIR=$LCPP/build/bin
for lib in libllama-common.so libllama.so; do
  [ -e "$LIBDIR/$lib" ] || { echo "build-chat-parse-ref: no $LIBDIR/$lib (mainline is not built)" >&2; exit 66; }
done
COMMIT=$(git_lcpp rev-parse --short=9 HEAD)
mkdir -p "$OUT"
TMP=$OUT/chat_parse_ref.tmp.$$
trap 'rm -f "$TMP" "$TMP.build"' EXIT
SRC_SHA=$(sha256sum < "$SRC" | cut -d' ' -f1)
g++ -std=c++17 -O2 -o "$TMP" "$SRC" -I"$LCPP/common" -I"$LCPP/include" -I"$LCPP/ggml/include" -I"$LCPP/vendor" \
  -L"$LIBDIR" -lllama-common -lllama -lggml -lggml-base -Wl,-rpath,"$LIBDIR"
printf 'source_sha256 %s\nlcpp_commit %s\n' "$SRC_SHA" "$COMMIT" > "$TMP.build"
mv -f "$TMP" "$OUT/chat_parse_ref"
mv -f "$TMP.build" "$OUT/chat_parse_ref.build"
echo "built $OUT/chat_parse_ref from chat_parse_ref.cpp sha256 ${SRC_SHA:0:12} against mainline at $COMMIT"
