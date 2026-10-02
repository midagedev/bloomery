#!/usr/bin/env bash
# Build the hidden-state oracle on the box: hidden_ref (tools/ref/hidden_ref.cpp) linked against llama.cpp
# mainline's libllama and libggml in the qwen35 profile's LCPP build (tools/ref/models/qwen35.sh), not the ik
# tree the node dumpers link: mainline is this model's oracle. Under the qwen35 profile only.
#
# The tree is checked, not trusted: no tracked change (a hand edit there would reach the oracle under the
# commit's name). Its build is the one the depth runners time; this links against it and builds nothing in it.
# Output: $BLOOMERY_DATA/bin/hidden_ref (HIDDEN_OUT moves it) and hidden_ref.build beside it, naming the sha256 of
# the hidden_ref.cpp it was built from and the mainline commit; hidden.sh refuses a binary whose record does not
# match its own tree. The compile writes a temporary name and only a finished link replaces the installed binary.
set -euo pipefail
# shellcheck source=tools/ref/ref-paths.sh
source "$(dirname "${BASH_SOURCE[0]}")/ref-paths.sh"
[ "$MODEL_NAME" = qwen35 ] ||
  { echo "build-hidden: hidden_ref is built under the qwen35 profile; this command picked $MODEL_NAME" >&2; exit 2; }
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
OUT=${HIDDEN_OUT:-$BLOOMERY_DATA/bin}
SRC=$HERE/tools/ref/hidden_ref.cpp
git_lcpp() { git -c safe.directory="$LCPP" -C "$LCPP" "$@"; }
if [ -n "$(git_lcpp status --porcelain --untracked-files=no)" ]; then
  echo "build-hidden: $LCPP has tracked changes; the oracle is the commit's build, not an edited tree" >&2
  exit 2
fi
COMMIT=$(git_lcpp rev-parse --short=9 HEAD)
LIB=$LCPP/build/bin
mkdir -p "$OUT"
TMP=$OUT/hidden_ref.tmp.$$
trap 'rm -f "$TMP" "$TMP.build"' EXIT
SRC_SHA=$(sha256sum "$SRC" | cut -d' ' -f1)
g++ -std=c++17 -O2 -o "$TMP" "$SRC" -I"$LCPP/include" -I"$LCPP/ggml/include" \
  -L"$LIB" -lllama -lggml -lggml-base -Wl,-rpath,"$LIB"
printf 'source_sha256 %s\nlcpp %s\nlcpp_commit %s\n' "$SRC_SHA" "$(readlink -f "$LCPP")" "$COMMIT" > "$TMP.build"
mv -f "$TMP" "$OUT/hidden_ref"
mv -f "$TMP.build" "$OUT/hidden_ref.build"
echo "built $OUT/hidden_ref from hidden_ref.cpp sha256 ${SRC_SHA:0:12} against $LCPP at $COMMIT"
