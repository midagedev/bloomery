#!/usr/bin/env bash
# Build the Clef image-input oracle on the box: dump_mtmd (tools/ref/clefvis/dump_mtmd.cpp) linked against llama.cpp
# mainline's libmtmd, libllama and libggml in the qwen35 profile's LCPP build (tools/ref/models/qwen35.sh), the tree
# hidden_ref links against: mainline has `qwen3vl_merger` natively, so no fork. Under the qwen35 profile only.
#
# The tree is checked, not trusted: no tracked change (a hand edit there would reach the oracle under the commit's
# name). Its build is the one the depth runners time; this links against it and builds nothing in it.
# Output: $BLOOMERY_DATA/bin/dump_mtmd (CLEFVIS_OUT moves it) and dump_mtmd.build beside it, naming the sha256 of the
# dump_mtmd.cpp it was built from, the mainline commit and the sha256 of the mtmd.h and mtmd-helper.h it saw;
# clefvis.sh refuses a binary whose record does not match its own tree. The compile writes a temporary name and only a
# finished link replaces the installed binary.
set -euo pipefail
# shellcheck source=tools/ref/ref-paths.sh
source "$(dirname "${BASH_SOURCE[0]}")/../ref-paths.sh"
[ "$MODEL_NAME" = qwen35 ] ||
  { echo "build-clefvis: dump_mtmd is built under the qwen35 profile; this command picked $MODEL_NAME" >&2; exit 2; }
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
OUT=${CLEFVIS_OUT:-$BLOOMERY_DATA/bin}
SRC=$HERE/tools/ref/clefvis/dump_mtmd.cpp
git_lcpp() { git -c safe.directory="$LCPP" -C "$LCPP" "$@"; }
if [ -n "$(git_lcpp status --porcelain --untracked-files=no)" ]; then
  echo "build-clefvis: $LCPP has tracked changes; the oracle is the commit's build, not an edited tree" >&2
  exit 2
fi
COMMIT=$(git_lcpp rev-parse --short=9 HEAD)
LIB=$LCPP/build/bin
for l in libmtmd.so libllama.so libggml.so libggml-base.so; do
  [ -e "$LIB/$l" ] || { echo "build-clefvis: no $LIB/$l (the mainline build is the depth runners'; it must carry mtmd)" >&2; exit 2; }
done
mkdir -p "$OUT"
TMP=$OUT/dump_mtmd.tmp.$$
trap 'rm -f "$TMP" "$TMP.build"' EXIT
SRC_SHA=$(sha256sum "$SRC" | cut -d' ' -f1)
H_SHA=$(cat "$LCPP/tools/mtmd/mtmd.h" "$LCPP/tools/mtmd/mtmd-helper.h" | sha256sum | cut -d' ' -f1)
g++ -std=c++17 -O2 -Wall -Wextra -Wno-unused-parameter -o "$TMP" "$SRC" -I"$LCPP/include" -I"$LCPP/ggml/include" \
  -I"$LCPP/tools/mtmd" -L"$LIB" -lmtmd -lllama -lggml -lggml-base -Wl,-rpath,"$LIB"
printf 'source_sha256 %s\nlcpp %s\nlcpp_commit %s\nmtmd_headers_sha256 %s\n' "$SRC_SHA" "$(readlink -f "$LCPP")" "$COMMIT" "$H_SHA" > "$TMP.build"
mv -f "$TMP" "$OUT/dump_mtmd"
mv -f "$TMP.build" "$OUT/dump_mtmd.build"
echo "built $OUT/dump_mtmd from dump_mtmd.cpp sha256 ${SRC_SHA:0:12} against $LCPP at $COMMIT"
