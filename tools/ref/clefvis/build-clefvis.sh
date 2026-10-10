#!/usr/bin/env bash
# Build the image-input oracle on the box: dump_mtmd (tools/ref/clefvis/dump_mtmd.cpp) linked against llama.cpp mainline's libmtmd,
# libllama and libggml in the profile's ORACLE tree (tools/ref/clefvis/profile.sh): mainline has `qwen3vl_merger` natively, so no
# fork. Clef-Flash's seat (BLOOMERY_MODEL=qwen35) links the tree hidden_ref links (53ed051ce) into $BLOOMERY_DATA/bin/dump_mtmd;
# the Qwen seats (qwen35moe, qwen4exp) one binary between them, from the tree beside it (36a73916e, tools/ref/qvis/build-lcpp-qvis.sh)
# into dump_mtmd_qvis, with the chat mode (DUMP_MTMD_CHAT: it links libllama-common for the model's chat template and parser). The
# post-decode callback of the helper changed with that tree's header (mtmd_helper_embd_batch): the build defines
# DUMP_MTMD_EMBD_BATCH when the header has it.
#
# The tree is checked, not trusted: no tracked change (a hand edit there would reach the oracle under the commit's name). Its build
# is the one the depth runners time; this links against it and builds nothing in it.
# Output: the binary and a `.build` record beside it, naming the sha256 of the dump_mtmd.cpp it was built from, the mainline commit
# and the sha256 of the mtmd.h and mtmd-helper.h it saw; clefvis.sh refuses a binary whose record does not match its own tree.
# The compile writes a temporary name and only a finished link replaces the installed binary (CLEFVIS_OUT moves the directory).
set -euo pipefail
# shellcheck source=tools/ref/ref-paths.sh
source "$(dirname "${BASH_SOURCE[0]}")/../ref-paths.sh"
# shellcheck source=tools/ref/clefvis/profile.sh
source "$(dirname "${BASH_SOURCE[0]}")/profile.sh"
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
OUT=${CLEFVIS_OUT:-$DATA/bin}
SRC=$HERE/tools/ref/clefvis/dump_mtmd.cpp
LCPP=$ORACLE
git_lcpp() { git -c safe.directory="$LCPP" -C "$LCPP" "$@"; }
if [ -n "$(git_lcpp status --porcelain --untracked-files=no)" ]; then
  echo "build-clefvis: $LCPP has tracked changes; the oracle is the commit's build, not an edited tree" >&2
  exit 2
fi
COMMIT=$(git_lcpp rev-parse --short=9 HEAD)
LIB=$LCPP/build/bin
libs=(libmtmd.so libllama.so libggml.so libggml-base.so)
[ "$FAMILY" != qvis ] || libs+=(libllama-common.so)
for l in "${libs[@]}"; do
  [ -e "$LIB/$l" ] || { echo "build-clefvis: no $LIB/$l (the mainline build is the depth runners'; it must carry mtmd)" >&2; exit 2; }
done
mkdir -p "$OUT"
NAME=$(basename "$BIN")
TMP=$OUT/$NAME.tmp.$$
trap 'rm -f "$TMP" "$TMP.build"' EXIT
SRC_SHA=$(sha256sum "$SRC" | cut -d' ' -f1)
H_SHA=$(cat "$LCPP/tools/mtmd/mtmd.h" "$LCPP/tools/mtmd/mtmd-helper.h" | sha256sum | cut -d' ' -f1)
defs=()
grep -q 'struct mtmd_helper_embd_batch' "$LCPP/tools/mtmd/mtmd-helper.h" && defs+=(-DDUMP_MTMD_EMBD_BATCH)
incs=(-I"$LCPP/include" -I"$LCPP/ggml/include" -I"$LCPP/tools/mtmd")
links=(-lmtmd -lllama -lggml -lggml-base)
if [ "$FAMILY" = qvis ]; then
  defs+=(-DDUMP_MTMD_CHAT)
  incs+=(-I"$LCPP/common" -I"$LCPP/vendor")
  links+=(-lllama-common)
fi
g++ -std=c++17 -O2 -Wall -Wextra -Wno-unused-parameter -Wno-unused-function "${defs[@]}" -o "$TMP" "$SRC" "${incs[@]}" -L"$LIB" "${links[@]}" -Wl,-rpath,"$LIB"
printf 'source_sha256 %s\nlcpp %s\nlcpp_commit %s\nmtmd_headers_sha256 %s\n' "$SRC_SHA" "$(readlink -f "$LCPP")" "$COMMIT" "$H_SHA" > "$TMP.build"
mv -f "$TMP" "$OUT/$NAME"
mv -f "$TMP.build" "$OUT/$NAME.build"
echo "built $OUT/$NAME from dump_mtmd.cpp sha256 ${SRC_SHA:0:12} against $LCPP at $COMMIT (${defs[*]:-no defines})"
