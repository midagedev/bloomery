#!/usr/bin/env bash
# Build the image-input oracle on the box: dump_mtmd (tools/ref/clefvis/dump_mtmd.cpp) linked against llama.cpp mainline's libmtmd,
# libllama and libggml in the profile's ORACLE tree (tools/ref/clefvis/profile.sh): mainline has `qwen3vl_merger` natively, so no
# fork. Clef-Flash's seat (BLOOMERY_MODEL=qwen35) links the tree hidden_ref links (53ed051ce) into $BLOOMERY_DATA/bin/dump_mtmd;
# the Qwen seats (qwen35moe, qwen4exp) one binary between them, from the tree beside it (36a73916e, tools/ref/qvis/build-lcpp-qvis.sh)
# into dump_mtmd_qvis, with the chat mode (DUMP_MTMD_CHAT: it links libllama-common for the model's chat template and parser);
# MiniCPM-V 4.7's seat (minicpmv47: the tower, hidden and chat arms, the chat mode linked as the Qwen seats' is) from mainline
# 358a420e (`just build-mc47-lcpp`, the tree with `minicpmv4_7`) into dump_mtmd_mc47. EmbeddingGemma 2's seat (gemma-embedding2)
# links the profile's DUMPER, tools/ref/emb2/dump_emb2.cpp, against mainline 6868a52cb (`just build-emb2-lcpp`) into dump_emb2. The
# post-decode callback of the helper changed with that tree's header (mtmd_helper_embd_batch): the build defines
# DUMP_MTMD_EMBD_BATCH when the header has it and the source reads it; the rendered prompt of the chat template became parts
# (common_chat_prompt) in the MiMo tree (cd0a5b6a1): DUMP_MTMD_CHAT_PROMPT_PARTS.
#
# The tree is checked, not trusted: no tracked change (a hand edit there would reach the oracle under the commit's name). Its build
# is the one the depth runners time; this links against it and builds nothing in it.
# Output: the binary and a `.build` record beside it, naming the sha256 of the dumper source it was built from and of the
# set_writer.h it includes, the mainline commit, the sha256 of the mtmd.h and mtmd-helper.h it saw, and what the tree's build
# was configured with (its CMakeCache.txt): the CUDA architectures, nvcc and its version, and the command-line options;
# clefvis.sh and emb2.sh refuse a binary whose record does not match their own tree.
# The compile writes a temporary name and only a finished link replaces the installed binary (CLEFVIS_OUT moves the directory).
set -euo pipefail
# shellcheck source=tools/ref/ref-paths.sh
source "$(dirname "${BASH_SOURCE[0]}")/../ref-paths.sh"
# shellcheck source=tools/ref/clefvis/profile.sh
source "$(dirname "${BASH_SOURCE[0]}")/profile.sh"
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
OUT=${CLEFVIS_OUT:-$DATA/bin}
SRC=$HERE/$DUMPER
HDR=$HERE/tools/ref/clefvis/set_writer.h
LCPP=$ORACLE
git_lcpp() { git -c safe.directory="$LCPP" -C "$LCPP" "$@"; }
dirty=$(git_lcpp status --porcelain --untracked-files=no) ||
  { echo "build-clefvis: cannot inspect tracked changes in $LCPP" >&2; exit 2; }
if [ -n "$dirty" ]; then
  echo "build-clefvis: $LCPP has tracked changes; the oracle is the commit's build, not an edited tree" >&2
  exit 2
fi
COMMIT=$(git_lcpp rev-parse --short=9 HEAD)
LIB=$LCPP/build/bin
CACHE=$LCPP/build/CMakeCache.txt
[ -f "$CACHE" ] || { echo "build-clefvis: no $CACHE (the mainline build is the depth runners')" >&2; exit 2; }
cache_value() { sed -n "s/^$1:[A-Z]*=//p" "$CACHE" | head -n 1; }
CUDA_ARCH=$(cache_value CMAKE_CUDA_ARCHITECTURES)
NVCC=$(cache_value CMAKE_CUDA_COMPILER)
NVCC_VERSION=-
if [ -n "$NVCC" ]; then
  NVCC_VERSION=$("$NVCC" --version 2>&1 | awk '/release/ && !seen { print; seen = 1 }') && [ -n "$NVCC_VERSION" ] ||
    { echo "build-clefvis: $NVCC (CMAKE_CUDA_COMPILER in $CACHE) does not print its release" >&2; exit 2; }
fi
libs=(libmtmd.so libllama.so libggml.so libggml-base.so)
case $FAMILY in qvis | mcvis) libs+=(libllama-common.so) ;; esac
for l in "${libs[@]}"; do
  [ -e "$LIB/$l" ] || { echo "build-clefvis: no $LIB/$l (the mainline build is the depth runners'; it must carry mtmd)" >&2; exit 2; }
done
mkdir -p "$OUT"
NAME=$(basename "$BIN")
TMP=$OUT/$NAME.tmp.$$
trap 'rm -f "$TMP" "$TMP.build"' EXIT
SRC_SHA=$(sha256sum "$SRC" | cut -d' ' -f1)
HDR_SHA=$(sha256sum "$HDR" | cut -d' ' -f1)
H_SHA=$(cat "$LCPP/tools/mtmd/mtmd.h" "$LCPP/tools/mtmd/mtmd-helper.h" | sha256sum | cut -d' ' -f1)
defs=()
if grep -q DUMP_MTMD_EMBD_BATCH "$SRC" && grep -q 'struct mtmd_helper_embd_batch' "$LCPP/tools/mtmd/mtmd-helper.h"; then
  defs+=(-DDUMP_MTMD_EMBD_BATCH)
fi
if grep -q '^struct common_chat_prompt {' "$LCPP/common/chat.h" 2> /dev/null; then
  defs+=(-DDUMP_MTMD_CHAT_PROMPT_PARTS)
fi
incs=(-I"$LCPP/include" -I"$LCPP/ggml/include" -I"$LCPP/tools/mtmd")
links=(-lmtmd -lllama -lggml -lggml-base)
if [ "$FAMILY" = qvis ] || [ "$FAMILY" = mcvis ]; then
  defs+=(-DDUMP_MTMD_CHAT)
  incs+=(-I"$LCPP/common" -I"$LCPP/vendor")
  links+=(-lllama-common)
fi
g++ -std=c++17 -O2 -Wall -Wextra -Wno-unused-parameter -Wno-unused-function "${defs[@]}" -o "$TMP" "$SRC" "${incs[@]}" -L"$LIB" "${links[@]}" -Wl,-rpath,"$LIB"
{
  printf 'source_sha256 %s\nset_writer_sha256 %s\nlcpp %s\nlcpp_commit %s\nmtmd_headers_sha256 %s\n' "$SRC_SHA" "$HDR_SHA" \
    "$(readlink -f "$LCPP")" "$COMMIT" "$H_SHA"
  printf 'cuda_architectures %s\ncuda_compiler %s\ncuda_compiler_version %s\n' "${CUDA_ARCH:--}" "${NVCC:--}" "$NVCC_VERSION"
  grep -E ':UNINITIALIZED=|^(GGML_CUDA|GGML_NATIVE|CMAKE_BUILD_TYPE):' "$CACHE" | sed 's/^/cmake /'
} > "$TMP.build"
mv -f "$TMP" "$OUT/$NAME"
mv -f "$TMP.build" "$OUT/$NAME.build"
echo "built $OUT/$NAME from $(basename "$SRC") sha256 ${SRC_SHA:0:12} against $LCPP at $COMMIT (${defs[*]:-no defines})"
