#!/usr/bin/env bash
# Build the MTP step oracle on the box: q35mtp_ref (tools/ref/q35mtp_ref.cpp) linked against llama.cpp mainline's
# libllama and libggml in the qwen35moe profile's MTP_LCPP tree (tools/ref/models/qwen35moe.sh), not the ik tree the
# node dumpers link. Under the qwen35moe profile only.
#
# The tree is checked, not trusted: no tracked change (a hand edit there would reach the oracle under the commit's
# name). Its build is the one that exists; this links against it and builds nothing in it. q35mtp_ref includes the
# tree's staging header src/llama-ext.h (the nextn embeddings), so it is built from that tree's sources at that commit.
# Output: $BLOOMERY_DATA/bin/q35mtp_ref (Q35MTP_OUT moves it) and q35mtp_ref.build beside it, naming the sha256 of the
# q35mtp_ref.cpp it was built from, the mainline commit and the md5 of the libllama it links; dump-q35mtp.sh refuses a
# binary whose record does not match its own tree. The compile writes a temporary name and only a finished link
# replaces the installed binary.
set -euo pipefail
# shellcheck source=tools/ref/ref-paths.sh
source "$(dirname "${BASH_SOURCE[0]}")/ref-paths.sh"
[ "$MODEL_NAME" = qwen35moe ] ||
  { echo "build-q35mtp-ref: q35mtp_ref is built under the qwen35moe profile; this command picked $MODEL_NAME" >&2; exit 2; }
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
OUT=${Q35MTP_OUT:-$BLOOMERY_DATA/bin}
SRC=$HERE/tools/ref/q35mtp_ref.cpp
L=$MTP_LCPP
git_lcpp() { git -c safe.directory="$L" -C "$L" "$@"; }
if [ -n "$(git_lcpp status --porcelain --untracked-files=no)" ]; then
  echo "build-q35mtp-ref: $L has tracked changes; the oracle is the commit's build, not an edited tree" >&2
  exit 2
fi
COMMIT=$(git_lcpp rev-parse --short=9 HEAD)
LIB=$L/build/bin
[ -f "$LIB/libllama.so.0" ] || { echo "build-q35mtp-ref: no $LIB/libllama.so.0: $L is not built" >&2; exit 66; }
[ -f "$LIB/libggml-cuda.so" ] || { echo "build-q35mtp-ref: no $LIB/libggml-cuda.so: $L has no CUDA backend" >&2; exit 66; }
LIB_MD5=$(md5sum < "$LIB/libllama.so.0" | cut -d' ' -f1)
mkdir -p "$OUT"
TMP=$OUT/q35mtp_ref.tmp.$$
trap 'rm -f "$TMP" "$TMP.build"' EXIT
SRC_SHA=$(sha256sum "$SRC" | cut -d' ' -f1)
g++ -std=c++17 -O2 -o "$TMP" "$SRC" -I"$L/include" -I"$L/ggml/include" -I"$L/src" \
  -L"$LIB" -lllama -lggml -lggml-base -Wl,-rpath,"$LIB"
printf 'source_sha256 %s\nlcpp %s\nlcpp_commit %s\nlibllama_md5 %s\n' "$SRC_SHA" "$(readlink -f "$L")" "$COMMIT" "$LIB_MD5" > "$TMP.build"
mv -f "$TMP" "$OUT/q35mtp_ref"
mv -f "$TMP.build" "$OUT/q35mtp_ref.build"
echo "built $OUT/q35mtp_ref from q35mtp_ref.cpp sha256 ${SRC_SHA:0:12} against $L at $COMMIT"
