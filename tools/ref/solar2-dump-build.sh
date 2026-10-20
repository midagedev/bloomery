#!/usr/bin/env bash
# Link solar2_dump (tools/ref/solar2-dump.cpp, with tools/ref/ref_io.h) against the existing PAI build of the solar-open2 profile
# ($PAI/build/bin: llama.cpp 6ea215d17 plus prometheusAIR's solar_open2 patch). This does not configure or rebuild that
# tree: a rebuild moves the libllama md5 the oracle sets name. Output: $BLOOMERY_DATA/bin/solar2_dump and solar2_dump.build
# beside it, the record tools/ref/solar2-dump.sh compares: the source sha256 (cpp then header), the tree's commit, the patch's
# sha256 and the md5 of the libllama the binary loads.
set -euo pipefail
export BLOOMERY_MODEL=solar-open2
# shellcheck source=tools/ref/ref-paths.sh
source "$(dirname "${BASH_SOURCE[0]}")/ref-paths.sh"
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
OUT=${SOLAR2_OUT:-$BLOOMERY_DATA/bin}
SRC=$HERE/tools/ref/solar2-dump.cpp
IO=$HERE/tools/ref/ref_io.h
LIBDIR=$PAI/build/bin
[ -f "$LIBDIR/libllama.so" ] || { echo "solar2-dump-build: no libllama.so in $LIBDIR (the PAI tree is built there)" >&2; exit 66; }
[ -f "$PAI/solar_open2.patch" ] || { echo "solar2-dump-build: no $PAI/solar_open2.patch (the patch the tree carries)" >&2; exit 66; }
COMMIT=$(git -c safe.directory='*' -C "$PAI" rev-parse --short=9 HEAD)
PATCH_SHA=$(sha256sum "$PAI/solar_open2.patch" | cut -d' ' -f1)
LIB_MD5=$(md5sum < "$(readlink -f "$LIBDIR/libllama.so")" | cut -d' ' -f1)
mkdir -p "$OUT"
TMP=$OUT/solar2_dump.tmp.$$
trap 'rm -f "$TMP" "$TMP.build"' EXIT
SRC_SHA=$(cat "$SRC" "$IO" | sha256sum | cut -d' ' -f1)
g++ -std=c++17 -O2 -o "$TMP" "$SRC" -I"$PAI/include" -I"$PAI/ggml/include" -I"$HERE/tools/ref" \
  -L"$LIBDIR" -lllama -lggml -lggml-base -Wl,-rpath,"$LIBDIR"
printf 'source_sha256 %s\npai_commit %s\npatch_sha256 %s\nlibllama_md5 %s\n' "$SRC_SHA" "$COMMIT" "$PATCH_SHA" "$LIB_MD5" > "$TMP.build"
mv -f "$TMP" "$OUT/solar2_dump"
mv -f "$TMP.build" "$OUT/solar2_dump.build"
echo "built $OUT/solar2_dump from solar2-dump.cpp and ref_io.h sha256 ${SRC_SHA:0:12} against $PAI at $COMMIT (patch ${PATCH_SHA:0:12}, libllama md5 $LIB_MD5)"
