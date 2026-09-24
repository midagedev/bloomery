#!/usr/bin/env bash
# Build the stage-1 oracle instrument on the box. Links libllama + libcommon (loads the model
# through llama.h); the flags come from ref-build-common.sh. Output: DUMP_OUT or $BLOOMERY_DATA/bin.
#
# The compile writes a temporary name and only a finished link is moved over dump_ref, so a failed
# build exits non-zero and leaves the installed binary and its record as they were. The record,
# dump_ref.build beside the binary, names the sha256 of the dump_ref.cpp it was built from and the
# ik tree it links; dump.sh refuses a binary whose record does not match its own tree's source.
set -euo pipefail
# shellcheck source=tools/ref/ref-build-common.sh
source "$(dirname "${BASH_SOURCE[0]}")/ref-build-common.sh"
OUT=${DUMP_OUT:-$REF_BIN}
SRC=$HERE/tools/ref/dump_ref.cpp
mkdir -p "$OUT"
TMP=$OUT/dump_ref.tmp.$$
trap 'rm -f "$TMP" "$TMP.build"' EXIT
SRC_SHA=$(sha256sum "$SRC" | cut -d' ' -f1)
ref_cxx -o "$TMP" "$SRC" "${REF_LLAMA_INC[@]}" "${REF_LLAMA_LINK[@]}"
printf 'source_sha256 %s\nik %s\n' "$SRC_SHA" "$(readlink -f "$IK")" > "$TMP.build"
mv -f "$TMP" "$OUT/dump_ref"
mv -f "$TMP.build" "$OUT/dump_ref.build"
echo "built $OUT/dump_ref from dump_ref.cpp sha256 ${SRC_SHA:0:12}"
