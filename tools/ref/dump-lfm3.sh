#!/usr/bin/env bash
# Dump the lfm3 profile's last-position logits sets, one a file of its LOGITS_SETS (tools/ref/models/lfm3.sh: the Q4_K_M, Q8_0
# and BF16 files into ref-lfm3-logits, ref-lfm3-q8-logits and ref-lfm3-bf16-logits), each by tools/ref/dump-nemotron.sh, the
# one dump script of these sets (its header has the prompts, the context, the build record and the staging).
# This script owns what is true of the three files together. Before the first dump every file must be the table's size and
# the directory's REVISION file must name the profile's LFM3_REVISION: tools/ref/fetch-lfm3.sh writes it once every file
# matches the table's bytes and sha256, so a file of another revision or a download that stopped half-way is no input.
# A caller's own BLOOMERY_REF_MODEL is ignored: each dump names its file. A dump that fails ends the run (its status is
# this script's), the sets before it stay.
# CPU only, no lease, no card (dump-nemotron.sh hides CUDA).
set -euo pipefail
# shellcheck source=tools/ref/ref-paths.sh
source "$(dirname "${BASH_SOURCE[0]}")/ref-paths.sh"
[ "$MODEL_NAME" = lfm3 ] || {
  echo "dump-lfm3: this dump runs under the lfm3 profile; this command picked $MODEL_NAME" >&2
  exit 2
}
DUMP=$(dirname "${BASH_SOURCE[0]}")/dump-nemotron.sh
for entry in "${LOGITS_SETS[@]}"; do
  IFS=: read -r set_name file bytes _ <<< "$entry"
  [ -f "$file" ] || {
    echo "dump-lfm3: no $file for $set_name — run just fetch-lfm3 (exit 66)" >&2
    exit 66
  }
  got=$(wc -c < "$file" | tr -d ' ')
  [ "$got" = "$bytes" ] || {
    echo "dump-lfm3: $file is $got bytes, the profile's table has $bytes — run just fetch-lfm3 (exit 66)" >&2
    exit 66
  }
  revision=$(cat "${file%/*}/REVISION" 2> /dev/null || true)
  [ "$revision" = "$LFM3_REVISION" ] || {
    echo "dump-lfm3: ${file%/*}/REVISION names '${revision:-nothing}', the profile pins $LFM3_REVISION — run just fetch-lfm3 (exit 66)" >&2
    exit 66
  }
done
for entry in "${LOGITS_SETS[@]}"; do
  IFS=: read -r set_name file _ <<< "$entry"
  BLOOMERY_MODEL=lfm3 BLOOMERY_REF_MODEL=$file bash "$DUMP" || {
    rc=$?
    echo "dump-lfm3: $set_name from $file failed (exit $rc)" >&2
    exit "$rc"
  }
done
echo "dump-lfm3: installed ${#LOGITS_SETS[@]} sets"
