#!/usr/bin/env bash
# Put the lfm3 profile's three GGUF files (tools/ref/models/lfm3.sh LOGITS_SETS) where the profile names them, at the revision it
# pins, then write the `REVISION` file beside them that tools/ref/dump-lfm3.sh reads. A file of the table's byte count and sha256
# stays; any other is downloaded to <file>.part from https://huggingface.co/$LFM3_REPO/resolve/$LFM3_REVISION/<name>, checked
# against the table (a mismatch removes the .part and ends the run, 65) and moved into place. REVISION is removed first and
# written last, so a directory with one is a directory whose three files matched (the profile keeps all three in one). The files go straight to their final directory: nothing is moved
# afterwards, and a reference set names the file's full path. No lease, no card; the network and the drive are the resources.
# LFM3_BASE_URL moves the download base (a file:// directory for the self-test, tools/check-lfm3-oracle.sh).
set -euo pipefail
# shellcheck source=tools/ref/ref-paths.sh
source "$(dirname "${BASH_SOURCE[0]}")/ref-paths.sh"
[ "$MODEL_NAME" = lfm3 ] || {
  echo "fetch-lfm3: this fetch runs under the lfm3 profile; this command picked $MODEL_NAME" >&2
  exit 2
}
BASE=${LFM3_BASE_URL:-https://huggingface.co/$LFM3_REPO/resolve/$LFM3_REVISION}
digest() { # <file>: its sha256, or the failure of sha256sum
  local sum
  sum=$(sha256sum < "$1") || return
  printf '%s\n' "${sum%% *}"
}
dir=
for entry in "${LOGITS_SETS[@]}"; do
  IFS=: read -r set_name file bytes sha <<< "$entry"
  dir=${file%/*}
  mkdir -p "$dir"
  rm -f "$dir/REVISION"
  if [ -f "$file" ] && [ "$(wc -c < "$file" | tr -d ' ')" = "$bytes" ] && [ "$(digest "$file")" = "$sha" ]; then
    echo "fetch-lfm3: $file is the pinned file already ($set_name)"
    continue
  fi
  part=$file.part
  rm -f "$part"
  echo "fetch-lfm3: $BASE/${file##*/} -> $part"
  curl -fL --retry 3 --max-time 600 -o "$part" "$BASE/${file##*/}"
  got_bytes=$(wc -c < "$part" | tr -d ' ')
  got_sha=$(digest "$part")
  if [ "$got_bytes" != "$bytes" ] || [ "$got_sha" != "$sha" ]; then
    rm -f "$part"
    echo "fetch-lfm3: ${file##*/} is $got_bytes bytes, sha256 $got_sha; the profile's table has $bytes bytes, $sha (exit 65)" >&2
    exit 65
  fi
  mv -f "$part" "$file"
  echo "fetch-lfm3: $file matches the table ($bytes bytes)"
done
printf '%s\n' "$LFM3_REVISION" > "$dir/REVISION"
echo "fetch-lfm3: $dir/REVISION is $LFM3_REVISION"
