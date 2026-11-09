#!/usr/bin/env bash
# Copy a split set to a mirror directory and verify every shard's sha256 (docs/cards/k31-mirror-copy.card). The lead's stop-gap
# until tools/ref/k31-mirror.sh (the k31mirror round) lands; writes no sidecar. Refuses unless the destination's free bytes
# exceed the set's bytes + 100 GiB. A shard whose copy already matches by length and sha is skipped.
set -u
SRC=${1:?first shard}; DST=${2:?mirror dir}
dir=$(dirname "$SRC"); base=$(basename "$SRC"); prefix=${base%-00001-of-*.gguf}; count=${base##*-of-}; count=${count%.gguf}
[ "$prefix" != "$base" ] || { echo "k31-mirror-copy: $SRC is not a first shard"; exit 64; }
shards=(); for i in $(seq -f %05g 1 "$count"); do shards+=("$prefix-$i-of-$count.gguf"); done
bytes=0; for s in "${shards[@]}"; do [ -f "$dir/$s" ] || { echo "k31-mirror-copy: missing $dir/$s"; exit 66; }; bytes=$((bytes + $(stat -c %s "$dir/$s"))); done
mkdir -p "$DST"; free=$(df --output=avail -B1 "$DST" | tail -1)
[ "$free" -gt $((bytes + 100 * 1024 * 1024 * 1024)) ] || { echo "k31-mirror-copy: $DST has $free B free, the set is $bytes B: under the set + 100 GiB"; exit 75; }
echo "[copy] $count shards, $((bytes / 1000000000)) GB -> $DST; predicted wall $((bytes / 2000000000 / 60 + 1)) min copy + $((bytes * 2 / 1500000000 / 60 + 1)) min sha"
t0=$(date +%s); fail=0
for s in "${shards[@]}"; do
  src="$dir/$s"; dst="$DST/$s"; t=$(date +%s)
  if [ -f "$dst" ] && [ "$(stat -c %s "$dst")" = "$(stat -c %s "$src")" ] && [ -f "$dst.sha256" ] && [ "$(cut -d' ' -f1 "$dst.sha256")" = "$(cut -d' ' -f1 "$src.sha256" 2>/dev/null || sha256sum "$src" | cut -d' ' -f1)" ]; then
    echo "[copy] $s skip (length and sha match)"; continue
  fi
  nice -n 10 cp --reflink=never "$src" "$dst" || { echo "[copy] $s cp FAILED"; fail=1; break; }
  a=$(nice -n 10 sha256sum "$src" | cut -d' ' -f1); b=$(nice -n 10 sha256sum "$dst" | cut -d' ' -f1)
  if [ "$a" = "$b" ]; then echo "$b  $s" > "$dst.sha256"; echo "[copy] $s ok $(( $(date +%s) - t )) s sha=$a"; else echo "[copy] $s sha MISMATCH src=$a dst=$b"; rm -f "$dst"; fail=1; break; fi
done
echo "[copy] done fail=$fail wall=$(( $(date +%s) - t0 )) s; $(df -h "$DST" | tail -1)"
exit $fail
