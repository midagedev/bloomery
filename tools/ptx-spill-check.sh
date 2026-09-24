#!/usr/bin/env bash
# PTX spill ratchet — the gate over ptx-scan's `spill` (ptxas spill stores, bytes) and `jit_local`
# (the driver JIT's local bytes per thread) columns. ptx-scan is an instrument and asserts nothing;
# this script reads its table for each named binary and holds every entry to the value pinned in the
# table file. Both columns are fixed at build time, so this is a compile-time ratchet: no timing.
#
# The failure it blocks: a kernel that starts spilling to local memory passes every bit gate — the
# spill changes speed, not bits — and nothing read those columns (an 8-byte ptxas spill in
# ds41_attn_seg_sel was found by eye). A new entry is a failure until it is pinned, so a new kernel
# cannot arrive spilling unseen.
#
# Usage: tools/ptx-spill-check.sh <table> <binary name>...
#   Runs `tools/ptx-scan.sh <binary>` for each binary (the recipe builds them first) and compares
#   its entry rows with the table's rows for that binary. The table (tools/ref/ptx-shapes.tsv) holds
#   `<binary> <entry> <spill> <jit_local>`, whitespace-separated; `#` starts a comment. A pinned
#   nonzero value carries a `# PIN(YYYY-MM-DD): <reason>` comment on the line above its row.
#
# Every violation prints one line, and the script runs every binary before it decides:
#   <bin> <entry> spill=<got> pinned=<want> ABOVE|BELOW   (either column; BELOW = lower the pin)
#   <bin> <entry> unpinned spill=<got> jit_local=<got>     (in the scan, not in the table)
#   <bin> <entry> stale                                    (in the table, not in the scan)
# Exit status: 0 when every binary's scan read and matched its pins; 1 on a violation or a scan
# that failed (ptx-scan's own nonzero exit, or a table it printed without rows); 2 on a usage error.
set -uo pipefail
TABLE=${1:-}
if [ -z "$TABLE" ] || [ $# -lt 2 ]; then
  echo "usage: ptx-spill-check.sh <table> <binary name>..." >&2
  exit 2
fi
shift
[ -r "$TABLE" ] || { echo "ptx-spill-check: no table $TABLE" >&2; exit 2; }
SCAN="${BASH_SOURCE[0]%/*}/ptx-scan.sh"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
rc=0
for BIN in "$@"; do
  bash "$SCAN" "$BIN" > "$TMP/$BIN.scan" 2> "$TMP/$BIN.err"
  src=$?
  banner=$(grep -m1 '^ptx-scan bin=' "$TMP/$BIN.scan")
  echo "${banner:-ptx-scan bin=target/release/$BIN (no banner)}"
  if [ "$src" -ne 0 ]; then
    echo "ptx-spill bin=$BIN scan rc=$src FAIL"
    sed 's/^/  ptx-scan stderr: /' "$TMP/$BIN.err" | head -5
    rc=1
    continue
  fi
  # The table rows: between the header line (`entry … jit_local`) and the md5 section.
  awk '
    /^entry[[:space:]]/ { h = 1; for (i = 1; i <= NF; i++) { if ($i == "spill") s = i; if ($i == "jit_local") j = i }; next }
    /^ptx-scan-md5:/ { exit }
    h && NF > 1 { print $1, $s, $j }
  ' "$TMP/$BIN.scan" > "$TMP/$BIN.got"
  awk -v b="$BIN" '!/^[[:space:]]*#/ && NF >= 4 && $1 == b { print $2, $3, $4 }' "$TABLE" > "$TMP/$BIN.want"
  if [ ! -s "$TMP/$BIN.got" ]; then
    echo "ptx-spill bin=$BIN scan printed no entry rows FAIL"
    rc=1
    continue
  fi
  out=$(awk -v b="$BIN" '
    NR == FNR { ws[$1] = $2; wj[$1] = $3; next }
    {
      seen[$1] = 1
      if (!($1 in ws)) { printf "%s %s unpinned spill=%s jit_local=%s\n", b, $1, $2, $3; next }
      if ($2 != ws[$1]) printf "%s %s spill=%s pinned=%s %s\n", b, $1, $2, ws[$1], ($2 + 0 > ws[$1] + 0 ? "ABOVE" : "BELOW")
      if ($3 != wj[$1]) printf "%s %s jit_local=%s pinned=%s %s\n", b, $1, $3, wj[$1], ($3 + 0 > wj[$1] + 0 ? "ABOVE" : "BELOW")
    }
    END { for (e in ws) if (!(e in seen)) printf "%s %s stale\n", b, e }
  ' "$TMP/$BIN.want" "$TMP/$BIN.got" | sort)
  n=$(wc -l < "$TMP/$BIN.got" | tr -d ' ')
  p=$(wc -l < "$TMP/$BIN.want" | tr -d ' ')
  nz=$(awk '$2 != 0 || $3 != 0 { printf "%s%s(spill=%s,jit_local=%s)", c, $1, $2, $3; c = "," }' "$TMP/$BIN.got")
  if [ -n "$out" ]; then
    echo "$out"
    v=$(printf '%s\n' "$out" | wc -l | tr -d ' ')
    echo "ptx-spill bin=$BIN entries=$n pinned=$p nonzero=${nz:-none} violations=$v FAIL"
    rc=1
  else
    echo "ptx-spill bin=$BIN entries=$n pinned=$p nonzero=${nz:-none} violations=0 PASS"
  fi
done
exit "$rc"
