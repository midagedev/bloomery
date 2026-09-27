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
#        tools/ptx-spill-check.sh --self-test   (a stub ptx-scan.sh on fixed scans, no build; check-recipes runs it)
#   Runs `tools/ptx-scan.sh <binary>` for each binary (the recipe builds them first) and compares
#   its entry rows with the table's rows for that binary. The table (tools/ref/ptx-shapes.tsv) holds
#   `<binary> <entry> <spill> <jit_local>`, whitespace-separated; `#` starts a comment. A pinned
#   nonzero value carries a `# PIN(YYYY-MM-DD): <reason>` comment on the line above its row.
#   The table is read once, into a copy every binary reads: a table given as a pipe or a process
#   substitution (`<(grep … tools/ref/ptx-shapes.tsv)`) is empty on a second read.
#
# Every violation prints one line, and the script runs every binary before it decides:
#   <bin> <entry> spill=<got> pinned=<want> ABOVE|BELOW   (either column; BELOW = lower the pin)
#   <bin> <entry> unpinned spill=<got> jit_local=<got>     (in the scan, not in the table)
#   <bin> <entry> stale                                    (in the table, not in the scan)
# Exit status: 0 when every binary's scan read and matched its pins; 1 on a violation or a scan
# that failed (ptx-scan's own nonzero exit, or a table it printed without rows); 2 on a usage error.
set -uo pipefail

self_test() {
  local t fails=0 out rc
  t=$(mktemp -d)
  cp "${BASH_SOURCE[0]}" "$t/ptx-spill-check.sh"
  # The stub scan prints the fixture scan of its binary, or fails as ptx-scan does on a missing binary.
  printf '%s\n' '#!/usr/bin/env bash' \
    '[ -f "${BASH_SOURCE[0]%/*}/$1.scan" ] || { echo "ptx-scan bin=target/release/$1 missing scan=failed"; exit 1; }' \
    'cat "${BASH_SOURCE[0]%/*}/$1.scan"' > "$t/ptx-scan.sh"
  scan() { # scan <bin> <entry spill jit_local>...
    local b=$1 r
    shift
    { echo "ptx-scan bin=target/release/$b modules=1"; echo "entry regs spill jit_local"
      for r in "$@"; do set -- $r; echo "$1 32 $2 $3"; done
      echo "ptx-scan-md5: method=m"; echo "k 0123 1"; } > "$t/$b.scan"
  }
  scan a "k1 0 0" "k2 8 0"
  scan b "k3 0 0"
  scan c "k4 0 0"
  scan d "k1 16 0"
  printf '%s\n' '# fixture' 'a k1 0 0' '# PIN(2000-01-01): fixture' 'a k2 8 0' 'b k3 0 0' 'd k1 0 0' 'd k9 0 0' > "$t/table"
  check() { # check <name> <want rc> <want line or -> <args...>
    local name=$1 want=$2 line=$3
    shift 3
    out=$(bash "$t/ptx-spill-check.sh" "$@" 2>&1); rc=$?
    if [ "$rc" != "$want" ] || { [ "$line" != - ] && ! grep -qxF -- "$line" <<< "$out"; }; then
      echo "ptx-spill-check self-test: $name: rc $rc (want $want), want line '$line' in:" >&2
      sed 's/^/  /' <<< "$out" >&2
      fails=$((fails + 1))
    fi
  }
  check "a table read twice through a process substitution" 0 "ptx-spill bin=b entries=1 pinned=1 nonzero=none violations=0 PASS" \
    <(cat "$t/table") a b
  check "a binary with no pinned row reads unpinned, not stale" 1 "c k4 unpinned spill=0 jit_local=0" "$t/table" a c
  check "a spill above its pin" 1 "d k1 spill=16 pinned=0 ABOVE" "$t/table" d
  check "a pinned row the scan lacks" 1 "d k9 stale" "$t/table" d
  check "a failed scan" 1 "ptx-spill bin=e scan rc=1 FAIL" "$t/table" e
  check "no binary" 2 - "$t/table"
  rm -rf "$t"
  [ "$fails" = 0 ] && echo "ptx-spill-check: self-test ok" || { echo "ptx-spill-check: self-test $fails failed" >&2; return 1; }
}
if [ "${1:-}" = --self-test ]; then
  self_test
  exit $?
fi

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
cat -- "$TABLE" > "$TMP/table" || { echo "ptx-spill-check: cannot read the table $TABLE" >&2; exit 2; }
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
  awk -v b="$BIN" '!/^[[:space:]]*#/ && NF >= 4 && $1 == b { print $2, $3, $4 }' "$TMP/table" > "$TMP/$BIN.want"
  if [ ! -s "$TMP/$BIN.got" ]; then
    echo "ptx-spill bin=$BIN scan printed no entry rows FAIL"
    rc=1
    continue
  fi
  # The pins by file name, not NR == FNR: a binary with no pinned row has an empty first file, and
  # NR == FNR would then read its scan as the pins and print every entry as stale.
  out=$(awk -v b="$BIN" '
    FILENAME == ARGV[1] { ws[$1] = $2; wj[$1] = $3; next }
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
