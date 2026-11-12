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
# A scan that skipped the driver JIT (`ptx-scan.sh --no-jit`: banner `jit=skipped(no-jit)`, the
# jit_local cells `skipped(no-jit)`) holds no jit_local reading, and this ratchet pins that column: it is
# refused by name, never read as a value. The script runs the scan itself without the flag, so this
# guards a caller that hands it another scan.
#
# The JIT cache. The driver JIT's reading of an entry (jit_regs, jit_local) is a function of the entry's PTX, the ptxas reading of
# it, the module's directives, the card's compute capability and the driver, so a row of the cache holds it under that key:
# the entry's digest (ptx-scan's md5 block: method, md5, lines), regs, smem, spill and reqntid, the scan's arch and ptxas
# version, a checksum of the module directives (`.version`, `.target`, `.address_size`), and the card's capability, the driver
# version and a checksum of its JIT compiler (libnvidia-ptxjitcompiler.so.1). The driver and capability come from nvidia-smi,
# which every card must agree on (the JIT's card is not known before the JIT), and ldconfig. Per binary:
#   every entry's key cached  -> the script reads `ptx-scan.sh --no-jit` (no card, no gate lock, no JIT) and takes the JIT
#                                columns from the cache; the banner names `jit=cached jit-cc=<cc> jit-driver=<id>`
#   an entry not cached       -> the binary is scanned in full, as without a cache, and its rows are added: the driver JITs a
#                                binary's modules whole, so one moved entry costs its binary's JIT, and the cached entries of
#                                that binary are read again and held to their rows (a row the JIT contradicts is a violation)
#   no key to be had          -> a scan with `calls=` in its banner (an entry that calls a device function has the callee's
#                                code outside its digest), no module kept, a column or md5 row missing: the binary is scanned in
#                                full and adds no row; the line below names why
# The cache file is PTX_SPILL_CACHE, else $XDG_CACHE_HOME or ~/.cache, then bloomery/ptx-spill-jit.tsv (not under $BLOOMERY_DATA:
# the ledger key digests that directory, and a file every run rewrites would move it): a header line, then `key jit_regs
# jit_local entry checksum` rows, tab-separated, replaced whole by a rename. A row that does not read (another shape, a cell that is no number, a
# checksum that does not match) is named with its line and not used; a row of another driver, JIT compiler or capability is
# counted and dropped at the next write; a file whose first line is not the header is never read or written. No cache path, a
# machine with no driver query or cards of two capabilities: no cache, every scan in full, the cause named.
#   PTX_SPILL_CACHE=<file>   the cache file; `off` uses none
#   PTX_SPILL_FULL=1         every binary scanned in full, the cache audited against it and refreshed (the cold run)
# Two lines name the cache: before the binaries `ptx-spill cache=<file> rows=<n> driver=<id>` (or `cache=off(<why>)`), and for each
# binary, before its verdict, `ptx-spill bin=<bin> jit entries=<n> jitted=<n> cached=<n> [missed=<n>] cache=<file>|off(<why>)`:
# jitted is the entries whose columns this run read from the driver (all of a binary that took the JIT), cached those read from
# the cache, missed the entries whose key was not cached (up to twelve are named on a line `ptx-spill bin=<bin> jit missed=<a>,<b>`
# when the cache held rows to lack them from). A cached row the JIT contradicts prints
#   <bin> <entry> cached jit_regs=<a> jit_local=<b>, the JIT read jit_regs=<c> jit_local=<d> CACHE-CONFLICT
# and the JIT's reading replaces the row. A cache that cannot be written prints `ptx-spill cache=<file> not written: <why>` and the
# verdict stands.
#
# Usage: tools/ptx-spill-check.sh <table>... <binary name>[:<entry substring>]...
#        tools/ptx-spill-check.sh --self-test   (a stub ptx-scan.sh on fixed scans, no build; check-recipes runs it)
#   Runs `tools/ptx-scan.sh <binary>` for each binary (the recipe builds them first) and compares
#   its entry rows with that binary's rows in the table its scan reads. A `<binary>:<substring>` argument scans only
#   the entries whose name holds the substring (ptx-scan's own filter) and the table's rows for the binary are
#   those entries: a binary that links a bundle another named binary already pins (the gpu bundle) pins only
#   its own device crate's entries. A table (tools/ref/ptx-shapes.tsv) holds
#   `<binary> <entry> <spill> <jit_local>`, whitespace-separated; `#` starts a comment. A pinned
#   nonzero value carries a `# PIN(YYYY-MM-DD): <reason>` comment on the line above its row.
#
#   One table per PTX arch and card capability, because `spill` follows the arch ptxas assembles for and
#   `jit_local` the capability the driver JITs for. The leading arguments that hold a `/` are the tables, and one that
#   ends in `/` is every `*.tsv` in that directory (the recipe names tools/ref/ptx-shapes.tsv and tools/ref/ptx-shapes/);
#   the rest are binaries. A table names its arch on one `# arch: <sm_NN>` line and its capability on
#   one `# jit-cc: <M.m>` line of its header. A table with no such line, with two, or with one whose value is not one
#   arch or capability, and two tables of one arch and capability, are refused. Each binary is read against the table
#   of its own scan's banner, `arch=` and `jit-cc=`; a scan with no such table fails by name. An arch no table pins
#   (BLOOMERY_ARCH, unset: tools/arch.sh's selection) ends the script before any scan, so a scan is never read against
#   another arch's pins.
#   Every table is read once, into a copy every binary reads: a table given as a pipe or a process
#   substitution (`<(grep … tools/ref/ptx-shapes.tsv)`) is empty on a second read.
#
# Every violation prints one line, and the script runs every binary before it decides:
#   <bin> <entry> spill=<got> pinned=<want> ABOVE|BELOW   (either column; BELOW = lower the pin)
#   <bin> <entry> unpinned spill=<got> jit_local=<got>     (in the scan, not in the table)
#   <bin> <entry> stale                                    (in the table, not in the scan)
#   ptx-spill bin=<bin> scan is no-jit (jit_local not read) FAIL   (a scan that skipped the JIT)
#   ptx-spill bin=<bin> no table for <arch> on cc <M.m> FAIL       (no table of the scan's arch and capability)
#   ptx-spill bin=<bin> scan rc=75 CONTENTION                      (the scan's JIT could not take a lock: no reading, no verdict)
#   ptx-spill bin=<bin> comparison=unread|entries=unread|pinned=unread FAIL   (the comparison or a count could not be read)
# Before its verdict each binary names the table it read: `ptx-spill bin=<bin> table=<path> arch=<arch> jit-cc=<M.m>`.
# Exit status: 0 when every binary's scan read and matched its pins; 1 on a violation or a scan
# that failed (ptx-scan's own nonzero exit but 75, or a table it printed without rows); 75 when at least one scan
# ended 75 (lock contention: ptx-scan's JIT step found a card lock, a batch hold or the timing lease taken) and no binary
# had a violation or a failed scan: a contended binary is no verdict and the rest read as usual, so a batch retries the
# gate instead of booking it red; 2 on a usage error
# (a table whose arch or capability lines are refused above included); 64 a BLOOMERY_ARCH that is no arch or that no
# table pins.
#
# BLOOMERY_BUILD_ONLY=1 (tools/gate-batch.sh's build lane puts it in an item's box env; tools/gate.sh and tools/gpu-gate.sh read the
# same variable). This script runs no build of its own: the recipe's cargo builds come first in its box command and are done when
# this call starts, and the scans below read what they built. So the build-only call ends here, 0, once its tables and arch read as
# in a run and every binary it names is built (a binary the recipe did not build is named, 2): no cache, no scan, no card. 0 or
# unset runs the check; any other value ends 64.
set -uo pipefail

self_test() {
  local t fails=0 out rc
  t=$(mktemp -d)
  cp "${BASH_SOURCE[0]}" "$t/ptx-spill-check.sh"
  cp "${BASH_SOURCE[0]%/*}/arch.sh" "$t/arch.sh"
  unset PTX_SPILL_FULL XDG_CACHE_HOME BLOOMERY_BUILD_ONLY
  export BLOOMERY_ARCH=sm_86
  export PTX_SPILL_CACHE=off
  # The stub scan prints the fixture scan of its binary (`--no-jit`: the `.nj` twin), or fails as ptx-scan does on a missing binary.
  # A fixture with a `.err` file also writes that to stderr, and one with a `.rc` file ends with that code (the scan of a JIT that
  # met a lock, 75). Every call is logged, and PTX_SCAN_KEEP gets the binary's `.ptx` fixture as its one module.
  printf '%s\n' '#!/usr/bin/env bash' \
    'nj=; [ "${1:-}" != --no-jit ] || { nj=.nj; shift; }' \
    'echo "${nj:-.full} $*" >> "${BASH_SOURCE[0]%/*}/calls.log"' \
    'f="${BASH_SOURCE[0]%/*}/$1${2:+.$2}$nj.scan"' \
    '[ -f "$f" ] || { echo "ptx-scan bin=target/release/$1 missing scan=failed"; exit 1; }' \
    '[ ! -f "${f%.scan}.err" ] || cat "${f%.scan}.err" >&2' \
    '[ -z "${PTX_SCAN_KEEP:-}" ] || { mkdir -p "$PTX_SCAN_KEEP"; cp "${BASH_SOURCE[0]%/*}/$1.ptx" "$PTX_SCAN_KEEP/mod1.ptx" 2> /dev/null; [ ! -f "${BASH_SOURCE[0]%/*}/$1.ptx2" ] || cp "${BASH_SOURCE[0]%/*}/$1.ptx2" "$PTX_SCAN_KEEP/mod2.ptx"; }' \
    'cat "$f"' \
    'exit "$(cat "${f%.scan}.rc" 2> /dev/null || echo 0)"' > "$t/ptx-scan.sh"
  # A scan's banner names the arch and capability it was read on: SA and SC, sm_86 and 8.6 unset.
  scan() { # scan <bin> <entry spill jit_local>...
    local b=$1 r
    shift
    { echo "ptx-scan bin=target/release/$b arch=${SA:-sm_86} modules=1 jit-card=fixture jit-cc=${SC:-8.6} jit-cuda=13.4"; echo "entry regs spill jit_local"
      for r in "$@"; do set -- $r; echo "$1 32 $2 $3"; done
      echo "ptx-scan-md5: method=m"; echo "k 0123 1"; } > "$t/$b.scan"
  }
  marked() { # marked <bin> <banner tail> <jit_local cell> <entry spill>...: a scan whose banner tail and jit_local cell are given
    local b=$1 banner=$2 cell=$3 r
    shift 3
    { echo "ptx-scan bin=target/release/$b arch=sm_86 modules=1$banner"; echo "entry regs spill jit_local"
      for r in "$@"; do set -- $r; echo "$1 32 $2 $cell"; done
      echo "ptx-scan-md5: method=m"; echo "k 0123 1"; } > "$t/$b.scan"
  }
  scan a "k1 0 0" "k2 8 0"
  scan b "k3 0 0"
  scan c "k4 0 0"
  scan d "k1 16 0"
  marked f " jit=skipped(no-jit)" "skipped(no-jit)" "k1 0"
  marked g " jit=skipped(no-jit)" 0 "k1 0"
  marked h "" "skipped(no-jit)" "k1 0"
  scan v "vis_a 0 0" "outside 0 0"
  scan v.vis_ "vis_a 0 0"
  scan v.vis_a "vis_a 0 0"
  scan v.outside "outside 0 0"
  scan v.vis_missing "vis_missing 0 0"
  # p75: a scan whose JIT met a held lock, as ptx-scan ends it: a banner with no rows, six module lines on stderr and the line that
  # names the contention (gpu-gate.sh's own last line in it) last, rc 75.
  hold='the batch hold /root/bloomery-batch.gpuhold (owner leadbatch) was still up after 4 s (CARD_BOUND 4 s) - contention, not a red gate (rc 75)'
  want75="ptx-scan: the driver JIT (oxart_jit) hit lock contention, not a failure (rc 75): gpu-gate.sh: oxart_jit: $hold"
  echo "ptx-scan bin=target/release/p75 section=.oxart bytes=1 modules=6 jit=contention scan=failed" > "$t/p75.scan"
  { for r in 1 2 3 4 5 6; do echo "ptx-scan: mod$r bundle=fixture bytes=1"; done; echo "$want75"; } > "$t/p75.err"
  echo 75 > "$t/p75.rc"
  # Scans on a cc 12.1 card: w at sm_86, x at sm_86 with only a cc 8.6 row, y on a capability no table pins, z at sm_121.
  SC=12.1 scan w "k1 0 0" "k2 8 24"
  SC=12.1 scan x "k1 0 0"
  SC=9.0 scan y "k1 0 0"
  SA=sm_121 SC=12.1 scan z "k1 0 16"
  printf '%s\n' '# fixture' '# arch: sm_86' '# jit-cc: 8.6' 'a k1 0 0' '# PIN(2000-01-01): fixture' 'a k2 8 0' 'b k3 0 0' \
    'd k1 0 0' 'd k9 0 0' 'f k1 0 0' 'g k1 0 0' 'h k1 0 0' 'v vis_a 0 0' 'v outside 0 0' 'w k1 0 0' 'w k2 8 0' 'x k1 0 0' 'y k1 0 0' 'z k1 0 0' \
    > "$t/table"
  mkdir "$t/side" "$t/empty"
  printf '%s\n' '# fixture' '# arch: sm_86' '# jit-cc: 12.1' 'w k1 0 0' '# PIN(2000-01-01): fixture' 'w k2 8 24' 'y k1 0 0' \
    > "$t/side/sm_86-cc12.1.tsv"
  printf '%s\n' '# fixture' '# arch: sm_121' '# jit-cc: 12.1' '# PIN(2000-01-01): fixture' 'z k1 0 16' > "$t/side/sm_121-cc12.1.tsv"
  tables=("$t/table" "$t/side/")
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
  # A scan that ends 75 met lock contention: no reading and no verdict, so no FAIL and rc 75; any real failure beside it is still 1.
  check "a scan that ends 75 is contention, rc 75" 75 "ptx-spill bin=p75 scan rc=75 CONTENTION" "$t/table" p75
  if grep -q 'FAIL' <<< "$out"; then
    echo "ptx-spill-check self-test: a scan that ends 75 printed a FAIL line:" >&2
    sed 's/^/  /' <<< "$out" >&2
    fails=$((fails + 1))
  fi
  check "a contended scan's own stderr line, behind its six module lines, reaches the output" 75 "  ptx-scan stderr: $want75" "$t/table" p75
  check "a contended binary then a green one: the green one reads, rc 75" 75 "ptx-spill bin=a entries=2 pinned=2 nonzero=k2(spill=8,jit_local=0) violations=0 PASS" "$t/table" p75 a
  check "a green binary then a contended one: rc 75" 75 "ptx-spill bin=p75 scan rc=75 CONTENTION" "$t/table" a p75
  check "two contended scans: rc 75" 75 "ptx-spill bin=p75 scan rc=75 CONTENTION" "$t/table" p75 p75
  check "a contended scan then a failed one: rc 1, the failure named" 1 "ptx-spill bin=e scan rc=1 FAIL" "$t/table" p75 e
  check "a failed scan then a contended one: rc 1, the contention named" 1 "ptx-spill bin=p75 scan rc=75 CONTENTION" "$t/table" e p75
  check "a contended scan then a spill above its pin: rc 1" 1 "d k1 spill=16 pinned=0 ABOVE" "$t/table" p75 d
  check "a scan that skipped the JIT (banner and cells)" 1 "ptx-spill bin=f scan is no-jit (jit_local not read) FAIL" "$t/table" f
  check "a no-jit banner over a numeric jit_local" 1 "ptx-spill bin=g scan is no-jit (jit_local not read) FAIL" "$t/table" g
  check "a skipped jit_local under a banner that ran the JIT" 1 "ptx-spill bin=h scan is no-jit (jit_local not read) FAIL" "$t/table" h
  check "a binary scanned under an entry filter pins only the filtered entries" 0 "ptx-spill bin=v entries=1 pinned=1 nonzero=none violations=0 PASS" "$t/table" v:vis_
  check "two filters of the same binary read their own pins" 0 "ptx-spill bin=v entries=1 pinned=1 nonzero=none violations=0 PASS" "$t/table" v:vis_a v:outside
  check "an unfiltered binary still checks every pin" 0 "ptx-spill bin=v entries=2 pinned=2 nonzero=none violations=0 PASS" "$t/table" v
  check "an unpinned entry inside the filter is refused" 1 "v vis_missing unpinned spill=0 jit_local=0" "$t/table" v:vis_missing
  scan v.vis_ "vis_other 0 0"
  check "a missing entry inside the filter is stale" 1 "v vis_a stale" "$t/table" v:vis_
  check "no binary" 2 - "$t/table"
  check "no table" 2 - a
  # The table: the scan's arch and capability pick it among the tables named.
  check "a cc 8.6 scan reads the cc 8.6 table" 0 "ptx-spill bin=a table=$t/table arch=sm_86 jit-cc=8.6" "${tables[@]}" a
  check "a cc 12.1 scan reads the cc 12.1 table" 0 "ptx-spill bin=w entries=2 pinned=2 nonzero=k2(spill=8,jit_local=24) violations=0 PASS" \
    "${tables[@]}" w
  check "a cc 12.1 scan is not read against a cc 8.6 row" 1 "x k1 unpinned spill=0 jit_local=0" "${tables[@]}" x
  check "a capability no table pins" 1 "ptx-spill bin=y no table for sm_86 on cc 9.0 FAIL" "${tables[@]}" y
  BLOOMERY_ARCH=sm_121 check "an sm_121 scan reads the sm_121 table" 0 \
    "ptx-spill bin=z table=$t/side/sm_121-cc12.1.tsv arch=sm_121 jit-cc=12.1" "${tables[@]}" z
  check "a directory with no table" 2 "ptx-spill-check: cannot read the table $t/empty/*.tsv" "$t/table" "$t/empty/" a
  # A table's header: one `# arch:` and one `# jit-cc:` line, each with one value of its shape.
  header() { # header <name> <header line>...: a table of a's two rows under those header lines
    local n=$1
    shift
    printf '%s\n' '# fixture' "$@" 'a k1 0 0' '# PIN(2000-01-01): fixture' 'a k2 8 0' > "$t/$n"
  }
  header noarch '# jit-cc: 8.6'
  header twoarch '# arch: sm_86' '# arch: sm_90' '# jit-cc: 8.6'
  header nocc '# arch: sm_86'
  header twocc '# arch: sm_86' '# jit-cc: 8.6' '# jit-cc: 12.1'
  header nocc-value '# arch: sm_86' '# jit-cc:'
  header twocc-values '# arch: sm_86' '# jit-cc: 12.1 8.6'
  header ccshape '# arch: sm_86' '# jit-cc: 86'
  header archshape '# arch: sm86' '# jit-cc: 8.6'
  header dup '# arch: sm_86' '# jit-cc: 8.6'
  check "a table that names no arch" 2 "ptx-spill-check: the table $t/noarch declares no \`# arch:\` line: its rows pin no arch" "$t/noarch" a
  check "a table that names two archs" 2 "ptx-spill-check: the table $t/twoarch declares 2 \`# arch:\` lines: a table pins one arch" "$t/twoarch" a
  check "a table that names no capability" 2 "ptx-spill-check: the table $t/nocc declares no \`# jit-cc:\` line: its rows pin no jit-cc" "$t/nocc" a
  check "a table that names a second capability" 2 \
    "ptx-spill-check: the table $t/twocc declares 2 \`# jit-cc:\` lines: a table pins one jit-cc" "$t/twocc" a
  check "a capability line with no value" 2 \
    "ptx-spill-check: the table $t/nocc-value declares '# jit-cc:', not one jit-cc as in '# jit-cc: 8.6'" "$t/nocc-value" a
  check "a capability line with two values" 2 \
    "ptx-spill-check: the table $t/twocc-values declares '# jit-cc: 12.1 8.6', not one jit-cc as in '# jit-cc: 8.6'" "$t/twocc-values" a
  check "a capability of another shape" 2 "ptx-spill-check: the table $t/ccshape declares '# jit-cc: 86', not one jit-cc as in '# jit-cc: 8.6'" "$t/ccshape" a
  check "an arch of another shape" 2 "ptx-spill-check: the table $t/archshape declares '# arch: sm86', not one arch as in '# arch: sm_86'" "$t/archshape" a
  check "two tables of one arch and capability" 2 \
    "ptx-spill-check: the tables $t/table and $t/dup both pin sm_86 on cc 8.6" "$t/table" "$t/dup" a
  # The arch: the scan is of the build's arch. An arch no table pins ends it before any scan.
  BLOOMERY_ARCH=sm_90 check "an arch no table pins" 64 \
    "ptx-spill-check: no ptx-shapes table for sm_90; the tables pin sm_86 on cc 8.6, sm_121 on cc 12.1, sm_86 on cc 12.1" "${tables[@]}" a
  if grep -q 'ptx-scan bin=' <<< "$out"; then
    echo "ptx-spill-check self-test: the refusal of another arch ran a scan:" >&2
    sed 's/^/  /' <<< "$out" >&2
    fails=$((fails + 1))
  fi
  BLOOMERY_ARCH=sm_86 check "the table's own arch is read against the table" 0 "ptx-spill bin=a entries=2 pinned=2 nonzero=k2(spill=8,jit_local=0) violations=0 PASS" "$t/table" a
  BLOOMERY_ARCH=sm86 check "a BLOOMERY_ARCH that is no arch" 64 "ptx-spill-check.sh: BLOOMERY_ARCH is a CUDA arch, sm_ and the capability's digits with an optional a or f (unset: auto), got 'sm86'" "$t/table" a
  # A failed read is no verdict. The md5sum and wc stand-ins of tools/ref/read-stubs.sh fail on a text that holds the word
  # fail-wc: rn's scan rows and pins both hold it (no violation, so a lost count read as PASS), rp's pins alone; a sort that
  # fails loses the comparison itself.
  mkdir "$t/stubs" "$t/nosort"
  # shellcheck disable=SC2016 # the inner shell expands them; a subshell, so the file's own check does not replace this one
  bash -c 'source "$1" && read_stubs "$2"' read-stubs "${BASH_SOURCE[0]%/*}/ref/read-stubs.sh" "$t/stubs"
  printf '#!/bin/sh\nexit 1\n' > "$t/nosort/sort"
  chmod +x "$t/nosort/sort"
  scan rn "fail-wc 0 0"
  scan rp "k1 0 0"
  printf '%s\n' '# fixture' '# arch: sm_86' '# jit-cc: 8.6' 'rn fail-wc 0 0' 'rp k1 0 0' 'rp k2 0 fail-wc' > "$t/reads"
  PATH=$t/stubs:$PATH check "an entry count that fails to read" 1 "ptx-spill bin=rn entries=unread FAIL" "$t/reads" rn
  PATH=$t/stubs:$PATH check "a pin count that fails to read" 1 "ptx-spill bin=rp pinned=unread FAIL" "$t/reads" rp
  PATH=$t/nosort:$PATH check "a comparison that fails to read" 1 "ptx-spill bin=a comparison=unread FAIL" "$t/table" a
  # The JIT cache. The stubs of the driver (nvidia-smi, ldconfig and its JIT compiler file) put it under a name the cases change.
  mkdir "$t/drv"
  printf '%s\n' '#!/bin/sh' '[ -z "${STUB_SMI_FAIL:-}" ] || exit 1' 'echo "${STUB_DRV:-615.71.09}, ${STUB_CC:-8.6}"' \
    '[ -z "${STUB_CARD2:-}" ] || echo "$STUB_CARD2"' > "$t/drv/nvidia-smi"
  printf '%s\n' '#!/bin/sh' "printf '\\tlibnvidia-ptxjitcompiler.so.1 (libc6,x86-64) => $t/jitlib\\n'" > "$t/drv/ldconfig"
  chmod +x "$t/drv/nvidia-smi" "$t/drv/ldconfig"
  echo jit-compiler-1 > "$t/jitlib"
  jscan() { # jscan <bin> <entry md5 spill jit_local>...: <bin>'s full scan, its --no-jit twin and its module, as the cache keys them
    local b=$1 v r e md sp jl at
    shift
    for v in "" .nj; do
      [ -n "$v" ] && at=" jit=skipped(no-jit)" || at=" jit-card=fixture jit-cc=${SC:-8.6} jit-cuda=13.4"
      { echo "ptx-scan bin=target/release/$b section=.oxart bytes=1 ptxas=/x/ptxas ptxas-version=13.3.73 arch=sm_86 modules=1$at${CALLS:+ calls=$CALLS}"
        echo "entry reqntid regs smem spill jit_regs jit_local"
        for r in "$@"; do
          read -r e md sp jl <<< "$r"
          if [ -n "$v" ]; then echo "$e 128 32 0 $sp skipped(no-jit) skipped(no-jit)"; else echo "$e 128 32 0 $sp 40 $jl"; fi
        done
        echo "ptx-scan-md5: method=m"
        for r in "$@"; do
          read -r e md sp jl <<< "$r"
          [ "$e" = "${NOMD:-}" ] || printf '%s %032x 12\n' "$e" "$md"
        done; } > "$t/$b$v.scan"
    done
    [ -n "${NOPTX:-}" ] || printf '%s\n' '// fixture' ".version ${PTXV:-8.7}" '.target sm_86' '.address_size 64' > "$t/$b.ptx"
  }
  assert() { # assert <name> <command...>: the command succeeds
    local name=$1
    shift
    if ! "$@"; then
      echo "ptx-spill-check self-test: $name" >&2
      fails=$((fails + 1))
    fi
  }
  ran() { grep -qxF -- "$1" "$t/calls.log"; }
  refute() { ! "$@"; }
  cached_rows() { grep -vc '^#' "$t/jit.tsv"; }
  jc() { PATH=$t/drv:$PATH PTX_SPILL_CACHE=$t/jit.tsv "$@"; }
  printf '%s\n' 'j k1 0 0' 'j k2 0 0' 'j k3 0 0' 'jc k1 0 0' 'jc k4 0 0' 'jx kx1 0 0' 'jn kn1 0 0' 'jm k1 0 0' 'jm k2 0 0' 'je ke1 0 0' \
    'jp kp1 0 0' >> "$t/table"
  JV='ptx-spill bin=j entries=3 pinned=3 nonzero=none violations=0 PASS'
  jscan j "k1 1 0 0" "k2 2 0 0" "k3 3 0 0"
  : > "$t/calls.log"
  jc check "a cold cache JITs every entry and fills the cache" 0 "ptx-spill bin=j jit entries=3 jitted=3 cached=0 missed=3 cache=$t/jit.tsv" "$t/table" j
  assert "a cold cache holds nothing to read: the binary is scanned in full once" test "$(cat "$t/calls.log")" = ".full j"
  assert "a cold run writes a row an entry" test "$(cached_rows)" = 3
  cold=$out
  : > "$t/calls.log"
  jc check "a warm cache serves every entry: no entry is JITted" 0 "ptx-spill bin=j jit entries=3 jitted=0 cached=3 cache=$t/jit.tsv" "$t/table" j
  assert "a warm run reads the --no-jit scan and runs no JIT" test "$(cat "$t/calls.log")" = ".nj j"
  assert "a warm run's verdict is a cold run's" test "$(grep -F 'ptx-spill bin=j entries=' <<< "$out")" = "$(grep -F 'ptx-spill bin=j entries=' <<< "$cold")"
  assert "a warm run's banner names the cache, not a JIT it did not run" grep -qxF "ptx-scan bin=target/release/j section=.oxart bytes=1 ptxas=/x/ptxas ptxas-version=13.3.73 arch=sm_86 modules=1 jit=cached jit-cc=8.6 jit-driver=8.6/615.71.09/$(cksum < "$t/jitlib" | tr ' ' -)" <<< "$out"
  assert "a warm run leaves the cache as it was" test "$(cached_rows)" = 3
  # An entry whose PTX moved is JITted, never served from the row of its old PTX: k2's digest moves and its JIT now reads 40 bytes.
  jscan j "k1 1 0 0" "k2 22 0 40" "k3 3 0 0"
  : > "$t/calls.log"
  jc check "a moved entry is JITted and its new reading is the verdict" 1 "j k2 jit_local=40 pinned=0 ABOVE" "$t/table" j
  assert "a moved entry takes its binary's JIT" ran ".full j"
  assert "the moved entry is named" grep -qxF "ptx-spill bin=j jit missed=k2" <<< "$out"
  assert "a moved entry reads as a miss beside two cached entries" grep -qxF "ptx-spill bin=j jit entries=3 jitted=3 cached=0 missed=1 cache=$t/jit.tsv" <<< "$out"
  assert "the moved entry's row joins the cache beside the old one" test "$(cached_rows)" = 4
  # A moved module directive (a new PTX ISA version) moves every key of the module, whatever its entries' digests.
  jscan j "k1 1 0 0" "k2 2 0 0" "k3 3 0 0"
  : > "$t/calls.log"
  jc check "the old PTX is served again from its rows" 0 "ptx-spill bin=j jit entries=3 jitted=0 cached=3 cache=$t/jit.tsv" "$t/table" j
  PTXV=9.0 jscan j "k1 1 0 0" "k2 2 0 0" "k3 3 0 0"
  jc check "a module whose .version moved is JITted whole" 0 "ptx-spill bin=j jit entries=3 jitted=3 cached=0 missed=3 cache=$t/jit.tsv" "$t/table" j
  jscan j "k1 1 0 0" "k2 2 0 0" "k3 3 0 0"
  # A driver change invalidates every row, and the next write drops the old driver's rows.
  rm -f "$t/jit.tsv"
  jc check "a cache is refilled after its file is gone" 0 "ptx-spill bin=j jit entries=3 jitted=3 cached=0 missed=3 cache=$t/jit.tsv" "$t/table" j
  STUB_DRV=616.00.01 jc check "a changed driver version invalidates every row" 0 "ptx-spill bin=j jit entries=3 jitted=3 cached=0 missed=3 cache=$t/jit.tsv" "$t/table" j
  assert "a changed driver version is named, its old rows dropped" grep -qxF "ptx-spill cache=$t/jit.tsv: 3 rows of another driver, JIT compiler or capability, dropped at the next write" <<< "$out"
  assert "the next write keeps only the current driver's rows" test "$(grep -c 616.00.01 "$t/jit.tsv"):$(grep -c 615.71.09 "$t/jit.tsv")" = 3:0
  echo jit-compiler-2 > "$t/jitlib"
  jc check "a changed JIT compiler invalidates every row" 0 "ptx-spill bin=j jit entries=3 jitted=3 cached=0 missed=3 cache=$t/jit.tsv" "$t/table" j
  # The fixture's JIT ran on cc 8.6: under a cc 12.1 key its reading is not written.
  STUB_CC=12.1 jc check "a changed capability invalidates every row" 0 \
    "ptx-spill cache=$t/jit.tsv: 3 rows of another driver, JIT compiler or capability, dropped at the next write" "$t/table" j
  assert "a JIT on another capability than the key's is not cached" grep -qxF "ptx-spill bin=j jit entries=3 jitted=3 cached=0 cache=off(the JIT ran on cc 8.6, the cache's key says cc 12.1)" <<< "$out"
  assert "a capability mismatch adds no row" test "$(cached_rows)" = 3
  # A row that does not read as it was written is refused by name and not used.
  rm -f "$t/jit.tsv"
  jc check "refill" 0 - "$t/table" j
  sed '3s/\t0\t/\t7\t/' "$t/jit.tsv" > "$t/jit.bad" && mv "$t/jit.bad" "$t/jit.tsv"
  jc check "a row whose JIT cell was edited is refused by its checksum" 0 "ptx-spill cache=$t/jit.tsv row 3 refused: its checksum does not match" "$t/table" j
  assert "an edited row is not used: its entry reads as a miss beside two cached entries" grep -qxF "ptx-spill bin=j jit entries=3 jitted=3 cached=0 missed=1 cache=$t/jit.tsv" <<< "$out"
  assert "the refused row is gone after the write" test "$(cached_rows)" = 3
  sed '2s/|[0-9a-f]\{32\}|/|ffffffffffffffffffffffffffffffff|/' "$t/jit.tsv" > "$t/jit.bad" && mv "$t/jit.bad" "$t/jit.tsv"
  jc check "a row whose digest was edited is refused by its checksum" 0 "ptx-spill cache=$t/jit.tsv row 2 refused: its checksum does not match" "$t/table" j
  echo 'short row' >> "$t/jit.tsv"
  jc check "a row of another shape is refused" 0 "ptx-spill cache=$t/jit.tsv row 5 refused: has 1 columns, not 5" "$t/table" j
  # A cached row the JIT contradicts: the JIT's reading wins, the conflict is a violation, and the row is replaced.
  rm -f "$t/jit.tsv"
  jscan jc "k1 1 0 0" "k4 4 0 0"
  jc check "refill" 0 - "$t/table" jc
  jscan jc "k1 1 0 8" "k4 44 0 0"
  jc check "a cached row the JIT contradicts is a violation" 1 "jc k1 cached jit_regs=40 jit_local=0, the JIT read jit_regs=40 jit_local=8 CACHE-CONFLICT" "$t/table" jc
  jc check "the contradicting reading replaces the row" 1 "ptx-spill bin=jc jit entries=2 jitted=0 cached=2 cache=$t/jit.tsv" "$t/table" jc
  # PTX_SPILL_FULL: every entry JITted, the cache audited and refreshed.
  jscan jc "k1 1 0 0" "k4 44 0 0"
  PTX_SPILL_FULL=1 jc check "a full run ignores the rows" 1 "jc k1 cached jit_regs=40 jit_local=8, the JIT read jit_regs=40 jit_local=0 CACHE-CONFLICT" "$t/table" jc
  PTX_SPILL_FULL=1 jc check "a full run JITs every entry" 0 "ptx-spill bin=jc jit entries=2 jitted=2 cached=0 missed=0 cache=$t/jit.tsv" "$t/table" jc
  assert "a full run says so" grep -qF "driver=8.6/615.71.09/" <<< "$out"
  # What cannot be keyed is JITted and cached for nothing.
  rm -f "$t/jit.tsv"
  jscan j "k1 1 0 0" "k2 2 0 0" "k3 3 0 0"
  jc check "refill" 0 - "$t/table" j
  CALLS=kx1:2 jscan jx "kx1 5 0 0"
  jc check "an entry that calls a device function is JITted every run" 0 "ptx-spill bin=jx jit entries=1 jitted=1 cached=0 cache=off(its entries call a device function: the callee's code is not in their digest)" "$t/table" jx
  assert "a binary that cannot be keyed adds no row" refute grep -q kx1 "$t/jit.tsv"
  NOPTX=1 jscan jn "kn1 6 0 0"
  jc check "a scan that kept no module cannot be keyed" 0 "ptx-spill bin=jn jit entries=1 jitted=1 cached=0 cache=off(the scan kept no module)" "$t/table" jn
  NOMD=k2 jscan jm "k1 7 0 0" "k2 8 0 0"
  jc check "a scan whose md5 block lacks an entry cannot be keyed" 0 "ptx-spill bin=jm jit entries=2 jitted=2 cached=0 cache=off(the md5 block has no row for k2)" "$t/table" jm
  assert "an unkeyable scan adds no row" test "$(cached_rows)" = 3
  # Entries shared by two binaries share rows, however many modules each binary carries: j has one module, jd two of one directive set.
  : > "$t/calls.log"
  jscan jd "k1 1 0 0" "k2 2 0 0" "k3 3 0 0"
  cp "$t/jd.ptx" "$t/jd.ptx2"
  printf '%s\n' 'jd k1 0 0' 'jd k2 0 0' 'jd k3 0 0' >> "$t/table"
  jc check "a binary of two modules reads the rows of a binary of one" 0 "ptx-spill bin=jd jit entries=3 jitted=0 cached=3 cache=$t/jit.tsv" "$t/table" jd
  assert "...with no JIT" test "$(cat "$t/calls.log")" = ".nj jd"
  printf '%s\n' '// fixture' '.version 9.0' '.target sm_86' '.address_size 64' > "$t/jd.ptx2"
  jc check "a second module of another directive set moves every key" 0 "ptx-spill bin=jd jit entries=3 jitted=3 cached=0 missed=3 cache=$t/jit.tsv" "$t/table" jd
  # Failures that are not the cache's stay as a full scan reports them.
  jc check "a failed --no-jit scan is a failed scan" 1 "ptx-spill bin=je scan rc=1 FAIL" "$t/table" je
  jscan jp "kp1 10 0 0"
  echo 75 > "$t/jp.rc"
  jc check "a miss whose JIT meets a lock is contention, not a verdict" 75 "ptx-spill bin=jp scan rc=75 CONTENTION" "$t/table" jp
  # What the cache cannot be read under: named, and the scans run in full.
  : > "$t/calls.log"
  STUB_CARD2="615.71.09, 12.1" jc check "cards of two capabilities leave the JIT's card unknown" 0 "ptx-spill cache=off(the cards differ in driver version or compute capability, and the JIT's card is not known before the JIT)" "$t/table" j
  assert "no cache: the scan runs in full, no --no-jit pass" test "$(cat "$t/calls.log")" = ".full j"
  STUB_SMI_FAIL=1 jc check "a machine with no driver query has no cache" 0 "ptx-spill cache=off(nvidia-smi names no card)" "$t/table" j
  echo hello > "$t/foreign"
  PTX_SPILL_CACHE=$t/foreign PATH=$t/drv:$PATH check "a file that is not a cache is never read" 0 \
    "ptx-spill cache=off($t/foreign is not a v1 cache of this script: its first line is not the cache header)" "$t/table" j
  assert "a file that is not a cache is never written" test "$(cat "$t/foreign")" = hello
  PTX_SPILL_CACHE=$t/nodir/jit.tsv PATH=$t/drv:$PATH check "a cache file that cannot be written is named, the verdict stands" 0 "$JV" "$t/table" j
  assert "an unwritable cache is named" grep -qF "ptx-spill cache=$t/nodir/jit.tsv not written: " <<< "$out"
  PTX_SPILL_CACHE='' HOME='' check "no cache path: every scan is a full scan" 0 "ptx-spill cache=off(no cache file: PTX_SPILL_CACHE, XDG_CACHE_HOME and HOME are unset)" "$t/table" j
  check "PTX_SPILL_CACHE=off: no cache" 0 "ptx-spill cache=off(PTX_SPILL_CACHE=off)" "$t/table" j
  # The default file: under XDG_CACHE_HOME, else ~/.cache, its directory made on the first write, never under BLOOMERY_DATA.
  PTX_SPILL_CACHE='' HOME=$t/home PATH=$t/drv:$PATH check "the cache's default file is under HOME" 0 \
    "ptx-spill bin=j jit entries=3 jitted=3 cached=0 missed=3 cache=$t/home/.cache/bloomery/ptx-spill-jit.tsv" "$t/table" j
  assert "its directory is made on the first write" test -s "$t/home/.cache/bloomery/ptx-spill-jit.tsv"
  PTX_SPILL_CACHE='' HOME=$t/home XDG_CACHE_HOME=$t/xdg PATH=$t/drv:$PATH check "XDG_CACHE_HOME moves it" 0 \
    "ptx-spill cache=$t/xdg/bloomery/ptx-spill-jit.tsv rows=0 driver=8.6/615.71.09/$(cksum < "$t/jitlib" | tr ' ' -)" "$t/table" j
  BLOOMERY_DATA=$t/data PTX_SPILL_CACHE='' HOME=$t/home PATH=$t/drv:$PATH check "BLOOMERY_DATA does not hold it" 0 \
    "ptx-spill cache=$t/home/.cache/bloomery/ptx-spill-jit.tsv rows=3 driver=8.6/615.71.09/$(cksum < "$t/jitlib" | tr ' ' -)" "$t/table" j
  # BLOOMERY_BUILD_ONLY (the header's last paragraph): the call ends 0 before any scan once its binaries are built. The cases run in a
  # directory whose target/release holds the built binaries a and b (and not c); the stub scan's call log shows whether a scan ran.
  mkdir -p "$t/bo/target/release"
  printf '#!/bin/sh\n' > "$t/bo/target/release/a"
  cp "$t/bo/target/release/a" "$t/bo/target/release/b"
  chmod +x "$t/bo/target/release/a" "$t/bo/target/release/b"
  bo() { # bo <name> <want rc> <want line or -> <want scans: none|some> <BLOOMERY_BUILD_ONLY value> <args...>
    local name=$1 want=$2 line=$3 scans=$4 v=$5 got=none
    shift 5
    rm -f "$t/calls.log"
    out=$(cd "$t/bo" && BLOOMERY_BUILD_ONLY=$v bash "${SPILL:-$t/ptx-spill-check.sh}" "$@" 2>&1); rc=$?
    [ ! -f "$t/calls.log" ] || got=some
    if [ "$rc" != "$want" ] || [ "$got" != "$scans" ] || { [ "$line" != - ] && ! grep -qxF -- "$line" <<< "$out"; }; then
      echo "ptx-spill-check self-test: $name: rc $rc (want $want), scans $got (want $scans), want line '$line' in:" >&2
      sed 's/^/  /' <<< "$out" >&2
      fails=$((fails + 1))
    fi
  }
  bo "build-only: the binaries are built, the call ends 0 and no scan runs" 0 \
    "ptx-spill-check.sh: BLOOMERY_BUILD_ONLY: 2 binary argument(s) built; no scan runs" none 1 "$t/table" a b:k3
  bo "build-only: a binary the recipe did not build is named first, rc 2, no scan" 2 \
    "ptx-spill-check.sh: no executable target/release/c — the recipe builds it before calling this" none 1 "$t/table" a c
  bo "build-only: a table that does not read ends as the run ends it, rc 2" 2 \
    "ptx-spill-check: the table $t/noarch declares no \`# arch:\` line: its rows pin no arch" none 1 "$t/noarch" a
  BLOOMERY_ARCH=sm_90 bo "build-only: an arch no table pins ends 64 before the binaries are read" 64 \
    "ptx-spill-check: no ptx-shapes table for sm_90; the tables pin sm_86 on cc 8.6" none 1 "$t/table" c
  bo "build-only: any other value ends 64, no scan" 64 \
    "ptx-spill-check.sh: BLOOMERY_BUILD_ONLY is 1 (build, run nothing) or 0/unset, got '2'" none 2 "$t/table" a
  bo "build-only: 0 runs the check as ever" 0 "ptx-spill bin=a entries=2 pinned=2 nonzero=k2(spill=8,jit_local=0) violations=0 PASS" some 0 "$t/table" a
  # FAIL-first: the script with the case removed reaches the stub scan under BLOOMERY_BUILD_ONLY=1.
  sed '/^case \${BLOOMERY_BUILD_ONLY:-0} in$/,/^esac$/d' "$t/ptx-spill-check.sh" > "$t/ptx-spill-nobo.sh"
  assert "build-only mutant: the sed removed the short-circuit from the copy, and the script has it" \
    test "$(grep -c '^    echo "ptx-spill-check.sh: BLOOMERY_BUILD_ONLY: \$# binary' "$t/ptx-spill-nobo.sh")$(grep -c '^    echo "ptx-spill-check.sh: BLOOMERY_BUILD_ONLY: \$# binary' "$t/ptx-spill-check.sh")" = 01
  SPILL=$t/ptx-spill-nobo.sh bo "build-only mutant: without the short-circuit the stub scan phase is reached" 0 \
    "ptx-spill bin=a entries=2 pinned=2 nonzero=k2(spill=8,jit_local=0) violations=0 PASS" some 1 "$t/table" a
  rm -rf "$t"
  [ "$fails" = 0 ] && echo "ptx-spill-check: self-test ok" || { echo "ptx-spill-check: self-test $fails failed" >&2; return 1; }
}
if [ "${1:-}" = --self-test ]; then
  self_test
  exit $?
fi

TABLES=()
while [ $# -gt 0 ]; do
  case $1 in */) TABLES+=("$1"*.tsv); shift ;; */*) TABLES+=("$1"); shift ;; *) break ;; esac
done
if [ "${#TABLES[@]}" = 0 ] || [ $# = 0 ]; then
  echo "usage: ptx-spill-check.sh <table>... <binary name>..." >&2
  exit 2
fi
SCAN="${BASH_SOURCE[0]%/*}/ptx-scan.sh"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
# shellcheck source=tools/arch.sh
source "${BASH_SOURCE[0]%/*}/arch.sh"
bloomery_arch ptx-spill-check.sh || exit $?
KEYS=()
for i in "${!TABLES[@]}"; do
  F=${TABLES[$i]}
  cat -- "$F" > "$TMP/table.$i" || { echo "ptx-spill-check: cannot read the table $F" >&2; exit 2; }
  # one `# <key>: <value>` line each, the value one arch (tools/arch.sh's shape) or one capability as a banner prints it
  for K in arch jit-cc; do
    if [ "$K" = arch ]; then S=$ARCH_SHAPE X=sm_86; else S='^[1-9][0-9]?\.[0-9]$' X=8.6; fi
    D=$(grep "^# $K:" "$TMP/table.$i")
    n=$(grep -c "^# $K:" "$TMP/table.$i")
    case $n in
      0) echo "ptx-spill-check: the table $F declares no \`# $K:\` line: its rows pin no $K" >&2; exit 2 ;;
      1) ;;
      *) echo "ptx-spill-check: the table $F declares $n \`# $K:\` lines: a table pins one $K" >&2; exit 2 ;;
    esac
    V=${D#"# $K: "}
    if ! [[ $V =~ $S ]]; then
      echo "ptx-spill-check: the table $F declares '$D', not one $K as in '# $K: $X'" >&2
      exit 2
    fi
    if [ "$K" = arch ]; then T_ARCH=$V; else T_CC=$V; fi
  done
  for j in "${!KEYS[@]}"; do
    if [ "${KEYS[$j]}" = "$T_ARCH $T_CC" ]; then
      echo "ptx-spill-check: the tables ${TABLES[$j]} and $F both pin $T_ARCH on cc $T_CC" >&2
      exit 2
    fi
  done
  KEYS+=("$T_ARCH $T_CC")
done
NAMED=$(for K in "${KEYS[@]}"; do echo "${K% *} on cc ${K#* }"; done | paste -sd, - | sed 's/,/, /g')
if ! printf '%s\n' "${KEYS[@]}" | grep -q "^$ARCH "; then
  echo "ptx-spill-check: no ptx-shapes table for $ARCH; the tables pin $NAMED" >&2
  exit 64
fi
case ${BLOOMERY_BUILD_ONLY:-0} in
  0) ;;
  1)
    for ARG in "$@"; do
      BIN=${ARG%%:*}
      [ -f "target/release/$BIN" ] && [ -x "target/release/$BIN" ] || { echo "ptx-spill-check.sh: no executable target/release/$BIN — the recipe builds it before calling this" >&2; exit 2; }
    done
    echo "ptx-spill-check.sh: BLOOMERY_BUILD_ONLY: $# binary argument(s) built; no scan runs"
    exit 0 ;;
  *) echo "ptx-spill-check.sh: BLOOMERY_BUILD_ONLY is 1 (build, run nothing) or 0/unset, got '$BLOOMERY_BUILD_ONLY'" >&2; exit 64 ;;
esac
# The JIT cache (header). CACHE_WHY names why it cannot be used; empty means it can. The default file is under the cache home,
# never in $BLOOMERY_DATA: tools/recipes.py's box manifest digests every top-level entry there into each gate's ledger key, so a
# file rewritten by every run would move the keys.
CACHE=
if [ -n "${PTX_SPILL_CACHE:-}" ]; then
  CACHE=$PTX_SPILL_CACHE
elif [ -n "${XDG_CACHE_HOME:-}" ]; then
  CACHE=$XDG_CACHE_HOME/bloomery/ptx-spill-jit.tsv
elif [ -n "${HOME:-}" ]; then
  CACHE=$HOME/.cache/bloomery/ptx-spill-jit.tsv
fi
CACHE_HEAD='# ptx-spill jit cache v1: key = method|md5|lines|regs|smem|spill|reqntid|arch|ptxas|module-header|cc/driver/jit-compiler; row = key jit_regs jit_local entry checksum'
CACHE_WHY=
DRV=
CC=
# An awk function, a row's checksum over the characters awk can index (a row holds printable ASCII and tabs).
AWK_SUM='function sum(s,   i, h) {
  if (!ready) { for (i = 1; i < 127; i++) ord[sprintf("%c", i)] = i; ready = 1 }
  h = 5381
  for (i = 1; i <= length(s); i++) h = (h * 33 + ord[substr(s, i, 1)]) % 2147483629
  return h
}
'
# cache_identity: DRV and CC, the one driver, JIT compiler and capability the card's JIT reads under; CACHE_WHY when they are not one.
cache_identity() {
  local smi n lib sum
  smi=$(nvidia-smi --query-gpu=driver_version,compute_cap --format=csv,noheader 2> /dev/null | sed 's/ //g' | sort -u) || smi=
  n=$(printf '%s\n' "$smi" | grep -c .)
  if [ "$n" = 0 ]; then
    CACHE_WHY="nvidia-smi names no card"
  elif [ "$n" != 1 ]; then
    CACHE_WHY="the cards differ in driver version or compute capability, and the JIT's card is not known before the JIT"
  elif ! [[ $smi =~ ^[0-9][0-9.]*,[0-9]+\.[0-9]+$ ]]; then
    CACHE_WHY="nvidia-smi names '$smi', not <driver version>,<compute capability>"
  else
    lib=$(PATH=$PATH:/sbin:/usr/sbin ldconfig -p 2> /dev/null | awk '
      /libnvidia-ptxjitcompiler\.so\.1 / { if ($0 ~ /x86-64|AArch64/ && !best) best = $NF; if (!any) any = $NF }
      END { print (best ? best : any) }')
    if [ -z "$lib" ] || ! sum=$(cksum < "$lib" 2> /dev/null); then
      CACHE_WHY="the driver's JIT compiler (libnvidia-ptxjitcompiler.so.1) cannot be read"
    else
      CC=${smi#*,}
      DRV="$CC/${smi%,*}/${sum// /-}"
    fi
  fi
}
# cache_load: the rows of $CACHE that read under this driver, into $TMP/cache.rows; every other row is named. A file that is not
# a v1 cache is never read or written.
cache_load() {
  : > "$TMP/cache.rows"
  [ -s "$CACHE" ] || return 0
  if [ "$(head -1 "$CACHE")" != "$CACHE_HEAD" ]; then
    CACHE_WHY="$CACHE is not a v1 cache of this script: its first line is not the cache header"
    return 0
  fi
  LC_ALL=C awk -F'\t' -v drv="$DRV" -v keep="$TMP/cache.rows" -v path="$CACHE" "$AWK_SUM"'
    NR == 1 || /^#/ || NF == 0 { next }
    {
      why = ""
      if (NF != 5) why = "has " NF " columns, not 5"
      else if (split($1, k, "|") != 11 || length(k[2]) != 32 || k[2] !~ /^[0-9a-f]+$/) why = "its key is not a v1 key"
      else if ($2 !~ /^[0-9]+$/ || $3 !~ /^[0-9]+$/) why = "a JIT cell is not a number"
      else if (sum($1 "\t" $2 "\t" $3 "\t" $4) != $5) why = "its checksum does not match"
      if (why != "") { printf "ptx-spill cache=%s row %d refused: %s\n", path, NR, why; next }
      if (k[11] != drv) { foreign++; next }
      print > keep
    }
    END { if (foreign) printf "ptx-spill cache=%s: %d rows of another driver, JIT compiler or capability, dropped at the next write\n", path, foreign }
  ' "$CACHE"
}
# scan_keys <scan> <module header digest>: `entry<TAB>key<TAB>jit_regs<TAB>jit_local` per entry of a scan. The key holds what the
# JIT's reading is a function of. Nonzero, the cause on stderr, when the scan lacks a column or a banner field or its md5 block an entry.
scan_keys() {
  LC_ALL=C awk -v hdr="$2" -v drv="$DRV" '
    function bad(m) { print m | "cat 1>&2"; failed = 1 }
    /^ptx-scan bin=/ {
      for (i = 1; i <= NF; i++) { if ($i ~ /^arch=/) arch = substr($i, 6); if ($i ~ /^ptxas-version=/) pv = substr($i, 15) }
      next
    }
    /^entry[ \t]/ { for (i = 1; i <= NF; i++) col[$i] = i; h = 1; next }
    /^ptx-scan-md5:/ { method = $0; sub(/^.*method=/, "", method); inmd5 = 1; next }
    inmd5 { if (NF == 3) { md[$1] = $2; ml[$1] = $3 }; next }
    h && NF > 1 { rows[++n] = $0 }
    END {
      split("entry regs smem spill reqntid jit_regs jit_local", need, " ")
      for (i = 1; i <= 7; i++) if (!(need[i] in col)) bad("the scan has no " need[i] " column")
      if (arch == "" || pv == "" || method == "") bad("the scan names no arch, ptxas-version or digest method")
      for (r = 1; r <= n && !failed; r++) {
        split(rows[r], f, " ")
        e = f[col["entry"]]
        if (!(e in md)) { bad("the md5 block has no row for " e); break }
        printf "%s\t%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s\t%s\t%s\n", e, method, md[e], ml[e], f[col["regs"]], f[col["smem"]], f[col["spill"]], \
          f[col["reqntid"]], arch, pv, hdr, drv, f[col["jit_regs"]], f[col["jit_local"]]
      }
      exit failed
    }' "$1"
}
# bin_keys <scan> <module dir>: $TMP/$BIN.keys of a scan, or nonzero and KEYS_WHY when the scan cannot be keyed: an entry that
# calls a device function has the callee's code outside its digest, and the module directives (.version, .target) are no entry's:
# their distinct lines over the binary's modules are in the key, so two binaries that carry one module share its entries' rows.
bin_keys() {
  local scan=$1 hl
  shift
  KEYS_WHY=
  if grep -q ' calls=' "$scan"; then
    KEYS_WHY="its entries call a device function: the callee's code is not in their digest"
    return 1
  fi
  set -- "$1"/mod[0-9]*.ptx
  if [ ! -f "$1" ]; then
    KEYS_WHY="the scan kept no module"
    return 1
  fi
  hl=$(grep -hE '^\.(version|target|address_size)' "$@" | LC_ALL=C sort -u)
  if [ -z "$hl" ]; then
    KEYS_WHY="no module declares a .version, .target or .address_size"
    return 1
  fi
  if ! scan_keys "$scan" "$(printf '%s\n' "$hl" | cksum | tr ' ' -)" > "$TMP/$BIN.keys" 2> "$TMP/$BIN.keys.err"; then
    KEYS_WHY=$(head -1 "$TMP/$BIN.keys.err")
    return 1
  fi
}
# look_up: $TMP/$BIN.look, `H<TAB>entry<TAB>jit_regs<TAB>jit_local` where the entry's key is cached, `M<TAB>entry` where it is not.
look_up() {
  awk -F'\t' '
    FILENAME == ARGV[1] { c[$1] = $2 "\t" $3; next }
    { if ($2 in c) print "H\t" $1 "\t" c[$2]; else print "M\t" $1 }
  ' "$TMP/cache.rows" "$TMP/$BIN.keys" > "$TMP/$BIN.look"
}
# cache_add: the rows of $TMP/$BIN.keys into $TMP/cache.rows and $CACHE, which is replaced whole (a file beside it, then a rename).
# Another run's rows written since this run read the file are lost to the rename: a lost row is one more JIT, never a wrong verdict.
cache_add() {
  : > "$TMP/cache.err"
  [ -n "${PTX_SPILL_CACHE:-}" ] || mkdir -p "${CACHE%/*}" 2>> "$TMP/cache.err"
  LC_ALL=C awk -F'\t' "$AWK_SUM"'
    $3 ~ /^[0-9]+$/ && $4 ~ /^[0-9]+$/ { r = $2 "\t" $3 "\t" $4 "\t" $1; print r "\t" sum(r) }
  ' "$TMP/$BIN.keys" > "$TMP/$BIN.new"
  cat "$TMP/cache.rows" "$TMP/$BIN.new" | LC_ALL=C awk -F'\t' '{ row[$1] = $0 } END { for (k in row) print row[k] }' | LC_ALL=C sort > "$TMP/cache.merged"
  if { echo "$CACHE_HEAD"; cat "$TMP/cache.merged"; } > "$CACHE.$$.tmp" 2>> "$TMP/cache.err" && mv -f "$CACHE.$$.tmp" "$CACHE" 2>> "$TMP/cache.err"; then
    cp "$TMP/cache.merged" "$TMP/cache.rows"
  else
    rm -f "$CACHE.$$.tmp"
    echo "ptx-spill cache=$CACHE not written: $(head -1 "$TMP/cache.err")"
  fi
}
full_scan() {
  bash "$SCAN" "$BIN" ${FILTER:+"$FILTER"} > "$TMP/$BIN.scan" 2> "$TMP/$BIN.err"
  src=$?
}
# read_scan: $TMP/$BIN.scan and .err hold a whole scan and src is its exit status. With the cache, an entry whose key is cached
# takes its JIT columns from the cache; a binary with an entry that is not cached is scanned in full (the driver JITs a binary's
# modules whole, so one moved entry costs its binary's JIT) and the rows that scan read are added. CACHED and JITTED count the
# entries by where their JIT columns came from, MISSED those the cache lacked, JIT_NOTE says why this binary skipped the cache;
# $TMP/$BIN.conf holds the cached rows the JIT contradicted.
read_scan() {
  local mods=$TMP/$BIN.mods cc
  CACHED=0 JITTED='' MISSED='' MISSED_NAMES='' JIT_NOTE=''
  : > "$TMP/$BIN.conf"
  if [ -n "$CACHE_WHY" ]; then
    full_scan
    return
  fi
  if [ "${PTX_SPILL_FULL:-}" != 1 ] && [ -s "$TMP/cache.rows" ]; then
    PTX_SCAN_KEEP=$mods bash "$SCAN" --no-jit "$BIN" ${FILTER:+"$FILTER"} > "$TMP/$BIN.nj" 2> "$TMP/$BIN.nj.err"
    src=$?
    if [ "$src" -ne 0 ]; then
      # the failure a full scan reports first, with its banner and stderr
      cp "$TMP/$BIN.nj" "$TMP/$BIN.scan"
      cp "$TMP/$BIN.nj.err" "$TMP/$BIN.err"
      return
    fi
    if ! bin_keys "$TMP/$BIN.nj" "$mods"; then
      JIT_NOTE=$KEYS_WHY
      full_scan
      return
    fi
    look_up
    if ! grep -q '^M' "$TMP/$BIN.look"; then
      awk -v hits="$TMP/$BIN.look" -v cc="$CC" -v drv="$DRV" '
        BEGIN { while ((getline line < hits) > 0) { split(line, f, "\t"); JR[f[2]] = f[3]; JL[f[2]] = f[4] }; close(hits) }
        /^ptx-scan bin=/ && !banner { sub(/ jit=skipped\(no-jit\)/, " jit=cached jit-cc=" cc " jit-driver=" drv); banner = 1 }
        /^entry[ \t]/ { for (i = 1; i <= NF; i++) { if ($i == "jit_regs") a = i; if ($i == "jit_local") b = i }; h = 1; print; next }
        /^ptx-scan-md5:/ { h = 0 }
        h && NF > 1 { $a = JR[$1]; $b = JL[$1] }
        { print }
      ' "$TMP/$BIN.nj" > "$TMP/$BIN.scan"
      cp "$TMP/$BIN.nj.err" "$TMP/$BIN.err"
      CACHED=$(grep -c '^H' "$TMP/$BIN.look")
      JITTED=0
      return
    fi
  fi
  PTX_SCAN_KEEP=$mods bash "$SCAN" "$BIN" ${FILTER:+"$FILTER"} > "$TMP/$BIN.scan" 2> "$TMP/$BIN.err"
  src=$?
  [ "$src" -eq 0 ] || return
  if ! bin_keys "$TMP/$BIN.scan" "$mods"; then
    JIT_NOTE=$KEYS_WHY
    return
  fi
  cc=$(sed -n 's/^ptx-scan bin=.* jit-cc=\([^ ]*\).*/\1/p' "$TMP/$BIN.scan" | head -1)
  if [ "$cc" != "$CC" ]; then
    JIT_NOTE="the JIT ran on cc ${cc:-(none)}, the cache's key says cc $CC"
    return
  fi
  look_up
  JITTED=$(grep -c . "$TMP/$BIN.keys")
  MISSED=$(grep -c '^M' "$TMP/$BIN.look")
  # which entries the cache lacked, when it held rows to lack them from and few lack them
  if [ -s "$TMP/cache.rows" ] && [ "$MISSED" -gt 0 ] && [ "$MISSED" -le 12 ]; then
    MISSED_NAMES=$(awk -F'\t' '$1 == "M" { print $2 }' "$TMP/$BIN.look" | paste -sd, -)
  fi
  awk -F'\t' -v bin="$BIN" '
    FILENAME == ARGV[1] { if ($1 == "H") { cr[$2] = $3; cl[$2] = $4 }; next }
    ($1 in cl) && ($3 != cr[$1] || $4 != cl[$1]) {
      printf "%s %s cached jit_regs=%s jit_local=%s, the JIT read jit_regs=%s jit_local=%s CACHE-CONFLICT\n", bin, $1, cr[$1], cl[$1], $3, $4
    }
  ' "$TMP/$BIN.look" "$TMP/$BIN.keys" > "$TMP/$BIN.conf"
  cache_add
}
if [ "$CACHE" = off ]; then
  CACHE_WHY="PTX_SPILL_CACHE=off"
elif [ -z "$CACHE" ]; then
  CACHE_WHY="no cache file: PTX_SPILL_CACHE, XDG_CACHE_HOME and HOME are unset"
else
  cache_identity
  [ -n "$CACHE_WHY" ] || cache_load
fi
if [ -n "$CACHE_WHY" ]; then
  echo "ptx-spill cache=off($CACHE_WHY)"
else
  echo "ptx-spill cache=$CACHE rows=$(grep -c . "$TMP/cache.rows") driver=$DRV${PTX_SPILL_FULL:+ full=$PTX_SPILL_FULL}"
fi
rc=0
contention=0
for ARG in "$@"; do
  BIN=${ARG%%:*}
  FILTER=
  case "$ARG" in *:*) FILTER=${ARG#*:} ;; esac
  read_scan
  banner=$(grep -m1 '^ptx-scan bin=' "$TMP/$BIN.scan")
  echo "${banner:-ptx-scan bin=target/release/$BIN (no banner)}"
  if [ "$src" -eq 75 ]; then
    # ptx-scan's last stderr line on a 75 names the contention and holds gpu-gate.sh's own last line (the batch reads the hold's owner from it).
    echo "ptx-spill bin=$BIN scan rc=75 CONTENTION"
    sed 's/^/  ptx-scan stderr: /' "$TMP/$BIN.err" | tail -1
    contention=1
    continue
  fi
  if [ "$src" -ne 0 ]; then
    echo "ptx-spill bin=$BIN scan rc=$src FAIL"
    sed 's/^/  ptx-scan stderr: /' "$TMP/$BIN.err" | head -5
    rc=1
    continue
  fi
  # A skipped JIT is no reading, and `skipped(no-jit)` in awk's arithmetic is 0: refused before any compare.
  if grep -qF 'skipped(no-jit)' "$TMP/$BIN.scan"; then
    echo "ptx-spill bin=$BIN scan is no-jit (jit_local not read) FAIL"
    rc=1
    continue
  fi
  # The scan's own arch and capability pick the table.
  BARCH=$(sed -n 's/.* arch=\([^ ]*\).*/\1/p' <<< "$banner")
  BCC=$(sed -n 's/.* jit-cc=\([^ ]*\).*/\1/p' <<< "$banner")
  T=
  for i in "${!KEYS[@]}"; do [ "${KEYS[$i]}" = "$BARCH $BCC" ] && T=$i; done
  if [ -z "$T" ]; then
    echo "ptx-spill bin=$BIN no table for ${BARCH:-(no arch)} on cc ${BCC:-(no jit-cc)} FAIL"
    rc=1
    continue
  fi
  echo "ptx-spill bin=$BIN table=${TABLES[$T]} arch=$BARCH jit-cc=$BCC"
  # The table rows: between the header line (`entry … jit_local`) and the md5 section.
  awk '
    /^entry[[:space:]]/ { h = 1; for (i = 1; i <= NF; i++) { if ($i == "spill") s = i; if ($i == "jit_local") j = i }; next }
    /^ptx-scan-md5:/ { exit }
    h && NF > 1 { print $1, $s, $j }
  ' "$TMP/$BIN.scan" > "$TMP/$BIN.got"
  awk -v b="$BIN" -v filter="$FILTER" '!/^[[:space:]]*#/ && NF >= 4 && $1 == b && index($2, filter) { print $2, $3, $4 }' "$TMP/table.$T" > "$TMP/$BIN.want"
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
  ' "$TMP/$BIN.want" "$TMP/$BIN.got" | sort) || { echo "ptx-spill bin=$BIN comparison=unread FAIL"; rc=1; continue; }
  n=$(wc -l < "$TMP/$BIN.got" | tr -d ' ') || { echo "ptx-spill bin=$BIN entries=unread FAIL"; rc=1; continue; }
  p=$(wc -l < "$TMP/$BIN.want" | tr -d ' ') || { echo "ptx-spill bin=$BIN pinned=unread FAIL"; rc=1; continue; }
  nz=$(awk '$2 != 0 || $3 != 0 { printf "%s%s(spill=%s,jit_local=%s)", c, $1, $2, $3; c = "," }' "$TMP/$BIN.got")
  # The JIT columns' sources. A cached row the JIT contradicted is a violation: the key did not fix what it was to fix.
  if [ -n "$CACHE_WHY" ]; then c="cache=off($CACHE_WHY)"; elif [ -n "$JIT_NOTE" ]; then c="cache=off($JIT_NOTE)"; else c="cache=$CACHE"; fi
  echo "ptx-spill bin=$BIN jit entries=$n jitted=${JITTED:-$n} cached=$CACHED${MISSED:+ missed=$MISSED} $c"
  [ -z "$MISSED_NAMES" ] || echo "ptx-spill bin=$BIN jit missed=$MISSED_NAMES"
  out=$({ [ -z "$out" ] || printf '%s\n' "$out"; cat "$TMP/$BIN.conf"; } | sort)
  if [ -n "$out" ]; then
    echo "$out"
    v=$(printf '%s\n' "$out" | wc -l | tr -d ' ') || v=unread
    echo "ptx-spill bin=$BIN entries=$n pinned=$p nonzero=${nz:-none} violations=$v FAIL"
    rc=1
  else
    echo "ptx-spill bin=$BIN entries=$n pinned=$p nonzero=${nz:-none} violations=0 PASS"
  fi
done
# A real failure anywhere is 1; contention alone is 75 (the header).
[ "$rc" != 0 ] || [ "$contention" = 0 ] || rc=75
exit "$rc"
