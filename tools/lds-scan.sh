#!/usr/bin/env bash
# Shared-load and spill scan — per entry of a gate binary's device code, the loads and stores its PTX
# body issues and what ptxas makes of them in SASS, one row an entry. An instrument, not a gate: it
# asserts nothing about a kernel. It answers the question ptx-scan's `spill` column leaves open (that
# column is ptxas's spill store and load bytes together): how many 4-byte slots went to local memory
# (SASS STL/LDL), and whether the shared-memory loads were merged into vectors before (PTX
# ld.shared.v2/.v4) or by ptxas (SASS LDS.64/LDS.128) — the reading that settled hc_gated_up_mix's
# spill in one command.
#
# Usage: tools/lds-scan.sh <binary name> [entry substring]
#   Reads target/release/<binary> the way tools/ptx-scan.sh does (the .oxart section through
#   target/release/oxart_ptx, one PTX file a module), assembles each module holding a matching entry
#   with ptxas for the card's arch and disassembles that entry alone (cuobjdump -sass -fun). The recipe
#   (`just lds-scan <binary> [entry substring]`; `--features deepseek41` for a V4.1 binary) builds the
#   binary and the extractor first; it needs no card. LDS_SCAN_PTXAS and LDS_SCAN_CUOBJDUMP (default
#   the box env's CUDA_TOOLKIT_PATH, else /usr/local/cuda), LDS_SCAN_ARCH (default sm_86) and
#   LDS_SCAN_EXTRACT (default target/release/oxart_ptx) override the tools.
#
# Output: a banner `lds-scan bin=… ptxas=… ptxas-version=… arch=… modules=N entries=M`, then per matching
# entry, by name ascending:
#   lds <entry> lines=<PTX body lines> ld.shared=<n> ld.shared.v2=<n> ld.shared.v4=<n> ld.f32=<n>
#       ld.global=<n> st.shared=<n> shfl=<n> fma=<n> cvt.f16=<n> | sass lines=<n> STL=<n> STL.64=<n>
#       STL.128=<n> LDL=<n> LDS=<n> LDS.64=<n> LDS.128=<n>
# (one line). The PTX counts are instructions in the entry's own text (its `.entry` line to the next
# `.entry` or `.func`, as ptx-scan reads a body): ld.shared counts every width, .v2 and .v4 the vector
# ones among them; ld.f32 is a generic-space scalar or vector f32 load. The SASS counts are instruction
# lines of the entry's disassembly: STL and LDL every width, the .64/.128 ones among them; LDS likewise.
# The SASS is this toolkit's ptxas; the driver JITs the same PTX at load and may allocate differently
# (ptx-scan's jit_local column is the card's reading).
#
# Exit status: 0 with rows; 1 when the scan failed (the banner ends in `scan=failed` and names what) or no
# entry matches; 2 on a usage error.
set -uo pipefail

# `bash tools/lds-scan.sh --self-test` (check-recipes runs it, on the Mac): the scan over a fixture binary
# whose extractor, objcopy, ptxas and cuobjdump are stubs on PATH — two modules, an entry with a device
# function after it, a SASS listing with every counted width and an LDSM that is no LDS — against rows
# counted by hand; then a filter that matches nothing, a cuobjdump that fails and a third argument.
self_test() {
  local t n=0 bad=0 out rc self
  self=$(cd "$(dirname "$0")" && pwd -P)/$(basename "$0")
  t=$(mktemp -d "${TMPDIR:-/tmp}/lds-scan-test.XXXXXX") || return 70
  # shellcheck disable=SC2064 # the path is fixed now
  trap "rm -rf '$t'" EXIT
  mkdir -p "$t/bin" "$t/fix" "$t/tree/target/release"
  # shellcheck disable=SC2016 # the stubs expand them when they run
  printf '%s\n' '#!/bin/sh' 'for a; do o=$a; done' 'echo section > "$o"' > "$t/bin/objcopy"
  # shellcheck disable=SC2016
  printf '%s\n' '#!/bin/sh' 'cp "$FIX"/mod1.ptx "$FIX"/mod2.ptx "$2"/' 'echo "mod1 a"' 'echo "mod2 b"' > "$t/bin/oxart_ptx"
  # shellcheck disable=SC2016
  printf '%s\n' '#!/bin/sh' 'case $1 in --version) echo "Cuda compilation tools, release 12.8, V12.8.93"; exit 0 ;; esac' \
    'for a; do case $a in *.ptx) i=$a ;; esac; o=$a; done' 'cp "$i" "$o"' > "$t/bin/ptxas"
  # shellcheck disable=SC2016
  printf '%s\n' '#!/bin/sh' '[ -f "$FIX/$3.sass" ] || { echo "cuobjdump stub: no function $3" >&2; exit 1; }' 'cat "$FIX/$3.sass"' > "$t/bin/cuobjdump"
  printf '%s\n' '#!/bin/sh' > "$t/tree/target/release/gate_x"
  chmod +x "$t/bin/"* "$t/tree/target/release/gate_x"
  cat > "$t/fix/mod1.ptx" << 'PTX'
.version 8.0
.visible .entry up_mix_4x320(
	.param .u64 a
)
{
	ld.shared.f32 	%f1, [%r1];
	ld.shared.f32 	%f2, [%r2];
	ld.shared.v4.f32 	{%f3,%f4,%f5,%f6}, [%r3];
	ld.shared.v2.f32 	{%f11,%f12}, [%r3];
	ld.global.nc.f32 	%f7, [%rd1];
	st.shared.f32 	[%r4], %f7;
	fma.rn.f32 	%f8, %f1, %f2, %f3;
	shfl.sync.bfly.b32 	%r5, %r6, 1, 31, -1;
	cvt.f32.f16 	%f9, %rs1;
	ld.f32 	%f10, [%rd2];
}
.func helper()
{
	ld.shared.f32 	%f1, [%r1];
}
.visible .entry up_down(
)
{
	ld.global.f32 	%f1, [%rd1];
}
PTX
  printf '%s\n' '.visible .entry other(' ')' '{' '	ret;' '}' > "$t/fix/mod2.ptx"
  cat > "$t/fix/up_mix_4x320.sass" << 'SASS'
		Function : up_mix_4x320
        /*0010*/                   STL [R1+0x4], R2 ;
        /*0020*/                   STL.64 [R1+0x8], R4 ;
        /*0030*/                   STL.128 [R1+0x10], R4 ;
        /*0040*/                   LDL R3, [R1+0x4] ;
        /*0050*/                   LDS.128 R8, [R0] ;
        /*0060*/                   LDS.64 R8, [R0] ;
        /*0070*/                   LDS R9, [R0+0x10] ;
        /*0080*/                   LDSM.16.M88.4 R12, [R0] ;
SASS
  printf '%s\n' '		Function : up_down' '        /*0010*/                   LDG.E R2, [R2.64] ;' > "$t/fix/up_down.sass"
  scan() { (cd "$t/tree" && env PATH="$t/bin:$PATH" FIX="$t/fix" LDS_SCAN_PTXAS="$t/bin/ptxas" LDS_SCAN_CUOBJDUMP="$t/bin/cuobjdump" \
    LDS_SCAN_EXTRACT="$t/bin/oxart_ptx" bash "$self" "$@") > "$t/out" 2>&1; }
  # case_ <name> <want rc> <want output, whole>
  case_() {
    n=$((n + 1))
    out=$(cat "$t/out")
    if [ "$rc" = "$2" ] && [ "$out" = "$3" ]; then
      echo "ok $1"
    else
      bad=$((bad + 1))
      echo "FAIL $1: rc $rc (want $2)"
      printf '%s\n' "$out" | sed 's/^/    got | /'
      printf '%s\n' "$3" | sed 's/^/   want | /'
    fi
  }
  local banner="lds-scan bin=target/release/gate_x ptxas=$t/bin/ptxas ptxas-version=12.8.93 arch=sm_86 modules=2"
  local mix="lds up_mix_4x320 lines=15 ld.shared=4 ld.shared.v2=1 ld.shared.v4=1 ld.f32=1 ld.global=1 st.shared=1 shfl=1 fma=1 cvt.f16=1 | sass lines=9 STL=3 STL.64=1 STL.128=1 LDL=1 LDS=3 LDS.64=1 LDS.128=1"
  local down="lds up_down lines=5 ld.shared=0 ld.shared.v2=0 ld.shared.v4=0 ld.f32=0 ld.global=1 st.shared=0 shfl=0 fma=0 cvt.f16=0 | sass lines=2 STL=0 STL.64=0 STL.128=0 LDL=0 LDS=0 LDS.64=0 LDS.128=0"
  rc=0; scan gate_x up_ || rc=$?
  case_ two-entries 0 "$banner entries=2 filter=up_
$down
$mix"
  rc=0; scan gate_x mix || rc=$?
  case_ one-entry 0 "$banner entries=1 filter=mix
$mix"
  rc=0; scan gate_x nomatch || rc=$?
  case_ no-match 1 "$banner entries=0 filter=nomatch
lds-scan: no entry of target/release/gate_x matches 'nomatch'"
  rc=0; scan gate_x other || rc=$?
  case_ cuobjdump-fails 1 "$banner entries=1 filter=other
cuobjdump stub: no function other
lds-scan bin=target/release/gate_x cuobjdump-failed=other scan=failed"
  rc=0; scan gate_x mix extra || rc=$?
  case_ third-argument 2 "usage: lds-scan.sh <binary name> [entry substring] | --self-test"
  echo "lds-scan: self-test $([ "$bad" = 0 ] && echo ok || echo FAIL) ($n cases, $bad failed)"
  [ "$bad" = 0 ]
}
if [ "${1:-}" = --self-test ]; then
  self_test
  exit
fi
if [ $# -lt 1 ] || [ $# -gt 2 ] || [ -z "$1" ]; then
  echo "usage: lds-scan.sh <binary name> [entry substring] | --self-test" >&2
  exit 2
fi
NAME=$1
FILTER=${2:-}
TOOLKIT=${CUDA_TOOLKIT_PATH:-/usr/local/cuda}
PTXAS=${LDS_SCAN_PTXAS:-$TOOLKIT/bin/ptxas}
CUOBJDUMP=${LDS_SCAN_CUOBJDUMP:-$TOOLKIT/bin/cuobjdump}
ARCH=${LDS_SCAN_ARCH:-sm_86}
EXTRACT=${LDS_SCAN_EXTRACT:-target/release/oxart_ptx}
BIN=target/release/$NAME
fail() { # fail <banner fields...>
  echo "lds-scan bin=$BIN $* scan=failed"
  exit 1
}
[ -x "$BIN" ] || { echo "lds-scan: no $BIN — run the matching just recipe first" >&2; fail missing; }
[ -x "$PTXAS" ] || fail "ptxas=none($PTXAS)"
[ -x "$CUOBJDUMP" ] || fail "cuobjdump=none($CUOBJDUMP)"
[ -x "$EXTRACT" ] || fail extract=none
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
if ! objcopy -O binary --only-section=.oxart "$BIN" "$TMP/section" 2> /dev/null || [ ! -s "$TMP/section" ]; then
  fail section=none
fi
"$EXTRACT" "$TMP/section" "$TMP" > "$TMP/list" || fail extract=failed
NMOD=$(grep -c '^mod[0-9]' "$TMP/list")
[ "$NMOD" -gt 0 ] || fail modules=0

# add <field> <ERE> <file>: ` <field>=<lines of <file> matching <ERE>>` onto ROW. It runs in this shell, not
# in a command substitution, so a grep that fails (2) fails the scan instead of printing an empty count.
add() {
  local n rc=0
  n=$(grep -cE -- "$2" "$3") || rc=$?
  [ "$rc" -le 1 ] || fail "grep-failed=${3##*/}"
  ROW+=" $1=$n"
}

# Every entry of every module, `<module> <name>` a line; the filter is a fixed substring of the name.
for ((i = 1; i <= NMOD; i++)); do
  sed -nE 's/^[[:space:]]*(\.visible[[:space:]]+)?\.entry[[:space:]]+([A-Za-z_][A-Za-z0-9_]*).*/\2/p' "$TMP/mod$i.ptx" |
    sed "s/^/$i /"
done | sort -k2 > "$TMP/entries"
if [ -n "$FILTER" ]; then
  awk -v f="$FILTER" 'index($2, f) > 0' "$TMP/entries" > "$TMP/match"
else
  cp "$TMP/entries" "$TMP/match"
fi
NENT=$(wc -l < "$TMP/match" | tr -d ' ')
VER=$("$PTXAS" --version 2> /dev/null | sed -n 's/.*, V\([0-9][0-9.]*\)$/\1/p')
echo "lds-scan bin=$BIN ptxas=$PTXAS ptxas-version=${VER:-unknown} arch=$ARCH modules=$NMOD entries=$NENT${FILTER:+ filter=$FILTER}"
if [ "$NENT" = 0 ]; then
  echo "lds-scan: no entry of $BIN matches '${FILTER}'"
  exit 1
fi
while read -r mod entry; do
  # The entry's own text: its `.entry` line to the next `.entry` or `.func` line.
  awk -v e="$entry" '
    $0 ~ "\\.entry[ \t]+" e "[ \t(]" || $0 ~ "\\.entry[ \t]+" e "$" { p = 1; print; next }
    p && /\.(entry|func)[ \t]/ { p = 0 }
    p' "$TMP/mod$mod.ptx" > "$TMP/e.ptx"
  lines=$(wc -l < "$TMP/e.ptx" | tr -d ' ')
  [ "$lines" -gt 0 ] || fail "body=none($entry)"
  if [ ! -f "$TMP/mod$mod.cubin" ]; then
    "$PTXAS" -arch="$ARCH" "$TMP/mod$mod.ptx" -o "$TMP/mod$mod.cubin" 2> "$TMP/mod$mod.err" ||
      { echo "lds-scan: ptxas failed on mod$mod: $(head -1 "$TMP/mod$mod.err")" >&2; fail "ptxas-failed=mod$mod"; }
  fi
  "$CUOBJDUMP" -sass -fun "$entry" "$TMP/mod$mod.cubin" > "$TMP/e.sass" || fail "cuobjdump-failed=$entry"
  slines=$(wc -l < "$TMP/e.sass" | tr -d ' ')
  grep -qE -- "Function : $entry\$" "$TMP/e.sass" || fail "sass=none($entry)"
  ROW="lds $entry lines=$lines"
  add ld.shared '\bld\.shared\.' "$TMP/e.ptx"
  add ld.shared.v2 '\bld\.shared\.v2\.' "$TMP/e.ptx"
  add ld.shared.v4 '\bld\.shared\.v4\.' "$TMP/e.ptx"
  add ld.f32 '\bld\.(v[24]\.)?f32\b' "$TMP/e.ptx"
  add ld.global '\bld\.global\.' "$TMP/e.ptx"
  add st.shared '\bst\.shared\.' "$TMP/e.ptx"
  add shfl '\bshfl\.' "$TMP/e.ptx"
  add fma '\bfma\.' "$TMP/e.ptx"
  add cvt.f16 '\bcvt\.f32\.f16\b' "$TMP/e.ptx"
  ROW+=" | sass lines=$slines"
  add STL '\bSTL\b' "$TMP/e.sass"
  add STL.64 '\bSTL\.64\b' "$TMP/e.sass"
  add STL.128 '\bSTL\.128\b' "$TMP/e.sass"
  add LDL '\bLDL\b' "$TMP/e.sass"
  add LDS '\bLDS\b' "$TMP/e.sass"
  add LDS.64 '\bLDS\.64\b' "$TMP/e.sass"
  add LDS.128 '\bLDS\.128\b' "$TMP/e.sass"
  echo "$ROW"
done < "$TMP/match"
