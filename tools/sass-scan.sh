#!/usr/bin/env bash
# SASS scan — the SASS twin of tools/ptx-scan.sh: per entry of a gate binary's device code, how many
# global loads are issued before the first wait on one (per loop, and along one execution path).
# An instrument, not a gate. It reads the PTX that ptx-scan reads (.oxart through oxart_ptx),
# assembles each module with ptxas for the card's arch, disassembles the cubin with cuobjdump, and
# hands the listings to tools/sass_inflight.py, whose header says what is counted and how.
#
# Usage: tools/sass-scan.sh [--exact] <binary name> [entry substring [decisions]]
#   --exact    the entry argument is a whole name, not a substring (`ds41_attn_seg` alone, not also
#              `ds41_attn_seg_sel`); it may stand anywhere among the arguments. Through the recipe it
#              goes after `--`, or just reads it as a recipe option:
#              `just sass-scan --features deepseek41 <binary> -- --exact <entry>`.
#   decisions  the path's conditional branches in order, t (taken) or n (not taken), comma-separated,
#              e.g. n,t,t,n; without them the path falls through every conditional branch. `list`
#              instead prints the matching entries' listings (<addr> <instruction>) to pick them from.
#   The recipe (`just sass-scan <binary>`; `--features deepseek41` for a V4.1 binary) builds the
#   binary and the extractor first. SASS_SCAN_PTXAS and SASS_SCAN_CUOBJDUMP (default the box env's
#   CUDA_TOOLKIT_PATH, else /usr/local/cuda), SASS_SCAN_ARCH (default sm_86) and SASS_SCAN_EXTRACT
#   (default target/release/oxart_ptx) override the tools.
#
# The SASS is what ptxas makes of the PTX here; the driver JITs the same PTX at load and may allocate
# differently, so a count here is this toolkit's reading of the kernel, named in the banner.
# Exit status: 0 with rows; 1 when the scan failed (the banner ends in `scan=failed` and names what),
# or no entry matches; 2 on a usage error.
set -uo pipefail
EXACT=
POS=()
for a in "$@"; do
  if [ "$a" = --exact ]; then EXACT=1; else POS+=("$a"); fi
done
NAME=${POS[0]:-}
FILTER=${POS[1]:-}
DECISIONS=${POS[2]:-}
if [ -z "$NAME" ] || [ ${#POS[@]} -gt 3 ] || { [ -n "$EXACT" ] && [ -z "$FILTER" ]; }; then
  echo "usage: sass-scan.sh [--exact] <binary name> [entry substring [decisions|list]]" >&2
  echo "       (--exact needs the entry argument)" >&2
  exit 2
fi
TOOLKIT=${CUDA_TOOLKIT_PATH:-/usr/local/cuda}
PTXAS=${SASS_SCAN_PTXAS:-$TOOLKIT/bin/ptxas}
CUOBJDUMP=${SASS_SCAN_CUOBJDUMP:-$TOOLKIT/bin/cuobjdump}
ARCH=${SASS_SCAN_ARCH:-sm_86}
EXTRACT=${SASS_SCAN_EXTRACT:-target/release/oxart_ptx}
BIN=target/release/$NAME
fail() { # fail <banner fields...>
  echo "sass-scan bin=$BIN $* scan=failed"
  exit 1
}
[ -x "$BIN" ] || { echo "sass-scan: no $BIN — run the matching just recipe first" >&2; fail missing; }
[ -x "$PTXAS" ] || fail "ptxas=none($PTXAS)"
[ -x "$CUOBJDUMP" ] || fail "cuobjdump=none($CUOBJDUMP)"
[ -x "$EXTRACT" ] || fail extract=none
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
if ! objcopy -O binary --only-section=.oxart "$BIN" "$TMP/section" 2>/dev/null || [ ! -s "$TMP/section" ]; then
  fail section=none
fi
"$EXTRACT" "$TMP/section" "$TMP" > "$TMP/list" || fail extract=failed
NMOD=$(grep -c '^mod[0-9]' "$TMP/list")
[ "$NMOD" -gt 0 ] || fail modules=0
for ((i = 1; i <= NMOD; i++)); do
  "$PTXAS" -arch="$ARCH" "$TMP/mod$i.ptx" -o "$TMP/mod$i.cubin" 2> "$TMP/mod$i.err" \
    || { echo "sass-scan: ptxas failed on mod$i: $(head -1 "$TMP/mod$i.err")" >&2; fail "ptxas-failed=mod$i"; }
  "$CUOBJDUMP" -sass "$TMP/mod$i.cubin" > "$TMP/mod$i.sass" \
    || fail "cuobjdump-failed=mod$i"
done
VER=$("$PTXAS" --version 2>/dev/null | sed -n 's/.*, V\([0-9][0-9.]*\)$/\1/p')
ARGS=(--banner "sass-scan bin=$BIN ptxas=$PTXAS ptxas-version=${VER:-unknown} arch=$ARCH modules=$NMOD")
[ -z "$FILTER" ] || ARGS+=(--filter "$FILTER")
[ -z "$EXACT" ] || ARGS+=(--exact)
[ -z "$DECISIONS" ] || ARGS+=(--decisions "$DECISIONS")
MODS=()
for ((i = 1; i <= NMOD; i++)); do MODS+=("$TMP/mod$i.sass"); done
python3 "${BASH_SOURCE[0]%/*}/sass_inflight.py" "${ARGS[@]}" "${MODS[@]}"
