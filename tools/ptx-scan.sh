#!/usr/bin/env bash
# PTX scan — read the device code a gate binary carries. An instrument, not a gate (always exits 0).
#
# `cargo oxide` puts the bundle's PTX text into the executable's `.oxart` ELF section verbatim.
# So a kernel's compiled shape — whether registers spilled to local (`__local_depot`), how many
# round trips that costs (ld.local/st.local), how wide the block is (`.reqntid`) — reads out of a
# byte scan with no device. Two root causes were found this way: flash's per-lane accumulator
# array spilling to a depot, and rms_norm's launch geometry leaving one resident warp.
#
# Asserting is the gates' job. This script only prints a table:
#   gate_p5 `no_local_depot`  — depot, ld.local and st.local are all 0 for the five flash kernels
#   gate_p4 `norm_geometry`   — .reqntid of rms_norm and norm_quant == RMS_THREADS
#   gate_p4 `argmax_geometry` — .reqntid of argmax == ARGMAX_THREADS
#
# Mind the spelling: PTX writes a fused multiply-add as `fma.rn.f32` (and `fma.rm.f32`). There is
# no `fma.f32` anywhere, so counting that silently returns 0 and the kernel reads as clean — here
# and in the gates, the `fma.` prefix is what is counted.
#
# Usage: tools/ptx-scan.sh <binary name> [entry substring]
#   Reads target/release/<binary>. The recipe (`just ptx-scan <binary>`) builds it first.
# Output puts entries that have a depot first, then by name ascending — a depot is the defect and
# the rest is context.
#
# The last four columns come from `ptxas -v` on the same PTX, not from the byte scan:
#   regs   registers per thread ("Used N registers")
#   smem   static shared memory per block, bytes (0 when ptxas prints no smem clause)
#   spill  spill stores + spill loads, bytes, summed from the function-properties line
#   blk/SM(static)  resident blocks per SM, derived from those three and .reqntid
# They are `-` when ptxas is absent, or when the entry carries no .reqntid to size a block.
# The occupancy arithmetic is the sm_86 column of the CUDA programming guide's technical
# specifications: 65536 registers and 100 KB of shared memory per SM, 48 resident warps,
# 16 resident blocks (8.0 allows 32, 8.6 does not), registers allocated 256 per warp, shared
# memory 128 bytes per block plus a 1 KB per-block driver reservation. Static only: dynamic
# shared memory is a launch argument and is nowhere in the PTX, so an entry that asks for
# some at launch is reported here as if it asked for none.
# Two more limits of this reading: the driver JITs the embedded PTX at load, so its assembler
# may differ from the toolkit ptxas by a register or two; and a spill in SASS is not a PTX
# local round-trip — an entry can show ld.local/st.local here and 0 spill bytes.
set -uo pipefail
NAME=${1:-}
FILTER=${2:-}
PTXAS=${PTX_SCAN_PTXAS:-/usr/local/cuda/bin/ptxas}
ARCH=${PTX_SCAN_ARCH:-sm_86}
# The module split needs NUL as a record separator, which is gawk, not mawk or BSD awk.
GAWK=$(command -v gawk || command -v awk)
if [ -z "$NAME" ]; then
  echo "usage: ptx-scan.sh <gate_bin_name> [entry-substring]" >&2
  exit 0
fi
BIN=target/release/$NAME
if [ ! -x "$BIN" ]; then
  echo "ptx-scan: no $BIN — run the matching just gate-gpu-* recipe first" >&2
  exit 0
fi
PTX=$(mktemp)
TBL=$(mktemp)
MODS=$(mktemp -d)
trap 'rm -rf "$PTX" "$TBL" "$MODS"' EXIT
if ! objcopy -O binary --only-section=.oxart "$BIN" "$PTX" 2>/dev/null || [ ! -s "$PTX" ]; then
  echo "ptx-scan: $BIN carries no .oxart section (host-only binary?)" >&2
  exit 0
fi
# The section is an OXIDEART container: the PTX text sits in it as one NUL-delimited run,
# alongside binary headers and a symbol table. Cut it out on NUL rather than stripping NULs,
# or the symbol table lands in what ptxas reads. One module per `.version` header.
NMOD=0
if [ -x "$PTXAS" ]; then
  NMOD=$(LC_ALL=C "$GAWK" -v dir="$MODS" '
    BEGIN { RS = "\0" }
    /\.version [0-9]/ { n++; print > (dir "/mod" n ".ptx") }
    END { print n + 0 }
  ' "$PTX" 2>/dev/null || echo 0)
  case "$NMOD" in
    ''|*[!0-9]*) NMOD=0 ;;
  esac
  # No module means the split failed, not that the bundle is empty — an awk without NUL
  # records is the likely reason, and it would otherwise read as four silent `-` columns.
  [ "$NMOD" != 0 ] || echo "ptx-scan: no PTX module found in .oxart ($GAWK without NUL records?)" >&2
  for MOD in "$MODS"/mod*.ptx; do
    [ -e "$MOD" ] || continue
    # A module ptxas rejects (a .version it does not know, a wrong -arch) would otherwise
    # read as four silent `-` columns under a banner that says ptxas ran.
    "$PTXAS" -arch="$ARCH" -v "$MOD" -o /dev/null 2>"$MOD.err" ||
      echo "ptx-scan: ptxas failed on $(basename "$MOD"): $(head -1 "$MOD.err")" >&2
    # Key on the name ptxas quotes, never on a substring: `flash_latent` is a prefix of
    # `flash_latent_seg`, which is a prefix of `flash_latent_seg_v2`.
    LC_ALL=C awk -v q="'" '
      index($0, "Compiling entry function") > 0 {
        i = index($0, q); rest = substr($0, i + 1); j = index(rest, q)
        name = substr(rest, 1, j - 1)
        next
      }
      name != "" && index($0, "bytes spill stores") > 0 {
        st = 0; ld = 0
        if (match($0, /[0-9]+ bytes spill stores/)) { s = substr($0, RSTART, RLENGTH); sub(/ .*/, "", s); st = s + 0 }
        if (match($0, /[0-9]+ bytes spill loads/))  { s = substr($0, RSTART, RLENGTH); sub(/ .*/, "", s); ld = s + 0 }
        spill[name] = st + ld
        next
      }
      name != "" && match($0, /Used [0-9]+ registers/) {
        s = substr($0, RSTART, RLENGTH); gsub(/[^0-9]/, "", s); regs[name] = s + 0
        smem[name] = 0
        if (match($0, /[0-9]+ bytes smem/)) { s = substr($0, RSTART, RLENGTH); sub(/ .*/, "", s); smem[name] = s + 0 }
        seen[name] = 1
        next
      }
      END { for (k in seen) printf "%s\t%d\t%d\t%d\n", k, regs[k], smem[k], spill[k] }
    ' "$MOD.err" >>"$TBL"
  done
else
  echo "ptx-scan: no $PTXAS — regs/smem/spill/blk columns are '-'" >&2
fi
echo "ptx-scan bin=$BIN section=.oxart bytes=$(wc -c <"$PTX" | tr -d ' ') ptxas=$([ -x "$PTXAS" ] && echo "$PTXAS arch=$ARCH modules=$NMOD" || echo none)${FILTER:+ filter=$FILTER}"
printf '%-28s %8s %6s %9s %9s %6s %8s %6s %6s %6s %14s\n' \
  entry reqntid depot ld.local st.local fma cvt.f16 regs smem spill 'blk/SM(static)'
LC_ALL=C tr -d '\000' <"$PTX" | LC_ALL=C awk -v filter="$FILTER" -v tbl="$TBL" '
BEGIN {
  while ((getline line < tbl) > 0) {
    split(line, f, "\t"); R[f[1]] = f[2]; S[f[1]] = f[3]; P[f[1]] = f[4]
  }
  close(tbl)
}
# Resident blocks per SM on sm_86 from the static resources alone; see the header comment
# for the constants and for what "static" leaves out.
function blocks(ntid, regs, smem,   wpb, rpw, lr, lw, ls, b) {
  if (ntid == "" || regs == "") return "-"
  wpb = int((ntid + 31) / 32)
  if (wpb < 1 || wpb > 48) return "-"
  if (regs < 1) regs = 1
  rpw = int((regs * 32 + 255) / 256) * 256
  lr = int(65536 / (rpw * wpb))
  lw = int(48 / wpb)
  ls = int(102400 / (int((smem + 127) / 128) * 128 + 1024))
  b = lr
  if (lw < b) b = lw
  if (ls < b) b = ls
  if (16 < b) b = 16
  return b
}
function flush(   have) {
  if (name == "") return
  have = (name in R)
  if (filter == "" || index(name, filter) > 0)
    printf "%d\t%-28s %8s %6s %9d %9d %6d %8d %6s %6s %6s %14s\n", (depot ? 0 : 1), name,
           (ntid == "" ? "-" : ntid), (depot ? "YES" : "no"), ldl, stl, fma, cvt,
           (have ? R[name] : "-"), (have ? S[name] : "-"), (have ? P[name] : "-"),
           (have ? blocks(ntid, R[name], S[name]) : "-")
  name = ""
}
function reset() { ntid = ""; depot = 0; ldl = 0; stl = 0; fma = 0; cvt = 0 }
function tally(line, needle,   n, s) { n = 0; s = line
  while ((i = index(s, needle)) > 0) { n++; s = substr(s, i + length(needle)) }
  return n }
/\.visible \.entry / {
  flush(); reset()
  name = $0
  sub(/.*\.visible \.entry /, "", name)
  sub(/\(.*/, "", name)
  next
}
name != "" {
  if (ntid == "" && index($0, ".reqntid ") > 0) {
    ntid = $0; sub(/.*\.reqntid /, "", ntid); sub(/[^0-9].*/, "", ntid)
  }
  if (index($0, "__local_depot") > 0) depot = 1
  ldl += tally($0, "ld.local")
  stl += tally($0, "st.local")
  fma += tally($0, "fma.")
  cvt += tally($0, "cvt.f32.f16")
}
END { flush() }
' | LC_ALL=C sort -k1,1n -k2,2 | cut -f2-
exit 0
