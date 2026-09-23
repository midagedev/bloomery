#!/usr/bin/env bash
# PTX scan — read the device code a gate binary carries, as a table. An instrument, not a gate: it
# asserts nothing about a kernel. But a scan that could not read the kernels is a failure, not a
# table — it exits 0 only when every PTX module came out of the container, assembled under ptxas
# and loaded under the driver's JIT.
#
# `cargo oxide` puts the bundle's PTX text into the executable's `.oxart` ELF section verbatim.
# So a kernel's compiled shape — whether registers spilled to local (`__local_depot`), how many
# round trips that costs (ld.local/st.local), how wide the block is (`.reqntid`) — reads out of a
# byte scan with no device. Two root causes were found this way: flash's per-lane accumulator
# array spilling to a depot, and rms_norm's launch geometry leaving one resident warp.
#
# The section is an oxide-artifacts container, and only its parser knows where a payload ends: a
# payload is stored with its length and padded to 8 bytes, so no NUL is promised after the PTX: a
# payload of 8k bytes runs straight into the container's next record (for the last payload, the
# entry-symbol table). The extractor
# target/release/oxart_ptx (crates/gpu-gates/src/bin/oxart_ptx.rs, the parser behind
# `bloomery_gpu_gates::ptx`) writes one file per bundle that carries PTX; every column below is
# read from those files and from nothing else in the section.
#
# Asserting is the gates' job. This script only prints a table:
#   gate_p5 `no_local_depot`        — depot, ld.local and st.local are all 0 for the four flash
#                                     kernels and the two kv_append entries
#   gate_p4 `norm_geometry`         — .reqntid of rms_norm and norm_quant == RMS_THREADS
#   gate_p4 `argmax_geometry`       — .reqntid of argmax == ARGMAX_THREADS
#   gate_p6 `router_shape`          — the fma floor and no depot for f32_gemv, q8_0_gemv and
#                                     q8_0_gemv_heads, and .reqntid of router_topk and expert_table
#   gate_p6 `q3k_half_decode_shape` — no clz and a hardware f16 convert in both Q3_K entries
#
# Mind the spelling: PTX writes a fused multiply-add as `fma.rn.f32` (and `fma.rm.f32`). There is
# no `fma.f32` anywhere, so counting that silently returns 0 and the kernel reads as clean — here
# and in the gates, the `fma.` prefix is what is counted.
#
# An entry's body is its own text only: it ends at the next `.entry` or `.func` line, so a device
# function's instructions never land on the entry above it. An entry that `call`s one has the
# callee's work outside its row; the banner names such entries (`calls=NAME:N,…`) and stderr says
# why — the same reading as bloomery_gpu_gates::ptx (`body`, `Counts::calls`).
#
# Usage: tools/ptx-scan.sh <binary name> [entry substring]
#   Reads target/release/<binary> and runs the extractor on its section. The recipe
#   (`just ptx-scan <binary>`; `--features deepseek41` for a V4.1 binary) builds all three first.
#   PTX_SCAN_PTXAS (default $CUDA_TOOLKIT_PATH/bin/ptxas, else /usr/local/cuda/bin/ptxas),
#   PTX_SCAN_ARCH (default sm_86), PTX_SCAN_EXTRACT (default target/release/oxart_ptx) and
#   PTX_SCAN_JIT (default oxart_jit, a target/release binary run through tools/gpu-gate.sh)
#   override the tools.
# Output puts entries that have a depot first, then by name ascending — a depot is the defect and
# the rest is context.
#
# Exit status. 0: a banner line, the header, one row per entry. 1: the scan failed — the banner line
# on stdout ends in `scan=failed`, names what failed, and no rows follow; the detail is on stderr:
#   missing              no target/release/<binary>
#   section=none         the binary carries no .oxart section (a host-only binary?)
#   extract=failed       the extractor rejected the section, or is not there
#   modules=0            the section carries no PTX payload
#   ptxas=none           no executable ptxas, so the ptxas columns cannot be read
#   ptxas-failed=modN    ptxas rejected these modules (comma-separated)
#   jit=failed           the driver did not load every module, or the gate lock was not free
#   ptxas-unread=NAME    ptxas assembled every module but reported nothing for these entries
#   jit-unread=NAME      the driver loaded every module but reported nothing for these entries
#   bytescan=failed      the byte scan itself failed
#   rows=0               no entry, or none the entry substring matches
# 2: a usage error.
#
# The next four columns come from `ptxas -v` on the same PTX, not from the byte scan:
#   regs   registers per thread ("Used N registers")
#   smem   static shared memory per block, bytes (0 when ptxas prints no smem clause)
#   spill  spill stores + spill loads, bytes, summed from the function-properties line
#   blk/SM(static)  resident blocks per SM, derived from those three and .reqntid
# blk/SM is `-` when the entry carries no .reqntid to size a block.
# The occupancy arithmetic is the sm_86 column of the CUDA programming guide's technical
# specifications: 65536 registers and 100 KB of shared memory per SM, 48 resident warps,
# 16 resident blocks (8.0 allows 32, 8.6 does not), registers allocated 256 per warp, shared
# memory 128 bytes per block plus a 1 KB per-block driver reservation. Static only: dynamic
# shared memory is a launch argument and is nowhere in the PTX, so an entry that asks for
# some at launch is reported here as if it asked for none.
# A spill in SASS is not a PTX local round-trip — an entry can show ld.local/st.local here and 0
# spill bytes.
#
# The last two columns are what the card runs. The kernels reach it as PTX and the driver JITs
# them at load with its own compiler, which need not be the toolkit's ptxas; oxart_jit loads every
# module the way cuda-core does (cuModuleLoadData, default JIT options) and reads, per entry:
#   jit_regs   CU_FUNC_ATTRIBUTE_NUM_REGS — the registers a launch of the entry occupies
#   jit_local  CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES — per-thread local memory (spill and local arrays)
# The banner names the card and the driver's CUDA version they came from (jit-card, jit-cuda). The
# load needs a card, so it runs under the gate lock: a scan waits while another GPU gate holds it.
set -uo pipefail
NAME=${1:-}
FILTER=${2:-}
# The toolkit the build uses (the box env's CUDA_TOOLKIT_PATH): toolkits differ here by a register
# on some entries, so the scan reads the one the bindings were generated from.
PTXAS=${PTX_SCAN_PTXAS:-${CUDA_TOOLKIT_PATH:-/usr/local/cuda}/bin/ptxas}
ARCH=${PTX_SCAN_ARCH:-sm_86}
EXTRACT=${PTX_SCAN_EXTRACT:-target/release/oxart_ptx}
JIT=${PTX_SCAN_JIT:-oxart_jit}
if [ -z "$NAME" ]; then
  echo "usage: ptx-scan.sh <gate_bin_name> [entry-substring]" >&2
  exit 2
fi
BIN=target/release/$NAME
# A failed scan prints its banner and no rows: two failed scans must never diff as two equal tables.
fail() { # fail <banner fields...>
  echo "ptx-scan bin=$BIN $* scan=failed"
  exit 1
}
if [ ! -x "$BIN" ]; then
  echo "ptx-scan: no $BIN — run the matching just gate-gpu-* recipe first" >&2
  fail missing
fi
PTX=$(mktemp)
TBL=$(mktemp)
MODS=$(mktemp -d)
trap 'rm -rf "$PTX" "$TBL" "$MODS"' EXIT
if ! objcopy -O binary --only-section=.oxart "$BIN" "$PTX" 2>/dev/null || [ ! -s "$PTX" ]; then
  echo "ptx-scan: $BIN carries no .oxart section (host-only binary?)" >&2
  fail section=none
fi
SEC="section=.oxart bytes=$(wc -c <"$PTX" | tr -d ' ')"
# The container is cut by its own parser, never here: one file per bundle that carries PTX.
if [ ! -x "$EXTRACT" ]; then
  echo "ptx-scan: no extractor $EXTRACT — cargo build --release -p bloomery-gpu-gates --bin oxart_ptx" >&2
  fail "$SEC extract=failed"
fi
if ! "$EXTRACT" "$PTX" "$MODS" >"$MODS/list"; then
  fail "$SEC extract=failed"
fi
sed 's/^/ptx-scan: /' "$MODS/list" >&2
NMOD=$(grep -c '^mod[0-9]' "$MODS/list")
# The version names the assembler the last four columns come from: a toolkit move changes them.
if [ -x "$PTXAS" ]; then
  PTXAS_VER=$("$PTXAS" --version 2>/dev/null | sed -n 's/.*, V\([0-9][0-9.]*\)$/\1/p')
  TOOLS="ptxas=$PTXAS ptxas-version=${PTXAS_VER:-unknown} arch=$ARCH"
else
  TOOLS=ptxas=none
fi
if [ "$NMOD" = 0 ]; then
  echo "ptx-scan: the .oxart section carries no PTX payload" >&2
  fail "$SEC $TOOLS modules=0"
fi
if [ ! -x "$PTXAS" ]; then
  echo "ptx-scan: no $PTXAS — regs/smem/spill/blk cannot be read" >&2
  fail "$SEC $TOOLS modules=$NMOD"
fi
MODFILES=()
for ((i = 1; i <= NMOD; i++)); do MODFILES+=("$MODS/mod$i.ptx"); done
FAILED=
for MOD in "${MODFILES[@]}"; do
  # A module ptxas rejects (a .version it does not know, a wrong -arch) has no ptxas columns.
  if ! "$PTXAS" -arch="$ARCH" -v "$MOD" -o /dev/null 2>"$MOD.err"; then
    echo "ptx-scan: ptxas failed on $(basename "$MOD"): $(head -1 "$MOD.err")" >&2
    FAILED=${FAILED:+$FAILED,}$(basename "$MOD" .ptx)
    continue
  fi
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
[ -z "$FAILED" ] || fail "$SEC $TOOLS modules=$NMOD ptxas-failed=$FAILED"
# The driver's JIT on the box's card, the same section: one `# card=… cuda_driver=…` line, then
# `<entry>\t<regs>\t<local>\t<shared>\t<max_threads>` per entry.
JITTBL=$MODS/jit
if ! bash tools/gpu-gate.sh "$JIT" "$PTX" >"$JITTBL" 2>"$MODS/jit.err"; then
  echo "ptx-scan: the driver JIT ($JIT) failed: $(tail -1 "$MODS/jit.err")" >&2
  fail "$SEC $TOOLS modules=$NMOD jit=failed"
fi
JITS=$(sed -n 's/^# card=\([^ ]*\) cuda_driver=\([^ ]*\)$/jit-card=\1 jit-cuda=\2/p' "$JITTBL")
if [ -z "$JITS" ]; then
  echo "ptx-scan: $JIT printed no card line" >&2
  fail "$SEC $TOOLS modules=$NMOD jit=failed"
fi
ROWS=$MODS/rows
UNREAD=$MODS/unread
JITUNREAD=$MODS/jitunread
CALLS=$MODS/calls
if ! LC_ALL=C awk -v filter="$FILTER" -v tbl="$TBL" -v unread="$UNREAD" -v jit="$JITTBL" \
  -v jitunread="$JITUNREAD" -v callsf="$CALLS" '
BEGIN {
  while ((getline line < tbl) > 0) {
    split(line, f, "\t"); R[f[1]] = f[2]; S[f[1]] = f[3]; P[f[1]] = f[4]
  }
  close(tbl)
  while ((getline line < jit) > 0) {
    if (substr(line, 1, 1) == "#") continue
    split(line, f, "\t"); JR[f[1]] = f[2]; JL[f[1]] = f[3]
  }
  close(jit)
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
function flush(   have, hj) {
  if (name == "") return
  have = (name in R)
  hj = (name in JR)
  if (!have) print name > unread
  if (!hj) print name > jitunread
  if (filter == "" || index(name, filter) > 0) {
    printf "%d\t%-28s %8s %6s %9d %9d %6d %8d %6s %6s %6s %14s %8s %9s\n", (depot ? 0 : 1), name,
           (ntid == "" ? "-" : ntid), (depot ? "YES" : "no"), ldl, stl, fma, cvt,
           (have ? R[name] : "-"), (have ? S[name] : "-"), (have ? P[name] : "-"),
           (have ? blocks(ntid, R[name], S[name]) : "-"),
           (hj ? JR[name] : "-"), (hj ? JL[name] : "-")
    if (calls > 0) printf "%s:%d\n", name, calls > callsf
  }
  name = ""
}
function reset() { ntid = ""; depot = 0; ldl = 0; stl = 0; fma = 0; cvt = 0; calls = 0 }
function tally(line, needle,   n, s) { n = 0; s = line
  while ((i = index(s, needle)) > 0) { n++; s = substr(s, i + length(needle)) }
  return n }
# `call` as its own mnemonic (call, call.uni, predicated), not `.callprototype` or `// callseq`.
# The character after a match is kept, so it cannot pass for the start of the next one.
function tally_call(line,   n, s) { n = 0; s = line
  while (match(s, /(^|[ \t{;])call[. \t]/)) { n++; s = substr(s, RSTART + RLENGTH - 1) }
  return n }
# An entry body ends at the next entry or at the end of its module, never in the next module.
FNR == 1 { flush(); reset() }
/\.visible \.entry / {
  flush(); reset()
  name = $0
  sub(/.*\.visible \.entry /, "", name)
  sub(/\(.*/, "", name)
  next
}
# ...and at any other declaration: a device function (`.func`, a definition or a prototype) is
# not the entry above it, and its instructions do not belong to that entry.
/(^|[ \t])\.(entry|func)([ \t(]|$)/ { flush(); reset(); next }
name != "" {
  if (ntid == "" && index($0, ".reqntid ") > 0) {
    ntid = $0; sub(/.*\.reqntid /, "", ntid); sub(/[^0-9].*/, "", ntid)
  }
  if (index($0, "__local_depot") > 0) depot = 1
  ldl += tally($0, "ld.local")
  stl += tally($0, "st.local")
  fma += tally($0, "fma.")
  cvt += tally($0, "cvt.f32.f16")
  calls += tally_call($0)
}
END { flush() }
' "${MODFILES[@]}" | LC_ALL=C sort -k1,1n -k2,2 | cut -f2- >"$ROWS"; then
  fail "$SEC $TOOLS modules=$NMOD bytescan=failed"
fi
if [ -s "$UNREAD" ]; then
  echo "ptx-scan: ptxas assembled every module but reported nothing for: $(paste -sd, "$UNREAD")" >&2
  fail "$SEC $TOOLS modules=$NMOD ptxas-unread=$(paste -sd, "$UNREAD")"
fi
if [ -s "$JITUNREAD" ]; then
  echo "ptx-scan: the driver loaded every module but reported nothing for: $(paste -sd, "$JITUNREAD")" >&2
  fail "$SEC $TOOLS modules=$NMOD $JITS jit-unread=$(paste -sd, "$JITUNREAD")"
fi
if [ ! -s "$ROWS" ]; then
  echo "ptx-scan: no entry${FILTER:+ name contains $FILTER}" >&2
  fail "$SEC $TOOLS modules=$NMOD${FILTER:+ filter=$FILTER} rows=0"
fi
CALLERS=
if [ -s "$CALLS" ]; then
  CALLERS=$(LC_ALL=C sort "$CALLS" | paste -sd, -)
  echo "ptx-scan: entries that call a device function — their rows count their own body, not the callee's: $CALLERS" >&2
fi
echo "ptx-scan bin=$BIN $SEC $TOOLS modules=$NMOD${FILTER:+ filter=$FILTER} $JITS${CALLERS:+ calls=$CALLERS}"
printf '%-28s %8s %6s %9s %9s %6s %8s %6s %6s %6s %14s %8s %9s\n' \
  entry reqntid depot ld.local st.local fma cvt.f16 regs smem spill 'blk/SM(static)' jit_regs jit_local
cat "$ROWS"
exit 0
