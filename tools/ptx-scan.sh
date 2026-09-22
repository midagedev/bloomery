#!/usr/bin/env bash
# PTX 스캔 — 게이트 바이너리가 싣고 있는 디바이스 코드를 읽는다. 계측기지 게이트가 아니다(항상 exit 0).
#
# `cargo oxide`는 번들의 PTX 텍스트를 실행 파일의 `.oxart` ELF 섹션에 그대로 넣는다. 그래서 커널의
# 컴파일된 모양 — 레지스터가 로컬로 쏟아졌는지(`__local_depot`), 그 왕복이 몇 번인지(ld.local/st.local),
# 블록 폭이 얼마인지(`.reqntid`) — 은 디바이스 없이 바이트 스캔으로 읽힌다. 원인 둘을 이렇게 찾았다:
# flash의 lane별 누산 배열이 디포로 쏟아진 것, rms_norm이 warp 하나만 상주하는 런치 기하.
#
# 단언하는 것은 게이트다. 이 스크립트는 표만 낸다:
#   gate_p5 `no_local_depot`  — flash 커널 다섯의 디포·ld.local·st.local이 전부 0
#   gate_p4 `norm_geometry`   — rms_norm·norm_quant의 .reqntid == RMS_THREADS
#   gate_p4 `argmax_geometry` — argmax의 .reqntid == ARGMAX_THREADS
#
# 철자 주의: PTX는 융합 곱셈-덧셈을 `fma.rn.f32`(와 `fma.rm.f32`)로 쓴다. `fma.f32`는 아무 데도 없어서
# 그걸 세면 조용히 0이 나오고 깨끗한 커널처럼 읽힌다 — 여기서도 게이트에서도 `fma.` 접두를 센다.
#
# 사용: tools/ptx-scan.sh <바이너리 이름> [엔트리 부분문자열]
#   target/release/<바이너리>를 읽는다. 레시피(`just ptx-scan <바이너리>`)가 빌드까지 한다.
# 출력은 디포를 가진 엔트리 먼저, 그 안에서 이름 오름차순 — 디포가 결함이고 나머지는 맥락이다.
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
