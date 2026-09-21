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
set -uo pipefail
NAME=${1:-}
FILTER=${2:-}
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
trap 'rm -f "$PTX"' EXIT
if ! objcopy -O binary --only-section=.oxart "$BIN" "$PTX" 2>/dev/null || [ ! -s "$PTX" ]; then
  echo "ptx-scan: $BIN carries no .oxart section (host-only binary?)" >&2
  exit 0
fi
echo "ptx-scan bin=$BIN section=.oxart bytes=$(wc -c <"$PTX" | tr -d ' ')${FILTER:+ filter=$FILTER}"
printf '%-28s %8s %6s %9s %9s %6s %8s\n' entry reqntid depot ld.local st.local fma cvt.f16
LC_ALL=C tr -d '\000' <"$PTX" | LC_ALL=C awk -v filter="$FILTER" '
function flush() {
  if (name == "") return
  if (filter == "" || index(name, filter) > 0)
    printf "%d\t%-28s %8s %6s %9d %9d %6d %8d\n", (depot ? 0 : 1), name,
           (ntid == "" ? "-" : ntid), (depot ? "YES" : "no"), ldl, stl, fma, cvt
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
