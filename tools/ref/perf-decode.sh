#!/usr/bin/env bash
# 디코드 스텝의 스레드별 perf 표를, 박스 전역 임대 안에서, 증인을 남기고 뜬다 (박스에서 실행).
#
#   bash tools/ref/perf-decode.sh [<remote-dir>]     # 기본은 이 트리
#
# 막는 실패 둘. ① 시작·정리를 스텝 비용으로 읽는 것: 레코드는 디코드 루프 안에서만 뜬다 —
# 바이너리가 스텝 표 머리줄을 찍은(= 프리필이 끝난) 뒤에 pid에 붙고, PERF_SECS 뒤에 떨어진다.
# N은 그 창이 루프 안에 끝나도록 넉넉히 준다(13 ms/스텝이면 400 스텝이 5 s — perf가 붙는 데도 시간이 든다). 루프가 창보다
# 먼저 끝나면 표 끝에 적는다. ② 프로파일 표를 손으로 뜨는 것: 표도 측정이라 임대 안에서만.
# 이벤트는 cpu-clock이다(AGENTS.md: 기본 IBS 이벤트는 이 CPU에서 심볼을 잘못 귀속한다).
set -euo pipefail
# shellcheck source=tools/ref/ref-paths.sh
source "${BASH_SOURCE[0]%/*}/ref-paths.sh"
export BLOOMERY_DATA
# shellcheck source=tools/ref/lease.sh
source "${BASH_SOURCE[0]%/*}/lease.sh"
TOKENS=${BLOOMERY_DECODE_TOKENS:-$REF_TOKENS}
N=${BLOOMERY_DECODE_N:-400}
SECS=${PERF_SECS:-2.0}
TOP=${PERF_TOP:-25}
if [ -n "${1:-}" ]; then BIN=$(decode_bin "$1"); else BIN=${BLOOMERY_DECODE_BIN:-$DECODE_BIN}; fi
[ -x "$BIN" ] || { echo "no decode binary at $BIN — run: just build-decode" >&2; exit 2; }
BIN_SHA=$(sha256sum "$BIN" | cut -c1-12)
OUT=$(mktemp -d /tmp/perf-decode.XXXXXX)

WITNESS=(head loadavg pressure-cpu threads model binary lock-holder busiest)
lease_take
witness pre
# Bounded like every child under the lease (lease_bounded's bound); perf attaches to the decode binary
# itself, the bound's child, whose pid is the main thread's tid in the tables below.
timeout --kill-after=10 "$LEASE_ARM_BOUND" "$BIN" -m "$MODEL" --tokens "$TOKENS" -n "$N" > "$OUT/decode.log" 2>&1 &
bound_pid=$!
pid=
while [ -z "$pid" ]; do
  kill -0 "$bound_pid" 2> /dev/null || { echo "decode exited before perf could attach" >&2; cat "$OUT/decode.log" >&2; exit 1; }
  pid=$(awk '{ print $1 }' "/proc/$bound_pid/task/$bound_pid/children" 2> /dev/null || true)
  [ -n "$pid" ] || sleep 0.01
done
# The step table's header is printed after the prefill returns; stdout is line-buffered.
until grep -q '^step ' "$OUT/decode.log" 2> /dev/null; do
  kill -0 "$pid" 2> /dev/null || { echo "decode exited before its step table" >&2; cat "$OUT/decode.log" >&2; exit 1; }
  sleep 0.01
done
# PERF_CALLERS="<symbol>...": record the main thread alone with DWARF call chains instead, and print
# each symbol's callers — a memmove or memset row names no call site on its own.
if [ -n "${PERF_CALLERS:-}" ]; then
  target=(-t "$pid" --call-graph "dwarf,16384")
else
  target=(-p "$pid")
fi
lease_bounded "$LEASE_ARM_BOUND" perf record -q -e cpu-clock -F 10000 "${target[@]}" -o "$OUT/perf.data" -- sleep "$SECS" 2> "$OUT/perf.err" || echo "[perf] record rc=$?: $(tr '\n' ' ' < "$OUT/perf.err")"
if kill -0 "$pid" 2> /dev/null; then inside=yes; else inside=no; fi
wait "$bound_pid"
witness post
steps_done=$(grep -cE '^ +[0-9]+ +[0-9]+ +[0-9.]+ ' "$OUT/decode.log")
grep -E 'decode steps in|per step' "$OUT/decode.log"
echo "[perf] window ${SECS}s after the prefill; decode still running when the window closed: $inside ($steps_done of $N steps)"
# The tables come from `perf script`, not `perf report --tid`: on this perf (6.8) the --tid view of
# a `-p` record dropped every qdot kernel from the main thread and showed it as 73 % barrier spin, while
# its own samples, listed one by one, were 68 % dot kernels.
ms_step=$(grep -E 'decode steps in' "$OUT/decode.log" | sed 's/.*mean \([0-9.]*\) ms.*/\1/')
lease_bounded "$LEASE_ARM_BOUND" perf script -i "$OUT/perf.data" -F tid,sym 2> /dev/null | awk -v main="$pid" '{print ($1 == main ? "main" : "workers"), $2}' \
  | sed 's/\.llvm\.[0-9]*$//' > "$OUT/samples.txt"
for who in main workers; do
  echo
  echo "=== $who, symbols: % of that group's samples, ms/step [derived: share x ${ms_step} ms/step$( [ "$who" = workers ] && echo ', per worker')] ==="
  awk -v who="$who" -v ms="$ms_step" -v top="$TOP" '
    $1 == who { n[$2]++; t++ }
    END {
      for (s in n) printf "%7.2f%% %7.3f  %s\n", 100 * n[s] / t, ms * n[s] / t, s | "sort -rn | head -n " top
      close("sort -rn | head -n " top)
      printf "  (%d samples)\n", t
    }' "$OUT/samples.txt"
done
# PERF_ANNOTATE="<symbol>..." prints each symbol's hottest instructions over every thread that ran it —
# the question a symbol table cannot answer when a closure is inlined into its dispatcher (a spin loop
# and a chunk's own rows are one symbol then).
for sym in ${PERF_ANNOTATE:-}; do
  echo
  echo "=== hottest instructions of $sym ==="
  lease_bounded "$LEASE_ARM_BOUND" perf annotate -i "$OUT/perf.data" --stdio -s "$sym" 2> /dev/null \
    | grep -E '^ +[0-9]+\.[0-9]+ +:' | sort -rn | head -n 12 || true
done
for sym in ${PERF_CALLERS:-}; do
  echo
  echo "=== main thread, callers of $sym ==="
  lease_bounded "$LEASE_ARM_BOUND" perf report -i "$OUT/perf.data" --stdio --no-children --percentage relative -S "$sym" -G 2> /dev/null \
    | grep -vE '^#|^$' | head -n 60 || true
done
# PERF_KEEP=1 leaves the record for reading afterwards (reading is not measuring; recording is).
if [ "${PERF_KEEP:-0}" = 1 ]; then echo "[perf] kept $OUT (main tid $pid)"; else rm -rf "$OUT"; fi
