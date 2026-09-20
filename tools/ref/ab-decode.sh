#!/usr/bin/env bash
# 같은 임대 안에서 여러 트리의 bloomery-decode를 번갈아 잰다 (박스에서 실행).
#
#   bash tools/ref/ab-decode.sh <remote-dir>... [현재 트리는 자동 포함]
#
# 막는 실패: 다른 시각의 절대값을 나란히 놓는 비교. 같은 커밋도 창이 바뀌면 5%가
# 움직이므로, 디스패치 경로를 만진 라운드의 속도 판정은 이 스크립트의 상대값으로만 한다.
# 각 <remote-dir>는 ~/repo/ 아래 이름(워크트리에서 `just build-decode`로 만든 것).
set -uo pipefail
MODEL=${BLOOMERY_REF_MODEL:-/models/small/DeepSeek-V2-Lite-Chat.Q3_K_M.gguf}
TOKENS=${BLOOMERY_DECODE_TOKENS:-100000,549,6077,280,7239,317}
N=${BLOOMERY_DECODE_N:-96}
ROUNDS=${BLOOMERY_AB_ROUNDS:-4}
bins=()
for d in "$@" "$(basename "$PWD")"; do
  b="$HOME/repo/$d/target/release/bloomery-decode"
  [ -x "$b" ] || { echo "no decode binary at $b — run just build-decode in that tree" >&2; exit 2; }
  bins+=("$d")
done
witness() {
  echo "--- witness $1 $(date -u +%Y-%m-%dT%H:%M:%SZ) load=$(cut -d' ' -f1-3 /proc/loadavg) io=$(grep '^some' /proc/pressure/io | cut -d' ' -f2) gpu=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader | tr '\n' ' ')"
  # 임대를 모르는 남의 프로세스는 이 줄에서만 보인다(다른 세션의 llama-server가 A/B를 오염시킨 적 있다).
  echo "    busiest: $(ps -eo comm,pcpu --sort=-pcpu --no-headers | head -n 4 | awk '{printf "%s %s%% | ", $1, $2}')"
}
exec 9>/root/bloomery-cpu.lock
flock -w 1800 9 || { echo "[lease] timed out" >&2; exit 75; }
witness pre
for r in $(seq "$ROUNDS"); do
  for d in "${bins[@]}"; do
    out=$("$HOME/repo/$d/target/release/bloomery-decode" -m "$MODEL" --tokens "$TOKENS" -n "$N" 2>&1) || { echo "r$r $d FAILED" >&2; exit 1; }
    toks=$(echo "$out" | grep -E 'decode steps' | sed 's/.*= //;s/ (.*//')
    pre=$(echo "$out" | grep -E '^prefill' | sed 's/.*= //')
    med=$(echo "$out" | awk '/^ +[0-9]+ +[0-9]+ +[0-9.]+ /{print $3}' | sort -n | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}')
    [ -n "$toks" ] || { echo "r$r $d produced no decode line" >&2; exit 1; }
    echo "r$r $d | prefill $pre | decode $toks | median ${med} ms"
  done
done
witness post
