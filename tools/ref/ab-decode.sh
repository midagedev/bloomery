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
IKBIN=${IKBIN:-/home/user/ik_llama.cpp/build/bin/llama-bench}
IK_BEST_FLAGS=${IK_BEST_FLAGS:--mla 3 -fa 1 -fmoe 1 -rtr 1}
# BLOOMERY_AB_ENVS="K=V;K=V K2=V2": 현재 트리의 같은 바이너리를 env만 바꿔 팔로 더 넣는다
# (바이트가 같은 레버의 A/B — 빌드 둘의 링크 배치 차이가 끼지 않는다).
IFS=';' read -r -a envs <<< "${BLOOMERY_AB_ENVS:-}"
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
# 러너가 실제로 읽은 값. 맥 셸의 BLOOMERY_AB_* 는 ssh를 그냥 넘지 않는다(레시피가 실어 보낸다) —
# 이 줄이 없으면 "env가 박스에 닿았나"를 바퀴 수를 세어 추측해야 했다.
echo "[config] rounds=$ROUNDS n=$N trees=${bins[*]} envs=${BLOOMERY_AB_ENVS:-} ik=${BLOOMERY_AB_IK:-0}"
witness pre
# 팔 목록: 트리 팔("tree:<dir>")과 env 팔("env:<K=V ...>")을 한 줄로 세우고 바퀴마다 한 칸씩
# 돌린다. 막는 실패: 위치 편향 — 고정 순서에서는 바퀴의 첫 팔이 0.3–0.8% 느리게 나왔고
# (같은 바이너리의 A/A로 확인), 그만큼의 차이를 가진 변경의 판정이 순서에 따라 뒤집힌다.
arms=()
sums=()
for d in "${bins[@]}"; do arms+=("tree:$d"); done
for e in "${envs[@]}"; do [ -n "$e" ] && arms+=("env:$e"); done
here=$(basename "$PWD")
for r in $(seq "$ROUNDS"); do
  for i in $(seq 0 $((${#arms[@]} - 1))); do
    a=${arms[$(((i + r - 1) % ${#arms[@]}))]}
    case $a in
      tree:*) d=${a#tree:}; label=$d; e="" ;;
      env:*) d=$here; e=${a#env:}; label="[$e]" ;;
    esac
    out=$(env $e "$HOME/repo/$d/target/release/bloomery-decode" -m "$MODEL" --tokens "$TOKENS" -n "$N" 2>&1) || { echo "r$r $label FAILED" >&2; exit 1; }
    toks=$(echo "$out" | grep -E 'decode steps' | sed 's/.*= //;s/ (.*//')
    pre=$(echo "$out" | grep -E '^prefill' | sed 's/.*= //')
    med=$(echo "$out" | awk '/^ +[0-9]+ +[0-9]+ +[0-9.]+ /{print $3}' | sort -n | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}')
    [ -n "$toks" ] || { echo "r$r $label produced no decode line" >&2; exit 1; }
    echo "r$r $label | prefill $pre | decode $toks | median ${med} ms"
    sums+=("$label|${toks%% *}")
  done
  # ik 팔(BLOOMERY_AB_IK=1): 가장 빠르게 잰 플래그 조합의 llama-bench를 같은 바퀴 안에 끼운다.
  # 막는 실패: 단발 헤드라인 하나를 ik의 다른 임대 숫자와 비교해 "넘었다"고 쓰는 것 —
  # 1% 안쪽의 차이는 번갈아 잰 표본 여러 개로만 말할 수 있다. -r 1: 바퀴가 곧 반복이다.
  if [ "${BLOOMERY_AB_IK:-0}" = 1 ]; then
    ik=$(CUDA_VISIBLE_DEVICES="" "$IKBIN" -m "$MODEL" -ngl 0 -t 32 -p 0 -n "$N" -r 1 $IK_BEST_FLAGS 2>&1 | grep -E "tg$N" | awk -F'|' '{print $(NF-1)}' | sed 's/ ±.*//;s/ //g')
    [ -n "$ik" ] || { echo "r$r ik produced no tg$N line" >&2; exit 1; }
    echo "r$r ik[$IK_BEST_FLAGS] | decode $ik tok/s"
  fi
done
# 팔별 평균 — 바퀴 표를 눈으로 더하다 틀리지 않게.
printf '%s\n' "${sums[@]}" | awk -F'|' '{s[$1]+=$2; n[$1]++} END{for(k in s) printf "mean %-40s %.2f tok/s (n=%d)\n", k, s[k]/n[k], n[k]}' | sort
witness post
