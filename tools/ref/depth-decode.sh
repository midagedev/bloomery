#!/usr/bin/env bash
# 컨텍스트 깊이별 디코드: 같은 임대 안에서 bloomery(트리별)와 ik를 깊이마다 번갈아 잰다 (박스에서 실행).
#
#   BLOOMERY_DEPTHS="6 1024 4096" bash tools/ref/depth-decode.sh [remote-dir]...
#
# 막는 실패: tg96(깊이 0)의 비교를 "디코드가 빠르다"로 일반화하는 것. 디코드 스텝의 어텐션 항은
# 캐시된 키 수에 선형이고, 깊이 0에서는 스텝의 몇 %라 어느 엔진의 기울기도 헤드라인에 안 보인다.
# bloomery 팔은 깊이 d의 고정 의사난수 프롬프트를 프리필한 뒤 N 스텝을 재고, ik 팔은
# `llama-bench -gp d,N`(pp d 뒤의 tg N)이다. 팔은 (엔진, 깊이) 쌍이고 바퀴마다 한 칸씩 돈다.
# 트리 팔: 인자 <remote-dir>... — ab-decode.sh의 방식이다(현재 트리 자동 포함, 각 트리는
# 워크트리에서 `just build-decode`로 만든 것). 결과 줄과 평균 줄은 트리 이름을 단다.
set -uo pipefail
MODEL=${BLOOMERY_REF_MODEL:-/models/small/DeepSeek-V2-Lite-Chat.Q3_K_M.gguf}
N=${BLOOMERY_DECODE_N:-96}
ROUNDS=${BLOOMERY_AB_ROUNDS:-3}
DEPTHS=${BLOOMERY_DEPTHS:-6 1024 4096}
IKBIN=${IKBIN:-/home/user/ik_llama.cpp/build/bin/llama-bench}
IK_BEST_FLAGS=${IK_BEST_FLAGS:--mla 3 -fa 1 -fmoe 1 -rtr 1}
trees=()
for d in "$@" "$(basename "$PWD")"; do
  b="$HOME/repo/$d/target/release/bloomery-decode"
  [ -x "$b" ] || { echo "no decode binary at $b — run just build-decode in that tree" >&2; exit 2; }
  trees+=("$d")
done
witness() {
  echo "--- witness $1 $(date -u +%Y-%m-%dT%H:%M:%SZ) load=$(cut -d' ' -f1-3 /proc/loadavg) io=$(grep '^some' /proc/pressure/io | cut -d' ' -f2) gpu=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader | tr '\n' ' ')"
  echo "    busiest: $(ps -eo comm,pcpu --sort=-pcpu --no-headers | head -n 4 | awk '{printf "%s %s%% | ", $1, $2}')"
}
# 깊이 d의 프롬프트: BOS(100000) 뒤에 LCG 난수 id [1000, 91000). 값은 시간에 안 걸린다 —
# 디코드 스텝은 어떤 토큰이든 층마다 전문가 여섯과 캐시된 키 전부를 읽는다.
prompt() {
  awk -v n="$1" 'BEGIN{s=12345; printf "100000"; for(i=1;i<n;i++){s=(s*1103515245+12345)%2147483648; printf ",%d", 1000+(s%90000)}}'
}
exec 9>/root/bloomery-cpu.lock
flock -w 1800 9 || { echo "[lease] timed out" >&2; exit 75; }
witness pre
# 팔 목록: 깊이마다 트리 팔("tree:<dir>:<depth>")을 세운 뒤 ik 팔 — 같은 깊이의 비교가 임대 안에서
# 이웃하도록. 바퀴마다 한 칸씩 돈다(위치 편향 — ab-decode.sh와 같은 이유).
arms=()
sums=()
for dep in $DEPTHS; do
  for d in "${trees[@]}"; do arms+=("tree:$d:$dep"); done
  arms+=("ik:$dep")
done
for r in $(seq "$ROUNDS"); do
  for i in $(seq 0 $((${#arms[@]} - 1))); do
    a=${arms[$(((i + r - 1) % ${#arms[@]}))]}
    case $a in
      tree:*)
        rest=${a#tree:}; dir=${rest%:*}; dep=${rest##*:}
        out=$("$HOME/repo/$dir/target/release/bloomery-decode" -m "$MODEL" --tokens "$(prompt "$dep")" -n "$N" 2>&1) || { echo "r$r $dir d=$dep FAILED" >&2; echo "$out" | tail -n 5 >&2; exit 1; }
        toks=$(echo "$out" | grep -E 'decode steps' | sed 's/.*= //;s/ (.*//')
        pre=$(echo "$out" | grep -E '^prefill' | sed 's/.*= //')
        [ -n "$toks" ] || { echo "r$r $dir d=$dep produced no decode line" >&2; exit 1; }
        echo "r$r $dir d=$dep | prefill $pre | decode $toks"
        sums+=("$dir d=$dep|${toks%% *}")
        ;;
      ik:*)
        dep=${a##*:}
        raw=$(CUDA_VISIBLE_DEVICES="" "$IKBIN" -m "$MODEL" -ngl 0 -t 32 -p 0 -n 0 -gp "$dep,$N" -r 1 $IK_BEST_FLAGS 2>&1)
        ik=$(echo "$raw" | grep -E "tg$N@pp$dep" | awk -F'|' '{print $(NF-1)}' | sed 's/ ±.*//;s/ //g')
        [ -n "$ik" ] || { echo "r$r $a produced no tg$N@pp$dep line" >&2; echo "$raw" | tail -n 8 >&2; exit 1; }
        echo "r$r ik d=$dep [$IK_BEST_FLAGS] | decode $ik tok/s"
        sums+=("ik d=$dep|$ik")
        ;;
    esac
  done
done
printf '%s\n' "${sums[@]}" | awk -F'|' '{s[$1]+=$2; n[$1]++} END{for(k in s) printf "mean %-24s %.2f tok/s (n=%d)\n", k, s[k]/n[k], n[k]}' | sort
witness post
