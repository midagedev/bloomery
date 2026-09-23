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
# 모델·ik 트리·llama-bench 기본값(BLOOMERY_REF_MODEL·IK·IKBIN 오버라이드는 그대로 받는다)은 빌드 스크립트와
# 같은 파일이 소유한다.
# shellcheck source=tools/ref/ref-paths.sh
source "${BASH_SOURCE[0]%/*}/ref-paths.sh"
# The lease and the witness fields.
# shellcheck source=tools/ref/lease.sh
source "${BASH_SOURCE[0]%/*}/lease.sh"
N=${BLOOMERY_DECODE_N:-96}
ROUNDS=${BLOOMERY_AB_ROUNDS:-3}
DEPTHS=${BLOOMERY_DEPTHS:-6 1024 4096}
trees=()
for d in "$@" "$(basename "$PWD")"; do
  # A tree is named by its remote directory; a path given by mistake is reduced to that name.
  d=$(basename "$d")
  b=$(decode_bin "$d")
  [ -x "$b" ] || { echo "no decode binary at $b — run just build-decode in that tree" >&2; exit 2; }
  trees+=("$d")
done
WITNESS=(head-load indent busiest model)
lease_take
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
        # The prompt of depth d is lcg_prompt's. Its values do not reach the time: a decode step
        # reads six experts per layer and every cached key whatever the token is.
        out=$("$(decode_bin "$dir")" -m "$MODEL" --tokens "$(lcg_prompt "$dep")" -n "$N" 2>&1) || { echo "r$r $dir d=$dep FAILED" >&2; echo "$out" | tail -n 5 >&2; exit 1; }
        toks=$(echo "$out" | grep -E 'decode steps' | sed 's/.*= //;s/ (.*//')
        pre=$(echo "$out" | grep -E '^prefill' | sed 's/.*= //')
        [ -n "$toks" ] || { echo "r$r $dir d=$dep produced no decode line" >&2; exit 1; }
        echo "r$r $dir d=$dep | prefill $pre | decode $toks"
        sums+=("$dir d=$dep|${toks%% *}")
        ;;
      ik:*)
        dep=${a##*:}
        # 분할이 의도다: IK_BEST_FLAGS는 플래그 여럿을 담은 한 문자열이다(ab-decode.sh와 같다).
        # shellcheck disable=SC2086
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
