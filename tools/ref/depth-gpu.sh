#!/usr/bin/env bash
# 컨텍스트 깊이별 GPU 디코드: 같은 임대 안에서 우리 엔진(generate)과 ik를 깊이마다 번갈아 잰다 (박스에서 실행).
#
#   BLOOMERY_GPU_ARMS="6:512 ik:0 1024:1120 ik:1024 4096:4192 ik:4096" bash tools/ref/depth-gpu.sh
#
# depth-decode.sh의 GPU 형제다. 그쪽은 CPU 엔진(bloomery-decode)의 트리 팔을 세우고, 이쪽은
# GPU 엔진의 `generate --time`을 세운다. 막는 실패는 같다: 깊이 0의 비율을 "디코드가 빠르다"로
# 일반화하는 것 — 디코드 스텝의 어텐션 항은 캐시된 키 수에 선형이라 깊이 0에서는 어느 엔진의
# 기울기도 헤드라인에 안 보인다. CPU 선에서 같은 실수가 "+2.2%"를 깊이 4096의 −39%로 뒤집었다.
#
# 팔 문법(순서가 곧 임대 안의 배치다. 바퀴마다 한 칸씩 돈다 — 위치 편향, ab-decode.sh와 같은 이유):
#   ab:<우리 팔>        같은 팔을 `--time`이 아니라 `--ab`로 돈다(안쪽 바퀴 수 BLOOMERY_AB_INNER).
#                        `--ab`에는 워밍 라운드가 있고 `--time`에는 없다 — 둘을 같은 임대 안에
#                        번갈아 세워야 그 차이가 계기 차이인지 창 표류인지 갈린다. 보고되는 값은
#                        base 팔(레버 전부 끔 = 실제 경로)의 바퀴별 p50 평균이다.
#                        BLOOMERY_AB_SET(기본 비어 있음)을 주면 `--ab-set`으로 넘기고, ROW 옆에
#                        모든 팔의 `ab` 줄을 `ARM` 접두로 같이 찍는다 — keyaxis 세트에서는
#                        base가 아니라 팔들의 차가 질문이기 때문이다.
#   toks=<id,id,...>:<ctx>[:<n>]  우리 팔인데 프롬프트를 LCG가 아니라 리터럴 id로 준다.
#                        헤드라인을 낸 그 프롬프트(id 0 = "The capital of France is")로 계기를
#                        재현할 때 쓴다 — LCG 팔과 같은 길이에서 값이 같아야 "토큰 값은 시간에
#                        안 걸린다"가 주장이 아니라 관측이 된다.
#   seed=<깊이>:<ctx>[:<n>]  우리 팔인데 프롬프트를 디코드하지 않고 KV 캐시를 직접 채워 그 깊이에
#                        선다(generate --seed-depth). 같은 깊이의 리터럴/LCG 팔과 pos·n_keys가 같고
#                        준비 시간만 없다 — 깊이 4096에서 팔 하나의 준비가 16초에서 0에 가까워진다.
#                        찍히는 토큰은 무의미하다(씨앗 행은 모델이 쓴 키가 아니다). 시간만 읽는다.
#                        `ab:seed=`는 없다 — --ab의 팔마다 reset()이 씨앗을 지운다(generate가 거부한다).
#                        씨앗 팔은 씨앗 팔과만 비교한다: 4바퀴 대조에서 깊이 4096의 씨앗 팔이 매 바퀴 0.7% 빨랐다
#                        (자 ±1% 안이지만 잡음이 아닌 편향 — 값 압축성이나 준비 시간 차의 발열이 후보). 씨앗 µs를
#                        진짜 프롬프트 기준선 옆에 놓지 않는다.
#   env=<K=V[,K=V]>:<나머지 팔>  우리 팔인데 그 호출에만 환경 변수를 건다. 레버 하나만 다른 두 팔을
#                        같은 임대 안에 번갈아 세우려고 있다(ab-decode.sh의 BLOOMERY_AB_ENVS와 같은 이유):
#                        커널 선택처럼 프로세스 시작 때 한 번 읽히는 레버는 프로세스를 갈라야 갈린다.
#                        변수 목록이 팔 이름에 붙으므로 평균 표에서 두 팔이 섞이지 않는다.
#   <깊이>:<ctx>[:<n>]   우리 팔. ctx는 generate의 --ctx이고, 세그먼트 수를 정한다
#                        (flash::segments_for(ctx) = ceil(ctx/seg_keys), seg_keys는 기본 텐서 코어
#                        패스에서 64, BLOOMERY_FLASH_MMA=0 스칼라 패스에서 128 — generate의 load 줄이
#                        찍는다). ctx는 깊이가 아니라 캐시 높이라
#                        죽은 세그먼트도 블록을 런치한다 — 그래서 ctx는 팔마다 명시한다.
#   ik:<깊이>            ik 팔. 깊이 0은 llama-bench의 평범한 tg N이고, 그 위는 -gp <깊이>,N이다.
#                        llama-bench는 cparams.n_ctx = n_prompt + n_gen으로 스스로 ctx를 맞추므로
#                        (examples/llama-bench/llama-bench.cpp:1046), 우리 팔의 ctx = 깊이+n이 같은 모양이다.
#
# 우리 팔의 프롬프트는 depth-decode.sh와 같은 LCG 의사난수 id다(BOS 뒤 고정 수열). 반복 프롬프트보다
# 캐시에 덜 친절하고, CPU 깊이 표가 쓴 것과 같은 프롬프트라 두 표가 같은 말을 한다.
set -uo pipefail
MODEL=${BLOOMERY_REF_MODEL:-/models/small/DeepSeek-V2-Lite-Chat.Q3_K_M.gguf}
N=${BLOOMERY_DECODE_N:-96}
# --time 팔의 워밍(generate --warm W): 앞 W 스텝은 돌되 통계에서 빠진다. 비면 안 붙는다. ab 팔에는
# 안 붙인다 — --ab는 팔마다 untimed 라운드를 이미 돌리고, generate가 둘을 같이 주면 거부한다.
WARM=${BLOOMERY_GEN_WARM:-}
ROUNDS=${BLOOMERY_AB_ROUNDS:-3}
# ik 트리·llama-bench 기본값(IK·IKBIN 오버라이드는 그대로 받는다)은 빌드 스크립트와 같은 파일이 소유한다.
# shellcheck source=tools/ref/ref-paths.sh
source "${BASH_SOURCE[0]%/*}/ref-paths.sh"
# 3090 기준선(216.6/204.6/189.7)이 쓴 그 조합. CPU 최속 조합에서 -rtr 1만 뺀 것이고 GPU 플래그
# 스윕은 아직 없다 — "이 플래그에서"의 숫자다.
IK_GPU_FLAGS=${IK_GPU_FLAGS:--ngl 99 -mla 3 -fa 1 -fmoe 1}
BIN=${BLOOMERY_GEN_BIN:-target/release/generate}
ARMS_SPEC=${BLOOMERY_GPU_ARMS:-6:512 ik:0 1024:1120 ik:1024 4096:4192 ik:4096}
# 카드 핀·증인 줄·옆 카드 판정·바이너리 신선도는 러너 넷이 같은 파일에서 읽는다.
# shellcheck source=tools/ref/timing-card.sh
source "${BASH_SOURCE[0]%/*}/timing-card.sh"
LOCK=/root/bloomery-cpu.lock
assert_fresh_binary "$BIN" || exit $?
[ -x "$IKBIN" ] || { echo "no llama-bench at $IKBIN" >&2; exit 2; }

# 행마다 전후로 남기는 증인. 공통 줄은 timing-card.sh가 낸다 — 여기서는 머리와 이 러너만 쓰는
# busiest 줄을 더한다.
witness() {
  echo "--- witness $1 $(now)"
  witness_card
  echo "    busiest: $(ps -eo comm,pcpu --sort=-pcpu --no-headers | head -n 4 | awk '{printf "%s %s%% | ", $1, $2}')"
}

# 깊이 d의 프롬프트: BOS(100000) 뒤에 LCG 난수 id [1000, 91000). depth-decode.sh와 같은 수열이다.
prompt() {
  awk -v n="$1" 'BEGIN{s=12345; printf "100000"; for(i=1;i<n;i++){s=(s*1103515245+12345)%2147483648; printf ",%d", 1000+(s%90000)}}'
}
med() { sort -n | awk '{a[NR]=$1} END{if(NR)print a[int((NR+1)/2)]}'; }

exec 9>"$LOCK"
echo "[lease] waiting for $LOCK ..."
flock -w 1800 9 || { echo "[lease] timed out after 30 min"; exit 75; }
echo "[lease] held by pid $$ at $(now)"
echo "[config] model=$MODEL n=$N rounds=$ROUNDS warm=${WARM:-0} ik_flags=$IK_GPU_FLAGS"
echo "[config] arms=$ARMS_SPEC"
echo "[config] timing_gpu=$TIMING_GPU other_gpu=$OTHER_GPU CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
witness pre
guard_other

read -r -a arms <<< "$ARMS_SPEC"
sums=()
for r in $(seq "$ROUNDS"); do
  for i in $(seq 0 $((${#arms[@]} - 1))); do
    a=${arms[$(((i + r - 1) % ${#arms[@]}))]}
    guard_other
    case $a in
      ik:*)
        dep=${a#ik:}
        witness "pre r$r ik d=$dep"
        if [ "$dep" = 0 ]; then
          # shellcheck disable=SC2086
          raw=$("$IKBIN" -m "$MODEL" -p 0 -n "$N" -r 1 $IK_GPU_FLAGS 2>&1)
          val=$(echo "$raw" | grep -E "tg$N" | awk -F'|' '{print $(NF-1)}' | sed 's/ ±.*//;s/ //g')
        else
          # shellcheck disable=SC2086
          raw=$("$IKBIN" -m "$MODEL" -p 0 -n 0 -gp "$dep,$N" -r 1 $IK_GPU_FLAGS 2>&1)
          val=$(echo "$raw" | grep -E "tg$N@pp$dep" | awk -F'|' '{print $(NF-1)}' | sed 's/ ±.*//;s/ //g')
        fi
        witness "post r$r ik d=$dep"
        [ -n "$val" ] || { echo "r$r $a produced no tg line" >&2; echo "$raw" | tail -n 8 >&2; exit 1; }
        echo "ROW r$r ik d=$dep | tok/s $val"
        sums+=("ik d=$dep|$val||")
        ;;
      *)
        use_ab=0
        case $a in ab:*) use_ab=1; a=${a#ab:} ;; esac
        arm_env=(); env_tag=
        case $a in
          env=*) rest0=${a#env=}; IFS=',' read -r -a arm_env <<< "${rest0%%:*}"; a=${rest0#*:}
                 env_tag="+$(echo "${arm_env[*]}" | tr ' ' '+')" ;;
        esac
        case $a in
          seed=*)
            rest=${a#seed=}; dep=${rest%%:*}; rest=${rest#*:}; ctx=${rest%%:*}
            toks=""; label="seed$env_tag"
            ;;
          toks=*)
            rest=${a#toks=}; toks=${rest%%:*}; rest=${rest#*:}; ctx=${rest%%:*}
            dep=$(echo "$toks" | awk -F, '{print NF}'); label="lit$env_tag"
            ;;
          *)
            dep=${a%%:*}; rest=${a#*:}; ctx=${rest%%:*}
            toks=$(prompt "$dep"); label="lcg$env_tag"
            ;;
        esac
        n=$N; case $rest in *:*) n=${rest#*:} ;; esac
        inst="time"; [ "$use_ab" = 1 ] && inst=ab  # 라벨 문자열(--time 모드)이다, time 명령이 아니다
        # --ab의 팔마다 reset()이 캐시를 0으로 되감아 씨앗을 지운다: 조용히 깊이 1을 재게 된다.
        if [ "$use_ab" = 1 ] && [[ "$label" == seed* ]]; then
          echo "ab:seed= 는 없다 — --ab의 팔마다 reset()이 씨앗을 지운다. seed=<깊이>:<ctx> 를 쓴다." >&2
          exit 2
        fi
        witness "pre r$r ours($label,$inst) d=$dep ctx=$ctx n=$n"
        t0=$(date +%s)
        if [ "$use_ab" = 1 ]; then
          out=$(env ${arm_env[@]+"${arm_env[@]}"} "$BIN" --tokens "$toks" -n "$n" --ctx "$ctx" --ab "${BLOOMERY_AB_INNER:-3}" ${BLOOMERY_AB_SET:+--ab-set "$BLOOMERY_AB_SET"} 2>&1)
        elif [[ "$label" == seed* ]]; then
          out=$(env ${arm_env[@]+"${arm_env[@]}"} "$BIN" --seed-depth "$dep" -n "$n" --ctx "$ctx" --time ${WARM:+--warm "$WARM"} 2>&1)
        else
          out=$(env ${arm_env[@]+"${arm_env[@]}"} "$BIN" --tokens "$toks" -n "$n" --ctx "$ctx" --time ${WARM:+--warm "$WARM"} 2>&1)
        fi
        rc=$?
        t1=$(date +%s)
        witness "post r$r ours($label,$inst) d=$dep ctx=$ctx n=$n"
        if [ $rc -ne 0 ]; then
          echo "r$r ours($label,$inst) d=$dep ctx=$ctx FAILED rc=$rc" >&2
          echo "$out" | tail -n 20 >&2
          exit 1
        fi
        if [ "$use_ab" = 1 ]; then
          # base 팔 = 레버 전부 끔 = 실제 경로. 나머지 팔은 이 라운드의 질문이 아니다.
          base=$(echo "$out" | grep -E '^ab arm=base ')
          [ -n "$base" ] || { echo "r$r ours($label,ab) produced no base arm line" >&2; echo "$out" | tail -n 20 >&2; exit 1; }
          p50=$(echo "$base" | sed 's/.*mean_p50_ms=\([0-9.]*\).*/\1/')
          sdp=$(echo "$base" | sed 's/.*sd_pct=\([0-9.-]*\).*/\1/')
          echo "ROW r$r ours($label,ab) d=$dep ctx=$ctx n=$n | base mean_p50 ${p50} ms | sd ${sdp}% | tok/s $(awk -v p="$p50" 'BEGIN{printf "%.2f", 1e3/p}') | inner_rounds ${BLOOMERY_AB_INNER:-3} | wall $((t1 - t0))s"
          # 평균 줄의 첫 열은 `--time` 행에서 mean_ms 기반이다. ab 행이 거기 내놓는 것은
          # base 팔의 바퀴별 p50 평균이므로, 통계 이름을 키에 박아 두 행을 같은 열에서
          # 잘못 읽지 않게 한다(열 하나에 통계 둘이 들어가는 것이 이 표의 유일한 함정이다).
          sums+=("ours($label,ab:base_p50) d=$dep ctx=$ctx n=$n|$(awk -v p="$p50" 'BEGIN{printf "%.4f", 1e3/p}')|$(awk -v p="$p50" 'BEGIN{printf "%.4f", 1e3/p}')|")
          # 팔 세트를 준 라운드는 팔들의 차가 질문이므로 모든 ab 줄을 그대로 남긴다.
          if [ -n "${BLOOMERY_AB_SET:-}" ]; then
            echo "$out" | grep -E '^ab (round|arm)=' | sed "s/^/ARM r$r d=$dep | /"
          fi
          continue
        fi
        smoke=$(echo "$out" | grep -E '^SMOKE ')
        [ -n "$smoke" ] || { echo "r$r ours($label) d=$dep ctx=$ctx produced no SMOKE line" >&2; echo "$out" | tail -n 20 >&2; exit 1; }
        p50=$(echo "$smoke" | sed 's/.*p50_ms=\([0-9.]*\).*/\1/')
        mean=$(echo "$smoke" | sed 's/.*mean_ms=\([0-9.]*\).*/\1/')
        nodes=$(echo "$out" | grep -E '^capture graph_nodes=' | sed 's/.*=//')
        # 워밍 질문을 행마다 공짜로 답하게 한다: --time에는 워밍 라운드가 없으므로, 클럭 램프가
        # 기제라면 첫 열 스텝이 마지막 열 스텝보다 느려야 하고 깊이가 깊을수록(프리필이 곧 워밍이다)
        # 그 차이가 작아야 한다. 스텝별 ms는 --time이 이미 찍고 있다.
        series=$(echo "$out" | awk '/^time step /{sub(/.*ms=/,""); print}')
        h10=$(echo "$series" | head -n 10 | med)
        t10=$(echo "$series" | tail -n 10 | med)
        last=$(echo "$out" | grep -cE '^step ')
        uniq_tok=$(echo "$out" | awk '/^step /{print $NF}' | sort -u | wc -l | tr -d ' ')
        warmcol=$(echo "$smoke" | sed -n 's/.*warm=\([0-9]*\).*/\1/p')
        ikref=$(echo "$out" | sed -n 's/^reference ik \([0-9.]*\) tok\/s at depth \([0-9]*\).*/\1@\2/p')
        echo "ROW r$r ours($label) d=$dep ctx=$ctx n=$n | p50 ${p50} ms | mean ${mean} ms | warm ${warmcol:-0} | ik_ref ${ikref:-?} | tok/s(p50) $(awk -v p="$p50" 'BEGIN{printf "%.2f", 1e3/p}') | tok/s(mean) $(awk -v m="$mean" 'BEGIN{printf "%.2f", 1e3/m}') | nodes ${nodes:-?} | first10_p50 ${h10} | last10_p50 ${t10} | wall $((t1 - t0))s | steps ${last} | distinct_tokens ${uniq_tok}"
        sums+=("ours($label) d=$dep ctx=$ctx n=$n|$(awk -v m="$mean" 'BEGIN{printf "%.4f", 1e3/m}')|$(awk -v p="$p50" 'BEGIN{printf "%.4f", 1e3/p}')|$uniq_tok")
        ;;
    esac
  done
done
echo
echo "=== 팔별 평균 (tok/s). 첫 열: --time 행은 mean_ms 기반(교차 엔진 비율용), ab:base_p50 행은"
echo "    base 팔의 바퀴별 p50 평균 기반. 끝 열의 p50은 계열 비교용이다."
echo "    distinct_tokens = 그 팔이 뽑은 서로 다른 토큰 수(바퀴 간 범위). 팔마다 다르면 MoE 전문가 집합이"
echo "    달라 시간이 교란된다 — ROW에 이미 있던 열을 평균 표에도 세운다. ik·ab 팔에는 없다. ==="
printf '%s\n' "${sums[@]}" | awk -F'|' '{
  s[$1]+=$2; n[$1]++; if($3!=""){sp[$1]+=$3; np[$1]++}
  if(mn[$1]==""||$2+0<mn[$1]+0)mn[$1]=$2; if(mx[$1]==""||$2+0>mx[$1]+0)mx[$1]=$2
  if($4!=""){if(tmn[$1]==""||$4+0<tmn[$1]+0)tmn[$1]=$4; if(tmx[$1]==""||$4+0>tmx[$1]+0)tmx[$1]=$4}
} END{for(k in s){
  spread = (mn[k]>0) ? 100*(mx[k]-mn[k])/mn[k] : 0
  dtok = (tmn[k]=="") ? "-" : ((tmn[k]==tmx[k]) ? tmn[k] : tmn[k] ".." tmx[k])
  printf "mean %-26s %8.2f tok/s(mean_ms)  [%s..%s, spread %.2f%%]  distinct_tokens %-7s %s (n=%d)\n", k, s[k]/n[k], mn[k], mx[k], spread, dtok, (np[k]?sprintf("%.2f tok/s(p50)", sp[k]/np[k]):""), n[k]}}' | sort
witness post
