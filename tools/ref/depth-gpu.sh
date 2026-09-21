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
#   toks=<id,id,...>:<ctx>[:<n>]  우리 팔인데 프롬프트를 LCG가 아니라 리터럴 id로 준다.
#                        헤드라인을 낸 그 프롬프트(id 0 = "The capital of France is")로 계기를
#                        재현할 때 쓴다 — LCG 팔과 같은 길이에서 값이 같아야 "토큰 값은 시간에
#                        안 걸린다"가 주장이 아니라 관측이 된다.
#   <깊이>:<ctx>[:<n>]   우리 팔. ctx는 generate의 --ctx이고, 세그먼트 수를 정한다
#                        (flash::segments_for(ctx) = ceil(ctx/128)). ctx는 깊이가 아니라 캐시 높이라
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
ROUNDS=${BLOOMERY_AB_ROUNDS:-3}
IKBIN=${IKBIN:-/home/user/ik_llama.cpp/build/bin/llama-bench}
# 3090 기준선(216.6/204.6/189.7)이 쓴 그 조합. CPU 최속 조합에서 -rtr 1만 뺀 것이고 GPU 플래그
# 스윕은 아직 없다 — "이 플래그에서"의 숫자다.
IK_GPU_FLAGS=${IK_GPU_FLAGS:--ngl 99 -mla 3 -fa 1 -fmoe 1}
BIN=${BLOOMERY_GEN_BIN:-target/release/generate}
ARMS_SPEC=${BLOOMERY_GPU_ARMS:-6:512 ik:0 1024:1120 ik:1024 4096:4192 ik:4096}
GPU_3090=GPU-307fa0f6-daae-24e5-6fd3-cd50620de6b1
GPU_A6000=GPU-8c129fa6-7382-35a5-2464-9ff01d99fcd4
# 시간을 재는 카드는 A6000이다(사용자, 2026-09-22 — 3090이 같은 날 부하 아래서 두 번 버스에서 떨어져
# 기준선을 A6000에서 다시 잡았다). 우리 바이너리와 ik 둘 다 이 카드에서 돈다 — env 파일의 3090 핀을 덮어쓴다.
# 3090은 게이트·빌드 카드다. 옛 3090 수치와 이 카드 수치는 같은 표에 놓지 않는다(증인이 카드를 적는다).
TIMING_GPU=${BLOOMERY_TIMING_GPU:-$GPU_A6000}
OTHER_GPU=$GPU_3090; [ "$TIMING_GPU" = "$GPU_3090" ] && OTHER_GPU=$GPU_A6000
export CUDA_VISIBLE_DEVICES=$TIMING_GPU
LOCK=/root/bloomery-cpu.lock
[ -x "$BIN" ] || { echo "no generate binary at $BIN — build it with cargo oxide first" >&2; exit 2; }
[ -x "$IKBIN" ] || { echo "no llama-bench at $IKBIN" >&2; exit 2; }

now() { date -u +%Y-%m-%dT%H:%M:%SZ; }

# 행마다 전후로 남기는 증인. 시간 카드의 컴퓨트 앱이 자기 프로세스뿐인지가 핵심이고, loadavg는
# 신호가 아니다(rig-log docs/quiet-machine.md) — IO 압력과 실제 프로세스 목록이 신호다.
# 첫 줄이 카드 이름과 전력 제한이다: 카드가 바뀐 날부터 카드 없는 숫자는 뜻이 없다.
witness() {
  echo "--- witness $1 $(now)"
  echo "    timing-card: $(nvidia-smi --query-gpu=name,power.limit,clocks.max.sm --format=csv,noheader -i "$TIMING_GPU")"
  echo "    3090-apps: [$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i "$GPU_3090" | tr '\n' ';')]"
  echo "    a6000-apps: [$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i "$GPU_A6000" | tr '\n' ';')]"
  echo "    gpu: $(nvidia-smi --query-gpu=index,utilization.gpu,power.draw,clocks.sm --format=csv,noheader | tr '\n' ';')"
  echo "    load=$(cut -d' ' -f1-3 /proc/loadavg) io=$(grep '^some' /proc/pressure/io | cut -d' ' -f2) llm.service=$(systemctl is-active llm.service || true)"
  echo "    busiest: $(ps -eo comm,pcpu --sort=-pcpu --no-headers | head -n 4 | awk '{printf "%s %s%% | ", $1, $2}')"
}

# 두 카드가 다 우리 것이다(사용자, 2026-09-22). 시간을 안 재는 옆 카드에 컴퓨트 앱이 있으면 우리 다른
# 라운드의 게이트·빌드일 가능성이 크다. 중단하지 않고 **증인에 남긴다** — 시간 카드의 수치가 옆 카드
# 부하에 흔들리는지는 이 증인 열로 나중에 판정한다(아직 잰 적 없다). 중단이 필요하면 BLOOMERY_OTHER_STRICT=1.
guard_other() {
  local apps
  apps=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i "$OTHER_GPU")
  [ -n "$apps" ] || return 0
  echo "[other-busy] $(now) 옆 카드($OTHER_GPU)에 컴퓨트 앱: [$(echo "$apps" | tr '\n' ';')]" >&2
  if [ "${BLOOMERY_OTHER_STRICT:-}" = 1 ]; then
    witness abort-other >&2
    exit 75
  fi
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
echo "[config] model=$MODEL n=$N rounds=$ROUNDS ik_flags=$IK_GPU_FLAGS"
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
        sums+=("ik d=$dep|$val")
        ;;
      *)
        use_ab=0
        case $a in ab:*) use_ab=1; a=${a#ab:} ;; esac
        case $a in
          toks=*)
            rest=${a#toks=}; toks=${rest%%:*}; rest=${rest#*:}; ctx=${rest%%:*}
            dep=$(echo "$toks" | awk -F, '{print NF}'); label="lit"
            ;;
          *)
            dep=${a%%:*}; rest=${a#*:}; ctx=${rest%%:*}
            toks=$(prompt "$dep"); label="lcg"
            ;;
        esac
        n=$N; case $rest in *:*) n=${rest#*:} ;; esac
        inst=time; [ "$use_ab" = 1 ] && inst=ab
        witness "pre r$r ours($label,$inst) d=$dep ctx=$ctx n=$n"
        t0=$(date +%s)
        if [ "$use_ab" = 1 ]; then
          out=$("$BIN" --tokens "$toks" -n "$n" --ctx "$ctx" --ab "${BLOOMERY_AB_INNER:-3}" 2>&1)
        else
          out=$("$BIN" --tokens "$toks" -n "$n" --ctx "$ctx" --time 2>&1)
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
          sums+=("ours($label,ab:base_p50) d=$dep ctx=$ctx n=$n|$(awk -v p="$p50" 'BEGIN{printf "%.4f", 1e3/p}')|$(awk -v p="$p50" 'BEGIN{printf "%.4f", 1e3/p}')")
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
        echo "ROW r$r ours($label) d=$dep ctx=$ctx n=$n | p50 ${p50} ms | mean ${mean} ms | tok/s(p50) $(awk -v p="$p50" 'BEGIN{printf "%.2f", 1e3/p}') | tok/s(mean) $(awk -v m="$mean" 'BEGIN{printf "%.2f", 1e3/m}') | nodes ${nodes:-?} | first10_p50 ${h10} | last10_p50 ${t10} | wall $((t1 - t0))s | steps ${last} | distinct_tokens ${uniq_tok}"
        sums+=("ours($label) d=$dep ctx=$ctx n=$n|$(awk -v m="$mean" 'BEGIN{printf "%.4f", 1e3/m}')|$(awk -v p="$p50" 'BEGIN{printf "%.4f", 1e3/p}')")
        ;;
    esac
  done
done
echo
echo "=== 팔별 평균 (tok/s). 첫 열: --time 행은 mean_ms 기반(교차 엔진 비율용), ab:base_p50 행은"
echo "    base 팔의 바퀴별 p50 평균 기반. 끝 열의 p50은 계열 비교용이다. ==="
printf '%s\n' "${sums[@]}" | awk -F'|' '{
  s[$1]+=$2; n[$1]++; if($3!=""){sp[$1]+=$3; np[$1]++}
  if(mn[$1]==""||$2+0<mn[$1]+0)mn[$1]=$2; if(mx[$1]==""||$2+0>mx[$1]+0)mx[$1]=$2
} END{for(k in s){
  spread = (mn[k]>0) ? 100*(mx[k]-mn[k])/mn[k] : 0
  printf "mean %-26s %8.2f tok/s(mean_ms)  [%s..%s, spread %.2f%%]  %s (n=%d)\n", k, s[k]/n[k], mn[k], mx[k], spread, (np[k]?sprintf("%.2f tok/s(p50)", sp[k]/np[k]):""), n[k]}}' | sort
witness post
