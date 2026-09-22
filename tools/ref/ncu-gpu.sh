#!/usr/bin/env bash
# 커널 카운터: 왜 비싼가를 Nsight Compute에 묻는다 (박스에서 실행).
#
#   BLOOMERY_NCU_DEPTHS="6 1024 4096" bash tools/ref/ncu-gpu.sh
#   BLOOMERY_NCU_KERNELS='flash' BLOOMERY_NCU_DEPTHS=4096 bash tools/ref/ncu-gpu.sh
#
# 러너 형제 셋 가운데 이것만 시간을 재지 않는다. depth-gpu.sh가 "얼마나 걸리는가"를,
# generate --ab의 프로브 팔이 "그 일이 얼마나 비싼가"를, 이 러너가 "왜 비싼가"를 낸다.
#
# **여기서 나온 µs는 기록이 아니다.** ncu는 카운터를 모으려고 커널을 직렬화하고 여러 번 재생하며,
# 그 사이 클럭을 고정한다(--clock-control base). 스텝 ms·tok/s의 원본은 임대 아래의 depth-gpu.sh
# 뿐이고, 이 파일이 내는 것은 비율·처리율·점유율·멈춘 사유다. 두 표를 같은 열에 놓지 않는다.
#
# 그래도 임대를 잡는다: 카드를 독점하고 클럭을 건드리므로, 이게 도는 동안 옆에서 잰 숫자는 무효다.
#
# 그래프: 우리 스텝은 캡처된 그래프 하나(648노드)다. ncu 2025.3의 --graph-profiling 기본값이
# node라 노드별 커널 카운터가 그대로 나온다. eager로도 같은 커널이 뜨므로, 의심스러우면
# BLOOMERY_NCU_MODE=eager로 한 번 더 돌려 두 표가 같은 말을 하는지 본다.
set -uo pipefail
MODEL=${BLOOMERY_REF_MODEL:-/models/small/DeepSeek-V2-Lite-Chat.Q3_K_M.gguf}
BIN=${BLOOMERY_GEN_BIN:-target/release/generate}
NCU=${NCU:-/usr/local/cuda/bin/ncu}
DEPTHS=${BLOOMERY_NCU_DEPTHS:-6 4096}
# 어느 커널을 볼 것인가. 정규식이고, 비우면 전부(느리다 — 스텝당 648노드다).
KERNELS=${BLOOMERY_NCU_KERNELS:-flash}
MODE=${BLOOMERY_NCU_MODE:-graph}
# 커널 하나당 몇 번의 런치를 모을 것인가. 재생 비용이 여기 붙는다.
COUNT=${BLOOMERY_NCU_COUNT:-16}
# **몇 개를 건너뛰고 모을 것인가 — 이 값이 틀리면 계기가 딴 깊이를 잰다.**
# generate는 프롬프트 P토큰을 `step(&tokens)` 하나로 먹이지만 그 안에서 **토큰마다 그래프를 한 번씩
# 재생한다**(model.rs `step`의 토큰 루프). 그러니 필터에 걸리는 런치는 프롬프트 토큰 하나당 27층 ×
# 2커널 = 54개이고, 깊이 d의 디코드에 닿으려면 (d + 버릴 스텝) × 54개를 건너뛰어야 한다.
# 실측 2026-09-22, 두 번 틀렸다: 건너뛰기 0으로 잡은 "깊이 4096"은 깊이 0이었고(9.15 → 10.85 µs,
# 깊이 6과 같음), 108(= 스텝 둘이라고 믿은 값)로 잡은 것은 깊이 2였다. 그리드는 증거가 아니다 —
# 세그먼트 수가 캐시 높이에서 나오므로 깊이 0에서도 528이다. 증거는 이 산술과, 지표가 깊이를 따라
# 움직이는 것뿐이다. 시간이 어느 커널에 가는가는 이 러너가 아니라 nsys-gpu.sh가 답한다(재생 경계로
# 스텝을 자른다). 여기서는 그 커널의 "왜"만 묻는다.
SKIP_STEPS=${BLOOMERY_NCU_SKIP_STEPS:-1}
# 필터에 걸리는 스텝당 런치 수. 기본 54는 KERNELS=flash일 때의 값이고, 필터를 바꾸면 nsys-gpu.sh의
# 표(재생당 런치 수)에서 읽어 넘긴다. BLOOMERY_NCU_SKIP은 절대값 덮어쓰기.
PER_STEP=${BLOOMERY_NCU_PER_STEP:-54}
OUTDIR=${BLOOMERY_NCU_OUT:-/root/bloomery-data/ncu}
LOCK=/root/bloomery-cpu.lock
# 카드 핀·증인 줄·바이너리 신선도는 러너 넷이 같은 파일에서 읽는다.
# shellcheck source=tools/ref/timing-card.sh
source "${BASH_SOURCE[0]%/*}/timing-card.sh"

# 섹션으로 묻는다(개별 메트릭 이름은 드라이버·ncu 판마다 흔들린다). 이 넷이 답하는 것:
#   SpeedOfLight       — 이 커널이 계산에 붙었나 메모리에 붙었나, 각각 피크의 몇 %인가
#   Occupancy          — 달성 점유율과 그것을 막는 것(레지스터·공유메모리·블록 수 중 무엇)
#   WarpStateStats     — 워프가 멈춘 사유별 비율. 깊이에서 가팔라지는 기울기의 이름이 여기 있다
#   MemoryWorkloadAnalysis — L1·L2 적중률과 DRAM 바이트. 헤드 16개가 같은 키 행을 읽는지 여기서 보인다
SECTIONS=${BLOOMERY_NCU_SECTIONS:-SpeedOfLight Occupancy WarpStateStats MemoryWorkloadAnalysis}

# 멈춘 사유는 섹션이 안 싣는다(이 ncu 판에 `smsp__pcsamp_*`가 없다 — PC 샘플링이 아니라
# 하드웨어 카운터 비율로 나온다). 이름으로 따로 묻는다. 단위는 "활성 워프당 비율"이라
# 사유별로 더하면 워프 하나가 매 사이클 어디에 서 있었는지의 분해가 된다.
#   long_scoreboard  전역 메모리 로드 대기 — 깊이에서 커지면 키 로드가 범인이다
#   barrier          블록 장벽 대기 — 타일마다 둘 있는 그 장벽
#   short_scoreboard 공유메모리·MIO 의존 대기
#   mio_throttle     MIO 큐 포화 — 공유메모리 처리율 벽
#   not_selected     발행할 수 있었는데 스케줄러가 딴 워프를 골랐다(= 점유가 충분하다는 신호)
#   no_instruction   명령 캐시 미스
STALLS=${BLOOMERY_NCU_STALLS:-long_scoreboard barrier short_scoreboard mio_throttle lg_throttle math_pipe_throttle wait not_selected selected no_instruction drain membar misc}

assert_fresh_binary "$BIN" || exit $?
[ -x "$NCU" ] || { echo "no ncu at $NCU" >&2; exit 2; }
# 카운터는 admin 전용이다(/proc/driver/nvidia/params의 RmProfilingAdminOnly: 1).
[ "$(id -u)" = 0 ] || { echo "ncu 카운터는 root가 필요하다(RmProfilingAdminOnly=1)" >&2; exit 77; }
mkdir -p "$OUTDIR"

witness() {
  echo "--- witness $1 $(now)"
  witness_card
}

# depth-gpu.sh와 같은 LCG 수열. 두 표가 같은 프롬프트를 말해야 나란히 읽힌다.
prompt() {
  awk -v n="$1" 'BEGIN{s=12345; printf "100000"; for(i=1;i<n;i++){s=(s*1103515245+12345)%2147483648; printf ",%d", 1000+(s%90000)}}'
}

exec 9>"$LOCK"
echo "[lease] waiting for $LOCK ..."
flock -w 1800 9 || { echo "[lease] timed out after 30 min"; exit 75; }
echo "[lease] held by pid $$ at $(now)"
echo "[config] ncu=$($NCU --version | sed -n 3p) mode=$MODE kernels='${KERNELS}' count=$COUNT depths='$DEPTHS'"
echo "[config] sections='$SECTIONS' out=$OUTDIR"
witness pre

sec_args=()
for s in $SECTIONS; do sec_args+=(--section "$s"); done
met_args=()
if [ -n "${BLOOMERY_NCU_METRICS:-}" ]; then
  # 지속시간만 같은 지표 하나로 물으면 재생이 한 번이라, 필터 없이 스텝 전체(648노드)를
  # 훑어도 몇 분이면 끝난다. "깊이 비용이 어느 커널에 있는가"는 이 모드로 답한다.
  met_args=(--metrics "$BLOOMERY_NCU_METRICS")
elif [ -n "$STALLS" ]; then
  list=""
  for s in $STALLS; do list="$list,smsp__warp_issue_stalled_${s}_per_warp_active"; done
  # 유출은 직접 묻는다 — 한 번 13~31배를 먹은 결함이고(nvlabs-ledger §5) 진단이 없었다.
  list="$list,l1tex__t_sector_hit_rate.pct,lts__t_sector_hit_rate.pct,dram__bytes_read.sum"
  met_args=(--metrics "${list#,}")
fi
kern_args=()
[ -n "$KERNELS" ] && kern_args=(--kernel-name "regex:$KERNELS" --kernel-name-base function)

rc_all=0
for d in $DEPTHS; do
  ctx=$((d + 128))
  skip=${BLOOMERY_NCU_SKIP:-$(( (d + SKIP_STEPS) * PER_STEP ))}
  out="$OUTDIR/ncu-d${d}-${MODE}-$(date -u +%H%M%S)"
  echo
  echo "=== 깊이 $d (ctx $ctx, mode $MODE, launch-skip $skip = ($d + $SKIP_STEPS) × $PER_STEP) → $out.txt"
  witness "pre d=$d"
  # -n 3: 프롬프트로 캐시를 채운 뒤 피드백 스텝 둘. 런치 수집은 --launch-count가 끊는다.
  "$NCU" --target-processes application-only --clock-control base \
         --graph-profiling node --launch-skip "$skip" --launch-count "$COUNT" \
         "${kern_args[@]}" "${sec_args[@]}" "${met_args[@]}" \
         --csv --log-file "$out.csv" \
         "$BIN" --tokens "$(prompt "$d")" -n 3 --ctx "$ctx" --mode "$MODE" \
         > "$out.txt" 2>&1
  rc=$?
  witness "post d=$d"
  echo "[rc] $rc"
  [ $rc -eq 0 ] || { rc_all=$rc; echo "--- 마지막 20줄"; tail -n 20 "$out.txt"; continue; }
  # 사람이 읽는 요약. 원본 CSV는 런치마다 한 줄씩이라 커널별로 중앙값을 낸다 —
  # 평균이 아니라 중앙값인 이유는 첫 런치가 콜드 캐시를 지고 오기 때문이다.
  echo "--- 요약(커널별 런치 중앙값). 원본은 $out.csv"
  python3 - "$out.csv" "${BLOOMERY_NCU_TOTALS:-}" <<'PY'
import csv, statistics, sys
KEEP = ("Duration", "gpu__time_duration.sum", "Compute (SM) Throughput", "Memory Throughput", "DRAM Throughput",
        "Achieved Occupancy", "Theoretical Occupancy", "Block Limit Registers",
        "Block Limit Shared Mem", "Block Limit Warps", "L1/TEX Hit Rate", "L2 Hit Rate",
        "Local Memory Spilling Requests", "Warp Cycles Per Issued Instruction")
rows = [r for r in csv.reader(open(sys.argv[1])) if len(r) > 14 and r[0] not in ("ID", "")]
by, shape = {}, {}
for r in rows:
    kernel, metric, unit, val = r[4].split("(")[0], r[12], r[13], r[14]
    # 그리드는 캐시 높이(--ctx)의 함수라 어느 깊이를 잡았는지 말해 주지 않는다 — 형상 기록일 뿐이다.
    shape.setdefault(kernel, set()).add((r[7], r[8]))
    try:
        v = float(val.replace(",", ""))
    except ValueError:
        continue
    by.setdefault((kernel, metric, unit), []).append(v)
if len(sys.argv) > 2 and sys.argv[2]:
    # 스텝 전체를 훑은 모드: 커널마다 (런치 수 × 지속시간 합)을 큰 것부터. 깊이 둘을 나란히
    # 놓으면 어느 커널이 깊이를 타는지가 한 열로 읽힌다.
    tot = []
    for (kernel, metric, unit), v in by.items():
        if "duration" in metric.lower():
            tot.append((sum(v), len(v), kernel, unit))
    grand = sum(t[0] for t in tot) or 1.0
    print(f"    {'커널':38s} {'런치':>5s} {'합':>12s} {'몫':>7s} {'평균':>10s}")
    for s, n, k, u in sorted(tot, reverse=True)[:28]:
        print(f"    {k[:38]:38s} {n:5d} {s:12.1f} {100 * s / grand:6.1f}% {s / n:10.1f} {u}")
    print(f"    {'합계':38s} {sum(t[1] for t in tot):5d} {grand:12.1f}   100.0%")
    raise SystemExit
for kernel in sorted({k for k, _, _ in by}):
    launches = max(len(v) for (kk, _, _), v in by.items() if kk == kernel)
    sh = " ".join(f"block={b} grid={g}" for b, g in sorted(shape.get(kernel, ())))
    print(f"  [{kernel}]  런치 {launches}  {sh}")
    named = [(m, u, s) for (kk, m, u), s in by.items() if kk == kernel and m in KEEP]
    for m, u, s in sorted(named, key=lambda x: KEEP.index(x[0])):
        print(f"    {m:34s} {statistics.median(s):12.3f} {u}")
    st = [(m.replace('smsp__warp_issue_stalled_', '').replace('_per_warp_active', ''), s)
          for (kk, m, _), s in by.items() if kk == kernel and m.startswith("smsp__warp_issue_stalled_")]
    if st:
        tot = sum(statistics.median(s) for _, s in st) or 1.0
        print("    멈춘 사유(활성 워프당 비율, 합 대비 %):")
        for name, s in sorted(st, key=lambda x: -statistics.median(x[1]))[:8]:
            med = statistics.median(s)
            print(f"      {name:26s} {med:9.4f}  {100 * med / tot:5.1f}%")
PY
done

witness post
echo "[lease] released at $(now)"
exit $rc_all
