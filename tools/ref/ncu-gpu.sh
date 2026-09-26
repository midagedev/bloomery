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
# 그래프: 우리 스텝은 캡처된 그래프 하나다. 러너가 --graph-profiling node를 직접 넘기므로(아래,
# ncu 2025.3.1 고정) 노드별 커널 카운터가 그대로 나온다 — 판의 기본값에 기대지 않는다. eager로도 같은 커널이 뜨므로, 의심스러우면
# BLOOMERY_NCU_MODE=eager로 한 번 더 돌려 두 표가 같은 말을 하는지 본다.
#
# The grouped GEMM form (BLOOMERY_NCU_FORM=gemm, `just ncu-gpu-gemm [ARM]`): the counters of one
# bench arm of the grouped int8 GEMM instead of a decode step. The profiled command is
# `gate_p8 --bench-kernels --bench-arm ARM` (BLOOMERY_NCU_GEMM_ARM, default gemm_q4k_moe_t4096:
# Qwen3's routed gate shape, 128 experts of 768 x 2048 Q4_K, top-8, T = 4096), which runs that one
# arm and none of the others, so every `gemm_q4k` launch in the process is the arm's and no skip has
# to be counted across the other arms (the depth form's two wrong skips above are what that saves).
# The arm launches its GEMM eagerly 64 times (the warm-up burst), then 7 x 64 timed eager launches,
# then graph replays; the default skip 64 (BLOOMERY_NCU_SKIP) steps over the warm-up and the count
# (BLOOMERY_NCU_COUNT) takes eager launches of the timed bursts. Sections SpeedOfLight,
# WarpStateStats and MemoryWorkloadAnalysis (BLOOMERY_NCU_SECTIONS), the stall reasons below, and
# the pipe metrics in GEMM_METRICS; the summary prints, per kernel, the throughput units with the
# top one named, the LSU data-pipe wavefronts per launch, issue-active, the IMMA tensor pipe, DRAM
# and the top stall reasons (medians over the collected launches). The profile runs under
# BLOOMERY_ARM_BOUND seconds (default 900). BLOOMERY_DRY=1 prints the command line and exits before
# the binary check and the lease; the depth form has no dry path and refuses the variable.
#
# The V4.1 prompt form (BLOOMERY_NCU_FORM=ds41pp, `just ncu-gpu-ds41-pp [P]`): one launch of each of the
# four attention projections (the joined qkv, q_b, wo_a heads, wo_b) at m = 8 in one full chunk of a
# layer >= 2 of a P-token prompt batch (BLOOMERY_NCU_PROMPT, default 512). The skip is the known trap, so
# it is not counted here: tools/ref/ds41pp.py ncu-plan derives it from the nsys prefill form's trace of
# the same command (BLOOMERY_NCU_TRACE, default the newest nsys-ds41-pp<P>-* sqlite in $BLOOMERY_DATA/nsys,
# with its .meta and run log): the eager q3k_gemv / ds41_q3k_gemv_heads_mcol launches in launch order
# before the chunk's qkv, and the shapes of the launches it reaches. The profile runs that command with
# --mode eager — the prompt batch is eager in both modes, and without the capture before it the
# filter's launch count is the trace's — and refuses a trace made by another binary or hot list.
# BLOOMERY_NCU_LAYER picks the layer (default: the first >= 2 whose full chunks carry only the four
# projections and no engram, compressor or indexer launch); the chunk is its middle full chunk. The
# summary checks every profiled launch's name, grid and block against the plan and each one's column
# count against its global loads (the m-column walk issues 8 + 3m global loads a warp-iteration; the row
# count and K come from the file's tensors) and prints nothing else on a mismatch; then, per launch, the
# block-step cycles (one iteration of a block's 8 warps) and each unit's demand in cycles next to them:
# issue, alu, fmaheavy (the pipe IDP.4A issues on, with IMAD and IMUL), fmalite, xu, lsu, the L1TEX data
# pipe, DRAM — from each pipe's instruction count at its peak rate and, for alu, fma, fmaheavy, fmalite
# and the LSU writeback, from ncu's own active cycles — with ncu's own pipe shares and the top stall
# reasons. Sections SpeedOfLight, LaunchStats, Occupancy (BLOOMERY_NCU_SECTIONS) and the metrics in
# DS41PP_METRICS; BLOOMERY_ARM_BOUND bounds the profile. The card is full, so before a profiled launch's
# first pass ncu saves every device allocation to host memory: this form's prompt runs for minutes.
set -uo pipefail
# 데이터 디렉터리 기본값(BLOOMERY_DATA 오버라이드는 그대로 받는다)은 빌드 스크립트와 같은 파일이 소유한다.
# 모델은 generate가 BLOOMERY_REF_MODEL에서 직접 연다 — tools/box.sh가 ref-paths.sh의 MODEL을 그 이름으로
# export한다(generate 자신에게는 기본값이 없다).
# shellcheck source=tools/ref/ref-paths.sh
source "${BASH_SOURCE[0]%/*}/ref-paths.sh"
BIN=${BLOOMERY_GEN_BIN:-target/release/generate}
# ncu는 판을 이름으로 고정한다. /usr/local/cuda/bin/ncu(13.0 래퍼)는 /opt/nvidia/nsight-compute에서 가장 새 판을
# 고르므로, 13.3을 설치한 뒤로는 2026.2.1이 돌았다. 그 판의 "Local Memory Spilling Requests"는 공유 메모리 스필을
# 따로 센다(새 "Shared Memory Spilling Requests" 행). 이전 표와 같은 뜻으로 읽으려고 2025.3.1에 둔다.
NCU=${NCU:-/opt/nvidia/nsight-compute/2025.3.1/ncu}
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
OUTDIR=${BLOOMERY_NCU_OUT:-$BLOOMERY_DATA/ncu}
# 카드 핀·증인 줄·바이너리 신선도는 러너 넷이 같은 파일에서 읽는다.
# shellcheck source=tools/ref/timing-card.sh
source "${BASH_SOURCE[0]%/*}/timing-card.sh"
# The lease and the witness fields.
# shellcheck source=tools/ref/lease.sh
source "${BASH_SOURCE[0]%/*}/lease.sh"

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

FORM=${BLOOMERY_NCU_FORM:-generate}
DRY=${BLOOMERY_DRY:-}
case $FORM in
  generate)
    [ -z "$DRY" ] || { echo "ncu-gpu.sh: BLOOMERY_DRY is read by the gemm and ds41pp forms only; the depth form has no dry path" >&2; exit 64; }
    ;;
  gemm | ds41pp) ;;
  *) echo "ncu-gpu.sh: BLOOMERY_NCU_FORM is generate, gemm or ds41pp, got '$FORM'" >&2; exit 64 ;;
esac

if [ "$FORM" = gemm ]; then
  GEMM_BIN=${BLOOMERY_NCU_GEMM_BIN:-target/release/gate_p8}
  GEMM_ARM=${BLOOMERY_NCU_GEMM_ARM:-gemm_q4k_moe_t4096}
  GEMM_KERNELS=${BLOOMERY_NCU_KERNELS:-^gemm_q4k}
  GEMM_SKIP=${BLOOMERY_NCU_SKIP:-64}
  GEMM_SECTIONS=${BLOOMERY_NCU_SECTIONS:-SpeedOfLight WarpStateStats MemoryWorkloadAnalysis}
  BOUND=${BLOOMERY_ARM_BOUND:-900}
  # The pipe metrics the summary names, in its order: the L1TEX unit's throughput; the LSU data-pipe
  # wavefronts (every global and shared access the unit's data stage serves) and the shared-memory
  # part of them; that pipe's busy share; shared-memory bank conflicts; issue-active; the IMMA
  # tensor pipe (int8 mma); the LSU instruction pipe; DRAM.
  GEMM_METRICS=l1tex__throughput.avg.pct_of_peak_sustained_active
  GEMM_METRICS+=,l1tex__data_pipe_lsu_wavefronts.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared.sum
  GEMM_METRICS+=,l1tex__data_pipe_lsu_wavefronts.avg.pct_of_peak_sustained_elapsed
  GEMM_METRICS+=,l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum
  GEMM_METRICS+=,smsp__issue_active.avg.pct_of_peak_sustained_active
  GEMM_METRICS+=,sm__pipe_tensor_op_imma_cycles_active.avg.pct_of_peak_sustained_active
  GEMM_METRICS+=,sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active
  GEMM_METRICS+=,dram__throughput.avg.pct_of_peak_sustained_elapsed
  case $GEMM_ARM in
    gemm_q4k_*_t*) ;;
    *) echo "ncu-gpu.sh: BLOOMERY_NCU_GEMM_ARM is a gate_p8 grouped-GEMM arm (gemm_q4k_<label>_t<T>), got '$GEMM_ARM'" >&2; exit 64 ;;
  esac
  case "$GEMM_SKIP:$COUNT" in
    *[!0-9:]* | :* | *:) echo "ncu-gpu.sh: BLOOMERY_NCU_SKIP and BLOOMERY_NCU_COUNT are whole numbers, got '$GEMM_SKIP' and '$COUNT'" >&2; exit 64 ;;
  esac
  sec_args=()
  for s in $GEMM_SECTIONS; do sec_args+=(--section "$s"); done
  list=$GEMM_METRICS
  for s in $STALLS; do list="$list,smsp__warp_issue_stalled_${s}_per_warp_active"; done
  out="$OUTDIR/ncu-gemm-${GEMM_ARM}-$(date -u +%H%M%S)"
  CMD=(timeout --kill-after=10 "$BOUND" "$NCU" --target-processes application-only --clock-control base
       --launch-skip "$GEMM_SKIP" --launch-count "$COUNT"
       --kernel-name "regex:$GEMM_KERNELS" --kernel-name-base function
       "${sec_args[@]}" --metrics "$list" --csv --log-file "$out.csv"
       "$GEMM_BIN" --bench-kernels --bench-arm "$GEMM_ARM")
  if [ -n "$DRY" ]; then
    echo "[dry] form=gemm bin=$GEMM_BIN arm=$GEMM_ARM kernels='$GEMM_KERNELS' skip=$GEMM_SKIP count=$COUNT bound=${BOUND}s timing_gpu=$TIMING_GPU CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    echo "[dry] ${CMD[*]}"
    exit 0
  fi
  assert_fresh_binary "$GEMM_BIN" || exit $?
  [ -x "$NCU" ] || { echo "no ncu at $NCU" >&2; exit 2; }
  [ "$(id -u)" = 0 ] || { echo "ncu 카운터는 root가 필요하다(RmProfilingAdminOnly=1)" >&2; exit 77; }
  mkdir -p "$OUTDIR"
  WITNESS=(head-open indent card model)
  lease_take
  echo "[config] ncu=$($NCU --version | sed -n 3p) form=gemm arm=$GEMM_ARM kernels='$GEMM_KERNELS' skip=$GEMM_SKIP count=$COUNT bound=${BOUND}s"
  echo "[config] sections='$GEMM_SECTIONS' out=$out"
  witness pre
  guard_other
  "${CMD[@]}" > "$out.txt" 2>&1
  rc=$?
  witness post
  echo "[rc] $rc"
  [ $rc -eq 0 ] || { echo "--- last 20 lines"; tail -n 20 "$out.txt"; }
  if [ -s "$out.csv" ]; then
    echo "--- summary (median over the collected launches; the durations are not numbers of record). Raw: $out.csv"
    lease_bounded "$LEASE_ARM_BOUND" python3 - "$out.csv" <<'PY'
import csv, statistics, sys
rows = [r for r in csv.reader(open(sys.argv[1])) if len(r) > 14 and r[0] not in ("ID", "")]
by, shape, ids = {}, {}, {}
for r in rows:
    kernel, metric, unit, val = r[4].split("(")[0], r[12], r[13], r[14]
    shape.setdefault(kernel, set()).add((r[7], r[8]))
    ids.setdefault(kernel, set()).add(r[0])
    try:
        v = float(val.replace(",", ""))
    except ValueError:
        continue
    # One name can come from two sections in two units (SpeedOfLight's "Memory Throughput" is a percentage,
    # MemoryWorkloadAnalysis's is byte/s), so the unit is part of the key, as in the generate form's summary.
    by.setdefault((kernel, metric, unit), []).append(v)
if not by:
    print("    the CSV holds no metric rows")
    raise SystemExit(3)
SOL = ("Duration", "Elapsed Cycles", "SM Active Cycles", "Compute (SM) Throughput", "Memory Throughput",
       "L1/TEX Cache Throughput", "L2 Cache Throughput", "DRAM Throughput")
UNITS = ("Compute (SM) Throughput", "L1/TEX Cache Throughput", "L2 Cache Throughput", "DRAM Throughput")
PIPE = (
    ("L1TEX throughput %", "l1tex__throughput.avg.pct_of_peak_sustained_active"),
    ("LSU data-pipe wavefronts / launch", "l1tex__data_pipe_lsu_wavefronts.sum"),
    ("  of them shared memory", "l1tex__data_pipe_lsu_wavefronts_mem_shared.sum"),
    ("LSU data-pipe busy %", "l1tex__data_pipe_lsu_wavefronts.avg.pct_of_peak_sustained_elapsed"),
    ("shared bank conflicts / launch", "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum"),
    ("issue-active %", "smsp__issue_active.avg.pct_of_peak_sustained_active"),
    ("tensor (IMMA) pipe %", "sm__pipe_tensor_op_imma_cycles_active.avg.pct_of_peak_sustained_active"),
    ("LSU instruction pipe %", "sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active"),
    ("DRAM %", "dram__throughput.avg.pct_of_peak_sustained_elapsed"),
)
missing = 0
for kernel in sorted({k for k, _, _ in by}):
    def variants(m):
        return sorted((u, statistics.median(s)) for (kk, mm, u), s in by.items() if kk == kernel and mm == m)
    def med(m, u):
        vs = [v for uu, v in variants(m) if uu == u]
        return vs[0] if vs else None
    sh = " ".join(f"block={b} grid={g}" for b, g in sorted(shape[kernel]))
    print(f"  [{kernel}]  launches {len(ids[kernel])}  {sh}")
    for m in SOL:
        for u, v in variants(m):
            print(f"    {m:36s} {v:14.3f} {u}")
    top = max(((med(m, "%"), m) for m in UNITS if med(m, "%") is not None), default=None)
    if top:
        print(f"    {'top unit':36s} {top[1]} at {top[0]:.1f} %")
    for label, m in PIPE:
        vs = variants(m)
        if not vs:
            missing += 1
            print(f"    {label:36s} {'NOT REPORTED':>14s}   {m}")
        for u, v in vs:
            print(f"    {label:36s} {v:14.4g} {u}")
    st = [(m.replace("smsp__warp_issue_stalled_", "").replace("_per_warp_active", ""), s)
          for (kk, m, _), s in by.items() if kk == kernel and m.startswith("smsp__warp_issue_stalled_")]
    if st:
        tot = sum(statistics.median(s) for _, s in st) or 1.0
        print("    stall reasons (per active warp, share of their sum):")
        for name, s in sorted(st, key=lambda x: -statistics.median(x[1]))[:6]:
            v = statistics.median(s)
            print(f"      {name:26s} {v:9.4f}  {100 * v / tot:5.1f}%")
raise SystemExit(3 if missing else 0)
PY
    prc=$?
    [ $prc -eq 0 ] || [ $rc -ne 0 ] || rc=$prc
  else
    echo "[summary] no CSV at $out.csv"
    [ $rc -ne 0 ] || rc=3
  fi
  echo "[lease] released at $(now)"
  exit $rc
fi

if [ "$FORM" = ds41pp ]; then
  [ "$MODEL_NAME" = deepseek41 ] || { echo "ncu-gpu.sh: the ds41pp form profiles generate_ds41; the profile is '$MODEL_NAME' (BLOOMERY_MODEL=deepseek41)" >&2; exit 64; }
  PP=${BASH_SOURCE[0]%/*}/ds41pp.py
  GEN=${BLOOMERY_GEN_BIN:-target/release/generate_ds41}
  P=${BLOOMERY_NCU_PROMPT:-512}
  case $P in
    '' | *[!0-9]* | [0-8]) echo "ncu-gpu.sh: BLOOMERY_NCU_PROMPT is a prompt length >= 9, got '$P'" >&2; exit 64 ;;
  esac
  TRACE=${BLOOMERY_NCU_TRACE:-$(ls -t "$BLOOMERY_DATA"/nsys/nsys-ds41-pp"$P"-n*-*.sqlite 2> /dev/null | head -n 1)}
  META=${TRACE%.sqlite}.meta
  RUNLOG=${TRACE%.sqlite}.txt
  meta() { [ -f "$META" ] && sed -n "s/^$1=//p" "$META"; }
  N=$(meta n)
  N=${N:-2}
  SECTIONS_PP=${BLOOMERY_NCU_SECTIONS:-SpeedOfLight LaunchStats Occupancy}
  BOUND=${BLOOMERY_ARM_BOUND:-900}
  DS41PP_METRICS=sm__cycles_elapsed.avg,sm__cycles_active.avg,smsp__inst_executed.sum
  DS41PP_METRICS+=,smsp__issue_active.avg.pct_of_peak_sustained_active,sm__warps_active.avg.pct_of_peak_sustained_active
  DS41PP_METRICS+=,sm__inst_executed_pipe_alu.sum,sm__inst_executed_pipe_fma.sum,sm__inst_executed_pipe_fmaheavy.sum
  DS41PP_METRICS+=,sm__inst_executed_pipe_fmalite.sum,sm__inst_executed_pipe_xu.sum,sm__inst_executed_pipe_lsu.sum
  DS41PP_METRICS+=,sm__inst_executed_pipe_cbu.sum,sm__inst_executed_pipe_adu.sum,sm__inst_executed_pipe_uniform.sum
  DS41PP_METRICS+=,sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active
  DS41PP_METRICS+=,sm__inst_executed_pipe_fmaheavy.avg.pct_of_peak_sustained_active
  DS41PP_METRICS+=,sm__inst_executed_pipe_fmalite.avg.pct_of_peak_sustained_active
  DS41PP_METRICS+=,sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_active
  DS41PP_METRICS+=,sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active
  DS41PP_METRICS+=,smsp__pipe_alu_cycles_active.avg,smsp__pipe_fma_cycles_active.avg
  DS41PP_METRICS+=,smsp__pipe_fmaheavy_cycles_active.avg,smsp__pipe_fmalite_cycles_active.avg
  DS41PP_METRICS+=,l1tex__lsuin_requests.sum,l1tex__lsu_writeback_active.avg
  DS41PP_METRICS+=,smsp__inst_executed_op_global_ld.sum,l1tex__data_pipe_lsu_wavefronts.sum
  DS41PP_METRICS+=,l1tex__throughput.avg.pct_of_peak_sustained_active
  DS41PP_METRICS+=,dram__bytes_read.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed
  out="$OUTDIR/ncu-ds41pp-pp${P}-$(date -u +%H%M%S)"
  mkdir -p "$OUTDIR"
  sec_args=()
  for s in $SECTIONS_PP; do sec_args+=(--section "$s"); done
  list=$DS41PP_METRICS
  for s in $STALLS; do list="$list,smsp__warp_issue_stalled_${s}_per_warp_active"; done
  # The command for a skip, a count and a kernel filter, into CMD.
  pp_cmd() {
    CMD=(timeout --kill-after=10 "$BOUND" "$NCU" --target-processes application-only --clock-control base
         --launch-skip "$1" --launch-count "$2"
         --kernel-name "regex:$3" --kernel-name-base function
         "${sec_args[@]}" --metrics "$list" --csv --log-file "$out.csv"
         "$GEN" --depth "$P" -n "$N" --mode eager --time)
  }
  if [ -n "$DRY" ]; then
    pp_cmd '<the plan skip>' '<the plan count>' '<the plan regex>'
    echo "[dry] form=ds41pp bin=$GEN trace=${TRACE:-<none: run just nsys-gpu-ds41-prefill $P first>} (sha256 $(meta sha256), hot list '$(meta hot_list)') bound=${BOUND}s timing_gpu=$TIMING_GPU CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES hot_list=${BLOOMERY_HOT_LIST:-<unset>}"
    echo "[dry] ${CMD[*]}"
    exit 0
  fi
  [ -n "$TRACE" ] && [ -f "$TRACE" ] || { echo "ncu-gpu.sh: no nsys prefill trace of P = $P (BLOOMERY_NCU_TRACE, or run just nsys-gpu-ds41-prefill $P first)" >&2; exit 2; }
  [ -f "$META" ] && [ -f "$RUNLOG" ] || { echo "ncu-gpu.sh: $TRACE has no .meta or run log beside it: the trace's binary and command are unknown" >&2; exit 2; }
  assert_fresh_binary "$GEN" || exit $?
  [ "$(meta sha256)" = "$BIN_SHA" ] || { echo "[skip] the trace was made by generate_ds41 sha256 $(meta sha256), this one is $BIN_SHA: its launch order is not this binary's — run just nsys-gpu-ds41-prefill $P again" >&2; exit 3; }
  [ "$(meta hot_list)" = "${BLOOMERY_HOT_LIST:-}" ] || { echo "[skip] the trace ran with BLOOMERY_HOT_LIST='$(meta hot_list)', this run with '${BLOOMERY_HOT_LIST:-}': another placement" >&2; exit 3; }
  [ -x "$NCU" ] || { echo "no ncu at $NCU" >&2; exit 2; }
  [ "$(id -u)" = 0 ] || { echo "ncu 카운터는 root가 필요하다(RmProfilingAdminOnly=1)" >&2; exit 77; }
  WITNESS=(head-open indent card model)
  lease_take
  plan_args=("$TRACE" "$P" "$RUNLOG" "$out.plan")
  [ -z "${BLOOMERY_NCU_LAYER:-}" ] || plan_args+=(--layer "$BLOOMERY_NCU_LAYER")
  echo "[plan] from $TRACE"
  lease_bounded "$LEASE_ARM_BOUND" python3 "$PP" ncu-plan "${plan_args[@]}" || exit $?
  pp_cmd "$(sed -n 's/^skip=//p' "$out.plan")" "$(sed -n 's/^count=//p' "$out.plan")" "$(sed -n 's/^regex=//p' "$out.plan")"
  SKIP=$(sed -n 's/^skip=//p' "$out.plan")
  COUNT=$(sed -n 's/^count=//p' "$out.plan")
  REGEX=$(sed -n 's/^regex=//p' "$out.plan")
  echo "[config] ncu=$($NCU --version | sed -n 3p) form=ds41pp P=$P n=$N skip=$SKIP count=$COUNT regex='$REGEX' bound=${BOUND}s"
  echo "[config] sections='$SECTIONS_PP' out=$out hot_list=${BLOOMERY_HOT_LIST:-<unset>}"
  witness pre
  guard_other
  t0=$(date +%s)
  "${CMD[@]}" > "$out.txt" 2>&1
  rc=$?
  witness post
  echo "[rc] $rc wall $(($(date +%s) - t0))s"
  [ $rc -eq 0 ] || { echo "--- last 20 lines"; tail -n 20 "$out.txt"; }
  want=$(sed -n 's/^plan_line=//p' "$out.plan")
  got=$(grep -m 1 '^plan ' "$out.txt")
  if [ "$want" != "$got" ]; then
    echo "[skip] the profiled run's plan line differs from the trace's:"
    echo "    trace:   $want"
    echo "    profile: $got"
    [ $rc -ne 0 ] || rc=3
  elif [ -s "$out.csv" ]; then
    echo "--- summary. Raw: $out.csv, the plan $out.plan, the run $out.txt"
    lease_bounded "$LEASE_ARM_BOUND" python3 "$PP" ncu-summary "$out.csv" "$out.plan" "$MODEL"
    prc=$?
    [ $prc -eq 0 ] || [ $rc -ne 0 ] || rc=$prc
  else
    echo "[summary] no CSV at $out.csv"
    [ $rc -ne 0 ] || rc=3
  fi
  echo "[lease] released at $(now)"
  exit $rc
fi

assert_fresh_binary "$BIN" || exit $?
[ -x "$NCU" ] || { echo "no ncu at $NCU" >&2; exit 2; }
# 카운터는 admin 전용이다(/proc/driver/nvidia/params의 RmProfilingAdminOnly: 1).
[ "$(id -u)" = 0 ] || { echo "ncu 카운터는 root가 필요하다(RmProfilingAdminOnly=1)" >&2; exit 77; }
mkdir -p "$OUTDIR"

# shellcheck disable=SC2034 # read by lease.sh's witness()
WITNESS=(head-open indent card model)

lease_take
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
  lease_bounded "$LEASE_ARM_BOUND" "$NCU" --target-processes application-only --clock-control base \
         --graph-profiling node --launch-skip "$skip" --launch-count "$COUNT" \
         "${kern_args[@]}" "${sec_args[@]}" "${met_args[@]}" \
         --csv --log-file "$out.csv" \
         "$BIN" --tokens "$(lcg_prompt "$d")" -n 3 --ctx "$ctx" --mode "$MODE" \
         > "$out.txt" 2>&1
  rc=$?
  witness "post d=$d"
  echo "[rc] $rc"
  [ $rc -eq 0 ] || { rc_all=$rc; echo "--- 마지막 20줄"; tail -n 20 "$out.txt"; continue; }
  # 사람이 읽는 요약. 원본 CSV는 런치마다 한 줄씩이라 커널별로 중앙값을 낸다 —
  # 평균이 아니라 중앙값인 이유는 첫 런치가 콜드 캐시를 지고 오기 때문이다.
  echo "--- 요약(커널별 런치 중앙값). 원본은 $out.csv"
  lease_bounded "$LEASE_ARM_BOUND" python3 - "$out.csv" "${BLOOMERY_NCU_TOTALS:-}" <<'PY'
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
