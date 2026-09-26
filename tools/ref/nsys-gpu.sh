#!/usr/bin/env bash
# 커널 타임라인: 스텝의 시간이 어느 커널에 가는가를 Nsight Systems에 묻는다 (박스에서 실행).
#
#   BLOOMERY_NSYS_DEPTHS="6 4096" bash tools/ref/nsys-gpu.sh
#
# 러너 형제 넷의 자리: depth-gpu.sh가 "얼마나 걸리는가"(기록), generate --ab의 프로브 팔이
# "그 일이 얼마나 비싼가", 이 러너가 "어느 커널이 그 시간을 먹는가", ncu-gpu.sh가 "그 커널은 왜
# 비싼가"다. ncu는 커널을 직렬화하고 클럭을 고정하므로 시간 분포를 묻는 계기가 아니다 — 그 질문은
# 여기서 답한다. nsys는 실제 실행을 그대로 두고 커널 시작·끝만 찍는다(그래프 노드 단위).
#
# **스텝 경계는 재생 횟수로 자른다.** generate는 프롬프트 P토큰을 `step(&tokens)` 하나로 먹이지만
# 그 안에서 토큰마다 그래프를 한 번씩 재생한다(model.rs `step`의 토큰 루프). 그러니 프로파일 안의
# 재생은 P + (n-1)개이고, 마지막 재생이 깊이 P + n - 2의 디코드 스텝이다. "몇 개를 건너뛴다"를
# 런치 수로 세면 틀린다 — 2026-09-22 실측: ncu가 54런치(= 스텝 하나라고 믿은 값)를 건너뛰고 잡은
# "깊이 4096"은 깊이 2였고, 그리드(캐시 높이에서 나온다)는 깊이 0부터 528이라 증거가 못 됐다.
# 이 러너는 재생마다 정확히 한 번 나오는 커널을 찾아 그것으로 경계를 긋고, 그 경계가 P + n - 1개인지
# 스스로 확인한다. 아니면 표를 내지 않는다.
#
# **여기서 나온 µs는 커널 시간이지 스텝 시간이 아니다.** 창 안 커널 합과 창의 벽시계(첫 시작~끝)를
# 같이 찍는다 — 둘의 차가 런치 사이 빈틈이고, 창 벽시계가 depth-gpu.sh의 스텝 ms와 맞아야 이 표를
# 그 기록 옆에 놓을 수 있다.
#
# qwen3moe (BLOOMERY_MODEL=qwen3moe, `just nsys-gpu-qwen3moe 6 1024 4096`): generate_qwen3moe
# prefills a prompt eight positions per eager pass, so the prompt form above would put ⌈D/8⌉
# multi-row passes into the trace. This profile runs the binary's seed form instead:
# `--seed-depth D -n N --ctx C --mode M --time`. The caches are filled by host copies (no kernel),
# then one eager one-row prefill pass feeds id 0 at position D - 1 and yields token 0, and N - 1
# graph replays follow: N windows, window r at depth D - 1 + r, windows 1 .. N - 1 being `time step
# 1 .. N - 1` of the same run. The boundary is named, not searched for: `embed_rows_q4k`, layer 0's
# embedding gather, one launch per pass and per replay (arch/qwen3moe/dispatch.rs `layer`, embed =
# slot 0); its count must be N or no table. C defaults to depth-qwen3moe.sh's height for the same
# D (D + BLOOMERY_DECODE_N rounded up to 256; BLOOMERY_GEN_CTX fixes it), because the flash grid is
# a function of C (flash_gqa::segments_for) and the table must be of the kernels that runner times.
# The seeded cache is a pattern, so routing differs from a real prompt's; the kernel shapes do not.
# N >= 3, so that the control window is a replay. Each profile runs under BLOOMERY_ARM_BOUND
# (seconds, default 900).
#
#
# qwen3moe prefill (BLOOMERY_NSYS_FORM=prefill, `just nsys-gpu-qwen3moe-prefill 4096`): the timed prompt
# of depth-qwen3moe.sh's `<P>` arm instead of a decode step. Each entry of BLOOMERY_NSYS_DEPTHS is a
# prompt length P >= 9 (a GEMM ubatch runs); the profiled command is that arm's,
# `generate_qwen3moe --tokens <lcg_prompt P> -n N --ctx C --mode M --time`, C as in the seed form. The
# prompt runs as the plan's K units (at the default ubatch size 4096 and P <= 4096, one ubatch: K = 1),
# each opening with one `embed_rows_q4k` launch, then the head; the N - 1 feedback steps are graph
# replays, each opening with one. So the boundary count is K + N - 1, K read from the run's own `time
# prompt ... passes=K` row, or no table; the prefill window runs from the first boundary to the
# (K+1)-th — the prompt's launches and its head. Its table is per kernel over that window (launches,
# total ms, share of the window's kernel sum), with the grouped GEMM entries summed, beside the
# window's wall and the binary's `time prompt ms=` (that wall also holds the host work before the first
# launch and the token's readback). N >= 2, so that a replay closes the window.
#
#   bash tools/ref/nsys-gpu.sh --analyze <out>.sqlite <depth> <n> [<out>.txt]   # tables again, no profile
#   BLOOMERY_DRY=1 ...                                                          # command lines, no lease
set -uo pipefail
# 데이터 디렉터리 기본값(BLOOMERY_DATA 오버라이드는 그대로 받는다)은 빌드 스크립트와 같은 파일이 소유한다.
# 모델은 generate가 BLOOMERY_REF_MODEL에서 직접 연다 — tools/box.sh가 ref-paths.sh의 MODEL을 그 이름으로
# export한다(generate 자신에게는 기본값이 없다).
# shellcheck source=tools/ref/ref-paths.sh
source "${BASH_SOURCE[0]%/*}/ref-paths.sh"
FORM=prompt
[ "$MODEL_NAME" != qwen3moe ] || FORM=seed
case ${BLOOMERY_NSYS_FORM:-} in
  '') ;;
  prefill)
    [ "$MODEL_NAME" = qwen3moe ] || { echo "nsys-gpu.sh: BLOOMERY_NSYS_FORM=prefill profiles generate_qwen3moe; the profile is '$MODEL_NAME'" >&2; exit 64; }
    FORM=prefill
    ;;
  *) echo "nsys-gpu.sh: BLOOMERY_NSYS_FORM is prefill or unset, got '$BLOOMERY_NSYS_FORM'" >&2; exit 64 ;;
esac
if [ "$FORM" != prompt ]; then BIN=${BLOOMERY_GEN_BIN:-target/release/generate_qwen3moe}; else BIN=${BLOOMERY_GEN_BIN:-target/release/generate}; fi
MARKER=${BLOOMERY_NSYS_MARKER:-embed_rows_q4k}
BOUND=${BLOOMERY_ARM_BOUND:-900}
DEPTH_N=${BLOOMERY_DECODE_N:-96}
GEN_CTX=${BLOOMERY_GEN_CTX:-}
DRY=${BLOOMERY_DRY:-}
NSYS=${NSYS:-/usr/local/cuda/bin/nsys}
if [ "$FORM" = prefill ]; then DEPTHS=${BLOOMERY_NSYS_DEPTHS:-4096}; else DEPTHS=${BLOOMERY_NSYS_DEPTHS:-6 4096}; fi
MODE=${BLOOMERY_NSYS_MODE:-graph}
# 디코드 스텝 수. 마지막 재생을 표로 내고, 그 앞 재생을 대조로 낸다.
NGEN=${BLOOMERY_NSYS_N:-4}
OUTDIR=${BLOOMERY_NSYS_OUT:-$BLOOMERY_DATA/nsys}
TOP=${BLOOMERY_NSYS_TOP:-24}

# The tables: sqlite, depth, n, top, form, the run's own output, the seed form's marker.
analyze() {
  timeout --kill-after=10 "${BLOOMERY_ARM_BOUND:-900}" python3 - "$@" <<'PY'
import re, sqlite3, sys
db, depth, ngen, top = sqlite3.connect(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
form, runlog = sys.argv[5], sys.argv[6]
names = dict(db.execute("SELECT id, value FROM StringIds"))
rows = db.execute("SELECT start, end, shortName, gridX, gridY, gridZ FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start").fetchall()
ks = [(s, e, names.get(n, str(n)).split("(")[0], (gx, gy, gz)) for s, e, n, gx, gy, gz in rows]
print(f"    커널 런치 전체 {len(ks)}개")
from collections import Counter
cnt = Counter(k[2] for k in ks)
if form == "prefill":
    # The prompt's K units (ubatches, a tail pass), each opening with one marker launch, then n - 1
    # graph replays, each opening with one. K is the run's own `time prompt ... passes=K`.
    marker = sys.argv[7]
    units = p_n = p_ms = None
    try:
        for line in open(runlog):
            m = re.match(r"time prompt n=(\d+) ms=([0-9.]+) tok/s=\S+ passes=(\d+) kind=(\w+)", line)
            if m:
                p_n, p_ms, units = int(m.group(1)), float(m.group(2)), int(m.group(3))
                print("    " + line.rstrip())
            elif line.startswith(("load ", "capture ", "step 0 ")):
                print("    " + line.rstrip())
    except OSError:
        pass
    if units is None:
        print(f"[boundary] no `time prompt n= ms= tok/s= passes= kind=` row in {runlog}: the unit count is unknown, no table")
        raise SystemExit(3)
    if p_n != depth:
        print(f"[boundary] the run's prompt is n={p_n}, the profile asked for P={depth}: no table")
        raise SystemExit(3)
    replays = units + ngen - 1
    bounds = [i for i, k in enumerate(ks) if k[2] == marker]
    print(f"[boundary] kernel {marker}: {len(bounds)} launches, expected {units} prompt unit(s) + n - 1 replays = {replays} "
          f"-> {'OK' if len(bounds) == replays else 'MISMATCH'}")
    if len(bounds) != replays:
        print("    most frequent names:")
        for n, c in cnt.most_common(12):
            print(f"      {c:7d}  {n}")
        raise SystemExit(3)
elif form == "seed":
    # One eager prefill pass for the seeded prompt's single id, then n - 1 graph replays.
    replays = ngen
    marker = sys.argv[7]
    bounds = [i for i, k in enumerate(ks) if k[2] == marker]
    cands = sorted(n for n, c in cnt.items() if c == replays)
    print(f"[boundary] kernel {marker}: {len(bounds)} launches, expected 1 prefill pass + n - 1 replays = {replays} "
          f"-> {'OK' if len(bounds) == replays else 'MISMATCH'}; once-per-pass candidates: {', '.join(cands) or 'none'}")
    if len(bounds) != replays:
        print("    most frequent names:")
        for n, c in cnt.most_common(12):
            print(f"      {c:7d}  {n}")
        raise SystemExit(3)
else:
    replays = depth + ngen - 1
    # 재생마다 정확히 한 번 나오는 커널이 경계다. 여럿이면 가장 먼저 나오는 것.
    cands = [n for n, c in cnt.items() if c == replays]
    if not cands:
        print(f"    경계 없음: 재생 {replays}회에 맞는 커널이 없다. 이름별 횟수 상위:")
        for n, c in cnt.most_common(12):
            print(f"      {c:7d}  {n}")
        raise SystemExit(3)
    first_idx = {n: next(i for i, k in enumerate(ks) if k[2] == n) for n in cands}
    marker = min(cands, key=first_idx.get)
    bounds = [i for i, k in enumerate(ks) if k[2] == marker]
    print(f"    경계 커널 {marker} (재생당 1회, {len(bounds)}회 — 기대 {replays}) 후보 {len(cands)}개")
bounds.append(len(ks))
def window(r):
    # 마지막 재생 뒤에는 헤드(argmax)가 붙는다 — 그래프 밖이지만 스텝의 일부이므로 같이 센다.
    return ks[bounds[r]:bounds[r + 1]]
def period(r):
    # 이 재생 시작에서 다음 재생 시작까지. 디코드 스텝이면 호스트 턴어라운드(sync·복사)까지 든
    # 값이라 depth-gpu.sh의 스텝 ms와 맞아야 하는 쪽은 이것이다. 마지막 재생에는 다음이 없다.
    return (ks[bounds[r + 1]][0] - ks[bounds[r]][0]) / 1e3 if bounds[r + 1] < len(ks) else float("nan")
def report(label, w, per):
    wall = (w[-1][1] - w[0][0]) / 1e3
    tot = sum(e - s for s, e, _, _ in w) / 1e3
    print(f"  [{label}] 커널 {len(w)}개  다음 재생까지 {per:9.1f} µs  첫~끝 {wall:9.1f} µs  커널 합 {tot:9.1f} µs  빈틈 {wall - tot:8.1f} µs")
    by = {}
    for s, e, n, g in w:
        d_ = by.setdefault(n, [0.0, 0, set()])
        d_[0] += (e - s) / 1e3; d_[1] += 1; d_[2].add(g)
    print(f"    {'커널':40s} {'런치':>5s} {'합 µs':>10s} {'몫':>7s} {'평균 µs':>9s}  그리드")
    for n, (s, c, g) in sorted(by.items(), key=lambda x: -x[1][0])[:top]:
        gs = " ".join("x".join(map(str, t)) for t in sorted(g)[:3])
        print(f"    {n[:40]:40s} {c:5d} {s:10.1f} {100 * s / (tot or 1):6.1f}% {s / c:9.2f}  {gs}")
if form == "prefill":
    w = ks[bounds[0]:bounds[units]]
    wall = (w[-1][1] - w[0][0]) / 1e6
    tot = sum(e - s for s, e, _, _ in w) / 1e6
    print(f"  [prefill window: P = {depth}, {units} unit(s) and the head] kernels {len(w)}  first-to-last {wall:.3f} ms  "
          f"kernel sum {tot:.3f} ms  gaps {wall - tot:.3f} ms  time prompt ms={p_ms:.3f} (outside the window {p_ms - wall:.3f} ms)")
    by = {}
    for s, e, n, g in w:
        d_ = by.setdefault(n, [0.0, 0, set()])
        d_[0] += (e - s) / 1e6; d_[1] += 1; d_[2].add(g)
    gemm = sum(v[0] for n, v in by.items() if re.fullmatch(r"gemm_q\dk", n))
    print(f"    grouped GEMM (gemm_q*k) {gemm:.3f} ms: {100 * gemm / (tot or 1):.1f} % of the kernel sum, "
          f"{100 * gemm / (wall or 1):.1f} % of the window, {100 * gemm / p_ms:.1f} % of time prompt")
    print(f"    {'kernel':40s} {'launches':>8s} {'total ms':>10s} {'% sum':>7s} {'mean ms':>9s}  grids")
    for n, (s, c, g) in sorted(by.items(), key=lambda x: -x[1][0])[:top]:
        gs = " ".join("x".join(map(str, t)) for t in sorted(g)[:3])
        print(f"    {n[:40]:40s} {c:8d} {s:10.3f} {100 * s / (tot or 1):6.1f}% {s / c:9.4f}  {gs}")
    r = replays - 1
    last = window(r)
    print(f"  [control: replay {r - units + 1} of {ngen - 1}, the last decode step] kernels {len(last)}  "
          f"first-to-last {(last[-1][1] - last[0][0]) / 1e3:.1f} µs  kernel sum {sum(e - s for s, e, _, _ in last) / 1e3:.1f} µs")
elif form == "seed":
    # The run's own lines: `time step i` is window i; window 0 is the prefill pass.
    timed = {}
    try:
        for line in open(runlog):
            m = re.match(r"time step (\d+)( warm)? ms=([0-9.]+)", line)
            if m:
                timed[int(m.group(1))] = float(m.group(3))
            elif line.startswith(("load ", "capture ", "seed ", "step 0 ", "SMOKE ")):
                print("    " + line.rstrip())
    except OSError:
        pass
    print(f"  {'window':>6s} {'depth':>6s} {'kind':>8s} {'time step ms':>12s} {'period µs':>10s} {'wall µs':>10s} {'kern sum':>10s} {'gap':>8s} {'kernels':>7s}")
    for r in range(replays):
        w = window(r)
        wall = (w[-1][1] - w[0][0]) / 1e3
        tot = sum(e - s for s, e, _, _ in w) / 1e3
        ts = f"{timed[r]:.4f}" if r in timed else ("-" if r else "token 0")
        print(f"  {r:6d} {depth - 1 + r:6d} {'prefill' if r == 0 else 'replay':>8s} {ts:>12s} {period(r):10.1f} {wall:10.1f} {tot:10.1f} {wall - tot:8.1f} {len(w):7d}")
    report(f"window 0 = depth {depth - 1} (the seeded prompt's one id, eager prefill pass)", window(0), period(0))
    report(f"window {replays - 2} = depth {depth + replays - 3} (replay)", window(replays - 2), period(replays - 2))
    report(f"window {replays - 1} = depth {depth + replays - 2} (last decode replay)", window(replays - 1), period(replays - 1))
else:
    # 프롬프트 첫 재생(깊이 0), 마지막 앞 재생, 마지막 재생. 처음과 끝의 차가 깊이 비용이다.
    report(f"재생 0 = 깊이 0 (프롬프트 첫 토큰)", window(0), period(0))
    report(f"재생 {replays - 2} = 깊이 {depth + ngen - 3}", window(replays - 2), period(replays - 2))
    report(f"재생 {replays - 1} = 깊이 {depth + ngen - 2} (마지막 디코드)", window(replays - 1), period(replays - 1))
PY
}

if [ "${1:-}" = --analyze ]; then
  [ $# -ge 4 ] || { echo "usage: nsys-gpu.sh --analyze <sqlite> <depth> <n> [<run log>]" >&2; exit 64; }
  analyze "$2" "$3" "$4" "$TOP" "$FORM" "${5:-/dev/null}" "$MARKER"
  exit $?
fi
if [ "$FORM" = seed ]; then
  for d in $DEPTHS; do
    case $d in
      '' | *[!0-9]* | 0) echo "nsys-gpu.sh: depth '$d' is a positive integer (the seed form feeds one id at D - 1)" >&2; exit 64 ;;
    esac
  done
  case $NGEN in
    '' | *[!0-9]* | [012]) echo "nsys-gpu.sh: BLOOMERY_NSYS_N is at least 3 in the seed form, got '$NGEN'" >&2; exit 64 ;;
  esac
  case $GEN_CTX in
    *[!0-9]* | 0) echo "nsys-gpu.sh: BLOOMERY_GEN_CTX is a positive integer, got '$GEN_CTX'" >&2; exit 64 ;;
  esac
fi
if [ "$FORM" = prefill ]; then
  for d in $DEPTHS; do
    case $d in
      '' | *[!0-9]* | [0-8]) echo "nsys-gpu.sh: prompt length '$d' is an integer >= 9 in the prefill form (a GEMM ubatch runs)" >&2; exit 64 ;;
    esac
  done
  case $NGEN in
    '' | *[!0-9]* | [01]) echo "nsys-gpu.sh: BLOOMERY_NSYS_N is at least 2 in the prefill form (a replay closes the window), got '$NGEN'" >&2; exit 64 ;;
  esac
  case $GEN_CTX in
    *[!0-9]* | 0) echo "nsys-gpu.sh: BLOOMERY_GEN_CTX is a positive integer, got '$GEN_CTX'" >&2; exit 64 ;;
  esac
fi
# The profiled command for depth $1 (the prompt and prefill forms' ids: $2), into CMD and CTX.
profile_cmd() {
  local d=$1 ids=${2:-}
  if [ "$FORM" = seed ]; then
    CTX=${GEN_CTX:-$(((d + DEPTH_N + 255) / 256 * 256))}
    CMD=(timeout --kill-after=10 "$BOUND" "$NSYS" profile -t cuda --cuda-graph-trace=node --cuda-event-trace=false
         --sample=none --cpuctxsw=none -o "$out" --force-overwrite true
         "$BIN" --seed-depth "$d" -n "$NGEN" --ctx "$CTX" --mode "$MODE" --time)
  elif [ "$FORM" = prefill ]; then
    CTX=${GEN_CTX:-$(((d + DEPTH_N + 255) / 256 * 256))}
    CMD=(timeout --kill-after=10 "$BOUND" "$NSYS" profile -t cuda --cuda-graph-trace=node --cuda-event-trace=false
         --sample=none --cpuctxsw=none -o "$out" --force-overwrite true
         "$BIN" --tokens "$ids" -n "$NGEN" --ctx "$CTX" --mode "$MODE" --time)
  else
    CTX=$((d + 128))
    CMD=(timeout --kill-after=10 "$BOUND" "$NSYS" profile -t cuda --cuda-graph-trace=node --cuda-event-trace=false
         --sample=none --cpuctxsw=none -o "$out" --force-overwrite true
         "$BIN" --tokens "$ids" -n "$NGEN" --ctx "$CTX" --mode "$MODE")
  fi
}
# 카드 핀·증인 줄·바이너리 신선도는 러너 넷이 같은 파일에서 읽는다.
# shellcheck source=tools/ref/timing-card.sh
source "${BASH_SOURCE[0]%/*}/timing-card.sh"
# The lease and the witness fields.
# shellcheck source=tools/ref/lease.sh
source "${BASH_SOURCE[0]%/*}/lease.sh"

if [ -n "$DRY" ]; then
  echo "[dry] form=$FORM model=$MODEL bin=$BIN mode=$MODE n=$NGEN depths='$DEPTHS' out=$OUTDIR timing_gpu=$TIMING_GPU CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  for d in $DEPTHS; do
    out="$OUTDIR/<name>"
    profile_cmd "$d" "<lcg_prompt $d>"
    if [ "$FORM" = seed ]; then
      echo "[dry] depth $d: ctx $CTX, windows expected $NGEN (1 prefill pass + $((NGEN - 1)) replays), boundary $MARKER"
      echo "[dry] depth $d: ${CMD[*]}"
    elif [ "$FORM" = prefill ]; then
      echo "[dry] prompt $d: ctx $CTX, boundaries expected K + $((NGEN - 1)) (K = the run's time prompt passes=), boundary $MARKER"
      echo "[dry] prompt $d: ${CMD[*]}"
    else
      echo "[dry] depth $d: ctx $CTX, replays expected $((d + NGEN - 1))"
      echo "[dry] depth $d: ${CMD[*]}"
    fi
  done
  exit 0
fi

assert_fresh_binary "$BIN" || exit $?
[ -x "$NSYS" ] || { echo "no nsys at $NSYS" >&2; exit 2; }
mkdir -p "$OUTDIR"

WITNESS=(head-open indent card model)

lease_take
echo "[config] nsys=$($NSYS --version) mode=$MODE n=$NGEN depths='$DEPTHS' out=$OUTDIR"
[ "$FORM" != seed ] || echo "[config] form=seed bin=$BIN boundary=$MARKER ctx=${GEN_CTX:-D+$DEPTH_N rounded up to 256} bound=${BOUND}s timing_gpu=$TIMING_GPU other_gpu=$OTHER_GPU"
[ "$FORM" != prefill ] || echo "[config] form=prefill bin=$BIN boundary=$MARKER ctx=${GEN_CTX:-P+$DEPTH_N rounded up to 256} bound=${BOUND}s timing_gpu=$TIMING_GPU other_gpu=$OTHER_GPU"
witness pre

rc_all=0
for d in $DEPTHS; do
  if [ "$FORM" = seed ]; then
    out="$OUTDIR/nsys-qwen3moe-d${d}-n${NGEN}-${MODE}-$(date -u +%H%M%S)"
    profile_cmd "$d"
    echo
    echo "=== depth $d (ctx $CTX, mode $MODE, 1 prefill pass + $((NGEN - 1)) replays expected) -> $out.nsys-rep"
    guard_other
  elif [ "$FORM" = prefill ]; then
    out="$OUTDIR/nsys-qwen3moe-pp${d}-n${NGEN}-${MODE}-$(date -u +%H%M%S)"
    profile_cmd "$d" "$(lcg_prompt "$d")"
    echo
    echo "=== prompt $d (ctx $CTX, mode $MODE, the prompt's units + $((NGEN - 1)) replays expected) -> $out.nsys-rep"
    guard_other
  else
    out="$OUTDIR/nsys-d${d}-${MODE}-$(date -u +%H%M%S)"
    profile_cmd "$d" "$(lcg_prompt "$d")"
    echo
    echo "=== 깊이 $d (ctx $CTX, mode $MODE, 재생 $((d + NGEN - 1))개 기대) → $out.nsys-rep"
  fi
  witness "pre d=$d"
  "${CMD[@]}" > "$out.txt" 2>&1
  rc=$?
  witness "post d=$d"
  echo "[rc] $rc"
  [ $rc -eq 0 ] || { rc_all=$rc; echo "--- 마지막 20줄"; tail -n 20 "$out.txt"; continue; }
  lease_bounded "$BOUND" "$NSYS" export --type sqlite --force-overwrite true -o "$out.sqlite" "$out.nsys-rep" > "$out.export.txt" 2>&1 \
    || { rc_all=$?; echo "[export] 실패"; tail -n 10 "$out.export.txt"; continue; }
  analyze "$out.sqlite" "$d" "$NGEN" "$TOP" "$FORM" "$out.txt" "$MARKER" || rc_all=$?
  echo "--- 원본: $out.nsys-rep, $out.sqlite"
done

witness post
echo "[lease] released at $(now)"
exit $rc_all
