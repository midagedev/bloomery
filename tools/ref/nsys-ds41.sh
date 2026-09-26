#!/usr/bin/env bash
# V4.1 kernel timeline: where one decode step's time goes, kernel time against the host tier's
# joins, asked of Nsight Systems on the timing card (run on the box, under the lease).
#
#   BLOOMERY_MODEL=deepseek41 tools/box.sh 'bash tools/ref/nsys-ds41.sh 6 1024'
#   just nsys-gpu-ds41 6 1024
#   bash tools/ref/nsys-ds41.sh --analyze <out>.sqlite <depth> <n> [<out>.txt]   # tables again, no profile
#
# The V4.1 sibling of nsys-gpu.sh. Its step boundary is the same (a kernel that runs exactly once
# per replay, counted against the replays the run makes) and so is its refusal: no table when the
# count is wrong. What differs is why this file exists:
#   - the binary is generate_ds41 at its default context: the placement plan's card expert prefix
#     depends on ctx_max, so a per-depth --ctx (nsys-gpu.sh's d + 128) would profile another
#     placement than the one depth-ds41.sh times;
#   - the run is `--depth D -n N --mode graph --time`, so the SMOKE and `time step` lines of the
#     profiled run sit next to its replay periods;
#   - every routed layer joins the host tier inside the replay. Per layer the stream runs
#     handoff → go → [HC_PRE, card experts, shared expert] → wait → post (chain/ffn.rs). The go and
#     the wait are stream memory-operation batches, which the trace records as nothing: the wait's
#     time is the gap in front of the layer's `ds41_ffn_post*` kernel. So per layer:
#       bridge   = post.start − handoff.end   (go to join as the card sees it: host compute plus
#                                              both signalling latencies)
#       overlap  = kernel time between the two (card work done while the host computes)
#       exposed  = post.start − end of the kernel before it (what the step waited for the host)
#     and exposed − the replay's median inter-kernel gap is the part the host tier added.
#
# Replays: generate_ds41 feeds its D ids as the `load` line's `prefill=` says. `batch` (the default)
# runs them eagerly through body::prefill — kernels with no graph node, one `time prompt ...
# kind=batch` row — so the replays are the N − 1 generated steps: replay r is `time step r + 1` at
# depth D + r. `steps` (BLOOMERY_PREFILL=steps) feeds one graph replay an id: D + N − 1 replays, the
# last at depth D + N − 2; replay D − 1 produces token 0 and is fed untimed, replays D .. D + N − 2
# are `time step 1 .. N−1`. The boundary is the replay's first graph kernel among those that run once
# a replay, searched among the graph's kernels only: the batch's eager kernels carry the same names,
# and the batch embed launches ds41_glue_embed_q3k once a prompt token, so over all kernels that one
# name alone counts D + N − 1 under the batch feed and would open each window at the replay's second
# node. The trace's own eager kernels decide the feed: the run log's `time prompt` row must say the
# same, or no table. The batch feed is what depth-ds41.sh times and leaves the caches as the steps
# would; at depth 1024 the step feed would add ≈ 1024 replays of wall and trace. The capture before
# the prompt executes no kernel.
#
# µs here are the trace's. A replay's period (its first kernel's start to the next replay's) holds
# the host turnaround too (argmax readback, the step's host half, the image copy) and is the value
# that must agree with the `time step` line of the same step. CPU sampling and context-switch
# tracing are off: the host tier spins a whole pool, and sampling it would perturb the bridge this
# runner measures.
#
# The prefill form (BLOOMERY_NSYS_FORM=prefill, `just nsys-gpu-ds41-prefill [P...]`): the prompt batch
# instead of a decode step. Each argument is a prompt length P >= 9 (default 512); the profiled command
# is `generate_ds41 --depth P -n N --mode graph --time` (N = BLOOMERY_NSYS_N, default 2, at least 2 so
# that a replay closes the window), depth-ds41.sh's `<P>` arm at another N: the fed ids are lease.sh's
# lcg_prompt P, in batches. The window runs from the prompt's first kernel to the first replay; inside
# it tools/ref/ds41pp.py cuts the layer-batches at `ds41_ffn_places` and the joins, counted against the
# run's `stat prefill split` and `stat prefill ced=` lines (no table on a wrong count), and prints per
# layer-batch the route window (the layer's first kernel to the route's last copy to the host), the
# union gap (the card holding only the shadow while the host runs the union) and the post, the kernel
# table of one layer-batch (BLOOMERY_NSYS_LAYER, default 2) and the mean over layers 2-39 grouped into
# terms, the card's idle time in the route window (the gaps), and the launch queue from the CUDA API
# trace (per layer-batch the enqueue calls, the most activities in flight when the queue fills, from
# which activity it is full, the calls longer than BLOOMERY_NSYS_BLOCKED_US µs, default 8, after and
# before that, and the queue model at those values). ds41pp.py's header has the cut. The profile writes
# <out>.meta beside the report (the binary's sha256, P, N, the command, the hot list): the ncu ds41pp
# form reads it and the sqlite to derive its launch skip. `--analyze <sqlite> <P> <n> <run log>` under
# BLOOMERY_NSYS_FORM=prefill re-prints the tables (the run log is required: the cut reads it).
#
# Environment: BLOOMERY_NSYS_N (N, default 8, 2 in the prefill form), BLOOMERY_NSYS_LAST (replays
# tabled, default 8), BLOOMERY_NSYS_TOP (kernel rows, default 24), BLOOMERY_NSYS_OUT (default
# $BLOOMERY_DATA/nsys), BLOOMERY_GEN_BIN (default target/release/generate_ds41), BLOOMERY_ARM_BOUND
# (seconds one profile may run, default 900), BLOOMERY_DRY=1 (the command lines, then exit before the
# binary check and the lease).
set -uo pipefail
# shellcheck source=tools/ref/ref-paths.sh
source "${BASH_SOURCE[0]%/*}/ref-paths.sh"

FORM=decode
case ${BLOOMERY_NSYS_FORM:-} in
  '') ;;
  prefill) FORM=prefill ;;
  *) echo "nsys-ds41.sh: BLOOMERY_NSYS_FORM is prefill or unset, got '$BLOOMERY_NSYS_FORM'" >&2; exit 64 ;;
esac
if [ "$FORM" = prefill ]; then NGEN=${BLOOMERY_NSYS_N:-2}; else NGEN=${BLOOMERY_NSYS_N:-8}; fi
LAST=${BLOOMERY_NSYS_LAST:-8}
TOP=${BLOOMERY_NSYS_TOP:-24}
LAYER=${BLOOMERY_NSYS_LAYER:-2}
BLOCKED_US=${BLOOMERY_NSYS_BLOCKED_US:-8}
DRY=${BLOOMERY_DRY:-}
PP=${BASH_SOURCE[0]%/*}/ds41pp.py

# The analysis: sqlite, depth, n, the run's own output (for SMOKE and `time step`), last, top.
analyze() {
  timeout --kill-after=10 "${BLOOMERY_ARM_BOUND:-900}" python3 - "$@" << 'PY'
import sqlite3, statistics, sys, re
from collections import defaultdict
db = sqlite3.connect(sys.argv[1])
depth, ngen, last, top = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[5]), int(sys.argv[6])
runlog = sys.argv[4]
names = dict(db.execute("SELECT id, value FROM StringIds"))
rows = db.execute("SELECT start, end, shortName, graphNodeId FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start").fetchall()
# The batch feed's kernels run outside the graph; the replays' are its nodes.
eager = [r for r in rows if r[3] is None]
ks = [(s, e, names.get(n, str(n)).split("(")[0]) for s, e, n, g in rows if g is not None]
def rows_of(table):
    try:
        return db.execute(f"SELECT start, end, bytes FROM {table} ORDER BY start").fetchall()
    except sqlite3.OperationalError:
        return []
copies, sets = rows_of("CUPTI_ACTIVITY_KIND_MEMCPY"), rows_of("CUPTI_ACTIVITY_KIND_MEMSET")
feed = "batch" if eager else "steps"
kind = None
try:
    for line in open(runlog):
        m = re.match(r"time prompt n=\d+ ms=\S+ tok/s=\S+ passes=\d+ kind=(\w+)", line)
        if m:
            kind = m.group(1)
except OSError:
    pass
print(f"    kernels {len(rows)} ({len(eager)} outside the graph: the prompt's feed is {feed})  memcpy "
      f"{len(copies)}  memset {len(sets)}  (stream memory operations leave no record)")
if kind is not None and kind != feed:
    print(f"    NO TABLE: the run log's time prompt row says kind={kind}, the trace's kernels say {feed}")
    raise SystemExit(3)
if eager and ks and eager[-1][0] > ks[0][0]:
    print("    NO TABLE: a graph replay runs before the batch's last eager kernel")
    raise SystemExit(3)
fed = depth if feed == "steps" else 0
replays = fed + ngen - 1
cnt = defaultdict(int)
for k in ks:
    cnt[k[2]] += 1
cands = [n for n, c in cnt.items() if c == replays]
if not cands:
    print(f"    NO BOUNDARY: no graph kernel runs {replays} times. Most frequent names:")
    for n, c in sorted(cnt.items(), key=lambda x: -x[1])[:12]:
        print(f"      {c:8d}  {n}")
    raise SystemExit(3)
first = {n: next(i for i, k in enumerate(ks) if k[2] == n) for n in cands}
marker = min(cands, key=first.get)
bounds = [i for i, k in enumerate(ks) if k[2] == marker]
expect = "depth + n - 1" if feed == "steps" else "n - 1"
print(f"[boundary] kernel {marker}: {len(bounds)} replay launches, expected {expect} = {replays} "
      f"-> {'OK' if len(bounds) == replays else 'MISMATCH'}; once-per-replay candidates: {', '.join(sorted(cands))}")
if len(bounds) != replays:
    raise SystemExit(3)
bounds.append(len(ks))
if eager:
    print(f"    the batch: {len(eager)} eager kernels, first to last {(eager[-1][1] - eager[0][0]) / 1e6:.1f} ms")

# The run's own lines: `time step i` is replay i - 1 under the batch feed; under the step feed
# token 0 comes out of replay depth - 1 and `time step i` is replay depth - 1 + i.
timed = {}
smoke = ""
try:
    for line in open(runlog):
        m = re.match(r"time step (\d+)( warm)? ms=([0-9.]+)", line)
        if m:
            timed[fed - 1 + int(m.group(1))] = float(m.group(3))
        if line.startswith("SMOKE"):
            smoke = line.rstrip()
        elif line.startswith(("plan ", "load ", "capture ", "fed ", "time prompt ")):
            print("    " + line.rstrip())
except OSError:
    pass
if smoke:
    print("    " + smoke)

def is_post(n):
    return n.startswith("ds41_ffn_post")

def replay(r):
    w = ks[bounds[r]:bounds[r + 1]]
    t0 = w[0][0]
    t_next = ks[bounds[r + 1]][0] if bounds[r + 1] < len(ks) else None
    wall = (w[-1][1] - t0) / 1e3
    ksum = sum(e - s for s, e, _ in w) / 1e3
    gaps = [(w[i][0] - w[i - 1][1]) / 1e3 for i in range(1, len(w))]
    joins, j_idx = [], set()
    last_handoff = None
    for i, (s, e, n) in enumerate(w):
        if n == "ds41_ffn_handoff":
            last_handoff = i
        elif is_post(n) and last_handoff is not None:
            h = last_handoff
            bridge = (s - w[h][1]) / 1e3
            over = sum(ee - ss for ss, ee, _ in w[h + 1:i]) / 1e3
            exposed = (s - w[i - 1][1]) / 1e3
            joins.append((bridge, over, exposed, w[i - 1][2]))
            j_idx.add(i)
            last_handoff = None
    other = [gaps[i - 1] for i in range(1, len(w)) if i not in j_idx]
    base = statistics.median(other) if other else 0.0
    big = max(range(1, len(w)), key=lambda i: w[i][0] - w[i - 1][1])
    c_in = [c for c in copies if t0 <= c[0] < (t_next or float("inf"))]
    s_in = [c for c in sets if t0 <= c[0] < (t_next or float("inf"))]
    return dict(w=w, wall=wall, ksum=ksum, gap=wall - ksum, joins=joins, base=base,
                period=(t_next - t0) / 1e3 if t_next else float("nan"),
                big=(gaps[big - 1], w[big - 1][2], w[big][2], big in j_idx),
                other_gap=sum(other), copies=c_in, sets=s_in)

lo = max(0, replays - last)
R = {r: replay(r) for r in range(lo, replays)}
print()
print(f"=== per replay (last {replays - lo}; µs unless noted; `time step` in ms from the same run)")
print(f"  {'replay':>6s} {'depth':>5s} {'time step':>10s} {'period':>9s} {'wall':>9s} {'kern sum':>9s} {'gap':>8s} "
      f"{'joins':>5s} {'bridge':>8s} {'overlap':>8s} {'exposed':>8s} {'exp-base':>8s} {'base':>5s} {'other gap':>9s} "
      f"{'H2D/D2H':>8s}  largest gap")
for r, x in R.items():
    j = x["joins"]
    b, o, e = (sum(t[k] for t in j) for k in range(3))
    xs = e - len(j) * x["base"]
    cp = sum(c[1] - c[0] for c in x["copies"] + x["sets"]) / 1e3
    ts = f"{timed[r]:.3f}" if r in timed else ("token 0" if r == depth - 1 else "fed")
    g, a, bb, isj = x["big"]
    print(f"  {r:6d} {r + (depth if feed == 'batch' else 0):5d} {ts:>10s} {x['period']:9.1f} {x['wall']:9.1f} {x['ksum']:9.1f} {x['gap']:8.1f} "
          f"{len(j):5d} {b:8.1f} {o:8.1f} {e:8.1f} {xs:8.1f} {x['base']:5.2f} {x['other_gap']:9.1f} {cp:8.1f}  "
          f"{g:.1f} ({a} -> {bb}{', join' if isj else ''})")

dec = [r for r in R if r in timed]
if dec:
    def mean(f):
        return statistics.fmean(f(R[r]) for r in dec)
    print()
    print(f"=== mean of the {len(dec)} timed decode replays above (ms)")
    tstep = statistics.fmean(timed[r] for r in dec)
    # The last replay has no next one, so its period is unknown: the period means pair only the
    # replays that have one, each with its own `time step` and wall.
    pr = [r for r in dec if R[r]["period"] == R[r]["period"]]
    per = statistics.fmean(R[r]["period"] for r in pr) / 1e3 if pr else float("nan")
    tper = statistics.fmean(timed[r] for r in pr) if pr else float("nan")
    wper = statistics.fmean(R[r]["wall"] for r in pr) / 1e3 if pr else float("nan")
    wall = mean(lambda x: x["wall"]) / 1e3
    ksum = mean(lambda x: x["ksum"]) / 1e3
    bridge = mean(lambda x: sum(t[0] for t in x["joins"])) / 1e3
    over = mean(lambda x: sum(t[1] for t in x["joins"])) / 1e3
    exp_ = mean(lambda x: sum(t[2] for t in x["joins"])) / 1e3
    othg = mean(lambda x: x["other_gap"]) / 1e3
    print(f"  time step {tstep:.3f} over {len(dec)} steps; over the {len(pr)} with a next replay: time step "
          f"{tper:.3f}  period {per:.3f} ({100 * (per / tper - 1):+.2f} %)  wall {wper:.3f}  "
          f"host turnaround (period - wall) {per - wper:.3f}")
    print(f"  kernel sum (GPU serial) {ksum:.3f}  gap {wall - ksum:.3f} = join gaps {exp_:.3f} + other gaps {othg:.3f}")
    print(f"  host bridge (go -> join, all layers) {bridge:.3f}  card overlap inside it {over:.3f}  "
          f"exposed {exp_:.3f}  (bridge - overlap = {bridge - over:.3f}, the rest is launch gaps inside the bridge)")

    # Per layer, averaged over the timed replays.
    nj = min(len(R[r]["joins"]) for r in dec)
    print()
    print(f"=== per routed layer, mean over the timed replays (µs): {nj} joins per replay")
    print(f"  {'join':>4s} {'bridge':>8s} {'overlap':>8s} {'exposed':>8s}  kernel before the join")
    for k in range(nj):
        vals = [R[r]["joins"][k] for r in dec]
        print(f"  {k:4d} {statistics.fmean(v[0] for v in vals):8.1f} {statistics.fmean(v[1] for v in vals):8.1f} "
              f"{statistics.fmean(v[2] for v in vals):8.1f}  {vals[0][3]}")

    # Kernels by time, per step.
    by = defaultdict(lambda: [0.0, 0])
    for r in dec:
        for s, e, n in R[r]["w"]:
            by[n][0] += (e - s) / 1e3
            by[n][1] += 1
    tot = sum(v[0] for v in by.values())
    print()
    print(f"=== kernels by time per step, mean over the timed replays (kernel sum {tot / len(dec):.1f} µs)")
    print(f"  {'kernel':44s} {'launch':>6s} {'µs':>9s} {'share':>7s} {'µs/launch':>9s}")
    for n, (s, c) in sorted(by.items(), key=lambda x: -x[1][0])[:top]:
        print(f"  {n[:44]:44s} {c / len(dec):6.0f} {s / len(dec):9.1f} {100 * s / tot:6.1f}% {s / c:9.2f}")
    rest = sorted(by.items(), key=lambda x: -x[1][0])[top:]
    if rest:
        print(f"  {'(' + str(len(rest)) + ' more kernels)':44s} {sum(v[1] for _, v in rest) / len(dec):6.0f} "
              f"{sum(v[0] for _, v in rest) / len(dec):9.1f} {100 * sum(v[0] for _, v in rest) / tot:6.1f}%")
    cps = [c for r in dec for c in R[r]["copies"]]
    sts = [c for r in dec for c in R[r]["sets"]]
    print(f"  memcpy per step: {len(cps) / len(dec):.1f} ({sum(c[1] - c[0] for c in cps) / 1e3 / len(dec):.1f} µs, "
          f"{sum(c[2] for c in cps) / len(dec):.0f} B)  memset per step: {len(sts) / len(dec):.1f} "
          f"({sum(c[1] - c[0] for c in sts) / 1e3 / len(dec):.1f} µs)")
PY
}

# The prefill form's tables: sqlite, P, n, the run log.
analyze_prefill() {
  timeout --kill-after=10 "${BLOOMERY_ARM_BOUND:-900}" python3 "$PP" tables "$1" "$2" "$3" "$4" --layer "$LAYER" --blocked-us "$BLOCKED_US"
}

if [ "${1:-}" = --analyze ]; then
  if [ "$FORM" = prefill ]; then
    [ $# -ge 5 ] || { echo "usage: BLOOMERY_NSYS_FORM=prefill nsys-ds41.sh --analyze <sqlite> <P> <n> <run log>" >&2; exit 64; }
    analyze_prefill "$2" "$3" "$4" "$5"
    exit $?
  fi
  [ $# -ge 4 ] || { echo "usage: nsys-ds41.sh --analyze <sqlite> <depth> <n> [<run log>]" >&2; exit 64; }
  analyze "$2" "$3" "$4" "${5:-/dev/null}" "$LAST" "$TOP"
  exit $?
fi

[ "$MODEL_NAME" = deepseek41 ] || {
  echo "nsys-ds41.sh: the profile is $MODEL_NAME — pick deepseek41 on the Mac side (BLOOMERY_MODEL=deepseek41)" >&2
  exit 64
}
DEPTHS=("$@")
if [ "$FORM" = prefill ]; then
  [ ${#DEPTHS[@]} -gt 0 ] || DEPTHS=(512)
  for d in "${DEPTHS[@]}"; do
    case $d in
      '' | *[!0-9]* | [0-8]) echo "nsys-ds41.sh: prompt length '$d' is an integer >= 9 in the prefill form" >&2; exit 64 ;;
    esac
  done
  case $NGEN in
    '' | *[!0-9]* | [01]) echo "nsys-ds41.sh: BLOOMERY_NSYS_N is at least 2 in the prefill form (a replay closes the window), got '$NGEN'" >&2; exit 64 ;;
  esac
else
  [ ${#DEPTHS[@]} -gt 0 ] || DEPTHS=(6)
  for d in "${DEPTHS[@]}"; do
    case $d in
      '' | *[!0-9]*) echo "nsys-ds41.sh: depth '$d' is not a number" >&2; exit 64 ;;
    esac
    [ "$d" -ge 1 ] || { echo "nsys-ds41.sh: depth $d: the run feeds at least one id" >&2; exit 64; }
  done
fi
BIN=${BLOOMERY_GEN_BIN:-target/release/generate_ds41}
NSYS=${NSYS:-/usr/local/cuda/bin/nsys}
OUTDIR=${BLOOMERY_NSYS_OUT:-$BLOOMERY_DATA/nsys}
BOUND=${BLOOMERY_ARM_BOUND:-900}
# The card pin, the card's witness lines, the other-card guard and the binary's freshness.
# shellcheck source=tools/ref/timing-card.sh
source "${BASH_SOURCE[0]%/*}/timing-card.sh"
# The lease and the witness fields.
# shellcheck source=tools/ref/lease.sh
source "${BASH_SOURCE[0]%/*}/lease.sh"

# The profiled command for depth or prompt length $1, into CMD; the report path is $out.
profile_cmd() {
  CMD=(timeout --kill-after=10 "$BOUND"
       "$NSYS" profile -t cuda --cuda-graph-trace=node --cuda-event-trace=false
       --sample=none --cpuctxsw=none -o "$out" --force-overwrite true
       "$BIN" --depth "$1" -n "$NGEN" --mode graph --time)
}

if [ -n "$DRY" ]; then
  echo "[dry] form=$FORM bin=$BIN n=$NGEN args='${DEPTHS[*]}' out=$OUTDIR timing_gpu=$TIMING_GPU CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES hot_list=${BLOOMERY_HOT_LIST:-<unset>}"
  for d in "${DEPTHS[@]}"; do
    out="$OUTDIR/<name>"
    profile_cmd "$d"
    echo "[dry] $d: ${CMD[*]}"
  done
  exit 0
fi

assert_fresh_binary "$BIN" || exit $?
[ -x "$NSYS" ] || { echo "no nsys at $NSYS" >&2; exit 2; }
mkdir -p "$OUTDIR"

# shellcheck disable=SC2034 # read by lease.sh's witness()
WITNESS=(head-open indent card busiest model mem pgmajfault)

lease_take
echo "[config] form=$FORM nsys=$($NSYS --version) n=$NGEN args='${DEPTHS[*]}' last=$LAST out=$OUTDIR bound=${BOUND}s"
echo "[config] timing_gpu=$TIMING_GPU other_gpu=$OTHER_GPU CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES hot_list=${BLOOMERY_HOT_LIST:-<unset>}"
witness pre
guard_other

rc_all=0
for d in "${DEPTHS[@]}"; do
  if [ "$FORM" = prefill ]; then
    out="$OUTDIR/nsys-ds41-pp${d}-n${NGEN}-$(date -u +%H%M%S)"
    echo
    echo "=== prompt $d (n $NGEN: the prompt's batches, then $((NGEN - 1)) replay(s)) -> $out.nsys-rep"
  else
    out="$OUTDIR/nsys-ds41-d${d}-n${NGEN}-$(date -u +%H%M%S)"
    echo
    echo "=== depth $d (n $NGEN: $((NGEN - 1)) replays after a batch feed, $((d + NGEN - 1)) after a step feed) -> $out.nsys-rep"
  fi
  profile_cmd "$d"
  guard_other
  witness "pre d=$d"
  t0=$(date +%s)
  "${CMD[@]}" > "$out.txt" 2>&1
  rc=$?
  t1=$(date +%s)
  witness "post d=$d"
  echo "[rc] $rc wall $((t1 - t0))s"
  [ $rc -eq 0 ] || { rc_all=$rc; echo "--- last 20 lines of $out.txt"; tail -n 20 "$out.txt"; continue; }
  if [ "$FORM" = prefill ]; then
    {
      echo "bin=$BIN_PATH"
      echo "sha256=$BIN_SHA"
      echo "P=$d"
      echo "n=$NGEN"
      echo "hot_list=${BLOOMERY_HOT_LIST:-}"
      echo "cmd=${CMD[*]}"
    } > "$out.meta"
  fi
  t0=$(date +%s)
  lease_bounded "$LEASE_ARM_BOUND" "$NSYS" export --type sqlite --force-overwrite true -o "$out.sqlite" "$out.nsys-rep" > "$out.export.txt" 2>&1 \
    || { rc_all=1; echo "[export] failed"; tail -n 10 "$out.export.txt"; continue; }
  echo "[export] $(($(date +%s) - t0))s"
  if [ "$FORM" = prefill ]; then
    analyze_prefill "$out.sqlite" "$d" "$NGEN" "$out.txt" || rc_all=$?
    echo "--- files: $out.nsys-rep, $out.sqlite, $out.txt, $out.meta"
  else
    analyze "$out.sqlite" "$d" "$NGEN" "$out.txt" "$LAST" "$TOP" || rc_all=$?
    echo "--- files: $out.nsys-rep, $out.sqlite, $out.txt"
  fi
done

witness post
echo "[lease] released at $(now)"
exit $rc_all
