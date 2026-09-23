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
# Replays: generate_ds41 feeds D ids, one graph replay each, then N − 1 generated steps, one replay
# each: D + N − 1 replays, the last at depth D + N − 2. Replay D − 1 produces token 0 and is fed
# untimed; replays D .. D + N − 2 are `time step 1 .. N−1`. The capture before the prompt executes
# no kernel.
#
# µs here are the trace's. A replay's period (its first kernel's start to the next replay's) holds
# the host turnaround too (argmax readback, the step's host half, the image copy) and is the value
# that must agree with the `time step` line of the same step. CPU sampling and context-switch
# tracing are off: the host tier spins a whole pool, and sampling it would perturb the bridge this
# runner measures.
#
# Environment: BLOOMERY_NSYS_N (N, default 8), BLOOMERY_NSYS_LAST (replays tabled, default 8),
# BLOOMERY_NSYS_TOP (kernel rows, default 24), BLOOMERY_NSYS_OUT (default $BLOOMERY_DATA/nsys),
# BLOOMERY_GEN_BIN (default target/release/generate_ds41), BLOOMERY_ARM_BOUND (seconds one profile
# may run, default 900).
set -uo pipefail
# shellcheck source=tools/ref/ref-paths.sh
source "${BASH_SOURCE[0]%/*}/ref-paths.sh"

NGEN=${BLOOMERY_NSYS_N:-8}
LAST=${BLOOMERY_NSYS_LAST:-8}
TOP=${BLOOMERY_NSYS_TOP:-24}

# The analysis: sqlite, depth, n, the run's own output (for SMOKE and `time step`), last, top.
analyze() {
  python3 - "$@" << 'PY'
import sqlite3, statistics, sys, re
from collections import defaultdict
db = sqlite3.connect(sys.argv[1])
depth, ngen, last, top = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[5]), int(sys.argv[6])
runlog = sys.argv[4]
names = dict(db.execute("SELECT id, value FROM StringIds"))
ks = [(s, e, names.get(n, str(n)).split("(")[0])
      for s, e, n in db.execute("SELECT start, end, shortName FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start")]
def rows(table):
    try:
        return db.execute(f"SELECT start, end, bytes FROM {table} ORDER BY start").fetchall()
    except sqlite3.OperationalError:
        return []
copies, sets = rows("CUPTI_ACTIVITY_KIND_MEMCPY"), rows("CUPTI_ACTIVITY_KIND_MEMSET")
print(f"    kernels {len(ks)}  memcpy {len(copies)}  memset {len(sets)}  (stream memory operations leave no record)")
replays = depth + ngen - 1
cnt = defaultdict(int)
for k in ks:
    cnt[k[2]] += 1
cands = [n for n, c in cnt.items() if c == replays]
if not cands:
    print(f"    NO BOUNDARY: no kernel runs {replays} times. Most frequent names:")
    for n, c in sorted(cnt.items(), key=lambda x: -x[1])[:12]:
        print(f"      {c:8d}  {n}")
    raise SystemExit(3)
first = {n: next(i for i, k in enumerate(ks) if k[2] == n) for n in cands}
marker = min(cands, key=first.get)
bounds = [i for i, k in enumerate(ks) if k[2] == marker]
print(f"[boundary] kernel {marker}: {len(bounds)} launches, expected depth + n - 1 = {replays} "
      f"-> {'OK' if len(bounds) == replays else 'MISMATCH'}; once-per-replay candidates: {', '.join(sorted(cands))}")
if len(bounds) != replays:
    raise SystemExit(3)
bounds.append(len(ks))

# The run's own lines: token 0 comes out of replay depth - 1, `time step i` is replay depth - 1 + i.
timed = {}
smoke = ""
try:
    for line in open(runlog):
        m = re.match(r"time step (\d+)( warm)? ms=([0-9.]+)", line)
        if m:
            timed[depth - 1 + int(m.group(1))] = float(m.group(3))
        if line.startswith("SMOKE"):
            smoke = line.rstrip()
        elif line.startswith(("plan ", "load ", "capture ", "fed ")):
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
    print(f"  {r:6d} {r:5d} {ts:>10s} {x['period']:9.1f} {x['wall']:9.1f} {x['ksum']:9.1f} {x['gap']:8.1f} "
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

if [ "${1:-}" = --analyze ]; then
  [ $# -ge 4 ] || { echo "usage: nsys-ds41.sh --analyze <sqlite> <depth> <n> [<run log>]" >&2; exit 64; }
  analyze "$2" "$3" "$4" "${5:-/dev/null}" "$LAST" "$TOP"
  exit $?
fi

[ "$MODEL_NAME" = deepseek41 ] || {
  echo "nsys-ds41.sh: the profile is $MODEL_NAME — pick deepseek41 on the Mac side (BLOOMERY_MODEL=deepseek41)" >&2
  exit 64
}
DEPTHS=("$@")
[ ${#DEPTHS[@]} -gt 0 ] || DEPTHS=(6)
for d in "${DEPTHS[@]}"; do
  case $d in
    '' | *[!0-9]*) echo "nsys-ds41.sh: depth '$d' is not a number" >&2; exit 64 ;;
  esac
  [ "$d" -ge 1 ] || { echo "nsys-ds41.sh: depth $d: the run feeds at least one id" >&2; exit 64; }
done
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

assert_fresh_binary "$BIN" || exit $?
[ -x "$NSYS" ] || { echo "no nsys at $NSYS" >&2; exit 2; }
mkdir -p "$OUTDIR"

WITNESS=(head-open indent card busiest model mem pgmajfault)

lease_take
echo "[config] nsys=$($NSYS --version) n=$NGEN depths='${DEPTHS[*]}' last=$LAST out=$OUTDIR bound=${BOUND}s"
echo "[config] timing_gpu=$TIMING_GPU other_gpu=$OTHER_GPU CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
witness pre
guard_other

rc_all=0
for d in "${DEPTHS[@]}"; do
  out="$OUTDIR/nsys-ds41-d${d}-n${NGEN}-$(date -u +%H%M%S)"
  echo
  echo "=== depth $d (n $NGEN, $((d + NGEN - 1)) replays expected) -> $out.nsys-rep"
  guard_other
  witness "pre d=$d"
  t0=$(date +%s)
  timeout --kill-after=10 "$BOUND" \
    "$NSYS" profile -t cuda --cuda-graph-trace=node --cuda-event-trace=false \
    --sample=none --cpuctxsw=none \
    -o "$out" --force-overwrite true \
    "$BIN" --depth "$d" -n "$NGEN" --mode graph --time \
    > "$out.txt" 2>&1
  rc=$?
  t1=$(date +%s)
  witness "post d=$d"
  echo "[rc] $rc wall $((t1 - t0))s"
  [ $rc -eq 0 ] || { rc_all=$rc; echo "--- last 20 lines of $out.txt"; tail -n 20 "$out.txt"; continue; }
  "$NSYS" export --type sqlite --force-overwrite true -o "$out.sqlite" "$out.nsys-rep" > "$out.export.txt" 2>&1 \
    || { rc_all=1; echo "[export] failed"; tail -n 10 "$out.export.txt"; continue; }
  analyze "$out.sqlite" "$d" "$NGEN" "$out.txt" "$LAST" "$TOP" || rc_all=$?
  echo "--- files: $out.nsys-rep, $out.sqlite, $out.txt"
done

witness post
echo "[lease] released at $(now)"
exit $rc_all
