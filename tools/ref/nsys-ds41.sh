#!/usr/bin/env bash
# V4.1 kernel timeline: where one decode step's time goes, kernel time against the host tier's
# joins, asked of Nsight Systems on the timing card (run on the box, under the lease).
#
#   BLOOMERY_MODEL=deepseek41 tools/box.sh 'bash tools/ref/nsys-ds41.sh 6 1024'
#   just nsys-gpu-ds41 6 1024
#   BLOOMERY_BOX_ENV='BLOOMERY_RESIDENCY=mid-p40-s1' just nsys-gpu-ds41 prose:64
#   bash tools/ref/nsys-ds41.sh --analyze <out>.sqlite <depth, P or arm> <n> [<out>.txt]
#       tables again, no profile; the depth argument is the number the analysis reads, and a
#       corpus arm (prose:<P>, code:<P>, the run form's spelling) is accepted as its P
#   bash tools/ref/nsys-ds41.sh --self-test
#       the decode-form analysis held to synthetic traces (no box, no nsys: the fixtures are
#       ds41copy.py's Synth, and every case runs this script's own --analyze)
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
# Arms (the decode form's arguments): a numeric depth feeds lease.sh's lcg_prompt; `prose:<P>` /
# `code:<P>` feed the first P ids of $BLOOMERY_DATA/engram/corpus-<name>.ids (one id a line) as
# `--tokens`, depth-ds41.sh's corpus arms, whose routing is the corpus's. Under the batch feed a
# corpus prompt of P ids behaves exactly as depth P — the replays are N − 1, the boundary count
# and the run log's `time prompt` row are checked the same — so the analysis needs no case of its
# own; a P the file cannot supply (P < 1, or past its line count) is refused before the lease.
#
# Placement: BLOOMERY_GEN_PLACE (a, the default, gate or bp) is the --place every profiled
# generate_ds41 loads by, named in the [config] line. Plan (a) loads on the card named A6000 and
# the gate plan on the one named 3090 (workstation::plan_a, plan_gate), so with the 3090 as the
# one timing card a is refused (64) and with the A6000 gate is. bp is plan (b′), both cards: it
# runs only in the two-card mode (BLOOMERY_TIMING_CARDS=a6000+3090, timing-card.sh; the profile's
# two-card line says what loads where), is refused (64) outside it, and inside it a and gate are
# (bp is the placement that sees both cards). The two-card precheck runs before the lease and the
# per-run check after each profile (an Xid, a card lost or off its cap, a load that named one
# card: that run's tables are refused, rc 1).
#
# Levers: whatever the caller exports through BLOOMERY_BOX_ENV reaches the profiled binary
# through the environment. The runner runs `generate_ds41 --levers` once before the lease and
# echoes the rows it reports as set — the binary's own reading of its own registry — and its
# refusal of a name no row names or a value a kind does not take is the runner's refusal, before
# the lease is waited for. A dry run prints BLOOMERY_RESIDENCY's value in its first line.
#
# The copy stream (BLOOMERY_RESIDENCY on, decode form): beside the per-replay tables the
# analysis tables what the swap's copy stream did — per step the H2D chunks (count, bytes, busy
# µs, and that busy split by what the engine stream was doing under it: its host-wait gaps, the
# waits in front of each `ds41_ffn_post*`; its kernels; neither), the D2D copies, the unpack
# kernels by name (`ds41_r8_q3k`), and the means per step mod 4 (the rule plans its flips at
# every fourth boundary). The copy stream is identified as a stream that is not the one the
# graph replays run on (the streamId of kernels with a graphNodeId, which only a replay's
# kernels carry); the cut is tools/ref/ds41copy.py's, and its self-test holds it to a synthetic
# trace. With the lever off the analysis prints the one line that says so instead of the tables.
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
# same, or no table. Under BLOOMERY_RESIDENCY the engine streams also run eager kernels between the
# replays (the boundary work): the batch is the eager engine-stream kernels that end before the
# first replay's first kernel, and the rest are named as boundary work, not a refusal.
# The batch feed is what depth-ds41.sh times and leaves the caches as the steps
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
# <out>.meta beside the report (the binary's sha256, P, N, the command): the ncu ds41pp
# form reads it and the sqlite to derive its launch skip. `--analyze <sqlite> <P> <n> <run log> [--plan FILE]`
# under BLOOMERY_NSYS_FORM=prefill re-prints the tables (the run log is required: the cut reads it); --plan
# FILE is the call's plan for a run log from before the engine printed it (`generate_ds41 --plan` with the
# run's arguments, or tools/flow/plans/), handed to ds41pp.py.
#
# Environment: BLOOMERY_NSYS_N (N, default 8, 2 in the prefill form), BLOOMERY_NSYS_LAST (replays
# tabled, default 8), BLOOMERY_NSYS_TOP (kernel rows, default 24), BLOOMERY_NSYS_OUT (default
# $BLOOMERY_DATA/nsys), BLOOMERY_GEN_BIN (default target/release/generate_ds41), BLOOMERY_GEN_PLACE
# and BLOOMERY_TIMING_CARDS (Placement above), BLOOMERY_ARM_BOUND (seconds one profile may run,
# default 900), BLOOMERY_DRY=1 (the command lines, then exit before the binary check and the
# lease; the decode recipe builds nothing under it either, like the prefill one).
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
CP=${BASH_SOURCE[0]%/*}/ds41copy.py
PLACE=${BLOOMERY_GEN_PLACE:-a}
case $PLACE in
  a | gate | bp) ;;
  *) echo "nsys-ds41.sh: BLOOMERY_GEN_PLACE is a (plan (a), on the A6000; the default), gate (the gate plan, on the 3090) or bp (plan (b′), on both cards), got '$PLACE'" >&2; exit 64 ;;
esac
# The corpus arms (prose:<P>, code:<P>): corpus-<name>.ids under $BLOOMERY_DATA/engram, one id a
# line; each file's id count is read once, into CORPUS_N_<name>, when an arm names it.
CORPORA="prose code"
corpus_file() { echo "${BLOOMERY_DATA:-}/engram/corpus-$1.ids"; }
# corpus_check <arm> <name> <P>: the file's id count read once; P outside 1..count is refused.
corpus_check() {
  local file var
  file=$(corpus_file "$2") var=CORPUS_N_$2
  if [ -z "${!var:-}" ]; then
    [ -r "$file" ] || { echo "nsys-ds41.sh: arm '$1': no $2 prompt file at $file (BLOOMERY_DATA)" >&2; exit 2; }
    printf -v "$var" '%s' "$(($(wc -l < "$file")))"
  fi
  if [ "$3" -lt 1 ] || [ "$3" -gt "${!var}" ]; then
    echo "nsys-ds41.sh: arm '$1': a $2 prompt of $3 ids; $file holds ${!var} (1..${!var})" >&2
    exit 64
  fi
}

# The analysis: sqlite, depth, n, the run's own output (for SMOKE and `time step`), last, top,
# and the copy-stream analyzer's path (ds41copy.py, its own last argument).
# python3 bounded by the arm bound where timeout exists (the box, every profile); a Mac has no
# timeout, and there --analyze and --self-test read a saved sqlite with no bound — the box's
# contract is unchanged.
py_bounded() {
  if command -v timeout >/dev/null 2>&1; then
    timeout --kill-after=10 "${BLOOMERY_ARM_BOUND:-900}" python3 "$@"
  else
    python3 "$@"
  fi
}
analyze() {
  py_bounded - "$@" << 'PY'
import os, sqlite3, statistics, sys
from collections import defaultdict
db = sqlite3.connect(sys.argv[1])
depth, ngen, last, top = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[5]), int(sys.argv[6])
runlog = sys.argv[4]
# ds41copy.py owns more than the copy-stream tables: the handoff kernel family the join cut
# matches below is its one definition, shared by both tools. Its module not loading refuses the
# whole analysis here, before any table: the join cut cannot run without it.
sys.path.insert(0, os.path.dirname(sys.argv[7]))
sys.path.insert(0, os.path.join(os.path.dirname(sys.argv[7]), "..", "bloomery"))
try:
    import ds41copy
except Exception as exc:
    print(f"    NO TABLE: the copy-stream analyzer ({sys.argv[7]}) did not load: {exc}")
    raise SystemExit(3)
import records
names = dict(db.execute("SELECT id, value FROM StringIds"))
rows = db.execute("SELECT start, end, shortName, streamId, deviceId, graphNodeId "
                  "FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start").fetchall()
# The batch feed's kernels run outside the graph on the engine streams — the (deviceId, streamId)
# pairs the replays' graph kernels run on; a streamId alone is per-context, and the two cards'
# contexts reuse ids. A kernel outside the graph on another stream (the copy stream's unpack,
# a draft's) is neither: under BLOOMERY_RESIDENCY the copy stream works inside the replays, and
# counting its kernels as the prompt's would refuse the trace and misread the feed.
eng = sorted({(r[4], r[3]) for r in rows if r[5] is not None})
eager = [r for r in rows if r[5] is None and (r[4], r[3]) in eng]
side = [r for r in rows if r[5] is None and (r[4], r[3]) not in eng]
ks = [(s, e, names.get(n, str(n)).split("(")[0]) for s, e, n, st, dv, g in rows if g is not None]
def rows_of(table):
    try:
        return db.execute(f"SELECT start, end, bytes FROM {table} ORDER BY start").fetchall()
    except sqlite3.OperationalError:
        return []
copies, sets = rows_of("CUPTI_ACTIVITY_KIND_MEMCPY"), rows_of("CUPTI_ACTIVITY_KIND_MEMSET")
# The run's own lines go through the record reader (tools/bloomery/records.py) — the record
# schema's one owner — not through a regex of this script's own: a row whose fields have drifted
# is still read there, by field name, and a run log that holds no `time prompt` row is named
# below instead of quietly skipping the feed check.
log_lines = None
try:
    with open(runlog) as f:
        log_lines = f.read().splitlines()
except OSError:
    pass
recs = records.read(log_lines) if log_lines is not None else []
runlog_given = bool(log_lines) and any(x.strip() for x in log_lines)
tp = records.first(recs, "time_prompt")
kind = tp["kind"] if tp is not None and "kind" in tp else None
# Under a residency the engine streams also run eager kernels between the replays (the boundary
# work), so the prompt batch is the eager engine-stream kernels that end before the first replay's
# first kernel; the rest are named on their own line below, not a refusal.
batch = [k for k in eager if not ks or k[1] <= ks[0][0]]
after = [k for k in eager if ks and k[1] > ks[0][0]]
feed = "batch" if batch else "steps"
side_note = f", {len(side)} on other streams (the copy stream's)" if side else ""
print(f"    kernels {len(rows)} ({len(batch)} outside the graph on the engine stream(s) {eng}: "
      f"the prompt's feed is {feed}{side_note})  memcpy {len(copies)}  memset {len(sets)}  "
      f"(stream memory operations leave no record)")
after_note = ""
if after:
    acnt = defaultdict(int)
    for k in after:
        acnt[names.get(k[2], str(k[2])).split("(")[0]] += 1
    after_note = " (" + ", ".join(f"{c} {n}" for n, c in sorted(acnt.items(), key=lambda x: -x[1])[:5]) + ")"
print(f"    eager engine-stream kernels after the first replay (boundary work under a residency): "
      f"{len(after)}{after_note}")
if runlog_given and kind is None:
    print("    NOTE: the run log carries no `time prompt` row the record reader reads: the check "
          "of the feed against the trace's kernels is skipped")
if kind is not None and kind != feed:
    print(f"    NO TABLE: the run log's time prompt row says kind={kind}, the trace's kernels say {feed}")
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
if batch:
    print(f"    the batch: {len(batch)} eager kernels, first to last {(batch[-1][1] - batch[0][0]) / 1e6:.1f} ms")

# The run's own lines: `time step i` is replay i - 1 under the batch feed; under the step feed
# token 0 comes out of replay depth - 1 and `time step i` is replay depth - 1 + i.
timed = {}
smoke = ""
for x in recs:
    if x.kind == "time_step":
        timed[fed - 1 + x["i"]] = x["ms"]
    elif x.kind == "smoke":
        smoke = x.line.rstrip()
    elif x.kind in ("plan", "load", "capture", "fed", "time_prompt"):
        print("    " + x.line.rstrip())
if smoke:
    print("    " + smoke)

def is_post(n):
    return n.startswith("ds41_ffn_post")

def replay(r):
    w = ks[bounds[r]:bounds[r + 1]]
    t0 = w[0][0]
    t_next = ks[bounds[r + 1]][0] if bounds[r + 1] < len(ks) else None
    # The last replay has no next one to bound its copies, so they are collected only to its
    # last kernel's end: the run's teardown after that (the readback, the runtime's own
    # transfers) belongs to no step.
    t_hi = t_next if t_next is not None else w[-1][1]
    wall = (w[-1][1] - t0) / 1e3
    ksum = sum(e - s for s, e, _ in w) / 1e3
    gaps = [(w[i][0] - w[i - 1][1]) / 1e3 for i in range(1, len(w))]
    joins, j_idx, jgaps = [], set(), []
    last_handoff = None
    for i, (s, e, n) in enumerate(w):
        if ds41copy.is_handoff(n):
            last_handoff = i
        elif is_post(n) and last_handoff is not None:
            h = last_handoff
            bridge = (s - w[h][1]) / 1e3
            over = sum(ee - ss for ss, ee, _ in w[h + 1:i]) / 1e3
            exposed = (s - w[i - 1][1]) / 1e3
            joins.append((bridge, over, exposed, w[i - 1][2]))
            j_idx.add(i)
            jgaps.append((w[i - 1][1], s))
            last_handoff = None
    other = [gaps[i - 1] for i in range(1, len(w)) if i not in j_idx]
    base = statistics.median(other) if other else 0.0
    if len(w) > 1:
        big_i = max(range(1, len(w)), key=lambda i: w[i][0] - w[i - 1][1])
        big = (gaps[big_i - 1], w[big_i - 1][2], w[big_i][2], big_i in j_idx)
    else:
        big = (0.0, w[0][2], w[0][2], False)  # a one-kernel replay has no gap to name
    c_in = [c for c in copies if t0 <= c[0] < t_hi]
    s_in = [c for c in sets if t0 <= c[0] < t_hi]
    return dict(w=w, wall=wall, ksum=ksum, gap=wall - ksum, joins=joins, base=base,
                period=(t_next - t0) / 1e3 if t_next else float("nan"),
                big=big,
                other_gap=sum(other), copies=c_in, sets=s_in, t0=t0, t1=t_next, jgaps=jgaps)

lo = max(0, replays - last)
R = {r: replay(r) for r in range(lo, replays)}
print()
print(f"=== per replay (last {replays - lo}; µs unless noted; `time step` in ms from the same run)")
print(f"  {'replay':>6s} {'depth':>5s} {'time step':>10s} {'period':>9s} {'wall':>9s} {'kern sum':>9s} {'gap':>8s} "
      f"{'joins':>5s} {'bridge':>8s} {'overlap':>8s} {'exposed':>8s} {'exp-base':>8s} {'base':>5s} {'other gap':>9s} "
      f"{'mem µs':>8s}  largest gap")
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
if not dec:
    print()
    if not runlog_given:
        print("=== no run log was given: the `time step` column and the mean tables need one "
              "(--analyze's <run log> argument)")
    elif not timed:
        print("=== the run log carries no `time step` rows the record reader reads: the mean "
              "tables are skipped")
    else:
        print(f"=== the {len(timed)} timed steps map to no replay among the {len(R)} tabled "
              f"(the depth argument does not match the run?): the mean tables are skipped")
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

# The copy stream beside the engine stream (the header's copy-stream paragraph): the windows
# and the host-wait gaps this cut found, the sqlite and the run log re-read by ds41copy.py.
print()
ds41copy.tables(sys.argv[1], runlog,
                [(r, R[r]["t0"], R[r]["t1"]) for r in R], {r: R[r]["jgaps"] for r in R}, top)
PY
}

# The prefill form's tables: sqlite, P, n, the run log, then ds41pp.py's --plan FILE when given.
analyze_prefill() {
  py_bounded "$PP" tables "$@" --layer "$LAYER" --blocked-us "$BLOCKED_US"
}

# The decode-form analysis held to synthetic traces, no box and no nsys: ds41copy.py's Synth
# writes the sqlite tables the analyzer queries, and every case runs this script's own --analyze
# and holds its exit code, its refusals and the numbers it prints.
self_test() {
  local t out rc n=0 bad=0 lastcp
  t=$(mktemp -d)
  trap 'rm -rf "$t"' RETURN
  python3 - "$t" "${BASH_SOURCE[0]%/*}/ds41copy.py" << 'FIX'
import os
import sys

t, cp = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.dirname(cp))
from ds41copy import CPY, ENG, MS, Synth  # its import also puts records' directory on the path
import records

RUNLOG = (
    "plan place=a card=A6000 ctx_max=4096 card_experts=200 (100 B) host_experts=88 (100 B) "
    "host_shadow=1 B 40 on 40 layers 20000000000\n"
    "load resident_bytes=1 shadow=host 0 unified_addressing=1 cards=[A6000] ctx=4096 layers=40 "
    "top_k=512 mode=graph place=a pin_main=on pinned=true prefill=batch ced=on group=1 in 1.0 s "
    "(runtime value)\n"
    "capture graph_nodes=784\n"
    "fed ids=6 first=[1,2,3,4] last=[5,6] depth_sequence_from=0\n"
    "time prompt n=6 ms=1.0000 tok/s=6.00 passes=1 kind=batch\n"
    "time step 1 ms=0.7000\n"
    "time step 2 ms=0.7000\n"
    "time step 3 ms=0.7000\n"
    "SMOKE mode=graph place=a prompt_tokens=6 depth=6 generated=3 warm=0 steps=3 p50_ms=0.7000 "
    "mean_ms=0.7000 tok/s(p50)=8.57\n")
# Every line of both logs must be a record the reader reads (the analysis echoes and checks
# them through it), or the fixture itself is wrong.
EVOLVED = RUNLOG.replace("time prompt n=6 ms=", "time prompt n=6 warm=0 ms=").replace(
    "kind=batch\n", "kind=steps\n")
for text in (RUNLOG, EVOLVED):
    unread = [x for x in text.splitlines() if records.read([x]) == []]
    assert not unread, f"fixture lines no record reads: {unread}"
open(os.path.join(t, "run.txt"), "w").write(RUNLOG)
open(os.path.join(t, "evolved.txt"), "w").write(EVOLVED)


def decode_synth(handoff, bare_last=False):
    """Three replays after one eager batch kernel (the batch feed); a D2H inside replay 0 and
    replay 2 each, and a teardown D2H after the last kernel, which no replay owns. `bare_last`
    leaves the last replay its marker kernel alone."""
    s = Synth()
    s.kern(0.90 * MS, 0.95 * MS, "ds41_glue_embed_q3k", ENG)
    for r in range(3):
        t0 = (1.0 + r) * MS
        if bare_last and r == 2:
            s.kern(t0 + 0.95 * MS, t0 + 0.98 * MS, "ds41_head_logits", ENG, graph=104)
            break
        s.kern(t0, t0 + 0.05 * MS, handoff, ENG, graph=101 + 10 * r)
        s.kern(t0 + 0.05 * MS, t0 + 0.15 * MS, "ds41_card_gate", ENG, graph=102 + 10 * r)
        s.kern(t0 + 0.60 * MS, t0 + 0.70 * MS, "ds41_ffn_post", ENG, graph=103 + 10 * r)
        s.kern(t0 + 0.95 * MS, t0 + 0.98 * MS, "ds41_head_logits", ENG, graph=104 + 10 * r)
    s.copy(1.20 * MS, 1.25 * MS, 2, 4096, CPY)
    s.copy(3.20 * MS, 3.25 * MS, 2, 4096, CPY)
    s.copy(5.00 * MS, 5.50 * MS, 2, 65536, CPY)
    return s


decode_synth("ds41_ffn_handoff").write(os.path.join(t, "plain.sqlite"))
decode_synth("ds41_ffn_handoff_tier").write(os.path.join(t, "tier.sqlite"))
decode_synth("ds41_ffn_handoff", bare_last=True).write(os.path.join(t, "one.sqlite"))
FIX
  case_() { # <name> <want_rc> <must contain> [<must not contain>]
    n=$((n + 1))
    if [ "$rc" = "$2" ] && printf '%s\n' "$out" | grep -qF -- "$3" \
       && { [ $# -lt 4 ] || ! printf '%s\n' "$out" | grep -qF -- "$4"; }; then
      echo "ok $1"
    else
      bad=$((bad + 1))
      echo "FAIL $1: rc $rc (want $2)"
      printf '%s\n' "$out" | sed 's/^/    got | /'
    fi
  }
  rc=0; out=$(bash "$0" --analyze "$t/plain.sqlite" 6 4 "$t/run.txt" 2>&1) || rc=$?
  case_ plain-handoff-joins 0 ": 1 joins per replay" ": 0 joins per replay"
  rc=0; out=$(bash "$0" --analyze "$t/tier.sqlite" 6 4 "$t/run.txt" 2>&1) || rc=$?
  case_ tier-handoff-joins 0 ": 1 joins per replay" ": 0 joins per replay"
  rc=0; out=$(bash "$0" --analyze "$t/tier.sqlite" 6 4 "$t/run.txt" 2>&1) || rc=$?
  lastcp=$(printf '%s\n' "$out" | awk '$1 == 2 && $2 == 8 {print $15}')
  n=$((n + 1))
  if [ "$rc" = 0 ] && [ "$lastcp" = "50.0" ]; then
    echo "ok teardown-not-billed"
  else
    bad=$((bad + 1))
    echo "FAIL teardown-not-billed: rc $rc, the last replay's mem µs is '${lastcp:-<no row>}' (want 50.0)"
  fi
  rc=0; out=$(bash "$0" --analyze "$t/plain.sqlite" 6 4 "$t/run.txt" 2>&1) || rc=$?
  case_ runlog-lines-echo 0 "plan place=a card=A6000"
  rc=0; out=$(bash "$0" --analyze "$t/one.sqlite" 6 4 "$t/run.txt" 2>&1) || rc=$?
  case_ one-kernel-replay 0 "0.0 (ds41_head_logits -> ds41_head_logits"
  rc=0; out=$(bash "$0" --analyze "$t/tier.sqlite" 6 4 "$t/evolved.txt" 2>&1) || rc=$?
  case_ drifted-feed-refused 3 "NO TABLE: the run log's time prompt row says kind=steps"
  rc=0; out=$(bash "$0" --analyze "$t/tier.sqlite" 6 4 2>&1) || rc=$?
  case_ no-log-named 0 "=== no run log was given" "=== mean of the"
  rc=0; out=$(bash "$0" --analyze "$t/tier.sqlite" prose:64 4 "$t/run.txt" 2>&1) || rc=$?
  case_ arm-accepted 0 "-> OK"
  rc=0; out=$(bash "$0" --analyze "$t/tier.sqlite" prose:x 4 2>&1) || rc=$?
  case_ arm-refused 64 "got 'prose:x'"
  echo "nsys-ds41.sh: self-test $([ "$bad" = 0 ] && echo ok || echo FAIL) ($n cases, $bad failed)"
  [ "$bad" = 0 ]
}

if [ "${1:-}" = --analyze ]; then
  if [ "$FORM" = prefill ]; then
    usage="usage: BLOOMERY_NSYS_FORM=prefill nsys-ds41.sh --analyze <sqlite> <P> <n> <run log> [--plan FILE]"
    case $# in
      5) PLAN=() ;;
      7) [ "$6" = --plan ] || { echo "$usage (got '$6' after the run log)" >&2; exit 64; }
         [ -f "$7" ] || { echo "nsys-ds41.sh: --plan $7 is not a file" >&2; exit 64; }
         PLAN=(--plan "$7") ;;
      *) echo "$usage" >&2; exit 64 ;;
    esac
    analyze_prefill "$2" "$3" "$4" "$5" "${PLAN[@]}"
    exit $?
  fi
  [ $# -ge 4 ] || { echo "usage: nsys-ds41.sh --analyze <sqlite> <depth, P or arm> <n> [<run log>]" >&2; exit 64; }
  # The depth argument is the number the analysis reads; a corpus arm (prose:<P>, code:<P>, the
  # run form's spelling of the same number) is accepted with its prefix stripped — anything else
  # non-numeric would die inside the analyzer's int() with a traceback instead of this refusal.
  A_DEPTH=$3
  case $A_DEPTH in
    prose:* | code:*) A_DEPTH=${A_DEPTH#*:} ;;
  esac
  case $A_DEPTH in
    '' | 0 | *[!0-9]*) echo "nsys-ds41.sh: --analyze's depth is a number >= 1 or a corpus arm prose:<P>/code:<P>, got '$3'" >&2; exit 64 ;;
  esac
  analyze "$2" "$A_DEPTH" "$4" "${5:-/dev/null}" "$LAST" "$TOP" "$CP"
  exit $?
fi

if [ "${1:-}" = --self-test ]; then
  self_test
  exit $?
fi

[ "$MODEL_NAME" = deepseek41 ] || {
  echo "nsys-ds41.sh: the profile is $MODEL_NAME — pick deepseek41 on the Mac side (BLOOMERY_MODEL=deepseek41)" >&2
  exit 64
}
DEPTHS=("$@")
# Per argument, by its index in DEPTHS: the depth or prompt length (A_DEP), the corpus its ids
# come from (A_CORP, empty for the lcg prompt a --depth feeds) and the label in output names.
A_DEP=() A_CORP=() A_NAME=()
if [ "$FORM" = prefill ]; then
  [ ${#DEPTHS[@]} -gt 0 ] || DEPTHS=(512)
  for d in "${DEPTHS[@]}"; do
    case $d in
      '' | *[!0-9]* | [0-8]) echo "nsys-ds41.sh: prompt length '$d' is an integer >= 9 in the prefill form" >&2; exit 64 ;;
    esac
    A_DEP+=("$d") A_CORP+=('') A_NAME+=("d$d")
  done
  case $NGEN in
    '' | *[!0-9]* | [01]) echo "nsys-ds41.sh: BLOOMERY_NSYS_N is at least 2 in the prefill form (a replay closes the window), got '$NGEN'" >&2; exit 64 ;;
  esac
else
  [ ${#DEPTHS[@]} -gt 0 ] || DEPTHS=(6)
  for d in "${DEPTHS[@]}"; do
    case $d in
      prose:* | code:*)
        c=${d%%:*} p=${d#*:}
        case $p in '' | *[!0-9]*) echo "nsys-ds41.sh: arm '$d' is <name>:<P>, P the count of ids the corpus file's head feeds" >&2; exit 64 ;; esac
        corpus_check "$d" "$c" "$p"
        A_DEP+=("$p") A_CORP+=("$c") A_NAME+=("$c$p")
        ;;
      '' | *[!0-9]*)
        echo "nsys-ds41.sh: depth '$d' is not a number (the corpus arms are prose:<P>, code:<P>)" >&2; exit 64 ;;
      *)
        [ "$d" -ge 1 ] || { echo "nsys-ds41.sh: depth $d: the run feeds at least one id" >&2; exit 64; }
        A_DEP+=("$d") A_CORP+=('') A_NAME+=("d$d")
        ;;
    esac
  done
fi
BIN=${BLOOMERY_GEN_BIN:-target/release/generate_ds41}
NSYS=${NSYS:-/usr/local/cuda/bin/nsys}
OUTDIR=${BLOOMERY_NSYS_OUT:-$BLOOMERY_DATA/nsys}
BOUND=${BLOOMERY_ARM_BOUND:-900}
# The card pin, the card's witness lines, the other-card guard and the binary's freshness; this
# runner has the two-card mode (the header's Placement).
# shellcheck disable=SC2034 # read by timing-card.sh when it is sourced next
TIMING_CARDS_RUNNER=1
# shellcheck source=tools/ref/timing-card.sh
source "${BASH_SOURCE[0]%/*}/timing-card.sh"
timing_cards_mode || exit $?
# In the two-card mode the profiled binary loads by bp alone; with one card, plan (a) needs the
# A6000 as the timing card and the gate plan the 3090 (workstation::plan_a, plan_gate).
TC_ARMS=()
for i in "${!A_NAME[@]}"; do TC_ARMS+=("${DEPTHS[$i]}" ours ours); done
# shellcheck disable=SC2034 # read by timing_cards_arms (timing-card.sh)
TIMING_CARDS_PLACE=bp TIMING_CARDS_PLACE_RAN=$PLACE
timing_cards_arms "$BIN" "${TC_ARMS[@]}" || exit $?
if [ "$PLACE" = a ] && [ "$TIMING_GPU" = "$GPU_3090" ]; then
  echo "nsys-ds41.sh: BLOOMERY_GEN_PLACE=a is plan (a), which loads on the A6000, and the timing card is the 3090 (BLOOMERY_TIMING_GPU=$TIMING_GPU): generate_ds41 would refuse the run; set BLOOMERY_GEN_PLACE=gate" >&2
  exit 64
fi
if [ "$PLACE" = bp ] && [ -z "$TIMING_CARDS" ]; then
  echo "nsys-ds41.sh: BLOOMERY_GEN_PLACE=bp is plan (b′), which loads on both cards (the A6000 and its 3090 expert tier); it runs in the two-card mode, BLOOMERY_TIMING_CARDS=a6000+3090" >&2
  exit 64
fi
if [ "$PLACE" = gate ] && [ "$TIMING_GPU" != "$GPU_3090" ]; then
  echo "nsys-ds41.sh: BLOOMERY_GEN_PLACE=gate is the gate plan, which loads on the 3090, and the timing card is $TIMING_GPU, not the 3090 ($GPU_3090): name the 3090 in BLOOMERY_TIMING_GPU, or leave BLOOMERY_GEN_PLACE at a" >&2
  exit 64
fi
# The lease and the witness fields.
# shellcheck source=tools/ref/lease.sh
source "${BASH_SOURCE[0]%/*}/lease.sh"

# arm_feed <i>: the feed arguments of arm <i> into FEED — a depth's `--depth D`, a corpus arm's
# `--tokens` over the first P ids of its file (a dry run prints the phrase it would run).
arm_feed() {
  local c=${A_CORP[$1]}
  if [ -z "$c" ]; then
    FEED=(--depth "${A_DEP[$1]}")
  elif [ -n "$DRY" ]; then
    FEED=(--tokens "\$(head -n ${A_DEP[$1]} $(corpus_file "$c") | paste -sd, -)")
  else
    FEED=(--tokens "$(head -n "${A_DEP[$1]}" "$(corpus_file "$c")" | paste -sd, -)")
  fi
}

# The profiled command for arm <i>, into CMD; the report path is $out.
profile_cmd() {
  arm_feed "$1"
  CMD=(timeout --kill-after=10 "$BOUND"
       "$NSYS" profile -t cuda --cuda-graph-trace=node --cuda-event-trace=false
       --sample=none --cpuctxsw=none -o "$out" --force-overwrite true
       "$BIN" "${FEED[@]}" -n "$NGEN" --mode graph --time --place "$PLACE")
}

if [ -n "$DRY" ]; then
  echo "[dry] form=$FORM bin=$BIN n=$NGEN place=$PLACE residency=${BLOOMERY_RESIDENCY:-<unset>} args='${DEPTHS[*]}' out=$OUTDIR timing_gpu=$TIMING_GPU CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  if [ -n "$TIMING_CARDS" ]; then
    echo "[dry] two cards: $TIMING_CARDS_NAME, the profile's two-card line: $TWO_CARD_PLACEMENT"
    tc_rc=0
    timing_cards_precheck '[dry] ' || tc_rc=$?
    if [ "$tc_rc" = 0 ]; then
      echo "[dry] two-card precheck: ok"
    else
      echo "[dry] two-card precheck: refused (rc $tc_rc): $TWOCARD_WHY — a real run stops here, before the lease"
    fi
  fi
  for i in "${!A_NAME[@]}"; do
    out="$OUTDIR/<name>"
    profile_cmd "$i"
    echo "[dry] ${DEPTHS[$i]}: ${CMD[*]}"
  done
  exit 0
fi

assert_fresh_binary "$BIN" || exit $?
[ -x "$NSYS" ] || { echo "no nsys at $NSYS" >&2; exit 2; }
mkdir -p "$OUTDIR"

# The binary's own reading of the levers the caller exported (BLOOMERY_BOX_ENV reaches the
# profiled run through the environment): the rows it reports as set are the [levers] lines, and
# its refusal of a name no row names or a value a kind does not take stops the runner here,
# before the lease is waited for. The full table stays beside the reports.
LEVERS_TXT="$OUTDIR/nsys-ds41-levers.txt"
if ! "$BIN" --levers > "$LEVERS_TXT" 2>&1; then
  echo "[levers] generate_ds41 --levers refused the environment:" >&2
  sed 's/^/    /' "$LEVERS_TXT" >&2
  exit 64
fi

# shellcheck disable=SC2034 # read by lease.sh's witness()
WITNESS=(head-open indent card busiest model mem pgmajfault)

lease_take
timing_cards_start
echo "[config] form=$FORM nsys=$($NSYS --version) n=$NGEN place=$PLACE args='${DEPTHS[*]}' last=$LAST out=$OUTDIR bound=${BOUND}s"
[ -z "$TIMING_CARDS" ] || echo "[config] two cards: $TIMING_CARDS_NAME, the profile's two-card line: $TWO_CARD_PLACEMENT"
echo "[config] timing_gpu=$TIMING_GPU other_gpu=$OTHER_GPU CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
LEVER_SET=$(awk '$3 == "set"' "$LEVERS_TXT")
if [ -n "$LEVER_SET" ]; then
  sed 's/^/[levers] /' <<< "$LEVER_SET"
else
  echo "[levers] none set beyond the defaults (the binary's full table: $LEVERS_TXT)"
fi
for c in $CORPORA; do
  var=CORPUS_N_$c
  [ -z "${!var:-}" ] || echo "[config] $c: the first P ids of $(corpus_file "$c") (${!var} ids)"
done
witness pre
guard_other

rc_all=0
for i in "${!A_NAME[@]}"; do
  d=${A_DEP[$i]}
  if [ "$FORM" = prefill ]; then
    out="$OUTDIR/nsys-ds41-pp${d}-n${NGEN}-$(date -u +%H%M%S)"
    echo
    echo "=== prompt $d (n $NGEN: the prompt's batches, then $((NGEN - 1)) replay(s)) -> $out.nsys-rep"
  elif [ -n "${A_CORP[$i]}" ]; then
    out="$OUTDIR/nsys-ds41-${A_CORP[$i]}${d}-n${NGEN}-$(date -u +%H%M%S)"
    echo
    echo "=== ${A_CORP[$i]} $d (n $NGEN: the corpus's first $d ids as a batch, then $((NGEN - 1)) replay(s)) -> $out.nsys-rep"
  else
    out="$OUTDIR/nsys-ds41-d${d}-n${NGEN}-$(date -u +%H%M%S)"
    echo
    echo "=== depth $d (n $NGEN: $((NGEN - 1)) replays after a batch feed, $((d + NGEN - 1)) after a step feed) -> $out.nsys-rep"
  fi
  profile_cmd "$i"
  guard_other
  witness "pre ${A_NAME[$i]}"
  t0=$(date +%s)
  "${CMD[@]}" > "$out.txt" 2>&1
  rc=$?
  t1=$(date +%s)
  witness "post ${A_NAME[$i]}"
  echo "[rc] $rc wall $((t1 - t0))s"
  [ $rc -eq 0 ] || { rc_all=$rc; echo "--- last 20 lines of $out.txt"; tail -n 20 "$out.txt"; continue; }
  # Two cards: an Xid, a card lost or off its cap, or a load that named one card refuses this
  # run's tables (timing_cards_arm; nothing to check with one card).
  if ! timing_cards_arm "$(cat "$out.txt")" ours; then
    rc_all=1
    echo "[two-cards] run refused: $TWOCARD_WHY"
    continue
  fi
  if [ "$FORM" = prefill ]; then
    {
      echo "bin=$BIN_PATH"
      echo "sha256=$BIN_SHA"
      echo "P=$d"
      echo "n=$NGEN"
      echo "place=$PLACE"
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
    analyze "$out.sqlite" "$d" "$NGEN" "$out.txt" "$LAST" "$TOP" "$CP" || rc_all=$?
    echo "--- files: $out.nsys-rep, $out.sqlite, $out.txt"
  fi
done

witness post
echo "[lease] released at $(now)"
exit $rc_all
