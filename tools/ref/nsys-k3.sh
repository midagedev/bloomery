#!/usr/bin/env bash
# K3.1 bp decode chain timeline: where the stage card's time goes between the host's signal and the next layer's go,
# asked of Nsight Systems (run on the box, both cards). Not a timed run: no lease, no card pick; like
# tools/ref/k31-step-stats.sh it takes /root/k31-probe.lock (one K3.1 process on the box at a time).
#
#   bash tools/ref/nsys-k3.sh [OUT]                         profile, export, tables (OUT default /root/k31-nsys)
#   bash tools/ref/nsys-k3.sh --analyze OUT/k3.sqlite [W [REPLAYS]]   tables again from a saved export, no profile
#                                                           (REPLAYS: the run's D + N - 1 step calls)
#   just nsys-gpu-kimik3 [OUT]
#
# The profiled command is k31-step-stats.sh's: generate_kimi_k3 --place bp --ctx 2048 --mode graph over the first D ids of
# corpus-prose.ids, -n N, BLOOMERY_HOST_ROOM=ROOM and BLOOMERY_STEP_STATS=1 (so the run's own `stat step` gap lines sit
# beside the trace). Env: K31_NS_D (64), K31_NS_N (24), K31_NS_ROOM (232G), K31_NS_MODEL (the K3.1 first shard),
# K31_NS_ENV (extra NAME=value words), K31_NS_W (steps the tables read, the last W; 20), NSYS (the toolkit's nsys).
#
# The cut. Per hosted layer the stage card's stream runs: front (mixer half, router, latent down, the handoff kernel
# `*handoff*`, the go batch) -> shadow (shared stack, the card's slots) -> wait (a stream memory operation on the host
# page and the tier's counter: the trace records nothing, so it is the longest idle gap between two handoffs) -> back
# (tier sum, adds, routed norm, latent up, close) -> the next layer's front. So, between handoff i and handoff i+1:
#   shadow = kernel time from handoff i's end to the wait gap
#   wait   = the longest idle gap (the host's leg as the card sees it, plus both signal latencies)
#   chain  = the wait's end to handoff i+1's end: what the host's `gap_*` counts after its signal; its kernels, busy time
#            and launch gaps (chain wall - busy; negative where a branch's kernels overlap the main stream's)
#   tier   = the tier card's (the device with no handoff kernel) last kernel or copy end before the wait's end; slack =
#            wait end - tier end (near zero: the stage waited for the tier card, not the host)
# An interval holding the once-per-step marker (the first stage kernel that runs D + N - 1 times, or one fewer, with the
# handoffs a whole number of steps) crosses a step and is left out. Only graph kernels are read (graphNodeId set).
set -uo pipefail
HERE=$(cd "${BASH_SOURCE[0]%/*}" && pwd)
# shellcheck source=tools/gate-bound.sh
source "$HERE/../gate-bound.sh"

analyze() {
  python3 - "$@" << 'PY'
import bisect, itertools, sqlite3, statistics, sys
from collections import Counter, defaultdict
db = sqlite3.connect(sys.argv[1]); W = int(sys.argv[2]); REPLAYS = int(sys.argv[3])
names = dict(db.execute("SELECT id, value FROM StringIds"))
K = [(s, e, d, names.get(n, str(n)).split("(")[0]) for s, e, d, n in db.execute(
    "SELECT start, end, deviceId, shortName FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE graphNodeId IS NOT NULL ORDER BY start")]
try:
    M = db.execute("SELECT start, end, deviceId, bytes, copyKind FROM CUPTI_ACTIVITY_KIND_MEMCPY ORDER BY start").fetchall()
except sqlite3.OperationalError:
    M = []
hcount = Counter(d for s, e, d, n in K if "handoff" in n)
if not hcount:
    print("NO TABLE: no graph kernel named *handoff*"); raise SystemExit(3)
stage = hcount.most_common(1)[0][0]
S = [k for k in K if k[2] == stage]
T = sorted([(s, e, n) for s, e, d, n in K if d != stage] + [(s, e, f"memcpy{kind}:{b}") for s, e, d, b, kind in M if d != stage])
Ts = [t[0] for t in T]
H = [i for i, k in enumerate(S) if "handoff" in k[3]]
cnt = Counter(k[3] for k in S)
hc = len(H)
print(f"[handoff] {dict(Counter(k[3] for k in S if 'handoff' in k[3]))}")
# The step marker: a stage kernel that runs once a replay. The run makes D + N - 1 step calls; the capture's own step
# may run no graph kernel, so a count one under is taken too, and the handoffs must be a whole number of steps.
cands = [n for n, c in cnt.items() if c in (REPLAYS, REPLAYS - 1) and hc % c == 0]
if not cands:
    near = sorted(cnt.items(), key=lambda x: abs(x[1] - REPLAYS))[:8]
    print(f"NO TABLE: no stage kernel runs {REPLAYS} (or one fewer) times with {hc} handoffs; nearest: {near}")
    raise SystemExit(3)
first = {n: next(i for i, k in enumerate(S) if k[3] == n) for n in cands}
marker = min(cands, key=first.get)
steps = cnt[marker]; per = hc // steps
print(f"[cut] stage device {stage}, {hc} handoffs, marker {marker} x {steps} -> {per} hosted layers a step; "
      f"tier device(s) {sorted({d for _, _, d, _ in K if d != stage})}, {len(T)} tier records")
Mk = [i for i, k in enumerate(S) if k[3] == marker]
lo_t = S[Mk[-W - 1]][1] if len(Mk) > W else S[0][0]   # the last W steps: after the (W+1)-th last marker
print(f"[window] the last {W} steps of {steps}: from t={lo_t}")
marks = set(Mk)
rows = []
for a, b in zip(H, H[1:]):
    if S[a][1] < lo_t:
        continue
    if any(a < m < b for m in marks):
        continue
    seq = S[a:b + 1]
    gaps = [(seq[j][0] - seq[j - 1][1], j) for j in range(1, len(seq))]
    g, j = max(gaps)
    ws, we = seq[j - 1][1], seq[j][0]
    shadow = sum(e - s for s, e, _, _ in seq[1:j])
    ch = seq[j:]
    busy = sum(e - s for s, e, _, _ in ch)
    wall = ch[-1][1] - ch[0][0]
    tend = None; tstart = None
    for s, e, n in itertools.islice(T, bisect.bisect_left(Ts, S[a][1]), None):
        if e <= we + 2000:
            tstart = s if tstart is None else tstart
            tend = e
        elif s > we:
            break
    rows.append(dict(iv=S[b][1] - S[a][1], shadow=shadow, wait=g, wall=wall, busy=busy, n=len(ch),
                     lg=wall - busy, ch=ch, tspan=(tend - tstart) if tend else None,
                     slack=(we - tend) if tend else None, ws=ws, we=we))
if not rows:
    print("NO TABLE: no layer interval in the window"); raise SystemExit(3)
def pct(xs, p):
    xs = sorted(xs); return xs[min(len(xs) - 1, int(p / 100 * len(xs)))]
def line(label, xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        print(f"  {label:36s} -"); return
    print(f"  {label:36s} p10 {pct(xs,10)/1e3:8.1f}  p50 {pct(xs,50)/1e3:8.1f}  p90 {pct(xs,90)/1e3:8.1f}  mean {statistics.mean(xs)/1e3:8.1f} us")
print(f"[layers] {len(rows)} intervals (handoff to handoff, one step's boundary left out)")
line("interval handoff->handoff", [r["iv"] for r in rows])
line("shadow kernel busy (under the host)", [r["shadow"] for r in rows])
line("wait gap (host leg + signal latency)", [r["wait"] for r in rows])
line("chain wall (wait end -> next handoff)", [r["wall"] for r in rows])
line("chain kernel busy", [r["busy"] for r in rows])
line("chain launch gaps (wall - busy)", [r["lg"] for r in rows])
line("tier card span (go -> last record)", [r["tspan"] for r in rows])
line("tier slack (wait end - tier end)", [r["slack"] for r in rows])
ns = Counter(r["n"] for r in rows)
print(f"  chain kernels a layer: {dict(sorted(ns.items()))}")
tight = sum(1 for r in rows if r["slack"] is not None and r["slack"] < 20000)
print(f"  layers whose wait ended < 20 us after the tier card's last record: {tight} of {len(rows)}")
per_name = defaultdict(lambda: [0, 0]); gap_after = defaultdict(int)
for r in rows:
    ch = r["ch"]
    for i, (s, e, _, n) in enumerate(ch):
        per_name[n][0] += 1; per_name[n][1] += e - s
        if i > 0:
            gap_after[n] += s - ch[i - 1][1]
L = len(rows)
print(f"[chain kernels] per layer, mean over {L} intervals: launches, busy us, idle before it us")
for n, (c, t) in sorted(per_name.items(), key=lambda x: -x[1][1] - gap_after[x[0]]):
    print(f"  {c / L:5.2f}  {t / L / 1e3:8.1f}  {gap_after[n] / L / 1e3:8.1f}  {n}")
med = sorted(rows, key=lambda r: r["wall"])[len(rows) // 2]
print(f"[one layer] the median chain ({med['n']} kernels, wall {med['wall']/1e3:.1f} us, busy {med['busy']/1e3:.1f} us): "
      f"offset from the wait's end, duration, idle before, name")
t0 = med["we"]; prev = med["ws"]
for s, e, _, n in med["ch"]:
    print(f"  {(s - t0)/1e3:8.1f} {(e - s)/1e3:8.1f} {(s - prev)/1e3:8.1f}  {n}")
    prev = e
PY
}

if [ "${1:-}" = "--analyze" ]; then
  [ -f "${2:-}" ] || { echo "nsys-k3: --analyze needs the sqlite" >&2; exit 64; }
  analyze "$2" "${3:-${K31_NS_W:-20}}" "${4:-$(( ${K31_NS_D:-64} + ${K31_NS_N:-24} - 1 ))}"; exit $?
fi

OUT=${1:-/root/k31-nsys}
D=${K31_NS_D:-64}; N=${K31_NS_N:-24}; ROOM=${K31_NS_ROOM:-232G}; W=${K31_NS_W:-20}
MODEL=${K31_NS_MODEL:-/models/Kimi-K3.1-Pro-IQ1_S/Kimi-K3.1-Pro-IQ1_S-00001-of-00031.gguf}
IDS=${BLOOMERY_DATA:-/root/bloomery-data}/kimi-k3/corpus-prose.ids
BIN=target/release/generate_kimi_k3
NSYS=${NSYS:-${CUDA_TOOLKIT_PATH:-/usr/local/cuda}/bin/nsys}
die() { echo "nsys-k3: $1" >&2; exit "${2:-1}"; }
[ -x "$BIN" ] || die "no $BIN (build it first)" 66
[ -x "$NSYS" ] || die "no nsys at $NSYS" 66
[ -f "$MODEL" ] || die "no model at $MODEL" 66
[ -f "$IDS" ] || die "no ids at $IDS" 66
[ "$N" -gt "$W" ] || die "K31_NS_N ($N) must pass K31_NS_W ($W)" 64
mkdir -p "$OUT"
exec 9>/root/k31-probe.lock; flock -w 7200 9 || die "k31-probe.lock held for 2 h" 75
TOKENS=$(head -n "$D" "$IDS" | paste -sd, -)
[ "$(echo "$TOKENS" | tr ',' '\n' | wc -l)" = "$D" ] || die "ids file shorter than $D"
{ sha256sum "$BIN"; "$NSYS" --version; echo "start=$(date -Iseconds) D=$D N=$N room=$ROOM W=$W"; } | tee "$OUT/summary.txt"
t0=$(date +%s)
bounded 1800 env BLOOMERY_REF_MODEL="$MODEL" BLOOMERY_HOST_ROOM="$ROOM" BLOOMERY_STEP_STATS=1 ${K31_NS_ENV:-} \
  "$NSYS" profile -t cuda --cuda-graph-trace=node --cuda-event-trace=false \
  --sample=none --cpuctxsw=none -o "$OUT/k3" --force-overwrite true \
  "$BIN" --place bp --ctx 2048 --mode graph --tokens "$TOKENS" -n "$N" > "$OUT/out.txt" 2> "$OUT/err.txt"
rc=$?
echo "rc=$rc wall_s=$(( $(date +%s) - t0 ))" | tee -a "$OUT/summary.txt"
grep '^load \|^stat summary\|^nvtier ' "$OUT/out.txt" "$OUT/err.txt" | tee -a "$OUT/summary.txt"
[ "$rc" = 0 ] || exit "$rc"
bounded 1800 "$NSYS" export --type sqlite --force-overwrite true -o "$OUT/k3.sqlite" "$OUT/k3.nsys-rep" > /dev/null || die "export failed"
bounded 1800 "$NSYS" stats -q -r cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum -f csv -o "$OUT/stats" "$OUT/k3.sqlite" > /dev/null 2>&1
analyze "$OUT/k3.sqlite" "$W" "$(( D + N - 1 ))" | tee "$OUT/chain.txt"
