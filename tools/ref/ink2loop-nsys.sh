#!/usr/bin/env bash
# One nsys trace of Inkling 2 Mini's decode on bloomery-serve, and the loop controller's table (round ink2loop; a probe, not a
# timing runner: tools/ref/nsys-gpu.sh has no serve form). Run under the lease on the box:
#
#   BLOOMERY_BOX_ENV='BLOOMERY_MODEL=inkling2' tools/box.sh 'tools/ref/lease-hold.sh --card docs/cards/ink2loop.card -- \
#       bash tools/ref/ink2loop-nsys.sh ~/repo/<tree>/target/release/bloomery-serve'
#   bash tools/ref/ink2loop-nsys.sh --analyze <out>.sqlite          the table again from a saved export
#
# Env: D (prompt ids, 512), N (n_predict, 128), PORT (18190), BOUND (seconds for the whole profile, 600), OUT (output dir).
# The server runs at the serving defaults (-m <file> --host 127.0.0.1 --port P), as tools/ref/depth-serve.sh starts it, under
# `nsys profile -t cuda --cuda-graph-trace=node`; one /completion of the first D ids of corpus-prose.ids, temperature 0,
# ignore_eos, cache_prompt off; then SIGINT to the captured nsys pid, which ends the trace.
set -uo pipefail
HERE=$(cd "${BASH_SOURCE[0]%/*}" && pwd)

analyze() {
  python3 -I - "$1" "${2:-48}" << 'PY'
import sqlite3, sys
from collections import Counter, defaultdict
db = sqlite3.connect(sys.argv[1])
per_replay = int(sys.argv[2])  # down launches a replay: 24 message visits x (attention triple + feed-forward) pairs
names = dict(db.execute("SELECT id, value FROM StringIds"))
rows = db.execute("SELECT start, end, shortName, gridX, graphNodeId FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start").fetchall()
ks = [(s, e, names.get(n, str(n)).split("(")[0], gx, g is not None) for s, e, n, gx, g in rows]
cnt = Counter(k[2] for k in ks)
print(f"kernels {len(ks)}; graph nodes {sum(1 for k in ks if k[4])}")
for n in ("loop_control_down", "loop_control_out", "loop_control_out3", "loop_control_down_rows"):
    print(f"  {n}: {cnt.get(n, 0)} launches")
nd = cnt.get("loop_control_down", 0)
if nd == 0 or nd % per_replay:
    print(f"NO TABLE: loop_control_down count {nd} is not a whole number of replays of {per_replay}")
    raise SystemExit(3)
R = nd // per_replay
# A replay's window runs from its first loop_control_down to the next replay's first: the controller's 48 down launches a
# replay are the clock, so no marker kernel is needed (the window is one decode step's wall, host turnaround included).
dwn = [i for i, k in enumerate(ks) if k[2] == "loop_control_down"]
bounds = [dwn[r * per_replay] for r in range(R)] + [len(ks)]
marker = "loop_control_down (every 48th)"
print(f"replays {R}; windows cut at every {per_replay}th loop_control_down")
# Skip the first two replays (the capture's own and the warm one) and the last (it has no next boundary).
use = range(2, R - 1)
agg = defaultdict(lambda: [0.0, 0, set(), 1e18, 0.0])
pair_span, pair_gap_after, ctrl_sum, kern_sum, wall, period = [], [], [], [], [], []
for r in use:
    w = ks[bounds[r]:bounds[r + 1]]
    wall.append((w[-1][1] - w[0][0]) / 1e3)
    period.append((ks[bounds[r + 1]][0] - ks[bounds[r]][0]) / 1e3 if bounds[r + 1] < len(ks) else float("nan"))
    kern_sum.append(sum(e - s for s, e, _, _, _ in w) / 1e3)
    cs = 0.0
    i = 0
    while i < len(w):
        s, e, n, gx, _ = w[i]
        if n == "loop_control_down":
            j = i + 1
            while j < len(w) and not w[j][2].startswith("loop_control_out"):
                j += 1
            if j < len(w):
                pair_span.append((w[j][1] - s) / 1e3)
                pair_gap_after.append((w[j][0] - e) / 1e3)  # down end to out start
            i = j
            continue
        i += 1
    for s, e, n, gx, _ in w:
        if n.startswith("loop_control"):
            d = (e - s) / 1e3
            cs += d
            a = agg[f"{n}[{gx}]"]
            a[0] += d; a[1] += 1; a[2].add(gx); a[3] = min(a[3], d); a[4] = max(a[4], d)
    ctrl_sum.append(cs)
m = lambda xs: sum(xs) / len(xs)
print(f"windows used {len(list(use))} (replays 2 .. {R - 2})")
print(f"  {'kernel':24s} {'launches/replay':>15s} {'mean us':>9s} {'min':>7s} {'max':>7s}  grid")
for n, (s, c, g, lo, hi) in sorted(agg.items()):
    print(f"  {n:24s} {c / len(list(use)):15.1f} {s / c:9.2f} {lo:7.2f} {hi:7.2f}")
print(f"controller kernel sum a replay: {m(ctrl_sum):8.1f} us   (min {min(ctrl_sum):.1f}, max {max(ctrl_sum):.1f})")
print(f"all kernels' sum a replay:      {m(kern_sum):8.1f} us   first-to-last {m(wall):.1f} us   period {m(period):.1f} us")
print(f"controller share of the kernel sum {100 * m(ctrl_sum) / m(kern_sum):.1f} %, of the period {100 * m(ctrl_sum) / m(period):.1f} %")
print(f"pairs (down start to out end): {len(pair_span) / len(list(use)):.1f} a replay, mean span {m(pair_span):.2f} us, "
      f"sum a replay {sum(pair_span) / len(list(use)):.1f} us; down end to out start mean {m(pair_gap_after):.2f} us")
by = defaultdict(lambda: [0.0, 0])
for r in use:
    for s, e, n, gx, _ in ks[bounds[r]:bounds[r + 1]]:
        by[n][0] += (e - s) / 1e3; by[n][1] += 1
tot = sum(v[0] for v in by.values())
print("top kernels a replay (sum us, launches, mean us):")
for n, (s, c) in sorted(by.items(), key=lambda x: -x[1][0])[:14]:
    print(f"  {n[:40]:40s} {s / len(list(use)):9.1f} {c / len(list(use)):7.1f} {s / c:8.2f}")
PY
}

if [ "${1:-}" = --analyze ]; then
  [ $# -ge 2 ] || { echo "usage: ink2loop-nsys.sh --analyze <sqlite> [down launches a replay]" >&2; exit 64; }
  analyze "$2" "${3:-48}"
  exit $?
fi
[ $# -eq 1 ] || { echo "usage: ink2loop-nsys.sh <bloomery-serve binary> | --analyze <sqlite>" >&2; exit 64; }
BIN=$1
[ -x "$BIN" ] || { echo "ink2loop-nsys.sh: no binary at $BIN" >&2; exit 2; }
# shellcheck source=tools/ref/ref-paths.sh
source "$HERE/ref-paths.sh"
[ "${MODEL_NAME:-}" = inkling2 ] || { echo "ink2loop-nsys.sh: BLOOMERY_MODEL=inkling2 (the profile is '${MODEL_NAME:-}')" >&2; exit 64; }
D=${D:-512} N=${N:-128} PORT=${PORT:-18190} BOUND=${BOUND:-600}
OUT=${OUT:-$BLOOMERY_DATA/nsys}
NSYS=${NSYS:-${CUDA_TOOLKIT_PATH:-/usr/local/cuda}/bin/nsys}
[ -x "$NSYS" ] || { echo "ink2loop-nsys.sh: no nsys at $NSYS" >&2; exit 2; }
IDS=${IDS:-$BLOOMERY_DATA/$MODEL_NAME/corpus-prose.ids}
[ "$(grep -c . "$IDS")" -ge "$D" ] || { echo "ink2loop-nsys.sh: $IDS holds fewer than $D ids" >&2; exit 2; }
mkdir -p "$OUT"
stamp=$(date -u +%H%M%S)
rep="$OUT/nsys-ink2loop-d$D-n$N-$stamp"
log="$rep.server.log"
body=$(mktemp "${TMPDIR:-/tmp}/ink2loop-body.XXXXXX")
resp=$(mktemp "${TMPDIR:-/tmp}/ink2loop-resp.XXXXXX")
head -n "$D" "$IDS" | python3 -c '
import json, sys
ids = [int(t) for t in sys.stdin.read().split() if t.strip()]
print(json.dumps({"prompt": ids, "n_predict": int(sys.argv[1]), "temperature": 0, "top_k": 1, "ignore_eos": True,
                  "cache_prompt": False, "return_tokens": False, "stream": False}))' "$N" > "$body"
echo "[config] bin=$BIN model=$MODEL D=$D N=$N port=$PORT nsys=$($NSYS --version) out=$rep"
echo "[card] $(nvidia-smi --query-gpu=name,power.limit,clocks.sm,memory.used --format=csv,noheader)"
echo "[apps-before] $(nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader | tr '\n' ';')"
timeout --kill-after=10 "$BOUND" env -u BLOOMERY_REF_MODEL "$NSYS" profile -t cuda --cuda-graph-trace=node \
  --cuda-event-trace=false --sample=none --cpuctxsw=none -o "$rep" --force-overwrite true \
  "$BIN" -m "$MODEL" --host 127.0.0.1 --port "$PORT" > "$log" 2>&1 &
pid=$!
echo "[pid] nsys-timeout $pid"
up=
for _ in $(seq 1 "$BOUND"); do
  kill -0 "$pid" 2> /dev/null || break
  if [ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2> /dev/null)" = 200 ]; then up=1; break; fi
  sleep 1
done
if [ -z "$up" ]; then
  echo "ink2loop-nsys.sh: the server did not answer /health; log tail:" >&2
  tail -n 20 "$log" >&2
  kill -INT "$pid" 2> /dev/null
  wait "$pid"
  exit 1
fi
echo "[health] up; apps: $(nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader | tr '\n' ';')"
code=$(timeout --kill-after=10 300 curl -sS -o "$resp" -w '%{http_code}' -H 'Content-Type: application/json' \
  --data-binary @"$body" "http://127.0.0.1:$PORT/completion")
echo "[request] http $code"
python3 -I -c '
import json, sys
t = json.load(open(sys.argv[1]))["timings"]
print("[timings]", {k: t[k] for k in ("prompt_n", "prompt_ms", "prompt_per_second", "predicted_n", "predicted_ms", "predicted_per_second", "cache_n") if k in t})' "$resp"
kill -INT "$pid" 2> /dev/null
for _ in $(seq 1 120); do kill -0 "$pid" 2> /dev/null || break; sleep 1; done
if kill -0 "$pid" 2> /dev/null; then kill -TERM "$pid" 2> /dev/null; fi
wait "$pid"
echo "[rc] nsys $?"
rm -f "$body" "$resp"
[ -f "$rep.nsys-rep" ] || { echo "ink2loop-nsys.sh: no $rep.nsys-rep; log tail:" >&2; tail -n 20 "$log" >&2; exit 1; }
"$NSYS" export --type sqlite --force-overwrite true -o "$rep.sqlite" "$rep.nsys-rep" > "$rep.export.txt" 2>&1 || { tail -n 10 "$rep.export.txt"; exit 1; }
echo "[trace] $rep.nsys-rep $rep.sqlite"
analyze "$rep.sqlite" 48
