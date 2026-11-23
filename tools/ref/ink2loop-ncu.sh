#!/usr/bin/env bash
# One Nsight Compute pass over Inkling 2 Mini's loop_control_down launches on bloomery-serve (round ink2loop; a probe beside
# tools/ref/ink2loop-nsys.sh, whose header has the run form). ncu serializes and clock-locks the kernel: the numbers are stall
# shares, issue rates and per-instruction samples, not microseconds of the step.
#
#   BLOOMERY_MODEL=inkling2 tools/box.sh 'tools/ref/lease-hold.sh --card docs/cards/<slug>.card -- \
#       bash tools/ref/ink2loop-ncu.sh ~/repo/<tree>/target/release/bloomery-serve'
#
# Env: KERNEL (loop_control_down), SKIP (launches skipped, 120: past the prompt pass's, into the decode graph), COUNT (2),
# D (512), N (16), PORT (18191), BOUND (900), OUT.
set -uo pipefail
HERE=$(cd "${BASH_SOURCE[0]%/*}" && pwd)
[ $# -eq 1 ] || { echo "usage: ink2loop-ncu.sh <bloomery-serve binary>" >&2; exit 64; }
BIN=$1
[ -x "$BIN" ] || { echo "ink2loop-ncu.sh: no binary at $BIN" >&2; exit 2; }
# shellcheck source=tools/ref/ref-paths.sh
source "$HERE/ref-paths.sh"
# The server and the request are bounded children of the owner in tools/gate-bound.sh (bounded_bg, bounded).
# shellcheck source=tools/gate-bound.sh
source "$HERE/../gate-bound.sh" || exit 70
[ "${MODEL_NAME:-}" = inkling2 ] || { echo "ink2loop-ncu.sh: BLOOMERY_MODEL=inkling2" >&2; exit 64; }
KERNEL=${KERNEL:-loop_control_down} SKIP=${SKIP:-120} COUNT=${COUNT:-2} D=${D:-512} N=${N:-16} PORT=${PORT:-18191} BOUND=${BOUND:-900}
OUT=${OUT:-$BLOOMERY_DATA/ncu}
NCU=${NCU:-${CUDA_TOOLKIT_PATH:-/usr/local/cuda}/bin/ncu}
[ -x "$NCU" ] || { echo "ink2loop-ncu.sh: no ncu at $NCU" >&2; exit 2; }
IDS=$BLOOMERY_DATA/$MODEL_NAME/corpus-prose.ids
mkdir -p "$OUT"
rep="$OUT/ncu-ink2loop-$KERNEL-$(date -u +%H%M%S)"
body=$(mktemp "${TMPDIR:-/tmp}/ink2loop-body.XXXXXX")
head -n "$D" "$IDS" | python3 -c '
import json, sys
ids = [int(t) for t in sys.stdin.read().split() if t.strip()]
print(json.dumps({"prompt": ids, "n_predict": int(sys.argv[1]), "temperature": 0, "top_k": 1, "ignore_eos": True,
                  "cache_prompt": False, "return_tokens": False, "stream": False}))' "$N" > "$body"
echo "[config] bin=$BIN kernel=$KERNEL skip=$SKIP count=$COUNT D=$D N=$N ncu=$($NCU --version | tail -1) out=$rep"
echo "[apps-before] $(nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader | tr '\n' ';')"
BG_OUT=$rep.server.log bounded_bg "$BOUND" env -u BLOOMERY_REF_MODEL "$NCU" --target-processes all --graph-profiling node \
  --kernel-name "regex:^${KERNEL}\$" --launch-skip "$SKIP" --launch-count "$COUNT" --set full --import-source yes \
  -o "$rep" --force-overwrite \
  "$BIN" -m "$MODEL" --host 127.0.0.1 --port "$PORT" || exit $?
pid=$BG_PID
echo "[pid] ncu-timeout $pid"
# server_end: SIGINT to the server so ncu writes its report, 120 s for it to end by itself, then the owner's stop for one
# that stays (TERM, then KILL); the server's status is in BG_RC.
server_end() {
  kill -INT "$pid" 2> /dev/null
  for _ in $(seq 1 120); do kill -0 "$pid" 2> /dev/null || break; sleep 1; done
  if kill -0 "$pid" 2> /dev/null; then bounded_bg_stop; else bounded_bg_wait; fi
}
up=
for _ in $(seq 1 "$BOUND"); do
  kill -0 "$pid" 2> /dev/null || break
  if [ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2> /dev/null)" = 200 ]; then up=1; break; fi
  sleep 1
done
[ -n "$up" ] || { echo "ink2loop-ncu.sh: the server did not answer /health" >&2; tail -n 20 "$rep.server.log" >&2; server_end; exit 1; }
echo "[health] up"
status=$(mktemp "${TMPDIR:-/tmp}/ink2loop-status.XXXXXX")
LEASE_BOUNDED_OUT=$status bounded 600 curl -sS -o /dev/null -w '%{http_code}' -H 'Content-Type: application/json' --data-binary @"$body" "http://127.0.0.1:$PORT/completion"
code=$(< "$status")
echo "[request] http $code"
server_end
echo "[rc] ncu $BG_RC"
rm -f "$body" "$status"
[ -f "$rep.ncu-rep" ] || { echo "ink2loop-ncu.sh: no $rep.ncu-rep; log tail:" >&2; tail -n 20 "$rep.server.log" >&2; exit 1; }
echo "[report] $rep.ncu-rep"
"$NCU" -i "$rep.ncu-rep" --page details 2>&1 | grep -a -E "^  loop_control|Grid Size|Registers Per Thread|Duration|Elapsed Cycles|SM Frequency|Achieved Occupancy|Theoretical Occupancy|Issue Slots Busy|Executed Ipc Active|Warp Cycles Per Issued|Warp Cycles Per Executed|Eligible Warps|Active Warps Per Scheduler|DRAM Throughput|L1/TEX Hit Rate|L2 Hit Rate|Memory Throughput|stall|Stall|Local|Shared Memory Configuration Size" | head -60
echo "--- sampled stall reasons by SASS instruction (the lines with the most samples)"
"$NCU" -i "$rep.ncu-rep" --page source --csv --print-source sass 2>&1 | python3 -I -c '
import csv, sys
rows = list(csv.reader(sys.stdin))
hdr = next((r for r in rows if "Address" in r or "Source" in r), None)
if hdr is None:
    print("no source page"); raise SystemExit
i_src = hdr.index("Source")
cols = [i for i, h in enumerate(hdr) if h.strip() in ("Warp Stall Sampling (All Samples)", "Warp Stall Sampling (Not-issued Samples)")]
data = [r for r in rows[rows.index(hdr) + 1:] if len(r) == len(hdr)]
if not cols:
    print("no sampling column in:", hdr[:12]); raise SystemExit
c = cols[0]
def num(x):
    try: return float(x.replace(",", ""))
    except ValueError: return 0.0
tot = sum(num(r[c]) for r in data) or 1.0
print(f"samples {tot:.0f} over {len(data)} instructions")
cum = 0.0
for k, r in enumerate(data):
    v = num(r[c])
    cum += v
    if v / tot >= 0.015:
        print(f"  {k:5d} {100 * v / tot:5.1f} %  cum {100 * cum / tot:5.1f} %  {r[i_src][:90]}")
'
