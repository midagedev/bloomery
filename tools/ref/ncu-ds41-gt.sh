#!/usr/bin/env bash
# The V4.1 prompt batch's card-expert tile kernels under Nsight Compute: DRAM bytes and duration of every
# ds41_expert_gate_up_tiles (GT) and q4k_gemv_tiles (DT) launch of one prose prompt call (run on the box,
# under the lease, lead-only; `just ncu-gpu-ds41-gt [P]`).
#
# ncu-gpu.sh cannot take it: its ds41pp form profiles the four attention projections from an nsys
# trace's plan, and its generic form feeds the LCG prompt, whose routing leaves the card experts under
# the host union. This one feeds generate_ds41 the first P (512) ids of
# $BLOOMERY_DATA/engram/corpus-prose.ids at --place a, -n 2, with the hot list BLOOMERY_HOT_LIST
# (required: without one the card holds no routed expert to tile), and collects three metrics of every
# GT and DT launch up to BLOOMERY_NCU_COUNT (96: two kernels over V4.1's 43 layers, and room) —
# dram__bytes_read.sum, gpu__time_duration.sum, launch__grid_size — in base units, one pass, at the
# card's own clock (--clock-control none: the durations stand beside the unprofiled card_in term).
# The summary: per kernel the launches, the largest grid, and over the launches whose grid is within
# 2 % of it (a layer's full block; the CED triangle shrinks the last layers' blocks, the decode steps'
# launches are smaller still) the median DRAM GB, the median ms and their launch count.
#
#   BLOOMERY_BOX_ENV='BLOOMERY_HOT_LIST=/root/bloomery-data/router/hotlist-384.txt' just ncu-gpu-ds41-gt 512
#   BLOOMERY_BOX_ENV='BLOOMERY_DRY=1 BLOOMERY_HOT_LIST=…' just ncu-gpu-ds41-gt 512  # the command, no lease
#   bash tools/ref/ncu-ds41-gt.sh --summary <out>.csv                            # the summary again
#
# Environment: BLOOMERY_HOT_LIST (required), BLOOMERY_NCU_COUNT, BLOOMERY_NCU_OUT (default
# $BLOOMERY_DATA/ncu), BLOOMERY_ARM_BOUND (seconds the profile may run, default 900), BLOOMERY_DRY.
set -uo pipefail
# shellcheck source=tools/ref/ref-paths.sh
source "${BASH_SOURCE[0]%/*}/ref-paths.sh"
KERNELS='^(ds41_expert_gate_up_tiles|q4k_gemv_tiles)$'
METRICS=dram__bytes_read.sum,gpu__time_duration.sum,launch__grid_size

summary() { # summary <csv>: the per-launch rows and each kernel's full-block medians; 3 on no rows
  python3 - "$1" << 'PY'
import csv, re, statistics, sys
SCALE = {"byte": 1, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9, "nsecond": 1, "usecond": 1e3,
         "msecond": 1e6, "second": 1e9, "": 1}
rows = list(csv.reader(open(sys.argv[1])))
hdr = next((i for i, r in enumerate(rows) if r and r[0] == "ID"), None)
if hdr is None:
    sys.exit(f"[summary] {sys.argv[1]} holds no ncu CSV header")
col = {n: i for i, n in enumerate(rows[hdr])}
need = ("ID", "Kernel Name", "Grid Size", "Metric Name", "Metric Unit", "Metric Value")
if any(n not in col for n in need):
    sys.exit(f"[summary] the CSV lacks one of {need}; header {rows[hdr]}")
per = {}
for r in rows[hdr + 1:]:
    if len(r) <= col["Metric Value"] or not r[col["Metric Name"]]:
        continue
    d = per.setdefault(int(r[col["ID"]]), dict(name=r[col["Kernel Name"]].split("(")[0], m={},
                       grid=tuple(int(x) for x in re.findall(r"\d+", r[col["Grid Size"]]))))
    unit = r[col["Metric Unit"]]
    if unit not in SCALE:
        sys.exit(f"[summary] metric {r[col['Metric Name']]} in unit '{unit}', which this summary does not scale")
    d["m"][r[col["Metric Name"]]] = float(r[col["Metric Value"]].replace(",", "")) * SCALE[unit]
if not per:
    print("[summary] no launch in the CSV")
    sys.exit(3)
print(f"{'id':>5s} {'kernel':28s} {'grid':>14s} {'blocks':>7s} {'dram_GB':>8s} {'ms':>8s}")
blocks = lambda g: g[0] * (g[1] if len(g) > 1 else 1) * (g[2] if len(g) > 2 else 1)
for i in sorted(per):
    d = per[i]
    print(f"{i:5d} {d['name'][:28]:28s} {'x'.join(map(str, d['grid'])):>14s} {blocks(d['grid']):7d} "
          f"{d['m'].get('dram__bytes_read.sum', float('nan')) / 1e9:8.3f} "
          f"{d['m'].get('gpu__time_duration.sum', float('nan')) / 1e6:8.3f}")
for name in sorted({d["name"] for d in per.values()}):
    ds = [d for d in per.values() if d["name"] == name]
    top = max(blocks(d["grid"]) for d in ds)
    full = [d for d in ds if blocks(d["grid"]) >= 0.98 * top]
    gb = statistics.median(d["m"]["dram__bytes_read.sum"] / 1e9 for d in full)
    ms = statistics.median(d["m"]["gpu__time_duration.sum"] / 1e6 for d in full)
    print(f"summary {name} launches={len(ds)} max_blocks={top} full_block_launches={len(full)} "
          f"dram_gb_median={gb:.3f} ms_median={ms:.3f}")
PY
}

if [ "${1:-}" = --summary ]; then
  [ $# = 2 ] || { echo "usage: ncu-ds41-gt.sh --summary <csv>" >&2; exit 64; }
  summary "$2"
  exit $?
fi

[ "$MODEL_NAME" = deepseek41 ] || { echo "ncu-ds41-gt.sh: the profile is '$MODEL_NAME' (BLOOMERY_MODEL=deepseek41)" >&2; exit 64; }
P=${1:-512}
COUNT=${BLOOMERY_NCU_COUNT:-96}
for v in "P:$P" "BLOOMERY_NCU_COUNT:$COUNT"; do
  case ${v#*:} in '' | *[!0-9]* | 0*) echo "ncu-ds41-gt.sh: ${v%%:*} is a positive integer, got '${v#*:}'" >&2; exit 64 ;; esac
done
HOT=${BLOOMERY_HOT_LIST:-}
[ -n "$HOT" ] || { echo "ncu-ds41-gt.sh: BLOOMERY_HOT_LIST is required (no hot list, no routed expert on the card to tile)" >&2; exit 64; }
GEN=target/release/generate_ds41
PROSE=${BLOOMERY_DATA:-}/engram/corpus-prose.ids
OUTDIR=${BLOOMERY_NCU_OUT:-$BLOOMERY_DATA/ncu}
# The ncu build ncu-gpu.sh pins (its header says why: the wrapper picks the newest installed).
NCU=${NCU:-/opt/nvidia/nsight-compute/2025.3.1/ncu}
DRY=${BLOOMERY_DRY:-}
# shellcheck source=tools/ref/timing-card.sh
source "${BASH_SOURCE[0]%/*}/timing-card.sh"
# shellcheck source=tools/ref/lease.sh
source "${BASH_SOURCE[0]%/*}/lease.sh"
[ "$TIMING_GPU" = "$GPU_A6000" ] || { echo "ncu-ds41-gt.sh: --place a loads on the A6000, the timing card is $TIMING_GPU" >&2; exit 64; }
export CUDA_VISIBLE_DEVICES=$TIMING_GPU

gt_cmd() { # gt_cmd <ids>: into CMD
  CMD=(timeout --kill-after=10 "$LEASE_ARM_BOUND" "$NCU" --target-processes application-only --clock-control none
       --graph-profiling node --launch-skip 0 --launch-count "$COUNT"
       --kernel-name "regex:$KERNELS" --kernel-name-base function
       --metrics "$METRICS" --print-units base --csv --log-file "$out.csv"
       "$GEN" --tokens "$1" -n 2 --place a)
}

if [ -n "$DRY" ]; then
  out="$OUTDIR/<name>"
  gt_cmd "<the first $P ids of corpus-prose.ids>"
  echo "[dry] P=$P count=$COUNT hot_list=$HOT timing_gpu=$TIMING_GPU CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  echo "[dry] ${CMD[*]}"
  exit 0
fi

[ -f "$HOT" ] || { echo "ncu-ds41-gt.sh: no hot list at $HOT" >&2; exit 66; }
[ -f "$PROSE" ] || { echo "ncu-ds41-gt.sh: no prose ids at $PROSE" >&2; exit 66; }
[ "$P" -le "$(wc -l < "$PROSE")" ] || { echo "ncu-ds41-gt.sh: $PROSE holds fewer than $P ids" >&2; exit 64; }
assert_fresh_binary "$GEN" || exit $?
[ -x "$NCU" ] || { echo "no ncu at $NCU" >&2; exit 2; }
[ "$(id -u)" = 0 ] || { echo "ncu 카운터는 root가 필요하다(RmProfilingAdminOnly=1)" >&2; exit 77; }
mkdir -p "$OUTDIR"
out="$OUTDIR/ncu-ds41-gt-p${P}-$(date -u +%H%M%S)"
gt_cmd "$(head -n "$P" "$PROSE" | paste -sd, -)"

# shellcheck disable=SC2034 # read by lease.sh's witness()
WITNESS=(head-open indent card model)
lease_take
echo "[config] ncu=$($NCU --version | sed -n 3p) P=$P count=$COUNT kernels='$KERNELS' metrics=$METRICS hot_list=$HOT clock-control=none out=$out"
witness pre
guard_other
t0=$(date +%s)
"${CMD[@]}" > "$out.txt" 2>&1
rc=$?
witness post
echo "[rc] $rc wall $(($(date +%s) - t0))s${OTHER_BUSY_TAG}"
[ $rc -eq 0 ] || { echo "--- last 20 lines"; tail -n 20 "$out.txt"; }
if [ -s "$out.csv" ]; then
  echo "--- summary. Raw: $out.csv, the run $out.txt"
  summary "$out.csv"
  src=$?
  [ $src -eq 0 ] || [ $rc -ne 0 ] || rc=$src
else
  echo "[summary] no CSV at $out.csv"
  [ $rc -ne 0 ] || rc=3
fi
echo "[lease] released at $(now)"
exit $rc
