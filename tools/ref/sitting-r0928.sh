#!/usr/bin/env bash
# Sitting r0928: the research probes of the 2026-09-28 idea audit, in one run from the Mac (lead-only).
# Each job is an existing recipe or runner with its prediction card (docs/cards/r0928-<slug>.card);
# this script owns only the order, the hold, the logs and the post-reads.
#
#   tools/ref/sitting-r0928.sh             # the sitting: the hold up, every job, the hold down
#   tools/ref/sitting-r0928.sh --dry-run   # each job's command and card check; nothing reaches the box
#   tools/ref/sitting-r0928.sh --self-test # the post-read parsers on fixed text; no box
#   tools/ref/sitting-r0928.sh --only a,b  # the named jobs only, in the sitting's order
#
# The hold: before the first job the script waits for a quiet box (a no-op tools/box.sh call, whose guard
# waits out the lease and every other hold), then puts /root/bloomery-r0928-hold up and exports
# BLOOMERY_HOLD_OWNER=r0928, so every job's box.sh passes this hold and no other track starts a build
# between two jobs. The hold comes down on every exit path (trap). A job that takes the lease does so
# inside its runner, card first (lease_take); the builds run before each lease, under the hold.
#
# Order (the page cache): the Qwen3 jobs, then the V4.1 jobs, the engram residency read right after them
# (its card asks for the cache the V4.1 prompts left), GLM, and last the two-card probes, whose 8 GiB
# host chunks would evict the model files ahead of the loads.
#
# Logs: every job's whole stdout and stderr go to <dir>/<job>.log before anything reads them (never
# through a pipe that could cut them); the post-reads read those files and write <dir>/<job>.post;
# <dir>/sitting.log is this script's own record: each job's start, end, rc and wall. <dir> is
# BLOOMERY_SITTING_DIR, default ./target/sitting-r0928-<UTC stamp> on the Mac.
#
# rc 75 (contention: the lease or a hold another owner put up, a busy A6000) retries the job up to
# twice; any other non-zero rc is recorded and the sitting goes on. The script's exit code is 1 when
# any job ended non-zero, else 0.
set -uo pipefail
ROOT=$(cd "${BASH_SOURCE[0]%/*}/../.." && pwd)
OWNER=r0928
HOLD=/root/bloomery-$OWNER-hold
HOT=/root/bloomery-data/router/hotlist-384.txt
# The held-out engram prompt: corpus-threads.ids from half its 4,515,000 lines (card r0928-engram-resident).
THREADS_OFFSET=2257500

# The jobs, in the sitting's order: name | card slug | predicted minutes | the command, run from ROOT with
# BLOOMERY_BOX_ENV set to the card plus the job's own entries (a `box=` prefix, `;`-separated, before `|`).
JOBS=(
  "q6k|r0928-q6k-ncu|3|mac=BLOOMERY_MODEL=qwen3moe|q6k_cmd"
  "kv512|r0928-q3kv-nsys512|2||just nsys-gpu-qwen3moe-prefill 512"
  "clock4096|r0928-q3clock-nsys4096|2||just nsys-gpu-qwen3moe-prefill 4096"
  "s|r0928-q3s-pp|3|mac=BLOOMERY_AB_ROUNDS=3|just depth-gpu-qwen3moe 1 2 3 4 5"
  "imma|r0928-imma-m128|1||just time-gpu-probe-imma"
  "join|r0928-join-nsys|4|box=BLOOMERY_HOT_LIST=$HOT;BLOOMERY_STEP_STATS=1;BLOOMERY_NSYS_N=8|just nsys-gpu-ds41 6"
  "enqueue|r0928-enqueue-g2|5|mac=BLOOMERY_AB_ROUNDS=1;box=BLOOMERY_HOT_LIST=$HOT;BLOOMERY_STEP_STATS=1|just depth-gpu-ds41 4096"
  "gt|r0928-gt-ncu|5|box=BLOOMERY_HOT_LIST=$HOT|just ncu-gpu-ds41-gt 512"
  "engram|r0928-engram-resident|3||engram_cmd"
  "union|r0928-union-split|4||just time-cpu-v41-host --threads 32 --rounds 3 --seconds 24 --warmup 3 --arms unionr8:4x8u0.125,unionr8:4x16u0.0625"
  "glm|r0928-glm-nsys|15||just nsys-gpu-glm5next 512"
  "tred|r0928-tred|1|box=BLOOMERY_OTHER_STRICT=1|just time-gpu-probe-tred"
  "hostreg|r0928-hostreg|2|box=BLOOMERY_OTHER_STRICT=1|just time-gpu-probe-hostreg"
)

field() { # field <job line> <n>: the n-th |-field (1-based)
  local IFS='|'
  # shellcheck disable=SC2206 # the split on | is the point
  local f=($1)
  echo "${f[$(($2 - 1))]:-}"
}

# The generic ncu form has no recipe: the box command a recipe would carry, precheck first.
Q6K_BOX='bash tools/ref/card-precheck.sh && cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin generate_qwen3moe && BLOOMERY_GEN_BIN=target/release/generate_qwen3moe BLOOMERY_NCU_KERNELS='"'"'^gemm_q6k$'"'"' BLOOMERY_NCU_SKIP=0 BLOOMERY_NCU_COUNT=2 BLOOMERY_NCU_DEPTHS=4096 BLOOMERY_NCU_MODE=eager bash tools/ref/ncu-gpu.sh'
q6k_cmd() { ./tools/box.sh "$Q6K_BOX"; }

# The engram read times nothing and takes no lease: the probe is built and checked first (a build under
# this sitting's hold), then run twice as a read-only command — the held-out prompt, then the prose
# control. A read-only command may not name target/, so the two runs go over stdin.
ENGRAM_RUNS="./target/release/probe_engram --ids \$BLOOMERY_DATA/engram/corpus-threads.ids --offset $THREADS_OFFSET --limit 4096 && ./target/release/probe_engram --ids \$BLOOMERY_DATA/engram/corpus-prose.ids --offset 0 --limit 4096"
engram_cmd() {
  BLOOMERY_BOX_ENV='' just probe-engram --check || return $?
  BLOOMERY_BOX_READONLY=1 ./tools/box.sh 'bash -s' <<< "$ENGRAM_RUNS"
}

# The union split needs bench_v41_host to tally the r8 union's gate-up and down apart (kinds
# unionr8_gu and unionr8_down); until it does, the job is skipped by name (the card's term does not exist).
skip_reason() { # skip_reason <job>: a reason on stdout when the job cannot run in this tree
  case $1 in
    union) grep -q 'unionr8_down' "$ROOT/crates/model/src/bin/bench_v41_host.rs" ||
      echo "bench_v41_host has no unionr8_gu / unionr8_down tallies (the proposed split is not in this tree)" ;;
  esac
}

# --- post-reads: each reads <job>.log (and, through read-only box commands, the runner's kept files) ---

# The first `<prefix> <path>` token on a line, a path ending in <suffix>.
path_in() { # path_in <log> <suffix>
  grep -oE "/[^ ,]+$2" "$1" | head -n 1
}

# The implied SM clock of the P = 4096 flash launches: 48 x 2,319,186 cycles (rig-log 09-26#q3swz-ncu,
# layer 24's launch) over the summed flash time.
clock_ghz() { # clock_ghz <summed flash us>
  awk -v us="$1" 'BEGIN { if (us <= 0) exit 1; printf "%.3f\n", 48 * 2319186 / (us * 1e3) }'
}

# s: per round, the least-squares slope of the prompt ms (P / pp_tok/s) over P, over the rows' mean p50.
s_fit() { # s_fit <log>
  python3 - "$1" << 'PY'
import re, statistics, sys
rows = {}
for line in open(sys.argv[1]):
    m = re.match(r"ROW r(\d+) (\S+) d=(\d+) .*\| p50 ([0-9.]+) ms .*\| pp_tok/s ([0-9.]+) \(n=(\d+), passes=(\d+)", line)
    if not m:
        continue
    r, label, d, p50, pp, n, passes = m.groups()
    tags = " ".join(t for t in ("[cold]", "[other-busy]") if t in line)
    rows.setdefault(int(r), []).append((int(n), float(pp), float(p50), int(passes), tags))
if not rows:
    sys.exit("s_fit: no ROW line with a p50 and a pp_tok/s column")
ss = []
for r, xs in sorted(rows.items()):
    bad = [x for x in xs if x[3] != 1 or x[4]]
    P = [x[0] for x in xs]
    t = [1e3 * x[0] / x[1] for x in xs]
    pm, tm = statistics.fmean(P), statistics.fmean(t)
    den = sum((p - pm) ** 2 for p in P)
    if len(xs) < 2 or den == 0:
        print(f"round {r}: fewer than two prompt lengths, no slope")
        continue
    slope = sum((p - pm) * (y - tm) for p, y in zip(P, t)) / den
    step = statistics.fmean(x[2] for x in xs)
    s = slope / step
    note = f" VOID: {len(bad)} row(s) with passes != 1 or [cold]/[other-busy]" if bad else ""
    if not bad:
        ss.append(s)
    print(f"round {r}: P {P} prompt ms {[round(y, 3) for y in t]} slope {slope:.3f} ms/position "
          f"step p50 {step:.3f} ms s {s:.3f}{note}")
if ss:
    print(f"s over {len(ss)} clean rounds: mean {statistics.fmean(ss):.3f} min {min(ss):.3f} max {max(ss):.3f}")
PY
}

# The join term's two halves: the nsys mean host bridge a replay (ms, the runner's line) and the kept,
# non-warm stat step records of the same run (leg_us, served), read on the box by records.py.
bridge_ms() { # bridge_ms <log>
  sed -nE 's/^  host bridge \(go -> join, all layers\) ([0-9.]+) .*/\1/p' "$1" | head -n 1
}
join_us() { # join_us <bridge ms> <mean leg us> <mean served>
  awk -v b="$1" -v l="$2" -v s="$3" 'BEGIN { if (s <= 0) exit 1; printf "%.2f\n", (b * 1e3 - l) / s }'
}
STAT_STEP_MEANS=$(cat << 'PY'
import json, statistics, subprocess, sys
out = subprocess.run([sys.executable, "tools/bloomery/records.py", "parse", sys.argv[1]], capture_output=True,
                     text=True, check=True).stdout
xs = []
for line in out.splitlines():
    kind, _, js = line.partition("\t")
    if kind == "stat_step":
        f = json.loads(js)
        if not f.get("warm"):
            xs.append(f)
if not xs:
    sys.exit("no non-warm stat step record")
print(f"stat_step n={len(xs)} leg_us_mean={statistics.fmean(x['leg_us'] for x in xs):.2f} "
      f"served_mean={statistics.fmean(x['served'] for x in xs):.3f} "
      f"served_min={min(x['served'] for x in xs)} served_max={max(x['served'] for x in xs)}")
PY
)

# The ncu CSV's throughput rows per launch (the q6k job's term and the units beside it), read on the box.
NCU_UNITS=$(cat << 'PY'
import csv, re, sys
rows = list(csv.reader(open(sys.argv[1])))
hdr = next((i for i, r in enumerate(rows) if r and r[0] == "ID"), None)
if hdr is None:
    sys.exit(f"{sys.argv[1]}: no ncu CSV header")
col = {n: i for i, n in enumerate(rows[hdr])}
seen = set()
for r in rows[hdr + 1:]:
    if len(r) <= col["Metric Value"]:
        continue
    name = r[col["Metric Name"]]
    if re.search(r"Throughput|Duration|SM Frequency|Elapsed Cycles|stalled_(long_scoreboard|barrier|mio_throttle|lg_throttle|short_scoreboard)", name):
        key = (r[col["ID"]], name)
        if key in seen:
            continue
        seen.add(key)
        print(f"launch {r[col['ID']]} {r[col['Kernel Name']].split('(')[0]} grid {r[col['Grid Size']]} | "
              f"{name} {r[col['Metric Value']]} {r[col['Metric Unit']]}")
PY
)

ro_box() { # ro_box <command>, the script on stdin: a read-only box command (syncs nothing, builds nothing)
  BLOOMERY_BOX_READONLY=1 ./tools/box.sh "$1"
}

post() { # post <job> <log>: the job's post-read into stdout
  local job=$1 log=$2 p b us means leg served
  case $job in
    q6k)
      p=$(sed -nE 's/.*원본은 (\/[^ ]+\.csv).*/\1/p' "$log" | head -n 1)
      [ -n "$p" ] || { echo "post q6k: no CSV path in the log"; return 1; }
      ro_box "python3 - $p" <<< "$NCU_UNITS"
      ;;
    kv512 | clock4096)
      p=$(path_in "$log" .sqlite)
      [ -n "$p" ] || { echo "post $job: no sqlite path in the log"; return 1; }
      if [ "$job" = kv512 ]; then
        ro_box "python3 tools/ref/nsys-table.py $p --steps 1 --pattern gemm_"
      else
        ro_box "python3 tools/ref/nsys-table.py $p --steps 1 --pattern gqa_prefill_flash" > "$log.flash"
        cat "$log.flash"
        us=$(awk '$1 ~ /^gqa_prefill_flash/ { s += $(NF - 1) } END { if (s > 0) print s }' "$log.flash")
        [ -n "$us" ] && echo "implied SM clock: 48 x 2,319,186 cycles over $us us = $(clock_ghz "$us") GHz"
      fi
      ;;
    s) s_fit "$log" ;;
    join)
      b=$(bridge_ms "$log")
      p=$(sed -nE 's/^--- files: [^,]+, [^,]+, ([^ ,]+\.txt).*/\1/p' "$log" | head -n 1)
      [ -n "$b" ] && [ -n "$p" ] || { echo "post join: bridge '$b' or run log '$p' missing from the log"; return 1; }
      means=$(ro_box "python3 - $p" <<< "$STAT_STEP_MEANS") || { echo "$means"; return 1; }
      echo "$means"
      leg=$(sed -nE 's/.*leg_us_mean=([0-9.]+).*/\1/p' <<< "$means")
      served=$(sed -nE 's/.*served_mean=([0-9.]+).*/\1/p' <<< "$means")
      echo "bridge $b ms a replay; (bridge - leg) / served = $(join_us "$b" "$leg" "$served") us a service"
      ;;
    enqueue) python3 "$ROOT/tools/bloomery/records.py" lines "$log" stat_prefill_split ;;
    gt) grep -E '^summary ' "$log" ;;
    imma) grep -E '^probe (check|imma) ' "$log" ;;
    engram) grep -E '^probe (check|engram) ' "$log" ;;
    union) grep -E 'unionr8' "$log" ;;
    glm) grep -E '^(#|--- |    time |    SMOKE)' "$log" | grep -vE '^--- sources' ;;
    tred) grep -E '^probe (check|peer|tred) ' "$log" ;;
    hostreg) grep -E '^probe (check|hostreg) ' "$log" ;;
  esac
}

self_test() {
  local d fail=0 got
  d=$(mktemp -d "${TMPDIR:-/tmp}/sitting-r0928.XXXXXX") || return 2
  cat > "$d/s.log" << 'EOF'
ROW r1 ours d=1 n=96 ctx=256 | tok/s(mean) 200 @ n=96, depth 1, A6000 | p50 5.000 ms | mean 5.1 ms | tok/s(p50) 200 | warm 0 | first10_p50 5 | last10_p50 5 | distinct_tokens 9 | nodes 700 | pp_tok/s 200.0 (n=1, passes=1) | slot 1/5 | wall 3s | majflt 0 (timed 0; ≤ 1 % of W 1 s)
ROW r1 ours d=3 n=96 ctx=256 | tok/s(mean) 200 @ n=96, depth 3, A6000 | p50 5.000 ms | mean 5.1 ms | tok/s(p50) 200 | warm 0 | first10_p50 5 | last10_p50 5 | distinct_tokens 9 | nodes 700 | pp_tok/s 400.0 (n=3, passes=1) | slot 3/5 | wall 3s | majflt 0 (timed 0; ≤ 1 % of W 1 s)
ROW r1 ours d=5 n=96 ctx=256 | tok/s(mean) 200 @ n=96, depth 5, A6000 | p50 5.000 ms | mean 5.1 ms | tok/s(p50) 200 | warm 0 | first10_p50 5 | last10_p50 5 | distinct_tokens 9 | nodes 700 | pp_tok/s 500.0 (n=5, passes=1) | slot 5/5 | wall 3s | majflt 0 (timed 0; ≤ 1 % of W 1 s)
ROW r2 ours d=1 n=96 ctx=256 | tok/s(mean) 200 @ n=96, depth 1, A6000 | p50 5.000 ms | mean 5.1 ms | tok/s(p50) 200 | warm 0 | first10_p50 5 | last10_p50 5 | distinct_tokens 9 | nodes 700 | pp_tok/s 200.0 (n=1, passes=2) | slot 1/5 | wall 3s | majflt 0 (timed 0; ≤ 1 % of W 1 s)
ROW r2 ours d=5 n=96 ctx=256 | tok/s(mean) 200 @ n=96, depth 5, A6000 | p50 5.000 ms | mean 5.1 ms | tok/s(p50) 200 | warm 0 | first10_p50 5 | last10_p50 5 | distinct_tokens 9 | nodes 700 | pp_tok/s 500.0 (n=5, passes=1) | slot 5/5 | wall 3s | majflt 0 (timed 0; ≤ 1 % of W 1 s)
EOF
  # Round 1: prompt ms 5, 7.5, 10 at P 1, 3, 5 — slope 1.25 ms a position over a 5 ms step, s = 0.25.
  got=$(s_fit "$d/s.log")
  grep -q '^round 1: .* slope 1.250 ms/position step p50 5.000 ms s 0.250$' <<< "$got" || { echo "self-test: s_fit round 1: $got" >&2; fail=1; }
  grep -q '^round 2: .*VOID: 1 row' <<< "$got" || { echo "self-test: s_fit did not void a passes=2 round: $got" >&2; fail=1; }
  grep -q '^s over 1 clean rounds: mean 0.250' <<< "$got" || { echo "self-test: s_fit summary: $got" >&2; fail=1; }
  printf '%s\n' "  time step 22.300 over 7 steps; ..." "  host bridge (go -> join, all layers) 15.600  card overlap inside it 1.2  exposed 3.4  (bridge - overlap = 14.4, the rest ...)" > "$d/j.log"
  got=$(bridge_ms "$d/j.log")
  [ "$got" = 15.600 ] || { echo "self-test: bridge_ms read '$got', want 15.600" >&2; fail=1; }
  # (15.6 ms - 15,000 us) / 40 services = 15.00 us.
  got=$(join_us 15.600 15000 40)
  [ "$got" = 15.00 ] || { echo "self-test: join_us gave '$got', want 15.00" >&2; fail=1; }
  join_us 1 1 0 > /dev/null 2>&1 && { echo "self-test: join_us divided by zero served" >&2; fail=1; }
  # 48 x 2,319,186 cycles in 72,000 us: 1.546 GHz.
  got=$(clock_ghz 72000)
  [ "$got" = 1.546 ] || { echo "self-test: clock_ghz gave '$got', want 1.546" >&2; fail=1; }
  echo "--- 원본: /root/bloomery-data/nsys/x.nsys-rep, /root/bloomery-data/nsys/x.sqlite" > "$d/n.log"
  got=$(path_in "$d/n.log" .sqlite)
  [ "$got" = /root/bloomery-data/nsys/x.sqlite ] || { echo "self-test: path_in read '$got'" >&2; fail=1; }
  got=$(skip_reason kv512)
  [ -z "$got" ] || { echo "self-test: kv512 has a skip reason: $got" >&2; fail=1; }
  rm -rf "$d"
  [ "$fail" = 0 ] && echo "sitting-r0928 self-test: ok"
  return "$fail"
}

# --- the sitting ---

MODE=run ONLY=
while [ $# -gt 0 ]; do
  case $1 in
    --dry-run) MODE=dry ;;
    --self-test) MODE=self ;;
    --only) ONLY=${2:-}; shift ;;
    *) echo "usage: sitting-r0928.sh [--dry-run | --self-test] [--only job,job]" >&2; exit 64 ;;
  esac
  shift
done
cd "$ROOT" || exit 2
[ "$MODE" != self ] || { self_test; exit $?; }

selected() { # selected <job>: 0 when --only is empty or names it
  [ -z "$ONLY" ] || [[ ",$ONLY," == *",$1,"* ]]
}
if [ -n "$ONLY" ]; then
  for n in ${ONLY//,/ }; do
    printf '%s\n' "${JOBS[@]}" | cut -d'|' -f1 | grep -qx "$n" || { echo "sitting-r0928: no job '$n'" >&2; exit 64; }
  done
fi

# The table first: every job's card, command and minutes, and the total of the jobs that run.
total=0
printf '%-9s %-24s %4s  %s\n' job card min command
for j in "${JOBS[@]}"; do
  name=$(field "$j" 1) card=$(field "$j" 2) min=$(field "$j" 3) envs=$(field "$j" 4) cmd=$(field "$j" 5)
  selected "$name" || continue
  why=$(skip_reason "$name")
  if [ -n "$why" ]; then
    printf '%-9s %-24s %4s  SKIP: %s\n' "$name" "$card" - "$why"
  else
    total=$((total + min))
    case $cmd in
      q6k_cmd) cmd="ncu-gpu.sh's generic form on generate_qwen3moe (the box command: --dry-run)" ;;
      engram_cmd) cmd="just probe-engram --check, then probe_engram twice read-only (--dry-run)" ;;
    esac
    printf '%-9s %-24s %4s  %s%s\n' "$name" "$card" "$min" "${envs:+[$envs] }" "$cmd"
  fi
done
echo "predicted: $total min of jobs [the cards' minutes], plus the builds each recipe runs before its lease (not in the cards)"

box_env_of() { # box_env_of <card> <envs>: BLOOMERY_BOX_ENV for the job
  local e=BLOOMERY_LEASE_CARD=docs/cards/$1.card part
  IFS=';' read -r -a parts <<< "$2"
  local in_box=''
  for part in ${parts[@]+"${parts[@]}"}; do
    case $part in
      box=*) in_box=1; e="$e ${part#box=}" ;;
      mac=*) in_box= ;;
      *) [ -z "$in_box" ] || e="$e $part" ;;
    esac
  done
  echo "$e"
}
mac_env_of() { # mac_env_of <envs>: the NAME=value entries the Mac side exports
  local part in_mac='' out=''
  IFS=';' read -r -a parts <<< "$1"
  for part in ${parts[@]+"${parts[@]}"}; do
    case $part in
      mac=*) in_mac=1; out="$out ${part#mac=}" ;;
      box=*) in_mac= ;;
      *) [ -z "$in_mac" ] || out="$out $part" ;;
    esac
  done
  echo "${out# }"
}

if [ "$MODE" = dry ]; then
  rc=0
  for j in "${JOBS[@]}"; do
    name=$(field "$j" 1) card=$(field "$j" 2) envs=$(field "$j" 4) cmd=$(field "$j" 5)
    selected "$name" || continue
    echo
    echo "=== $name"
    echo "[dry] card: $(python3 tools/ref/card.py check "docs/cards/$card.card" 2>&1 | tail -n 1)" || rc=1
    python3 tools/ref/card.py check "docs/cards/$card.card" > /dev/null 2>&1 || rc=1
    why=$(skip_reason "$name")
    [ -z "$why" ] || { echo "[dry] SKIP: $why"; continue; }
    menv=$(mac_env_of "$envs")
    benv=$(box_env_of "$card" "$envs") || rc=1
    echo "[dry] BLOOMERY_BOX_ENV='$benv'"
    case $cmd in
      q6k_cmd) echo "[dry] command: ${menv:+$menv }./tools/box.sh '$Q6K_BOX'" ;;
      engram_cmd) echo "[dry] command: just probe-engram --check, then BLOOMERY_BOX_READONLY=1 ./tools/box.sh 'bash -s' with stdin: $ENGRAM_RUNS" ;;
      *) echo "[dry] command: ${menv:+$menv }$cmd" ;;
    esac
  done
  echo
  echo "[dry] hold: $HOLD (owner $OWNER), put up after a quiet guard, taken down on exit"
  exit "$rc"
fi

DIR=${BLOOMERY_SITTING_DIR:-$ROOT/target/sitting-$OWNER-$(date -u +%Y%m%dT%H%M%SZ)}
mkdir -p "$DIR" || exit 2
SLOG=$DIR/sitting.log
say() { echo "[sitting $(date -u +%H:%M:%SZ)] $*" | tee -a "$SLOG"; }

# A quiet box first: the no-op's guard waits out the lease and every hold (BLOOMERY_BOX_WAIT, 30 min).
say "waiting for a quiet box (the lease and every other hold down)"
BLOOMERY_HOLD_OWNER='' ./tools/box.sh 'true' >> "$SLOG" 2>&1 || { say "the box did not go quiet (rc $?); no hold, nothing ran"; exit 75; }
drop_hold() { BLOOMERY_BOX_READONLY=1 ./tools/box.sh "rm -f $HOLD" >> "$SLOG" 2>&1 && say "hold $HOLD down"; }
BLOOMERY_BOX_READONLY=1 ./tools/box.sh "touch $HOLD" >> "$SLOG" 2>&1 || { say "could not put $HOLD up"; exit 2; }
trap drop_hold EXIT
trap 'exit 130' INT TERM
export BLOOMERY_HOLD_OWNER=$OWNER
say "hold $HOLD up; logs in $DIR"

red=()
for j in "${JOBS[@]}"; do
  name=$(field "$j" 1) card=$(field "$j" 2) min=$(field "$j" 3) envs=$(field "$j" 4) cmd=$(field "$j" 5)
  selected "$name" || continue
  why=$(skip_reason "$name")
  [ -z "$why" ] || { say "$name: SKIP — $why"; continue; }
  log=$DIR/$name.log
  benv=$(box_env_of "$card" "$envs")
  menv=$(mac_env_of "$envs")
  for try in 1 2 3; do
    say "$name: start (try $try, card $card, predicted $min min) BLOOMERY_BOX_ENV='$benv'${menv:+ mac: $menv}"
    t0=$(date +%s)
    # shellcheck disable=SC2086 # the command and the Mac env are word lists
    case $cmd in
      q6k_cmd | engram_cmd) (for kv in $menv; do export "${kv?}"; done; BLOOMERY_BOX_ENV=$benv "$cmd") > "$log" 2>&1 ;;
      *) env $menv BLOOMERY_BOX_ENV="$benv" $cmd > "$log" 2>&1 ;;
    esac
    rc=$?
    say "$name: end rc=$rc wall $(($(date +%s) - t0)) s (predicted $min min)"
    [ "$rc" = 75 ] || break
    say "$name: rc 75 is contention; retrying"
  done
  if [ "$rc" = 0 ]; then
    post "$name" "$log" > "$DIR/$name.post" 2>&1 || say "$name: the post-read failed (rc $?), see $DIR/$name.post"
    sed "s/^/  [$name] /" "$DIR/$name.post" | tee -a "$SLOG"
  else
    red+=("$name:$rc")
  fi
done
say "done; non-zero: ${red[*]:-none}"
[ ${#red[@]} = 0 ]
