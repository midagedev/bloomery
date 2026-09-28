#!/usr/bin/env bash
# Sitting r0928: the research probes of the 2026-09-28 idea audit, in one run from the Mac (lead-only).
# Each job is an existing recipe or runner with its prediction card (docs/cards/r0928-<slug>.card);
# this script owns only the order, the hold, the logs and the post-reads.
# Predicted: 62 min of jobs, the cards' minutes summed (58 while the union job is skipped; --dry-run sums the
# jobs that run), plus the builds each recipe runs before its lease.
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
# Order (the page cache): the Qwen3 jobs, the Qwen3 half of the A6000 A/A last among them (aaq3: its twelve
# loads read the file the Qwen3 jobs left cached), then the V4.1 jobs with the V4.1 half of the A/A last among
# them (aav41, whose prose:512 arms read the engram rows of prose ids 0..511), the engram residency read right
# after them (its card asks for the cache the V4.1 prompts left; its warm control is those 512 ids), GLM, and last the two-card probes, whose 8 GiB host chunks
# would evict the model files ahead of the loads. Every V4.1 job runs the default placement, the id prefix (no
# BLOOMERY_HOT_LIST): each loads the same host set, the experts past each layer's prefix, so the first V4.1
# load reads it in and the later ones find it cached. The A/A is two jobs of one card: one lease a model,
# each next to its model's jobs.
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
# The held-out engram prompt: corpus-threads.ids from half its 4,515,000 lines (card r0928-engram-resident).
THREADS_OFFSET=2257500

# The jobs, in the sitting's order: name | card slug | predicted minutes | the command, run from ROOT with
# BLOOMERY_BOX_ENV set to the card plus the job's own entries (a `box=` prefix, `;`-separated, before `|`).
# The A/A's same-binary arms: V4.1's second arm is the first run alone (@BLOOMERY_AB_LOAD=arm, which the runner
# strips before the binary sees it), so the two share argv and environment; Qwen3's is the ours binary named
# again as a bin: arm, the runner's only same-binary form. Its $PWD stays literal on the Mac (a word of $cmd, a
# single-quoted recipe line, box.sh's "$*") and expands on the box after box.sh's cd into the remote tree.
JOBS=(
  "q6k|r0928-q6k-ncu|3|mac=BLOOMERY_MODEL=qwen3moe|q6k_cmd"
  "kv512|r0928-q3kv-nsys512|2||just nsys-gpu-qwen3moe-prefill 512"
  "clock4096|r0928-q3clock-nsys4096|2||just nsys-gpu-qwen3moe-prefill 4096"
  "s|r0928-q3s-pp|3|mac=BLOOMERY_AB_ROUNDS=3|just depth-gpu-qwen3moe 1 2 3 4 5"
  "aaq3|r0928-aa-ruler|5|mac=BLOOMERY_AB_ROUNDS=6|just depth-gpu-qwen3moe 512 bin:\$PWD/target/release/generate_qwen3moe:512"
  "imma|r0928-imma-m128|1||just time-gpu-probe-imma"
  "join|r0928-join-nsys|4|box=BLOOMERY_STEP_STATS=1;BLOOMERY_NSYS_N=8|just nsys-gpu-ds41 6"
  "enqueue|r0928-enqueue-g2|5|mac=BLOOMERY_AB_ROUNDS=1;box=BLOOMERY_STEP_STATS=1|just depth-gpu-ds41 4096"
  "aav41|r0928-aa-ruler|12|mac=BLOOMERY_AB_ROUNDS=3|just depth-gpu-ds41 prose:512 prose:512@BLOOMERY_AB_LOAD=arm"
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
ENGRAM_RUNS="./target/release/probe_engram --ids \$BLOOMERY_DATA/engram/corpus-threads.ids --offset $THREADS_OFFSET --limit 4096 && ./target/release/probe_engram --ids \$BLOOMERY_DATA/engram/corpus-prose.ids --offset 0 --limit 512"
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
# non-warm stat step records of the same run (leg_us, served; host_slots printed beside them as the
# placement's witness), read on the box by records.py. The run's plan records must all say hot_list none.
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
xs, plans = [], []
for line in out.splitlines():
    kind, _, js = line.partition("\t")
    if kind == "plan":
        plans.append(json.loads(js).get("hot_list"))
    elif kind == "stat_step":
        f = json.loads(js)
        if not f.get("warm"):
            xs.append(f)
if not plans:
    sys.exit("no plan record: the run's placement is unknown")
if any(h != "none" for h in plans):
    sys.exit(f"the run's plan records say hot_list {plans}; this job runs the id prefix (hot_list none)")
if not xs:
    sys.exit("no non-warm stat step record")
print(f"stat_step n={len(xs)} leg_us_mean={statistics.fmean(x['leg_us'] for x in xs):.2f} "
      f"served_mean={statistics.fmean(x['served'] for x in xs):.3f} "
      f"served_min={min(x['served'] for x in xs)} served_max={max(x['served'] for x in xs)} "
      f"host_slots_mean={statistics.fmean(x['host_slots'] for x in xs):.1f} plans={len(plans)} hot_list=none")
PY
)

# plan_none <log>: every plan record in the log (stripped, so an indented echo counts) says hot_list none,
# the id prefix; a log with no plan record fails, since the job's placement is then unknown.
PLAN_NONE=$(cat << 'PY'
import importlib.util, sys
spec = importlib.util.spec_from_file_location("records", sys.argv[1])
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)
schema = R.load()
hs = [r.get("hot_list") for r in (schema.read_line(x.strip()) for x in open(sys.argv[2], errors="replace"))
      if r is not None and r.kind == "plan"]
if not hs:
    sys.exit("plan: no plan record in the log; the job's placement is unknown")
bad = [h for h in hs if h != "none"]
if bad:
    sys.exit(f"plan: {len(bad)} of {len(hs)} plan record(s) ran a hot list ({bad[0]}); this job runs the id prefix")
print(f"plan: {len(hs)} record(s), hot_list none (the id prefix)")
PY
)
plan_none() { python3 - "$ROOT/tools/bloomery/records.py" "$1" <<< "$PLAN_NONE"; }

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

# The A/A (card r0928-aa-ruler): per model two same-binary arms, a (the runner's `prose` / `ours`) and b
# (`prose@BLOOMERY_AB_LOAD=arm` / the one `bin:<tree>` label), two terms each: decode tok/s(mean) and pp tok/s,
# read from every counted row (tag ROW). A V4.1 row's plan and time prompt records, echoed before it, are read
# by tools/bloomery/records.py by kind and field: the row is void unless its plan is place a with hot_list none
# (the id prefix) and exactly one time prompt record's tok/s is its row's pp_tok/s. generate_qwen3moe's lines have no
# records.py schema, so a Qwen3 row is read from its ROW line alone. A row tagged [cold], [cpu-busy] or
# [other-busy] is void; a void row is printed with its values and counts in no term. Per term: each arm's
# per-round values, mean and SD (sample SD in % of its mean), the pooled SD with its df, and the mean
# difference b / a - 1 against h = t(0.975, df) x s_p x sqrt(1/n_a + 1/n_b), card.py's interval. A term with
# fewer than two clean rounds in either arm is void.
AA_POST=$(cat << 'PY'
import importlib.util, math, re, statistics, sys
records_py, tdist_py, log, model = sys.argv[1:5]


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


t975 = module(tdist_py, "tdist").t975
schema = module(records_py, "records").load() if model == "v41" else None
TAG = re.compile(r"^(ROW|WARMUP|DISCARD|PRIME|COLD|FAIL) r(\d+) (\S+) [dp]=(\d+)")
DECODE = re.compile(r"\| tok/s\(mean\) ([0-9.]+) @")
PP = re.compile(r"\| pp_tok/s ([0-9.]+) \(n=(\d+)")
BAD = ("[cold]", "[cpu-busy]", "[other-busy]")
arm_a = {"v41": "prose", "q3": "ours"}[model]
arm_b = "prose@BLOOMERY_AB_LOAD=arm" if model == "v41" else None
plan, pending, rows = None, [], []
for raw in open(log, errors="replace"):
    line = raw.strip()
    m = TAG.match(line)
    if m:
        label = m.group(3)
        if m.group(1) == "ROW" and (label == arm_a or label == arm_b or (model == "q3" and label.startswith("bin:"))):
            dec, pp = DECODE.search(line), PP.search(line)
            if not dec or not pp:
                sys.exit(f"aa: a {label} row without tok/s(mean) or pp_tok/s: {line}")
            rows.append((int(m.group(2)), label, float(dec.group(1)), float(pp.group(1)), int(pp.group(2)), plan,
                         pending, [t for t in BAD if t in line]))
        pending = []
        continue
    if schema is None:
        continue
    rec = schema.read_line(line)
    if rec is None:
        continue
    if rec.kind == "plan":
        plan = rec
    elif rec.kind == "time_prompt":
        pending.append(rec)
if model == "q3":
    bins = sorted({r[1] for r in rows if r[1].startswith("bin:")})
    if len(bins) > 1:
        sys.exit(f"aa: more than one bin: label {bins}; the A/A has one second arm")
    arm_b = bins[0] if bins else "bin:?"
val = {}
for r, label, dec, pp, n, pl, recs, tags in rows:
    why = " ".join(tags)
    if model == "v41" and not why:
        tps = [x["tok/s"] for x in recs]
        if pl is None:
            why = "no plan record before it"
        elif pl.get("place") != "a" or pl.get("hot_list") != "none":
            why = f"its plan ran place={pl.get('place')} hot_list={pl.get('hot_list')}"
        elif tps != [pp]:
            why = f"time prompt tok/s {tps} is not its pp_tok/s {pp}"
    if why:
        print(f"aa {model} r{r} {label} VOID ({why}): decode {dec} pp {pp}")
        continue
    val.setdefault(("decode", label), {})[r] = dec
    val.setdefault((f"pp{n}", label), {})[r] = pp
terms = sorted({t for t, _ in val}, key=lambda t: (t != "decode", t)) or ["decode"]
for term in terms:
    xs = {}
    for label in (arm_a, arm_b):
        per = val.get((term, label), {})
        xs[label] = [per[r] for r in sorted(per)]
        if len(per) > 1:
            v = xs[label]
            print(f"aa {model} {term} {label}: {' '.join(f'r{r} {per[r]:.2f}' for r in sorted(per))} | n {len(v)} "
                  f"mean {statistics.fmean(v):.2f} sd {statistics.stdev(v) / statistics.fmean(v) * 100:.3f} %")
        elif per:
            print(f"aa {model} {term} {label}: {' '.join(f'r{r} {per[r]:.2f}' for r in sorted(per))} | n 1")
    na, nb = len(xs[arm_a]), len(xs[arm_b])
    if na < 2 or nb < 2:
        print(f"aa {model} {term} void: {arm_a} {na} clean round(s), {arm_b} {nb}; a term needs two in each arm")
        continue
    sa = statistics.stdev(xs[arm_a]) / statistics.fmean(xs[arm_a]) * 100
    sb = statistics.stdev(xs[arm_b]) / statistics.fmean(xs[arm_b]) * 100
    df = na + nb - 2
    sp = math.sqrt(((na - 1) * sa ** 2 + (nb - 1) * sb ** 2) / df)
    d = (statistics.fmean(xs[arm_b]) / statistics.fmean(xs[arm_a]) - 1) * 100
    h = t975(df) * sp * math.sqrt(1 / na + 1 / nb)
    cause = "an order or warm-up effect" if model == "v41" else \
        "an order, warm-up or driver-shape effect (ours --arm-sync against bin: --tokens)"
    print(f"aa {model} {term} pooled sd {sp:.3f} % df {df} | diff b/a {d:+.3f} % h {h:.3f} % (t {t975(df):.3f}) "
          f"{'inside' if abs(d) <= h else 'OUTSIDE: ' + cause}")
PY
)
# aa_decide <pooled sd %>...: the card's rule on the largest term (0.35..0.63 in, under below, over above).
aa_decide() {
  awk 'BEGIN { if (ARGC < 2) exit 1; m = ARGV[1]; for (i = 2; i < ARGC; i++) if (ARGV[i] + 0 > m + 0) m = ARGV[i]
    v = m + 0 < 0.35 ? "below" : (m + 0 > 0.63 ? "above" : "in")
    a = v == "in" ? "the ruler stays 0.6 %" : "sd_same_binary, card.py SD_DEFAULT and the AGENTS.md ruler take " m " %"
    printf "aa verdict: largest pooled sd %s %% over %d term(s): %s — %s\n", m, ARGC - 1, v, a }' "$@"
}
# aa_post <log> <v41|q3> [<the other half's post file>]: the terms of one job, then the verdict over them and,
# when the other half's post is there, over both models' terms.
aa_post() {
  local out sds other=${3:-}
  out=$(python3 - "$ROOT/tools/bloomery/records.py" "$ROOT/tools/ref/tdist.py" "$1" "$2" <<< "$AA_POST") || { echo "$out"; return 1; }
  echo "$out"
  if [ -n "$other" ]; then
    if [ -r "$other" ]; then
      echo "aa: with the other half's terms from $other"
      out=$(printf '%s\n' "$out"; grep -E '^aa (v41|q3) [^ ]+ pooled sd ' "$other")
    else
      echo "aa: no post at $other (that half did not run or ended non-zero): the verdict is this half's alone"
    fi
  fi
  sds=$(sed -nE 's/^aa (v41|q3) [^ ]+ pooled sd ([0-9.]+) %.*/\2/p' <<< "$out")
  [ -n "$sds" ] || { echo "aa verdict: void (no term has two clean rounds in both arms)"; return 1; }
  # shellcheck disable=SC2086 # the pooled SDs, one word each
  aa_decide $sds
}

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
    enqueue)
      plan_none "$log" || return 1
      python3 "$ROOT/tools/bloomery/records.py" lines "$log" stat_prefill_split
      ;;
    imma) grep -E '^probe (check|imma) ' "$log" ;;
    engram) grep -E '^probe (check|engram) ' "$log" ;;
    aaq3) aa_post "$log" q3 ;;
    aav41) aa_post "$log" v41 "$DIR/aaq3.post" ;;
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
  # The header's total is the job table's.
  got=$(printf '%s\n' "${JOBS[@]}" | awk -F'|' '{ s += $3 } END { print s }')
  grep -q "^# Predicted: $got min of jobs," "$ROOT/tools/ref/sitting-r0928.sh" || { echo "self-test: the header's total is not the job table's $got min" >&2; fail=1; }
  got=$(skip_reason kv512)
  [ -z "$got" ] || { echo "self-test: kv512 has a skip reason: $got" >&2; fail=1; }
  # aa, Qwen3: three clean rounds; the warm-up and a FAIL row count in no term. decode: ours 200, 202, 198
  # (SD 2 over 200, 1.000 %), bin: 206, 208, 204 (2 over 206, 0.971 %); pooled sqrt((2 x 1 + 2 x 0.9426) / 4) =
  # 0.986 % at df 4, diff +3.000 % against h = 2.776 x 0.986 x sqrt(2/3) = 2.234: outside. pp: ours 7800, 7878,
  # 7722 (1.000 %), bin: 7800, 7839, 7761 (0.500 %); pooled 0.791 %, h 1.792, diff 0: inside.
  aa_q3_row() { # aa_q3_row <tag> <round> <label> <decode tok/s> <pp tok/s> [<tags>]
    echo "$1 r$2 $3 d=512 n=96 ctx=768 | tok/s(mean) $4 @ n=96, depth 512, A6000 | p50 5.000 ms | mean 5.000 ms | tok/s(p50) $4 | warm 0 | first10_p50 5 | last10_p50 5 | distinct_tokens 9 | nodes 700 | pp_tok/s $5 (n=512, passes=1, kind=ubatch) | wall 21s | majflt 0 (timed 0; ≤ 0 % of W 1 s)${6:+ $6}"
  }
  {
    aa_q3_row WARMUP 0 ours 150.00 5000.00
    aa_q3_row ROW 1 ours 200.00 7800.00
    aa_q3_row ROW 1 bin:bloomery 206.00 7800.00
    aa_q3_row ROW 2 bin:bloomery 208.00 7839.00
    aa_q3_row ROW 2 ours 202.00 7878.00
    echo "FAIL r3 ours d=512 rc=1 | no SMOKE line"
    aa_q3_row ROW 3 ours 198.00 7722.00
    aa_q3_row ROW 3 bin:bloomery 204.00 7761.00
  } > "$d/aaq3.log"
  got=$(aa_post "$d/aaq3.log" q3 2>&1)
  printf '%s\n' "$got" > "$d/aaq3.post"
  grep -q '^aa q3 decode ours: r1 200.00 r2 202.00 r3 198.00 | n 3 mean 200.00 sd 1.000 %$' <<< "$got" || { echo "self-test: aa q3 decode ours: $got" >&2; fail=1; }
  grep -q '^aa q3 decode bin:bloomery: .* | n 3 mean 206.00 sd 0.971 %$' <<< "$got" || { echo "self-test: aa q3 decode bin: $got" >&2; fail=1; }
  grep -q '^aa q3 decode pooled sd 0.986 % df 4 | diff b/a +3.000 % h 2.234 % (t 2.776) OUTSIDE: an order, warm-up or driver-shape' <<< "$got" || { echo "self-test: aa q3 decode pooled: $got" >&2; fail=1; }
  grep -q '^aa q3 pp512 pooled sd 0.791 % df 4 | diff b/a +0.000 % h 1.792 % (t 2.776) inside$' <<< "$got" || { echo "self-test: aa q3 pp512 pooled: $got" >&2; fail=1; }
  grep -q '^aa verdict: largest pooled sd 0.986 % over 2 term(s): above — ' <<< "$got" || { echo "self-test: aa q3 verdict: $got" >&2; fail=1; }
  # aa, V4.1: three clean rounds, and four void rows (r4 a [cold] row and a row whose time prompt record is not
  # its pp column, r5 a plan that ran a hot list and a [cpu-busy] row). decode: 30, 30.3, 29.7 (1.000 %) and 30,
  # 30.15, 29.85 (0.500 %), pooled 0.791 %; pp: 200, 201, 199 (0.500 %) and 200, 200.5, 199.5 (0.250 %), pooled
  # sqrt((2 x 0.25 + 2 x 0.0625) / 4) = 0.395 %, h 0.896. With the Qwen3 half's post the verdict is over four terms.
  aa_v41_arm() { # aa_v41_arm <round> <label> <hot_list> <decode tok/s> <pp tok/s> <time prompt tok/s> [<tags>]
    echo "[load] r$1 1 arm(s): - one process, one load"
    echo "    plan place=a card=A6000 ctx_max=32768 card_experts=2668 (44000000000 B) host_experts=12692 (214000000000 B) host_shadow=0 B n_l=70..71 on 38 layers card_budget=none hot_list=$3"
    echo "time prompt n=512 ms=2560.0000 tok/s=$6 passes=1 kind=batch"
    echo "ROW r$1 $2 d=512 n=96 | tok/s(mean) $4 @ n=96, depth 512, A6000 | place a | p50 33.3 ms | mean 33.3 ms | pp_tok/s $5 (n=512, passes=1, kind=batch) | majflt 0 (timed 0; ≤ 0 % of W 5 s) | slot 1/1 | wall 90s${7:+ $7}"
  }
  aa_b=prose@BLOOMERY_AB_LOAD=arm
  {
    aa_v41_arm 1 prose none 30.00 200.00 200.00
    aa_v41_arm 1 "$aa_b" none 30.00 200.00 200.00
    aa_v41_arm 2 "$aa_b" none 30.15 200.50 200.50
    aa_v41_arm 2 prose none 30.30 201.00 201.00
    aa_v41_arm 3 prose none 29.70 199.00 199.00
    aa_v41_arm 3 "$aa_b" none 29.85 199.50 199.50
    aa_v41_arm 4 prose none 31.00 210.00 210.00 '[cold]'
    aa_v41_arm 4 "$aa_b" none 31.00 210.00 209.00
    aa_v41_arm 5 prose /tmp/other-hot.txt 31.00 210.00 210.00
    aa_v41_arm 5 "$aa_b" none 31.00 210.00 210.00 '[cpu-busy]'
  } > "$d/aav41.log"
  got=$(aa_post "$d/aav41.log" v41 "$d/aaq3.post" 2>&1)
  grep -q '^aa v41 decode pooled sd 0.791 % df 4 | diff b/a +0.000 % h 1.792 % (t 2.776) inside$' <<< "$got" || { echo "self-test: aa v41 decode pooled: $got" >&2; fail=1; }
  grep -q '^aa v41 pp512 pooled sd 0.395 % df 4 | diff b/a +0.000 % h 0.896 % (t 2.776) inside$' <<< "$got" || { echo "self-test: aa v41 pp512 pooled: $got" >&2; fail=1; }
  [ "$(grep -c '^aa v41 r[45] .* VOID (' <<< "$got")" = 4 ] || { echo "self-test: aa v41 did not void the four r4/r5 rows: $got" >&2; fail=1; }
  grep -q '^aa v41 r5 prose VOID (its plan ran place=a hot_list=/tmp/other-hot.txt)' <<< "$got" || { echo "self-test: aa v41 kept a row whose plan ran a hot list: $got" >&2; fail=1; }
  grep -q '^aa v41 r4 prose@BLOOMERY_AB_LOAD=arm VOID (time prompt tok/s \[209.0\] is not its pp_tok/s 210.0)' <<< "$got" || { echo "self-test: aa v41 time prompt cross-check: $got" >&2; fail=1; }
  grep -q '^aa verdict: largest pooled sd 0.986 % over 4 term(s): above — ' <<< "$got" || { echo "self-test: aa joint verdict: $got" >&2; fail=1; }
  got=$(aa_post "$d/aav41.log" v41 "$d/absent.post" 2>&1)
  grep -q '^aa verdict: largest pooled sd 0.791 % over 2 term(s): above — ' <<< "$got" || { echo "self-test: aa v41-alone verdict: $got" >&2; fail=1; }
  # One clean round a arm is no term: the verdict is void and the post fails.
  aa_q3_row ROW 1 ours 200.00 7800.00 > "$d/aaq3-one.log"
  aa_q3_row ROW 1 bin:bloomery 206.00 7800.00 >> "$d/aaq3-one.log"
  got=$(aa_post "$d/aaq3-one.log" q3 2>&1) && { echo "self-test: aa_post passed with no term: $got" >&2; fail=1; }
  grep -q '^aa verdict: void' <<< "$got" || { echo "self-test: aa void verdict: $got" >&2; fail=1; }
  # The rule's edges: 0.35 and 0.63 are in the band; the largest term decides.
  got=$(aa_decide 0.35)
  [ "$(cut -d: -f3 <<< "$got" | cut -d' ' -f2)" = in ] || { echo "self-test: aa_decide at 0.35: $got" >&2; fail=1; }
  got=$(aa_decide 0.349)
  [ "$(cut -d: -f3 <<< "$got" | cut -d' ' -f2)" = below ] || { echo "self-test: aa_decide under 0.35: $got" >&2; fail=1; }
  got=$(aa_decide 0.2 0.63)
  [ "$(cut -d: -f3 <<< "$got" | cut -d' ' -f2)" = in ] || { echo "self-test: aa_decide at 0.63: $got" >&2; fail=1; }
  got=$(aa_decide 0.631 0.2)
  [ "$(cut -d: -f3 <<< "$got" | cut -d' ' -f2)" = above ] || { echo "self-test: aa_decide over 0.63: $got" >&2; fail=1; }
  got=$(aa_decide 0.2 0.3)
  [ "$got" = "aa verdict: largest pooled sd 0.3 % over 2 term(s): below — sd_same_binary, card.py SD_DEFAULT and the AGENTS.md ruler take 0.3 %" ] || { echo "self-test: aa_decide below: $got" >&2; fail=1; }
  # plan_none: an id-prefix log passes; one plan that ran a list, or no plan at all, fails.
  aa_v41_arm 1 prose none 30.00 200.00 200.00 > "$d/plan.log"
  got=$(plan_none "$d/plan.log" 2>&1)
  [ "$got" = "plan: 1 record(s), hot_list none (the id prefix)" ] || { echo "self-test: plan_none on an id-prefix log: $got" >&2; fail=1; }
  aa_v41_arm 2 prose /tmp/other-hot.txt 30.00 200.00 200.00 >> "$d/plan.log"
  got=$(plan_none "$d/plan.log" 2>&1) && { echo "self-test: plan_none passed a plan that ran a hot list: $got" >&2; fail=1; }
  grep -q '^plan: 1 of 2 plan record(s) ran a hot list (/tmp/other-hot.txt)' <<< "$got" || { echo "self-test: plan_none's refusal: $got" >&2; fail=1; }
  echo "ROW r1 prose d=512 n=96 | tok/s(mean) 30.00" > "$d/noplan.log"
  got=$(plan_none "$d/noplan.log" 2>&1) && { echo "self-test: plan_none passed a log with no plan record: $got" >&2; fail=1; }
  # The join's stat step means (the box side, run here on a fixed run log): the warm step counts in no mean;
  # (28,700 + 29,300) / 2 = 29,000 us, host slots (198 + 200) / 2 = 199.0. A plan that ran a list fails.
  step_row() { # step_row <i> <warm flag or -> <leg us> <host slots>
    local w=" $2"
    [ "$2" != - ] || w=
    echo "stat step $1$w served=40 leg_us=$3 straggle_us=1.0 straggle_max_us=2.0 host_slots=$4 host_w2=0.5000 overlap=0 union=0 go_early=0 parks=0 majflt=0 minflt=0 vram_free=1 eng_warm=0 eng_cold=0 eng_direct=0 eng_wait_us=0.0 eng_helper_us=0.0 eng_classify_us=0.0"
  }
  {
    aa_v41_arm 1 prose none 30.00 200.00 200.00 | sed -n 's/^    //p'
    step_row 0 warm 40000.0 150
    step_row 1 - 28700.0 198
    step_row 2 - 29300.0 200
  } > "$d/join-run.txt"
  got=$(python3 - "$d/join-run.txt" <<< "$STAT_STEP_MEANS" 2>&1)
  [ "$got" = "stat_step n=2 leg_us_mean=29000.00 served_mean=40.000 served_min=40 served_max=40 host_slots_mean=199.0 plans=1 hot_list=none" ] || { echo "self-test: stat step means: $got" >&2; fail=1; }
  sed 's/hot_list=none/hot_list=\/tmp\/other-hot.txt/' "$d/join-run.txt" > "$d/join-hot.txt"
  got=$(python3 - "$d/join-hot.txt" <<< "$STAT_STEP_MEANS" 2>&1) && { echo "self-test: stat step means passed a plan that ran a hot list: $got" >&2; fail=1; }
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
