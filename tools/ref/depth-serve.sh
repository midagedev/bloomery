#!/usr/bin/env bash
# Decode and prompt rate of any model through `bloomery-serve`, one server an arm and a round, under the timing
# lease (runs on the box).
#
#   BLOOMERY_MODEL=mimo2 just depth-serve <tree>:<D> [<tree>:<D> | env=K=V[,K=V]:<tree>:<D>] ...
#   BLOOMERY_MODEL=mimo2 BLOOMERY_BOX_ENV='BLOOMERY_DRY=1' just depth-serve bloomery-base:4032 bloomery-r5c:4032
#   bash tools/ref/depth-serve.sh --summary [LOG]    the two tables again from a log's ROW lines (stdin without LOG)
#   bash tools/ref/depth-serve.sh --self-test        the stub test (the Mac and the box; tools/ref/card-tests/run.sh runs it)
#
# Why a server. `generate` refuses families the engine's AnyEngine does not open, and a per-family generate_* is a
# second copy of the decode loop. `bloomery-serve` seats every family (crates/gpu-gates/src/bin/shared/serve_seats/)
# and answers a /completion with llama-server's `timings`, so one runner times any model, through the path a user
# runs, at the serving defaults.
#
# Arms. `<tree>:<D>` is the binary ~/repo/<tree>/target/release/bloomery-serve (the tree is a directory name under
# ~/repo, as ab-decode.sh's) and the first <D> ids of the model's prose corpus as the prompt
# ($BLOOMERY_DATA/<profile name>/corpus-prose.ids, one id a line; DEPTH_SERVE_IDS names another file). `env=K=V,K=V:`
# before an arm sets those levers for that arm's server only (lever-arms.sh checks them against the lever registry,
# as the other depth runners do); the variables show in the arm's row label. The model is BLOOMERY_MODEL's profile
# (tools/ref/models/<name>.sh): its file is `-m`, and the seat is the file's architecture's.
#
# Per arm and round, one fresh server (the load is the price of a round and is outside every timed number):
#   1. the server at the serving defaults — `-m <file> --host 127.0.0.1 --port P --ctx-size C`, no --parallel, no
#      lever — under `timeout` (BLOOMERY_ARM_BOUND, default 900 s, covers its load and request), BLOOMERY_REF_MODEL
#      removed from its environment as gate_mimo2_serve does (the file is named by -m);
#   2. its `listening on http://127.0.0.1:P` line and a 200 from /health, which the seat prints once the model is
#      loaded; the line's ctx must be C, else the arm fails by name;
#   3. one /completion of the D ids, temperature 0, `n_predict` N, ignore_eos, cache_prompt off (lcpp-warm.sh's
#      lcpp_srv_request: it fails on a response with no `timings`, a prompt_n other than D, a cache_n above 0 and a
#      predicted_n other than N), the witness blocks and the major-fault count (pgmajfault) around it;
#   4. TERM to the server's captured pid and a wait for its exit (KILL after 30 s).
# No warm-up request is sent: every arm's single request meets the same first-request state of its fresh server, and
# the load-to-load scatter is in the ruler. The rates are llama-server's (lcpp-warm.sh's header has the derivation):
# tg is timings.predicted_per_second, the N - 1 decode steps over their own wall, pp is timings.prompt_per_second,
# the D ids over the prompt's wall; neither clock holds the HTTP exchange or the tokenization.
#
# Context. bloomery-serve feeds the generated token k at position D + k - 1 and never feeds the last one, so a
# request stays inside the context when D + N - 1 <= C (the `n + len > ctx_max` stop in crates/serve/src/genloop.rs).
# An arm past that is refused before the lease: at the defaults (C 4160, N 128) the longest D is 4033, and D 4096 at
# C 4160 takes DEPTH_SERVE_N of at most 65.
#
# Rows and tables. One row a server: `ROW r<k> <label> D=<D> | tg_tok/s | pp_tok/s | load_s | majflt`, the label the
# tree (and `@K=V,...`), a ` [cpu-busy]` or ` [other-busy]` tag when the guards saw a co-tenant. Then per arm the means,
# and for every arm after the first arm of its D the difference against that first arm in tok/s and in percent with
# its 95 % interval: the pooled-SD two-sample interval (tools/ref/card.py's half_width at df 2N - 2, the ruler of
# AGENTS.md 「Know the ruler」), arm minus first, so + is the arm that is faster. Arms at different D are never compared.
#
# Order. The arms rotate by one place every round (arm k of round r is arm (k + r - 1) mod n, ab-decode.sh's rule),
# BLOOMERY_AB_ROUNDS rounds, default 4.
#
# Knobs (each reaches the box through BLOOMERY_BOX_ENV). None is a BLOOMERY_ name, which would have to be a row of the lever
# registry (tools/check-levers.sh) though no binary reads it; a BLOOMERY_SERVE_ variable set in the environment is refused
# by name, so a caller who spells a knob that way is told and not silently run at the default:
#   DEPTH_SERVE_CTX          the server's --ctx-size C, default 4160
#   DEPTH_SERVE_N            n_predict N, at least 2 (tg needs a step), default 128
#   DEPTH_SERVE_PORT         the loopback port, default 18180; the server's own listening line must name it, so a
#                            foreign process on it makes the server fail to bind and the arm fail by name; 0 is the
#                            kernel's pick
#   DEPTH_SERVE_REQ_BOUND    seconds the request may take, default 300
#   DEPTH_SERVE_IDS          the prose ids file, default $BLOOMERY_DATA/<profile name>/corpus-prose.ids
#   DEPTH_SERVE_LOAD_EST     seconds one load takes, for the predicted wall line (mimo2: 41, the lead's figure)
#
# Refused before the lease, by name: no model, an arm that does not parse, a tree or its binary missing (rc 2), a
# binary older than the Cargo files and sources its dep-info lists (rc 3, timing-card.sh's assert_fresh_binary), ids
# missing or short, a context that does not hold D + N - 1. After the lease an arm fails (rc 1, the server stopped,
# its log's tail printed) on a server that exits or does not answer /health within its bound, a listening line with
# another port or ctx, or a request that fails (no timings among them). Contention is the lease's and the guards'
# (rc 75). BLOOMERY_DRY=1 prints the plan (arms, order, commands, predicted wall) and the binaries' state, takes no
# lease and starts nothing; the recipe builds nothing under it.
set -uo pipefail
__d=${BASH_SOURCE[0]%/*}
[ "$__d" != "${BASH_SOURCE[0]}" ] || __d=.
ROOT=$(cd "$__d/../.." && pwd)
HERE=$ROOT/tools/ref
unset __d
CARD_PY=$HERE/card.py

MODE=run
case ${1:-} in
  --self-test) MODE=selftest; shift ;;
  --summary) MODE=summary; shift ;;
esac
if [ "$MODE" = selftest ]; then
  # The stub profile on any host: no tier, no card, and none of the caller's runner variables.
  unset BLOOMERY_TIER BLOOMERY_FIXTURE_VARIANT BLOOMERY_REF_MODEL BLOOMERY_REF_MODEL_PROFILE BLOOMERY_TIMING_CARDS \
    BLOOMERY_TIMING_GPU BLOOMERY_DRY BLOOMERY_AB_ROUNDS BLOOMERY_ARM_BOUND BLOOMERY_LEASE_CARD BLOOMERY_CPU_BUSY_COMMS
  BLOOMERY_MODEL=mimo2
elif [ "$MODE" = run ] && [ -z "${BLOOMERY_MODEL:-}${BLOOMERY_REF_MODEL_PROFILE:-}" ]; then
  echo "depth-serve.sh: no model: BLOOMERY_MODEL names the profile (tools/ref/models/<name>.sh), e.g. BLOOMERY_MODEL=mimo2" >&2
  exit 64
fi

LEVER_ARM_RUNNER=depth-serve.sh
LEVER_REGISTRY=$ROOT/crates/levers/src/registry.rs
if [ "$MODE" != summary ]; then
  # shellcheck source=tools/ref/ref-paths.sh
  source "$HERE/ref-paths.sh"
  # The timing card, the stale-binary rule, the guards and, through it, the lease and the witness block.
  # shellcheck source=tools/ref/timing-card.sh
  source "$HERE/timing-card.sh"
  # The server arms' start, request and stop (the llama-server runners' functions; the server here is bloomery-serve).
  # shellcheck source=tools/ref/lcpp-warm.sh
  source "$HERE/lcpp-warm.sh"
  # shellcheck source=tools/ref/lever-arms.sh
  source "$HERE/lever-arms.sh" || exit 2
fi

SERVE_BIN=target/release/bloomery-serve
# The runner's own waits per server beyond the load and the request, for the wall line: two witness blocks and two
# cpu guards (about a second each), the health poll's half second, the stop's fifth of a second, curl and python.
OVERHEAD_S=5
WITNESS=(head-open indent card busiest model mem pgmajfault)

ds_refuse() {
  echo "depth-serve.sh: $2" >&2
  exit "$1"
}
arm_refuse() { ds_refuse 64 "arm '$1': $2"; }

# ds_posint <name> <value> <min>: refuse by name a value that is not a whole number of at least <min>.
ds_posint() {
  case $2 in '' | *[!0-9]*) ds_refuse 64 "$1 is a whole number, got '$2'" ;; esac
  [ "$2" -ge "$3" ] || ds_refuse 64 "$1 is at least $3, got $2"
}

# The two tables from a file of ROW lines (python3; card.py's half_width is the interval).
ds_tables() {
  python3 -I - "$CARD_PY" "$1" << 'PY'
import importlib.util
import math
import re
import statistics
import sys

spec = importlib.util.spec_from_file_location('card', sys.argv[1])
card = importlib.util.module_from_spec(spec)
spec.loader.exec_module(card)
ROW = re.compile(r'^ROW r(\d+) (\S+) D=(\d+) \| tg_tok/s (\S+) \| pp_tok/s (\S+) \| load_s (\d+) \| majflt (\d+)(.*)$')
arms, order = {}, []
for line in open(sys.argv[2], errors='replace'):
    m = ROW.match(line.rstrip('\n'))
    if not m:
        continue
    _, label, dep, tg, pp, load, maj, tags = m.groups()
    key = (label, int(dep))
    if key not in arms:
        arms[key] = []
        order.append(key)
    arms[key].append({'tg': float(tg), 'pp': float(pp), 'load': int(load), 'maj': int(maj), 'tags': tags})
if not arms:
    print('depth-serve.sh: no ROW lines to summarize', file=sys.stderr)
    sys.exit(1)

print('=== per arm (tok/s): tg = timings.predicted_per_second, pp = timings.prompt_per_second; one fresh server a row ===')
for label, dep in order:
    rs = arms[(label, dep)]
    parts = []
    for name in ('tg', 'pp'):
        xs = [x[name] for x in rs]
        m = statistics.mean(xs)
        if len(xs) > 1:
            s = statistics.stdev(xs)
            parts.append(f'{name} {m:.2f} tok/s (sd {s:.2f} = {100 * s / m:.2f} %, [{min(xs):.2f}..{max(xs):.2f}], n={len(xs)})')
        else:
            parts.append(f'{name} {m:.2f} tok/s (n=1)')
    n = len(rs)
    cpu = sum('[cpu-busy]' in x['tags'] for x in rs)
    other = sum('[other-busy]' in x['tags'] for x in rs)
    load = statistics.mean(x['load'] for x in rs)
    maj = sum(x['maj'] for x in rs)
    print(f'mean {label} D={dep} | ' + ' | '.join(parts) + f' | load {load:.1f} s | majflt {maj} | [cpu-busy {cpu}/{n}] [other-busy {other}/{n}]')

print('=== difference against the first arm of each D (arm - first; 95 % interval: card.py half_width, pooled SD, df 2n - 2) ===')
firsts = {}
for key in order:
    firsts.setdefault(key[1], key)
t975 = card.t_table()
for key in order:
    label, dep = key
    first = firsts[dep]
    if key == first:
        continue
    parts = []
    for name in ('tg', 'pp'):
        b = [x[name] for x in arms[first]]
        a = [x[name] for x in arms[key]]
        m0 = statistics.mean(b)
        d = statistics.mean(a) - m0
        head = f'{name} {d:+.2f} tok/s ({100 * d / m0:+.2f} %)'
        if len(a) != len(b):
            parts.append(f'{head}, no interval: {len(a)} rounds against {len(b)}')
        elif len(a) < 2:
            parts.append(f'{head}, one round: no interval')
        else:
            sp = math.sqrt((statistics.stdev(b) ** 2 + statistics.stdev(a) ** 2) / 2)
            h = card.half_width(t975, sp, len(a))[0]
            parts.append(f'{head} ± {h:.2f} tok/s (± {100 * h / m0:.2f} %) [{"clears 0" if abs(d) > h else "holds 0"}]')
    print(f'diff {label} vs {first[0]} D={dep} | ' + ' | '.join(parts))
PY
}

# --summary [LOG]: the tables of a log's (or stdin's) ROW lines.
ds_summary_mode() {
  local rows rc
  rows=$(mktemp "${TMPDIR:-/tmp}/depth-serve-rows.XXXXXX") || exit 2
  if [ -n "${1:-}" ]; then
    [ -r "$1" ] || { rm -f "$rows"; ds_refuse 2 "no log at $1"; }
    grep -E '^ROW r[0-9]+ ' "$1" > "$rows"
  else
    grep -E '^ROW r[0-9]+ ' > "$rows"
  fi
  ds_tables "$rows"
  rc=$?
  rm -f "$rows"
  return "$rc"
}

# A compute process on the timing card as an arm starts is another round's functional run (a functional run is
# refused while the card is busy, not while the lease is held, so one can start between two arms). The server loads
# onto that card, so wait for it, polling every 10 s, and refuse after 10 minutes, rc 75: contention, not a result.
ds_guard_timing() {
  local apps i
  for ((i = 0; i < 60; i++)); do
    apps=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i "$TIMING_GPU")
    if [ -z "$apps" ]; then
      [ "$i" = 0 ] || echo "[timing-busy] $(now) the timing card is free after $((i * 10)) s" >&2
      return 0
    fi
    if [ "$i" = 0 ]; then
      echo "[timing-busy] $(now) compute apps on the timing card ($TIMING_GPU): [$(echo "$apps" | tr '\n' ';')]; waiting up to 10 min" >&2
      witness wait-timing >&2
    fi
    sleep 10
  done
  echo "[timing-busy] $(now) still busy after 10 min: [$(echo "$apps" | tr '\n' ';')]" >&2
  witness abort-timing >&2
  exit 75
}

# ds_parse_arms <arm>...: A_TREE, A_DEP, A_ENV (the env= list or empty), A_LABEL (the tree and `@list`), each arm's
# refusals by name.
ds_parse_arms() {
  local a rest envs tree dep label seen=' '
  A_TREE=() A_DEP=() A_ENV=() A_LABEL=()
  for a in "$@"; do
    rest=$a envs=''
    case $rest in
      env=*)
        rest=${rest#env=}
        case $rest in *:*) ;; *) arm_refuse "$a" "an env= list with no ':<tree>:<D>' after it" ;; esac
        envs=${rest%%:*} rest=${rest#*:}
        [ -n "$envs" ] || arm_refuse "$a" "an empty env= list"
        ;;
    esac
    case $rest in *:*) ;; *) arm_refuse "$a" "want [env=K=V[,K=V]:]<tree>:<D>" ;; esac
    tree=${rest%%:*} dep=${rest#*:}
    [[ $tree =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || arm_refuse "$a" "'$tree' is no tree name (a directory under ~/repo: letters, digits, '.', '_', '-')"
    [[ $dep =~ ^[1-9][0-9]*$ ]] || arm_refuse "$a" "'$dep' is no prompt length (a whole number of ids, at least 1)"
    [ -z "$envs" ] || arm_envs_ok "$a" "$envs"
    label=$tree${envs:+@$envs}
    case $seen in *" $label:$dep "*) arm_refuse "$a" "listed twice (an A/A needs a second tree or an env= list)" ;; esac
    seen+="$label:$dep "
    A_TREE+=("$tree") A_DEP+=("$dep") A_ENV+=("$envs") A_LABEL+=("$label")
  done
}

# ds_ids_file: the prose ids' file.
ds_ids_file() { echo "${DEPTH_SERVE_IDS:-$BLOOMERY_DATA/$MODEL_NAME/corpus-prose.ids}"; }

# ds_ids_check <max D>: the ids file's refusals, by name: readable, holding <max D> lines, each of them an id.
ds_ids_check() {
  local file nids bad
  file=$(ds_ids_file)
  [ -r "$file" ] || ds_refuse 2 "no prose ids at $file (one id a line; DEPTH_SERVE_IDS names another file)"
  nids=$(($(wc -l < "$file")))
  [ "$1" -le "$nids" ] || ds_refuse 64 "a prompt of $1 ids; $file holds $nids"
  bad=$(head -n "$1" "$file" | grep -nvE '^[0-9]+$' | head -n 1)
  [ -z "$bad" ] || ds_refuse 2 "line ${bad%%:*} of the first $1 of $file ('${bad#*:}') is no id: the file is one id a line"
}

# ds_ids <D>: the first D ids, comma-separated.
ds_ids() { head -n "$1" "$(ds_ids_file)" | paste -sd, -; }

# ds_resolve_bins: A_BIN, A_SHA and A_MTIME per arm, each tree checked once. A run refuses (2 missing, 3 stale) before
# the lease; a dry run prints the state and goes on.
ds_resolve_bins() {
  local i j dir rc
  A_BIN=() A_SHA=() A_MTIME=()
  for ((i = 0; i < ${#A_LABEL[@]}; i++)); do
    dir=$HOME/repo/${A_TREE[i]}
    A_BIN+=("$dir/$SERVE_BIN")
    for ((j = 0; j < i; j++)); do
      [ "${A_TREE[j]}" != "${A_TREE[i]}" ] || break
    done
    if [ "$j" -lt "$i" ]; then
      A_SHA+=("${A_SHA[j]}") A_MTIME+=("${A_MTIME[j]}")
      continue
    fi
    if [ ! -d "$dir" ]; then
      [ -n "$DRY" ] || ds_refuse 2 "arm '${A_LABEL[i]}': no tree $dir"
      echo "[dry] tree ${A_TREE[i]}: no $dir (a run refuses, rc 2)"
      A_SHA+=(-) A_MTIME+=(-)
      continue
    fi
    # assert_fresh_binary walks the tree it stands in (its dep-info's sources, the Cargo files): from the arm's own.
    pushd "$dir" > /dev/null || ds_refuse 2 "arm '${A_LABEL[i]}': cannot enter $dir"
    assert_fresh_binary "${A_BIN[i]}"
    rc=$?
    popd > /dev/null || exit 2
    if [ "$rc" != 0 ]; then
      [ -n "$DRY" ] || exit "$rc"
      echo "[dry] tree ${A_TREE[i]}: ${A_BIN[i]} would be refused (rc $rc)"
      A_SHA+=(-) A_MTIME+=(-)
    else
      A_SHA+=("$BIN_SHA") A_MTIME+=("$BIN_MTIME")
    fi
  done
}

# ds_server_cmd <arm index>: SRV_CMD, the arm's server command.
ds_server_cmd() {
  SRV_CMD=("${A_BIN[$1]}" -m "$MODEL" --host 127.0.0.1 --port "$PORT" --ctx-size "$CTX")
}

# ds_print_plan: the plan, for a dry run and the head of a real one.
ds_print_plan() {
  local i r k n=${#A_LABEL[@]} line envs servers=$((${#A_LABEL[@]} * ROUNDS)) est w
  echo "[plan] model=$MODEL_NAME file=$MODEL ids=$(ds_ids_file)"
  echo "[plan] servers=$servers (${n} arms x $ROUNDS rounds, one load each) n_predict=$N ctx-size=$CTX port=$PORT arm_bound=${LEASE_ARM_BOUND}s req_bound=${REQ_BOUND}s"
  for ((i = 0; i < n; i++)); do
    ds_server_cmd "$i"
    envs=''
    [ -z "${A_ENV[i]}" ] || envs=" ${A_ENV[i]//,/ }"
    echo "[plan] arm $((i + 1)): ${A_LABEL[i]} D=${A_DEP[i]}: timeout --kill-after=10 $LEASE_ARM_BOUND env -u BLOOMERY_REF_MODEL${envs} ${SRV_CMD[*]}"
  done
  for ((r = 1; r <= ROUNDS; r++)); do
    line=''
    for ((i = 0; i < n; i++)); do
      k=$(((i + r - 1) % n))
      line="$line ${A_LABEL[k]}:${A_DEP[k]}"
    done
    echo "[plan] order r$r:$line"
  done
  echo "[plan] request: POST /completion, the first D ids of the prose corpus, n_predict=$N, temperature 0, ignore_eos, cache_prompt off"
  est=${DEPTH_SERVE_LOAD_EST:-}
  [ -n "$est" ] || case $MODEL_NAME in mimo2) est=41 ;; esac
  if [ -n "$est" ]; then
    w=$((servers * (est + OVERHEAD_S)))
    echo "[plan] predicted wall >= $servers servers x (load $est s + overhead $OVERHEAD_S s) = $w s (~$((w / 60)) min) [derived: the load from DEPTH_SERVE_LOAD_EST, not measured here; the overhead from this runner's own waits]; each request adds D/pp + (N-1)/tg, known from the first row"
  else
    echo "[plan] predicted wall: $servers servers x (load + request + $OVERHEAD_S s overhead); this model's load time is not known here (DEPTH_SERVE_LOAD_EST=<s> prices it)"
  fi
}

# ds_fail <round> <index> <why>: the arm's FAIL line and the server's log tail, the server stopped, exit 1.
ds_fail() {
  echo "FAIL r$1 ${A_LABEL[$2]} D=${A_DEP[$2]}: $3${CPU_BUSY_TAG:-}${OTHER_BUSY_TAG:-}" >&2
  if [ -r "${SRV_LOG:-}" ]; then
    echo "--- the server's log ($SRV_LOG), last 20 lines" >&2
    tail -n 20 "$SRV_LOG" >&2
  fi
  lcpp_srv_stop
  exit 1
}

# ds_run_arm <round> <index>: one server, one request, its row (appended to ROWS_FILE) or a ds_fail.
ds_run_arm() {
  local r=$1 i=$2 label=${A_LABEL[$2]} dep=${A_DEP[$2]} ids t0 load tg pp maj listen ctx
  local -a envs
  CPU_BUSY_TAG='' OTHER_BUSY_TAG=''
  guard_other
  ds_guard_timing
  guard_cpu "pre r$r $label D=$dep"
  BIN_PATH=${A_BIN[$i]} BIN_SHA=${A_SHA[$i]} BIN_MTIME=${A_MTIME[$i]}
  ids=$(ds_ids "$dep")
  ds_server_cmd "$i"
  envs=()
  [ -z "${A_ENV[$i]}" ] || IFS=, read -r -a envs <<< "${A_ENV[$i]}"
  SRV_LOG=$LOGDIR/r$r-$(echo "$label" | tr -c 'A-Za-z0-9._\n-' _)-D$dep.log
  t0=$(date +%s)
  lcpp_srv_start "$LEASE_ARM_BOUND" "$SRV_LOG" -u BLOOMERY_REF_MODEL ${envs[@]+"${envs[@]}"} ||
    ds_fail "$r" "$i" "$SRV_WHY"
  load=$(($(date +%s) - t0))
  listen=$(grep -m1 'listening on http://' "$SRV_LOG")
  echo "[server] r$r $label D=$dep: $listen (loaded and listening in $load s)"
  if [ "$PORT" != 0 ] && [ "$SRV_PORT" != "$PORT" ]; then
    ds_fail "$r" "$i" "the server listens on port $SRV_PORT, not the --port $PORT it was given"
  fi
  ctx=$(sed -n 's/.* ctx=\([0-9][0-9]*\) .*/\1/p' <<< "$listen")
  [ -n "$ctx" ] || ds_fail "$r" "$i" "the listening line names no ctx=, so the context the arm ran at is not proven"
  [ "$ctx" = "$CTX" ] || ds_fail "$r" "$i" "the listening line names ctx=$ctx, not the --ctx-size $CTX the arm asked"
  witness "pre r$r $label D=$dep"
  if ! lcpp_srv_request "$ids" "$N" "$REQ_BOUND"; then
    witness "post r$r $label D=$dep (request failed)"
    ds_fail "$r" "$i" "$SRV_WHY"
  fi
  witness "post r$r $label D=$dep"
  guard_cpu "post r$r $label D=$dep"
  maj=$((SRV_F1 - SRV_F0))
  tg=$(awk -v v="$SRV_PRED_TPS" 'BEGIN { printf "%.2f", v }')
  pp=$(awk -v v="$SRV_PROMPT_TPS" 'BEGIN { printf "%.2f", v }')
  if ! awk -v v="$tg" 'BEGIN { exit !(v > 0) }'; then
    ds_fail "$r" "$i" "timings.predicted_per_second is '$SRV_PRED_TPS', not a rate"
  fi
  lcpp_srv_stop
  echo "ROW r$r $label D=$dep | tg_tok/s $tg | pp_tok/s $pp | load_s $load | majflt $maj${CPU_BUSY_TAG:-}${OTHER_BUSY_TAG:-}" | tee -a "$ROWS_FILE"
}

# ds_main <arm>...: the run (or, under BLOOMERY_DRY, the plan).
ds_main() {
  local i r k n v maxd need
  for v in $(compgen -v BLOOMERY_SERVE_); do
    ds_refuse 64 "$v is set, and nothing reads it: this runner's knobs are DEPTH_SERVE_CTX, DEPTH_SERVE_N, DEPTH_SERVE_PORT, DEPTH_SERVE_REQ_BOUND, DEPTH_SERVE_IDS and DEPTH_SERVE_LOAD_EST (a BLOOMERY_ name would have to be a row of the lever registry)"
  done
  DRY=${BLOOMERY_DRY:-}
  ROUNDS=${BLOOMERY_AB_ROUNDS:-4}
  N=${DEPTH_SERVE_N:-128}
  CTX=${DEPTH_SERVE_CTX:-4160}
  PORT=${DEPTH_SERVE_PORT:-18180}
  REQ_BOUND=${DEPTH_SERVE_REQ_BOUND:-300}
  ds_posint BLOOMERY_AB_ROUNDS "$ROUNDS" 1
  ds_posint DEPTH_SERVE_N "$N" 2
  ds_posint DEPTH_SERVE_CTX "$CTX" 1
  ds_posint DEPTH_SERVE_PORT "$PORT" 0
  [ "$PORT" -le 65535 ] || ds_refuse 64 "DEPTH_SERVE_PORT is a port, got $PORT"
  ds_posint DEPTH_SERVE_REQ_BOUND "$REQ_BOUND" 1
  ds_posint BLOOMERY_ARM_BOUND "$LEASE_ARM_BOUND" 1
  [ $# -ge 1 ] || ds_refuse 64 "no arms: <tree>:<D> names a tree under ~/repo and the prompt length in ids"
  ds_parse_arms "$@"
  n=${#A_LABEL[@]}
  maxd=0
  for ((i = 0; i < n; i++)); do
    [ "${A_DEP[i]}" -le "$maxd" ] || maxd=${A_DEP[i]}
    need=$((A_DEP[i] + N - 1))
    [ "$need" -le "$CTX" ] || arm_refuse "${A_LABEL[i]}:${A_DEP[i]}" "D ${A_DEP[i]} and n_predict $N take $need positions, past --ctx-size $CTX: bloomery-serve feeds the generated token k at position D + k - 1 and never the last, so D + n_predict - 1 must not pass the context (crates/serve/src/genloop.rs); lower DEPTH_SERVE_N or raise DEPTH_SERVE_CTX"
  done
  ds_ids_check "$maxd"
  [ -e "$MODEL" ] || ds_refuse 2 "no model file at $MODEL (the $MODEL_NAME profile's, BLOOMERY_REF_MODEL moves it)"
  ds_resolve_bins
  if [ -z "$DRY" ] && [ -z "$(lcpp_srv_majflt 2> /dev/null)" ]; then
    ds_refuse 2 "/proc/vmstat gives no pgmajfault: every row's majflt column would read 0"
  fi
  ds_print_plan
  if [ -n "$DRY" ]; then
    echo "[dry] nothing built, no lease taken, no server started"
    return 0
  fi
  [ -n "${BLOOMERY_CPU_BUSY_COMMS:-}" ] || CPU_BUSY_COMMS="$CPU_BUSY_COMMS bloomery-serve llama-server"
  trap lcpp_srv_stop EXIT
  trap 'exit 143' TERM
  trap 'exit 130' INT
  # ROUNDS (global above) is what lease_take checks an ab card against.
  lease_take
  LOGDIR=$(mktemp -d "${TMPDIR:-/tmp}/depth-serve.XXXXXX") || exit 2
  ROWS_FILE=$LOGDIR/rows.txt
  : > "$ROWS_FILE"
  echo "[config] model=$MODEL_NAME n=$N ctx=$CTX port=$PORT rounds=$ROUNDS arm_bound=${LEASE_ARM_BOUND}s req_bound=${REQ_BOUND}s timing_gpu=$TIMING_GPU other_gpu=$OTHER_GPU CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES logs=$LOGDIR"
  echo "[config] arms=$* (the server at the serving defaults beside --ctx-size, no warm-up request, one fresh server a row)"
  witness pre
  guard_other
  for ((r = 1; r <= ROUNDS; r++)); do
    for ((i = 0; i < n; i++)); do
      k=$(((i + r - 1) % n))
      ds_run_arm "$r" "$k"
    done
  done
  echo
  ds_tables "$ROWS_FILE"
  witness post
}

# ----------------------------------------------------------------------------------------------------------------
# `bash tools/ref/depth-serve.sh --self-test`: arm parsing, rotation, the refusals, the server stopped after each arm
# and the tables' math, on the stub server (python3, curl, 127.0.0.1), no lease, no card, no model. Each case runs in
# its own directory under one temporary root with its own HOME, data and TMPDIR; the stub logs its pid there, and the
# cleanup signals only those pids. ds_main runs in a subshell with the lease, the witness block and the guards
# replaced by markers (the real stale-binary rule, the real servers' start, request and stop stay).
ds_selftest() {
  # T is read by the EXIT trap, after this function has returned: not local.
  local n=0 fails=0 k C RC lever knob stub=$HERE/stub-llama-server.py
  command -v python3 > /dev/null || { echo "depth-serve.sh --self-test: python3 is not on PATH" >&2; exit 2; }
  T=$(mktemp -d "${TMPDIR:-/tmp}/depth-serve-self-test.XXXXXX") || exit 2
  st_cleanup() {
    local f p
    for f in "$T"/*/stub-srv-pids "$T"/*/listener-pid; do
      [ -r "$f" ] || continue
      while read -r p; do
        # Only a pid the case wrote and whose command is still the stub or the listener it started.
        if kill -0 "$p" 2> /dev/null && ps -o command= -p "$p" 2> /dev/null | grep -qE 'stub-llama-server|port-holder'; then kill "$p" 2> /dev/null; fi
      done < "$f"
    done
    rm -rf "$T"
  }
  trap st_cleanup EXIT
  trap 'exit 143' TERM
  trap 'exit 130' INT
  lever=$(lever_rows_py "$LEVER_REGISTRY" | awk '$2 == "Parsed" { print $1; exit }')
  [ -n "$lever" ] || { echo "depth-serve.sh --self-test: the lever registry has no parsed lever to set on an arm" >&2; exit 2; }

  ok() { n=$((n + 1)); echo "ok $1"; }
  bad() {
    n=$((n + 1)) fails=$((fails + 1))
    echo "FAIL $1: $2"
    [ ! -r "$C/out" ] || sed 's/^/    | /' "$C/out" | head -n 40
  }
  # st_case <name>: a fresh case directory with its home, a model file, 5,000 prose ids and (where there is none) a
  # `timeout` for the PATH.
  st_case() {
    C=$T/$1
    mkdir -p "$C/home/repo" "$C/data/mimo2" "$C/bin"
    : > "$C/model.gguf"
    seq 1 5000 > "$C/data/mimo2/corpus-prose.ids"
    if ! command -v timeout > /dev/null; then
      printf '#!/bin/sh\ncase $1 in --kill-after=*) shift ;; esac\nshift\nexec "$@"\n' > "$C/bin/timeout"
      chmod +x "$C/bin/timeout"
    fi
  }
  # st_tree <name> <tps> [NAME=VALUE...]: ~/repo/<name> with Cargo files and a binary newer than them, a wrapper that
  # dumps its environment and runs the stub at that decode rate, listing `ctx=` as bloomery-serve's record does.
  st_tree() {
    local name=$1 d=$C/home/repo/$1 tps=$2
    shift 2
    mkdir -p "$d/target/release"
    : > "$d/Cargo.toml"
    : > "$d/Cargo.lock"
    touch -t 202601010000 "$d/Cargo.toml" "$d/Cargo.lock"
    printf '#!/bin/sh\nenv > "%s/env.%s.$$"\nexec env STUB_SRV_TPS=%s STUB_SRV_LISTEN_CTX=argv %s python3 "%s" "$@"\n' "$C" "$name" "$tps" "$*" "$stub" > "$d/$SERVE_BIN"
    chmod +x "$d/$SERVE_BIN"
    touch -t 202601020000 "$d/$SERVE_BIN"
  }
  st_port() { python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])'; }
  # st_run <arg>...: ds_main in a subshell, its output in $C/out and its rc in RC. Extra variables go through ST_ENV.
  ST_ENV=() ST_PORT='' ST_BOUND=''
  st_run() {
    ( export HOME=$C/home TMPDIR=$C BLOOMERY_REF_MODEL=$C/model.gguf PATH=$C/bin:$PATH
      MODEL=$C/model.gguf BLOOMERY_DATA=$C/data
      export DEPTH_SERVE_PORT=${ST_PORT:-$(st_port)}
      # shellcheck disable=SC2163 # kv is a NAME=VALUE string
      for kv in ${ST_ENV[@]+"${ST_ENV[@]}"}; do export "$kv"; done
      [ -z "${ST_BOUND:-}" ] || LEASE_ARM_BOUND=$ST_BOUND
      lease_take() { echo "[lease-stub] taken"; }
      witness() { echo "[witness-stub] $1"; }
      guard_other() { :; }
      ds_guard_timing() { :; }
      guard_cpu() { :; }
      lcpp_srv_majflt() { echo 7; }
      ds_main "$@"
    ) > "$C/out" 2>&1
    RC=$?
  }
  # st_check <name> <rc> <fixed string>...: the last run's rc, and every string in its output.
  st_check() {
    local name=$1 want=$2 s
    shift 2
    if [ "$RC" != "$want" ]; then bad "$name" "rc $RC, want $want"; return; fi
    for s in "$@"; do
      grep -qF -- "$s" "$C/out" || { bad "$name" "no line holds '$s'"; return; }
    done
    ok "$name"
  }
  # st_absent <name> <fixed string>...: the last run's output holds none of them.
  st_absent() {
    local name=$1 s
    shift
    for s in "$@"; do
      if grep -qF -- "$s" "$C/out"; then bad "$name" "a line holds '$s'"; return; fi
    done
    ok "$name"
  }
  # st_servers <name> <count>: the stub started <count> times and every pid it logged is gone.
  st_servers() {
    local p alive=0 count=0
    if [ -r "$C/stub-srv-pids" ]; then
      while read -r p; do
        count=$((count + 1))
        ! kill -0 "$p" 2> /dev/null || alive=$((alive + 1))
      done < "$C/stub-srv-pids"
    fi
    if [ "$count" != "$2" ]; then bad "$1" "$count servers started, want $2"; return; fi
    if [ "$alive" != 0 ]; then bad "$1" "$alive of $count servers still up after the run"; return; fi
    ok "$1"
  }

  # --- the plan: parsing, rotation, a dry run that starts nothing.
  st_case dry
  st_tree alpha 20
  st_tree beta 21
  ST_ENV=(BLOOMERY_DRY=1 BLOOMERY_AB_ROUNDS=2)
  st_run alpha:6 "env=$lever=1:beta:1024"
  st_check dry-plan 0 "[plan] arm 1: alpha D=6:" "[plan] arm 2: beta@$lever=1 D=1024:" \
    "[plan] order r1: alpha:6 beta@$lever=1:1024" "[plan] order r2: beta@$lever=1:1024 alpha:6" \
    "env -u BLOOMERY_REF_MODEL $lever=1 $C/home/repo/beta/$SERVE_BIN -m $C/model.gguf --host 127.0.0.1 --port " \
    "--ctx-size 4160" "[plan] servers=4 (2 arms x 2 rounds" "[dry] nothing built, no lease taken, no server started"
  st_absent dry-takes-no-lease "[lease-stub]" "[witness-stub]"
  [ ! -e "$C/stub-srv-pids" ] && ok dry-starts-no-server || bad dry-starts-no-server "the stub was started"
  rm -rf "$C/home/repo/alpha/target"
  st_run alpha:6
  st_check dry-names-a-missing-binary 0 "no binary at $C/home/repo/alpha/$SERVE_BIN" "[dry] tree alpha: $C/home/repo/alpha/$SERVE_BIN would be refused (rc 2)"
  ST_ENV=()

  # --- the refusals before the lease.
  st_case refuse
  st_tree alpha 20
  st_run
  st_check refuse-no-arms 64 "no arms"
  st_run alpha
  st_check refuse-arm-syntax 64 "want [env=K=V[,K=V]:]<tree>:<D>"
  st_run alpha:0
  st_check refuse-arm-zero-length 64 "'0' is no prompt length"
  st_run alpha:x
  st_check refuse-arm-length 64 "'x' is no prompt length"
  st_run ../alpha:6
  st_check refuse-arm-tree 64 "'../alpha' is no tree name"
  st_run env=:alpha:6
  st_check refuse-arm-empty-env 64 "an empty env= list"
  st_run env=NOT_A_LEVER_AT_ALL=1:alpha:6
  st_check refuse-arm-env-no-lever 64 "NOT_A_LEVER_AT_ALL is no row of the lever registry"
  st_run alpha:6 alpha:6
  st_check refuse-arm-twice 64 "listed twice"
  st_run alpha:4160
  st_check refuse-context 64 "take 4287 positions, past --ctx-size 4160"
  ST_ENV=(BLOOMERY_DRY=1)
  st_run alpha:4033
  st_check refuse-context-edge 0 "[plan] arm 1: alpha D=4033:"
  # The spelling the registry would need a row for, built so that this file names no such variable.
  knob=BLOOMERY_SERVE_
  ST_ENV=("${knob}CTX=4160")
  st_run alpha:6
  st_check refuse-prefixed-knob 64 "${knob}CTX is set, and nothing reads it" "DEPTH_SERVE_CTX"
  ST_ENV=()
  ST_ENV=(DEPTH_SERVE_CTX=7000)
  st_run alpha:6000
  st_check refuse-ids-short 64 "a prompt of 6000 ids; $C/data/mimo2/corpus-prose.ids holds 5000"
  ST_ENV=()
  st_run nosuch:6
  st_check refuse-no-tree 2 "arm 'nosuch': no tree $C/home/repo/nosuch"
  mkdir -p "$C/home/repo/bare"
  : > "$C/home/repo/bare/Cargo.toml"
  st_run bare:6
  st_check refuse-no-binary 2 "no binary at $C/home/repo/bare/$SERVE_BIN"
  touch -t 202601030000 "$C/home/repo/alpha/Cargo.lock"
  st_run alpha:6
  st_check refuse-stale-binary 3 "[stale-binary] $C/home/repo/alpha/$SERVE_BIN" "is older than its sources"
  st_absent refuse-stale-binary-before-the-lease "[lease-stub]"
  [ ! -e "$C/stub-srv-pids" ] && ok refuse-stale-binary-starts-no-server || bad refuse-stale-binary-starts-no-server "the stub was started"
  touch -t 202601010000 "$C/home/repo/alpha/Cargo.lock"
  rm -f "$C/data/mimo2/corpus-prose.ids"
  st_run alpha:6
  st_check refuse-ids-missing 2 "no prose ids at $C/data/mimo2/corpus-prose.ids"
  seq 1 5000 > "$C/data/mimo2/corpus-prose.ids"
  sed -i.bak '3s/.*/x3/' "$C/data/mimo2/corpus-prose.ids"
  st_run alpha:6
  st_check refuse-ids-not-an-id 2 "line 3 of the first 6 of $C/data/mimo2/corpus-prose.ids ('x3') is no id"
  seq 1 5000 > "$C/data/mimo2/corpus-prose.ids"
  rm -f "$C/model.gguf"
  st_run alpha:6
  st_check refuse-no-model-file 2 "no model file at $C/model.gguf"
  RC=0
  env -u BLOOMERY_MODEL -u BLOOMERY_REF_MODEL_PROFILE bash "$HERE/depth-serve.sh" alpha:6 > "$C/out" 2>&1 || RC=$?
  st_check refuse-no-model 64 "no model: BLOOMERY_MODEL names the profile"

  # --- a run: the order, the one lease, the witness blocks around every request, the fixed port, the servers stopped.
  st_case rotate
  st_tree a 20
  st_tree b 21
  st_tree c 22
  ST_PORT=$(st_port)
  ST_ENV=(BLOOMERY_AB_ROUNDS=3)
  st_run a:6 b:6 c:6
  st_check rotate-run 0 "[lease-stub] taken" "[config] model=mimo2 n=128 ctx=4160 port=$ST_PORT rounds=3"
  got=$(grep '^ROW' "$C/out" | awk '{ printf "%s%s ", $2, $3 }')
  [ "$got" = "r1a r1b r1c r2b r2c r2a r3c r3a r3b " ] && ok rotate-order || bad rotate-order "rows in order [$got]"
  st_servers rotate-servers-stopped 9
  [ "$(grep -c -- "--port $ST_PORT --ctx-size 4160" "$C/stub-srv-argv")" = 9 ] && ok rotate-fixed-port-and-ctx || bad rotate-fixed-port-and-ctx "the stub's argv lacks --port $ST_PORT --ctx-size 4160 nine times"
  [ "$(grep -c '^\[lease-stub\]' "$C/out")" = 1 ] && ok rotate-one-lease || bad rotate-one-lease "the lease was not taken once"
  seq=$(grep -E '^\[lease-stub\]|^\[witness-stub\] (pre r|post r)|^ROW' "$C/out" | head -n 4 | sed 's/ | pp_tok.*//' | tr '\n' '|')
  [ "$seq" = "[lease-stub] taken|[witness-stub] pre r1 a D=6|[witness-stub] post r1 a D=6|ROW r1 a D=6 | tg_tok/s 20.00|" ] &&
    ok rotate-witness-around-the-request || bad rotate-witness-around-the-request "got [$seq]"
  # shellcheck disable=SC2144 # a glob of at most 9 files, read by grep -l
  if grep -lq '^BLOOMERY_REF_MODEL=' "$C"/env.* 2> /dev/null; then bad rotate-server-env "a server saw BLOOMERY_REF_MODEL"; else ok rotate-server-env; fi
  st_check rotate-tables 0 "mean a D=6 | tg 20.00 tok/s (sd 0.00 = 0.00 %, [20.00..20.00], n=3) | pp 60.00 tok/s" \
    "diff b vs a D=6 | tg +1.00 tok/s (+5.00 %) ± 0.00 tok/s (± 0.00 %) [clears 0] | pp +0.00 tok/s (+0.00 %) ± 0.00 tok/s (± 0.00 %) [holds 0]" \
    "diff c vs a D=6 | tg +2.00 tok/s (+10.00 %)"
  ST_PORT=
  ST_ENV=()

  # --- a lever arm: its variables reach that arm's servers only.
  st_case envarm
  st_tree a 20
  ST_ENV=(BLOOMERY_AB_ROUNDS=1)
  st_run a:6 "env=$lever=1:a:6"
  st_check envarm-run 0 "ROW r1 a D=6" "ROW r1 a@$lever=1 D=6"
  # shellcheck disable=SC2012 # the dumps are named by the wrapper
  [ "$(grep -l "^$lever=1\$" "$C"/env.a.* 2> /dev/null | wc -l | tr -d ' ')" = 1 ] && ok envarm-env-reaches-one-arm || bad envarm-env-reaches-one-arm "$lever=1 was not in exactly one server's environment"
  st_servers envarm-servers-stopped 2
  ST_ENV=()

  # --- a server's failure under the lease: named, the server stopped, no row.
  st_case notimings
  st_tree nt 20 STUB_SRV_NOTIMINGS=1
  ST_ENV=(BLOOMERY_AB_ROUNDS=1)
  st_run nt:6
  st_check refuse-missing-timings 1 "FAIL r1 nt D=6: no timings in the /completion response"
  st_absent refuse-missing-timings-no-row "ROW "
  st_servers refuse-missing-timings-server-stopped 1
  st_case nohealth
  st_tree nh 20 STUB_SRV_HANG=1
  ST_ENV=(BLOOMERY_AB_ROUNDS=1)
  ST_BOUND=3
  st_run nh:6
  ST_BOUND=
  st_check refuse-no-health 1 "FAIL r1 nh D=6: llama-server did not answer /health within 3 s"
  st_servers refuse-no-health-server-stopped 1
  st_case ctxmismatch
  st_tree cm 20 STUB_SRV_LISTEN_CTX=1234
  ST_ENV=(BLOOMERY_AB_ROUNDS=1)
  st_run cm:6
  st_check refuse-ctx-mismatch 1 "the listening line names ctx=1234, not the --ctx-size 4160"
  st_servers refuse-ctx-mismatch-server-stopped 1
  st_case noctx
  st_tree nc 20
  sed -i.bak 's/STUB_SRV_LISTEN_CTX=argv //' "$C/home/repo/nc/$SERVE_BIN"
  touch -t 202601020000 "$C/home/repo/nc/$SERVE_BIN"
  ST_ENV=(BLOOMERY_AB_ROUNDS=1)
  st_run nc:6
  st_check refuse-no-ctx-word 1 "the listening line names no ctx="
  st_case busyport
  st_tree bp 20
  ST_PORT=$(st_port)
  python3 -c 'import os, socket, sys, time
s = socket.socket()
s.bind(("127.0.0.1", int(sys.argv[1])))
s.listen()
open(sys.argv[2], "w").write(str(os.getpid()) + "\n")
time.sleep(60)  # port-holder' "$ST_PORT" "$C/listener-pid" &
  holder=$!
  # The holder writes its pid once it listens: a loaded host starts python slowly, so wait up to 30 s.
  for ((k = 0; k < 100; k++)); do [ -s "$C/listener-pid" ] && break; sleep 0.3; done
  if [ -s "$C/listener-pid" ]; then
    ST_ENV=(BLOOMERY_AB_ROUNDS=1)
    st_run bp:6
    st_check refuse-busy-port 1 "FAIL r1 bp D=6: llama-server exited 1 before it answered /health" "Address already in use"
  else
    : > "$C/out"
    bad refuse-busy-port "the port holder did not listen within 30 s"
  fi
  kill "$holder" 2> /dev/null
  wait "$holder" 2> /dev/null
  ST_PORT=
  ST_ENV=()

  # --- the tables' math, against card.py's half_width and by hand (a: 100 102 98 100, b: 101 103 99 101 over 4 rounds;
  # sd 1.633 each, so h = t(0.975, 6) 2.447 x 1.633 x sqrt(2/4) = 2.826 tok/s).
  st_case tables
  : > "$C/rows"
  r=0
  for tg in "100 101" "102 103" "98 99" "100 101"; do
    r=$((r + 1))
    set -- $tg
    echo "ROW r$r a D=6 | tg_tok/s $1.00 | pp_tok/s 60.00 | load_s 41 | majflt 3" >> "$C/rows"
    echo "ROW r$r b D=6 | tg_tok/s $2.00 | pp_tok/s 60.00 | load_s 43 | majflt 5 [cpu-busy]" >> "$C/rows"
  done
  echo "ROW r1 b D=1024 | tg_tok/s 18.00 | pp_tok/s 300.00 | load_s 41 | majflt 0" >> "$C/rows"
  python3 -I - "$CARD_PY" > "$C/want" << 'PY'
import importlib.util
import math
import statistics
import sys

spec = importlib.util.spec_from_file_location('card', sys.argv[1])
card = importlib.util.module_from_spec(spec)
spec.loader.exec_module(card)
a, b = [100, 102, 98, 100], [101, 103, 99, 101]
sp = math.sqrt((statistics.stdev(a) ** 2 + statistics.stdev(b) ** 2) / 2)
h = card.half_width(card.t_table(), sp, len(a))[0]
d = statistics.mean(b) - statistics.mean(a)
print(f'{d:+.2f} tok/s ({100 * d / statistics.mean(a):+.2f} %) ± {h:.2f} tok/s (± {100 * h / statistics.mean(a):.2f} %)')
PY
  want=$(cat "$C/want")
  ds_tables "$C/rows" > "$C/out" 2>&1
  RC=$?
  st_check tables-against-card-py 0 "diff b vs a D=6 | tg $want [holds 0]"
  st_check tables-by-hand 0 "diff b vs a D=6 | tg +1.00 tok/s (+1.00 %) ± 2.83 tok/s (± 2.83 %) [holds 0]" \
    "mean a D=6 | tg 100.00 tok/s (sd 1.63 = 1.63 %, [98.00..102.00], n=4)" "[cpu-busy 4/4] [other-busy 0/4]" "majflt 20"
  st_absent tables-no-cross-depth-diff "diff b vs a D=1024"
  RC=0
  bash "$HERE/depth-serve.sh" --summary "$C/rows" > "$C/out" 2>&1 || RC=$?
  st_check summary-mode 0 "diff b vs a D=6 | tg $want [holds 0]"
  echo "ROW r1 a D=6 | tg_tok/s 20.00 | pp_tok/s 60.00 | load_s 41 | majflt 0" > "$C/rows"
  echo "ROW r1 b D=6 | tg_tok/s 21.00 | pp_tok/s 60.00 | load_s 41 | majflt 0" >> "$C/rows"
  echo "ROW r2 b D=6 | tg_tok/s 21.00 | pp_tok/s 60.00 | load_s 41 | majflt 0" >> "$C/rows"
  RC=0
  bash "$HERE/depth-serve.sh" --summary "$C/rows" > "$C/out" 2>&1 || RC=$?
  st_check tables-unequal-rounds 0 "tg +1.00 tok/s (+5.00 %), no interval: 2 rounds against 1"
  head -n 2 "$C/rows" > "$C/rows1"
  RC=0
  bash "$HERE/depth-serve.sh" --summary "$C/rows1" > "$C/out" 2>&1 || RC=$?
  st_check tables-one-round 0 "tg +1.00 tok/s (+5.00 %), one round: no interval"
  echo "no rows here" > "$C/rows"
  RC=0
  bash "$HERE/depth-serve.sh" --summary "$C/rows" > "$C/out" 2>&1 || RC=$?
  st_check summary-no-rows 1 "no ROW lines to summarize"

  if [ "$fails" = 0 ]; then
    echo "depth-serve.sh self-test: $n checks ok"
  else
    echo "depth-serve.sh self-test: $fails of $n checks FAILED"
  fi
  [ "$fails" = 0 ]
}

case $MODE in
  selftest) ds_selftest ;;
  summary) ds_summary_mode "$@" ;;
  *) ds_main "$@" ;;
esac
