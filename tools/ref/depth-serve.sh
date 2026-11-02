#!/usr/bin/env bash
# Decode and prompt rate of any model through `bloomery-serve`, one server an arm and a round, under the timing
# lease (runs on the box, or on a contributor's machine under tools/box.sh's local mode).
#
#   BLOOMERY_MODEL=mimo2 just depth-serve <tree>:<D> [<tree>:<D> | env=K=V[,K=V]:<tree>:<D>] ...
#   BLOOMERY_MODEL=minimax-m3 DEPTH_SERVE_TARBALL=<release tarball> just depth-serve <tree>@portable:<D> ...
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
# ~/repo, as ab-decode.sh's) and `<tree>@portable:<D>` the release build's, ~/repo/<tree>/target/portable/release/
# bloomery-serve (tools/release/build.sh builds it into its own target dir and packs it): the release's headline is
# timed on the binary the tarball holds, so a @portable arm needs DEPTH_SERVE_TARBALL and its sha256 must equal the
# tarball member's (below). Both name the first <D> ids of the model's prose corpus as the prompt
# ($BLOOMERY_DATA/<profile name>/corpus-prose.ids, one id a line; DEPTH_SERVE_IDS names another file). `env=K=V,K=V:`
# before an arm sets those levers for that arm's server only (lever-arms.sh checks them against the lever registry,
# as the other depth runners do); the variables show in the arm's row label. The model is BLOOMERY_MODEL's profile
# (tools/ref/models/<name>.sh): its file is `-m`, and the seat is the file's architecture's.
#
# Per arm and round, one fresh server (the load is the price of a round and is outside every timed number):
#   0. the card's used memory back within DEPTH_SERVE_MEM_SLACK_MIB of the sitting's baseline (read once, after the lease
#      is taken): a server's memory is released after its process is gone, and a load that starts before that finds
#      the card smaller than the sitting's plan (waits DEPTH_SERVE_MEM_WAIT_S, then the arm fails by name);
#   1. the server at the serving defaults — `-m <file> --host 127.0.0.1 --port P`, no --ctx-size (unless
#      DEPTH_SERVE_CTX names one), no --place (unless DEPTH_SERVE_PLACE does), no --parallel, no lever — under
#      `timeout` (BLOOMERY_ARM_BOUND, default 900 s, covers its load, request and hook), BLOOMERY_REF_MODEL removed
#      from its environment as gate_mimo2_serve does (the file is named by -m);
#   2. its `listening on http://127.0.0.1:P` line and a 200 from /health, which the seat prints once the model is
#      loaded; the line's ctx must be DEPTH_SERVE_CTX when that is set, else the arm fails by name;
#   2b. the `plan` record of its log (tools/bloomery/records.py, kind plan or plan38): the placement and the context
#      the plan picked go into the row's label, and the record's decisions — every field but the host and card
#      readings (room, plan_room, card_free, read) — must equal those of the first load of the arm's group (the
#      arms of one label, whatever their D), else the arm fails `plan drifted`. A first load's plan must also be
#      the one `<binary> --plan` prints in the user's environment (no CUDA_VISIBLE_DEVICES), probed under the lease
#      before round 1: the timed key is then the user's key (DEPTH_SERVE_USER_PLAN);
#   3. one /completion of the D ids, temperature 0, `n_predict` N, ignore_eos, cache_prompt off (lcpp-warm.sh's
#      lcpp_srv_request: it fails on a response with no `timings`, a prompt_n other than D, a cache_n above 0 and a
#      predicted_n other than N), the witness blocks and the major-fault count (pgmajfault) around it;
#   3b. DEPTH_SERVE_AFTER, on round 1's last server of each arm group, after its timed request (below);
#   4. TERM to the server's captured pid and a wait for its exit (KILL after 30 s).
# No warm-up request is sent: every arm's single request meets the same first-request state of its fresh server, and
# the load-to-load scatter is in the ruler. The rates are llama-server's (lcpp-warm.sh's header has the derivation):
# tg is timings.predicted_per_second, the N - 1 decode steps over their own wall, pp is timings.prompt_per_second,
# the D ids over the prompt's wall; neither clock holds the HTTP exchange or the tokenization.
#
# Context. bloomery-serve feeds the generated token k at position D + k - 1 and never feeds the last one, so a
# request stays inside the context when D + N - 1 <= C (the `n + len > ctx_max` stop in crates/serve/src/genloop.rs).
# DEPTH_SERVE_CTX unset is the serving default: no --ctx-size, the unset rule picks the context (the largest step
# under the model's limit that the machine holds). Its pick is the `ctx_max` of the plan record and is in every row's
# label; the context a request must fit is the one the listening line names, so an arm past it fails by name after the
# load, before its request. With DEPTH_SERVE_CTX set (a deliberate row), an arm past it is refused before the lease:
# at C 4160 and N 128 the longest D is 4033, and D 4096 at C 4160 takes DEPTH_SERVE_N of at most 65.
#
# Rows and tables. One row a server: `ROW r<k> <label> D=<D> | tg_tok/s | pp_tok/s | load_s | majflt`, the label the
# tree (`@portable`, and `@K=V,...`) and the plan's own picks, `/place=<word>/ctx=<ctx_max>`, a ` [cpu-busy]` or
# ` [other-busy]` tag when the guards saw a co-tenant, and ` [cold]` when the faults of its request could have cost
# COLD_PCT percent of its window or more (cold-blocks.sh's cold_check: the machine-wide fault count over the request,
# at COLD_US a fault, against prompt_ms + predicted_ms); a [cold] row is in the log and out of every mean. Then per
# arm the means over its warm rows (a `[cold k/n]` count ends the line, and an arm with no warm row prints no mean),
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
#   DEPTH_SERVE_CTX          the server's --ctx-size C, default unset (the serving default: the plan picks the context)
#   DEPTH_SERVE_PLACE        the server's --place, `a` or `bp`, default unset (the placement rule's own pick); the plan
#                            record's place is in the row's label whichever way it was chosen
#   DEPTH_SERVE_N            n_predict N, at least 2 (tg needs a step), default 128
#   DEPTH_SERVE_PORT         the loopback port, default 18180; the server's own listening line must name it, so a
#                            foreign process on it makes the server fail to bind and the arm fail by name; 0 is the
#                            kernel's pick
#   DEPTH_SERVE_REQ_BOUND    seconds the request may take, default 300
#   DEPTH_SERVE_IDS          the prose ids file, default $BLOOMERY_DATA/<profile name>/corpus-prose.ids
#   DEPTH_SERVE_LOAD_EST     seconds one load takes, for the predicted wall line (mimo2: 41, the lead's figure)
#   DEPTH_SERVE_TARBALL      the release tarball (`bloomery-<v>-<platform>.tar.gz`, tools/release/build.sh's) whose
#                            `<name>/bin/bloomery-serve` every @portable arm's binary must equal by sha256, read before
#                            the lease; the witness blocks' `binary:` line names the tarball and the sha256. Required
#                            by a @portable arm, refused beside none
#   DEPTH_SERVE_AFTER        a command line (bash -c) run under the lease (lease_bounded) on round 1's last server of
#                            each arm group — the arms of one label, whatever their D — after that server's timed request
#                            and before it stops, with BLOOMERY_SESSION_URL (http://127.0.0.1:<port>), _PID (the pid of
#                            the server's `timeout`, alive while the server is), _STDERR (the server's log) and _DIR (a
#                            directory of its own) in its environment: the functional checks of a release sitting.
#                            Its lines go to this log between the request and the row,
#                            `HOOK r<k> <label> D=<D> rc=<n>` ends them, a hook that exits other than 0 makes the run's
#                            exit status 1 (the rounds still run). A hook never runs before its server's timed request:
#                            ds_hook_run refuses (70)
#   DEPTH_SERVE_AFTER_BOUND  seconds the hook may take, default 300, under BLOOMERY_ARM_BOUND (the server's whole life)
#   DEPTH_SERVE_USER_PLAN    `check` (default): the first load of an arm group must print the plan its `--plan` printed
#                            in the user's environment, else the arm fails; `note`: the difference is printed
#                            (`[user-plan]`) and the run goes on, for a card set the user's environment does not see
#                            as the timing environment does (the 3090 alone beside an A6000); `off`: no probe
#   DEPTH_SERVE_MEM_WAIT_S   seconds the card's used memory may take to return to the sitting's baseline before a
#                            server starts, default 60
#   DEPTH_SERVE_MEM_SLACK_MIB  MiB above the baseline that still count as returned, default 128
#
# Refused before the lease, by name: no model, an arm that does not parse, a tree or its binary missing (rc 2), a
# binary older than the Cargo files and sources its dep-info lists (rc 3, timing-card.sh's assert_fresh_binary; a
# @portable arm's is the portable build's own dep-info), a @portable arm without DEPTH_SERVE_TARBALL or a tarball no
# arm reads, a tarball that is missing or holds no member (rc 2), a @portable binary that is not the tarball's member
# (rc 65), ids missing or short, a DEPTH_SERVE_CTX that does not hold D + N - 1, a place or a mode that is none of
# the words above. After the lease an arm fails (rc 1, the server stopped, its log's tail printed) on a server that
# exits or does not answer /health within its bound, a listening line with another port or ctx (or, DEPTH_SERVE_CTX
# unset, a context that does not hold D + N - 1), a log with no plan record, a plan that drifted from the group's
# first, a user-environment plan that is not the timing plan, a card whose memory did not return, or a request that
# fails (no timings among them). Contention is the lease's and the guards' (rc 75). BLOOMERY_DRY=1 prints the plan (arms, order, commands, predicted wall) and the binaries' state, takes no
# lease and starts nothing; the recipe builds nothing under it.
#
# Local mode. Under tools/box.sh's local mode (BLOOMERY_BOX_CARD=local; timing-card.sh's header has its refusals) the
# arms time this machine's one card. Before the lease a kernel journal that does not answer, or shows a user other than
# root no line, refuses the run (69); after every arm's server has stopped, an Xid since the last arm's check or the card
# lost fails the arm, and a failed arm's line names them beside its own reason. The recipe runs in the contributor's tree
# and builds it for an arm of that tree's name, so such an arm whose ~/repo/<tree> is another directory, or none, is
# refused before the lease (64).
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
    BLOOMERY_TIMING_GPU BLOOMERY_DRY BLOOMERY_AB_ROUNDS BLOOMERY_ARM_BOUND BLOOMERY_LEASE_CARD BLOOMERY_CPU_BUSY_COMMS \
    BLOOMERY_BOX_CARD
  # shellcheck disable=SC2046 # the names are words
  unset $(compgen -v DEPTH_SERVE_)
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
  # The timing card, the stale-binary rule, the guards and, through it, the lease and the witness block; this runner
  # has local mode (the header's Local mode) and the two-card mode (BLOOMERY_TIMING_CARDS=a6000+3090: both cards in
  # the server's view, each arm's Xid, cap and answer checked by timing_cards_arm).
  TIMING_CARDS_RUNNER=1 TIMING_LOCAL_RUNNER=1
  # shellcheck source=tools/ref/timing-card.sh
  source "$HERE/timing-card.sh"
  # The server arms' start, request and stop (the llama-server runners' functions; the server here is bloomery-serve).
  # shellcheck source=tools/ref/lcpp-warm.sh
  source "$HERE/lcpp-warm.sh"
  # The cold tag's rule (COLD_US, COLD_PCT, cold_check): one copy, the depth runners'.
  # shellcheck source=tools/ref/cold-blocks.sh
  source "$HERE/cold-blocks.sh"
  # shellcheck source=tools/ref/lever-arms.sh
  source "$HERE/lever-arms.sh" || exit 2
  # shellcheck source=tools/ref/arm-place.sh
  source "$HERE/arm-place.sh" || exit 2
fi

SERVE_BIN=target/release/bloomery-serve
# The release build's binary (tools/release/build.sh builds into its own target dir), the one a tarball packs.
PORTABLE_BIN=target/portable/release/bloomery-serve
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

def warm_rows(rs):
    # A [cold] row's window paid for its page faults: it stays in the log and leaves every mean and difference.
    return [x for x in rs if '[cold]' not in x['tags']]


print('=== per arm (tok/s): tg = timings.predicted_per_second, pp = timings.prompt_per_second; one fresh server a row; [cold] rows out of the means ===')
for label, dep in order:
    rs = arms[(label, dep)]
    warm = warm_rows(rs)
    parts = []
    for name in ('tg', 'pp'):
        xs = [x[name] for x in warm]
        if not xs:
            continue
        m = statistics.mean(xs)
        if len(xs) > 1:
            s = statistics.stdev(xs)
            parts.append(f'{name} {m:.2f} tok/s (sd {s:.2f} = {100 * s / m:.2f} %, [{min(xs):.2f}..{max(xs):.2f}], n={len(xs)})')
        else:
            parts.append(f'{name} {m:.2f} tok/s (n=1)')
    n = len(rs)
    cpu = sum('[cpu-busy]' in x['tags'] for x in rs)
    other = sum('[other-busy]' in x['tags'] for x in rs)
    cold = n - len(warm)
    load = statistics.mean(x['load'] for x in rs)
    maj = sum(x['maj'] for x in rs)
    if not warm:
        parts = [f'no warm row: every one of its {n} row(s) is [cold]']
    print(f'mean {label} D={dep} | ' + ' | '.join(parts) + f' | load {load:.1f} s | majflt {maj} | [cpu-busy {cpu}/{n}] [other-busy {other}/{n}] [cold {cold}/{n}]')

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
    if not warm_rows(arms[first]) or not warm_rows(arms[key]):
        none = first[0] if not warm_rows(arms[first]) else label
        print(f'diff {label} vs {first[0]} D={dep} | no diff: {none} has no warm row')
        continue
    for name in ('tg', 'pp'):
        b = [x[name] for x in warm_rows(arms[first])]
        a = [x[name] for x in warm_rows(arms[key])]
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

# ds_parse_arms <arm>...: A_TREE, A_KIND (tree or portable), A_DEP, A_ENV (the env= list or empty), A_LABEL (the tree,
# `@portable` and `@list`), A_GROUP (the first arm of the same label: the arm group) and A_HOOK (1 for the arm that is
# its group's last server of round 1, which runs in index order), each arm's refusals by name.
ds_parse_arms() {
  local a rest envs tree dep label kind i j seen=' '
  local -a last
  A_TREE=() A_KIND=() A_DEP=() A_ENV=() A_LABEL=() A_GROUP=() A_HOOK=()
  last=()
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
    case $rest in *:*) ;; *) arm_refuse "$a" "want [env=K=V[,K=V]:]<tree>:<D> or <tree>@portable:<D>" ;; esac
    tree=${rest%%:*} dep=${rest#*:} kind=tree
    case $tree in
      *@portable) kind=portable tree=${tree%@portable} ;;
      *@*) arm_refuse "$a" "'${tree#*@}' after '@' is no binary kind (@portable is the release build, target/portable; a tree alone is its target/release)" ;;
    esac
    [[ $tree =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || arm_refuse "$a" "'$tree' is no tree name (a directory under ~/repo: letters, digits, '.', '_', '-')"
    [[ $dep =~ ^[1-9][0-9]*$ ]] || arm_refuse "$a" "'$dep' is no prompt length (a whole number of ids, at least 1)"
    [ -z "$envs" ] || arm_envs_ok "$a" "$envs"
    label=$tree
    [ "$kind" != portable ] || label+=@portable
    label+=${envs:+@$envs}
    case $seen in *" $label:$dep "*) arm_refuse "$a" "listed twice (an A/A needs a second tree or an env= list)" ;; esac
    seen+="$label:$dep "
    A_TREE+=("$tree") A_KIND+=("$kind") A_DEP+=("$dep") A_ENV+=("$envs") A_LABEL+=("$label")
  done
  for ((i = 0; i < ${#A_LABEL[@]}; i++)); do
    for ((j = 0; j < i; j++)); do
      [ "${A_LABEL[j]}" != "${A_LABEL[i]}" ] || break
    done
    A_GROUP+=("$j") A_HOOK+=(0)
    last[j]=$i
  done
  for j in ${last[@]+"${last[@]}"}; do A_HOOK[j]=1; done
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

# ds_sha256 <file>: the file's sha256, empty when it cannot be read.
ds_sha256() { sha256sum "$1" 2> /dev/null | cut -d' ' -f1; }

# ds_member_sha <tarball> <member>: the sha256 of the member's bytes; 1 when the tarball holds none.
ds_member_sha() {
  local out
  out=$(tar -xzOf "$1" "$2" 2> /dev/null | sha256sum) || return 1
  out=${out%% *}
  [ -n "$out" ] || return 1
  echo "$out"
}

# ds_tarball_refuse <rc> <why>: a run refuses before the lease; a dry run prints the refusal and goes on.
ds_tarball_refuse() {
  [ -n "$DRY" ] || ds_refuse "$1" "$2"
  echo "[dry] $2 (a run refuses, rc $1)"
}

# ds_tarball_resolve: TB (DEPTH_SERVE_TARBALL), TB_MEMBER (`<name>/bin/bloomery-serve`, build.sh's layout) and TB_SHA (the
# member's sha256, empty when the tarball was refused in a dry run). A @portable arm needs the tarball, and a tarball no
# arm reads is a knob that checks nothing.
ds_tarball_resolve() {
  local i first='' base
  TB=${DEPTH_SERVE_TARBALL:-} TB_MEMBER='' TB_SHA=''
  for ((i = 0; i < ${#A_LABEL[@]}; i++)); do
    [ "${A_KIND[i]}" != portable ] || { first=${A_LABEL[i]}; break; }
  done
  if [ -z "$first" ]; then
    [ -z "$TB" ] || ds_refuse 64 "DEPTH_SERVE_TARBALL is set, and no arm is @portable: nothing is checked against it (a tree arm is not the release's binary)"
    return 0
  fi
  [ -n "$TB" ] || ds_refuse 64 "arm '$first': a @portable arm is the release's build, and DEPTH_SERVE_TARBALL names the tarball it must equal: it is unset"
  base=${TB##*/}
  case $base in
    *.tar.gz) ;;
    *) ds_refuse 64 "DEPTH_SERVE_TARBALL '$TB' is no *.tar.gz (build.sh's bloomery-<version>-<platform>.tar.gz)" ;;
  esac
  TB_MEMBER=${base%.tar.gz}/bin/bloomery-serve
  if [ ! -r "$TB" ]; then
    ds_tarball_refuse 2 "no tarball at $TB (DEPTH_SERVE_TARBALL)"
  elif ! TB_SHA=$(ds_member_sha "$TB" "$TB_MEMBER"); then
    TB_SHA=''
    ds_tarball_refuse 2 "the tarball $TB holds no $TB_MEMBER"
  fi
}

# ds_portable_identity <arm index>: A_IDENT for a @portable arm, whose binary must be the tarball's member by sha256
# (rc 65 before the lease; a dry run prints the refusal). Prints the binary's line.
ds_portable_identity() {
  local i=$1 full
  if [ -z "$TB_SHA" ]; then
    A_IDENT+=(-)
    return 0
  fi
  full=$(ds_sha256 "${A_BIN[i]}")
  if [ "$full" != "$TB_SHA" ]; then
    if [ -z "$DRY" ]; then
      ds_refuse 65 "arm '${A_LABEL[i]}': ${A_BIN[i]} (sha256 $full) is not the member $TB_MEMBER of $TB (sha256 $TB_SHA)"
    fi
    echo "[dry] tree ${A_LABEL[i]}: ${A_BIN[i]} is not the tarball's member (a run refuses, rc 65): sha256 $full, the member's $TB_SHA"
    A_IDENT+=(-)
    return 0
  fi
  A_IDENT+=("$full (the member $TB_MEMBER of $TB)")
  echo "[binary] ${A_BIN[i]} is the member $TB_MEMBER of $TB: sha256 $full"
}

# ds_resolve_bins: A_BIN, A_SHA, A_MTIME and A_IDENT (what the witness's binary line prints) per arm, each binary checked
# once. A run refuses (2 missing, 3 stale, 65 not the tarball's) before the lease; a dry run prints the state and goes on.
ds_resolve_bins() {
  local i j dir rc why rel
  A_BIN=() A_SHA=() A_MTIME=() A_IDENT=()
  ds_tarball_resolve
  for ((i = 0; i < ${#A_LABEL[@]}; i++)); do
    dir=$HOME/repo/${A_TREE[i]}
    rel=$SERVE_BIN
    [ "${A_KIND[i]}" != portable ] || rel=$PORTABLE_BIN
    A_BIN+=("$dir/$rel")
    for ((j = 0; j < i; j++)); do
      [ "${A_TREE[j]}:${A_KIND[j]}" != "${A_TREE[i]}:${A_KIND[i]}" ] || break
    done
    if [ "$j" -lt "$i" ]; then
      A_SHA+=("${A_SHA[j]}") A_MTIME+=("${A_MTIME[j]}") A_IDENT+=("${A_IDENT[j]}")
      continue
    fi
    # Local mode runs the recipe in the contributor's tree, and the recipe builds that tree's target/release for an arm
    # of its name (a @portable arm builds nothing).
    if [ "${A_KIND[i]}" = tree ] && [ -n "$TIMING_LOCAL" ] && [ "${A_TREE[i]}" = "${ROOT##*/}" ] \
      && [ "$(cd "$dir" 2> /dev/null && pwd -P)" != "$(cd "$ROOT" && pwd -P)" ]; then
      why="$dir is not this runner's tree $ROOT, which has its name: under local mode the depth-serve recipe builds the tree it runs in, so run it from $dir"
      [ -n "$DRY" ] || ds_refuse 64 "arm '${A_LABEL[i]}': $why"
      echo "[dry] tree ${A_TREE[i]}: $why (a run refuses, rc 64)"
      A_SHA+=(-) A_MTIME+=(-) A_IDENT+=(-)
      continue
    fi
    if [ ! -d "$dir" ]; then
      [ -n "$DRY" ] || ds_refuse 2 "arm '${A_LABEL[i]}': no tree $dir"
      echo "[dry] tree ${A_TREE[i]}: no $dir (a run refuses, rc 2)"
      A_SHA+=(-) A_MTIME+=(-) A_IDENT+=(-)
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
      A_SHA+=(-) A_MTIME+=(-) A_IDENT+=(-)
      continue
    fi
    A_SHA+=("$BIN_SHA") A_MTIME+=("$BIN_MTIME")
    if [ "${A_KIND[i]}" = portable ]; then
      ds_portable_identity "$i"
    else
      A_IDENT+=("$BIN_SHA")
    fi
  done
}

# ds_server_cmd <arm index>: SRV_CMD, the arm's server command: the serving defaults beside the placement and the context
# the caller asked for.
ds_server_cmd() {
  SRV_CMD=("${A_BIN[$1]}" -m "$MODEL" --host 127.0.0.1 --port "$PORT")
  [ -z "$PLACE" ] || SRV_CMD+=(--place "$PLACE")
  [ -z "$CTX" ] || SRV_CMD+=(--ctx-size "$CTX")
}

# ds_log_name <round> <label> <D> <kind>: the arm's server log under LOGDIR, its label made a file name.
ds_log_name() { echo "$LOGDIR/$4$1-$(echo "$2" | tr -c 'A-Za-z0-9._\n-' _)-D$3.log"; }

# ds_print_plan: the plan, for a dry run and the head of a real one.
ds_print_plan() {
  local i r k n=${#A_LABEL[@]} line envs servers=$((${#A_LABEL[@]} * ROUNDS)) est w
  echo "[plan] model=$MODEL_NAME file=$MODEL ids=$(ds_ids_file)"
  echo "[plan] servers=$servers (${n} arms x $ROUNDS rounds, one load each) n_predict=$N ctx-size=${CTX:-unset (the plan picks it; its ctx_max is in each row label)} place=${PLACE:-unset} port=$PORT arm_bound=${LEASE_ARM_BOUND}s req_bound=${REQ_BOUND}s"
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
  if [ "$USER_PLAN" != off ]; then
    echo "[plan] user-environment plan (DEPTH_SERVE_USER_PLAN=$USER_PLAN): before round 1, under the lease, each arm group's first arm runs with --plan, CUDA_VISIBLE_DEVICES removed; its plan record must be the one the group's first load prints"
  fi
  [ -z "$AFTER" ] || echo "[plan] hook: DEPTH_SERVE_AFTER (bound ${AFTER_BOUND}s) after the timed request of round 1's last server of each arm group"
  echo "[plan] memory: a server starts when each timing card's used memory is within ${MEM_SLACK_MIB} MiB of the sitting's baseline (waits up to ${MEM_WAIT_S}s)"
  est=${DEPTH_SERVE_LOAD_EST:-}
  [ -n "$est" ] || case $MODEL_NAME in mimo2) est=41 ;; esac
  if [ -n "$est" ]; then
    w=$((servers * (est + OVERHEAD_S)))
    echo "[plan] predicted wall >= $servers servers x (load $est s + overhead $OVERHEAD_S s) = $w s (~$((w / 60)) min) [derived: the load from DEPTH_SERVE_LOAD_EST, not measured here; the overhead from this runner's own waits]; each request adds D/pp + (N-1)/tg, known from the first row"
  else
    echo "[plan] predicted wall: $servers servers x (load + request + $OVERHEAD_S s overhead); this model's load time is not known here (DEPTH_SERVE_LOAD_EST=<s> prices it)"
  fi
}

# ds_fail <round> <index> <why>: the server stopped, the arm's FAIL line and the server's log tail, exit 1. In local
# mode the line also names an Xid since the last arm's check or the card lost, unless that check is the <why>.
ds_fail() {
  local why=$3
  lcpp_srv_stop
  [ -n "$TWOCARD_WHY" ] || timing_cards_arm "$(cat "${SRV_LOG:-/dev/null}" 2> /dev/null)" serve || why="$why; $TWOCARD_WHY"
  echo "FAIL r$1 ${A_LABEL[$2]} D=${A_DEP[$2]}: $why${CPU_BUSY_TAG:-}${OTHER_BUSY_TAG:-}" >&2
  if [ -r "${SRV_LOG:-}" ]; then
    echo "--- the server's log ($SRV_LOG), last 20 lines" >&2
    tail -n 20 "$SRV_LOG" >&2
  fi
  exit 1
}

# The plan record. tools/bloomery/records.py reads it by kind (`plan`, or `plan38` of the qwen3moe family seats): the
# record's fields all but the readings the host and the card are measured at (room, plan_room, card_free, read), which
# move with the page cache and a co-tenant, are the plan's decisions, and two loads are one key when they print the
# same ones.
RECORDS_PY=$ROOT/tools/bloomery/records.py

# ds_plan_py <records.py> <log>: the python of ds_plan_read, a function so that no here-document sits in a substitution.
ds_plan_py() {
  python3 -I - "$1" "$2" << 'PY'
import importlib.util
import sys

spec = importlib.util.spec_from_file_location('records', sys.argv[1])
records = importlib.util.module_from_spec(spec)
spec.loader.exec_module(records)
READINGS = ('room', 'plan_room', 'card_free', 'read')
lines = open(sys.argv[2], errors='replace').read().splitlines()
for name in ('generate_ds41', 'generate_qwen3moe'):
    plans = [r for r in records.read(lines, name) if r.kind in ('plan', 'plan38') and not r.loose]
    if plans:
        r = plans[0]
        print(r.raw['place'])
        print(r.raw['ctx_max'])
        print('\t'.join('%s=%s' % (k, v) for k, v in r.raw.items() if k not in READINGS))
        sys.exit(0)
print('ERR the log holds no plan record that tools/bloomery/records.py reads whole (kinds plan and plan38 of generate_ds41 and generate_qwen3moe)')
sys.exit(1)
PY
}

# ds_plan_read <log>: PLAN_PLACE, PLAN_CTX (the plan's ctx_max) and PLAN_KEY (the decisions, tab-separated name=value) of
# the log's first plan record; 1 with PLAN_WHY when the log holds none.
ds_plan_read() {
  local out rest
  PLAN_PLACE='' PLAN_CTX='' PLAN_KEY='' PLAN_WHY=''
  out=$(ds_plan_py "$RECORDS_PY" "$1" 2> /dev/null) || {
    PLAN_WHY=${out#ERR }
    [ -n "$PLAN_WHY" ] || PLAN_WHY="tools/bloomery/records.py could not read $1"
    return 1
  }
  PLAN_PLACE=${out%%$'\n'*} rest=${out#*$'\n'}
  PLAN_CTX=${rest%%$'\n'*} PLAN_KEY=${rest#*$'\n'}
}

# ds_plan_diff <key> <key>: the fields that differ, `name old -> new`, comma-separated.
ds_plan_diff() {
  python3 -I - "$1" "$2" << 'PY'
import sys

a = dict(x.split('=', 1) for x in sys.argv[1].split('\t'))
b = dict(x.split('=', 1) for x in sys.argv[2].split('\t'))
print(', '.join('%s %s -> %s' % (k, a.get(k, '-'), b.get(k, '-')) for k in list(a) + [k for k in b if k not in a] if a.get(k) != b.get(k)))
PY
}

# ds_row_label <index>: the arm's label and the plan's own picks of the load that just ran.
ds_row_label() { echo "${A_LABEL[$1]}/place=$PLAN_PLACE/ctx=$PLAN_CTX"; }

# ds_plan_group <round> <index>: the load's plan against its arm group's first (G_KEY): a drift fails the arm by name. The
# group's first load is also held to the user's environment's plan (ds_user_plan_compare).
ds_plan_group() {
  local r=$1 i=$2 g=${A_GROUP[$2]} diff
  if [ -z "${G_KEY[$g]:-}" ]; then
    G_KEY[g]=$PLAN_KEY G_FIRST[g]="r$r ${A_LABEL[i]} D=${A_DEP[i]}"
    ds_user_plan_compare "$r" "$i"
    return 0
  fi
  [ "${G_KEY[$g]}" != "$PLAN_KEY" ] || return 0
  diff=$(ds_plan_diff "${G_KEY[$g]}" "$PLAN_KEY")
  ds_fail "$r" "$i" "plan drifted: $diff (against the arm's first load, ${G_FIRST[$g]}): the rows are not one key"
}

# ds_user_plan_probe: before round 1, under the lease (the card is ours and idle), each arm group's first arm with --plan
# and no CUDA_VISIBLE_DEVICES, which is the placement a user's run of the same command gets: its plan record is UP_KEY of
# the group. A probe that fails ends the run under `check` and is noted under `note`.
ds_user_plan_probe() {
  local i rc out why t0
  local -a envs
  UP_KEY=()
  [ "$USER_PLAN" != off ] || return 0
  for ((i = 0; i < ${#A_LABEL[@]}; i++)); do
    [ "${A_GROUP[i]}" = "$i" ] || continue
    ds_server_cmd "$i"
    envs=()
    [ -z "${A_ENV[i]}" ] || IFS=, read -r -a envs <<< "${A_ENV[i]}"
    rc=0 t0=$SECONDS
    lease_capture "$PLAN_PROBE_BOUND" env -u BLOOMERY_REF_MODEL -u CUDA_VISIBLE_DEVICES ${envs[@]+"${envs[@]}"} "${SRV_CMD[@]}" --plan || rc=$?
    out=$(ds_log_name 0 "${A_LABEL[i]}" "${A_DEP[i]}" user-plan-)
    printf '%s\n' "$LEASE_OUT" > "$out"
    why=''
    if [ "$rc" != 0 ]; then
      why="--plan without CUDA_VISIBLE_DEVICES exited $rc ($out): $(tail -n 1 "$out")"
    elif ! ds_plan_read "$out"; then
      why="--plan without CUDA_VISIBLE_DEVICES: $PLAN_WHY ($out)"
    fi
    if [ -n "$why" ]; then
      if [ "$USER_PLAN" = note ]; then
        echo "[user-plan] ${A_LABEL[i]}: $why (DEPTH_SERVE_USER_PLAN=note: the run goes on, the group's plan is not compared)"
        continue
      fi
      echo "FAIL user-plan ${A_LABEL[i]}: $why" >&2
      exit 1
    fi
    UP_KEY[i]=$PLAN_KEY
    echo "[user-plan] ${A_LABEL[i]}: --plan without CUDA_VISIBLE_DEVICES prints place=$PLAN_PLACE ctx_max=$PLAN_CTX (the probe took $((SECONDS - t0)) s)"
  done
}

# ds_user_plan_compare <round> <index>: the group's first load against its user-environment probe.
ds_user_plan_compare() {
  local g=${A_GROUP[$2]} diff msg
  [ -n "${UP_KEY[$g]:-}" ] || return 0
  if [ "${UP_KEY[$g]}" = "$PLAN_KEY" ]; then
    echo "[user-plan] ${A_LABEL[$2]}: the user's environment prints the plan the timing environment loads"
    return 0
  fi
  diff=$(ds_plan_diff "${UP_KEY[$g]}" "$PLAN_KEY")
  msg="the plan the user's environment prints (no CUDA_VISIBLE_DEVICES) is not the plan the timing environment loads: $diff"
  if [ "$USER_PLAN" = note ]; then
    echo "[user-plan] r$1 ${A_LABEL[$2]} D=${A_DEP[$2]}: $msg (DEPTH_SERVE_USER_PLAN=note: the run goes on)"
    return 0
  fi
  ds_fail "$1" "$2" "$msg"
}

# The card's memory between servers. A server's memory is released once its process is gone, and a load that starts before
# that finds a smaller card than the sitting's plan assumed: the next arm waits until each timing card's used memory is
# within MEM_SLACK_MIB of what it was when the lease was taken.

# ds_mem_used <gpu>: the card's used memory in MiB; empty when nvidia-smi gives no whole number (a unified-memory device
# answers [N/A]).
ds_mem_used() {
  local v
  v=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" 2> /dev/null | head -n 1 | tr -d ' ')
  case $v in '' | *[!0-9]*) return 0 ;; esac
  echo "$v"
}

# ds_mem_baseline: MEM_GPU and MEM_BASE, each timing card and its used MiB now; a card with no reading is named and skipped.
ds_mem_baseline() {
  local g v
  MEM_GPU=() MEM_BASE=()
  for g in $TIMING_GPU ${TIMING_GPU2:-}; do
    v=$(ds_mem_used "$g")
    if [ -z "$v" ]; then
      echo "[mem-baseline] the timing card $g gives no used-memory reading: the wait between servers is skipped"
      continue
    fi
    MEM_GPU+=("$g") MEM_BASE+=("$v")
    echo "[mem-baseline] the sitting's baseline is $v MiB on the timing card $g: a server's memory must be back within $MEM_SLACK_MIB MiB of it before the next load"
  done
}

# ds_mem_wait <round> <index>: every baselined card back within the slack, polling every half second for MEM_WAIT_S; the arm
# fails by name past it.
ds_mem_wait() {
  local k g v polls
  [ ${#MEM_GPU[@]} -gt 0 ] || return 0
  for ((k = 0; k < ${#MEM_GPU[@]}; k++)); do
    g=${MEM_GPU[k]} polls=0
    while :; do
      v=$(ds_mem_used "$g")
      [ -n "$v" ] || ds_fail "$1" "$2" "the timing card $g gives no used-memory reading now (${MEM_BASE[k]} MiB at the sitting's baseline): the wait for the last server's memory cannot tell"
      if [ "$v" -le $((MEM_BASE[k] + MEM_SLACK_MIB)) ]; then break; fi
      if [ "$polls" -ge $((MEM_WAIT_S * 2)) ]; then
        ds_fail "$1" "$2" "the timing card $g holds $v MiB, above the sitting's baseline ${MEM_BASE[k]} MiB + $MEM_SLACK_MIB MiB slack, $MEM_WAIT_S s after the last server stopped: its memory has not been released"
      fi
      if [ "$polls" = 0 ]; then
        echo "[mem-wait] r$1 ${A_LABEL[$2]} D=${A_DEP[$2]}: used $v MiB, baseline ${MEM_BASE[k]} MiB + $MEM_SLACK_MIB MiB slack: waiting for the last server's memory"
      fi
      polls=$((polls + 1))
      sleep 0.5
    done
    [ "$polls" = 0 ] || echo "[mem-wait] r$1 ${A_LABEL[$2]} D=${A_DEP[$2]}: back at ${v:-?} MiB after $((polls / 2)).$((polls % 2 * 5)) s"
  done
}

# ds_hook_run <round> <index>: DEPTH_SERVE_AFTER on the server that has just answered its timed request, under the lease
# (lease_bounded). Its first request must never be a functional one: the hook runs after the timed request of this
# very server (SRV_TIMED_PID) or is refused (70).
ds_hook_run() {
  local r=$1 i=$2 label=${A_LABEL[$2]} dep=${A_DEP[$2]} dir rc=0
  if [ -z "${SRV_PID:-}" ] || [ "${SRV_TIMED_PID:-}" != "$SRV_PID" ]; then
    ds_refuse 70 "ds_hook_run: refused: the hook of r$r $label D=$dep would run before the timed request of its server (the first request a server answers is the timed one)"
  fi
  dir=$(ds_log_name "$r" "$label" "$dep" session-)
  dir=${dir%.log}
  mkdir -p "$dir" || ds_refuse 2 "cannot make the hook's directory $dir"
  echo "[hook] r$r $label D=$dep: DEPTH_SERVE_AFTER after its timed request (bound ${AFTER_BOUND}s, http://127.0.0.1:$SRV_PORT, session dir $dir)"
  lease_bounded "$AFTER_BOUND" env "BLOOMERY_SESSION_URL=http://127.0.0.1:$SRV_PORT" "BLOOMERY_SESSION_PID=$SRV_PID" \
    "BLOOMERY_SESSION_STDERR=$SRV_LOG" "BLOOMERY_SESSION_DIR=$dir" bash -c "$AFTER" || rc=$?
  echo "HOOK r$r $label D=$dep rc=$rc"
  if [ "$rc" != 0 ]; then HOOK_FAILS=$((HOOK_FAILS + 1)); fi
  HOOK_RUNS=$((HOOK_RUNS + 1))
}

# ds_run_arm <round> <index>: one server, one request, its row (appended to ROWS_FILE) or a ds_fail.
ds_run_arm() {
  local r=$1 i=$2 label=${A_LABEL[$2]} dep=${A_DEP[$2]} ids t0 load tg pp maj listen ctx need win row
  local -a envs
  CPU_BUSY_TAG='' OTHER_BUSY_TAG='' COLD_TAG='' SRV_TIMED_PID=''
  guard_other
  ds_guard_timing
  ds_mem_wait "$r" "$i"
  guard_cpu "pre r$r $label D=$dep"
  BIN_PATH=${A_BIN[$i]} BIN_SHA=${A_IDENT[$i]} BIN_MTIME=${A_MTIME[$i]}
  ids=$(ds_ids "$dep")
  ds_server_cmd "$i"
  envs=()
  [ -z "${A_ENV[$i]}" ] || IFS=, read -r -a envs <<< "${A_ENV[$i]}"
  SRV_LOG=$(ds_log_name "$r" "$label" "$dep" r)
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
  if [ -n "$CTX" ]; then
    [ "$ctx" = "$CTX" ] || ds_fail "$r" "$i" "the listening line names ctx=$ctx, not the --ctx-size $CTX the arm asked"
  else
    need=$((dep + N - 1))
    [ "$need" -le "$ctx" ] || ds_fail "$r" "$i" "D $dep and n_predict $N take $need positions, past the context the server listens on ($ctx), which the unset rule picked: lower DEPTH_SERVE_N or name a DEPTH_SERVE_CTX that holds them"
  fi
  ds_plan_read "$SRV_LOG" || ds_fail "$r" "$i" "$PLAN_WHY: the placement and the context the plan picked are not proven"
  ds_plan_group "$r" "$i"
  row=$(ds_row_label "$i")
  witness "pre r$r $label D=$dep"
  if ! lcpp_srv_request "$ids" "$N" "$REQ_BOUND"; then
    witness "post r$r $label D=$dep (request failed)"
    ds_fail "$r" "$i" "$SRV_WHY"
  fi
  SRV_TIMED_PID=$SRV_PID
  witness "post r$r $label D=$dep"
  guard_cpu "post r$r $label D=$dep"
  maj=$((SRV_F1 - SRV_F0))
  win=$(awk -v a="$SRV_PROMPT_MS" -v b="$SRV_PRED_MS" 'BEGIN { printf "%.2f", (a + b) / 1000 }')
  cold_check "$maj" "$win"
  if [ -n "$COLD_TAG" ]; then
    echo "[cold] r$r $label D=$dep: $maj major faults x $COLD_US us could cost up to $MAJ_BOUND % of the $win s window: the row is out of the headline mean"
  fi
  tg=$(awk -v v="$SRV_PRED_TPS" 'BEGIN { printf "%.2f", v }')
  pp=$(awk -v v="$SRV_PROMPT_TPS" 'BEGIN { printf "%.2f", v }')
  if ! awk -v v="$tg" 'BEGIN { exit !(v > 0) }'; then
    ds_fail "$r" "$i" "timings.predicted_per_second is '$SRV_PRED_TPS', not a rate"
  fi
  if [ -n "$AFTER" ] && [ "$r" = 1 ] && [ "${A_HOOK[$i]}" = 1 ]; then ds_hook_run "$r" "$i"; fi
  lcpp_srv_stop
  # Local mode: an Xid since the last arm's check, or the card lost, fails the arm (timing-card.sh).
  timing_cards_arm "$(cat "$SRV_LOG")" serve || ds_fail "$r" "$i" "$TWOCARD_WHY"
  echo "ROW r$r $row D=$dep | tg_tok/s $tg | pp_tok/s $pp | load_s $load | majflt $maj${CPU_BUSY_TAG:-}${OTHER_BUSY_TAG:-}${COLD_TAG:-}" | tee -a "$ROWS_FILE"
}

# ds_main <arm>...: the run (or, under BLOOMERY_DRY, the plan).
ds_main() {
  # A two-card request this machine cannot honour ends here, under its mode's tag (an opted-in runner says it itself).
  if [ -n "${BLOOMERY_TIMING_CARDS:-}" ] && [ -n "$TIMING_CARDS_WHY" ]; then
    if [ -n "$TIMING_LOCAL" ]; then echo "[timing-card] refused: $TIMING_CARDS_WHY" >&2; else echo "[timing-cards] refused: $TIMING_CARDS_WHY" >&2; fi
    exit 64
  fi
  local i r k n v maxd need
  # Local mode refused (timing-card.sh printed why as this file sourced it) ends the run here, dry or not.
  [ -z "$TIMING_LOCAL" ] || [ -z "$TIMING_CARDS_WHY" ] || exit 64
  for v in $(compgen -v BLOOMERY_SERVE_); do
    ds_refuse 64 "$v is set, and nothing reads it: this runner's knobs are DEPTH_SERVE_CTX, DEPTH_SERVE_PLACE, DEPTH_SERVE_N, DEPTH_SERVE_PORT, DEPTH_SERVE_REQ_BOUND, DEPTH_SERVE_IDS, DEPTH_SERVE_LOAD_EST, DEPTH_SERVE_TARBALL, DEPTH_SERVE_AFTER, DEPTH_SERVE_AFTER_BOUND, DEPTH_SERVE_USER_PLAN, DEPTH_SERVE_MEM_WAIT_S and DEPTH_SERVE_MEM_SLACK_MIB (a BLOOMERY_ name would have to be a row of the lever registry)"
  done
  DRY=${BLOOMERY_DRY:-}
  ROUNDS=${BLOOMERY_AB_ROUNDS:-4}
  N=${DEPTH_SERVE_N:-128}
  CTX=${DEPTH_SERVE_CTX:-}
  PLACE=${DEPTH_SERVE_PLACE:-}
  PORT=${DEPTH_SERVE_PORT:-18180}
  REQ_BOUND=${DEPTH_SERVE_REQ_BOUND:-300}
  AFTER=${DEPTH_SERVE_AFTER:-}
  AFTER_BOUND=${DEPTH_SERVE_AFTER_BOUND:-300}
  USER_PLAN=${DEPTH_SERVE_USER_PLAN:-check}
  MEM_WAIT_S=${DEPTH_SERVE_MEM_WAIT_S:-60}
  MEM_SLACK_MIB=${DEPTH_SERVE_MEM_SLACK_MIB:-128}
  PLAN_PROBE_BOUND=120
  ds_posint BLOOMERY_AB_ROUNDS "$ROUNDS" 1
  ds_posint DEPTH_SERVE_N "$N" 2
  [ -z "$CTX" ] || ds_posint DEPTH_SERVE_CTX "$CTX" 1
  ds_posint DEPTH_SERVE_PORT "$PORT" 0
  [ "$PORT" -le 65535 ] || ds_refuse 64 "DEPTH_SERVE_PORT is a port, got $PORT"
  ds_posint DEPTH_SERVE_REQ_BOUND "$REQ_BOUND" 1
  ds_posint DEPTH_SERVE_AFTER_BOUND "$AFTER_BOUND" 1
  ds_posint DEPTH_SERVE_MEM_WAIT_S "$MEM_WAIT_S" 0
  ds_posint DEPTH_SERVE_MEM_SLACK_MIB "$MEM_SLACK_MIB" 0
  ds_posint BLOOMERY_ARM_BOUND "$LEASE_ARM_BOUND" 1
  case $PLACE in '' | a | bp) ;; *) ds_refuse 64 "DEPTH_SERVE_PLACE is a or bp (unset: the plan's own pick), got '$PLACE'" ;; esac
  case $USER_PLAN in check | note | off) ;; *) ds_refuse 64 "DEPTH_SERVE_USER_PLAN is check, note or off, got '$USER_PLAN'" ;; esac
  if [ -n "$AFTER" ] && [ "$AFTER_BOUND" -ge "$LEASE_ARM_BOUND" ]; then
    ds_refuse 64 "DEPTH_SERVE_AFTER_BOUND $AFTER_BOUND s must be under BLOOMERY_ARM_BOUND $LEASE_ARM_BOUND s: the server's bound covers its load, its request and the hook"
  fi
  [ $# -ge 1 ] || ds_refuse 64 "no arms: <tree>:<D> names a tree under ~/repo and the prompt length in ids"
  ds_parse_arms "$@"
  n=${#A_LABEL[@]}
  maxd=0
  for ((i = 0; i < n; i++)); do
    [ "${A_DEP[i]}" -le "$maxd" ] || maxd=${A_DEP[i]}
    need=$((A_DEP[i] + N - 1))
    [ -z "$CTX" ] || [ "$need" -le "$CTX" ] || arm_refuse "${A_LABEL[i]}:${A_DEP[i]}" "D ${A_DEP[i]} and n_predict $N take $need positions, past --ctx-size $CTX: bloomery-serve feeds the generated token k at position D + k - 1 and never the last, so D + n_predict - 1 must not pass the context (crates/serve/src/genloop.rs); lower DEPTH_SERVE_N or raise DEPTH_SERVE_CTX"
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
  timing_cards_before_lease
  # ROUNDS (global above) is what lease_take checks an ab card against.
  lease_take
  timing_cards_start
  LOGDIR=$(mktemp -d "${TMPDIR:-/tmp}/depth-serve.XXXXXX") || exit 2
  ROWS_FILE=$LOGDIR/rows.txt
  : > "$ROWS_FILE"
  G_KEY=() G_FIRST=() UP_KEY=() HOOK_FAILS=0 HOOK_RUNS=0 SRV_TIMED_PID=''
  echo "[config] model=$MODEL_NAME n=$N ctx=${CTX:-unset} port=$PORT rounds=$ROUNDS place=${PLACE:-unset} arm_bound=${LEASE_ARM_BOUND}s req_bound=${REQ_BOUND}s timing_gpu=$TIMING_GPU other_gpu=$OTHER_GPU CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES logs=$LOGDIR"
  echo "[config] arms=$* (the server at the serving defaults beside --ctx-size and --place when named, no warm-up request, one fresh server a row)"
  # The witness blocks' binary line (timing-card.sh's card field) names BIN_PATH and BIN_SHA: the first arm's for the
  # sitting's first block, each arm's own in its blocks.
  BIN_PATH=${A_BIN[0]} BIN_SHA=${A_IDENT[0]} BIN_MTIME=${A_MTIME[0]}
  witness pre
  guard_other
  ds_mem_baseline
  ds_user_plan_probe
  for ((r = 1; r <= ROUNDS; r++)); do
    for ((i = 0; i < n; i++)); do
      k=$(((i + r - 1) % n))
      ds_run_arm "$r" "$k"
    done
  done
  echo
  ds_tables "$ROWS_FILE"
  witness post
  if [ "$HOOK_FAILS" != 0 ]; then
    echo "[hook] $HOOK_FAILS of $HOOK_RUNS hook run(s) failed: the run's exit status is 1"
    return 1
  fi
}

# ----------------------------------------------------------------------------------------------------------------
# `bash tools/ref/depth-serve.sh --self-test`: arm parsing, rotation, the refusals, the server stopped after each arm,
# local mode's checks and the tables' math, on the stub server (python3, curl, 127.0.0.1), no card, no model. Each case
# runs in its own directory under one temporary root with its own HOME, data and TMPDIR; the stub logs its pid there,
# and the cleanup signals only those pids. ds_main runs in a subshell with the lease (outside the local cases, which
# take the real one in their own lock directory), the witness block and the guards replaced by markers (the real
# stale-binary rule, the real servers' start, request and stop stay).
ds_selftest() {
  # T is read by the EXIT trap, after this function has returned: not local.
  local n=0 fails=0 k C RC lever knob gb10 own stub=$HERE/stub-llama-server.py
  command -v python3 > /dev/null || { echo "depth-serve.sh --self-test: python3 is not on PATH" >&2; exit 2; }
  T=$(mktemp -d "${TMPDIR:-/tmp}/depth-serve-self-test.XXXXXX") || exit 2
  st_cleanup() {
    local f p
    for f in "$T"/*/stub-srv-pids "$T"/*/listener-pid; do
      [ -r "$f" ] || continue
      while read -r p; do
        # Only a pid the case wrote and whose command is still the stub or the listener it started.
        if kill -0 "$p" 2> /dev/null && ps -o command= -p "$p" 2> /dev/null | grep -E 'stub-llama-server|port-holder' > /dev/null; then kill "$p" 2> /dev/null; fi
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
  # st_bin <name> <path in the tree> <tps> [NAME=VALUE...]: ~/repo/<name> with Cargo files and a binary newer than them,
  # a wrapper that dumps its environment, prints the `plan` record bloomery-serve prints before its load (`--plan` ends
  # there; a probe's dump is envprobe.*, a server's env.*) and runs the stub at that decode rate, listing `ctx=` as
  # bloomery-serve's record does. The wrapper's sequences, one word a load (a probe prints the first), come from the
  # run's environment: ST_CTXSEQ the plan's context where no --ctx-size is given (ST_CTXFIXED, default 16384, past the
  # sequence), ST_EXPSEQ the card experts, ST_TPSSEQ the decode rate; ST_PLAN_USER_CARD the card a probe without
  # CUDA_VISIBLE_DEVICES names; the tree's own NAME=VALUE words reach the wrapper as well as the stub. The plan's room,
  # plan_room and card_free readings change with every load, as a host's do.
  st_bin() {
    local name=$1 rel=$2 tps=$3 d=$C/home/repo/$1
    shift 3
    mkdir -p "$d/${rel%/*}"
    : > "$d/Cargo.toml"
    : > "$d/Cargo.lock"
    touch -t 202601010000 "$d/Cargo.toml" "$d/Cargo.lock"
    printf '#!/bin/sh\nC=%s NAME=%s TPS=%s STUB=%s\nEXTRA="%s"\n' "$C" "$name" "$tps" "$stub" "$*" > "$d/$rel"
    cat >> "$d/$rel" << 'WRAP'
for kv in $EXTRA; do export "$kv"; done
pick() {
  i=1
  for w in $1; do
    if [ "$i" = "$2" ]; then echo "$w"; return; fi
    i=$((i + 1))
  done
  echo "$3"
}
ctx= place=a prev=
for a in "$@"; do
  case $prev in --ctx-size) ctx=$a ;; --place) place=$a ;; esac
  prev=$a
done
probe=
case " $* " in *" --plan "*) probe=1 ;; esac
n=0
if [ -n "$probe" ]; then
  env > "$C/envprobe.$NAME.$$"
else
  env > "$C/env.$NAME.$$"
  n=$(cat "$C/loads.$NAME" 2> /dev/null || echo 0)
  n=$((n + 1))
  echo "$n" > "$C/loads.$NAME"
fi
# A probe prints what the first load prints.
k=$n
[ -z "$probe" ] || k=1
ctxarg=
if [ -z "$ctx" ]; then
  ctx=$(pick "${ST_CTXSEQ:-}" "$k" "${ST_CTXFIXED:-16384}")
  ctxarg="--ctx $ctx"
fi
card=Stub
[ -n "$probe" ] && [ -z "${CUDA_VISIBLE_DEVICES:-}" ] && card=${ST_PLAN_USER_CARD:-Stub}
exp=$(pick "${ST_EXPSEQ:-}" "$k" 1063)
room=$((263890399232 + n * 4096))
echo "plan place=$place card=$card ctx_max=$ctx card_experts=$exp (33857090004 B) host_experts=6233 (198524141568 B) host_shadow=0 B n_l=18..19 on 57 layers card_budget=none paged=0 arena=0 arena_slots=0 room=$room plan_room=$((room - 1000)) card_free=$((50668240896 - n * 512)) devices=[stage:$card:cuda0:50952404992] cuda_order=cuda0" >&2
[ -z "$probe" ] || exit 0
tps=$(pick "${ST_TPSSEQ:-}" "$n" "$TPS")
exec env STUB_SRV_TPS=$tps STUB_SRV_LISTEN_CTX=argv $EXTRA python3 "$STUB" "$@" $ctxarg
WRAP
    chmod +x "$d/$rel"
    touch -t 202601020000 "$d/$rel"
  }
  st_tree() {
    local name=$1 tps=$2
    shift 2
    st_bin "$name" "$SERVE_BIN" "$tps" "$@"
  }
  st_portable() {
    local name=$1 tps=$2
    shift 2
    st_bin "$name" "$PORTABLE_BIN" "$tps" "$@"
  }
  # st_tarball <tree> [version]: ST_TB, a release tarball of ~/repo/<tree>'s portable binary packed as
  # tools/release/build.sh packs it ($NAME.tar.gz holding $NAME/bin/bloomery-serve), and ST_TBM, the member's name.
  st_tarball() {
    local name=bloomery-${2:-9.9.9}-linux-x86_64-cuda-sm86 stage=$C/tb
    rm -rf "$stage"
    mkdir -p "$stage/$name/bin"
    cp "$C/home/repo/$1/$PORTABLE_BIN" "$stage/$name/bin/bloomery-serve"
    tar -czf "$stage/$name.tar.gz" -C "$stage" "$name"
    ST_TB=$stage/$name.tar.gz ST_TBM=$name/bin/bloomery-serve
  }
  st_port() { python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])'; }
  # st_seq <name>: the next line of $C/<name>.seq (the last again once they run out), the position kept in $C/<name>.i
  # (a command substitution is a subshell, so a variable would not carry it).
  st_seq() {
    local i v
    i=$(cat "$C/$1.i" 2> /dev/null || echo 0)
    echo $((i + 1)) > "$C/$1.i"
    v=$(sed -n "$((i + 1))p" "$C/$1.seq")
    [ -n "$v" ] || v=$(tail -n 1 "$C/$1.seq")
    echo "$v"
  }
  # st_run <arg>...: ds_main in a subshell, its output in $C/out and its rc in RC. Extra variables go through ST_ENV.
  # Under ST_LOCAL the subshell sources timing-card.sh again in local mode, with $T/local's stub nvidia-smi and kernel
  # journal first on PATH, and the lease is the real lease_take in the case's lock directory, its card check left out.
  ST_ENV=() ST_PORT='' ST_BOUND='' ST_LOCAL=''
  st_run() {
    ( export HOME=$C/home TMPDIR=$C BLOOMERY_REF_MODEL=$C/model.gguf PATH=$C/bin:$PATH
      MODEL=$C/model.gguf BLOOMERY_DATA=$C/data
      export DEPTH_SERVE_PORT=${ST_PORT:-$(st_port)}
      # shellcheck disable=SC2163 # kv is a NAME=VALUE string
      for kv in ${ST_ENV[@]+"${ST_ENV[@]}"}; do export "$kv"; done
      if [ -n "$ST_LOCAL" ]; then
        export BLOOMERY_BOX_CARD=local BLOOMERY_LOCK_DIR=$C/locks BLOOMERY_LEASE_TRACE=$C/trace PATH=$T/local:$PATH
        # The source at the top pinned this shell's CUDA_VISIBLE_DEVICES in box mode; a local run starts from the caller's view.
        unset BLOOMERY_LEASE_LOCK BLOOMERY_LEASE_HELD CUDA_VISIBLE_DEVICES
        command -v flock > /dev/null || PATH=$HERE/card-tests/bin:$PATH
        mkdir -p "$C/locks"
        # shellcheck source=tools/ref/timing-card.sh
        source "$HERE/timing-card.sh"
        lease_card() { :; }
      else
        lease_take() { echo "[lease-stub] taken"; }
        TIMING_GPU=GPU-11111111-1111-1111-1111-111111111111
      fi
      [ -z "${ST_BOUND:-}" ] || LEASE_ARM_BOUND=$ST_BOUND
      # The block's tag, then the real host lines of the card field (the binary line among them).
      witness() { echo "[witness-stub] $1"; __witness_host; }
      guard_other() { :; }
      ds_guard_timing() { :; }
      guard_cpu() { :; }
      lcpp_srv_majflt() { if [ -r "$C/mf.seq" ]; then st_seq mf; else echo 7; fi; }
      ds_mem_used() {
        local v
        if [ -r "$C/mem.seq" ]; then
          v=$(st_seq mem)
          [ "$v" = none ] || echo "$v"
        fi
      }
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
  ST_ENV=(BLOOMERY_DRY=1 BLOOMERY_AB_ROUNDS=2 DEPTH_SERVE_CTX=4160)
  st_run alpha:6 "env=$lever=1:beta:1024"
  st_check dry-plan 0 "[plan] arm 1: alpha D=6:" "[plan] arm 2: beta@$lever=1 D=1024:" \
    "[plan] order r1: alpha:6 beta@$lever=1:1024" "[plan] order r2: beta@$lever=1:1024 alpha:6" \
    "env -u BLOOMERY_REF_MODEL $lever=1 $C/home/repo/beta/$SERVE_BIN -m $C/model.gguf --host 127.0.0.1 --port " \
    "--ctx-size 4160" "ctx-size=4160" "[plan] servers=4 (2 arms x 2 rounds" "[dry] nothing built, no lease taken, no server started"
  st_absent dry-takes-no-lease "[lease-stub]" "[witness-stub]"
  [ ! -e "$C/stub-srv-pids" ] && ok dry-starts-no-server || bad dry-starts-no-server "the stub was started"
  # The serving defaults: no --ctx-size (the plan picks the context) and no --place; a place asked for is passed on.
  ST_ENV=(BLOOMERY_DRY=1 BLOOMERY_AB_ROUNDS=1)
  st_run alpha:6
  st_check dry-plan-ctx-unset 0 "ctx-size=unset (the plan picks it" "place=unset"
  st_absent dry-plan-ctx-unset-no-flag "--ctx-size" "--place"
  ST_ENV=(BLOOMERY_DRY=1 BLOOMERY_AB_ROUNDS=1 DEPTH_SERVE_PLACE=bp)
  st_run alpha:6
  st_check dry-plan-place 0 "place=bp" "-m $C/model.gguf --host 127.0.0.1 --port " " --place bp"
  ST_ENV=(BLOOMERY_DRY=1 BLOOMERY_AB_ROUNDS=2)
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
  st_check refuse-arm-syntax 64 "want [env=K=V[,K=V]:]<tree>:<D>" "<tree>@portable:<D>"
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
  st_run alpha@nosuch:6
  st_check refuse-arm-binary-kind 64 "arm 'alpha@nosuch:6': 'nosuch' after '@' is no binary kind" "@portable"
  ST_ENV=(DEPTH_SERVE_CTX=4160)
  st_run alpha:4160
  st_check refuse-context 64 "take 4287 positions, past --ctx-size 4160"
  ST_ENV=(BLOOMERY_DRY=1 DEPTH_SERVE_CTX=4160)
  st_run alpha:4033
  st_check refuse-context-edge 0 "[plan] arm 1: alpha D=4033:"
  ST_ENV=(DEPTH_SERVE_PLACE=gate)
  st_run alpha:6
  st_check refuse-place 64 "DEPTH_SERVE_PLACE is a or bp (unset: the plan's own pick), got 'gate'"
  ST_ENV=(DEPTH_SERVE_USER_PLAN=maybe)
  st_run alpha:6
  st_check refuse-user-plan-mode 64 "DEPTH_SERVE_USER_PLAN is check, note or off, got 'maybe'"
  ST_ENV=(DEPTH_SERVE_AFTER=true DEPTH_SERVE_AFTER_BOUND=900 BLOOMERY_ARM_BOUND=900)
  st_run alpha:6
  st_check refuse-hook-bound 64 "DEPTH_SERVE_AFTER_BOUND 900 s must be under BLOOMERY_ARM_BOUND 900 s"
  st_absent refuse-hook-bound-before-the-lease "[lease-stub]"
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
  ST_ENV=(BLOOMERY_AB_ROUNDS=3 DEPTH_SERVE_CTX=4160)
  st_run a:6 b:6 c:6
  st_check rotate-run 0 "[lease-stub] taken" "[config] model=mimo2 n=128 ctx=4160 port=$ST_PORT rounds=3"
  got=$(grep '^ROW' "$C/out" | awk '{ sub(/\/.*/, "", $3); printf "%s%s ", $2, $3 }')
  [ "$got" = "r1a r1b r1c r2b r2c r2a r3c r3a r3b " ] && ok rotate-order || bad rotate-order "rows in order [$got]"
  st_servers rotate-servers-stopped 9
  [ "$(grep -c -- "--port $ST_PORT --ctx-size 4160" "$C/stub-srv-argv")" = 9 ] && ok rotate-fixed-port-and-ctx || bad rotate-fixed-port-and-ctx "the stub's argv lacks --port $ST_PORT --ctx-size 4160 nine times"
  [ "$(grep -c '^\[lease-stub\]' "$C/out")" = 1 ] && ok rotate-one-lease || bad rotate-one-lease "the lease was not taken once"
  seq=$(grep -E '^\[lease-stub\]|^\[witness-stub\] (pre r|post r)|^ROW' "$C/out" | head -n 4 | sed 's/ | pp_tok.*//' | tr '\n' '|')
  [ "$seq" = "[lease-stub] taken|[witness-stub] pre r1 a D=6|[witness-stub] post r1 a D=6|ROW r1 a/place=a/ctx=4160 D=6 | tg_tok/s 20.00|" ] &&
    ok rotate-witness-around-the-request || bad rotate-witness-around-the-request "got [$seq]"
  # shellcheck disable=SC2144 # a glob of at most 9 files, read by grep -l
  if grep -lq '^BLOOMERY_REF_MODEL=' "$C"/env.* 2> /dev/null; then bad rotate-server-env "a server saw BLOOMERY_REF_MODEL"; else ok rotate-server-env; fi
  st_check rotate-tables 0 "mean a/place=a/ctx=4160 D=6 | tg 20.00 tok/s (sd 0.00 = 0.00 %, [20.00..20.00], n=3) | pp 60.00 tok/s" \
    "diff b/place=a/ctx=4160 vs a/place=a/ctx=4160 D=6 | tg +1.00 tok/s (+5.00 %) ± 0.00 tok/s (± 0.00 %) [clears 0] | pp +0.00 tok/s (+0.00 %) ± 0.00 tok/s (± 0.00 %) [holds 0]" \
    "diff c/place=a/ctx=4160 vs a/place=a/ctx=4160 D=6 | tg +2.00 tok/s (+10.00 %)"
  # Every witness block names the binary it measures.
  [ "$(grep -c "^    binary: $C/home/repo/a/$SERVE_BIN sha256=" "$C/out")" = 7 ] && ok rotate-witness-names-the-binary || bad rotate-witness-names-the-binary "the seven witness blocks that name arm a (the sitting's first and a pre and a post a round) do not each carry its binary line"
  ST_PORT=
  ST_ENV=()

  # --- a lever arm: its variables reach that arm's servers only.
  st_case envarm
  st_tree a 20
  ST_ENV=(BLOOMERY_AB_ROUNDS=1)
  st_run a:6 "env=$lever=1:a:6"
  st_check envarm-run 0 "ROW r1 a/place=a/ctx=16384 D=6" "ROW r1 a@$lever=1/place=a/ctx=16384 D=6"
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
  ST_ENV=(BLOOMERY_AB_ROUNDS=1 DEPTH_SERVE_CTX=4160)
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

  # --- the serving defaults: no --ctx-size and no --place; the placement and the context the plan printed are in the
  # row's label.
  st_case ctxunset
  st_tree a 20
  ST_ENV=(BLOOMERY_AB_ROUNDS=1)
  st_run a:6
  st_check ctx-unset-run 0 "[config] model=mimo2 n=128 ctx=unset" "rounds=1 place=unset" "ROW r1 a/place=a/ctx=16384 D=6 | tg_tok/s 20.00" \
    "[mem-baseline] the timing card" "gives no used-memory reading: the wait between servers is skipped"
  [ "$(grep -c -- '--ctx-size' "$C/stub-srv-argv")" = 0 ] && ok ctx-unset-server-gets-no-flag || bad ctx-unset-server-gets-no-flag "the stub was given --ctx-size"
  # Nothing can hold D + n_predict - 1 against a context the plan has not picked: the load tells, and the arm fails by name.
  st_case ctxunset-n
  st_tree a 20
  ST_ENV=(BLOOMERY_AB_ROUNDS=1 DEPTH_SERVE_N=16384)
  st_run a:6
  st_check ctx-unset-refuses-after-the-load 1 "FAIL r1 a D=6: D 6 and n_predict 16384 take 16389 positions, past the context the server listens on (16384)"
  st_absent ctx-unset-refusal-no-request "ROW "
  st_servers ctx-unset-refusal-stops-the-server 1
  st_case place
  st_tree a 20
  ST_ENV=(BLOOMERY_AB_ROUNDS=1 DEPTH_SERVE_PLACE=bp)
  st_run a:6
  st_check place-run 0 "rounds=1 place=bp" "ROW r1 a/place=bp/ctx=16384 D=6"
  [ "$(grep -c -- ' --place bp' "$C/stub-srv-argv")" = 1 ] && ok place-reaches-the-server || bad place-reaches-the-server "the stub was not given --place bp"

  # --- the binary: a @portable arm is the release's build, equal by sha256 to its tarball's member, before the lease.
  st_case portable
  st_portable p 20
  st_tarball p
  ST_ENV=(BLOOMERY_AB_ROUNDS=1)
  st_run p@portable:6
  st_check refuse-portable-no-tarball 64 "arm 'p@portable': a @portable arm is the release's build, and DEPTH_SERVE_TARBALL names the tarball it must equal"
  want=$(sha256sum "$C/home/repo/p/$PORTABLE_BIN" | cut -d' ' -f1)
  printf '# a later build\n' >> "$C/home/repo/p/$PORTABLE_BIN"
  touch -t 202601020000 "$C/home/repo/p/$PORTABLE_BIN"
  got=$(sha256sum "$C/home/repo/p/$PORTABLE_BIN" | cut -d' ' -f1)
  ST_ENV=(BLOOMERY_AB_ROUNDS=1 "DEPTH_SERVE_TARBALL=$ST_TB")
  st_run p@portable:6
  st_check refuse-portable-sha 65 "arm 'p@portable': $C/home/repo/p/$PORTABLE_BIN (sha256 $got) is not the member $ST_TBM of $ST_TB (sha256 $want)"
  st_absent refuse-portable-sha-before-the-lease "[lease-stub]"
  [ ! -e "$C/stub-srv-pids" ] && ok refuse-portable-sha-starts-no-server || bad refuse-portable-sha-starts-no-server "the stub was started"
  ST_ENV=(BLOOMERY_DRY=1 "DEPTH_SERVE_TARBALL=$ST_TB")
  st_run p@portable:6
  st_check portable-sha-dry 0 "[dry] tree p@portable: $C/home/repo/p/$PORTABLE_BIN is not the tarball's member (a run refuses, rc 65)"
  st_portable p 20
  ST_ENV=(BLOOMERY_AB_ROUNDS=1 "DEPTH_SERVE_TARBALL=$ST_TB")
  st_run p@portable:6
  st_check portable-run 0 "[binary] $C/home/repo/p/$PORTABLE_BIN is the member $ST_TBM of $ST_TB: sha256 $want" \
    "ROW r1 p@portable/place=a/ctx=16384 D=6 | tg_tok/s 20.00"
  [ "$(grep -c "^    binary: $C/home/repo/p/$PORTABLE_BIN sha256=$want (the member $ST_TBM of $ST_TB) mtime=" "$C/out")" -ge 2 ] &&
    ok portable-witness-names-the-tarball || bad portable-witness-names-the-tarball "the witness blocks carry no binary line naming the tarball and its sha256"
  st_case portable-other
  st_tree t 20
  st_portable p 20
  st_tarball p
  ST_ENV=("DEPTH_SERVE_TARBALL=$ST_TB")
  st_run t:6
  st_check refuse-tarball-without-portable-arm 64 "DEPTH_SERVE_TARBALL is set, and no arm is @portable"
  ST_ENV=("DEPTH_SERVE_TARBALL=$C/none.tar.gz")
  st_run p@portable:6
  st_check refuse-tarball-missing 2 "no tarball at $C/none.tar.gz"
  mkdir -p "$C/tb2/other/bin"
  cp "$C/home/repo/p/$PORTABLE_BIN" "$C/tb2/other/bin/bloomery-serve"
  tar -czf "$C/tb2/bloomery-0.0.0-x.tar.gz" -C "$C/tb2" other
  ST_ENV=("DEPTH_SERVE_TARBALL=$C/tb2/bloomery-0.0.0-x.tar.gz")
  st_run p@portable:6
  st_check refuse-tarball-no-member 2 "the tarball $C/tb2/bloomery-0.0.0-x.tar.gz holds no bloomery-0.0.0-x/bin/bloomery-serve"
  # The portable build's own dep-info (target/portable/release/bloomery-serve.d) is the freshness rule's.
  st_case portable-fresh
  st_portable q 20
  mkdir -p "$C/home/repo/q/crates/x/src"
  : > "$C/home/repo/q/crates/x/src/lib.rs"
  touch -t 202601010000 "$C/home/repo/q/crates/x/src/lib.rs"
  printf 'target/portable/release/bloomery-serve: crates/x/src/lib.rs\n' > "$C/home/repo/q/$PORTABLE_BIN.d"
  st_tarball q
  ST_ENV=(BLOOMERY_DRY=1 "DEPTH_SERVE_TARBALL=$ST_TB")
  st_run q@portable:6
  st_check portable-fresh 0 "[binary] $C/home/repo/q/$PORTABLE_BIN sha256=" "(newer than its sources: dep-info)"
  touch -t 202601030000 "$C/home/repo/q/crates/x/src/lib.rs"
  ST_ENV=("DEPTH_SERVE_TARBALL=$ST_TB")
  st_run q@portable:6
  st_check refuse-portable-stale 3 "[stale-binary] $C/home/repo/q/$PORTABLE_BIN" "is older than its sources" "crates/x/src/lib.rs"
  st_absent refuse-portable-stale-before-the-lease "[lease-stub]"
  ST_ENV=()

  # --- the plan: every load of one arm group prints the same decisions (its readings may move), else `plan drifted`.
  st_case plandrift
  st_tree pd 20
  ST_ENV=(BLOOMERY_AB_ROUNDS=2 "ST_EXPSEQ=1063 1100")
  st_run pd:6
  st_check refuse-plan-drift 1 "FAIL r2 pd D=6: plan drifted: card_experts 1063 -> 1100 (against the arm's first load, r1 pd D=6)"
  st_servers plan-drift-stops-the-server 2
  [ "$(grep -c '^ROW r1 ' "$C/out")" = 1 ] && [ "$(grep -c '^ROW r2 ' "$C/out")" = 0 ] && ok plan-drift-keeps-the-rows-before-it || bad plan-drift-keeps-the-rows-before-it "round 1's row is not the only one"
  st_case ctxdrift
  st_tree cd 20
  ST_ENV=(BLOOMERY_AB_ROUNDS=2 "ST_CTXSEQ=16384 8192")
  st_run cd:6
  st_check refuse-ctx-drift 1 "FAIL r2 cd D=6: plan drifted: ctx_max 16384 -> 8192"
  st_case planreadings
  st_tree pr 20
  ST_ENV=(BLOOMERY_AB_ROUNDS=3)
  st_run pr:6
  st_check plan-readings-are-no-drift 0 "ROW r3 pr/place=a/ctx=16384 D=6"
  [ "$(cat "$C"/depth-serve.*/r*-pr-D6.log | grep -o ' room=[0-9]*' | sort -u | wc -l | tr -d ' ')" = 3 ] &&
    ok plan-readings-moved || bad plan-readings-moved "the three loads did not print three room readings"
  # Two arm groups, two plans: a candidate may plan as it likes; only a group's own loads are one key.
  st_case plantwo
  st_tree a 20
  st_tree b 21 ST_CTXFIXED=8192
  ST_ENV=(BLOOMERY_AB_ROUNDS=2)
  st_run a:6 b:6
  st_check plan-per-group 0 "ROW r2 a/place=a/ctx=16384 D=6" "ROW r2 b/place=a/ctx=8192 D=6"

  # --- the user's environment: `--plan` without CUDA_VISIBLE_DEVICES prints the plan the timing environment loads.
  st_case userplan
  st_tree up 20
  ST_ENV=(BLOOMERY_AB_ROUNDS=1 ST_PLAN_USER_CARD=Other)
  st_run up:6
  st_check refuse-user-plan-differs 1 "FAIL r1 up D=6: the plan the user's environment prints (no CUDA_VISIBLE_DEVICES) is not the plan the timing environment loads: card Other -> Stub"
  st_absent refuse-user-plan-differs-no-row "ROW "
  st_servers refuse-user-plan-differs-server-stopped 1
  st_case userplan-note
  st_tree up 20
  ST_ENV=(BLOOMERY_AB_ROUNDS=1 ST_PLAN_USER_CARD=Other DEPTH_SERVE_USER_PLAN=note)
  st_run up:6
  st_check user-plan-note 0 "[user-plan] r1 up D=6: the plan the user's environment prints" "card Other -> Stub" "DEPTH_SERVE_USER_PLAN=note: the run goes on" "ROW r1 up/place=a/ctx=16384 D=6"
  st_case userplan-same
  st_tree up 20
  ST_ENV=(BLOOMERY_AB_ROUNDS=1)
  st_run up:6
  st_check user-plan-same 0 "[user-plan] up: the user's environment prints the plan the timing environment loads" "ROW r1 up/place=a/ctx=16384 D=6"
  # The probe is a process on the card: it runs once the lease is held and before the first server starts.
  order=$(grep -n -E '^\[lease-stub\]|^\[user-plan\] up: --plan|^\[server\] r1' "$C/out" | sed 's/^[0-9]*:\[\([a-z-]*\)\].*/\1/' | tr '\n' ' ')
  [ "$order" = "lease-stub user-plan server " ] && ok user-plan-probe-between-the-lease-and-the-first-server || bad user-plan-probe-between-the-lease-and-the-first-server "order [$order]"
  # shellcheck disable=SC2144 # a glob of the probes' dumps, read by grep -l
  [ "$(grep -l '^CUDA_VISIBLE_DEVICES=' "$C"/envprobe.up.* 2> /dev/null | wc -l | tr -d ' ')" = 0 ] && [ "$(compgen -G "$C/envprobe.up.*" | wc -l | tr -d ' ')" = 1 ] &&
    ok user-plan-probe-has-no-cuda-visible-devices || bad user-plan-probe-has-no-cuda-visible-devices "the probe ran with CUDA_VISIBLE_DEVICES, or not once"
  # shellcheck disable=SC2144
  [ "$(grep -l '^CUDA_VISIBLE_DEVICES=' "$C"/env.up.* 2> /dev/null | wc -l | tr -d ' ')" = 1 ] && ok user-plan-server-keeps-cuda-visible-devices || bad user-plan-server-keeps-cuda-visible-devices "the server ran without CUDA_VISIBLE_DEVICES"
  st_case userplan-off
  st_tree up 20
  ST_ENV=(BLOOMERY_AB_ROUNDS=1 ST_PLAN_USER_CARD=Other DEPTH_SERVE_USER_PLAN=off)
  st_run up:6
  st_check user-plan-off 0 "ROW r1 up/place=a/ctx=16384 D=6"
  st_absent user-plan-off-no-probe "[user-plan]"
  if compgen -G "$C/envprobe.*" > /dev/null; then bad user-plan-off-starts-no-probe "a probe ran"; else ok user-plan-off-starts-no-probe; fi

  # --- the cold tag: 0.3.10's speed sitting read its first row at 34323 major faults (78.50 against 99.83 tok/s) into a
  # mean of two; the row is [cold], out of the headline mean and counted in its line (cold-blocks.sh's rule).
  st_case cold
  st_tree w 22.18
  printf '0\n0\n34323\n34323\n34325\n' > "$C/mf.seq"
  ST_ENV=(BLOOMERY_AB_ROUNDS=2 "ST_TPSSEQ=22.18 20.54" DEPTH_SERVE_CTX=4300)
  st_run w:512
  st_check cold-row 0 "| majflt 34323 [cold]" "[cold] r1 w D=512: 34323 major faults x 75 us" "of the 6.45 s window" "ROW r2 w/place=a/ctx=4300 D=512 | tg_tok/s 20.54 | pp_tok/s 5120.00 | load_s"
  st_check cold-mean 0 "mean w/place=a/ctx=4300 D=512 | tg 20.54 tok/s (n=1) | pp 5120.00 tok/s (n=1)" "[cold 1/2]"
  [ "$(grep '^ROW r2 ' "$C/out" | grep -c '\[cold\]')" = 0 ] && ok cold-warm-row-untagged || bad cold-warm-row-untagged "round 2's row is tagged"
  st_case tables-cold
  cat > "$C/rows" << 'ROWS'
ROW r1 rel0310 D=512 | tg_tok/s 22.18 | pp_tok/s 78.50 | load_s 39 | majflt 34323 [cold]
ROW r1 rel0310 D=4096 | tg_tok/s 20.81 | pp_tok/s 97.99 | load_s 15 | majflt 14
ROW r2 rel0310 D=4096 | tg_tok/s 21.21 | pp_tok/s 95.52 | load_s 15 | majflt 2
ROW r2 rel0310 D=512 | tg_tok/s 20.54 | pp_tok/s 99.83 | load_s 14 | majflt 0
ROWS
  RC=0
  bash "$HERE/depth-serve.sh" --summary "$C/rows" > "$C/out" 2>&1 || RC=$?
  st_check tables-cold-out-of-the-mean 0 "mean rel0310 D=512 | tg 20.54 tok/s (n=1) | pp 99.83 tok/s (n=1) | load 26.5 s | majflt 34323 | [cpu-busy 0/2] [other-busy 0/2] [cold 1/2]" \
    "mean rel0310 D=4096 | tg 21.01 tok/s (sd 0.28 = 1.35 %, [20.81..21.21], n=2) | pp 96.75 tok/s (sd 1.75 = 1.81 %, [95.52..97.99], n=2) | load 15.0 s | majflt 16 | [cpu-busy 0/2] [other-busy 0/2] [cold 0/2]"
  cat > "$C/rows" << 'ROWS'
ROW r1 a D=6 | tg_tok/s 20.00 | pp_tok/s 60.00 | load_s 41 | majflt 0
ROW r2 a D=6 | tg_tok/s 22.00 | pp_tok/s 60.00 | load_s 41 | majflt 0
ROW r3 a D=6 | tg_tok/s 99.00 | pp_tok/s 60.00 | load_s 41 | majflt 9 [cold]
ROW r1 b D=6 | tg_tok/s 21.00 | pp_tok/s 60.00 | load_s 41 | majflt 0
ROW r2 b D=6 | tg_tok/s 23.00 | pp_tok/s 60.00 | load_s 41 | majflt 0
ROW r3 b D=6 | tg_tok/s 20.00 | pp_tok/s 60.00 | load_s 41 | majflt 0
ROW r1 c D=6 | tg_tok/s 20.00 | pp_tok/s 60.00 | load_s 41 | majflt 7 [cold]
ROWS
  RC=0
  bash "$HERE/depth-serve.sh" --summary "$C/rows" > "$C/out" 2>&1 || RC=$?
  st_check tables-cold-arms 0 "mean a D=6 | tg 21.00 tok/s (sd 1.41 = 6.73 %, [20.00..22.00], n=2) | pp 60.00 tok/s (sd 0.00 = 0.00 %, [60.00..60.00], n=2) | load 41.0 s | majflt 9 | [cpu-busy 0/3] [other-busy 0/3] [cold 1/3]" \
    "diff b vs a D=6 | tg +0.33 tok/s (+1.59 %), no interval: 3 rounds against 2 | pp +0.00 tok/s (+0.00 %), no interval: 3 rounds against 2" \
    "mean c D=6 | no warm row: every one of its 1 row(s) is [cold]" \
    "diff c vs a D=6 | no diff: c has no warm row"

  # --- DEPTH_SERVE_AFTER: on round 1's last server of each arm group, after its timed request and before the server
  # stops, with the server's address, pid, log and a directory of its own; never in round 2, never before the request.
  st_case hook
  st_tree a 20
  st_tree b 21
  cat > "$C/hook.sh" << 'HOOK'
echo "hook-ran last=$(tail -n 1 "$TMPDIR/stub-srv-reqs" | cut -d' ' -f1-2) health=$(curl -s -o /dev/null -w '%{http_code}' "$BLOOMERY_SESSION_URL/health") pid=$(kill -0 "$BLOOMERY_SESSION_PID" 2> /dev/null && echo alive) log=$(basename "$BLOOMERY_SESSION_STDERR") dir=$([ -d "$BLOOMERY_SESSION_DIR" ] && echo ok) url=$BLOOMERY_SESSION_URL"
HOOK
  ST_PORT=$(st_port)
  ST_ENV=(BLOOMERY_AB_ROUNDS=2 "DEPTH_SERVE_AFTER=bash $C/hook.sh")
  st_run a:6 a:12 b:6
  st_check hook-run 0 "[hook] r1 a D=12: DEPTH_SERVE_AFTER after its timed request" "hook-ran last=1 12 health=200 pid=alive log=r1-a-D12.log dir=ok url=http://127.0.0.1:$ST_PORT" \
    "[hook] r1 b D=6: DEPTH_SERVE_AFTER after its timed request" "hook-ran last=1 6 health=200 pid=alive log=r1-b-D6.log dir=ok url=http://127.0.0.1:$ST_PORT" \
    "HOOK r1 a D=12 rc=0" "HOOK r1 b D=6 rc=0"
  [ "$(grep -c '^hook-ran ' "$C/out")" = 2 ] && ok hook-once-per-group-in-round-1 || bad hook-once-per-group-in-round-1 "the hook did not run exactly twice"
  st_servers hook-servers-stopped 6
  ST_PORT=
  st_case hookfail
  st_tree a 20
  ST_ENV=(BLOOMERY_AB_ROUNDS=2 "DEPTH_SERVE_AFTER=echo red; exit 3")
  st_run a:6
  st_check hook-failure-is-a-red-run 1 "HOOK r1 a D=6 rc=3" "[hook] 1 of 1 hook run(s) failed" "ROW r2 a/place=a/ctx=16384 D=6" "mean a/place=a/ctx=16384 D=6"
  st_case hookbefore
  RC=0
  ( A_LABEL=(a)
    A_DEP=(6)
    LOGDIR=$C AFTER='echo must-not-run' AFTER_BOUND=5 SRV_PID=123 SRV_PORT=1 SRV_LOG=$C/log SRV_TIMED_PID=''
    ds_hook_run 1 0 ) > "$C/out" 2>&1 || RC=$?
  st_check refuse-hook-before-the-timed-request 70 "ds_hook_run: refused: the hook of r1 a D=6 would run before the timed request of its server"
  st_absent refuse-hook-before-the-timed-request-ran-nothing "must-not-run"

  # --- the memory between servers: the next load starts when the card's used memory is back at the sitting's baseline.
  st_case mem
  st_tree a 20
  st_tree b 21
  printf '100\n100\n1250\n1250\n130\n' > "$C/mem.seq"
  ST_ENV=(BLOOMERY_AB_ROUNDS=1)
  st_run a:6 b:6
  st_check mem-wait-returns 0 "[mem-baseline] the sitting's baseline is 100 MiB" "[mem-wait] r1 b D=6: used 1250 MiB, baseline 100 MiB + 128 MiB slack: waiting for the last server's memory" \
    "[mem-wait] r1 b D=6: back at 130 MiB" "ROW r1 b/place=a/ctx=16384 D=6"
  st_absent mem-wait-quiet-when-clean "[mem-wait] r1 a D=6"
  st_case memstuck
  st_tree a 20
  st_tree b 21
  printf '100\n100\n1250\n' > "$C/mem.seq"
  ST_ENV=(BLOOMERY_AB_ROUNDS=1 DEPTH_SERVE_MEM_WAIT_S=2)
  st_run a:6 b:6
  st_check refuse-mem-not-back 1 "FAIL r1 b D=6: the timing card" "holds 1250 MiB, above the sitting's baseline 100 MiB + 128 MiB slack, 2 s after the last server stopped"
  st_absent refuse-mem-not-back-no-row "ROW r1 b"
  st_servers refuse-mem-not-back-one-server 1
  st_case memlost
  st_tree a 20
  st_tree b 21
  printf '100\n100\nnone\n' > "$C/mem.seq"
  ST_ENV=(BLOOMERY_AB_ROUNDS=1)
  st_run a:6 b:6
  st_check refuse-mem-no-reading 1 "FAIL r1 b D=6: the timing card" "gives no used-memory reading now (100 MiB at the sitting's baseline)"
  ST_ENV=()

  # --- local mode (tools/box.sh's BLOOMERY_BOX_CARD=local): $T/local's nvidia-smi lists one made-up card and answers
  # its name, and once STUB_GONE is set no longer does; its journalctl shows a kernel line, under STUB_XID an Xid line
  # once every server the case started has stopped, and fails under STUB_JOURNAL=fail. The box
  # (/root/bloomery-cpu.lock) refuses local mode.
  if [ -e /root/bloomery-cpu.lock ]; then
    echo "skip local, local-journal, local-cards, local-xid, local-xid-failed-arm, local-gone, local-tree-elsewhere, local-tree-here: /root/bloomery-cpu.lock exists (the box), where local mode is refused"
  else
    gb10=GPU-22222222-2222-2222-2222-222222222222
    mkdir -p "$T/local"
    cat > "$T/local/nvidia-smi" << SMI
#!/bin/sh
case "\$*" in
  *--query-gpu=uuid\\ *) echo $gb10 ;;
  *--query-gpu=name\\ *)
    [ -z "\$STUB_GONE" ] || { echo 'GPU is lost (stub)' >&2; exit 15; }
    echo 'NVIDIA GB10 (stub)'
    ;;
esac
SMI
    cat > "$T/local/journalctl" << 'JCTL'
#!/bin/sh
[ "$STUB_JOURNAL" != fail ] || { echo 'journalctl: the stub fails' >&2; exit 1; }
echo '1790428800.000000 ws kernel: Linux version (stub)'
[ -n "$STUB_XID" ] || exit 0
if [ -r "$TMPDIR/stub-srv-pids" ]; then
  while read -r p; do ! kill -0 "$p" 2> /dev/null || exit 0; done < "$TMPDIR/stub-srv-pids"
fi
echo '1790428806.570394 ws kernel: NVRM: Xid (PCI:000f:01:00): 13, pid=4242, name=bloomery-serve, Graphics Exception'
JCTL
    chmod +x "$T/local/nvidia-smi" "$T/local/journalctl"
    ST_LOCAL=1
    st_case local
    st_tree a 20
    ST_ENV=(BLOOMERY_AB_ROUNDS=1)
    st_run a:6
    st_check local 0 "[timing-card] Xid reader: journalctl -k" "[lease] held by pid" "[timing-card] Xid count from @" \
      "timing_gpu=$gb10 other_gpu= CUDA_VISIBLE_DEVICES=$gb10 " "ROW r1 a/place=a/ctx=16384 D=6 | tg_tok/s 20.00"
    st_absent local-refuses-nothing "refused"
    st_case local-journal
    st_tree a 20
    ST_ENV=(BLOOMERY_AB_ROUNDS=1 STUB_JOURNAL=fail)
    st_run a:6
    st_check local-journal 69 "depth-serve.sh: refused before the lease: journalctl -k does not answer (rc 1)"
    st_absent local-journal-takes-no-lease "[lease]"
    st_case local-cards
    st_tree a 20
    ST_ENV=(BLOOMERY_AB_ROUNDS=1 BLOOMERY_TIMING_CARDS=a6000+3090 STUB_JOURNAL=fail)
    st_run a:6
    st_check local-cards 64 "[timing-card] refused: BLOOMERY_TIMING_CARDS=a6000+3090 and BLOOMERY_BOX_CARD=local"
    st_absent local-cards-before-the-journal "journalctl" "[lease]"
    ST_ENV=(BLOOMERY_AB_ROUNDS=1 BLOOMERY_TIMING_CARDS=a6000+3090 BLOOMERY_DRY=1)
    st_run a:6
    st_check local-cards-dry 64 "[timing-card] refused: BLOOMERY_TIMING_CARDS=a6000+3090 and BLOOMERY_BOX_CARD=local"
    st_absent local-cards-dry-no-plan "[plan]"
    st_case local-xid
    st_tree a 20
    ST_ENV=(BLOOMERY_AB_ROUNDS=1 STUB_XID=1)
    st_run a:6
    st_check local-xid 1 "FAIL r1 a D=6: 1 NVRM Xid line(s) since the last arm's check (1 since the lease was taken); last: 1790428806.570394 ws kernel: NVRM: Xid"
    st_absent local-xid-no-row "ROW "
    st_servers local-xid-server-stopped 1
    st_case local-xid-failed-arm
    st_tree nt 20 STUB_SRV_NOTIMINGS=1
    ST_ENV=(BLOOMERY_AB_ROUNDS=1 STUB_XID=1)
    st_run nt:6
    st_check local-xid-failed-arm 1 "FAIL r1 nt D=6: no timings in the /completion response" \
      "; 1 NVRM Xid line(s) since the last arm's check (1 since the lease was taken)"
    st_case local-gone
    st_tree a 20
    ST_ENV=(BLOOMERY_AB_ROUNDS=1 STUB_GONE=1)
    st_run a:6
    st_check local-gone 1 "FAIL r1 a D=6: the timing card ($gb10) does not answer nvidia-smi (rc 15)"
    [ "$(grep -o 'does not answer nvidia-smi' "$C/out" | wc -l | tr -d ' ')" = 1 ] && ok local-gone-named-once || bad local-gone-named-once "the lost card is not named exactly once"
    # An arm named as this tree: ~/repo/<name> another tree of that name is refused, ~/repo/<name> a link to this
    # tree is not.
    own=${ROOT##*/}
    if [[ $own =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
      st_case local-tree-elsewhere
      st_tree "$own" 20
      ST_ENV=(BLOOMERY_AB_ROUNDS=1)
      st_run "$own:6"
      st_check local-tree-elsewhere 64 "arm '$own': $C/home/repo/$own is not this runner's tree $ROOT"
      st_case local-tree-here
      ln -s "$ROOT" "$C/home/repo/$own"
      ST_ENV=(BLOOMERY_AB_ROUNDS=1 BLOOMERY_DRY=1)
      st_run "$own:6"
      st_check local-tree-here 0 "[plan] arm 1: $own D=6:"
      st_absent local-tree-here-not-refused "is not this runner's tree"
    else
      echo "skip local-tree-elsewhere, local-tree-here: this tree's name '$own' is no tree name"
    fi
    ST_LOCAL=''
    ST_ENV=()
  fi

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
