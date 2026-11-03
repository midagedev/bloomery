#!/usr/bin/env bash
# bloomery — the Mac static tier in one command, the round loop's form: fmt-check, lint, the scoped combos and every
# tools/check-*.sh, check-recipes as its blocks (tools/check-recipes.sh --jobs), all of them jobs of one scheduler,
# tools/static-sched.py: at most a cpu count at once (lint and combos count the cores they compile on), the longest
# recorded first (~/.cache/bloomery/static-times.tsv), combos after lint (one target directory, one cargo lock). Every
# job at once oversubscribed the Mac and turned timing self-tests red under the load alone.
#
# A run with a BASE keys every job but lint and combos (tools/recipes.py static-key: the content of the tree the job
# reads, the commit of a revision its keyspec names, the command and the environment, the rustflags among it) and skips
# a job whose key is green in the ledger; a skipped step or block is a summary line of its own. Lint and combos source
# $HOME/opt/bloomery-mac-env.sh (tools/mac-check.sh), which no key reads and which may export rustflags: a key green
# under one setup would skip the lint under another. --fresh keys and records but skips nothing (a proof run); a full
# run (no BASE) keys nothing.
#
# The cargo lane runs `lint`, not `check`: clippy is rustc plus lints, so every error `cargo check`
# reports is a clippy error too (proved FAIL-first at birth: a type-error mutant is red under this
# loop's lint step), and the two keep separate artifacts — a `check` pass after `lint` rebuilds the
# workspace members it touched (measured: 23 crates), which the loop does not need. `mac-check.sh
# check` stays as a mode; the box's landing runs it there.
#
# The combos step is scoped: `mac-check.sh combos --base BASE --ledger LEDGER` checks only the shapes
# an input file of which changed since BASE, or whose input key is not green in the ledger, and a
# green shape lands in the ledger — a rerun at unchanged inputs runs none (the full `combos`, and
# `mac-static` with no BASE, stay the lead's landing form). A change to tools/recipes.py or
# tools/mac-check.sh moves every key (they are in every shape's inputs), so a tooling round re-checks
# every shape.
#
#   tools/mac-static.sh [--fresh] [BASE|--full]   one line a step (rc, wall, the step's own summary), exits
#                                       non-zero naming every red step; BASE scopes the combos step
#                                       (a rev, or A..B) and keys the jobs; --full or no argument runs
#                                       it unscoped and unkeyed
#   tools/mac-static.sh --self-test     the job list, the scheduling (through tools/static-sched.py on stub jobs), the
#                                       rc/wall capture, the skip lines, the red-step naming, the check-script
#                                       derivation, the parallel check runner and the argument plumbing, against stub
#                                       jobs and fixture check scripts; runs no cargo and no check script
#                                       (check-recipes runs it)
#   tools/mac-static.sh --checks [--list] [--no-base] [--skip NAME]…
#                                       the check scripts alone, in parallel, at most one a cpu (what `just gate` runs:
#                                       the justfile's check-scripts): one line a check in byte order of the name
#                                       (rc, wall, the last line of its log), then every red check's log whole and the
#                                       red names; every check runs, a red one stops no other, and the exit is the first
#                                       red check's rc in that order. Each run keeps its results in a directory of its own
#                                       (target/mac-static/checks/run.XXXXXX, removed when the run ends), so runs that overlap
#                                       read their own rcs. --skip leaves a check out; --no-base runs
#                                       check-model-tax on its tables alone, with no revision to hold them to (a checkout
#                                       with no origin/main: tools/contrib-check.sh); --list prints each name it would
#                                       run with its arguments, one a line, and runs none
#
# The check scripts are every tools/check-*.sh, in byte order of the name, but those SKIP names: a new script joins the
# loop, `just gate` and tools/contrib-check.sh by existing, with no list to add its name to. A skip that names no script is
# refused (64), so a stale entry cannot read as a script left out on purpose.
#
# Exit: the first red step's rc, or 0. 64: an argument this script does not take. 69: no cpu count (getconf, sysctl and
# nproc all fail), or results that cannot be cleared, created or removed (named; a run that already ended red keeps its rc). A step
# whose rc file is missing or empty (killed before it ended, or a result that could not be written) is red, named `no rc`,
# never 0. Jobs' full output and their rc/wall land in target/mac-static/run.XXXXXX/<job>.{log,rc,wall} (a skipped one also
# <job>.skip; sched.tsv is the run's timeline; --checks: a run directory under target/mac-static/checks/, whose red logs come
# back whole on stderr); check-recipes' step files are its blocks' report; the cargo steps also write the mac-check logs
# (target/mac-check-*.log) as they always did.
#
# The environment is mac-check's (the pinned toolchain, $HOME/opt/bloomery-mac-env.sh for the cross
# check): every cargo step calls tools/mac-check.sh, which owns that setup. The ledger is
# ~/.cache/bloomery/mac-static-ledger.tsv, shared between trees on purpose: a green key is a fact
# about the inputs, not the tree; the shapes' keys and the jobs' keys hash different headers, so they never meet.
# Another round's tree has its own target directory; nothing here reads another tree's.
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
LEDGER=$HOME/.cache/bloomery/mac-static-ledger.tsv
# The check scripts the round loop leaves out, by name (tools/check-NAME.sh), the reason on the line above each. None today.
SKIP=()
TIMES=$HOME/.cache/bloomery/static-times.tsv
# The cores a cargo step holds in the schedule: cargo compiles on every core while a crate graph allows.
CARGO_CPUS=4

say() { printf '%s\n' "$*" >&2; }

# check_names DIR SKIP…: the NAMEs of DIR's check-NAME.sh in byte order, one a line, but SKIP's. A SKIP that names no script and
# a DIR with no script are refused (64).
check_names() {
  local dir=$1 f n s hit
  local -a all=()
  shift
  for f in "$dir"/check-*.sh; do
    [ -e "$f" ] || { say "mac-static.sh: no check-*.sh in $dir"; return 64; }
    n=${f##*/check-}
    all+=("${n%.sh}")
  done
  for s in "$@"; do
    hit=
    for n in "${all[@]}"; do [ "$n" != "$s" ] || hit=1; done
    [ -n "$hit" ] || { say "mac-static.sh: skip '$s' names no $dir/check-$s.sh"; return 64; }
  done
  for n in "${all[@]}"; do
    hit=
    for s in "$@"; do [ "$n" != "$s" ] || hit=1; done
    [ -n "$hit" ] || printf '%s\n' "$n"
  done | LC_ALL=C sort
}

names=$(check_names "$HERE/tools" ${SKIP[@]+"${SKIP[@]}"}) || exit $?
SCRIPTS=()
while IFS= read -r n; do SCRIPTS+=("$n"); done <<< "$names"
unset names n

# combos_argv BASE LEDGER: the mac-check combos argv for the combos step (the scoped form only when
# BASE is non-empty — the full form is the lead's landing one).
combos_argv() {
  local a=(combos)
  [ -z "$1" ] || a+=(--base "$1")
  [ -z "$2" ] || [ -z "$1" ] || a+=(--ledger "$2")
  printf '%s\n' "${a[@]}"
}

# check_argv NAME BASE NOBASE: the arguments of tools/check-NAME.sh. check-model-tax holds its rows to the revision BASE names (the
# A of A..B; origin/main on a full run), so an edit of its tables cannot raise a count unargued; a non-empty NOBASE is the run
# with no revision to hold them to (its tables alone: a checkout with no origin/main).
check_argv() {
  case $1 in
    model-tax)
      if [ -n "${3:-}" ]; then printf '%s\n' check; else printf '%s\n' check --base "$(base_rev "$2")"; fi
      ;;
  esac
}
# base_rev BASE: the revision a base names, origin/main when there is none.
base_rev() {
  local b=${1%%..*}
  printf '%s\n' "${b:-origin/main}"
}

# ncpu: the cpu count, the bound of --checks' parallel jobs. No count is refused (69), never a guessed one.
ncpu() {
  local n
  n=$(getconf _NPROCESSORS_ONLN 2> /dev/null || sysctl -n hw.ncpu 2> /dev/null || nproc 2> /dev/null || true)
  case $n in
    '' | *[!0-9]* | 0) say "mac-static.sh: no cpu count (getconf _NPROCESSORS_ONLN, sysctl -n hw.ncpu and nproc all failed)"; return 69 ;;
  esac
  printf '%s\n' "$n"
}

# step_inner NAME D CMD…: one step, serial in its caller — its rc, wall and full output to D. The rc is the file's, never the
# wrapper's; a result that cannot be written is named and ends 69 (the rc file, written last, is then missing: the summary reads
# `no rc`).
step_inner() {
  local name=$1 d=$2 t0=$SECONDS rc=0
  shift 2
  "$@" > "$d/$name.log" 2>&1 < /dev/null || rc=$?
  { echo "$((SECONDS - t0))" > "$d/$name.wall" && echo "$rc" > "$d/$name.rc"; } \
    || { say "mac-static.sh: cannot write $name's result in $d"; return 69; }
}

# clear_results D: D's rc, wall and pids files of an earlier run removed. One that stays would read as this run's result, so a
# failure is named and ends 69.
clear_results() {
  rm -f "$1"/*.rc "$1"/*.wall "$1"/pids || { say "mac-static.sh: cannot clear the earlier results in $1"; return 69; }
}

# remove_run_dir DIR RC: the EXIT trap of a --checks run (RC is the status it ends with): DIR removed. One that stays is named and
# ends the run 69, unless it ended red already.
remove_run_dir() {
  rm -rf "$1" || { say "mac-static.sh: cannot remove the results in $1"; [ "$2" != 0 ] || exit 69; }
}

# step_keyspec NAME BASE: what a check script's input key reads (tools/recipes.py static-key). Each scans the tree;
# check-model-tax also holds its rows to the base revision.
step_keyspec() {
  case $1 in
    model-tax) printf 'tree,rev:%s\n' "$(base_rev "$2")" ;;
    *) echo tree ;;
  esac
}

# job_lines BASE CARGO_CMDV…: every job of the run as tools/static-sched.py's JOBS lines, run from this tree's root:
# fmt-check, lint, combos after lint, each check script, check-recipes as its blocks. A command carries its arguments
# quoted (%q) for the scheduler's `bash -c`: a ledger path or a revision with spaces stays one argument.
job_lines() {
  local base=$1 s w cmdv
  shift
  local -a args
  printf 'fmt-check\t-\ttree\ttools/mac-check.sh fmt-check\n'
  printf 'lint\tcpus=%s\t-\ttools/mac-check.sh lint\n' "$CARGO_CPUS"
  printf -v cmdv '%q ' "$@"
  printf 'combos\tafter=lint,cpus=%s\t-\ttools/mac-check.sh %s\n' "$CARGO_CPUS" "${cmdv% }"
  for s in "${SCRIPTS[@]}"; do
    if [ "$s" = recipes ]; then
      bash "$HERE/tools/check-recipes.sh" --jobs
      continue
    fi
    args=()
    while IFS= read -r w; do args+=("$w"); done < <(check_argv "$s" "$base")
    printf -v cmdv '%q ' ${args[@]+"${args[@]}"}
    printf 'check-%s\t-\t%s\tbash tools/check-%s.sh%s\n' "$s" "$(step_keyspec "$s" "$base")" "$s" "${args[*]+ ${cmdv% }}"
  done
}

# run_jobs D JOBS MODE: the scheduler over JOBS from this tree's root, its files to D. MODE: full (no keys), keyed (a
# job whose key is green is skipped), fresh (keyed and recorded, nothing skipped). Then check-recipes' step files from
# its blocks' report, the step's wall being the span from its first block's start to its last block's end.
run_jobs() {
  local d=$1 jobs=$2 mode=$3 rc=0
  local -a keyed=()
  if [ "$mode" != full ]; then
    (cd "$HERE" && python3 tools/recipes.py static-key "$jobs") > "$d/keys.tsv" || {
      say "mac-static: tools/recipes.py static-key failed; every job runs"
      mode=full
    }
  fi
  [ "$mode" = full ] || keyed=(--keys "$d/keys.tsv" --ledger "$LEDGER" --tree "${HERE##*/}")
  [ "$mode" != fresh ] || keyed+=(--no-skip)
  (cd "$HERE" && python3 tools/static-sched.py run --out "$d" --times "$TIMES" ${keyed[@]+"${keyed[@]}"} "$jobs") || rc=$?
  [ "$rc" = 0 ] || { say "mac-static: tools/static-sched.py ended rc $rc"; return "$rc"; }
  if grep -q '^check-recipes\.' "$jobs"; then
    rc=0
    bash "$HERE/tools/check-recipes.sh" --report "$d" > "$d/check-recipes.log" 2>&1 || rc=$?
    echo "$rc" > "$d/check-recipes.rc"
    awk -F'\t' '$1 ~ /^check-recipes\./ && $5 == "run" { if (s == "" || $2 < s) s = $2; if ($3 > e) e = $3 }
      END { printf "%d\n", (s == "" ? 0 : e - s) }' "$d/sched.tsv" > "$d/check-recipes.wall"
  fi
}

# run_check_jobs DIR BASE NOBASE JOBS D NAME…: DIR's check-NAME.sh for each NAME, in the order given, at most JOBS at a time under
# its check_argv; each one's output, rc and wall land in D (step_inner). The window is the jobs launched minus the jobs waited
# on, and a full one waits for the oldest pid (the Mac's bash 3.2 has no `wait -n`). Every pid is captured at start, written to
# D/pids, and waited on; no polling.
run_check_jobs() {
  local dir=$1 base=$2 nobase=$3 jobs=$4 d=$5 n w i=0
  local -a pids=() args
  shift 5
  : > "$d/pids" || { say "mac-static.sh: cannot write $d/pids"; return 69; }
  for n in "$@"; do
    [ $((${#pids[@]} - i)) -lt "$jobs" ] || { wait "${pids[$i]}" || true; i=$((i + 1)); }
    args=()
    while IFS= read -r w; do args+=("$w"); done < <(check_argv "$n" "$base" "$nobase")
    ( step_inner "check-$n" "$d" bash "$dir/check-$n.sh" ${args[@]+"${args[@]}"} ) &
    pids+=($!)
    echo "check-$n $!" >> "$d/pids"
  done
  [ "${#pids[@]}" = 0 ] || wait "${pids[@]}" || true # a killed wrapper is the summary's `no rc` case, not this script's end
}

# run_checks DIR BASE NOBASE LIST JOBS D SKIP…: DIR's check scripts (check_names: all but SKIP) through run_check_jobs, their
# output, rc and wall in D, then summarize_checks. LIST non-empty prints each NAME and its arguments, one a line, and runs none.
run_checks() {
  local dir=$1 base=$2 nobase=$3 list=$4 jobs=$5 d=$6 listed n w
  local -a names=() args
  shift 6
  listed=$(check_names "$dir" "$@") || return $?
  [ -n "$listed" ] || return 0 # every check skipped: a here-string of nothing would still read one empty name
  while IFS= read -r n; do names+=("$n"); done <<< "$listed"
  if [ -n "$list" ]; then
    for n in "${names[@]}"; do
      args=()
      while IFS= read -r w; do args+=("$w"); done < <(check_argv "$n" "$base" "$nobase")
      printf '%s\n' "$n${args[*]+ ${args[*]}}"
    done
    return 0
  fi
  mkdir -p "$d" || { say "mac-static.sh: cannot create $d"; return 69; }
  clear_results "$d" || return $?
  run_check_jobs "$dir" "$base" "$nobase" "$jobs" "$d" "${names[@]}" || return $?
  summarize_checks "$d" "$jobs" "${names[@]}"
}

# step_line D NAME: the step's summary line on stdout, `mac-static: <step> rc <n> in <s> s [— the step's own closing line]`: the
# cargo steps' closing line comes from their log, check-recipes' is its count of blocks run and skipped, a check's is its log's
# last non-empty line (cut at 160 characters), and a skipped step's names its green row. A step that
# is red, or whose rc file is missing, empty or not a number, joins REDS; FIRSTRC is the rc of the first red one (1 for no rc), 0
# while none is.
REDS=()
FIRSTRC=0
step_line() {
  local d=$1 name=$2 rc='' wall extra=
  [ ! -f "$d/$name.rc" ] || rc=$(cat "$d/$name.rc" 2> /dev/null) || rc=
  case $rc in
    '' | *[!0-9]*)
      echo "mac-static: $name rc ? no rc in $((SECONDS - T0)) s (no rc file: killed before its wrapper ran, or its result could not be written)"
      REDS+=("$name (no rc)")
      [ "$FIRSTRC" != 0 ] || FIRSTRC=1
      return 0
      ;;
  esac
  wall=$(cat "$d/$name.wall")
  case $name in
    lint) extra=$(grep -E '^mac-check: (lint \^warning|lint rc)' "$d/$name.log" | tail -1 || true) ;;
    combos) extra=$(grep -E '^mac-check: combos rc' "$d/$name.log" | tail -1 || true) ;;
    fmt-check) extra="rustfmt diff blocks: $(grep -cE '^diff' "$d/$name.log" || true)" ;;
    check-recipes) extra=$(grep -E '^check-recipes: [0-9]+ blocks ran' "$d/$name.log" | tail -1 || true) ;;
    check-*) extra=$(grep -v '^[[:space:]]*$' "$d/$name.log" | tail -1 | cut -c1-160 || true) ;;
  esac
  [ ! -f "$d/$name.skip" ] || extra="skipped, its input key is $(cat "$d/$name.skip")"
  echo "mac-static: $name rc $rc in $wall s${extra:+ — $extra}"
  if [ "$rc" != 0 ]; then
    REDS+=("$name (rc $rc)")
    [ "$FIRSTRC" != 0 ] || FIRSTRC=$rc
  fi
}

# summarize_checks D JOBS NAME…: one step_line a check in the order given, a total line, then the verdict: ok, or every red check's
# log whole (stderr) and one line naming the red ones, ending with the first red check's rc.
summarize_checks() {
  local d=$1 jobs=$2 n
  shift 2
  REDS=()
  FIRSTRC=0
  for n in "$@"; do step_line "$d" "check-$n"; done
  echo "mac-static: $# checks in $((SECONDS - T0)) s, at most $jobs at once"
  if [ "${#REDS[@]}" != 0 ]; then
    for n in "$@"; do
      [ -f "$d/check-$n.rc" ] && [ "$(cat "$d/check-$n.rc")" != 0 ] || continue
      say "mac-static: ---- check-$n, its log ----"
      cat "$d/check-$n.log" >&2
    done
    say "mac-static: red checks: ${REDS[*]}"
    return "$FIRSTRC"
  fi
  echo "mac-static: checks ok"
}

# summarize D BASE: one line a step in a fixed order (fmt-check, lint, combos, every check),
# then the verdict. Prints `mac-static: <step> rc <n> in <s> s [— the step's own closing line]`; the
# cargo steps' closing line comes from their log, check-recipes' is its count of blocks run and skipped, a
# skipped step's names its green row. Then one line naming every skipped job (steps and check-recipes' blocks)
# when any was. Ends 0, or prints one line naming every red step and ends with the first red rc.
summarize() {
  local d=$1 base=$2 s f skipped=''
  REDS=()
  FIRSTRC=0
  step_line "$d" fmt-check
  step_line "$d" lint
  step_line "$d" combos
  for s in "${SCRIPTS[@]}"; do step_line "$d" "check-$s"; done
  for f in "$d"/*.skip; do
    [ -e "$f" ] || continue
    f=${f##*/}
    skipped="$skipped ${f%.skip}"
  done
  [ -z "$skipped" ] || echo "mac-static: skipped by a green input key:$skipped"
  echo "mac-static: total $((SECONDS - T0)) s, base ${base:-none (full run)}, ledger ${base:+$LEDGER}"
  if [ "${#REDS[@]}" != 0 ]; then
    say "mac-static: red steps: ${REDS[*]}"
    return "$FIRSTRC"
  fi
  echo "mac-static: ok"
}

self_test() {
  local fails=0 d out rc k in_scripts in_skip sp cmd got
  local -a pids=()
  fail() { say "mac-static self-test FAIL: $*"; fails=$((fails + 1)); }
  T0=$SECONDS

  # combos_argv: the full form without a base, the scoped form with both, no ledger without a base
  out=$(combos_argv '' "$LEDGER" | tr '\n' ' ')
  [ "$out" = "combos " ] || fail "no base derives '$out'"
  out=$(combos_argv main '' | tr '\n' ' ')
  [ "$out" = "combos --base main " ] || fail "a base without a ledger derives '$out'"
  out=$(combos_argv 'HEAD~3..HEAD' "$HOME/x.tsv" | tr '\n' ' ')
  [ "$out" = "combos --base HEAD~3..HEAD --ledger $HOME/x.tsv " ] || fail "a base and a ledger derive '$out'"

  # check_argv: only check-model-tax takes arguments, a base compares its rows with, origin/main when the run has none
  out=$(check_argv model-tax '' | tr '\n' ' ')
  [ "$out" = "check --base origin/main " ] || fail "a full run derives model-tax '$out'"
  out=$(check_argv model-tax 0123abc | tr '\n' ' ')
  [ "$out" = "check --base 0123abc " ] || fail "a base derives model-tax '$out'"
  out=$(check_argv model-tax 'a1..b2' | tr '\n' ' ')
  [ "$out" = "check --base a1 " ] || fail "a range derives model-tax '$out'"
  out=$(check_argv model-tax 0123abc 1 | tr '\n' ' ')
  [ "$out" = "check " ] || fail "a run with no base derives model-tax '$out', not its tables alone"
  out=$(check_argv arch main | tr '\n' ' ')
  [ -z "$out" ] || fail "a check with no arguments derives '$out'"

  # every tools/check-*.sh is a step or on SKIP, never both: a check script the loop leaves out for no reason never runs in it
  local f name unlisted='' doubled=''
  for f in "$HERE"/tools/check-*.sh; do
    name=${f##*/check-}
    name=${name%.sh}
    case " ${SCRIPTS[*]} " in *" $name "*) in_scripts=1 ;; *) in_scripts= ;; esac
    case " ${SKIP[*]+${SKIP[*]}} " in *" $name "*) in_skip=1 ;; *) in_skip= ;; esac
    [ -z "$in_scripts" ] || [ -z "$in_skip" ] || doubled="$doubled $name"
    [ -n "$in_scripts" ] || [ -n "$in_skip" ] || unlisted="$unlisted $name"
  done
  [ -z "$unlisted" ] || fail "tools/check-*.sh that is neither a step nor on SKIP:$unlisted"
  [ -z "$doubled" ] || fail "tools/check-*.sh that is a step and on SKIP:$doubled"

  # check_names and run_checks on fixture check scripts: the derivation (byte order, the skip), the refusals (a skip that names
  # no script, a directory with none), the arguments (model-tax's base, or none), every check running whatever one's rc, one log a
  # check, and --list running none
  local cd_ lines
  cd_=$(mktemp -d)
  for name in aa modelsx model-tax red skipme zz; do
    printf '#!/usr/bin/env bash\necho "$@" >> "%s/ran-%s"\necho "%s out"\n' "$cd_" "$name" "$name" > "$cd_/check-$name.sh"
  done
  printf '#!/usr/bin/env bash\necho "$@" >> "%s/ran-red"\necho "red out"\nexit 7\n' "$cd_" > "$cd_/check-red.sh"
  : > "$cd_/other.sh"
  out=$(check_names "$cd_" | tr '\n' ' ')
  [ "$out" = "aa model-tax modelsx red skipme zz " ] || fail "check_names derives '$out', not the scripts in byte order (model-tax before modelsx)"
  out=$(check_names "$cd_" skipme zz | tr '\n' ' ')
  [ "$out" = "aa model-tax modelsx red " ] || fail "check_names with a skip derives '$out'"
  rc=0; out=$(check_names "$cd_" nosuch 2>&1) || rc=$?
  [ "$rc" = 64 ] || fail "a skip that names no script ends rc $rc, not 64"
  case $out in *"skip 'nosuch' names no"*) ;; *) fail "a stale skip is not named: $out" ;; esac
  mkdir "$cd_/empty"
  rc=0; out=$(check_names "$cd_/empty" 2>&1) || rc=$?
  [ "$rc" = 64 ] || fail "a directory with no check script ends rc $rc, not 64"
  case $out in *"no check-*.sh in"*) ;; *) fail "an empty directory is not named: $out" ;; esac
  rc=0; out=$(run_checks "$cd_" '' '' 1 2 "$cd_/o" skipme zz 2>&1) || rc=$?
  lines=$(printf '%s' "$out" | tr '\n' ' ')
  [ "$rc" = 0 ] && [ "$lines" = "aa model-tax check --base origin/main modelsx red" ] || fail "--list ends rc $rc and prints '$lines'"
  [ ! -e "$cd_/ran-aa" ] && [ ! -e "$cd_/o" ] || fail "--list ran a check script"
  rc=0; out=$(run_checks "$cd_" '' 1 1 2 "$cd_/o" skipme zz 2>&1) || rc=$?
  lines=$(printf '%s' "$out" | tr '\n' ' ')
  [ "$rc" = 0 ] && [ "$lines" = "aa model-tax check modelsx red" ] || fail "--list with no base ends rc $rc and prints '$lines'"
  rc=0; out=$(run_checks "$cd_" 'a1..b2' '' '' 2 "$cd_/o" red skipme zz 2>&1) || rc=$?
  [ "$rc" = 0 ] || fail "a run of green scripts ends rc $rc: $out"
  case $out in *"mac-static: check-aa rc 0 in "*" s — aa out"*) ;; *) fail "a green check's line has no closing line from its log: $out" ;; esac
  case $out in *"mac-static: checks ok"*) ;; *) fail "a green run does not end ok: $out" ;; esac
  [ "$(cat "$cd_/ran-model-tax")" = "check --base a1" ] || fail "model-tax ran with '$(cat "$cd_/ran-model-tax")', not 'check --base a1'"
  [ -z "$(cat "$cd_/ran-aa")" ] || fail "a check with no arguments ran with '$(cat "$cd_/ran-aa")'"
  [ ! -e "$cd_/ran-skipme" ] && [ ! -e "$cd_/ran-zz" ] || fail "a skipped check script ran"
  [ "$(cat "$cd_/o/check-aa.log")" = "aa out" ] && [ "$(cat "$cd_/o/check-modelsx.log")" = "modelsx out" ] || fail "a check's output is not in its own log"
  rm -f "$cd_"/ran-*
  rc=0; out=$(run_checks "$cd_" '' '' '' 2 "$cd_/o" skipme 2>&1) || rc=$?
  [ "$rc" = 7 ] || fail "a run with a red script ends rc $rc, not that script's 7"
  case $out in *"mac-static: check-red rc 7 in "*) ;; *) fail "the red script is not named: $out" ;; esac
  case $out in *"red checks: check-red (rc 7)"*) ;; *) fail "the red-check line does not name it: $out" ;; esac
  case $out in *"---- check-red, its log ----"*"red out"*) ;; *) fail "the red check's log did not come back whole: $out" ;; esac
  case $out in *"---- check-zz, its log ----"*) fail "a green check's log came back: $out" ;; esac
  [ "$(cat "$cd_/ran-model-tax")" = "check --base origin/main" ] || fail "a run with no base ran model-tax with '$(cat "$cd_/ran-model-tax")'"
  [ -e "$cd_/ran-zz" ] || fail "a red check stopped the checks that follow it: every check runs"
  rc=0; out=$(run_checks "$cd_" '' '' '' 2 "$cd_/o" aa model-tax modelsx red skipme zz 2>&1) || rc=$?
  [ "$rc" = 0 ] && [ -z "$out" ] || fail "every check skipped: rc $rc, out '$out', not rc 0 and nothing run"
  rc=0; out=$(run_checks "$cd_" '' '' 1 2 "$cd_/o" aa model-tax modelsx red skipme zz 2>&1) || rc=$?
  [ "$rc" = 0 ] && [ -z "$out" ] || fail "every check skipped, --list: rc $rc, out '$out', not rc 0 and no line"
  rm -rf "$cd_"

  # the parallel runner, on fixture checks. Meet: two checks that each mark their start and wait for the other's mark (they ran at
  # once; one after another the first waits out its bound, 60 s, and ends rc 9). Bound: six checks that each record how many are
  # live when they start; at JOBS 2 none sees a third. Order: the check that ends last is the first in byte order and the first
  # red, and the lines and the exit follow the order of the names, not of the ends.
  cd_=$(mktemp -d)
  export CHECK_MEET=$cd_/meet CHECK_LIVE=$cd_/live CHECK_SEEN=$cd_/seen
  mkdir "$CHECK_MEET" "$CHECK_LIVE"
  : > "$CHECK_SEEN"
  # shellcheck disable=SC2016 # the check's own shell expands them
  meet='touch "$CHECK_MEET/%s"; n=0; until [ -e "$CHECK_MEET/%s" ]; do n=$((n + 1)); [ "$n" -le 600 ] || exit 9; sleep 0.1; done'
  mkdir "$cd_/meetd" "$cd_/boundd" "$cd_/orderd"
  printf '#!/usr/bin/env bash\n'"$meet"'\n' a b > "$cd_/meetd/check-meeta.sh"
  printf '#!/usr/bin/env bash\n'"$meet"'\n' b a > "$cd_/meetd/check-meetb.sh"
  for k in 1 2 3 4 5 6; do
    # shellcheck disable=SC2016
    printf '#!/usr/bin/env bash\nmkdir "$CHECK_LIVE/b%s"\nls "$CHECK_LIVE" | wc -l | tr -d " " >> "$CHECK_SEEN"\nsleep 0.2\nrmdir "$CHECK_LIVE/b%s"\n' "$k" "$k" > "$cd_/boundd/check-b$k.sh"
  done
  printf '#!/usr/bin/env bash\nsleep 0.6\necho "aa out"\nexit 3\n' > "$cd_/orderd/check-aa.sh"
  printf '#!/usr/bin/env bash\necho "mm out"\nexit 7\n' > "$cd_/orderd/check-mm.sh"
  printf '#!/usr/bin/env bash\necho "zz out"\n' > "$cd_/orderd/check-zz.sh"
  rc=0; out=$(run_checks "$cd_/meetd" '' '' '' 2 "$cd_/o-meet" 2>&1) || rc=$?
  [ "$rc" = 0 ] || fail "two checks that wait for each other did not meet within 60 s: the checks ran one after another ($rc): $out"
  rc=0; out=$(run_checks "$cd_/boundd" '' '' '' 2 "$cd_/o-bound" 2>&1) || rc=$?
  [ "$rc" = 0 ] || fail "the six bound checks ended rc $rc: $out"
  [ "$(wc -l < "$CHECK_SEEN" | tr -d ' ')" = 6 ] || fail "$(wc -l < "$CHECK_SEEN" | tr -d ' ') of the six bound checks ran"
  k=$(sort -n "$CHECK_SEEN" | tail -1)
  [ "$k" -le 2 ] || fail "$k checks were live at once under a bound of 2"
  rc=0; out=$(run_checks "$cd_/orderd" '' '' '' 3 "$cd_/o-order" 2>&1) || rc=$?
  [ "$rc" = 3 ] || fail "two red checks end rc $rc, not the first in byte order's 3: $out"
  lines=$(printf '%s\n' "$out" | grep '^mac-static: check-' | awk '{ printf "%s ", $2 }')
  [ "$lines" = "check-aa check-mm check-zz " ] || fail "the lines run in the order '$lines', not the names' (aa mm zz)"
  case $out in *"---- check-aa, its log ----"*"---- check-mm, its log ----"*) ;; *) fail "the red logs do not follow the names' order: $out" ;; esac
  case $out in *"red checks: check-aa (rc 3) check-mm (rc 7)"*) ;; *) fail "the red-check line does not name both in order: $out" ;; esac
  [ "$(cat "$cd_/o-order/check-aa.log")" = "aa out" ] && [ "$(cat "$cd_/o-order/check-mm.log")" = "mm out" ] || fail "a check's output is not in its own log"
  [ "$(wc -l < "$cd_/o-order/pids" | tr -d ' ')" = 3 ] || fail "the pids file does not list the three checks"
  unset CHECK_MEET CHECK_LIVE CHECK_SEEN
  rm -rf "$cd_"

  # Each entry reads its own results. Run 1's model-tax ends red while zlate waits for a release file; run 2 finishes green before
  # run 1 resumes. Sharing a result directory would let run 1 read run 2's zero for model-tax and end green.
  local fx p1 rc1 rc2 mode
  for mode in --checks --full; do
    fx=$(mktemp -d)
    mkdir "$fx/tools" "$fx/sync"
    # the scheduler and what it imports (recipes.py reads the job file, and the table reader)
    cp "$HERE/tools/mac-static.sh" "$HERE/tools/static-sched.py" "$HERE/tools/recipes.py" "$fx/tools/"
    mkdir "$fx/tools/bloomery"
    cp "$HERE/tools/bloomery/fixture_dirs.py" "$fx/tools/bloomery/"
    printf '#!/usr/bin/env bash\n[ "${CHECK_ROLE:-}" != slow ] || exit 7\n' > "$fx/tools/check-model-tax.sh"
    printf '#!/usr/bin/env bash\nexit 0\n' > "$fx/tools/mac-check.sh"
    chmod +x "$fx/tools/mac-check.sh"
    # shellcheck disable=SC2016 # the check's own shell expands them
    printf '#!/usr/bin/env bash\n[ "$CHECK_ROLE" != slow ] || { n=0; until [ -e "$CHECK_SYNC/release" ]; do n=$((n + 1)); [ "$n" -le 600 ] || exit 9; sleep 0.1; done; }\n' > "$fx/tools/check-zlate.sh"
    rc1=0; rc2=0
    # HOME is the fixture's, so the scheduler's times and the ledger are the fixture's, not the real files
    HOME=$fx CHECK_SYNC=$fx/sync CHECK_ROLE=slow bash "$fx/tools/mac-static.sh" "$mode" > "$fx/out1" 2>&1 & p1=$!
    k=0
    until [ "$(cat "$fx"/target/mac-static/check-model-tax.rc "$fx"/target/mac-static/run.*/check-model-tax.rc "$fx"/target/mac-static/checks/check-model-tax.rc "$fx"/target/mac-static/checks/run.*/check-model-tax.rc 2> /dev/null)" = 7 ]; do
      k=$((k + 1))
      [ "$k" -le 600 ] || { fail "the first run's model-tax result did not land within 60 s"; break; }
      sleep 0.1
    done
    out=$(HOME=$fx CHECK_SYNC=$fx/sync CHECK_ROLE=green bash "$fx/tools/mac-static.sh" "$mode" 2>&1) || rc2=$?
    touch "$fx/sync/release"
    wait "$p1" || rc1=$?
    [ "$rc2" = 0 ] || fail "$mode: the overlapping green run ends rc $rc2: $out"
    [ "$rc1" = 7 ] || fail "$mode: the red run ends rc $rc1 when a green run overlaps it, not its model-tax's 7: $(cat "$fx/out1")"
    case $(cat "$fx/out1") in *"check-model-tax (rc 7)"*) ;; *) fail "$mode: the red run does not name model-tax: $(cat "$fx/out1")" ;; esac
    if [ "$mode" = --checks ]; then
      [ -z "$(ls -A "$fx/target/mac-static/checks" 2> /dev/null)" ] || fail "a --checks run left its results behind: $(ls -A "$fx/target/mac-static/checks")"
    fi
    rm -rf "$fx"
  done
  echo "mac-static: overlap result cases 2 exercised (--checks and --full)"

  # earlier results that cannot be cleared are never this run's. Prior green results in a directory of mode 555 (a user who writes
  # through a mode, root, skips this case by name): the run ends 69 naming the clearing and runs no check, where it read the old zeros.
  # A prior result that is a directory (rm -f refuses it for every user) holds the same for the clearing alone. A result a step cannot
  # write is named (69), a result file that is empty reads as no rc, and a run directory that cannot be removed is named (69 for a
  # green run; a red one keeps its own status).
  fx=$(mktemp -d)
  mkdir "$fx/checks" "$fx/u1" "$fx/u2"
  # shellcheck disable=SC2016
  printf '#!/usr/bin/env bash\necho "$@" >> "%s/ran-aa"\n' "$fx" > "$fx/checks/check-aa.sh"
  # shellcheck disable=SC2016
  printf '#!/usr/bin/env bash\necho "$@" >> "%s/ran-red"\nexit 7\n' "$fx" > "$fx/checks/check-red.sh"
  for name in aa red; do echo 0 > "$fx/u1/check-$name.rc"; echo 1 > "$fx/u1/check-$name.wall"; echo "stale $name" > "$fx/u1/check-$name.log"; done
  chmod -R a-w "$fx/u1"
  if { : > "$fx/u1/probe"; } 2> /dev/null; then
    say "mac-static self-test: the unwritable-directory case is skipped (uid $(id -u) writes through mode 555)"
  else
    rc=0; out=$(run_checks "$fx/checks" '' '' '' 2 "$fx/u1" 2>&1) || rc=$?
    [ "$rc" = 69 ] || fail "a run over prior green results in an unwritable directory ends rc $rc, not 69: $out"
    case $out in *"cannot clear"*"$fx/u1"*) ;; *) fail "the results that could not be cleared are not named: $out" ;; esac
    [ ! -e "$fx/ran-red" ] || fail "a check ran with the earlier results in place"
  fi
  chmod -R u+w "$fx/u1"
  mkdir "$fx/u2/check-red.rc"
  echo 0 > "$fx/u2/check-aa.rc"; echo 1 > "$fx/u2/check-aa.wall"
  rc=0; out=$(run_checks "$fx/checks" '' '' '' 2 "$fx/u2" 2>&1) || rc=$?
  [ "$rc" = 69 ] || fail "a run whose earlier result cannot be removed ends rc $rc, not 69: $out"
  case $out in *"cannot clear"*"$fx/u2"*) ;; *) fail "the earlier result that stays is not named: $out" ;; esac
  [ ! -e "$fx/ran-aa" ] || fail "a check ran with an earlier result it could not remove"
  mkdir "$fx/u3" "$fx/u3/x.rc"
  rc=0; out=$(step_inner x "$fx/u3" true 2>&1) || rc=$?
  [ "$rc" = 69 ] || fail "a step that cannot write its rc ends rc $rc, not 69: $out"
  case $out in *"cannot write"*"x"*) ;; *) fail "the result a step cannot write is not named: $out" ;; esac
  : > "$fx/u3/check-aa.rc"; echo 1 > "$fx/u3/check-aa.wall"
  rc=0; out=$(summarize_checks "$fx/u3" 1 aa 2>&1) || rc=$?
  [ "$rc" = 1 ] || fail "an empty rc file ends rc $rc, not no rc's 1: $out"
  case $out in *"check-aa rc ? no rc"*"red checks: check-aa (no rc)"*) ;; *) fail "an empty rc file is not read as no rc: $out" ;; esac
  mkdir "$fx/u4"
  rc=0; out=$(remove_run_dir "$fx/u4/." 0 2>&1) || rc=$?
  [ "$rc" = 69 ] || fail "a run directory that cannot be removed ends a green run rc $rc, not 69: $out"
  case $out in *"cannot remove the results in $fx/u4/."*) ;; *) fail "the run directory that stays is not named: $out" ;; esac
  rc=0; out=$(remove_run_dir "$fx/u4/." 7 2>&1) || rc=$?
  [ "$rc" = 0 ] && [ -n "$out" ] || fail "a run directory that cannot be removed changes a red run's status (rc $rc) or is not named: $out"
  rm -rf "$fx"

  # --checks on this tree: its --list is every check script (SKIP is the round loop's, --skip is its own), its arguments are refused by name
  out=$(bash "$HERE/tools/mac-static.sh" --checks --list | awk '{ printf "%s ", $1 }')
  [ "$out" = "$(check_names "$HERE/tools" | tr '\n' ' ')" ] || fail "--checks --list prints '$out', not every tools/check-*.sh"
  out=$(bash "$HERE/tools/mac-static.sh" --checks --list | grep '^model-tax ')
  [ "$out" = "model-tax check --base origin/main" ] || fail "--checks --list prints model-tax as '$out'"
  out=$(bash "$HERE/tools/mac-static.sh" --checks --list --no-base | grep '^model-tax ')
  [ "$out" = "model-tax check" ] || fail "--checks --no-base --list prints model-tax as '$out', not its tables alone"
  rc=0; out=$(bash "$HERE/tools/mac-static.sh" --checks --list --skip nosuch 2>&1) || rc=$?
  [ "$rc" = 64 ] || fail "--checks --skip of no script ends rc $rc, not 64"
  rc=0; out=$(bash "$HERE/tools/mac-static.sh" --checks --skip 2>&1) || rc=$?
  [ "$rc" = 64 ] || fail "--checks --skip with no name ends rc $rc, not 64"
  rc=0; out=$(bash "$HERE/tools/mac-static.sh" --checks --bogus 2>&1) || rc=$?
  [ "$rc" = 64 ] || fail "--checks --bogus ends rc $rc, not 64"
  rc=0; out=$(bash "$HERE/tools/mac-static.sh" --fresh --checks --list 2>&1) || rc=$?
  [ "$rc" = 64 ] || fail "--fresh --checks ends rc $rc, not 64: a --checks run has no keys to skip by, so --fresh would be dropped unsaid"
  out=$(ncpu) || fail "ncpu ends rc $?"
  case $out in '' | *[!0-9]* | 0) fail "ncpu prints '$out', not a positive whole number" ;; esac

  # job_lines: fmt-check, lint, combos after lint (the cargo steps holding CARGO_CPUS), one job a check script, the
  # model-tax key holding the base revision, check-recipes as its blocks and not as one step
  out=$(job_lines '' combos --base main)
  [ "$(printf '%s\n' "$out" | head -3 | cut -f1,2 | tr '\t\n' ' ')" = "fmt-check - lint cpus=$CARGO_CPUS combos after=lint,cpus=$CARGO_CPUS " ] \
    || fail "the cargo jobs' lines: $(printf '%s\n' "$out" | head -3)"
  grep -qx "combos	after=lint,cpus=$CARGO_CPUS	-	tools/mac-check.sh combos --base main" <<< "$out" || fail "the combos job is not the combos argv, unkeyed"
  grep -qx "lint	cpus=$CARGO_CPUS	-	tools/mac-check.sh lint" <<< "$out" || fail "the lint job is not mac-check lint, unkeyed (it sources a setup script no key reads): $(printf '%s\n' "$out" | grep '^lint	')"
  for s in "${SCRIPTS[@]}"; do
    [ "$s" = recipes ] || grep -q "^check-$s	" <<< "$out" || fail "check-$s is not a job"
  done
  ! grep -q '^check-recipes	' <<< "$out" || fail "check-recipes is a job of its own, not its blocks"
  grep -q '^check-recipes\.cardtests	' <<< "$out" || fail "check-recipes' blocks are not jobs"
  grep -q '^check-recipes\.gatebatch\.1	' <<< "$out" && grep -Eq '^check-recipes\.selftest\.1	(-|cpus=2)	' <<< "$out" || fail "gate-batch's and recipes.py's self-test parts are not jobs"
  grep -qx 'check-model-tax	-	tree,rev:origin/main	bash tools/check-model-tax.sh check --base origin/main' <<< "$out" \
    || fail "the model-tax job does not key the base revision: $(printf '%s\n' "$out" | grep '^check-model-tax')"
  # a job's command keeps its argument boundaries as the scheduler's `bash -c` reads it: a ledger path with a space reaches
  # mac-check as one argument, and so does a revision with spaces reach check-model-tax
  sp='/tmp/x y/ledger.tsv'
  out=$(job_lines '' combos --ledger "$sp")
  cmd=$(printf '%s\n' "$out" | awk -F'\t' '$1 == "combos" { print $4 }')
  got=$(bash -c "set -- $cmd; printf '%s|' \"\$@\"")
  [ "$got" = "tools/mac-check.sh|combos|--ledger|$sp|" ] || fail "a ledger path with a space does not reach mac-check as one argument: $got"
  out=$(job_lines 'main@{1 day ago}' combos)
  cmd=$(printf '%s\n' "$out" | awk -F'\t' '$1 == "check-model-tax" { print $4 }')
  got=$(bash -c "set -- $cmd; printf '%s|' \"\$@\"")
  [ "$got" = "bash|tools/check-model-tax.sh|check|--base|main@{1 day ago}|" ] || fail "a revision with spaces does not reach check-model-tax as one argument: $got"

  # the scheduling, through tools/static-sched.py on stub jobs: two jobs that each mark their start and wait for the
  # other's mark ran at once (one after the other, the first waits out its bound, 120 s, and ends rc 9), whatever the
  # load delays either start by; rcs and walls land; a job after a slow one starts after it and reads its own wall
  d=$(mktemp -d)
  T0=$SECONDS
  # shellcheck disable=SC2016 # the job's own shell expands them
  meet='touch "$1/$2.on"; n=0; until [ -e "$1/$3.on" ]; do n=$((n + 1)); [ "$n" -le 600 ] || exit 9; sleep 0.2; done'
  {
    printf 'meet-a\t-\t-\tbash -c %q _ %q a b\n' "$meet" "$d"
    printf 'meet-b\t-\t-\tbash -c %q _ %q b a\n' "$meet" "$d"
    printf 'false-step\t-\t-\texit 3\n'
    printf 'true-step\t-\t-\techo ok\n'
    printf 'slow-step\t-\t-\tsleep 3\n'
    printf 'after-slow\tafter=slow-step\t-\ttrue\n'
  } > "$d/jobs.tsv"
  rc=0; TIMES=$d/times.tsv run_jobs "$d" "$d/jobs.tsv" full || rc=$? # stub rows stay out of the real times file
  [ "$rc" = 0 ] || fail "the scheduler over stub jobs ended rc $rc"
  [ "$(cat "$d/meet-a.rc")" = 0 ] && [ "$(cat "$d/meet-b.rc")" = 0 ] \
    || fail "two jobs that wait for each other did not meet within 120 s: the jobs ran one after another"
  [ "$(cat "$d/true-step.rc")" = 0 ] || fail "a true step's rc is not 0"
  [ "$(cat "$d/false-step.rc")" = 3 ] || fail "an exit-3 step's rc is not 3"
  [ "$(cat "$d/slow-step.wall")" -ge 3 ] || fail "a 3 s step reads a wall of $(cat "$d/slow-step.wall") s"
  [ "$(cat "$d/after-slow.wall")" -lt 3 ] || fail "a step after a slow one reads a wall of $(cat "$d/after-slow.wall") s, not its own"
  awk -F'\t' '$1 == "slow-step" { e = $3 } $1 == "after-slow" { s = $2 } END { exit !(s >= e) }' "$d/sched.tsv" \
    || fail "a step after another started before it ended: $(cat "$d/sched.tsv")"
  [ "$(cat "$d/true-step.log")" = ok ] || fail "a step's output did not land in its log"
  [ ! -e "$d/check-recipes.rc" ] || fail "a run with no check-recipes job wrote its step files"
  echo 0 > "$d/fmt-check.rc"; echo 1 > "$d/fmt-check.wall"; : > "$d/fmt-check.log"
  echo 3 > "$d/lint.rc"; echo 1 > "$d/lint.wall"; printf '%s\n' 'mac-check: lint rc 3' > "$d/lint.log"
  echo 0 > "$d/combos.rc"; echo 2 > "$d/combos.wall"; printf '%s\n' 'mac-check: combos rc 0; 3 shapes run' > "$d/combos.log"
  for s in "${SCRIPTS[@]}"; do echo 0 > "$d/check-$s.rc"; echo 1 > "$d/check-$s.wall"; : > "$d/check-$s.log"; done
  rm -f "$d/check-loads.rc"
  rc=0; out=$(summarize "$d" '' 2>&1) || rc=$?
  [ "$rc" = 3 ] || fail "a red loop ends rc $rc, not the first red step's 3"
  case $out in *"lint rc 3 in 1 s"*"mac-check: lint rc 3"*) ;; *) fail "the lint line or its summary is wrong: $out" ;; esac
  case $out in *"check-loads rc ? no rc"*) ;; *) fail "a step without an rc file is not named: $out" ;; esac
  case $out in *"red steps: lint (rc 3) check-loads (no rc)"*) ;; *) fail "the red-step line does not name both: $out" ;; esac
  case $out in *"base none (full run)"*) ;; *) fail "a full run does not say so: $out" ;; esac
  echo 3 > "$d/check-loads.rc"
  echo 0 > "$d/lint.rc"
  rc=0; out=$(summarize "$d" main 2>&1) || rc=$?
  [ "$rc" = 3 ] || fail "a red check script ends rc $rc, not 3"
  case $out in *"red steps: check-loads (rc 3)"*) ;; *) fail "the red check script is not named: $out" ;; esac
  case $out in *"base main"*"ledger $LEDGER"*) ;; *) fail "a scoped run does not name its base and ledger: $out" ;; esac
  for s in "${SCRIPTS[@]}"; do echo 0 > "$d/check-$s.rc"; done
  rc=0; out=$(summarize "$d" main 2>&1) || rc=$?
  [ "$rc" = 0 ] || fail "a green loop ends rc $rc"
  case $out in *"mac-static: ok"*) ;; *) fail "a green loop does not end ok: $out" ;; esac
  case $out in *"skipped by a green input key"*) fail "a run with no skip names one: $out" ;; esac
  # a skipped step's line names its green row, and the run names every skipped job, a check-recipes block among them
  echo 'green at 2026-10-19 10:00 in t1, key 0123456789ab' > "$d/check-arch.skip"
  : > "$d/check-recipes.gatebatch.skip"
  printf '%s\n' 'check-recipes: 40 blocks ran, 15 skipped by a green input key' > "$d/check-recipes.log"
  rc=0; out=$(summarize "$d" main 2>&1) || rc=$?
  [ "$rc" = 0 ] || fail "a run with skipped jobs ends rc $rc"
  case $out in *"check-arch rc 0 in 1 s — skipped, its input key is green at 2026-10-19 10:00 in t1"*) ;; *) fail "a skipped step's line: $out" ;; esac
  case $out in *"check-recipes rc 0 in 1 s — check-recipes: 40 blocks ran, 15 skipped"*) ;; *) fail "check-recipes' line does not count its blocks: $out" ;; esac
  case $out in *"skipped by a green input key: check-arch check-recipes.gatebatch"*) ;; *) fail "the skipped jobs are not named: $out" ;; esac
  rm -rf "$d"

  if [ "$fails" != 0 ]; then
    say "mac-static: self-test $fails failed"
    return 1
  fi
  echo "mac-static: self-test ok"
}

T0=$SECONDS
BASE=
MODE=keyed
if [ "${1:-}" = --fresh ]; then
  MODE=fresh
  shift
fi
case ${1:-} in
  --self-test) [ $# = 1 ] && [ "$MODE" = keyed ] || { say "mac-static.sh: --self-test takes nothing"; exit 64; }; self_test; exit $? ;;
  --checks)
    [ "$MODE" = keyed ] || { say "mac-static.sh: --fresh is a run's, --checks takes only its own arguments"; exit 64; }
    shift
    list=
    nobase=
    jobs=1
    cskip=()
    while [ $# -gt 0 ]; do
      case $1 in
        --list) list=1; shift ;;
        --no-base) nobase=1; shift ;;
        --skip) [ $# -ge 2 ] || { say "mac-static.sh: --skip takes a check name"; exit 64; }; cskip+=("$2"); shift 2 ;;
        *) say "usage: tools/mac-static.sh --checks [--list] [--no-base] [--skip NAME]… ('$1' is not one)"; exit 64 ;;
      esac
    done
    [ -n "$list" ] || jobs=$(ncpu) || exit $?
    check_dir=$HERE/target/mac-static/checks
    if [ -z "$list" ]; then
      mkdir -p "$check_dir" || { say "mac-static.sh: cannot create $check_dir"; exit 69; }
      check_dir=$(mktemp -d "$check_dir/run.XXXXXX") || { say "mac-static.sh: cannot create a run directory in $check_dir"; exit 69; }
      trap 'remove_run_dir "$check_dir" $?' EXIT
    fi
    rc=0
    run_checks "$HERE/tools" '' "$nobase" "$list" "$jobs" "$check_dir" ${cskip[@]+"${cskip[@]}"} || rc=$?
    exit "$rc"
    ;;
  --full) [ $# = 1 ] || { say "mac-static.sh: --full takes nothing"; exit 64; } ;;
  "") : ;;
  -*) say "usage: tools/mac-static.sh [--fresh] [BASE|--full] | --checks [--list] [--no-base] [--skip NAME]… | --self-test ('$1' is not one)"; exit 64 ;;
  *) BASE=$1; [ $# = 1 ] || { say "mac-static.sh: one optional BASE, got $#: $*"; exit 64; } ;;
esac
[ -n "$BASE" ] || MODE=full

D=$HERE/target/mac-static
mkdir -p "$D" || { say "mac-static.sh: cannot create $D"; exit 69; }
D=$(mktemp -d "$D/run.XXXXXX") || { say "mac-static.sh: cannot create a run directory in $D"; exit 69; }
echo "mac-static: results ${D#"$HERE/"}"
CARGO=()
while IFS= read -r w; do CARGO+=("$w"); done < <(combos_argv "$BASE" "$LEDGER")
job_lines "$BASE" "${CARGO[@]}" > "$D/jobs.tsv"
run_jobs "$D" "$D/jobs.tsv" "$MODE" || true # a scheduler that ended early leaves its steps without rc files: the summary names them
summarize "$D" "$BASE"
