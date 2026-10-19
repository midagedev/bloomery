#!/usr/bin/env bash
# bloomery — the Mac static tier in one command, the round loop's form. The loop a round runs before
# it reports was a sequence (mac-check check, mac-check combos, mac-check lint, fmt-check, the eight
# tools/check-*.sh one after another) whose walls measured 2026-10-07 (load ~5, this Mac): the cargo
# steps 0.6-40 s each but the eight scripts ~240 s serial, `check-recipes` alone 172 s. Two lanes cut
# it to the longer lane: the scripts lane (fmt-check and every tools/check-*.sh, one job each) and the cargo
# lane (lint, then the scoped combos, serial — they share the tree's target directory, and two cargos
# on one directory serialize on its lock anyway).
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
#   tools/mac-static.sh [BASE|--full]   one line a step (rc, wall, the step's own summary), exits
#                                       non-zero naming every red step; BASE scopes the combos step
#                                       (a rev, or A..B); --full or no argument runs it unscoped
#   tools/mac-static.sh --self-test     the lanes' orchestration, the rc/wall capture, the red-step
#                                       naming, the check-script derivation, the parallel check runner and the
#                                       argument plumbing, against /bin/true, /bin/false and sleep stubs and fixture
#                                       check scripts; runs no cargo and no check script (check-recipes runs it)
#   tools/mac-static.sh --checks [--list] [--no-base] [--skip NAME]…
#                                       the check scripts alone, in parallel, at most one a cpu (what `just gate` runs:
#                                       the justfile's check-scripts): one line a check in byte order of the name
#                                       (rc, wall, the last line of its log), then every red check's log whole and the
#                                       red names; every check runs, a red one stops no other, and the exit is the first
#                                       red check's rc in that order. --skip leaves a check out; --no-base runs
#                                       check-model-tax on its tables alone, with no revision to hold them to (a checkout
#                                       with no origin/main: tools/contrib-check.sh); --list prints each name it would
#                                       run with its arguments, one a line, and runs none
#
# The check scripts are every tools/check-*.sh, in byte order of the name, but those SKIP names: a new script joins the
# loop, `just gate` and tools/contrib-check.sh by existing, with no list to add its name to. A skip that names no script is
# refused (64), so a stale entry cannot read as a script left out on purpose.
#
# Exit: the first red step's rc, or 0. 64: an argument this script does not take. 69: no cpu count (getconf, sysctl and
# nproc all fail). A step that wrote no rc file (killed before its wrapper ran) is red, named `no rc`. Steps' full output
# and their rc/wall land in target/mac-static/<step>.{log,rc,wall} (--checks: target/mac-static/checks/); the cargo lane's
# steps also write the mac-check logs (target/mac-check-*.log) as they always did.
#
# The environment is mac-check's (the pinned toolchain, $HOME/opt/bloomery-mac-env.sh for the cross
# check): every cargo step calls tools/mac-check.sh, which owns that setup. The ledger is
# ~/.cache/bloomery/mac-static-ledger.tsv, shared between trees on purpose: a green key is a fact
# about the inputs, not the tree. Another round's tree has its own target directory; nothing here
# reads another tree's.
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
LEDGER=$HOME/.cache/bloomery/mac-static-ledger.tsv
# The check scripts the round loop leaves out, by name (tools/check-NAME.sh), the reason on the line above each. None today.
SKIP=()

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

# step_inner NAME D CMD…: one step, serial in its caller — its rc, wall and full output to D. Never
# fails its caller: the rc is the file's, not the wrapper's.
step_inner() {
  local name=$1 d=$2 t0=$SECONDS rc=0
  shift 2
  "$@" > "$d/$name.log" 2>&1 < /dev/null || rc=$?
  echo "$rc" > "$d/$name.rc"
  echo "$((SECONDS - t0))" > "$d/$name.wall"
}

# run_steps D CARGO_CMDV…: both lanes. The scripts lane: one background job a script (fmt-check and
# every tools/check-*.sh). The cargo lane: one background job running lint then the combos step
# serially (one target directory, one lock). Every job's pid is captured at start, written to
# D/pids, and waited on; no polling.
run_steps() {
  local d=$1 base=$2
  shift 2
  local -a cargo=("$@") pids=() names=()
  : > "$d/pids"
  ( step_inner fmt-check "$d" "$HERE/tools/mac-check.sh" fmt-check ) & pids+=($!) && names+=(fmt-check)
  local s
  local -a args
  local w
  for s in "${SCRIPTS[@]}"; do
    args=()
    while IFS= read -r w; do args+=("$w"); done < <(check_argv "$s" "$base")
    ( step_inner "check-$s" "$d" bash "$HERE/tools/check-$s.sh" ${args[@]+"${args[@]}"} ) & pids+=($!) && names+=(check-$s)
  done
  ( step_inner lint "$d" "$HERE/tools/mac-check.sh" lint
    step_inner combos "$d" "$HERE/tools/mac-check.sh" "${cargo[@]}" ) & pids+=($!) && names+=(cargo-lane)
  local i
  for i in "${!pids[@]}"; do echo "${names[$i]} ${pids[$i]}" >> "$d/pids"; done
  wait "${pids[@]}" || true # a killed wrapper is the summary's `no rc` case, not this script's end
}

# run_check_jobs DIR BASE NOBASE JOBS D NAME…: DIR's check-NAME.sh for each NAME, in the order given, at most JOBS at a time under
# its check_argv; each one's output, rc and wall land in D (step_inner). The window is the jobs launched minus the jobs waited
# on, and a full one waits for the oldest pid (the Mac's bash 3.2 has no `wait -n`). Every pid is captured at start, written to
# D/pids, and waited on; no polling.
run_check_jobs() {
  local dir=$1 base=$2 nobase=$3 jobs=$4 d=$5 n w i=0
  local -a pids=() args
  shift 5
  : > "$d/pids"
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
  mkdir -p "$d"
  rm -f "$d"/*.rc "$d"/*.wall "$d"/pids 2> /dev/null || true
  run_check_jobs "$dir" "$base" "$nobase" "$jobs" "$d" "${names[@]}"
  summarize_checks "$d" "$jobs" "${names[@]}"
}

# step_line D NAME: the step's summary line on stdout, `mac-static: <step> rc <n> in <s> s [— the step's own closing line]`: the
# cargo steps' closing line comes from their log, a check's is its log's last non-empty line (cut at 160 characters). A step that
# is red, or wrote no rc, joins REDS; FIRSTRC is the rc of the first red one (1 for no rc), 0 while none is.
REDS=()
FIRSTRC=0
step_line() {
  local d=$1 name=$2 rc wall extra=
  if [ ! -f "$d/$name.rc" ]; then
    echo "mac-static: $name rc ? no rc in $((SECONDS - T0)) s (killed before its wrapper ran)"
    REDS+=("$name (no rc)")
    [ "$FIRSTRC" != 0 ] || FIRSTRC=1
    return 0
  fi
  rc=$(cat "$d/$name.rc")
  wall=$(cat "$d/$name.wall")
  case $name in
    lint) extra=$(grep -E '^mac-check: (lint \^warning|lint rc)' "$d/$name.log" | tail -1 || true) ;;
    combos) extra=$(grep -E '^mac-check: combos rc' "$d/$name.log" | tail -1 || true) ;;
    fmt-check) extra="rustfmt diff blocks: $(grep -cE '^diff' "$d/$name.log" || true)" ;;
    check-*) extra=$(grep -v '^[[:space:]]*$' "$d/$name.log" | tail -1 | cut -c1-160 || true) ;;
  esac
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
  echo "mac-static: $# checks in $((SECONDS - T0)) s, at most $jobs at once, logs in $d"
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
# cargo steps' closing line comes from their log. Ends 0, or prints one line naming every red step
# and ends with the first red rc.
summarize() {
  local d=$1 base=$2 s
  REDS=()
  FIRSTRC=0
  step_line "$d" fmt-check
  step_line "$d" lint
  step_line "$d" combos
  for s in "${SCRIPTS[@]}"; do step_line "$d" "check-$s"; done
  echo "mac-static: total $((SECONDS - T0)) s, base ${base:-none (full run)}, ledger ${base:+$LEDGER}"
  if [ "${#REDS[@]}" != 0 ]; then
    say "mac-static: red steps: ${REDS[*]}"
    return "$FIRSTRC"
  fi
  echo "mac-static: ok"
}

self_test() {
  local fails=0 d out rc k in_scripts in_skip
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
  out=$(ncpu) || fail "ncpu ends rc $?"
  case $out in '' | *[!0-9]* | 0) fail "ncpu prints '$out', not a positive whole number" ;; esac

  # the lanes: two steps that each mark their start and wait for the other's mark (they ran at once;
  # on serial lanes the first waits out its bound, 120 s, and ends rc 9), whatever the load delays
  # either start by; rcs and walls land, a failing step is red and named, a step with no rc file is
  # red as `no rc`
  d=$(mktemp -d)
  T0=$SECONDS
  np=0
  # shellcheck disable=SC2016 # the step's own shell expands them
  meet='touch "$1/$2.on"; n=0; until [ -e "$1/$3.on" ]; do n=$((n + 1)); [ "$n" -le 600 ] || exit 9; sleep 0.2; done'
  ( step_inner meet-a "$d" bash -c "$meet" _ "$d" a b ) & pids[$np]=$!; np=$((np + 1))
  ( step_inner meet-b "$d" bash -c "$meet" _ "$d" b a ) & pids[$np]=$!; np=$((np + 1))
  ( step_inner false-step "$d" bash -c 'exit 3' ) & pids[$np]=$!; np=$((np + 1))
  ( step_inner true-step "$d" bash -c 'echo ok' ) & pids[$np]=$!; np=$((np + 1))
  # a step's wall is its own seconds, not the lane's: a function step moves this subshell's clock by 60 s (no sleep), and the
  # next step in the same lane reads under 30 s, whatever the load; a wall taken from the lane's start would read 60 or more
  clock_step() { SECONDS=$((SECONDS + 60)); }
  ( step_inner clock-step "$d" clock_step; step_inner after-clock "$d" true ) & pids[$np]=$!; np=$((np + 1))
  wait "${pids[@]}" || true
  [ "$(cat "$d/meet-a.rc")" = 0 ] && [ "$(cat "$d/meet-b.rc")" = 0 ] \
    || fail "two steps that wait for each other did not meet within 120 s: the lanes ran one after another"
  [ "$(cat "$d/true-step.rc")" = 0 ] || fail "a true step's rc is not 0"
  [ "$(cat "$d/false-step.rc")" = 3 ] || fail "an exit-3 step's rc is not 3"
  [ "$(cat "$d/clock-step.wall")" -ge 60 ] || fail "a step that moved the clock by 60 s reads a wall of $(cat "$d/clock-step.wall") s"
  [ "$(cat "$d/after-clock.wall")" -lt 30 ] || fail "a step after a slow one in its lane reads a wall of $(cat "$d/after-clock.wall") s, not its own"
  [ "$(cat "$d/true-step.log")" = ok ] || fail "a step's output did not land in its log"
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
  rm -rf "$d"

  if [ "$fails" != 0 ]; then
    say "mac-static: self-test $fails failed"
    return 1
  fi
  echo "mac-static: self-test ok"
}

T0=$SECONDS
BASE=
case ${1:-} in
  --self-test) [ $# = 1 ] || { say "mac-static.sh: --self-test takes nothing"; exit 64; }; self_test; exit $? ;;
  --checks)
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
    rc=0
    run_checks "$HERE/tools" '' "$nobase" "$list" "$jobs" "$HERE/target/mac-static/checks" ${cskip[@]+"${cskip[@]}"} || rc=$?
    exit "$rc"
    ;;
  --full) [ $# = 1 ] || { say "mac-static.sh: --full takes nothing"; exit 64; } ;;
  "") : ;;
  -*) say "usage: tools/mac-static.sh [BASE|--full] | --checks [--list] [--no-base] [--skip NAME]… | --self-test ('$1' is not one)"; exit 64 ;;
  *) BASE=$1; [ $# = 1 ] || { say "mac-static.sh: one optional BASE, got $#: $*"; exit 64; } ;;
esac

D=$HERE/target/mac-static
mkdir -p "$D"
rm -f "$D"/*.rc "$D"/*.wall "$D"/pids 2> /dev/null || true
CARGO=()
while IFS= read -r w; do CARGO+=("$w"); done < <(combos_argv "$BASE" "$LEDGER")
run_steps "$D" "$BASE" "${CARGO[@]}"
summarize "$D" "$BASE"
