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
#                                       naming, the check-script derivation and the argument plumbing, against
# /bin/true, /bin/false and sleep stubs and fixture check scripts; runs no cargo and no check script (check-recipes runs it)
#   tools/mac-static.sh --checks [--list] [--skip NAME]…
#                                       the check scripts one after another in the foreground, the first red ends
#                                       it with that script's rc (what `just gate` runs: the justfile's
#                                       check-scripts); --skip leaves a check out, --list prints each name it
#                                       would run with its arguments and runs none
#
# The check scripts are every tools/check-*.sh, in byte order of the name, but those SKIP names: a new script joins the
# loop, `just gate` and tools/contrib-check.sh by existing, with no list to add its name to. A skip that names no script is
# refused (64), so a stale entry cannot read as a script left out on purpose.
#
# Exit: the first red step's rc, or 0. 64: an argument this script does not take. A step that wrote
# no rc file (killed before its wrapper ran) is red, named `no rc`. Steps' full output and their
# rc/wall land in target/mac-static/<step>.{log,rc,wall}; the cargo lane's steps also write the
# mac-check logs (target/mac-check-*.log) as they always did.
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

# check_argv NAME BASE: the arguments of tools/check-NAME.sh. check-model-tax holds its rows to the revision BASE names (the
# A of A..B; origin/main on a full run), so an edit of its tables cannot raise a count unargued.
check_argv() {
  case $1 in
    model-tax) printf '%s\n' check --base "$(base_rev "$2")" ;;
  esac
}
# base_rev BASE: the revision a base names, origin/main when there is none.
base_rev() {
  local b=${1%%..*}
  printf '%s\n' "${b:-origin/main}"
}

# run_checks DIR BASE LIST SKIP…: DIR's check scripts (check_names: all but SKIP) one after another in the foreground, each under
# its check_argv, a `mac-static: check-NAME` line before each. The first red ends it with that script's rc. LIST non-empty prints
# each NAME and its arguments, one a line, and runs none.
run_checks() {
  local dir=$1 base=$2 list=$3 listed n rc w
  local -a names=() args
  shift 3
  listed=$(check_names "$dir" "$@") || return $?
  [ -n "$listed" ] || return 0 # every check skipped: a here-string of nothing would still read one empty name
  while IFS= read -r n; do names+=("$n"); done <<< "$listed"
  for n in "${names[@]}"; do
    args=()
    while IFS= read -r w; do args+=("$w"); done < <(check_argv "$n" "$base")
    if [ -n "$list" ]; then
      printf '%s\n' "$n${args[*]+ ${args[*]}}"
      continue
    fi
    say "mac-static: check-$n"
    rc=0
    bash "$dir/check-$n.sh" ${args[@]+"${args[@]}"} < /dev/null || rc=$?
    [ "$rc" = 0 ] || { say "mac-static: check-$n rc $rc"; return "$rc"; }
  done
}

# step_inner NAME D CMD…: one step, serial in its caller — its rc, wall and full output to D. Never
# fails its caller: the rc is the file's, not the wrapper's.
step_inner() {
  local name=$1 d=$2 t0=$SECONDS rc=0
  shift 2
  "$@" > "$d/$name.log" 2>&1 || rc=$?
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

# summarize D BASE: one line a step in a fixed order (fmt-check, lint, combos, every check),
# then the verdict. Prints `mac-static: <step> rc <n> in <s> s [— the step's own closing line]`; the
# cargo steps' closing line comes from their log. Ends 0, or prints one line naming every red step
# and ends with the first red rc.
summarize() {
  local d=$1 base=$2 red=0 firstrc=0 s name rc wall extra
  local -a reds=()
  one_line() { # NAME: the step's summary line and its red bookkeeping
    local name=$1
    if [ ! -f "$d/$name.rc" ]; then
      echo "mac-static: $name rc ? no rc in $((SECONDS - T0)) s (killed before its wrapper ran)"
      reds+=("$name (no rc)")
      [ "$red" != 0 ] || { red=1; firstrc=1; }
      return
    fi
    rc=$(cat "$d/$name.rc")
    wall=$(cat "$d/$name.wall")
    extra=
    case $name in
      lint) extra=$(grep -E '^mac-check: (lint \^warning|lint rc)' "$d/$name.log" | tail -1 || true) ;;
      combos) extra=$(grep -E '^mac-check: combos rc' "$d/$name.log" | tail -1 || true) ;;
      fmt-check) extra="rustfmt diff blocks: $(grep -cE '^diff' "$d/$name.log" || true)" ;;
    esac
    echo "mac-static: $name rc $rc in $wall s${extra:+ — $extra}"
    if [ "$rc" != 0 ]; then
      reds+=("$name (rc $rc)")
      [ "$red" != 0 ] || { red=1; firstrc=$rc; }
    fi
  }
  one_line fmt-check
  one_line lint
  one_line combos
  for s in "${SCRIPTS[@]}"; do one_line "check-$s"; done
  echo "mac-static: total $((SECONDS - T0)) s, base ${base:-none (full run)}, ledger ${base:+$LEDGER}"
  if [ "$red" != 0 ]; then
    say "mac-static: red steps: ${reds[@]+"${reds[*]}"}"
    return "$firstrc"
  fi
  echo "mac-static: ok"
}

self_test() {
  local fails=0 d out rc k in_scripts in_skip
  local -a pids=()
  fail() { say "mac-static self-test FAIL: $*"; fails=$((fails + 1)); }

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
  # no script, a directory with none), the arguments (model-tax's base), the first red ending the run, and --list running none
  local cd_ lines
  cd_=$(mktemp -d)
  for name in aa modelsx model-tax red skipme zz; do
    printf '#!/usr/bin/env bash\necho "$@" >> "%s/ran-%s"\n' "$cd_" "$name" > "$cd_/check-$name.sh"
  done
  printf '#!/usr/bin/env bash\necho "$@" >> "%s/ran-red"\nexit 7\n' "$cd_" > "$cd_/check-red.sh"
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
  rc=0; out=$(run_checks "$cd_" '' 1 skipme zz 2>&1) || rc=$?
  lines=$(printf '%s' "$out" | tr '\n' ' ')
  [ "$rc" = 0 ] && [ "$lines" = "aa model-tax check --base origin/main modelsx red" ] || fail "--list ends rc $rc and prints '$lines'"
  [ ! -e "$cd_/ran-aa" ] || fail "--list ran a check script"
  rc=0; out=$(run_checks "$cd_" 'a1..b2' '' red skipme zz 2>&1) || rc=$?
  [ "$rc" = 0 ] || fail "a run of green scripts ends rc $rc: $out"
  [ "$(cat "$cd_/ran-model-tax")" = "check --base a1" ] || fail "model-tax ran with '$(cat "$cd_/ran-model-tax")', not 'check --base a1'"
  [ -z "$(cat "$cd_/ran-aa")" ] || fail "a check with no arguments ran with '$(cat "$cd_/ran-aa")'"
  [ ! -e "$cd_/ran-skipme" ] && [ ! -e "$cd_/ran-zz" ] || fail "a skipped check script ran"
  rm -f "$cd_"/ran-*
  rc=0; out=$(run_checks "$cd_" '' '' skipme 2>&1) || rc=$?
  [ "$rc" = 7 ] || fail "a run with a red script ends rc $rc, not that script's 7"
  case $out in *"mac-static: check-red rc 7"*) ;; *) fail "the red script is not named: $out" ;; esac
  [ "$(cat "$cd_/ran-model-tax")" = "check --base origin/main" ] || fail "a run with no base ran model-tax with '$(cat "$cd_/ran-model-tax")'"
  [ ! -e "$cd_/ran-zz" ] || fail "a script after the red one ran"
  rc=0; out=$(run_checks "$cd_" '' '' aa model-tax modelsx red skipme zz 2>&1) || rc=$?
  [ "$rc" = 0 ] && [ -z "$out" ] || fail "every check skipped: rc $rc, out '$out', not rc 0 and nothing run"
  rc=0; out=$(run_checks "$cd_" '' 1 aa model-tax modelsx red skipme zz 2>&1) || rc=$?
  [ "$rc" = 0 ] && [ -z "$out" ] || fail "every check skipped, --list: rc $rc, out '$out', not rc 0 and no line"
  rm -rf "$cd_"

  # --checks on this tree: its --list is every check script (SKIP is the round loop's, --skip is its own), its arguments are refused by name
  out=$(bash "$HERE/tools/mac-static.sh" --checks --list | awk '{ printf "%s ", $1 }')
  [ "$out" = "$(check_names "$HERE/tools" | tr '\n' ' ')" ] || fail "--checks --list prints '$out', not every tools/check-*.sh"
  rc=0; out=$(bash "$HERE/tools/mac-static.sh" --checks --list --skip nosuch 2>&1) || rc=$?
  [ "$rc" = 64 ] || fail "--checks --skip of no script ends rc $rc, not 64"
  rc=0; out=$(bash "$HERE/tools/mac-static.sh" --checks --skip 2>&1) || rc=$?
  [ "$rc" = 64 ] || fail "--checks --skip with no name ends rc $rc, not 64"
  rc=0; out=$(bash "$HERE/tools/mac-static.sh" --checks --bogus 2>&1) || rc=$?
  [ "$rc" = 64 ] || fail "--checks --bogus ends rc $rc, not 64"

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
    cskip=()
    while [ $# -gt 0 ]; do
      case $1 in
        --list) list=1; shift ;;
        --skip) [ $# -ge 2 ] || { say "mac-static.sh: --skip takes a check name"; exit 64; }; cskip+=("$2"); shift 2 ;;
        *) say "usage: tools/mac-static.sh --checks [--list] [--skip NAME]… ('$1' is not one)"; exit 64 ;;
      esac
    done
    rc=0
    run_checks "$HERE/tools" '' "$list" ${cskip[@]+"${cskip[@]}"} || rc=$?
    exit "$rc"
    ;;
  --full) [ $# = 1 ] || { say "mac-static.sh: --full takes nothing"; exit 64; } ;;
  "") : ;;
  -*) say "usage: tools/mac-static.sh [BASE|--full] | --checks [--list] [--skip NAME]… | --self-test ('$1' is not one)"; exit 64 ;;
  *) BASE=$1; [ $# = 1 ] || { say "mac-static.sh: one optional BASE, got $#: $*"; exit 64; } ;;
esac

D=$HERE/target/mac-static
mkdir -p "$D"
rm -f "$D"/*.rc "$D"/*.wall "$D"/pids 2> /dev/null || true
CARGO=()
while IFS= read -r w; do CARGO+=("$w"); done < <(combos_argv "$BASE" "$LEDGER")
run_steps "$D" "$BASE" "${CARGO[@]}"
summarize "$D" "$BASE"
