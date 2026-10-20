#!/usr/bin/env bash
# bloomery — the Mac static tier in one command, the round loop's form. The loop a round runs before
# it reports was a sequence (mac-check check, mac-check combos, mac-check lint, fmt-check, the eight
# tools/check-*.sh one after another) whose walls measured 2026-10-07 (load ~5, this Mac): the cargo
# steps 0.6-40 s each but the eight scripts ~240 s serial, `check-recipes` alone 172 s. Two lanes cut
# it to the longer lane: the scripts lane (fmt-check and every tools/check-*.sh, one job each) and the cargo
# lane (lint, then the scoped combos, serial — they share the tree's target directory, and two cargos
# on one directory serialize on its lock anyway).
#
# On a Linux host, `--native` (below) runs these lanes directly there, which is the faster path when such a host is up.
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
#   tools/mac-static.sh --native [--slice NAME.slice] [BASE|--full]
#                                       the same two lanes run directly on a Linux host, where the recipes' own cargo lines are the
#                                       cargo lane: fmt-check and clippy as the fmt-check and lint recipes spell them
#                                       (tools/recipes.py box-command), and the combos step's shapes as `cargo check` with no
#                                       --target. The lint is held to tools/lint-ratchet.txt as mac-check.sh holds it. The combos
#                                       ledger is the host's own file (~/.cache/bloomery/native-static-ledger.tsv, keyed as the
#                                       Mac's is). BLOOMERY_GIT_COMMIT is fixed for the cargo steps (as mac-check.sh fixes it), so a
#                                       HEAD move rebuilds nothing. Linux only (64 elsewhere). The lanes run plain unless --slice
#                                       is given: then the run goes in that systemd slice (systemd-run --scope) under a bound of
#                                       NATIVE_BOUND seconds (killed with rc 124). A slice whose unit file the host has not installed
#                                       is refused (69), since systemd-run would make one with no limits.
#   tools/mac-static.sh --self-test     the lanes' orchestration, the rc/wall capture, the red-step
#                                       naming, the check-script derivation, the parallel check runner and the
#                                       argument plumbing, against /bin/true, /bin/false and sleep stubs and fixture
#                                       check scripts; runs no cargo and no check script (check-recipes runs it)
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
# `tools/mac-check.sh` stays the Mac's: the cross check, the Mac's toolchain and its disk floor. The native lane has none of them and
# repeats only mac-check.sh's lint ratchet and ledger line, each in a few lines, because that script cannot run on Linux.
#
# The check scripts are every tools/check-*.sh, in byte order of the name, but those SKIP names: a new script joins the
# loop, `just gate` and tools/contrib-check.sh by existing, with no list to add its name to. A skip that names no script is
# refused (64), so a stale entry cannot read as a script left out on purpose.
#
# Exit: the first red step's rc, or 0. 64: an argument this script does not take. 69: no cpu count (getconf, sysctl and
# nproc all fail), or results that cannot be cleared, created or removed (named; a run that already ended red keeps its rc). A step
# whose rc file is missing or empty (killed before its wrapper ran, or a result that could not be written) is red, named `no rc`,
# never 0. Steps' full output and their rc/wall land in target/mac-static/run.XXXXXX/<step>.{log,rc,wall} (--checks: a run directory under
# target/mac-static/checks/, whose red logs come back whole on stderr); the cargo lane's steps also write the mac-check logs
# (target/mac-check-*.log) as they always did.
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
# The native lane (the header's --native): its combos ledger and the bound in seconds of a run in a slice.
NATIVE_LEDGER=$HOME/.cache/bloomery/native-static-ledger.tsv
NATIVE_BOUND=1500
NATIVE=

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

# native_words RECIPE VERB: the recipe's box command (tools/recipes.py box-command) as one word a line. It must be one plain
# `cargo VERB …` with no shell syntax (the character set mac-check.sh's derive holds a command to), since it runs as words, never
# through a shell. Anything else is refused (64) naming why.
native_words() {
  local cmd
  local -a w
  cmd=$(python3 "$HERE/tools/recipes.py" box-command "$1") || return $?
  if [[ ! $cmd =~ ^[A-Za-z0-9_,./=:+\ -]+$ ]]; then
    say "mac-static.sh: the $1 box command has shell syntax (a character outside [A-Za-z0-9_,./=:+ -]), which --native does not run: $cmd"
    return 64
  fi
  read -ra w <<< "$cmd"
  if [ "${w[0]}" != cargo ] || [ "${w[1]:-}" != "$2" ]; then
    say "mac-static.sh: the $1 box command is not \`cargo $2 …\`: $cmd"
    return 64
  fi
  printf '%s\n' "${w[@]}"
}

# native_cargo WORD…: cargo in this tree with the version commit fixed (a serve build script reads it: a HEAD move would rebuild its
# dependents), no stdin.
native_cargo() {
  (cd "$HERE" && BLOOMERY_GIT_COMMIT=static-check "$@") < /dev/null
}

# native_fmt: the fmt-check recipe's cargo line.
native_fmt() {
  local line
  local -a w=()
  while IFS= read -r line; do w+=("$line"); done <<< "$(native_words fmt-check fmt)"
  native_cargo "${w[@]}"
}

# native_ratchet FILE: the one number line of FILE (tools/lint-ratchet.txt; `#` lines are its rule), or 70.
native_ratchet() {
  local lines
  if [ ! -f "$1" ]; then
    say "mac-static.sh: no ratchet file $1 to hold the lint to"
    return 70
  fi
  lines=$(grep -vE '^[[:space:]]*(#|$)' "$1" || true)
  if ! [[ $lines =~ ^[0-9]+$ ]]; then
    say "mac-static.sh: $1 must hold exactly one line with the warning count, a bare number; it holds: ${lines:-nothing}"
    return 70
  fi
  echo "$lines"
}

# native_lint: the lint recipe's cargo line, its `^warning:` count held to the ratchet; the last line says which way it stands.
native_lint() {
  local line out count ratchet rc=0 vrc=0
  local -a w=()
  while IFS= read -r line; do w+=("$line"); done <<< "$(native_words lint clippy)"
  ratchet=$(native_ratchet "$HERE/tools/lint-ratchet.txt") || return $?
  out=$(mktemp)
  native_cargo "${w[@]}" > "$out" 2>&1 || rc=$?
  cat "$out"
  count=$(grep -c '^warning:' "$out" || true)
  rm -f "$out"
  if [ "$rc" != 0 ]; then
    echo "native: lint rc $rc (^warning: $count, the ratchet $ratchet in tools/lint-ratchet.txt)"
    return "$rc"
  fi
  if [ "$count" -gt "$ratchet" ]; then
    echo "native: lint ^warning: $count, above the ratchet $ratchet in tools/lint-ratchet.txt — red"
    vrc=1
  elif [ "$count" -lt "$ratchet" ]; then
    echo "native: lint ^warning: $count, below the ratchet $ratchet in tools/lint-ratchet.txt — the recorded value can come down"
  else
    echo "native: lint ^warning: $count = the ratchet $ratchet in tools/lint-ratchet.txt"
  fi
  return "$vrc"
}

# native_combos BASE LEDGER: `cargo check` for each shape tools/recipes.py combos lists (scoped to BASE and held to LEDGER when BASE is
# given: a green shape is appended to LEDGER as `<key>\tgreen\t<date>\t<tree>`). Every shape runs, a red one too; the last line counts
# them. The first red shape's rc is the result.
native_combos() {
  local base=$1 ledger=$2 list kind label cmd key n=0 rc=0 crc t0=$SECONDS t1 skips=0 total= r
  local -a ra=(combos) w reds=()
  [ -z "$base" ] || ra+=(--base "$base")
  [ -z "$ledger" ] || [ -z "$base" ] || ra+=(--ledger "$ledger")
  list=$(python3 "$HERE/tools/recipes.py" "${ra[@]}") || return $?
  while IFS=$'\t' read -r kind label cmd key; do
    case $kind in
      skip) skips=$((skips + 1)) ;;
      total) total=$label ;;
      combo)
        if [[ ! $cmd =~ ^[A-Za-z0-9_,./=:+\ -]+$ ]]; then
          say "mac-static.sh: the shape '$label' has shell syntax in its command, which --native does not run: $cmd"
          return 64
        fi
        read -ra w <<< "$cmd"
        if [ "${w[0]}" != cargo ] || [ "${w[1]:-}" != check ]; then
          say "mac-static.sh: the shape '$label' is not \`cargo check …\`: $cmd"
          return 64
        fi
        echo "native: shape $label: $cmd"
        t1=$SECONDS
        crc=0
        native_cargo "${w[@]}" || crc=$?
        echo "native: shape rc $crc in $((SECONDS - t1)) s: $label"
        n=$((n + 1))
        if [ "$crc" != 0 ]; then
          reds+=("$label: $cmd")
          [ "$rc" != 0 ] || rc=$crc
        elif [ -n "$ledger" ] && [ -n "$base" ]; then
          mkdir -p "$(dirname "$ledger")"
          printf '%s\tgreen\t%s\t%s\n' "$key" "$(date '+%Y-%m-%d %H:%M')" "${HERE##*/}" >> "$ledger"
        fi
        ;;
      *)
        say "mac-static.sh: recipes.py combos printed a line that is no combo, skip or total: $kind"
        return 64
        ;;
    esac
  done <<< "$list"
  if [ "$n" = 0 ] && [ -z "$base" ]; then
    say "mac-static.sh: recipes.py combos lists no build shape"
    return 70
  fi
  for r in "${!reds[@]}"; do echo "native: red shape ${reds[$r]}"; done
  echo "native: combos rc $rc; $n shapes run, ${#reds[@]} red${base:+, $skips skipped (scoped)}, in $((SECONDS - t0)) s (recipes.py combos: $total)"
  return "$rc"
}

# native_host UNAME: 0 on Linux, else 64 naming where the Mac's tier is.
native_host() {
  [ "$1" = Linux ] || { say "mac-static.sh: --native runs on a Linux host; this is $1: a Mac runs just mac-static"; return 64; }
}

# enter_slice SLICE ARGS…: this script again as --native, inside the slice and under the bound; its rc is the result. A slice whose
# unit file the host has not installed is 69 (systemd-run would make one with no limits).
enter_slice() {
  local slice=$1 lm fp rc=0
  shift
  command -v systemd-run > /dev/null || { say "mac-static.sh: --slice needs systemd-run"; exit 69; }
  lm=$(systemctl show -p LoadState --value "$slice" 2> /dev/null) || lm=
  fp=$(systemctl show -p FragmentPath --value "$slice" 2> /dev/null) || fp=
  if [ "$lm" != loaded ] || [ -z "$fp" ]; then
    say "mac-static.sh: $slice is not installed on this host (LoadState '${lm:-unknown}', FragmentPath '${fp:-none}'): systemd-run would make a slice with no limits; install its unit file first"
    exit 69
  fi
  systemd-run --scope --quiet --slice="$slice" -- env STATIC_IN_SLICE=1 timeout --kill-after=10 "$NATIVE_BOUND" bash "$0" --native --slice "$slice" "$@" || rc=$?
  [ "$rc" != 124 ] || say "mac-static.sh: --native ran past its bound of $NATIVE_BOUND s in $slice and was killed (rc 124)"
  exit "$rc"
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
  if [ -n "$NATIVE" ]; then
    ( step_inner fmt-check "$d" native_fmt ) & pids+=($!) && names+=(fmt-check)
  else
    ( step_inner fmt-check "$d" "$HERE/tools/mac-check.sh" fmt-check ) & pids+=($!) && names+=(fmt-check)
  fi
  local s
  local -a args
  local w
  for s in "${SCRIPTS[@]}"; do
    args=()
    while IFS= read -r w; do args+=("$w"); done < <(check_argv "$s" "$base")
    ( step_inner "check-$s" "$d" bash "$HERE/tools/check-$s.sh" ${args[@]+"${args[@]}"} ) & pids+=($!) && names+=(check-$s)
  done
  if [ -n "$NATIVE" ]; then
    ( step_inner lint "$d" native_lint
      step_inner combos "$d" native_combos "$base" "${base:+$LEDGER}" ) & pids+=($!) && names+=(cargo-lane)
  else
    ( step_inner lint "$d" "$HERE/tools/mac-check.sh" lint
      step_inner combos "$d" "$HERE/tools/mac-check.sh" "${cargo[@]}" ) & pids+=($!) && names+=(cargo-lane)
  fi
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
# cargo steps' closing line comes from their log, a check's is its log's last non-empty line (cut at 160 characters). A step that
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
    lint) extra=$(grep -E '^(mac-check|native): (lint \^warning|lint rc)' "$d/$name.log" | tail -1 || true) ;;
    combos) extra=$(grep -E '^(mac-check|native): combos rc' "$d/$name.log" | tail -1 || true) ;;
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

  # Each entry reads its own results. Run 1's model-tax ends red while zlate waits for a release file; run 2 finishes green before
  # run 1 resumes. Sharing a result directory would let run 1 read run 2's zero for model-tax and end green.
  local fx p1 rc1 rc2 mode
  for mode in --checks --full; do
    fx=$(mktemp -d)
    mkdir "$fx/tools" "$fx/sync"
    cp "$HERE/tools/mac-static.sh" "$fx/tools/mac-static.sh"
    printf '#!/usr/bin/env bash\n[ "${CHECK_ROLE:-}" != slow ] || exit 7\n' > "$fx/tools/check-model-tax.sh"
    printf '#!/usr/bin/env bash\nexit 0\n' > "$fx/tools/mac-check.sh"
    chmod +x "$fx/tools/mac-check.sh"
    # shellcheck disable=SC2016 # the check's own shell expands them
    printf '#!/usr/bin/env bash\n[ "$CHECK_ROLE" != slow ] || { n=0; until [ -e "$CHECK_SYNC/release" ]; do n=$((n + 1)); [ "$n" -le 600 ] || exit 9; sleep 0.1; done; }\n' > "$fx/tools/check-zlate.sh"
    rc1=0; rc2=0
    CHECK_SYNC=$fx/sync CHECK_ROLE=slow bash "$fx/tools/mac-static.sh" "$mode" > "$fx/out1" 2>&1 & p1=$!
    k=0
    until [ "$(cat "$fx"/target/mac-static/check-model-tax.rc "$fx"/target/mac-static/run.*/check-model-tax.rc "$fx"/target/mac-static/checks/check-model-tax.rc "$fx"/target/mac-static/checks/run.*/check-model-tax.rc 2> /dev/null)" = 7 ]; do
      k=$((k + 1))
      [ "$k" -le 600 ] || { fail "the first run's model-tax result did not land within 60 s"; break; }
      sleep 0.1
    done
    out=$(CHECK_SYNC=$fx/sync CHECK_ROLE=green bash "$fx/tools/mac-static.sh" "$mode" 2>&1) || rc2=$?
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

  # --native (the header): the host refusal, the recipes' command shapes, the lint ratchet, the combos run and its ledger, and the
  # whole run in a fixture tree with a fake cargo and recipes.py. No cargo runs.
  fx=$(mktemp -d)
  mkdir "$fx/tools" "$fx/bin" "$fx/home"
  rc=0; native_host Darwin 2> /dev/null || rc=$?
  [ "$rc" = 64 ] || fail "--native on Darwin ends rc $rc, not 64"
  native_host Linux || fail "--native on Linux is refused"
  # the recipes' commands as words: one plain cargo line of the right verb, nothing else
  cat > "$fx/tools/recipes.py" << 'PY'
import os, sys
here = os.path.dirname(os.path.abspath(__file__))
open(os.path.join(here, "recipes.log"), "a").write(" ".join(sys.argv[1:]) + "\n")
if sys.argv[1] == "box-command":
    print(open(os.path.join(here, "cmd." + sys.argv[2])).read().strip())
elif sys.argv[1] == "combos":
    sys.stdout.write(open(os.path.join(here, "combos.out")).read())
PY
  printf '%s\n' 'cargo clippy --workspace --all-targets --features gpu,x/y' > "$fx/tools/cmd.lint"
  printf '%s\n' 'cargo fmt --all -- --check' > "$fx/tools/cmd.fmt-check"
  out=$(HERE=$fx native_words lint clippy | tr '\n' ' ')
  [ "$out" = "cargo clippy --workspace --all-targets --features gpu,x/y " ] || fail "native_words derives '$out'"
  printf '%s\n' 'cargo clippy --workspace | tee /tmp/x' > "$fx/tools/cmd.lint"
  rc=0; out=$(HERE=$fx native_words lint clippy 2>&1) || rc=$?
  [ "$rc" = 64 ] && case $out in *"has shell syntax"*) true ;; *) false ;; esac || fail "a piped box command ends rc $rc: $out"
  printf '%s\n' 'cargo check --workspace' > "$fx/tools/cmd.lint"
  rc=0; out=$(HERE=$fx native_words lint clippy 2>&1) || rc=$?
  [ "$rc" = 64 ] && case $out in *"is not \`cargo clippy"*) true ;; *) false ;; esac || fail "a box command of another verb ends rc $rc: $out"
  # the ratchet
  printf '# rule\n\n7\n' > "$fx/tools/lint-ratchet.txt"
  [ "$(native_ratchet "$fx/tools/lint-ratchet.txt")" = 7 ] || fail "native_ratchet does not read 7 past comments and blanks"
  printf '1\n2\n' > "$fx/tools/two-numbers.txt"
  rc=0; native_ratchet "$fx/tools/two-numbers.txt" > /dev/null 2>&1 || rc=$?
  [ "$rc" = 70 ] || fail "a ratchet file with two numbers ends rc $rc, not 70"
  rc=0; native_ratchet "$fx/tools/none.txt" > /dev/null 2>&1 || rc=$?
  [ "$rc" = 70 ] || fail "a missing ratchet file ends rc $rc, not 70"
  # a cargo that prints $WARN warnings and ends $CARGO_RC (or 101 for a package named redpkg), and logs its argument words and the commit word
  cat > "$fx/bin/cargo" << 'SH'
#!/bin/sh
echo "$BLOOMERY_GIT_COMMIT $*" >> "$FX/cargo.log"
case "$*" in *redpkg*) echo "error: red" >&2; exit 101 ;; esac
n=0
while [ "$n" -lt "${WARN:-0}" ]; do echo "warning: w$n"; n=$((n + 1)); done
exit "${CARGO_RC:-0}"
SH
  chmod +x "$fx/bin/cargo"
  export FX=$fx
  printf '%s\n' 'cargo clippy --workspace --all-targets' > "$fx/tools/cmd.lint"
  out=$(PATH="$fx/bin:$PATH" WARN=7 HERE=$fx native_lint 2>&1) && rc=0 || rc=$?
  [ "$rc" = 0 ] && case $out in *"native: lint ^warning: 7 = the ratchet 7 in tools/lint-ratchet.txt"*) true ;; *) false ;; esac || fail "lint at the ratchet: rc $rc: $out"
  out=$(PATH="$fx/bin:$PATH" WARN=8 HERE=$fx native_lint 2>&1) && rc=0 || rc=$?
  [ "$rc" = 1 ] && case $out in *"native: lint ^warning: 8, above the ratchet 7"*"— red") true ;; *) false ;; esac || fail "lint above the ratchet: rc $rc: $out"
  out=$(PATH="$fx/bin:$PATH" WARN=5 HERE=$fx native_lint 2>&1) && rc=0 || rc=$?
  [ "$rc" = 0 ] && case $out in *"native: lint ^warning: 5, below the ratchet 7"*) true ;; *) false ;; esac || fail "lint below the ratchet: rc $rc: $out"
  out=$(PATH="$fx/bin:$PATH" WARN=2 CARGO_RC=101 HERE=$fx native_lint 2>&1) && rc=0 || rc=$?
  [ "$rc" = 101 ] && case $out in *"native: lint rc 101 (^warning: 2, the ratchet 7"*) true ;; *) false ;; esac || fail "a red clippy: rc $rc: $out"
  grep -q '^static-check clippy --workspace --all-targets$' "$fx/cargo.log" || fail "the lint did not run the recipe's words under the fixed commit word: $(cat "$fx/cargo.log")"
  # the combos: every shape runs, a red one too; a green one joins the ledger only in the scoped form; the counts close the step
  printf 'combo\tapp test\tcargo check -p okpkg\tkeyone\ncombo\tbad\tcargo check -p redpkg\tkeytwo\ncombo\tlast\tcargo check -p lastpkg --features a,b\tkeythree\nskip\tx\twhy\ntotal\t3 commands\n' > "$fx/tools/combos.out"
  : > "$fx/cargo.log"; : > "$fx/tools/recipes.log"; rm -f "$fx/ledger.tsv"
  out=$(PATH="$fx/bin:$PATH" HERE=$fx native_combos main "$fx/ledger.tsv" 2>&1) && rc=0 || rc=$?
  [ "$rc" = 101 ] || fail "a red shape ends combos rc $rc, not its 101"
  case $out in *"native: red shape bad: cargo check -p redpkg"*"native: combos rc 101; 3 shapes run, 1 red, 1 skipped (scoped), in "*" s (recipes.py combos: 3 commands)"*) ;; *) fail "the combos closing lines are wrong: $out" ;; esac
  [ "$(grep -c '' "$fx/cargo.log")" = 3 ] || fail "combos ran $(grep -c '' "$fx/cargo.log") shapes, not all three after a red one"
  [ "$(cut -f1,2 "$fx/ledger.tsv" | tr '\t\n' ' _')" = "keyone green_keythree green_" ] || fail "the ledger holds '$(cut -f1,2 "$fx/ledger.tsv" 2> /dev/null | tr '\t\n' ' _')', not the two green shapes"
  [ "$(cat "$fx/tools/recipes.log")" = "combos --base main --ledger $fx/ledger.tsv" ] || fail "recipes.py got '$(cat "$fx/tools/recipes.log")'"
  rm -f "$fx/ledger.tsv"; : > "$fx/tools/recipes.log"
  out=$(PATH="$fx/bin:$PATH" HERE=$fx native_combos '' '' 2>&1) && rc=0 || rc=$?
  [ ! -e "$fx/ledger.tsv" ] && [ "$(cat "$fx/tools/recipes.log")" = combos ] || fail "the full form wrote a ledger or passed flags: $(cat "$fx/tools/recipes.log")"
  printf 'combo\tp\tcargo check -p a && rm -rf x\tk\ntotal\t1\n' > "$fx/tools/combos.out"
  rc=0; out=$(PATH="$fx/bin:$PATH" HERE=$fx native_combos '' '' 2>&1) || rc=$?
  [ "$rc" = 64 ] || fail "a shape with shell syntax ends rc $rc, not 64: $out"
  printf 'total\t0\n' > "$fx/tools/combos.out"
  rc=0; out=$(PATH="$fx/bin:$PATH" HERE=$fx native_combos '' '' 2>&1) || rc=$?
  [ "$rc" = 70 ] || fail "a full run with no shape ends rc $rc, not 70: $out"
  rc=0; out=$(PATH="$fx/bin:$PATH" HERE=$fx native_combos main "$fx/ledger.tsv" 2>&1) || rc=$?
  [ "$rc" = 0 ] || fail "a scoped run with every shape skipped ends rc $rc, not 0: $out"
  # enter_slice: a slice with no unit file is refused, an installed one runs this script again inside it, under the bound
  cat > "$fx/bin/systemctl" << 'SH'
#!/bin/sh
case "$*" in
  *LoadState*) echo "${LOAD:-loaded}" ;;
  *FragmentPath*) echo "${FRAG-/etc/systemd/system/x.slice}" ;;
esac
SH
  cat > "$fx/bin/systemd-run" << 'SH'
#!/bin/sh
echo "$*" >> "$FX/systemd-run.log"
exit "${RUN_RC:-0}"
SH
  chmod +x "$fx/bin/systemctl" "$fx/bin/systemd-run"
  rc=0; out=$(PATH="$fx/bin:$PATH" FX=$fx LOAD=not-found enter_slice x.slice 2>&1) || rc=$?
  [ "$rc" = 69 ] && case $out in *"x.slice is not installed"*"no limits"*) true ;; *) false ;; esac || fail "a slice with no unit ends rc $rc: $out"
  rc=0; out=$(PATH="$fx/bin:$PATH" FX=$fx FRAG='' enter_slice x.slice 2>&1) || rc=$?
  [ "$rc" = 69 ] || fail "a slice with no unit file ends rc $rc, not 69 (systemd-run would make a transient one with no limits): $out"
  [ ! -e "$fx/systemd-run.log" ] || fail "a refused slice still ran systemd-run"
  rc=0; out=$(PATH="$fx/bin:$PATH" FX=$fx RUN_RC=3 enter_slice x.slice origin/main 2>&1) || rc=$?
  [ "$rc" = 3 ] || fail "the run's rc 3 ends enter_slice rc $rc"
  [ "$(cat "$fx/systemd-run.log")" = "--scope --quiet --slice=x.slice -- env STATIC_IN_SLICE=1 timeout --kill-after=10 $NATIVE_BOUND bash $0 --native --slice x.slice origin/main" ] ||
    fail "systemd-run got '$(cat "$fx/systemd-run.log")'"
  rc=0; out=$(PATH="$fx/bin:$PATH" FX=$fx RUN_RC=124 enter_slice x.slice 2>&1) || rc=$?
  [ "$rc" = 124 ] && case $out in *"past its bound of $NATIVE_BOUND s"*) true ;; *) false ;; esac || fail "a run killed at the bound ends rc $rc: $out"
  rm -f "$fx/systemd-run.log"
  # the whole run in a fixture tree: Linux (a uname stub), the fake cargo, recipes.py and one check script
  cp "$HERE/tools/mac-static.sh" "$fx/tools/mac-static.sh"
  printf '#!/usr/bin/env bash\necho "aa out"\n' > "$fx/tools/check-aa.sh"
  printf '#!/bin/sh\necho Linux\n' > "$fx/bin/uname"
  chmod +x "$fx/bin/uname"
  printf 'combo\tapp test\tcargo check -p okpkg\tkeyone\ntotal\t1 commands\n' > "$fx/tools/combos.out"
  printf '%s\n' 'cargo clippy --workspace --all-targets' > "$fx/tools/cmd.lint"
  : > "$fx/cargo.log"; : > "$fx/tools/recipes.log"
  rc=0
  out=$(env -i PATH="$fx/bin:/usr/bin:/bin" HOME="$fx/home" FX="$fx" WARN=7 bash "$fx/tools/mac-static.sh" --native origin/main 2>&1) || rc=$?
  [ "$rc" = 0 ] || fail "the native run in the fixture tree ends rc $rc: $out"
  for k in "mac-static: native lane, combos ledger $fx/home/.cache/bloomery/native-static-ledger.tsv" \
    "mac-static: fmt-check rc 0 in " "mac-static: lint rc 0 in " "native: lint ^warning: 7 = the ratchet 7" "mac-static: combos rc 0 in " "native: combos rc 0; 1 shapes run, 0 red, 0 skipped (scoped)" \
    "mac-static: check-aa rc 0 in " "aa out" "mac-static: total " "base origin/main, ledger $fx/home/.cache/bloomery/native-static-ledger.tsv" "mac-static: ok"; do
    case $out in *"$k"*) ;; *) fail "the native run's output lacks '$k': $out" ;; esac
  done
  case $out in *", slice "*) fail "a native run with no --slice names a slice: $out" ;; esac
  [ ! -e "$fx/systemd-run.log" ] || fail "a native run with no --slice ran systemd-run: $(cat "$fx/systemd-run.log")"
  [ "$(cut -f1,2 "$fx/home/.cache/bloomery/native-static-ledger.tsv")" = "$(printf 'keyone\tgreen')" ] || fail "the native run's ledger is '$(cat "$fx/home/.cache/bloomery/native-static-ledger.tsv" 2>&1)'"
  grep -q '^combos --base origin/main --ledger '"$fx"'/home/.cache/bloomery/native-static-ledger.tsv$' "$fx/tools/recipes.log" || fail "recipes.py combos got: $(cat "$fx/tools/recipes.log")"
  grep -q '^static-check fmt --all -- --check$' "$fx/cargo.log" || fail "fmt-check ran '$(cat "$fx/cargo.log")'"
  rc=0
  out=$(env -i PATH="$fx/bin:/usr/bin:/bin" HOME="$fx/home" FX="$fx" WARN=9 bash "$fx/tools/mac-static.sh" --native 2>&1) || rc=$?
  [ "$rc" = 1 ] && case $out in *"mac-static: lint rc 1 in "*"above the ratchet 7"*"base none (full run), ledger "*"red steps: lint (rc 1)"*) true ;; *) false ;; esac || fail "an over-ratchet native run ends rc $rc: $out"
  rc=0; out=$(env -i PATH="$fx/bin:/usr/bin:/bin" HOME="$fx/home" FX="$fx" bash "$fx/tools/mac-static.sh" --native --bogus 2>&1) || rc=$?
  [ "$rc" = 64 ] || fail "--native --bogus ends rc $rc, not 64"
  # the same run in a slice: the fake systemd-run execs what follows its --, the fake timeout drops its flags
  printf '#!/bin/sh\nwhile [ "$1" != -- ]; do shift; done; shift\nexec "$@"\n' > "$fx/bin/systemd-run"
  printf '#!/bin/sh\nwhile [ "$#" -gt 0 ]; do case "$1" in -*|[0-9]*) shift ;; *) break ;; esac; done\nexec "$@"\n' > "$fx/bin/timeout"
  chmod +x "$fx/bin/systemd-run" "$fx/bin/timeout"
  rc=0; out=$(env -i PATH="$fx/bin:/usr/bin:/bin" HOME="$fx/home" FX="$fx" WARN=7 bash "$fx/tools/mac-static.sh" --native --slice x.slice origin/main 2>&1) || rc=$?
  [ "$rc" = 0 ] || fail "the native run in a slice ends rc $rc: $out"
  case $out in *"mac-static: native lane, combos ledger $fx/home/.cache/bloomery/native-static-ledger.tsv, slice x.slice, bound $NATIVE_BOUND s"*"mac-static: ok"*) ;; *) fail "the native run in a slice does not say so: $out" ;; esac
  for a in '--slice' '--slice --full' '--slice x' '--slice a/b.slice'; do
    rc=0; out=$(env -i PATH="$fx/bin:/usr/bin:/bin" HOME="$fx/home" FX="$fx" bash "$fx/tools/mac-static.sh" --native $a 2>&1) || rc=$?
    [ "$rc" = 64 ] && case $out in *"--slice takes a slice unit name"*) true ;; *) false ;; esac || fail "--native $a ends rc $rc, not 64: $out"
  done
  unset FX
  rm -rf "$fx"

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
  --native)
    native_host "$(uname -s)" || exit $?
    shift
    slice=
    if [ "${1:-}" = --slice ]; then
      case ${2:-} in
        -* | '' | */*) say "mac-static.sh: --slice takes a slice unit name, NAME.slice"; exit 64 ;;
        *.slice) slice=$2 ;;
        *) say "mac-static.sh: --slice takes a slice unit name, NAME.slice, got '$2'"; exit 64 ;;
      esac
      shift 2
    fi
    case ${1:-} in
      --full) [ $# = 1 ] || { say "mac-static.sh: --full takes nothing"; exit 64; } ;;
      "") : ;;
      -*) say "usage: tools/mac-static.sh --native [--slice NAME.slice] [BASE|--full] ('$1' is not one)"; exit 64 ;;
      *) BASE=$1; [ $# = 1 ] || { say "mac-static.sh: one optional BASE, got $#: $*"; exit 64; } ;;
    esac
    [ -z "$slice" ] || [ -n "${STATIC_IN_SLICE:-}" ] || enter_slice "$slice" ${BASE:+"$BASE"}
    NATIVE=1
    LEDGER=$NATIVE_LEDGER
    ;;
  --full) [ $# = 1 ] || { say "mac-static.sh: --full takes nothing"; exit 64; } ;;
  "") : ;;
  -*) say "usage: tools/mac-static.sh [BASE|--full] | --native [--slice NAME.slice] [BASE|--full] | --checks [--list] [--no-base] [--skip NAME]… | --self-test ('$1' is not one)"; exit 64 ;;
  *) BASE=$1; [ $# = 1 ] || { say "mac-static.sh: one optional BASE, got $#: $*"; exit 64; } ;;
esac

D=$HERE/target/mac-static
mkdir -p "$D" || { say "mac-static.sh: cannot create $D"; exit 69; }
D=$(mktemp -d "$D/run.XXXXXX") || { say "mac-static.sh: cannot create a run directory in $D"; exit 69; }
echo "mac-static: results ${D#"$HERE/"}"
[ -z "$NATIVE" ] || echo "mac-static: native lane, combos ledger $LEDGER${slice:+, slice $slice, bound $NATIVE_BOUND s}"
CARGO=()
while IFS= read -r w; do CARGO+=("$w"); done < <(combos_argv "$BASE" "$LEDGER")
run_steps "$D" "$BASE" "${CARGO[@]}"
summarize "$D" "$BASE"
