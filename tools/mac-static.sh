#!/usr/bin/env bash
# bloomery — the Mac static tier in one command, the round loop's form. The loop a round runs before
# it reports was a sequence (mac-check check, mac-check combos, mac-check lint, fmt-check, the eight
# tools/check-*.sh one after another) whose walls measured 2026-10-07 (load ~5, this Mac): the cargo
# steps 0.6-40 s each but the eight scripts ~240 s serial, `check-recipes` alone 172 s. Two lanes cut
# it to the longer lane: the scripts lane (fmt-check and the eight checks, one job each) and the cargo
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
#                                       naming and the argument plumbing, against /bin/true,
# /bin/false and sleep stubs; runs no cargo and no check script (check-recipes runs it)
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
SCRIPTS=(arch comments levers loads recipes rustflags unsafe waits)

say() { printf '%s\n' "$*" >&2; }

# combos_argv BASE LEDGER: the mac-check combos argv for the combos step (the scoped form only when
# BASE is non-empty — the full form is the lead's landing one).
combos_argv() {
  local a=(combos)
  [ -z "$1" ] || a+=(--base "$1")
  [ -z "$2" ] || [ -z "$1" ] || a+=(--ledger "$2")
  printf '%s\n' "${a[@]}"
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
# the eight tools/check-*.sh). The cargo lane: one background job running lint then the combos step
# serially (one target directory, one lock). Every job's pid is captured at start, written to
# D/pids, and waited on; no polling.
run_steps() {
  local d=$1 base=$2
  shift 2
  local -a cargo=("$@") pids=() names=()
  : > "$d/pids"
  ( step_inner fmt-check "$d" "$HERE/tools/mac-check.sh" fmt-check ) & pids+=($!) && names+=(fmt-check)
  local s
  for s in "${SCRIPTS[@]}"; do
    ( step_inner "check-$s" "$d" bash "$HERE/tools/check-$s.sh" ) & pids+=($!) && names+=(check-$s)
  done
  ( step_inner lint "$d" "$HERE/tools/mac-check.sh" lint
    step_inner combos "$d" "$HERE/tools/mac-check.sh" "${cargo[@]}" ) & pids+=($!) && names+=(cargo-lane)
  local i
  for i in "${!pids[@]}"; do echo "${names[$i]} ${pids[$i]}" >> "$d/pids"; done
  wait "${pids[@]}" || true # a killed wrapper is the summary's `no rc` case, not this script's end
}

# summarize D BASE: one line a step in a fixed order (fmt-check, lint, combos, the eight checks),
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
  local fails=0 d out rc t k
  local -a pids=()
  fail() { say "mac-static self-test FAIL: $*"; fails=$((fails + 1)); }

  # combos_argv: the full form without a base, the scoped form with both, no ledger without a base
  out=$(combos_argv '' "$LEDGER" | tr '\n' ' ')
  [ "$out" = "combos " ] || fail "no base derives '$out'"
  out=$(combos_argv main '' | tr '\n' ' ')
  [ "$out" = "combos --base main " ] || fail "a base without a ledger derives '$out'"
  out=$(combos_argv 'HEAD~3..HEAD' "$HOME/x.tsv" | tr '\n' ' ')
  [ "$out" = "combos --base HEAD~3..HEAD --ledger $HOME/x.tsv " ] || fail "a base and a ledger derive '$out'"

  # the lanes: two sleeping steps overlap (their serial sum is far more than the wall), rcs and walls
  # land, a failing step is red and named, a step with no rc file is red as `no rc`
  d=$(mktemp -d)
  T0=$SECONDS
  np=0
  ( step_inner sleep-a "$d" bash -c 'sleep 2' ) & pids[$np]=$!; np=$((np + 1))
  ( step_inner sleep-b "$d" bash -c 'sleep 2' ) & pids[$np]=$!; np=$((np + 1))
  ( step_inner false-step "$d" bash -c 'exit 3' ) & pids[$np]=$!; np=$((np + 1))
  ( step_inner true-step "$d" bash -c 'echo ok' ) & pids[$np]=$!; np=$((np + 1))
  wait "${pids[@]}" || true
  t=$((SECONDS - T0))
  [ "$t" -lt 4 ] || fail "two 2 s steps took $t s serial, not parallel"
  [ "$(cat "$d/true-step.rc")" = 0 ] || fail "a true step's rc is not 0"
  [ "$(cat "$d/false-step.rc")" = 3 ] || fail "an exit-3 step's rc is not 3"
  [ "$(cat "$d/true-step.wall")" -lt 3 ] || fail "a true step's wall is not its own"
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
  --full) [ $# = 1 ] || { say "mac-static.sh: --full takes nothing"; exit 64; } ;;
  "") : ;;
  -*) say "usage: tools/mac-static.sh [BASE|--full] | --self-test ('$1' is not one)"; exit 64 ;;
  *) BASE=$1; [ $# = 1 ] || { say "mac-static.sh: one optional BASE, got $#: $*"; exit 64; } ;;
esac

D=$HERE/target/mac-static
mkdir -p "$D"
rm -f "$D"/*.rc "$D"/*.wall "$D"/pids 2> /dev/null || true
CARGO=()
while IFS= read -r w; do CARGO+=("$w"); done < <(combos_argv "$BASE" "$LEDGER")
run_steps "$D" "$BASE" "${CARGO[@]}"
summarize "$D" "$BASE"
