#!/usr/bin/env bash
# bloomery — the Mac static tier in one command, the round loop's form: fmt-check, lint, the scoped combos and every
# tools/check-*.sh, check-recipes as its blocks (tools/check-recipes.sh --jobs), all of them jobs of one scheduler,
# tools/static-sched.py: at most a cpu count at once (lint and combos count the cores they compile on), the longest
# recorded first (~/.cache/bloomery/static-times.tsv), combos after lint (one target directory, one cargo lock). Every
# job at once oversubscribed the Mac and turned timing self-tests red under the load alone.
#
# A run with a BASE keys every job but combos (tools/recipes.py static-key: the content of the tree the job reads, the
# commit of a revision its keyspec names, the command and the environment, the rustflags among it) and skips a job whose
# key is green in the ledger; a skipped step or block is a summary line of its own. --fresh keys and records but skips
# nothing (a proof run); a full run (no BASE) keys nothing.
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
#   tools/mac-static.sh --self-test     the job list, the scheduling (through tools/static-sched.py on stub
#                                       jobs), the rc/wall capture, the skip lines, the red-step naming and
# the argument plumbing; runs no cargo and no check script (check-recipes runs it)
#
# Exit: the first red step's rc, or 0. 64: an argument this script does not take. A step that wrote
# no rc file (killed before it ended) is red, named `no rc`. Jobs' full output and their rc/wall land in
# target/mac-static/<job>.{log,rc,wall} (a skipped one also <job>.skip; sched.tsv is the run's timeline);
# check-recipes' step files are its blocks' report; the cargo steps also write the mac-check logs
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
TIMES=$HOME/.cache/bloomery/static-times.tsv
# The cores a cargo step holds in the schedule: cargo compiles on every core while a crate graph allows.
CARGO_CPUS=4
SCRIPTS=(arch bounded comments defaults emb2 levers loads model-tax oracle-tree ptx-names records recipes rustflags unsafe waits)

say() { printf '%s\n' "$*" >&2; }

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
  printf 'lint\tcpus=%s\ttree\ttools/mac-check.sh lint\n' "$CARGO_CPUS"
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

# summarize D BASE: one line a step in a fixed order (fmt-check, lint, combos, every check),
# then the verdict. Prints `mac-static: <step> rc <n> in <s> s [— the step's own closing line]`; the
# cargo steps' closing line comes from their log, check-recipes' is its count of blocks run and skipped, a
# skipped step's names its green row. Then one line naming every skipped job (steps and check-recipes' blocks)
# when any was. Ends 0, or prints one line naming every red step and ends with the first red rc.
summarize() {
  local d=$1 base=$2 red=0 firstrc=0 s name rc wall extra f skipped=''
  local -a reds=()
  one_line() { # NAME: the step's summary line and its red bookkeeping
    local name=$1
    if [ ! -f "$d/$name.rc" ]; then
      echo "mac-static: $name rc ? no rc in $((SECONDS - T0)) s (killed before it ended)"
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
      check-recipes) extra=$(grep -E '^check-recipes: [0-9]+ blocks ran' "$d/$name.log" | tail -1 || true) ;;
    esac
    [ ! -f "$d/$name.skip" ] || extra="skipped, its input key is $(cat "$d/$name.skip")"
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
  for f in "$d"/*.skip; do
    [ -e "$f" ] || continue
    f=${f##*/}
    skipped="$skipped ${f%.skip}"
  done
  [ -z "$skipped" ] || echo "mac-static: skipped by a green input key:$skipped"
  echo "mac-static: total $((SECONDS - T0)) s, base ${base:-none (full run)}, ledger ${base:+$LEDGER}"
  if [ "$red" != 0 ]; then
    say "mac-static: red steps: ${reds[@]+"${reds[*]}"}"
    return "$firstrc"
  fi
  echo "mac-static: ok"
}

self_test() {
  local fails=0 d out rc sp cmd got
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

  # every tools/check-*.sh is a step: a check script SCRIPTS leaves out never runs in the round loop
  local f name unlisted=''
  for f in "$HERE"/tools/check-*.sh; do
    name=${f##*/check-}
    name=${name%.sh}
    case " ${SCRIPTS[*]} " in *" $name "*) ;; *) unlisted="$unlisted $name" ;; esac
  done
  [ -z "$unlisted" ] || fail "tools/check-*.sh that SCRIPTS does not list:$unlisted"

  # job_lines: fmt-check, lint, combos after lint (the cargo steps holding CARGO_CPUS), one job a check script, the
  # model-tax key holding the base revision, check-recipes as its blocks and not as one step
  out=$(job_lines '' combos --base main)
  [ "$(printf '%s\n' "$out" | head -3 | cut -f1,2 | tr '\t\n' ' ')" = "fmt-check - lint cpus=$CARGO_CPUS combos after=lint,cpus=$CARGO_CPUS " ] \
    || fail "the cargo jobs' lines: $(printf '%s\n' "$out" | head -3)"
  grep -qx "combos	after=lint,cpus=$CARGO_CPUS	-	tools/mac-check.sh combos --base main" <<< "$out" || fail "the combos job is not the combos argv, unkeyed"
  for s in "${SCRIPTS[@]}"; do
    [ "$s" = recipes ] || grep -q "^check-$s	" <<< "$out" || fail "check-$s is not a job"
  done
  ! grep -q '^check-recipes	' <<< "$out" || fail "check-recipes is a job of its own, not its blocks"
  grep -q '^check-recipes\.cardtests	' <<< "$out" || fail "check-recipes' blocks are not jobs"
  grep -q '^check-recipes\.gatebatch\.1	' <<< "$out" || fail "gate-batch's self-test parts are not jobs"
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
  --full) [ $# = 1 ] || { say "mac-static.sh: --full takes nothing"; exit 64; } ;;
  "") : ;;
  -*) say "usage: tools/mac-static.sh [--fresh] [BASE|--full] | --self-test ('$1' is not one)"; exit 64 ;;
  *) BASE=$1; [ $# = 1 ] || { say "mac-static.sh: one optional BASE, got $#: $*"; exit 64; } ;;
esac
[ -n "$BASE" ] || MODE=full

D=$HERE/target/mac-static
mkdir -p "$D"
rm -f "$D"/*.rc "$D"/*.wall "$D"/*.skip "$D"/pids "$D"/sched.tsv 2> /dev/null || true
CARGO=()
while IFS= read -r w; do CARGO+=("$w"); done < <(combos_argv "$BASE" "$LEDGER")
job_lines "$BASE" "${CARGO[@]}" > "$D/jobs.tsv"
run_jobs "$D" "$D/jobs.tsv" "$MODE" || true # a scheduler that ended early leaves its steps without rc files: the summary names them
summarize "$D" "$BASE"
