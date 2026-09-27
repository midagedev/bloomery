#!/usr/bin/env bash
# Mutant runner — the FAIL-first loop over a set of mutants: for each, apply it, run the gate (red is a
# kill), restore the sources from this runner's own copies, check them against the md5s recorded before
# any mutant, and run the gate again (green is the clean run the kill is read against).
#
# Usage: tools/mutant-run.sh [--allow-dirty] [--retries N] [--out DIR] --crate NAME [--crate NAME]...
#                            <mutant dir> -- <gate command...>
#        tools/mutant-run.sh --self-test   (a temp git repo, fake mutants and a fake gate; check-recipes runs it)
#
#   <mutant dir>   holds the mutants, `<name>.patch` or `<name>.diff`, each `git apply`-able on the tree as
#                  it is (every one is checked before the first runs), in name order. A mutant edits files
#                  only: one that creates, deletes, renames or changes the mode of a file is refused — a
#                  copy cannot undo it.
#   --crate NAME   a crate the mutants live in (repeatable). Every gate run, mutated and clean, must print
#                  cargo's `Compiling NAME v…` line for each: box.sh syncs by content, so a changed file is
#                  rebuilt, and a run without the line ran a binary built from other source.
#   --allow-dirty  run over files with uncommitted changes (they are restored to that content). Without
#                  it such a tree is refused, so a restore cannot stand between the tree and real edits.
#   --retries N    runs of the gate while it exits 75 (lock contention), default 5; the attempts append
#                  to one log.
#   --out DIR      the copies, the md5 record and one log per run (default a new temp dir; never inside
#                  the tree, which box.sh syncs).
#   The gate command runs from the tree's top directory, stdout and stderr to `<out>/<run>.log`.
#
# Output, one line per run: `mut-<name> rc=<rc> <HH:MM:SS> killed|SURVIVED|NOT-BUILT|NOT-RUN|NO-COMPILE <crate>`
# and `clean-after-<name> rc=<rc> <HH:MM:SS> green|RED|NO-COMPILE <crate>`, then a summary line.
# A kill is any other nonzero rc of a run that compiled, a gate's timeout (124, 137) included. Not a kill:
# a mutant that does not compile (NOT-BUILT: cargo printed `error: could not compile`), and a run the gate
# runner refused or could not start (NOT-RUN: 64 a runner usage error, 69 a lock file it cannot open —
# tools/gpu-gate.sh's codes — 126 or 127 no command).
# The runner never calls git checkout, restore or stash. On every exit — the end, an error, a signal — a
# trap restores a mutated file from its copy, and the md5 check against the record runs; a file that
# changed while the gate ran (another editor) has that content saved under <out>/changed/ before the
# restore, and the run ends 3.
#
# Exit status: 0 every mutant killed and every clean run green, each with its Compiling lines; 1 a
# mutant survived or did not build, a clean run was red, or a Compiling line was missing; 2 a usage
# error or a refusal before the first mutant; 3 a restore that did not reach the recorded md5, or a
# file changed during a run; 75 the gate still exited 75 after the retries; 128+n a signal.
set -uo pipefail

md5_of() {
  if command -v md5sum > /dev/null 2>&1; then md5sum < "$1" | cut -c1-32; else md5 -q < "$1"; fi
}

self_test() {
  local t me fails=0 out rc pid
  me=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")
  t=$(mktemp -d)
  mkdir -p "$t/repo/src" "$t/muts" "$t/bad"
  local g="git -C $t/repo -c user.name=t -c user.email=t@t -c core.hooksPath=/dev/null"
  $g init -q
  printf 'fn a() {\n    one();\n}\n' > "$t/repo/src/a.rs"
  printf 'fn b() {\n    two();\n}\n' > "$t/repo/src/b.rs"
  $g add src
  $g commit -qm fixture
  cp "$t/repo/src/a.rs" "$t/a.rs"
  cp "$t/repo/src/b.rs" "$t/b.rs"
  mkpatch() { # mkpatch <dir> <name> <file> <new second line>
    printf 'fn x() {\n    %s\n}\n' "$4" > "$t/repo/src/$3"
    $g diff > "$1/$2.patch"
    cp "$t/$3" "$t/repo/src/$3"
  }
  # The fake gate: prints cargo's Compiling line, then reads the tree. A marker in a source picks what a
  # mutant does to the run; with no marker the gate is green.
  cat > "$t/gate.sh" << 'EOF'
#!/usr/bin/env bash
s=$(cat src/a.rs src/b.rs)
case "$s" in *QUIET*) ;; *) echo "   Compiling fake v0.1.0 (/x)" ;; esac
case "$s" in
  *KILL*) exit 1 ;;
  *SURVIVE*) exit 0 ;;
  *NOBUILD*) echo "error: could not compile \`fake\` (lib)"; exit 101 ;;
  *QUIET*) exit 1 ;;
  *NOTRUN*) exit 69 ;;
  *BREAKCOPY*) printf 'x\n' >> "$T_OUT/orig/src/a.rs"; exit 1 ;;
  *EDIT*) printf '// an edit made during the run\n' >> src/a.rs; exit 1 ;;
  *SIGNAL*) for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do [ -s "$T_PID" ] && break; sleep 0.1; done
            kill -TERM "$(cat "$T_PID")"; exit 1 ;;
esac
exit 0
EOF
  chmod +x "$t/gate.sh"
  mkpatch "$t/muts" m1 a.rs 'KILL();'
  mkpatch "$t/muts" m2 b.rs 'KILL();'
  local orig_a orig_b
  orig_a=$(md5_of "$t/a.rs")
  orig_b=$(md5_of "$t/b.rs")
  run() { # run <out dir> <mutant dir> [flags...] — the runner on the fixture; sets out and rc
    local o=$1 m=$2
    shift 2
    rm -rf "$o"
    out=$(cd "$t/repo" && T_OUT=$o T_PID=$t/pid bash "$me" --out "$o" --crate fake "$@" "$m" -- "$t/gate.sh" 2>&1)
    rc=$?
  }
  expect() { # expect <case> <want rc> <want substring>
    if [ "$rc" != "$2" ] || ! grep -qF -- "$3" <<< "$out"; then
      echo "mutant-run self-test: $1: rc $rc (want $2), want '$3' in:" >&2
      sed 's/^/  /' <<< "$out" >&2
      fails=$((fails + 1))
    fi
  }
  clean() { # the fixture's sources are the committed ones
    if [ "$(md5_of "$t/repo/src/a.rs")" != "$orig_a" ] || [ "$(md5_of "$t/repo/src/b.rs")" != "$orig_b" ]; then
      echo "mutant-run self-test: $1: a source is not the committed one after the run" >&2
      fails=$((fails + 1))
      cp "$t/a.rs" "$t/repo/src/a.rs"
      cp "$t/b.rs" "$t/repo/src/b.rs"
    fi
  }
  one() { # one <name> <file> <line> — a mutant dir holding one mutant
    rm -rf "$t/one"
    mkdir "$t/one"
    mkpatch "$t/one" "$1" "$2" "$3"
  }

  run "$t/o" "$t/muts"
  expect "two mutants, both killed" 0 "mutant-run: 2 mutants, 2 killed, 0 survived, 0 not built, 0 not run; clean runs 2 green — PASS"
  expect "the clean run after m2" 0 "clean-after-m2 rc=0"
  clean "two mutants"
  one s a.rs 'SURVIVE();'
  run "$t/o" "$t/one"
  expect "a surviving mutant" 1 "mut-s rc=0"
  expect "a surviving mutant, its verdict" 1 "SURVIVED"
  clean "a surviving mutant"
  one n a.rs 'NOBUILD();'
  run "$t/o" "$t/one"
  expect "a mutant that does not compile is not a kill" 1 "NOT-BUILT"
  clean "a mutant that does not compile"
  one q a.rs 'QUIET();'
  run "$t/o" "$t/one"
  expect "a run without the Compiling line" 1 "NO-COMPILE fake"
  clean "a run without the Compiling line"
  one r a.rs 'NOTRUN();'
  run "$t/o" "$t/one"
  expect "a gate runner that could not run the gate is not a kill" 1 "mut-r rc=69"
  expect "a gate runner that could not run the gate, its verdict" 1 "NOT-RUN"
  clean "a gate runner that could not run the gate"
  one c a.rs 'BREAKCOPY();'
  run "$t/o" "$t/one"
  expect "a restore from a broken copy" 3 "RESTORE FAILED: src/a.rs"
  cp "$t/a.rs" "$t/repo/src/a.rs"
  clean "a restore from a broken copy"
  one e a.rs 'EDIT();'
  run "$t/o" "$t/one"
  expect "a file edited while the gate ran" 3 "src/a.rs changed while mut-e ran"
  if ! grep -qF 'EDIT();' "$t/o/changed/src/a.rs" 2> /dev/null; then
    echo "mutant-run self-test: the content edited during the run was not saved" >&2
    fails=$((fails + 1))
  fi
  clean "a file edited while the gate ran"
  # A signal: the runner in the background, its pid from $! for the gate to signal.
  one g a.rs 'SIGNAL();'
  rm -f "$t/pid"
  (cd "$t/repo" && T_OUT=$t/o T_PID=$t/pid exec bash "$me" --out "$t/o" --crate fake "$t/one" -- "$t/gate.sh" > "$t/sig.out" 2>&1) &
  pid=$!
  echo "$pid" > "$t/pid"
  wait "$pid"
  rc=$?
  out=$(cat "$t/sig.out")
  expect "a TERM while the gate runs" 143 "stopped by TERM during mut-g"
  clean "a TERM while the gate runs"
  # Refusals before the first mutant.
  printf '// uncommitted\n' >> "$t/repo/src/a.rs"
  local dirty
  dirty=$(md5_of "$t/repo/src/a.rs")
  run "$t/o" "$t/muts"
  expect "uncommitted changes to a mutated file" 2 "uncommitted changes to src/a.rs"
  run "$t/o" "$t/muts" --allow-dirty
  expect "--allow-dirty" 0 "PASS"
  if [ "$(md5_of "$t/repo/src/a.rs")" != "$dirty" ]; then
    echo "mutant-run self-test: --allow-dirty did not restore the uncommitted content" >&2
    fails=$((fails + 1))
  fi
  cp "$t/a.rs" "$t/repo/src/a.rs"
  printf 'new\n' > "$t/repo/src/c.rs"
  (cd "$t/repo" && git add -N src/c.rs && git diff > "$t/bad/create.patch" && git rm -q --cached src/c.rs)
  rm "$t/repo/src/c.rs"
  run "$t/o" "$t/bad"
  expect "a mutant that creates a file" 2 "creates, deletes, renames or changes the mode"
  run "$t/repo/o" "$t/muts"
  expect "an out dir inside the tree" 2 "inside the tree"
  clean "the refusals"
  rm -rf "$t"
  [ "$fails" = 0 ] && echo "mutant-run: self-test ok" || { echo "mutant-run: self-test $fails failed" >&2; return 1; }
}
if [ "${1:-}" = --self-test ]; then
  self_test
  exit $?
fi

usage() {
  echo "mutant-run: $1" >&2
  echo "usage: mutant-run.sh [--allow-dirty] [--retries N] [--out DIR] --crate NAME [--crate NAME]... <mutant dir> -- <gate command...>" >&2
  exit 2
}
ALLOW_DIRTY=0
RETRIES=5
OUT=
CRATES=()
MUTDIR=
while [ $# -gt 0 ]; do
  case "$1" in
    --allow-dirty) ALLOW_DIRTY=1 ;;
    --retries) [ $# -ge 2 ] || usage "--retries takes a count"; RETRIES=$2; shift ;;
    --out) [ $# -ge 2 ] || usage "--out takes a directory"; OUT=$2; shift ;;
    --crate) [ $# -ge 2 ] || usage "--crate takes a crate name"; CRATES+=("$2"); shift ;;
    --) shift; break ;;
    -*) usage "unknown flag $1" ;;
    *) [ -z "$MUTDIR" ] || usage "a second mutant dir $1 (the gate command goes after --)"; MUTDIR=$1 ;;
  esac
  shift
done
[ $# -gt 0 ] || usage "no gate command after --"
[ -n "$MUTDIR" ] || usage "no mutant dir"
[ -d "$MUTDIR" ] || usage "no mutant dir $MUTDIR"
[ ${#CRATES[@]} -gt 0 ] || usage "no --crate: name the crate whose Compiling line proves each run rebuilt"
case "$RETRIES" in '' | *[!0-9]* | 0) usage "--retries $RETRIES is not a count from 1" ;; esac
MUTDIR=$(cd "$MUTDIR" && pwd)
ROOT=$(git rev-parse --show-toplevel 2> /dev/null) || usage "not inside a git tree: $(pwd)"
cd "$ROOT" || exit 2
if [ -z "$OUT" ]; then
  OUT=$(mktemp -d "${TMPDIR:-/tmp}/mutant-run.XXXXXX")
else
  mkdir -p "$OUT" || usage "cannot make --out $OUT"
fi
OUT=$(cd "$OUT" && pwd -P)
case "$OUT/" in "$ROOT"/*) usage "--out $OUT is inside the tree $ROOT, which box.sh syncs" ;; esac

shopt -s nullglob
PATCHES=("$MUTDIR"/*.patch "$MUTDIR"/*.diff)
shopt -u nullglob
[ ${#PATCHES[@]} -gt 0 ] || usage "no *.patch or *.diff in $MUTDIR"
: > "$OUT/files.all"
for p in "${PATCHES[@]}"; do
  git apply --check "$p" 2> "$OUT/apply.err" || { sed 's/^/  /' "$OUT/apply.err" >&2; usage "$p does not apply to the tree"; }
  [ -z "$(git apply --summary "$p")" ] || usage "$p creates, deletes, renames or changes the mode of a file; a copy cannot undo that"
  git apply --numstat "$p" | cut -f3- >> "$OUT/files.all"
done
sort -u "$OUT/files.all" > "$OUT/files"
dirty=
while IFS= read -r f; do
  git ls-files --error-unmatch -- "$f" > /dev/null 2>&1 || usage "$f is not a tracked file"
  git diff --quiet HEAD -- "$f" || dirty="$dirty $f"
done < "$OUT/files"
if [ -n "$dirty" ] && [ "$ALLOW_DIRTY" = 0 ]; then
  usage "uncommitted changes to${dirty}; commit them, or pass --allow-dirty to have them restored as they are"
fi

# The record first, then the copies, each copy checked against the record: a restore is judged by
# the record, never by the copy it came from.
: > "$OUT/orig.md5"
while IFS= read -r f; do
  printf '%s  %s\n' "$(md5_of "$f")" "$f" >> "$OUT/orig.md5"
  mkdir -p "$OUT/orig/$(dirname "$f")"
  cp "$f" "$OUT/orig/$f" || usage "cannot copy $f to $OUT/orig"
done < "$OUT/files"
while read -r sum f; do
  [ "$(md5_of "$OUT/orig/$f")" = "$sum" ] || usage "the copy of $f in $OUT/orig does not match it"
done < "$OUT/orig.md5"

APPLIED=     # the run a mutant is applied for, empty when the tree is clean
FINAL_RC=0

# Every file back to its recorded md5; a file that is not what the mutant left has that content saved
# first. Prints one line per defect and returns 1 on any.
restore() {
  local bad=0 sum f now
  while read -r sum f; do
    if [ -n "$APPLIED" ]; then
      now=$(awk -v f="$f" '{ p = $0; sub(/^[^ ]+  /, "", p) } p == f { print $1 }' "$OUT/applied.md5")
      if [ "$(md5_of "$f")" != "$now" ]; then
        mkdir -p "$OUT/changed/$(dirname "$f")"
        cp "$f" "$OUT/changed/$f"
        echo "mutant-run: $f changed while $APPLIED ran; that content is saved at $OUT/changed/$f" >&2
        bad=1
      fi
      cp "$OUT/orig/$f" "$f" || true
    fi
    now=$(md5_of "$f")
    if [ "$now" != "$sum" ]; then
      echo "mutant-run: RESTORE FAILED: $f md5 $now, recorded $sum (copy at $OUT/orig/$f)" >&2
      bad=1
    fi
  done < "$OUT/orig.md5"
  APPLIED=
  return "$bad"
}
on_exit() {
  local rc=$?
  [ "$FINAL_RC" = 0 ] || rc=$FINAL_RC
  restore || rc=3
  exit "$rc"
}
trap on_exit EXIT
trap 'echo "mutant-run: stopped by INT${APPLIED:+ during $APPLIED}" >&2; exit 130' INT
trap 'echo "mutant-run: stopped by TERM${APPLIED:+ during $APPLIED}" >&2; exit 143' TERM

# run_gate <run> — the gate under the retries, appended to one log; sets GRC.
run_gate() {
  local run=$1 try=1
  : > "$OUT/$run.log"
  while :; do
    echo "== $run attempt $try $(date +%H:%M:%S)" >> "$OUT/$run.log"
    "${CMD[@]}" >> "$OUT/$run.log" 2>&1 < /dev/null
    GRC=$?
    [ "$GRC" = 75 ] && [ "$try" -lt "$RETRIES" ] || break
    echo "$run attempt $try rc=75 (lock contention), again" >&2
    try=$((try + 1))
  done
}
# compiled <run> — the first crate whose Compiling line the run's log lacks, or nothing.
compiled() {
  local c
  for c in "${CRATES[@]}"; do
    grep -qF "Compiling $c v" "$OUT/$1.log" || { echo "$c"; return; }
  done
}

CMD=("$@")
echo "mutant-run: ${#PATCHES[@]} mutants from $MUTDIR over $(wc -l < "$OUT/files" | tr -d ' ') files, logs in $OUT"
killed=0 survived=0 notbuilt=0 notrun=0 green=0 fail=0
for p in "${PATCHES[@]}"; do
  name=$(basename "$p")
  name=${name%.*}
  restore || { FINAL_RC=3; exit 3; }
  git apply "$p" 2> "$OUT/apply.err" || { sed 's/^/  /' "$OUT/apply.err" >&2; echo "mutant-run: $p did not apply" >&2; FINAL_RC=1; exit 1; }
  APPLIED=mut-$name
  : > "$OUT/applied.md5"
  while IFS= read -r f; do printf '%s  %s\n' "$(md5_of "$f")" "$f" >> "$OUT/applied.md5"; done < "$OUT/files"
  run_gate "mut-$name"
  miss=$(compiled "mut-$name")
  if [ "$GRC" = 75 ]; then
    verdict="contention after $RETRIES attempts"
  elif [ "$GRC" = 64 ] || [ "$GRC" = 69 ] || [ "$GRC" = 126 ] || [ "$GRC" = 127 ]; then
    verdict=NOT-RUN; notrun=$((notrun + 1)); fail=1
  elif grep -qF 'error: could not compile' "$OUT/mut-$name.log"; then
    verdict=NOT-BUILT; notbuilt=$((notbuilt + 1)); fail=1
  elif [ -n "$miss" ]; then
    verdict="NO-COMPILE $miss"; fail=1
  elif [ "$GRC" = 0 ]; then
    verdict=SURVIVED; survived=$((survived + 1)); fail=1
  else
    verdict=killed; killed=$((killed + 1))
  fi
  echo "mut-$name rc=$GRC $(date +%H:%M:%S) $verdict"
  restore || { FINAL_RC=3; exit 3; }
  [ "$GRC" != 75 ] || { FINAL_RC=75; exit 75; }
  run_gate "clean-after-$name"
  miss=$(compiled "clean-after-$name")
  if [ "$GRC" = 75 ]; then
    echo "clean-after-$name rc=75 $(date +%H:%M:%S) contention after $RETRIES attempts"
    FINAL_RC=75; exit 75
  elif [ -n "$miss" ]; then
    echo "clean-after-$name rc=$GRC $(date +%H:%M:%S) NO-COMPILE $miss"; fail=1
  elif [ "$GRC" = 0 ]; then
    echo "clean-after-$name rc=0 $(date +%H:%M:%S) green"; green=$((green + 1))
  else
    # A gate red with no mutant makes every later kill meaningless: stop here.
    echo "clean-after-$name rc=$GRC $(date +%H:%M:%S) RED"
    echo "mutant-run: the gate is red on the clean tree; stopping" >&2
    FINAL_RC=1; exit 1
  fi
done
verdict=PASS
[ "$fail" = 0 ] || verdict=FAIL
echo "mutant-run: ${#PATCHES[@]} mutants, $killed killed, $survived survived, $notbuilt not built, $notrun not run; clean runs $green green — $verdict"
[ "$fail" = 0 ] || FINAL_RC=1
exit "$FINAL_RC"
