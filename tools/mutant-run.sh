#!/usr/bin/env bash
# Mutant runner — the FAIL-first loop over a set of mutants: for each, apply it, run the gate (red is a
# kill), restore the sources from this runner's own copies, check them against the md5s recorded before
# any mutant, and run the gate again (green is the clean run the kill is read against).
#
# Usage: tools/mutant-run.sh [--allow-dirty] [--retries N] [--out DIR] [--copy DIR]
#                            [--clean each|once|first] [--expect NAME=ERE]...
#                            --crate NAME [--crate NAME]... <mutant dir> -- <gate command...>
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
#   --copy DIR     before the first mutant, rsync this tree's tracked files and untracked non-ignored
#                  files into DIR (created if absent; an existing DIR is refreshed by checksum, the rule
#                  mac-remote.sh uses: -rlpc). target/ and .git are not transferred and are not removed
#                  from DIR, so a rebuild there stays warm and box.sh's remote ~/repo/<basename of DIR>
#                  stays the same directory between runs. Any other file already in DIR is removed.
#                  Every apply, restore and gate then runs in DIR; the tree this runner
#                  was started in is not written. DIR must not be that tree, inside it, or a parent of it
#                  (refused by name), and --out must not be DIR or inside it. A flock on DIR
#                  (`.mutant-run.lock`, held until this process exits) refuses a second run of the same
#                  DIR by name, so two runs cannot share it. Every line this runner
#                  writes names DIR. The gate's own text in the run log is left as the gate wrote it.
#   --clean W      each (the default): a clean run after every mutant. once: one clean run, after the last
#                  mutant — for a gate whose run and rebuild cost minutes. first: one clean run before any
#                  mutant and none after. A leading run that is not green (red, no Compiling line, or
#                  still exiting 75) stops with a named error and no mutant runs. The restore and its md5
#                  check still follow every mutant. A red clean run under once means no kill above can be
#                  read. first does not also run once's final clean: the leading run is the one clean run,
#                  and a green one is what those kills are read against.
#   --expect NAME=ERE
#                  repeatable, one per mutant. NAME is killed only when the gate exits nonzero and the run
#                  log matches ERE (an extended regex, a line at a time). A nonzero exit that would
#                  otherwise be a kill and does not match is RED-OTHER. A mutant with no --expect keeps
#                  the rule below. NOT-RUN, NOT-BUILT, NO-COMPILE and a zero exit are unchanged.
#   The gate command runs from the tree's top directory (from DIR with --copy), stdout and stderr to
#   `<out>/<run>.log`.
#
# Output, one line per run: `mut-<name> rc=<rc> <HH:MM:SS> killed|SURVIVED|NOT-BUILT|NOT-RUN|NO-COMPILE <crate>|RED-OTHER|LOST-DAEMON`
# and `clean-after-<name> rc=<rc> <HH:MM:SS> green|RED|NO-COMPILE <crate>` (under --clean once, one
# `clean-once …` line after the last mutant; under --clean first, one `clean-first …` line before any
# mutant), then a summary line. With any --expect, the summary also counts `red-other`.
# A kill is any other nonzero rc of a run that compiled, a gate's timeout (124, 137) included, except
# where --expect says the log does not match (RED-OTHER; the run is then not PASS). Not a kill:
# a mutant that does not compile (NOT-BUILT: cargo printed `error: could not compile`), a run the gate
# runner refused or could not start (NOT-RUN: 64 a runner usage error, 69 a lock file it cannot open —
# tools/gpu-gate.sh's codes — 126 or 127 no command), and a client that lost the daemon (LOST-DAEMON:
# rc 84, its own count, not RED). That line also prints `python3 tools/boxq.py attach sNNN` when the
# log names `job sNNN`.
# The runner never calls git checkout, restore or stash. On every exit — the end, an error, a signal — a
# trap restores a mutated file from its copy, and the md5 check against the record runs; a file that
# changed while the gate ran (another editor) has that content saved under <out>/changed/ before the
# restore, and the run ends 3.
#
# Exit status: 0 every mutant killed and every clean run green, each with its Compiling lines; 1 a
# mutant survived, was RED-OTHER or LOST-DAEMON or did not build, a clean run was red or lost the
# daemon, or a Compiling line was missing;
# 2 a usage error or a refusal before the first mutant; 3 a restore that did not reach the recorded md5,
# or a file changed during a run; 75 the gate still exited 75 after the retries; 128+n a signal.
set -uo pipefail

md5_of() {
  if command -v md5sum > /dev/null 2>&1; then md5sum < "$1" | cut -c1-32; else md5 -q < "$1"; fi
}

self_test() {
  local t me fails=0 out rc pid
  me=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")
  local boxq; boxq=$(dirname "$me")/boxq.py  # the owner of rc 84 (a lost daemon); the fake gate runs it
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
  # mutant does to the run; with no marker the gate is green, or red under T_CLEANRED (a gate red only
  # on the clean tree).
  cat > "$t/gate.sh" << 'EOF'
#!/usr/bin/env bash
s=$(cat src/a.rs src/b.rs)
case "$s" in *QUIET*) ;; *) echo "   Compiling fake v0.1.0 (/x)" ;; esac
case "$s" in
  *KILL*) echo "killed-by-clause"; exit 1 ;;
  *OTHER*) echo "unrelated-refusal"; exit 1 ;;
  *COPYDIR*) echo "ran-in=$(pwd -P)"
    if [ ! -f "${T_SRC:-}/src/a.rs" ]; then echo source-missing
    elif grep -q COPYDIR "${T_SRC}/src/a.rs"; then echo source-was-written
    else echo source-untouched; fi
    exit 1 ;;
  *SURVIVE*) exit 0 ;;
  *NOBUILD*) echo "error: could not compile \`fake\` (lib)"; exit 101 ;;
  *QUIET*) exit 1 ;;
  *NOTRUN*) exit 69 ;;
  *LOST*) BOXQ_SSH=false BOXQ_POLL_S=0.01 python3 "$T_BOXQ" attach s184; exit $? ;;
  *BREAKCOPY*) printf 'x\n' >> "$T_OUT/orig/src/a.rs"; exit 1 ;;
  *EDIT*) printf '// an edit made during the run\n' >> src/a.rs; exit 1 ;;
  *SIGNAL*) for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do [ -s "$T_PID" ] && break; sleep 0.1; done
            kill -TERM "$(cat "$T_PID")"; exit 1 ;;
esac
[ -z "${T_CLEANRED:-}" ] || exit 1
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
    out=$(cd "$t/repo" && T_OUT=$o T_PID=$t/pid T_SRC=$t/repo T_BOXQ=$boxq T_CLEANRED=${CLEANRED:-} bash "$me" --out "$o" --crate fake "$@" "$m" -- "$t/gate.sh" 2>&1)
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
  # --clean once: one clean run after the last mutant; the restore check still follows every mutant.
  run "$t/o" "$t/muts" --clean once
  expect "--clean once" 0 "mutant-run: 2 mutants, 2 killed, 0 survived, 0 not built, 0 not run; clean runs 1 green — PASS"
  expect "--clean once, its one clean run" 0 "clean-once rc=0"
  if grep -qF clean-after- <<< "$out"; then
    echo "mutant-run self-test: --clean once ran a clean run after a mutant" >&2
    fails=$((fails + 1))
  fi
  clean "--clean once"
  CLEANRED=1 run "$t/o" "$t/muts" --clean once
  expect "--clean once, a gate red only on the clean tree" 1 "no kill above can be read"
  expect "--clean once, a gate red only on the clean tree, after the last mutant" 1 "mut-m2 rc=1"
  clean "--clean once, a gate red only on the clean tree"
  CLEANRED=1 run "$t/o" "$t/muts"
  expect "--clean each, a gate red only on the clean tree" 1 "clean-after-m1 rc=1"
  clean "--clean each, a gate red only on the clean tree"
  one c a.rs 'BREAKCOPY();'
  run "$t/o" "$t/one" --clean once
  expect "--clean once, a restore from a broken copy" 3 "RESTORE FAILED: src/a.rs"
  cp "$t/a.rs" "$t/repo/src/a.rs"
  clean "--clean once, a restore from a broken copy"
  run "$t/o" "$t/muts" --clean some
  expect "--clean with another value" 2 "--clean some is not each or once"
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
  # --copy: the source bytes stay put and the gate's cwd is DIR. A second run refreshes DIR.
  tree_md5() {
    local root=$1 f
    (cd "$root" && find . -path ./.git -prune -o -type f -print | LC_ALL=C sort | while IFS= read -r f; do
      md5_of "$f"
      printf ' %s\n' "$f"
    done)
  }
  one y a.rs 'COPYDIR();'
  rm -rf "$t/workcopy"
  local srcmd copydir l
  srcmd=$(tree_md5 "$t/repo")
  run "$t/o" "$t/one" --copy "$t/workcopy"
  expect "--copy, the mutant is killed in DIR" 0 "killed"
  if [ ! -d "$t/workcopy" ]; then
    echo "mutant-run self-test: --copy: DIR was not created" >&2
    fails=$((fails + 1))
  else
    copydir=$(cd "$t/workcopy" && pwd -P)
    [ "$(basename "$copydir")" = workcopy ] || { echo "mutant-run self-test: --copy: DIR's basename is not the one given" >&2; fails=$((fails + 1)); }
    while IFS= read -r l; do
      [ -n "$l" ] || continue
      case "$l" in
        *"$copydir"*) ;;
        *) echo "mutant-run self-test: --copy: a log line does not name DIR: $l" >&2; fails=$((fails + 1)) ;;
      esac
    done <<< "$out"
    if ! grep -qF "ran-in=$copydir" "$t/o/mut-y.log"; then
      echo "mutant-run self-test: --copy: the gate did not run in $copydir" >&2
      fails=$((fails + 1))
    fi
    if ! grep -qF source-untouched "$t/o/mut-y.log"; then
      echo "mutant-run self-test: --copy: the source tree was written while the mutant was applied" >&2
      fails=$((fails + 1))
    fi
    printf 'stale\n' > "$copydir/stale-left.rs"
    mkdir -p "$copydir/target"
    printf 'warm\n' > "$copydir/target/warm"
    run "$t/o" "$t/one" --copy "$t/workcopy"
    expect "--copy, a second run refreshes DIR" 0 "dir=$copydir"
    if [ -e "$copydir/stale-left.rs" ]; then
      echo "mutant-run self-test: --copy left a file the tree does not have" >&2
      fails=$((fails + 1))
    fi
    if [ "$(cat "$copydir/target/warm" 2> /dev/null)" != warm ]; then
      echo "mutant-run self-test: --copy removed the copy's target/" >&2
      fails=$((fails + 1))
    fi
    if [ "$(tree_md5 "$t/repo")" != "$srcmd" ]; then
      echo "mutant-run self-test: --copy: a source md5 changed" >&2
      fails=$((fails + 1))
    fi
    if [ -n "$(git -C "$t/repo" status --porcelain)" ]; then
      echo "mutant-run self-test: --copy left the source tree dirty" >&2
      fails=$((fails + 1))
    fi
  fi
  clean "--copy"
  run "$t/o" "$t/muts" --copy "$t/repo/insidecopy"
  expect "--copy inside the tree" 2 "inside the tree"
  if [ -e "$t/repo/insidecopy" ]; then
    echo "mutant-run self-test: --copy created a directory inside the source tree" >&2
    fails=$((fails + 1))
    rm -rf "$t/repo/insidecopy"
  fi
  clean "--copy inside the tree"
  # --clean first: a red clean gate stops the run before any mutant.
  CLEANRED=1 run "$t/o" "$t/muts" --clean first
  expect "--clean first, a red clean gate" 1 "before any mutant"
  if grep -qE '^mut-' <<< "$out"; then
    echo "mutant-run self-test: --clean first ran a mutant after a red clean gate" >&2
    fails=$((fails + 1))
  fi
  clean "--clean first, a red clean gate"
  run "$t/o" "$t/muts" --clean first
  expect "--clean first, the leading clean run" 0 "clean-first rc=0"
  expect "--clean first, the mutants ran" 0 "mut-m2 rc=1"
  expect "--clean first, one clean run" 0 "clean runs 1 green — PASS"
  if grep -qF clean-after- <<< "$out" || grep -qF clean-once <<< "$out"; then
    echo "mutant-run self-test: --clean first ran a clean run after a mutant" >&2
    fails=$((fails + 1))
  fi
  clean "--clean first"
  # --expect: an unmatched nonzero exit is RED-OTHER, not a kill. A match stays a kill.
  one o a.rs 'OTHER();'
  run "$t/o" "$t/one" --expect 'o=killed-by-clause'
  expect "--expect, an unmatched red" 1 "RED-OTHER"
  expect "--expect, the summary counts red-other" 1 "1 red-other"
  if grep -qF '— PASS' <<< "$out"; then
    echo "mutant-run self-test: --expect: an unmatched red was PASS" >&2
    fails=$((fails + 1))
  fi
  clean "--expect, an unmatched red"
  one k a.rs 'KILL();'
  run "$t/o" "$t/one" --expect 'k=killed-by-clause'
  expect "--expect, a matched red is a kill" 0 "killed"
  expect "--expect, a matched red passes" 0 "— PASS"
  clean "--expect, a matched red"
  run "$t/o" "$t/one" --expect 'k=['
  expect "--expect, a pattern that is not an extended regex" 2 "not an extended regex"
  run "$t/o" "$t/one" --expect 'no-such=clause'
  expect "--expect, a name that is not a mutant" 2 "matches no mutant"
  run "$t/o" "$t/muts" --expect 'm1=a' --expect 'm1=b'
  expect "--expect given twice" 2 "given twice"
  # rc 84 is a lost daemon: its own count, not a kill and not RED, and the line names the attach command.
  one d a.rs 'LOST();'
  run "$t/o" "$t/one"
  expect "a daemon poll that returns 84" 1 "LOST-DAEMON"
  expect "a daemon poll that returns 84 names the attach command" 1 "python3 tools/boxq.py attach s184"
  if grep -qE 'mut-d rc=84 .*killed' <<< "$out" || grep -qE 'mut-d rc=84 .*RED' <<< "$out"; then
    echo "mutant-run self-test: rc 84 was killed or RED" >&2
    fails=$((fails + 1))
  fi
  expect "a daemon poll that returns 84 has its own count" 1 "1 lost-daemon"
  clean "a daemon poll that returns 84"
  # A second run cannot share a --copy DIR while the first holds its lock.
  one y a.rs 'KILL();'
  rm -rf "$t/held"
  mkdir -p "$t/held"
  python3 -c 'import fcntl, os, sys, time
fcntl.flock(os.open(sys.argv[1], os.O_RDWR | os.O_CREAT, 0o644), fcntl.LOCK_EX)
time.sleep(30)' "$t/held/.mutant-run.lock" &
  hold=$!
  sleep 0.2
  run "$t/o" "$t/one" --copy "$t/held"
  expect "a second run of the same --copy DIR" 2 "in use by another mutant-run"
  kill "$hold" 2> /dev/null || true
  wait "$hold" 2> /dev/null || true
  clean "a second run of the same --copy DIR"
  rm -rf "$t"
  [ "$fails" = 0 ] && echo "mutant-run: self-test ok" || { echo "mutant-run: self-test $fails failed" >&2; return 1; }
}
if [ "${1:-}" = --self-test ]; then
  self_test
  exit $?
fi

# A line this runner writes. With --copy it names DIR; without, it is echo.
logline() { if [ -n "${COPY:-}" ]; then echo "$1 dir=$COPY"; else echo "$1"; fi; }
logerr() { logline "$1" >&2; }

# norm_abs PATH — PATH with . and .. collapsed. PATH may not exist. Relative to the cwd.
norm_abs() {
  local p=$1 part rest n
  local -a out=()
  case "$p" in
    /*) ;;
    *) p="$(pwd)/$p" ;;
  esac
  rest=${p#/}
  while [ -n "$rest" ]; do
    part=${rest%%/*}
    case "$rest" in
      */*) rest=${rest#*/} ;;
      *) rest= ;;
    esac
    case "$part" in
      '' | .) ;;
      ..)
        n=${#out[@]}
        if [ "$n" -gt 0 ]; then
          unset "out[$((n - 1))]"
          if [ "${#out[@]}" -eq 0 ]; then out=(); else out=("${out[@]}"); fi
        fi
        ;;
      *) out+=("$part") ;;
    esac
  done
  if [ "${#out[@]}" -eq 0 ]; then printf '/\n'; return; fi
  printf '/%s' "${out[@]}"
  printf '\n'
}

# phys_abs PATH — norm_abs, then the longest existing ancestor through pwd -P.
phys_abs() {
  local p rest=
  p=$(norm_abs "$1")
  while [ ! -d "$p" ]; do
    [ "$p" = / ] && break
    rest="/$(basename "$p")$rest"
    p=$(dirname "$p")
  done
  if [ -d "$p" ]; then p=$(cd "$p" && pwd -P); fi
  printf '%s%s\n' "$p" "$rest"
}

# copy_tree SRC DST — DST gets SRC's tracked files and untracked non-ignored files.
# -rlpc is mac-remote.sh's checksum rsync: a changed file is sent, an unchanged one keeps its
# mtime. A files-from line is prefixed with ./ because rsync drops a line that starts with '#'
# or ';'. target/ and .git are not sent and not deleted, so DST's build stays warm. Any other
# file in DST that the list does not name is removed. git init makes DST its own work tree, so
# git apply patches DST even when DST sits inside some other repository.
copy_tree() {
  local src=$1 dst=$2 f list=$OUT/copy.list keep=$OUT/copy.keep rel
  command -v rsync > /dev/null 2>&1 || usage "rsync is not on PATH; --copy needs it"
  : > "$list"
  : > "$keep"
  while IFS= read -r -d '' f; do
    case "$f" in
      '' | target | target/* | .git | .git/* | .mutant-run.lock) continue ;;
      *$'\n'*) usage "--copy refuses a path that contains a newline" ;;
    esac
    [ -e "$src/$f" ] || [ -L "$src/$f" ] || continue
    printf './%s\n' "$f" >> "$list"
    printf '%s\n' "$f" >> "$keep"
  done < <(git -C "$src" ls-files --cached --others --exclude-standard -z)
  mkdir -p "$dst" || usage "cannot make --copy $dst"
  git -C "$dst" init -q > "$OUT/git-init.err" 2>&1 ||
    usage "cannot make a git work tree in $dst ($(tr '\n' ' ' < "$OUT/git-init.err"))"
  if [ -s "$list" ]; then
    rsync -rlpc --files-from="$list" "$src/" "$dst/" > "$OUT/rsync.err" 2>&1 ||
      usage "rsync into $dst failed ($(tr '\n' ' ' < "$OUT/rsync.err"))"
  fi
  LC_ALL=C sort -u "$keep" -o "$keep"
  : > "$OUT/copy.dest"
  while IFS= read -r -d '' f; do
    rel=${f#"$dst"/}
    case "$rel" in
      .git | .git/* | target | target/* | .mutant-run.lock) continue ;;
    esac
    if [ -d "$f" ] && [ ! -L "$f" ]; then continue; fi
    case "$rel" in *$'\n'*) usage "--copy refuses an existing path that contains a newline" ;; esac
    printf '%s\n' "$rel" >> "$OUT/copy.dest"
  done < <(find "$dst" -mindepth 1 \( -path "$dst/.git" -o -path "$dst/.git/*" -o -path "$dst/target" -o -path "$dst/target/*" \) -prune -o -print0)
  while IFS= read -r rel; do
    [ -n "$rel" ] || continue
    rm -f "$dst/$rel" || usage "cannot remove $dst/$rel"
  done < <(awk 'NR==FNR { keep[$0] = 1; next } !($0 in keep) { print }' "$keep" "$OUT/copy.dest")
  while IFS= read -r -d '' f; do
    rel=${f#"$dst"/}
    case "$rel" in
      .git | .git/* | target | target/* | .mutant-run.lock) continue ;;
    esac
    rmdir "$f" 2> /dev/null || true
  done < <(find "$dst" -mindepth 1 -depth \( -path "$dst/.git" -o -path "$dst/.git/*" -o -path "$dst/target" -o -path "$dst/target/*" \) -prune -o -type d -print0)
}

# expect_for NAME — prints NAME's ERE and returns 0, or returns 1 when NAME has no --expect.
expect_for() {
  local i
  [ "${#EXPECT_NAMES[@]}" -gt 0 ] || return 1
  for i in "${!EXPECT_NAMES[@]}"; do
    if [ "${EXPECT_NAMES[$i]}" = "$1" ]; then
      printf '%s' "${EXPECT_ERES[$i]}"
      return 0
    fi
  done
  return 1
}

# check_expects — every --expect names a mutant and its ERE compiles. Before the first mutant.
check_expects() {
  local i name ere bn p found erc
  [ "${#EXPECT_NAMES[@]}" -gt 0 ] || return 0
  for i in "${!EXPECT_NAMES[@]}"; do
    name=${EXPECT_NAMES[$i]}
    ere=${EXPECT_ERES[$i]}
    found=0
    for p in "${PATCHES[@]}"; do
      bn=$(basename "$p")
      bn=${bn%.*}
      if [ "$bn" = "$name" ]; then found=1; break; fi
    done
    [ "$found" = 1 ] || usage "--expect $name matches no mutant"
    erc=0
    grep -E -q -e "$ere" /dev/null 2> "$OUT/expect.err" || erc=$?
    [ "$erc" != 2 ] || usage "--expect $name: pattern is not an extended regex ($(tr '\n' ' ' < "$OUT/expect.err"))"
  done
}

usage() {
  logerr "mutant-run: $1"
  if [ -n "${COPY:-}" ] || [ "${#EXPECT_NAMES[@]}" -gt 0 ] || [ "${CLEAN:-each}" = first ]; then
    logerr "usage: mutant-run.sh [--allow-dirty] [--retries N] [--out DIR] [--copy DIR] [--clean each|once|first] [--expect NAME=ERE] --crate NAME [--crate NAME]... <mutant dir> -- <gate command...>"
  else
    echo "usage: mutant-run.sh [--allow-dirty] [--retries N] [--out DIR] [--clean each|once] --crate NAME [--crate NAME]... <mutant dir> -- <gate command...>" >&2
  fi
  exit 2
}
ALLOW_DIRTY=0
RETRIES=5
CLEAN=each
OUT=
COPY=
CRATES=()
EXPECT_NAMES=()
EXPECT_ERES=()
MUTDIR=
while [ $# -gt 0 ]; do
  case "$1" in
    --allow-dirty) ALLOW_DIRTY=1 ;;
    --retries) [ $# -ge 2 ] || usage "--retries takes a count"; RETRIES=$2; shift ;;
    --out) [ $# -ge 2 ] || usage "--out takes a directory"; OUT=$2; shift ;;
    --copy)
      [ $# -ge 2 ] || usage "--copy takes a directory"
      [ -n "$2" ] || usage "--copy takes a directory"
      [ -z "$COPY" ] || usage "a second --copy $2"
      COPY=$2
      shift ;;
    --clean) [ $# -ge 2 ] || usage "--clean takes each, once or first"; CLEAN=$2; shift ;;
    --expect)
      [ $# -ge 2 ] || usage "--expect takes NAME=ERE"
      case "$2" in
        *=*) ;;
        *) usage "--expect $2 is not NAME=ERE" ;;
      esac
      ename=${2%%=*}
      eere=${2#*=}
      [ -n "$ename" ] || usage "--expect $2 has an empty name"
      [ -n "$eere" ] || usage "--expect $2 has an empty pattern"
      if [ "${#EXPECT_NAMES[@]}" -gt 0 ]; then
        for ei in "${!EXPECT_NAMES[@]}"; do
          [ "${EXPECT_NAMES[$ei]}" = "$ename" ] && usage "--expect $ename given twice"
        done
      fi
      EXPECT_NAMES+=("$ename")
      EXPECT_ERES+=("$eere")
      shift ;;
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
case "$CLEAN" in each | once | first) ;; *) usage "--clean $CLEAN is not each or once or first" ;; esac
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
if [ -n "$COPY" ]; then
  if [ -e "$COPY" ] && [ ! -d "$COPY" ]; then usage "--copy $COPY is not a directory"; fi
  COPY=$(phys_abs "$COPY")
  ROOT_P=$(cd "$ROOT" && pwd -P)
  [ "$COPY" != / ] || usage "--copy / contains the tree $ROOT"
  if [ "$COPY" = "$ROOT" ] || [ "$COPY" = "$ROOT_P" ]; then usage "--copy $COPY is the tree $ROOT"; fi
  case "$COPY/" in
    "$ROOT"/* | "$ROOT_P"/*) usage "--copy $COPY is inside the tree $ROOT" ;;
  esac
  case "$ROOT_P/" in
    "$COPY"/*) usage "--copy $COPY contains the tree $ROOT" ;;
  esac
  # --out inside DIR would be deleted when DIR is refreshed.
  if [ "$OUT" = "$COPY" ] || [ "${OUT#"$COPY"/}" != "$OUT" ]; then
    usage "--out $OUT is inside --copy $COPY"
  fi
fi

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
check_expects
# Validation above is read-only on the tree we were started in. With --copy, every later read and
# write of a source file happens in DIR.
if [ -n "$COPY" ]; then
  mkdir -p "$COPY" || usage "cannot make --copy $COPY"
  exec 9>>"$COPY/.mutant-run.lock" || usage "cannot open the lock for --copy $COPY"
  # flock(2) on fd 9 through python3's fcntl (the Mac has no flock(1)): the lock is the open file
  # description's, so it stays held by this shell after python3 exits.
  python3 -c 'import fcntl; fcntl.flock(9, fcntl.LOCK_EX | fcntl.LOCK_NB)' 2> /dev/null ||
    usage "--copy $COPY is in use by another mutant-run"
  copy_tree "$ROOT" "$COPY"
  cd "$COPY" || usage "cannot enter --copy $COPY"
  ROOT=$COPY
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
        logerr "mutant-run: $f changed while $APPLIED ran; that content is saved at $OUT/changed/$f"
        bad=1
      fi
      cp "$OUT/orig/$f" "$f" || true
    fi
    now=$(md5_of "$f")
    if [ "$now" != "$sum" ]; then
      logerr "mutant-run: RESTORE FAILED: $f md5 $now, recorded $sum (copy at $OUT/orig/$f)"
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
trap 'logerr "mutant-run: stopped by INT${APPLIED:+ during $APPLIED}"; exit 130' INT
trap 'logerr "mutant-run: stopped by TERM${APPLIED:+ during $APPLIED}"; exit 143' TERM

# run_gate <run> — the gate under the retries, appended to one log; sets GRC.
run_gate() {
  local run=$1 try=1
  : > "$OUT/$run.log"
  while :; do
    logline "== $run attempt $try $(date +%H:%M:%S)" >> "$OUT/$run.log"
    "${CMD[@]}" >> "$OUT/$run.log" 2>&1 < /dev/null
    GRC=$?
    [ "$GRC" = 75 ] && [ "$try" -lt "$RETRIES" ] || break
    logerr "$run attempt $try rc=75 (lock contention), again"
    try=$((try + 1))
  done
}
# lost_verdict <run> — LOST-DAEMON plus the attach command. The id is the log's `job sNNN`.
lost_verdict() {
  local id
  id=$(grep -oE 'job s[0-9]+' "$OUT/$1.log" | head -1 | cut -d' ' -f2)
  if [ -n "$id" ]; then
    printf 'LOST-DAEMON python3 tools/boxq.py attach %s' "$id"
  else
    printf 'LOST-DAEMON python3 tools/boxq.py attach'
  fi
}

# compiled <run> — the first crate whose Compiling line the run's log lacks, or nothing.
compiled() {
  local c
  for c in "${CRATES[@]}"; do
    grep -qF "Compiling $c v" "$OUT/$1.log" || { echo "$c"; return; }
  done
}

# clean_run <run> — the gate on the restored tree; a red run stops the runner: every kill is read against it.
clean_run() {
  local run=$1
  run_gate "$run"
  miss=$(compiled "$run")
  if [ "$GRC" = 75 ]; then
    logline "$run rc=75 $(date +%H:%M:%S) contention after $RETRIES attempts"
    FINAL_RC=75; exit 75
  elif [ "$GRC" = 84 ]; then
    logline "$run rc=84 $(date +%H:%M:%S) $(lost_verdict "$run")"
    logerr "mutant-run: the gate's daemon poll was lost on the clean tree; no kill can be read"
    FINAL_RC=1; exit 1
  elif [ -n "$miss" ]; then
    logline "$run rc=$GRC $(date +%H:%M:%S) NO-COMPILE $miss"
    if [ "$run" = clean-first ]; then
      logerr "mutant-run: the clean run before any mutant did not compile $miss; no mutant will run"
      FINAL_RC=1; exit 1
    fi
    fail=1
  elif [ "$GRC" = 0 ]; then
    logline "$run rc=0 $(date +%H:%M:%S) green"; green=$((green + 1))
  else
    logline "$run rc=$GRC $(date +%H:%M:%S) RED"
    if [ "$run" = clean-first ]; then
      logerr "mutant-run: the gate is red on the clean tree before any mutant; no mutant will run"
    elif [ "$CLEAN" = once ]; then
      logerr "mutant-run: the gate is red on the clean tree after the last mutant; no kill above can be read"
    else
      logerr "mutant-run: the gate is red on the clean tree; stopping"
    fi
    FINAL_RC=1; exit 1
  fi
}

CMD=("$@")
logline "mutant-run: ${#PATCHES[@]} mutants from $MUTDIR over $(wc -l < "$OUT/files" | tr -d ' ') files, logs in $OUT (clean runs: $CLEAN)"
killed=0 survived=0 notbuilt=0 notrun=0 redother=0 lost=0 green=0 fail=0
if [ "$CLEAN" = first ]; then clean_run clean-first; fi
for p in "${PATCHES[@]}"; do
  name=$(basename "$p")
  name=${name%.*}
  restore || { FINAL_RC=3; exit 3; }
  git apply "$p" 2> "$OUT/apply.err" || { sed 's/^/  /' "$OUT/apply.err" >&2; logerr "mutant-run: $p did not apply"; FINAL_RC=1; exit 1; }
  APPLIED=mut-$name
  : > "$OUT/applied.md5"
  while IFS= read -r f; do printf '%s  %s\n' "$(md5_of "$f")" "$f" >> "$OUT/applied.md5"; done < "$OUT/files"
  run_gate "mut-$name"
  miss=$(compiled "mut-$name")
  if [ "$GRC" = 75 ]; then
    verdict="contention after $RETRIES attempts"
  elif [ "$GRC" = 84 ]; then
    verdict=$(lost_verdict "mut-$name")
    lost=$((lost + 1)); fail=1
  elif [ "$GRC" = 64 ] || [ "$GRC" = 69 ] || [ "$GRC" = 126 ] || [ "$GRC" = 127 ]; then
    verdict=NOT-RUN; notrun=$((notrun + 1)); fail=1
  elif grep -qF 'error: could not compile' "$OUT/mut-$name.log"; then
    verdict=NOT-BUILT; notbuilt=$((notbuilt + 1)); fail=1
  elif [ -n "$miss" ]; then
    verdict="NO-COMPILE $miss"; fail=1
  elif [ "$GRC" = 0 ]; then
    verdict=SURVIVED; survived=$((survived + 1)); fail=1
  elif ere=$(expect_for "$name"); then
    if grep -a -E -q -e "$ere" "$OUT/mut-$name.log"; then
      verdict=killed; killed=$((killed + 1))
    else
      verdict=RED-OTHER; redother=$((redother + 1)); fail=1
    fi
  else
    verdict=killed; killed=$((killed + 1))
  fi
  logline "mut-$name rc=$GRC $(date +%H:%M:%S) $verdict"
  restore || { FINAL_RC=3; exit 3; }
  [ "$GRC" != 75 ] || { FINAL_RC=75; exit 75; }
  [ "$CLEAN" = each ] || continue
  clean_run "clean-after-$name"
done
[ "$CLEAN" = once ] && clean_run clean-once
verdict=PASS
[ "$fail" = 0 ] || verdict=FAIL
sum="mutant-run: ${#PATCHES[@]} mutants, $killed killed, $survived survived, $notbuilt not built, $notrun not run"
if [ "${#EXPECT_NAMES[@]}" -gt 0 ] || [ "$redother" -gt 0 ]; then
  sum="$sum, $redother red-other"
fi
if [ "$lost" -gt 0 ]; then
  sum="$sum, $lost lost-daemon"
fi
sum="$sum; clean runs $green green — $verdict"
logline "$sum"
[ "$fail" = 0 ] || FINAL_RC=1
exit "$FINAL_RC"
