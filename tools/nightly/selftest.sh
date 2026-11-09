#!/usr/bin/env bash
# `bash tools/nightly/selftest.sh --self-test`: the commit label of tools/nightly/units.sh (`tree_commit`) and the two scripts
# that print it, tools/contrib-check.sh and tools/nightly/run.sh, and `archive_tree`, on temp repositories. No card, no box, no cargo, no network.
# One line per check, `ok <name>` or `FAIL <name>: got … want …`, the verdict last (bash 3.2 on the Mac).
# --output-only checks the production output-allocation and receipt blocks without running any units.
#
# What it blocks: a `git status` that fails (a corrupt index, an unreadable tree) prints nothing, and a test of that output
# reads nothing as a clean tree. A run on such a tree then published its HEAD as the commit it tested, without `+local`, and went
# on. The label is `?` or the run stops (contrib-check 70, nightly rc 2 RED) naming the tree and git's exit code. The three other
# scripts that name a tree's commit (gate-batch.sh, ptx-compare.sh, narrow-scan.sh) test that in their own self-tests.
set -uo pipefail

if [ "${1:-}" != --self-test ]; then
  echo "usage: tools/nightly/selftest.sh --self-test" >&2
  exit 64
fi

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
# shellcheck source=tools/nightly/units.sh
source "$HERE/units.sh"

if [ "${2:-}" = --output-only ]; then
  python3 - "${OUTPUT_TEST_ROOT:-$ROOT}" << 'PY'
import pathlib, subprocess, sys, tempfile

root = pathlib.Path(sys.argv[1]).resolve()
bad = checks = 0
def check(name, ok, detail):
    global bad, checks
    print(f"{'ok' if ok else 'FAIL'} {name}: {detail}")
    checks += 1
    bad += not ok

with tempfile.TemporaryDirectory(prefix="output-owner-") as tmp:
    t = pathlib.Path(tmp)
    def run(body, *args, cwd=None):
        return subprocess.run(["bash", "-c", body, "_", *map(str, args)],
                              input="release\n", cwd=cwd, capture_output=True, text=True, timeout=10)
    common = 'set -euo pipefail; ROOT=$1; OUT=$2; DRY=0; RC=73; die() { echo "$*" >&2; exit "$RC"; }; '
    contrib = (root / "tools/contrib-check.sh").read_text()
    allocation = contrib[contrib.index('STAMP=$(date '):contrib.index("add fmt ")]
    body = common + 'date() { echo same-second; }; TIER=t0; BOUND=10; '
    body += 'run_unit() { UNIT_RC=0; bash -c "$1" > "$2" 2>&1 || UNIT_RC=$?; UNIT_SECS=0; }; unit_failure() { :; }; '
    body += allocation + '\nadd "$3" "$4"; echo ready; read -r release;\n'
    body += contrib[contrib.index('# ---- run ----'):]
    first = subprocess.Popen(["bash", "-c", body, "_", str(t), str(t / "unused"), "inject-fail", "exit 101"],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        ready = first.stdout.readline().strip()
        other = run(body, t, t / "unused", "fmt", "true")
        stdout, stderr = first.communicate("release\n", timeout=10)
        check("contrib-check keeps a requested failure after a same-second run",
              ready == "ready" and first.returncode == 1 and other.returncode == 0 and "inject-fail rc=101" in stdout,
              f"first={first.returncode} other={other.returncode} {stdout.strip()} {stderr.strip()}")
    finally:
        if first.poll() is None:
            first.kill()
            first.communicate()

    gate = (root / "tools/gate-batch.sh").read_text()
    end = gate.index('RUNLOG=$OUT/run.log')
    gate = gate[gate.rfind('if [ "$DRY" = 0 ]; then', 0, end):end]
    body = common + 'OUT_AUTO=1; ' + gate + '\nprintf "%s\\n" "$OUT"'
    a, b = (run(body, root, t / "automatic") for _ in range(2))
    check("gate-batch automatic outputs stay private",
          a.returncode == b.returncode == 0 and a.stdout != b.stdout, (a.stdout + b.stdout).strip())
    replay = (root / "tools/batch-replay.sh").read_text()
    begin = replay.index('WORK=$(mktemp ')
    replay_out = replay[begin:replay.index('# The measured seconds,')]
    for name, allocation in (("gate-batch", gate), ("batch-replay", replay_out)):
        out = t / name
        body = common + 'source "$ROOT/tools/nightly/units.sh"; ' + allocation
        owner = subprocess.Popen(["bash", "-c", body + '\necho ready; read -r release', "_", str(root), str(out)],
                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            ready = owner.stdout.readline().strip()
            rival = run(body, root, out)
            owner.stdin.write("release\n")
            owner.stdin.flush()
            stdout, stderr = owner.communicate(timeout=10)
            check(f"{name} has one output owner before any log",
                  ready == "ready" and rival.returncode == 73 and owner.returncode == 0,
                  f"owner={owner.returncode} rival={rival.returncode} {rival.stderr.strip()} {stderr.strip()}")
            retry = run(body, root, out)
            check(f"{name} releases an unused output directory", retry.returncode == 0, retry.stderr.strip())
            artifact = out / ("run.log" if name == "gate-batch" else "new.rc")
            artifact.write_text("saved\n")
            retry = run(body, root, out)
            check(f"{name} preserves existing output", retry.returncode == 73 and artifact.read_text() == "saved\n",
                  retry.stderr.strip())
            linkout = t / f"{name}-dangling"
            linkout.mkdir()
            (linkout / artifact.name).symlink_to(linkout / "missing")
            retry = run(body, root, linkout)
            check(f"{name} preserves a dangling output symlink", retry.returncode == 73, retry.stderr.strip())
        finally:
            if owner.poll() is None:
                owner.kill()
                owner.communicate()

    work, out = t / "work", t / "receipts"
    out.mkdir()
    for arm, rc in (("base", 0), ("new", 1)):
        (work / arm / "tools").mkdir(parents=True)
        (work / arm / "tools/gate-batch.sh").write_text(f"exit {rc}\n")
    arm_fn = replay[replay.index('run_arm() {'):replay.index('# A base with the budget refusal')]
    end_check = replay[replay.index('for arm in base new; do', replay.index('wait "$newpid"')):replay.index('# Both walls')]
    body = 'set -euo pipefail; WORK=$1; OUT=$2; LIST=/dev/null; ' + arm_fn
    body += '\nrun_arm base; run_arm new;\n' + end_check.replace(
        '  rc=$(cat ', '  printf 0 > "$OUT/base.rc"; printf 0 > "$OUT/new.rc"\n  rc=$(cat ')
    result = run(body, work, out)
    check("batch-replay refuses its own failed arm despite foreign rc files",
          result.returncode == 70 and "new arm ended rc=1" in result.stderr, result.stderr.strip())
    (t / "fixture.list").write_text("gate-probe\n")
    (work / "new/tools/gate-batch.sh").write_text('[ -f "$4" ] || exit 65\necho list-ok\n')
    inputs = replay[replay.index('LIST=$1 RUNLOG=$2'):replay.index('command -v just')]
    for path in ("fixture.list", str(t / "fixture.list")):
        body = 'set -euo pipefail; WORK=$1; OUT=$2; set -- "$3" "$3"; ' + inputs + arm_fn
        result = run(body + '\nrun_arm new; cat "$OUT/new.log"', work, out, path, cwd=t)
        check(f"batch-replay keeps the list path across arm directories: {path}",
              result.returncode == 0 and result.stdout.strip() == "list-ok", result.stdout + result.stderr)

    repo = t / "repo"
    (repo / "tools/ref").mkdir(parents=True)
    (repo / "tools/nightly").mkdir()
    for path in ("justfile", "tools/gate-batch.sh", "tools/ref/ref-paths.sh",
                 "tools/ref/real-only.sh", "tools/ref/fixture-only.sh", "tools/nightly/units.sh"):
        (repo / path).write_text("# fixture\n")
    recipe = "\ngate-probe:\n    ./tools/box.sh 'probe {{arch}}'\n"
    (repo / "justfile").write_text(recipe.replace("{{arch}}", "sm_86"))
    (repo / "tools/arch.sh").write_text((root / "tools/arch.sh").read_text())
    def git(*args):
        return subprocess.run(["git", "-C", str(repo), "-c", "user.name=fixture", "-c", "user.email=fixture@example.invalid",
                               "-c", "commit.gpgsign=false", *args], check=True, capture_output=True, text=True).stdout.strip()
    git("init", "-q", "--template=")
    git("add", ".")
    git("commit", "-qm", "without arch")
    no_arch = git("rev-parse", "HEAD")
    (repo / "justfile").write_text("arch := 'sm_86'\n" + recipe)
    git("add", ".")
    git("commit", "-qm", "legacy")
    legacy = git("rev-parse", "HEAD")
    (repo / "tools/gate-batch-resources.tsv").write_text("fixture\tm\t1\n")
    (repo / "tools/gate-batch.sh").write_text("# --over-budget-ok\n")
    (repo / "justfile").write_text("arch := `bash tools/arch.sh`\n" + recipe)
    git("add", ".")
    git("commit", "-qm", "current")
    copies = replay[replay.index('for arm in base new; do', replay.index('# The measured seconds,')):replay.index('# Each arm')]
    probe = replay[replay.index("BASE_BUDGET_OK=''"):replay.index('run_arm new --over-budget-ok')]
    table = replay[replay.index('python3 - "$LIST"'):replay.index('# The fake box:')]
    for base, expected in ((no_arch, ""), (legacy, ""), ("HEAD", "1")):
        mutants = ["none", "resources", "arch-helper", "arch-freeze"]
        mutants += ["arch-absent"] if base == no_arch else (["budget"] if base == "HEAD" else [])
        for mutant in mutants:
            work = t / f"copy-{base}-{mutant}"
            work.mkdir()
            (work / "secs.tsv").write_text("gate-probe\t7\tA\n")
            code = copies.replace('cp "$ROOT/tools/gate-batch-resources.tsv"', ':') if mutant == "resources" else copies
            if mutant == "arch-helper":
                code = code.replace('cp "$ROOT/tools/arch.sh"', ':')
            if mutant == "arch-freeze":
                code = code.replace('mv "$WORK/$arm/justfile.resolved" "$WORK/$arm/justfile"', ':')
            if mutant == "arch-absent":
                code = code.replace('if grep -q \'^arch := \' "$WORK/$arm/justfile"; then', 'if true; then')
            code += probe.replace('then BASE_BUDGET_OK=1;', 'then BASE_BUDGET_OK=;') if mutant == "budget" else probe
            code += table + '\necho "budget=$BASE_BUDGET_OK"'
            result = run('set -euo pipefail; export BLOOMERY_ARCH=sm_121a; ROOT=$1; WORK=$2; BASE_REV=$3; LIST=$4; ' + code,
                         repo, work, base, t / "fixture.list")
            keyed = True
            for arm, arch in (("base", "sm_121a" if base == "HEAD" else "sm_86"), ("new", "sm_121a")):
                command_table = work / arm / "table.tsv"
                if not command_table.exists() or command_table.read_text() != f"probe {arch}\tgate-probe\t7\n":
                    keyed = False
                    continue
                box = work / arm / "tools/box.sh"
                box.write_text('#!/bin/sh\nprintf "%s\\n" "$*"\n')
                box.chmod(0o755)
                actual = subprocess.run(["env", "BLOOMERY_ARCH=sm_86", "just", "gate-probe"], cwd=work / arm, capture_output=True, text=True)
                keyed = keyed and actual.returncode == 0 and actual.stdout.strip() == f"probe {arch}"
            valid = (result.returncode == 0 and result.stdout.splitlines()[-1] == f"budget={expected}" and keyed
                     and (work / "new/tools/gate-batch-resources.tsv").read_text() == "fixture\tm\t1\n"
                     and (work / "base/tools/gate-batch-resources.tsv").exists() == (base == "HEAD")) if (work / "new/tools/gate-batch-resources.tsv").exists() else False
            check(f"batch-replay base={base} snapshot and command keys mutant={mutant}", valid == (mutant == "none"),
                  f"rc={result.returncode} keyed={keyed} {result.stdout.strip()} {result.stderr.strip()}")
    work = t / "copy-invalid-arch"
    work.mkdir()
    result = run('set -euo pipefail; export BLOOMERY_ARCH=; ROOT=$1; WORK=$2; BASE_REV=HEAD; ' + copies,
                 repo, work)
    check("batch-replay refuses a failed architecture selection before making command keys",
          result.returncode != 0 and "got ''" in result.stderr and not (work / "base/table.tsv").exists(),
          f"rc={result.returncode} {result.stderr.strip()}")
print(f"output-owner self-test: {checks - bad} of {checks} ok")
sys.exit(bool(bad))
PY
  exit $?
fi

# An empty T would put every repository, fixture and nightly state below at the filesystem root.
T=$(mktemp -d "${TMPDIR:-/tmp}/tree-commit.XXXXXX") || {
  echo "tree-commit self-test: cannot create a scratch directory under ${TMPDIR:-/tmp}: nothing was checked" >&2
  exit 70
}
trap 'rm -rf "$T"' EXIT
fails=0 checks=0
check() {
  checks=$((checks + 1))
  if [ "$2" = "$3" ]; then
    echo "ok $1"
  else
    echo "FAIL $1: got [$2] want [$3]"
    fails=$((fails + 1))
  fi
}
# has NAME HAYSTACK NEEDLE: `ok` when HAYSTACK contains NEEDLE, else the haystack's first line in the failure.
has() {
  case $2 in
    *"$3"*) check "$1" ok ok ;;
    *) check "$1" "${2%%$'\n'*}" "…$3…" ;;
  esac
}

g() { git -c user.email=t@t -c user.name=t -c commit.gpgsign=false "$@"; }
bash "$HERE/selftest.sh" --self-test --output-only
check output-directory-contracts "$?" 0
# mkdir_lock: of eight takers that race for one directory, exactly one wins, in each of 20 rounds. A mkdir that reads the
# directory another taker made as its own lets two win.
for k in $(seq 20); do
  racers=()
  for j in 1 2 3 4 5 6 7 8; do
    (mkdir_lock "$T/lock.$k" && echo "$k") &
    racers+=("$!")
  done
  wait "${racers[@]}"
done > "$T/lock.won"
check mkdir-lock-one-winner "$(sort "$T/lock.won" | uniq -d | wc -l | tr -d ' ') $(wc -l < "$T/lock.won" | tr -d ' ')" "0 20"
# mkrepo DIR: a repository with one commit and one tracked file.
mkrepo() {
  git init -q --template= "$1" && echo a > "$1/f" && g -C "$1" add f && g -C "$1" commit -q -m one
}
sha8() { git -C "$1" rev-parse --short=8 HEAD; }
corrupt_index() { printf garbage > "$1/.git/index"; }

# ---- tree_commit ----
mkrepo "$T/clean"
tree_commit "$T/clean"
check clean "$?|$TREE_COMMIT|$TREE_GIT_RC" "0|$(sha8 "$T/clean")|0"
check clean-parts "$TREE_SHA|$TREE_DIRTY" "$(sha8 "$T/clean")|0"
tree_commit "$T/clean" 12
check width "$?|$TREE_SHA|$TREE_COMMIT" "0|$(git -C "$T/clean" rev-parse --short=12 HEAD)|$(git -C "$T/clean" rev-parse --short=12 HEAD)"

mkrepo "$T/untracked"
echo b > "$T/untracked/new"
tree_commit "$T/untracked"
check untracked-is-local "$?|$TREE_COMMIT" "0|$(sha8 "$T/untracked")+local"
check untracked-parts "$TREE_SHA|$TREE_DIRTY" "$(sha8 "$T/untracked")|1"

mkrepo "$T/modified"
echo b >> "$T/modified/f"
tree_commit "$T/modified"
check modified-is-local "$?|$TREE_COMMIT" "0|$(sha8 "$T/modified")+local"

# The failure mode: HEAD reads, status fails with 128 and prints nothing. Never the bare commit, never `+local` guessed.
mkrepo "$T/badindex"
corrupt_index "$T/badindex"
check badindex-head-reads "$(sha8 "$T/badindex" > /dev/null 2>&1 && echo yes)" yes
tree_commit "$T/badindex" 2> /dev/null
check badindex-refused "$?|$TREE_COMMIT|$TREE_GIT_RC" "2|?|128"
check badindex-no-part-states-a-clean-tree "$TREE_SHA|$TREE_DIRTY" "?|?"

git init -q --template= "$T/nocommit"
tree_commit "$T/nocommit"
check no-head "$?|$TREE_COMMIT|$TREE_GIT_RC|$TREE_SHA|$TREE_DIRTY" "1|?|0|?|?"

mkdir "$T/plain"
GIT_CEILING_DIRECTORIES=$T tree_commit "$T/plain"
check not-a-repository "$?|$TREE_COMMIT|$TREE_GIT_RC" "1|?|0"

# ---- tools/contrib-check.sh: the real script in the real tree, its index pointed at a corrupt file ----
# Only the injected unit could run if the stop were missing, so a regression ends this check red and short, never in a full run.
printf garbage > "$T/index.bad"
out=$(cd "$ROOT" && GIT_INDEX_FILE=$T/index.bad CONTRIB_INJECT_FAIL=1 bash tools/contrib-check.sh --tier t0 --only '^inject-fail$' 2> "$T/cc.err")
rc=$?
check contrib-check-stops "$rc" 70
check contrib-check-no-header "$out" ""
has contrib-check-names-the-tree "$(cat "$T/cc.err")" "git status failed in $ROOT (git exit 128)"

# A tree with no readable HEAD (an unpacked archive) is the `?` label, not the stop: the header names commit=? and the units run.
out=$(cd "$ROOT" && GIT_DIR=$T/no-git CONTRIB_INJECT_FAIL=1 bash tools/contrib-check.sh --tier t0 --only '^inject-fail$' 2> "$T/cc-archive.err")
rc=$?
check contrib-check-archive-runs "$rc" 1
has contrib-check-archive-header "$out" "contrib-check tier=t0 commit=?"
has contrib-check-archive-unit-ran "$out" "inject-fail rc=101"

# The check units are tools/mac-static.sh's list (the one discovery of check scripts) under the names check-<name>: two of the cheap
# ones run for real, so a loop of its own that names them otherwise, or none, ends this check red.
out=$(cd "$ROOT" && bash tools/contrib-check.sh --tier t0 --only '^check-(rustflags|waits)$' 2> "$T/cc-units.err")
rc=$?
check contrib-check-check-units-run "$rc" 0
has contrib-check-check-unit-rustflags "$out" "check-rustflags rc=0"
has contrib-check-check-unit-waits "$out" "check-waits rc=0"
has contrib-check-check-units-done "$out" "DONE tier=t0 total=2 red=0"

# ---- tools/nightly/run.sh: a clone under a temp NIGHTLY_HOME, no fetch ----
# nightly_run NAME CLONE: the files of the one run it leaves in $T/NAME/state/<date>/. flock is a stub on the nightly's PATH
# ($HOME/bin comes first there), so a host without it (macOS) runs the same code. The clone has no rust-toolchain.toml, so a run
# that passes the tree check ends at "no channel": rc 2 either way, and the commit file and the failed line tell the cases apart.
nightly_run() {
  local name=$1 home=$T/$1/home
  mkdir -p "$home/bin" "$T/$name/state"
  printf '#!/bin/sh\nexit 0\n' > "$home/bin/flock"
  chmod +x "$home/bin/flock"
  mv "$2" "$home/repo"
  NIGHTLY_HOME=$home NIGHTLY_STATE=$T/$name/state NIGHTLY_NO_FETCH=1 bash "$HERE/run.sh" > /dev/null 2>&1
  NIGHTLY_RC=$?
  NIGHTLY_DIR=
  for NIGHTLY_DIR in "$T/$name"/state/[0-9]*; do break; done
  NIGHTLY_CLONE=$home/repo
}
nightly_file() { cat "$NIGHTLY_DIR/$1" 2> /dev/null || echo "(no $1)"; }

mkrepo "$T/fx-n1"
corrupt_index "$T/fx-n1"
nightly_run n1 "$T/fx-n1"
check nightly-badindex-rc "$NIGHTLY_RC|$(nightly_file rc)|$(nightly_file status)" "2|2|RED"
has nightly-badindex-names-the-clone "$(nightly_file failed)" "git status of $NIGHTLY_CLONE failed (git exit 128)"
check nightly-badindex-publishes-no-commit "$(nightly_file commit)" "(no commit)"

mkrepo "$T/fx-n2"
nightly_run n2 "$T/fx-n2"
check nightly-clean-commit "$(nightly_file commit)" "$(sha8 "$NIGHTLY_CLONE")"
has nightly-clean-goes-on "$(nightly_file failed)" "no channel"

mkrepo "$T/fx-n3"
echo b > "$T/fx-n3/new"
nightly_run n3 "$T/fx-n3"
check nightly-dirty-commit "$(nightly_file commit)" "$(sha8 "$NIGHTLY_CLONE")+local"

git init -q --template= "$T/fx-n4"
nightly_run n4 "$T/fx-n4"
check nightly-no-head-rc "$NIGHTLY_RC|$(nightly_file status)" "2|RED"
has nightly-no-head-named "$(nightly_file failed)" "HEAD of $NIGHTLY_CLONE cannot be read"

# ---- archive_tree ----
# The race: a tar that stops at the end-of-archive marker (bsdtar) exits while git still writes the zero padding of its last
# block, and git dies of SIGPIPE. The shims in tools/nightly/archive-shim make it deterministic on any tar: `git archive` to a
# pipe is late with its last 4096 bytes, `tar -x` from a pipe stops after the marker; a file operand and every other command
# are the real ones.
REAL_GIT=$(command -v git) REAL_TAR=$(command -v tar)
SHIM=$HERE/archive-shim
shimmed() { (PATH="$SHIM:$PATH" SHIM_REAL_GIT=$REAL_GIT SHIM_REAL_TAR=$REAL_TAR "$@"); }
mkrepo "$T/arc"
arc_sha=$(git -C "$T/arc" rev-parse HEAD)
echo changed > "$T/arc/f"
mkdir "$T/arc-pipe"
shimmed bash -c 'git -C "$1" archive --format=tar "$2" | tar -x -C "$3"; echo "git ${PIPESTATUS[0]} tar ${PIPESTATUS[1]}"' _ "$T/arc" "$arc_sha" "$T/arc-pipe" > "$T/arc-pipe.out" 2>&1
check archive-shims-lose-the-pipe "$(tail -n 1 "$T/arc-pipe.out")" "git 141 tar 0"
mkdir "$T/arc-out" "$T/arc-tmp"
r=$(shimmed env TMPDIR="$T/arc-tmp" bash -c 'source "$1"; archive_tree "$2" "$3" "$4"; echo "$?|$ARCHIVE_GIT_RC|$ARCHIVE_TAR_RC"' _ "$HERE/units.sh" "$T/arc" "$arc_sha" "$T/arc-out" 2> /dev/null)
check archive-tree-survives-a-late-last-write "$r" "0|0|0"
check archive-tree-is-the-commit "$(cat "$T/arc-out/f" 2> /dev/null)|$(ls "$T/arc-out")" "a|f"
check archive-tree-leaves-no-temp-file "$(ls -A "$T/arc-tmp")" ""
r=$(shimmed env TMPDIR="$T/arc-tmp" bash -c 'source "$1"; archive_tree "$2" "$3" "$4"; echo "$?|$ARCHIVE_GIT_RC|$ARCHIVE_TAR_RC"' _ "$HERE/units.sh" "$T/arc" no-such-rev "$T/arc-out" 2> /dev/null)
check archive-tree-bad-rev-names-git "$r|$(ls -A "$T/arc-tmp")" "1|128|-|"

# ---- run_unit: nothing of the unit outlives it ----
# Each way a unit ends (by itself, at its bound, on a TERM to the runner) with each thing it may leave behind (nothing,
# a child holding its output, a unit that ignores TERM), under each monitor and in both output forms. The runner holds
# a lock on fd 9 as tools/nightly/run.sh does and traps TERM. Once the runner is done, nothing of the unit's process
# group lives and the lock is free. The monitors: tools/ref/cancel-stubs.sh's forwarding stand-in for GNU timeout, none
# (an empty UNIT_TIMEOUT, a host without timeout) and the host's own timeout when it has one. With no monitor a unit has
# no bound. The bound path counts the bound from the unit's readiness, not from its spawn: the unit writes its pid file
# and its ready line, then sends SIGALRM to its monitor, which is what the monitor's own timer raises when the bound runs
# out; the monitor's timer (60 s) is only the hang guard, and the KILL after the bound is 0.5 s. A unit never ends by
# itself in the term and bound paths (its 60 s dwell is also a hang guard): what ends it is the path under test. A slow
# unit (bound path only) is ready a second after its spawn, as on a loaded machine: the bound still comes from readiness.
# PIN(2026-10-19): the bound path's bound was 0.5 s of wall clock against a unit that ended itself after 2 s, so a loaded host
# that ran the bound out late read as a surviving unit. The assertions are unchanged (the runner's code, nothing left, no
# unit-survived); the bound now runs out on the event, once the unit is ready.
# shellcheck source=tools/ref/cancel-stubs.sh
source "$ROOT/tools/ref/cancel-stubs.sh"
mkdir "$T/ru-bin"
cancel_timeout "$T/ru-bin/stub-timeout"
printf '#!/bin/sh\nshift 2\nexec "$RU_MON" --kill-after=0.5 60 "$@"\n' > "$T/ru-bin/short"
mkdir "$T/ru-ps"
printf '#!/bin/sh\nexit 1\n' > "$T/ru-ps/ps"
chmod +x "$T/ru-bin/stub-timeout" "$T/ru-bin/short" "$T/ru-ps/ps"
cat > "$T/ru-cell" << 'CELL'
source "$UNITS"
source "$GB"
exec 9> "$LOCKF"
! command -v flock > /dev/null || flock -n 9 || exit 75
trap 'echo trapped; exit 143' TERM
run_unit "$UNIT" "$LOG" 60 $MODE
echo "returned rc=$UNIT_RC"
CELL
ru_self=$(ps -o pgid= -p $$ | tr -d ' ')
ru_lock=-
! command -v flock > /dev/null || ru_lock=free
ru_monitors="stub:$T/ru-bin/stub-timeout none:"
if command -v timeout > /dev/null; then ru_monitors="$ru_monitors gnu:timeout"; fi
for ru_mon in $ru_monitors; do
  mon=${ru_mon#*:}
  for ru_mode in stream quiet; do
    for ru_path in end bound term; do
      [ -n "$mon" ] || [ "$ru_path" != bound ] || continue
      for ru_left in none child deaf slow; do
        [ "$ru_left" != slow ] || [ "$ru_path" = bound ] || continue
        tag=run-unit-$ru_path-$ru_left-${ru_mon%%:*}-$ru_mode pidf=$T/$tag.pid
        unit='ps -o pgid= -p $$ > "$PIDF";'
        case $ru_left in child) unit="$unit sleep 1000 &" ;; deaf) unit="$unit trap '' TERM;" ;; slow) unit="sleep 1; $unit" ;; esac
        unit="$unit echo unit-ready;"
        [ "$ru_path" != bound ] || unit="$unit kill -ALRM \$PPID;"
        case $ru_path in end) unit="$unit exit 3" ;; *) unit="$unit sleep 60; echo unit-survived" ;; esac
        ru_timeout=$mon
        [ "$ru_path" != bound ] || ru_timeout=$T/ru-bin/short
        env UNITS="$HERE/units.sh" GB="$ROOT/tools/gate-bound.sh" UNIT_TIMEOUT="$ru_timeout" RU_MON="$mon" \
          CANCEL_TIMEOUT_FORWARD=1 LEASE_STOP_GRACE=3 UNIT="$unit" PIDF="$pidf" LOG="$T/$tag.log" LOCKF="$T/$tag.lock" \
          MODE="$([ "$ru_mode" = quiet ] && echo quiet)" bash "$T/ru-cell" > "$T/$tag.out" 2>&1 &
        rp=$! i=0
        if [ "$ru_path" = term ]; then
          # ready: the unit's line in its log, or the runner gone (its output tells why); 1200 passes is the failsafe
          until grep -qs unit-ready "$T/$tag.log" || ! cancel_alive "$rp" || [ "$i" = 1200 ]; do sleep 0.1; i=$((i + 1)); done
          kill -TERM "$rp"
        fi
        i=0
        while cancel_alive "$rp" && [ "$i" -lt 1200 ]; do sleep 0.1; i=$((i + 1)); done
        ! cancel_alive "$rp" || kill -KILL "$rp"
        wait "$rp"
        rc=$?
        g=$(tr -d ' ' < "$pidf" 2> /dev/null) || g=''
        # A unit left in this script's own group has no group to check, and a KILL to it would end this script.
        [ "$g" != "$ru_self" ] || g=own-group
        left=$(ps -eo pid=,pgid=,stat= | awk -v g="${g:-none}" '$2 == g && $3 !~ /^Z/ { printf " %s", $1 }')
        lock=-
        if command -v flock > /dev/null; then flock -n "$T/$tag.lock" true && lock=free || lock=held; fi
        ru_g=${g##*[!0-9]*}
        [ -z "$ru_g" ] || kill -KILL -- "-$ru_g" 2> /dev/null
        case $ru_path:${ru_mon%%:*}:$ru_left in
          end:*) want="0|returned rc=3" ;;
          bound:gnu:deaf) want="0|returned rc=137" ;;
          bound:*) want="0|returned rc=124" ;;
          term:*) want="143|trapped" ;;
        esac
        [ "$ru_mode" = quiet ] || want="${want%%|*}|unit-ready ${want#*|}"
        # A monitor KILLed in the stop leaves the shell's "Killed" notice in the output.
        check "$tag" "$rc|$(grep -v Killed "$T/$tag.out" | tr '\n' ' ')|$(grep -v Killed "$T/$tag.log")|${g:-no-pgid}$left|$lock" \
          "$want |unit-ready|${ru_g:-a-group-of-its-own}|$ru_lock"
      done
    done
  done
  # A unit that leaves a child, and a process list that cannot be read: whether the child is gone is unknown, so the
  # unit's 0 becomes 70, and the log says why.
  r=$(env UNITS="$HERE/units.sh" GB="$ROOT/tools/gate-bound.sh" UNIT_TIMEOUT="$mon" LOG="$T/ru-ps.log" PATH="$T/ru-ps:$PATH" \
    bash -c 'source "$UNITS"; source "$GB"; run_unit "sleep 1000 & echo x" "$LOG" 60 quiet; echo "rc=$UNIT_RC"')
  check "run-unit-ps-fails-${ru_mon%%:*}" "$r|$(grep -c 'cannot read' "$T/ru-ps.log")" "rc=70|1"
  # A unit gets the signals at their defaults whatever the monitor: a writer whose reader has gone dies of SIGPIPE (141),
  # and a write past the file size limit dies of SIGXFSZ (153).
  env UNITS="$HERE/units.sh" GB="$ROOT/tools/gate-bound.sh" UNIT_TIMEOUT="$mon" LOG="$T/ru-sig.log" BIG="$T/ru-sig.big" \
    bash -c 'source "$UNITS"; source "$GB"; run_unit "yes | head -n 1 > /dev/null; s=\${PIPESTATUS[0]}; (ulimit -f 0; echo x > \"\$BIG\") 2> /dev/null; echo sig \$s \$?" "$LOG" 60 quiet'
  check "run-unit-default-signals-${ru_mon%%:*}" "$(grep '^sig ' "$T/ru-sig.log")" "sig 141 153"
  # A unit that leaves a child holding its output, and a group KILL that is refused as one to another user's process
  # is: run_unit returns at once in both forms, a unit that passed reports 70 and one that failed keeps its own code.
  for ru_mode in stream quiet; do
    tag=run-unit-kill-refused-${ru_mon%%:*}-$ru_mode
    env UNITS="$HERE/units.sh" GB="$ROOT/tools/gate-bound.sh" UNIT_TIMEOUT="$mon" LOG="$T/$tag.log" PIDF="$T/$tag.pid" \
      MODE="$([ "$ru_mode" = quiet ] && echo quiet)" bash -c 'source "$UNITS"; source "$GB"
      kill() { if [ "$1" = -KILL ]; then echo "kill: ($3) - Operation not permitted" >&2; return 1; fi; builtin kill "$@"; }
      for c in "exit 0" "exit 3"; do run_unit "sleep 1000 & echo \$! >> \"\$PIDF\"; echo x; $c" "$LOG" 60 $MODE; echo "rc=$UNIT_RC"; done' \
      > "$T/$tag.out" 2>&1 &
    # run_unit returns at once: one that waited on the leftover child would take its 1000 s; 600 passes is the failsafe
    rp=$! i=0
    while cancel_alive "$rp" && [ "$i" -lt 600 ]; do sleep 0.1; i=$((i + 1)); done
    r=returned
    ! cancel_alive "$rp" || { r=hung; kill -KILL "$rp"; }
    wait "$rp" 2> /dev/null
    for p in $(cat "$T/$tag.pid"); do ! cancel_alive "$p" || kill -KILL "$p"; done
    want='rc=70 rc=3 '
    [ "$ru_mode" = quiet ] || want='x [bound] cannot clear process group rc=70 x [bound] cannot clear process group rc=3 '
    check "$tag" "$r|$(sed 's/^\(\[bound\] cannot clear process group\) .*/\1/' "$T/$tag.out" | tr '\n' ' ')" "returned|$want"
  done
done
# Printing the log fails (a stdout open only for reading): a unit that passed reports cat's code, and one that failed keeps
# its own. Each unit prints a line, so cat has a write to fail. A closed stdout is no such case for every cat: uutils' ends
# 0 there.
r=$(env UNITS="$HERE/units.sh" GB="$ROOT/tools/gate-bound.sh" UNIT_TIMEOUT="$T/ru-bin/stub-timeout" LOG="$T/ru-cat.log" \
  bash -c 'source "$UNITS"; source "$GB"; for c in "echo x; exit 0" "echo x; exit 3"; do run_unit "$c" "$LOG" 60 1< /dev/null 2> /dev/null; echo "rc=$UNIT_RC"; done')
check run-unit-print-fails "$(tr '\n' ' ' <<< "$r")" "rc=1 rc=3 "

echo "tree-commit self-test: $([ "$fails" = 0 ] && echo ok || echo FAILED) ($checks checks, $fails failures)"
[ "$fails" = 0 ]
