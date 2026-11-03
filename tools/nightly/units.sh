#!/usr/bin/env bash
# The one unit runner and the one commit label: a library, sourced by tools/nightly/run.sh (the nightly host run) and
# tools/contrib-check.sh (a contributor's off-box check), and, for the label alone, by tools/gate-batch.sh,
# tools/ptx-compare.sh and tools/narrow-scan.sh. Output directory ownership is shared by gate-batch.sh and batch-replay.sh,
# and gate-batch.sh takes its mutexes with mkdir_lock.
# A unit is a `name<TAB>command` line of `python3 tools/recipes.py
# host-nightly`, or any command a caller names; this file runs one and reads its outcome, and a caller decides what to do
# with it.
#
#   run_unit CMD LOG BOUND [quiet]
#     runs `bash -c CMD` with stdin from /dev/null under `timeout --kill-after=10 BOUND+30` (the 30 s leave room for the
#     unit's own tools/gate.sh bound to speak first); the output goes to LOG; unless the fourth word is `quiet`, run_unit
#     then prints LOG to stdout, once the unit has ended and not while it runs.
#     The unit is a child of the caller's shell through tools/gate-bound.sh's run_bounded, which the caller sources, and
#     its monitor leads a process group of its own: a TERM, INT or HUP to the caller stops the unit and that group, then
#     reaches the caller's own trap. However the unit ends (by itself, at its bound, on that stop), when run_unit returns
#     nothing of the group lives (run_bounded --clear KILLs what is left), so nothing the unit started keeps the run
#     waiting or holds the caller's descriptors (the nightly lock on fd 9). A process the unit moves out of the group
#     (setsid) is outside this.
#     Sets:
#       UNIT_RC     the command's exit code (124/137: the bound ran out; 70: the group's KILL was refused, something of
#                   the group outlived it, or the process list could not be read); when that is 0 and printing LOG
#                   fails, cat's code
#       UNIT_SECS   its wall time in whole seconds
#       UNIT_KIND   `cargo` when CMD runs tools/gate.sh, else `script`
#       UNIT_P, UNIT_F, UNIT_I   the sums of the log's `test result:` passed, failed and ignored counts
#     UNIT_TIMEOUT names the bounding command (default `timeout`, which run_bounded runs as tools/gate-bound.sh's
#     bound_timeout monitor); the empty string runs the unit unbounded, in a process group of its own: the caller says so.
#   unit_failure NAME RC LOG BOUND
#     prints what a red unit's `failed` file holds: one `NAME: <test>` line per libtest `... FAILED` line, or else one
#     `NAME: <why>` line (timeout, a filter that matched no test, a build failure, the rc) and the first few red lines (a rustfmt
#     `Diff in` line included).
#   tree_commit DIR [WIDTH]
#     the label a run's evidence is attributed to: the short commit (WIDTH hex digits, default 8) of the tree at DIR, with
#     `+local` appended when its working tree holds uncommitted files. Sets TREE_SHA (the short commit), TREE_DIRTY (1 when
#     the working tree holds uncommitted files, else 0), TREE_COMMIT (the label: TREE_SHA, `+local` when dirty) and
#     TREE_GIT_RC (the exit code of the git call that failed, else 0), and returns 0 when the label is exact. A caller with
#     another spelling of the dirty mark composes it from TREE_SHA and TREE_DIRTY. It never states a commit it could not
#     qualify: on any failure TREE_SHA, TREE_DIRTY and TREE_COMMIT are all `?`.
#       1  HEAD cannot be read (no commit, or not a repository); `git status` is not asked
#       2  HEAD reads and `git status` fails (a corrupt index, an unreadable tree): whether the tree is clean is unknown, and a
#          caller must stop and name the tree and TREE_GIT_RC
#
#   archive_tree REPO REV DEST
#     extracts the tree of commit REV in REPO into the existing directory DEST, what `git archive` carries and never the working
#     tree (tools/ptx-compare.sh and tools/narrow-scan.sh read their base commit this way). The archive goes through a file,
#     not a pipe: a tar that stops reading at the end-of-archive marker (bsdtar, the Mac's) exits while git is still writing
#     the zero padding of its last block, and git dies of SIGPIPE (rc 141) on an archive tar read whole. Sets ARCHIVE_GIT_RC
#     and ARCHIVE_TAR_RC (`-` when tar did not run) and returns 0 when both are 0, else 1 (70 when it has no temporary file).
#
#   mkdir_lock DIR
#     makes DIR and returns 0, or returns 1 when it exists or cannot be made (that one named on stderr): the lock a caller
#     takes by making a directory and drops with rmdir. One mkdir(2), whose EEXIST is a loss. uutils coreutils mkdir (Ubuntu
#     26.04's) reads a directory another process made between its own check and its mkdir(2) as made, so two takers could
#     both win.
#
# The callers stay in step by sharing this code: a change to how a unit's counts or failures are read, or to how a tree is
# labelled or archived, lands here once.

# shellcheck disable=SC2034  # the UNIT_*, TREE_* and ARCHIVE_* variables are this library's result interface, read by the sourcing script
UNIT_TIMEOUT=${UNIT_TIMEOUT-timeout}

mkdir_lock() {
  python3 -c 'import os, sys
try:
    os.mkdir(sys.argv[1])
except FileExistsError:
    sys.exit(1)' "$1"
}

# output_dir_take DIR reserves a caller's output namespace until output_dir_drop.
# A marker left by an interrupted owner is refused rather than guessed stale.
output_dir_take() {
  local dir=$1
  OUTPUT_DIR_LOCK=
  mkdir -p "$dir" && [ -d "$dir" ] && [ -w "$dir" ] || {
    echo "output directory: cannot write $dir" >&2
    return 73
  }
  mkdir_lock "$dir/.run-owner" || {
    echo "output directory: $dir already has an owner (.run-owner)" >&2
    return 73
  }
  OUTPUT_DIR_LOCK=$dir/.run-owner
}

output_dir_drop() {
  [ -n "${OUTPUT_DIR_LOCK:-}" ] || return 0
  rmdir "$OUTPUT_DIR_LOCK" || return 73
  OUTPUT_DIR_LOCK=
}

run_unit() {
  local cmd=$1 log=$2 bound=$3 mode=${4:-} t0=$SECONDS rc
  local -a bounded=()
  if [ -n "$UNIT_TIMEOUT" ]; then
    bounded=("$UNIT_TIMEOUT" --kill-after=10 "$((bound + 30))")
  else
    # No monitor makes the unit a process group of its own. Job control would print job notices into the unit's output
    # (bash 5.2 does). Python starts with SIGPIPE, SIGXFSZ and SIGXFZ ignored and an exec keeps them so: the unit gets them
    # back at their defaults, as subprocess's restore_signals gives a child.
    bounded=(python3 -c 'import os, signal, sys
os.setpgid(0, 0)
for s in "SIGPIPE", "SIGXFSZ", "SIGXFZ":
    if hasattr(signal, s):
        signal.signal(getattr(signal, s), signal.SIG_DFL)
os.execvp(sys.argv[1], sys.argv[1:])')
  fi
  UNIT_RC=0 BIN_SIGNAL=
  BLOOMERY_GATE_BOUND=$bound run_bounded --clear "${bounded[@]}" bash -c "$cmd" < /dev/null > "$log" 2>&1 || UNIT_RC=$?
  if [ "$mode" != quiet ]; then
    cat "$log" || {
      rc=$?
      [ "$UNIT_RC" != 0 ] || UNIT_RC=$rc
    }
  fi
  # The signal the unit was stopped on goes on to the caller's trap.
  [ -z "$BIN_SIGNAL" ] || kill -s "$BIN_SIGNAL" $$
  UNIT_SECS=$((SECONDS - t0))
  UNIT_KIND=script
  case $cmd in *gate.sh*) UNIT_KIND=cargo ;; esac
  read -r UNIT_P UNIT_F UNIT_I < <(awk '/^test result: / { for (k = 2; k <= NF; k++) { if ($k == "passed;") p += $(k - 1); if ($k == "failed;") f += $(k - 1); if ($k == "ignored;") g += $(k - 1) } } END { print p + 0, f + 0, g + 0 }' "$log")
}

unit_failure() {
  local name=$1 rc=$2 log=$3 bound=$4 named why
  named=$(sed -n 's/^test \(.*\) \.\.\. FAILED$/\1/p' "$log")
  if [ -n "$named" ]; then
    printf '%s\n' "$named" | sed "s/^/$name: /"
  else
    case $rc in
      124 | 137) why="timeout after ${bound}s (rc $rc)" ;;
      78) why="a test filter matched no test (rc 78)" ;;
      *) if grep -q 'could not compile\|^error\(\[E[0-9]*\]\)\?:' "$log"; then why="build failure (rc $rc)"; else why="rc $rc"; fi ;;
    esac
    echo "$name: $why"
    # a script's red lines sit anywhere in its output, not in its last 80: name the first few
    grep -E 'FAIL|[Ff]ailed|[Ee]rror|^Diff in' "$log" | grep -v '^ok ' | head -n 8 | cut -c1-220 | sed 's/^/    | /'
  fi
}

tree_commit() {
  local dir=$1 width=${2:-8} dirty
  TREE_GIT_RC=0
  if ! TREE_SHA=$(git -C "$dir" rev-parse --short="$width" HEAD 2> /dev/null); then
    TREE_SHA="?" TREE_DIRTY="?" TREE_COMMIT="?"
    return 1
  fi
  # A status that fails prints nothing, which reads as a clean tree: the assignment's own exit code decides.
  dirty=$(git -C "$dir" status --porcelain --untracked-files=normal) || {
    TREE_GIT_RC=$?
    TREE_SHA="?" TREE_DIRTY="?" TREE_COMMIT="?"
    return 2
  }
  TREE_DIRTY=0
  TREE_COMMIT=$TREE_SHA
  if [ -n "$dirty" ]; then
    TREE_DIRTY=1
    TREE_COMMIT=$TREE_COMMIT+local
  fi
  return 0
}

archive_tree() {
  local repo=$1 rev=$2 dest=$3 f
  ARCHIVE_GIT_RC=0 ARCHIVE_TAR_RC=-
  f=$(mktemp "${TMPDIR:-/tmp}/archive-tree.XXXXXX") || return 70
  git -C "$repo" archive --format=tar -o "$f" "$rev" || {
    ARCHIVE_GIT_RC=$?
    rm -f "$f"
    return 1
  }
  tar -x -f "$f" -C "$dest"
  ARCHIVE_TAR_RC=$?
  rm -f "$f"
  [ "$ARCHIVE_TAR_RC" = 0 ]
}
