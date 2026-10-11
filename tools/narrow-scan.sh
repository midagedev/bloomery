#!/usr/bin/env bash
# `just narrow BASE [HEAD]`: the scans `just affected --narrow` needs, taken for you.
#   tools/narrow-scan.sh BASE [HEAD] [--out DIR] [--dry-run]
#   tools/narrow-scan.sh BASE --worktree [--out DIR] [--dry-run]
#   tools/narrow-scan.sh --self-test
#
# What it does, in order:
#   1. Resolves BASE and HEAD (default: this tree's HEAD commit) to commits. The scans read the commits through
#      `git archive`, never the working tree: uncommitted edits are not scanned (a note says so). `--worktree` is
#      the mode for a tree that may not commit (a round): the head side is this tree's current content, uncommitted
#      and untracked files included, synced as box.sh syncs it (its own tools/box.sh from this tree), and the base
#      side stays BASE's commit. BASE must be an ancestor of this tree's HEAD, so that `tools/affected-gates.sh BASE`
#      (the working tree against the merge base) reads the change the two scans compare; a fingerprint of the tree
#      (tracked diff against BASE plus each untracked file's md5) is taken before the scans and again before the
#      affected run, and a tree that moved in between ends the run (rc 1) instead of narrowing on mixed content.
#   2. Derives the scan set from HEAD's tree (`tools/recipes.py scan-set`; with `--worktree`, this tree as it is): the
#      fewest bins of bloomery-gpu-gates whose scans cover every kernel carrier (each device lib, and each bin with
#      kernels of its own), each with the cargo features it builds under.
#   3. For each side in turn, BASE then HEAD: extracts the commit into DIR/src-<side>, syncs it to that side's remote
#      dir with its own tools/box.sh, builds the set there (`cargo oxide build`, as the ptx-scan recipe does) and runs
#      `ptx-scan.sh --no-jit` on each bin. This tree's tools/ptx-scan.sh is the scanner on both sides (the commit's
#      own may predate --no-jit), shipped over stdin; the extractor (oxart_ptx) is the commit's.
#   4. Splits the box's output into DIR/<side>/<bin>.log and runs
#      `tools/affected-gates.sh BASE..HEAD --narrow --scan <base log> <head log> ...` (`BASE --narrow …` under
#      `--worktree`) with every pair, printing its output: the narrowed list with its why-lines, and the `recipes:`
#      line to paste. DIR/narrowed.txt keeps it.
#
# The remote dirs are per track and persist across runs: ~/repo/bloomery-<track>-narrow-base and -narrow-head, <track>
# this tree's directory name without "bloomery-" (the main tree's pair: ~/repo/bloomery-narrow-base and -narrow-head), so
# two tracks never wait on each other's dirs. A second run builds incrementally: the BASE side costs a sync and the scans
# (no scan is cached; they run again), the HEAD side the crates the change touched. A pair belongs to this recipe alone,
# and one run uses it at a time: a lock under ~/.cache/bloomery named by the pair (a second run on it waits up to
# NARROW_WAIT seconds, default 1800, naming the holder; the main tree's pair keeps the lock name older copies of this
# script took for every tree). NARROW_BASE_REMOTE and NARROW_HEAD_REMOTE name other dirs. The names follow the
# auxiliary-dir rule (bloomery-<track>-<suffix>): once the track's worktree is gone, `just box-tracks --remove
# <track>-narrow-base <track>-narrow-head` clears them (the main tree's: `narrow-base narrow-head`).
#
# Stopping (INT/TERM): the side in flight ends from both ends, by pids it recorded and never by a name match. Its box
# command writes the pid of the `timeout` it started to .narrow-scan-<side>-<pid>.pid in the remote dir (removed when the
# runner ends, and by the next sync if a run died); the stop reads that file in one READONLY box call (no sync, no guard
# wait) and signals that pid, whose TERM `timeout` passes to the runner's process group, then signals the local box.sh's
# process group (job control makes it a group of its own: box.sh, its ssh and rsync) by the pid written under DIR at its
# start. A stop that finds no file prints so (the runner had not started or had ended).
#
# It runs no gate and takes no timing lease, no hold, no card and no gate lock: `--no-jit` loads nothing on a card, and
# the box commands are builds and ptxas. box.sh's own guard still waits while a sitting's lease or hold is up (rc 75 is
# contention: run again, BLOOMERY_BOX_WAIT=3600 waits longer). BLOOMERY_CARD, BLOOMERY_BOX_ENV and BLOOMERY_TIER are
# unset for the box calls, so nothing card-related reaches the box. The scans are sm_86's: a BLOOMERY_ARCH naming another
# arch is refused (64) before anything runs.
#
# DIR (default ~/.cache/bloomery/narrow/<tree>/<base8>-<head8>, <base8>-worktree under `--worktree`) keeps: scan-set.tsv, runner.sh (the script each side
# runs), <side>.build.log (the box's stderr: the builds), <side>.scan.out, <side>/<bin>.log (the scan logs a landing
# cites), narrowed.txt and MANIFEST; <side>.box.pid while the side runs.
#
# NARROW_BOUND (default 7200 s, a hang guard: no box-job length needs approval) bounds the whole run: each side's box command runs under `timeout` with what
# is left. A cut build keeps its compiled crates, so running again continues it. Exit codes: 0 done; 64 usage; 75
# contention (the box guard, or the lock); 70 the box's toolchain; 1 a build or scan failed; 124 the bound; 130 INT and
# 143 TERM; the `affected` run's own code otherwise.
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PKG=bloomery-gpu-gates # the package the ptx-scan recipe builds; tools/recipes.py SCAN_PACKAGE
ARCH=sm_86 # the arch of the scans: tools/ptx-scan.sh's blk/SM column is its arithmetic and it refuses another (BLOOMERY_ARCH, tools/arch.sh)
USAGE="usage: narrow-scan.sh BASE [HEAD | --worktree] [--out DIR] [--dry-run] | --self-test"

die() { # die <rc> <message>
  echo "narrow-scan: $2" >&2
  exit "$1"
}

md5_of() { # md5_of <file>: the Mac has md5, the box md5sum
  if command -v md5sum > /dev/null; then md5sum "$1" | cut -d' ' -f1; else md5 -q "$1"; fi
}

# str_key <text>: a short key of the text, for a lock's name.
str_key() {
  printf '%s' "$1" | if command -v md5sum > /dev/null; then md5sum | cut -c1-12; else md5 -q | cut -c1-12; fi
}

# worktree_key <base sha>: a fingerprint of this tree's content against BASE: the tracked diff (binary) and each
# untracked, not ignored file's md5 by path (git lists them in path order). Equal keys are the same content; the ignored files box.sh also syncs
# are not in it (affected-gates reads git's view of the change too).
worktree_key() {
  {
    git -C "$ROOT" diff --binary --no-renames "$1" || echo "diff failed"
    git -C "$ROOT" ls-files --others --exclude-standard -z | while IFS= read -r -d '' f; do
      printf '%s %s\n' "$(md5_of "$ROOT/$f" 2> /dev/null || echo unreadable)" "$f"
    done
  } | if command -v md5sum > /dev/null; then md5sum | cut -d' ' -f1; else md5 -q; fi
}

# make_runner <scan-set.tsv> <ptx-scan.sh> <runner out>: the script a side runs on the box. It writes ptx-scan.sh to a
# temp file, builds each feature group (the builds' output goes to stderr: stdout carries the scan blocks only), builds
# the extractor, and prints each bin's scan between `=== narrow-scan begin <bin>` and `end <bin> rc=<n>` markers.
make_runner() {
  local tsv=$1 scanner=$2 out=$3 bin feats covers f b groups='' line
  local bins=() featl=()
  while IFS=$'\t' read -r bin feats covers; do
    [ -n "$bin" ] || continue
    case $bin in *[!A-Za-z0-9_-]* | -*) echo "narrow-scan: scan-set.tsv: bin name '$bin' is not a cargo bin name" >&2; return 2 ;; esac
    case $feats in *[!a-z0-9_,-]*) echo "narrow-scan: scan-set.tsv: features '$feats' of $bin are not a feature list" >&2; return 2 ;; esac
    bins[${#bins[@]}]=$bin
    featl[${#featl[@]}]=$feats
  done < "$tsv"
  [ "${#bins[@]}" -gt 0 ] || { echo "narrow-scan: $tsv names no bin: nothing to scan" >&2; return 2; }
  if grep -qx 'NARROW_PTX_SCAN_EOF' "$scanner"; then
    echo "narrow-scan: $scanner holds the here-document's end marker" >&2
    return 2
  fi
  {
    echo 'set -uo pipefail'
    echo 'P=$(mktemp) || exit 70'
    echo "cat > \"\$P\" <<'NARROW_PTX_SCAN_EOF'"
    cat "$scanner"
    [ "$(tail -c 1 "$scanner" | wc -l)" -eq 1 ] || printf '\n'
    echo 'NARROW_PTX_SCAN_EOF'
    # one `cargo oxide build` per distinct feature list, its bins in the set's order
    for f in "${featl[@]}"; do
      case " $groups " in *" $f "*) continue ;; esac
      groups="$groups $f"
      line="cargo oxide build --arch $ARCH -- -p $PKG --features $f --release"
      for b in "${!bins[@]}"; do
        [ "${featl[$b]}" = "$f" ] && line="$line --bin ${bins[$b]}"
      done
      echo 't=$SECONDS'
      echo "$line >&2 </dev/null || { echo 'narrow-scan: the build with features $f failed' >&2; exit 1; }"
      echo "echo \"narrow-scan: wall: build, features $f: \$((SECONDS - t)) s\" >&2"
    done
    echo 't=$SECONDS'
    echo "cargo build --release -p $PKG --bin oxart_ptx >&2 </dev/null || { echo 'narrow-scan: the extractor build failed' >&2; exit 1; }"
    echo "echo \"narrow-scan: wall: build, extractor: \$((SECONDS - t)) s\" >&2"
    for b in "${bins[@]}"; do
      echo 't=$SECONDS'
      echo "echo '=== narrow-scan begin $b'; bash \"\$P\" --no-jit $b 2>&1; echo \"=== narrow-scan end $b rc=\$?\""
      echo "echo \"narrow-scan: wall: scan $b: \$((SECONDS - t)) s\" >&2"
    done
    echo 'rm -f "$P"'
  } > "$out"
}

# split_logs <scan.out> <dir> <bin>...: cuts the box's stdout into <dir>/<bin>.log. Non-zero when a block is missing,
# unterminated, nested, or its scan ended non-zero; the reasons are printed.
split_logs() {
  local file=$1 dir=$2
  shift 2
  mkdir -p "$dir"
  awk -v dir="$dir" -v want="$*" '
    BEGIN { n = split(want, W, " "); for (i = 1; i <= n; i++) need[W[i]] = 1 }
    /^=== narrow-scan begin / {
      if (cur != "") { print "block " cur " has no end marker"; bad = 1; close(file) }
      cur = $4; file = dir "/" cur ".log"; printf "" > file; seen[cur] = "open"; next
    }
    /^=== narrow-scan end / {
      if ($4 != cur) { print "end marker for " $4 " outside its block (inside " (cur == "" ? "none" : cur) ")"; bad = 1; next }
      rc = $5; sub(/^rc=/, "", rc); seen[cur] = (rc == "0" ? "ok" : "rc=" rc); close(file); cur = ""; next
    }
    cur != "" { print >> file }
    END {
      if (cur != "") { print "block " cur " has no end marker: the box output was cut"; bad = 1 }
      for (b in need) {
        if (!(b in seen)) { print "no scan block for " b; bad = 1 }
        else if (seen[b] != "ok") { print "the scan of " b " ended " seen[b] " (its log: " dir "/" b ".log)"; bad = 1 }
      }
      exit bad
    }' "$file"
}

# lock_take <lock dir> <wait seconds> <who>: one run at a time uses a pair of persistent remote dirs. A holder whose pid
# is gone is taken over. Prints what it waits for; rc 75 at the bound.
lock_take() {
  local lockd=$1 wait=$2 who=$3 waited=0 pid
  mkdir -p "$(dirname "$lockd")"
  while ! mkdir "$lockd" 2> /dev/null; do
    pid=$(cat "$lockd/pid" 2> /dev/null || true)
    if [ -n "$pid" ] && ! kill -0 "$pid" 2> /dev/null; then
      echo "narrow-scan: the lock's holder (pid $pid, $(cat "$lockd/who" 2> /dev/null)) is gone: taking it over" >&2
      rm -rf "$lockd"
      continue
    fi
    if [ "$waited" -ge "$wait" ]; then
      echo "narrow-scan: $lockd is held by pid ${pid:-?} ($(cat "$lockd/who" 2> /dev/null)); this pair of remote dirs takes one run at a time (contention, not failure: run again)" >&2
      return 75
    fi
    echo "narrow-scan: waiting for pid ${pid:-?} ($(cat "$lockd/who" 2> /dev/null)): this pair of remote dirs takes one run at a time (${waited}s of ${wait}s)" >&2
    sleep 30
    waited=$((waited + 30))
  done
  echo "$$" > "$lockd/pid"
  echo "$who" > "$lockd/who"
}

lock_release() { # lock_release <lock dir>: only the holder's own
  [ "$(cat "$1/pid" 2> /dev/null)" = "$$" ] && rm -rf "$1"
  return 0
}

# side_command <bound s> <pid file>: the one box.sh argument each side runs. The runner arrives on stdin, so the command writes it to a
# temp file and runs it under `timeout`, started in the background so that its pid (the `timeout`, which passes a TERM to the runner's
# process group) is written to <pid file> in the remote dir; the file goes when the runner ends. The command's rc comes back whatever
# came before it (the guard's 75, the backend's 70). The leading `:` names `cargo oxide`, which makes box.sh check the pinned backend
# before the command, as it does for a recipe.
side_command() {
  printf '%s' ': cargo oxide && T=$(mktemp) && cat > "$T" && { timeout --kill-after=10 '"$1"' bash "$T" </dev/null & echo $! > '"$2"'; wait $!; }; rc=$?; rm -f "$T" '"$2"'; exit $rc'
}

# stop_command <pid file>: the READONLY box command of a stop: signals the pid the side command recorded, if the file is there.
stop_command() {
  printf '%s' 'p=$(cat '"$1"' 2>/dev/null); if [ -n "$p" ]; then kill -TERM "$p" && echo "narrow-scan: signalled the recorded remote pid $p"; else echo "narrow-scan: no remote pid recorded in '"$1"' (the runner had not started or had ended)"; fi'
}

# remote_stop <remote dir> <pid file>: the stop's one READONLY box call (no sync, no guard wait: a sitting does not delay it),
# bounded at 30 s.
remote_stop() {
  local bp k=0
  env -u BLOOMERY_CARD -u BLOOMERY_BOX_ENV -u BLOOMERY_TIER BLOOMERY_REMOTE="$1" BLOOMERY_BOX_READONLY=1 BLOOMERY_BOX_WAIT=0 \
    "$ROOT/tools/box.sh" "$(stop_command "$2")" < /dev/null >&2 2>&1 &
  bp=$!
  while kill -0 "$bp" 2> /dev/null && [ "$k" -lt 150 ]; do
    sleep 0.2
    k=$((k + 1))
  done
  if kill -0 "$bp" 2> /dev/null; then
    kill -TERM "$bp" 2> /dev/null || true
    echo "narrow-scan: the stop's box call did not return in 30 s: the runner may still be running on the box ($1/$2; 'just box-gc --dry-run' lists orphans)" >&2
  fi
}

# stop_run <rc>: INT or TERM. The side in flight (ACTIVE_*, set by run_side) is ended from both ends, then the EXIT trap releases the lock.
ACTIVE_NAME='' ACTIVE_DIR='' ACTIVE_RPIDF='' ACTIVE_PIDF=''
stop_run() {
  local bp
  trap '' INT TERM # a second signal does not cut the stop short
  if [ -n "$ACTIVE_NAME" ]; then
    echo "narrow-scan: stopped: ending the $ACTIVE_NAME side ($ACTIVE_DIR/$ACTIVE_RPIDF on the box, $ACTIVE_PIDF here)" >&2
    remote_stop "$ACTIVE_DIR" "$ACTIVE_RPIDF"
    if [ -f "$ACTIVE_PIDF" ]; then
      bp=$(cat "$ACTIVE_PIDF")
      kill -TERM -- "-$bp" 2> /dev/null || kill -TERM "$bp" 2> /dev/null || true
    fi
  fi
  exit "$1"
}

# run_side <name> <rev> <remote dir> <runner> <out dir> <bound s> <bin>...: one side's sync, build and scans on the box.
# `rev` is a commit (extracted from `git archive`) or `worktree`: this tree as it is, synced by its own tools/box.sh. The box.sh runs in
# the background and is waited for, so a stop signal is handled at once (a trap waits for a foreground command); its pid goes to
# <out>/<name>.box.pid, and its stderr through a fifo to <out>/<name>.build.log and ours.
run_side() {
  local name=$1 rev=$2 dir=$3 runner=$4 out=$5 bound=$6 src rc t0 cmd fifo tp bp
  shift 6
  if [ "$rev" = worktree ]; then
    src=$ROOT
  else
    src=$out/src-$name
    rm -rf "$src"
    mkdir -p "$src"
    git -C "$ROOT" archive --format=tar "$rev" | tar -x -C "$src"
    rc=("${PIPESTATUS[@]}")
    [ "${rc[0]}" = 0 ] && [ "${rc[1]}" = 0 ] || die 1 "git archive $rev failed (git ${rc[0]}, tar ${rc[1]})"
  fi
  ACTIVE_RPIDF=.narrow-scan-$name-$$.pid
  cmd=$(side_command "$bound" "$ACTIVE_RPIDF")
  echo "narrow-scan: $name: $rev -> $dir (bound ${bound}s)" >&2
  t0=$SECONDS
  fifo=$out/$name.build.fifo
  rm -f "$fifo"
  mkfifo "$fifo" || die 70 "cannot make $fifo"
  tee "$out/$name.build.log" < "$fifo" >&2 &
  tp=$!
  # git must not find this tree's repository above the archive (box.sh stamps its HEAD into the build as the commit). Job control
  # makes the subshell the leader of a process group of its own, so a stop reaches box.sh's ssh and rsync with the one pid.
  set -m
  (
    cd "$src" && exec env -u BLOOMERY_CARD -u BLOOMERY_BOX_ENV -u BLOOMERY_TIER GIT_CEILING_DIRECTORIES="$out" BLOOMERY_REMOTE="$dir" \
      ./tools/box.sh "$cmd" < "$runner" 2>&1 > "$out/$name.scan.out"
  ) > "$fifo" &
  bp=$!
  set +m
  ACTIVE_PIDF=$out/$name.box.pid
  echo "$bp" > "$ACTIVE_PIDF"
  ACTIVE_NAME=$name ACTIVE_DIR=$dir
  wait "$bp"
  rc=$?
  ACTIVE_NAME=''
  wait "$tp" 2> /dev/null
  rm -f "$fifo" "$ACTIVE_PIDF"
  SIDE_WALL=$((SECONDS - t0))
  [ "$rev" = worktree ] || rm -rf "$src"
  case $rc in
    0) ;;
    75) echo "narrow-scan: $name: rc 75: the box is held by a sitting or a lease (contention, not failure; nothing was built): run again, BLOOMERY_BOX_WAIT=3600 waits longer" >&2 ;;
    124 | 137) echo "narrow-scan: $name: the bound of ${bound}s ended the box command (rc $rc); the dir keeps its compiled crates: run again to continue" >&2 ;;
    70) echo "narrow-scan: $name: rc 70: the box's cuda-oxide backend is missing for this commit's pin (tools/box.sh header)" >&2 ;;
    *) echo "narrow-scan: $name: the box command ended rc $rc (the commit's build or a scan: $out/$name.build.log)" >&2 ;;
  esac
  return "$rc"
}

self_test() {
  local t n=0 bad=0 want rc got
  unset BLOOMERY_ARCH
  t=$(mktemp -d "${TMPDIR:-/tmp}/narrow-scan-test.XXXXXX") || die 70 "self-test: no temporary directory"
  # ok <name> <condition rc>
  ok() {
    n=$((n + 1))
    if [ "$2" = 0 ]; then echo "ok $1"; else bad=$((bad + 1)); echo "FAIL $1"; fi
  }
  printf '%s\t%s\t%s\n' bloomery-serve-ds41 deepseek41,vision 'lib a; lib b' gate_kquant gpu 'bin gate_kquant' gate_swap gpu 'bin gate_swap' > "$t/set.tsv"
  printf '#!/usr/bin/env bash\necho "scan $*"\n' > "$t/ptx-scan.sh"
  # the runner: one build per distinct feature list, the bins grouped in the set's order, one extractor build, one scan
  # block per bin, the scanner embedded verbatim
  make_runner "$t/set.tsv" "$t/ptx-scan.sh" "$t/runner.sh"; ok "runner written" $?
  bash -n "$t/runner.sh"; ok "runner parses" $?
  [ "$(grep -c '^cargo oxide build' "$t/runner.sh")" = 2 ]; ok "runner: one cargo oxide build per feature list" $?
  grep -qF "cargo oxide build --arch $ARCH -- -p $PKG --features deepseek41,vision --release --bin bloomery-serve-ds41 >&2" "$t/runner.sh"; ok "runner: the V4.1 group" $?
  grep -qF "cargo oxide build --arch $ARCH -- -p $PKG --features gpu --release --bin gate_kquant --bin gate_swap >&2" "$t/runner.sh"; ok "runner: the gpu group takes both bins" $?
  [ "$(grep -c "^cargo build --release -p $PKG --bin oxart_ptx" "$t/runner.sh")" = 1 ]; ok "runner: one extractor build" $?
  [ "$(grep -c -- '--no-jit' "$t/runner.sh")" = 3 ]; ok "runner: one --no-jit scan per bin" $?
  sed -n "/^cat > \"\$P\" <<'NARROW_PTX_SCAN_EOF'$/,/^NARROW_PTX_SCAN_EOF$/p" "$t/runner.sh" | sed '1d;$d' | cmp -s - "$t/ptx-scan.sh"; ok "runner: ptx-scan.sh embedded verbatim" $?
  [ "$(grep -c '^echo "narrow-scan: wall: ' "$t/runner.sh")" = 6 ]; ok "runner: a wall line per build group, the extractor and each scan" $?
  case $(side_command 900 .p-1) in *"cargo oxide"*) ;; *) false ;; esac; ok "side command: names cargo oxide, so box.sh checks the backend" $?
  case $(side_command 900 .p-1) in *"timeout --kill-after=10 900 bash"*) ;; *) false ;; esac; ok "side command: the runner runs under the bound" $?
  case $(side_command 900 .p-1) in *'echo $! > .p-1'*) ;; *) false ;; esac; ok "side command: the runner's pid is recorded in the named file" $?
  case $(side_command 900 .p-1) in *'rm -f "$T" .p-1'*) ;; *) false ;; esac; ok "side command: the pid file goes when the runner ends" $?
  sc=$(stop_command .p-1 2> /dev/null)
  ro=$(sed -n "s/^    ro_build='\(.*\)'$/\1/p" "$ROOT/tools/box.sh")
  [ -n "$sc" ] && [ -n "$ro" ] && ! [[ $sc =~ $ro ]] && [[ $sc != *target/* ]]; ok "drift: box.sh's READONLY mode accepts the stop command (no build word, no target/)" $?
  got=$(cd "$t" && bash -c "$(stop_command .no-such.pid 2> /dev/null)" 2>&1)
  grep -qF "no remote pid recorded in .no-such.pid" <<< "$got"; ok "stop command: with no pid file it says so and signals nothing" $?
  grep -q "the build with features gpu failed" "$t/runner.sh" && grep -q "the extractor build failed" "$t/runner.sh"; ok "runner: a failed build ends it by name" $?
  # split_logs
  {
    echo 'noise before'; echo '=== narrow-scan begin a'; echo 'row 1'; echo 'row 2'; echo '=== narrow-scan end a rc=0'
    echo '=== narrow-scan begin b'; echo 'row 3'; echo '=== narrow-scan end b rc=0'
  } > "$t/good.out"
  got=$(split_logs "$t/good.out" "$t/good" a b 2>&1); rc=$?
  [ "$rc" = 0 ] && [ "$(cat "$t/good/a.log")" = "$(printf 'row 1\nrow 2')" ] && [ "$(cat "$t/good/b.log")" = row\ 3 ]; ok "split: two blocks, noise outside them dropped" $?
  { echo '=== narrow-scan begin a'; echo 'row'; echo '=== narrow-scan end a rc=1'; } > "$t/rc1.out"
  got=$(split_logs "$t/rc1.out" "$t/rc1" a 2>&1); rc=$?
  [ "$rc" != 0 ] && grep -qF "the scan of a ended rc=1" <<< "$got"; ok "split: a scan that ended rc=1 is refused by name" $?
  { echo '=== narrow-scan begin a'; echo 'row'; } > "$t/cut.out"
  got=$(split_logs "$t/cut.out" "$t/cut" a 2>&1); rc=$?
  [ "$rc" != 0 ] && grep -qF "has no end marker: the box output was cut" <<< "$got"; ok "split: a cut output is refused by name" $?
  got=$(split_logs "$t/good.out" "$t/miss" a b c 2>&1); rc=$?
  [ "$rc" != 0 ] && grep -qF "no scan block for c" <<< "$got"; ok "split: a bin with no block is refused by name" $?
  { echo '=== narrow-scan begin a'; echo '=== narrow-scan begin b'; echo '=== narrow-scan end b rc=0'; } > "$t/nest.out"
  got=$(split_logs "$t/nest.out" "$t/nest" a b 2>&1); rc=$?
  [ "$rc" != 0 ] && grep -qF "block a has no end marker" <<< "$got"; ok "split: a begin inside a block is refused by name" $?
  # the lock: a free lock is taken, a held one is refused at its bound naming the holder, a dead holder is taken over
  lock_take "$t/lock.d" 0 "test run" 2> /dev/null; ok "lock: a free lock is taken" $?
  got=$( (lock_take "$t/lock.d" 0 "second" 2>&1); echo "rc=$?")
  grep -q "rc=75" <<< "$got" && grep -qF "held by pid $$ (test run)" <<< "$got"; ok "lock: a held lock is refused rc 75 naming its holder" $?
  lock_release "$t/lock.d"
  [ ! -d "$t/lock.d" ]; ok "lock: release removes the holder's own" $?
  mkdir "$t/lock.d" && echo 99999999 > "$t/lock.d/pid" && echo gone > "$t/lock.d/who"
  got=$(lock_take "$t/lock.d" 0 "after the dead one" 2>&1); rc=$?
  [ "$rc" = 0 ] && grep -qF "is gone: taking it over" <<< "$got"; ok "lock: a dead holder is taken over" $?
  rm -rf "$t/lock.d"
  mkdir "$t/lock.d" && echo 1 > "$t/lock.d/pid"
  lock_release "$t/lock.d"
  [ -d "$t/lock.d" ]; ok "lock: release leaves another holder's lock" $?
  # the runner's build lines are the ptx-scan recipe's: a recipe edit that this script does not follow is red
  recipe=$(awk '/^ptx-scan /{f=1; next} f && /^[^ ]/{exit} f' "$ROOT/justfile")
  grep -qF "cargo oxide build --arch {{arch}} -- -p $PKG --features {{FEATURES}} --release --bin {{BIN}}" <<< "$recipe"; ok "drift: the ptx-scan recipe still builds with this runner's cargo oxide line" $?
  grep -qF "cargo build --release -p $PKG --bin oxart_ptx && bash tools/ptx-scan.sh" <<< "$recipe"; ok "drift: the ptx-scan recipe still builds the extractor as this runner does" $?
  grep -qF "NARROW_PTX_SCAN_EOF" "$ROOT/tools/ptx-scan.sh"; [ $? = 1 ]; ok "drift: ptx-scan.sh does not hold the runner's here-document marker" $?
  # another arch is refused by name before any git or box call, and the scan's own arch is not
  got=$(env BLOOMERY_ARCH=sm_121 bash "$0" HEAD --dry-run 2>&1); rc=$?
  [ "$rc" = 64 ] && grep -qF "narrow-scan: BLOOMERY_ARCH is 'sm_121': the scan pairs are sm_86's" <<< "$got"; ok "arch: another BLOOMERY_ARCH is refused by name (64)" $?
  # --worktree: this script in a throwaway repository beside stub tools (no box, no card), so the whole run goes through
  # the real argument handling, the side loop and the affected call. The box stub logs which content each side synced.
  w=$t/bloomery-wt
  mkdir -p "$w/tools" "$t/home"
  cp "$0" "$w/tools/narrow-scan.sh"
  printf '%s\n' 'import os, sys' 'open(os.environ["STUB_LOG"], "a").write("recipes " + " ".join(sys.argv[1:]) + "\n")' \
    'print("gate_x\tgpu\tbin gate_x")' > "$w/tools/recipes.py"
  printf '#!/usr/bin/env bash\necho scan\n' > "$w/tools/ptx-scan.sh"
  printf '#!/usr/bin/env bash\necho "affected $*" >> "$STUB_LOG"\necho "affected stub"\n' > "$w/tools/affected-gates.sh"
  cat > "$w/tools/box.sh" <<'NARROW_STUB_EOF'
#!/usr/bin/env bash
here=$(cd "$(dirname "$0")/.." && pwd)
# The stop's read-only call: its command runs in the fake remote dir, as box.sh's READONLY mode runs it on the box.
if [ "${BLOOMERY_BOX_READONLY:-0}" = 1 ]; then
  echo "readonly $BLOOMERY_REMOTE" >> "$STUB_LOG"
  cd "$STUB_REMOTE/$(basename "$BLOOMERY_REMOTE")" || exit 66
  exec bash -c "$*"
fi
side=head
case $BLOOMERY_REMOTE in rb | *narrow-base) side=base ;; esac
# STUB_HANG: the box.sh itself is held before it starts anything on the box (the sync, the guard's wait).
if [ -n "${STUB_HANG:-}" ]; then
  mkdir -p "$STUB_REMOTE/$(basename "$BLOOMERY_REMOTE")"
  echo $$ > "$STUB_REMOTE/hang.pid"
  sleep 60 &
  echo $! > "$STUB_REMOTE/hang-sleep.pid"
  wait $!
  exit 0
fi
# STUB_BLOCK: the box command itself runs in the fake remote dir, with a runner that never ends.
if [ -n "${STUB_BLOCK:-}" ]; then
  mkdir -p "$STUB_REMOTE/$(basename "$BLOOMERY_REMOTE")" && cd "$STUB_REMOTE/$(basename "$BLOOMERY_REMOTE")" || exit 70
  echo "$side $BLOOMERY_REMOTE blocked" >> "$STUB_LOG"
  printf 'exec sleep 60\n' | PATH="$STUB_BIN:$PATH" bash -c "$*"
  exit $?
fi
cat > /dev/null
new=no
[ -e "$here/new.txt" ] && new=yes
echo "$side $BLOOMERY_REMOTE marker=$(cat "$here/marker.txt") new=$new" >> "$STUB_LOG"
[ "$side" = head ] && [ -n "${STUB_EDIT:-}" ] && echo more >> "$here/marker.txt"
echo '=== narrow-scan begin gate_x'
echo 'ptx-scan: mod0 bundle=b bytes=1'
echo '=== narrow-scan end gate_x rc=0'
NARROW_STUB_EOF
  chmod +x "$w/tools/"*.sh
  git -C "$w" init -q && git -C "$w" config user.email test@test && git -C "$w" config user.name test && git -C "$w" config commit.gpgsign false
  echo base > "$w/marker.txt"
  git -C "$w" add -A && git -C "$w" commit -q -m base
  wb=$(git -C "$w" rev-parse HEAD)
  echo edited > "$w/marker.txt"
  echo new > "$w/new.txt"
  # wt_run <stub log> <narrow-scan args...>: the run's output in $got, its code in $rc
  wt_run() {
    local log=$1
    shift
    : > "$log"
    got=$(HOME="$t/home" STUB_LOG="$log" NARROW_BASE_REMOTE=rb NARROW_HEAD_REMOTE=rh bash "$w/tools/narrow-scan.sh" "$@" 2>&1)
    rc=$?
  }
  wt_run "$t/wt.log" "$wb" --worktree --out "$t/wt-out"
  [ "$rc" = 0 ]; ok "worktree: the run ends rc 0" $?
  grep -qx "head rh marker=edited new=yes" "$t/wt.log"; ok "worktree: the head side synced the edited and the untracked file" $?
  grep -qx "base rb marker=base new=no" "$t/wt.log"; ok "worktree: the base side is BASE's commit" $?
  grep -qx "recipes scan-set" "$t/wt.log"; ok "worktree: the scan set is read from the tree, no commit named" $?
  grep -qx "affected $wb --narrow --scan $t/wt-out/base/gate_x.log $t/wt-out/head/gate_x.log" "$t/wt.log"; ok "worktree: affected reads the working tree against BASE with the pair" $?
  grep -qx "head=worktree@$wb" "$t/wt-out/MANIFEST" && grep -q '^worktree_key=[0-9a-f]\{32\}$' "$t/wt-out/MANIFEST"; ok "worktree: the MANIFEST names the tree and its key" $?
  wt_run "$t/cm.log" "$wb" "$wb" --out "$t/cm-out"
  [ "$rc" = 0 ] && grep -qx "head rh marker=base new=no" "$t/cm.log" && grep -qx "recipes scan-set $wb" "$t/cm.log" && grep -qx "affected $wb..$wb --narrow --scan $t/cm-out/base/gate_x.log $t/cm-out/head/gate_x.log" "$t/cm.log"
  ok "worktree: a commit head side still reads the commit, not the edit (the contrast)" $?
  STUB_EDIT=1 wt_run "$t/edit.log" "$wb" --worktree --out "$t/edit-out"
  [ "$rc" = 1 ] && grep -qF "the working tree changed during the run" <<< "$got"; ok "worktree: a tree edited during the run is refused by name" $?
  wt_run "$t/arg.log" "$wb" "$wb" --worktree --out "$t/arg-out"
  [ "$rc" = 64 ] && grep -qF -- "--worktree takes no HEAD" <<< "$got"; ok "worktree: a HEAD beside --worktree is refused rc 64 by name" $?
  orphan=$(git -C "$w" commit-tree "$(git -C "$w" rev-parse 'HEAD^{tree}')" -m orphan)
  wt_run "$t/anc.log" "$orphan" --worktree --out "$t/anc-out"
  [ "$rc" = 64 ] && grep -qF "is not an ancestor of this tree's HEAD" <<< "$got" && [ ! -s "$t/anc.log" ]; ok "worktree: a BASE that is not an ancestor is refused rc 64 by name, nothing synced" $?
  # Per-track remote dirs (the header): the tree's directory name minus "bloomery-" names its pair, the main tree keeps the old
  # names. Each case runs a copy of the throwaway tree under another directory name; --dry-run prints the pair and touches no box.
  mkdir -p "$t/names"
  for name in bloomery bloomery-aa bloomery-bb plain 'bloomery-a+b'; do cp -R "$w" "$t/names/$name"; done
  # dirs_of <tree name> [VAR=value…]: the dry run's output in $got, its code in $rc
  dirs_of() {
    local name=$1
    shift
    got=$(env HOME="$t/home" STUB_LOG="$t/dirs.log" "$@" bash "$t/names/$name/tools/narrow-scan.sh" "$wb" --dry-run 2>&1)
    rc=$?
  }
  dirs_of bloomery-aa
  grep -qF "base dir ~/repo/bloomery-aa-narrow-base, head dir ~/repo/bloomery-aa-narrow-head," <<< "$got"; ok "dirs: a track's pair is ~/repo/bloomery-<track>-narrow-base and -narrow-head" $?
  aa=$got
  dirs_of bloomery-bb
  [ "$got" != "$aa" ] && grep -qF "~/repo/bloomery-bb-narrow-base" <<< "$got" && ! grep -qF "bloomery-aa" <<< "$got"; ok "dirs: two tracks get distinct dirs" $?
  dirs_of bloomery
  grep -qF "base dir ~/repo/bloomery-narrow-base, head dir ~/repo/bloomery-narrow-head," <<< "$got"; ok "dirs: the main tree keeps ~/repo/bloomery-narrow-base and -narrow-head (the unchanged names)" $?
  dirs_of plain
  grep -qF "base dir ~/repo/bloomery-plain-narrow-base," <<< "$got"; ok "dirs: a tree not named bloomery-<track> is a track of its own name" $?
  dirs_of bloomery-aa NARROW_BASE_REMOTE=/x/base
  grep -qF "base dir /x/base, head dir ~/repo/bloomery-aa-narrow-head," <<< "$got"; ok "dirs: NARROW_BASE_REMOTE overrides the base dir, the head dir stays the track's" $?
  dirs_of bloomery-aa NARROW_HEAD_REMOTE=/x/head
  grep -qF "base dir ~/repo/bloomery-aa-narrow-base, head dir /x/head," <<< "$got"; ok "dirs: NARROW_HEAD_REMOTE overrides the head dir, the base dir stays the track's" $?
  dirs_of 'bloomery-a+b'
  [ "$rc" = 64 ] && grep -qF "is not a box directory name" <<< "$got"; ok "dirs: a tree name outside [A-Za-z0-9._-] is refused by name (64)" $?
  dirs_of 'bloomery-a+b' NARROW_BASE_REMOTE=/x/b NARROW_HEAD_REMOTE=/x/h
  [ "$rc" = 0 ] && grep -qF "base dir /x/b, head dir /x/h," <<< "$got"; ok "dirs: both overrides given, the tree's name is not read" $?
  # The stop and the lock. trk_run <tree> <stub log> <args…>: a run on the tree's own dirs that does not wait for the lock.
  trk_run() {
    local tree=$1 log=$2
    shift 2
    : > "$log"
    got=$(HOME="$t/home" STUB_LOG="$log" NARROW_WAIT=0 bash "$tree/tools/narrow-scan.sh" "$@" 2>&1)
    rc=$?
  }
  # The fake `timeout` execs the command, so the runner is one process (the real one ends its whole process group on a TERM).
  mkdir -p "$t/fakebin" "$t/remote"
  printf '%s\n' '#!/bin/sh' 'shift 2' 'exec "$@"' > "$t/fakebin/timeout"
  chmod +x "$t/fakebin/timeout"
  : > "$t/stop-stub.log"
  HOME="$t/home" STUB_LOG="$t/stop-stub.log" STUB_BLOCK=1 STUB_REMOTE="$t/remote" STUB_BIN="$t/fakebin" \
    bash "$t/names/bloomery-aa/tools/narrow-scan.sh" "$wb" --worktree --out "$t/stop-out" > "$t/stop.log" 2>&1 &
  sp=$!
  pf='' k=0
  while [ -z "$pf" ] && [ "$k" -lt 100 ] && kill -0 "$sp" 2> /dev/null; do
    for f in "$t"/remote/*/.narrow-scan-*.pid; do [ ! -e "$f" ] || pf=$f; done
    [ -n "$pf" ] || sleep 0.2
    k=$((k + 1))
  done
  [ -n "$pf" ]; ok "stop: the box command records the pid of the runner it started" $?
  rp=$(cat "$pf" 2> /dev/null || echo 0)
  case $rp in '' | *[!0-9]*) rp=0 ;; esac
  [ "$rp" -gt 1 ] && kill -0 "$rp" 2> /dev/null; ok "stop: … and that pid is the live runner" $?
  trk_run "$t/names/bloomery-aa" "$t/same.log" "$wb" "$wb" --out "$t/same-out"
  [ "$rc" = 75 ] && grep -qF "is held by pid $sp" <<< "$got"; ok "lock: a second run on the same pair is refused rc 75, naming the holder" $?
  trk_run "$t/names/bloomery-bb" "$t/bb.log" "$wb" "$wb" --out "$t/bb-out"
  [ "$rc" = 0 ] && grep -qx "base ~/repo/bloomery-bb-narrow-base marker=base new=no" "$t/bb.log" && grep -qx "head ~/repo/bloomery-bb-narrow-head marker=base new=no" "$t/bb.log"
  ok "lock: another track's run is not held up by it and runs on its own pair" $?
  kill -TERM "$sp"
  k=0
  while kill -0 "$sp" 2> /dev/null && [ "$k" -lt 100 ]; do sleep 0.2; k=$((k + 1)); done
  gone=0
  if kill -0 "$sp" 2> /dev/null; then kill -KILL "$sp" 2> /dev/null; gone=1; fi
  src=0
  { wait "$sp"; src=$?; } 2> /dev/null
  [ "$gone" = 0 ] && [ "$src" = 143 ]; ok "stop: a TERM ends the run at once, rc 143" $?
  [ "$rp" -gt 1 ] && ! kill -0 "$rp" 2> /dev/null; ok "stop: the recorded remote pid is gone" $?
  grep -qx "readonly ~/repo/bloomery-aa-narrow-base" "$t/stop-stub.log" && grep -qF "signalled the recorded remote pid $rp" "$t/stop.log"
  ok "stop: it signalled exactly that pid, through a read-only call on the side's dir" $?
  [ -z "$(ls -d "$t/home/.cache/bloomery"/narrow-scan*.lock.d 2> /dev/null)" ]; ok "stop: the lock is released" $?
  # The main tree's pair keeps the lock older copies of the script took for every tree: a live holder of it holds that pair, and no other.
  mkdir -p "$t/home/.cache/bloomery/narrow-scan.lock.d"
  echo $$ > "$t/home/.cache/bloomery/narrow-scan.lock.d/pid"
  echo "an older copy" > "$t/home/.cache/bloomery/narrow-scan.lock.d/who"
  trk_run "$t/names/bloomery" "$t/legacy.log" "$wb" "$wb" --out "$t/legacy-out"
  [ "$rc" = 75 ] && grep -qF "is held by pid $$ (an older copy)" <<< "$got"; ok "lock: the main tree's pair is held by a run of an older copy (the old lock name), rc 75 naming it" $?
  trk_run "$t/names/bloomery-bb" "$t/legacy-bb.log" "$wb" "$wb" --out "$t/legacy-bb-out"
  [ "$rc" = 0 ]; ok "lock: … and a track's own pair is not" $?
  rm -rf "$t/home/.cache/bloomery/narrow-scan.lock.d"
  # A stop before the box command has started a runner: nothing is recorded there, and the local box.sh is ended by its pid file.
  # hang_case <TERM|INT> <rc>: a run with the stub held before it starts anything on the box, stopped by that signal. The run starts
  # with INT at its default: a background job, or any step of a parallel runner (mac-static), has it ignored, and an ignored signal
  # cannot be trapped.
  hang_case() {
    local sig=$1 want=$2 hp hs hc k gone src
    : > "$t/hang-stub.log"
    rm -f "$t/remote/hang.pid" "$t/remote/hang-sleep.pid"
    HOME="$t/home" STUB_LOG="$t/hang-stub.log" STUB_HANG=1 STUB_REMOTE="$t/remote" \
      python3 -c 'import os, signal, sys; signal.signal(signal.SIGINT, signal.SIG_DFL); os.execvp("bash", ["bash"] + sys.argv[1:])' \
      "$t/names/bloomery-bb/tools/narrow-scan.sh" "$wb" "$wb" --out "$t/hang-out-$sig" > "$t/hang-$sig.log" 2>&1 &
    hp=$!
    k=0
    while [ ! -s "$t/remote/hang.pid" ] && [ "$k" -lt 100 ] && kill -0 "$hp" 2> /dev/null; do sleep 0.2; k=$((k + 1)); done
    hs=$(cat "$t/remote/hang.pid" 2> /dev/null || echo 0)
    case $hs in '' | *[!0-9]*) hs=0 ;; esac
    [ "$hs" -gt 1 ] && [ "$(cat "$t/hang-out-$sig/base.box.pid" 2> /dev/null)" = "$hs" ]; ok "stop $sig: the pid file under DIR holds the local box.sh's pid" $?
    kill -"$sig" "$hp"
    k=0
    while kill -0 "$hp" 2> /dev/null && [ "$k" -lt 100 ]; do sleep 0.2; k=$((k + 1)); done
    gone=0
    if kill -0 "$hp" 2> /dev/null; then kill -KILL "$hp" 2> /dev/null; gone=1; fi
    src=0
    { wait "$hp"; src=$?; } 2> /dev/null
    [ "$gone" = 0 ] && [ "$src" = "$want" ] && grep -qF "no remote pid recorded in .narrow-scan-base-$hp.pid" "$t/hang-$sig.log"
    ok "stop $sig: before any runner started it says so and ends rc $want" $?
    [ "$hs" -gt 1 ] && ! kill -0 "$hs" 2> /dev/null; ok "stop $sig: … and the local box.sh is gone" $?
    hc=$(cat "$t/remote/hang-sleep.pid" 2> /dev/null || echo 0)
    case $hc in '' | *[!0-9]*) hc=0 ;; esac
    [ "$hc" -gt 1 ] && ! kill -0 "$hc" 2> /dev/null; ok "stop $sig: … and so is the process it had started (the ssh), by the job's process group" $?
    if [ "$hc" -gt 1 ]; then kill -TERM "$hc" 2> /dev/null || true; fi
  }
  hang_case TERM 143
  hang_case INT 130
  # usage refusals
  for args in "" "--bogus a" "a b c" "x --out"; do
    # shellcheck disable=SC2086
    got=$(bash "$0" $args 2>&1); rc=$?
    [ "$rc" = 64 ]; ok "usage: '$args' is refused rc 64" $?
  done
  got=$(bash "$0" no-such-rev-1234 2>&1); rc=$?
  [ "$rc" = 64 ] && grep -qF "'no-such-rev-1234' is not a commit" <<< "$got"; ok "usage: an unknown BASE is refused by name" $?
  rm -rf "$t"
  want=ok
  [ "$bad" = 0 ] || want=FAIL
  echo "narrow-scan: self-test $want ($n cases, $bad failed)"
  [ "$bad" = 0 ]
}

if [ "${1:-}" = --self-test ]; then
  self_test
  exit
fi

# The scan pairs are sm_86's (ARCH): ptx-scan.sh refuses another arch, and a box build at one would end there.
[ "${BLOOMERY_ARCH-$ARCH}" = "$ARCH" ] || die 64 "BLOOMERY_ARCH is '$BLOOMERY_ARCH': the scan pairs are $ARCH's (tools/ptx-scan.sh refuses another arch: its blk/SM column is $ARCH's occupancy arithmetic)"

BASE='' HEAD_REV='' OUT='' DRY=0 WORKTREE=0
while [ $# -gt 0 ]; do
  case $1 in
    --out)
      [ $# -ge 2 ] || die 64 "--out needs a directory; $USAGE"
      OUT=$2
      shift 2
      ;;
    --dry-run) DRY=1; shift ;;
    --worktree) WORKTREE=1; shift ;;
    -*) die 64 "'$1' is not a flag of this tool; $USAGE" ;;
    *)
      if [ -z "$BASE" ]; then BASE=$1
      elif [ -z "$HEAD_REV" ]; then HEAD_REV=$1
      else die 64 "'$1' after BASE and HEAD; $USAGE"
      fi
      shift
      ;;
  esac
done
[ -n "$BASE" ] || die 64 "$USAGE"
[ "$WORKTREE" = 0 ] || [ -z "$HEAD_REV" ] || die 64 "--worktree takes no HEAD ('$HEAD_REV'): its head side is this tree's current content; $USAGE"
HEAD_REV=${HEAD_REV:-HEAD}
BASE_SHA=$(git -C "$ROOT" rev-parse --verify --quiet "$BASE^{commit}") || die 64 "'$BASE' is not a commit of this repository"
HEAD_SHA=$(git -C "$ROOT" rev-parse --verify --quiet "$HEAD_REV^{commit}") || die 64 "'$HEAD_REV' is not a commit of this repository"
HEAD_NAME=${HEAD_SHA:0:8}
if [ "$WORKTREE" = 1 ]; then
  [ "$(git -C "$ROOT" merge-base "$BASE_SHA" "$HEAD_SHA" 2> /dev/null)" = "$BASE_SHA" ] \
    || die 64 "--worktree: '$BASE' is not an ancestor of this tree's HEAD ${HEAD_SHA:0:8}; the affected run reads the working tree against the merge base, so the base side must be that commit"
  HEAD_NAME=worktree
  WT_KEY0=$(worktree_key "$BASE_SHA")
fi
[ -n "$OUT" ] || OUT=$HOME/.cache/bloomery/narrow/$(basename "$ROOT")/${BASE_SHA:0:8}-$HEAD_NAME
case $OUT in /*) ;; *) OUT=$PWD/$OUT ;; esac
mkdir -p "$OUT" || die 70 "cannot make $OUT"
# The pair of remote dirs (the header): this tree's track is its directory name without "bloomery-"; the main tree has none.
TRACK=''
if [ -z "${NARROW_BASE_REMOTE:-}" ] || [ -z "${NARROW_HEAD_REMOTE:-}" ]; then
  case $(basename "$ROOT") in
    bloomery) ;;
    bloomery-?*) TRACK=$(basename "$ROOT") TRACK=${TRACK#bloomery-} ;;
    *) TRACK=$(basename "$ROOT") ;;
  esac
  case $TRACK in
    *[!A-Za-z0-9._-]*) die 64 "this tree's name '$TRACK' is not a box directory name ([A-Za-z0-9._-]): name the remote dirs with NARROW_BASE_REMOTE and NARROW_HEAD_REMOTE" ;;
  esac
fi
PAIR='~/repo/bloomery'
[ -z "$TRACK" ] || PAIR=$PAIR-$TRACK
BASE_DIR=${NARROW_BASE_REMOTE:-$PAIR-narrow-base}
HEAD_DIR=${NARROW_HEAD_REMOTE:-$PAIR-narrow-head}
BOUND=${NARROW_BOUND:-7200}
case $BOUND in '' | *[!0-9]*) die 64 "NARROW_BOUND is whole seconds, got '$BOUND'" ;; esac
if [ "$WORKTREE" = 1 ]; then
  HEAD_LABEL="the working tree (HEAD ${HEAD_SHA:0:8}, key $WT_KEY0)"
  SCAN_SET_REV=()
  AFFECTED_RANGE=$BASE_SHA
  HEAD_SRC=worktree
else
  [ -z "$(git -C "$ROOT" status --porcelain 2> /dev/null | head -n 1)" ] || echo "narrow-scan: note: this tree has uncommitted changes; the scans read the commits ${BASE_SHA:0:8} and ${HEAD_SHA:0:8} only (--worktree reads the tree)" >&2
  HEAD_LABEL=${HEAD_SHA:0:8}
  SCAN_SET_REV=("$HEAD_SHA")
  AFFECTED_RANGE=$BASE_SHA..$HEAD_SHA
  HEAD_SRC=$HEAD_SHA
fi

echo "narrow-scan: ${BASE_SHA:0:8}..$HEAD_LABEL (no gate, no lease, no hold, no card, no gate lock: --no-jit scans); out $OUT"
python3 "$ROOT/tools/recipes.py" scan-set ${SCAN_SET_REV[@]+"${SCAN_SET_REV[@]}"} > "$OUT/scan-set.tsv" 2> "$OUT/scan-set.err" || {
  cat "$OUT/scan-set.err" >&2
  die 2 "the scan set could not be derived from $HEAD_LABEL"
}
cat "$OUT/scan-set.err" >&2
BINS=()
while IFS=$'\t' read -r b _; do [ -z "$b" ] || BINS[${#BINS[@]}]=$b; done < "$OUT/scan-set.tsv"
make_runner "$OUT/scan-set.tsv" "$ROOT/tools/ptx-scan.sh" "$OUT/runner.sh" || die 2 "the runner could not be written"
echo "narrow-scan: scan set (bin, features, carriers it covers):"
sed 's/^/  /' "$OUT/scan-set.tsv"
if [ "$DRY" = 1 ]; then
  echo "narrow-scan: --dry-run: base dir $BASE_DIR, head dir $HEAD_DIR, bound ${BOUND}s; the runner each side runs is $OUT/runner.sh"
  exit 0
fi

LOCKD=$HOME/.cache/bloomery/narrow-scan-$(str_key "$BASE_DIR $HEAD_DIR").lock.d
# The main tree's pair is the one every copy of this script shared, under one lock of this name before the pairs were per track: it
# keeps that lock, so a run of an older copy (a tree not yet rebased) still excludes this one from the dirs.
[ "$BASE_DIR $HEAD_DIR" != '~/repo/bloomery-narrow-base ~/repo/bloomery-narrow-head' ] || LOCKD=$HOME/.cache/bloomery/narrow-scan.lock.d
lock_take "$LOCKD" "${NARROW_WAIT:-1800}" "$(basename "$ROOT") ${BASE_SHA:0:8}..$HEAD_NAME since $(date +%H:%M:%S)" || exit $?
trap 'lock_release "$LOCKD"' EXIT
trap 'stop_run 130' INT
trap 'stop_run 143' TERM

T0=$SECONDS
WALL_BASE=- WALL_HEAD=-
for side in base head; do
  if [ "$side" = base ]; then rev=$BASE_SHA dir=$BASE_DIR; else rev=$HEAD_SRC dir=$HEAD_DIR; fi
  left=$((BOUND - (SECONDS - T0)))
  [ "$left" -ge 60 ] || die 124 "the bound of ${BOUND}s is spent before the $side side (the base side took ${WALL_BASE}s): run again, the dirs keep their compiled crates"
  run_side "$side" "$rev" "$dir" "$OUT/runner.sh" "$OUT" "$left" "${BINS[@]}" || exit $?
  if [ "$side" = base ]; then WALL_BASE=$SIDE_WALL; else WALL_HEAD=$SIDE_WALL; fi
  split_logs "$OUT/$side.scan.out" "$OUT/$side" "${BINS[@]}" > "$OUT/$side.split.err" 2>&1 || {
    cat "$OUT/$side.split.err" >&2
    die 1 "$side: the box output does not hold every scan ($OUT/$side.scan.out)"
  }
done

echo "narrow-scan: bundles each scan carried (base | head):"
for b in "${BINS[@]}"; do
  echo "  $b: $(sed -n 's/^ptx-scan: mod[0-9]* bundle=\([^ ]*\) .*/\1/p' "$OUT/base/$b.log" | tr '\n' ' ')| $(sed -n 's/^ptx-scan: mod[0-9]* bundle=\([^ ]*\) .*/\1/p' "$OUT/head/$b.log" | tr '\n' ' ')"
done
ARGS=()
for b in "${BINS[@]}"; do ARGS[${#ARGS[@]}]=--scan; ARGS[${#ARGS[@]}]=$OUT/base/$b.log; ARGS[${#ARGS[@]}]=$OUT/head/$b.log; done
if [ "$WORKTREE" = 1 ]; then
  WT_KEY1=$(worktree_key "$BASE_SHA")
  [ "$WT_KEY1" = "$WT_KEY0" ] || die 1 "the working tree changed during the run (key $WT_KEY0 -> $WT_KEY1): the head scans and the affected list would read different content; run again with the tree still"
fi
"$ROOT/tools/affected-gates.sh" "$AFFECTED_RANGE" --narrow "${ARGS[@]}" > "$OUT/narrowed.txt" 2> "$OUT/narrowed.err"
arc=$?
cat "$OUT/narrowed.txt"
cat "$OUT/narrowed.err" >&2
{
  echo "base=$BASE_SHA"
  if [ "$WORKTREE" = 1 ]; then
    echo "head=worktree@$HEAD_SHA"
    echo "worktree_key=$WT_KEY1"
  else
    echo "head=$HEAD_SHA"
  fi
  echo "base_dir=$BASE_DIR"
  echo "head_dir=$HEAD_DIR"
  echo "wall_base_s=$WALL_BASE"
  echo "wall_head_s=$WALL_HEAD"
  echo "ptx_scan_md5=$(md5_of "$ROOT/tools/ptx-scan.sh")"
  echo "affected_rc=$arc"
} > "$OUT/MANIFEST"
echo "narrow-scan: done: base side ${WALL_BASE}s, head side ${WALL_HEAD}s, affected rc $arc; scan logs and the list are under $OUT"
exit "$arc"
