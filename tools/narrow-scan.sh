#!/usr/bin/env bash
# `just narrow BASE [HEAD]`: the scans `just affected --narrow` needs, taken for you.
#   tools/narrow-scan.sh BASE [HEAD] [--out DIR] [--dry-run]
#   tools/narrow-scan.sh --self-test
#
# What it does, in order:
#   1. Resolves BASE and HEAD (default: this tree's HEAD commit) to commits. The scans read the commits through
#      `git archive`, never the working tree: uncommitted edits are not scanned (a note says so).
#   2. Derives the scan set from HEAD's tree (`tools/recipes.py scan-set`): the fewest bins of bloomery-gpu-gates whose
#      scans cover every kernel carrier (each device lib, and each bin with kernels of its own), each with the cargo
#      features it builds under.
#   3. For each side in turn, BASE then HEAD: extracts the commit into DIR/src-<side>, syncs it to that side's remote
#      dir with its own tools/box.sh, builds the set there (`cargo oxide build`, as the ptx-scan recipe does) and runs
#      `ptx-scan.sh --no-jit` on each bin. This tree's tools/ptx-scan.sh is the scanner on both sides (the commit's
#      own may predate --no-jit), shipped over stdin; the extractor (oxart_ptx) is the commit's.
#   4. Splits the box's output into DIR/<side>/<bin>.log and runs
#      `tools/affected-gates.sh BASE..HEAD --narrow --scan <base log> <head log> ...` with every pair, printing its
#      output: the narrowed list with its why-lines, and the `recipes:` line to paste. DIR/narrowed.txt keeps it.
#
# The remote dirs persist across runs (~/repo/bloomery-narrow-base and ~/repo/bloomery-narrow-head), so a second
# run builds incrementally: the BASE side costs a sync and the scans, the HEAD side the crates the change touched.
# They belong to this recipe alone, and one run uses them at a time: a lock under ~/.cache/bloomery (a second run
# waits up to NARROW_WAIT seconds, default 1800, naming the holder). NARROW_BASE_REMOTE and NARROW_HEAD_REMOTE name other dirs. Remove them
# with `just box-tracks --remove narrow-base narrow-head` once nothing needs the warm builds.
#
# It runs no gate and takes no timing lease, no hold, no card and no gate lock: `--no-jit` loads nothing on a card, and
# the box commands are builds and ptxas. box.sh's own guard still waits while a sitting's lease or hold is up (rc 75 is
# contention: run again, BLOOMERY_BOX_WAIT=3600 waits longer). BLOOMERY_CARD, BLOOMERY_BOX_ENV and BLOOMERY_TIER are
# unset for the box calls, so nothing card-related reaches the box.
#
# DIR (default ~/.cache/bloomery/narrow/<tree>/<base8>-<head8>) keeps: scan-set.tsv, runner.sh (the script each side
# runs), <side>.build.log (the box's stderr: the builds), <side>.scan.out, <side>/<bin>.log (the scan logs a landing
# cites), narrowed.txt and MANIFEST.
#
# NARROW_BOUND (default 7200 s, a hang guard: no box-job length needs approval) bounds the whole run: each side's box command runs under `timeout` with what
# is left. A cut build keeps its compiled crates, so running again continues it. Exit codes: 0 done; 64 usage; 75
# contention (the box guard, or the lock); 70 the box's toolchain; 1 a build or scan failed; 124 the bound; the
# `affected` run's own code otherwise.
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PKG=bloomery-gpu-gates # the package the ptx-scan recipe builds; tools/recipes.py SCAN_PACKAGE
USAGE="usage: narrow-scan.sh BASE [HEAD] [--out DIR] [--dry-run] | --self-test"

die() { # die <rc> <message>
  echo "narrow-scan: $2" >&2
  exit "$1"
}

md5_of() { # md5_of <file>: the Mac has md5, the box md5sum
  if command -v md5sum > /dev/null; then md5sum "$1" | cut -d' ' -f1; else md5 -q "$1"; fi
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
      line="cargo oxide build --arch sm_86 -- -p $PKG --features $f --release"
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

# lock_take <lock dir> <wait seconds> <who>: one run at a time uses the two persistent remote dirs. A holder whose pid
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
      echo "narrow-scan: $lockd is held by pid ${pid:-?} ($(cat "$lockd/who" 2> /dev/null)); the remote dirs take one run at a time (contention, not failure: run again)" >&2
      return 75
    fi
    echo "narrow-scan: waiting for pid ${pid:-?} ($(cat "$lockd/who" 2> /dev/null)): the persistent remote dirs take one run at a time (${waited}s of ${wait}s)" >&2
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

# side_command <bound s>: the one box.sh argument each side runs. The runner arrives on stdin, so the command writes it to a
# temp file and runs it under `timeout`; its rc comes back whatever came before it (the guard's 75, the backend's 70). The
# leading `:` names `cargo oxide`, which makes box.sh check the pinned backend before the command, as it does for a recipe.
side_command() {
  printf '%s' ': cargo oxide && T=$(mktemp) && cat > "$T" && timeout --kill-after=10 '"$1"' bash "$T" </dev/null; rc=$?; rm -f "$T"; exit $rc'
}

# run_side <name> <rev> <remote dir> <runner> <out dir> <bound s> <bin>...: one side's sync, build and scans on the box.
run_side() {
  local name=$1 rev=$2 dir=$3 runner=$4 out=$5 bound=$6 src rc t0 cmd
  shift 6
  src=$out/src-$name
  rm -rf "$src"
  mkdir -p "$src"
  git -C "$ROOT" archive --format=tar "$rev" | tar -x -C "$src"
  rc=("${PIPESTATUS[@]}")
  [ "${rc[0]}" = 0 ] && [ "${rc[1]}" = 0 ] || die 1 "git archive $rev failed (git ${rc[0]}, tar ${rc[1]})"
  cmd=$(side_command "$bound")
  echo "narrow-scan: $name: $rev -> $dir (bound ${bound}s)" >&2
  t0=$SECONDS
  # git must not find this tree's repository above the archive (box.sh stamps its HEAD into the build as the commit).
  (
    cd "$src" && env -u BLOOMERY_CARD -u BLOOMERY_BOX_ENV -u BLOOMERY_TIER GIT_CEILING_DIRECTORIES="$out" BLOOMERY_REMOTE="$dir" \
      ./tools/box.sh "$cmd" < "$runner" 2>&1 > "$out/$name.scan.out"
  ) | tee "$out/$name.build.log" >&2
  rc=${PIPESTATUS[0]}
  SIDE_WALL=$((SECONDS - t0))
  rm -rf "$src"
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
  grep -qF "cargo oxide build --arch sm_86 -- -p $PKG --features deepseek41,vision --release --bin bloomery-serve-ds41 >&2" "$t/runner.sh"; ok "runner: the V4.1 group" $?
  grep -qF "cargo oxide build --arch sm_86 -- -p $PKG --features gpu --release --bin gate_kquant --bin gate_swap >&2" "$t/runner.sh"; ok "runner: the gpu group takes both bins" $?
  [ "$(grep -c "^cargo build --release -p $PKG --bin oxart_ptx" "$t/runner.sh")" = 1 ]; ok "runner: one extractor build" $?
  [ "$(grep -c -- '--no-jit' "$t/runner.sh")" = 3 ]; ok "runner: one --no-jit scan per bin" $?
  sed -n "/^cat > \"\$P\" <<'NARROW_PTX_SCAN_EOF'$/,/^NARROW_PTX_SCAN_EOF$/p" "$t/runner.sh" | sed '1d;$d' | cmp -s - "$t/ptx-scan.sh"; ok "runner: ptx-scan.sh embedded verbatim" $?
  [ "$(grep -c '^echo "narrow-scan: wall: ' "$t/runner.sh")" = 6 ]; ok "runner: a wall line per build group, the extractor and each scan" $?
  case $(side_command 900) in *"cargo oxide"*) ;; *) false ;; esac; ok "side command: names cargo oxide, so box.sh checks the backend" $?
  case $(side_command 900) in *"timeout --kill-after=10 900 bash"*) ;; *) false ;; esac; ok "side command: the runner runs under the bound" $?
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
  grep -qF "cargo oxide build --arch sm_86 -- -p $PKG --features {{FEATURES}} --release --bin {{BIN}}" <<< "$recipe"; ok "drift: the ptx-scan recipe still builds with this runner's cargo oxide line" $?
  grep -qF "cargo build --release -p $PKG --bin oxart_ptx && bash tools/ptx-scan.sh" <<< "$recipe"; ok "drift: the ptx-scan recipe still builds the extractor as this runner does" $?
  grep -qF "NARROW_PTX_SCAN_EOF" "$ROOT/tools/ptx-scan.sh"; [ $? = 1 ]; ok "drift: ptx-scan.sh does not hold the runner's here-document marker" $?
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

BASE='' HEAD_REV='' OUT='' DRY=0
while [ $# -gt 0 ]; do
  case $1 in
    --out)
      [ $# -ge 2 ] || die 64 "--out needs a directory; $USAGE"
      OUT=$2
      shift 2
      ;;
    --dry-run) DRY=1; shift ;;
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
HEAD_REV=${HEAD_REV:-HEAD}
BASE_SHA=$(git -C "$ROOT" rev-parse --verify --quiet "$BASE^{commit}") || die 64 "'$BASE' is not a commit of this repository"
HEAD_SHA=$(git -C "$ROOT" rev-parse --verify --quiet "$HEAD_REV^{commit}") || die 64 "'$HEAD_REV' is not a commit of this repository"
[ -n "$OUT" ] || OUT=$HOME/.cache/bloomery/narrow/$(basename "$ROOT")/${BASE_SHA:0:8}-${HEAD_SHA:0:8}
case $OUT in /*) ;; *) OUT=$PWD/$OUT ;; esac
mkdir -p "$OUT" || die 70 "cannot make $OUT"
BASE_DIR=${NARROW_BASE_REMOTE:-'~/repo/bloomery-narrow-base'}
HEAD_DIR=${NARROW_HEAD_REMOTE:-'~/repo/bloomery-narrow-head'}
BOUND=${NARROW_BOUND:-7200}
case $BOUND in '' | *[!0-9]*) die 64 "NARROW_BOUND is whole seconds, got '$BOUND'" ;; esac
[ -z "$(git -C "$ROOT" status --porcelain 2> /dev/null | head -n 1)" ] || echo "narrow-scan: note: this tree has uncommitted changes; the scans read the commits ${BASE_SHA:0:8} and ${HEAD_SHA:0:8} only" >&2

echo "narrow-scan: ${BASE_SHA:0:8}..${HEAD_SHA:0:8} (no gate, no lease, no hold, no card, no gate lock: --no-jit scans); out $OUT"
python3 "$ROOT/tools/recipes.py" scan-set "$HEAD_SHA" > "$OUT/scan-set.tsv" 2> "$OUT/scan-set.err" || {
  cat "$OUT/scan-set.err" >&2
  die 2 "the scan set could not be derived from ${HEAD_SHA:0:8}"
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

LOCKD=$HOME/.cache/bloomery/narrow-scan.lock.d
lock_take "$LOCKD" "${NARROW_WAIT:-1800}" "$(basename "$ROOT") ${BASE_SHA:0:8}..${HEAD_SHA:0:8} since $(date +%H:%M:%S)" || exit $?
trap 'lock_release "$LOCKD"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

T0=$SECONDS
WALL_BASE=- WALL_HEAD=-
for side in base head; do
  if [ "$side" = base ]; then rev=$BASE_SHA dir=$BASE_DIR; else rev=$HEAD_SHA dir=$HEAD_DIR; fi
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
"$ROOT/tools/affected-gates.sh" "$BASE_SHA..$HEAD_SHA" --narrow "${ARGS[@]}" > "$OUT/narrowed.txt" 2> "$OUT/narrowed.err"
arc=$?
cat "$OUT/narrowed.txt"
cat "$OUT/narrowed.err" >&2
{
  echo "base=$BASE_SHA"
  echo "head=$HEAD_SHA"
  echo "base_dir=$BASE_DIR"
  echo "head_dir=$HEAD_DIR"
  echo "wall_base_s=$WALL_BASE"
  echo "wall_head_s=$WALL_HEAD"
  echo "ptx_scan_md5=$(md5_of "$ROOT/tools/ptx-scan.sh")"
  echo "affected_rc=$arc"
} > "$OUT/MANIFEST"
echo "narrow-scan: done: base side ${WALL_BASE}s, head side ${WALL_HEAD}s, affected rc $arc; scan logs and the list are under $OUT"
exit "$arc"
