#!/usr/bin/env bash
# `just ptx-compare BASE BIN [--features F]`: the add and move classes' ptx-scan proof (AGENTS.md's change-class table) as one command.
#   tools/ptx-compare.sh BASE BIN [--features F] [--base-only] [--no-cache]
#   tools/ptx-compare.sh --self-test
#
# What it does. Scans BIN twice with the driver JIT (the add class reads the JIT columns), at BASE and at this tree, and prints one table
# (tools/recipes.py `ptx-compare`, which reads both logs through tools/ref/ptx-canon.py's parse_scan, the reader `just affected --narrow`
# and `just canon-check` share): the entries on each side, the entries added, the entries removed, and the existing entries whose any
# column or ptx-scan-md5 moved, each with its moved columns.
#   1. BASE (any rev that names a commit) is read through `git archive`, never the working tree, into a temp dir; this tree's
#      tools/ptx-scan.sh is laid over it, so both sides run one scanner (the extractor, oxart_ptx, is each side's own). The base
#      side is synced by BASE's own tools/box.sh to ~/repo/bloomery-<track>-ptxcmp-base (<track> this tree's directory name without
#      "bloomery-"; the main tree's: ~/repo/bloomery-ptxcmp-base), a remote dir of its own that persists, so a second run builds
#      incrementally; `just box-tracks --remove <track>-ptxcmp-base` clears it once the worktree is gone.
#   2. The head side is this tree as box.sh syncs it, uncommitted and untracked files included, into the caller's remote dir (what
#      `just ptx-scan` uses), scanned by the same command as the ptx-scan recipe's.
#   3. Both calls are box.sh calls of this environment: with BLOOMERY_BOX_QUEUE=1 they are queued jobs, and a gate-batch item
#      (`'ptx-compare:BASE generate_ds41 --features gpu,deepseek41'`) carries its lane's card in BLOOMERY_BOX_ENV into both. The JIT
#      runs under tools/gpu-gate.sh as the ptx-scan recipe's does. Both sides are scanned in this order, base first.
#
# The base side's cache. A scan of BASE is the same whenever its inputs are: the BASE sha, BIN, the features, the arch and the md5 of
# this tree's tools/ptx-scan.sh. A run whose key has target/ptx-compare/<base12>-<bin>-<features>-<scanner8>/base.scan prints `ptx-compare:
# base <sha> from cache <file>` and makes no box call for the base side. The file is written whole (a temp file and a rename) only after
# the side ended rc 0 and the log reads as a full scan (tools/recipes.py ptx-compare of the log against itself). The toolkit on the box is
# not in the key: a ptxas or driver move shows as moved rows against the cached base, so after one run with --no-cache (the run's result
# replaces the entry) or remove target/ptx-compare. `--base-only` runs the base side alone (a round opens with it) and prints where the
# cache is; the head side and the table come at the round's end, from the same command without it.
#
# Output files (target/ptx-compare/<key>/): base.scan, head.scan (the scan logs, stdout of the box calls), base.build.log, head.build.log
# (their stderr: the builds), compare.txt (the table).
#
# Exit codes: 0 nothing existing moved (the head holds every base entry with each column and digest equal, and maybe more: `added`
# says which); 3 an existing entry's row or digest moved, or the bundle set did; 4 an entry was removed; 5 both; 2 a log or a pair
# refused by name (a failed scan, a --no-jit scan, two bins, two digest methods, two toolchains: tools/recipes.py's message); 64 usage; 1 a side's box command
# or the base archive failed (named, with its log); 75 the box is held by a sitting or a lease (contention, nothing built: run again);
# 70 the box's toolchain (box.sh's exit 70). Those are the codes a gate-batch item reads: red unless 0.
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PKG=bloomery-gpu-gates # the package the ptx-scan recipe builds
ARCH=sm_86 # the ptx scan is sm_86's (tools/scan-args.sh refuses another BLOOMERY_ARCH before this runs)
USAGE="usage: ptx-compare.sh BASE BIN [--features F] [--base-only] [--no-cache] | --self-test"

die() { # die <rc> <message>
  echo "ptx-compare: $2" >&2
  exit "$1"
}

md5_of() { # md5_of <file>: the Mac has md5, the box md5sum
  if command -v md5sum > /dev/null; then md5sum "$1" | cut -d' ' -f1; else md5 -q "$1"; fi
}

# scan_command <features> <bin>: the box command of a side. The build lines are the ptx-scan recipe's (the self-test holds them equal) with
# the builds' output on stderr, so stdout is the scan log; the scan's own stderr joins it, as `just narrow`'s runner does.
scan_command() {
  printf '%s' "cargo oxide build --arch $ARCH -- -p $PKG --features $1 --release --bin $2 --bin oxart_jit >&2 && cargo build --release -p $PKG --bin oxart_ptx >&2 && bash tools/ptx-scan.sh $2 2>&1"
}

# side_run <name> <dir to run box.sh in> <remote dir or ''> <command> <scan out> <build log>: one side's box call; its rc.
# stdin is /dev/null (a queued call refuses input); the build's stderr is kept in <build log> and shown.
side_run() {
  local name=$1 dir=$2 remote=$3 cmd=$4 out=$5 blog=$6 rc
  echo "ptx-compare: $name: ${remote:-the own remote dir of this tree}" >&2
  (
    cd "$dir" || exit 70
    if [ -n "$remote" ]; then export BLOOMERY_REMOTE="$remote"; fi
    # git must not find this tree's repository above an archive (box.sh stamps its HEAD into the build as the commit)
    if [ "$dir" != "$ROOT" ]; then GIT_CEILING_DIRECTORIES=$(dirname "$dir"); export GIT_CEILING_DIRECTORIES; fi
    exec ./tools/box.sh "$cmd" < /dev/null > "$out" 2> >(tee "$blog" >&2)
  )
  rc=$?
  return "$rc"
}

# side_failed <name> <rc> <build log>: the named line of a failed side; the exit code it maps to (75, 83 and 70 as box.sh gave
# them, so gate-batch's dir-lock retry sees an 83; else 1).
side_failed() {
  case $2 in
    75) echo "ptx-compare: $1: rc 75: the box is held by a sitting or a lease (contention, not failure; nothing was built): run again, BLOOMERY_BOX_WAIT=3600 waits longer" >&2; return 75 ;;
    83) echo "ptx-compare: $1: rc 83: directory lock refused ($3)" >&2; return "$2" ;;
    70) echo "ptx-compare: $1: rc 70: the box's toolchain (box.sh's exit 70: the cuda-oxide backend of this commit's pin is missing)" >&2; return 70 ;;
    *) echo "ptx-compare: $1: the box command ended rc $2 (the build or the scan: $3)" >&2; return 1 ;;
  esac
}

# valid_scan <log>: 0 when the log reads as a full scan of one bin (recipes.py's reader, against itself: identical).
valid_scan() {
  python3 "$ROOT/tools/recipes.py" ptx-compare "$1" "$1" > /dev/null 2>&1
}

# run_compare <base rev> <bin> <features> <base only 0|1> <no cache 0|1>
run_compare() {
  local rev=$1 bin=$2 feats=$3 baseonly=$4 nocache=$5 sha key dir track bremote src rc scanner hl tmp
  local arc=()
  sha=$(git -C "$ROOT" rev-parse --verify --quiet "$rev^{commit}") || die 64 "'$rev' names no commit of this repository"
  case $bin in '' | *[!A-Za-z0-9_-]* | -*) die 64 "'$bin' is no binary name" ;; esac
  case $feats in '' | *[!A-Za-z0-9_,/-]*) die 64 "--features '$feats' is no cargo feature list" ;; esac
  scanner=$(md5_of "$ROOT/tools/ptx-scan.sh" | cut -c1-8)
  key=${sha:0:12}-$bin-${feats//[,\/]/+}-$scanner
  dir=$ROOT/target/ptx-compare/$key
  mkdir -p "$dir" || die 70 "cannot make $dir"
  track=$(basename "$ROOT")
  track=${track#bloomery-}
  if [ "$track" = bloomery ]; then bremote="~/repo/bloomery-ptxcmp-base"; else bremote="~/repo/bloomery-$track-ptxcmp-base"; fi
  # --- the base side
  if [ "$nocache" = 0 ] && [ -s "$dir/base.scan" ] && valid_scan "$dir/base.scan"; then
    echo "ptx-compare: base ${sha:0:12} from cache $dir/base.scan" >&2
  else
    [ ! -e "$dir/base.scan" ] || echo "ptx-compare: base cache $dir/base.scan is $([ "$nocache" = 1 ] && echo 'bypassed (--no-cache)' || echo 'not a full scan'): scanning again" >&2
    src=$(mktemp -d "${TMPDIR:-/tmp}/ptx-compare-base.XXXXXX") || die 70 "no temporary directory"
    git -C "$ROOT" archive --format=tar "$sha" | tar -x -C "$src"
    arc=("${PIPESTATUS[@]}")
    if [ "${arc[0]}" != 0 ] || [ "${arc[1]}" != 0 ]; then
      rm -rf "$src"
      die 1 "git archive ${sha:0:12} failed (git ${arc[0]}, tar ${arc[1]})"
    fi
    # one scanner on both sides; -p keeps its mtime, so a second run's base tree has the id the first synced (box.sh's tree id)
    cp -p "$ROOT/tools/ptx-scan.sh" "$src/tools/ptx-scan.sh" || { rm -rf "$src"; die 70 "cannot lay tools/ptx-scan.sh over the base tree"; }
    tmp=$(mktemp "$dir/base.scan.XXXXXX") || die 70 "cannot create base scan"
    side_run "base ${sha:0:12}" "$src" "$bremote" "$(scan_command "$feats" "$bin")" "$tmp" "$dir/base.build.log"
    rc=$?
    rm -rf "$src"
    if [ "$rc" != 0 ]; then
      rm -f "$tmp"
      side_failed "base ${sha:0:12}" "$rc" "$dir/base.build.log"
      return $?
    fi
    if ! valid_scan "$tmp"; then
      echo "ptx-compare: the base side ended 0 but its log is no full scan of $bin; $(python3 "$ROOT/tools/recipes.py" ptx-compare "$tmp" "$tmp" 2>&1 | tail -1): kept as $dir/base.scan.bad" >&2
      mv -f "$tmp" "$dir/base.scan.bad"
      return 2
    fi
    mv -f "$tmp" "$dir/base.scan" || return 70
  fi
  if [ "$baseonly" = 1 ]; then
    echo "ptx-compare: base ${sha:0:12} of $bin ($feats) is cached: $dir/base.scan"
    return 0
  fi
  # --- the head side: this tree as box.sh syncs it
  hl=$(git -C "$ROOT" rev-parse --short=8 HEAD 2> /dev/null || echo unknown)
  [ -z "$(git -C "$ROOT" status --porcelain 2> /dev/null)" ] || hl=$hl+uncommitted
  tmp=$(mktemp "$dir/head.scan.XXXXXX") || die 70 "cannot create head scan"
  side_run "head $hl" "$ROOT" "" "$(scan_command "$feats" "$bin")" "$tmp" "$dir/head.build.log"
  rc=$?
  if [ "$rc" != 0 ]; then
    rm -f "$tmp"
    side_failed "head $hl" "$rc" "$dir/head.build.log"
    return $?
  fi
  # this call's private scan is the one compared; head.scan is published after, for the next reader
  python3 "$ROOT/tools/recipes.py" ptx-compare "$dir/base.scan" "$tmp" --base-label "${sha:0:12}" --head-label "$hl" |
    tee "$dir/compare.txt"
  rc=${PIPESTATUS[0]}
  mv -f "$tmp" "$dir/head.scan" || return 70
  return "$rc"
}

# --- the self-test: both sides on stub box.sh and ptx-scan.sh in a temp git repo, the table through the real recipes.py
self_test() {
  local t n=0 bad=0 rc out base_sha cases
  t=$(mktemp -d "${TMPDIR:-/tmp}/ptx-compare-test.XXXXXX") || die 70 "self-test: no temporary directory"
  # shellcheck disable=SC2064 # the path is fixed now
  trap "rm -rf '$t'" EXIT
  ok() { # ok <name> <condition rc>
    n=$((n + 1))
    if [ "$2" = 0 ]; then echo "ok $1"; else bad=$((bad + 1)); echo "FAIL $1"; [ -z "${out:-}" ] || printf '%s\n' "$out" | sed 's/^/    | /'; fi
  }
  # --- a scan log the stub box.sh prints: entries `name:digest-char`, the header and the md5 block of tools/ref/ptx-canon.py's reader
  mklog() { # mklog <bin> <entry:c>... : a full scan log on stdout
    local b=$1 e
    shift
    echo "ptx-scan: mod1 bundle=bloomery-gpu bytes=100"
    echo "ptx-scan bin=target/release/$b section=.oxart bytes=9 ptxas=/p ptxas-version=13.3.73 arch=sm_86 modules=1 jit-card=NVIDIA_RTX_A6000 jit-cc=8.6 jit-cuda=13.4"
    echo "entry reqntid depot ld.local st.local fma cvt.f16 regs smem spill blk/SM(static) jit_regs jit_local"
    for e in "$@"; do printf '%-24s 256 no 0 0 0 0 12 0 0 6 12 0\n' "${e%%:*}"; done
    echo "ptx-scan-md5: method=decl1"
    for e in "$@"; do printf '%s %s 41\n' "${e%%:*}" "$(printf '%32s' '' | tr ' ' "${e##*:}")"; done
  }
  # --- the stand-in tree: a git repo whose tools/ are this script, a stub box.sh and ptx-scan.sh, and the real readers
  mkdir -p "$t/repo/tools/ref" "$t/stub" "$t/home"
  cp "$0" "$t/repo/tools/ptx-compare.sh"
  cp "$ROOT/tools/recipes.py" "$t/repo/tools/recipes.py"
  cp "$ROOT/tools/ref/ptx-canon.py" "$t/repo/tools/ref/ptx-canon.py"
  [ ! -d "$ROOT/tools/bloomery" ] || cp -R "$ROOT/tools/bloomery" "$t/repo/tools/bloomery"
  printf '%s\n' '#!/usr/bin/env bash' 'echo "scanner v1"' > "$t/repo/tools/ptx-scan.sh"
  # the stub box.sh: logs its call (cwd's name, the remote dir, queue, the command, the md5 of the ptx-scan.sh it sees), then prints
  # the canned scan of its side (base when BLOOMERY_REMOTE names ptxcmp-base) on stdout and a build line on stderr; STUB_RC_<SIDE> fails it
  cat > "$t/repo/tools/box.sh" << 'STUB'
#!/usr/bin/env bash
side=head
case ${BLOOMERY_REMOTE:-} in *ptxcmp-base) side=base ;; esac
printf '%s\t%s\t%s\tgit_ceiling=%s\tscanner=%s\t%s\n' "$side" "$(basename "$PWD")" "${BLOOMERY_REMOTE:-own}" "${GIT_CEILING_DIRECTORIES:-none}" \
  "$(md5sum tools/ptx-scan.sh 2> /dev/null | cut -c1-8 || echo none)" "$1" >> "$STUB_DIR/box.log"
[ -t 0 ] && echo "stub box.sh: stdin is a terminal" >> "$STUB_DIR/box.log"
echo "   Compiling stub v0.1.0 ($side)" >&2
var=STUB_RC_$(echo "$side" | tr a-z A-Z)
rc=${!var:-0}
[ "$rc" = 0 ] || exit "$rc"
[ "$side" = base ] || sleep "${STUB_SLEEP_HEAD:-0}"
if [ "$side" = head ] && [ -n "${STUB_HEAD_SCAN:-}" ]; then cat "$STUB_HEAD_SCAN"; else cat "$STUB_DIR/$side.scan"; fi
STUB
  chmod +x "$t/repo/tools/box.sh" "$t/repo/tools/ptx-scan.sh" "$t/repo/tools/ptx-compare.sh"
  (
    cd "$t/repo" || exit 70
    export GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_SYSTEM=/dev/null HOME="$t/home"
    git init -q . && git add -A && git -c user.name=t -c user.email=t@t commit -q -m base
  ) || die 70 "self-test: the stand-in repo could not be made"
  base_sha=$(git -C "$t/repo" rev-parse HEAD)
  # this tree's scanner is not the one BASE committed (a change under test may carry a newer ptx-scan.sh): the base side must run this one
  echo 'echo "scanner v2 (the working tree)"' >> "$t/repo/tools/ptx-scan.sh"
  export STUB_DIR="$t/stub" HOME="$t/home" GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_SYSTEM=/dev/null
  mklog gx alpha:1 beta:2 > "$t/stub/base.scan"
  # run [VAR=value...] -- ARGS...: the stand-in's ptx-compare.sh; rc and out are the result, the box log is reset
  run() {
    local envs=()
    while [ "$1" != -- ]; do envs+=("$1"); shift; done
    shift
    : > "$t/stub/box.log"
    rc=0
    out=$(env ${envs[@]+"${envs[@]}"} bash "$t/repo/tools/ptx-compare.sh" "$@" 2>&1 < /dev/null) || rc=$?
  }
  calls() { grep -c "^$1	" "$t/stub/box.log" || true; }
  has() { case $out in *"$1"*) return 0 ;; *) return 1 ;; esac; }
  # 1. a cold run, the head with one entry more: the base is scanned in its own remote dir from the archive, the head from the tree
  mklog gx alpha:1 beta:2 gamma:3 > "$t/stub/head.scan"
  run -- "$base_sha" gx --features gpu
  [ "$rc" = 0 ] && has '  added         1: gamma' && has '  changed       none' && has 'ptx-compare: added-only (rc 0)'; ok "cold run: an added entry is rc 0 and the table names it" $?
  has "base=${base_sha:0:12} head=" && has "  base entries  2" && has "  head entries  3"; ok "cold run: the table names the base commit and counts both sides" $?
  [ "$(calls base)" = 1 ] && [ "$(calls head)" = 1 ]; ok "cold run: one box call a side" $?
  grep -q "^base	.*	~/repo/bloomery-repo-ptxcmp-base	" "$t/stub/box.log"; ok "cold run: the base side runs in a remote dir of its own (…-ptxcmp-base)" $?
  grep -q "^head	repo	own	git_ceiling=none" "$t/stub/box.log"; ok "cold run: the head side runs in this tree with its own remote dir" $?
  grep -q "^base	ptx-compare-base\." "$t/stub/box.log" && grep -q "git_ceiling=${TMPDIR:-/tmp}" "$t/stub/box.log"; ok "cold run: the base side runs in a temp archive, away from this tree's git" $?
  [ "$(md5_of "$t/repo/tools/ptx-scan.sh" | cut -c1-8)" = "$(grep '^base' "$t/stub/box.log" | sed -n 's/.*scanner=\([0-9a-f]*\).*/\1/p')" ]; ok "cold run: the base side runs this tree's tools/ptx-scan.sh" $?
  grep -q -- "cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin gx --bin oxart_jit >&2 && cargo build --release -p bloomery-gpu-gates --bin oxart_ptx >&2 && bash tools/ptx-scan.sh gx 2>&1" "$t/stub/box.log"; ok "cold run: the box command builds the bin and oxart_jit (the JIT is run) and scans it" $?
  ! grep -q -- '--no-jit' "$t/stub/box.log"; ok "cold run: no --no-jit anywhere" $?
  [ -s "$t/repo/target/ptx-compare/${base_sha:0:12}-gx-gpu-$(md5_of "$t/repo/tools/ptx-scan.sh" | cut -c1-8)/base.scan" ]; ok "cold run: the base scan is cached under target/ptx-compare/<key>/" $?
  # 2. the same run again: the base from the cache, one call (the head)
  run -- "$base_sha" gx --features gpu
  [ "$rc" = 0 ] && has "from cache" && [ "$(calls base)" = 0 ] && [ "$(calls head)" = 1 ]; ok "second run: the base from cache, one box call (the head)" $?
  # 3. another feature list is another key; --no-cache scans the base again
  run -- "$base_sha" gx --features gpu,deepseek41
  [ "$rc" = 0 ] && [ "$(calls base)" = 1 ]; ok "another feature list: a cache miss, the base is scanned" $?
  run -- "$base_sha" gx --features gpu --no-cache
  [ "$rc" = 0 ] && has "bypassed (--no-cache)" && [ "$(calls base)" = 1 ]; ok "--no-cache: the base is scanned again" $?
  # 4. what moved: the exit code
  mklog gx alpha:1 beta:2 > "$t/stub/head.scan"
  run -- "$base_sha" gx --features gpu
  [ "$rc" = 0 ] && has '  added         none' && has 'ptx-compare: identical (rc 0)'; ok "head = base: rc 0 and added none" $?
  mklog gx alpha:1 beta:9 > "$t/stub/head.scan"
  run -- "$base_sha" gx --features gpu
  [ "$rc" = 3 ] && has '  changed       1: beta (ptx-scan-md5:'; ok "an existing entry's digest moved: rc 3, named" $?
  mklog gx alpha:1 > "$t/stub/head.scan"
  run -- "$base_sha" gx --features gpu
  [ "$rc" = 4 ] && has '  removed       1: beta'; ok "an entry removed: rc 4, named" $?
  mklog gx alpha:7 gamma:3 > "$t/stub/head.scan"
  run -- "$base_sha" gx --features gpu
  [ "$rc" = 5 ] && has '  added         1: gamma' && has '  removed       1: beta' && has '  changed       1: alpha'; ok "changed and removed: rc 5" $?
  # 5. a head scan that is no full scan of the bin, and a no-jit head, are refused by name (2), never read as unmoved
  mklog gy alpha:1 beta:2 > "$t/stub/head.scan"
  run -- "$base_sha" gx --features gpu
  [ "$rc" = 2 ] && has 'a pair is two scans of one binary'; ok "a scan of another bin: refused, rc 2" $?
  { mklog gx alpha:1 beta:2 | sed -e 's/jit-card=[^ ]* jit-cc=[^ ]* jit-cuda=[^ ]*/jit=skipped(no-jit)/' -e 's/ 12 0$/ skipped(no-jit) skipped(no-jit)/'; } > "$t/stub/head.scan"
  run -- "$base_sha" gx --features gpu
  [ "$rc" = 2 ] && has 'holds no JIT columns'; ok "a --no-jit head: refused, rc 2" $?
  # 6. --base-only: the base side alone, cached for the round's end
  mklog gx alpha:1 beta:2 gamma:3 > "$t/stub/head.scan"
  rm -rf "$t/repo/target"
  run -- "$base_sha" gx --features gpu --base-only
  [ "$rc" = 0 ] && [ "$(calls base)" = 1 ] && [ "$(calls head)" = 0 ] && has 'is cached:' && ! has 'ptx-compare: added'; ok "--base-only: the base side alone, no table" $?
  run -- "$base_sha" gx --features gpu
  [ "$rc" = 0 ] && has "from cache" && [ "$(calls base)" = 0 ]; ok "--base-only then the run: the head side only, the table" $?
  # 7. a side's failure: named, its code mapped, nothing cached
  rm -rf "$t/repo/target"
  run STUB_RC_BASE=75 -- "$base_sha" gx --features gpu
  [ "$rc" = 75 ] && has 'rc 75: the box is held' && [ "$(calls head)" = 0 ]; ok "base side rc 75: contention, passed on, the head side not run" $?
  [ ! -e "$t/repo/target/ptx-compare/${base_sha:0:12}-gx-gpu-$(md5_of "$t/repo/tools/ptx-scan.sh" | cut -c1-8)/base.scan" ]; ok "base side failed: nothing cached" $?
  run STUB_RC_BASE=101 -- "$base_sha" gx --features gpu
  [ "$rc" = 1 ] && has 'the box command ended rc 101'; ok "base side rc 101: named, rc 1" $?
  run STUB_RC_HEAD=70 -- "$base_sha" gx --features gpu
  [ "$rc" = 70 ] && has "the box's toolchain"; ok "head side rc 70: the toolchain, passed on" $?
  run STUB_RC_HEAD=1 -- "$base_sha" gx --features gpu
  [ "$rc" = 1 ] && has 'head' && has 'the box command ended rc 1'; ok "head side rc 1: named, rc 1" $?
  run STUB_RC_HEAD=83 -- "$base_sha" gx --features gpu
  [ "$rc" = 83 ] && has 'rc 83: directory lock refused'; ok "head side rc 83: passed on, so gate-batch's dir-lock retry sees it" $?
  # two calls with one cache key at once: a slow one scanning changed PTX and a fast one scanning the base's; each compares
  # its own scan (a shared temp file let the fast one's publish stand in for the slow one's head)
  mklog gx alpha:9 beta:2 > "$t/stub/changed.scan"
  mklog gx alpha:1 beta:2 > "$t/stub/same.scan"
  : > "$t/stub/box.log"
  ( cd "$t" && STUB_HEAD_SCAN="$t/stub/changed.scan" STUB_SLEEP_HEAD=2 bash "$t/repo/tools/ptx-compare.sh" "$base_sha" gx --features gpu > "$t/slow.out" 2>&1 < /dev/null; echo $? > "$t/slow.rc" ) &
  slow=$!
  sleep 1
  # mid-scan, the slow call's head scan is a private name, never the shared one another call would truncate
  shared=$(find "$t/repo/target" -name head.scan.tmp 2> /dev/null | head -1)
  ( cd "$t" && STUB_HEAD_SCAN="$t/stub/same.scan" bash "$t/repo/tools/ptx-compare.sh" "$base_sha" gx --features gpu > "$t/fast.out" 2>&1 < /dev/null; echo $? > "$t/fast.rc" ) &
  fast=$!
  wait "$slow" "$fast"
  out="slow rc $(cat "$t/slow.rc"), fast rc $(cat "$t/fast.rc"), shared temp '${shared}'"
  [ -z "$shared" ] && [ "$(cat "$t/slow.rc")" = 3 ] && [ "$(cat "$t/fast.rc")" = 0 ] && grep -q '  changed       1: alpha' "$t/slow.out"; ok "two calls at once: each compares its own private head scan" $?
  # 8. usage
  run -- no-such-rev gx
  [ "$rc" = 64 ] && has "names no commit"; ok "a rev that names no commit: 64" $?
  run -- "$base_sha" 'g x'
  [ "$rc" = 64 ] && has "is no binary name"; ok "a bin that is no name: 64" $?
  run -- "$base_sha" gx --features 'gpu deepseek41'
  [ "$rc" = 64 ] && has "is no cargo feature list"; ok "a feature list that is none: 64" $?
  run -- "$base_sha"
  [ "$rc" = 64 ] && has "usage:"; ok "a missing bin: usage, 64" $?
  # 9. the build lines are the ptx-scan recipe's (a drift here means the head side scans something the recipe does not)
  recipe=$(awk '/^ptx-scan /{f=1; next} f && /^[^ ]/{exit} f' "$ROOT/justfile")
  case $recipe in
    *"cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features {{FEATURES}} --release --bin {{BIN}} "*"--bin oxart_jit"*"&& cargo build --release -p bloomery-gpu-gates --bin oxart_ptx && bash tools/ptx-scan.sh "*) rc=0 ;;
    *) rc=1 ;;
  esac
  out=$recipe
  ok "drift: the ptx-scan recipe's build lines are the ones this scan_command writes" $rc
  cases=$n
  echo "ptx-compare: self-test $([ "$bad" = 0 ] && echo ok || echo FAIL) ($cases cases, $bad failed)"
  [ "$bad" = 0 ]
}

# --- the command line
case ${1:-} in
  --self-test) self_test; exit ;;
esac
[ $# -ge 2 ] || die 64 "$USAGE"
REV=$1 BIN=$2
shift 2
FEATS=gpu BASEONLY=0 NOCACHE=0
while [ $# -gt 0 ]; do
  case $1 in
    --features) [ $# -ge 2 ] || die 64 "$USAGE"; FEATS=$2; shift 2 ;;
    --base-only) BASEONLY=1; shift ;;
    --no-cache) NOCACHE=1; shift ;;
    *) die 64 "$USAGE (got '$1': a feature goes through --features)" ;;
  esac
done
run_compare "$REV" "$BIN" "$FEATS" "$BASEONLY" "$NOCACHE"
