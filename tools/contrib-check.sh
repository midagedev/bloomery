#!/usr/bin/env bash
# bloomery — a contributor's off-box check: the tests a machine with no workstation, no card and no model file can run.
# The design and what is left is docs/contrib/local-check.md; the result is a contributor's evidence in a pull request,
# never landing evidence (the maintainers' box record decides, AGENTS.md "Never run a gate where it cannot run").
#
#   tools/contrib-check.sh [--tier t0|t1] [--only REGEX]   (`just contrib-check`; default t1 on x86_64 Linux, else t0)
#       --only REGEX  runs just the units whose name matches the extended regex (a red unit again, say `--only '^fmt$'`)
#       --self-test   the status boundary on a scratch copy of this tree — no unit of the copy runs but the `fmt` one
#
#   T0  any OS with the pinned nightly:
#         fmt                    cargo fmt --all -- --check
#         test-<crate>           cargo test -p <crate>, for each crate `recipes.py pure-crates` lists
#         recipes-self-test      python3 tools/recipes.py --self-test
#         <tools/…/x.py>         each Python tool self-test tools/check-recipes.sh lists (read from its `selftests=(` array: the
#                                one owner of that list), run as that script runs it
#         check-<name>           every tools/check-*.sh. None is left out: each reads the tree and runs no cargo and no box
#                                command, and `recipes.py host-nightly` runs the same set on a Linux host (all ran green on a
#                                Mac). check-rustflags holds the two cargo files against each other, never this machine's CPU
#   T1  x86_64 or aarch64 Linux with no card: T0, then every unit of `python3 tools/recipes.py host-nightly` that T0 did
#       not already run by name (the check scripts and recipes-self-test run once), each `bash tools/gate.sh --release
#       --locked -p <crate>`. Needs what tools/nightly/install.sh lists: the CUDA headers (cuda.h, curand.h,
#       crt/host_config.h under CUDA_TOOLKIT_PATH), libclang for bindgen, python3-numpy and python3-pil.
#   T2  the card and fixture tier is not built: `--tier t2` prints that and exits 64.
#
# RUSTFLAGS: the repo's .cargo/config.toml targets Zen 3 (znver3) on x86_64 Linux, and a binary built for it may stop with an
# illegal instruction on another CPU (docs/BUILD.md, "The CPU flag"). On x86_64 Linux this script sets `-C target-cpu=native` for every
# cargo unit unless RUSTFLAGS is already set, in which case yours stands and the header line says so. The nightly's own
# machine sets x86-64-v3 (tools/nightly/run.sh); the unit runner is the same code (tools/nightly/units.sh).
#
# Output: a header (commit, host, toolchain, RUSTFLAGS), one line per unit `<name> rc=<N> <secs>s` (a red unit then names its
# failing tests and its first red lines, then prints that unit's log whole — the CI step keeps this stdout, and the file
# under target/ is not uploaded), a `DONE tier=… total=… red=… wall=…s` line; each unit's full log is also in
# target/contrib-check/<time>/<name>.log. A host without GNU `timeout` (macOS) runs units unbounded, and the header says so;
# the bound elsewhere is CONTRIB_UNIT_BOUND seconds a unit (default 1200), BLOOMERY_GATE_BOUND for tools/gate.sh.
# CONTRIB_INJECT_FAIL=1 adds one deliberately failing unit, `inject-fail`: the red path end to end.
#
# Exit: 0 every unit green; 1 a unit red; 64 usage (a tier this host cannot run included; `--tier t2`); 69 a prerequisite is
# missing (named, with how it is made); 70 a list this script reads (the self-tests, the pure crates, the nightly's units) is
# empty, or `--only` matches no unit.
set -uo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT" || exit 69
# shellcheck source=tools/nightly/units.sh
source "$ROOT/tools/nightly/units.sh"

say() { printf '%s\n' "$*" >&2; }

# self_test [RUN_ONLY_REGEX]: the status boundary on a scratch copy of this tree. The copy carries a readable HEAD,
# so a clean run passes the boundary (the header publishes the commit without +local, and the unit the extended
# regex names runs) and a corrupt-index run stops 70 naming the tree and Git's exit code before the header. The
# old unchecked substitution would turn the corrupt-index run green: the assertions keep the boundary. A run of
# the copy needs what one run of one unit needs, so the self-test needs this host's toolchain.
self_test() {
  local run_only=${1:-'^fmt$'} d out rc scratch fails=0
  command -v git > /dev/null || { say "contrib-check: --self-test: no git on PATH: nothing was run"; return 70; }
  d=$(mktemp -d "${TMPDIR:-/tmp}/contrib-selftest.XXXXXX") || { say "contrib-check: --self-test: cannot create scratch directory"; return 1; }
  cp -a "$ROOT" "$d/repo" || { say "contrib-check: --self-test: the copy of $ROOT failed"; return 1; }
  scratch=$d/repo
  rm -rf "$scratch/.git" "$scratch/target"
  if ! git -C "$scratch" init -q || ! git -C "$scratch" add -A ||
    ! git -C "$scratch" -c user.name=contrib-check -c user.email=contrib-check@local commit -q --no-gpg-sign -m tree; then
    say "contrib-check: --self-test: the scratch HEAD did not come up in $scratch"
    rm -rf "$d"
    return 1
  fi
  # the clean case: the boundary publishes the commit and the run goes on. contrib-check reads the status of
  # the directory it runs in, so the copy runs from its own root.
  out=$(cd "$scratch" && bash "$scratch/tools/contrib-check.sh" --only "$run_only" 2>&1) && rc=0 || rc=$?
  case $out in
    *"+local"*) say "contrib-check: self-test: the clean run's header marked the commit +local: $out"; fails=$((fails + 1)) ;;
    *"contrib-check tier="*) ;;
    *) say "contrib-check: self-test: the clean run printed no evidence header rc $rc: $out"; fails=$((fails + 1)) ;;
  esac
  [ "$rc" = 0 ] || { say "contrib-check: self-test: the clean run ended rc $rc: $out"; fails=$((fails + 1)); }
  # the corrupt-index case: the run stops 70 naming the tree and Git's exit code, and the header is never printed
  printf 'x' > "$scratch/.git/index"
  out=$(cd "$scratch" && bash "$scratch/tools/contrib-check.sh" --only "$run_only" 2>&1) || rc=$?
  rc=${rc:-0}
  out=$(cd "$scratch" && bash "$scratch/tools/contrib-check.sh" --only "$run_only" 2>&1) && rc=0 || rc=$?
  [ "$rc" = 70 ] || { say "contrib-check: self-test: the corrupt-index run ended rc $rc, want 70: $out"; fails=$((fails + 1)); }
  case $out in
    *"failed (git exit 128)"*) ;;
    *) say "contrib-check: self-test: the corrupt-index run does not name Git's exit 128: $out"; fails=$((fails + 1)) ;;
  esac
  case $out in
    *"contrib-check tier="*) say "contrib-check: self-test: the corrupt-index run printed the evidence header: $out"; fails=$((fails + 1)) ;;
  esac
  # the no-HEAD case (a source archive: no .git, so commit=?): the run keeps going — the ? path the checked
  # assignment cannot treat, and its unit runs
  rm -rf "$scratch/.git"
  out=$(cd "$scratch" && bash "$scratch/tools/contrib-check.sh" --only "$run_only" 2>&1) && rc=0 || rc=$?
  case $out in
    *"commit=?"*) ;;
    *) say "contrib-check: self-test: the archive run's header does not keep the ? path rc $rc: $out"; fails=$((fails + 1)) ;;
  esac
  [ "$rc" = 0 ] || { say "contrib-check: self-test: the archive run ended rc $rc: $out"; fails=$((fails + 1)); }
  rm -rf "$d"
  if [ "$fails" != 0 ]; then
    say "contrib-check: self-test $fails failed"
    return 1
  fi
  say "contrib-check: self-test ok"
  return 0
}

OS=$(uname -s)
ARCH=$(uname -m)
HOST_LINUX_X86=0
if [ "$OS" = Linux ] && [ "$ARCH" = x86_64 ]; then HOST_LINUX_X86=1; fi
HOST_TIER=0
if [ "$OS" = Linux ] && { [ "$ARCH" = x86_64 ] || [ "$ARCH" = aarch64 ]; }; then HOST_TIER=1; fi

TIER=
ONLY=
while [ $# -gt 0 ]; do
  case $1 in
    --tier)
      [ $# -ge 2 ] || { say "contrib-check: --tier takes t0, t1 or t2"; exit 64; }
      TIER=$2
      shift 2
      ;;
    --only)
      [ $# -ge 2 ] || { say "contrib-check: --only takes a regex of unit names"; exit 64; }
      ONLY=$2
      shift 2
      ;;
    --self-test)
      shift
      rc=0
      self_test "$@" || rc=$?
      exit "$rc"
      ;;
    --tier=*)
      TIER=${1#--tier=}
      shift
      ;;
    -h | --help)
      sed -n '2,/^set -uo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      say "contrib-check: unknown argument '$1' (usage: contrib-check.sh [--tier t0|t1] [--only REGEX])"
      exit 64
      ;;
  esac
done
if [ -z "$TIER" ]; then
  if [ "$HOST_LINUX_X86" = 1 ]; then TIER=t1; else TIER=t0; fi
fi
case $TIER in
  t0) ;;
  t1)
    if [ "$HOST_TIER" != 1 ]; then
      say "contrib-check: tier t1 runs the host tier, which needs x86_64 or aarch64 Linux (this is $OS $ARCH); use --tier t0"
      exit 64
    fi
    ;;
  t2)
    echo "T2 is not built yet, see docs/contrib/local-check.md"
    exit 64
    ;;
  *)
    say "contrib-check: no tier '$TIER' (t0, t1; t2 is not built)"
    exit 64
    ;;
esac

# ---- prerequisites: the toolchain rust-toolchain.toml pins, python3 ----
command -v python3 > /dev/null || { say "contrib-check: no python3 on PATH (the unit lists come from tools/recipes.py)"; exit 69; }
command -v rustup > /dev/null || { say "contrib-check: no rustup on PATH; install it from https://rustup.rs, then rerun"; exit 69; }
CHANNEL=$(sed -n 's/^channel *= *"\(.*\)"/\1/p' rust-toolchain.toml | head -1)
[ -n "$CHANNEL" ] || { say "contrib-check: no channel in rust-toolchain.toml"; exit 69; }
if ! rustup toolchain list | grep -q "^$CHANNEL"; then
  say "contrib-check: the pinned toolchain $CHANNEL is not installed: rustup toolchain install $CHANNEL --profile minimal -c rustfmt"
  exit 69
fi
export RUSTUP_TOOLCHAIN=$CHANNEL
# The pinned toolchain's own bin directory goes first on PATH: a cargo or rustc another package manager put earlier (Homebrew's)
# ignores RUSTUP_TOOLCHAIN and is another version (tools/mac-check.sh does the same).
TBIN=$(dirname "$(rustup which cargo --toolchain "$CHANNEL")")
[ -x "$TBIN/cargo" ] || { say "contrib-check: rustup found no cargo for $CHANNEL"; exit 69; }
PATH=$TBIN:$PATH
export PATH
if [ ! -x "$TBIN/cargo-fmt" ] || ! cargo fmt --version > /dev/null 2>&1; then
  say "contrib-check: rustfmt is missing from $CHANNEL: rustup component add rustfmt --toolchain $CHANNEL"
  exit 69
fi
if [ "$(command -v rustc)" != "$TBIN/rustc" ]; then
  say "contrib-check: rustc resolves to $(command -v rustc), not the pinned toolchain's $TBIN/rustc"
  exit 69
fi

RUSTFLAGS_NOTE="not set (the repo's .cargo/config.toml applies)"
if [ "$HOST_LINUX_X86" = 1 ]; then
  if [ -n "${RUSTFLAGS:-}" ]; then
    RUSTFLAGS_NOTE="$RUSTFLAGS (yours)"
  else
    export RUSTFLAGS='-C target-cpu=native'
    RUSTFLAGS_NOTE="$RUSTFLAGS"
  fi
fi

BOUND=${CONTRIB_UNIT_BOUND:-1200}
case $BOUND in '' | *[!0-9]* | 0) say "contrib-check: CONTRIB_UNIT_BOUND must be a positive whole number of seconds, got '$BOUND'"; exit 64 ;; esac
TIMEOUT_NOTE="timeout ${BOUND}s a unit"
if command -v timeout > /dev/null; then
  UNIT_TIMEOUT=timeout
elif command -v gtimeout > /dev/null; then
  UNIT_TIMEOUT=gtimeout
else
  UNIT_TIMEOUT=
  TIMEOUT_NOTE="no GNU timeout on this host: units run unbounded"
fi
unset BLOOMERY_BOX_ENV BLOOMERY_DATA

# ---- the units: lists from their owners, never copied ----
STAMP=$(date +%Y%m%d-%H%M%S)
LOGDIR=$ROOT/target/contrib-check/$STAMP
mkdir -p "$LOGDIR" || { say "contrib-check: cannot make $LOGDIR"; exit 69; }
PLAN=$LOGDIR/plan.tsv
: > "$PLAN"
add() { printf '%s\t%s\n' "$1" "$2" >> "$PLAN"; }

add fmt 'cargo fmt --all -- --check'

crates=$(python3 tools/recipes.py pure-crates --names) || { say "contrib-check: tools/recipes.py pure-crates failed"; exit 70; }
[ -n "$crates" ] || { say "contrib-check: recipes.py pure-crates lists no crate"; exit 70; }
for c in $crates; do add "test-$c" "cargo test -p $c"; done

add recipes-self-test 'python3 tools/recipes.py --self-test'

# check-recipes.sh's `selftests=(` array: lines `    "tools/x.py --self-test"`.
selftests=$(sed -n '/^  selftests=($/,/^  )$/p' tools/check-recipes.sh | sed -n 's/^ *"\(tools\/[^"]*\)"$/\1/p')
[ -n "$selftests" ] || { say "contrib-check: no Python self-test found in tools/check-recipes.sh's selftests=( array"; exit 70; }
while IFS= read -r t; do add "${t%% *}" "python3 $t"; done <<< "$selftests"

for c in tools/check-*.sh; do
  [ -f "$c" ] || { say "contrib-check: no tools/check-*.sh"; exit 70; }
  b=$(basename "$c" .sh)
  add "$b" "bash $c"
done

if [ "$TIER" = t1 ]; then
  nightly=$(python3 tools/recipes.py host-nightly) || { say "contrib-check: tools/recipes.py host-nightly failed"; exit 70; }
  [ -n "$nightly" ] || { say "contrib-check: recipes.py host-nightly lists no unit"; exit 70; }
  while IFS=$'\t' read -r name cmd; do
    # the check scripts and recipes-self-test are the same commands T0 ran under the same names
    if cut -f1 "$PLAN" | grep -qxF "$name"; then continue; fi
    add "$name" "$cmd"
  done <<< "$nightly"
fi
if [ "${CONTRIB_INJECT_FAIL:-}" = 1 ]; then
  add inject-fail "echo 'test injected::deliberate_failure ... FAILED'; echo 'test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out'; exit 101"
fi

if [ -n "$ONLY" ]; then
  : > "$PLAN.only"
  while IFS=$'\t' read -r name cmd; do
    [[ $name =~ $ONLY ]]
    case $? in
      0) printf '%s\t%s\n' "$name" "$cmd" >> "$PLAN.only" ;;
      2) say "contrib-check: --only '$ONLY' is not an extended regex"; exit 64 ;;
    esac
  done < "$PLAN"
  [ -s "$PLAN.only" ] || { say "contrib-check: --only '$ONLY' matches no unit"; exit 70; }
  mv "$PLAN.only" "$PLAN"
fi

# ---- the header: the block a pull request pastes ----
cpu=$(sed -n 's/^model name[^:]*: *//p' /proc/cpuinfo 2> /dev/null | head -1)
[ -n "$cpu" ] || cpu=$(sysctl -n machdep.cpu.brand_string 2> /dev/null || echo unknown)
commit=$(git rev-parse --short=8 HEAD 2> /dev/null || echo "?")
# The status is a checked assignment, as in tools/box.sh: a status that fails (a corrupt index) leaves the
# commit's dirt unreadable, and the header would attribute the runs to $commit without its +local — the run
# stops 70 naming the tree and Git's exit code before the header. No readable HEAD keeps the "?" path (a
# source archive has none; its header prints commit=? and its unit runs).
if [ "$commit" != "?" ]; then
  st=$(git status --porcelain --untracked-files=normal 2> /dev/null) && st_rc=0 || st_rc=$?
  [ "$st_rc" = 0 ] || { say "contrib-check: git status of $ROOT failed (git exit $st_rc)"; exit 70; }
  [ -z "$st" ] || commit=$commit+local
fi
echo "contrib-check tier=$TIER commit=$commit"
echo "host: $(uname -srm), $cpu"
echo "toolchain: $(rustc --version) ($CHANNEL); RUSTFLAGS: $RUSTFLAGS_NOTE; $TIMEOUT_NOTE"
echo "logs: $LOGDIR"

# ---- run ----
START=$SECONDS
TOTAL=0
RED=0
REDNAMES=
while IFS=$'\t' read -r name cmd; do
  TOTAL=$((TOTAL + 1))
  log=$LOGDIR/${name//\//_}.log
  run_unit "$cmd" "$log" "$BOUND" quiet
  printf '%s rc=%s %ss\n' "$name" "$UNIT_RC" "$UNIT_SECS"
  if [ "$UNIT_RC" != 0 ]; then
    RED=$((RED + 1))
    REDNAMES="$REDNAMES $name"
    unit_failure "$name" "$UNIT_RC" "$log" "$BOUND" | sed 's/^/  /'
    echo "  log: $log"
    # The whole log. CI keeps this step's stdout; target/contrib-check is not uploaded.
    sed 's/^/  | /' "$log"
  fi
done < "$PLAN"
echo "DONE tier=$TIER total=$TOTAL red=$RED wall=$((SECONDS - START))s${REDNAMES:+ red:$REDNAMES}"
[ "$RED" = 0 ]
