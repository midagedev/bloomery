#!/usr/bin/env bash
# bloomery — a contributor's off-box check: the tests a machine with no workstation, no card and no model file can run.
# The design and what is left is docs/contrib/local-check.md; the result is a contributor's evidence in a pull request,
# never landing evidence (the maintainers' box record decides, AGENTS.md "Never run a gate where it cannot run").
#
#   tools/contrib-check.sh [--tier t0|t1] [--only REGEX]   (`just contrib-check`; default t1 on x86_64 Linux, else t0)
#       --only REGEX  runs just the units whose name matches the extended regex (a red unit again, say `--only '^fmt$'`)
#
#   T0  any OS with the pinned toolchain:
#         fmt                    cargo fmt --all -- --check
#         lab-<name>-lock        cargo metadata --locked on each labs/<name>/Cargo.toml: a lab is a workspace of its own, so a
#                                root crate's new dependency leaves its lock stale until the change that added it refreshes it
#         lab-<name>-fmt         cargo fmt --check on that lab (the root fmt does not reach it). Building and running a lab is
#                                its contributor's verify script on their card (CONTRIBUTING.md)
#         test-<crate>           cargo test -p <crate>, for each crate `recipes.py pure-crates` lists
#         check-<name>           every tools/check-*.sh, as `tools/mac-static.sh --checks --list --no-base` lists them with
#                                their arguments (check-model-tax with no base revision: its tables against the code alone).
#                                None is left out: each reads the tree and runs no cargo and no box command, and
#                                `recipes.py host-nightly` runs the same set on a Linux host (all ran green on a Mac).
#                                check-rustflags holds the two cargo files against each other, never this machine's CPU.
#                                check-recipes runs `tools/recipes.py --self-test` and each Python tool self-test of its
#                                `selftests=(` array, so neither is a unit of its own here
#   T1  x86_64 or aarch64 Linux with no card: T0, then every unit of `python3 tools/recipes.py host-nightly` that T0 did
#       not already run (the check scripts run once, and recipes-self-test runs inside check-recipes), each
#       `bash tools/gate.sh --release --locked -p <crate>`. Needs what tools/nightly/install.sh lists: the CUDA headers
#       (cuda.h, curand.h, crt/host_config.h under CUDA_TOOLKIT_PATH), libclang for bindgen, python3-numpy and python3-pil.
#   T2  the card and fixture tier is not built: `--tier t2` prints that and exits 64.
#
# RUSTFLAGS: the repo's .cargo/config.toml targets Zen 3 (znver3) on x86_64 Linux, and a binary built for it may stop with an
# illegal instruction on another CPU (docs/BUILD.md, "The CPU flag"). On x86_64 Linux this script sets `-C target-cpu=native` for every
# cargo unit unless RUSTFLAGS is already set, in which case yours stands and the header line says so. The nightly's own
# machine sets x86-64-v3 (tools/nightly/run.sh); the unit runner is the same code (tools/nightly/units.sh).
#
# Output: a header (commit, host, CPU features, toolchain, RUSTFLAGS), one line per unit `<name> rc=<N> <secs>s` (a red unit then names its
# failing tests and its first red lines, then prints that unit's log whole — the CI step keeps this stdout, and the file
# under target/ is not uploaded), a `DONE tier=… total=… red=… wall=…s` line; each unit's full log is also in
# target/contrib-check/<time>.<unique>/<name>.log. A host without GNU `timeout` (macOS) runs units unbounded, and the header says so;
# the bound elsewhere is CONTRIB_UNIT_BOUND seconds a unit (default 1200), BLOOMERY_GATE_BOUND for tools/gate.sh.
# CONTRIB_INJECT_FAIL=1 adds one deliberately failing unit, `inject-fail`: the red path end to end.
#
# Exit: 0 every unit green; 1 a unit red; 64 usage (a tier this host cannot run included; `--tier t2`); 69 a prerequisite is
# missing (named, with how it is made); 70 a list this script reads (the pure crates, the nightly's units) is
# empty, `--only` matches no unit, or `git status` fails in the tree (whether it is clean is unknown, so no commit is named).
set -uo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT" || exit 69
# shellcheck source=tools/nightly/units.sh
source "$ROOT/tools/nightly/units.sh"
# shellcheck source=tools/gate-bound.sh
source "$ROOT/tools/gate-bound.sh" || exit 70 # run_unit's run_bounded

say() { printf '%s\n' "$*" >&2; }

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

# ---- the tree: the commit the evidence is attributed to ----
# `?` is the label of a tree with no readable HEAD (an unpacked archive), and the header says so. A tree whose HEAD reads but whose
# `git status` fails is not that: its state is unknown, and the run stops before it names a commit it cannot qualify.
tree_commit "$ROOT"
case $? in
  0 | 1) ;;
  *)
    say "contrib-check: git status failed in $ROOT (git exit $TREE_GIT_RC): whether the tree is clean is unknown, so no commit can be named; repair the repository and rerun"
    exit 70
    ;;
esac
commit=$TREE_COMMIT

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
mkdir -p "$ROOT/target/contrib-check" || { say "contrib-check: cannot make the log parent directory"; exit 69; }
LOGDIR=$(mktemp -d "$ROOT/target/contrib-check/$STAMP.XXXXXX") || { say "contrib-check: cannot make a private log directory"; exit 69; }
PLAN=$LOGDIR/plan.tsv
: > "$PLAN"
add() { printf '%s\t%s\n' "$1" "$2" >> "$PLAN"; }

add fmt 'cargo fmt --all -- --check'

for m in labs/*/Cargo.toml; do
  [ -f "$m" ] || continue
  lab=$(basename "$(dirname "$m")")
  add "lab-$lab-lock" "cargo metadata --locked --format-version 1 --manifest-path $m > /dev/null"
  add "lab-$lab-fmt" "cargo fmt --all --manifest-path $m -- --check"
done

crates=$(python3 tools/recipes.py pure-crates --names) || { say "contrib-check: tools/recipes.py pure-crates failed"; exit 70; }
[ -n "$crates" ] || { say "contrib-check: recipes.py pure-crates lists no crate"; exit 70; }
for c in $crates; do add "test-$c" "cargo test -p $c"; done

# The check scripts and the arguments each takes come from tools/mac-static.sh, the one owner of that discovery; --no-base because a
# contributor's checkout has no revision to hold check-model-tax's tables to (that unit's output says so).
checks=$(bash tools/mac-static.sh --checks --list --no-base) || { say "contrib-check: tools/mac-static.sh --checks --list failed"; exit 70; }
[ -n "$checks" ] || { say "contrib-check: tools/mac-static.sh --checks lists no check script"; exit 70; }
while IFS= read -r line; do
  b=check-${line%% *}
  add "$b" "bash tools/$b.sh${line#"${line%% *}"}"
done <<< "$checks"

if [ "$TIER" = t1 ]; then
  nightly=$(python3 tools/recipes.py host-nightly) || { say "contrib-check: tools/recipes.py host-nightly failed"; exit 70; }
  [ -n "$nightly" ] || { say "contrib-check: recipes.py host-nightly lists no unit"; exit 70; }
  while IFS=$'\t' read -r name cmd; do
    # the check scripts are the same commands T0 ran under the same names; check-recipes ran recipes-self-test
    if [ "$name" = recipes-self-test ] || cut -f1 "$PLAN" | grep -qxF "$name"; then continue; fi
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
cpu_features=
feature_source=unavailable
case "$OS" in
  Linux)
    feature_source='/proc/cpuinfo, first reported set'
    cpu_features=$(awk '/^(flags|Features)[[:space:]]*:/ { sub(/^[^:]*:[[:space:]]*/, ""); print; exit }' /proc/cpuinfo 2> /dev/null)
    ;;
  Darwin)
    feature_source='sysctl hw.optional=1'
    cpu_features=$(sysctl -a 2> /dev/null | LC_ALL=C sort | sed -n 's/^hw\.optional\.\([^:]*\): 1$/\1/p' | tr '\n' ' ')
    cpu_features=${cpu_features% }
    ;;
esac
[ -n "$cpu_features" ] || cpu_features=unavailable
echo "contrib-check tier=$TIER commit=$commit"
echo "host: $(uname -srm), $cpu"
echo "cpu features ($feature_source): $cpu_features"
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
