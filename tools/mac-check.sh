#!/usr/bin/env bash
# bloomery — the static tier on the Mac. `check` and `lint` run the `check` and `lint` recipes' own
# box command (`tools/recipes.py box-command`: the recipe is the one owner of its flags) with
# `--target x86_64-unknown-linux-gnu` after the cargo subcommand: a cross check that type-checks and
# lints the Linux build and links no target code, so no x86_64 linker is needed. `fmt-check` runs
# `fmt-check`'s command and `fmt` the same command without its `-- --check` (it writes the tree), both
# with the pinned toolchain's rustfmt. `test` runs `cargo test -p <crate>` natively (no --target: the
# host is aarch64-apple-darwin) for each pure crate, the list `tools/recipes.py pure-crates` derives
# from the crate graph and the sources (its one owner; the rule is in its docstring): a crate with no
# device root in its closure and no x86_64 or Linux-only code the Mac would build. Every other crate's
# tests and every gate never run here: their binaries are Linux ones, and the box judges them. A Mac
# result is development-loop evidence; landing evidence is the box's record. `combos` cross-checks, as
# `check` does, every build shape the recipes compile — one `cargo check -p <package> [--profile test]
# [--features …] <targets>` a shape, the list `tools/recipes.py combos` derives from every recipe's cargo
# calls (its one owner; the rule is its «build shapes» paragraph): the check recipe builds one feature
# set, the gate recipes others (`--features gpu` alone, a lib with no feature, a test build). It runs every
# shape, a red one too, names each red shape at the end, and prints the ones the list leaves out by name.
#
#   tools/mac-check.sh check|lint|fmt|fmt-check|test|combos [--base SPEC] [--ledger FILE]
#   tools/mac-check.sh --self-test   the derivation, the refusals, the ratchet, the target directory,
#                                    the test totals, the toolchain and prerequisite checks against a
#                                    fake HOME, and the disk floor against a fake df; runs no cargo
#                                    (check-recipes runs it)
#
# `combos --base SPEC --ledger FILE` is the round loop's scoped form (tools/mac-static.sh runs it):
# recipes.py prints only the shapes an input file of which changed since SPEC, or whose input key is
# not green in the ledger, each combo line carrying its 64-hex key; a shape that checks green is
# appended to the ledger as `<key>\tgreen\t<date>\t<tree>`, so the next run at the same inputs skips
# it. The full `combos` (no flags) stays the lead's landing form. --ledger without --base is a named 64.
#
# Exit: cargo's own code (`test`: the first crate's that is not 0; `combos`: the first red shape's).
# `lint` also ends 1 when its `^warning:` count (the box's ruler, one per target a warning appears in)
# is above the ratchet, the one number in tools/lint-ratchet.txt. 64: a mode or a box command this
# script does not run, a malformed `recipes.py combos` line, or not on macOS. 69: a prerequisite is missing (named, with how it is made), or the volume that holds target/ is below the disk floor (the `mac-check: disk:` line). 70: no ratchet in
# tools/lint-ratchet.txt (missing, or not exactly one number line), no pure crate to test, or no build shape (`combos`).
#
# The toolchain is this script's: the channel rust-toolchain.toml pins, at
# $HOME/.rustup/toolchains/<channel>-<host>/bin, goes first on PATH, and cargo, cargo-clippy,
# clippy-driver, cargo-fmt and rustfmt must each resolve there — a missing one is named, never taken
# from Homebrew (whose cargo and rustfmt are other versions). RUSTC, when set, must be that rustc.
# `fmt`, `fmt-check` and `test` need nothing else (`test`: the host std, which comes with the
# toolchain). `check`, `lint` and `combos` first source one file outside the
# repository, $HOME/opt/bloomery-mac-env.sh (a fake HOME moves it; the self-test does), which sets
#   CUDA_TOOLKIT_PATH   a directory whose include/cuda.h bindgen reads (headers only)
#   BINDGEN_EXTRA_CLANG_ARGS_x86_64_unknown_linux_gnu   `--sysroot=<dir>` with <dir>/usr/include/stdlib.h
#   LIBCLANG_PATH       the directory of libclang.dylib
# The build goes to the tree's own target/ (the workspace root's, as on the box), set here after the
# env file: cargo's metadata hash of a workspace member leaves its absolute path out, so two trees in
# one target directory reuse each other's rmeta files. An inherited CARGO_TARGET_DIR is overridden,
# with a line that names it.
#
# Disk floor. Before its first cargo command, every mode that writes target/ (check, lint, test,
# combos; fmt and fmt-check write nothing there and skip this) reads the free bytes of the volume
# that holds the tree's target/ with df -Pk: below MIN_FREE_GIB the mode stops with exit 69 and one
# `mac-check: disk:` line naming the free GiB, the floor and the mount point — a run must never
# start and die of ENOSPC halfway. A df that fails or does not parse is also 69, naming why, never a
# pass. BLOOMERY_MIN_FREE_GIB overrides the floor; an override that is not a positive integer is a
# named error (64).
# How each piece is made, once per Mac (the box is `ws`):
#   toolchain  rustup toolchain install <channel> --profile minimal -c clippy,rustfmt \
#                -t x86_64-unknown-linux-gnu      (the x86_64-linux std is what --target needs)
#   CUDA       mkdir -p ~/opt/cuda-13.3 && rsync -a ws:/usr/local/cuda-13.3/include ~/opt/cuda-13.3/
#   sysroot    mkdir -p ~/opt/linux-sysroot/usr && rsync -a ws:/usr/include ~/opt/linux-sysroot/usr/
#   libclang   xcode-select --install             (/Library/Developer/CommandLineTools/usr/lib)
#   env file   export CUDA_TOOLKIT_PATH=$HOME/opt/cuda-13.3 LIBCLANG_PATH=/Library/Developer/CommandLineTools/usr/lib
#              export BINDGEN_EXTRA_CLANG_ARGS_x86_64_unknown_linux_gnu="--sysroot=$HOME/opt/linux-sysroot -I$HOME/opt/linux-sysroot/usr/include/x86_64-linux-gnu"
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
TARGET=x86_64-unknown-linux-gnu
MIN_FREE_GIB=3 # the disk floor (the header's «Disk floor»): 2× the larger of one cold check's and one cold combos' growth of a tree's target/, rounded up to a whole GiB

say() { printf '%s\n' "$*" >&2; }

# The recipe whose box command a mode runs, and the cargo subcommand that command must be.
recipe_of() {
  case $1 in
    check) echo check ;;
    lint) echo lint ;;
    fmt | fmt-check) echo fmt-check ;;
    *) say "mac-check.sh: no mode '$1' (check, lint, fmt, fmt-check; test has no recipe)"; return 64 ;;
  esac
}
verb_of() {
  case $1 in
    check) echo check ;;
    lint) echo clippy ;;
    fmt | fmt-check) echo fmt ;;
    *) say "mac-check.sh: no mode '$1' (check, lint, fmt, fmt-check)"; return 64 ;;
  esac
}

# test_argv CRATE: the native test of one pure crate, one word a line — what the box's gate runner
# gives cargo for a whole package, without its bound and filters.
test_argv() {
  if [[ ! $1 =~ ^[a-z0-9][a-z0-9_-]*$ ]]; then
    say "mac-check.sh: '$1' is not a crate name"
    return 64
  fi
  printf '%s\n' cargo test -p "$1"
}

# totals LOG: the sum of the log's `test result:` lines — "P passed, F failed, I ignored in N test
# binaries" — or 1 when it has none.
totals() {
  awk '/^test result: /{ for (i = 1; i <= NF; i++) { if ($(i+1) ~ /^passed/) p += $i; if ($(i+1) ~ /^failed/) f += $i; if ($(i+1) ~ /^ignored/) g += $i }; n++ }
       END { if (!n) exit 1; printf "%d passed, %d failed, %d ignored in %d test binaries\n", p, f, g, n }' "$1"
}

# combo_lines [TABS]: recipes.py combos' output on stdin, checked line by line — `combo<TAB>label<TAB>command`
# (TABS=3, the --base form: a fourth field, the 64-hex input key), `skip<TAB>recipe<TAB>why`, one
# `total<TAB>counts` last — or 64 naming the first line that is not one, and 70 when it lists no shape.
combo_lines() {
  local tabs=${1:-2} line kind a b k n=0 no=0 total=0
  while IFS= read -r line; do
    no=$((no + 1))
    IFS=$'\t' read -r kind a b k <<< "$line"
    if [ "$total" != 0 ]; then
      say "mac-check.sh: recipes.py combos line $no follows its total line: $line"
      return 64
    fi
    case $kind in
      combo | skip)
        local want=2
        [ "$kind" = combo ] && want=$tabs
        if [ -z "$a" ] || [ -z "$b" ] || [ "$(printf '%s' "$line" | tr -cd '\t' | wc -c | tr -d ' ')" != "$want" ]; then
          say "mac-check.sh: recipes.py combos line $no is not $kind<TAB>…<TAB>…: $line"
          return 64
        fi
        if [ "$kind" = combo ]; then
          if [ "$tabs" = 3 ] && [[ ! $k =~ ^[0-9a-f]{64}$ ]]; then
            say "mac-check.sh: recipes.py combos line $no has no 64-hex input key: $line"
            return 64
          fi
          n=$((n + 1))
        fi
        ;;
      total) total=1 ;;
      *)
        say "mac-check.sh: recipes.py combos line $no is not a combo, skip or total line: $line"
        return 64
        ;;
    esac
  done
  if [ "$total" = 0 ]; then
    say "mac-check.sh: recipes.py combos printed no total line (cut short?)"
    return 64
  fi
  if [ "$n" = 0 ] && [ "$tabs" != 3 ]; then # the --base form may list no shape to run: every one skipped
    say "mac-check.sh: recipes.py combos lists no build shape"
    return 70
  fi
}

# derive MODE CMD: the cargo argv this script runs for the box command CMD, one word a line. CMD
# must be one plain `cargo <the mode's subcommand> …` — no shell syntax, quoting, variable or
# environment prefix, no --target of its own; `fmt` takes fmt-check's command, which must end
# `-- --check`, without those two words. Anything else is refused with 64, naming why.
derive() {
  local mode=$1 cmd=$2 verb w n
  local -a words
  verb=$(verb_of "$mode") || return 64
  if [[ -z ${cmd// /} ]]; then
    say "mac-check.sh: the $mode box command is empty"
    return 64
  fi
  if [[ ! $cmd =~ ^[A-Za-z0-9_,./=:+\ -]+$ ]]; then
    say "mac-check.sh: the $mode box command has shell syntax (a character outside [A-Za-z0-9_,./=:+ -]), which this script does not run: $cmd"
    return 64
  fi
  read -ra words <<< "$cmd"
  if [ "${words[0]}" != cargo ] || [ "${#words[@]}" -lt 2 ] || [ "${words[1]}" != "$verb" ]; then
    say "mac-check.sh: the $mode box command is not \`cargo $verb …\`: $cmd"
    return 64
  fi
  for w in "${words[@]}"; do
    case $w in
      --target | --target=*)
        say "mac-check.sh: the $mode box command names its own target ($w); the cross check adds --target $TARGET"
        return 64
        ;;
    esac
  done
  n=${#words[@]}
  if [ "$mode" = fmt ]; then
    if [ "$n" -lt 4 ] || [ "${words[n - 2]}" != -- ] || [ "${words[n - 1]}" != --check ]; then
      say "mac-check.sh: fmt-check's box command does not end \`-- --check\`, so fmt has no writing form of it: $cmd"
      return 64
    fi
    n=$((n - 2))
  fi
  printf '%s\n' cargo "$verb"
  case $mode in check | lint) printf '%s\n' --target "$TARGET" ;; esac
  [ "$n" -le 2 ] || printf '%s\n' "${words[@]:2:n-2}"
}

# The host part of a rustup toolchain name for `uname -s` `uname -m`.
host_triple() {
  case "$1 $2" in
    "Darwin arm64" | "Darwin aarch64") echo aarch64-apple-darwin ;;
    "Darwin x86_64") echo x86_64-apple-darwin ;;
    *) say "mac-check.sh: runs on macOS; this is $1 $2 — on the box the recipes are just check / lint / fmt-check"; return 64 ;;
  esac
}

# The channel rust-toolchain.toml pins.
channel() {
  local c
  c=$(sed -nE 's/^[[:space:]]*channel[[:space:]]*=[[:space:]]*"([^"]+)".*/\1/p' "$HERE/rust-toolchain.toml")
  if [ -z "$c" ] || [ "$(printf '%s\n' "$c" | wc -l | tr -d ' ')" != 1 ]; then
    say "mac-check.sh: no single channel = \"…\" line in $HERE/rust-toolchain.toml"
    return 70
  fi
  echo "$c"
}

# target_dir ROOT: the tree's own target directory.
target_dir() {
  echo "${1%/}/target"
}

# The disk floor, the header's «Disk floor». floor_gib resolves it (the constant, or
# BLOOMERY_MIN_FREE_GIB, which must be a positive integer of GiB or a named 64). df_able walks up to
# the deepest existing ancestor of a path — df errors on a path that does not exist, and the volume
# of the deepest existing ancestor is where mkdir -p creates the rest (a symlink or mount inside the
# missing span cannot change it). disk_ok reads the Available column of `df -Pk` (POSIX output,
# macOS and Linux alike) and prints one verdict line: 0 above the floor, 69 below it or when df
# fails or does not parse — named, never a pass.
floor_gib() {
  if [ -n "${BLOOMERY_MIN_FREE_GIB+x}" ]; then
    case $BLOOMERY_MIN_FREE_GIB in
      '' | 0 | *[!0-9]*) say "mac-check.sh: BLOOMERY_MIN_FREE_GIB='$BLOOMERY_MIN_FREE_GIB' is not a positive integer of GiB"; return 64 ;;
    esac
    echo $((10#$BLOOMERY_MIN_FREE_GIB))
  else
    echo "$MIN_FREE_GIB"
  fi
}

df_able() { # the deepest existing ancestor of $1
  local p=$1
  while [ ! -e "$p" ]; do
    case $p in
      / | '') echo /; return ;;
      *) p=${p%/*} ;;
    esac
  done
  echo "$p"
}

disk_ok() { # $1 a path whose volume must hold the floor free; one verdict line, 0 or 69
  local probe out rc=0 floor line
  floor=$(floor_gib) || return $?
  probe=$(df_able "$1")
  out=$(df -Pk "$probe" 2>&1) || rc=$?
  [ "$rc" = 0 ] || { say "mac-check: disk: df -Pk $probe failed (rc $rc): $(printf '%s\n' "$out" | tail -1)"; return 69; }
  rc=0
  line=$(printf '%s\n' "$out" | awk -v kib=$((10#$floor * 1048576)) -v floor="$floor" -v probe="$probe" '
    END {
      if (NR < 2 || $4 !~ /^[0-9]+$/) {
        printf "mac-check: disk: df -Pk %s output does not parse (the Available column): %s\n", probe, $0 > "/dev/stderr"
        exit 1
      }
      mount = $6; for (i = 7; i <= NF; i++) mount = mount " " $i
      if ($4 + 0 < kib) {
        printf "mac-check: disk: %.1f GiB free on %s (for %s), below the %s GiB floor — free space and run again\n", $4 / 1048576, mount, probe, floor > "/dev/stderr"
        exit 1
      }
      printf "mac-check: disk: %.1f GiB free on %s (for %s), above the %s GiB floor\n", $4 / 1048576, mount, probe, floor
    }') || rc=$?
  [ "$rc" = 0 ] || return 69
  say "$line"
}

# load_env: source $HOME/opt/bloomery-mac-env.sh, or 69 naming it.
load_env() {
  local f=$HOME/opt/bloomery-mac-env.sh
  if [ ! -f "$f" ]; then
    say "mac-check.sh: missing $f, the cross check's environment — the header of tools/mac-check.sh says what it exports"
    return 69
  fi
  # shellcheck disable=SC1090
  . "$f" || { say "mac-check.sh: sourcing $f failed"; return 69; }
}

# prereqs MODE TOOLCHAIN_BIN: every piece the mode needs, with the toolchain already first on PATH.
# Each missing one is named with how it is made; the return is 69 when any is.
prereqs() {
  local mode=$1 tc=$2 miss=0 t got rlib sysroot
  local tcname=${tc%/bin}
  tcname=${tcname##*/}
  if [ ! -x "$tc/cargo" ] || [ ! -x "$tc/rustc" ]; then
    say "mac-check.sh: missing the toolchain rust-toolchain.toml pins, $tc/{cargo,rustc}: rustup toolchain install ${tcname%-*-*-*} --profile minimal -c clippy,rustfmt -t $TARGET"
    miss=1
  fi
  local -a tools=(cargo)
  case $mode in
    lint) tools+=(cargo-clippy clippy-driver) ;;
    fmt | fmt-check) tools+=(cargo-fmt rustfmt) ;;
  esac
  for t in "${tools[@]}"; do
    got=$(command -v "$t" || true)
    if [ "$got" != "$tc/$t" ]; then
      say "mac-check.sh: $t resolves to '${got:-nothing}', not $tc/$t: the pinned toolchain lacks it (rustup component add --toolchain $tcname clippy rustfmt)"
      miss=1
    fi
  done
  if [ -n "${RUSTC:-}" ] && [ "$RUSTC" != "$tc/rustc" ]; then
    say "mac-check.sh: RUSTC is $RUSTC, not $tc/rustc, the toolchain rust-toolchain.toml pins: unset it or point it there"
    miss=1
  fi
  if [ "$mode" = test ]; then
    local host
    host=$(printf '%s\n' "$tcname" | awk -F- '{ print $(NF-2) "-" $(NF-1) "-" $NF }')
    rlib=$(ls "$tc/../lib/rustlib/$host/lib"/libstd-*.rlib 2> /dev/null | head -1 || true)
    if [ -z "$rlib" ]; then
      say "mac-check.sh: missing the $host std in $tcname (it comes with the toolchain): rustup toolchain install ${tcname%-*-*-*} --profile minimal -c clippy,rustfmt -t $TARGET"
      miss=1
    fi
  fi
  case $mode in check | lint | combos) ;; *) [ "$miss" = 0 ] || return 69; return 0 ;; esac
  rlib=$(ls "$tc/../lib/rustlib/$TARGET/lib"/libstd-*.rlib 2> /dev/null | head -1 || true)
  if [ -z "$rlib" ]; then
    say "mac-check.sh: missing the $TARGET std in $tcname: rustup target add --toolchain $tcname $TARGET"
    miss=1
  fi
  if [ -z "${CUDA_TOOLKIT_PATH:-}" ] || [ ! -f "$CUDA_TOOLKIT_PATH/include/cuda.h" ]; then
    say "mac-check.sh: missing cuda.h under CUDA_TOOLKIT_PATH='${CUDA_TOOLKIT_PATH:-}': mkdir -p ~/opt/cuda-13.3 && rsync -a ws:/usr/local/cuda-13.3/include ~/opt/cuda-13.3/"
    miss=1
  fi
  sysroot=$(printf '%s\n' "${BINDGEN_EXTRA_CLANG_ARGS_x86_64_unknown_linux_gnu:-}" | tr ' ' '\n' | sed -n 's/^--sysroot=//p' | head -1)
  if [ -z "$sysroot" ] || [ ! -f "$sysroot/usr/include/stdlib.h" ]; then
    say "mac-check.sh: missing the Linux sysroot's usr/include/stdlib.h (--sysroot='$sysroot' in BINDGEN_EXTRA_CLANG_ARGS_x86_64_unknown_linux_gnu): mkdir -p ~/opt/linux-sysroot/usr && rsync -a ws:/usr/include ~/opt/linux-sysroot/usr/"
    miss=1
  fi
  if [ -z "${LIBCLANG_PATH:-}" ] || [ ! -f "$LIBCLANG_PATH/libclang.dylib" ]; then
    say "mac-check.sh: missing libclang.dylib under LIBCLANG_PATH='${LIBCLANG_PATH:-}': xcode-select --install (it lands in /Library/Developer/CommandLineTools/usr/lib)"
    miss=1
  fi
  [ "$miss" = 0 ] || return 69
}

# setup MODE TOOLCHAIN_BIN ROOT: the environment a run of MODE gets — the env file for check, lint and
# combos, the pinned toolchain first on PATH, the prerequisites, the disk floor, and for check, lint,
# test and combos the tree's own CARGO_TARGET_DIR, whatever the environment held (a line names what
# it replaces). Runs in the caller's shell.
setup() {
  case $1 in check | lint | combos) load_env || return $? ;; esac
  export PATH="$2:$PATH"
  prereqs "$1" "$2" || return $?
  case $1 in
    check | lint | test | combos)
      local own
      own=$(target_dir "$3")
      [ -z "${CARGO_TARGET_DIR:-}" ] || [ "$CARGO_TARGET_DIR" = "$own" ] ||
        say "mac-check: CARGO_TARGET_DIR=$CARGO_TARGET_DIR from the environment is overridden with the tree's own $own (a shared directory serves another tree's rmeta)"
      export CARGO_TARGET_DIR=$own
      disk_ok "$own" || return $? # the disk floor: no mode that writes target/ starts below it
      ;;
  esac
}

# ratchet FILE: the one number line of FILE (tools/lint-ratchet.txt; `#` lines are its rule), or 70.
ratchet() {
  local lines
  if [ ! -f "$1" ]; then
    say "mac-check.sh: no ratchet file $1 to hold the lint to"
    return 70
  fi
  lines=$(grep -vE '^[[:space:]]*(#|$)' "$1" || true)
  if ! [[ $lines =~ ^[0-9]+$ ]]; then
    say "mac-check.sh: $1 must hold exactly one line with the warning count, a bare number; it holds: ${lines:-nothing}"
    return 70
  fi
  echo "$lines"
}

# verdict COUNT RATCHET: the lint's last line; 1 when COUNT is above RATCHET.
verdict() {
  if [ "$1" -gt "$2" ]; then
    echo "mac-check: lint ^warning: $1, above the ratchet $2 in tools/lint-ratchet.txt — red"
    return 1
  elif [ "$1" -lt "$2" ]; then
    echo "mac-check: lint ^warning: $1, below the ratchet $2 in tools/lint-ratchet.txt — the recorded value can come down"
  else
    echo "mac-check: lint ^warning: $1 = the ratchet $2 in tools/lint-ratchet.txt"
  fi
}

self_test() {
  local fails=0 out t m cmd fake tc ch host a b tab why drc
  fail() { say "mac-check self-test FAIL: $*"; fails=$((fails + 1)); }

  # the derivation from the real justfile: each mode's recipe gives `cargo <verb> …`, the cross
  # target goes right after the subcommand, and the recipe's own words follow unchanged
  for m in check lint fmt fmt-check; do
    cmd=$(python3 "$HERE/tools/recipes.py" box-command "$(recipe_of "$m")" 2> /dev/null) || { fail "recipes.py box-command $(recipe_of "$m") failed"; continue; }
    out=$(derive "$m" "$cmd" 2> /dev/null | tr '\n' ' ') || { fail "the $m recipe's box command refused: $cmd"; continue; }
    case $m in
      fmt-check) [ "$out" = "$cmd " ] || fail "fmt-check derives '$out' from '$cmd'" ;;
      fmt) [ "$out" = "${cmd% -- --check} " ] || fail "fmt derives '$out' from '$cmd'" ;;
      *)
        t=$(verb_of "$m")
        [ "$out" = "cargo $t --target $TARGET ${cmd#cargo "$t" } " ] || fail "$m derives '$out' from '$cmd'"
        ;;
    esac
  done
  out=$(derive check "cargo check -p a --features x,y/z" | tr '\n' ' ')
  [ "$out" = "cargo check --target $TARGET -p a --features x,y/z " ] || fail "a plain check derives '$out'"
  out=$(derive fmt "cargo fmt --all -- --check" | tr '\n' ' ')
  [ "$out" = "cargo fmt --all " ] || fail "fmt derives '$out' from a check command"
  # the refusals: 64, and the line names why
  for t in "check|cargo test -p x|not \`cargo check" \
    "check|cargo clippy --workspace|not \`cargo check" \
    "lint|cargo check --workspace|not \`cargo clippy" \
    "fmt-check|cargo check --workspace|not \`cargo fmt" \
    "fmt|cargo fmt --all|does not end" \
    "fmt|cargo fmt --all --check|does not end" \
    "check|cargo oxide build -p x|not \`cargo check" \
    "check|cargo check --workspace && echo x|shell syntax" \
    "check|cargo check \$F|shell syntax" \
    "check|cargo check 'x'|shell syntax" \
    "check|X=1 cargo check|not \`cargo check" \
    "check|cargo check --target aarch64-apple-darwin|its own target" \
    "check|cargo check --target=x|its own target" \
    "check|cargo|not \`cargo check" \
    "check| |empty" \
    "nope|cargo check|no mode"; do
    IFS='|' read -r m cmd why <<< "$t"
    out=$(derive "$m" "$cmd" 2>&1) && fail "'$m' accepted '$cmd'" || {
      [ $? = 64 ] || fail "'$m' '$cmd' refused with another rc"
      case $out in *"$why"*) ;; *) fail "'$m' '$cmd' refused without naming '$why': $out" ;; esac
    }
  done

  # test: one plain `cargo test -p <crate>` a crate, a name that is not one refused; the totals of a log
  out=$(test_argv bloomery-sampler | tr '\n' ' ')
  [ "$out" = "cargo test -p bloomery-sampler " ] || fail "test derives '$out' for bloomery-sampler"
  for t in "a b" "--workspace" "" "x;y" "-p"; do
    test_argv "$t" > /dev/null 2>&1 && fail "test accepted the crate name '$t'" || { [ $? = 64 ] || fail "test refused '$t' with another rc"; }
  done
  t=$(mktemp -d)
  printf '%s\n' '     Running unittests src/lib.rs' 'test result: ok. 19 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.01s' \
    'test a ... FAILED' 'test result: FAILED. 3 passed; 2 failed; 39 ignored; 0 measured; 0 filtered out; finished in 0.20s' > "$t/log"
  out=$(totals "$t/log") || fail "totals of a log with two result lines failed"
  [ "$out" = "22 passed, 2 failed, 40 ignored in 2 test binaries" ] || fail "totals read '$out'"
  printf '%s\n' 'error[E0425]: cannot find value `RUSAGE_THREAD` in crate `libc`' > "$t/log"
  totals "$t/log" > /dev/null && fail "a log with no result line gave totals"
  rm -rf "$t"
  # the crates test runs: the pure list recipes.py derives, each one a name test_argv takes
  out=$(python3 "$HERE/tools/recipes.py" pure-crates --names 2> /dev/null) || fail "recipes.py pure-crates --names failed"
  [ -n "$out" ] || fail "recipes.py pure-crates --names lists no crate"
  for t in $out; do test_argv "$t" > /dev/null || fail "pure crate '$t' is not a name test takes"; done

  # combos: the real list is well formed and each shape's command is one check derives; a malformed list
  # is refused by name
  tab=$'\t'
  out=$(python3 "$HERE/tools/recipes.py" combos 2> /dev/null) || fail "recipes.py combos failed"
  combo_lines <<< "$out" || fail "recipes.py combos' own output refused"
  while IFS=$'\t' read -r t m cmd; do
    [ "$t" = combo ] || continue
    derive check "$cmd" > /dev/null 2>&1 || fail "the shape '$m' has a command check does not derive: $cmd"
  done <<< "$out"
  case $out in *"combo${tab}bloomery-gpu-gates build [gpu]: "*) ;; *) fail "recipes.py combos lists no bloomery-gpu-gates build shape with gpu alone" ;; esac
  case $out in *"combo${tab}bloomery-gpu-gates test [no features]: "*) ;; *) fail "recipes.py combos lists no bloomery-gpu-gates test shape with no feature" ;; esac
  a="combo${tab}p build [x]: 1 targets${tab}cargo check -p p --bin b"
  b="total${tab}1 commands"
  for t in "$a|nope${tab}x${tab}y|not a combo, skip or total" \
    "$a|combo${tab}x|not combo<TAB>" \
    "$a|skip${tab}${tab}why|not skip<TAB>" \
    "$a|combo${tab}x${tab}y${tab}z|not combo<TAB>" \
    "$a|$b|$a|follows its total" \
    "$a|no total line" \
    "skip${tab}r${tab}why|$b|lists no build shape"; do
    why=${t##*|}
    printf '%s\n' "${t%|*}" | tr '|' '\n' > "${TMPDIR:-/tmp}/mac-check-combos.$$"
    out=$(combo_lines < "${TMPDIR:-/tmp}/mac-check-combos.$$" 2>&1) && fail "combo_lines accepted '${t%|*}'" || {
      case $out in *"$why"*) ;; *) fail "combo_lines refused '${t%|*}' without naming '$why': $out" ;; esac
    }
  done
  rm -f "${TMPDIR:-/tmp}/mac-check-combos.$$"
  printf '%s\n' "$a" "$b" | combo_lines || fail "combo_lines refused a list of one shape and its total"
  # the --base form: combo lines carry a 64-hex key (3 tabs), skip lines keep 2, a list whose every
  # shape skips is accepted, and a combo line without a key is refused by name
  k=0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
  printf '%s\n' "combo${tab}p build [x]: 1 targets${tab}cargo check -p p --bin b${tab}$k" "skip${tab}r${tab}why" "total${tab}0 of 1 commands to run" | combo_lines 3 ||
    fail "combo_lines 3 refused a scoped list of one keyed shape, one skip and its total"
  printf '%s\n' "skip${tab}r${tab}why" "total${tab}0 of 1 commands to run" | combo_lines 3 || fail "combo_lines 3 refused a scoped list with no shape to run"
  for t in "combo${tab}p build [x]: 1 targets${tab}cargo check -p p --bin b${tab}nothex|no 64-hex input key" \
    "combo${tab}p build [x]: 1 targets${tab}cargo check -p p --bin b${tab}${k}a|no 64-hex input key" \
    "combo${tab}p build [x]: 1 targets${tab}cargo check -p p|not combo<TAB>"; do
    why=${t##*|}
    out=$(printf '%s\n' "${t%|*}" "total${tab}1 commands" | combo_lines 3 2>&1) && fail "combo_lines 3 accepted '${t%|*}'" || {
      case $out in *"$why"*) ;; *) fail "combo_lines 3 refused '${t%|*}' without naming '$why': $out" ;; esac
    }
  done
  # the combos block's own refusals, before any cargo: 64, named
  for t in "--bogus|not '--bogus'" "--base|needs a value" "--base main --ledger|needs a value" "--ledger /tmp/x|--ledger reads the keys"; do
    args=${t%%|*}
    why=${t##*|}
    out=$("$HERE/tools/mac-check.sh" combos $args < /dev/null 2>&1) && fail "combos $args accepted" || {
      [ $? = 64 ] || fail "combos $args refused with another rc"
      case $out in *"$why"*) ;; *) fail "combos $args refused without naming '$why': $out" ;; esac
    }
  done

  # the host triple and the channel
  [ "$(host_triple Darwin arm64)" = aarch64-apple-darwin ] || fail "Darwin arm64 is not aarch64-apple-darwin"
  host_triple Linux x86_64 > /dev/null 2>&1 && fail "Linux accepted as the Mac" || { [ $? = 64 ] || fail "Linux refused with another rc"; }
  ch=$(channel) || fail "no channel read from rust-toolchain.toml"

  # one target directory a tree: two roots never share one
  a=$(target_dir /x/bloomery)
  b=$(target_dir /x/bloomery-foo/)
  [ "$a" = /x/bloomery/target ] && [ "$b" = /x/bloomery-foo/target ] || fail "two roots give target directories '$a' and '$b'"

  # the ratchet and the verdict
  t=$(mktemp -d)
  printf '%s\n' '48' '# the rule' > "$t/a"
  [ "$(ratchet "$t/a")" = 48 ] || fail "the ratchet of a file with one number line is not that number"
  for m in 'missing' '' '# only the rule' '48 on main' '48|49' 'x48'; do
    rm -f "$t/b"
    [ "$m" = missing ] || printf '%s\n' "$m" | tr '|' '\n' > "$t/b"
    ratchet "$t/b" > /dev/null 2>&1 && fail "a ratchet file '$m' gave a ratchet" || { [ $? = 70 ] || fail "the ratchet file '$m' ended with another rc"; }
  done
  if a=$(ratchet "$HERE/tools/lint-ratchet.txt"); then
    verdict $((a + 1)) "$a" > /dev/null && fail "$((a + 1)) warnings over the file's ratchet of $a passed"
  else
    fail "tools/lint-ratchet.txt gives no ratchet"
  fi
  verdict 49 48 > /dev/null && fail "49 warnings over a ratchet of 48 passed"
  verdict 48 48 > /dev/null || fail "48 warnings at a ratchet of 48 failed"
  verdict 47 48 | grep -q 'can come down' || fail "47 under 48 does not say the value can come down"
  rm -rf "$t"

  # setup, against a fake HOME with a fake Homebrew bin first on the inherited PATH: the complete
  # HOME passes every mode and resolves every tool to the pinned toolchain; each missing piece is named
  host=$(host_triple "$(uname -s)" "$(uname -m)" 2> /dev/null || echo aarch64-apple-darwin)
  fake=$(mktemp -d)
  tc=$fake/.rustup/toolchains/$ch-$host/bin
  mkdir -p "$tc" "$fake/.rustup/toolchains/$ch-$host/lib/rustlib/$TARGET/lib" "$fake/.rustup/toolchains/$ch-$host/lib/rustlib/$host/lib" "$fake/opt/cuda-13.3/include" \
    "$fake/opt/linux-sysroot/usr/include" "$fake/clt/lib" "$fake/brew/bin"
  for t in cargo rustc cargo-clippy clippy-driver cargo-fmt rustfmt; do
    printf '#!/bin/sh\nexit 0\n' > "$tc/$t"
    printf '#!/bin/sh\nexit 0\n' > "$fake/brew/bin/$t"
  done
  chmod +x "$tc"/* "$fake/brew/bin"/*
  touch "$fake/.rustup/toolchains/$ch-$host/lib/rustlib/$TARGET/lib/libstd-0.rlib" "$fake/.rustup/toolchains/$ch-$host/lib/rustlib/$host/lib/libstd-0.rlib" "$fake/opt/cuda-13.3/include/cuda.h" \
    "$fake/opt/linux-sysroot/usr/include/stdlib.h" "$fake/clt/lib/libclang.dylib"
  # a fake df first on the probe's PATH (brew/bin is first there): the disk floor's verdict must not
  # depend on the host's real disk. One POSIX data line whose Available column is DISK_KB KiB on
  # /fake/mount; DISK_RC makes df fail, DISK_BAD output that does not parse.
  cat > "$fake/brew/bin/df" << 'DF'
#!/bin/sh
if [ -n "${DISK_RC:-}" ]; then echo "df: fake failure" >&2; exit "$DISK_RC"; fi
if [ -n "${DISK_BAD:-}" ]; then printf 'Filesystem 1024-blocks Used Available Capacity Mounted on\ngarbage line\n'; exit 0; fi
printf 'Filesystem 1024-blocks Used Available Capacity Mounted on\n/dev/fake 100000000 1000 %s 1%% /fake/mount\n' "${DISK_KB:-999999999}"
DF
  chmod +x "$fake/brew/bin/df"
  write_env() { # RUSTC
    cat > "$fake/opt/bloomery-mac-env.sh" << EOF
export PATH=\$HOME/brew/bin:\$PATH${1:+ RUSTC=$1}
export CUDA_TOOLKIT_PATH=\$HOME/opt/cuda-13.3 LIBCLANG_PATH=\$HOME/clt/lib
export BINDGEN_EXTRA_CLANG_ARGS_x86_64_unknown_linux_gnu="--sysroot=\$HOME/opt/linux-sysroot -I\$HOME/opt/linux-sysroot/usr/include/x86_64-linux-gnu"
export CARGO_TARGET_DIR=\$HOME/tgt
EOF
  }
  probe() { # MODE -> setup's rc and lines, then what each tool resolves to and the target directory
    (HOME=$fake && PATH=$fake/brew/bin:/usr/bin:/bin && unset CARGO_TARGET_DIR RUSTC && setup "$1" "$tc" /x/bloomery-foo &&
      for t in cargo cargo-clippy rustfmt; do echo "resolved $t $(command -v "$t")"; done && echo "target ${CARGO_TARGET_DIR:-none}") 2>&1
  }
  write_env
  for m in check lint fmt fmt-check test combos; do
    out=$(probe "$m") || fail "the complete fake HOME fails $m: $out"
    for t in cargo cargo-clippy rustfmt; do
      case $out in *"resolved $t $tc/$t"*) ;; *) fail "$m with Homebrew first on PATH resolves $t elsewhere: $out" ;; esac
    done
  done
  # the env file exports a shared directory: check still gets the tree's own, and says what it replaced
  out=$(probe check) || true
  case $out in *"target /x/bloomery-foo/target"*) ;; *) fail "an inherited shared CARGO_TARGET_DIR is not replaced by the tree's own: $out" ;; esac
  case $out in *"CARGO_TARGET_DIR=$fake/tgt from the environment is overridden"*) ;; *) fail "the override of an inherited CARGO_TARGET_DIR is not named: $out" ;; esac
  out=$(probe fmt) || true
  case $out in *"target none"*) ;; *) fail "fmt sets a target directory: $out" ;; esac
  out=$(probe test) || true
  case $out in *"target /x/bloomery-foo/target"*) ;; *) fail "test does not build in the tree's own target directory: $out" ;; esac
  # the disk floor: the modes that write target/ print the verdict line and stop below the floor
  # with 69 and the named line; fmt and fmt-check, which write nothing there, do not ask; a failing
  # or unparsable df is a named 69; the override is honored when positive, a named 64 when not
  for m in check lint test combos; do
    out=$(probe "$m") || fail "the complete fake HOME fails $m: $out"
    case $out in *"mac-check: disk: "*" GiB free on /fake/mount "*" floor"*) ;; *) fail "$m prints no disk verdict line: $out" ;; esac
  done
  for m in fmt fmt-check; do
    out=$(probe "$m") || fail "the complete fake HOME fails $m: $out"
    case $out in *"mac-check: disk:"*) fail "$m asks for the disk floor it cannot need: $out" ;; esac
  done
  drc=0; out=$(DISK_KB=1024 probe check) || drc=$?
  [ "$drc" = 69 ] || fail "check below the floor ended rc $drc, not 69: $out"
  case $out in *"mac-check: disk: "*" below the "*" GiB floor"*) ;; *) fail "check below the floor names no disk line: $out" ;; esac
  drc=0; out=$(DISK_RC=1 probe lint) || drc=$?
  [ "$drc" = 69 ] || fail "lint with a failing df ended rc $drc, not 69: $out"
  case $out in *"failed (rc 1): df: fake failure"*) ;; *) fail "a failing df is not named: $out" ;; esac
  drc=0; out=$(DISK_BAD=1 probe combos) || drc=$?
  [ "$drc" = 69 ] || fail "combos with unparsable df output ended rc $drc, not 69: $out"
  case $out in *"output does not parse"*) ;; *) fail "unparsable df output is not named: $out" ;; esac
  drc=0; out=$(BLOOMERY_MIN_FREE_GIB=abc probe check) || drc=$?
  [ "$drc" = 64 ] || fail "BLOOMERY_MIN_FREE_GIB=abc ended rc $drc, not 64: $out"
  case $out in *"is not a positive integer of GiB"*) ;; *) fail "a bad override is not named: $out" ;; esac
  drc=0; out=$(BLOOMERY_MIN_FREE_GIB=0 probe test) || drc=$?
  [ "$drc" = 64 ] || fail "BLOOMERY_MIN_FREE_GIB=0 ended rc $drc, not 64: $out"
  drc=0; out=$(BLOOMERY_MIN_FREE_GIB=999999 probe check) || drc=$?
  [ "$drc" = 69 ] || fail "an override above the free space ended rc $drc, not 69: $out"
  case $out in *"below the 999999 GiB floor"*) ;; *) fail "the override's floor is not the one named: $out" ;; esac
  # each piece removed in turn: MODE|path|what the line names
  for t in "check|$fake/opt/cuda-13.3/include/cuda.h|missing cuda.h" \
    "check|$fake/opt/linux-sysroot/usr/include/stdlib.h|stdlib.h" \
    "lint|$fake/clt/lib/libclang.dylib|libclang.dylib" \
    "check|$fake/.rustup/toolchains/$ch-$host/lib/rustlib/$TARGET/lib/libstd-0.rlib|rustup target add" \
    "lint|$tc/cargo-clippy|cargo-clippy resolves to '$fake/brew/bin/cargo-clippy'" \
    "fmt|$tc/rustfmt|rustfmt resolves to '$fake/brew/bin/rustfmt'" \
    "fmt-check|$tc/cargo-fmt|cargo-fmt resolves to '$fake/brew/bin/cargo-fmt'" \
    "fmt|$tc/cargo|rustup toolchain install $ch" \
    "test|$fake/.rustup/toolchains/$ch-$host/lib/rustlib/$host/lib/libstd-0.rlib|missing the $host std" \
    "test|$tc/cargo|rustup toolchain install $ch" \
    "check|$fake/opt/bloomery-mac-env.sh|missing $fake/opt/bloomery-mac-env.sh" \
    "combos|$fake/opt/bloomery-mac-env.sh|missing $fake/opt/bloomery-mac-env.sh" \
    "combos|$fake/opt/cuda-13.3/include/cuda.h|missing cuda.h"; do
    IFS='|' read -r m cmd why <<< "$t"
    mv "$cmd" "$cmd.away"
    out=$(probe "$m") && fail "$m passed without $cmd" || {
      [ $? = 69 ] || fail "$m without $cmd ended with another rc: $out"
      case $out in *"$why"*) ;; *) fail "$m without $cmd does not name '$why': $out" ;; esac
    }
    mv "$cmd.away" "$cmd"
  done
  # fmt and test need no env file, CUDA, sysroot or libclang
  mv "$fake/opt/bloomery-mac-env.sh" "$fake/env.away"
  mv "$fake/opt/cuda-13.3/include/cuda.h" "$fake/cuda.h.away"
  for m in fmt fmt-check test; do
    out=$(probe "$m") || fail "$m asks for the env file or cuda.h: $out"
  done
  mv "$fake/env.away" "$fake/opt/bloomery-mac-env.sh"
  mv "$fake/cuda.h.away" "$fake/opt/cuda-13.3/include/cuda.h"
  write_env /opt/other/rustc
  out=$(probe check) && fail "another RUSTC passed" || {
    case $out in *"RUSTC is /opt/other/rustc"*) ;; *) fail "another RUSTC is not named: $out" ;; esac
  }
  rm -rf "$fake"

  if [ "$fails" != 0 ]; then
    say "mac-check: self-test $fails failed"
    return 1
  fi
  echo "mac-check: self-test ok"
}

case ${1:-} in
  --self-test) [ $# = 1 ] || { say "mac-check.sh: --self-test takes nothing"; exit 64; }; self_test; exit $? ;;
  check | lint | fmt | fmt-check | test) [ $# = 1 ] || { say "mac-check.sh: one mode, got $#: $*"; exit 64; } ;;
  combos) : ;; # its own block takes --base SPEC and --ledger FILE
  *) say "usage: tools/mac-check.sh check|lint|fmt|fmt-check|test|combos | --self-test"; exit 64 ;;
esac
MODE=$1
[ "$MODE" != combos ] || shift
HOST=$(host_triple "$(uname -s)" "$(uname -m)") || exit $?
CHANNEL=$(channel) || exit $?
TC=$HOME/.rustup/toolchains/$CHANNEL-$HOST/bin

if [ "$MODE" = test ]; then
  CRATES=$(python3 "$HERE/tools/recipes.py" pure-crates --names) || exit $?
  if [ -z "$CRATES" ]; then
    say "mac-check.sh: tools/recipes.py pure-crates selects no crate; nothing to test natively"
    exit 70
  fi
  setup test "$TC" "$HERE" || exit $?
  mkdir -p "$HERE/target"
  LOG=$HERE/target/mac-check-test.log
  LOCK0=$(md5 -q "$HERE/Cargo.lock")
  HEAD="mac-check: cargo test -p <crate>, natively on $HOST, for the pure crates of tools/recipes.py pure-crates ($("$TC/rustc" --version 2> /dev/null || echo "$TC/rustc"); CARGO_TARGET_DIR=$CARGO_TARGET_DIR)"
  say "$HEAD; log $LOG"
  echo "$HEAD" > "$LOG"
  rc=0
  ran=()
  for c in $CRATES; do
    A=$(test_argv "$c") || exit 64
    ARGV=()
    while IFS= read -r w; do ARGV+=("$w"); done <<< "$A"
    echo "mac-check: ${ARGV[*]}" >> "$LOG"
    crc=0
    (cd "$HERE" && "$TC/cargo" "${ARGV[@]:1}") >> "$LOG" 2>&1 || crc=$?
    ran+=("$c=$crc")
    [ "$rc" != 0 ] || rc=$crc
  done
  cat "$LOG"
  [ "$(md5 -q "$HERE/Cargo.lock")" = "$LOCK0" ] || say "mac-check: cargo rewrote Cargo.lock in this tree (a dependency edit's lock refresh): commit it with the edit"
  echo "mac-check: test rc $rc; ${#ran[@]} crates (crate=rc): ${ran[*]}; $(totals "$LOG" || echo "no test result line")"
  exit "$rc"
fi
if [ "$MODE" = combos ]; then
  BASE= LEDGER=
  while [ $# -gt 0 ]; do
    case $1 in
      --base) [ $# -ge 2 ] || { say "mac-check.sh: --base needs a value"; exit 64; }; BASE=$2; shift 2 ;;
      --ledger) [ $# -ge 2 ] || { say "mac-check.sh: --ledger needs a value"; exit 64; }; LEDGER=$2; shift 2 ;;
      *) say "mac-check.sh: combos takes --base SPEC and --ledger FILE, not '$1'"; exit 64 ;;
    esac
  done
  [ -z "$LEDGER" ] || [ -n "$BASE" ] || { say "mac-check.sh: --ledger reads the keys the --base form prints; give --base too"; exit 64; }
  RC_ARGS=(combos)
  [ -z "$BASE" ] || RC_ARGS+=(--base "$BASE")
  [ -z "$LEDGER" ] || RC_ARGS+=(--ledger "$LEDGER")
  LIST=$(python3 "$HERE/tools/recipes.py" "${RC_ARGS[@]}") || exit $?
  combo_lines $([ -n "$BASE" ] && echo 3) <<< "$LIST" || exit $?
  setup combos "$TC" "$HERE" || exit $?
  mkdir -p "$HERE/target"
  LOG=$HERE/target/mac-check-combos.log
  LOCK0=$(md5 -q "$HERE/Cargo.lock")
  HEAD="mac-check: cargo check --target $TARGET for each build shape of tools/recipes.py combos${BASE:+ scoped to --base $BASE}${LEDGER:+, ledger $LEDGER} ($("$TC/rustc" --version 2> /dev/null || echo "$TC/rustc"); CARGO_TARGET_DIR=$CARGO_TARGET_DIR)"
  say "$HEAD; log $LOG"
  echo "$HEAD" > "$LOG"
  rc=0
  n=0
  TOTAL=
  reds=()
  skips_scoped=0
  while IFS=$'\t' read -r -u 3 kind label cmd key; do
    case $kind in
      skip)
        if [ -n "$BASE" ]; then
          skips_scoped=$((skips_scoped + 1))
          echo "mac-check: skipped (scoped): $label — $cmd" >> "$LOG"
        else
          echo "mac-check: left out by name: $label — $cmd" >> "$LOG"
        fi
        ;;
      total) TOTAL=$label ;;
      combo)
        DERIVED=$(derive check "$cmd") || exit 64
        ARGV=()
        while IFS= read -r w; do ARGV+=("$w"); done <<< "$DERIVED"
        echo "mac-check: shape $label: ${ARGV[*]}" >> "$LOG"
        t0=$SECONDS
        crc=0
        (cd "$HERE" && "$TC/cargo" "${ARGV[@]:1}") >> "$LOG" 2>&1 || crc=$?
        echo "mac-check: shape rc $crc in $((SECONDS - t0)) s: $label" >> "$LOG"
        n=$((n + 1))
        if [ "$crc" != 0 ]; then
          reds+=("$label: ${ARGV[*]}")
          [ "$rc" != 0 ] || rc=$crc
        elif [ -n "$LEDGER" ]; then
          mkdir -p "$(dirname "$LEDGER")"
          printf '%s\tgreen\t%s\t%s\n' "$key" "$(date '+%Y-%m-%d %H:%M')" "${HERE##*/}" >> "$LEDGER"
        fi
        ;;
    esac
  done 3<<< "$LIST"
  cat "$LOG"
  [ "$(md5 -q "$HERE/Cargo.lock")" = "$LOCK0" ] || say "mac-check: cargo rewrote Cargo.lock in this tree (a dependency edit's lock refresh): commit it with the edit"
  for r in "${reds[@]+"${reds[@]}"}"; do echo "mac-check: red shape $r"; done
  echo "mac-check: combos rc $rc; $n shapes run, ${#reds[@]} red${BASE:+, $skips_scoped skipped (scoped)}, in $SECONDS s (recipes.py combos: $TOTAL)"
  exit "$rc"
fi
RECIPE=$(recipe_of "$MODE")
CMD=$(python3 "$HERE/tools/recipes.py" box-command "$RECIPE") || exit $?
DERIVED=$(derive "$MODE" "$CMD") || exit $?
ARGV=()
while IFS= read -r w; do ARGV+=("$w"); done <<< "$DERIVED"
setup "$MODE" "$TC" "$HERE" || exit $?

mkdir -p "$HERE/target"
LOG=$HERE/target/mac-check-$MODE.log
LOCK0=$(md5 -q "$HERE/Cargo.lock")
ORIGIN="the $RECIPE recipe's box command"
[ "$MODE" != fmt ] || ORIGIN="$ORIGIN without -- --check"
HEAD="mac-check: ${ARGV[*]}  ($ORIGIN; $("$TC/rustc" --version 2> /dev/null || echo "$TC/rustc"), $("$TC/rustfmt" --version 2> /dev/null || echo rustfmt); CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-none})"
# check and lint: where the build went and how many of the tree's own crates it compiled
built() { echo "target ${CARGO_TARGET_DIR:-none}, $(grep -cE '^ *(Checking|Compiling) bloomery-' "$LOG" || true) bloomery crates checked or compiled"; }
say "$HEAD; log $LOG"
echo "$HEAD" > "$LOG"
rc=0
(cd "$HERE" && "$TC/cargo" "${ARGV[@]:1}") >> "$LOG" 2>&1 || rc=$?
cat "$LOG"
[ "$(md5 -q "$HERE/Cargo.lock")" = "$LOCK0" ] || say "mac-check: cargo rewrote Cargo.lock in this tree (a dependency edit's lock refresh): commit it with the edit"
case $MODE in
  check) echo "mac-check: check rc $rc; $(built)"; exit "$rc" ;;
  fmt | fmt-check) echo "mac-check: $MODE rc $rc"; exit "$rc" ;;
esac
RATCHET=$(ratchet "$HERE/tools/lint-ratchet.txt") || exit $?
COUNT=$(grep -c '^warning:' "$LOG" || true)
if [ "$rc" != 0 ]; then
  echo "mac-check: lint rc $rc (^warning: $COUNT, the ratchet $RATCHET in tools/lint-ratchet.txt); $(built)"
  exit "$rc"
fi
vrc=0
v=$(verdict "$COUNT" "$RATCHET") || vrc=$?
echo "$v; $(built)"
exit "$vrc"
