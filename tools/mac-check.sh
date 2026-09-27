#!/usr/bin/env bash
# bloomery — the static tier on the Mac. `check` and `lint` run the `check` and `lint` recipes' own
# box command (`tools/recipes.py box-command`: the recipe is the one owner of its flags) with
# `--target x86_64-unknown-linux-gnu` after the cargo subcommand: a cross check that type-checks and
# lints the Linux build and links no target code, so no x86_64 linker is needed. `fmt-check` runs
# `fmt-check`'s command and `fmt` the same command without its `-- --check` (it writes the tree), both
# with the pinned toolchain's rustfmt. Tests and gates never run here: their binaries are Linux ones,
# and the box judges them.
#
#   tools/mac-check.sh check|lint|fmt|fmt-check
#   tools/mac-check.sh --self-test   the derivation, the refusals, the ratchet, the target directory and
#                                    the toolchain and prerequisite checks against a fake HOME; runs no
#                                    cargo (check-recipes runs it)
#
# Exit: cargo's own code. `lint` also ends 1 when its `^warning:` count (the box's ruler, one per
# target a warning appears in) is above the ratchet, the last `**N on main` of AGENTS.md's Known
# state. 64: a mode or a box command this script does not run, or not on macOS. 69: a prerequisite
# is missing (named, with how it is made). 70: no ratchet in AGENTS.md.
#
# The toolchain is this script's: the channel rust-toolchain.toml pins, at
# $HOME/.rustup/toolchains/<channel>-<host>/bin, goes first on PATH, and cargo, cargo-clippy,
# clippy-driver, cargo-fmt and rustfmt must each resolve there — a missing one is named, never taken
# from Homebrew (whose cargo and rustfmt are other versions). RUSTC, when set, must be that rustc.
# `fmt` and `fmt-check` need nothing else. `check` and `lint` first source one file outside the
# repository, $HOME/opt/bloomery-mac-env.sh (a fake HOME moves it; the self-test does), which sets
#   CUDA_TOOLKIT_PATH   a directory whose include/cuda.h bindgen reads (headers only)
#   BINDGEN_EXTRA_CLANG_ARGS_x86_64_unknown_linux_gnu   `--sysroot=<dir>` with <dir>/usr/include/stdlib.h
#   LIBCLANG_PATH       the directory of libclang.dylib
# The build goes to the tree's own target/ (the workspace root's, as on the box), set here after the
# env file: cargo's metadata hash of a workspace member leaves its absolute path out, so two trees in
# one target directory reuse each other's rmeta files. An inherited CARGO_TARGET_DIR is overridden,
# with a line that names it.
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

say() { printf '%s\n' "$*" >&2; }

# The recipe whose box command a mode runs, and the cargo subcommand that command must be.
recipe_of() {
  case $1 in
    check) echo check ;;
    lint) echo lint ;;
    fmt | fmt-check) echo fmt-check ;;
    *) say "mac-check.sh: no mode '$1' (check, lint, fmt, fmt-check)"; return 64 ;;
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
  case $mode in check | lint) ;; *) [ "$miss" = 0 ] || return 69; return 0 ;; esac
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

# setup MODE TOOLCHAIN_BIN ROOT: the environment a run of MODE gets — the env file for check and lint,
# the pinned toolchain first on PATH, the prerequisites, and for check and lint the tree's own
# CARGO_TARGET_DIR, whatever the environment held (a line names what it replaces). Runs in the caller's shell.
setup() {
  case $1 in check | lint) load_env || return $? ;; esac
  export PATH="$2:$PATH"
  prereqs "$1" "$2" || return $?
  case $1 in
    check | lint)
      local own
      own=$(target_dir "$3")
      [ -z "${CARGO_TARGET_DIR:-}" ] || [ "$CARGO_TARGET_DIR" = "$own" ] ||
        say "mac-check: CARGO_TARGET_DIR=$CARGO_TARGET_DIR from the environment is overridden with the tree's own $own (a shared directory serves another tree's rmeta)"
      export CARGO_TARGET_DIR=$own
      ;;
  esac
}

# ratchet FILE: the last `**N on main` of the file (AGENTS.md's Known state), or 70.
ratchet() {
  local n
  n=$(grep -oE '\*\*[0-9]+ on main' "$1" | tail -1 | grep -oE '[0-9]+' || true)
  if [ -z "$n" ]; then
    say "mac-check.sh: no \`**N on main\` warning count in $1 to hold the lint to"
    return 70
  fi
  echo "$n"
}

# verdict COUNT RATCHET: the lint's last line; 1 when COUNT is above RATCHET.
verdict() {
  if [ "$1" -gt "$2" ]; then
    echo "mac-check: lint ^warning: $1, above the ratchet $2 in AGENTS.md — red"
    return 1
  elif [ "$1" -lt "$2" ]; then
    echo "mac-check: lint ^warning: $1, below the ratchet $2 in AGENTS.md — the recorded value can come down"
  else
    echo "mac-check: lint ^warning: $1 = the ratchet $2 in AGENTS.md"
  fi
}

self_test() {
  local fails=0 out t m cmd fake tc ch host a b
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
  printf '%s\n' 'x **175 on main `a`** (y), 169 through, **48 on main after `del2`** (z).' > "$t/a.md"
  [ "$(ratchet "$t/a.md")" = 48 ] || fail "the ratchet of a Known-state line is not its last bold count"
  printf '%s\n' 'no count here' > "$t/b.md"
  ratchet "$t/b.md" > /dev/null 2>&1 && fail "a file without a count gave a ratchet" || { [ $? = 70 ] || fail "no ratchet ended with another rc"; }
  ratchet "$HERE/AGENTS.md" | grep -qxE '[0-9]+' || fail "AGENTS.md gives no ratchet"
  verdict 49 48 > /dev/null && fail "49 warnings over a ratchet of 48 passed"
  verdict 48 48 > /dev/null || fail "48 warnings at a ratchet of 48 failed"
  verdict 47 48 | grep -q 'can come down' || fail "47 under 48 does not say the value can come down"
  rm -rf "$t"

  # setup, against a fake HOME with a fake Homebrew bin first on the inherited PATH: the complete
  # HOME passes every mode and resolves every tool to the pinned toolchain; each missing piece is named
  host=$(host_triple "$(uname -s)" "$(uname -m)" 2> /dev/null || echo aarch64-apple-darwin)
  fake=$(mktemp -d)
  tc=$fake/.rustup/toolchains/$ch-$host/bin
  mkdir -p "$tc" "$fake/.rustup/toolchains/$ch-$host/lib/rustlib/$TARGET/lib" "$fake/opt/cuda-13.3/include" \
    "$fake/opt/linux-sysroot/usr/include" "$fake/clt/lib" "$fake/brew/bin"
  for t in cargo rustc cargo-clippy clippy-driver cargo-fmt rustfmt; do
    printf '#!/bin/sh\nexit 0\n' > "$tc/$t"
    printf '#!/bin/sh\nexit 0\n' > "$fake/brew/bin/$t"
  done
  chmod +x "$tc"/* "$fake/brew/bin"/*
  touch "$fake/.rustup/toolchains/$ch-$host/lib/rustlib/$TARGET/lib/libstd-0.rlib" "$fake/opt/cuda-13.3/include/cuda.h" \
    "$fake/opt/linux-sysroot/usr/include/stdlib.h" "$fake/clt/lib/libclang.dylib"
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
  for m in check lint fmt fmt-check; do
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
  # each piece removed in turn: MODE|path|what the line names
  for t in "check|$fake/opt/cuda-13.3/include/cuda.h|missing cuda.h" \
    "check|$fake/opt/linux-sysroot/usr/include/stdlib.h|stdlib.h" \
    "lint|$fake/clt/lib/libclang.dylib|libclang.dylib" \
    "check|$fake/.rustup/toolchains/$ch-$host/lib/rustlib/$TARGET/lib/libstd-0.rlib|rustup target add" \
    "lint|$tc/cargo-clippy|cargo-clippy resolves to '$fake/brew/bin/cargo-clippy'" \
    "fmt|$tc/rustfmt|rustfmt resolves to '$fake/brew/bin/rustfmt'" \
    "fmt-check|$tc/cargo-fmt|cargo-fmt resolves to '$fake/brew/bin/cargo-fmt'" \
    "fmt|$tc/cargo|rustup toolchain install $ch" \
    "check|$fake/opt/bloomery-mac-env.sh|missing $fake/opt/bloomery-mac-env.sh"; do
    IFS='|' read -r m cmd why <<< "$t"
    mv "$cmd" "$cmd.away"
    out=$(probe "$m") && fail "$m passed without $cmd" || {
      [ $? = 69 ] || fail "$m without $cmd ended with another rc: $out"
      case $out in *"$why"*) ;; *) fail "$m without $cmd does not name '$why': $out" ;; esac
    }
    mv "$cmd.away" "$cmd"
  done
  # fmt needs no env file, CUDA, sysroot or libclang
  mv "$fake/opt/bloomery-mac-env.sh" "$fake/env.away"
  mv "$fake/opt/cuda-13.3/include/cuda.h" "$fake/cuda.h.away"
  for m in fmt fmt-check; do
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
  check | lint | fmt | fmt-check) [ $# = 1 ] || { say "mac-check.sh: one mode, got $#: $*"; exit 64; } ;;
  *) say "usage: tools/mac-check.sh check|lint|fmt|fmt-check | --self-test"; exit 64 ;;
esac
MODE=$1
HOST=$(host_triple "$(uname -s)" "$(uname -m)") || exit $?
CHANNEL=$(channel) || exit $?
TC=$HOME/.rustup/toolchains/$CHANNEL-$HOST/bin
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
RATCHET=$(ratchet "$HERE/AGENTS.md") || exit $?
COUNT=$(grep -c '^warning:' "$LOG" || true)
if [ "$rc" != 0 ]; then
  echo "mac-check: lint rc $rc (^warning: $COUNT, the ratchet $RATCHET in AGENTS.md); $(built)"
  exit "$rc"
fi
vrc=0
v=$(verdict "$COUNT" "$RATCHET") || vrc=$?
echo "$v; $(built)"
exit "$vrc"
