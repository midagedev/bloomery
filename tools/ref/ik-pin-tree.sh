#!/usr/bin/env bash
# Make an ik tree at one pinned commit with its libggml built, and print the tree's path on stdout (progress goes to
# stderr). Runs under tools/box.sh as root; the ik trees belong to `user`, so every git and build step in them runs
# as that user (the shared repo's objects stay user-owned: never chown).
#
#   bash tools/ref/ik-pin-tree.sh tools/ref/iq4xs-ik.pin
#
# The pin file is one line, the commit's 40 hex digits; its name without `-ik.pin` names the tree,
# /home/user/ik-<name>-<first 8 hex>, a worktree of the shared repo (/home/user/ik_llama.cpp), so a pin move makes a
# new tree and the old one stays. The shared tree is never checked out, reset or built: the references that read
# it keep their build. The commit is fetched into the shared repo's object store when it is not there.
#
# The tree is checked, not trusted, on every run: HEAD is the pin and nothing outside build/ is modified or untracked.
# The configure differs from the shared tree's by GGML_CUDA alone: after it, every GGML_* and LLAMA_* cache entry, the
# build type and the compilers' flags are compared with the shared tree's, and a difference stops the build, so the
# two libggml differ by their sources. A cache that cannot be read, or holds none of those entries, ends the run with
# 2: no comparison, no build. Only the `ggml` target is built (the harnesses link libggml, never libllama),
# CPU only: no card is needed, and CUDA's template instances are most of an ik build.
#
# No lease: the build is niced and bounded, and tools/box.sh's guard has already waited out a sitting. Rate
# harnesses and timed runs never link a tree made here without their own lease.
#   BLOOMERY_BUILD_BOUND=<s>  the build's timeout, default 1800
#   bash tools/ref/ik-pin-tree.sh --self-test   the dirty-tree filter and the cache comparison on fixtures, no box
set -euo pipefail
# shellcheck source=tools/gate-bound.sh
source "$(dirname "${BASH_SOURCE[0]}")/../gate-bound.sh" || exit 70 # bounded, the one owner of a bounded child
# shellcheck source=tools/grep-answer.sh
source "$(dirname "${BASH_SOURCE[0]}")/../grep-answer.sh" || exit 70 # grep's exit rule: 1 is an answer, 2 no verdict
SRC_TREE=/home/user/ik_llama.cpp
JOBS=24 # of the box's 64 threads: other rounds build beside it
BOUND=${BLOOMERY_BUILD_BOUND:-1800}

# dirty_outside_build STATUS: the lines of a `git status --porcelain` text that are not the build directory.
dirty_outside_build() { printf '%s\n' "$1" | grep_answer ik-pin-tree -v '^?? build/'; }
# cache_flags TREE: the cache entries that decide the two libggml builds' difference, sorted by option for join. A cache
# that cannot be read, or has none of them, is a nonzero status.
cache_flags() {
  grep -E '^(GGML_[A-Z0-9_]*|LLAMA_[A-Z0-9_]*|CMAKE_BUILD_TYPE|CMAKE_(C|CXX)_(COMPILER|FLAGS|FLAGS_RELEASE)):[A-Z]+=' \
    "$1/build/CMakeCache.txt" | grep -v -e '_FOUND:' -e '^GGML_CUDA:BOOL=' | LC_ALL=C sort -t= -k1,1
}
# cache_differ SHARED PINNED: the options whose values differ, one `option<TAB>shared value<TAB>pinned value` line each
# on stdout (none: the caches agree), and on stderr the options only one of the two has. A value is everything after
# the first `=`, so `-march=native` is one value. A cache cache_flags cannot read: 2, no verdict.
cache_differ() {
  local shared pinned only
  shared=$(cache_flags "$1") || { echo "ik-pin-tree: cannot read the cache options of the shared tree $1" >&2; return 2; }
  pinned=$(cache_flags "$2") || { echo "ik-pin-tree: cannot read the cache options of the pinned tree $2" >&2; return 2; }
  only=$(LC_ALL=C join -t= -v1 -v2 <(printf '%s\n' "$shared") <(printf '%s\n' "$pinned")) || return 2
  [ -z "$only" ] || { echo "ik-pin-tree: cache options only one of $1 and $2 has:"; echo "$only"; } >&2
  awk '
    function key(s) { return substr(s, 1, index(s, "=") - 1) }
    function val(s) { return substr(s, index(s, "=") + 1) }
    NR == FNR { a[key($0)] = val($0); next }
    { k = key($0); if ((k in a) && a[k] != val($0)) printf "%s\t%s\t%s\n", k, a[k], val($0) }
  ' <(printf '%s\n' "$shared") <(printf '%s\n' "$pinned")
}

if [ "${1:-}" = --self-test ]; then
  set +e
  t=$(mktemp -d "${TMPDIR:-/tmp}/ik-pin-tree-test.XXXXXX") || exit 70
  trap 'rm -rf "$t"' EXIT
  bad=0 tab=$'\t'
  check() { # NAME GOT WANT
    [ "$2" = "$3" ] && return 0
    echo "ik-pin-tree self-test FAIL $1: got '$2', want '$3'" >&2
    bad=$((bad + 1))
  }
  # tree NAME LINES…: $t/NAME/build/CMakeCache.txt with the lines (noise first, as a real cache has).
  tree() {
    local d=$t/$1
    shift
    mkdir -p "$d/build"
    { echo '// cache'; echo 'CMAKE_HOME_DIRECTORY:INTERNAL=/x'; echo 'GGML_AVX2_FOUND:BOOL=ON'; printf '%s\n' "$@"; } > "$d/build/CMakeCache.txt"
  }
  same=('CMAKE_BUILD_TYPE:STRING=Release' 'CMAKE_CXX_FLAGS:STRING=-march=native -O2' 'CMAKE_C_FLAGS:STRING=-g=1=2' 'GGML_AVX2:BOOL=ON')
  tree shared "${same[@]}" 'GGML_CUDA:BOOL=ON'
  tree same "${same[@]}" 'GGML_CUDA:BOOL=OFF'
  tree flags 'CMAKE_BUILD_TYPE:STRING=Release' 'CMAKE_CXX_FLAGS:STRING=-march=native -O3' 'CMAKE_C_FLAGS:STRING=-g=1=2' 'GGML_AVX2:BOOL=ON'
  tree more "${same[@]}" 'GGML_NEW:BOOL=ON'
  tree none
  # run NAME: cache_differ on shared and NAME; OUT, ERR and RC.
  run() {
    OUT=$(cache_differ "$t/shared" "$t/$1" 2> "$t/err")
    RC=$?
    ERR=$(cat "$t/err")
  }
  # Two caches that agree, to the `=` inside a value: no difference, no warning.
  run same
  check same-rc "$RC" 0
  check same-out "$OUT" ''
  check same-err "$ERR" ''
  run flags
  check flags-rc "$RC" 0
  check flags-out "$OUT" "CMAKE_CXX_FLAGS:STRING${tab}-march=native -O2${tab}-march=native -O3"
  # An option only the pinned tree has is a warning, not a difference.
  run more
  check more-rc "$RC" 0
  check more-out "$OUT" ''
  check more-err "$(printf '%s\n' "$ERR" | tail -n 1)" 'GGML_NEW:BOOL=ON'
  # A cache that cannot be read, or has no entries, is no verdict and never an empty difference.
  run absent
  check absent-rc "$RC" 2
  case $ERR in *"pinned tree $t/absent"*) ;; *) check absent-err "$ERR" "a line naming $t/absent" ;; esac
  cache_differ "$t/absent" "$t/same" 2> "$t/err" > /dev/null
  check shared-absent-rc "$?" 2
  case $(cat "$t/err") in *"shared tree $t/absent"*) ;; *) check shared-absent-err "$(cat "$t/err")" "a line naming $t/absent" ;; esac
  run none
  check none-rc "$RC" 2
  # The dirty filter lets the build directory through; a grep that fails is not an empty answer.
  check dirty-build "$(dirty_outside_build '?? build/')" ''
  check dirty-other "$(dirty_outside_build "$(printf '?? build/\n M ggml/a.c')")" ' M ggml/a.c'
  mkdir "$t/stub"
  printf '#!/bin/sh\necho "grep: stub" >&2\nexit 2\n' > "$t/stub/grep"
  chmod +x "$t/stub/grep"
  (PATH=$t/stub:$PATH dirty_outside_build ' M ggml/a.c' > /dev/null 2>&1)
  check dirty-grep-fails "$?" 2
  echo "ik-pin-tree self-test: $([ "$bad" = 0 ] && echo ok || echo FAIL) ($bad failures)"
  [ "$bad" = 0 ]
  exit
fi

pin_file=${1:?usage: ik-pin-tree.sh <pin file> | --self-test}

read -r PIN < "$pin_file" || { echo "ik-pin-tree: cannot read $pin_file" >&2; exit 64; }
[[ $PIN =~ ^[0-9a-f]{40}$ ]] || { echo "ik-pin-tree: $pin_file holds '$PIN', not a 40-digit lower-case commit" >&2; exit 64; }
name=$(basename "$pin_file" -ik.pin)
IK=/home/user/ik-$name-${PIN:0:8}

# From the user's home: box.sh runs this in root's remote directory, which the user cannot stat.
export USER_PATH=/usr/local/bin:/usr/bin:/bin
as_user() { (cd /home/user && sudo -u user env HOME=/home/user PATH="$USER_PATH" "$@"); }
ugit() { as_user git -C "$1" "${@:2}"; }

if [ ! -d "$IK" ]; then
  if ! ugit "$SRC_TREE" cat-file -e "$PIN^{commit}" 2> /dev/null; then
    echo "ik-pin-tree: fetching $PIN into $SRC_TREE" >&2
    ugit "$SRC_TREE" fetch --no-tags origin "$PIN" >&2
  fi
  ugit "$SRC_TREE" worktree add --detach "$IK" "$PIN" >&2
fi
head=$(ugit "$IK" rev-parse HEAD)
[ "$head" = "$PIN" ] || { echo "ik-pin-tree: $IK HEAD is $head, not the pin $PIN" >&2; exit 1; }
dirty=$(ugit "$IK" status --porcelain --untracked-files=all) ||
  { echo "ik-pin-tree: cannot inspect tracked changes in $IK" >&2; exit 2; }
dirty=$(dirty_outside_build "$dirty")
[ -z "$dirty" ] || { echo "ik-pin-tree: $IK has changes outside its commit:" >&2; echo "$dirty" >&2; exit 1; }

as_user cmake -S "$IK" -B "$IK/build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF > /dev/null
differ=$(cache_differ "$SRC_TREE" "$IK")
if [ -n "$differ" ]; then
  echo "ik-pin-tree: $IK/build is configured differently from $SRC_TREE/build (option, value in $SRC_TREE, value in $IK):" >&2
  echo "$differ" >&2
  exit 1
fi

t0=$(date +%s)
# shellcheck disable=SC2016 # the inner shell expands USER_PATH and its two arguments
bounded "$BOUND" bash -c 'cd /home/user && exec sudo -u user env HOME=/home/user PATH="$USER_PATH" nice -n 19 cmake --build "$0" -j "$1" --target ggml' \
  "$IK/build" "$JOBS" >&2
echo "ik-pin-tree: libggml of $IK ($name, ${PIN:0:8}) built in $(($(date +%s) - t0)) s" >&2
[ -f "$IK/build/ggml/src/libggml.so" ] || { echo "ik-pin-tree: $IK/build/ggml/src/libggml.so is not there after the build" >&2; exit 1; }
echo "$IK"
