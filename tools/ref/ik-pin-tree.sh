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
# two libggml differ by their sources. Only the `ggml` target is built (the harnesses link libggml, never libllama),
# CPU only: no card is needed, and CUDA's template instances are most of an ik build.
#
# No lease: the build is niced and bounded, and tools/box.sh's guard has already waited out a sitting. Rate
# harnesses and timed runs never link a tree made here without their own lease.
#   BLOOMERY_BUILD_JOBS=<n>   the build's -j, default 24 (the box has 64 threads and other rounds run on it)
#   BLOOMERY_BUILD_BOUND=<s>  the build's timeout, default 1800
set -euo pipefail
# shellcheck source=tools/gate-bound.sh
source "$(dirname "${BASH_SOURCE[0]}")/../gate-bound.sh" || exit 70 # bounded, the one owner of a bounded child
pin_file=${1:?usage: ik-pin-tree.sh <pin file>}
SRC_TREE=/home/user/ik_llama.cpp
JOBS=${BLOOMERY_BUILD_JOBS:-24}
BOUND=${BLOOMERY_BUILD_BOUND:-1800}

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
dirty=$(printf '%s\n' "$dirty" | grep -v '^?? build/' || true)
[ -z "$dirty" ] || { echo "ik-pin-tree: $IK has changes outside its commit:" >&2; echo "$dirty" >&2; exit 1; }

as_user cmake -S "$IK" -B "$IK/build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF > /dev/null
cache_flags() {
  grep -E '^(GGML_[A-Z0-9_]*|LLAMA_[A-Z0-9_]*|CMAKE_BUILD_TYPE|CMAKE_(C|CXX)_(COMPILER|FLAGS|FLAGS_RELEASE)):[A-Z]+=' \
    "$1/build/CMakeCache.txt" | grep -v -e '_FOUND:' -e '^GGML_CUDA:BOOL=' | LC_ALL=C sort
}
differ=$(LC_ALL=C join -t= <(cache_flags "$SRC_TREE") <(cache_flags "$IK") | awk -F= '$2 != $3')
only=$(LC_ALL=C join -t= -v1 -v2 <(cache_flags "$SRC_TREE") <(cache_flags "$IK") || true)
[ -z "$only" ] || { echo "ik-pin-tree: cache options only one of $SRC_TREE and $IK has:"; echo "$only"; } >&2
if [ -n "$differ" ]; then
  echo "ik-pin-tree: $IK/build is configured differently from $SRC_TREE/build (option=$SRC_TREE=$IK):" >&2
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
