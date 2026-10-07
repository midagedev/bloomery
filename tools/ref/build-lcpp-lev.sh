#!/usr/bin/env bash
# Builds lev's /v1/systemone oracle: llama.cpp mainline's server at the commit that added the decision
# server (LEV_LCPP_COMMIT in models/qwen35.sh, PR #29818), plus tools/ref/lev/lcpp-ids.patch, a logging-only
# patch (one `bloomery-ids` line a decision task: the task's prompt ids), because the server writes the
# rendered prompts' ids nowhere (its own `prompt token` debug block is commented out) and the ids are the
# first pin of lev's gate (tools/ref/lev_ref.py reads them, crates/decision/tests/lev.rs holds our encoder
# to them). The patch changes no value the server computes.
#
#   BLOOMERY_MODEL=qwen35 BLOOMERY_REMOTE='~/repo/bloomery-<track>' tools/box.sh 'setsid -f bash tools/ref/build-lcpp-lev.sh </dev/null >/dev/null 2>&1'
#   ssh ws 'cat /root/lcpp-lev/pid; tail -n 3 /root/lcpp-lev/log; cat /root/lcpp-lev/rc'
#
# A clone of the pin's tree on the box (/home/user/llama.cpp-a4cb4c61, its objects shared with
# `git clone --shared`) at LEV_LCPP, detached at LEV_LCPP_COMMIT, the patch applied, then built with the
# CMake flags of the mainline build the hidden-state oracle uses (/home/user/llama.cpp-mainline:
# GGML_CUDA=ON, CMAKE_CUDA_ARCHITECTURES=86, LLAMA_CURL=OFF, Release, nvcc from cuda-13.3), target
# llama-server. Every git and build step runs as the trees' owner `user`. The build runs under
# `taskset -c 0-31 nice -n 19` with 16 jobs and a 29-minute bound, and starts only while the timing lease
# is free (rc 75). An existing tree at another commit, or whose tracked changes are not exactly the patch,
# is refused (rc 65) rather than moved. The log ends with the one line lev_ref.py checks a dump against:
# `lev-oracle commit <sha> patch <sha256> server <md5 of llama-server>`. Writes pid, log and rc (last;
# absent = running or killed) under /root/lcpp-lev.
set -uo pipefail
HERE=$(cd "${BASH_SOURCE[0]%/*}" && pwd)
# shellcheck source=tools/ref/ref-paths.sh
source "$HERE/ref-paths.sh"
[ "$MODEL_NAME" = qwen35 ] || { echo "build-lcpp-lev: the profile is $MODEL_NAME; run under BLOOMERY_MODEL=qwen35" >&2; exit 64; }
: "${LEV_LCPP:?models/qwen35.sh names no LEV_LCPP}" "${LEV_LCPP_COMMIT:?models/qwen35.sh names no LEV_LCPP_COMMIT}"
PATCH=$HERE/lev/lcpp-ids.patch
TREE=$LEV_LCPP
SRC=/home/user/llama.cpp-a4cb4c61
MAINLINE=/home/user/llama.cpp-mainline
NVCC=/usr/local/cuda-13.3/bin/nvcc
USER_PATH=/usr/local/cuda-13.3/bin:/usr/bin:/bin
DIR=/root/lcpp-lev
as_user() { sudo -u user env HOME=/home/user "PATH=$USER_PATH" "$@"; }
ugit() { as_user git -c safe.directory="$TREE" -C "$TREE" "$@"; }

mkdir -p "$DIR"
rm -f "$DIR/rc"
echo $$ > "$DIR/pid"
exec > "$DIR/log" 2>&1
# shellcheck source=tools/ref/lease.sh
source "$HERE/lease.sh"
fail() {
  echo "build-lcpp-lev: $1"
  echo "${2:-2}" > "$DIR/rc"
  exit "${2:-2}"
}
# Every git and build step from /: the track directory holds a .git file naming the Mac's worktree,
# which git would find from there, and the user `user` cannot enter /root.
cd / || fail "cannot cd /"
echo "build-lcpp-lev: start $(now) pid $$ tree $TREE pin $LEV_LCPP_COMMIT"
lease_free || case $? in
  1) fail "the timing lease is held (a sitting is running); start again after it" 75 ;;
  *) fail "the timing lease cannot be tested (above); not starting" 70 ;;
esac
patch_sha=$(sha256sum "$PATCH" | cut -d' ' -f1)
echo "build-lcpp-lev: patch $PATCH sha256 $patch_sha"
# `user` cannot read /root: the patch goes through /tmp.
APPLY=/tmp/lcpp-lev-ids.$$.patch
cp "$PATCH" "$APPLY" && chmod a+r "$APPLY" || fail "cannot copy the patch to $APPLY"
trap 'rm -f "$APPLY"' EXIT
if [ -d "$TREE/.git" ] || [ -f "$TREE/.git" ]; then
  have=$(ugit rev-parse HEAD) || fail "$TREE is not a git tree"
  [ "$have" = "$LEV_LCPP_COMMIT" ] || fail "$TREE is at $have, the pin is $LEV_LCPP_COMMIT: move it by hand, knowing which dumps came from which" 65
  want=$(as_user git -C "$TREE" apply --numstat "$APPLY" | cut -f3 | sort | tr '\n' ' ')
  changed=$(ugit diff --name-only | sort | tr '\n' ' ')
  [ "$want" = "$changed" ] || fail "$TREE has tracked changes [$changed] other than the patch's [$want]" 65
else
  as_user git clone --shared --no-checkout "$SRC" "$TREE" || fail "clone of $SRC failed"
  ugit checkout --detach "$LEV_LCPP_COMMIT" || fail "checkout of $LEV_LCPP_COMMIT failed"
  as_user git -C "$TREE" apply "$APPLY" || fail "the patch does not apply to $LEV_LCPP_COMMIT"
fi
echo "build-lcpp-lev: HEAD $(ugit log -1 --format='%H %cI %s')"
echo "build-lcpp-lev: tracked changes: $(ugit diff --stat | tail -n 1)"
echo "build-lcpp-lev: mainline's command-line options ($MAINLINE/build/CMakeCache.txt):"
grep -E ':UNINITIALIZED=|^GGML_CUDA:BOOL=|^CMAKE_BUILD_TYPE:|^CMAKE_CUDA_COMPILER:' "$MAINLINE/build/CMakeCache.txt" | sed 's/^/    /'
t0=$(date +%s)
as_user cmake -S "$TREE" -B "$TREE/build" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DLLAMA_CURL=OFF \
  -DCMAKE_BUILD_TYPE=Release "-DCMAKE_CUDA_COMPILER=$NVCC" || fail "cmake configure failed"
timeout --kill-after=10 1740 taskset -c 0-31 nice -n 19 sudo -u user env HOME=/home/user "PATH=$USER_PATH" \
  cmake --build "$TREE/build" --config Release -j 16 --target llama-server
rc=$?
echo "build-lcpp-lev: build rc $rc in $(($(date +%s) - t0)) s, end $(now)"
if [ "$rc" = 0 ]; then
  echo "lev-oracle commit $LEV_LCPP_COMMIT patch $patch_sha server $(md5sum "$TREE/build/bin/llama-server" | cut -d' ' -f1)"
fi
echo "$rc" > "$DIR/rc"
