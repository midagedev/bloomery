#!/usr/bin/env bash
# Builds the oracle tree of the Qwen3.6 / Qwen3.8 image-input sets: llama.cpp mainline at 36a73916e, the commit whose
# mtmd, qwen35moe and qwen4exp (the fixes after 53ed051ce) the Qwen families' sets are dumped from. The Clef sets stay at
# /home/user/llama.cpp-mainline (53ed051ce); this tree sits beside it and moves nothing there.
#
#   BLOOMERY_REMOTE='~/repo/bloomery-<track>' tools/box.sh 'setsid -f bash tools/ref/qvis/build-lcpp-qvis.sh </dev/null >/dev/null 2>&1'
#   BLOOMERY_BOX_READONLY=1 tools/box.sh 'cat /root/lcpp-qvis/pid; tail -n 3 /root/lcpp-qvis/log; cat /root/lcpp-qvis/rc'
#
# A clone of the box's local clone /home/user/llama.cpp (objects shared, `git clone --shared`) at /home/user/llama.cpp-36a73916,
# the pin fetched from GitHub and checked out detached. Every git and build step runs as the trees' owner `user`. The
# flags are mainline's (/home/user/llama.cpp-mainline/build/CMakeCache.txt: GGML_CUDA=ON, CMAKE_CUDA_ARCHITECTURES=86,
# LLAMA_CURL=OFF, Release, nvcc from cuda-13.3), the targets the oracle links: llama, mtmd and llama-common (the chat
# templates the E sets run). The build runs under `taskset -c 0-31 nice -n 19` with 16 jobs and a 29-minute bound, and
# starts only while the timing lease is free (rc 75). A tree at another commit is refused (rc 65), never moved; a tree with
# tracked changes is refused too: the oracle is the commit's build. Writes pid, log and rc (last; absent = running or
# killed) under /root/lcpp-qvis.
set -uo pipefail
HERE=$(cd "${BASH_SOURCE[0]%/*}" && pwd)
TREE=/home/user/llama.cpp-36a73916
PIN=36a73916ee0cb3b457f356066afabd47cce68884
SRC=/home/user/llama.cpp
MAINLINE=/home/user/llama.cpp-mainline
NVCC=/usr/local/cuda-13.3/bin/nvcc
USER_PATH=/usr/local/cuda-13.3/bin:/usr/bin:/bin
DIR=/root/lcpp-qvis
as_user() { sudo -u user env HOME=/home/user "PATH=$USER_PATH" "$@"; }
ugit() { as_user git -c safe.directory="$TREE" -C "$TREE" "$@"; }

mkdir -p "$DIR"
rm -f "$DIR/rc"
echo $$ > "$DIR/pid"
exec > "$DIR/log" 2>&1
# shellcheck source=tools/ref/lease.sh
source "$HERE/../lease.sh"
fail() {
  echo "build-lcpp-qvis: $1"
  echo "${2:-2}" > "$DIR/rc"
  exit "${2:-2}"
}
# Every git and build step from /: the track directory holds a .git file naming the Mac's worktree, which git would find
# from there, and the user `user` cannot enter /root.
cd / || fail "cannot cd /"
echo "build-lcpp-qvis: start $(now) pid $$ tree $TREE pin $PIN"
lease_free || case $? in
  1) fail "the timing lease is held (a sitting is running); start again after it" 75 ;;
  *) fail "the timing lease cannot be tested (above); not starting" 70 ;;
esac
if [ -d "$TREE/.git" ] || [ -f "$TREE/.git" ]; then
  have=$(ugit rev-parse HEAD) || fail "$TREE is not a git tree"
  [ "$have" = "$PIN" ] || fail "$TREE is at $have, the pin is $PIN: move it by hand, knowing which sets came from which" 65
  [ -z "$(ugit status --porcelain --untracked-files=no)" ] || fail "$TREE has tracked changes; the oracle is the commit's build" 65
else
  as_user git clone --shared --no-checkout "$SRC" "$TREE" || fail "clone of $SRC failed"
  ugit fetch --no-tags https://github.com/ggml-org/llama.cpp "$PIN" || fail "fetch of $PIN failed"
  ugit checkout --detach "$PIN" || fail "checkout of $PIN failed"
fi
echo "build-lcpp-qvis: HEAD $(ugit log -1 --format='%H %cI %s')"
echo "build-lcpp-qvis: short $(ugit rev-parse --short=9 HEAD)"
echo "build-lcpp-qvis: mainline's command-line options ($MAINLINE/build/CMakeCache.txt):"
grep -E ':UNINITIALIZED=|^GGML_CUDA:BOOL=|^CMAKE_BUILD_TYPE:|^CMAKE_CUDA_COMPILER:' "$MAINLINE/build/CMakeCache.txt" | sed 's/^/    /'
t0=$(date +%s)
as_user cmake -S "$TREE" -B "$TREE/build" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DLLAMA_CURL=OFF \
  -DCMAKE_BUILD_TYPE=Release "-DCMAKE_CUDA_COMPILER=$NVCC" || fail "cmake configure failed"
timeout --kill-after=10 1740 taskset -c 0-31 nice -n 19 sudo -u user env HOME=/home/user "PATH=$USER_PATH" \
  cmake --build "$TREE/build" --config Release -j 16 --target llama mtmd llama-common
rc=$?
echo "build-lcpp-qvis: build rc $rc in $(($(date +%s) - t0)) s, end $(now)"
if [ "$rc" = 0 ]; then
  for l in libmtmd.so libllama.so libllama-common.so libggml.so libggml-base.so; do
    [ -e "$TREE/build/bin/$l" ] || { echo "build-lcpp-qvis: no $TREE/build/bin/$l"; rc=2; }
  done
fi
echo "$rc" > "$DIR/rc"
