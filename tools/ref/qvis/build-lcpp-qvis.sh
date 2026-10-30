#!/usr/bin/env bash
# Builds the oracle tree of the Qwen3.6 / Qwen3.8 image-input sets: llama.cpp mainline at 36a73916e, the commit whose
# mtmd, qwen35moe and qwen4exp (the fixes after 53ed051ce) the Qwen families' sets are dumped from. The Clef sets stay at
# /home/user/llama.cpp-mainline (53ed051ce); this tree sits beside it and moves nothing there.
#
#   BLOOMERY_REMOTE='~/repo/bloomery-<track>' tools/box.sh 'setsid -f bash tools/ref/qvis/build-lcpp-qvis.sh </dev/null >/dev/null 2>&1'
#   BLOOMERY_BOX_READONLY=1 tools/box.sh 'cat /root/lcpp-qvis/pid; tail -n 3 /root/lcpp-qvis/log; cat /root/lcpp-qvis/rc'
#
# One script builds every mainline oracle tree; the pin, the tree, the sentinel directory, the targets and the card's
# architecture come by environment, and the defaults below are the Qwen tree on the box. MiniCPM-V 4.7's tree (`just
# build-mc47-lcpp`) is the same build at another pin, EmbeddingGemma 2's (`just build-emb2-lcpp`) at another pin with the
# server and the tokenizer among the targets:
#   LCPP_BUILD_PIN      the full 40-hex commit (default 36a73916ee0cb3b457f356066afabd47cce68884); anything else is refused (rc 64)
#   LCPP_BUILD_TREE     the tree (default /home/user/llama.cpp-36a73916)
#   LCPP_BUILD_DIR      where pid, log and rc go (default /root/lcpp-qvis); one directory a tree, so a build never overwrites
#                       another tree's rc
#   LCPP_BUILD_TARGETS  the targets, comma-separated (default llama,mtmd,llama-common)
#   LCPP_BUILD_CUDA_ARCH  CMAKE_CUDA_ARCHITECTURES (default 86, the box's two cards; 121a on a DGX Spark's GB10)
#   LCPP_BUILD_CUDA     the CUDA toolkit nvcc comes from (default /usr/local/cuda-13.3)
#   LCPP_BUILD_CPU_ONLY 1 builds the CPU backend alone (GGML_CUDA=OFF, GGML_NATIVE=ON, no nvcc); 0, the default, the CUDA
#                       build above; any other value is refused (rc 64). Mistral Large 4's oracle tree is the CPU-only one
#
# A clone of the box's local clone /home/user/llama.cpp (objects shared, `git clone --shared`) at the tree, the pin fetched from
# GitHub and checked out detached. Every git and build step runs as the trees' owner `user` when the script runs as root (the
# box), else as the caller (a contributor's machine under tools/box.sh's local mode). The
# flags are mainline's (/home/user/llama.cpp-mainline/build/CMakeCache.txt: GGML_CUDA=ON, CMAKE_CUDA_ARCHITECTURES=86,
# LLAMA_CURL=OFF, Release, nvcc from cuda-13.3), the targets the oracle links: llama, mtmd and llama-common (the chat
# templates the E sets run). The build runs under `taskset -c 0-31 nice -n 19` with 16 jobs and a 29-minute bound, and
# starts only while the timing lease is free (rc 75). A tree at another commit is refused (rc 65), never moved; a tree with
# tracked changes is refused too: the oracle is the commit's build. Writes pid, log and rc (last; absent = running or
# killed) under the sentinel directory.
set -uo pipefail
HERE=$(cd "${BASH_SOURCE[0]%/*}" && pwd)
TREE=${LCPP_BUILD_TREE:-/home/user/llama.cpp-36a73916}
PIN=${LCPP_BUILD_PIN:-36a73916ee0cb3b457f356066afabd47cce68884}
SRC=/home/user/llama.cpp
MAINLINE=/home/user/llama.cpp-mainline
CUDA=${LCPP_BUILD_CUDA:-/usr/local/cuda-13.3}
CUDA_ARCH=${LCPP_BUILD_CUDA_ARCH:-86}
CPU_ONLY=${LCPP_BUILD_CPU_ONLY:-0}
IFS=, read -ra TARGETS <<< "${LCPP_BUILD_TARGETS:-llama,mtmd,llama-common}"
NVCC=$CUDA/bin/nvcc
USER_PATH=$CUDA/bin:/usr/bin:/bin
DIR=${LCPP_BUILD_DIR:-/root/lcpp-qvis}
if [ "$(id -u)" = 0 ]; then AS_USER=(sudo -u user env HOME=/home/user "PATH=$USER_PATH"); else AS_USER=(env "PATH=$USER_PATH"); fi
as_user() { "${AS_USER[@]}" "$@"; }
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
[[ $PIN =~ ^[0-9a-f]{40}$ ]] || fail "LCPP_BUILD_PIN='$PIN' is not a full commit id (40 lower-case hex digits)" 64
case $CPU_ONLY in
  0 | 1) ;;
  *) fail "LCPP_BUILD_CPU_ONLY='$CPU_ONLY' is not 0 or 1" 64 ;;
esac
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
  dirty=$(ugit status --porcelain --untracked-files=no) || fail "cannot inspect tracked changes in $TREE"
  [ -z "$dirty" ] || fail "$TREE has tracked changes; the oracle is the commit's build" 65
else
  as_user git clone --shared --no-checkout "$SRC" "$TREE" || fail "clone of $SRC failed"
  ugit fetch --no-tags https://github.com/ggml-org/llama.cpp "$PIN" || fail "fetch of $PIN failed"
  ugit checkout --detach "$PIN" || fail "checkout of $PIN failed"
fi
echo "build-lcpp-qvis: HEAD $(ugit log -1 --format='%H %cI %s')"
echo "build-lcpp-qvis: short $(ugit rev-parse --short=9 HEAD)"
echo "build-lcpp-qvis: mainline's command-line options ($MAINLINE/build/CMakeCache.txt):"
grep -E ':UNINITIALIZED=|^GGML_CUDA:BOOL=|^CMAKE_BUILD_TYPE:|^CMAKE_CUDA_COMPILER:' "$MAINLINE/build/CMakeCache.txt" | sed 's/^/    /'
if [ "$CPU_ONLY" = 1 ]; then
  BACKEND_FLAGS=(-DGGML_CUDA=OFF -DGGML_NATIVE=ON)
  echo "build-lcpp-qvis: targets ${TARGETS[*]}, CPU backend only"
else
  BACKEND_FLAGS=(-DGGML_CUDA=ON "-DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH" "-DCMAKE_CUDA_COMPILER=$NVCC")
  echo "build-lcpp-qvis: targets ${TARGETS[*]}, CUDA architecture $CUDA_ARCH, $NVCC"
fi
t0=$(date +%s)
as_user cmake -S "$TREE" -B "$TREE/build" "${BACKEND_FLAGS[@]}" -DLLAMA_CURL=OFF \
  -DCMAKE_BUILD_TYPE=Release || fail "cmake configure failed"
rc=0
lease_bounded 1740 taskset -c 0-31 nice -n 19 "${AS_USER[@]}" \
  cmake --build "$TREE/build" --config Release -j 16 --target "${TARGETS[@]}" || rc=$?
echo "build-lcpp-qvis: build rc $rc in $(($(date +%s) - t0)) s, end $(now)"
if [ "$rc" = 0 ]; then
  for l in libmtmd.so libllama.so libllama-common.so libggml.so libggml-base.so; do
    [ -e "$TREE/build/bin/$l" ] || { echo "build-lcpp-qvis: no $TREE/build/bin/$l"; rc=2; }
  done
fi
echo "$rc" > "$DIR/rc"
