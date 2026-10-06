#!/usr/bin/env bash
# The V4.1 vision oracle's engine: smalinin's llama.cpp fork, branch my_build_deepseek41, at FORK_SHA — the
# llama.cpp tree that runs V4.1 with the reference's three image rules (src/models/deepseek41.cpp: `is_media`
# for an embd batch, no engram row there, `exp_probs_b_vl` in the expert pick; llama-kv-cells.h: an embd
# cell holds LLAMA_TOKEN_NULL, which blocks the engram lookback after it). visref_fork.cpp beside this file
# feeds it a reference set's ids and image rows (tools/ref/vision/visref.sh).
#
#   BLOOMERY_REMOTE='~/repo/bloomery-visref' tools/box.sh 'bash tools/ref/vision/build-fork.sh'
#
# Writes, under ROOT (VISREF_FORK_ROOT, default ~/repo/bloomery-visref-fork):
#   src/                          the fork at FORK_SHA, fetched from GitHub by its full id, detached
#   src/build/                    its CMake build: GGML_CUDA=ON, sm_86, Release, nvcc of cuda-13.3 (the flags
#                                 of the mainline trees the depth runners time); targets llama, llama-tokenize
#   target/release/visref_fork    visref_fork.cpp linked against src/build's libllama, libggml and libggml-base, the
#                                 path tools/gpu-gate.sh runs it by (cd ROOT first)
#   build.log                     the last build's own lines
# Idempotent: a tree at FORK_SHA is reused and cmake --build rebuilds only what changed; the harness is
# compiled whenever this script runs. A tree at another commit is refused (65): moving it would change the
# sets already dumped from it, which name FORK_SHA. Every step is bounded (VISREF_BUILD_BOUND
# seconds for the CMake build, default 1800). It starts only while the timing lease is free (75): nvcc beside
# a sitting would mark its rows [cpu-busy].
set -uo pipefail
FORK_URL=https://github.com/smalinin/llama.cpp
FORK_SHA=cfd8adcf6fb86ca523ad72fe21e8d2ac3860cd8e
HERE=$(cd "${BASH_SOURCE[0]%/*}" && pwd)
ROOT=${VISREF_FORK_ROOT:-$HOME/repo/bloomery-visref-fork}
BOUND=${VISREF_BUILD_BOUND:-1800}
NVCC=/usr/local/cuda-13.3/bin/nvcc
case $BOUND in
  '' | *[!0-9]* | 0) echo "build-fork.sh: VISREF_BUILD_BOUND is whole seconds from 1, got '$BOUND'" >&2; exit 64 ;;
esac
SRC=$ROOT/src
BIN=$ROOT/target/release
mkdir -p "$BIN"
exec > >(tee "$ROOT/build.log") 2>&1
fail() { echo "build-fork.sh: $1"; exit "${2:-2}"; }
# shellcheck source=tools/ref/lease-probe.sh
source "$HERE/../lease-probe.sh"
lease_free || case $? in
  1) fail "the timing lease is held (a sitting is running); start again after it" 75 ;;
  *) fail "the timing lease cannot be tested (above); not starting" 70 ;;
esac
echo "build-fork.sh: start $(now) fork $FORK_URL@$FORK_SHA into $ROOT"
if [ -d "$SRC/.git" ]; then
  have=$(git -C "$SRC" rev-parse HEAD) || fail "$SRC is not a git tree"
  [ "$have" = "$FORK_SHA" ] || fail "$SRC is at $have, not $FORK_SHA: remove it by hand, knowing which sets came from it" 65
  [ -z "$(git -C "$SRC" status --porcelain --untracked-files=no)" ] || fail "$SRC has local edits: the fork is the pin, not a copy of it" 65
else
  rm -rf "$SRC"
  git init -q "$SRC" || fail "git init $SRC failed"
  timeout --kill-after=10 600 git -C "$SRC" fetch -q --depth 1 "$FORK_URL" "$FORK_SHA" || fail "fetch of $FORK_SHA from $FORK_URL failed"
  git -C "$SRC" checkout -q --detach FETCH_HEAD || fail "checkout of $FORK_SHA failed"
fi
echo "build-fork.sh: HEAD $(git -C "$SRC" log -1 --format='%H %cI %s')"
timeout --kill-after=10 600 cmake -S "$SRC" -B "$SRC/build" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DCMAKE_BUILD_TYPE=Release "-DCMAKE_CUDA_COMPILER=$NVCC" -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF > "$ROOT/cmake.log" 2>&1 \
  || { tail -n 30 "$ROOT/cmake.log"; fail "cmake configure failed (whole log: $ROOT/cmake.log)"; }
t0=$SECONDS
timeout --kill-after=10 "$BOUND" taskset -c 0-31 nice -n 19 cmake --build "$SRC/build" --config Release -j 16 \
  --target llama llama-tokenize > "$ROOT/make.log" 2>&1
rc=$?
[ "$rc" = 0 ] || { tail -n 40 "$ROOT/make.log"; fail "cmake --build failed, rc $rc (whole log: $ROOT/make.log)" "$rc"; }
echo "build-fork.sh: llama and llama-tokenize built in $((SECONDS - t0)) s"
LIB=$SRC/build/bin
[ -f "$LIB/libllama.so" ] || fail "no $LIB/libllama.so after the build"
timeout --kill-after=10 300 g++ -std=c++17 -O2 -Wall -Wextra -o "$BIN/visref_fork.tmp" "$HERE/visref_fork.cpp" \
  -I"$SRC/include" -I"$SRC/ggml/include" -L"$LIB" -lllama -lggml -lggml-base -Wl,-rpath,"$LIB" \
  || fail "the harness does not compile"
mv "$BIN/visref_fork.tmp" "$BIN/visref_fork"
echo "fork $FORK_SHA" > "$BIN/visref_fork.build"
echo "build-fork.sh: $BIN/visref_fork (fork $FORK_SHA, md5 $(md5sum < "$BIN/visref_fork" | cut -c1-32)) $(now)"
