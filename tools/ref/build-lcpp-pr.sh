#!/usr/bin/env bash
# Builds a ggml-org/llama.cpp pull request's head in a tree of its own, with the CMake flags of the
# mainline build the depth runners already time (/home/user/llama.cpp-mainline), for a model mainline
# does not build: the GLM-5.3-Flash arms of tools/ref/depth-glm5next.sh run PR #27752's and PR
# #27754's (models/glm5next.sh LCPP27752, LCPP27754). Then, with `smoke`, a greedy completion of the
# oracle prompt through the branch, the correctness check before any of its rows is timed: a branch
# that runs another model is not a reference (V4.1's PR #28696 once did).
#
#   BLOOMERY_REMOTE='~/repo/bloomery-<track>' tools/box.sh 'setsid -f bash tools/ref/build-lcpp-pr.sh build 27752 </dev/null >/dev/null 2>&1'
#   ssh ws 'cat /root/lcpp-pr27752/pid; tail -n 3 /root/lcpp-pr27752/log; cat /root/lcpp-pr27752/rc'
#   BLOOMERY_MODEL=glm5next BLOOMERY_CARD=a6000 tools/box.sh 'bash tools/ref/build-lcpp-pr.sh smoke 27752'
#
# build <N>: a clone of the local mainline clone /home/user/llama.cpp (its objects shared, `git clone
# --shared`) at /home/user/llama.cpp-pr<N>, the PR head fetched from GitHub (`pull/<N>/head`) and
# checked out detached; the commit goes in the log and the tree's HEAD. Every git and build step runs
# as the trees' owner `user` (a root-owned object in a shared store blocks that user later). The
# flags are mainline's: its CMakeCache.txt shows GGML_CUDA=ON, CMAKE_CUDA_ARCHITECTURES=86,
# LLAMA_CURL=OFF, Release, nvcc from cuda-13.3; every other option at its default, which the log
# prints beside mainline's for the options that differ. Targets: llama-bench (the timed arms),
# llama-server (the server and MTP arms), llama-completion (the smoke) and llama-tokenize (its ids). The build runs under
# `taskset -c 0-31 nice -n 19` with 16 jobs and a 30-minute bound, and starts only while the timing
# lease is free (rc 75): nvcc beside a sitting would tag its rows [cpu-busy]. An existing tree at
# another commit is refused (rc 65) rather than moved: a moved tree changes rows already timed from
# it. Writes pid, log and rc (last; absent = running or killed) under /root/lcpp-pr<N>.
#
# smoke <N>: llama-completion of the profile's REF_TOKENS text ("The capital of France is") for 5
# tokens at temperature 0, the branch's placement (LCPP<N>_CLI_FLAGS, under LCPP<N>_ENV), on the
# card box.sh put in view; then llama-tokenize of prompt plus completion, whose ids after the prompt's
# are compared by the caller with generate_glm5next's greedy tokens on the same ids. The oracle's
# first generated token is 12089 (ik and ours agree, gate_glm5next_e2e): the line `smoke first=<id>`
# says whether the branch produced it. Not timed and no lease; it loads the whole file (199.7 GB
# through the mapping, 48.7 GB of it to the card).
set -uo pipefail
mode=${1:-} pr=${2:-}
case $mode:$pr in
  build:[0-9]* | smoke:[0-9]*) ;;
  *) echo "usage: build-lcpp-pr.sh build|smoke <PR number>" >&2; exit 64 ;;
esac
case $pr in *[!0-9]*) echo "build-lcpp-pr.sh: the PR is a number, got '$pr'" >&2; exit 64 ;; esac
TREE=/home/user/llama.cpp-pr$pr
HERE=$(cd "${BASH_SOURCE[0]%/*}" && pwd)
SRC=/home/user/llama.cpp
MAINLINE=/home/user/llama.cpp-mainline
NVCC=/usr/local/cuda-13.3/bin/nvcc
USER_PATH=/usr/local/cuda-13.3/bin:/usr/bin:/bin
as_user() { sudo -u user env HOME=/home/user "PATH=$USER_PATH" "$@"; }
ugit() { as_user git -C "$TREE" "$@"; }

if [ "$mode" = smoke ]; then
  # shellcheck source=tools/ref/ref-paths.sh
  source "$HERE/ref-paths.sh"
  [ "$MODEL_NAME" = glm5next ] || { echo "build-lcpp-pr.sh smoke: the profile is $MODEL_NAME, not glm5next" >&2; exit 64; }
  flags_var=LCPP${pr}_CLI_FLAGS env_var=LCPP${pr}_ENV
  flags=${!flags_var:-}
  [ -n "$flags" ] || { echo "build-lcpp-pr.sh smoke: models/glm5next.sh names no $flags_var" >&2; exit 64; }
  envs=${!env_var:-}
  for b in llama-completion llama-tokenize; do
    [ -x "$TREE/build/bin/$b" ] || { echo "build-lcpp-pr.sh smoke: no $TREE/build/bin/$b (build first)" >&2; exit 2; }
  done
  prompt="The capital of France is"
  echo "smoke: $TREE $(git -c safe.directory="$TREE" -C "$TREE" rev-parse --short=10 HEAD) env [$envs] flags [$flags]"
  # shellcheck disable=SC2086 # the profile keeps its flags and NAME=VALUE words as one string
  # stdin closed: a chat template in the file turns conversation mode on where -no-cnv is not taken,
  # and a completion waiting on stdin would hold the card until its bound
  out=$(timeout --kill-after=10 300 env $envs "$TREE/build/bin/llama-completion" -m "$MODEL" -p "$prompt" -n 5 \
    --temp 0 --seed 0 -no-cnv --no-warmup -c 512 $flags < /dev/null 2> "${TMPDIR:-/tmp}/lcpp-pr$pr-smoke.err")
  rc=$?
  echo "smoke: llama-completion rc $rc; its stderr: ${TMPDIR:-/tmp}/lcpp-pr$pr-smoke.err"
  grep -E 'load_tensors|model type|CUDA0 model buffer|CPU_Mapped model buffer|error' "${TMPDIR:-/tmp}/lcpp-pr$pr-smoke.err" | head -n 12 | sed 's/^/smoke: /'
  [ "$rc" = 0 ] || exit "$rc"
  echo "smoke: text [$out]"
  ids=$("$TREE/build/bin/llama-tokenize" -m "$MODEL" -p "$out" --ids --log-disable --no-parse-special 2> /dev/null | tail -n 1)
  echo "smoke: ids of prompt + completion $ids (the prompt's are $REF_TOKENS)"
  first=$(tr -d '[] ' <<< "$ids" | cut -d, -f6)
  echo "smoke first=${first:-none} (the oracle's 12089)"
  exit 0
fi

DIR=/root/lcpp-pr$pr
mkdir -p "$DIR"
rm -f "$DIR/rc"
echo $$ > "$DIR/pid"
exec > "$DIR/log" 2>&1
# shellcheck source=tools/ref/lease.sh
source "$HERE/lease.sh"
fail() {
  echo "build-lcpp-pr: $1"
  echo "${2:-2}" > "$DIR/rc"
  exit "${2:-2}"
}
# Every git and build step from /: the track directory holds a .git file naming the Mac's worktree,
# which git would find from there, and the user `user` cannot enter /root.
cd / || fail "cannot cd /"
echo "build-lcpp-pr: start $(now) pid $$ PR #$pr tree $TREE"
lease_free || case $? in
  1) fail "the timing lease is held (a sitting is running); start again after it" 75 ;;
  *) fail "the timing lease cannot be tested (above); not starting" 70 ;;
esac
head=$(git ls-remote https://github.com/ggml-org/llama.cpp "pull/$pr/head" | cut -f1)
[ -n "$head" ] || fail "no pull/$pr/head on GitHub"
echo "build-lcpp-pr: pull/$pr/head is $head"
if [ -d "$TREE/.git" ]; then
  have=$(ugit rev-parse HEAD) || fail "$TREE is not a git tree"
  [ "$have" = "$head" ] || fail "$TREE is at $have, the PR head is $head: move it by hand, knowing which rows came from which" 65
else
  as_user git clone --shared --no-checkout "$SRC" "$TREE" || fail "clone of $SRC failed"
  ugit fetch https://github.com/ggml-org/llama.cpp "pull/$pr/head" || fail "fetch of pull/$pr/head failed"
  ugit checkout --detach FETCH_HEAD || fail "checkout of $head failed"
fi
echo "build-lcpp-pr: HEAD $(ugit log -1 --format='%H %cI %s')"
echo "build-lcpp-pr: mainline's command-line options ($MAINLINE/build/CMakeCache.txt):"
grep -E ':UNINITIALIZED=|^GGML_CUDA:BOOL=|^CMAKE_BUILD_TYPE:|^CMAKE_CUDA_COMPILER:' "$MAINLINE/build/CMakeCache.txt" | sed 's/^/    /'
t0=$(date +%s)
as_user cmake -S "$TREE" -B "$TREE/build" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DLLAMA_CURL=OFF \
  -DCMAKE_BUILD_TYPE=Release "-DCMAKE_CUDA_COMPILER=$NVCC" || fail "cmake configure failed"
echo "build-lcpp-pr: options that differ from mainline's cache:"
comm -13 <(grep -E '^(GGML|LLAMA)_[A-Z0-9_]+:[A-Z]+=' "$MAINLINE/build/CMakeCache.txt" | sort) \
  <(grep -E '^(GGML|LLAMA)_[A-Z0-9_]+:[A-Z]+=' "$TREE/build/CMakeCache.txt" | sort) | sed 's/^/    /'
timeout --kill-after=10 1800 taskset -c 0-31 nice -n 19 sudo -u user env HOME=/home/user "PATH=$USER_PATH" \
  cmake --build "$TREE/build" --config Release -j 16 --target llama-bench llama-completion llama-tokenize llama-server
rc=$?
echo "build-lcpp-pr: build rc $rc in $(($(date +%s) - t0)) s, end $(now)"
if [ "$rc" = 0 ]; then
  md5sum "$TREE/build/bin/llama-bench" "$TREE/build/bin/llama-completion" "$TREE/build/bin/llama-tokenize" "$TREE/build/bin/llama-server"
  "$TREE/build/bin/llama-bench" --help 2>&1 | grep -E -- '-n-cpu-moe|-nopo|--no-op-offload' | head -n 3
fi
echo "$rc" > "$DIR/rc"
