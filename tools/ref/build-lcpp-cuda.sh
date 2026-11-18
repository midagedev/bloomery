#!/usr/bin/env bash
# A CUDA build of a reference llama.cpp tree for the card this machine times, in a build directory of its own.
#
#   BLOOMERY_REMOTE='~/repo/bloomery-<track>' tools/box.sh 'setsid -f bash tools/ref/build-lcpp-cuda.sh build llama.cpp-v41 build-sm120 120a </dev/null >/dev/null 2>&1'
#   ssh ws 'cat /root/lcpp-cuda-llama.cpp-v41-build-sm120/pid; tail -n 3 /root/lcpp-cuda-llama.cpp-v41-build-sm120/log; cat /root/lcpp-cuda-llama.cpp-v41-build-sm120/rc'
#
# build <tree> <build-dir> <arch> [<target>...]: /home/user/<tree>'s sources configured into /home/user/<tree>/<build-dir> with
# GGML_CUDA=ON, CMAKE_CUDA_ARCHITECTURES=<arch> (the card's compute capability: `nvidia-smi --query-gpu=compute_cap`, 12.0 is 120a,
# 8.6 is 86), LLAMA_CURL=OFF, Release, nvcc from cuda-13.3 (the flags of the trees' existing CUDA builds), and
# LLAMA_USE_PREBUILT_UI=OFF (the server's web UI is fetched from a hub bucket at build time and a tree older than the bucket's
# latest stops on its asset check; the references' server needs no UI); every other option at its default. Targets default to
# llama-bench and llama-server (the timed arms and the warm server arms of the depth runners).
#
# The build directory is never `build`: a profile's `build/` is the tree the oracle sets and the references' rows name (a rebuild
# moves the md5 of the libraries they pin, and an existing build of another architecture keeps the rows it timed), so a card that is
# not the one the tree was built for gets a directory beside it, and the depth runners are pointed at it by LCPPBIN (the
# llama-server beside it follows; GLM's pull-request trees take LCPP<N>BIN and LCPP<N>SRV). An existing build directory is
# reconfigured and built again (cmake and make skip what is current), so a killed build resumes.
#
# Every git and build step runs as the tree's owner (`user` for the trees cloned there, root for one made as root), under
# `taskset -c 0-31 nice -n 19` with 16 jobs and a bound (LCPP_BUILD_BOUND, default 2400 s), and starts only while the timing lease
# is free (rc 75): nvcc beside a sitting would tag its rows [cpu-busy]. A build is not timed. Writes pid, log and rc (last; absent =
# running or killed) under /root/lcpp-cuda-<tree>-<build-dir>/.
#
# Exit: 0 built; 2 a step failed; 64 usage; 65 the tree is not a llama.cpp checkout; 75 the timing lease is held.
set -uo pipefail
mode=${1:-} tree=${2:-} bdir=${3:-} arch=${4:-}
if [ "$mode" != build ] || [ -z "$tree" ] || [ -z "$bdir" ] || [ -z "$arch" ]; then
  echo "usage: build-lcpp-cuda.sh build <tree under /home/user> <build dir> <arch, e.g. 120a> [<target>...]" >&2
  exit 64
fi
shift 4
case $tree in '' | *[!A-Za-z0-9._-]* | .*) echo "build-lcpp-cuda.sh: the tree is a directory name under /home/user, got '$tree'" >&2; exit 64 ;; esac
case $bdir in '' | *[!A-Za-z0-9._-]* | .*) echo "build-lcpp-cuda.sh: the build dir is a directory name, got '$bdir'" >&2; exit 64 ;; esac
[ "$bdir" != build ] || { echo "build-lcpp-cuda.sh: the build dir is never 'build' (the oracle's and the rows' tree); name another" >&2; exit 64; }
case $arch in [0-9][0-9] | [0-9][0-9][0-9] | [0-9][0-9]a | [0-9][0-9][0-9]a) ;; *) echo "build-lcpp-cuda.sh: the arch is the compute capability without its dot (120a, 90a, 86), got '$arch'" >&2; exit 64 ;; esac
targets=("$@")
[ "${#targets[@]}" -gt 0 ] || targets=(llama-bench llama-server)
TREE=/home/user/$tree
HERE=$(cd "${BASH_SOURCE[0]%/*}" && pwd)
NVCC=/usr/local/cuda-13.3/bin/nvcc
USER_PATH=/usr/local/cuda-13.3/bin:/usr/bin:/bin
BOUND=${LCPP_BUILD_BOUND:-2400}
DIR=/root/lcpp-cuda-$tree-$bdir
# The steps run as the tree's owner: `user` for the trees cloned there, root for one that was made as root (a build directory cannot be
# made in a tree its builder does not own).
OWNER=$(stat -c %U "$TREE" 2> /dev/null) || { echo "build-lcpp-cuda.sh: $TREE is not there" >&2; exit 65; }
OWNER_HOME=$(getent passwd "$OWNER" | cut -d: -f6)
[ -n "$OWNER_HOME" ] || { echo "build-lcpp-cuda.sh: no home directory for $OWNER, the owner of $TREE" >&2; exit 65; }
as_user() { sudo -u "$OWNER" env "HOME=$OWNER_HOME" "PATH=$USER_PATH" "$@"; }
mkdir -p "$DIR"
rm -f "$DIR/rc"
echo $$ > "$DIR/pid"
exec > "$DIR/log" 2>&1
# shellcheck source=tools/ref/lease.sh
source "$HERE/lease.sh"
fail() {
  echo "build-lcpp-cuda: $1"
  echo "${2:-2}" > "$DIR/rc"
  exit "${2:-2}"
}
# From /: the track directory holds a .git file naming the Mac's worktree, which git would find from there, and `user` cannot
# enter /root.
cd / || fail "cannot cd /"
echo "build-lcpp-cuda: start $(now) pid $$ tree $TREE build dir $bdir arch $arch targets ${targets[*]}"
[ -f "$TREE/CMakeLists.txt" ] && [ -f "$TREE/include/llama.h" ] || fail "$TREE is not a llama.cpp checkout (no CMakeLists.txt or include/llama.h)" 65
lease_free || case $? in
  1) fail "the timing lease is held (a sitting is running); start again after it" 75 ;;
  *) fail "the timing lease cannot be tested (above); not starting" 70 ;;
esac
echo "build-lcpp-cuda: HEAD $(as_user git -c "safe.directory=$TREE" -C "$TREE" log -1 --format='%H %cI %s')"
as_user git -c "safe.directory=$TREE" -C "$TREE" status --porcelain --untracked-files=no | sed 's/^/build-lcpp-cuda: tracked change: /'
if [ -f "$TREE/build/CMakeCache.txt" ]; then
  echo "build-lcpp-cuda: the tree's own build ($TREE/build/CMakeCache.txt), left as it is:"
  grep -E ':UNINITIALIZED=|^GGML_CUDA:BOOL=|^CMAKE_BUILD_TYPE:' "$TREE/build/CMakeCache.txt" | sed 's/^/    /'
fi
t0=$(date +%s)
# A web UI a failed earlier build of this directory fetched is stale for this configuration, which embeds none: remove it.
for u in dist dist.tar.gz dist.tar.gz.sha256 ui-gzip; do
  [ ! -e "$TREE/$bdir/tools/ui/$u" ] || as_user rm -rf "$TREE/$bdir/tools/ui/$u"
done
as_user cmake -S "$TREE" -B "$TREE/$bdir" -DGGML_CUDA=ON "-DCMAKE_CUDA_ARCHITECTURES=$arch" -DLLAMA_CURL=OFF \
  -DLLAMA_USE_PREBUILT_UI=OFF -DCMAKE_BUILD_TYPE=Release "-DCMAKE_CUDA_COMPILER=$NVCC" || fail "cmake configure failed"
grep -E '^CMAKE_CUDA_ARCHITECTURES:|^GGML_CUDA:BOOL=' "$TREE/$bdir/CMakeCache.txt" | sed 's/^/    configured: /'
rc=0
lease_bounded "$BOUND" taskset -c 0-31 nice -n 19 sudo -u "$OWNER" env "HOME=$OWNER_HOME" "PATH=$USER_PATH" \
  cmake --build "$TREE/$bdir" --config Release -j 16 --target "${targets[@]}" || rc=$?
echo "build-lcpp-cuda: build rc $rc in $(($(date +%s) - t0)) s, end $(now)"
if [ "$rc" = 0 ]; then
  for t in "${targets[@]}"; do
    [ -x "$TREE/$bdir/bin/$t" ] && md5sum "$TREE/$bdir/bin/$t"
  done
  [ ! -x "$TREE/$bdir/bin/llama-bench" ] || "$TREE/$bdir/bin/llama-bench" --help 2>&1 | grep -E -- '-n-cpu-moe|-nopo' | head -n 2
fi
echo "$rc" > "$DIR/rc"
exit "$rc"
