#!/usr/bin/env bash
# Build the V4.1 candidate-mask oracle on the box: the ik tree CAND_IK (models/deepseek41.sh) and a
# dump_ref linked against it, for the `-sep` decode-step variants and d1c (refset family
# cand-deepseek41). Runs under tools/box.sh (as root); the ik trees belong to `user`, so every git and
# cmake step in them runs as that user.
#
# The tree is a worktree of /home/user/ik_llama.cpp: ik main at PIN (it carries the separate V4.1 graph,
# V41_SEPARATE, which builds the candidate mask) with PICK, #2507's index-key fix, cherry-picked and
# nothing else, because ik main's shared ds4_build_comp still rotates the pooled latent in place before
# the index keys read it. The tree is checked, not trusted, on every run: HEAD's parent is PIN, the
# commit's patch is PICK's (git patch-id), and nothing is modified or untracked outside build/. A tree
# that is anything else is refused — a hand edit there would reach the oracle under this build's name.
# The `# build` line every set of the family carries is HEAD's short hash (dump.sh reads it), which
# the family pins (crates/refset/src/arch/deepseek41/mod.rs, CAND_BUILD).
#
# The cmake flags are the ones the V4.1 oracle tree (REF_TREE, the profile's IK) was configured with:
# after the configure every GGML_* and LLAMA_* cache entry is compared with REF_TREE's, and a
# difference stops the build, so the two trees differ by their sources alone.
#
# The ik build uses every core for minutes, so it runs under the machine-wide CPU lease with an
# exclusive card (BLOOMERY_LEASE_CARD); dump_ref is then linked by build-dump.sh into DUMP_OUT, the
# variants' STEP_BIN_DIR. The installed $BLOOMERY_DATA/bin/dump_ref is never touched: it is linked
# against REF_TREE and every other V4.1 set is dumped by it.
set -euo pipefail
PIN=ed27bf7e
PICK=49ef19d0
SRC_TREE=/home/user/ik_llama.cpp
export BLOOMERY_MODEL=deepseek41
# shellcheck source=tools/ref/ref-paths.sh
source "$(dirname "${BASH_SOURCE[0]}")/ref-paths.sh"
REF_TREE=$IK
export IK=$CAND_IK
export DUMP_OUT=$BLOOMERY_DATA/bin-cand
# shellcheck source=tools/ref/lease.sh
source "$(dirname "${BASH_SOURCE[0]}")/lease.sh"

# From the user's home: box.sh runs this in root's remote directory, which git as the user cannot stat.
as_user() { (cd /home/user && sudo -u user env HOME=/home/user PATH=/usr/local/cuda/bin:/usr/bin:/bin "$@"); }
ugit() { as_user git -C "$1" "${@:2}"; }

pin_full=$(ugit "$SRC_TREE" rev-parse "$PIN^{commit}")
if [ ! -d "$IK" ]; then
  ugit "$SRC_TREE" worktree add -b v41/cand-oracle "$IK" "$PIN"
  ugit "$IK" cherry-pick "$PICK"
fi
parent=$(ugit "$IK" rev-parse HEAD^)
[ "$parent" = "$pin_full" ] || { echo "build-ik-cand: $IK HEAD's parent is $parent, not $PIN ($pin_full)" >&2; exit 1; }
want_id=$(ugit "$IK" show "$PICK" | as_user git patch-id --stable | cut -d' ' -f1)
got_id=$(ugit "$IK" show HEAD | as_user git patch-id --stable | cut -d' ' -f1)
[ "$got_id" = "$want_id" ] ||
  { echo "build-ik-cand: $IK HEAD's patch (patch-id $got_id) is not $PICK's ($want_id)" >&2; exit 1; }
dirty=$(ugit "$IK" status --porcelain --untracked-files=all | grep -v '^?? build/' || true)
[ -z "$dirty" ] || { echo "build-ik-cand: $IK has changes outside its commits:" >&2; echo "$dirty" >&2; exit 1; }
BUILD=$(ugit "$IK" rev-parse --short=8 HEAD)
echo "build-ik-cand: $IK = $PIN + $PICK as $BUILD"

as_user cmake -S "$IK" -B "$IK/build" -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc > /dev/null
cache_flags() { grep -E '^(GGML|LLAMA)_[A-Z0-9_]*:[A-Z]+=' "$1/build/CMakeCache.txt" | grep -v '_FOUND:' | LC_ALL=C sort; }
# PIN is newer than REF_TREE, so an option one of them lacks is printed; an option both have must agree.
differ=$(LC_ALL=C join -t= <(cache_flags "$REF_TREE") <(cache_flags "$IK") | awk -F= '$2 != $3')
only=$(LC_ALL=C join -t= -v1 -v2 <(cache_flags "$REF_TREE") <(cache_flags "$IK") || true)
[ -z "$only" ] || { echo "build-ik-cand: cache options only one of $REF_TREE and $IK has:"; echo "$only"; }
if [ -n "$differ" ]; then
  echo "build-ik-cand: $IK/build is configured differently from $REF_TREE/build (option=$REF_TREE=$IK):" >&2
  echo "$differ" >&2
  exit 1
fi

lease_take
t0=$(date +%s)
# Bounded (BLOOMERY_BUILD_BOUND, default 1800 s): the build holds every core under the lease, and a
# hung one must end rather than hold the machine. The bound runs as the tree's user, inside as_user,
# so it signals the build it started.
as_user timeout --kill-after=10 "${BLOOMERY_BUILD_BOUND:-1800}" cmake --build "$IK/build" -j "$(nproc)" --target llama common
echo "build-ik-cand: ik build in $(($(date +%s) - t0)) s"
bash "$(dirname "${BASH_SOURCE[0]}")/build-dump.sh"
lease_release
echo "build-ik-cand: $DUMP_OUT/dump_ref against $IK ($BUILD)"
