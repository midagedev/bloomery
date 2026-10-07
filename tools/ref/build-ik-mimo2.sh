#!/usr/bin/env bash
# Build the MiMo-V2.6-Flash oracle on the box: the ik tree the mimo2 profile names (IK, models/mimo2.sh)
# and a dump_ref linked against it into $BLOOMERY_DATA/$REF_BIN_NAME. Runs under tools/box.sh (as
# root); the ik trees belong to `user`, so every git and cmake step in them runs as that user.
#
# The tree is a worktree of /home/user/ik_llama.cpp: upstream ik at PIN, the commit that loads MiMo's
# fused blk.N.attn_qkv (the oracle tree REF_TREE predates it), with PICKS cherry-picked in order: upstream's
# per-layer fix of that load (5bf8f0fe, #2583: PIN sizes every layer's fused attn_qkv by layer 0's KV heads,
# and MiMo's sliding-window layers have twice its KV heads), and of the five commits REF_TREE carries on its
# own base the one upstream lacks (the index-key fix, PR #2507, not merged). The other four are upstream at
# PIN under their own commits — the V4.1
# model (a7616195, #2455; the oracle's copy conflicts with it, as does the V4 stream-name rename that
# follows it, which touches only the deepseek4 files), the engram prefetch and the indexed branch's
# per-thread sinks (d412fc75, #2511), both of which cherry-pick empty. The tree is checked, not trusted,
# on every run: the commit PICKS' count back from HEAD is PIN, each commit's patch is its pick's (git
# patch-id), and nothing is modified or untracked outside build/ and PROVENANCE. A tree that is anything
# else is refused — a hand edit there would reach the oracle under this build's name. The `# build` line
# every set of the family carries is HEAD's short hash (dump.sh reads it), which the family pins
# (crates/refset/src/arch/mimo2/mod.rs, IK_BUILD).
#
# The cmake flags are REF_TREE's: after the configure every GGML_* and LLAMA_* cache entry is compared
# with REF_TREE's, and a difference stops the build, so the two trees differ by their sources alone.
#
# The ik build uses every core for minutes, so it runs under the machine-wide CPU lease with an
# exclusive card (BLOOMERY_LEASE_CARD); dump_ref is then linked by build-dump.sh into
# $BLOOMERY_DATA/$REF_BIN_NAME. The installed $BLOOMERY_DATA/bin/dump_ref is never touched: it is
# linked against REF_TREE. The shared repository's .git must hold no entry another user owns, before
# and after (a root-owned object there breaks every tree's git as `user`).
set -euo pipefail
PIN=043ced9a
PICKS=(49ef19d0 5bf8f0fe)
SRC_TREE=/home/user/ik_llama.cpp
REF_TREE=/home/user/ik-idxkey
export BLOOMERY_MODEL=mimo2
# shellcheck source=tools/ref/ref-paths.sh
source "$(dirname "${BASH_SOURCE[0]}")/ref-paths.sh"
[ "$IK" != "$REF_TREE" ] ||
  { echo "build-ik-mimo2: the mimo2 profile's IK is $REF_TREE, not a tree of its own" >&2; exit 2; }
export IK
[ -n "${REF_BIN_NAME:-}" ] || { echo "build-ik-mimo2: the mimo2 profile names no REF_BIN_NAME" >&2; exit 2; }
export DUMP_OUT=$BLOOMERY_DATA/$REF_BIN_NAME
# shellcheck source=tools/ref/lease.sh
source "$(dirname "${BASH_SOURCE[0]}")/lease.sh"

# From the user's home: box.sh runs this in root's remote directory, which git as the user cannot stat.
as_user() { (cd /home/user && sudo -u user env HOME=/home/user PATH=/usr/local/cuda/bin:/usr/bin:/bin "$@"); }
ugit() { as_user git -C "$1" "${@:2}"; }

foreign() { find "$SRC_TREE/.git" ! -user user -print -quit; }
f=$(foreign)
[ -z "$f" ] || { echo "build-ik-mimo2: $SRC_TREE/.git already holds an entry not owned by user: $f" >&2; exit 1; }
pin_full=$(ugit "$SRC_TREE" rev-parse "$PIN^{commit}")
if [ ! -d "$IK" ]; then
  ugit "$SRC_TREE" worktree add -b mimo2/oracle "$IK" "$PIN"
  for c in "${PICKS[@]}"; do ugit "$IK" cherry-pick "$c"; done
fi
base=$(ugit "$IK" rev-parse "HEAD~${#PICKS[@]}")
[ "$base" = "$pin_full" ] ||
  { echo "build-ik-mimo2: $IK HEAD~${#PICKS[@]} is $base, not $PIN ($pin_full)" >&2; exit 1; }
for i in "${!PICKS[@]}"; do
  c=${PICKS[$i]}
  at="HEAD~$((${#PICKS[@]} - 1 - i))"
  want=$(ugit "$IK" show "$c" | as_user git patch-id --stable | cut -d' ' -f1)
  got=$(ugit "$IK" show "$at" | as_user git patch-id --stable | cut -d' ' -f1)
  [ "$got" = "$want" ] ||
    { echo "build-ik-mimo2: $IK $at's patch (patch-id $got) is not $c's ($want)" >&2; exit 1; }
done
dirty=$(ugit "$IK" status --porcelain --untracked-files=all | grep -v -e '^?? build/' -e '^?? PROVENANCE$' || true)
[ -z "$dirty" ] || { echo "build-ik-mimo2: $IK has changes outside its commits:" >&2; echo "$dirty" >&2; exit 1; }
BUILD=$(ugit "$IK" rev-parse --short=8 HEAD)
echo "build-ik-mimo2: $IK = $PIN + ${PICKS[*]} as $BUILD"
{
  echo "base $PIN $(ugit "$IK" log -1 --format=%s "$PIN")"
  for c in "${PICKS[@]}"; do echo "pick $c $(ugit "$IK" log -1 --format=%s "$c")"; done
  echo "head $BUILD"
} | as_user tee "$IK/PROVENANCE" > /dev/null

as_user cmake -S "$IK" -B "$IK/build" -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DGGML_IQK_FA_ALL_QUANTS=ON > /dev/null
cache_flags() { grep -E '^(GGML|LLAMA)_[A-Z0-9_]*:[A-Z]+=' "$1/build/CMakeCache.txt" | grep -v '_FOUND:' | LC_ALL=C sort; }
# PIN is newer than REF_TREE's base, so an option one of them lacks is printed; an option both have must agree.
differ=$(LC_ALL=C join -t= <(cache_flags "$REF_TREE") <(cache_flags "$IK") | awk -F= '$2 != $3')
only=$(LC_ALL=C join -t= -v1 -v2 <(cache_flags "$REF_TREE") <(cache_flags "$IK") || true)
[ -z "$only" ] || { echo "build-ik-mimo2: cache options only one of $REF_TREE and $IK has:"; echo "$only"; }
if [ -n "$differ" ]; then
  echo "build-ik-mimo2: $IK/build is configured differently from $REF_TREE/build (option=$REF_TREE=$IK):" >&2
  echo "$differ" >&2
  exit 1
fi

lease_take
t0=$(date +%s)
# Bounded (BLOOMERY_BUILD_BOUND, default 1800 s): the build holds every core under the lease, and a
# hung one must end rather than hold the machine. The bound runs as the tree's user, inside as_user,
# so it signals the build it started.
as_user timeout --kill-after=10 "${BLOOMERY_BUILD_BOUND:-1800}" cmake --build "$IK/build" -j "$(nproc)" \
  --target llama common llama-tokenize
echo "build-ik-mimo2: ik build in $(($(date +%s) - t0)) s"
bash "$(dirname "${BASH_SOURCE[0]}")/build-dump.sh"
lease_release
f=$(foreign)
[ -z "$f" ] || { echo "build-ik-mimo2: $SRC_TREE/.git now holds an entry not owned by user: $f" >&2; exit 1; }
echo "build-ik-mimo2: $DUMP_OUT/dump_ref against $IK ($BUILD)"
