#!/usr/bin/env bash
# Builds the mistral.rs binary the depth runners' mrs arms run (MRSBIN, tools/ref/models/qwen3moe.sh)
# in its recommended CUDA form: `--features "cuda flash-attn"`, the build upstream's release,
# install.sh and docs make on Ampere. Without flash-attn its prompt attention runs the eager
# naive_sdpa path, which is not the engine users run. The tree is /home/user/mistral.rs, built as
# its owner `user`, into its own target/. The binary it replaces is kept once, by commit, as SAVE.
#
# It starts only while the timing lease is free (rc 75 otherwise) and does not hold it: a 20-40 min
# lease would stop every round that waits on it before a box command. So no sitting may open while
# it runs: guard_cpu (lease.sh) tags a V4.1 row it meets [cpu-busy], a tagged row is a lost row, and
# depth-qwen3moe.sh has no CPU guard at all. The flash kernels compile as parallel nvcc jobs, half
# of `available_parallelism()` (`thread_percentage(0.5)` in mistralrs-flash-attn/build.rs through
# cudaforge, no env override), so the build runs under `taskset -c 0-31` — 16 jobs, not 32 — at nice
# 19, in a systemd scope with MemoryHigh and MemoryMax: it cannot push out the page cache a V4.1
# load holds, and past MemoryMax it dies by name (rc != 0) instead of the machine swapping.
# CUDA_COMPUTE_CAP stays unset, as in the build it replaces: cudaforge prints rerun-if-env-changed
# for it, so setting it would rebuild every kernel crate, not only flash-attn; detection reads 8.6
# on both cards.
#
# Detached run on the box (it writes pid, log and rc under DIR; rc last, absent = running or killed):
#   BLOOMERY_REMOTE='~/repo/bloomery-ee-mrs' tools/box.sh 'setsid -f bash tools/ref/build-mrs.sh </dev/null >/dev/null 2>&1'
#   ssh ws 'cat /root/mrs-build/pid; tail -n 3 /root/mrs-build/log; cat /root/mrs-build/rc'
#
# Environment: MRS_DIR (default /home/user/mistral.rs), MRS_BUILD_DIR (default /root/mrs-build),
# MRS_SAVE (default /home/user/mistralrs-noflash-<commit>), MRS_FEATURES (default "cuda flash-attn"),
# MRS_MEM_HIGH / MRS_MEM_MAX (default 96G / 140G).
set -uo pipefail
MRS=${MRS_DIR:-/home/user/mistral.rs}
DIR=${MRS_BUILD_DIR:-/root/mrs-build}
FEATURES=${MRS_FEATURES:-cuda flash-attn}
MEM_HIGH=${MRS_MEM_HIGH:-96G}
MEM_MAX=${MRS_MEM_MAX:-140G}
mkdir -p "$DIR"
rm -f "$DIR/rc"
echo $$ > "$DIR/pid"
exec > "$DIR/log" 2>&1
# shellcheck source=tools/ref/lease.sh
source "${BASH_SOURCE[0]%/*}/lease.sh"

USER_PATH=/usr/local/cuda/bin:/home/user/.cargo/bin:/usr/bin:/bin
# systemd-run, taskset and nice exec a binary, so the build line spells this out instead of calling it.
as_user() { sudo -u user env HOME=/home/user "PATH=$USER_PATH" "$@"; }
fail() {
  echo "build-mrs: $1"
  echo "${2:-2}" > "$DIR/rc"
  exit "${2:-2}"
}

echo "build-mrs: start $(now) pid $$"
commit=$(as_user git -C "$MRS" rev-parse --short=9 HEAD) || fail "no git tree at $MRS"
SAVE=${MRS_SAVE:-/home/user/mistralrs-noflash-$commit}
BIN=$MRS/target/release/mistralrs
echo "build-mrs: tree $MRS commit $commit features [$FEATURES]"
echo "build-mrs: local changes, built in:"
as_user git -C "$MRS" status --short
if [ -e "$BIN" ] && [ ! -e "$SAVE" ]; then
  cp -p "$BIN" "$SAVE" || fail "could not keep $BIN as $SAVE"
fi
[ ! -e "$SAVE" ] || md5sum "$SAVE"
[ ! -e "$BIN" ] || md5sum "$BIN"

flock -n "$LEASE_LOCK" true || fail "the timing lease is held (a sitting is running); start again after it" 75
free -g | head -n 2
t0=$(date +%s)
systemd-run --scope --quiet -p "MemoryHigh=$MEM_HIGH" -p "MemoryMax=$MEM_MAX" \
  taskset -c 0-31 nice -n 19 sudo -u user env HOME=/home/user "PATH=$USER_PATH" \
  bash -c "cd '$MRS' && cargo build --release -p mistralrs-cli --features '$FEATURES'"
rc=$?
echo "build-mrs: cargo rc $rc in $(($(date +%s) - t0)) s, end $(now)"
if [ "$rc" = 0 ]; then
  md5sum "$BIN"
  "$BIN" --version
  echo "build-mrs: flash_fwd names in the binary: $(strings "$BIN" | grep -c flash_fwd)"
fi
echo "$rc" > "$DIR/rc"
