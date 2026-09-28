#!/usr/bin/env bash
# The two-card probe runner (lead-only): one gpu-gates probe binary under the machine-wide lease with
# both cards in view, the A6000 as device 0 and the 3090 as device 1, between witness blocks. For the
# probes whose question is the pair of cards or the host memory both reach (probe_tred, the exchange
# between the cards; probe_hostreg, a registration both can use). time-gate.sh cannot take them: it
# puts the timing card alone in view.
#
#   tools/ref/probe-twocard.sh <probe bin> [args...]     # default args: --time
#   BLOOMERY_DRY=1 tools/ref/probe-twocard.sh <bin> ...  # the command and the cards, then exit 0
#   tools/ref/probe-twocard.sh --self-test               # the argument and dry paths, no GPU
#
# Before the lease: the binary must be newer than its sources (timing-card.sh's assert_fresh_binary,
# rc 2 or 3). Under the lease, before the run: a compute process on the A6000 ends the run (rc 75,
# contention: the probe would share the card), and one on the 3090 is guard_other's — recorded as
# [other-busy], or rc 75 under BLOOMERY_OTHER_STRICT=1, which the sitting sets: both cards are the
# measured object here, not only the timing card. The probe runs under lease_bounded
# (BLOOMERY_ARM_BOUND, default 900 s) and its exit code is this script's.
set -euo pipefail
# shellcheck source=tools/ref/ref-paths.sh
source "${BASH_SOURCE[0]%/*}/ref-paths.sh"
# shellcheck source=tools/ref/timing-card.sh
source "${BASH_SOURCE[0]%/*}/timing-card.sh"
# shellcheck source=tools/ref/lease.sh
source "${BASH_SOURCE[0]%/*}/lease.sh"

if [ "${1:-}" = --self-test ]; then
  me=${BASH_SOURCE[0]}
  fail=0
  out=$(BLOOMERY_DRY=1 bash "$me" probe_tred --check 2>&1) || { echo "self-test: the dry run failed: $out" >&2; fail=1; }
  grep -qx "\[dry\] CUDA_VISIBLE_DEVICES=$GPU_A6000,$GPU_3090" <<< "$out" ||
    { echo "self-test: the dry run does not put the A6000 first and the 3090 second: $out" >&2; fail=1; }
  grep -qx '\[dry\] command: target/release/probe_tred --check' <<< "$out" ||
    { echo "self-test: the dry run's command is not the probe's: $out" >&2; fail=1; }
  out=$(BLOOMERY_DRY=1 bash "$me" probe_tred 2>&1) || fail=1
  grep -qx '\[dry\] command: target/release/probe_tred --time' <<< "$out" ||
    { echo "self-test: no arguments is not --time: $out" >&2; fail=1; }
  for bad in '' '../x' 'a/b' 'gate_p8' 'probe_'; do
    rc=0
    BLOOMERY_DRY=1 bash "$me" "$bad" > /dev/null 2>&1 || rc=$?
    [ "$rc" = 64 ] || { echo "self-test: '$bad' was not refused with 64 (rc $rc)" >&2; fail=1; }
  done
  [ "$fail" = 0 ] && echo "probe-twocard self-test: ok"
  exit "$fail"
fi

NAME=${1:-}
case $NAME in
  probe_[a-z0-9]*) case $NAME in *[!a-z0-9_]*) NAME= ;; esac ;;
  *) NAME= ;;
esac
[ -n "$NAME" ] || { echo "probe-twocard.sh: the first argument is a probe_<name> binary of crates/gpu-gates, got '${1:-}'" >&2; exit 64; }
shift
ARGS=("$@")
[ ${#ARGS[@]} -gt 0 ] || ARGS=(--time)
BIN=target/release/$NAME
export CUDA_VISIBLE_DEVICES=$TIMING_GPU,$OTHER_GPU

if [ -n "${BLOOMERY_DRY:-}" ]; then
  echo "[dry] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  echo "[dry] command: $BIN ${ARGS[*]}"
  exit 0
fi

assert_fresh_binary "$BIN" || exit $?
WITNESS=(head indent card busiest model)
lease_take
witness pre
apps=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i "$TIMING_GPU")
if [ -n "$apps" ]; then
  echo "[timing-busy] $(now) compute apps on the A6000 ($TIMING_GPU): [$(echo "$apps" | tr '\n' ';')] — the probe would share it" >&2
  witness abort-timing >&2
  exit 75
fi
guard_other
rc=0
lease_bounded "$LEASE_ARM_BOUND" "$BIN" "${ARGS[@]}" || rc=$?
guard_other
witness post
echo "$NAME ${ARGS[*]} rc=$rc${OTHER_BUSY_TAG}"
exit $rc
