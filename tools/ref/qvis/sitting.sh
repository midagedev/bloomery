#!/usr/bin/env bash
# One sitting of a Qwen seat's image-input oracle sets on the box: the card run of every set, then the CPU twins, in one
# detached, bounded driver with an rc sentinel (the chat ids are written before it: `just dump-ref-qvis <model> --ids`, CPU
# only, a second). Qwen3.8's is a 111 GB load per run: the whole sitting is one approved window (CLEFVIS_BOUND, below), started only
# while the lead's batch hold /root/bloomery-batch.gpuhold is down (polled every 60 s) so the batch's GPU lanes keep the cards.
#
#   BLOOMERY_MODEL=qwen4exp BLOOMERY_REMOTE='~/repo/bloomery-<track>' tools/box.sh 'setsid -f bash tools/ref/qvis/sitting.sh qwen4exp 5400 </dev/null >/dev/null 2>&1'
#   BLOOMERY_BOX_READONLY=1 tools/box.sh 'cat /root/qvis-sitting-qwen4exp/pid; tail -n 5 /root/qvis-sitting-qwen4exp/log; cat /root/qvis-sitting-qwen4exp/rc'
#
# Writes pid, log, start/end epoch lines and rc (last; absent = running or killed) under /root/qvis-sitting-<model>. The rc is
# the first non-zero phase's, 124 when the bound ended it. The card pick is box.sh's (under its default pin gpu-gate's `any`: an idle
# A6000, else the 3090); the dump's `-ncmoe` follows the card (profile.sh). Both phases run under one `timeout` of the bound, so
# a hung dump ends the sitting and its children (the timeout's own process group), never a bare kill of a recorded pid.
set -uo pipefail
model=${1:?usage: sitting.sh <qwen35moe|qwen4exp> [bound seconds]} bound=${2:-3000}
case $model in qwen35moe | qwen4exp) ;; *) echo "sitting.sh: the seat is qwen35moe or qwen4exp, got $model" >&2; exit 64 ;; esac
case $bound in '' | *[!0-9]*) echo "sitting.sh: the bound is whole seconds, got '$bound'" >&2; exit 64 ;; esac
HERE=$(cd "${BASH_SOURCE[0]%/*}/../../.." && pwd)
cd "$HERE" || exit 2
D=/root/qvis-sitting-$model
mkdir -p "$D"
rm -f "$D/rc"
echo $$ > "$D/pid"
exec > "$D/log" 2>&1
now() { date -u +%Y-%m-%dT%H:%M:%SZ; }
echo "sitting: $model start-wait $(now) bound ${bound}s"
while [ -e /root/bloomery-batch.gpuhold ]; do
  echo "sitting: the batch hold is up ($(cat /root/bloomery-batch.gpuhold 2> /dev/null)); polling in 60 s"
  sleep 60
done
echo "start $(date +%s) $(now)"
export BLOOMERY_MODEL=$model CLEFVIS_BOUND=$bound CLEFVIS_RUN_BOUND=${QVIS_RUN_BOUND:-1700}
rc=0
phase() { # <name> <args…>: one clefvis.sh call under the sitting's remaining bound
  local name=$1 t0 left prc
  shift
  t0=$(date +%s)
  left=$((bound - (t0 - START)))
  if [ "$left" -le 0 ]; then echo "phase $name: the bound is spent"; [ "$rc" != 0 ] || rc=124; return; fi
  timeout --kill-after=30 "$left" bash tools/ref/clefvis/clefvis.sh "$@"
  prc=$?
  echo "phase $name rc=$prc wall=$(($(date +%s) - t0)) s"
  [ "$rc" != 0 ] || rc=$prc
}
START=$(date +%s)
phase card
phase twin --cpu-twin
echo "end $(date +%s) $(now) wall=$(($(date +%s) - START)) s rc=$rc"
echo "$rc" > "$D/rc"
