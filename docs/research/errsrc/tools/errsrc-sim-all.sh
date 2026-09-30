#!/usr/bin/env bash
# errsrc: run exact_ref --act ARM over all 33 prompts (steps 0..30, the f64 set's span) on the box,
# one CPU-lease per prompt, logs under the track's target/errsrc-sim/ARM/p<id>.log; then copy back.
# usage: ERRSRC_OUT=DIR errsrc-sim-all.sh ARM [ARM ...] (run from the errsrc track's tree; its box dir is
# ~/repo/bloomery-errsrc)
set -euo pipefail
S=${ERRSRC_OUT:?ERRSRC_OUT unset: the local directory the logs are copied back into}
cd "$(dirname "$0")/../../../.."
for ARM in "$@"; do
  ./tools/box.sh "cargo build --release -p bloomery-gpu-gates --bin exact_ref && mkdir -p target/errsrc-sim/$ARM && for id in \$(seq 0 32); do flock -w 3600 /root/bloomery-cpu.lock ./target/release/exact_ref --prompt-id \$id --steps 30 --act $ARM > target/errsrc-sim/$ARM/p\$id.log || exit 1; echo \"$ARM p\$id done\"; done"
  mkdir -p "$S/errsrc-sim/$ARM"
  scp -q "ws:~/repo/bloomery-errsrc/target/errsrc-sim/$ARM/p*.log" "$S/errsrc-sim/$ARM/"
done
echo all-done
