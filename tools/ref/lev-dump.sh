#!/usr/bin/env bash
# lev's oracle dump (tools/ref/lev_ref.py run), as `just dump-lev` runs it: the server runs through tools/gpu-gate.sh,
# which owns the card pick, the card's gate lock and the bound, so it never shares a card with another track's gate
# (tools/ptx-scan.sh takes the lock for oxart_jit the same way). gpu-gate.sh runs ./target/release/<name>: the script
# links itself there as lev_dump and calls the runner on it; run under that name it is the dump. The model and the
# server tree are the qwen35 profile's (LEV_MODEL, LEV_LCPP); the arguments are lev_ref.py run's.
set -euo pipefail
# The link resolved: tools/gpu-gate.sh runs a hard-linked copy of the link from target/release/.run.<pid>/.
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.."
if [ "$(basename "${BASH_SOURCE[0]}")" != lev_dump ]; then
  mkdir -p target/release
  ln -sf "$PWD/tools/ref/lev-dump.sh" target/release/lev_dump
  BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh lev_dump "$@"
  exit $?
fi
# shellcheck source=tools/ref/ref-paths.sh
source tools/ref/ref-paths.sh
exec python3 tools/ref/lev_ref.py run --model "$LEV_MODEL" --tree "$LEV_LCPP" "$@"
