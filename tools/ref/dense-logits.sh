#!/usr/bin/env bash
# Write a dense profile's logits oracle sets (LOGITS_SETS of tools/ref/models/qwen38d.sh, qwen38nv.sh or nanbeige43.sh): for each `<set>:<P>`,
# llama.cpp mainline's logits after the first P ids of PROSE_IDS and after each of the LOGITS_FORCED ids that follow them
# (teacher forced on the corpus), by dense_logits_ref (tools/ref/dense_logits_ref.cpp) through tools/ref/dense-logits.py, into
# $BLOOMERY_DATA/<set>/ — the sets the profile's refset family checks (logits-qwen38d, logits-qwen38nv: crates/refset/src/arch/qwen35/dense.rs;
# logits-nanbeige43: crates/refset/src/arch/llama/dense.rs). The profile names its file's identity (LOGITS_TARGET, a key of
# dense-logits.py's TARGETS), which also gives the architecture the harness checks the file against.
#
# The binary must be the one tools/ref/build-dense-logits.sh built from this tree's dense_logits_ref.cpp against the profile's
# LCPP at its pin, and the corpus must have the profile's sha256: either refused names what differs (dense-logits.py checks
# the record, the tree, the libraries and the model file's size and sha256 before it opens an engine). The whole file goes
# on the card box.sh puts in view (-ngl 99, the card's gate lock, tools/gpu-gate.sh), the prompt in ubatches of
# LOGITS_UBATCH in one llama_decode, then the last prompt id and every forced id one at a time (mainline's serving shape).
# Each set is written to <set>.new and moved over the old set only when the writer exits 0 (its manifest's `# complete`
# trailer is then written). No lease: a functional oracle, not a timed run.
#
# `--cpu-twin` writes each set's CPU twin instead, into $BLOOMERY_DATA/<set>.cpu/: the same binary and ids with no card in
# view, mainline's CPU kernels (activations as q8_K per 256) standing in for its CUDA MMQ (q8_1 per 32), another 8-bit
# realization of the same rule: the oracle's own floor that a gate's band is derived from
# (`python3 tools/ref/dense-logits.py floor <set>...`). No gate reads a twin.
#
#   dense-logits.sh [--cpu-twin]            every set
#   dense-logits.sh [--cpu-twin] <set>...   the sets named
set -euo pipefail
# shellcheck source=tools/ref/ref-paths.sh
source "$(dirname "${BASH_SOURCE[0]}")/ref-paths.sh"
# shellcheck source=tools/ref/lease.sh
source "$(dirname "${BASH_SOURCE[0]}")/lease.sh"
{ declare -p LOGITS_SETS > /dev/null 2>&1 && [ -n "${LOGITS_TARGET:-}" ]; } ||
  { echo "dense-logits.sh: the $MODEL_NAME profile names no logits sets (LOGITS_SETS, LOGITS_TARGET); this command picked it" >&2; exit 2; }
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
twin=0
if [ "${1:-}" = --cpu-twin ]; then twin=1; shift; fi
BIN=$HERE/target/release/dense_logits_ref
if [ ! -x "$BIN" ] || [ ! -f "$BIN.build.json" ]; then
  echo "dense-logits.sh: no $BIN with its .build.json (build it: just build-dense-logits-ref)" >&2
  exit 2
fi
for want in "$@"; do
  known=0
  for s in "${LOGITS_SETS[@]}"; do [ "${s%%:*}" = "$want" ] && known=1; done
  [ "$known" = 1 ] || { echo "dense-logits.sh: $want is no set of the $MODEL_NAME profile" >&2; exit 64; }
done
[ -f "$MODEL" ] || { echo "dense-logits.sh: the profile's model $MODEL is not there" >&2; exit 2; }
rc=0
for s in "${LOGITS_SETS[@]}"; do
  IFS=: read -r set_name p <<<"$s"
  if [ $# -gt 0 ] && ! printf '%s\n' "$@" | grep -qx "$set_name"; then continue; fi
  dir=$BLOOMERY_DATA/$set_name backend=cuda bound=1500
  if [ "$twin" = 1 ]; then dir=$dir.cpu backend=cpu bound=1200; fi
  rm -rf "$dir.new" "$dir.new.staging"
  t0=$SECONDS
  # The writer's own bound is the call's; this one is its backstop.
  if lease_bounded $((bound + 60)) python3 "$HERE/tools/ref/dense-logits.py" run --kind reference --backend "$backend" \
      --target "$LOGITS_TARGET" --binary "$BIN" --model "$MODEL" --corpus "$PROSE_IDS" --corpus-sha256 "$PROSE_SHA256" \
      --prompt-count "$p" --forced-count "$LOGITS_FORCED" --prefill batch --ctx "$LOGITS_CTX" --ubatch "$LOGITS_UBATCH" \
      --bound "$bound" --out "$dir.new"; then
    rm -rf "$dir"
    mv "$dir.new" "$dir"
    echo "dense-logits.sh: $set_name ($p ids + $LOGITS_FORCED forced, $backend) -> $dir in $((SECONDS - t0)) s"
  else
    rc=$?
    echo "dense-logits.sh: $set_name failed (rc $rc, $((SECONDS - t0)) s); the old set, if any, stays; the evidence is in $dir.new.staging" >&2
  fi
done
exit "$rc"
