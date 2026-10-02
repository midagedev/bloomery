#!/usr/bin/env bash
# Write the qwen35 profile's hidden-state oracle sets (tools/ref/models/qwen35.sh HIDDEN_SETS): for each `<set>:<P>`,
# llama.cpp mainline's result_norm of every position of the first P ids of PROSE_IDS, through hidden_ref
# (tools/ref/hidden_ref.cpp), into $BLOOMERY_DATA/<set>/ — the sets the refset family hidden-qwen35 checks and
# gate-gpu-clef-hidden reads.
#
# The binary must be the one built from this tree's hidden_ref.cpp (its .build record's source sha256) against the
# profile's LCPP at its current commit, and the ids file must have the profile's sha256: either refused names what
# differs. Every layer on the card box.sh puts in view (-ngl 99), the context the longest set's P, ubatches of 512
# (llama.cpp's default). Each set is written to <set>.staging and moved over the old set only when hidden_ref
# exits 0 (its manifest's `# complete` trailer is then written). No lease: a functional oracle, not a timed run.
#
# `--cpu-twin` writes each set's CPU twin instead, into $BLOOMERY_DATA/<set>.cpu/: the same binary and ids with no
# card in view (CUDA_VISIBLE_DEVICES empty, -ngl 0, -t 16), so mainline's CPU kernels (activations as q8_K per 256)
# stand in for its CUDA MMQ (q8_1 per 32) — another 8-bit realization of the same rule, the oracle's own floor that
# gate_clef_hidden's bands are derived from (tools/ref/hidden-diff.py <set>.cpu <set>). No gate reads a twin.
#
#   hidden.sh [--cpu-twin]            every set
#   hidden.sh [--cpu-twin] <set>...   the sets named
set -euo pipefail
# shellcheck source=tools/ref/ref-paths.sh
source "$(dirname "${BASH_SOURCE[0]}")/ref-paths.sh"
[ "$MODEL_NAME" = qwen35 ] ||
  { echo "hidden.sh: the hidden-state sets are the qwen35 profile's; this command picked $MODEL_NAME" >&2; exit 2; }
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
twin=0
if [ "${1:-}" = --cpu-twin ]; then twin=1; shift; fi
BIN=$BLOOMERY_DATA/bin/hidden_ref
want_src=$(sha256sum "$HERE/tools/ref/hidden_ref.cpp" | cut -d' ' -f1)
want_commit=$(git -c safe.directory="$LCPP" -C "$LCPP" rev-parse --short=9 HEAD)
if ! grep -qx "source_sha256 $want_src" "$BIN.build" 2>/dev/null ||
   ! grep -qx "lcpp_commit $want_commit" "$BIN.build" 2>/dev/null; then
  echo "hidden.sh: $BIN is not this tree's hidden_ref at $LCPP $want_commit (rebuild: just build-ref-hidden)" >&2
  exit 2
fi
got_sha=$(sha256sum "$PROSE_IDS" | cut -d' ' -f1)
[ "$got_sha" = "$PROSE_SHA256" ] ||
  { echo "hidden.sh: $PROSE_IDS has sha256 $got_sha, the profile pins $PROSE_SHA256" >&2; exit 2; }
ctx=0
for s in "${HIDDEN_SETS[@]}"; do p=${s#*:}; [ "$p" -gt "$ctx" ] && ctx=$p; done
rc=0
for s in "${HIDDEN_SETS[@]}"; do
  set_name=${s%%:*} p=${s#*:}
  if [ $# -gt 0 ] && ! printf '%s\n' "$@" | grep -qx "$set_name"; then continue; fi
  dir=$BLOOMERY_DATA/$set_name
  where=(-ngl 99) run=(env)
  if [ "$twin" = 1 ]; then dir=$dir.cpu where=(-ngl 0 -t 16) run=(env CUDA_VISIBLE_DEVICES=); fi
  rm -rf "$dir.staging"
  if "${run[@]}" BLOOMERY_REF_BUILD="$want_commit" BLOOMERY_REF_TOKENS_SHA256="$PROSE_SHA256" \
      timeout --kill-after=10 900 "$BIN" -m "$MODEL" --tokens-file "$PROSE_IDS" --tokens-count "$p" \
      --out "$dir.staging" -c "$ctx" -ub 512 "${where[@]}"; then
    rm -rf "$dir"
    mv "$dir.staging" "$dir"
    echo "hidden.sh: $set_name ($p ids) -> $dir"
  else
    rc=$?
    echo "hidden.sh: $set_name failed (rc $rc); the old set, if any, stays" >&2
  fi
done
exit "$rc"
