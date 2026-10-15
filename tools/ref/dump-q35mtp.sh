#!/usr/bin/env bash
# Write the Qwen3.6 MTP step oracle set (tools/ref/models/qwen35moe.sh MTP_*): llama.cpp mainline's MTP draft step of
# the prompt's first MTP_PROMPT prose ids, fed teacher-forced (step i: token ids[i+1], position i, the trunk's h_nextn
# row i), every step run in order and the graph nodes MTP_TAPS dumped at the steps MTP_ROWS, by q35mtp_ref
# (tools/ref/q35mtp_ref.cpp), into $BLOOMERY_DATA/ref-mtp/qwen35moe_mainline_prose64 — the set the refset family
# mtp-qwen35moe-mainline checks and gate_qwen35moe_mtp's clause (m) reads.
#
# The binary must be the one built from this tree's q35mtp_ref.cpp (its .build record's source sha256) against the profile's
# MTP_LCPP at its current commit and libllama (build-q35mtp-ref.sh), and the ids file must have the profile's sha256:
# either refused names what differs. The set is written to <set>.staging and moved over the old set only when q35mtp_ref
# exits 0 and its manifest has the `# mtp` line and the `# complete` trailer. Every layer and the head are on the card
# (-ngl 999), the context the prompt's length, one ubatch. The run goes through tools/gpu-gate.sh, as lev-dump.sh's does:
# the card's gate lock, the bound (Q35MTP_BOUND, default 1500 s) and the card pick (BLOOMERY_GATE_CARD); no timing lease
# (a functional oracle). The script links itself as target/release/q35mtp_dump and runs under that name inside the runner.
#
# Q35MTP_SET moves the set (a trial run installs nothing under the family's name). `--cpu` runs the same binary on the host instead (no card in view, -ngl 0, -t 16, no gate lock) into the same set,
# for a card whose MTP context refuses; the set's `# backend` line says which one wrote it.
#
#   dump-q35mtp.sh [--cpu] [--tap NAME]...   the default taps are the profile's MTP_TAPS; --tap replaces them
set -euo pipefail
REAL=$(readlink -f "${BASH_SOURCE[0]}")
cd "$(dirname "$REAL")/../.."
cpu=0
if [ "${1:-}" = --cpu ]; then cpu=1; shift; fi
if [ "$(basename "${BASH_SOURCE[0]}")" != q35mtp_dump ] && [ "$cpu" = 0 ]; then
  mkdir -p target/release
  ln -sf ../../tools/ref/dump-q35mtp.sh target/release/q35mtp_dump
  BLOOMERY_GATE_BOUND=${Q35MTP_BOUND:-1500} BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh q35mtp_dump "$@"
  exit $?
fi
# shellcheck source=tools/ref/ref-paths.sh
source tools/ref/ref-paths.sh
[ "$MODEL_NAME" = qwen35moe ] ||
  { echo "dump-q35mtp: the MTP set is the qwen35moe profile's; this command picked $MODEL_NAME" >&2; exit 2; }
BIN=$BLOOMERY_DATA/bin/q35mtp_ref
want_src=$(sha256sum tools/ref/q35mtp_ref.cpp | cut -d' ' -f1)
want_commit=$(git -c safe.directory="$MTP_LCPP" -C "$MTP_LCPP" rev-parse --short=9 HEAD)
want_lib=$(md5sum < "$MTP_LCPP/build/bin/libllama.so.0" | cut -d' ' -f1)
if ! grep -qx "source_sha256 $want_src" "$BIN.build" 2>/dev/null ||
   ! grep -qx "lcpp_commit $want_commit" "$BIN.build" 2>/dev/null ||
   ! grep -qx "libllama_md5 $want_lib" "$BIN.build" 2>/dev/null; then
  echo "dump-q35mtp: $BIN is not this tree's q35mtp_ref at $MTP_LCPP $want_commit (rebuild: just build-ref-q35mtp)" >&2
  exit 2
fi
got_sha=$(sha256sum "$MTP_IDS" | cut -d' ' -f1)
[ "$got_sha" = "$MTP_IDS_SHA256" ] ||
  { echo "dump-q35mtp: $MTP_IDS has sha256 $got_sha, the profile pins $MTP_IDS_SHA256" >&2; exit 2; }
[ -f "$MTP_MODEL" ] || { echo "dump-q35mtp: no $MTP_MODEL" >&2; exit 66; }
taps=()
while [ $# -gt 0 ]; do
  case $1 in
    --tap) [ $# -ge 2 ] || { echo "dump-q35mtp: --tap wants a name" >&2; exit 64; }; taps+=(--tap "$2"); shift 2 ;;
    *) echo "dump-q35mtp: unknown argument '$1' (--cpu, --tap NAME)" >&2; exit 64 ;;
  esac
done
if [ ${#taps[@]} = 0 ]; then for t in "${MTP_TAPS[@]}"; do taps+=(--tap "$t"); done; fi
SET=${Q35MTP_SET:-$BLOOMERY_DATA/ref-mtp/qwen35moe_mainline_prose64}
STAGE=$SET.staging
rm -rf "$STAGE"
mkdir -p "$(dirname "$SET")"
drop_incomplete_stage() {
  [ -d "$STAGE" ] || return 0
  rm -rf "$STAGE"
  echo "[staging] removed the incomplete $STAGE" >&2
}
trap drop_incomplete_stage EXIT
where=(-ngl 999) run=(env)
if [ "$cpu" = 1 ]; then where=(-ngl 0 -t 16) run=(env CUDA_VISIBLE_DEVICES=); fi
"${run[@]}" BLOOMERY_REF_BUILD="$want_commit" BLOOMERY_REF_TOKENS_SHA256="$MTP_IDS_SHA256" \
  timeout --kill-after=10 "${Q35MTP_RUN_BOUND:-1200}" "$BIN" -m "$MTP_MODEL" --arch qwen35moe \
  --tokens-file "$MTP_IDS" --tokens-count "$MTP_PROMPT" --rows "$MTP_ROWS" "${taps[@]}" \
  --out "$STAGE" -c "$MTP_PROMPT" "${where[@]}"
grep -q '^# complete' "$STAGE/MANIFEST.tsv" ||
  { echo "dump-q35mtp: the staged set has no completion trailer — not installing it" >&2; exit 1; }
grep -q $'^# mtp\t' "$STAGE/MANIFEST.tsv" ||
  { echo "dump-q35mtp: the staged set has no # mtp line (no MTP context ran) — not installing it" >&2; exit 1; }
rm -rf "$SET"
mv "$STAGE" "$SET"
echo "dump-q35mtp: installed $SET"
