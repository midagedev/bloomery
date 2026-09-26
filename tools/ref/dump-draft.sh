#!/usr/bin/env bash
# Produce the DSpark draft oracle set: every node ik_llama.cpp's V4.1 draft computes while the
# target decodes a fixed prompt greedily with the draft on, as raw f32, into
# $BLOOMERY_DATA/ref-draft/<set>/ (dump_draft.cpp's header describes the files and the manifest).
#
# ref-draft is a root of its own: nothing here ever writes under ref/ or the ref_* sets the target
# gates read. Stage, then swap, as dump.sh does: the dump goes to <set>.staging, is installed only
# with its `# complete` trailer, and never replaces a set of another model file.
#
# The set: the first 64 ids of the code corpus (router trace's stream; its sha256 is pinned below),
# 32 target positions decoded with the draft, ik's DSpark stage at block width 3 (the width of the
# 09-13 per-position accept rates). 64 + 32 stays below depth 123, where ik's past-window mask and
# the reference's still agree. The target is the deepseek41 profile's model on the card box.sh pins
# (the 3090): every layer offloaded, every layer's routed experts on the host, the draft whole on the
# card (its own --n-cpu-moe 0; it inherits the target's otherwise). GGML_CUDA_NO_PINNED_WEIGHTS=1 is
# the profile's IK_GPU_ENV and load-bearing for the same reason (models/deepseek41.sh).
#
# The dump runs under the CPU lease (the target's host experts page in from the model files and use
# every core) and the 3090 gate lock, and refuses a card that still runs a compute process (rc 75).
#
# Pick the profile on the Mac side: BLOOMERY_MODEL=deepseek41 ./tools/box.sh 'bash tools/ref/dump-draft.sh'.
#   DRAFT_SET_NAME   the set's name under ref-draft (default code64_n32_w3)
#   REF_THREADS      ik's -t (default 32)
set -euo pipefail
export IK=${DRAFT_IK:-/home/user/ik-dspark-draft}
# shellcheck source=tools/ref/ref-paths.sh
source "${BASH_SOURCE[0]%/*}/ref-paths.sh"
# shellcheck source=tools/ref/lease.sh
source "${BASH_SOURCE[0]%/*}/lease.sh"
[ "$MODEL_NAME" = deepseek41 ] ||
  { echo "dump-draft.sh: the draft oracle is DeepSeek-V4.1's; this command picked the $MODEL_NAME profile" >&2; exit 2; }

BASE_SHA=db517b69
PATCHED="src/graphs/build_deepseek4.cpp src/llama-hparams.cpp src/llama-hparams.h src/llama-load-tensors.cpp "
DRAFT_MODEL=/models/DeepSeek-V4.1-Flash-DSpark/DeepSeek-V4.1-Flash-Fp8-128x742M-MXFP4_MOE.tl37.gguf
TOKENS_FILE=$BLOOMERY_DATA/engram/corpus-code.ids
TOKENS_SHA256=e51b69554b6ede3404b2fa82a174af12c82ced5efab9444ee58842c1cd0a4b5b
N_PROMPT=64
N_PREDICT=32
SPEC=dspark:n_max=3
SET=${DRAFT_SET_NAME:-code64_n32_w3}
THREADS=${REF_THREADS:-32}
case $SET in
  */*|.*|*.staging|*.old|'') echo "dump-draft.sh: '$SET' cannot name a set" >&2; exit 2 ;;
esac
case $THREADS in
  ''|*[!0-9]*|0*) echo "dump-draft.sh: REF_THREADS must be a positive integer, got '$THREADS'" >&2; exit 2 ;;
esac

BIN=$BLOOMERY_DATA/bin/dump_draft
[ -x "$BIN" ] || { echo "no dump_draft at $BIN — run: just build-ref-dump-draft" >&2; exit 2; }
# The binary must load ik from the draft tree's build, and the tree must still be BASE_SHA plus the
# four patched files (build-dump-draft.sh checks their contents; this checks nothing moved since).
IK_REAL=$(readlink -f "$IK")
for lib in libllama.so libggml.so; do
  got=$(ldd "$BIN" 2>/dev/null | awk -v l="$lib" '$1 == l { print $3 }' || true)
  case $(readlink -f "$got" 2>/dev/null) in
    "$IK_REAL"/build/*) ;;
    *) echo "[foreign-lib] $BIN loads $lib from '${got:-nowhere}', not from $IK/build —" \
         "rebuild it: just build-ref-dump-draft" >&2; exit 3 ;;
  esac
done
ikgit() { git -c safe.directory='*' -C "$IK" "$@"; }
head=$(ikgit rev-parse --short=8 HEAD)
changed=$(ikgit diff --name-only "$BASE_SHA" | sort | tr '\n' ' ')
[ "$head" = "$BASE_SHA" ] && [ "$changed" = "$PATCHED" ] ||
  { echo "dump-draft.sh: $IK is at $head with [$changed] changed, not $BASE_SHA + [$PATCHED] — rebuild it" >&2; exit 3; }
BUILD=$(cat "$BLOOMERY_DATA/bin/dump_draft.build")
[ "$BUILD" = "$BASE_SHA+dsv41-draft 4eeb56aa1d75" ] ||
  { echo "dump-draft.sh: $BIN was built as '$BUILD'" >&2; exit 3; }

[ -f "$TOKENS_FILE" ] || { echo "dump-draft.sh: no ids file at $TOKENS_FILE" >&2; exit 2; }
got=$(sha256sum "$TOKENS_FILE" | cut -d' ' -f1)
[ "$got" = "$TOKENS_SHA256" ] ||
  { echo "dump-draft.sh: $TOKENS_FILE has sha256 $got, this set names $TOKENS_SHA256" >&2; exit 2; }

# The 3090 gates serialize on the gate lock (tools/gpu-gate.sh); the dump takes it too, so no gate
# shares the card's memory with it, then checks that nothing else runs there.
exec 8>/root/bloomery-gate.lock
echo "[gate-lock] waiting for /root/bloomery-gate.lock ..."
flock -w 1800 8 || { echo "[gate-lock] timed out after 30 min" >&2; exit 75; }
busy=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>&1 || true)
if [ -n "$busy" ]; then
  echo "dump-draft.sh: the card (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all}) runs a compute process: $busy" >&2
  exit 75
fi

WITNESS=(head-epoch loadavg pressure-io mem pgmajfault read-sectors gpus gpu-apps lock-holder model)
lease_take
witness pre-dump

ROOT=$BLOOMERY_DATA/ref-draft
REF=$ROOT/$SET
STAGE=$ROOT/$SET.staging
mkdir -p "$ROOT"
rm -rf "$STAGE"
mkdir -p "$STAGE"
t0=$(date +%s)
# Bounded like dump.sh's dumper (BLOOMERY_DUMP_BOUND, default 1800 s — five times the V4.1 CPU dump on
# record, rig-log 2026-09-23): a hung dumper must end rather than hold the lease.
lease_bounded "${BLOOMERY_DUMP_BOUND:-1800}" env GGML_CUDA_NO_PINNED_WEIGHTS=1 BLOOMERY_REF_WRITE=1 BLOOMERY_REF_DIR="$STAGE" BLOOMERY_REF_BUILD="$BUILD" \
  BLOOMERY_REF_TOKENS_SHA256="$TOKENS_SHA256" \
  "$BIN" -m "$MODEL" -md "$DRAFT_MODEL" --expect-arch deepseek41 --expect-draft-arch dflash \
    --tokens-file "$TOKENS_FILE" --tokens-count "$N_PROMPT" -n "$N_PREDICT" --spec-type "$SPEC" \
    -ngl 999 --n-cpu-moe 999 -draft "--n-cpu-moe 0" -c 512 -t "$THREADS" --defer-experts
echo "dump-draft.sh: dump in $(($(date +%s) - t0)) s"
witness post-dump

grep -q '^# complete' "$STAGE/MANIFEST.tsv" ||
  { echo "dump-draft.sh: the staged set has no completion trailer — not installing it" >&2; exit 1; }
model_of() { awk -F'\t' '$1 == "# model_file" || $1 == "# draft_model_file" { printf "%s ", $2 }' "$1"; }
if [ -f "$REF/MANIFEST.tsv" ] && [ "$(model_of "$REF/MANIFEST.tsv")" != "$(model_of "$STAGE/MANIFEST.tsv")" ]; then
  echo "dump-draft.sh: $REF holds a set of [$(model_of "$REF/MANIFEST.tsv")], this dump is of" \
    "[$(model_of "$STAGE/MANIFEST.tsv")] — not replacing it (the staged set stays in $STAGE)" >&2
  exit 1
fi
rm -rf "$REF.old"
if [ -d "$REF" ]; then mv "$REF" "$REF.old"; fi
mv "$STAGE" "$REF"
rm -rf "$REF.old"
echo "draft tensors: $(grep -c $'^tensor\t' "$REF/MANIFEST.tsv")  inputs: $(grep -c $'^input\t' "$REF/MANIFEST.tsv" || true)" \
  " blocks: $(grep -c $'^verify\t' "$REF/MANIFEST.tsv" || true)  files: $(find "$REF" -type f | wc -l)"
grep $'^# blocks\t' "$REF/MANIFEST.tsv"
echo "build: $BUILD  threads: $THREADS  set: $REF"
