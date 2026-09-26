#!/usr/bin/env bash
# ik-greedy.sh — ik_llama.cpp's greedy continuation of one prompt on the V4.1 file (or the qwen3moe one),
# CPU only, one token per llama_decode (-b 1's path), under the machine-wide CPU lease: the reference of the
# V4.1 step gate's --greedy (G3, crates/gpu-gates/src/bin/gate_deepseek41_step.rs) and of the qwen3moe e2e
# gate's greedy arm (crates/gpu-gates/src/bin/gate_qwen3moe_e2e.rs).
#
#   BLOOMERY_MODEL=deepseek41 tools/box.sh 'bash tools/ref/ik-greedy.sh'
#   just ik-greedy-ds41 [PROMPT]
#   just ik-greedy-qwen3moe          (prompts 0-7, the e2e gate's set)
#
# The profile picks the directories: deepseek41 writes under $BLOOMERY_DATA/greedy-ds41 and builds into
# bin/deepseek41, qwen3moe under $BLOOMERY_DATA/qwen3moe/greedy and bin/qwen3moe. The file names below are
# the V4.1 ones; qwen3moe's are the same names in its own directory.
#
# The prompt is tools/ref/prompts.tsv's row PROMPT (default 0), whose ids there are the V2-Lite tokenizer's.
# This runner takes its text and reads it with the V4.1 file's own tokenizer (llama-tokenize from the
# profile's tree, no BOS: the file sets tokenizer.ggml.add_bos_token to false), so both engines start from
# the same ids; neither engine tokenizes during the run. Row 0's ids must equal the profile's REF_TOKENS —
# the oracle's sequence is that same text — or the run stops (rc 1): a tokenizer that reads the text
# otherwise is not this model's. argmax_ref stops at the model's EOS and records it; row 0 reaches EOS after
# three tokens, so a longer continuation needs another row (PROMPT=7, "Once upon a time").
#
# The binary is tools/ref/argmax_ref.cpp (--gen N, --step-prefill), built against the profile's tree
# ($IK, the oracle's) into $BLOOMERY_DATA/bin/deepseek41/: the shared $BLOOMERY_DATA/bin/argmax_ref is
# linked to the V2-Lite profile's tree. A binary that would load libllama or libggml from outside $IK is
# refused (rc 3), as dump.sh refuses a dumper.
#
# The tokenizer is TOKENIZE (default under deepseek41: the fixed /home/user/ik-tilde tree, as oracle.sh;
# otherwise the profile's tree's llama-tokenize). Under deepseek41 it must read
# `~` as a symbol, as mainline and HF do, so that `~/` is one word and one id; it is probed on the file
# first, the probe crates/tokenizer/tools/oracle.sh runs:
#   llama-tokenize -m $MODEL -p '~/' --ids --log-disable     -> [71520]
# and any other answer (a tree whose `~` is in neither P nor S returns [96, 17]) stops the run (rc 65)
# before the lease: a prompt file written by that tree would carry the split. The fixed tree is
# /home/user/ik-tilde (TOKENIZE=/home/user/ik-tilde/build/bin/llama-tokenize). The qwen3moe profile has no
# such probe: its vocabulary's `~/` id has not been pinned.
#
# Writes $BLOOMERY_DATA/greedy-ds41/prompt<P>.tsv (id, text, the V4.1 ids) and greedy-ik-cpu-<N>-p<P>.tsv
# (argmax_ref's row: gen_ids and gen_margins), and <N>-p<P>.log with the witness blocks. N is GEN, default 64.
# Bounded by timeout (IK_GREEDY_BOUND seconds, default 900). The lease is lease_take's (tools/ref/lease.sh):
# the run needs a card, BLOOMERY_LEASE_CARD=docs/cards/<slug>.card through BLOOMERY_BOX_ENV, written
# for that run; a loop over prompts takes the lease once per prompt, with the one card.
set -euo pipefail
# shellcheck source=tools/ref/ref-paths.sh
source "${BASH_SOURCE[0]%/*}/ref-paths.sh"
# shellcheck source=tools/ref/lease.sh
source "${BASH_SOURCE[0]%/*}/lease.sh"
case $MODEL_NAME in
  deepseek41) OUTDIR_NAME=greedy-ds41 ;;
  qwen3moe)   OUTDIR_NAME=qwen3moe/greedy ;;
  *) echo "ik-greedy.sh: the profile is $MODEL_NAME — pick deepseek41 or qwen3moe on the Mac side (BLOOMERY_MODEL=...)" >&2
     exit 64 ;;
esac
GEN=${GEN:-64}
PROMPT=${PROMPT:-0}
BOUND=${IK_GREEDY_BOUND:-900}
for n in "$GEN" "$BOUND"; do
  case $n in ''|*[!0-9]*|0*) echo "ik-greedy.sh: GEN and IK_GREEDY_BOUND are positive integers, got '$n'" >&2; exit 64 ;; esac
done
case $PROMPT in ''|*[!0-9]*) echo "ik-greedy.sh: PROMPT is a row id, got '$PROMPT'" >&2; exit 64 ;; esac
HERE=$(cd "${BASH_SOURCE[0]%/*}/../.." && pwd)
OUTDIR=$BLOOMERY_DATA/$OUTDIR_NAME
BINDIR=$BLOOMERY_DATA/bin/$MODEL_NAME
BIN=$BINDIR/argmax_ref
# Default: under deepseek41 the fixed tree (as crates/tokenizer/tools/oracle.sh), since the profile's $IK
# tree splits `~/`; the qwen3moe profile keeps its own tree's tokenizer.
if [ "$MODEL_NAME" = deepseek41 ]; then
  TOK=${TOKENIZE:-/home/user/ik-tilde/build/bin/llama-tokenize}
else
  TOK=${TOKENIZE:-$IK/build/bin/llama-tokenize}
fi
mkdir -p "$OUTDIR" "$BINDIR"
[ -x "$TOK" ] || { echo "ik-greedy.sh: no $TOK — build the tree first" >&2; exit 2; }
if [ "$MODEL_NAME" = deepseek41 ]; then
  # shellcheck disable=SC2088 # the literal '~/' is the probe's text
  probe=$(CUDA_VISIBLE_DEVICES="" "$TOK" -m "$MODEL" -p '~/' --ids --log-disable 2>/dev/null | tail -n 1)
  [ "$probe" = '[71520]' ] || {
    echo "ik-greedy.sh: $TOK tokenizes '~/' as $probe on $MODEL, not [71520]: its \`~\` is not a symbol" \
      "— TOKENIZE=/home/user/ik-tilde/build/bin/llama-tokenize is the tree that reads it as one" >&2
    exit 65
  }
fi

ARGMAX_OUT=$BINDIR bash "$HERE/tools/ref/build-argmax.sh"
IK_REAL=$(readlink -f "$IK")
for lib in libllama.so libggml.so; do
  got=$(ldd "$BIN" 2>/dev/null | awk -v l="$lib" '$1 == l { print $3 }' || true)
  case $(readlink -f "$got" 2>/dev/null) in
    "$IK_REAL"/build/*) ;;
    *) echo "[foreign-lib] $BIN loads $lib from '${got:-nowhere}', not from $IK/build" >&2; exit 3 ;;
  esac
done

TEXT=$(awk -F '\t' -v p="$PROMPT" '!/^#/ && $1 == p { print $2; exit }' "$HERE/tools/ref/prompts.tsv")
[ -n "$TEXT" ] || { echo "ik-greedy.sh: tools/ref/prompts.tsv has no row $PROMPT" >&2; exit 2; }
IDS=$(CUDA_VISIBLE_DEVICES="" "$TOK" -m "$MODEL" -p "$TEXT" --ids --log-disable --no-parse-special 2>/dev/null \
      | tail -n 1 | tr -d '[] ')
[ -n "$IDS" ] || { echo "ik-greedy.sh: llama-tokenize printed no ids for '$TEXT'" >&2; exit 1; }
if [ "$PROMPT" = 0 ] && [ "$IDS" != "$REF_TOKENS" ]; then
  echo "ik-greedy.sh: prompt 0 ('$TEXT') reads as '$IDS' under $MODEL, the profile's REF_TOKENS are '$REF_TOKENS'" >&2
  exit 1
fi
PROMPTS=$OUTDIR/prompt$PROMPT.tsv
printf '# prompt %s of tools/ref/prompts.tsv, read by %s on %s\n%s\t%s\t%s\n' "$PROMPT" "$TOK" "$(basename "$MODEL")" "$PROMPT" "$TEXT" "$IDS" > "$PROMPTS"

OUT=$OUTDIR/greedy-ik-cpu-$GEN-p$PROMPT.tsv
LOG=$OUTDIR/$GEN-p$PROMPT.log
: > "$LOG"
say() { printf '%s\n' "$*" | tee -a "$LOG"; }
HEAD_REV=$(git -c safe.directory="$IK" -C "$IK" rev-parse --short HEAD)
say "ik-greedy.sh: tree $IK at $HEAD_REV, model $MODEL, prompt $PROMPT ids $IDS, $GEN greedy steps, bound ${BOUND}s"

# greedy_witness <tag>: lease.sh's fields, then this run's tree and binary, into the log as well.
WITNESS=(head-epoch loadavg pressure-cpu pressure-io mem pgmajfault read-sectors gpu-apps lock-holder)
greedy_witness() {
  {
    witness "$1"
    echo "tree: $IK head=$HEAD_REV binary: $BIN sha256=$(sha256sum "$BIN" | cut -c1-12)"
  } | tee -a "$LOG"
}

lease_take
echo "[lease] held by pid $$ at $(now), card $BLOOMERY_LEASE_CARD sha256=$(sha256sum "$LEASE_TREE/$BLOOMERY_LEASE_CARD" | cut -c1-64)" >> "$LOG"
greedy_witness pre-greedy
t0=$(date +%s)
rc=0
CUDA_VISIBLE_DEVICES="" timeout --kill-after=10 "$BOUND" \
  "$BIN" -m "$MODEL" --prompts "$PROMPTS" --gen "$GEN" --step-prefill -ngl 0 -c "$REF_CTX" -t 32 \
  "${REF_DUMP_ARGS[@]}" > "$OUT.partial" 2>> "$LOG" &
pid=$!
say "ik-greedy.sh: argmax_ref under pid $pid ($(cat "/proc/$pid/comm" 2>/dev/null || echo gone))"
wait "$pid" || rc=$?
t1=$(date +%s)
greedy_witness post-greedy
lease_release
if [ "$rc" != 0 ]; then
  say "ik-greedy.sh: argmax_ref exited $rc after $((t1 - t0)) s; the log's tail:"
  tail -n 15 "$LOG" >&2
  rm -f "$OUT.partial"
  exit "$rc"
fi
mv "$OUT.partial" "$OUT"
say "greedy tag=greedy-ik-cpu-$GEN-p$PROMPT tree=$IK head=$HEAD_REV model=$(basename "$MODEL") prompt=$IDS gen=$GEN wall_s=$((t1 - t0)) out=$OUT"
