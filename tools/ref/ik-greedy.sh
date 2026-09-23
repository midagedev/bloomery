#!/usr/bin/env bash
# ik-greedy.sh — ik_llama.cpp's greedy continuation of one prompt on the V4.1 file, CPU only, one token per
# llama_decode (-b 1's path), under the machine-wide CPU lease: the reference of the V4.1 step gate's
# --greedy (G3, crates/gpu-gates/src/bin/gate_deepseek41_step.rs).
#
#   BLOOMERY_MODEL=deepseek41 tools/box.sh 'bash tools/ref/ik-greedy.sh'
#   just ik-greedy-ds41 [PROMPT]
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
# Writes $BLOOMERY_DATA/greedy-ds41/prompt<P>.tsv (id, text, the V4.1 ids) and greedy-ik-cpu-<N>-p<P>.tsv
# (argmax_ref's row: gen_ids and gen_margins), and <N>-p<P>.log with the witness blocks. N is GEN, default 64.
# Bounded by timeout (IK_GREEDY_BOUND seconds, default 900).
set -euo pipefail
# shellcheck source=tools/ref/ref-paths.sh
source "${BASH_SOURCE[0]%/*}/ref-paths.sh"
[ "$MODEL_NAME" = deepseek41 ] || {
  echo "ik-greedy.sh: the profile is $MODEL_NAME — pick deepseek41 on the Mac side (BLOOMERY_MODEL=deepseek41)" >&2
  exit 64
}
GEN=${GEN:-64}
PROMPT=${PROMPT:-0}
BOUND=${IK_GREEDY_BOUND:-900}
for n in "$GEN" "$BOUND"; do
  case $n in ''|*[!0-9]*|0*) echo "ik-greedy.sh: GEN and IK_GREEDY_BOUND are positive integers, got '$n'" >&2; exit 64 ;; esac
done
case $PROMPT in ''|*[!0-9]*) echo "ik-greedy.sh: PROMPT is a row id, got '$PROMPT'" >&2; exit 64 ;; esac
HERE=$(cd "${BASH_SOURCE[0]%/*}/../.." && pwd)
OUTDIR=$BLOOMERY_DATA/greedy-ds41
BINDIR=$BLOOMERY_DATA/bin/deepseek41
BIN=$BINDIR/argmax_ref
TOK=$IK/build/bin/llama-tokenize
mkdir -p "$OUTDIR" "$BINDIR"
[ -x "$TOK" ] || { echo "ik-greedy.sh: no $TOK — build the tree first" >&2; exit 2; }

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

witness() {
  local dev sectors
  dev=$(df --output=source "$MODEL" 2>/dev/null | tail -n 1) || true
  sectors=$(awk '{print $3}' "/sys/class/block/${dev#/dev/}/stat" 2>/dev/null || echo '?')
  {
    echo "--- witness $1 $(date -u +%Y-%m-%dT%H:%M:%SZ) epoch $(date +%s) ---"
    echo "loadavg: $(cat /proc/loadavg)"
    echo "pressure-cpu: $(grep '^some' /proc/pressure/cpu | head -n1)"
    echo "pressure-io: $(grep '^some' /proc/pressure/io | head -n1)"
    echo "mem: $(grep -E '^(MemAvailable|Cached):' /proc/meminfo | tr -s ' ' | tr '\n' ' ')"
    echo "pgmajfault: $(awk '$1 == "pgmajfault" {print $2}' /proc/vmstat)"
    echo "read-sectors: $sectors ($dev, 512 B each)"
    echo "gpu-apps: [$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | tr '\n' ';')]"
    echo "lock-holder-pid: $$"
    echo "tree: $IK head=$HEAD_REV binary: $BIN sha256=$(sha256sum "$BIN" | cut -c1-12)"
  } | tee -a "$LOG"
}

LOCK=/root/bloomery-cpu.lock
exec 9>"$LOCK"
say "[lease] waiting for $LOCK ..."
flock -w 1800 9 || { say "[lease] timed out after 30 min"; exit 75; }
say "[lease] acquired $(date -u +%H:%M:%SZ)"
witness pre-greedy
t0=$(date +%s)
rc=0
CUDA_VISIBLE_DEVICES="" timeout --kill-after=10 "$BOUND" \
  "$BIN" -m "$MODEL" --prompts "$PROMPTS" --gen "$GEN" --step-prefill -ngl 0 -c "$REF_CTX" -t 32 \
  "${REF_DUMP_ARGS[@]}" > "$OUT.partial" 2>> "$LOG" &
pid=$!
say "ik-greedy.sh: argmax_ref under pid $pid ($(cat "/proc/$pid/comm" 2>/dev/null || echo gone))"
wait "$pid" || rc=$?
t1=$(date +%s)
witness post-greedy
exec 9>&-
if [ "$rc" != 0 ]; then
  say "ik-greedy.sh: argmax_ref exited $rc after $((t1 - t0)) s; the log's tail:"
  tail -n 15 "$LOG" >&2
  rm -f "$OUT.partial"
  exit "$rc"
fi
mv "$OUT.partial" "$OUT"
say "greedy tag=greedy-ik-cpu-$GEN-p$PROMPT tree=$IK head=$HEAD_REV model=$(basename "$MODEL") prompt=$IDS gen=$GEN wall_s=$((t1 - t0)) out=$OUT"
