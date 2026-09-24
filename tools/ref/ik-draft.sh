#!/usr/bin/env bash
# ik's V4.1 decode on a real-text prompt, plain or with the DSpark draft, under the lease on the
# timing card (run on the box, lead-only):
#
#   just time-ik-draft prose 96 dspark
#   BLOOMERY_MODEL=deepseek41 tools/box.sh 'bash tools/ref/ik-draft.sh <corpus> <N> [plain|dspark]'
#
# The ik twin of `just time-gpu-ds41 --tokens <512 ids> -n N`: the prompt is the first 512 ids of
# $BLOOMERY_DATA/engram/corpus-<corpus>.ids, so an ik row and one of ours are about the same
# prompt. It blocks two failures: an ik draft row read against one of ours on another prompt (the
# acceptance rate is a property of the text), and a draft row whose prompt ik tokenized to other
# ids than ours (llama-cli takes text, not ids).
#
# The prompt. llama-cli reads text, so the ids are decoded by our tokenizer
# (bloomery-tokenize --decode, special tokens rendered as their text) and checked twice: before
# the lease, llama-tokenize (vocabulary only) must turn the text back into the same 512 ids, and
# after the run, the ids llama-cli printed under --verbose-prompt must be those ids. Either
# mismatch refuses the row (rc 65) with the first differing position. llama-cli's -f drops one
# trailing newline, so its file carries one more than the text; --no-escape keeps a backslash in
# the text a backslash.
#
# The run: llama-cli -m MODEL $IK_GPU_FLAGS -c CTX -n N --temp 0 --ignore-eos, under
# `env $IK_GPU_ENV` (the profile's comment says why that variable is load-bearing). --temp 0 is
# ik's greedy sampler; --ignore-eos because generate_ds41 has no end-of-generation stop and
# always emits N tokens. CTX is 512 + N + 64 rounded up to 256 — llama-cli's default context is
# the model's own (over a million positions), not the prompt's. The dspark arm adds
# `-md $DSPARK_MODEL --spec-type $STAGE`. ik builds the DSpark draft's parameters from the
# target's own (common/speculative.cpp: only an MTP stage clears them), so the draft inherits
# -ngl and --n-cpu-moe, and with its three layers under --n-cpu-moe 34 every draft expert runs
# on the host; BLOOMERY_IK_DRAFT_PARAMS (passed as ik's --draft-params) changes that.
#
# The summary line per run:
#   ik-draft corpus=<c> arm=<a> n=<N> depth=512 card=<card> tok_s=<x> eval_ms=<y> decoded=<k>
#   step_tok_s=<z> accepted=<a> drafted=<d> | <ik's eval line> | <ik's statistics line>
# tok_s, eval_ms and decoded are ik's own `main: eval time` numbers: every emitted token counts
# once, drafted or not, so tok_s is positions per second. ik's clock starts when the first
# generated token is emitted — its compute is the prompt's evaluation, inside `prompt eval time` —
# and that token is in the count, so eval_ms spans decoded - 1 decode steps and tok_s reads
# decoded / (decoded - 1) high. step_tok_s is (decoded - 1) / eval_ms, derived:
# the rate over the same positions as generate_ds41's timed steps (its N - 1 steps after token 0).
# accepted and drafted are ik's `#acc tokens` and `#gen tokens` of the stage's `statistics` line
# (plain: `-`). ik's whole output follows, between `--- ik raw` markers.
#
# Environment: BLOOMERY_IK_SPEC (the dspark arm's stage, default `dspark`; `dspark:n_max=4` sets
# ik's canonical keys), BLOOMERY_IK_NCMOE (replaces --n-cpu-moe in IK_GPU_FLAGS: the 3090 holds
# the dense weights, not six layers of experts), BLOOMERY_IK_DRAFT_PARAMS, BLOOMERY_IK_CTX,
# BLOOMERY_ARM_BOUND (seconds, default 900), BLOOMERY_TOKENIZE_BIN (default
# target/release/bloomery-tokenize), BLOOMERY_TIMING_GPU (timing-card.sh: the card).
#
# Exit codes: 0 row printed, 64 usage, 2 a missing binary or corpus, 3 a stale tokenizer binary,
# 65 the prompt did not round-trip, 1 ik failed or printed no eval line, 124/137 the bound,
# 75 the lease was not free within 30 min.
set -uo pipefail
# shellcheck source=tools/ref/ref-paths.sh
source "${BASH_SOURCE[0]%/*}/ref-paths.sh"
[ "$MODEL_NAME" = deepseek41 ] || {
  echo "ik-draft.sh: the profile is $MODEL_NAME — pick deepseek41 on the Mac side (BLOOMERY_MODEL=deepseek41)" >&2
  exit 64
}
usage() { echo "usage: ik-draft.sh <corpus> <N> [plain|dspark]" >&2; exit 64; }
CORPUS=${1:-}
N=${2:-}
ARM=${3:-dspark}
[ -n "$CORPUS" ] || usage
case $CORPUS in *[!A-Za-z0-9_-]*) usage ;; esac
case $N in '' | *[!0-9]* | 0) usage ;; esac
case $ARM in plain | dspark) ;; *) usage ;; esac
STAGE=${BLOOMERY_IK_SPEC:-dspark}
case $STAGE in dspark | dspark:*) ;; *) echo "ik-draft.sh: BLOOMERY_IK_SPEC must be dspark[:k=v,...], got '$STAGE'" >&2; exit 64 ;; esac
BOUND=${BLOOMERY_ARM_BOUND:-900}
PROMPT_N=512
CTX=${BLOOMERY_IK_CTX:-$(((PROMPT_N + N + 64 + 255) / 256 * 256))}
case $CTX in *[!0-9]*) echo "ik-draft.sh: BLOOMERY_IK_CTX must be a number, got '$CTX'" >&2; exit 64 ;; esac
[ "$CTX" -ge $((PROMPT_N + N)) ] || { echo "ik-draft.sh: ctx $CTX < prompt $PROMPT_N + n $N" >&2; exit 64; }
FLAGS=$IK_GPU_FLAGS
if [ -n "${BLOOMERY_IK_NCMOE:-}" ]; then
  case $BLOOMERY_IK_NCMOE in *[!0-9]*) echo "ik-draft.sh: BLOOMERY_IK_NCMOE must be a number" >&2; exit 64 ;; esac
  FLAGS=$(echo " $FLAGS " | sed -E "s/ (--n-cpu-moe|-ncmoe) [0-9]+ / /")
  FLAGS="${FLAGS# } --n-cpu-moe $BLOOMERY_IK_NCMOE"
  FLAGS=${FLAGS% }
fi
DRAFT_PARAMS=${BLOOMERY_IK_DRAFT_PARAMS:-}

# The card pin (CUDA_VISIBLE_DEVICES), its witness lines and the other-card guard; the lease.
# shellcheck source=tools/ref/timing-card.sh
source "${BASH_SOURCE[0]%/*}/timing-card.sh"
# shellcheck source=tools/ref/lease.sh
source "${BASH_SOURCE[0]%/*}/lease.sh"

CLI=$IK/build/bin/llama-cli
LTOK=$IK/build/bin/llama-tokenize
TOK=${BLOOMERY_TOKENIZE_BIN:-target/release/bloomery-tokenize}
IDS=$BLOOMERY_DATA/engram/corpus-$CORPUS.ids
for f in "$CLI" "$LTOK"; do [ -x "$f" ] || { echo "ik-draft.sh: no $f" >&2; exit 2; }; done
[ -f "$IDS" ] || { echo "ik-draft.sh: no corpus $IDS" >&2; exit 2; }
if [ "$ARM" = dspark ] && [ ! -f "$DSPARK_MODEL" ]; then
  echo "ik-draft.sh: no draft $DSPARK_MODEL" >&2
  exit 2
fi
# Our tokenizer decodes the prompt; the freshness check is the one the timing runners use.
assert_fresh_binary "$TOK" || exit $?

WORK=$(mktemp -d "${TMPDIR:-/tmp}/ik-draft.XXXXXX")
trap 'rm -rf "$WORK"' EXIT

# The 512 ids, one per line, and the comma list the two checks compare against.
head -n "$PROMPT_N" "$IDS" > "$WORK/ids"
[ "$(wc -l < "$WORK/ids")" -eq "$PROMPT_N" ] || { echo "ik-draft.sh: $IDS has fewer than $PROMPT_N ids" >&2; exit 2; }
WANT=$(paste -sd, "$WORK/ids")
"$TOK" -m "$MODEL" --decode -f "$WORK/ids" > "$WORK/text" || { echo "ik-draft.sh: $TOK --decode failed" >&2; exit 1; }
cp "$WORK/text" "$WORK/prompt"
printf '\n' >> "$WORK/prompt"

# first_diff <want list> <got list>: the first position where two comma lists differ.
first_diff() {
  awk -v a="$1" -v b="$2" 'BEGIN{na=split(a,x,","); nb=split(b,y,","); n=(na>nb)?na:nb
    for(i=1;i<=n;i++) if(x[i]!=y[i]) {printf "position %d: want %s got %s (want %d ids, got %d)\n", i-1, x[i], y[i], na, nb; exit}}'
}

# Check 1, before the lease: the text tokenizes back to the ids.
GOT=$("$LTOK" -m "$MODEL" -f "$WORK/text" --ids --log-disable 2> "$WORK/ltok.err" | tr -d '[] \n')
if [ "$GOT" != "$WANT" ]; then
  echo "ik-draft.sh: the decoded prompt does not re-tokenize to the corpus ids under llama-tokenize:" >&2
  echo "    $(first_diff "$WANT" "$GOT")" >&2
  tail -n 5 "$WORK/ltok.err" >&2
  exit 65
fi

CARD_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$TIMING_GPU" | sed 's/^NVIDIA //; s/^GeForce //; s/^RTX //')
CLI_SHA=$(sha256sum "$CLI" | cut -c1-12)
IK_HEAD=$(git -c safe.directory="$IK" -C "$IK" rev-parse --short=8 HEAD 2> /dev/null || echo '?')
IK_DIRTY=$(git -c safe.directory="$IK" -C "$IK" status --porcelain --untracked-files=no 2> /dev/null | wc -l | tr -d ' ')
DRAFT_SHA=-
[ "$ARM" = plain ] || DRAFT_SHA=$(sha256sum "$DSPARK_MODEL" | cut -c1-12)
WITNESS=(head-open indent card busiest model mem pgmajfault)
ik_witness() { echo "    ik: $CLI sha256=$CLI_SHA head=$IK_HEAD dirty_files=$IK_DIRTY"; }

args=(-m "$MODEL")
# shellcheck disable=SC2206
args+=($FLAGS)
args+=(-c "$CTX" -n "$N" -f "$WORK/prompt" --no-escape --temp 0 --ignore-eos --no-display-prompt --verbose-prompt)
if [ "$ARM" = dspark ]; then
  args+=(-md "$DSPARK_MODEL" --spec-type "$STAGE")
  [ -z "$DRAFT_PARAMS" ] || args+=(--draft-params "$DRAFT_PARAMS")
fi

STAGE_SHOWN=none
DRAFT_SHOWN=none
if [ "$ARM" = dspark ]; then
  STAGE_SHOWN=$STAGE
  DRAFT_SHOWN="$DSPARK_MODEL sha256=$DRAFT_SHA draft_params=${DRAFT_PARAMS:-<inherited from the target>}"
fi

lease_take
echo "[config] corpus=$CORPUS ids=$IDS prompt=$PROMPT_N n=$N arm=$ARM stage=$STAGE_SHOWN ctx=$CTX card=$CARD_NAME arm_bound=${BOUND}s"
echo "[config] ik: $CLI model=$MODEL flags=$FLAGS env=$IK_GPU_ENV"
echo "[config] draft: $DRAFT_SHOWN"
echo "[config] timing_gpu=$TIMING_GPU other_gpu=$OTHER_GPU CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[config] prompt check 1 (llama-tokenize): $PROMPT_N ids round-trip"
guard_other
witness "pre ik $ARM $CORPUS n=$N"
ik_witness
t0=$(date +%s)
# shellcheck disable=SC2086
timeout --kill-after=10 "$BOUND" env $IK_GPU_ENV "$CLI" "${args[@]}" < /dev/null > "$WORK/raw" 2>&1
rc=$?
t1=$(date +%s)
witness "post ik $ARM $CORPUS n=$N"
ik_witness
echo "--- ik raw begin (rc $rc, wall $((t1 - t0))s)"
cat "$WORK/raw"
echo "--- ik raw end"

if [ $rc -ne 0 ]; then
  echo "ik-draft.sh: llama-cli rc $rc" >&2
  tail -n 20 "$WORK/raw" >&2
  exit "$rc"
fi

# Check 2: the ids llama-cli itself tokenized the prompt to (--verbose-prompt's list).
PGOT=$(awk -v n="$PROMPT_N" '/number of tokens in prompt = /{on=1; next} on && /^ *[0-9]+ -> \x27/{print $1; if(++k==n) exit}' "$WORK/raw" | paste -sd,)
PCOUNT=$(sed -n 's/.*number of tokens in prompt = \([0-9]*\).*/\1/p' "$WORK/raw" | head -n 1)
if [ "$PCOUNT" != "$PROMPT_N" ] || [ "$PGOT" != "$WANT" ]; then
  echo "ik-draft.sh: llama-cli's prompt is not the corpus ids (it counted ${PCOUNT:-?}):" >&2
  echo "    $(first_diff "$WANT" "$PGOT")" >&2
  exit 65
fi
echo "[check] prompt check 2 (llama-cli --verbose-prompt): $PCOUNT ids, identical"

EVAL=$(grep -E 'main: +eval time = ' "$WORK/raw" | head -n 1)
[ -n "$EVAL" ] || { echo "ik-draft.sh: ik printed no 'main: eval time' line" >&2; exit 1; }
eval_ms=$(echo "$EVAL" | sed -E 's/.*eval time = +([0-9.]+) ms.*/\1/')
tok_s=$(echo "$EVAL" | sed -E 's/.*, +([0-9.]+) tokens per second.*/\1/')
decoded=$(echo "$EVAL" | sed -E 's/.* ms \/ +([0-9]+) tokens.*/\1/')
step_tok_s=$(awk -v k="$decoded" -v ms="$eval_ms" 'BEGIN{if(k>1 && ms>0) printf "%.2f", (k-1)*1e3/ms; else print "-"}')
STATS=-
accepted=-
drafted=-
if [ "$ARM" = dspark ]; then
  STATS=$(grep -E 'statistics dspark: ' "$WORK/raw" | head -n 1)
  [ -n "$STATS" ] || { echo "ik-draft.sh: the dspark arm printed no 'statistics dspark' line — did the stage run?" >&2; exit 1; }
  accepted=$(echo "$STATS" | sed -nE 's/.*#acc tokens = ([0-9]+).*/\1/p')
  drafted=$(echo "$STATS" | sed -nE 's/.*#gen tokens = ([0-9]+).*/\1/p')
fi
echo "ik-draft corpus=$CORPUS arm=$ARM n=$N depth=$PROMPT_N card=$CARD_NAME tok_s=$tok_s eval_ms=$eval_ms decoded=$decoded step_tok_s=$step_tok_s accepted=$accepted drafted=$drafted | ${EVAL#"${EVAL%%[![:space:]]*}"} | ${STATS}"
