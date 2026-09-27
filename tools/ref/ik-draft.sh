#!/usr/bin/env bash
# ik's decode on a real-text prompt under the lease on the timing card (run on the box, lead-only):
# V4.1 plain or with the DSpark draft — or mainline llama.cpp's, plain (arm `lcpp`) — and Qwen3.8
# (qwen4exp) plain or with its shared MTP draft (arm `mtp`), the run the MTP acceptance is read from:
#
#   just time-ik-draft prose 96 dspark
#   just time-lcpp-prompt prose 96
#   just time-ik-mtp prose 256
#   BLOOMERY_MODEL=deepseek41 tools/box.sh 'bash tools/ref/ik-draft.sh <corpus> <N> [plain|dspark|lcpp]'
#   BLOOMERY_MODEL=qwen4exp tools/box.sh 'bash tools/ref/ik-draft.sh <corpus> <N> [plain|mtp]'
#   BLOOMERY_BOX_ENV=BLOOMERY_DRY=1 just time-lcpp-prompt prose 96   # the checks and command lines up
#                                    # to the lease, then exit 0: no load, no timing
#
# The profile picks the arms: deepseek41 plain, dspark (the default) and lcpp; qwen4exp plain and mtp
# (the default). Any other profile, or an arm its profile does not have, is refused (64).
#
# The reference twin of `just time-gpu-ds41 --tokens <512 ids> -n N`: the prompt is the first 512
# ids of the profile's corpus file — deepseek41's $BLOOMERY_DATA/engram/corpus-<corpus>.ids, qwen4exp's
# $BLOOMERY_DATA/qwen4exp/corpus-<corpus>.ids (its own vocabulary; router-trace.sh's rule), whose prose
# file must match the profile's CORPUS_PROSE_SHA256 — so a reference row and one of ours are about
# the same prompt. It blocks two failures: an ik draft row read against one of ours on another prompt (the
# acceptance rate is a property of the text), and a draft row whose prompt ik tokenized to other
# ids than ours (llama-cli takes text, not ids).
#
# The prompt. llama-cli (and mainline's llama-completion) reads text, so the ids are decoded — by our
# tokenizer for deepseek41 (bloomery-tokenize --decode, special tokens rendered as their text), and for
# qwen4exp, whose pre-tokenizer our tokenizer does not load, by tools/ref/gguf-decode.py from the
# model file's own vocabulary under the same piece rule — and checked twice: before
# the lease, llama-tokenize (vocabulary only) must turn the text back into the same 512 ids, and
# after the run, the ids llama-cli printed under --verbose-prompt must be those ids. Either
# mismatch refuses the row (rc 65) with the first differing position. llama-cli's -f drops one
# trailing newline, so its file carries one more than the text; --no-escape keeps a backslash in
# the text a backslash.
#
# The ik run: llama-cli -m MODEL $FLAGS -c CTX -n N --temp 0 --ignore-eos, under `env $ENV`, where
# FLAGS and ENV are the profile's IK_GPU_FLAGS and IK_GPU_ENV (deepseek41) or IK_MTP_FLAGS and
# IK_MTP_ENV (qwen4exp; the profiles' comments say why the environment is load-bearing). --temp 0 is
# ik's greedy sampler; --ignore-eos because generate_ds41 has no end-of-generation stop and
# always emits N tokens. CTX is 512 + N + 64 rounded up to 256 — llama-cli's default context is
# the model's own (over a million positions), not the prompt's. The dspark arm adds
# `-md $DSPARK_MODEL --spec-type $STAGE`. ik builds the DSpark draft's parameters from the
# target's own (common/speculative.cpp: only an MTP stage clears them), so the draft inherits
# -ngl and --n-cpu-moe, and with its three layers under the profile's --n-cpu-moe (IK_NCMOE)
# every draft expert runs on the host; BLOOMERY_IK_DRAFT_PARAMS (passed as ik's --draft-params) changes that.
# The mtp arm adds `-md $MTP_MODEL -ngld 99 --spec-type $STAGE`: an MTP stage drops the target's CPU-MoE
# overrides for the draft, and -ngld 99 puts all of its one layer on the card.
#
# The mtp arm's stages. BLOOMERY_IK_SPEC is a list of `mtp:...` stages separated by `;` (tools/box.sh
# splits BLOOMERY_BOX_ENV on spaces, so a list that crosses it has none) or whitespace, default
# `mtp:n_max=3,p_min=0.0`; each must name n_max and p_min (ik's defaults are not this runner's to
# guess), and each is one llama-cli run, one after the other under one lease. ik's MTP draft
# (common/speculative.cpp mtp_speculative_gen_draft) always proposes the first draft token; it then
# stops before the next one once a proposed token's full-vocabulary softmax maximum is below p_min, and
# at p_min 0 it computes no probability and proposes n_max tokens every window. ik logs no per-token
# probability, so a list of several p_min values is the only view of the gate this reference gives: the
# threshold table below, not a table of acceptance by probability bucket.
#
# The lcpp run: the profile's LCPP tree, $LCPP/build/bin/llama-completion -m MODEL $LCPP_CLI_FLAGS
# -c CTX -n N -f <prompt> --no-escape --temp 0 --ignore-eos --no-display-prompt --verbose-prompt
# -no-cnv. Mainline's llama-cli is the chat front end; llama-completion is the one-shot binary,
# and it turns conversation mode on by itself when the file has a chat template (-no-cnv keeps
# the text a plain completion). --temp 0 is an argmax (src/llama-sampler.cpp, temp <= 0). Both
# checks are the same, against that tree's llama-tokenize (whose -f reads the file verbatim but
# applies escapes unless --no-escape) and llama-completion's --verbose-prompt list, which has
# ik's `N -> 'piece'` shape. No environment: mainline keeps the host experts on the file
# mapping (the profile's LCPP_GPU_FLAGS comment).
#
# The ik build. Every ik row carries `ik_build=<8 hex>[-dirty]`, read from $IK as tools/ref/dump.sh
# reads the build tag of an oracle set, so a row names the build its numbers are of; the witness line
# after each block repeats it with the binary's sha256.
#
# The summary line per run:
#   ik-draft corpus=<c> arm=<a> n=<N> depth=512 card=<card> tok_s=<x> eval_ms=<y> decoded=<k>
#   step_tok_s=<z> accepted=<a> drafted=<d> ik_build=<b> | <ik's eval line> | <ik's statistics line>
# tok_s, eval_ms and decoded are ik's own `main: eval time` numbers: every emitted token counts
# once, drafted or not, so tok_s is positions per second. ik's clock starts when the first
# generated token is emitted — its compute is the prompt's evaluation, inside `prompt eval time` —
# and that token is in the count, so eval_ms spans decoded - 1 decode steps and tok_s reads
# decoded / (decoded - 1) high. step_tok_s is (decoded - 1) / eval_ms, derived:
# the rate over the same positions as generate_ds41's timed steps (its N - 1 steps after token 0).
# accepted and drafted are ik's `#acc tokens` and `#gen tokens` of the stage's `statistics` line
# (plain: `-`). ik's whole output follows, between `--- ik raw` markers.
#
# The mtp arm prints one more line per stage, from the same statistics line (windows = `#gen drafts`,
# ik's count of verify passes that carried a draft):
#   ik-mtp corpus=<c> n=<N> stage=<s> n_max=<m> p_min=<p> windows=<w> E=<e> a1=<r> rows=<q> a_geo=<g>
#   ik_build=<b>
# E = 1 + accepted / windows, the tokens a window keeps with the target's own; a1 = `#acc drafts` /
# windows, the share of windows whose first draft token was kept; rows = drafted / windows; a_geo the
# constant per-step acceptance a with 1 + a + ... + a^(m) = E (bisection), the design's `a` at a full
# head. Two [check] lines guard the reading: rows != n_max at p_min 0 (ik proposes n_max every window
# then), and decoded - 1 against windows + accepted, which differ by the windows without a draft and
# the last window's overshoot (at most n_max): a gap past n_max means windows are not what E divides by.
# With more than one stage, the threshold table follows, rows in the order the stages were given:
#   ik-mtp-threshold p_min=<p> E=<e> rows=<q> drafted=<d> accepted=<a> marginal=<r|->
# marginal is (accepted' - accepted) / (drafted' - drafted) against the row above it: the acceptance
# of the draft tokens the step between the two thresholds removes. The window starts move between
# stages (another cut, another alignment), so it is an estimate over the same N positions, not a
# per-token count.
#
# The lcpp arm prints the same line under the name lcpp-prompt, from common_perf_print's
# `eval time = <ms> ms / <runs> runs` (common/sampling.cpp). Mainline counts decode calls, not
# emitted tokens: the first token comes out of the prompt's evaluation and the last one is never
# decoded, so runs is N - 1, decoded is runs + 1, and its tokens per second is already the step
# rate — tok_s and step_tok_s are the same number. runs != N - 1 is printed as a [check] line.
# Its output sits between `--- lcpp raw` markers.
#
# Environment: BLOOMERY_IK_SPEC (the dspark arm's stage, default `dspark`; `dspark:n_max=4` sets
# ik's canonical keys; the mtp arm's stage list, above), BLOOMERY_IK_NCMOE (replaces --n-cpu-moe in
# the ik flags: the 3090 holds the dense weights, not six layers of experts), BLOOMERY_LCPP_NCMOE (the
# same for LCPP_CLI_FLAGS), BLOOMERY_IK_DRAFT_PARAMS, BLOOMERY_IK_CTX (both engines' context),
# BLOOMERY_ARM_BOUND (seconds per run, default 900), BLOOMERY_TOKENIZE_BIN (default
# target/release/bloomery-tokenize), BLOOMERY_TIMING_GPU (timing-card.sh: the card), BLOOMERY_DRY=1
# (above: every check that runs before the lease, the command lines, exit 0).
#
# Exit codes: 0 every row printed, 64 usage, 2 a missing binary or corpus, 3 a stale tokenizer binary,
# 65 the prompt did not round-trip or the corpus is not the pinned one, 1 the reference failed or
# printed no eval or statistics line, 124/137 the bound, 75 the lease was not free within 30 min.
# The mtp arm stops at the first stage that fails.
set -uo pipefail
# shellcheck source=tools/ref/ref-paths.sh
source "${BASH_SOURCE[0]%/*}/ref-paths.sh"
case $MODEL_NAME in
  deepseek41) ARMS='plain dspark lcpp' DEFAULT_ARM=dspark CORPORA=$BLOOMERY_DATA/engram ;;
  qwen4exp) ARMS='plain mtp' DEFAULT_ARM=mtp CORPORA=$BLOOMERY_DATA/qwen4exp ;;
  *)
    echo "ik-draft.sh: the profile is $MODEL_NAME — pick deepseek41 or qwen4exp on the Mac side (BLOOMERY_MODEL=…)" >&2
    exit 64
    ;;
esac
usage() { echo "usage: ik-draft.sh <corpus> <N> [${ARMS// /|}]   (profile $MODEL_NAME)" >&2; exit 64; }
CORPUS=${1:-}
N=${2:-}
ARM=${3:-$DEFAULT_ARM}
[ -n "$CORPUS" ] || usage
case $CORPUS in *[!A-Za-z0-9_-]*) usage ;; esac
case $N in '' | *[!0-9]* | 0) usage ;; esac
case " $ARMS " in *" $ARM "*) ;; *) usage ;; esac
case $ARM in lcpp) ENGINE=lcpp ;; *) ENGINE=ik ;; esac
DRY=${BLOOMERY_DRY:-}
STAGES=(none)
case $ARM in
  dspark)
    STAGE=${BLOOMERY_IK_SPEC:-dspark}
    case $STAGE in dspark | dspark:*) ;; *) echo "ik-draft.sh: BLOOMERY_IK_SPEC must be dspark[:k=v,...], got '$STAGE'" >&2; exit 64 ;; esac
    STAGES=("$STAGE")
    ;;
  mtp)
    SPEC_LIST=${BLOOMERY_IK_SPEC:-mtp:n_max=3,p_min=0.0}
    read -r -a STAGES <<< "${SPEC_LIST//;/ }"
    [ "${#STAGES[@]}" -gt 0 ] || { echo "ik-draft.sh: BLOOMERY_IK_SPEC names no mtp stage" >&2; exit 64; }
    for s in "${STAGES[@]}"; do
      case $s in
        mtp:*n_max=*) ;;
        *) echo "ik-draft.sh: each mtp stage is mtp:n_max=<m>,p_min=<p>[,...], got '$s'" >&2; exit 64 ;;
      esac
      [[ $s =~ [:,]n_max=[1-9][0-9]*(,|$) && $s =~ [:,]p_min=[0-9]+(\.[0-9]+)?(,|$) ]] || {
        echo "ik-draft.sh: the mtp stage '$s' must name n_max (an integer >= 1) and p_min (a decimal >= 0)" >&2
        exit 64
      }
    done
    ;;
esac
BOUND=${BLOOMERY_ARM_BOUND:-900}
PROMPT_N=512
CTX=${BLOOMERY_IK_CTX:-$(((PROMPT_N + N + 64 + 255) / 256 * 256))}
case $CTX in *[!0-9]*) echo "ik-draft.sh: BLOOMERY_IK_CTX must be a number, got '$CTX'" >&2; exit 64 ;; esac
[ "$CTX" -ge $((PROMPT_N + N)) ] || { echo "ik-draft.sh: ctx $CTX < prompt $PROMPT_N + n $N" >&2; exit 64; }
if [ "$ENGINE" = lcpp ]; then
  FLAGS=$LCPP_CLI_FLAGS NCMOE_VAR=BLOOMERY_LCPP_NCMOE
  RUN_ENV=
elif [ "$MODEL_NAME" = qwen4exp ]; then
  FLAGS=$IK_MTP_FLAGS NCMOE_VAR=BLOOMERY_IK_NCMOE
  RUN_ENV=$IK_MTP_ENV
else
  FLAGS=$IK_GPU_FLAGS NCMOE_VAR=BLOOMERY_IK_NCMOE
  RUN_ENV=$IK_GPU_ENV
fi
NCMOE=${!NCMOE_VAR:-}
if [ -n "$NCMOE" ]; then
  case $NCMOE in *[!0-9]*) echo "ik-draft.sh: $NCMOE_VAR must be a number" >&2; exit 64 ;; esac
  FLAGS=$(echo " $FLAGS " | sed -E "s/ (--n-cpu-moe|-ncmoe) [0-9]+ / /")
  FLAGS="${FLAGS# } --n-cpu-moe $NCMOE"
  FLAGS=${FLAGS% }
fi
DRAFT_PARAMS=${BLOOMERY_IK_DRAFT_PARAMS:-}

# The card pin (CUDA_VISIBLE_DEVICES), its witness lines and the other-card guard; the lease.
# shellcheck source=tools/ref/timing-card.sh
source "${BASH_SOURCE[0]%/*}/timing-card.sh"
# shellcheck source=tools/ref/lease.sh
source "${BASH_SOURCE[0]%/*}/lease.sh"

if [ "$ENGINE" = ik ]; then
  TREE=$IK CLI=$IK/build/bin/llama-cli LTOK=$IK/build/bin/llama-tokenize
  LTOK_ARGS=(--ids --log-disable)
else
  TREE=$LCPP CLI=$LCPP/build/bin/llama-completion LTOK=$LCPP/build/bin/llama-tokenize
  LTOK_ARGS=(--ids --log-disable --no-escape)
fi
case $ARM in
  dspark) DRAFT_FILE=$DSPARK_MODEL ;;
  mtp) DRAFT_FILE=$MTP_MODEL ;;
  *) DRAFT_FILE= ;;
esac
TOK=${BLOOMERY_TOKENIZE_BIN:-target/release/bloomery-tokenize}
DECODER=${BASH_SOURCE[0]%/*}/gguf-decode.py
IDS=$CORPORA/corpus-$CORPUS.ids
WORK=$(mktemp -d "${TMPDIR:-/tmp}/ik-draft.XXXXXX") || { echo "ik-draft.sh: no scratch directory under ${TMPDIR:-/tmp}" >&2; exit 2; }
trap 'rm -rf "$WORK"' EXIT

# decode_prompt: the prompt ids in $WORK/ids to text on stdout.
decode_prompt() {
  if [ "$MODEL_NAME" = qwen4exp ]; then
    python3 "$DECODER" -m "$MODEL" -f "$WORK/ids"
  else
    "$TOK" -m "$MODEL" --decode -f "$WORK/ids"
  fi
}
if [ "$MODEL_NAME" = qwen4exp ]; then
  DECODE_SHOWN="python3 $DECODER -m $MODEL -f $WORK/ids > $WORK/text"
else
  DECODE_SHOWN="$TOK -m $MODEL --decode -f $WORK/ids > $WORK/text"
fi

# stage_args <stage>: the arm's draft arguments for one stage, into SARGS.
stage_args() {
  SARGS=()
  case $ARM in
    dspark) SARGS=(-md "$DSPARK_MODEL" --spec-type "$1") ;;
    mtp) SARGS=(-md "$MTP_MODEL" -ngld 99 --spec-type "$1") ;;
    *) return 0 ;;
  esac
  [ -z "$DRAFT_PARAMS" ] || SARGS+=(--draft-params "$DRAFT_PARAMS")
}

args=(-m "$MODEL")
# shellcheck disable=SC2206
args+=($FLAGS)
args+=(-c "$CTX" -n "$N" -f "$WORK/prompt" --no-escape --temp 0 --ignore-eos --no-display-prompt --verbose-prompt)
[ "$ENGINE" = ik ] || args+=(-no-cnv)
if [ -n "$DRY" ]; then
  echo "[dry] engine=$ENGINE arm=$ARM corpus=$CORPUS ids=$IDS prompt=$PROMPT_N n=$N ctx=$CTX timing_gpu=$TIMING_GPU CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  echo "[dry] decode: $DECODE_SHOWN"
  echo "[dry] check 1: $LTOK -m $MODEL -f $WORK/text ${LTOK_ARGS[*]}"
  for s in "${STAGES[@]}"; do
    stage_args "$s"
    echo "[dry] run: timeout --kill-after=10 $BOUND env $RUN_ENV $CLI ${args[*]}${SARGS[*]:+ ${SARGS[*]}} < /dev/null"
  done
fi
for f in "$CLI" "$LTOK"; do
  [ -x "$f" ] && continue
  echo "ik-draft.sh: no $f" >&2
  [ "$ENGINE" = ik ] || [ "$f" != "$LTOK" ] || echo "    build it with: cmake --build $LCPP/build --target llama-tokenize" >&2
  exit 2
done
[ -f "$IDS" ] || { echo "ik-draft.sh: no corpus $IDS" >&2; exit 2; }
if [ -n "$DRAFT_FILE" ] && [ ! -f "$DRAFT_FILE" ]; then
  echo "ik-draft.sh: no draft $DRAFT_FILE" >&2
  exit 2
fi
if [ "$MODEL_NAME" = qwen4exp ]; then
  # The prose ids are pinned by the profile: a regenerated file would be another prompt.
  if [ "$CORPUS" = prose ]; then
    got=$(sha256sum "$IDS" | cut -d' ' -f1)
    [ "$got" = "$CORPUS_PROSE_SHA256" ] || {
      echo "ik-draft.sh: $IDS has sha256 $got, not the profile's CORPUS_PROSE_SHA256 $CORPUS_PROSE_SHA256" >&2
      exit 65
    }
  fi
else
  # Our tokenizer decodes the prompt; the freshness check is the one the timing runners use.
  assert_fresh_binary "$TOK" || exit $?
fi

# The 512 ids, one per line, and the comma list the two checks compare against.
head -n "$PROMPT_N" "$IDS" > "$WORK/ids"
[ "$(wc -l < "$WORK/ids")" -eq "$PROMPT_N" ] || { echo "ik-draft.sh: $IDS has fewer than $PROMPT_N ids" >&2; exit 2; }
WANT=$(paste -sd, "$WORK/ids")
decode_prompt > "$WORK/text" || { echo "ik-draft.sh: the prompt decode ($DECODE_SHOWN) failed" >&2; exit 1; }
cp "$WORK/text" "$WORK/prompt"
printf '\n' >> "$WORK/prompt"

# first_diff <want list> <got list>: the first position where two comma lists differ.
first_diff() {
  awk -v a="$1" -v b="$2" 'BEGIN{na=split(a,x,","); nb=split(b,y,","); n=(na>nb)?na:nb
    for(i=1;i<=n;i++) if(x[i]!=y[i]) {printf "position %d: want %s got %s (want %d ids, got %d)\n", i-1, x[i], y[i], na, nb; exit}}'
}

# Check 1, before the lease: the text tokenizes back to the ids.
GOT=$("$LTOK" -m "$MODEL" -f "$WORK/text" "${LTOK_ARGS[@]}" 2> "$WORK/ltok.err" | tr -d '[] \n')
if [ "$GOT" != "$WANT" ]; then
  echo "ik-draft.sh: the decoded prompt does not re-tokenize to the corpus ids under llama-tokenize:" >&2
  echo "    $(first_diff "$WANT" "$GOT")" >&2
  tail -n 5 "$WORK/ltok.err" >&2
  exit 65
fi

CARD_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$TIMING_GPU" | sed 's/^NVIDIA //; s/^GeForce //; s/^RTX //')
CLI_SHA=$(sha256sum "$CLI" | cut -c1-12)
# GIT_OPTIONAL_LOCKS=0 keeps `git status` from rewriting the index of a tree this root process does not own.
TREE_HEAD=$(git -c safe.directory="$TREE" -C "$TREE" rev-parse --short=9 HEAD 2> /dev/null || echo '?')
TREE_DIRTY=$(GIT_OPTIONAL_LOCKS=0 git -c safe.directory="$TREE" -C "$TREE" status --porcelain --untracked-files=no 2> /dev/null | wc -l | tr -d ' ')
# The build tag as tools/ref/dump.sh records an oracle set's: 8 hex digits, `-dirty` for a tree with
# uncommitted changes to tracked files.
BUILD=$(git -c safe.directory='*' -C "$TREE" rev-parse --short=8 HEAD 2> /dev/null || echo unknown)
if [ "$BUILD" != unknown ] && ! GIT_OPTIONAL_LOCKS=0 git -c safe.directory='*' -C "$TREE" diff --quiet HEAD 2> /dev/null; then
  BUILD="$BUILD-dirty"
fi
DRAFT_SHA=-
[ -z "$DRAFT_FILE" ] || DRAFT_SHA=$(sha256sum "$DRAFT_FILE" | cut -c1-12)
WITNESS=(head-open indent card busiest model mem pgmajfault)
ref_witness() { echo "    $ENGINE: $CLI sha256=$CLI_SHA head=$TREE_HEAD dirty_files=$TREE_DIRTY build=$BUILD"; }

DRAFT_SHOWN=none
[ -z "$DRAFT_FILE" ] || DRAFT_SHOWN="$DRAFT_FILE sha256=$DRAFT_SHA draft_params=${DRAFT_PARAMS:-<inherited from the target>}"

if [ -n "$DRY" ]; then
  echo "[dry] prompt check 1 ($LTOK): $PROMPT_N ids round-trip"
  ref_witness | sed 's/^   /[dry]/'
  echo "[dry] stops before the lease"
  exit 0
fi

lease_take
echo "[config] corpus=$CORPUS ids=$IDS prompt=$PROMPT_N n=$N arm=$ARM stages=${STAGES[*]} ctx=$CTX card=$CARD_NAME arm_bound=${BOUND}s"
echo "[config] $ENGINE: $CLI model=$MODEL flags=$FLAGS env=$RUN_ENV build=$BUILD"
echo "[config] draft: $DRAFT_SHOWN"
echo "[config] timing_gpu=$TIMING_GPU other_gpu=$OTHER_GPU CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[config] prompt check 1 ($LTOK): $PROMPT_N ids round-trip"

# stage_key <stage> <key>: a stage's numeric value for a canonical key (n_max, p_min).
stage_key() { sed -nE "s/.*[:,]$2=([0-9.]+)(,.*)?$/\1/p" <<< "$1"; }

# run_stage <stage>: one run and its row(s); returns nonzero on the first failure. The mtp arm's
# numbers are kept in T_* for the threshold table.
T_PMIN=() T_E=() T_ROWS=() T_DRAFTED=() T_ACC=()
run_stage() {
  local stage=$1 tag rc t0 t1
  stage_args "$stage"
  tag="$ENGINE $ARM $CORPUS n=$N${SARGS[*]:+ $stage}"
  echo "[config] stage: $stage"
  guard_other
  witness "pre $tag"
  ref_witness
  t0=$(date +%s)
  # shellcheck disable=SC2086
  timeout --kill-after=10 "$BOUND" env $RUN_ENV "$CLI" "${args[@]}" ${SARGS[@]+"${SARGS[@]}"} < /dev/null > "$WORK/raw" 2>&1
  rc=$?
  t1=$(date +%s)
  witness "post $tag"
  ref_witness
  echo "--- $ENGINE raw begin (rc $rc, wall $((t1 - t0))s)"
  cat "$WORK/raw"
  echo "--- $ENGINE raw end"

  if [ $rc -ne 0 ]; then
    echo "ik-draft.sh: ${CLI##*/} rc $rc" >&2
    tail -n 20 "$WORK/raw" >&2
    return "$rc"
  fi

  # Check 2: the ids llama-cli itself tokenized the prompt to (--verbose-prompt's list).
  local pgot pcount
  pgot=$(awk -v n="$PROMPT_N" '/number of tokens in prompt = /{on=1; next} on && /^ *[0-9]+ -> \x27/{print $1; if(++k==n) exit}' "$WORK/raw" | paste -sd,)
  pcount=$(sed -n 's/.*number of tokens in prompt = \([0-9]*\).*/\1/p' "$WORK/raw" | head -n 1)
  if [ "$pcount" != "$PROMPT_N" ] || [ "$pgot" != "$WANT" ]; then
    echo "ik-draft.sh: ${CLI##*/}'s prompt is not the corpus ids (it counted ${pcount:-?}):" >&2
    echo "    $(first_diff "$WANT" "$pgot")" >&2
    return 65
  fi
  echo "[check] prompt check 2 (${CLI##*/} --verbose-prompt): $pcount ids, identical"

  local eval_line eval_ms runs tok_s step_tok_s decoded
  if [ "$ENGINE" = lcpp ]; then
    eval_line=$(grep -E '^[a-z_]+: +eval time = ' "$WORK/raw" | head -n 1)
    [ -n "$eval_line" ] || { echo "ik-draft.sh: llama-completion printed no 'eval time' line" >&2; return 1; }
    eval_ms=$(echo "$eval_line" | sed -E 's/.*eval time = +([0-9.]+) ms.*/\1/')
    runs=$(echo "$eval_line" | sed -E 's/.* ms \/ +([0-9]+) runs.*/\1/')
    tok_s=$(echo "$eval_line" | sed -E 's/.*, +([0-9.]+) tokens per second.*/\1/')
    step_tok_s=$(awk -v k="$runs" -v ms="$eval_ms" 'BEGIN{if(k>0 && ms>0) printf "%.2f", k*1e3/ms; else print "-"}')
    [ "$runs" = $((N - 1)) ] || echo "[check] llama-completion ran $runs decode steps, not n - 1 = $((N - 1))"
    echo "lcpp-prompt corpus=$CORPUS arm=$ARM n=$N depth=$PROMPT_N card=$CARD_NAME tok_s=$tok_s eval_ms=$eval_ms decoded=$((runs + 1)) step_tok_s=$step_tok_s accepted=- drafted=- | ${eval_line#"${eval_line%%[![:space:]]*}"} | -"
    return 0
  fi
  eval_line=$(grep -E 'main: +eval time = ' "$WORK/raw" | head -n 1)
  [ -n "$eval_line" ] || { echo "ik-draft.sh: ik printed no 'main: eval time' line" >&2; return 1; }
  eval_ms=$(echo "$eval_line" | sed -E 's/.*eval time = +([0-9.]+) ms.*/\1/')
  tok_s=$(echo "$eval_line" | sed -E 's/.*, +([0-9.]+) tokens per second.*/\1/')
  decoded=$(echo "$eval_line" | sed -E 's/.* ms \/ +([0-9]+) tokens.*/\1/')
  step_tok_s=$(awk -v k="$decoded" -v ms="$eval_ms" 'BEGIN{if(k>1 && ms>0) printf "%.2f", (k-1)*1e3/ms; else print "-"}')
  local stats=- accepted=- drafted=- windows acc_drafts
  if [ "$ARM" != plain ]; then
    stats=$(grep -E "statistics $ARM: " "$WORK/raw" | head -n 1)
    [ -n "$stats" ] || { echo "ik-draft.sh: the $ARM arm printed no 'statistics $ARM' line — did the stage run?" >&2; return 1; }
    accepted=$(echo "$stats" | sed -nE 's/.*#acc tokens = ([0-9]+).*/\1/p')
    drafted=$(echo "$stats" | sed -nE 's/.*#gen tokens = ([0-9]+).*/\1/p')
  fi
  echo "ik-draft corpus=$CORPUS arm=$ARM n=$N depth=$PROMPT_N card=$CARD_NAME tok_s=$tok_s eval_ms=$eval_ms decoded=$decoded step_tok_s=$step_tok_s accepted=$accepted drafted=$drafted ik_build=$BUILD | ${eval_line#"${eval_line%%[![:space:]]*}"} | ${stats}"
  [ "$ARM" = mtp ] || return 0

  windows=$(echo "$stats" | sed -nE 's/.*#gen drafts = ([0-9]+).*/\1/p')
  acc_drafts=$(echo "$stats" | sed -nE 's/.*#acc drafts = ([0-9]+).*/\1/p')
  if [ -z "$windows" ] || [ -z "$acc_drafts" ] || [ -z "$accepted" ] || [ -z "$drafted" ] || [ "$windows" = 0 ]; then
    echo "ik-draft.sh: the statistics line has no drafts to divide by: $stats" >&2
    return 1
  fi
  local n_max p_min line
  n_max=$(stage_key "$stage" n_max)
  p_min=$(stage_key "$stage" p_min)
  line=$(awk -v w="$windows" -v ad="$acc_drafts" -v acc="$accepted" -v gen="$drafted" -v m="$n_max" '
    function ev(a,   s, t, i) { s = 1; t = 1; for (i = 1; i <= m; i++) { t *= a; s += t } return s }
    BEGIN {
      e = 1 + acc / w
      lo = 0; hi = 1
      if (e >= ev(1)) g = "1.000"; else { for (i = 0; i < 60; i++) { mid = (lo + hi) / 2; if (ev(mid) < e) lo = mid; else hi = mid } g = sprintf("%.3f", lo) }
      printf "windows=%d E=%.3f a1=%.3f rows=%.3f a_geo=%s", w, e, ad / w, gen / w, g
    }')
  echo "ik-mtp corpus=$CORPUS n=$N stage=$stage n_max=$n_max p_min=$p_min $line ik_build=$BUILD"
  awk -v p="$p_min" -v w="$windows" -v gen="$drafted" -v m="$n_max" 'BEGIN { exit !(p + 0 == 0 && gen != m * w) }' &&
    echo "[check] p_min 0 but rows $drafted / $windows != n_max $n_max: ik did not propose n_max every window"
  local gap=$((decoded - 1 - windows - accepted))
  [ "${gap#-}" -le "$n_max" ] ||
    echo "[check] decoded - 1 = $((decoded - 1)) but windows + accepted = $((windows + accepted)): a gap of $gap is more than n_max $n_max — E's windows are not the verify passes"
  T_PMIN+=("$p_min") T_E+=("$(sed -nE 's/.* E=([0-9.]+) .*/\1/p' <<< "$line")")
  T_ROWS+=("$(sed -nE 's/.* rows=([0-9.]+) .*/\1/p' <<< "$line")") T_DRAFTED+=("$drafted") T_ACC+=("$accepted")
  return 0
}

for s in "${STAGES[@]}"; do
  run_stage "$s" || exit $?
done

if [ "${#T_PMIN[@]}" -gt 1 ]; then
  echo "--- mtp threshold table (ik's rule: the first draft token always, then stop below p_min; p is ik's full-vocabulary softmax maximum)"
  for i in "${!T_PMIN[@]}"; do
    marginal=-
    if [ "$i" -gt 0 ]; then
      marginal=$(awk -v a="${T_ACC[i - 1]}" -v b="${T_ACC[i]}" -v c="${T_DRAFTED[i - 1]}" -v d="${T_DRAFTED[i]}" \
        'BEGIN { if (c != d) printf "%.3f", (a - b) / (c - d); else print "-" }')
    fi
    echo "ik-mtp-threshold p_min=${T_PMIN[i]} E=${T_E[i]} rows=${T_ROWS[i]} drafted=${T_DRAFTED[i]} accepted=${T_ACC[i]} marginal=$marginal"
  done
fi
