#!/usr/bin/env bash
# ik's V4.1 decode on a real-text prompt, plain or with the DSpark draft — or mainline llama.cpp's,
# plain (arm `lcpp`) — under the lease on the timing card (run on the box, lead-only):
#
#   just time-ik-draft prose 96 dspark
#   just time-lcpp-prompt prose 96
#   BLOOMERY_MODEL=deepseek41 tools/box.sh 'bash tools/ref/ik-draft.sh <corpus> <N> [plain|dspark|lcpp]'
#   BLOOMERY_BOX_ENV=BLOOMERY_DRY=1 just time-lcpp-prompt prose 96   # the checks and command lines up
#                                    # to the lease, then exit 0: no load, no timing
#
# The reference twin of `just time-gpu-ds41 --tokens <512 ids> -n N`: the prompt is the first 512
# ids of $BLOOMERY_DATA/engram/corpus-<corpus>.ids, so a reference row and one of ours are about
# the same prompt. It blocks two failures: an ik draft row read against one of ours on another prompt (the
# acceptance rate is a property of the text), and a draft row whose prompt ik tokenized to other
# ids than ours (llama-cli takes text, not ids).
#
# The prompt. llama-cli (and mainline's llama-completion) reads text, so the ids are decoded by our tokenizer
# (bloomery-tokenize --decode, special tokens rendered as their text) and checked twice: before
# the lease, llama-tokenize (vocabulary only) must turn the text back into the same 512 ids, and
# after the run, the ids llama-cli printed under --verbose-prompt must be those ids. Either
# mismatch refuses the row (rc 65) with the first differing position. llama-cli's -f drops one
# trailing newline, so its file carries one more than the text; --no-escape keeps a backslash in
# the text a backslash.
#
# The ik run: llama-cli -m MODEL $IK_GPU_FLAGS -c CTX -n N --temp 0 --ignore-eos, under
# `env $IK_GPU_ENV` (the profile's comment says why that variable is load-bearing). --temp 0 is
# ik's greedy sampler; --ignore-eos because generate_ds41 has no end-of-generation stop and
# always emits N tokens. CTX is 512 + N + 64 rounded up to 256 — llama-cli's default context is
# the model's own (over a million positions), not the prompt's. The dspark arm adds
# `-md $DSPARK_MODEL --spec-type $STAGE`. ik builds the DSpark draft's parameters from the
# target's own (common/speculative.cpp: only an MTP stage clears them), so the draft inherits
# -ngl and --n-cpu-moe, and with its three layers under the profile's --n-cpu-moe (IK_NCMOE)
# every draft expert runs on the host; BLOOMERY_IK_DRAFT_PARAMS (passed as ik's --draft-params) changes that.
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
# The lcpp arm prints the same line under the name lcpp-prompt, from common_perf_print's
# `eval time = <ms> ms / <runs> runs` (common/sampling.cpp). Mainline counts decode calls, not
# emitted tokens: the first token comes out of the prompt's evaluation and the last one is never
# decoded, so runs is N - 1, decoded is runs + 1, and its tokens per second is already the step
# rate — tok_s and step_tok_s are the same number. runs != N - 1 is printed as a [check] line.
# Its output sits between `--- lcpp raw` markers.
#
# The warm-up. Under the lease the command runs twice: first once, discarded, then the timed run, the
# same command line on the same prompt — the warm-up reads the pages the timed run will (the file
# mapping, the host experts, the engram rows its ids touch), as the depth runners' warm-ups and server
# arms' discarded request do (docs/fair-measure.md 2.1). Each run sits between its own witness blocks
# (`pre warmup …`, `pre …`); a warm-up that fails ends the script with its rc and its tail, before the
# timed run. The cold tag is the depth runners' (tools/ref/cold-blocks.sh): /proc/vmstat's pgmajfault
# across the timed process (the whole process: llama-cli prints no line where its clock starts), priced at
# COLD_US a fault against the window W = eval_ms, the clock the row's rate comes from; the summary line
# ends in `majflt=<n> (whole process; ≤ <x> % of W <w> s)` and, at COLD_PCT or more, ` [cold]`.
#
# Two cards. BLOOMERY_TIMING_CARDS=a6000+3090 (timing-card.sh's two-card mode) runs the arm on both cards,
# the A6000 as device 0 and the 3090 as device 1, for the separate "A6000+3090" table. The engine's flags
# must place the model over the two cards (-ts or -ot), or the arm is refused by name before anything
# runs: the lcpp arm's LCPP_CLI_FLAGS carry the profile's two-card -ts; the profile has no ik two-card
# line, so an ik arm runs only with an IK_GPU_FLAGS the caller sets. Before the lease timing_cards_precheck refuses a card that does not answer
# or answers under another name (69), a 3090 off its 250 W cap (78) and a kernel journal it cannot read
# (69); every witness block prints both cards' limits and clocks, the 3090's cap and the Xid count since
# the lease was taken (witness_cards); a compute process on either card is waited out (guard_cards); after
# the warm-up and after the timed run an NVRM Xid, a card lost or off its cap, or an engine whose
# ggml_cuda_init lines do not show the A6000 and the 3090 as devices 0 and 1 ends the script (rc 1) with
# the reason. The summary line's card field reads `A6000+3090`.
#
# Environment: BLOOMERY_IK_SPEC (the dspark arm's stage, default `dspark`; `dspark:n_max=4` sets
# ik's canonical keys), BLOOMERY_IK_NCMOE (replaces --n-cpu-moe in IK_GPU_FLAGS: the 3090 holds
# the dense weights, not six layers of experts; ik's llama-cli hands the count to its loader, which
# keeps the last N expert layers on the host, not llama-bench's first N — the profile's IK_GPU_FLAGS
# comment), BLOOMERY_LCPP_NCMOE (the same for LCPP_CLI_FLAGS; mainline's is the first N in both binaries),
# BLOOMERY_IK_DRAFT_PARAMS, BLOOMERY_IK_CTX (both engines' context),
# BLOOMERY_ARM_BOUND (seconds, default 900, each run's), BLOOMERY_TOKENIZE_BIN (default
# target/release/bloomery-tokenize), BLOOMERY_TIMING_GPU (timing-card.sh: the card), BLOOMERY_TIMING_CARDS
# (a6000+3090: the two-card mode above), BLOOMERY_DRY=1
# (above: every check that runs before the lease, the command lines, exit 0).
#
# Exit codes: 0 row printed, 64 usage (a two-card run whose flags place nothing on two cards included),
# 2 a missing binary or corpus or no pgmajfault counter, 3 a stale tokenizer binary, 65 the prompt did
# not round-trip, 1 the reference failed or printed no eval line, or a two-card check after a run failed,
# 124/137 the bound, 69/78 a two-card precheck, 75 the lease was not free within 30 min or a card stayed busy.
set -uo pipefail
# shellcheck source=tools/ref/ref-paths.sh
source "${BASH_SOURCE[0]%/*}/ref-paths.sh"
[ "$MODEL_NAME" = deepseek41 ] || {
  echo "ik-draft.sh: the profile is $MODEL_NAME — pick deepseek41 on the Mac side (BLOOMERY_MODEL=deepseek41)" >&2
  exit 64
}
usage() { echo "usage: ik-draft.sh <corpus> <N> [plain|dspark|lcpp]" >&2; exit 64; }
CORPUS=${1:-}
N=${2:-}
ARM=${3:-dspark}
[ -n "$CORPUS" ] || usage
case $CORPUS in *[!A-Za-z0-9_-]*) usage ;; esac
case $N in '' | *[!0-9]* | 0) usage ;; esac
case $ARM in plain | dspark) ENGINE=ik ;; lcpp) ENGINE=lcpp ;; *) usage ;; esac
DRY=${BLOOMERY_DRY:-}
STAGE=${BLOOMERY_IK_SPEC:-dspark}
case $STAGE in dspark | dspark:*) ;; *) echo "ik-draft.sh: BLOOMERY_IK_SPEC must be dspark[:k=v,...], got '$STAGE'" >&2; exit 64 ;; esac
BOUND=${BLOOMERY_ARM_BOUND:-900}
PROMPT_N=512
CTX=${BLOOMERY_IK_CTX:-$(((PROMPT_N + N + 64 + 255) / 256 * 256))}
case $CTX in *[!0-9]*) echo "ik-draft.sh: BLOOMERY_IK_CTX must be a number, got '$CTX'" >&2; exit 64 ;; esac
[ "$CTX" -ge $((PROMPT_N + N)) ] || { echo "ik-draft.sh: ctx $CTX < prompt $PROMPT_N + n $N" >&2; exit 64; }
if [ "$ENGINE" = ik ]; then
  FLAGS=$IK_GPU_FLAGS NCMOE_VAR=BLOOMERY_IK_NCMOE
  RUN_ENV=$IK_GPU_ENV
else
  FLAGS=$LCPP_CLI_FLAGS NCMOE_VAR=BLOOMERY_LCPP_NCMOE
  RUN_ENV=
fi
NCMOE=${!NCMOE_VAR:-}
if [ -n "$NCMOE" ]; then
  case $NCMOE in *[!0-9]*) echo "ik-draft.sh: $NCMOE_VAR must be a number" >&2; exit 64 ;; esac
  FLAGS=$(echo " $FLAGS " | sed -E "s/ (--n-cpu-moe|-ncmoe) [0-9]+ / /")
  FLAGS="${FLAGS# } --n-cpu-moe $NCMOE"
  FLAGS=${FLAGS% }
fi
DRAFT_PARAMS=${BLOOMERY_IK_DRAFT_PARAMS:-}

# The card pin (CUDA_VISIBLE_DEVICES), its witness lines and the other-card guard; the lease. This runner
# has the two-card mode (the header's Two cards).
TIMING_CARDS_RUNNER=1
# shellcheck source=tools/ref/timing-card.sh
source "${BASH_SOURCE[0]%/*}/timing-card.sh"
timing_cards_mode || exit $?
# shellcheck source=tools/ref/lease.sh
source "${BASH_SOURCE[0]%/*}/lease.sh"
# The cold tag (COLD_US, COLD_PCT, cold_check) and the fault counter (majflt_now, majflt_require).
# shellcheck source=tools/ref/cold-blocks.sh
source "${BASH_SOURCE[0]%/*}/cold-blocks.sh" || exit 2
if [ -n "$TIMING_CARDS" ]; then
  case " $FLAGS " in
    *" -ts "* | *" --tensor-split "* | *" -ot "* | *" --override-tensor "*) ;;
    *)
      echo "ik-draft.sh: BLOOMERY_TIMING_CARDS=a6000+3090 and the $ENGINE flags place nothing on two cards ($FLAGS): the profile has no $ENGINE two-card line, so a two-card run names its placement itself (-ts or -ot in $([ "$ENGINE" = ik ] && echo IK_GPU_FLAGS || echo LCPP_CLI_FLAGS)); without one the loader's own split would pick the layers" >&2
      exit 64
      ;;
  esac
fi

if [ "$ENGINE" = ik ]; then
  TREE=$IK CLI=$IK/build/bin/llama-cli LTOK=$IK/build/bin/llama-tokenize
  LTOK_ARGS=(--ids --log-disable)
else
  TREE=$LCPP CLI=$LCPP/build/bin/llama-completion LTOK=$LCPP/build/bin/llama-tokenize
  LTOK_ARGS=(--ids --log-disable --no-escape)
fi
TOK=${BLOOMERY_TOKENIZE_BIN:-target/release/bloomery-tokenize}
IDS=$BLOOMERY_DATA/engram/corpus-$CORPUS.ids
WORK=$(mktemp -d "${TMPDIR:-/tmp}/ik-draft.XXXXXX")
trap 'rm -rf "$WORK"' EXIT

args=(-m "$MODEL")
# shellcheck disable=SC2206
args+=($FLAGS)
args+=(-c "$CTX" -n "$N" -f "$WORK/prompt" --no-escape --temp 0 --ignore-eos --no-display-prompt --verbose-prompt)
[ "$ENGINE" = ik ] || args+=(-no-cnv)
if [ "$ARM" = dspark ]; then
  args+=(-md "$DSPARK_MODEL" --spec-type "$STAGE")
  [ -z "$DRAFT_PARAMS" ] || args+=(--draft-params "$DRAFT_PARAMS")
fi
if [ -n "$DRY" ]; then
  echo "[dry] engine=$ENGINE arm=$ARM corpus=$CORPUS ids=$IDS prompt=$PROMPT_N n=$N ctx=$CTX timing_gpu=$TIMING_GPU CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  echo "[dry] decode: $TOK -m $MODEL --decode -f $WORK/ids > $WORK/text"
  echo "[dry] check 1: $LTOK -m $MODEL -f $WORK/text ${LTOK_ARGS[*]}"
  echo "[dry] run: timeout --kill-after=10 $BOUND env $RUN_ENV $CLI ${args[*]} < /dev/null"
fi
for f in "$CLI" "$LTOK"; do
  [ -x "$f" ] && continue
  echo "ik-draft.sh: no $f" >&2
  [ "$ENGINE" = ik ] || [ "$f" != "$LTOK" ] || echo "    build it with: cmake --build $LCPP/build --target llama-tokenize" >&2
  exit 2
done
[ -f "$IDS" ] || { echo "ik-draft.sh: no corpus $IDS" >&2; exit 2; }
if [ "$ARM" = dspark ] && [ ! -f "$DSPARK_MODEL" ]; then
  echo "ik-draft.sh: no draft $DSPARK_MODEL" >&2
  exit 2
fi
# Our tokenizer decodes the prompt; the freshness check is the one the timing runners use.
assert_fresh_binary "$TOK" || exit $?

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
GOT=$("$LTOK" -m "$MODEL" -f "$WORK/text" "${LTOK_ARGS[@]}" 2> "$WORK/ltok.err" | tr -d '[] \n')
if [ "$GOT" != "$WANT" ]; then
  echo "ik-draft.sh: the decoded prompt does not re-tokenize to the corpus ids under llama-tokenize:" >&2
  echo "    $(first_diff "$WANT" "$GOT")" >&2
  tail -n 5 "$WORK/ltok.err" >&2
  exit 65
fi

if [ -n "$TIMING_CARDS" ]; then
  CARD_NAME=$TIMING_CARDS_NAME
else
  CARD_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$TIMING_GPU" | sed 's/^NVIDIA //; s/^GeForce //; s/^RTX //')
fi
CLI_SHA=$(sha256sum "$CLI" | cut -c1-12)
# GIT_OPTIONAL_LOCKS=0 keeps `git status` from rewriting the index of a tree this root process does not own.
TREE_HEAD=$(git -c safe.directory="$TREE" -C "$TREE" rev-parse --short=9 HEAD 2> /dev/null || echo '?')
TREE_DIRTY=$(GIT_OPTIONAL_LOCKS=0 git -c safe.directory="$TREE" -C "$TREE" status --porcelain --untracked-files=no 2> /dev/null | wc -l | tr -d ' ')
DRAFT_SHA=-
[ "$ARM" != dspark ] || DRAFT_SHA=$(sha256sum "$DSPARK_MODEL" | cut -c1-12)
WITNESS=(head-open indent card busiest model mem pgmajfault)
ref_witness() { echo "    $ENGINE: $CLI sha256=$CLI_SHA head=$TREE_HEAD dirty_files=$TREE_DIRTY"; }

STAGE_SHOWN=none
DRAFT_SHOWN=none
if [ "$ARM" = dspark ]; then
  STAGE_SHOWN=$STAGE
  DRAFT_SHOWN="$DSPARK_MODEL sha256=$DRAFT_SHA draft_params=${DRAFT_PARAMS:-<inherited from the target>}"
fi

if [ -n "$DRY" ]; then
  echo "[dry] prompt check 1 ($LTOK): $PROMPT_N ids round-trip"
  ref_witness | sed 's/^   /[dry]/'
  if [ -n "$TIMING_CARDS" ]; then
    tc_rc=0
    timing_cards_precheck '[dry] ' || tc_rc=$?
    if [ "$tc_rc" = 0 ]; then echo "[dry] two-card precheck: ok"; else echo "[dry] two-card precheck: refused (rc $tc_rc): $TWOCARD_WHY — a real run stops here, before the lease"; fi
  fi
  echo "[dry] warm-up: the run line above once, discarded, then the same line timed"
  echo "[dry] stops before the lease"
  exit 0
fi
majflt_require ik-draft.sh
timing_cards_precheck || {
  rc=$?
  echo "ik-draft.sh: two cards, refused before the lease: $TWOCARD_WHY" >&2
  exit "$rc"
}

lease_take
timing_cards_start
echo "[config] corpus=$CORPUS ids=$IDS prompt=$PROMPT_N n=$N arm=$ARM stage=$STAGE_SHOWN ctx=$CTX card=$CARD_NAME arm_bound=${BOUND}s"
echo "[config] $ENGINE: $CLI model=$MODEL flags=$FLAGS env=$RUN_ENV"
echo "[config] draft: $DRAFT_SHOWN"
echo "[config] timing_gpu=$TIMING_GPU other_gpu=$OTHER_GPU CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[config] prompt check 1 ($LTOK): $PROMPT_N ids round-trip"
[ -z "$TIMING_CARDS" ] || echo "[config] two cards: $TIMING_CARDS_NAME, CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, the placement in the flags above"

# run_once <label> <output file>: the command line between its witness blocks, after the card guard; RUN_RC,
# RUN_WALL, and the pgmajfault change across the process in RUN_FAULTS. A non-zero rc, or (two cards) a
# failed check after it, ends the script with the output's tail.
run_once() {
  local label=$1 out=$2 t0 f0
  guard_other
  witness "pre $label $ENGINE $ARM $CORPUS n=$N"
  ref_witness
  t0=$(date +%s) f0=$(majflt_now)
  # shellcheck disable=SC2086
  timeout --kill-after=10 "$BOUND" env $RUN_ENV "$CLI" "${args[@]}" < /dev/null > "$out" 2>&1
  RUN_RC=$?
  RUN_FAULTS=$(($(majflt_now) - f0)) RUN_WALL=$(($(date +%s) - t0))
  witness "post $label $ENGINE $ARM $CORPUS n=$N"
  ref_witness
  if [ "$RUN_RC" -ne 0 ]; then
    echo "ik-draft.sh: ${CLI##*/} rc $RUN_RC ($label)" >&2
    tail -n 20 "$out" >&2
    exit "$RUN_RC"
  fi
  if ! timing_cards_arm "$(cat "$out")"; then
    echo "ik-draft.sh: two cards ($label): $TWOCARD_WHY" >&2
    tail -n 20 "$out" >&2
    exit 1
  fi
}
run_once warmup "$WORK/warm"
echo "[warmup] the run once, discarded (rc 0, wall ${RUN_WALL}s, majflt $RUN_FAULTS${TWOCARD_DEVS:+, devices $TWOCARD_DEVS}): the timed run below reads the pages it read"
run_once timed "$WORK/raw"
rc=$RUN_RC
echo "--- $ENGINE raw begin (rc $rc, wall ${RUN_WALL}s)"
cat "$WORK/raw"
echo "--- $ENGINE raw end"

# cold_col <eval ms>: the timed run's fault column and cold tag (cold_check) against W = eval_ms, into
# COLD_COL. An eval time that is no positive number is a failed row, not a window of 0.
cold_col() {
  local w
  w=$(awk -v ms="$1" 'BEGIN { if (ms + 0 > 0) printf "%.4f", ms / 1e3 }')
  [ -n "$w" ] || { echo "ik-draft.sh: eval time '$1' ms is no window for the cold tag" >&2; exit 1; }
  cold_check "$RUN_FAULTS" "$w"
  COLD_COL="majflt=$RUN_FAULTS (whole process; ≤ $MAJ_BOUND % of W $w s)$COLD_TAG"
}

# Check 2: the ids llama-cli itself tokenized the prompt to (--verbose-prompt's list).
PGOT=$(awk -v n="$PROMPT_N" '/number of tokens in prompt = /{on=1; next} on && /^ *[0-9]+ -> \x27/{print $1; if(++k==n) exit}' "$WORK/raw" | paste -sd,)
PCOUNT=$(sed -n 's/.*number of tokens in prompt = \([0-9]*\).*/\1/p' "$WORK/raw" | head -n 1)
if [ "$PCOUNT" != "$PROMPT_N" ] || [ "$PGOT" != "$WANT" ]; then
  echo "ik-draft.sh: ${CLI##*/}'s prompt is not the corpus ids (it counted ${PCOUNT:-?}):" >&2
  echo "    $(first_diff "$WANT" "$PGOT")" >&2
  exit 65
fi
echo "[check] prompt check 2 (${CLI##*/} --verbose-prompt): $PCOUNT ids, identical"

if [ "$ENGINE" = lcpp ]; then
  EVAL=$(grep -E '^[a-z_]+: +eval time = ' "$WORK/raw" | head -n 1)
  [ -n "$EVAL" ] || { echo "ik-draft.sh: llama-completion printed no 'eval time' line" >&2; exit 1; }
  eval_ms=$(echo "$EVAL" | sed -E 's/.*eval time = +([0-9.]+) ms.*/\1/')
  runs=$(echo "$EVAL" | sed -E 's/.* ms \/ +([0-9]+) runs.*/\1/')
  tok_s=$(echo "$EVAL" | sed -E 's/.*, +([0-9.]+) tokens per second.*/\1/')
  step_tok_s=$(awk -v k="$runs" -v ms="$eval_ms" 'BEGIN{if(k>0 && ms>0) printf "%.2f", k*1e3/ms; else print "-"}')
  [ "$runs" = $((N - 1)) ] || echo "[check] llama-completion ran $runs decode steps, not n - 1 = $((N - 1))"
  cold_col "$eval_ms"
  echo "lcpp-prompt corpus=$CORPUS arm=$ARM n=$N depth=$PROMPT_N card=$CARD_NAME tok_s=$tok_s eval_ms=$eval_ms decoded=$((runs + 1)) step_tok_s=$step_tok_s accepted=- drafted=- | ${EVAL#"${EVAL%%[![:space:]]*}"} | - | $COLD_COL"
  exit 0
fi
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
cold_col "$eval_ms"
echo "ik-draft corpus=$CORPUS arm=$ARM n=$N depth=$PROMPT_N card=$CARD_NAME tok_s=$tok_s eval_ms=$eval_ms decoded=$decoded step_tok_s=$step_tok_s accepted=$accepted drafted=$drafted | ${EVAL#"${EVAL%%[![:space:]]*}"} | ${STATS} | $COLD_COL"
