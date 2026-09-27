#!/usr/bin/env bash
# shellcheck shell=bash
# The qwen4exp (Qwen3.8-Flash-Next) profile: everything the reference engine and the harnesses need
# that is a property of the *model*, not of the machine. Sourced by ref-paths.sh when
# BLOOMERY_MODEL=qwen4exp; never executed, and it exports nothing. Same shape as qwen3moe.sh.
#
# The name is the GGUF general.architecture value (shard 1's header says qwen4exp), so `grep qwen4exp`
# finds this profile, the metadata keys (qwen4exp.*), the oracle sets and their refset family
# (crates/refset/src/arch/qwen4exp) at once.
#
#   MODEL           shard 1 of unsloth's UD-Q4_K_XL split set (four shards; the first holds the header
#                   and no tensor); the loader follows split.count from there, and the first shard's
#                   full path is the identity every oracle set states. The mtp-*.gguf files beside it
#                   are not loaded. BLOOMERY_REF_MODEL moves it; a caller's own MODEL= is ignored, as in
#                   qwen3moe.sh
#   IK              moves the whole ik tree. The default is the tree the V4.1, qwen35moe and glm5next
#                   profiles name: its build carries this architecture (src/graphs/build_qwen4exp.cpp;
#                   libllama.so holds build_qwen4exp) and the installed dump_ref links against it, so one
#                   dumper serves every profile ([foreign-lib] in dump.sh otherwise)
#   REF_CTX         the one context the batch set is produced at
#   REF_SET_CPU     ref_qwen4exp under $BLOOMERY_DATA: ik's CPU dump
#   REF_SET_CUDA    ref_cuda_qwen4exp: a name only — no CUDA set is made for this model yet
#   REF_TOKENS      the oracle's token ids; dump.sh's BLOOMERY_REF_TOKENS overrides them
#   REF_DUMP_LEASE  1: every ik reference run takes the machine-wide CPU lease, and the dump pages in
#                   the whole 111.3 GB split set. Not overridable from here.
#   ref_step_variant  the decode-step variants, below
#
#   LCPP            the mainline llama.cpp tree the `lcpp:<D>` and `lcpppp:<P>` arms of
#                   depth-qwen3moe.sh run (`just depth-gpu-qwen4exp`): mainline itself, which builds
#                   this architecture (src/models/qwen4exp.cpp), so no PR branch; LCPPBIN moves its
#                   llama-bench alone
#   LCPP_GPU_FLAGS  the hand-set arm: every layer and the head on the card, flash attention on,
#                   the PLE table read into RAM up front (-lzm off: 270 GB of RAM holds it, and the
#                   lazy path costs the baseline its prefill), and the routed stacks of the first
#                   26 layers on the host (-ncmoe 26): 22 layers' experts, 1.57..1.84 GB each, fill
#                   the A6000's ~51 GB after ~4.9 GB of non-expert weights and ~10 GB left for the
#                   -ub 4096 arm's compute and output buffers [derived]. The fit arms (lcppfit,
#                   lcppppfit) let llama-bench place to the byte; each row publishes the faster.
#                   -t 32 is the host's cores, spelled out as deepseek41.sh does: the host layers'
#                   experts run on them. No flag sweep has been run
#   TWO_CARD_PLACEMENT  the two-card mode (BLOOMERY_TIMING_CARDS=a6000+3090, tools/ref/timing-card.sh; the
#                   default file only, empty otherwise, and depth-qwen3moe.sh then refuses the mode by name):
#                   the "A6000+3090" table's mainline line, LCPP_GPU_FLAGS at -ncmoe 21 and -ts 42.5/6.5
#                   (QWEN38_TS), and what they place. llama-bench's -sm layer (its default) gives device d
#                   the layers il whose il / 49 is below the d-th cumulative -ts fraction — 48 layers and the
#                   output as a 49th slot (src/llama-model.cpp:1521-1546 at 53ed051ce) — the A6000 device 0
#                   and the 3090 device 1; -ncmoe K keeps layers 0..K-1's experts on the host, so the 3090
#                   takes the last expert layers and the output. The A6000 (device 0) keeps 22 expert layers,
#                   the one-card line's count, now layers 21-42 with layers 0-20 beside them; the 3090 (device
#                   1) layers 43-47 with their experts and the output. 42.5/6.5 puts the boundary at 0.867,
#                   between slot 42 (0.857) and slot 43 (0.878). The 3090's five: its usable 24,176 MiB
#                   (25.35 GB) less the one-card line's ~10 GB reserve for the -ub 4096 buffers (the output,
#                   and its logits, are on the 3090 now) and at most all ~4.9 GB of non-expert weights leave
#                   ~10.5 GB, five layers at the band's top of 1.84 GB. The A6000's bound: 22 x 1.84 GB and at
#                   most the 4.9 GB beside them is 45.4 GB of its ~47.45 GB usable, ~2 GB where the one-card
#                   line kept ~10 at the band's middle [derived: the band is this comment's, per-layer bytes
#                   not read; no two-card load has been run]. The fit arms drop -ncmoe and -ts and let
#                   llama-bench's fit place over both cards (common/fit.cpp sets tensor_split per device):
#                   they are the check, and a line that does not load is a FAIL row
#
# No REF_DUMP_ARGS: without --defer-experts the loader populates the whole split set, which fits in
# the page cache (unlike V4.1's), so a later dump of the same file reads nothing from the device.
#
# Deliberately unset: IK_BEST_FLAGS, REF_PROMPTS and the ik and mistral.rs reference arms
# (IK_GPU_FLAGS, MRS, MRS_FLAGS): the public table's reference is mainline llama.cpp, and nobody has
# read whether the mistral.rs tree opens this file. Every script
# that reads them runs under `set -u`, so such a script stops at the unset name instead of running
# an engine at flags nobody chose.
#
# SC2034: every name here is read by the file that sources this one, which shellcheck does not
# see from this file alone.
# shellcheck disable=SC2034
MODEL_NAME=qwen4exp
: "${IK:=/home/user/ik-idxkey}"
QWEN38_FILE=/models/Qwen3.8-Flash-Next/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
MODEL=${BLOOMERY_REF_MODEL:-$QWEN38_FILE}
: "${REF_CTX:=512}"
: "${REF_SET_CPU:=ref_qwen4exp}"
: "${REF_SET_CUDA:=ref_cuda_qwen4exp}"
: "${LCPP:=/home/user/llama.cpp-mainline}"
: "${LCPPBIN:=$LCPP/build/bin/llama-bench}"
TWO_CARD_PLACEMENT=
if [ "${BLOOMERY_TIMING_CARDS:-}" = a6000+3090 ] && [ "$MODEL" = "$QWEN38_FILE" ]; then
  : "${QWEN38_TS:=42.5/6.5}"
  : "${LCPP_GPU_FLAGS:=-ngl 99 -fa on -lzm off -ncmoe 21 -t 32 -ts $QWEN38_TS}"
  TWO_CARD_PLACEMENT="lcpp -ncmoe 21 -ts $QWEN38_TS: the A6000 (device 0) layers 0-42, 21-42 with their experts; the 3090 (device 1) layers 43-47 with their experts and the output [derived, models/qwen4exp.sh]"
fi
: "${LCPP_GPU_FLAGS:=-ngl 99 -fa on -lzm off -ncmoe 26 -t 32}"
# "The capital of France is" under this model's tokenizer: what `$IK/build/bin/llama-tokenize
# -m $MODEL -p "The capital of France is" --ids --log-disable --no-parse-special` prints (it loads
# the vocabulary only). Five ids and no BOS: the file sets tokenizer.ggml.add_bos_token to false.
# Changing them invalidates the whole set.
REF_TOKENS=760,6511,314,9338,369
REF_DUMP_LEASE=1

# Decode-step variants, `dump.sh <variant>` (`just dump-ref-qwen4exp <variant>`): one decode step
# dumped after a quiet prefill (dump_ref.cpp, --decode-step), each into a set of its own — dump.sh
# refuses REF_SET_CPU and REF_SET_CUDA for them. `ref_step_variant <name>` sets STEP_SET, STEP_CTX,
# STEP_PREFILL, STEP_TOKENS (or STEP_TOKENS_FILE and STEP_TOKENS_SHA256) and STEP_ARGS as
# models/deepseek41.sh describes, and returns 1 for a name it does not know. The suffix
# `-every-node` runs the prefill under the dumped schedule instead of the fused one
# (--prefill-every-node), so the caches and the recurrent state the step reads carry a dumped
# prefill's arithmetic.
#   step4  the oracle's five ids: a quiet prefill of 4, the step at position 4, -c 512
#   d1k    a quiet prefill of 1,024 ids of prose, the step at position 1,024, -c 2048
#   d3k    a quiet prefill of 3,000 ids of prose, the step at position 3,000, -c 4096: past the cells
#          the QSA layers keep whole. ik's indexer cuts when its width, top_k + pool - 1 = 2,051 cells,
#          is under n_kv (build_qwen4exp.cpp, qwen4exp_qsa_mask), and n_kv at this step is at least
#          the 3,001 cells in use (3,008 at the KV pad of 32, 3,072 at flash attention's 256), so the
#          step's attention reads a selection.
#          3,000 is a multiple of the pool (4): the step's own block holds one cell
# The prose is $BLOOMERY_DATA/qwen4exp/corpus-prose.ids: the tokenizer oracle's prose text
# ($BLOOMERY_DATA/tokenizer-qwen3moe/prose.txt, ik's docs/**/*.md and README.md, md5
# c6bb074439479420faffa33257424c00, the text the qwen3moe, qwen35moe and glm5next sets are of) through
# ik-idxkey's llama-tokenize (md5 8dd4fe8a9bb773f80f8345c87f019b1b) with TOKENIZER_VOCAB = MODEL,
# --no-parse-special, one id per line, 76,180 ids. The file is byte for byte qwen35moe's
# corpus-prose.ids, kept under this profile's name so a change on either side cannot move the ids a
# set of the other is of; the sha256 below is the check.
ref_step_variant() {
  local name=$1 every_node=0
  case $name in
    *-every-node) every_node=1; name=${name%-every-node} ;;
  esac
  STEP_TOKENS='' STEP_TOKENS_FILE='' STEP_TOKENS_SHA256=''
  STEP_ARGS=()
  case $name in
    step4) STEP_SET=ref_qwen4exp_step4; STEP_CTX=512;  STEP_PREFILL=4; STEP_TOKENS=$REF_TOKENS ;;
    d1k)   STEP_SET=ref_qwen4exp_d1k;   STEP_CTX=2048; STEP_PREFILL=1024 ;;
    d3k)   STEP_SET=ref_qwen4exp_d3k;   STEP_CTX=4096; STEP_PREFILL=3000 ;;
    *)     return 1 ;;
  esac
  case $name in
    d1k|d3k) STEP_TOKENS_FILE=$BLOOMERY_DATA/qwen4exp/corpus-prose.ids
             STEP_TOKENS_SHA256=dca5f89b2903f9ffd2f4e20eec18a9fdf3561bcb531a721a114e4fab68e925be ;;
  esac
  if [ "$every_node" = 1 ]; then
    STEP_SET=${STEP_SET}_every_node
    STEP_ARGS+=(--prefill-every-node)
  fi
}
