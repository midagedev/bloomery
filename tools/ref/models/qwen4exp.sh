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
# No REF_DUMP_ARGS: without --defer-experts the loader populates the whole split set, which fits in
# the page cache (unlike V4.1's), so a later dump of the same file reads nothing from the device.
#
# Deliberately unset: IK_BEST_FLAGS, REF_PROMPTS and the reference-line arms (IK_GPU_FLAGS, LCPP,
# LCPP_GPU_FLAGS, MRS, MRS_FLAGS). No flag has been chosen or measured for this model; the depth and
# prompt-processing lines against ik, llama.cpp and mistral.rs belong to a later round. Every script
# that reads them runs under `set -u`, so such a script stops at the unset name instead of running
# an engine at flags nobody chose.
#
# SC2034: every name here is read by the file that sources this one, which shellcheck does not
# see from this file alone.
# shellcheck disable=SC2034
MODEL_NAME=qwen4exp
: "${IK:=/home/user/ik-idxkey}"
MODEL=${BLOOMERY_REF_MODEL:-/models/Qwen3.8-Flash-Next/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf}
: "${REF_CTX:=512}"
: "${REF_SET_CPU:=ref_qwen4exp}"
: "${REF_SET_CUDA:=ref_cuda_qwen4exp}"
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
