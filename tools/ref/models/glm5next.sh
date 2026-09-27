#!/usr/bin/env bash
# shellcheck shell=bash
# The glm5next (GLM-5.3-Flash) profile: everything the reference engine and the harnesses need that is
# a property of the *model*, not of the machine. Sourced by ref-paths.sh when BLOOMERY_MODEL=glm5next;
# never executed, and it exports nothing. Same shape as qwen3moe.sh.
#
# The name is the GGUF general.architecture value (shard 1's header says glm5next), so `grep glm5next`
# finds this profile, the metadata keys (glm5next.*), the oracle sets and their refset family
# (crates/refset/src/arch/glm5next) at once.
#
#   MODEL           shard 1 of unsloth's UD-Q4_K_XL split set (six shards); the loader follows
#                   split.count from there, and the first shard's full path is the identity every
#                   oracle set states. BLOOMERY_REF_MODEL moves it; a caller's own MODEL= is ignored,
#                   as in qwen3moe.sh
#   IK              moves the whole ik tree. The default is the tree the V4.1, qwen3moe and qwen35moe
#                   profiles name: its build carries this architecture (libllama.so holds glm5next and
#                   "GLM5NEXT: hyper_connection.count is required") and the installed dump_ref links
#                   against it, so one dumper serves every profile ([foreign-lib] in dump.sh otherwise)
#   REF_CTX         the one context the batch set is produced at
#   REF_SET_CPU     ref_glm5next under $BLOOMERY_DATA: ik's CPU dump
#   REF_SET_CUDA    ref_cuda_glm5next: a name only — no CUDA set is made for this model yet, and the
#                   file does not fit on one card at -ngl 99 (justfile, dump-ref-glm5next-cuda)
#   REF_TOKENS      the oracle's token ids; dump.sh's BLOOMERY_REF_TOKENS overrides them
#   REF_DUMP_LEASE  1: every ik reference run takes the machine-wide CPU lease, and the dump pages in
#                   the whole 199.7 GB split set, most of the page cache — more than the machine can
#                   share with a timed run. Not overridable from here.
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
MODEL_NAME=glm5next
: "${IK:=/home/user/ik-idxkey}"
MODEL=${BLOOMERY_REF_MODEL:-/models/GLM-5.3-Flash-UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf}
: "${REF_CTX:=512}"
: "${REF_SET_CPU:=ref_glm5next}"
: "${REF_SET_CUDA:=ref_cuda_glm5next}"
# "The capital of France is" under this model's tokenizer: what `$IK/build/bin/llama-tokenize
# -m $MODEL -p "The capital of France is" --ids --log-disable --no-parse-special` prints (it loads
# the vocabulary only). Five ids and no BOS: the file's pre-tokenizer is glm4, for which the
# vocabulary loader clears the BOS id, so the first id is 'The' and not tokenizer.ggml.bos_token_id.
# Changing them invalidates the whole set.
REF_TOKENS=785,6722,315,9621,374
REF_DUMP_LEASE=1

# Decode-step variants, `dump.sh <variant>` (`just dump-ref-glm5next <variant>`): one decode step
# dumped after a quiet prefill (dump_ref.cpp, --decode-step), each into a set of its own — dump.sh
# refuses REF_SET_CPU and REF_SET_CUDA for them. `ref_step_variant <name>` sets STEP_SET, STEP_CTX,
# STEP_PREFILL, STEP_TOKENS (or STEP_TOKENS_FILE and STEP_TOKENS_SHA256) and STEP_ARGS as
# models/deepseek41.sh describes, and returns 1 for a name it does not know. The suffix
# `-every-node` runs the prefill under the dumped schedule instead of the fused one
# (--prefill-every-node), so the caches and the recurrent state the step reads carry a dumped
# prefill's arithmetic.
#   step4  the oracle's five ids: a quiet prefill of 4, the step at position 4, -c 512
#   d1k    a quiet prefill of 1,024 ids of prose, the step at position 1,024, -c 2048
# The prose is $BLOOMERY_DATA/glm5next/corpus-prose.ids: the tokenizer oracle's `prose.nps.ids` under
# this vocabulary (set tokenizer-glm5next, written by crates/tokenizer/tools/oracle.sh with
# TOKENIZER_VOCAB = MODEL and ik-idxkey's llama-tokenize; the text is ik's docs/**/*.md and
# README.md, md5 c6bb074439479420faffa33257424c00, the prose the qwen3moe and qwen35moe sets are of),
# one id per line, 71,727 ids, copied out of tokenizer-glm5next/ so a rerun of that oracle on a moved
# ik tree cannot change the ids a set is of; the sha256 below is the check.
ref_step_variant() {
  local name=$1 every_node=0
  case $name in
    *-every-node) every_node=1; name=${name%-every-node} ;;
  esac
  STEP_TOKENS='' STEP_TOKENS_FILE='' STEP_TOKENS_SHA256=''
  STEP_ARGS=()
  case $name in
    step4) STEP_SET=ref_glm5next_step4; STEP_CTX=512;  STEP_PREFILL=4; STEP_TOKENS=$REF_TOKENS ;;
    d1k)   STEP_SET=ref_glm5next_d1k;   STEP_CTX=2048; STEP_PREFILL=1024 ;;
    *)     return 1 ;;
  esac
  case $name in
    d1k) STEP_TOKENS_FILE=$BLOOMERY_DATA/glm5next/corpus-prose.ids
         STEP_TOKENS_SHA256=8af07981c1749170b57d424cff5274b89be063c1eb44ffa3a443ddc16642bb64 ;;
  esac
  if [ "$every_node" = 1 ]; then
    STEP_SET=${STEP_SET}_every_node
    STEP_ARGS+=(--prefill-every-node)
  fi
}
