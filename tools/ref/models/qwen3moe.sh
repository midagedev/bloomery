#!/usr/bin/env bash
# shellcheck shell=bash
# The qwen3moe (Qwen3-30B-A3B-Instruct-2507) profile: everything the reference engine and the harnesses
# need that is a property of the *model*, not of the machine. Sourced by ref-paths.sh when
# BLOOMERY_MODEL=qwen3moe; never executed, and it exports nothing. Same shape as deepseek2.sh.
#
# The name is the GGUF general.architecture value, so `grep qwen3moe` finds the module
# (crates/model/src/arch/qwen3moe), its metadata keys (qwen3moe.*), this profile and the oracle sets.
#
#   MODEL           the one file (no split set). BLOOMERY_REF_MODEL moves it; a caller's own MODEL= is
#                   ignored, as in deepseek2.sh
#   IK              moves the whole ik tree. The default is the tree the V4.1 profile names: it builds
#                   this architecture (src/graphs/build_qwen3.cpp, build_qwen3moe) and the installed
#                   dump_ref links against it, so one dumper serves both profiles ([foreign-lib] in
#                   dump.sh otherwise)
#   REF_CTX         the one context the reference files are produced at
#   REF_SET_CPU     ref_qwen3moe under $BLOOMERY_DATA: ik's CPU dump
#   REF_SET_CUDA    ref_cuda_qwen3moe: ik's CUDA dump, when one is made
#   REF_TOKENS      the oracle's token ids; dump.sh's BLOOMERY_REF_TOKENS overrides them
#   REF_DUMP_LEASE  1: every ik reference run takes the machine-wide CPU lease, and the dump pages in
#                   the whole 18.6 GB file. Not overridable from here.
#   ref_step_variant  the decode-step variants, below
#
# Deliberately unset: IK_BEST_FLAGS, IK_GPU_FLAGS and REF_PROMPTS. No flag sweep and no prompt set
# exist for this model yet, and every script that reads them runs under `set -u`, so such a script
# stops at the unset name instead of running ik at flags nobody measured.
#
# SC2034: every name here is read by the file that sources this one, which shellcheck does not
# see from this file alone.
# shellcheck disable=SC2034
MODEL_NAME=qwen3moe
: "${IK:=/home/user/ik-idxkey}"
MODEL=${BLOOMERY_REF_MODEL:-/models/Qwen3-30B-A3B/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf}
: "${REF_CTX:=512}"
: "${REF_SET_CPU:=ref_qwen3moe}"
: "${REF_SET_CUDA:=ref_cuda_qwen3moe}"
# "The capital of France is" under this model's tokenizer: what `$IK/build/bin/llama-tokenize
# -m $MODEL -p "The capital of France is" --ids --log-disable --no-parse-special` prints (it loads
# the vocabulary only). Five ids and no BOS: the file sets tokenizer.ggml.add_bos_token to false.
# Changing them invalidates the whole set.
REF_TOKENS=785,6722,315,9625,374
REF_DUMP_LEASE=1

# Decode-step variants, `dump.sh <variant>` (`just dump-ref-qwen3moe <variant>`): one decode step
# dumped after a quiet prefill (dump_ref.cpp, --decode-step), each into a set of its own — dump.sh
# refuses REF_SET_CPU and REF_SET_CUDA for them. `ref_step_variant <name>` sets STEP_SET, STEP_CTX,
# STEP_PREFILL, STEP_TOKENS (or STEP_TOKENS_FILE and STEP_TOKENS_SHA256) and STEP_ARGS as
# models/deepseek41.sh describes, and returns 1 for a name it does not know. The suffix
# `-every-node` runs the prefill under the dumped schedule instead of the fused one
# (--prefill-every-node), so the caches the step reads carry a dumped prefill's arithmetic.
#   step4  the oracle's five ids: a quiet prefill of 4, the step at position 4, -c 512
ref_step_variant() {
  local name=$1 every_node=0
  case $name in
    *-every-node) every_node=1; name=${name%-every-node} ;;
  esac
  STEP_TOKENS='' STEP_TOKENS_FILE='' STEP_TOKENS_SHA256=''
  STEP_ARGS=()
  case $name in
    step4) STEP_SET=ref_qwen3moe_step4; STEP_CTX=512; STEP_PREFILL=4; STEP_TOKENS=$REF_TOKENS ;;
    *)     return 1 ;;
  esac
  if [ "$every_node" = 1 ]; then
    STEP_SET=${STEP_SET}_every_node
    STEP_ARGS+=(--prefill-every-node)
  fi
}
