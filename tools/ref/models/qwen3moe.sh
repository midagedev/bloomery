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
#   IK_GPU_FLAGS    ik's decode with the whole model on the timing card, the `ik:<D>` arms of
#                   depth-qwen3moe.sh
#   IK_GPU_DEFAULT_FLAGS  the same at llama-bench's defaults, the `ikdef:<D>` arms: the two merges
#                   in IK_GPU_FLAGS are opt-in, and the interleaved pair says which set is faster
#   LCPP            the mainline llama.cpp tree the `lcpp:<D>` arms run: mainline itself, not the V4.1
#                   PR branch the deepseek41 profile names (qwen3moe needs no port); LCPPBIN moves its
#                   llama-bench alone, as IKBIN does ik's
#   LCPP_GPU_FLAGS  mainline's decode with the whole model on the timing card, the `lcpp:<D>` arms.
#                   Both flag sets are chosen by reading each tree's CLI and docs; no flag sweep has
#                   been run, so a row is "at these flags". depth-qwen3moe.sh's header gives each
#                   flag's reason and the flags left out.
#
# Deliberately unset: IK_BEST_FLAGS and REF_PROMPTS. No CPU flag sweep and no prompt set exist for
# this model yet, and every script that reads them runs under `set -u`, so such a script stops at
# the unset name instead of running ik at flags nobody measured.
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
: "${IK_GPU_FLAGS:=-ngl 99 -fa 1 -fmoe 1 -mqkv 1 -muge 1}"
: "${IK_GPU_DEFAULT_FLAGS:=-ngl 99}"
: "${LCPP:=/home/user/llama.cpp-mainline}"
: "${LCPPBIN:=$LCPP/build/bin/llama-bench}"
: "${LCPP_GPU_FLAGS:=-ngl 99 -fa on}"

# Decode-step variants, `dump.sh <variant>` (`just dump-ref-qwen3moe <variant>`): one decode step
# dumped after a quiet prefill (dump_ref.cpp, --decode-step), each into a set of its own — dump.sh
# refuses REF_SET_CPU and REF_SET_CUDA for them. `ref_step_variant <name>` sets STEP_SET, STEP_CTX,
# STEP_PREFILL, STEP_TOKENS (or STEP_TOKENS_FILE and STEP_TOKENS_SHA256) and STEP_ARGS as
# models/deepseek41.sh describes, and returns 1 for a name it does not know. The suffix
# `-every-node` runs the prefill under the dumped schedule instead of the fused one
# (--prefill-every-node), so the caches the step reads carry a dumped prefill's arithmetic.
#   step4  the oracle's five ids: a quiet prefill of 4, the step at position 4, -c 512
#   d1k    a quiet prefill of 1,024 ids of prose, the step at position 1,024, -c 2048
#   d4k    a quiet prefill of 4,096 ids of prose, the step at position 4,096, -c 4608
# The prose of d1k and d4k is $BLOOMERY_DATA/qwen3moe/corpus-prose.ids: the tokenizer oracle's
# `prose.nps.ids` under this model's vocabulary (ik's docs/**/*.md and README.md, --no-parse-special,
# one id per line), copied out of tokenizer-qwen3moe/ so a rerun of that oracle on a moved ik tree
# cannot change the ids a set is of; the sha256 below is the check.
ref_step_variant() {
  local name=$1 every_node=0
  case $name in
    *-every-node) every_node=1; name=${name%-every-node} ;;
  esac
  STEP_TOKENS='' STEP_TOKENS_FILE='' STEP_TOKENS_SHA256=''
  STEP_ARGS=()
  case $name in
    step4) STEP_SET=ref_qwen3moe_step4; STEP_CTX=512;  STEP_PREFILL=4; STEP_TOKENS=$REF_TOKENS ;;
    d1k)   STEP_SET=ref_qwen3moe_d1k;   STEP_CTX=2048; STEP_PREFILL=1024 ;;
    d4k)   STEP_SET=ref_qwen3moe_d4k;   STEP_CTX=4608; STEP_PREFILL=4096 ;;
    *)     return 1 ;;
  esac
  case $name in
    d1k|d4k) STEP_TOKENS_FILE=$BLOOMERY_DATA/qwen3moe/corpus-prose.ids
             STEP_TOKENS_SHA256=9444bc5b4e2a7c5f4caac4f1aa7b6ef8e945b114fdecac52aca36a1de4e76cf6 ;;
  esac
  if [ "$every_node" = 1 ]; then
    STEP_SET=${STEP_SET}_every_node
    STEP_ARGS+=(--prefill-every-node)
  fi
}
