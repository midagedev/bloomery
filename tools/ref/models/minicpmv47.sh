#!/usr/bin/env bash
# shellcheck shell=bash
# The MiniCPM-V 4.7 35B-A3B text-model profile: a qwen35moe file (the Qwen3.6-35B-A3B trunk, 40 layers, 256 experts, 8 used,
# head 256) with its own vocabulary (248,144 tokens), `rope.section_order = zyxt`, and a declared next-token layer (block_count
# 41, nextn_predict_layers 1) whose tensors are stripped (blk.40 carries none). Sourced by ref-paths.sh when
# BLOOMERY_MODEL=minicpmv47; never executed, and it exports nothing. Same shape as qwen35moe.sh, of which it is the sibling.
#
#   MODEL           the one file (no split set): the Q4_K_M of the 35B-A3B text model. BLOOMERY_REF_MODEL moves it; a caller's
#                   own MODEL= is ignored, as in qwen35moe.sh
#   IK              qwen35moe.sh's tree. ik refuses this file as it is (it reads nextn_predict_layers = 1 and asks for blk.40),
#                   so an ik run needs a copy whose header says block_count 40 and nextn_predict_layers 0, the tensor data
#                   untouched (tools/ref/gguf-respell.py)
#   REF_SET_CPU     ref_minicpmv47 under $BLOOMERY_DATA: a name only — no ik set is dumped from this file yet
#   REF_SET_CUDA    ref_cuda_minicpmv47: a name only
#   REF_TOKENS      the ids ik's llama-tokenize prints for the oracle prompt under this file's vocabulary
#   REF_DUMP_LEASE  1: every ik reference run takes the machine-wide CPU lease. Not overridable from here.
#
# Deliberately unset: LCPP*, MRS*, IK_GPU_FLAGS, IK_BEST_FLAGS, REF_PROMPTS and `ref_step_variant` (dump.sh refuses a
# decode-step dump by name). Every script that reads them runs under `set -u` and stops at the unset name.
#
# SC2034: every name here is read by the file that sources this one, which shellcheck does not see from this file alone.
# shellcheck disable=SC2034
MODEL_NAME=qwen35moe
: "${IK:=/home/user/ik-idxkey}"
MODEL=${BLOOMERY_REF_MODEL:-/models/MiniCPM-V-4.7-35B-A3B/MiniCPM-V-4_7-35B-A3B-Q4_K_M.gguf}
: "${REF_CTX:=512}"
: "${REF_SET_CPU:=ref_minicpmv47}"
: "${REF_SET_CUDA:=ref_cuda_minicpmv47}"
# "The capital of France is" under this model's tokenizer: what `$IK/build/bin/llama-tokenize
# -m $MODEL -p "The capital of France is" --ids --log-disable --no-parse-special` prints (it loads the vocabulary only).
# Five ids and no BOS (tokenizer.ggml.add_bos_token is false). They equal qwen35moe.sh's: the two vocabularies agree on
# every id of this prompt, and on all 76,179 ids of the tokenizer oracle's prose text (equal sha256 of both id lists).
REF_TOKENS=760,6511,314,9338,369
REF_DUMP_LEASE=1
