#!/usr/bin/env bash
# shellcheck shell=bash
# The deepseek41 (DeepSeek-V4.1-Flash) profile: everything the reference engine and the harnesses
# need that is a property of the *model*, not of the machine. Sourced by ref-paths.sh when
# BLOOMERY_MODEL=deepseek41; never executed, and it exports nothing. Same shape as deepseek2.sh.
#
# The name is the GGUF general.architecture value (shard 1's header says deepseek41), so
# `grep deepseek41` finds the port's arch, its metadata keys (deepseek41.engram.*), this profile
# and the oracle set at once.
#
#   MODEL           shard 1 of the split set; the loader follows split.count from there.
#                   BLOOMERY_REF_MODEL moves it; a caller's own MODEL= is ignored, as in deepseek2.sh
#   IK              moves the whole ik tree. The default is our V4.1 port (PR #2455), the tree the
#                   V2-Lite dumper already links; its V4.1 path has only run on the CPU (-ngl 0)
#   REF_CTX         the one context the reference files are produced at
#   REF_SET_CPU     ref_deepseek41 under $BLOOMERY_DATA — flat, a sibling of ref, ref_cuda and
#                   ref_cuda_v2, because every reader resolves a set as $BLOOMERY_DATA/<one name>
#                   (dump.sh's staging and .old siblings, the gates' BLOOMERY_REF_SET)
#   REF_SET_CUDA    ref_cuda_deepseek41: a name only — no CUDA set is made for this model
#   REF_TOKENS      the oracle's token ids; dump.sh's BLOOMERY_REF_TOKENS overrides them
#   REF_DUMP_LEASE  1: the dump pages in hundreds of GiB of weights, so dump.sh runs it under the
#                   machine-wide CPU lease and witnesses it. Not overridable from here.
#
# Deliberately unset: IK_BEST_FLAGS, IK_GPU_FLAGS and REF_PROMPTS. No flag sweep and no prompt set
# exist for this model, and every script that reads them runs under `set -u`, so such a script
# stops at the unset name instead of running ik at flags nobody measured.
#
# SC2034: every name here is read by the file that sources this one, which shellcheck does not
# see from this file alone.
# shellcheck disable=SC2034
MODEL_NAME=deepseek41
: "${IK:=/home/user/ik_llama.cpp}"
MODEL=${BLOOMERY_REF_MODEL:-/models/DeepSeek-V4.1-Flash-Q3_K_M-engramQ8-tokembdBF16-attnQ8/DeepSeek-V4.1-Flash-Q3_K_M-00001-of-00009.gguf}
: "${REF_CTX:=512}"
: "${REF_SET_CPU:=ref_deepseek41}"
: "${REF_SET_CUDA:=ref_cuda_deepseek41}"
# "The capital of France is" under this model's tokenizer: what `$IK/build/bin/llama-tokenize
# -m $MODEL -p "The capital of France is" --ids --log-disable --no-parse-special` prints for
# shard 1 (it loads the vocabulary only). Five ids and no BOS: the file sets
# tokenizer.ggml.add_bos_token to false, so nothing is prepended (its BOS would be id 0).
# Changing them invalidates the whole set.
REF_TOKENS=671,6102,294,8760,344
REF_DUMP_LEASE=1
