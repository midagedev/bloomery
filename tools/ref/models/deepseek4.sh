#!/usr/bin/env bash
# shellcheck shell=bash
# The deepseek4 (DeepSeek-V4-Flash-0731) profile: everything the reference engine and the harnesses
# need that is a property of the *model*, not of the machine. Sourced by ref-paths.sh when
# BLOOMERY_MODEL=deepseek4; never executed, and it exports nothing. Same shape as deepseek2.sh.
#
# The name is the GGUF general.architecture value, so `grep deepseek4` finds this profile, the
# metadata keys (deepseek4.*) and the oracle sets. The engine reads the file in the deepseek41 module
# (crates/model/src/arch/deepseek41, its Model::Deepseek4) — the one architecture whose module is
# named after another string.
#
#   MODEL           shard 1 of unsloth's UD-Q3_K_M split set (4 shards, 1,328 tensors); the loader
#                   follows split.count from there. BLOOMERY_REF_MODEL moves it; a caller's own MODEL=
#                   is ignored, as in deepseek2.sh. Its routed gate and up stacks are IQ3_XXS (layer
#                   26 MXFP4), the down stacks MXFP4, the dense weights Q8_0; it carries no MTP layer
#   IK              moves the whole ik tree. The default is the tree the V4.1 profile names: its loader
#                   and graph build deepseek4 (src/llama-hparams.cpp, src/graphs/build_deepseek4.cpp)
#                   and it has a llama-bench
#   LCPP            the llama.cpp tree for this model: the V4.1 port's branch the deepseek41 profile
#                   names, which carries mainline's deepseek4 (src/llama-arch.cpp) and has a
#                   llama-bench; LCPPBIN moves its llama-bench alone, as IKBIN does ik's
#
# Deliberately unset: REF_TOKENS, REF_PROMPTS, IK_BEST_FLAGS, IK_GPU_FLAGS, LCPP_GPU_FLAGS and the
# decode-step variants. No oracle set, token list, placement flags or flag sweep exist for this model
# yet, and every script that reads them runs under `set -u` or checks for them (dump.sh), so such a
# script stops at the unset name instead of running a reference nobody set up.
#
# SC2034: every name here is read by the file that sources this one, which shellcheck does not
# see from this file alone.
# shellcheck disable=SC2034
MODEL_NAME=deepseek4
: "${IK:=/home/user/ik-idxkey}"
MODEL=${BLOOMERY_REF_MODEL:-/models/DeepSeek-V4-Flash-0731-UD-Q3_K_M/DeepSeek-V4-Flash-0731-UD-Q3_K_M-00001-of-00004.gguf}
: "${LCPP:=/home/user/llama.cpp-v41}"
: "${LCPPBIN:=$LCPP/build/bin/llama-bench}"
: "${REF_CTX:=512}"
: "${REF_SET_CPU:=ref_deepseek4}"
: "${REF_SET_CUDA:=ref_cuda_deepseek4}"
REF_DUMP_LEASE=1
