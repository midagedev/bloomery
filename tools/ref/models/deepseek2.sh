#!/usr/bin/env bash
# shellcheck shell=bash
# The deepseek2 (DeepSeek-V2-Lite) profile: everything the reference engine and the harnesses
# need that is a property of the *model*, not of the machine. Sourced by ref-paths.sh when
# BLOOMERY_MODEL=deepseek2 (its default); never executed, and it exports nothing.
#
# The name is the GGUF general.architecture value, so `grep deepseek2` finds the module, the
# metadata keys, this profile and the gates at once.
#
# Every value here keeps the override form the runner that reads it had before, so an
# environment that worked against the runners works against the profile unchanged:
#   MODEL           BLOOMERY_REF_MODEL moves it; a caller's own MODEL= is ignored, as in ref_paths.h
#   IK              moves the whole ik tree (IKBIN follows it, in ref-paths.sh)
#   IK_BEST_FLAGS   the CPU decode flags measured fastest for this model
#   IK_GPU_FLAGS    the same question with the layers offloaded: the CPU combination with
#                   -rtr 1 dropped; no GPU flag sweep has been run, so these are "at these flags"
#   REF_PROMPTS     the prompt set the argmax/greedy files and the expert union are taken over
#   REF_CTX         the one context the reference files are produced at
#   REF_SET_CPU     oracle directories under $BLOOMERY_DATA: ik's CPU dump and its CUDA dump
#   REF_SET_CUDA
#
# There is no PPL set for this model yet; when one is added it belongs here, beside REF_PROMPTS.
#
# SC2034: every name here is read by the file that sources this one, which shellcheck does not
# see from this file alone.
# shellcheck disable=SC2034
MODEL_NAME=deepseek2
: "${IK:=/home/user/ik_llama.cpp}"
MODEL=${BLOOMERY_REF_MODEL:-/models/small/DeepSeek-V2-Lite-Chat.Q3_K_M.gguf}
: "${IK_BEST_FLAGS:=-mla 3 -fa 1 -fmoe 1 -rtr 1}"
: "${IK_GPU_FLAGS:=-ngl 99 -mla 3 -fa 1 -fmoe 1}"
: "${REF_PROMPTS:=$(cd "${BASH_SOURCE[0]%/*}/.." && pwd)/prompts.tsv}"
: "${REF_CTX:=512}"
: "${REF_SET_CPU:=ref}"
: "${REF_SET_CUDA:=ref_cuda}"
