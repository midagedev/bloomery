#!/usr/bin/env bash
# shellcheck shell=bash
# The nemotron_h (Nemotron 3 Nano 4B) profile: what the reference harness needs that is a
# property of the model. Sourced by ref-paths.sh when BLOOMERY_MODEL=nemotron_h; never
# executed, and it exports nothing.
#
# The name is the GGUF general.architecture value, so `grep nemotron_h` finds this profile,
# the metadata keys and the logits set (crates/refset/src/arch/nemotron_h).
#
#   MODEL           the Q4_K_M file the tokenizer family already names. One shard.
#                   BLOOMERY_REF_MODEL moves it; a caller's own MODEL= is ignored.
#   LCPP            mainline llama.cpp. ik has no nemotron_h, so this tree is the oracle.
#                   nemotron_ref links its existing build and does not rebuild it: a rebuild
#                   moves the md5s the tokenizer family pins.
#   IK              no ik oracle. The default is only so IKBIN can be formed; nothing here runs it.
#
# No lease and no card: the file is a few gigabytes and the dump hides CUDA.
#
# SC2034: every name here is read by the file that sources this one.
# shellcheck disable=SC2034
MODEL_NAME=nemotron_h
: "${IK:=/home/user/ik-idxkey}"
: "${LCPP:=/home/user/llama.cpp-mainline}"
MODEL=${BLOOMERY_REF_MODEL:-/models/NVIDIA-Nemotron-3-Nano-4B-GGUF/NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf}
