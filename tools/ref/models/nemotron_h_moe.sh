#!/usr/bin/env bash
# shellcheck shell=bash
# The nemotron_h_moe (Nemotron 3.5 Bolt 74B-A9B, NVFP4 24 GB) profile: what the reference harness needs that is a
# property of the model. Sourced by ref-paths.sh when BLOOMERY_MODEL=nemotron_h_moe; never
# executed, and it exports nothing.
#
# The name is the GGUF general.architecture value, so `grep nemotron_h_moe` finds this profile,
# the metadata keys and the logits set (crates/refset/src/arch/nemotron_h_moe).
#
#   MODEL           the single-file 24 GB GGUF. BLOOMERY_REF_MODEL moves it; a caller's own MODEL= is ignored.
#   LCPP            mainline llama.cpp at cd0a5b6a1; its loader accepts per-layer expert counts. The file's
#                   layers hold 240 to 496 routed experts, and the older tree the nemotron_h profile names sizes
#                   every layer at expert_count. A tree of its own, so the tokenizer pins (crates/refset/src/arch/tokenizer.rs)
#                   stay the Nano's. ik has no nemotron_h_moe.
#   IK              no ik oracle. The default is only so IKBIN can be formed; nothing here runs it.
#
# No lease and no card: the dump hides CUDA (tools/ref/dump-nemotron.sh, the one script for both nemotron profiles).
#
# SC2034: every name here is read by the file that sources this one.
# shellcheck disable=SC2034
MODEL_NAME=nemotron_h_moe
: "${IK:=/home/user/ik-idxkey}"
: "${LCPP:=/home/user/llama.cpp-mainline-bolt}"
MODEL=${BLOOMERY_REF_MODEL:-/models/nemotron-3.5-bolt/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf}
