#!/usr/bin/env bash
# shellcheck shell=bash
# The minimax-m3 (MiniMax-M3-UD-IQ4_XS, 428B total / 23B active, unsloth) profile: what the reference harness needs that is a
# property of the model. Sourced by ref-paths.sh when BLOOMERY_MODEL=minimax-m3; never executed, and it exports nothing.
#
# The name is the GGUF general.architecture value, so `grep minimax-m3` finds this profile, the registry row and the oracle
# family (crates/refset/src/arch/minimaxm3).
#
#   MODEL   the first shard of the six-shard UD-IQ4_XS split set (unsloth/MiniMax-M3-GGUF, 208.5 GB). BLOOMERY_REF_MODEL moves
#           it; a caller's own MODEL= is ignored. The path is the family's identity (refset::arch::minimaxm3::MODEL): a set
#           names its file's full path, so the file is not moved or deleted before its sets are re-dumped.
#   LCPP    mainline llama.cpp at cd0a5b6a1 (/home/user/llama.cpp-mainline-bolt, a CPU build, no patch): src/models/minimax-m3.cpp is
#           mainline's own graph (the Solar-Open2 PAI tree carries the same file, older), so the oracle is mainline and the public
#           baseline is the same engine. ik is not the oracle here; the default IK is only so IKBIN can be formed.
#
# The dump (tools/ref/minimaxm3-dump.sh) hides CUDA and takes the CPU lease.
#
# SC2034: every name here is read by the file that sources this one.
# shellcheck disable=SC2034
MODEL_NAME=minimax-m3
: "${IK:=/home/user/ik-idxkey}"
: "${LCPP:=/home/user/llama.cpp-mainline-bolt}"
MODEL=${BLOOMERY_REF_MODEL:-/models/MiniMax-M3-UD-IQ4_XS/MiniMax-M3-UD-IQ4_XS-00001-of-00006.gguf}
