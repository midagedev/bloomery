#!/usr/bin/env bash
# shellcheck shell=bash
# The minimax-m3 profile (MiniMax-M3.1-Flash, 428B total / 23B active): what the reference harness needs that is a property of the
# model. Sourced by ref-paths.sh when BLOOMERY_MODEL=minimax-m3; never executed, and it exports nothing.
#
# The name is the GGUF general.architecture value, so `grep minimax-m3` finds this profile, the registry row and the oracle
# families (crates/refset/src/arch/minimaxm31, crates/refset/src/arch/minimaxm3).
#
#   MODEL   the vendor's checkpoint folder (MiniMaxAI/MiniMax-M3.1-Flash, 48 safetensors shards, the MXFP8 projections and NVFP4
#           routed experts as stored): the engine and the gates read it in its own formats, with no GGUF. BLOOMERY_REF_MODEL
#           moves it. The fixture tier's model is the folder `fixture cut` writes (tools/fixture-dirs.tsv, tools/fixture-cuts/
#           minimaxm31.json), resolved by ref-paths.sh. The oracle is transformers' f32 dump of this folder (refset family
#           tf-minimaxm31, `just dump-ref-minimaxm31`, tools/ref/mm31-dump.sh).
#   LCPP    mainline llama.cpp at cd0a5b6a1 (/home/user/llama.cpp-mainline-bolt, a CPU build, no patch): the route through MiniMax-M3's
#           GGUF (MiniMax-M3-UD-IQ4_XS, unsloth/MiniMax-M3-GGUF, 208.5 GB), which `just dump-ref-minimaxm3` names as its model
#           (BLOOMERY_REF_MODEL, the first shard of the six-shard split set; the file is the family lcpp-minimaxm3's identity, so it
#           is not moved or deleted before its sets are re-dumped). That route is kept as code and is unverified for M3.1; ik is not
#           the oracle here, the default IK is only so IKBIN can be formed.
#
# SC2034: every name here is read by the file that sources this one.
# shellcheck disable=SC2034
MODEL_NAME=minimax-m3
: "${IK:=/home/user/ik-idxkey}"
: "${LCPP:=/home/user/llama.cpp-mainline-bolt}"
MODEL=${BLOOMERY_REF_MODEL:-/data/moved/models/MiniMax-M3.1-Flash}
