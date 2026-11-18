#!/usr/bin/env bash
# shellcheck shell=bash
# The hy_v4_flash (Tencent Hy4-flash, AngelSlim/Hy4-flash-GGUF Q4_K_M) profile: what the gates need that is a property of the model.
# Sourced by ref-paths.sh when BLOOMERY_MODEL=hy_v4_flash; never executed, and it exports nothing.
#
# The name is the GGUF general.architecture value, so `grep hy_v4_flash` finds this profile, the registry row
# (crates/model/src/arch/hy4), the reader's type table (crates/gguf/src/arch/hy_v4_flash.rs) and the family
# (crates/refset/src/arch/hy4). The fixture directory is tools/fixture-dirs.tsv's `hy4`.
#
#   MODEL   shard 1 of the two-shard Q4_K_M split set (168 GB), the file the gates' header tables were read from. Shard 1 holds
#           the metadata, the vocabulary and the trunk; the engram table (per_layer_token_embd, 59 GiB of scale-less F8_E4M3
#           rows) is in shard 2, which the reader opens from shard 1. BLOOMERY_REF_MODEL moves it. The path is the file's
#           identity: a set names its file's full path, so the file is not moved or deleted before the sets are re-dumped.
#
#   LCPP    llama.cpp at the head of pull request #32517, 0a054d0b02454c998e068381d15f3f6615bdd987 ("Model : add hy_v4_flash
#           support", /home/user/llama.cpp-hy4, the CPU backend alone, built once for the tokenizer's oracle): src/models/
#           hy-v4-flash.cpp is the PR's own graph, so the oracle (tools/ref/hy4-dump.sh) and the format's reference are one
#           engine. The tree is not rebuilt for a dump: a rebuild moves the libllama md5 the oracle sets name. The default IK
#           is only so IKBIN can be formed: ik has no Hy4 graph.
#
# The dump (tools/ref/hy4-dump.sh) hides CUDA and takes the CPU lease.
#
# SC2034: every name here is read by the file that sources this one.
# shellcheck disable=SC2034
MODEL_NAME=hy_v4_flash
: "${IK:=/home/user/ik-idxkey}"
: "${LCPP:=/home/user/llama.cpp-hy4}"
MODEL=${BLOOMERY_REF_MODEL:-/models/Hy4-flash-GGUF/Hy4-flash-Q4_K_M-00001-of-00002.gguf}
