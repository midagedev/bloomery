#!/usr/bin/env bash
# shellcheck shell=bash
# The kimi-k3 (Kimi K3, unsloth/Kimi-K3-GGUF UD-IQ1_S) profile: what the gates need that is a property of the model. Sourced by
# ref-paths.sh when BLOOMERY_MODEL=kimi-k3; never executed, and it exports nothing.
#
# The name is the GGUF general.architecture value, so `grep kimi-k3` finds this profile, the registry row (crates/model/src/arch/kimi_k3)
# and the reader's type table (crates/gguf/src/arch/kimi_k3.rs).
#
#   MODEL   shard 1 of the fourteen-shard UD-IQ1_S split set. Shard 1 holds the metadata and no tensors; the tensors are in shards 2
#           to 14, so the reader opens the whole set from shard 1. BLOOMERY_REF_MODEL moves it. The path is the file's identity: a set
#           names its file's full path, so the file is not moved or deleted before the sets are re-dumped.
#
# No reference engine is named: mainline llama.cpp is Kimi K3's numeric oracle, but no recipe of this profile builds or runs it yet; the
# default IK is only so IKBIN can be formed.
#
# SC2034: every name here is read by the file that sources this one.
# shellcheck disable=SC2034
MODEL_NAME=kimi-k3
: "${IK:=/home/user/ik-idxkey}"
MODEL=${BLOOMERY_REF_MODEL:-/models/Kimi-K3-UD-IQ1_S/Kimi-K3-UD-IQ1_S-00001-of-00014.gguf}
