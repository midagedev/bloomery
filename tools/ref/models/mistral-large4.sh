#!/usr/bin/env bash
# shellcheck shell=bash
# The mistral-large4 (Mistral-Large-4-1T-A52B, UD-IQ1_S) profile: what the reference harness needs that is a property of the
# model. Sourced by ref-paths.sh when BLOOMERY_MODEL=mistral-large4; never executed, and it exports nothing.
#
# The name is the GGUF general.architecture value, so `grep mistral-large4` finds this profile, the reader
# (crates/model/src/arch/mistrallarge4) and the oracle family (crates/refset/src/arch/mistrallarge4).
#
#   MODEL   shard 1 of the six-shard UD-IQ1_S split set (unsloth/Mistral-Large-4-1T-A52B-GGUF, 223 GB). The files live
#           on the root drive and /models/Mistral-Large-4-1T-A52B-UD-IQ1_S is a symlink to their directory.
#           BLOOMERY_REF_MODEL moves it; a caller's own MODEL= is ignored. The path is the family's identity
#           (refset::arch::mistrallarge4::MODEL): a set names its file's full path, so the file is not moved or
#           deleted before its sets are re-dumped.
#   LCPP    mainline llama.cpp at PR 31376's merge, 78feec062d50ec9e8940244cac918ac4480b60cb
#           (/home/user/llama.cpp-ml4, the CPU backend alone, built by `just build-ml4-lcpp`):
#           src/models/mistral-large4.cpp is mainline's own graph, so the oracle (tools/ref/mistrallarge4-dump.sh)
#           and the public baseline are one engine. A CUDA build for the baseline lines is another tree at the same
#           commit: a rebuild of this one moves the libllama md5 the oracle sets name. The default IK is only so
#           IKBIN can be formed: ik has no mistral-large4 architecture.
#
# The dump (tools/ref/mistrallarge4-dump.sh) hides CUDA and takes the CPU lease.
#
# Deliberately unset: IK_BEST_FLAGS, REF_SET_CPU, REF_TOKENS and the reference-line arms (LCPPBIN,
# LCPP_GPU_FLAGS, MRS, MRS_FLAGS). The file does not fit one card, and no baseline flag has been chosen or
# measured for this model; the depth and prompt-processing lines against mainline belong to the placement
# phase. Every script that reads them runs under `set -u`, so such a script stops at the unset name instead of
# running an engine at flags nobody chose.
#
# SC2034: every name here is read by the file that sources this one.
# shellcheck disable=SC2034
MODEL_NAME=mistral-large4
: "${IK:=/home/user/ik-idxkey}"
: "${LCPP:=/home/user/llama.cpp-ml4}"
MODEL=${BLOOMERY_REF_MODEL:-/models/Mistral-Large-4-1T-A52B-UD-IQ1_S/Mistral-Large-4-1T-A52B-UD-IQ1_S-00001-of-00006.gguf}
