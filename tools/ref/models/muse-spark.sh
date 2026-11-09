#!/usr/bin/env bash
# shellcheck shell=bash
# The muse-spark (Muse-Spark-Open-816B-A42B, UD-IQ2_XXS) profile: what the reference harness needs that is a
# property of the model. Sourced by ref-paths.sh when BLOOMERY_MODEL=muse-spark; never executed, and it exports
# nothing.
#
# The name is the GGUF general.architecture value, so `grep muse-spark` finds this profile, the reader
# (crates/model/src/arch/musespark) and the oracle family (crates/refset/src/arch/musespark).
#
#   MODEL   shard 1 of the six-shard UD-IQ2_XXS split set (unsloth/Muse-Spark-Open-816B-A42B-GGUF, 225.6 GB;
#           muse-dl.sh lands it here). BLOOMERY_REF_MODEL moves it; a caller's own MODEL= is ignored. The
#           path is the family's identity (refset::arch::musespark::MODEL): a set names its file's full path,
#           so the file is not moved or deleted before its sets are re-dumped.
#   LCPP    mainline llama.cpp at PR 30616's merge, ba3c05f71da85cd0439e84ba42b0478b24d5db16
#           (/home/user/llama.cpp-muse, built CPU+CUDA by ~/.cache/bloomery/muse/build-llama-muse.sh):
#           src/models/muse-spark.cpp is mainline's own graph, so the oracle (tools/ref/musespark-dump.sh)
#           and the public baseline are one engine. ik has no muse-spark; the default IK is only so IKBIN
#           can be formed.
#
# The dump (tools/ref/musespark-dump.sh) hides CUDA and takes the CPU lease.
#
# Deliberately unset: IK_BEST_FLAGS, REF_SET_CPU, REF_TOKENS and the reference-line arms (LCPPBIN,
# LCPP_GPU_FLAGS, MRS, MRS_FLAGS). The file does not fit one card, and no baseline flag has been chosen or
# measured for this model; the depth and prompt-processing lines against mainline belong to the placement
# phase (the design's phase (c)). Every script that reads them runs under `set -u`, so such a script stops
# at the unset name instead of running an engine at flags nobody chose.
#
# SC2034: every name here is read by the file that sources this one.
# shellcheck disable=SC2034
MODEL_NAME=muse-spark
: "${IK:=/home/user/ik-idxkey}"
: "${LCPP:=/home/user/llama.cpp-muse}"
MODEL=${BLOOMERY_REF_MODEL:-/data/moved/models/Muse-Spark-Open-816B-A42B-UD-IQ2_XXS/Muse-Spark-Open-816B-A42B-UD-IQ2_XXS-00001-of-00006.gguf}
