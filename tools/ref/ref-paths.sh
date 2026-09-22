#!/usr/bin/env bash
# shellcheck shell=bash
# The default paths of the reference model, the data directory and the ik tree, for the build
# scripts (through ref-build-common.sh) and the runners alike. Sourced, never executed; it only
# defines variables, and exports none: a script whose children read one exports it itself.
#
# One place so that a harness built against one ik tree is not timed against another's
# llama-bench, and no runner reads a model or a data directory the harnesses and gates do not.
# Kept apart from ref-build-common.sh because that file is build-only (the caller sets -e first,
# it defines compiler flags) and the runners never compile.
#
# MODEL and BLOOMERY_DATA come from the variables the C++ harnesses (ref_paths.h) and the Rust
# gates (bloomery_gpu_gates::{ref_model_path, data_dir}) read, with the same defaults, so one
# override reaches all three: BLOOMERY_REF_MODEL moves the model, BLOOMERY_DATA the data
# directory. An empty variable counts as unset, as in ref_paths.h. BLOOMERY_REF_MODEL itself is
# left as the caller set it; MODEL is the name the scripts use.
#
# IK and IKBIN each keep their override: IK moves the whole tree (and IKBIN with it), IKBIN
# alone points at a different llama-bench.
#
# SC2034: the sourcing script reads these, which shellcheck does not see in this file alone.
# shellcheck disable=SC2034
: "${IK:=/home/user/ik_llama.cpp}"
IKBIN=${IKBIN:-$IK/build/bin/llama-bench}
MODEL=${BLOOMERY_REF_MODEL:-/models/small/DeepSeek-V2-Lite-Chat.Q3_K_M.gguf}
: "${BLOOMERY_DATA:=/root/bloomery-data}"
