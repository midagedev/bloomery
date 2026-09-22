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
# gates (bloomery_gpu_gates::{ref_model_path, data_dir}) read, so one override reaches all three:
# BLOOMERY_REF_MODEL moves the model, BLOOMERY_DATA the data directory. ref_paths.h and data_dir
# carry the same defaults; ref_model_path has none, and tools/box.sh exports this file's MODEL as
# BLOOMERY_REF_MODEL into every box command. An empty variable counts as unset, as in
# ref_paths.h. BLOOMERY_REF_MODEL itself is left as the caller set it; MODEL is the name the
# scripts use.
#
# IK and IKBIN each keep their override: IK moves the whole tree (and IKBIN with it), IKBIN
# alone points at a different llama-bench.
#
# What is a property of the model — the model file, the ik tree and flags, the prompt set, the
# reference context, the oracle directories — lives in models/<architecture>.sh and BLOOMERY_MODEL
# picks one. The name is the GGUF general.architecture value. What is a property of the machine
# — BLOOMERY_DATA, and IKBIN's place inside a tree — stays here, shared by every model.
#
# SC2034: the sourcing script reads these, which shellcheck does not see in this file alone.
# shellcheck disable=SC2034
: "${BLOOMERY_MODEL:=deepseek2}"
__ref_paths_profile="${BASH_SOURCE[0]%/*}/models/$BLOOMERY_MODEL.sh"
if [ ! -f "$__ref_paths_profile" ]; then
  echo "ref-paths.sh: no model profile for BLOOMERY_MODEL='$BLOOMERY_MODEL'" >&2
  echo "  expected $__ref_paths_profile" >&2
  exit 64
fi
# shellcheck source=tools/ref/models/deepseek2.sh
source "$__ref_paths_profile"
unset __ref_paths_profile
IKBIN=${IKBIN:-$IK/build/bin/llama-bench}
: "${BLOOMERY_DATA:=/root/bloomery-data}"
