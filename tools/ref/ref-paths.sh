#!/usr/bin/env bash
# shellcheck shell=bash
# The ik tree's default location, for the build scripts (through ref-build-common.sh) and the
# timing runners alike. Sourced, never executed; it only defines variables.
#
# One place so that a harness built against one ik tree is not timed against another's
# llama-bench. Kept apart from ref-build-common.sh because that file is build-only (the caller
# sets -e first, it defines compiler flags) and the runners never compile.
#
# IK and IKBIN each keep their override: IK moves the whole tree (and IKBIN with it), IKBIN
# alone points at a different llama-bench.
#
# SC2034: the sourcing script reads these, which shellcheck does not see in this file alone.
# shellcheck disable=SC2034
: "${IK:=/home/user/ik_llama.cpp}"
IKBIN=${IKBIN:-$IK/build/bin/llama-bench}
