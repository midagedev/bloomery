#!/usr/bin/env bash
# shellcheck shell=bash
# The flint (Flint, unsloth/flint-GGUF UD-Q4_K_XL) profile: what the gates need that is a property of the model. Sourced by
# ref-paths.sh when BLOOMERY_MODEL=flint; never executed, and it exports nothing.
#
# The name is the GGUF general.architecture value, so `grep flint` finds this profile, the registry row (crates/model/src/arch/flint)
# and the card body (crates/gpu-flint).
#
#   MODEL   the complete Q4_K_XL file (25,054,688,704 bytes, one file; crates/gpu-flint/src/two_card_tests.rs pins its size and
#           sha256). BLOOMERY_REF_MODEL moves it. The gpu-flint lib gates read the file as BLOOMERY_MODEL, the name its test harnesses
#           take for a model path, so their recipes hand this profile's file to the test under that name. The family has no fixture
#           file (no row in tools/fixture-dirs.tsv), so the fixture tier runs none of them.
#
# No reference engine is named: mainline llama.cpp is Flint's numeric oracle, but no recipe of this profile builds or runs it; the
# default IK is only so IKBIN can be formed.
#
# SC2034: every name here is read by the file that sources this one.
# shellcheck disable=SC2034
MODEL_NAME=flint
: "${IK:=/home/user/ik-idxkey}"
MODEL=${BLOOMERY_REF_MODEL:-/models/flint-GGUF/flint-UD-Q4_K_XL.gguf}
