#!/usr/bin/env bash
# shellcheck shell=bash
# The raon1nano (Raon-1-nano-preview Q8_0, 3.1B, KRAFTON) profile: what the reference harness needs that is a
# property of the model. Sourced by ref-paths.sh when BLOOMERY_MODEL=raon1nano; never executed, and it exports
# nothing.
#
# The name is the GGUF general.architecture value, so `grep raon1nano` finds this profile, the registry row and
# the oracle family (crates/refset/src/arch/raon1nano).
#
#   MODEL   the Q8_0 file (3.4 GB, one file; the family's real file and its own fixture — small enough to stand
#           in both roles, so its fixture-dirs.tsv row is `self` and a fixture-tier dump or gate is refused by
#           name on the file, never a cut written). BLOOMERY_REF_MODEL moves it; a caller's own MODEL= is
#           ignored. The path is the family's identity (refset::arch::raon1nano::MODEL): a set names its file's
#           full path, so the file is not moved or deleted before its sets are re-dumped.
#   LCPP    KRAFTON's llama.cpp branch raon-1-nano-preview at LCPP_COMMIT (/home/user/llama.cpp-4d72ff0d, a CPU
#           build, no patch): src/models/raon1nano.cpp is the vendor's own graph, so the oracle is that branch
#           and the public baseline is the same engine, named as a vendor patch in every table. The tree is the
#           raon-nano oracle round's build (its MTP draft head commit). ik has no Raon graph; the default IK is
#           only so IKBIN can be formed.
#
# The dump (tools/ref/raon1nano-dump.sh) hides CUDA and takes the CPU lease.
#
# SC2034: every name here is read by the file that sources this one.
# shellcheck disable=SC2034
MODEL_NAME=raon1nano
: "${IK:=/home/user/ik-idxkey}"
: "${LCPP:=/home/user/llama.cpp-4d72ff0d}"
LCPP_COMMIT=4d72ff0d45d5795aa4bd9655a77501269b52e1bc
MODEL=${BLOOMERY_REF_MODEL:-/data/models/Raon-1-nano-gguf/raon-1-nano-Q8_0.gguf}
