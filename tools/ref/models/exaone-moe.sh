#!/usr/bin/env bash
# shellcheck shell=bash
# The exaone-moe (LG K-EXAONE-2.0-750B-A37B) profile: what the gates need that is a property of the model. Sourced by
# ref-paths.sh when BLOOMERY_MODEL=exaone-moe; never executed, and it exports nothing.
#
# The name is the GGUF general.architecture value, so `grep exaone-moe` finds this profile, the registry row
# (crates/model/src/arch/exaone_moe) and the family (crates/refset/src/arch/exaone_moe).
#
#   MODEL   the fixture GGUF (bloomery-quantize --hf of the cut folder /models/fixtures/exaoneh, its rules
#           tools/fixture-cuts/exaoneh.json): the bootstrap mix of crates/gguf/src/arch/exaone_moe.rs — the routed stacks
#           Q4_K, the trunk and the head Q8_0, F32 norms, router and correction bias — the one file both engines read
#           (the engine opens no FP8 folder: an FP8 attention site has no card column). BLOOMERY_REF_MODEL moves it. The
#           path is the family's identity (refset::arch::exaone_moe::MODEL): a set names its file's full path, so the
#           file is not moved or deleted before the sets are re-dumped. The real 752 GB FP8 checkpoint's own bootstrap
#           quantization names its file here when it is written.
#
#   LCPP    mainline llama.cpp at 7bcbc04fb carrying the exaone-moe oracle patch (exaone-moe-oracle.patch: the
#           sliding-window, clamp and arch lists the file's per-layer arrays need), the CPU backend alone: the oracle
#           (tools/ref/exaone-moe-dump.sh) and the tokenizer's reference (llama-tokenize). ik has no exaone-moe; the
#           default IK is only so IKBIN can be formed.
#
# The dump (tools/ref/exaone-moe-dump.sh) hides CUDA and takes the CPU lease.
#
# SC2034: every name here is read by the file that sources this one.
# shellcheck disable=SC2034
MODEL_NAME=exaone-moe
: "${IK:=/home/user/ik-idxkey}"
: "${LCPP:=/root/agents/specs/exaone-r6a/work/llama.cpp}"
MODEL=${BLOOMERY_REF_MODEL:-/models/fixtures/exaoneh-gguf/exaoneh-gguf-fixture-00001-of-00001.gguf}
