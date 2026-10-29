#!/usr/bin/env bash
# shellcheck shell=bash
# The qwen38d (Qwen3.8-27B, dense) profile: what the reference engine and the harnesses need that is a property of the
# *model*, not of the machine. Sourced by ref-paths.sh when BLOOMERY_MODEL=qwen38d; never executed, and it exports
# nothing. The file's GGUF general.architecture is qwen35 (the profile's name picks the file, the architecture picks the
# engine), so `grep qwen35` finds the metadata keys (qwen35.*) and the engine's reader, and `grep qwen38d` this profile,
# its oracle sets and their refset family (crates/refset/src/arch/qwen35/dense.rs).
#
#   MODEL           lmstudio-community's Qwen3.8-27B Q4_K_M (rev 5a7da681, 16,810,714,336 bytes, sha256 e00082f7...:
#                   tools/ref/dense-logits.py TARGETS holds the identity the writer checks before it opens the file).
#                   BLOOMERY_REF_MODEL moves it. The fixture tier's model is the NVFP4 cut folder tools/fixture-dirs.tsv
#                   names for this profile (tools/fixture-cuts/qwen38nv.json), whose GGUF twin and oracle sets are the
#                   qwen38nv profile's (tools/ref/models/qwen38nv.sh); this file is the real tier's
#   LCPP            the mainline llama.cpp tree that is the numeric oracle: the qwen35 profile's, at LCPP_PIN
#                   (src/models/qwen35.cpp, LLM_ARCH_QWEN35). dense_logits_ref (tools/ref/dense_logits_ref.cpp) links against its
#                   build/bin libraries (tools/ref/build-dense-logits.sh), which tools/ref/emb2/oracle-tree.sh proves are
#                   the commit's build. IK is the default the scripts that expand IKBIN beside every profile take; no ik
#                   oracle exists for this profile
#   PROSE_IDS       the prose corpus, the qwen35 profile's file: the 27B's llama-tokenize --ids --no-parse-special over the
#                   same prose text ($BLOOMERY_DATA/tokenizer-qwen3moe/prose.txt, md5 c6bb074439479420faffa33257424c00) gives
#                   76,175 ids with the sha256 the qwen35 profile pins, so the ids are one file and PROSE_SHA256 is the check
#   LOGITS_TARGET   tools/ref/dense-logits.py TARGETS' name for MODEL: its size, sha256, vocabulary and architecture
#   LOGITS_SETS     the oracle sets, `<set>:<P>`: the first P ids of PROSE_IDS are the prompt and the LOGITS_FORCED ids after
#                   them the forced continuation (teacher forced on the corpus), each row mainline's logits after the prompt
#                   or after a forced id (tools/ref/dense-logits.sh)
#   LOGITS_CTX/UBATCH  the context every set is produced at (P = 4,096 and 32 forced ids fit; the card engine's
#                   context is 4,352 too) and llama.cpp's default ubatch
#
# SC2034: every name here is read by the file that sources this one, which shellcheck does not see from this file alone.
# shellcheck disable=SC2034
MODEL_NAME=qwen38d
: "${IK:=/home/user/ik-idxkey}"
MODEL=${BLOOMERY_REF_MODEL:-/models/Qwen3.8-27B/Qwen3.8-27B-Q4_K_M.gguf}
: "${LCPP:=/home/user/llama.cpp-mainline}"
LCPP_PIN=53ed051ce5e8193652e449f43216ca3859454f49
PROSE_IDS=${BLOOMERY_DATA:-/root/bloomery-data}/qwen35/corpus-prose.ids
PROSE_SHA256=4bdf4171c7e1d0fd37d61b4fd13fdfb66c0dea1806d03b0398f851da5a350223
LOGITS_TARGET=qwen38-27b
LOGITS_FORCED=32
LOGITS_CTX=4352
LOGITS_UBATCH=512
LOGITS_SETS=(ref_qwen38d_logits_p64:64 ref_qwen38d_logits_p4096:4096)
