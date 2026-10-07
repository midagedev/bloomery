#!/usr/bin/env bash
# shellcheck shell=bash
# The qwen35 (Qwen3.5 dense; Cloudflare Clef's backbone) profile: what the reference engine and the
# harnesses need that is a property of the *model*, not of the machine. Sourced by ref-paths.sh when
# BLOOMERY_MODEL=qwen35; never executed, and it exports nothing.
#
# The name is the GGUF general.architecture value, so `grep qwen35` finds this profile, the metadata keys
# (qwen35.*), the hidden-state oracle sets and their refset family (crates/refset/src/arch/qwen35).
#
#   MODEL           Clef's text backbone (Cloudflare/clef, Qwen3_5ForConditionalGeneration) converted by
#                   mainline's convert_hf_to_gguf.py (--no-mtp: the checkpoint carries no MTP block) to BF16
#                   and quantized by its llama-quantize to Q4_K_M: tensor types Q4_K, Q6_K and F32, 16.55 GB.
#                   The head's and tokenizer's files (joint_head*, tokenizer*, config.json) sit in
#                   $BLOOMERY_DATA/clef/27b/. BLOOMERY_REF_MODEL moves it
#   LCPP            the mainline llama.cpp tree that converts, quantizes and is the numeric oracle
#                   (src/models/qwen35.cpp, LLM_ARCH_QWEN35 at 53ed051ce); hidden_ref links against its
#                   build/bin libraries (tools/ref/build-hidden.sh)
#   IK              no ik oracle exists for this profile; the default tree is set for the scripts that
#                   expand IKBIN beside every profile
#   PROSE_IDS       the prose corpus under this vocabulary: the tokenizer oracle's prose text
#                   ($BLOOMERY_DATA/tokenizer-qwen3moe/prose.txt, md5 c6bb074439479420faffa33257424c00)
#                   through mainline's llama-tokenize -m $MODEL --ids --no-parse-special, one id per line,
#                   76,175 ids; PROSE_SHA256 is the check
#   HIDDEN_SETS     the hidden-state oracle sets, `<set>:<P>[:<file>]`: the first P ids of PROSE_IDS
#                   through <file> (MODEL when absent), each llama.cpp's result_norm of every position
#                   (tools/ref/hidden.sh). The flashq8 sets are Clef-Flash as bartowski publishes it at
#                   Q8_0 (rev d7f376ea; its tokenizer is the 27B's byte for byte, so the ids are the same);
#                   the lev sets are LEV_MODEL below (Qwen3.5-4B, 32 layers of width 2560, its head tied to the
#                   token embedding; the same vocabulary, so the same ids)
#
#   LEV_LCPP        the tree of lev's /v1/systemone oracle (tools/ref/lev_ref.py): mainline llama.cpp at
#                   LEV_LCPP_COMMIT, the commit that added the decision server (PR #29818), built with
#                   tools/ref/lev/lcpp-ids.patch (tools/ref/build-lcpp-lev.sh). Its own tree, so LCPP above
#                   (the hidden-state oracle, which has no decision server) does not move
#   LEV_MODEL       ggml-org/lev-GGUF's Q4_K_M (3,011,777,440 bytes, a Qwen3.5-4B dense file, `qwen35`,
#                   `qwen35.decision.type = lev`), under /root/models: the box's /models is not a round's
#
# SC2034: every name here is read by the file that sources this one.
# shellcheck disable=SC2034
MODEL_NAME=qwen35
: "${IK:=/home/user/ik-idxkey}"
MODEL=${BLOOMERY_REF_MODEL:-/models/clef-27b/clef-27b-Q4_K_M.gguf}
: "${LCPP:=/home/user/llama.cpp-mainline}"
PROSE_IDS=${BLOOMERY_DATA:-/root/bloomery-data}/qwen35/corpus-prose.ids
PROSE_SHA256=4bdf4171c7e1d0fd37d61b4fd13fdfb66c0dea1806d03b0398f851da5a350223
FLASH_Q8=/models/clef-flash/Cloudflare_clef-flash-Q8_0.gguf
LEV_MODEL=/root/models/lev-4b/lev-Q4_K_M.gguf
HIDDEN_SETS=(ref_qwen35_hidden_p64:64 ref_qwen35_hidden_p600:600 ref_qwen35_hidden_p4096:4096
  "ref_qwen35_flashq8_hidden_p64:64:$FLASH_Q8" "ref_qwen35_flashq8_hidden_p600:600:$FLASH_Q8"
  "ref_qwen35_flashq8_hidden_p4096:4096:$FLASH_Q8"
  "ref_qwen35_lev_hidden_p64:64:$LEV_MODEL" "ref_qwen35_lev_hidden_p600:600:$LEV_MODEL"
  "ref_qwen35_lev_hidden_p4096:4096:$LEV_MODEL")
LEV_LCPP=/home/user/llama.cpp-lev
LEV_LCPP_COMMIT=a4cb4c61fd9d9c2066c7c1747821d3d65b8943bd
