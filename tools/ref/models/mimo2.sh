#!/usr/bin/env bash
# shellcheck shell=bash
# The mimo2 (MiMo-V2.6-Flash) profile: everything the reference engine and the harnesses need that is
# a property of the *model*, not of the machine. Sourced by ref-paths.sh when BLOOMERY_MODEL=mimo2;
# never executed, and it exports nothing. Same shape as qwen3moe.sh.
#
# The name is the GGUF general.architecture value (shard 1's header says mimo2), so `grep mimo2`
# finds this profile, the metadata keys (mimo2.*), the oracle sets and their refset family
# (crates/refset/src/arch/mimo2) at once.
#
#   MODEL           shard 1 of the MOPD checkpoint's MXFP4 split set (two shards; the first holds the
#                   header and no tensor, the second the 167,363,516,512 B of tensors); the loader
#                   follows split.count from there, and the first shard's full path is the identity
#                   every oracle set states. The mtp-*.gguf file beside it is not loaded.
#                   BLOOMERY_REF_MODEL moves it; a caller's own MODEL= is ignored, as in qwen3moe.sh
#   IK              moves the whole ik tree. The default is the mimo2 oracle tree, not the one the
#                   V4.1, qwen3moe, qwen35moe, glm5next and qwen4exp profiles name: this file's
#                   attention is the fused attn_qkv, which ik loads from upstream 043ced9a on, past
#                   that tree's base. /home/user/ik-mimo2 is ik at 043ced9a with the one oracle commit
#                   upstream lacks cherry-picked (tools/ref/build-ik-mimo2.sh, which names the other
#                   four, checks the tree and writes its PROVENANCE); the family pins its HEAD (IK_BUILD)
#   REF_BIN_NAME    the directory under $BLOOMERY_DATA holding the dump_ref linked against IK
#                   (bin-mimo2; build-ik-mimo2.sh writes it); the installed bin/dump_ref stays the
#                   shared tree's
#   REF_CTX         the one context the batch set is produced at
#   REF_SET_CPU     ref_mimo2 under $BLOOMERY_DATA: ik's CPU dump
#   REF_SET_CUDA    ref_cuda_mimo2: a name only — no CUDA set is made for this model yet, and the
#                   file does not fit on one card at -ngl 99 (justfile, dump-ref-mimo2-cuda)
#   REF_TOKENS      the oracle's token ids; dump.sh's BLOOMERY_REF_TOKENS overrides them
#   REF_DUMP_LEASE  1: every ik reference run takes the machine-wide CPU lease, and the dump pages in
#                   the whole 167.4 GB split set, most of the page cache — more than the machine can
#                   share with a timed run. Not overridable from here.
#   ROUTER_TRACE_ARGS  harness arguments every router trace of this model adds (router-trace.sh), an
#                   array: --all-logits, below
#   ref_step_variant  the decode-step variants, below
#
# No REF_DUMP_ARGS: without --defer-experts the loader populates the whole split set, which fits in
# the page cache (unlike V4.1's), so a later dump of the same file reads nothing from the device.
#
# Deliberately unset: IK_BEST_FLAGS, REF_PROMPTS and the reference-line arms (IK_GPU_FLAGS, LCPP,
# LCPP_GPU_FLAGS, MRS, MRS_FLAGS). Mainline llama.cpp builds this architecture
# (src/models/mimo2.cpp), so it is the public baseline's engine, but no flag has been chosen or
# measured for this model; the depth and prompt-processing lines against it belong to a later
# round. Every script that reads them runs under `set -u`, so such a script stops at the unset
# name instead of running an engine at flags nobody chose.
#
# SC2034: every name here is read by the file that sources this one, which shellcheck does not
# see from this file alone.
# shellcheck disable=SC2034
MODEL_NAME=mimo2
: "${IK:=/home/user/ik-mimo2}"
: "${REF_BIN_NAME:=bin-mimo2}"
MODEL=${BLOOMERY_REF_MODEL:-/models/MiMo-V2.6-Flash-MOPD/MiMo-V2.6-Flash-MOPD-MXFP4-00001-of-00002.gguf}
: "${REF_CTX:=512}"
: "${REF_SET_CPU:=ref_mimo2}"
: "${REF_SET_CUDA:=ref_cuda_mimo2}"
# "The capital of France is" under this model's tokenizer: what `$IK/build/bin/llama-tokenize
# -m /models/MiMo-V2.6-Flash-RL/MiMo-V2.6-Flash-RL-MXFP4-00001-of-00002.gguf -p "The capital of France is"
# --ids --log-disable --no-parse-special` prints (it loads the vocabulary only; the RL checkpoint's
# vocabulary is MOPD's — 152,576 tokens, merges and types byte-identical between the two files'
# headers). Five ids and no BOS: the file sets tokenizer.ggml.add_bos_token to false, so the first
# id is 'The' and not a BOS id. Changing them invalidates the whole set.
REF_TOKENS=785,6722,315,9625,374
REF_DUMP_LEASE=1
# The router trace (router_trace.cpp) reads every layer's `ffn_moe_topk-<layer>` for every token of a
# decode call. ik's build_mimo2 hands the last layer's attention the output rows (src/graphs/
# build_mimo2.cpp: attn_out_ids), so that layer's FFN, and its router, sees only the rows the call asks
# logits for: one, the last token, by default. The graphs of the profiles traced so far (V4.1, glm5next,
# qwen4exp) select the output rows after the last layer. --all-logits asks for every row (n_outputs =
# n_tokens: the selection is the identity), so layer 47's selection covers the call's tokens; without it
# the harness stops at that layer with a shape error. Layer 0 is dense, so the set holds layers 1-47
# (the dense layer 0 and 47 routed, 256 experts and 8 used: crates/model/src/arch/mimo2/program.rs, its
# tests' `real` and ROUTER).
ROUTER_TRACE_ARGS=(--all-logits)

# Decode-step variants, `dump.sh <variant>` (`just dump-ref-mimo2 <variant>`): one decode step
# dumped after a quiet prefill (dump_ref.cpp, --decode-step), each into a set of its own — dump.sh
# refuses REF_SET_CPU and REF_SET_CUDA for them. `ref_step_variant <name>` sets STEP_SET, STEP_CTX,
# STEP_PREFILL, STEP_TOKENS (or STEP_TOKENS_FILE and STEP_TOKENS_SHA256) and STEP_ARGS as
# models/deepseek41.sh describes, and returns 1 for a name it does not know. The suffix
# `-every-node` runs the prefill under the dumped schedule instead of the fused one
# (--prefill-every-node), so the caches and the recurrent state the step reads carry a dumped
# prefill's arithmetic.
#   step4  the oracle's five ids: a quiet prefill of 4, the step at position 4, -c 512
#   d1k    a quiet prefill of 1,024 ids of prose, the step at position 1,024, -c 2048: past the
#          128-position sliding window, so every SWA layer's step reads a window of real cells
#   d4096  a quiet prefill of 4,096 ids of prose, the step at position 4,096, -c 4608: the nine
#          full-attention layers (hybrid_layer_pattern's 0s) read the whole 4,097-position context
# The prose is $BLOOMERY_DATA/mimo2/corpus-prose.ids: the tokenizer oracle's prose text
# ($BLOOMERY_DATA/tokenizer-qwen3moe/prose.txt, ik's docs/**/*.md and README.md, md5
# c6bb074439479420faffa33257424c00, the text the qwen3moe, qwen35moe, glm5next and qwen4exp sets
# are of) through ik-idxkey's llama-tokenize (md5 8dd4fe8a9bb773f80f8345c87f019b1b) with the RL
# checkpoint's vocabulary, --no-parse-special, one id per line, 74,364 ids — this vocabulary's
# count; the other profiles' ids under theirs differ, so the file is this profile's own. The
# sha256 below is the check.
ref_step_variant() {
  local name=$1 every_node=0
  case $name in
    *-every-node) every_node=1; name=${name%-every-node} ;;
  esac
  STEP_TOKENS='' STEP_TOKENS_FILE='' STEP_TOKENS_SHA256=''
  STEP_ARGS=()
  case $name in
    step4)  STEP_SET=ref_mimo2_step4;  STEP_CTX=512;  STEP_PREFILL=4;    STEP_TOKENS=$REF_TOKENS ;;
    d1k)    STEP_SET=ref_mimo2_d1k;    STEP_CTX=2048; STEP_PREFILL=1024 ;;
    d4096)  STEP_SET=ref_mimo2_d4096;  STEP_CTX=4608; STEP_PREFILL=4096 ;;
    *)     return 1 ;;
  esac
  case $name in
    d1k|d4096) STEP_TOKENS_FILE=$BLOOMERY_DATA/mimo2/corpus-prose.ids
               STEP_TOKENS_SHA256=9444bc5b4e2a7c5f4caac4f1aa7b6ef8e945b114fdecac52aca36a1de4e76cf6 ;;
  esac
  if [ "$every_node" = 1 ]; then
    STEP_SET=${STEP_SET}_every_node
    STEP_ARGS+=(--prefill-every-node)
  fi
}
