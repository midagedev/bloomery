#!/usr/bin/env bash
# shellcheck shell=bash
# The qwen35moe (Qwen3.6-35B-A3B) profile: everything the reference engine and the harnesses need
# that is a property of the *model*, not of the machine. Sourced by ref-paths.sh when
# BLOOMERY_MODEL=qwen35moe; never executed, and it exports nothing. Same shape as qwen3moe.sh.
#
# The name is the GGUF general.architecture value, so `grep qwen35moe` finds this profile, the
# metadata keys (qwen35moe.*), the oracle sets and their refset family (crates/refset/src/arch/qwen35moe).
#
#   MODEL           the one file (no split set): lmstudio's Q4_K_M (tensor types Q4_K, Q6_K and F32).
#                   The oracle sets are dumped from this file; the UD-Q4_K_XL and UD-Q6_K files beside
#                   it share the architecture and the vocabulary but are not the oracle's file.
#                   BLOOMERY_REF_MODEL moves it; a caller's own MODEL= is ignored, as in qwen3moe.sh
#   IK              moves the whole ik tree. The default is the tree the V4.1 and qwen3moe profiles name:
#                   it builds this architecture (src/graphs/build_qwen35.cpp, build_qwen35moe; the gated
#                   delta rule in src/llama-delta-net.cpp) and the installed dump_ref links against it,
#                   so one dumper serves every profile ([foreign-lib] in dump.sh otherwise)
#   REF_CTX         the one context the batch set is produced at
#   REF_SET_CPU     ref_qwen35moe under $BLOOMERY_DATA: ik's CPU dump
#   REF_SET_CUDA    ref_cuda_qwen35moe: a name only — no CUDA set is made for this model yet
#   REF_TOKENS      the oracle's token ids; dump.sh's BLOOMERY_REF_TOKENS overrides them
#   REF_DUMP_LEASE  1: every ik reference run takes the machine-wide CPU lease, and the dump pages in
#                   the whole 21.2 GB file. Not overridable from here.
#   ref_step_variant  the decode-step variants, below
#
#   LCPP            the mainline llama.cpp tree the `lcpp:<D>` and `lcpppp:<P>` arms of
#                   depth-qwen3moe.sh run (`just depth-gpu-qwen35moe`): mainline itself, which builds
#                   this architecture (src/models/qwen35moe.cpp, LLM_ARCH_QWEN35MOE at 53ed051ce), so
#                   no PR branch; LCPPBIN moves its llama-bench alone
#   LCPP_GPU_FLAGS  qwen3moe.sh's: every layer and the head on the card, flash attention on. No
#                   flag sweep has been run on this model
#   MRS             the mistral.rs tree the `mrs:<D>`, `mrspa0:<D>` and `mrspp:<P>` arms run, qwen3moe.sh's
#                   (its release build). It opens this file as `qwen35moe` through its native Qwen3-Next
#                   loader (mistralrs-core/src/gguf/normal_registry.rs: the Qwen35Moe schema and the
#                   Qwen3Next adapter; normal_bindings.rs bind_qwen3_next maps the GDN tensors); MRSBIN
#                   moves its `mistralrs` binary alone
#   MRS_FLAGS       qwen3moe.sh's: `--format gguf`. Under PagedAttention (the mrs and mrspp arms) this
#                   model's prompt runs in chunks of at most 512 tokens whatever P: its hybrid cache sends
#                   the prompt down the scheduler's recurrent path (mistralrs-core/src/paged_attention/
#                   scheduler.rs, select_recurrent_prompt_batch: chunk = min(max_prefill_chunk_tokens,
#                   budget)), not the up to 4096 a step qwen3moe's prompt gets. An mrspp row of this
#                   profile is a prefill in 512-token chunks
#
# Deliberately unset: IK_GPU_FLAGS, IK_GPU_DEFAULT_FLAGS, IK_BEST_FLAGS and REF_PROMPTS. Nobody has
# read whether ik's fused flags (-fmoe, -mqkv, -muge) apply to build_qwen35moe, and no CPU flag sweep
# and no prompt set exist for this model. Every script that reads them runs under `set -u`
# (depth-qwen3moe.sh reads an engine's names only when it has arms), so such a script stops at the
# unset name instead of running an engine at flags nobody chose.
#
# SC2034: every name here is read by the file that sources this one, which shellcheck does not
# see from this file alone.
# shellcheck disable=SC2034
MODEL_NAME=qwen35moe
: "${IK:=/home/user/ik-idxkey}"
MODEL=${BLOOMERY_REF_MODEL:-/models/Qwen3.6-35B-A3B/Qwen3.6-35B-A3B-Q4_K_M.gguf}
: "${REF_CTX:=512}"
: "${REF_SET_CPU:=ref_qwen35moe}"
: "${REF_SET_CUDA:=ref_cuda_qwen35moe}"
# "The capital of France is" under this model's tokenizer: what `$IK/build/bin/llama-tokenize
# -m $MODEL -p "The capital of France is" --ids --log-disable --no-parse-special` prints (it loads
# the vocabulary only). Five ids and no BOS: the file sets tokenizer.ggml.add_bos_token to false.
# Changing them invalidates the whole set.
REF_TOKENS=760,6511,314,9338,369
REF_DUMP_LEASE=1
: "${LCPP:=/home/user/llama.cpp-mainline}"
: "${LCPPBIN:=$LCPP/build/bin/llama-bench}"
: "${LCPP_GPU_FLAGS:=-ngl 99 -fa on}"
: "${MRS:=/home/user/mistral.rs}"
: "${MRSBIN:=$MRS/target/release/mistralrs}"
: "${MRS_FLAGS:=--format gguf}"

# Decode-step variants, `dump.sh <variant>` (`just dump-ref-qwen35moe <variant>`): one decode step
# dumped after a quiet prefill (dump_ref.cpp, --decode-step), each into a set of its own — dump.sh
# refuses REF_SET_CPU and REF_SET_CUDA for them. `ref_step_variant <name>` sets STEP_SET, STEP_CTX,
# STEP_PREFILL, STEP_TOKENS (or STEP_TOKENS_FILE and STEP_TOKENS_SHA256) and STEP_ARGS as
# models/deepseek41.sh describes, and returns 1 for a name it does not know. The suffix
# `-every-node` runs the prefill under the dumped schedule instead of the fused one
# (--prefill-every-node), so the caches and the recurrent state the step reads carry a dumped
# prefill's arithmetic.
#   step4  the oracle's five ids: a quiet prefill of 4, the step at position 4, -c 512
#   d1k    a quiet prefill of 1,024 ids of prose, the step at position 1,024, -c 2048
#   u1s4   the first 5 ids of the prose, a prefill of 4 in ubatches of one token (-ub 1), the step at
#          position 4, -c 512
#   u1s5   the same with the first 6 ids: a prefill of 5, the step at position 5
# u1s4 and u1s5 exist for one check, run with -every-node: the prefill's tokens each run as a graph
# of one token under the dumped schedule, the graph the dumped step runs, so u1s5's recurrent state
# as it enters the step is u1s4's step output bit for bit if that output is the state ik carries.
# The prose is $BLOOMERY_DATA/qwen35moe/corpus-prose.ids: qwen3moe's corpus-prose.ids recipe under
# this vocabulary — the tokenizer oracle's prose text ($BLOOMERY_DATA/tokenizer-qwen3moe/prose.txt,
# ik's docs/**/*.md and README.md, md5 c6bb074439479420faffa33257424c00, the text whose ids under the
# qwen3moe vocabulary are that profile's pinned file) through llama-tokenize --no-parse-special, one
# id per line, 76,180 ids; the sha256 below is the check.
ref_step_variant() {
  local name=$1 every_node=0
  case $name in
    *-every-node) every_node=1; name=${name%-every-node} ;;
  esac
  STEP_TOKENS='' STEP_TOKENS_FILE='' STEP_TOKENS_SHA256=''
  STEP_ARGS=()
  case $name in
    step4) STEP_SET=ref_qwen35moe_step4; STEP_CTX=512;  STEP_PREFILL=4; STEP_TOKENS=$REF_TOKENS ;;
    d1k)   STEP_SET=ref_qwen35moe_d1k;   STEP_CTX=2048; STEP_PREFILL=1024 ;;
    u1s4)  STEP_SET=ref_qwen35moe_u1s4;  STEP_CTX=512;  STEP_PREFILL=4; STEP_ARGS+=(-ub 1) ;;
    u1s5)  STEP_SET=ref_qwen35moe_u1s5;  STEP_CTX=512;  STEP_PREFILL=5; STEP_ARGS+=(-ub 1) ;;
    *)     return 1 ;;
  esac
  case $name in
    d1k|u1s4|u1s5) STEP_TOKENS_FILE=$BLOOMERY_DATA/qwen35moe/corpus-prose.ids
                   STEP_TOKENS_SHA256=dca5f89b2903f9ffd2f4e20eec18a9fdf3561bcb531a721a114e4fab68e925be ;;
  esac
  if [ "$every_node" = 1 ]; then
    STEP_SET=${STEP_SET}_every_node
    STEP_ARGS+=(--prefill-every-node)
  fi
}
