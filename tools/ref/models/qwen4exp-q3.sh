#!/usr/bin/env bash
# shellcheck shell=bash
# The qwen4exp profile over unsloth's UD-Q3_K_XL split set: everything the qwen4exp profile sets,
# with this file and its own set names. Sourced by ref-paths.sh when BLOOMERY_MODEL=qwen4exp-q3;
# never executed, and it exports nothing.
#
# The name keeps the qwen4exp prefix (the GGUF general.architecture value, shard 1's header says
# qwen4exp), so `grep qwen4exp` finds this profile beside the Q4 one.
#
#   MODEL           shard 1 of unsloth's UD-Q3_K_XL split set (three shards; the first holds the
#                   header and no tensor): the file the ik-qwen4exp-q3 refset family pins and
#                   gate_qwen4exp_e2e loads under this profile, as BLOOMERY_REF_MODEL moves it in
#                   qwen4exp.sh. Not the file the MTP draft set or the qwen38 residency and
#                   two-card gates read: those run under the qwen4exp profile
#   IK              the Q4 profile's tree, whose build carries this architecture: one dumper serves
#                   both files
#   REF_SET_CPU     ref_qwen4exp_q3 under $BLOOMERY_DATA: ik's CPU dump of this file (refset family
#                   ik-qwen4exp-q3; docs/cards/q38q3ref-dump.card is the dump's lease card)
#   REF_SET_CUDA    ref_cuda_qwen4exp_q3: a name only — no CUDA set is made for this model yet
#   REF_TOKENS      the Q4 profile's: the same model's tokenizer, so the same five ids
#   REF_DUMP_LEASE  the Q4 profile's 1: the dump pages in the whole split set under the CPU lease
#   ref_step_variant  the Q4 profile's decode-step variants (step4, d1k, d3k, the -every-node
#                   suffix) with this file's set names, over the same corpus file
#                   ($BLOOMERY_DATA/qwen4exp/corpus-prose.ids and its sha256): the ids are a
#                   property of the tokenizer, which this file shares with the Q4 one, so the
#                   shared file with the pin is the check
#
# Deliberately unset: LCPP_GPU_FLAGS (and with it the two-card placement, already empty here): the
# depth runner's hand-set reference arm is sized for the Q4 file's expert stacks, and no sizing has
# been derived for this one. Every script that reads it runs under `set -u`, so such a script stops
# at the unset name instead of running an engine at flags nobody chose, as qwen4exp.sh does for its
# deliberately unset arms.
#
# SC2034: every name here is read by the file that sources this one, which shellcheck does not
# see from this file alone.
# shellcheck disable=SC2034
__q38q3_dir=${BASH_SOURCE[0]%/*}
[ "$__q38q3_dir" != "${BASH_SOURCE[0]}" ] || __q38q3_dir=.
# shellcheck source=tools/ref/models/qwen4exp.sh
source "$__q38q3_dir/qwen4exp.sh"
unset __q38q3_dir
QWEN38_Q3_FILE=/models/Qwen3.8-Flash-Next-UD-Q3_K_XL/Qwen3.8-Flash-Next-UD-Q3_K_XL-00001-of-00003.gguf
MODEL=${BLOOMERY_REF_MODEL:-$QWEN38_Q3_FILE}
REF_SET_CPU=ref_qwen4exp_q3
REF_SET_CUDA=ref_cuda_qwen4exp_q3
# QWEN38_FILE (the Q4 file) has no reader past the sourced profile's own two-card line, which this
# profile's MODEL already leaves false; unset with the Q4 reference arms, so the only /models
# directory the sourced state defines is this profile's own.
unset LCPP_GPU_FLAGS QWEN38_NCMOE QWEN38_TS QWEN38_FILE

# Decode-step variants, `dump.sh <variant>` (`just dump-ref-qwen4exp-q3 <variant>`): the qwen4exp
# profile's cases with this file's set names (the refset family's, one for one), everything else —
# the contexts, the prefill lengths, the corpus file and its sha256, the -every-node suffix — the
# qwen4exp profile's own function's.
ref_step_variant() {
  local name=$1 every_node=0
  case $name in
    *-every-node) every_node=1; name=${name%-every-node} ;;
  esac
  STEP_TOKENS='' STEP_TOKENS_FILE='' STEP_TOKENS_SHA256=''
  STEP_ARGS=()
  case $name in
    step4) STEP_SET=ref_qwen4exp_q3_step4; STEP_CTX=512;  STEP_PREFILL=4;    STEP_TOKENS=$REF_TOKENS ;;
    d1k)   STEP_SET=ref_qwen4exp_q3_d1k;   STEP_CTX=2048; STEP_PREFILL=1024 ;;
    d3k)   STEP_SET=ref_qwen4exp_q3_d3k;   STEP_CTX=4096; STEP_PREFILL=3000 ;;
    *)     return 1 ;;
  esac
  case $name in
    d1k|d3k) STEP_TOKENS_FILE=$BLOOMERY_DATA/qwen4exp/corpus-prose.ids
             STEP_TOKENS_SHA256=dca5f89b2903f9ffd2f4e20eec18a9fdf3561bcb531a721a114e4fab68e925be ;;
  esac
  if [ "$every_node" = 1 ]; then
    STEP_SET=${STEP_SET}_every_node
    STEP_ARGS+=(--prefill-every-node)
  fi
}
