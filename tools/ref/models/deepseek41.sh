#!/usr/bin/env bash
# shellcheck shell=bash
# The deepseek41 (DeepSeek-V4.1-Flash) profile: everything the reference engine and the harnesses
# need that is a property of the *model*, not of the machine. Sourced by ref-paths.sh when
# BLOOMERY_MODEL=deepseek41; never executed, and it exports nothing. Same shape as deepseek2.sh.
#
# The name is the GGUF general.architecture value (shard 1's header says deepseek41), so
# `grep deepseek41` finds the port's arch, its metadata keys (deepseek41.engram.*), this profile
# and the oracle set at once.
#
#   MODEL           shard 1 of the split set; the loader follows split.count from there.
#                   BLOOMERY_REF_MODEL moves it; a caller's own MODEL= is ignored, as in deepseek2.sh
#   IK              moves the whole ik tree. The default is our V4.1 port (PR #2455) with the index-key
#                   fix (PR #2507) on the local branch v41/idxkey-fix, a worktree of the V2-Lite
#                   profile's tree; its V4.1 path has only run on the CPU (-ngl 0). The two profiles'
#                   trees differ, so dump.sh refuses a dump_ref built against the other one ([foreign-lib])
#   REF_CTX         the one context the reference files are produced at
#   REF_SET_CPU     ref_deepseek41 under $BLOOMERY_DATA — flat, a sibling of ref, ref_cuda and
#                   ref_cuda_v2, because every reader resolves a set as $BLOOMERY_DATA/<one name>
#                   (dump.sh's staging and .old siblings, the gates' BLOOMERY_REF_SET)
#   REF_SET_CUDA    ref_cuda_deepseek41: a name only — no CUDA set is made for this model
#   REF_TOKENS      the oracle's token ids; dump.sh's BLOOMERY_REF_TOKENS overrides them
#   REF_DUMP_LEASE  1: the dump pages in hundreds of GiB of weights, so dump.sh runs it under the
#                   machine-wide CPU lease and witnesses it. Not overridable from here.
#   REF_DUMP_ARGS   dumper arguments every dump of this model adds: --defer-experts, which skips the
#                   loader's MAP_POPULATE of the whole file set (more than the page cache holds) and
#                   changes nothing the graph computes; the dump faults in what it touches
#   ref_step_variant  the decode-step variants, below
#
#   IK_GPU_FLAGS    ik's decode on the timing card, placed to mirror design §5 (a) (depth-ds41.sh reads
#                   it): every layer offloaded, and the routed experts of the first 34 layers on the
#                   CPU, the last 6 layers' whole on the card. Plan (a) keeps a prefix of every
#                   layer's experts on the A6000 — 2,414 experts, 40,490,311,680 B
#                   (crates/model/tests/placement.rs) — and ik moves a layer's 384 experts as one
#                   tensor, so the closest it has is whole layers: 6 × 384 = 2,304 experts [derived, at
#                   plan (a)'s bytes per expert]; a 7th layer does not fit beside the dense weights [derived:
#                   7 × 6.44 GB of experts + 8.15 GB dense > the card's 48,592 MiB].
#                   The expected host work per token is the same (about 204 routed expert products,
#                   ours about 202 [derived]). -t 32 is llama-bench's own default, spelled out;
#                   --defer-experts skips the loader's MAP_POPULATE of a file set larger than the page
#                   cache. No flag sweep has been run: these are "at these flags".
#                   Known not to load today: with -ngl 999 the loader puts the whole host
#                   context (the 34 layers' experts and the engram tables, about 400 GiB) in one
#                   CUDA_Host buffer instead of the mmap branch, and the pinned allocation fails
#                   (`ggml_cuda_host_malloc: mmap of 410412.18 MiB failed`); GGML_CUDA_NO_PINNED=1
#                   would malloc the same bytes on a 256 GB box. `llama-perplexity -ngl 0` opens the
#                   same file by mmap, so the split is between -ngl 0 and -ngl > 0 or in llama-bench's
#                   own override path; until that is found depth-ds41.sh's ik arms fail at load
#
# Deliberately unset: IK_BEST_FLAGS and REF_PROMPTS. No CPU flag sweep and no prompt set exist for
# this model, and every script that reads them runs under `set -u`, so such a script stops at the
# unset name instead of running ik at flags nobody measured.
#
# SC2034: every name here is read by the file that sources this one, which shellcheck does not
# see from this file alone.
# shellcheck disable=SC2034
MODEL_NAME=deepseek41
: "${IK:=/home/user/ik-idxkey}"
MODEL=${BLOOMERY_REF_MODEL:-/models/DeepSeek-V4.1-Flash-Q3_K_M-engramQ8-tokembdBF16-attnQ8/DeepSeek-V4.1-Flash-Q3_K_M-00001-of-00009.gguf}
: "${IK_GPU_FLAGS:=-ngl 999 --n-cpu-moe 34 -t 32 --defer-experts}"
: "${REF_CTX:=512}"
: "${REF_SET_CPU:=ref_deepseek41}"
: "${REF_SET_CUDA:=ref_cuda_deepseek41}"
# "The capital of France is" under this model's tokenizer: what `$IK/build/bin/llama-tokenize
# -m $MODEL -p "The capital of France is" --ids --log-disable --no-parse-special` prints for
# shard 1 (it loads the vocabulary only). Five ids and no BOS: the file sets
# tokenizer.ggml.add_bos_token to false, so nothing is prepended (its BOS would be id 0).
# Changing them invalidates the whole set.
REF_TOKENS=671,6102,294,8760,344
REF_DUMP_LEASE=1
REF_DUMP_ARGS=(--defer-experts)

# Decode-step variants, `dump.sh <variant>` (`just dump-ref-v41 <variant>`): one decode step dumped
# after a quiet prefill (dump_ref.cpp, --decode-step), each into a set of its own — dump.sh refuses
# REF_SET_CPU and REF_SET_CUDA for them. `ref_step_variant <name>` sets
#   STEP_SET           the set's name under $BLOOMERY_DATA
#   STEP_CTX           -c
#   STEP_PREFILL       tokens evaluated quietly first; the step is the next one, at position STEP_PREFILL
#   STEP_TOKENS        the ids as a comma list, or
#   STEP_TOKENS_FILE   a file of ids, one per line, whose first STEP_PREFILL + 1 are the sequence, and
#   STEP_TOKENS_SHA256 that file's sha256: dump.sh refuses a file that no longer has it
#   STEP_ARGS          further dumper arguments
# and returns 1 for a name it does not know. Two suffixes, in either order, change one thing each
# and add it to the set name: `-unfused` turns ik's fused indexer top-k op off (--no-fused-idx-topk),
# so the indexer's scores and its TOP_K are nodes; `-every-node` runs the prefill under the dumped
# schedule instead of the fused one (--prefill-every-node), so the caches the step reads carry a
# dumped prefill's arithmetic — step4-every-node is the variant a batch set's last token can check.
#   step4  the oracle's five ids: a quiet prefill of 4, the step at position 4, -c 512
#   d1     the indexer top-k overridden to 64, so the indexer, the row gather and mask_to_idx run at a
#          short prefix: prefill 301, the step at 301 (a csa group completes there), -c 512
#   d1n    d1 without the override: the file's top-k (512) at prefill 301, so neither kind builds an indexer
#          (top_k >= pad256(n_vis) for csa and hca) while the window mask already hides keys older than 128 —
#          the step set of a chain that has no indexer yet
#   d2     the model's own top-k: prefill 1,025, the step at 1,025, -c 2048
# d1, d1n and d2 read the prose stream the router trace ran over (router/prose/MANIFEST.tsv names it):
# /root/bloomery-data/engram/corpus-prose-all.ids, 4,670,384 lines, sha256
# f7785d0fc84a4a7e3673a220120be8c6735ab226084b2a506d16412ec64f71a0 — d1 and d1n its first 302 ids, d2 its
# first 1,026.
ref_step_variant() {
  local name=$1 unfused=0 every_node=0
  while :; do
    case $name in
      *-unfused)    unfused=1;    name=${name%-unfused} ;;
      *-every-node) every_node=1; name=${name%-every-node} ;;
      *) break ;;
    esac
  done
  STEP_TOKENS='' STEP_TOKENS_FILE='' STEP_TOKENS_SHA256=''
  STEP_ARGS=()
  case $name in
    step4) STEP_SET=ref_deepseek41_step4; STEP_CTX=512;  STEP_PREFILL=4; STEP_TOKENS=$REF_TOKENS ;;
    d1)    STEP_SET=ref_deepseek41_d1;    STEP_CTX=512;  STEP_PREFILL=301
           STEP_ARGS=(--override-kv deepseek41.attention.indexer.top_k=int:64) ;;
    d1n)   STEP_SET=ref_deepseek41_d1n;   STEP_CTX=512;  STEP_PREFILL=301 ;;
    d2)    STEP_SET=ref_deepseek41_d2;    STEP_CTX=2048; STEP_PREFILL=1025 ;;
    *)     return 1 ;;
  esac
  case $name in
    d1|d1n|d2) STEP_TOKENS_FILE=$BLOOMERY_DATA/engram/corpus-prose-all.ids
               STEP_TOKENS_SHA256=f7785d0fc84a4a7e3673a220120be8c6735ab226084b2a506d16412ec64f71a0 ;;
  esac
  if [ "$unfused" = 1 ]; then
    STEP_SET=${STEP_SET}_unfused
    STEP_ARGS+=(--no-fused-idx-topk)
  fi
  if [ "$every_node" = 1 ]; then
    STEP_SET=${STEP_SET}_every_node
    STEP_ARGS+=(--prefill-every-node)
  fi
}
