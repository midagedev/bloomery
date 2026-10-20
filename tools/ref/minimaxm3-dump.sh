#!/usr/bin/env bash
# Dump the MiniMax-M3 oracle sets: mainline llama.cpp's (cd0a5b6a1, CPU build, no patch) forward pass over the five prompts of
# tools/ref/data/minimaxm3 on the profile's UD-IQ4_XS file, from ONE model load, as dump_ref-format sets under $BLOOMERY_DATA:
#   ref_minimaxm3_a  41 ids   every per-layer tap below, 64 greedy tokens
#   ref_minimaxm3_b  55 ids   (Korean) each layer output l_out and the router's expert ids, 64 greedy tokens
#   ref_minimaxm3_c  116 ids  (Rust) the same as b
#   ref_minimaxm3_d  1,536 ids of wikitext-2: the last position's logits only
#   ref_minimaxm3_e  2,560 ids of wikitext-2 (d's ids and 1,024 more): the sparse attention's block scores and selection
# Each set also holds topk.tsv (the last position's 40 largest logits with their f64 log-softmax) and greedy.tsv (the argmax
# tokens with their margins). Set names are the family's (crates/refset/src/arch/minimaxm3); MiniMax-M3 has no fixture file yet,
# so BLOOMERY_TIER=fixture is refused by name (exit 66, ref-paths.sh).
#
# Why a fifth prompt: sparse attention picks 16 blocks of 128 positions per indexer head, so a context of up to 2,048
# positions attends every block and nothing is selected. d (1,536 ids, 12 blocks) is below that line; e (2,560 ids, 20 blocks) has
# queries in blocks 16..19, which have 17..20 candidates and drop 1..4 of them. e is the only set that holds a selection.
#
# The harness is the shared one: tools/ref/cb_dump.cpp is the dumper, tools/ref/cb-dump.sh the build and the run (staging, lease,
# install); this file is MiniMax's part of them, the tree, the taps and the prompts. `bash tools/ref/minimaxm3-dump.sh --build`
# links the dumper against the profile's mainline tree (tools/ref/minimaxm3-dump-build.sh is that call).
#
# The taps are the graph's cb() names (src/models/minimax-m3.cpp, llama-graph.cpp build_moe_ffn), a layer's name `<node>-<il>`:
#   attn_norm, Qcur_normed, Kcur_normed   the layer input and the per-head q/k norms
#   Qcur, Kcur                            three times a layer and ubatch, in this order: the projection (MUL_MAT), its reshape,
#                                         then the rotated tensor (ROPE); a consumer proves the row by its op column
#   msa_bs                                the block-max scores of the indexer ([blocks, 4 heads, tokens]; sparse layers 3..59), -FLT_MAX
#                                         on a block the causal mask hides
#   msa_bsf                               msa_bs plus the local-block bias (+1e30 on the token's own block)
#   msa_block_mask                        [blocks, tokens, 4 heads] f16 widened: 0 on the blocks the top-16 selected, -inf on the rest.
#                                         `blocks` is the cache window (padded to 256 positions), not the positions so far, and a
#                                         token with fewer than 16 visible blocks has the hidden ones among its 16, in the order the
#                                         argsort leaves equal scores: compare the selection on the blocks the token sees
#   msa_fattn, kqv_out                    the flash attention output and the attention output before wo
#   ffn_inp, ffn_norm                     the residual and the ffn norm
#   ffn_moe_probs, ffn_moe_topk, ffn_moe_weights_scaled   the router: sigmoid probabilities, the chosen expert ids (an i32 node:
#                                         the f32 file, its i32 twin and its `int` row), their normalized weights times 2
#   ffn_moe_out, ffn_shexp, ffn_out, l_out, result_norm
# msa_* exist in the batch graph only (the greedy continuation runs the gather graph, whose block scores are unnamed), and the
# top-16 index itself is unnamed in both: the selection is the zeros of msa_block_mask.
#
# CPU only: CUDA_VISIBLE_DEVICES is emptied and the dumper asks for no GPU layers, so the reference is a CPU reduction order, as
# dump.sh's. -fa on: the sparse attention runs only with flash attention (the graph falls back to dense attention without it), and
# -b 4096 so e is one llama_decode (n_batch bounds it). The run takes the CPU lease (REF_DUMP_LEASE=0 only for a dry reading).
# DUMP_DRY=1 prints what would run and stops with 0, before the lease and anything on disk.
#
# Wall [derived, docs/cards/minimaxm3-dump.card]: the 208.5 GB file is mapped, and this tree's CPU backend repacks only the types it
# has an interleave for on this CPU (AVX2, no AVX-512: Q4_0, Q4_K, IQ4_NL, MXFP4; Q6_K and Q8_0 only on ARM and RISC-V, Q2_K on AVX-512),
# none of which this file holds, so nothing becomes anonymous memory: all 208.5 GB are read through the page cache (they fit the
# 251 GB box with ~28 GB to spare). Prefill 4,308 ids x 2 x 23.5 G MACs = 202 TFLOP, of which the routed 111 TFLOP run unrepacked
# IQ3_S/IQ4_XS dot products (the one term derivation does not close), greedy 192 tokens at ~19 GB a token over ~120 GB/s, ~1.4 GB of
# node files.
set -euo pipefail
export BLOOMERY_MODEL=minimax-m3
# shellcheck source=tools/ref/ref-paths.sh
source "${BASH_SOURCE[0]%/*}/ref-paths.sh"
# shellcheck source=tools/ref/cb-dump.sh
source "${BASH_SOURCE[0]%/*}/cb-dump.sh"

CB_FAMILY=minimaxm3
CB_ENV=MINIMAXM3
CB_SCRIPT=minimaxm3-dump.sh
CB_TREE=$LCPP
CB_PATCH=
CB_ARCH=minimax-m3
CB_TITLE="minimaxm3-dump — mainline llama.cpp's intermediate tensors, raw f32, little-endian"
CB_IDS=minimaxm3
ROUTER=ffn_moe_topk
TAPS=attn_norm+Qcur_normed+Kcur_normed+Qcur+Kcur+msa_bs+msa_bsf+msa_block_mask+msa_fattn+kqv_out+ffn_inp+ffn_norm+ffn_moe_probs+$ROUTER+ffn_moe_weights_scaled+ffn_moe_out+ffn_shexp+ffn_out+l_out+result_norm
# name:ids:nodes:greedy — nodes joined by `+`.
CB_PROMPTS=("a:a:$TAPS:64" "b:b:l_out+$ROUTER:64" "c:c:l_out+$ROUTER:64" "d:d::0" "e:e:msa_bs+msa_bsf+msa_block_mask:0")
CB_FLAGS=(-b 4096 -fa on)

case ${1:-} in
  --build) cb_dump_build ;;
  '')
    # shellcheck source=tools/ref/lease.sh
    source "${BASH_SOURCE[0]%/*}/lease.sh"
    cb_dump_run
    ;;
  *) echo "minimaxm3-dump.sh: no argument (the dump) or --build, got $1" >&2; exit 64 ;;
esac
