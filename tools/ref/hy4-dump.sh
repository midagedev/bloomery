#!/usr/bin/env bash
# Dump the Hy4-flash oracle sets: llama.cpp's PR #32517 tree (0a054d0b0, CPU build, no patch, models/hy_v4_flash.sh) forward pass over the
# five prompts of tools/ref/data/hy4 on the profile's Q4_K_M file (168 GB in two shards), from ONE model load, as dump_ref-format sets
# under $BLOOMERY_DATA:
#   ref_hy4_a  English paragraph, 32 ids           every tap below, 32 greedy tokens
#   ref_hy4_b  Korean paragraph, 31 ids            the layer taps (TAPS), 32 greedy tokens
#   ref_hy4_c  Rust function head, 30 ids          the same as b
#   ref_hy4_d  "The capital of France is", 5 ids   the same as b
#   ref_hy4_e  the ids of a, b and c in order, 93   every tap below, 32 greedy tokens. The one prompt past the fixture's top_k and window (64
#                                                   both) that ends on a partial pooling group (93 = 23 x 4 + 1): it alone reaches the indexer's
#                                                   selection (23 pools against the 15 plus the open group it picks), the window layers'
#                                                   truncation and the pooled tail (the one-token open group)
# Each set also holds result_output (the last position's whole logits row), topk.tsv (its 40 largest logits with their f64 log-softmax)
# and greedy.tsv (the argmax tokens with their margins). Set names are the family's (crates/refset/src/arch/hy4). Under
# BLOOMERY_TIER=fixture the same prompts run on the fixture file (/models/fixtures/hy4: five real layers, the dense lead, a sparse layer
# with an engram site, a window layer, a window layer with a site and the last sparse layer, on the real hidden of 3,072 with the real 512
# routed experts and vocabulary) and the sets are fx_ref_hy4_{a,b,c,d,e}.
#
# The prompts are raw text, no chat template and no BOS (the file's tokenizer adds none: llama-tokenize gives the same ids with and
# without --no-bos), so the model continues a sentence, a paragraph and a function head. The ids come from the tree's own llama-tokenize,
# taken once and kept in tools/ref/data/hy4 beside their text (crates/tokenizer/tools/hy4-oracle.sh --prompts); the dumper never tokenizes
# (a tokenizer difference would read as a kernel bug).
#
# The harness is the shared one: tools/ref/cb_dump.cpp is the dumper, tools/ref/cb-dump.sh the build and the run (staging, lease,
# install); this file is Hy4-flash's part of them, the tree, the taps and the prompts. `bash tools/ref/hy4-dump.sh --build` links the
# dumper against the profile's tree (tools/ref/hy4-dump-build.sh is that call; the tree is built once, by the tokenizer's oracle round).
#
# The taps are the graph's cb() names (src/models/hy-v4-flash.cpp, llama-graph.cpp build_moe_ffn), a layer's name `<node>-<il>`; a name a
# layer's kind does not have is absent from that layer (a tap the graph never computes is no file, never an error). TAPS, every prompt:
#   hc_init                       the embedding repeated into the four streams (unnumbered)
#   engram_out                    the streams after a site's engram write (the layers of `ple.layers`)
#   attn_norm, ffn_norm           the attention site's and the feed-forward site's input after the hyper-connection's pre-mix and the norm
#   attn_out                      the attention output after the output projection and its sandwich norm
#   hc_attn_out, l_out            the streams after the attention site and after the layer
#   ffn_out                       the feed-forward block's output after its sandwich norm (the dense layer's, or the routed stack plus the shared one)
#   ffn_moe_topk                  the router's chosen expert ids (an i32 node: the f32 file, its i32 twin and its `int` row)
#   ffn_moe_weights_scaled        their normalized weights times expert_weights_scale
#   hc_head, result_norm          the head's fold of the four streams, and its norm
# TAPS_A, prompt a, beside TAPS:
#   engram_rows, engram_delta     the gathered rows of a site and the delta it writes into the streams
#   hc_mixes, hc_pre, hc_post_gate, hc_head_mixes, hc_head_pre   the hyper-connection's mixes and the pre-mix's output
#   q_resid, kv_cmpr, attn_kqv, attn_gated   the latent attention's query residual, the cache latent, the attention output before and
#                                 after the output gate
#   indexer_q, indexer_k, indexer_gate, indexer_pool_k, indexer_weights, indexer_score, indexer_sel   the sparse layers' selection
#                                 (`indexer_sel`: the cells the picked pools leave visible, the fused top-k mask's output; the graph's
#                                 `indexer_top_k` exists only on the unfused path, which this CPU build does not take)
#   ffn_moe_logits, ffn_moe_probs, ffn_moe_probs_biased, ffn_moe_weights_norm   the router
#   ffn_moe_out, ffn_shexp        the routed stack's output (in the experts' latent) and the shared stack
# The greedy continuation runs with no node asked for, so a set's `ffn_moe_topk` covers the prompt's own positions only.
#
# CPU only: CUDA_VISIBLE_DEVICES is emptied and the dumper asks for no GPU layers, so the reference is a CPU reduction order, as dump.sh's.
# Flash attention is the library's own choice (no -fa). The run takes the CPU lease (REF_DUMP_LEASE=0 only for a dry reading).
# DUMP_DRY=1 prints what would run and stops with 0, before the lease and anything on disk.
#
# Wall [derived, docs/cards/hy4-fx-dump.card]: the fixture's dump is seconds. The real file's dump has no card yet (a card is written
# before its first run); its bound is the harness's 1,800 s unless BLOOMERY_DUMP_BOUND says more.
set -euo pipefail
export BLOOMERY_MODEL=hy_v4_flash
# shellcheck source=tools/ref/ref-paths.sh
source "${BASH_SOURCE[0]%/*}/ref-paths.sh"
# shellcheck source=tools/ref/cb-dump.sh
source "${BASH_SOURCE[0]%/*}/cb-dump.sh"

CB_FAMILY=hy4
CB_ENV=HY4
CB_SCRIPT=hy4-dump.sh
CB_TREE=$LCPP
CB_PATCH=
CB_ARCH=hy_v4_flash
CB_TITLE="hy4-dump — llama.cpp PR #32517's intermediate tensors, raw f32, little-endian"
CB_IDS=hy4
TAPS=hc_init+engram_out+attn_norm+attn_out+hc_attn_out+ffn_norm+ffn_out+l_out+ffn_moe_topk+ffn_moe_weights_scaled+hc_head+result_norm
TAPS_A=$TAPS+engram_rows+engram_delta+hc_mixes+hc_pre+hc_post_gate+hc_head_mixes+hc_head_pre+q_resid+kv_cmpr+attn_kqv+attn_gated+indexer_q+indexer_k+indexer_gate+indexer_pool_k+indexer_weights+indexer_score+indexer_sel+ffn_moe_logits+ffn_moe_probs+ffn_moe_probs_biased+ffn_moe_weights_norm+ffn_moe_out+ffn_shexp
# name:ids:nodes:greedy — nodes joined by `+`.
CB_PROMPTS=("a:a:$TAPS_A:32" "b:b:$TAPS:32" "c:c:$TAPS:32" "d:d:$TAPS:32" "e:e:$TAPS_A:32")
CB_FLAGS=()

case ${1:-} in
  --build) cb_dump_build ;;
  '')
    # shellcheck source=tools/ref/lease.sh
    source "${BASH_SOURCE[0]%/*}/lease.sh"
    cb_dump_run
    ;;
  *) echo "hy4-dump.sh: no argument (the dump) or --build, got $1" >&2; exit 64 ;;
esac
