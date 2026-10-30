#!/usr/bin/env bash
# Dump the Mistral Large 4 oracle sets: mainline llama.cpp's (PR 31376's merge, 78feec062, CPU build, no
# patch) forward pass over the two prompts of tools/ref/data/mistrallarge4 on the profile's UD-IQ1_S file, from
# ONE model load, as dump_ref-format sets under $BLOOMERY_DATA:
#   ref_mistrallarge4_a  an English sentence, 18 ids: every tap below, 32 greedy tokens
#   ref_mistrallarge4_b  a Rust function with a number, 42 ids: the same taps, 32 greedy tokens
# Each set also holds topk.tsv (the last position's 40 largest logits with their f64 log-softmax) and
# greedy.tsv (the argmax tokens with their margins). Set names are the family's
# (crates/refset/src/arch/mistrallarge4).
#
# The harness is the shared one: tools/ref/cb_dump.cpp is the dumper, tools/ref/cb-dump.sh the build and the
# run (staging, lease, install); this file is Mistral Large 4's part of them, the tree, the taps and the prompts.
# `bash tools/ref/mistrallarge4-dump.sh --build` links the dumper against the ml4 tree
# (tools/ref/mistrallarge4-dump-build.sh is that call).
#
# The ids are tokens of the ml4 tree's llama-tokenize on the file, taken once and kept in
# tools/ref/data/mistrallarge4 beside their text (a.txt and b.txt hold no final newline: one would be an id);
# the dumper never tokenizes (a tokenizer difference would read as a kernel bug). The file sets add_bos_token,
# so the ids are the text's own (--no-bos), the same mode the tokenizer oracle's set takes. To take them: one
# llama-tokenize a prompt, reshaped one id a line —
#   CUDA_VISIBLE_DEVICES= $LCPP/build/bin/llama-tokenize -m "$MODEL" -f TEXT --ids --log-disable \
#     --no-bos --no-escape | tr -d '[] ' | tr ',' '\n' | grep -v '^$' > tools/ref/data/mistrallarge4/NAME.ids
#
# The taps are the graph's cb() names (src/models/mistral-large4.cpp, llama-graph.cpp build_moe_ffn), a layer's
# name `<node>-<il>`; a name the graph gives twice or more in a layer (`kv_cmpr`: the cache view, then the
# normed latent; `indexer_k`: the projection, its norm, the rope, the Hadamard rotation) is one file an
# occurrence:
#   attn_norm                                   the layer input, normed
#   indexer_q_ssmax, indexer_k, indexer_weights the lightning indexer's SSMax-scaled queries, its key at each
#                                              stage and its head weights, on the 44 layers with a full indexer
#   top_k                                       the selected positions (an i32 node: the f32 file, its i32 twin
#                                              and its `int` row), one tensor a full indexer layer. A layer that
#                                              reuses the previous full layer's selection names the same tensor
#                                              again, so it is dumped once, under the last name it was given:
#                                              layer il reads the first `top_k-<j>` with j >= il (16, 17 and 18
#                                              read `top_k-18`)
#   Qcur_ssmax                                  the absorbed query, rope part in place, times the SSMax scale
#   kv_cmpr, Kcur                               the latent and the cache key (the NoPE layers' rope part is
#                                              zero-filled), on the layers that own a cache: 39..71 read 35's
#   ffn_inp, ffn_shexp, ffn_out                 the attention residual, the shared expert and the FFN output
#                                              (layer 0 is dense: `ffn_out` is its FFN)
#   ffn_moe_logits, ffn_moe_probs               the router's logits and their softmax
#   ffn_moe_topk                                the chosen expert ids (an i32 node)
#   ffn_moe_weights, ffn_moe_weights_norm       the chosen probabilities and the weights the experts are summed
#                                              with (renormalised by their sum; the scale is 1, so no
#                                              `ffn_moe_weights_scaled` exists)
#   ffn_moe_out                                 the experts' weighted sum
#   l_out                                       the layer output
#   result_norm                                 the head's norm
# `result_output` (the softcapped logits: /30, tanh, ×30) is always the last logits row.
#
# CPU only: CUDA_VISIBLE_DEVICES is emptied and the dumper asks for no GPU layers, so the reference is a
# CPU reduction order, as dump.sh's. The run takes the CPU lease (REF_DUMP_LEASE=0 only for a dry reading, or a
# round the lead has told not to take it). DUMP_DRY=1 prints what would run and stops with 0, before the lease and
# anything on disk.
#
# Wall [derived, docs/cards/mistrallarge4-dump.card]: the 223 GB file is mapped, and this tree's CPU backend
# repacks none of its types on this AVX2 CPU (IQ1_S, IQ2_XXS, Q6_K and Q8_0 stay page-cache reads; they fit the
# 251 GB box); the first run pages in what 60 ids and 64 greedy tokens route to.
set -euo pipefail
export BLOOMERY_MODEL=mistral-large4
# shellcheck source=tools/ref/ref-paths.sh
source "${BASH_SOURCE[0]%/*}/ref-paths.sh"
# shellcheck source=tools/ref/cb-dump.sh
source "${BASH_SOURCE[0]%/*}/cb-dump.sh"

CB_FAMILY=mistrallarge4
CB_ENV=MISTRALLARGE4
CB_SCRIPT=mistrallarge4-dump.sh
CB_TREE=$LCPP
CB_PATCH=
CB_ARCH=mistral-large4
CB_TITLE="mistrallarge4-dump — mainline llama.cpp's intermediate tensors, raw f32, little-endian"
CB_IDS=mistrallarge4
TAPS=attn_norm+indexer_q_ssmax+indexer_k+indexer_weights+top_k+Qcur_ssmax+kv_cmpr+Kcur+ffn_inp+ffn_moe_logits+ffn_moe_probs+ffn_moe_topk+ffn_moe_weights+ffn_moe_weights_norm+ffn_moe_out+ffn_shexp+ffn_out+l_out+result_norm
# name:ids:nodes:greedy — nodes joined by `+`.
CB_PROMPTS=("a:a:$TAPS:32" "b:b:$TAPS:32")
CB_FLAGS=()

case ${1:-} in
  --build) cb_dump_build ;;
  '')
    # shellcheck source=tools/ref/lease.sh
    source "${BASH_SOURCE[0]%/*}/lease.sh"
    cb_dump_run
    ;;
  *) echo "mistrallarge4-dump.sh: no argument (the dump) or --build, got $1" >&2; exit 64 ;;
esac
