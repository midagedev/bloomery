#!/usr/bin/env bash
# Dump the Kimi K3 oracle sets: mainline llama.cpp's (78feec062, CPU build, no patch, models/kimi-k3.sh) forward pass over the four prompts
# of tools/ref/data/kimi-k3 on the profile's UD-IQ1_S file (594 GB in fourteen shards), from ONE model load, as dump_ref-format sets
# under $BLOOMERY_DATA:
#   ref_kimik3_a  English paragraph, 35 ids           every tap below, 32 greedy tokens
#   ref_kimik3_b  Korean paragraph, 46 ids            the layer taps (TAPS), 32 greedy tokens
#   ref_kimik3_c  Rust function head, 48 ids          the same as b
#   ref_kimik3_d  "The capital of France is", 5 ids   the same as b
# Each set also holds result_output (the last position's whole logits row), topk.tsv (its 40 largest logits with their f64 log-softmax)
# and greedy.tsv (the argmax tokens with their margins). Set names are the family's (crates/refset/src/arch/kimi_k3). Under
# BLOOMERY_TIER=fixture the same prompts run on the fixture file (/models/fixtures/kimik3: four layers of the real file's two mixers on
# a hidden of 512) and the sets are fx_ref_kimik3_{a,b,c,d}.
#
# The prompts are raw text, no chat template and no BOS (the file's tokenizer adds none: llama-tokenize gives the same ids with and
# without --no-bos), so the model continues a sentence, a paragraph and a function head. The ids come from the tree's own llama-tokenize,
# taken once and kept in tools/ref/data/kimi-k3 beside their text (the texts hold no final newline: one would be an id); the dumper never
# tokenizes (a tokenizer difference would read as a kernel bug).
#
# The harness is the shared one: tools/ref/cb_dump.cpp is the dumper, tools/ref/cb-dump.sh the build and the run (staging, lease,
# install); this file is Kimi K3's part of them, the tree, the taps and the prompts. `bash tools/ref/kimi-k3-dump.sh --build` links the
# dumper against the profile's tree (tools/ref/kimi-k3-dump-build.sh is that call; `just build-k3-lcpp` builds the tree).
#
# The taps are the graph's cb() names (src/models/kimi-k3.cpp, llama-graph.cpp build_moe_ffn), a layer's name `<node>-<il>`; a name a
# layer's kind does not have is absent from that layer. TAPS, every prompt:
#   attn_norm, ffn_norm       the attention site's and the ffn site's input, after the cross-layer residual's mix and the norm
#   res_probs                 that mix's softmax over the banked checkpoints and the current stream, occurrence 0 the attention site's
#                             and occurrence 1 the ffn site's (layer 0's attention site has an empty bank and no node: its one
#                             `res_probs-0` is the ffn site's), and the head's final mix once more, unnumbered
#   kda_out, mla_out          the attention output after its output projection: the delta layers' (69) and the latent layers' (24)
#   prefix_sum_attn           the stream after the attention site; on a bank layer (every 12th, layer 0 first) the stream restarts from
#                             the attention output, so that layer's `kda_out` or `mla_out` is this node and has no node of its own
#   ffn_moe_topk              the router's chosen expert ids (an i32 node: the f32 file, its i32 twin and its `int` row)
#   ffn_moe_weights_norm      their normalized weights (the scale is 1: no *_scaled node)
#   ffn_out, l_out            the ffn site's output (the dense layer's, or the routed stack plus the shared one) and the stream after it
#   result_norm               the head's norm of the last row
# TAPS_A, prompt a, beside TAPS:
#   kda_q_conv, kda_k_conv, kda_v_conv   the short convolutions' outputs after SiLU
#   kda_g1, kda_beta, kda_scan_out       the decay gate, the erase gate and the scan's output
#   kda_g2, kda_normed                   the output gate and the RMS-normed scan output
#   mla_gated                            the gated attention output before the output projection
#   ffn_moe_logits, ffn_moe_probs, ffn_moe_probs_biased, ffn_moe_weights   the router: logits, sigmoid probabilities, the selection
#                                        scores with the bias, the chosen probabilities
#   ffn_moe_situ, ffn_moe_out, ffn_shexp the routed stack's activation, its summed output (in the experts' latent) and the shared stack
# The greedy continuation runs with no node asked for, so a set's `ffn_moe_topk` covers the prompt's own positions only.
#
# CPU only: CUDA_VISIBLE_DEVICES is emptied and the dumper asks for no GPU layers, so the reference is a CPU reduction order, as dump.sh's.
# -fa on: the engine's latent attention is the flash one. The run takes the CPU lease (REF_DUMP_LEASE=0 only for a dry reading).
# DUMP_DRY=1 prints what would run and stops with 0, before the lease and anything on disk.
#
# Wall [derived, docs/cards/kimi-k3-dump.card]: the tree maps each of the fourteen shards with MAP_POPULATE (no tensor of this arch is
# lazy), so the load reads the whole 594 GB once through the page cache (251 GB: the last ~240 GB stay), and every prompt and greedy
# token then faults in the experts the cache does not hold. The real tier's bound is 90 minutes, the harness's 30 for the fixture.
set -euo pipefail
export BLOOMERY_MODEL=kimi-k3
# shellcheck source=tools/ref/ref-paths.sh
source "${BASH_SOURCE[0]%/*}/ref-paths.sh"
# shellcheck source=tools/ref/cb-dump.sh
source "${BASH_SOURCE[0]%/*}/cb-dump.sh"

CB_FAMILY=kimik3
CB_ENV=KIMIK3
CB_SCRIPT=kimi-k3-dump.sh
CB_TREE=$LCPP
CB_PATCH=
CB_ARCH=kimi-k3
CB_TITLE="kimi-k3-dump — mainline llama.cpp's intermediate tensors, raw f32, little-endian"
CB_IDS=kimi-k3
TAPS=attn_norm+ffn_norm+res_probs+kda_out+mla_out+prefix_sum_attn+ffn_moe_topk+ffn_moe_weights_norm+ffn_out+l_out+result_norm
TAPS_A=$TAPS+kda_q_conv+kda_k_conv+kda_v_conv+kda_g1+kda_beta+kda_scan_out+kda_g2+kda_normed+mla_gated+ffn_moe_logits+ffn_moe_probs+ffn_moe_probs_biased+ffn_moe_weights+ffn_moe_situ+ffn_moe_out+ffn_shexp
# name:ids:nodes:greedy — nodes joined by `+`.
CB_PROMPTS=("a:a:$TAPS_A:32" "b:b:$TAPS:32" "c:c:$TAPS:32" "d:d:$TAPS:32")
CB_FLAGS=(-fa on)
# The real file's dump reads about 1.6 TB through the page cache; the fixture's is seconds.
[ "${BLOOMERY_TIER:-real}" = fixture ] || : "${BLOOMERY_DUMP_BOUND:=5400}"

case ${1:-} in
  --build) cb_dump_build ;;
  '')
    # shellcheck source=tools/ref/lease.sh
    source "${BASH_SOURCE[0]%/*}/lease.sh"
    cb_dump_run
    ;;
  *) echo "kimi-k3-dump.sh: no argument (the dump) or --build, got $1" >&2; exit 64 ;;
esac
