#!/usr/bin/env bash
# Dump the Raon-1-nano-preview oracle sets: KRAFTON's llama.cpp branch raon-1-nano-preview (CPU build, no patch,
# models/raon1nano.sh) on the Q8_0 file over the four prompts of tools/ref/data/raon1 (the two dialects share the
# vocabulary and the prompts: the nano tree's llama-tokenize reproduces every one of them), from ONE model load, as
# dump_ref-format sets under $BLOOMERY_DATA:
#   ref_raon1nano_a  English paragraph   every per-layer tap below, 32 greedy tokens
#   ref_raon1nano_b  Korean paragraph    each layer output, 32 greedy tokens (the file routes no experts, so there
#                                        is no router tap: mini's b named the router's ids, nano's names l_out)
#   ref_raon1nano_c  Rust function       the same as b
#   ref_raon1nano_d  1,536 ids of wikitext-2: the last position's logits only (more than 16 positions: the RDA
#                    chunked scan runs many chunks, the MLA layers attend a context of 1,536)
# Each set also holds result_output (the last position's whole logits row), topk.tsv (its 40 largest logits with
# their f64 log-softmax) and greedy.tsv (the argmax tokens with their margins). Set names are the family's
# (crates/refset/src/arch/raon1nano); the Q8_0 file is the family's own fixture, so BLOOMERY_TIER=fixture is
# refused by name (exit 66, ref-paths.sh's fixture_dump_tier: the real file stands, no cut is dumped).
#
# The harness is the shared one: tools/ref/cb_dump.cpp is the dumper, tools/ref/cb-dump.sh the build and the run
# (staging, lease, install); this file is the nano dialect's part of them, the tree, the taps and the prompts.
# `bash tools/ref/raon1nano-dump.sh --build` links the dumper against the profile's tree
# (tools/ref/raon1nano-dump-build.sh is that call). The tree is the oracle round's own build of the pinned commit;
# cb_dump_build refuses a tree at another commit or with tracked changes, and the run a library that moved.
#
# The taps are the graph's cb() names (src/models/raon1nano.cpp, delta-net-base.cpp, llama-graph.cpp), a layer's
# name `<node>-<il>`; a name that a layer's kind does not have is absent from that layer, and hc_* repeat once a
# site (the tap census of the oracle round, taps.txt):
#   attn_norm                  the attention site's input after the mHC pre-mix and the norm
#   hc_mixes, hc_pre, hc_post, hc_comb   an mHC site's selector rows (the folded chart's parametrization: the α
#                              scales baked into the fn rows), the pre-mix weights, the post weights and the
#                              doubly stochastic mixing matrix (twice a layer: the attention site first, then
#                              the ffn site)
#   rda_conv                   the one fused q|k|v short convolution after SiLU (RDA layers: all but 3, 7, .., 23)
#   rda_g1, rda_v_scaled, rda_beta   the decay gate, the value scaled by the write/erase ratio, the erase gate
#   rda_scan_out, new_state    the scan's output and the recurrent state after the ubatch ([128, 128, 16])
#   rda_out                    the layer output after wo (the output gate and the scan's norm are folded into the
#                              file's tensors: no rda_g2, no rda_normed tap)
#   fattn_mla, kqv_out, mla_gated   the latent layers' attention output, the value projection's, and the gated
#                              output before wo (MLA layers: 3, 7, .., 23)
#   hc_attn_out                the residual streams after the attention site's post-mix ([1536, 4 streams, tokens])
#   ffn_norm                   the ffn site's input after the mHC pre-mix and the norm
#   ffn_up, ffn_swiglu         the dense block's fused [gate|up] rows and their SwiGLU output (no router: the
#                              file routes no experts, so every ffn_moe_* tap of mini's list is gone)
#   ple_out                    the per-layer input's gated add into the streams, once a trunk layer
#   inp_per_layer              the per-layer input embedding table's rows of the prompt (no -il suffix)
#   l_out, result_norm         the residual streams after the ffn site's post-mix, and the normed mean of the
#                              streams at the last row (the head is the tied token table)
#
# CPU only: CUDA_VISIBLE_DEVICES is emptied and the dumper asks for no GPU layers, so the reference is a CPU
# reduction order, as dump.sh's. -fa on: the engine's attention is the flash-attention one. The run takes the CPU
# lease (REF_DUMP_LEASE=0 only for a dry reading). DUMP_DRY=1 prints what would run and stops with 0, before the
# lease and anything on disk.
#
# Wall [derived, docs/cards/raon1nano-dump.card].
set -euo pipefail
export BLOOMERY_MODEL=raon1nano
# shellcheck source=tools/ref/ref-paths.sh
source "${BASH_SOURCE[0]%/*}/ref-paths.sh"
# shellcheck source=tools/ref/cb-dump.sh
source "${BASH_SOURCE[0]%/*}/cb-dump.sh"

CB_FAMILY=raon1nano
CB_ENV=RAON1NANO
CB_SCRIPT=raon1nano-dump.sh
CB_TREE=$LCPP
CB_PATCH=
CB_ARCH=raon1nano
CB_TITLE="raon1nano-dump — KRAFTON's llama.cpp branch raon-1-nano-preview intermediate tensors, raw f32, little-endian"
CB_IDS=raon1
TAPS=attn_norm+hc_mixes+hc_pre+hc_post+hc_comb+rda_conv+rda_g1+rda_v_scaled+rda_beta+rda_scan_out+new_state+rda_out+fattn_mla+kqv_out+mla_gated+hc_attn_out+ffn_norm+ffn_up+ffn_swiglu+ple_out+inp_per_layer+l_out+result_norm
# name:ids:nodes:greedy — nodes joined by `+`.
CB_PROMPTS=("a:a:$TAPS:32" "b:b:l_out:32" "c:c:l_out:32" "d:d::0")
CB_FLAGS=(-fa on)

case ${1:-} in
  --build) cb_dump_build ;;
  '')
    # shellcheck source=tools/ref/lease.sh
    source "${BASH_SOURCE[0]%/*}/lease.sh"
    cb_dump_run
    ;;
  *) echo "raon1nano-dump.sh: no argument (the dump) or --build, got $1" >&2; exit 64 ;;
esac
