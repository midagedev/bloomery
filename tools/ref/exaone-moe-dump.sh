#!/usr/bin/env bash
# Dump the K-EXAONE-2.0 oracle sets: the patched mainline llama.cpp's (7bcbc04fb + exaone-moe-oracle.patch, CPU build,
# models/exaone-moe.sh) forward pass over the four prompts of tools/ref/data/exaone-moe on the fixture GGUF (the bootstrap
# mix of the cut folder /models/fixtures/exaoneh), from ONE model load, as dump_ref-format sets under $BLOOMERY_DATA:
#   ref_exaone_a  English paragraph                every tap below, 32 greedy tokens
#   ref_exaone_b  Korean paragraph                 the layer taps (TAPS), 32 greedy tokens
#   ref_exaone_c  Rust function head               the same as b
#   ref_exaone_d  "The capital of France is"       the same as b
# Each set also holds result_output (the last position's whole logits row), topk.tsv (its 40 largest logits with their f64
# log-softmax) and greedy.tsv (the argmax tokens with their margins). Set names are the family's
# (crates/refset/src/arch/exaone_moe). The real tier of this profile is the same GGUF: the engine opens no FP8 folder.
#
# The prompts are raw text, no chat template; the ids come from the tree's own llama-tokenize taken once and kept in
# tools/ref/data/exaone-moe beside their text (the texts hold no final newline: one would be an id), BOS included as the
# file's tokenizer adds it; the dumper never tokenizes (a tokenizer difference would read as a kernel bug).
#
# The harness is the shared one: tools/ref/cb_dump.cpp is the dumper, tools/ref/cb-dump.sh the build and the run (staging,
# lease, install); this file is K-EXAONE-2.0's part of them, the tree, the patch and the prompts. `bash
# tools/ref/exaone-moe-dump.sh --build` links the dumper against the profile's tree.
#
# The taps are the graph's cb() names (src/models/exaone-moe.cpp, llama-graph.cpp build_moe_ffn), a layer's name
# `<node>-<il>`; a name a layer's kind does not have is absent from that layer (the dense blocks 0 and 1 and the next-token
# block 6 have no router nodes, the routed ones no ffn_gate). TAPS, every prompt:
#   attn_norm, ffn_norm       the attention site's and the ffn site's input, after the residual and the norm
#   ffn_moe_topk              the router's chosen expert ids (an i32 node: the f32 file, its i32 twin and its `int` row)
#   ffn_moe_weights_norm      their normalized weights (the softmax-free sigmoid rule, the bias out)
#   ffn_moe_weights_scaled    the scale 2.5 applied: the weights the experts take
#   ffn_out, l_out            the ffn site's output (the dense lead's, or the routed stack plus the shared one) and the
#                             stream after it
#   result_norm               the head's norm of the last row
# TAPS_A, prompt a, beside TAPS:
#   Qcur_normed, Kcur_normed  the query and the key after their per-head RMS norms, before rope
#   Qcur, Kcur, Vcur          the roped query and key, and the value
#   attn_out, ffn_inp         the attention output after its projection; the residual sum it joins
#   ffn_moe_logits, ffn_moe_probs, ffn_moe_probs_biased   the router: logits, sigmoid probabilities, the selection scores
#                             with the correction bias
#   ffn_moe_weights, ffn_moe_weights_sum, ffn_moe_weights_sum_clamped   the chosen probabilities, their row sum and its
#                             F16 floor
#   ffn_moe_out, ffn_shexp    the routed stack's summed output and the shared expert's
# The greedy continuation runs with no node asked for, so a set's `ffn_moe_topk` covers the prompt's own positions only.
#
# CPU only: CUDA_VISIBLE_DEVICES is emptied and the dumper asks for no GPU layers, so the reference is a CPU reduction
# order, as dump.sh's. -fa on: the engine's attention is the flash one. The run takes the CPU lease (REF_DUMP_LEASE=0 only
# for a dry reading). DUMP_DRY=1 prints what would run and stops with 0, before the lease and anything on disk.
#
# Wall [derived, docs/cards/kimi-k3-dump.card's fixture bound]: the fixture GGUF is 8.5 GB and the load maps it once; every
# prompt and greedy token then reads it from the page cache. The bound is the harness's 1800 s.
set -euo pipefail
export BLOOMERY_MODEL=exaone-moe
# shellcheck source=tools/ref/ref-paths.sh
source "${BASH_SOURCE[0]%/*}/ref-paths.sh"
# shellcheck source=tools/ref/cb-dump.sh
source "${BASH_SOURCE[0]%/*}/cb-dump.sh"

CB_FAMILY=exaone
CB_ENV=EXAONE
CB_SCRIPT=exaone-moe-dump.sh
CB_TREE=$LCPP
CB_PATCH=/root/agents/specs/exaone-r6a/exaone-moe-oracle.patch
CB_PATCH_TAG=exaone-moe
CB_ARCH=exaone-moe
CB_TITLE="exaone-moe-dump — the patched llama.cpp tree's intermediate tensors, raw f32, little-endian"
CB_IDS=exaone-moe
TAPS=attn_norm+ffn_norm+ffn_moe_topk+ffn_moe_weights_norm+ffn_moe_weights_scaled+ffn_out+l_out+result_norm
TAPS_A=$TAPS+Qcur_normed+Kcur_normed+Qcur+Kcur+Vcur+attn_out+ffn_inp+ffn_moe_logits+ffn_moe_probs+ffn_moe_probs_biased+ffn_moe_weights+ffn_moe_weights_sum+ffn_moe_weights_sum_clamped+ffn_moe_out+ffn_shexp
# name:ids:nodes:greedy — nodes joined by `+`.
CB_PROMPTS=("a:a:$TAPS_A:32" "b:b:$TAPS:32" "c:c:$TAPS:32" "d:d:$TAPS:32")
CB_FLAGS=(-fa on)

case ${1:-} in
  --build) cb_dump_build ;;
  '')
    # shellcheck source=tools/ref/lease.sh
    source "${BASH_SOURCE[0]%/*}/lease.sh"
    cb_dump_run
    ;;
  *) echo "exaone-moe-dump.sh: no argument (the dump) or --build, got $1" >&2; exit 64 ;;
esac
