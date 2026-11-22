#!/usr/bin/env bash
# Dump the Inkling 2 oracle sets: the patched llama.cpp PR #32695 tree's (6196d9802 + tools/ref/inkling2-scale.patch, CPU
# build, models/inkling2.sh) forward pass over the five prompts of tools/ref/data/inkling2 on the Mini's UD-Q4_K_XL file
# (17.6 GB, one shard), from ONE model load, as dump_ref-format sets under $BLOOMERY_DATA:
#   ref_inkling2_a  English paragraph, 35 ids        every tap below (TAPS_A), 32 greedy tokens
#   ref_inkling2_b  Korean paragraph, 35 ids         the layer taps (TAPS), 32 greedy tokens
#   ref_inkling2_c  Rust function head, 39 ids       the same as b
#   ref_inkling2_d  "The capital of France is", 5 ids  the same as b
#   ref_inkling2_e  the harbor narrative, 1038 ids   the same as b. The one prompt past the sliding window (the Mini's and
#                   the fixture's windowed layers cap at 1,024 keys): its last 14 positions attend exactly 1,024 keys with
#                   the oldest dropped while the global layers (slots 5, 11, 17, 23 and their looped copies) see the whole
#                   prefix, and it is the only prompt split across ubatches (512 + 512 + 14 at -ub 512), so the short
#                   convolutions' recurrent state and the KV carry across ubatch boundaries and the core's passes two and
#                   three read windows their own pass filled. a-d never leave the first ubatch or reach the window.
# Each set also holds result_output (the last position's whole logits row, the unpadded 200,058), topk.tsv (its 40 largest
# logits with their f64 log-softmax) and greedy.tsv (the argmax tokens with their margins). Set names are the family's
# (crates/refset/src/arch/inkling2). Under BLOOMERY_TIER=fixture the same prompts run on the fixture file
# (/models/fixtures/inkling2: thirteen layers of the Mini — the stem block whole, one whole core block, a coda layer of the
# Q8_0 experts, 25 visits — at the real hidden, window and extent) and the sets are fx_ref_inkling2_{a..e}.
#
# The Small arm (`bash tools/ref/inkling2-dump.sh --small`): the same tree, patch, prompts and taps on the Small's UD-Q2_K_XL file
# (six shards, 191 GB, the first shard names it; the family's base program: 48 stored layers, no loop, so a tap's slot is the
# layer and no loop_* tap exists) into ref_inkling2s_{a..e}, the family lcpp-inkling2s. It refuses to start under 200 GiB of
# MemAvailable (the file is mapped, not copied, but the page cache is what keeps the greedy steps off the disk) and refuses the
# fixture tier. DUMP_ONLY="d" dumps the named prompts only and DUMP_SKIP_DONE=1 leaves a set that already holds its trailer, so
# the long run restarts per prompt: one prompt a call, d first as the smoke. Wall [derived]: docs/cards/inkling2s-dump.card.
#
# The prompts are raw text, no chat template and no BOS (the file's header states add_bos_token = false, and
# crates/tokenizer/tools/inkling2-oracle.sh checked the ids are the same with and without --no-bos before it wrote them),
# so the model continues a sentence, a paragraph and a function head. The ids come from this tree's own llama-tokenize,
# kept in tools/ref/data/inkling2 beside their text; the dumper never tokenizes (a tokenizer difference would read as a
# kernel bug).
#
# The harness is the shared one: tools/ref/cb_dump.cpp is the dumper, tools/ref/cb-dump.sh the build and the run (staging,
# lease, install); this file is Inkling 2's part of them, the tree, the patch and the prompts. `bash
# tools/ref/inkling2-dump.sh --build` links the dumper against the profile's tree
# (tools/ref/inkling2-dump-build.sh is that call; the tree is built once, by `just build-inkling2-lcpp`).
#
# The taps are the graph's cb() names (src/models/inkling2.cpp, llama-graph.cpp build_attn and build_moe_ffn). A looped
# visit's tap carries its CACHE SLOT, not a visit index: the graph loops `for (il = 0; il < n_layer; ++il)` over the slots
# (n_layer_all: 48 on the Mini) with `ip = hparams.loop_layer(il)` the stored layer whose weights it shares
# (llama-hparams.cpp loop_layer, llama-model.cpp's sharing pass), and every cb names `<node>-<il>`
# (llama-context.cpp graph_get_cb). The slot layout on this tree: the stem 0-5, the core's passes in order — pass 0 the
# slots 6-17, pass 1 the 18-29, pass 2 the 30-41 — and the coda shifted past them at 42-47 (loop_layer's looped span is
# [core_start, core_start + passes*core) with the coda at `slot - loop_n_extra`; master's inkling2-loop.cpp appends the
# passes after the physical layers instead, so the two trees' slot NUMBERS differ while the visits are the same). One
# unique number a visit, so each visit's files are unambiguous with no pass suffix. Two renames the graph itself makes: a
# block-boundary visit's attention output is not added to a residual, so the tensor cb'd `attn_out` is re-cb'd `ffn_inp`
# (the last name wins — `attn_out-<slot>` is absent at the boundary slots 0, 6, 12, 18, 24, 30, 36, 42, its rows the
# `ffn_inp` file); and on the NVFP4 visits the patch's per-expert scale renames `ffn_moe_gate_up` to
# `ffn_moe_gate_up_scaled` and `ffn_moe_down` to `ffn_moe_down_scaled` (the Q8_0 coda's, scale-less, keep the plain
# names). A name a visit's kind does not have is absent from it (a tap the graph never computes is no file, never an
# error). TAPS, every prompt:
#   norm                    build_norm's one name, by occurrence: occurrence 0 of the bare `norm` is the embedding norm
#                           (the model's input, tok_norm); a local visit's `norm-<slot>` occurrences are its attention input
#                           after the block-residual mix (and the message controller on looped visits) then its ffn norm; a
#                           global visit's the q, k and v input norms then the ffn norm (three input norms, one a query)
#   Qcur, Kcur, Vcur        q, k and v after their short convolutions and epilogues (the q channel scale and the L2 clip at
#                           22.627; k's clip; v's gainless RMS norm); Rcur beside them the relative projection after its own
#                           clip at 8.0 (the log-scaled tau of the global layers already multiplied in)
#   kq_b                    the relative-bias table rows the layer gathers by key distance, the per-query sink logit
#                           already subtracted — the exact tensor build_attn adds to kq
#   kqv_out                 the attention output BEFORE its output gate (build_attn's return; wo is the model's own)
#   attn_out                the layer's attention output — after the gate, wo and the post-conv
#   ffn_inp                 the residual stream entering the FFN: the block-residual add's output, a fresh stream at a
#                           block boundary
#   ffn_moe_topk            the router's chosen expert ids (an i32 node: the f32 file, its i32 twin and its `int` row)
#   ffn_moe_weights_scaled  their sigmoid scores' selected sum, normalized (+1e-20) and times route_scale 8.0
#   ffn_out, l_out          the FFN output after its post-conv, and the layer output
#   loop_q_inp, loop_ffn_inp  the loop controllers' outputs: the q attention input and the ffn mix after the message merge,
#                           on the core visits of passes >= 1 only (the slots 18-41)
#   result_norm             the head's input after the output-residual mix and its norm (unnumbered)
# TAPS_A, prompt a, beside TAPS: the attention's own stages and the router's and the expert stack's whole ladder —
#   kq, kq_plus_kq_b, kq_soft_max, kqv   the logits, the bias added, the softmax (the sink in its denominator) and the value
#                           mix (attention is never flash here: kq_b forces the dense path, so the dump and the card engine
#                           compare the same graph shape)
#   ffn_moe_logits, ffn_moe_probs, ffn_moe_probs_biased   the router: its f32 logits, the sigmoid scores, the selection
#                           scores with the correction bias (the bias steers selection only)
#   ffn_moe_weights, ffn_moe_weights_sum, ffn_moe_weights_sum_clamped, ffn_moe_weights_norm   the chosen scores, their row
#                           sum, the +1e-20 and the normalized weights
#   ffn_moe_gate_up, ffn_moe_gate, ffn_moe_up   the fused stack's rows and their gate/up halves; on the NVFP4 visits the
#                           fused rows are renamed ffn_moe_gate_up_scaled (the per-expert scale the patch loads, applied)
#   ffn_moe_situ            the SITU activation at beta 4.0, linear beta 25.0
#   ffn_moe_down, ffn_moe_down_scaled, ffn_moe_weighted, ffn_moe_out   the down projection (the NVFP4 visits' under its
#                           scaled name), the weighted expert outputs and their sum
# The greedy continuation runs with no node asked for, so a set's `ffn_moe_topk` covers the prompt's own positions only.
#
# CPU only: CUDA_VISIBLE_DEVICES is emptied and the dumper asks for no GPU layers, so the reference is a CPU reduction
# order, as dump.sh's. Flash attention stays the library's own choice (unset -fa). The run takes the CPU lease
# (REF_DUMP_LEASE=0 only for a dry reading). DUMP_DRY=1 prints what would run and stops with 0, before the lease and
# anything on disk.
#
# Wall [derived, docs/cards/hy4-fx-dump.card's fixture bound]: the fixture's dump (5.0 GB mapped once) is minutes; the
# Mini's (17.6 GB from the page cache) ~ a minute a prompt with the e set's taps (the kq_b rows alone are ~100 MB a
# windowed visit at its 1,038 positions, so e lands near 10 GB — the harness's 1800 s bound holds; raise
# BLOOMERY_DUMP_BOUND only if a slower box needs it).
set -euo pipefail
export BLOOMERY_MODEL=inkling2
SMALL_MODEL=/models/Inkling-2-Small-GGUF/UD-Q2_K_XL/Inkling-2-Small-UD-Q2_K_XL-00001-of-00006.gguf
if [ "${1:-}" = --small ]; then
  [ "${BLOOMERY_TIER:-real}" = real ] || { echo "inkling2-dump.sh --small: the Small has no fixture; the real tier only (exit 66)" >&2; exit 66; }
  export BLOOMERY_REF_MODEL=$SMALL_MODEL
fi
# shellcheck source=tools/ref/ref-paths.sh
source "${BASH_SOURCE[0]%/*}/ref-paths.sh"
# shellcheck source=tools/ref/cb-dump.sh
source "${BASH_SOURCE[0]%/*}/cb-dump.sh"

CB_FAMILY=inkling2
CB_ENV=INKLING2
CB_SCRIPT=inkling2-dump.sh
CB_TREE=$LCPP
CB_PATCH=${BASH_SOURCE[0]%/*}/inkling2-scale.patch
CB_PATCH_TAG=inkling2-scale
CB_ARCH=inkling2
CB_TITLE="inkling2-dump — the patched llama.cpp PR #32695 tree's intermediate tensors, raw f32, little-endian"
CB_IDS=inkling2
TAPS=norm+Qcur+Kcur+Vcur+Rcur+kq_b+kqv_out+attn_out+ffn_inp+ffn_moe_topk+ffn_moe_weights_scaled+ffn_out+l_out+loop_q_inp+loop_ffn_inp+result_norm
TAPS_A=$TAPS+kq+kq_plus_kq_b+kq_soft_max+kqv+ffn_moe_logits+ffn_moe_probs+ffn_moe_probs_biased+ffn_moe_weights+ffn_moe_weights_sum+ffn_moe_weights_sum_clamped+ffn_moe_weights_norm+ffn_moe_gate_up+ffn_moe_gate_up_scaled+ffn_moe_gate+ffn_moe_up+ffn_moe_situ+ffn_moe_down+ffn_moe_down_scaled+ffn_moe_weighted+ffn_moe_out
# name:ids:nodes:greedy — nodes joined by `+`.
CB_PROMPTS=("a:a:$TAPS_A:32" "b:b:$TAPS:32" "c:c:$TAPS:32" "d:d:$TAPS:32" "e:e:$TAPS:32")
CB_FLAGS=()
[ "${1:-}" != --small ] || CB_SET_FAMILY=inkling2s

# small_memory_floor: refuse to load the Small's 191 GB with under 200 GiB of MemAvailable, naming the figure.
small_memory_floor() {
  local kb
  kb=$(awk '$1 == "MemAvailable:" { print $2 }' /proc/meminfo)
  [ -n "$kb" ] && [ "$kb" -ge $((200 * 1024 * 1024)) ] || {
    echo "inkling2-dump.sh --small: MemAvailable is $((${kb:-0} / 1024 / 1024)) GiB, under the 200 GiB the Small's 191 GB file needs resident (exit 75)" >&2
    exit 75
  }
}

case ${1:-} in
  --build) cb_dump_build ;;
  '')
    # shellcheck source=tools/ref/lease.sh
    source "${BASH_SOURCE[0]%/*}/lease.sh"
    cb_dump_run ;;
  --small)
    # shellcheck source=tools/ref/lease.sh
    source "${BASH_SOURCE[0]%/*}/lease.sh"
    small_memory_floor
    cb_dump_run ;;
  *) echo "inkling2-dump.sh: no argument (the dump), --small or --build, got $1" >&2; exit 64 ;;
esac
