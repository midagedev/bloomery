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
# The variant `d3k` (`bash tools/ref/mistrallarge4-dump.sh d3k`, `just dump-ref-mistrallarge4 d3k`) dumps the deep set past the
# lightning indexer's top-k (2,048 positions, so the selection is a cut and not every position):
#   ref_mistrallarge4_d3k  3,001 ids of prose, a decode-step set: the first 3,000 ids run as a quiet prefill (no node asked for)
#                          and the nodes below are dumped at the one step on the 3,001st id, at position 3,000 (`# prefill 3000`,
#                          `# decode_pos 3000`; refset's RefManifest::step reads it), the same taps one row a node and no greedy
#                          continuation. The step's selection is 2,048 of 3,001 positions on every full indexer layer.
# The ids are the first D3K_IDS of the tokenizer oracle's prose under this vocabulary (`just dump-ref-tokenizer mistrallarge4`,
# $BLOOMERY_DATA/ref-tokenizer-mistrallarge4/prose.nps.ids: the same text the other families' d3k sets take, no BOS as a.ids and
# b.ids), written to $BLOOMERY_DATA/mistrallarge4/d3k.ids for the dumper (the set's `# tokens` line is their record); D3K_IDS
# is refset's D3K_POSITION + 1. MISTRALLARGE4_TAG=mix256g names the Mix-256G file's ref_mistrallarge4_mix256g_d3k. The real
# files only: the fixture's top-k is 16 and its walk past it is the e2e gate's own.
#
# The variant `d3k-scores` (`just dump-ref-mistrallarge4 d3k-scores`) dumps the same deep step into its own set, with one tap more,
# the lightning indexer's per-row score:
#   ref_mistrallarge4_d3k_scores  `d3k`'s ids and taps plus `indexer_score`: each full indexer layer's score of every cached
#                          row at the step, the values its `top_k` is cut from, so a gate reads mainline's own per-row score and
#                          not a proxy of it. The UD-IQ1_S file only (a tag is refused): the Mix-256G file's deep set has no
#                          score tap.
#
# The harness is the shared one: tools/ref/cb_dump.cpp is the dumper, tools/ref/cb-dump.sh the build and the
# run (staging, lease, install); this file is Mistral Large 4's part of them, the tree, the taps and the prompts.
# `bash tools/ref/mistrallarge4-dump.sh --build` links the dumper against the ml4 tree
# (`just build-ref-mistrallarge4` is that call).
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
#   indexer_score                               the per-row score each full indexer layer cuts its list from
#                                              (`d3k-scores` only): `n_kv` wide at a decode step, a row whose cache
#                                              slot is padding masked to -inf; the fused op names one tensor a layer,
#                                              the unfused path names five, the last of them the masked score
#   top_k                                       the selected positions (an i32 node: the f32 file, its i32 twin
#                                              and its `int` row), one tensor a full indexer layer. A layer that
#                                              reuses the previous full layer's selection names the same tensor
#                                              again, so it is dumped once, under the last name it was given:
#                                              layer il reads the first `top_k-<j>` with j >= il (16, 17 and 18
#                                              read `top_k-18`)
#   Qcur_ssmax                                  the absorbed query, rope part in place, times the SSMax scale
#   fattn_mla                                   the flash's weighted latent sum before the value up-projection
#                                              (the CPU build takes the flash path; the card gate's `att`
#                                              tap)
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
# 251 GB box); the first run pages in what 60 ids and 64 greedy tokens route to. The d3k variant's wall is
# docs/cards/mistrallarge4-d3k-dump.card's (and the Mix-256G file's, mistrallarge4-mix256g-d3k-dump.card): six ubatches of 512
# ids touch every expert, so the whole file is paged in once, then the prefill is CPU-bound.
set -euo pipefail
export BLOOMERY_MODEL=mistral-large4
# shellcheck source=tools/ref/ref-paths.sh
source "${BASH_SOURCE[0]%/*}/ref-paths.sh"
# shellcheck source=tools/ref/cb-dump.sh
source "${BASH_SOURCE[0]%/*}/cb-dump.sh"

CB_FAMILY=mistrallarge4
# The fixture tier's sets name the cut they were dumped from, fx_ref_mistrallarge4e128_{a,b} (refset's fixture::A and fixture::B): the
# 128-expert fixture (/models/fixtures/mistrallarge4-e128) and a cut with another expert count never share a set path.
[ "${BLOOMERY_TIER:-real}" != fixture ] || CB_SET_FAMILY=mistrallarge4e128
CB_ENV=MISTRALLARGE4
CB_SCRIPT=mistrallarge4-dump.sh
CB_TREE=$LCPP
CB_PATCH=
CB_ARCH=mistral-large4
CB_TITLE="mistrallarge4-dump — mainline llama.cpp's intermediate tensors, raw f32, little-endian"
CB_IDS=mistrallarge4
TAPS=attn_norm+indexer_q_ssmax+indexer_k+indexer_weights+top_k+Qcur_ssmax+kv_cmpr+Kcur+fattn_mla+ffn_inp+ffn_moe_logits+ffn_moe_probs+ffn_moe_topk+ffn_moe_weights+ffn_moe_weights_norm+ffn_moe_out+ffn_shexp+ffn_out+l_out+result_norm
# name:ids:nodes:greedy — nodes joined by `+`. MISTRALLARGE4_TAG=mix256g names the sets of the Mix-256G file
# (BLOOMERY_REF_MODEL, `just dump-ref-mistrallarge4-mix256g`) ref_mistrallarge4_mix256g_{a,b}, over the same ids.
tag=${MISTRALLARGE4_TAG:+${MISTRALLARGE4_TAG}_}
CB_PROMPTS=("${tag}a:a:$TAPS:32" "${tag}b:b:$TAPS:32")
# The deep set with the per-row index score (`d3k-scores`).
TAPS_SCORES=$TAPS+indexer_score
CB_FLAGS=()

# The deep set's ids: D3K_IDS = refset's D3K_POSITION + 1 (the step's own id on top of the prefill).
D3K_IDS=3001
D3K_CORPUS=$BLOOMERY_DATA/ref-tokenizer-mistrallarge4/prose.nps.ids
D3K_FILE=$BLOOMERY_DATA/mistrallarge4/d3k.ids

# d3k_ids: the first D3K_IDS ids of the tokenizer oracle's prose as the dumper's ids file. The oracle's set must be finished
# (its `# complete` trailer), hold at least D3K_IDS ids and have an id on each of those lines; anything else ends the dump by
# name. A dry run (DUMP_DRY=1) reads the same and writes nothing. The ids file's sha256 is printed, the log's record of the slice.
d3k_ids() {
  local man=${D3K_CORPUS%/*}/MANIFEST.tsv have
  grep -q '^# complete' "$man" 2>/dev/null || {
    echo "mistrallarge4-dump.sh: $man is missing or has no completion trailer — run: just dump-ref-tokenizer mistrallarge4 (exit 66)" >&2
    exit 66
  }
  [ -r "$D3K_CORPUS" ] || { echo "mistrallarge4-dump.sh: no $D3K_CORPUS (exit 66)" >&2; exit 66; }
  have=$(wc -l < "$D3K_CORPUS")
  [ "$have" -ge "$D3K_IDS" ] || {
    echo "mistrallarge4-dump.sh: $D3K_CORPUS holds $have ids, the deep set wants $D3K_IDS (exit 65)" >&2
    exit 65
  }
  head -n "$D3K_IDS" "$D3K_CORPUS" | awk '$0 !~ /^[0-9]+$/ { bad = NR } END { exit bad ? 1 : 0 }' || {
    echo "mistrallarge4-dump.sh: a line among the first $D3K_IDS of $D3K_CORPUS is not an id (exit 65)" >&2
    exit 65
  }
  if [ "${DUMP_DRY:-0}" = 1 ]; then
    printf '[dry] d3k ids\t%s first %s of %s (not written)\n' "$D3K_FILE" "$D3K_IDS" "$D3K_CORPUS"
    return 0
  fi
  mkdir -p "${D3K_FILE%/*}"
  head -n "$D3K_IDS" "$D3K_CORPUS" > "$D3K_FILE.tmp.$$"
  mv -f "$D3K_FILE.tmp.$$" "$D3K_FILE"
  echo "d3k ids: $D3K_IDS of $D3K_CORPUS, sha256 $(sha256sum < "$D3K_FILE" | cut -d' ' -f1)"
}

case ${1:-} in
  --build) cb_dump_build ;;
  '')
    # shellcheck source=tools/ref/lease.sh
    source "${BASH_SOURCE[0]%/*}/lease.sh"
    cb_dump_run
    ;;
  d3k)
    [ "${BLOOMERY_TIER:-real}" != fixture ] || {
      echo "mistrallarge4-dump.sh: d3k is the real files' deep set (the fixture's top-k is 16: its walk past it is the e2e gate's own), BLOOMERY_TIER=fixture does not dump it (exit 64)" >&2
      exit 64
    }
    d3k_ids
    # One prompt: the ids file by absolute path, every tap, no greedy continuation, a decode step after a quiet prefill.
    CB_PROMPTS=("${tag}d3k:$D3K_FILE:$TAPS:0:step")
    # shellcheck source=tools/ref/lease.sh
    source "${BASH_SOURCE[0]%/*}/lease.sh"
    cb_dump_run
    ;;
  d3k-scores)
    [ "${BLOOMERY_TIER:-real}" != fixture ] || {
      echo "mistrallarge4-dump.sh: d3k-scores is the real file's deep set, BLOOMERY_TIER=fixture does not dump it (exit 64)" >&2
      exit 64
    }
    [ -z "$tag" ] || {
      echo "mistrallarge4-dump.sh: d3k-scores is the UD-IQ1_S file's set, MISTRALLARGE4_TAG=${MISTRALLARGE4_TAG} names another file's (exit 64)" >&2
      exit 64
    }
    d3k_ids
    CB_PROMPTS=("d3k_scores:$D3K_FILE:$TAPS_SCORES:0:step")
    # shellcheck source=tools/ref/lease.sh
    source "${BASH_SOURCE[0]%/*}/lease.sh"
    cb_dump_run
    ;;
  *) echo "mistrallarge4-dump.sh: no argument (the dump), d3k, d3k-scores or --build, got $1" >&2; exit 64 ;;
esac
