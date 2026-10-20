#!/usr/bin/env bash
# Dump the Solar-Open2 oracle sets: the PAI tree's (llama.cpp 6ea215d17 + prometheusAIR's solar_open2 patch) forward pass over
# the four prompts of tools/ref/data/solar2 on the profile's Q4_K_M file, from ONE model load, as dump_ref-format sets under
# $BLOOMERY_DATA: ref_solar2_a (41 ids: every node of the graph's per-layer taps, 64 greedy tokens), ref_solar2_b (40 ids, Korean) and
# ref_solar2_c (114 ids, Rust): each layer output l_out and 64 greedy tokens, ref_solar2_d (1,536 ids of wikitext-2): the last
# position's logits only. Each set also holds topk.tsv (the last position's 40 largest logits with their f64 log-softmax) and
# greedy.tsv (the argmax tokens with their margins). Set names are the family's (crates/refset/src/arch/solar2); under
# BLOOMERY_TIER=fixture each is the fixture file's, behind `fx_`, with its `# fixture` line.
#
# The harness is the shared one: tools/ref/cb_dump.cpp is the dumper, tools/ref/cb-dump.sh the build and the run (staging, lease,
# install); this file is Solar's part of them, the tree, the taps and the prompts. `bash tools/ref/solar2-dump.sh --build` links
# the dumper against the PAI tree (tools/ref/solar2-dump-build.sh is that call).
#
# The ids are tokens of the PAI build's llama-tokenize on the file (add_bos_token is false: no BOS), taken once and kept in
# tools/ref/data/solar2; the dumper never tokenizes (a tokenizer difference would read as a kernel bug). prompt a/b/c text:
# /home/user/stage/sorc/prompts/{a,b,c}.txt (sorc round, 2026-10-20); d is the first 1,536 ids of
# /home/user/eval/wikitext-2-raw/wiki.test.raw.
#
# CPU only: CUDA_VISIBLE_DEVICES is emptied and the dumper asks for no GPU layers, so the reference is a CPU reduction order, as
# dump.sh's. The run takes the CPU lease (REF_DUMP_LEASE=0 only for a dry reading). DUMP_DRY=1 prints what would run and stops
# with 0, before the lease and anything on disk.
#
# Wall: load 151.8 GB through the CPU backend, which repacks every Q4_K/Q6_K weight to its 8x8 interleave in anonymous memory
# (~150 GB of the 251 GB box RAM; page cache gives way): 185 s warm, +~50 s cold at the root drive's ~3 GB/s; prefill 1,731 ids x
# 2 x 13.9 G MACs ~ 48 TFLOP and 3 x 1.3 s of expert reads for the 1,536-id prompt's ubatches, greedy 192 tokens at 8.7 GB a token
# over ~120 GB/s, and the ~0.6 GB of writes: 66 s together on the first run. Total 251 s measured, bound 1800 s.
set -euo pipefail
export BLOOMERY_MODEL=solar-open2
# shellcheck source=tools/ref/ref-paths.sh
source "${BASH_SOURCE[0]%/*}/ref-paths.sh"
# shellcheck source=tools/ref/cb-dump.sh
source "${BASH_SOURCE[0]%/*}/cb-dump.sh"

CB_FAMILY=solar2
CB_ENV=SOLAR2
CB_SCRIPT=solar2-dump.sh
CB_TREE=$PAI
CB_PATCH=$PAI/solar_open2.patch
CB_PATCH_TAG=pai
CB_ARCH=solar-open2
CB_TITLE="solar2-dump — the PAI tree's intermediate tensors, raw f32, little-endian"
CB_IDS=solar2
# name:ids:nodes:greedy — nodes joined by `+`; the names are the patch's cb() names.
TAPS=model.embed_tokens+attn_norm+kda_g1+kda_beta_neg_eigval+kda_out+attn_out+attn_gate_sigmoid+attn_gated+attn_proj+ffn_inp+ffn_norm+ffn_moe_out+ffn_shexp+ffn_out+l_out+result_norm
CB_PROMPTS=("a:a:$TAPS:64" "b:b:l_out:64" "c:c:l_out:64" "d:d::0")
CB_FLAGS=()

case ${1:-} in
  --build) cb_dump_build ;;
  '')
    # shellcheck source=tools/ref/lease.sh
    source "${BASH_SOURCE[0]%/*}/lease.sh"
    cb_dump_run
    ;;
  *) echo "solar2-dump.sh: no argument (the dump) or --build, got $1" >&2; exit 64 ;;
esac
