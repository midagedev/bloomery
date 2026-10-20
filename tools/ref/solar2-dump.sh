#!/usr/bin/env bash
# Dump the Solar-Open2 oracle sets: the PAI tree's (llama.cpp 6ea215d17 + prometheusAIR's solar_open2 patch) forward pass over
# the four prompts of tools/ref/data/solar2 on the profile's Q4_K_M file, from ONE model load, as dump_ref-format sets under
# $BLOOMERY_DATA: ref_solar2_a (41 ids: every node of the graph's per-layer taps, 64 greedy tokens), ref_solar2_b (40 ids, Korean) and
# ref_solar2_c (114 ids, Rust): each layer output l_out and 64 greedy tokens, ref_solar2_d (1,536 ids of wikitext-2): the last
# position's logits only. Each set also holds topk.tsv (the last position's 40 largest logits with their f64 log-softmax) and
# greedy.tsv (the argmax tokens with their margins). Set names are the family's (crates/refset/src/arch/solar2); under
# BLOOMERY_TIER=fixture each is the fixture file's, behind `fx_`, with its `# fixture` line.
#
# The ids are tokens of the PAI build's llama-tokenize on the file (add_bos_token is false: no BOS), taken once and kept in
# tools/ref/data/solar2 beside the texts' sha256 below; the dumper never tokenizes (a tokenizer difference would read as a
# kernel bug). prompt a/b/c text: /home/user/stage/sorc/prompts/{a,b,c}.txt (sorc round, 2026-10-20); d is the first 1,536 ids of
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
# shellcheck source=tools/ref/lease.sh
source "${BASH_SOURCE[0]%/*}/lease.sh"
HERE=$(cd "${BASH_SOURCE[0]%/*}/../.." && pwd)
fixture_dump_tier solar2-dump.sh

BIN="${SOLAR2_BIN:-$BLOOMERY_DATA/bin/solar2_dump}"
REC="$BIN.build"
[ -x "$BIN" ] || { echo "solar2-dump: no $BIN — run: bash tools/ref/solar2-dump-build.sh" >&2; exit 2; }
want=$(cat "$HERE/tools/ref/solar2-dump.cpp" "$HERE/tools/ref/ref_io.h" | sha256sum | cut -d' ' -f1)
rec() { awk -v k="$1" '$1 == k { print $2; exit }' "$REC"; }
[ "$(rec source_sha256)" = "$want" ] || {
  echo "[stale-binary] $BIN was linked from another solar2-dump.cpp or ref_io.h — run: bash tools/ref/solar2-dump-build.sh" >&2
  exit 3
}
# The binary must still be the tree it was linked against: its commit, patch and library.
COMMIT=$(git -c safe.directory='*' -C "$PAI" rev-parse --short=9 HEAD)
PATCH_SHA=$(sha256sum "$PAI/solar_open2.patch" | cut -d' ' -f1)
LIB_MD5=$(md5sum < "$(readlink -f "$PAI/build/bin/libllama.so")" | cut -d' ' -f1)
[ "$(rec pai_commit) $(rec patch_sha256) $(rec libllama_md5)" = "$COMMIT $PATCH_SHA $LIB_MD5" ] || {
  echo "[stale-binary] $BIN was linked against another PAI build than $PAI now holds ($(rec pai_commit) $(rec libllama_md5)): rebuild it" >&2
  exit 3
}
BUILD="$COMMIT+pai-${PATCH_SHA:0:12}"

# name:ids:nodes:greedy — nodes joined by `+`; the names are the patch's cb() names.
TAPS=model.embed_tokens+attn_norm+kda_g1+kda_beta_neg_eigval+kda_out+attn_out+attn_gate_sigmoid+attn_gated+attn_proj+ffn_inp+ffn_norm+ffn_moe_out+ffn_shexp+ffn_out+l_out+result_norm
PROMPTS=("a:a:$TAPS:64" "b:b:l_out:64" "c:c:l_out:64" "d:d::0")
STAGE="$BLOOMERY_DATA/ref-solar2.staging"
FIXTURE_LINE=$(fixture_line solar2-dump.sh)
ARGS=(-m "$MODEL" --out "$STAGE" --build "$BUILD" -c 4096 -ub 512 -t "${SOLAR2_THREADS:-32}")
[ -z "$FIXTURE_LINE" ] || ARGS+=(--fixture-line "$FIXTURE_LINE")
for p in "${PROMPTS[@]}"; do
  IFS=: read -r name ids nodes greedy <<< "$p"
  ARGS+=(--prompt "$name:$HERE/tools/ref/data/solar2/$ids.ids:$nodes:$greedy")
done

if [ "${DUMP_DRY:-0}" = 1 ]; then
  printf '[dry] tier\t%s\tprofile %s\n' "${BLOOMERY_TIER:-real}" "$MODEL_NAME"
  for p in "${PROMPTS[@]}"; do printf '[dry] set\t%sref_solar2_%s\n' "$SET_PREFIX" "${p%%:*}"; done
  printf '[dry] # model\t%s\n[dry] # build\t%s\n[dry] dumper\t%s\n' "$MODEL" "$BUILD" "$BIN"
  [ -z "$FIXTURE_LINE" ] || printf '[dry] %s\n' "$FIXTURE_LINE"
  printf '[dry] argv\t%q' "$BIN"
  printf ' %q' "${ARGS[@]}"
  printf '\n[dry] lease\ttaken (a dry run takes none)\n'
  exit 0
fi

DUMP_BOUND=${BLOOMERY_DUMP_BOUND:-1800}
WITNESS=(head-epoch loadavg pressure-io mem pgmajfault read-sectors lock-holder model)
if [ "${REF_DUMP_LEASE:-1}" = 1 ]; then
  lease_take
  witness pre-dump
fi
rm -rf "$STAGE"
mkdir -p "$STAGE"
# The staged sets are removed on every exit unless each carries its completion trailer: a dump that does not finish costs nothing.
drop_incomplete_sets() {
  local d
  for d in "$STAGE"/*/; do
    [ -d "$d" ] || continue
    grep -qs '^# complete' "$d/MANIFEST.tsv" || { rm -rf "$d"; echo "[staging] removed the incomplete $d" >&2; }
  done
}
trap drop_incomplete_sets EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
export CUDA_VISIBLE_DEVICES=""
lease_bounded "$DUMP_BOUND" "$BIN" "${ARGS[@]}"
if [ "${REF_DUMP_LEASE:-1}" = 1 ]; then witness post-dump; fi
for p in "${PROMPTS[@]}"; do
  name=${p%%:*}
  grep -q '^# complete' "$STAGE/$name/MANIFEST.tsv" || { echo "solar2-dump: $name has no completion trailer — not installing it" >&2; exit 1; }
done
for p in "${PROMPTS[@]}"; do
  name=${p%%:*}
  SET="$BLOOMERY_DATA/${SET_PREFIX}ref_solar2_$name"
  rm -rf "$SET.old"
  if [ -d "$SET" ]; then mv "$SET" "$SET.old"; fi
  mv "$STAGE/$name" "$SET"
  rm -rf "$SET.old"
  echo "installed $SET ($(grep -c '^tensor' "$SET/MANIFEST.tsv") tensors)"
done
rmdir "$STAGE"
echo "build: $BUILD  model: $MODEL"
