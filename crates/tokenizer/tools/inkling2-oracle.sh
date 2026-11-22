#!/usr/bin/env bash
# inkling2-oracle.sh — the reference ids of the inkling2 prompts, from the tree that will dump the
# family's oracle sets: llama-tokenize of the PR #32695 tree writes each .ids beside its .txt under
# the data directory (`--prompts tools/ref/data/inkling2`, one decimal id a line). The dumper never
# tokenizes (a tokenizer difference would read as a kernel bug), so its prompts read these files.
#
# The reference is the tree of llama.cpp PR #32695 ("convert: register the looped Inkling 2
# architecture separately") at the commit PIN — /home/user/llama.cpp-pr32695, the CPU build
# tools/ref/build-lcpp-pr.sh `build 32695 cpu` writes (LCPP names the tree; models/inkling2.sh names
# it too). The tree's runtime refuses the Mini's and the fixture's files (the fused NVFP4 stack's
# per-expert .scale tensors its own converter writes are never created by the loader), but
# llama-tokenize loads the vocabulary only, so the ids stand. The binary's own --version must name
# the tree's HEAD, so the ids come from the commit the header records. MODEL is the Mini's single
# UD-Q4_K_XL shard (the vocabulary is in it; BLOOMERY_REF_MODEL of the profile moves it). No lease,
# no card (CUDA_VISIBLE_DEVICES is emptied).
#
# The BOS question: the file's header states tokenizer.ggml.add_bos_token = false, and the script
# checks the ids are the same with and without --no-bos before it writes a file, so a file that
# grew a BOS would be refused by name. The pre-tokenizer is the file's own `inkling2` (the gpt-4o
# pretokenizer regex; the tokenizer round's cases oracle is that round's).
set -euo pipefail

PIN=6196d9802e0757a5218f16f91d9a6830793ba800
LCPP=${LCPP:-/home/user/llama.cpp-pr32695}
MODEL=${MODEL:-/models/Inkling-2-Mini-GGUF/Inkling-2-Mini-UD-Q4_K_XL.gguf}
TOKENIZE=$LCPP/build/bin/llama-tokenize
[ -x "$TOKENIZE" ] || { echo "inkling2-oracle: no $TOKENIZE" >&2; exit 66; }
[ -f "$MODEL" ] || { echo "inkling2-oracle: no vocabulary file at $MODEL" >&2; exit 66; }
commit=$(git -c safe.directory='*' -C "$LCPP" rev-parse HEAD)
[ "$commit" = "$PIN" ] || { echo "inkling2-oracle: the tree $LCPP is at $commit, the pin is $PIN" >&2; exit 65; }
dirty=$(git -c safe.directory='*' -C "$LCPP" status --porcelain --untracked-files=no | wc -l | tr -d ' ')
[ "$dirty" = 0 ] || { echo "inkling2-oracle: the tree $LCPP has $dirty tracked change(s)" >&2; exit 65; }
version=$(CUDA_VISIBLE_DEVICES= "$TOKENIZE" --version 2>&1 | sed -n 1p)
case $version in
  *"commit ${commit:0:9}"*) ;;
  *) echo "inkling2-oracle: the binary reports '$version', the tree is at $commit" >&2; exit 65 ;;
esac

[ "${1:-}" = --prompts ] || { echo "usage: inkling2-oracle.sh --prompts <data directory>" >&2; exit 64; }
dir=${2:?"inkling2-oracle: --prompts DIR"}

tmp=$(mktemp)
trap 'rm -f "$tmp" "$tmp.raw"' EXIT

# ids <flag...> — llama-tokenize --ids of $tmp, a space-separated list. A failed tokenize is a
# return 1, never an empty list.
ids() {
  CUDA_VISIBLE_DEVICES= "$TOKENIZE" -m "$MODEL" -f "$tmp" --ids --log-disable --no-bos --no-escape "$@" > "$tmp.raw" || return 1
  tr -d '[] ' < "$tmp.raw" | tr ',' ' ' | sed 's/ *$//'
}

for txt in "$dir"/*.txt; do
  cp "$txt" "$tmp"
  out=${txt%.txt}.ids
  list=$(ids) || { echo "inkling2-oracle: llama-tokenize failed on $txt" >&2; exit 65; }
  [ -n "$list" ] || { echo "inkling2-oracle: $txt has no ids" >&2; exit 65; }
  printf '%s\n' "$list" | tr ' ' '\n' > "$out"
  # The file adds no BOS: the same ids with and without --no-bos.
  CUDA_VISIBLE_DEVICES= "$TOKENIZE" -m "$MODEL" -f "$txt" --ids --log-disable --no-escape > "$tmp.raw"
  [ "$(tr -d '[] ' < "$tmp.raw" | tr ',' '\n' | sed '/^$/d' | tr '\n' ' ' | sed 's/ *$//')" = "$(tr '\n' ' ' < "$out" | sed 's/ *$//')" ] \
    || { echo "inkling2-oracle: $txt tokenizes differently with a BOS" >&2; exit 65; }
  echo "inkling2-oracle: $out $(wc -l < "$out" | tr -d ' ') ids" >&2
done
