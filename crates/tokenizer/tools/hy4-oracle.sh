#!/usr/bin/env bash
# hy4-oracle.sh — the reference ids of Hy4-flash's vocabulary for tests/hy4.rs, on stdout as the file tests/hy4-ids.tsv: one
# row per case of tests/cases.txt and tests/cases-hy4.txt (printf %b escapes), the text's bytes in hex, then llama-tokenize's ids
# with special tokens parsed (its default) and with --no-parse-special, each a space-separated list.
#
#   MODEL=<shard 1 of the Hy4-flash GGUF> bash crates/tokenizer/tools/hy4-oracle.sh > crates/tokenizer/tests/hy4-ids.tsv
#   MODEL=<shard 1> bash crates/tokenizer/tools/hy4-oracle.sh --prompts tools/ref/data/hy4   # the .ids beside each .txt
#
# The reference is the tree of llama.cpp PR #32517 ("Model : add hy_v4_flash support") at the commit PIN, built on the box with
# the CPU backend (LCPP names the tree; its `hy_v4` pre-tokenizer is the deepseek-v3 set of regexes, the vocabulary is the
# model's own). The binary's own `--version` must name the tree's HEAD, so the pins come from the commit the header records.
# MODEL is the first shard of the file (the metadata and the tokenizer; llama-tokenize opens every shard of a split, so the
# other shard must be there). No lease, no card (CUDA_VISIBLE_DEVICES is emptied; llama-tokenize loads the vocabulary only).
set -euo pipefail

PIN=0a054d0b02454c998e068381d15f3f6615bdd987
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TESTS=$HERE/../tests
LCPP=${LCPP:-/home/user/llama.cpp-hy4}
TOKENIZE=$LCPP/build/bin/llama-tokenize
[ -n "${MODEL:-}" ] || { echo "hy4-oracle: MODEL names the first shard of the Hy4-flash GGUF (it is unset)" >&2; exit 64; }
[ -x "$TOKENIZE" ] || { echo "hy4-oracle: no $TOKENIZE" >&2; exit 66; }
[ -f "$MODEL" ] || { echo "hy4-oracle: no vocabulary file at $MODEL" >&2; exit 66; }
commit=$(git -c safe.directory='*' -C "$LCPP" rev-parse HEAD)
[ "$commit" = "$PIN" ] || { echo "hy4-oracle: the tree $LCPP is at $commit, the pin is $PIN" >&2; exit 65; }
dirty=$(git -c safe.directory='*' -C "$LCPP" status --porcelain --untracked-files=no | wc -l | tr -d ' ')
[ "$dirty" = 0 ] || { echo "hy4-oracle: the tree $LCPP has $dirty tracked change(s)" >&2; exit 65; }
version=$(CUDA_VISIBLE_DEVICES= "$TOKENIZE" --version 2>&1 | sed -n 1p)
case $version in
  *"commit ${commit:0:9}"*) ;;
  *) echo "hy4-oracle: the binary reports '$version', the tree is at $commit" >&2; exit 65 ;;
esac

tmp=$(mktemp)
trap 'rm -f "$tmp" "$tmp.raw"' EXIT

# ids <flag...> — llama-tokenize --ids of $tmp, a space-separated list. A failed tokenize is a return 1, never an empty list: the
# callers read it as `x=$(ids) || exit`, because `set -e` does not reach a command substitution inside an argument.
ids() {
  CUDA_VISIBLE_DEVICES= "$TOKENIZE" -m "$MODEL" -f "$tmp" --ids --log-disable --no-bos --no-escape "$@" > "$tmp.raw" || return 1
  tr -d '[] ' < "$tmp.raw" | tr ',' ' ' | sed 's/ *$//'
}

if [ "${1:-}" = "--prompts" ]; then
  dir=${2:?"hy4-oracle: --prompts DIR"}
  for txt in "$dir"/*.txt; do
    cp "$txt" "$tmp"
    out=${txt%.txt}.ids
    list=$(ids) || { echo "hy4-oracle: llama-tokenize failed on $txt" >&2; exit 65; }
    [ -n "$list" ] || { echo "hy4-oracle: $txt has no ids" >&2; exit 65; }
    printf '%s\n' "$list" | tr ' ' '\n' > "$out"
    # The file adds no BOS: the same ids with and without --no-bos.
    CUDA_VISIBLE_DEVICES= "$TOKENIZE" -m "$MODEL" -f "$txt" --ids --log-disable --no-escape > "$tmp.raw"
    [ "$(tr -d '[] ' < "$tmp.raw" | tr ',' '\n' | sed '/^$/d' | tr '\n' ' ' | sed 's/ *$//')" = "$(tr '\n' ' ' < "$out" | sed 's/ *$//')" ] \
      || { echo "hy4-oracle: $txt tokenizes differently with a BOS" >&2; exit 65; }
    echo "hy4-oracle: $out $(wc -l < "$out" | tr -d ' ') ids" >&2
  done
  exit 0
fi

printf '# reference\tllama.cpp PR #32517 (Little0o0/llama.cpp) at %s, %s\n' "$commit" "$version"
printf '# vocabulary\tthe first shard of the Hy4-flash Q4_K_M GGUF, tokenizer.ggml.pre hy_v4\n'
printf '# columns\ttext bytes in hex, ids (special tokens parsed), ids (--no-parse-special)\n'
n=0
for cases in "$TESTS/cases.txt" "$TESTS/cases-hy4.txt"; do
  while IFS= read -r line || [ -n "$line" ]; do
    case $line in '#'*) continue ;; esac
    n=$((n + 1))
    printf '%b' "$line" > "$tmp"
    hex=$(od -An -v -tx1 "$tmp" | tr -d ' \n')
    parsed=$(ids) || { echo "hy4-oracle: llama-tokenize failed on case $n" >&2; exit 65; }
    plain=$(ids --no-parse-special) || { echo "hy4-oracle: llama-tokenize failed on case $n (no parse)" >&2; exit 65; }
    # An empty text is the one case with no ids.
    if [ -n "$hex" ] && { [ -z "$parsed" ] || [ -z "$plain" ]; }; then
      echo "hy4-oracle: case $n has no ids: $line" >&2
      exit 65
    fi
    printf '%s\t%s\t%s\n' "$hex" "$parsed" "$plain"
  done < "$cases"
done
printf '# complete\t%s\n' "$n"
