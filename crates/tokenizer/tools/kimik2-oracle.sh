#!/usr/bin/env bash
# kimik2-oracle.sh — the reference ids of Kimi K3's vocabulary for tests/kimik2.rs, on stdout as the file
# tests/kimik2-ids.tsv: one row per case of tests/cases.txt and tests/cases-kimik2.txt (printf %b escapes), the
# text's bytes in hex, then llama-tokenize's ids with special tokens parsed (its default) and with
# --no-parse-special, each a space-separated list.
#
# The reference is mainline llama.cpp built on the box (its `kimi-k2` pre-tokenizer is the hand-written
# `unicode_regex_split_custom_kimi_k2` of src/unicode.cpp): LCPP names the tree, MODEL the vocabulary file, the
# first shard of the K3 file (the metadata and the tokenizer, no tensors). The binary's own `--version` must
# name the tree's HEAD, so the pins come from the commit the header records. No lease, no card
# (CUDA_VISIBLE_DEVICES is emptied; llama-tokenize loads the vocabulary only).
#
# llama-tokenize opens every shard of a split before it reads the vocabulary, so shard 1 alone is refused while the
# later shards are missing or partial. vocab-only-gguf.py writes a copy of shard 1's metadata with no tensors and
# checks every key but `split.*` against it; `MODEL=<that copy>` runs this script on it, and the pins' header says so.
#
#   bash crates/tokenizer/tools/kimik2-oracle.sh > crates/tokenizer/tests/kimik2-ids.tsv
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TESTS=$HERE/../tests
LCPP=${LCPP:-/home/user/llama.cpp-ml4}
MODEL=${MODEL:-/models/Kimi-K3-UD-IQ1_S/Kimi-K3-UD-IQ1_S-00001-of-00014.gguf}
TOKENIZE=$LCPP/build/bin/llama-tokenize
[ -x "$TOKENIZE" ] || { echo "kimik2-oracle: no $TOKENIZE" >&2; exit 66; }
[ -f "$MODEL" ] || { echo "kimik2-oracle: no vocabulary file at $MODEL" >&2; exit 66; }
commit=$(git -c safe.directory='*' -C "$LCPP" rev-parse HEAD)
dirty=$(git -c safe.directory='*' -C "$LCPP" status --porcelain --untracked-files=no | wc -l | tr -d ' ')
[ "$dirty" = 0 ] || { echo "kimik2-oracle: the tree $LCPP has $dirty tracked change(s)" >&2; exit 65; }
version=$(CUDA_VISIBLE_DEVICES= "$TOKENIZE" --version 2>&1 | sed -n 1p)
case $version in
  *"commit ${commit:0:9}"*) ;;
  *) echo "kimik2-oracle: the binary reports '$version', the tree is at $commit" >&2; exit 65 ;;
esac

tmp=$(mktemp)
trap 'rm -f "$tmp" "$tmp.raw"' EXIT

# ids <flag...> — llama-tokenize --ids of $tmp, a space-separated list.
ids() {
  CUDA_VISIBLE_DEVICES= "$TOKENIZE" -m "$MODEL" -f "$tmp" --ids --log-disable --no-bos --no-escape "$@" > "$tmp.raw"
  tr -d '[] ' < "$tmp.raw" | tr ',' ' ' | sed 's/ *$//'
}

printf '# reference\tllama.cpp (mainline) at %s, %s\n' "$commit" "$version"
printf '# vocabulary\t%s\n' "$MODEL"
printf '# columns\ttext bytes in hex, ids (special tokens parsed), ids (--no-parse-special)\n'
n=0
for cases in "$TESTS/cases.txt" "$TESTS/cases-kimik2.txt"; do
  while IFS= read -r line || [ -n "$line" ]; do
    case $line in '#'*) continue ;; esac
    n=$((n + 1))
    printf '%b' "$line" > "$tmp"
    hex=$(od -An -v -tx1 "$tmp" | tr -d ' \n')
    printf '%s\t%s\t%s\n' "$hex" "$(ids)" "$(ids --no-parse-special)"
  done < "$cases"
done
printf '# complete\t%s\n' "$n"
