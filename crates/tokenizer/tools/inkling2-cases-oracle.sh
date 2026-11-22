#!/usr/bin/env bash
# inkling2-cases-oracle.sh — the reference ids of the Inkling 2 vocabulary for tests/inkling2.rs, on stdout as the file
# tests/inkling2-ids.tsv: one row per case of tests/cases.txt and tests/cases-inkling2.txt (printf %b escapes), the text's bytes
# in hex, then llama-tokenize's ids with special tokens parsed (its default) and with --no-parse-special, each a
# space-separated list. Both released files (the Mini's `inkling2` pre, the Small's `gpt-4o`) are run through it and must write
# the same table, which the gate holds each file to.
#
#   MODEL=<the Mini's or the Small's first shard> bash crates/tokenizer/tools/inkling2-cases-oracle.sh > crates/tokenizer/tests/inkling2-ids.tsv
#
# The reference is the family's oracle tree (LCPP, PR #32695 at PIN carrying tools/ref/inkling2-scale.patch, which touches
# no tokenizer file): its HEAD must be PIN and the binary's own --version must name it. llama-tokenize opens every shard of a
# split, so the other shards must be there. No lease, no card (CUDA_VISIBLE_DEVICES is emptied; the vocabulary only).
set -euo pipefail

PIN=6196d9802e0757a5218f16f91d9a6830793ba800
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TESTS=$HERE/../tests
LCPP=${LCPP:-/home/user/llama.cpp-pr32695-scale}
TOKENIZE=$LCPP/build/bin/llama-tokenize
[ -n "${MODEL:-}" ] || { echo "inkling2-cases-oracle: MODEL names the first shard of an Inkling 2 GGUF (it is unset)" >&2; exit 64; }
[ -x "$TOKENIZE" ] || { echo "inkling2-cases-oracle: no $TOKENIZE" >&2; exit 66; }
[ -f "$MODEL" ] || { echo "inkling2-cases-oracle: no vocabulary file at $MODEL" >&2; exit 66; }
commit=$(git -c safe.directory='*' -C "$LCPP" rev-parse HEAD)
[ "$commit" = "$PIN" ] || { echo "inkling2-cases-oracle: the tree $LCPP is at $commit, the pin is $PIN" >&2; exit 65; }
version=$(CUDA_VISIBLE_DEVICES= "$TOKENIZE" --version 2>&1 | sed -n 1p)
case $version in
  *"commit ${commit:0:9}"*) ;;
  *) echo "inkling2-cases-oracle: the binary reports '$version', the tree is at $commit" >&2; exit 65 ;;
esac
pre=$(python3 -I - "$MODEL" <<'PY'
import struct, sys
with open(sys.argv[1], "rb") as f:
    buf = f.read(64 << 20)
at = buf.find(b"tokenizer.ggml.pre")
if at < 0:
    sys.exit("no tokenizer.ggml.pre in the first 64 MiB")
at += len(b"tokenizer.ggml.pre")
(ty,) = struct.unpack_from("<I", buf, at)
(n,) = struct.unpack_from("<Q", buf, at + 4)
print(buf[at + 12 : at + 12 + n].decode())
PY
)

tmp=$(mktemp)
trap 'rm -f "$tmp" "$tmp.raw"' EXIT

# ids <flag...> — llama-tokenize --ids of $tmp, a space-separated list. A failed tokenize is a return 1, never an empty list.
ids() {
  CUDA_VISIBLE_DEVICES= "$TOKENIZE" -m "$MODEL" -f "$tmp" --ids --log-disable --no-bos --no-escape "$@" > "$tmp.raw" || return 1
  tr -d '[] ' < "$tmp.raw" | tr ',' ' ' | sed 's/ *$//'
}

printf '# reference\tllama.cpp PR #32695 at %s (%s), the oracle tree with tools/ref/inkling2-scale.patch\n' "$commit" "$version"
printf '# vocabulary\tInkling 2, tokenizer.ggml.pre %s (the Mini spells inkling2, the Small gpt-4o; both write this table)\n' "$pre"
printf '# columns\ttext bytes in hex, ids (special tokens parsed), ids (--no-parse-special)\n'
n=0
for cases in "$TESTS/cases.txt" "$TESTS/cases-inkling2.txt"; do
  while IFS= read -r line || [ -n "$line" ]; do
    case $line in '#'*) continue ;; esac
    n=$((n + 1))
    printf '%b' "$line" > "$tmp"
    hex=$(od -An -v -tx1 "$tmp" | tr -d ' \n')
    parsed=$(ids) || { echo "inkling2-cases-oracle: llama-tokenize failed on case $n" >&2; exit 65; }
    plain=$(ids --no-parse-special) || { echo "inkling2-cases-oracle: llama-tokenize failed on case $n (no parse)" >&2; exit 65; }
    # An empty text is the one case with no ids.
    if [ -n "$hex" ] && { [ -z "$parsed" ] || [ -z "$plain" ]; }; then
      echo "inkling2-cases-oracle: case $n has no ids: $line" >&2
      exit 65
    fi
    printf '%s\t%s\t%s\n' "$hex" "$parsed" "$plain"
  done < "$cases"
done
printf '# complete\t%s\n' "$n"
