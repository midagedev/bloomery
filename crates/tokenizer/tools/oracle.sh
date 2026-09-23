#!/usr/bin/env bash
# oracle.sh — reference token ids for the tokenizer gate, written next to the exact text they came
# from.
#
# Not a measurement: no lease, no timing, no GPU (CUDA_VISIBLE_DEVICES is emptied; llama-tokenize
# loads the vocabulary only). The text sets are engram-corpus.sh's, gathered the same way from the
# same trees, so the gate covers what the engram rounds tokenized:
#
#   code       $IK/src/*.cpp
#   prose      $IK/docs/**/*.md and $IK/README.md
#   prose-all  every Markdown file under $IK
#   threads    $IK/github-data/**/*.md, byte order
#   korean     this tree's docs/**/*.md that are at least 20 % Hangul, byte order
#
# plus the hand-picked cases in crates/tokenizer/tests/cases.txt (one per line, printf %b
# escapes; an empty line is the empty string; a line starting with '#' is a comment).
#
# Each set is tokenized twice: with special-token parsing (llama-tokenize's default) and with
# --no-parse-special (what engram-corpus.sh used). Output, under $BLOOMERY_DATA/tokenizer/:
#
#   <set>.txt            the text, exactly as the reference read it
#   <set>.ids            one id per line, special tokens parsed
#   <set>.nps.ids        one id per line, --no-parse-special
#   cases/<nn>.{txt,ids,nps.ids}
#   MANIFEST.tsv         the tokenizer binary, the vocabulary file, and each set's md5s and counts
#
# Usage: bash crates/tokenizer/tools/oracle.sh
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
# IK (the text trees) and BLOOMERY_DATA default as in every runner here.
# shellcheck source=tools/ref/ref-paths.sh
source "$ROOT/tools/ref/ref-paths.sh"
TOKENIZE=${TOKENIZE:-/home/user/ik-idxkey/build/bin/llama-tokenize}
V41_DIR=${BLOOMERY_V41_DIR:-/models/DeepSeek-V4.1-Flash-Q3_K_M-engramQ8-tokembdBF16-attnQ8}

[ -x "$TOKENIZE" ] || { echo "oracle: no llama-tokenize at $TOKENIZE" >&2; exit 66; }
SHARD=$(find "$V41_DIR" -maxdepth 1 -name '*-00001-of-*.gguf' | sort | head -1)
[ -n "$SHARD" ] || { echo "oracle: no first shard under $V41_DIR" >&2; exit 66; }

OUT=$BLOOMERY_DATA/tokenizer
rm -rf "$OUT.new"
mkdir -p "$OUT.new/cases"

# gather <name> — engram-corpus.sh's gather(), same finds, same sort orders.
gather() {
  case $1 in
    code)
      find "$IK/src" -maxdepth 1 -name '*.cpp' | sort | xargs -d '\n' cat -- ;;
    prose)
      { find "$IK/docs" -name '*.md' | sort | xargs -d '\n' cat --; cat -- "$IK/README.md"; } ;;
    prose-all)
      find "$IK" -name '*.md' -not -path '*/.git/*' | sort | xargs -d '\n' cat -- ;;
    threads)
      find "$IK/github-data" -name '*.md' | LC_ALL=C sort | xargs -d '\n' cat -- ;;
    korean)
      find "$ROOT/docs" -name '*.md' | LC_ALL=C sort | python3 -c '
import sys
for path in sys.stdin.read().splitlines():
    t = open(path, encoding="utf-8").read()
    if sum("가" <= c <= "힣" for c in t) >= 0.2 * max(1, len(t)):
        print(path)
' | xargs -d '\n' cat -- ;;
  esac
}

# ids <text file> <out prefix> — both parse modes, one id per line.
ids() {
  one "$1" "$2.ids"
  one "$1" "$2.nps.ids" --no-parse-special
}

# one <text file> <out file> [flag] — llama-tokenize --ids, reshaped to one id per line.
one() {
  CUDA_VISIBLE_DEVICES= "$TOKENIZE" -m "$SHARD" -f "$1" --ids --log-disable "${@:3}" > "$2.raw"
  tr -d '[] ' < "$2.raw" | tr ',' '\n' | { grep -v '^$' || true; } > "$2"
  rm -f "$2.raw"
}

{
  printf '# tokenizer\t%s\t%s\n' "$TOKENIZE" "$(md5sum < "$TOKENIZE" | cut -d' ' -f1)"
  printf '# vocabulary\t%s\n' "$SHARD"
  printf '# text tree\t%s\n' "$IK"
  printf 'set\tbytes\ttext_md5\tids\tnps_ids\n'
} > "$OUT.new/MANIFEST.tsv"

for name in code prose prose-all threads korean; do
  gather "$name" > "$OUT.new/$name.txt"
  ids "$OUT.new/$name.txt" "$OUT.new/$name"
  printf '%s\t%s\t%s\t%s\t%s\n' "$name" "$(wc -c < "$OUT.new/$name.txt")" \
    "$(md5sum < "$OUT.new/$name.txt" | cut -d' ' -f1)" \
    "$(wc -l < "$OUT.new/$name.ids")" "$(wc -l < "$OUT.new/$name.nps.ids")" >> "$OUT.new/MANIFEST.tsv"
done

n=0
while IFS= read -r line || [ -n "$line" ]; do
  case $line in '#'*) continue ;; esac
  n=$((n + 1))
  f=$(printf '%s/cases/%02d' "$OUT.new" "$n")
  printf '%b' "$line" > "$f.txt"
  ids "$f.txt" "$f"
done < "$ROOT/crates/tokenizer/tests/cases.txt"
printf 'cases\t%s\n' "$n" >> "$OUT.new/MANIFEST.tsv"

rm -rf "$OUT"
mv "$OUT.new" "$OUT"
cat "$OUT/MANIFEST.tsv"
