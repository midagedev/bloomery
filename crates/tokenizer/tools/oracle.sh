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
# The vocabulary is V4.1's first shard unless TOKENIZER_VOCAB names another GGUF file; its sets go
# under $BLOOMERY_DATA/$TOKENIZER_SET (default `tokenizer`, V4.1's), so each vocabulary keeps a set
# of its own — the swap at the end replaces the whole directory.
#
# Usage: bash crates/tokenizer/tools/oracle.sh
#        TOKENIZER_VOCAB=/models/…/file.gguf TOKENIZER_SET=tokenizer-qwen3moe bash crates/tokenizer/tools/oracle.sh
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
# IK (the text trees) and BLOOMERY_DATA default as in every runner here.
# shellcheck source=tools/ref/ref-paths.sh
source "$ROOT/tools/ref/ref-paths.sh"
# ik with its k_ucat_map listing `~` as a symbol, as mainline and HF do; back to the V4.1 profile's ik
# tree once that tree carries the same list.
TOKENIZE=${TOKENIZE:-/home/user/ik-tilde/build/bin/llama-tokenize}
# The V4.1 file set: the deepseek41 profile's choice, exported by tools/box.sh.
V41_DIR=${BLOOMERY_V41_DIR:?BLOOMERY_V41_DIR unset — run through tools/box.sh, which exports it from the deepseek41 profile}

[ -x "$TOKENIZE" ] || { echo "oracle: no llama-tokenize at $TOKENIZE" >&2; exit 66; }
if [ -n "${TOKENIZER_VOCAB:-}" ]; then
  SHARD=$TOKENIZER_VOCAB
  [ -f "$SHARD" ] || { echo "oracle: no vocabulary file at $SHARD" >&2; exit 66; }
else
  SHARD=$(find "$V41_DIR" -maxdepth 1 -name '*-00001-of-*.gguf' | sort | head -1)
  [ -n "$SHARD" ] || { echo "oracle: no first shard under $V41_DIR" >&2; exit 66; }
  # The reference's `~` is a symbol, so `~/` is one word and one id; a tree whose `~` is in neither
  # P nor S returns [96, 17].
  probe=$(CUDA_VISIBLE_DEVICES= "$TOKENIZE" -m "$SHARD" -p '~/' --ids --log-disable)
  [ "$probe" = '[71520]' ] || {
    echo "oracle: $TOKENIZE tokenizes '~/' as $probe on $SHARD, not [71520]: its \`~\` is not a symbol" >&2
    exit 65
  }
fi
SET=${TOKENIZER_SET:-tokenizer}
case $SET in
  ''|*/*|.*) echo "oracle: '$SET' cannot name a set" >&2; exit 64 ;;
esac

OUT=$BLOOMERY_DATA/$SET
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
