#!/usr/bin/env bash
# oracle.sh — reference token ids for the tokenizer gate, written next to the exact text they came
# from. The writer behind `just dump-ref-tokenizer`; `just gate-tokenizer` only reads the sets,
# through refset's tokenizer families (crates/refset/src/arch/tokenizer.rs), which refuse a set that
# this script did not finish or that another reference executable, library or vocabulary wrote.
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
# Each text is tokenized twice: with special-token parsing (llama-tokenize's default) and with
# --no-parse-special (what engram-corpus.sh used). Output, under $BLOOMERY_DATA/<set>/:
#
#   <name>.txt           the text, exactly as the reference read it
#   <name>.ids           one id per line, special tokens parsed
#   <name>.nps.ids       one id per line, --no-parse-special
#   cases/<nn>.{txt,ids,nps.ids}
#   MANIFEST.tsv         the executable and the libllama.so it loads (path, md5), the vocabulary
#                        file, the cases file (path, md5), one row per text (the corpus names, then
#                        cases/<nn>: bytes, text md5, id counts), and the `# complete` trailer
#
# The vocabularies, each with its reference tree and its set (`profile` below owns the table;
# crates/refset/src/arch/tokenizer.rs pins the same executables and libraries by md5):
#
#   v41       V4.1's first shard                  ik-tilde  (`~` is a symbol, as in mainline and HF)  ref-tokenizer-v41
#   qwen3moe  the Qwen3-MoE file                  ik-tokref (the unicode_tolower fix)                  ref-tokenizer-qwen3moe
#   glm5next  GLM-5.3-Flash's first shard         ik-tokref                                            ref-tokenizer-glm5next
#   qwen38    Qwen3.8-Flash-Next's first shard    ik-tokref                                            ref-tokenizer-qwen38
#
# The sets' directories are `ref-tokenizer-*`, not the `tokenizer*` of the format this script wrote
# before it wrote a trailer: a checkout that still runs that format's gate rewrites those and cannot
# touch these.
#
# Every vocabulary reads the same texts: the default profile's ik tree and this tree's docs/. `all`
# takes one snapshot of docs/ per vocabulary, in turn.
#
# Usage: bash crates/tokenizer/tools/oracle.sh v41|qwen3moe|glm5next|qwen38|all
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
CASES=$ROOT/crates/tokenizer/tests/cases.txt

case ${1:-} in
  v41 | qwen3moe | glm5next | qwen38) [ $# = 1 ] || { echo "usage: oracle.sh v41|qwen3moe|glm5next|qwen38|all" >&2; exit 64; } ;;
  all)
    [ $# = 1 ] || { echo "usage: oracle.sh v41|qwen3moe|glm5next|qwen38|all" >&2; exit 64; }
    for v in v41 qwen3moe glm5next qwen38; do
      echo "oracle: $v"
      bash "${BASH_SOURCE[0]}" "$v"
    done
    exit 0
    ;;
  *) echo "usage: oracle.sh v41|qwen3moe|glm5next|qwen38|all" >&2; exit 64 ;;
esac
VOCAB_NAME=$1

# IK (the text trees) and BLOOMERY_DATA default as in every runner here. The texts are the default
# profile's: another profile's IK tree would give this vocabulary other texts than its siblings.
# shellcheck source=tools/ref/ref-paths.sh
source "$ROOT/tools/ref/ref-paths.sh"
[ "$BLOOMERY_MODEL" = deepseek2 ] || {
  echo "oracle: the texts are the default profile's (deepseek2); BLOOMERY_MODEL=$BLOOMERY_MODEL would take another IK tree" >&2
  exit 64
}

# profile <name>: TOKENIZE (the reference executable), SHARD (the vocabulary file) and SET (the
# directory under $BLOOMERY_DATA) of one vocabulary.
profile() {
  case $1 in
    v41)
      # The V4.1 file set is the deepseek41 profile's choice, exported by tools/box.sh.
      local dir=${BLOOMERY_V41_DIR:?BLOOMERY_V41_DIR unset — run through tools/box.sh, which exports it from the deepseek41 profile}
      TOKENIZE=/home/user/ik-tilde/build/bin/llama-tokenize
      SHARD=$(find "$dir" -maxdepth 1 -name '*-00001-of-*.gguf' | sort | head -1)
      [ -n "$SHARD" ] || { echo "oracle: no first shard under $dir" >&2; exit 66; }
      SET=ref-tokenizer-v41
      ;;
    qwen3moe)
      TOKENIZE=/home/user/ik-tokref/build/bin/llama-tokenize
      SHARD=${BLOOMERY_QWEN3MOE_VOCAB:-/models/Qwen3-30B-A3B/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf}
      SET=ref-tokenizer-qwen3moe
      ;;
    glm5next)
      TOKENIZE=/home/user/ik-tokref/build/bin/llama-tokenize
      SHARD=/models/GLM-5.3-Flash-UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf
      SET=ref-tokenizer-glm5next
      ;;
    qwen38)
      TOKENIZE=/home/user/ik-tokref/build/bin/llama-tokenize
      SHARD=/models/Qwen3.8-Flash-Next/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
      SET=ref-tokenizer-qwen38
      ;;
  esac
}
profile "$VOCAB_NAME"

[ -x "$TOKENIZE" ] || { echo "oracle: no llama-tokenize at $TOKENIZE" >&2; exit 66; }
[ -f "$SHARD" ] || { echo "oracle: no vocabulary file at $SHARD" >&2; exit 66; }
LIBLLAMA=$(ldd "$TOKENIZE" | awk '$1 == "libllama.so" { print $3 }')
[ -f "$LIBLLAMA" ] || { echo "oracle: $TOKENIZE loads no libllama.so (ldd names '$LIBLLAMA')" >&2; exit 66; }
if [ "$VOCAB_NAME" = v41 ]; then
  # The reference's `~` is a symbol, so `~/` is one word and one id; a tree whose `~` is in neither
  # P nor S returns [96, 17].
  probe=$(CUDA_VISIBLE_DEVICES= "$TOKENIZE" -m "$SHARD" -p '~/' --ids --log-disable)
  [ "$probe" = '[71520]' ] || {
    echo "oracle: $TOKENIZE tokenizes '~/' as $probe on $SHARD, not [71520]: its \`~\` is not a symbol" >&2
    exit 65
  }
fi

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

# row <name> — the text <name>'s manifest row: bytes, text md5 and the two id counts.
row() {
  printf '%s\t%s\t%s\t%s\t%s\n' "$1" "$(wc -c < "$OUT.new/$1.txt")" \
    "$(md5sum < "$OUT.new/$1.txt" | cut -d' ' -f1)" \
    "$(wc -l < "$OUT.new/$1.ids")" "$(wc -l < "$OUT.new/$1.nps.ids")"
}

{
  printf '# tokenizer\t%s\t%s\n' "$TOKENIZE" "$(md5sum < "$TOKENIZE" | cut -d' ' -f1)"
  printf '# libllama\t%s\t%s\n' "$LIBLLAMA" "$(md5sum < "$LIBLLAMA" | cut -d' ' -f1)"
  printf '# vocabulary\t%s\n' "$SHARD"
  printf '# text tree\t%s\n' "$IK"
  printf '# cases\t%s\t%s\n' "$CASES" "$(md5sum < "$CASES" | cut -d' ' -f1)"
  printf 'set\tbytes\ttext_md5\tids\tnps_ids\n'
} > "$OUT.new/MANIFEST.tsv"

for name in code prose prose-all threads korean; do
  gather "$name" > "$OUT.new/$name.txt"
  ids "$OUT.new/$name.txt" "$OUT.new/$name"
  row "$name" >> "$OUT.new/MANIFEST.tsv"
done

n=0
while IFS= read -r line || [ -n "$line" ]; do
  case $line in '#'*) continue ;; esac
  n=$((n + 1))
  name=$(printf 'cases/%02d' "$n")
  printf '%b' "$line" > "$OUT.new/$name.txt"
  ids "$OUT.new/$name.txt" "$OUT.new/$name"
  row "$name" >> "$OUT.new/MANIFEST.tsv"
done < "$CASES"
printf '# complete\t%s\n' "$n" >> "$OUT.new/MANIFEST.tsv"

rm -rf "$OUT"
mv "$OUT.new" "$OUT"
cat "$OUT/MANIFEST.tsv"
