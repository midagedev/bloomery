#!/usr/bin/env bash
# engram-corpus.sh — token streams for `engram-reuse`, tokenized by the reference engine.
#
# Not a measurement: it takes no lease, times nothing and touches no GPU. `llama-tokenize`
# loads the vocabulary only (0.65 s against the split set, measured), so the 444 GiB of
# tensors is never read.
#
# There is no large text corpus on this box, so the sets below are what it does have. Each is
# reported separately because their reuse is the question — code repeats itself and prose does
# not, and mixing them would average the answer away.
#
#   code       the reference engine's own C++ sources          ($IK/src/*.cpp)
#   prose      the reference engine's docs and README          ($IK/docs/**/*.md, $IK/README.md)
#   prose-all  every Markdown file in that tree. Measured: 626 of its 695 files are under
#              github-data/ — scraped issue, pull-request and discussion threads, so it is
#              conversational text interleaved with pasted logs, benchmark tables and code
#              blocks, not clean prose. It is the long-stream witness (millions of tokens),
#              not a prose measurement; its repetition is an upper bound, because near-identical
#              build instructions and log formats recur across threads.
#   threads    those github-data/ threads alone, in byte order — the conversational set
#   korean     this repository's own Korean prose: every docs/**/*.md at least 20 % Hangul by
#              character, in byte order — the one non-English set on the box
#
# The two newer sets sort under LC_ALL=C, so their file order does not depend on the box's
# locale; the first three keep the locale sort their existing .ids files were built with.
# `korean` reads the synced tree, so its text is the commit box.sh last synced — record the
# commit next to the .ids md5 when a number rests on it.
#
# Output: $BLOOMERY_DATA/engram/corpus-<name>.ids, one decimal token id per line. Text, not
# packed u32, so the file greps, diffs and truncates like everything else under $BLOOMERY_DATA.
#
# Usage: bash tools/ref/engram-corpus.sh [code|prose|prose-all|threads|korean]...
#        (default: the first three)
set -euo pipefail

IK=${IK:-/home/user/ik_llama.cpp}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TOKENIZE=${TOKENIZE:-$IK/build/bin/llama-tokenize}
DATA=${BLOOMERY_DATA:-/root/bloomery-data}
V41_DIR=${BLOOMERY_V41_DIR:-/models/DeepSeek-V4.1-Flash-Q3_K_M-engramQ8-tokembdBF16-attnQ8}

[ -x "$TOKENIZE" ] || { echo "engram-corpus: no llama-tokenize at $TOKENIZE" >&2; exit 66; }

# The split set's first shard: llama.cpp follows split.count from there, and with vocab_only
# it never opens the others.
SHARD=$(find "$V41_DIR" -maxdepth 1 -name '*-00001-of-*.gguf' | sort | head -1)
[ -n "$SHARD" ] || { echo "engram-corpus: no first shard under $V41_DIR" >&2; exit 66; }

OUT=$DATA/engram
mkdir -p "$OUT"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# gather <name> — concatenate that set's files into $TMP/<name>.txt
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
    if sum("\uac00" <= c <= "\ud7a3" for c in t) >= 0.2 * max(1, len(t)):
        print(path)
' | xargs -d '\n' cat -- ;;
    *)
      echo "engram-corpus: unknown set '$1' (code, prose, prose-all, threads, korean)" >&2; exit 64 ;;
  esac
}

printf '| set | source bytes | tokens | bytes/token | ids file |\n'
printf '|---|---:|---:|---:|---|\n'

sets=("$@")
[ ${#sets[@]} -gt 0 ] || sets=(code prose prose-all)

for name in "${sets[@]}"; do
  src=$TMP/$name.txt
  gather "$name" > "$src"
  bytes=$(wc -c < "$src")
  # --no-parse-special: a serving stream does not turn a literal "<|end▁of▁sentence|>" inside a
  # source file into a control token, and this text is full of them.
  "$TOKENIZE" -m "$SHARD" -f "$src" --ids --log-disable --no-parse-special > "$TMP/$name.raw"
  tr -d '[] ' < "$TMP/$name.raw" | tr ',' '\n' | grep -v '^$' > "$OUT/corpus-$name.ids"
  tokens=$(wc -l < "$OUT/corpus-$name.ids")
  printf '| %s | %s | %s | %.2f | %s |\n' \
    "$name" "$bytes" "$tokens" "$(echo "$bytes $tokens" | awk '{print $1/$2}')" \
    "$OUT/corpus-$name.ids"
done
