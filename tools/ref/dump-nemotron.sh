#!/usr/bin/env bash
# Dump four last-position logit rows into $BLOOMERY_DATA/ref-nemotron-logits.
# CPU only: CUDA_VISIBLE_DEVICES is emptied and nemotron_ref asks for no GPU layers.
# The prompts are prefixes of the tokenizer oracle's copies of the corpora the other
# families use (engram-corpus.sh): code.txt is IK src/*.cpp, prose.txt is IK docs and
# README. llama-tokenize adds BOS because the file sets add_bos_token; the counts
# include that id. Mainline is not rebuilt.
set -euo pipefail
# shellcheck source=tools/ref/ref-paths.sh
source "$(dirname "${BASH_SOURCE[0]}")/ref-paths.sh"
[ "$MODEL_NAME" = nemotron_h ] || {
  echo "dump-nemotron: this dump runs under the nemotron_h profile; this command picked $MODEL_NAME" >&2
  exit 2
}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SRC=$HERE/tools/ref/nemotron_ref.cpp
IO=$HERE/tools/ref/ref_io.h
BIN=${NEMOTRON_BIN:-$BLOOMERY_DATA/bin/nemotron_ref}
REC=$BIN.build
[ -x "$BIN" ] || { echo "dump-nemotron: no $BIN — run just build-ref-nemotron" >&2; exit 2; }
# cpp then the shared header, the same order build-nemotron-ref.sh records. A header-only edit is a mismatch.
want=$(cat "$SRC" "$IO" | sha256sum | cut -d' ' -f1)
got=$(awk '$1 == "source_sha256" { print $2; exit }' "$REC")
[ "$want" = "$got" ] || {
  echo "dump-nemotron: $BIN was linked from another nemotron_ref.cpp or ref_io.h — run just build-ref-nemotron" >&2
  exit 2
}
TOKENIZE=$LCPP/build/bin/llama-tokenize
[ -x "$TOKENIZE" ] || { echo "dump-nemotron: no llama-tokenize at $TOKENIZE" >&2; exit 66; }
LIBLLAMA=$(ldd "$TOKENIZE" | awk '$1 == "libllama.so" || $1 == "libllama.so.0" { print $3; exit }')
[ -f "$LIBLLAMA" ] || { echo "dump-nemotron: $TOKENIZE loads no libllama.so (ldd names '$LIBLLAMA')" >&2; exit 66; }
TOK_MD5=$(md5sum < "$TOKENIZE" | cut -d' ' -f1)
LIB_MD5=$(md5sum < "$LIBLLAMA" | cut -d' ' -f1)
# The pin literals live in the build script, which wrote this record. A live
# binary that does not match it is not the one nemotron_ref was linked against.
want_build=$(awk '$1 == "build" { $1 = ""; sub(/^ /, ""); print; exit }' "$REC")
[ "llama-tokenize $TOK_MD5 libllama.so $LIB_MD5" = "$want_build" ] || {
  echo "dump-nemotron: $TOKENIZE or $LIBLLAMA differs from the build $BIN was linked against ($want_build)" >&2
  exit 2
}
export BLOOMERY_REF_BUILD="$want_build"
export CUDA_VISIBLE_DEVICES=""

TEXT=$BLOOMERY_DATA/ref-tokenizer-nemotron
[ -f "$TEXT/code.txt" ] || { echo "dump-nemotron: no $TEXT/code.txt (the tokenizer oracle's code corpus)" >&2; exit 66; }
[ -f "$TEXT/prose.txt" ] || { echo "dump-nemotron: no $TEXT/prose.txt (the tokenizer oracle's prose corpus)" >&2; exit 66; }

# ids_of <text> <lines> <count> <out> — tokenize a line prefix. No --no-bos: the file adds BOS.
ids_of() {
  local text=$1 lines=$2 count=$3 out=$4 raw got
  raw=$(mktemp)
  head -n "$lines" "$text" > "$raw.in"
  "$TOKENIZE" -m "$MODEL" -f "$raw.in" --ids --log-disable --no-escape > "$raw.bracket"
  { tr -d '[] ' < "$raw.bracket" | tr ',' '\n' | grep -v '^$' || true; } > "$raw"
  got=$(wc -l < "$raw")
  if [ "$got" -lt "$count" ]; then
    echo "dump-nemotron: the first $lines lines of $text tokenize to $got ids, need $count" >&2
    rm -f "$raw" "$raw.in" "$raw.bracket"
    exit 2
  fi
  head -n "$count" "$raw" > "$out"
  rm -f "$raw" "$raw.in" "$raw.bracket"
}

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
ids_of "$TEXT/code.txt" 80 29 "$WORK/code.ids"
ids_of "$TEXT/prose.txt" 400 512 "$WORK/prose.ids"
head -n 5 "$WORK/code.ids" > "$WORK/code5.ids"
head -n 29 "$WORK/code.ids" > "$WORK/code29.ids"
head -n 64 "$WORK/prose.ids" > "$WORK/prose64.ids"
cp "$WORK/prose.ids" "$WORK/prose512.ids"

SET=$BLOOMERY_DATA/ref-nemotron-logits
STAGE=$BLOOMERY_DATA/ref-nemotron-logits.staging
rm -rf "$STAGE"
mkdir -p "$STAGE"
trap 'drop_incomplete_stage; rm -rf "$WORK"' EXIT
"$BIN" -m "$MODEL" --out "$STAGE" -c 512 -ub 512 -t 8 \
  --prompt "code5:5:$WORK/code5.ids" \
  --prompt "code29:29:$WORK/code29.ids" \
  --prompt "prose64:64:$WORK/prose64.ids" \
  --prompt "prose512:512:$WORK/prose512.ids"
grep -q '^# complete' "$STAGE/MANIFEST.tsv" || {
  echo "dump-nemotron: the staged set has no completion trailer — not installing it" >&2
  exit 1
}
rows=$(grep -c '^prompt	' "$STAGE/MANIFEST.tsv")
[ "$rows" = 4 ] || { echo "dump-nemotron: the staged set has $rows prompt rows, want 4" >&2; exit 1; }
rm -rf "$SET"
mv "$STAGE" "$SET"
echo "dump-nemotron: installed $SET (4 prompts)"
