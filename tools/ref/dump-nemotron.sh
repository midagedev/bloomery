#!/usr/bin/env bash
# Dump the last-position logit rows of the profile's prompts (ten for each), for the profile the
# command runs under: nemotron_h (Nano 4B) into
# $BLOOMERY_DATA/ref-nemotron-logits, nemotron_h_moe (Bolt 74B-A9B) into $BLOOMERY_DATA/ref-nemotron-moe-logits, lfm3 (LFM3-1.7B)
# into the set its profile's LOGITS_SETS names for the model file (tools/ref/dump-lfm3.sh runs it once a file). The script,
# its checks and the harness source are one for all; the profile picks the model file, the set, the binary, the
# architecture nemotron_ref must find in the file and the prompts.
# The prompts are 1 to 4096 ids: a few short ones, the counts on each side of a 256-row
# chunk (255, 256, 257), and two long ones (1024 prose, 4096 code).
# CPU only: CUDA_VISIBLE_DEVICES is emptied and nemotron_ref asks for no GPU layers.
# The prompts are prefixes of the tokenizer oracle's copies of the corpora the other
# families use (engram-corpus.sh): code.txt is IK src/*.cpp, prose.txt is IK docs and
# README. The Nano file sets add_bos_token, so llama-tokenize adds BOS and the counts include that id. The Bolt
# file does not (add_bos_token false): on the same text its ids are the Nano's without the leading BOS (id 1),
# so a Bolt prompt of n ids is the Nano's ids 2 to n+1.
# The lfm3 files set add_bos_token false, as the Bolt's does: a prompt is the text's own ids. Its text is the
# nemotron tokenizer oracle's, which no model owns.
# Mainline is not rebuilt.
# BLOOMERY_TIER=fixture dumps the profile's set on its fixture file (ref-paths.sh: the model is its first shard): the set
# is its real twin's name behind `fx_` (fixture_dump_tier) and carries the file's `# fixture` line (fixture_line), which
# nemotron_ref writes beside `# model`. The Bolt's fixture is fx_ref-nemotron-moe-logits; the Nano has no fixture tier.
set -euo pipefail
# shellcheck source=tools/ref/ref-paths.sh
source "$(dirname "${BASH_SOURCE[0]}")/ref-paths.sh"
case $MODEL_NAME in
  nemotron_h) SET_NAME=ref-nemotron-logits BIN_NAME=nemotron_ref ;;
  nemotron_h_moe) SET_NAME=ref-nemotron-moe-logits BIN_NAME=nemotron_ref_moe ;;
  lfm3)
    # The set of the model file, one entry of the profile's LOGITS_SETS (<set>:<file>:<bytes>:<sha256>).
    SET_NAME=
    BIN_NAME=nemotron_ref_lfm3
    for entry in "${LOGITS_SETS[@]}"; do
      IFS=: read -r set_of file_of _ <<< "$entry"
      [ "$file_of" != "$MODEL" ] || SET_NAME=$set_of
    done
    [ -n "$SET_NAME" ] || {
      echo "dump-nemotron: $MODEL is none of the lfm3 profile's LOGITS_SETS files; no set is named for it (exit 2)" >&2
      exit 2
    }
    ;;
  *)
    echo "dump-nemotron: this dump runs under the nemotron_h, nemotron_h_moe or lfm3 profile; this command picked $MODEL_NAME" >&2
    exit 2
    ;;
esac
# Every profile dumps the ten prompts at one context: the Nano's gate (gate_nemotron_e2e, CTX 4160) holds its
# prompt path on every one; lfm3's window (256 rows) is cut by the 255, 256 and 257 prompts.
CTX=4096
PROMPTS=(code5 code29 code4096 prose1 prose64 prose255 prose256 prose257 prose512 prose1024)
fixture_dump_tier dump-nemotron.sh
if [ -n "$SET_PREFIX" ] && [ "$MODEL_NAME" != nemotron_h_moe ]; then
  echo "dump-nemotron: BLOOMERY_TIER=fixture: $SET_NAME has no fixture twin (crates/refset names only the Bolt's, fx_ref-nemotron-moe-logits) (exit 66)" >&2
  exit 66
fi
SET_NAME=$SET_PREFIX$SET_NAME
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SRC=$HERE/tools/ref/nemotron_ref.cpp
IO=$HERE/tools/ref/ref_io.h
BIN=${NEMOTRON_BIN:-$BLOOMERY_DATA/bin/$BIN_NAME}
REC=$BIN.build
[ -x "$BIN" ] || { echo "dump-nemotron: no $BIN — run just build-ref-nemotron (build-ref-nemotron-moe under nemotron_h_moe, build-ref-lfm3 under lfm3)" >&2; exit 2; }
# cpp then the shared header, the same order build-nemotron-ref.sh records. A header-only edit is a mismatch.
want=$(cat "$SRC" "$IO" | sha256sum | cut -d' ' -f1)
got=$(awk '$1 == "source_sha256" { print $2; exit }' "$REC")
[ "$want" = "$got" ] || {
  echo "dump-nemotron: $BIN was linked from another nemotron_ref.cpp or ref_io.h — run just build-ref-nemotron (build-ref-nemotron-moe under nemotron_h_moe, build-ref-lfm3 under lfm3)" >&2
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
FIXTURE_LINE=$(fixture_line dump-nemotron.sh)
FIXTURE_ARGS=()
[ -z "$FIXTURE_LINE" ] || FIXTURE_ARGS=(--fixture-line "$FIXTURE_LINE")
export CUDA_VISIBLE_DEVICES=""

TEXT=$BLOOMERY_DATA/ref-tokenizer-nemotron
[ -f "$TEXT/code.txt" ] || { echo "dump-nemotron: no $TEXT/code.txt (the tokenizer oracle's code corpus)" >&2; exit 66; }
[ -f "$TEXT/prose.txt" ] || { echo "dump-nemotron: no $TEXT/prose.txt (the tokenizer oracle's prose corpus)" >&2; exit 66; }

# ids_of <text> <lines> <count> <out> — tokenize a line prefix. No --no-bos: the file's own add_bos_token decides.
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
ids_of "$TEXT/code.txt" 2000 4096 "$WORK/code.ids"
ids_of "$TEXT/prose.txt" 2000 1024 "$WORK/prose.ids"
for n in 5 29 4096; do head -n "$n" "$WORK/code.ids" > "$WORK/code$n.ids"; done
for n in 1 64 255 256 257 512 1024; do head -n "$n" "$WORK/prose.ids" > "$WORK/prose$n.ids"; done
ARGS=()
for name in "${PROMPTS[@]}"; do ARGS+=(--prompt "$name:${name##*[a-z]}:$WORK/$name.ids"); done

SET=$BLOOMERY_DATA/$SET_NAME
STAGE=$SET.staging
rm -rf "$STAGE"
mkdir -p "$STAGE"
trap 'drop_incomplete_stage; rm -rf "$WORK"' EXIT
"$BIN" -m "$MODEL" --arch "$MODEL_NAME" --out "$STAGE" -c "$CTX" -ub 512 -t 8 ${FIXTURE_ARGS[@]+"${FIXTURE_ARGS[@]}"} "${ARGS[@]}"
grep -q '^# complete' "$STAGE/MANIFEST.tsv" || {
  echo "dump-nemotron: the staged set has no completion trailer — not installing it" >&2
  exit 1
}
rows=$(grep -c '^prompt	' "$STAGE/MANIFEST.tsv")
[ "$rows" = "${#PROMPTS[@]}" ] || { echo "dump-nemotron: the staged set has $rows prompt rows, want ${#PROMPTS[@]}" >&2; exit 1; }
rm -rf "$SET"
mv "$STAGE" "$SET"
echo "dump-nemotron: installed $SET (${#PROMPTS[@]} prompts)"
