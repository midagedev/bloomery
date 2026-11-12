#!/usr/bin/env bash
# exaone-oracle.sh — the reference ids of K-EXAONE-2.0's vocabulary for tests/exaone.rs, on stdout
# as one of two files over the cases of tests/cases.txt and tests/cases-exaone.txt (printf %b
# escapes), the text's bytes in hex first:
#   - default: tests/exaone-ids.tsv, one row per case whose text is already NFC — llama-tokenize's
#     ids with special tokens parsed (its default) and with --no-parse-special;
#   - --hf: tests/exaone-nfc-ids.tsv, one row per case whose text is not (decomposed Hangul,
#     combining accents) — the Hugging Face tokenizer's ids in both columns (those rows hold no
#     special token, so the modes are one). The engine NFC-normalizes before this pre-tokenizer
#     because the model's tokenizer.json does; the reference normalizes nothing on this path
#     (tokenizer.ggml.normalizer.nfc is read at load and applied by its UGM tokenizer alone), so
#     the reference's own ids of a non-NFC text are not what the engine is held to.
#
# The reference is mainline llama.cpp built on the box (its `exaone-moe` pre-tokenizer runs through
# the std::regex fallback of src/unicode.cpp; no hand-written splitter knows it): LCPP names the
# tree, BUILD its build dir, MODEL the vocab-only GGUF convert_hf_to_gguf.py writes from the hub
# folder (the tokenizer, no tensors; the same file as refset::arch::exaone_moe::VOCABULARY). The
# binary's own --version must name the tree's HEAD, so the pins come from the commit the header
# records. The --hf side reads the hub tokenizer.json itself (HFTOK) with the `tokenizers` of a
# venv (HFVENV). No lease, no card (CUDA_VISIBLE_DEVICES is emptied; llama-tokenize loads the
# vocabulary only).
#
#   bash crates/tokenizer/tools/exaone-oracle.sh > crates/tokenizer/tests/exaone-ids.tsv
#   bash crates/tokenizer/tools/exaone-oracle.sh --hf > crates/tokenizer/tests/exaone-nfc-ids.tsv
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TESTS=$HERE/../tests
LCPP=${LCPP:-/root/agents/specs/exaone/work/llama.cpp}
BUILD=${BUILD:-/root/agents/specs/exaone-r3/lcpp-build}
MODEL=${MODEL:-/data/models/K-EXAONE-2.0-vocab/exaone-moe-vocab.gguf}
HFVENV=${HFVENV:-/root/agents/specs/exaone-r3/venv}
HFTOK=${HFTOK:-/root/agents/specs/exaone/hf/tokenizer.json}
TOKENIZE=$BUILD/bin/llama-tokenize
[ -x "$TOKENIZE" ] || { echo "exaone-oracle: no $TOKENIZE" >&2; exit 66; }
[ -f "$MODEL" ] || { echo "exaone-oracle: no vocabulary file at $MODEL" >&2; exit 66; }
[ -f "$HFTOK" ] || { echo "exaone-oracle: no $HFTOK" >&2; exit 66; }
commit=$(git -c safe.directory='*' -C "$LCPP" rev-parse HEAD)
dirty=$(git -c safe.directory='*' -C "$LCPP" status --porcelain --untracked-files=no | wc -l | tr -d ' ')
[ "$dirty" = 0 ] || { echo "exaone-oracle: the tree $LCPP has $dirty tracked change(s)" >&2; exit 65; }
version=$(CUDA_VISIBLE_DEVICES= "$TOKENIZE" --version 2>&1 | sed -n 1p)
built=$(printf '%s' "$version" | sed -n 's/.*commit \([0-9a-f]\{7,\}\).*/\1/p')
case $commit in
  "$built"*) [ -n "$built" ] ;;
  *) false ;;
esac || { echo "exaone-oracle: the binary reports '$version', the tree is at $commit" >&2; exit 65; }
case ${1:-} in
  --hf) [ -x "$HFVENV/bin/python" ] || { echo "exaone-oracle: no $HFVENV/bin/python for the --hf side" >&2; exit 66; } ;;
  "") ;;
  *) echo "exaone-oracle: unknown argument '$1' (the one flag is --hf)" >&2; exit 64 ;;
esac

tmp=$(mktemp)
trap 'rm -f "$tmp" "$tmp.raw"' EXIT

# ids <flag...> — llama-tokenize --ids of $tmp, a space-separated list.
ids() {
  CUDA_VISIBLE_DEVICES= "$TOKENIZE" -m "$MODEL" -f "$tmp" --ids --log-disable --no-bos --no-escape "$@" > "$tmp.raw"
  tr -d '[] ' < "$tmp.raw" | tr ',' ' ' | sed 's/ *$//'
}

# hf-ids — the Hugging Face tokenizer's ids of $tmp (add_special_tokens off: the header names the
# special tokens, the template is not the tokenizer's to add), a space-separated list.
hf-ids() {
  "$HFVENV/bin/python" - "$HFTOK" "$tmp" <<'PY'
import sys
from tokenizers import Tokenizer
tok = Tokenizer.from_file(sys.argv[1])
with open(sys.argv[2], "rb") as f:
    ids = tok.encode(f.read().decode("utf-8", "replace"), add_special_tokens=False).ids
print(" ".join(str(i) for i in ids))
PY
}

# is-nfc — 0 when $tmp's text is NFC (a byte no decoder reads is one U+FFFD, which NFC keeps).
is-nfc() {
  python3 - "$tmp" <<'PY'
import sys, unicodedata
with open(sys.argv[1], "rb") as f:
    sys.exit(0 if unicodedata.is_normalized("NFC", f.read().decode("utf-8", "replace")) else 1)
PY
}

if [ "${1:-}" = "--hf" ]; then
  printf '# reference\tHugging Face tokenizers (the hub tokenizer.json, its NFC normalizer) over %s\n' "$HFTOK"
  printf '# vocabulary\t%s\n' "$MODEL"
  printf '# columns\ttext bytes in hex, ids (NFC applied; both modes — no special token in these rows), ids (the same)\n'
else
  printf '# reference\tllama.cpp (mainline) at %s, %s\n' "$commit" "$version"
  printf '# vocabulary\t%s\n' "$MODEL"
  printf '# columns\ttext bytes in hex, ids (special tokens parsed), ids (--no-parse-special)\n'
fi
n=0
for cases in "$TESTS/cases.txt" "$TESTS/cases-exaone.txt"; do
  while IFS= read -r line || [ -n "$line" ]; do
    case $line in '#'*) continue ;; esac
    printf '%b' "$line" > "$tmp"
    if is-nfc; then want=""; else want="--hf"; fi
    [ "$want" = "${1:-}" ] || continue
    n=$((n + 1))
    hex=$(od -An -v -tx1 "$tmp" | tr -d ' \n')
    if [ "${1:-}" = "--hf" ]; then
      printf '%s\t%s\t%s\n' "$hex" "$(hf-ids)" "$(hf-ids)"
    else
      printf '%s\t%s\t%s\n' "$hex" "$(ids)" "$(ids --no-parse-special)"
    fi
  done < "$cases"
done
printf '# complete\t%s\n' "$n"
