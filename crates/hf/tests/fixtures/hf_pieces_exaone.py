"""The pre-tokenizer words of Hugging Face's `tokenizers` for K-EXAONE-2.0-750B-A37B's tokenizer.json
(crates/tokenizer, `exaone_moe_hf_pieces_are_hugging_faces`).

usage (Python 3 with tokenizers==0.21.0 from PyPI):
    python3 hf_pieces_exaone.py <folder> | gzip -9n > exaone-moe-hf-pieces.json.gz

<folder> holds tokenizer.json of LG's K-EXAONE-2.0-750B-A37B (the hub folder /root/agents/specs/exaone/hf, whose
sha256 the header carries). The texts are hf_pieces_qwen2.py's corpus plus the exaone splitter's own cases (EXAONE
below: the letter word joined by single spaces with Hangul and Han in it, marks after letters and alone, contractions
of both ASCII cases and of the U+017F fold, one `\r`, `\n` or `/` behind a symbol run, the codepoints Unicode 15.0
added inside a word), each case a text and the byte length of each piece `pre_tokenizer.pre_tokenize_str` gives.
LG's added tokens hold the runs of spaces, tabs and newlines, so every text that spells one is dropped (as
hf_pieces_qwen2.write_pieces drops it): the pipeline cuts an added token out before its pre-tokenizer runs.
"""
import json
import sys
from pathlib import Path

import hf_corpus
import hf_encode
import hf_encode_minimax as minimax

# The exaone splitter's own cases beyond the qwen2 corpus. None spells an added token: no double
# space, no double tab, no double newline.
EXAONE = [
    # The letter word joined by single spaces, Hangul and Han letters of it (no CJK alternative).
    "hello world foo", "\ud55c\uad6d\uc5b4 \ud14d\uc2a4\ud2b8", "\u4e2d\u6587 abc", "\u4e2d\u6587\ud55c\uad6d\uc5b4 english", "abc\u4e2d\u6587 def",
    "Hello,\uc138\uacc4! world", "\ud55c\uae00123abc", "\u4e2d\u6587123", "1\ud55c\uae00", "\uac00\ub098\ub2e4 \ub77c\ub9c8\ubc14 \uc0ac\uc544\uc790", "\u4e2d\u3000\u6587 x",
    # Marks after letters, before them and alone; a mark after a space ends the word.
    "e\u0301 x y", "a\u0301b c", "\u0301a", "a \u0301b", "x\u0301\u0308y", "\u0301\u0302",
    "\u00e9\u0301cole \u00e8\u0308tude", "\ud55c\u0301\uae00", "\u4e2d\u0301\u6587 abc", "\U0001d165x", "x\U0001d165y",
    # Contractions of both ASCII cases and of the U+017F fold, in a word and at its ends.
    "IT'S WE'RE I'M YOU'LL HE'D", "it's we're i'm you'll he'd", "rock'n'roll",
    "x'\u017f y", "x'\u017ft y", "a'S b'd c'M", "'ll 'LL 'lL 'D", "Won't", "DON'T", "y'ALL", "that's",
    # One `\r`, `\n` or `/` behind a symbol run; the `\s*[\r\n]` whitespace alternative.
    "x/\n/y", "x//y", "a.\n", "a \r\n b", "a\r\n\r\nb", "\r\n\r\nx", "a\rb", "end\r\n",
    "x.\r\ny", "tab\tnewline\nx", "?!\nx", "/slash/", "a/b/c", "x=1;\ny=2;", "<tag>\n</tag>",
    # The word a codepoint Unicode 15.0 added would join under the reference's classes.
    "x\U00013440y", "a\U00011f04b", "\U0001e4d0\U0001e4f0", "x \U0001e030y", "a\U0001e08fy",
    # The optional character before a letter word.
    '"hello world', "(a b) c", "\u00ab\u4e2d\u6587\u00bb \ud55c\uae00", "1.5 x", "v1.2 x", "3rd place", "42km run",
]
EXAONE += [f"word'{w}" for w in
           ["s", "S", "t", "T", "re", "RE", "rE", "Re", "ve", "VE", "vE", "Ve", "m", "M",
            "ll", "LL", "lL", "Ll", "d", "D", "\u017f", "\u017ft", "n", "x", "st"]]
EXAONE += [f"it'{w}y" for w in
           ["s", "S", "t", "T", "re", "RE", "ve", "VE", "m", "M", "ll", "LL", "d", "D", "\u017f"]]


def main():
    folder = Path(sys.argv[1])
    data, tok = hf_corpus.read_tokenizer(folder)
    spellings = [a["content"] for a in json.loads(data)["added_tokens"]]
    texts = list(dict.fromkeys(hf_encode.corpus() + minimax.case_texts() + EXAONE))
    texts = [t for t in texts if not any(s in t for s in spellings)]
    cases = ({"text": t, "pieces": [len(p) for p, _ in tok.pre_tokenizer.pre_tokenize_str(t)]} for t in texts)
    hf_corpus.emit(data, "LG's K-EXAONE-2.0-750B-A37B (the hub folder /root/agents/specs/exaone/hf)/tokenizer.json",
                   cases)


if __name__ == "__main__":
    main()
