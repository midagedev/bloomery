"""The pre-tokenizer words of Hugging Face's `tokenizers` for Qwen3-30B-A3B-Instruct-2507's tokenizer.json
(crates/tokenizer, `qwen2_hf_pieces_are_hugging_faces`).

usage (Python 3 with tokenizers==0.21.0 from PyPI):
    python3 hf_pieces_qwen2.py <folder> | gzip -9n > qwen2-hf-pieces.json.gz

<folder> holds tokenizer.json of mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit at REV (the Qwen team's file, unchanged).
The texts are hf_encode.py's corpus (contractions in every case and fold, letters with marks, spaces, numbers, Korean,
code, emoji, codepoints of every Unicode version) and hf_encode_minimax.py's letter-category pairs, but those that spell
an added token (which `tokenizers` cuts out before its pre-tokenizer runs). Each case is a text and the byte length of
each piece `pre_tokenizer.pre_tokenize_str` gives, which is the number of ByteLevel characters of the piece. Ids do not
show a split that no merge crosses; the pieces do.
"""
import json
import sys
from pathlib import Path

import hf_corpus
import hf_encode
import hf_encode_minimax as minimax

REV = "e9675aa3ca5f900ccef55267914466d55ab325fa"


def main():
    data, tok = hf_corpus.read_tokenizer(Path(sys.argv[1]))
    spellings = [a["content"] for a in json.loads(data)["added_tokens"]]
    texts = list(dict.fromkeys(hf_encode.corpus() + minimax.case_texts()))
    texts = [t for t in texts if not any(s in t for s in spellings)]
    cases = ({"text": t, "pieces": [len(p) for p, _ in tok.pre_tokenizer.pre_tokenize_str(t)]} for t in texts)
    hf_corpus.emit(data, f"mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit@{REV}/tokenizer.json", cases)


if __name__ == "__main__":
    main()
