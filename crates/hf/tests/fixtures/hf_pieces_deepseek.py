"""The pre-tokenizer words of Hugging Face's `tokenizers` for DeepSeek-V4.1-Flash-1017's tokenizer.json
(read by deepseek_v3_hf_pieces_are_hugging_faces in crates/tokenizer).

usage (Python 3 with tokenizers==0.21.0 from PyPI):
    python3 hf_pieces_deepseek.py <folder> | gzip -9n > deepseek-v3-hf-pieces.json.gz

<folder> holds tokenizer.json of deepseek-ai/DeepSeek-V4.1-Flash-1017 at REV. The corpus is hf_pieces_qwen2.py's
(`write_pieces`): hf_encode.py's corpus and hf_encode_minimax.py's letter-category pairs but those that spell an added
token, each case a text and the byte length of each piece `pre_tokenizer.pre_tokenize_str` gives.
"""
import sys
from pathlib import Path

from hf_pieces_qwen2 import write_pieces

REV = "1bae0e1088993b9b82de08f59de5be5a34e42170"


def main():
    write_pieces(Path(sys.argv[1]), f"deepseek-ai/DeepSeek-V4.1-Flash-1017@{REV}/tokenizer.json")


if __name__ == "__main__":
    main()
