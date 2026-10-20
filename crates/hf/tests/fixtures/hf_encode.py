"""The Hugging Face encode corpus for crates/model-hf/src/bin/prepare.rs's served-tokenizer test.

usage (Python 3 with tokenizers==0.21.0 from PyPI):
    python3 hf_encode.py <folder> | gzip -9n > qwen3.8-hf-encode.json.gz

<folder> holds tokenizer.json of nvidia/Qwen3.8-Flash-Next-NVFP4 at REV (FP4-Local at b9ce18a6 ships the same
file). Each case is a text and the ids `Tokenizer.encode(text, add_special_tokens=False)` gives. The texts,
after the tokenizer gate's own cases that are UTF-8, reach every part of the pipeline the importer accepts:
added tokens before NFC; codepoints Unicode 15.0 and 15.1 added; NFC's composition and mark reordering, with
characters later than the Unicode 9.0 its tables hold; every alternative of the qwen35 regex (contractions in
every case HF's Unicode folding matches and in folds it does not, letter runs with marks, single numbers,
punctuation runs, each Unicode space before letters, newlines and the end); byte-level BPE on Korean, code,
emoji and long digit runs; and one codepoint of every 211 assigned ones (private use left out) in six
contexts.
"""
import codecs
import hashlib
import json
import sys
import unicodedata
from pathlib import Path

import tokenizers
from tokenizers import Tokenizer

REV = "fc694b54fb0174e0913e6adf86691ef85a4ead47"

APOSTROPHES = ["'", "\u2019"]
# Each contraction's letters in ASCII cases, U+017F (folds to s), and look-alikes that fold to none of them.
CONTRACTIONS = ["s", "S", "\u017f", "t", "T", "re", "RE", "rE", "ve", "Ve", "m", "M", "ll", "lL", "d", "D",
                "\u00df", "\u212a", "\ufb05", "\ufb06", "\u0130", "\u0131", "\uff33", "\u0455", "st", "\u017ft"]
SPACES = ["\t", "\n", "\x0b", "\x0c", "\r", " ", "\x1c", "\x1d", "\x1e", "\x1f", "\x85", "\xa0", "\u1680",
          "\u180e", "\u2000", "\u2003", "\u2007", "\u200a", "\u200b", "\u2028", "\u2029", "\u202f", "\u205f",
          "\u2060", "\u3000", "\ufeff"]
MARKS = ["\u0301", "\u0308", "\u0327", "\u0323", "\u031b", "\u20dd", "\ufe0f", "\u200d", "\u0e48", "\u094d",
         "\u302a", "\u05b4", "\u0670", "\u1ab0", "\U0001d165"]
BASES = ["a", "e", "o", "A", ".", " ", "1", "'", "\u3131", "\ud55c", "\u0915", "\u0e01", "", "\n"]
NUMBERS = ["0", "7", "12", "123", "1234", "1234567890", "3.14159265358979", "1,000,000", "-42", "1e-10",
           "0x7fffffff", "2026-10-20", "+82 10-1234-5678", "\u0661\u0662\u0663", "\u0967\u0968\u0969",
           "\uff11\uff12\uff13", "\u00b2\u00b3", "\u2160\u216b", "\u00bd\u2153", "\u2460\u2461", "\U0001d7ce\U0001d7cf",
           "9" * 64, "1 2 3 4", "v1.2.3-rc4"]
TEXTS = [
    "hello\n\nworld", "end ", "대한민국의 수도는 서울이고, 인구는 약 5,100만 명입니다.",
    "\u1112\u1161\u11ab\u1100\u116e\u11a8\u110b\u1165", "ㄱㅏㄴ ㅎㅏㄴ", "한글123abc!?", "까요? 그런데요... 「인용」",
    "\u1100\u1161\u11a8 가\u11a8",
    "ש\u05b8\u05c1לו\u05b9ם สว\u0e31สด\u0e35 नमस\u094dत\u0947", "你好，世界！这是一个测试。", "∀x∈ℝ: x² ≥ 0 ⇒ √x",
    "€100 £5 ¥300 ₩₩1000 $9.99",
    "\U0001f1f0\U0001f1f7\U0001f1fa\U0001f1f8 1\ufe0f\u20e3 #\ufe0f\u20e3 \U0001f3f4\U000e0067\U000e0062\U000e0073\U000e0063\U000e0074\U000e007f",
    "\U0001f9d1\U0001f3fb\u200d\U0001f4bb\U0001f600\U0001f600\U0001f600", "☺☺\ufe0e ©\ufe0f",
    "def f(x):\n    return x ** 2  # square\n", "fn main() {\n\tprintln!(\"{}\", 1_000);\n}\n",
    "const a = [1, 2, 3].map((x) => x * 2);", "if (a != b && c <= d) { return -1; }",
    "SELECT * FROM t WHERE id = '42';", "for i in $(seq 1 10); do echo \"$i\"; done",
    "{\"key\": [1, 2.5, null, true], \"k\\u00e9y\": \"v\"}", "<div class=\"x\">a &amp; b</div>",
    "^(?:[a-z0-9._%+-]+)@[a-z]+\\.[a-z]{2,}$", "    \n\t\t\n  x", "x\u00a0=\u00a0y;\u3000z",
    "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n", "a<|im_end|>b<|endoftext|>c",
    "<|im_start|><|im_start|>", "<|im_", "<|endoftext|", "text <|im_end|> text", "<think>\n\n</think>\n\n",
    "\u0000\u0001\u007f\u0080\u009f", "\ue000\uf8ff\U000f0000", "\ufdd0\ufffe\uffff", "��",
    "a\u0300\u0301\u0302\u0303\u0304\u0305\u0306\u0307\u0308\u0309", "Z\u0351\u0352\u0353\u0354\u0355",
    "\u212b \u2126 \u212a Å Ω K", "ẛ\u0323 q\u0307\u0323 ḋ\u0323", "가\u0301 \u1100\u1161\u0301", "क\u093c ऩ ড\u09bc",
    "\U000110ab \U00011099\U000110ba",
    # Letters, marks and numbers Unicode 15.0 and 15.1 added, which HF's regex (Unicode 14.0) has no class for.
    " \U0001d2c0\U0001d2c1 x", "x\U00011f04y \U00011f50", "a \U0001e4f0\U0001e4d0", "\u0cf3 \u0ece",
    "\U00031350\U0002ebf0x", "\U00013440 \U0001e030",
    # Composed or reordered under later Unicode than the 9.0 tables HF normalizes with.
    "\U00011935\U00011930", "a\u0305\u0eba", "x\u0301\u1df9", "\U00016d63\U00016d67", "\U000113c2\U000113c2",
]


def cases_txt():
    """The tokenizer gate's hand-picked strings (crates/tokenizer/tests/cases.txt), each line decoded as
    `printf %b` decodes it; HF takes only the ones that are UTF-8."""
    path = Path(__file__).resolve().parents[3] / "tokenizer/tests/cases.txt"
    out = []
    for line in path.read_bytes().split(b"\n"):
        if line.startswith(b"#"):
            continue
        try:
            out.append(codecs.escape_decode(line)[0].decode("utf-8"))
        except UnicodeDecodeError:
            pass
    return out


def corpus():
    out = cases_txt()
    add = out.append
    for a in APOSTROPHES:
        for c in CONTRACTIONS:
            for pre, post in [("", ""), ("it", ""), ("x", "x"), (" ", " "), ("1", "1"), ("", "\u0301"),
                              ("we", "s"), ("\u00e9", ".")]:
                add(f"{pre}{a}{c}{post}")
    for w in SPACES:
        for t in [f"a{w}b", f"{w}{w}a", f"a{w}", f"a{w}\n", f"{w}\n{w}", f"a {w}b", f"1{w}{w}2", f".{w}x"]:
            add(t)
    for b in BASES:
        for m in MARKS:
            add(f"{b}{m}x")
        add(f"{b}\u0301\u0327")
        add(f"{b}\u0327\u0301")
        add(f"{b}\u0308\u0323\u031b")
    for n in NUMBERS:
        for t in [n, f"x{n}y", f" {n} ", f"\ud55c{n}"]:
            add(t)
    out.extend(TEXTS)
    assigned = [c for c in range(0x110000) if unicodedata.category(chr(c)) not in ("Cn", "Co", "Cs")]
    for c in assigned[::211]:
        ch = chr(c)
        add(f"x{ch}y {ch}1'{ch}.{ch}{ch} \n{ch}")
    composed = [c for c in range(0x110000)
                if not 0xD800 <= c < 0xE000 and unicodedata.decomposition(chr(c))
                and not unicodedata.decomposition(chr(c)).startswith("<")]
    for c in composed[::23]:
        add("a" + unicodedata.normalize("NFD", chr(c)) + "b")
    return list(dict.fromkeys(out))


def main():
    folder = sys.argv[1]
    data = open(f"{folder}/tokenizer.json", "rb").read()
    tok = Tokenizer.from_str(data.decode("utf-8"))
    head = {
        "tokenizers": tokenizers.__version__,
        "unicode": unicodedata.unidata_version,
        "source": f"nvidia/Qwen3.8-Flash-Next-NVFP4@{REV}/tokenizer.json",
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    cases = [json.dumps({"text": t, "ids": tok.encode(t, add_special_tokens=False).ids}, ensure_ascii=False)
             for t in corpus()]
    sys.stdout.write(json.dumps(head, ensure_ascii=False)[:-1] + ', "cases": [\n' + ",\n".join(cases) + "\n]}\n")


if __name__ == "__main__":
    main()
