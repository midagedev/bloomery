#!/usr/bin/env python3
"""Write the ERNIE 5.5 vocabulary's pins from Hugging Face's `tokenizers`.

    ernie55_pins.py <folder> <cases-ernie55.txt> <ids.tsv> <pieces.tsv>
    ernie55_pins.py --self-test

`ids.tsv` rows are `<text bytes in hex>\t<ids>\t<ids>` (the folder's `tokenizer.json`, special tokens
not added; no case holds a special token, so both parse modes hold one list) and `pieces.tsv` rows
`<text bytes in hex>\t<each pre-tokenizer piece's bytes in hex, space-separated>`, both closed by a
`# complete` trailer. A case is one line of the cases file; a line starting `# ` is a comment.

`--self-test` runs the case decoder, the byte-level alphabet and the writer on stand-in tokenizers
(no `tokenizers` package, no folder).
"""
import os
import re
import sys
import tempfile


def decode(line):
    def sub(m):
        s = m.group(0)
        c = s[1]
        if c == "n":
            return "\n"
        if c == "t":
            return "\t"
        if c == "r":
            return "\r"
        if c == "\\":
            return "\\"
        return chr(int(s[2:], 16))

    return re.sub(r"\\(?:[ntr\\]|x[0-9A-Fa-f]{2}|u[0-9A-Fa-f]{4}|U[0-9A-Fa-f]{8})", sub, line)


def byte_level_table():
    """GPT-2's byte-level alphabet: the character each ByteLevel piece spells, mapped to its byte."""
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def write(tok, version, cases, ids_out, pieces_out):
    """Both pin files of `cases` (a path) through `tok`; the number of cases."""
    back = byte_level_table()
    head = (
        f"# reference\tHugging Face tokenizers {version}, Tokenizer.from_file(<folder>/tokenizer.json)\n"
        "# vocabulary\tERNIE-5.5-Flash-Preview's tokenizer.json (pre-tokenizer ernie5)\n"
    )
    ids_rows, piece_rows = [], []
    for line in open(cases, encoding="utf-8").read().split("\n"):
        if not line or line.startswith("# "):
            continue
        text = decode(line)
        raw = text.encode("utf-8")
        ids = tok.encode(text, add_special_tokens=False).ids
        cols = " ".join(map(str, ids))
        ids_rows.append(f"{raw.hex()}\t{cols}\t{cols}")
        pieces = []
        for piece, _ in tok.pre_tokenizer.pre_tokenize_str(text):
            pieces.append(bytes(back[ch] for ch in piece).hex())
        assert b"".join(bytes.fromhex(p) for p in pieces) == raw, text
        piece_rows.append(f"{raw.hex()}\t{' '.join(pieces)}")
    with open(ids_out, "w") as f:
        f.write(head + "# columns\ttext bytes in hex, ids (special tokens parsed), ids (--no-parse-special)\n")
        f.write("\n".join(ids_rows) + f"\n# complete\t{len(ids_rows)}\n")
    with open(pieces_out, "w") as f:
        f.write(head + "# columns\ttext bytes in hex, the pre-tokenizer's pieces in hex\n")
        f.write("\n".join(piece_rows) + f"\n# complete\t{len(piece_rows)}\n")
    return len(ids_rows)


def main(folder, cases, ids_out, pieces_out):
    import tokenizers
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(f"{folder}/tokenizer.json")
    print(write(tok, tokenizers.__version__, cases, ids_out, pieces_out), "cases")


class _Encoding:
    def __init__(self, ids):
        self.ids = ids


class _Stand:
    """A tokenizer that numbers each byte of the text and cuts the text at spaces, the pieces spelled
    in the byte-level alphabet (a space is U+0120)."""

    def __init__(self):
        spell = {b: c for c, b in byte_level_table().items()}
        self.pre_tokenizer = self
        self.spell = spell

    def encode(self, text, add_special_tokens):
        assert add_special_tokens is False
        return _Encoding(list(text.encode("utf-8")))

    def pre_tokenize_str(self, text):
        pieces, cur = [], b""
        for b in text.encode("utf-8"):
            if b == 0x20 and cur:
                pieces.append(cur)
                cur = b""
            cur += bytes([b])
        if cur:
            pieces.append(cur)
        return [("".join(self.spell[b] for b in p), (0, 0)) for p in pieces]


def self_test():
    assert decode(r"a\nb\tc\r\\d") == "a\nb\tc\r\\d"
    assert decode(r"\x20\x41é\U0001f600") == " Aé\U0001f600"
    assert decode("plain") == "plain"
    table = byte_level_table()
    assert len(table) == 256 and sorted(table.values()) == list(range(256))
    assert table["Ġ"] == 0x20 and table["Ċ"] == 0x0A and table["A"] == 0x41
    with tempfile.TemporaryDirectory() as d:
        cases, ids, pieces = (os.path.join(d, n) for n in ("cases", "ids", "pieces"))
        open(cases, "w", encoding="utf-8").write("# a comment\nab cd\n\nx\\x20y\n")
        n = write(_Stand(), "stand-in", cases, ids, pieces)
        assert n == 2, n
        ids_lines = open(ids).read().split("\n")
        assert ids_lines[2].startswith("# columns") and ids_lines[-2] == "# complete\t2", ids_lines
        assert ids_lines[3] == "6162206364\t97 98 32 99 100\t97 98 32 99 100", ids_lines
        piece_lines = open(pieces).read().split("\n")
        assert piece_lines[3] == "6162206364\t6162 206364", piece_lines
        assert piece_lines[4] == "782079\t78 2079", piece_lines
        assert piece_lines[-2] == "# complete\t2", piece_lines
        # A piece list that does not join back to the text is refused by the writer's own check.
        broken = _Stand()
        broken.pre_tokenize_str = lambda text: [("a", (0, 0))]
        try:
            write(broken, "stand-in", cases, ids, pieces)
        except AssertionError:
            pass
        else:
            raise SystemExit("the writer wrote pieces that do not join back to the text")
    print("ernie55_pins: self-test ok")


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-test"]:
        self_test()
    elif len(sys.argv) == 5:
        main(*sys.argv[1:5])
    else:
        raise SystemExit(__doc__)
