#!/usr/bin/env python3
"""Token ids back to text from a GGUF file's own vocabulary, for a vocabulary bloomery-tokenize does not load.

    tools/ref/gguf-decode.py -m <gguf> -f <ids file>
    tools/ref/gguf-decode.py --self-test

bloomery-tokenize --decode loads the whole tokenizer, and its loader refuses a pre-tokenizer the crate
does not run (crates/tokenizer/src/vocab.rs, `UnsupportedPre`) — Qwen3.8's `qwen35` among them. Decoding
needs no pre-tokenizer: it reads `tokenizer.ggml.tokens` and `tokenizer.ggml.token_type` from the header
and writes each id's piece as crates/tokenizer/src/decode.rs `pieces` does (llama_token_to_piece with
special tokens rendered):

  CONTROL, USER_DEFINED, UNKNOWN   the token text as written
  NORMAL                           the text mapped back through the GPT-2 byte alphabet
  BYTE                             the byte its `<0xXX>` text names

The ids file holds decimal ids separated by anything else (`llama-tokenize --ids` output, one id per
line, a comma list). The pieces go to stdout as raw bytes, nothing appended. The caller's round trip
(tools/ref/ik-draft.sh: the reference's llama-tokenize must give the same ids back) is the check on
the text, not this tool.

Refused by name, never decoded to a plausible text: a file whose tokenizer.ggml.model is not gpt2, a
header without the two arrays or with arrays of different lengths, an id outside the vocabulary, an
UNUSED or unknown token type, a NORMAL token with a character outside the byte alphabet (decode.rs
writes the reference's `[UNK_BYTE_…]` marker for it), a BYTE token whose text is not `<0xXX>`.

Exit status: 0; 2 a file missing or unreadable; 64 usage; 65 a header this reader refuses or an id it
cannot decode.
"""
import importlib.util
import os
import re
import struct
import sys
import tempfile

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("gguf_ranges", os.path.join(_here, "gguf-ranges.py"))
gr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gr)

NORMAL, UNKNOWN, CONTROL, USER_DEFINED, UNUSED, BYTE = 1, 2, 3, 4, 5, 6
TYPE_NAMES = {NORMAL: "NORMAL", UNKNOWN: "UNKNOWN", CONTROL: "CONTROL", USER_DEFINED: "USER_DEFINED",
              UNUSED: "UNUSED", BYTE: "BYTE"}
BYTE_TEXT = re.compile(r"<0x([0-9A-Fa-f]{2})>")
KEYS = ("tokenizer.ggml.model", "tokenizer.ggml.tokens", "tokenizer.ggml.token_type")


class Refusal(Exception):
    def __init__(self, code, msg):
        super().__init__(msg)
        self.code = code


def byte_alphabet():
    """GPT-2's byte -> codepoint table inverted: codepoint -> byte (crates/tokenizer/src/unicode.rs)."""
    kept = list(range(0x21, 0x7F)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    table, n = {}, 0
    for b in range(256):
        if b in kept:
            table[b] = b
        else:
            table[b] = 256 + n
            n += 1
    return {cp: b for b, cp in table.items()}


def vocab(path):
    """(model, tokens as str, token types) from the header's tokenizer keys; the other keys are skipped."""
    try:
        f = open(path, "rb")
    except OSError as e:
        raise Refusal(2, f"{path}: {e.strerror}") from e
    with f:
        size = os.fstat(f.fileno()).st_size

        def take(n):
            b = f.read(n)
            if len(b) != n:
                raise Refusal(65, f"{path}: the header ends at byte {f.tell()}, inside a field")
            return b

        def u32():
            return struct.unpack("<I", take(4))[0]

        def u64():
            return struct.unpack("<Q", take(8))[0]

        def string():
            n = u64()
            if n > size:
                raise Refusal(65, f"{path}: a string of {n} bytes in a file of {size}")
            return take(n)

        def value(t, keep):
            if t in gr.SCALAR:
                fmt = gr.SCALAR[t]
                return struct.unpack(fmt, take(struct.calcsize(fmt)))[0]
            if t == 8:
                return string()
            if t == 9:
                et, cnt = u32(), u64()
                if not keep and et in gr.SCALAR:
                    step = struct.calcsize(gr.SCALAR[et])
                    if cnt * step > size:
                        raise Refusal(65, f"{path}: an array of {cnt} values in a file of {size}")
                    f.seek(cnt * step, 1)
                    return None
                if cnt > size:
                    raise Refusal(65, f"{path}: an array of {cnt} values in a file of {size}")
                out = [value(et, keep) for _ in range(cnt)]
                return out if keep else None
            raise Refusal(65, f"{path}: a metadata value of type {t}")

        if take(4) != b"GGUF":
            raise Refusal(65, f"{path} is not a GGUF file")
        version = u32()
        if version not in (2, 3):
            raise Refusal(65, f"{path}: GGUF version {version} (this reader knows 2 and 3)")
        u64()  # tensor count: the tensor infos follow the metadata and are not read
        n_kv = u64()
        meta = {}
        for _ in range(n_kv):
            k = string().decode("utf-8", "replace")
            v = value(u32(), k in KEYS)
            if k in KEYS:
                meta[k] = v
    for k in KEYS:
        if k not in meta:
            raise Refusal(65, f"{path}: no {k} in the header")
    model = meta["tokenizer.ggml.model"]
    if not isinstance(model, bytes) or model != b"gpt2":
        raise Refusal(65, f"{path}: tokenizer.ggml.model is {model!r}, not gpt2 (the byte alphabet is gpt2's)")
    tokens, types = meta["tokenizer.ggml.tokens"], meta["tokenizer.ggml.token_type"]
    if not all(isinstance(t, bytes) for t in tokens):
        raise Refusal(65, f"{path}: tokenizer.ggml.tokens is not an array of strings")
    if not all(isinstance(t, int) for t in types):
        raise Refusal(65, f"{path}: tokenizer.ggml.token_type is not an array of integers")
    if len(tokens) != len(types):
        raise Refusal(65, f"{path}: {len(tokens)} tokens but {len(types)} token types")
    try:
        texts = [t.decode("utf-8", "strict") for t in tokens]
    except UnicodeDecodeError as e:
        raise Refusal(65, f"{path}: a token text is not UTF-8 ({e})") from e
    return model.decode(), texts, types


def piece(tid, text, ty, alphabet):
    """One token's bytes, or a Refusal naming the token."""
    if ty in (CONTROL, USER_DEFINED, UNKNOWN):
        return text.encode("utf-8")
    if ty == NORMAL:
        out = bytearray()
        for ch in text:
            b = alphabet.get(ord(ch))
            if b is None:
                raise Refusal(65, f"id {tid} ({text!r}, NORMAL): U+{ord(ch):04X} is outside the byte alphabet")
            out.append(b)
        return bytes(out)
    if ty == BYTE:
        m = BYTE_TEXT.fullmatch(text)
        if not m:
            raise Refusal(65, f"id {tid}: a BYTE token whose text {text!r} is not <0xXX>")
        return bytes([int(m.group(1), 16)])
    raise Refusal(65, f"id {tid} ({text!r}): token type {TYPE_NAMES.get(ty, ty)} has no piece")


def read_ids(path):
    try:
        with open(path, encoding="ascii", errors="strict") as f:
            text = f.read()
    except OSError as e:
        raise Refusal(2, f"{path}: {e.strerror}") from e
    except UnicodeDecodeError as e:
        raise Refusal(65, f"{path}: not ASCII ({e})") from e
    return [int(x) for x in re.findall(r"\d+", text)]


def decode(path, ids):
    _, tokens, types = vocab(path)
    alphabet = byte_alphabet()
    out = bytearray()
    for tid in ids:
        if tid >= len(tokens):
            raise Refusal(65, f"id {tid} is outside the vocabulary of {len(tokens)} tokens")
        out += piece(tid, tokens[tid], types[tid], alphabet)
    return bytes(out)


def main(argv):
    if argv == ["--self-test"]:
        return self_test()
    opts = dict(zip(argv[::2], argv[1::2]))
    model = opts.pop("-m", None) or opts.pop("--model", None)
    ids_path = opts.pop("-f", None) or opts.pop("--file", None)
    if len(argv) % 2 or opts or not model or not ids_path:
        print(__doc__.split("\n\n")[1], file=sys.stderr)
        return 64
    try:
        text = decode(model, read_ids(ids_path))
    except Refusal as e:
        print(f"gguf-decode.py: {e}", file=sys.stderr)
        return e.code
    sys.stdout.buffer.write(text)
    return 0


def _kv_strings(k, values):
    kb = k.encode()
    body = b"".join(struct.pack("<Q", len(v.encode())) + v.encode() for v in values)
    return struct.pack("<Q", len(kb)) + kb + struct.pack("<I", 9) + struct.pack("<IQ", 8, len(values)) + body


def _kv_ints(k, values):
    kb = k.encode()
    body = b"".join(struct.pack("<i", v) for v in values)
    return struct.pack("<Q", len(kb)) + kb + struct.pack("<I", 9) + struct.pack("<IQ", 5, len(values)) + body


def _kv_scalar_u32(k, v):
    return gr._kv_scalar(k, 4, v)


def self_test():
    """A small gpt2 vocabulary in a fixture GGUF: every piece rule, and each refusal by name."""
    tokens = ["Hello", "Ġworld", "Ċ", "<|im_end|>", "<0x41>", "cafÃ©", "中", "<unused>",
              "<0xZZ>"]
    types = [NORMAL, NORMAL, NORMAL, CONTROL, BYTE, NORMAL, NORMAL, UNUSED, BYTE]
    ok = True

    def expect_refusal(name, fn, needle):
        nonlocal ok
        try:
            fn()
        except Refusal as e:
            if needle not in str(e) or e.code != 65:
                print(f"gguf-decode self-test: {name}: refused with {e.code} {e}, wanted {needle!r}", file=sys.stderr)
                ok = False
            return
        print(f"gguf-decode self-test: {name}: not refused", file=sys.stderr)
        ok = False

    with tempfile.TemporaryDirectory() as root:
        good = os.path.join(root, "v.gguf")
        base = [gr._kv_string("general.architecture", "tst"), gr._kv_scalar("tst.block_count", 4, 1)]
        gr.write_gguf(good, base + [gr._kv_string("tokenizer.ggml.model", "gpt2"),
                                    _kv_strings("tokenizer.ggml.tokens", tokens),
                                    _kv_ints("tokenizer.ggml.token_type", types),
                                    _kv_scalar_u32("tokenizer.ggml.eos_token_id", 3)],
                      [("token_embd.weight", 0, [4, 9], b"\0" * 144)])
        # Ġ is the space, Ċ the newline, a CONTROL token keeps its text, <0x41> is 'A', and the byte
        # alphabet turns "cafÃ©" into the UTF-8 bytes of "café".
        got = decode(good, [0, 1, 2, 3, 4, 5])
        want = "Hello world\n<|im_end|>Acafé".encode("utf-8")
        if got != want:
            print(f"gguf-decode self-test: decode {got!r}, want {want!r}", file=sys.stderr)
            ok = False
        ids_file = os.path.join(root, "ids")
        with open(ids_file, "w", encoding="ascii") as f:
            f.write("[0, 1]\n2\n")
        if read_ids(ids_file) != [0, 1, 2]:
            print("gguf-decode self-test: the ids file did not read as [0, 1, 2]", file=sys.stderr)
            ok = False
        expect_refusal("an id past the vocabulary", lambda: decode(good, [9]), "outside the vocabulary of 9")
        expect_refusal("a character outside the alphabet", lambda: decode(good, [6]), "U+4E2D is outside")
        expect_refusal("an UNUSED token", lambda: decode(good, [7]), "UNUSED has no piece")
        expect_refusal("a BYTE token without <0xXX>", lambda: decode(good, [8]), "is not <0xXX>")
        spm = os.path.join(root, "spm.gguf")
        gr.write_gguf(spm, base + [gr._kv_string("tokenizer.ggml.model", "llama"),
                                   _kv_strings("tokenizer.ggml.tokens", tokens[:2]),
                                   _kv_ints("tokenizer.ggml.token_type", types[:2])], [])
        expect_refusal("a llama (SentencePiece) vocabulary", lambda: decode(spm, [0]), "not gpt2")
        short = os.path.join(root, "short.gguf")
        gr.write_gguf(short, base + [gr._kv_string("tokenizer.ggml.model", "gpt2"),
                                     _kv_strings("tokenizer.ggml.tokens", tokens[:3]),
                                     _kv_ints("tokenizer.ggml.token_type", types[:2])], [])
        expect_refusal("arrays of different lengths", lambda: decode(short, [0]), "3 tokens but 2 token types")
        bare = os.path.join(root, "bare.gguf")
        gr.write_gguf(bare, base + [gr._kv_string("tokenizer.ggml.model", "gpt2")], [])
        expect_refusal("no token array", lambda: decode(bare, [0]), "no tokenizer.ggml.tokens")
    if not ok:
        return 1
    print("gguf-decode: self-test ok")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
