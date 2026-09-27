#!/usr/bin/env python3
"""The reduced head's row list of an MTP draft: the vocabulary ids its head scores, ranked from what the
target emits and what the corpora hold.

    tools/ref/draft-vocab.py --target <first shard> --rows N --out FILE \\
        [--greedy TSV ...] [--corpus IDS ...] [--keep ID,...]
    tools/ref/draft-vocab.py --self-test

--target is the target model's first shard. Its header gives the vocabulary (`tokenizer.ggml.tokens`),
its digest and the end and padding ids, which the list always holds. The digest is SHA-256 over every
token in id order: its UTF-8 byte count (u64, little-endian), then the bytes. That is the array's
payload as the file stores it, and `model::arch::qwen35moe::place::vocab_sha256` computes the same. A
token whose bytes are not UTF-8, or that holds U+FFFD, is refused by both. The Rust reader decodes
lossily, so it cannot see such a token's bytes.

Each --greedy file is ik's greedy continuations (`tools/ref/argmax_ref.cpp --gen`). Its rows are read
by the names on its `#id` column line, never by position. The ids a row contributes are those under
`gen_ids` (the target's own greedy tokens) and under `top5_ids`. Each --corpus file is one id per line
(`corpus-prose.ids`, `corpus-code.ids`).

Every id gets two frequencies: its share of the greedy ids and its share of the corpus ids. A source
with no files adds 0. An id's score is the sum of the two, and the ids are ranked by that score, ties
to the lower id. The list is the end id, the padding id and every --keep id, then the ranked ids until
it holds N. It is written in ascending order:

    # bloomery mtp-head-rows 1 vocab=<n> rows=<N> vocab_sha256=<64 hex digits>
    <id>
    ...

One line goes on stdout:
    rows=<N> vocab=<n> greedy_ids=<g> greedy_cover=<x> corpus_ids=<c> corpus_cover=<y>
Each cover is the share of that source's ids the list holds. They are in-sample: the list was chosen from
those same ids.

Refused, each by name: an id at or past the vocabulary in any input; a greedy file with no `#id` line,
or one without the `gen_ids` and `top5_ids` columns, or a row of another width; a non-integer corpus
line; N outside 1..vocab-1; fewer distinct ids in the inputs and the kept ids than N (a list is never
padded with ids no input names); a header that is not GGUF 2 or 3, or has no token array.

The list moves how often the draft is accepted, never the tokens the target emits: those are always the
target's own argmax.

Exit status: 0; 2 a file missing or unreadable; 64 a usage error; 65 a malformed input.
"""
import hashlib
import os
import struct
import sys
import tempfile
from collections import Counter

FORMAT = "# bloomery mtp-head-rows 1"
TOKENS = "tokenizer.ggml.tokens"
EOS = "tokenizer.ggml.eos_token_id"
PAD = "tokenizer.ggml.padding_token_id"
# GGUF metadata scalar types (v2/v3): struct formats; 8 is a string, 9 an array.
SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


class Refusal(Exception):
    """A named refusal: the message and the exit status."""

    def __init__(self, code, msg):
        super().__init__(msg)
        self.code = code


def header_tokens(path):
    """(the token array's entries as raw bytes, the end id, the padding id) of a GGUF header."""
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
            if t in SCALAR:
                fmt = SCALAR[t]
                return struct.unpack(fmt, take(struct.calcsize(fmt)))[0]
            if t == 8:
                return string()
            if t == 9:
                et, cnt = u32(), u64()
                if et in SCALAR and not keep:
                    step = struct.calcsize(SCALAR[et])
                    if cnt * step > size:
                        raise Refusal(65, f"{path}: an array of {cnt} values in a file of {size}")
                    f.seek(cnt * step, 1)
                    return None
                out = [value(et, keep) for _ in range(cnt)]
                return out if keep else None
            raise Refusal(65, f"{path}: a metadata value of type {t}")

        if take(4) != b"GGUF":
            raise Refusal(65, f"{path} is not a GGUF file")
        version = u32()
        if version not in (2, 3):
            raise Refusal(65, f"{path}: GGUF version {version} (this reader knows 2 and 3)")
        u64()
        n_kv = u64()
        meta = {}
        for _ in range(n_kv):
            k = string().decode("utf-8", "replace")
            meta[k] = value(u32(), k == TOKENS)
    tokens = meta.get(TOKENS)
    if not isinstance(tokens, list) or not all(isinstance(t, bytes) for t in tokens):
        raise Refusal(65, f"{path}: {TOKENS} is absent or not an array of strings")
    for i, t in enumerate(tokens):
        try:
            text = t.decode("utf-8")
        except UnicodeDecodeError:
            raise Refusal(65, f"{path}: token {i} is not UTF-8, so the engine's reader cannot digest its bytes") from None
        if "�" in text:
            raise Refusal(65, f"{path}: token {i} holds U+FFFD, which the engine's reader cannot tell from bytes that are not UTF-8")
    ids = []
    for key in (EOS, PAD):
        v = meta.get(key)
        if not isinstance(v, int) or not 0 <= v < len(tokens):
            raise Refusal(65, f"{path}: {key} is {v!r}, not an id of the {len(tokens)}-token vocabulary")
        ids.append(v)
    return tokens, ids[0], ids[1]


def vocab_sha256(tokens):
    """The vocabulary's digest (module docstring)."""
    h = hashlib.sha256()
    for t in tokens:
        h.update(struct.pack("<Q", len(t)))
        h.update(t)
    return h.hexdigest()


def check_id(path, line, v, vocab):
    if not 0 <= v < vocab:
        raise Refusal(65, f"{path}:{line}: id {v} is past the vocabulary of {vocab}")
    return v


def read_greedy(path, vocab):
    """The ids of a greedy tsv's gen_ids and top5_ids columns, by the names on its `#id` line."""
    try:
        lines = open(path, encoding="utf-8").read().splitlines()
    except OSError as e:
        raise Refusal(2, f"{path}: {e.strerror}") from e
    names = None
    out = []
    for n, line in enumerate(lines, 1):
        if line.startswith("#id\t"):
            names = ["id"] + line.split("\t")[1:]
            for col in ("gen_ids", "top5_ids"):
                if col not in names:
                    raise Refusal(65, f"{path}:{n}: the column line names no {col}")
            continue
        if not line or line.startswith("#"):
            continue
        if names is None:
            raise Refusal(65, f"{path}:{n}: a row before the `#id` column line")
        fields = line.split("\t")
        if len(fields) != len(names):
            raise Refusal(65, f"{path}:{n}: {len(fields)} fields, the column line names {len(names)}")
        for col in ("gen_ids", "top5_ids"):
            for tok in fields[names.index(col)].split(","):
                try:
                    v = int(tok.strip())
                except ValueError:
                    raise Refusal(65, f"{path}:{n}: {col} holds {tok!r}, not an id") from None
                out.append(check_id(path, n, v, vocab))
    if names is None:
        raise Refusal(65, f"{path}: no `#id` column line")
    return out


def read_corpus(path, vocab):
    """One id a line."""
    try:
        lines = open(path, encoding="utf-8").read().splitlines()
    except OSError as e:
        raise Refusal(2, f"{path}: {e.strerror}") from e
    out = []
    for n, line in enumerate(lines, 1):
        if not line.isdigit():
            raise Refusal(65, f"{path}:{n}: {line!r} is not an id")
        out.append(check_id(path, n, int(line), vocab))
    return out


def choose(vocab, rows, greedy, corpus, keep):
    """The list: the kept ids, then the ranked ones until `rows`, ascending."""
    if not 0 < rows < vocab:
        raise Refusal(64, f"--rows {rows}: a list holds 1 to {vocab - 1} rows (the whole vocabulary is the full head)")
    score = Counter()
    for src in (greedy, corpus):
        if src:
            for v, c in Counter(src).items():
                score[v] += c / len(src)
    chosen = set(keep)
    if len(chosen) > rows:
        raise Refusal(64, f"--rows {rows} is fewer than the {len(chosen)} ids the list always holds")
    for v, _ in sorted(score.items(), key=lambda kv: (-kv[1], kv[0])):
        if len(chosen) == rows:
            break
        chosen.add(v)
    if len(chosen) < rows:
        raise Refusal(65, f"the inputs and the kept ids name {len(chosen)} distinct ids, fewer than --rows {rows}")
    return sorted(chosen)


def cover(src, chosen):
    s = set(chosen)
    return sum(1 for v in src if v in s) / len(src) if src else 0.0


def write(path, vocab, digest, ids):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{FORMAT} vocab={vocab} rows={len(ids)} vocab_sha256={digest}\n")
        for v in ids:
            f.write(f"{v}\n")


def run(target, rows, out, greedy_paths, corpus_paths, keep):
    tokens, eos, pad = header_tokens(target)
    vocab = len(tokens)
    for v in keep:
        check_id("--keep", 1, v, vocab)
    greedy = [v for p in greedy_paths for v in read_greedy(p, vocab)]
    corpus = [v for p in corpus_paths for v in read_corpus(p, vocab)]
    ids = choose(vocab, rows, greedy, corpus, [eos, pad] + keep)
    write(out, vocab, vocab_sha256(tokens), ids)
    return (f"rows={len(ids)} vocab={vocab} greedy_ids={len(greedy)} greedy_cover={cover(greedy, ids):.4f} "
            f"corpus_ids={len(corpus)} corpus_cover={cover(corpus, ids):.4f}")


def main(argv):
    if argv == ["--self-test"]:
        return self_test()
    target, rows, out, greedy, corpus, keep = None, None, None, [], [], []
    i = 0
    try:
        while i < len(argv):
            a = argv[i]
            if i + 1 >= len(argv):
                raise Refusal(64, f"{a} takes a value")
            v = argv[i + 1]
            if a == "--target":
                target = v
            elif a == "--rows":
                rows = int(v)
            elif a == "--out":
                out = v
            elif a == "--greedy":
                greedy.append(v)
            elif a == "--corpus":
                corpus.append(v)
            elif a == "--keep":
                keep.extend(int(x) for x in v.split(","))
            else:
                raise Refusal(64, f"unknown flag {a}")
            i += 2
        if target is None or rows is None or out is None:
            raise Refusal(64, "--target, --rows and --out are required")
        print(run(target, rows, out, greedy, corpus, keep))
    except ValueError as e:
        print(f"draft-vocab: {e}\n{__doc__}", file=sys.stderr)
        return 64
    except Refusal as e:
        print(f"draft-vocab: {e}", file=sys.stderr)
        return e.code
    return 0


# ---- the self-test ----

def _gguf(path, tokens, eos, pad):
    """A header-only GGUF 3 file of five keys: the architecture, an unrelated array, the token array,
    the end and padding ids."""
    b = bytearray(b"GGUF") + struct.pack("<IQQ", 3, 0, 5)

    def s(x):
        b.extend(struct.pack("<Q", len(x)) + x)

    s(b"general.architecture")
    b.extend(struct.pack("<I", 8))
    s(b"qwen4exp")
    s(b"qwen4exp.attention.compress_ratios")
    b.extend(struct.pack("<IIQ", 9, 5, 3) + struct.pack("<3i", 0, 0, 4))
    s(TOKENS.encode())
    b.extend(struct.pack("<IIQ", 9, 8, len(tokens)))
    for t in tokens:
        s(t)
    s(EOS.encode())
    b.extend(struct.pack("<II", 4, eos))
    s(PAD.encode())
    b.extend(struct.pack("<II", 4, pad))
    with open(path, "wb") as f:
        f.write(b)


def self_test():
    ok = True

    def expect(what, cond):
        nonlocal ok
        print(f"{'ok  ' if cond else 'FAIL'} {what}")
        ok = ok and cond

    def refused(what, code, fn):
        try:
            fn()
        except Refusal as e:
            expect(f"{what} (exit {e.code}: {e})", e.code == code)
            return
        expect(f"{what}: not refused", False)

    # The digest pins shared with place.rs's vocab_digest_of_two_tokens: tokens "a", "b".
    expect("digest of a, b", vocab_sha256([b"a", b"b"]) ==
           "cf6ab613e3942391f88ed698557e1680f160bd10e88c6b668c50360c10930e2b")
    expect("digest of a, bc, Ġx, <|im_end|>", vocab_sha256([t.encode() for t in ["a", "bc", "Ġx", "<|im_end|>"]]) ==
           "5e71a36312a596cecc96e69e404f0c5ed96a6c0ab2856b4ca9c1e0aeb5a7f0b9")
    with tempfile.TemporaryDirectory() as d:
        tokens = [f"t{i}".encode() for i in range(12)]
        target = os.path.join(d, "t.gguf")
        _gguf(target, tokens, eos=11, pad=10)
        got, eos, pad = header_tokens(target)
        expect("header: tokens, eos, pad", (got, eos, pad) == (tokens, 11, 10))
        greedy = os.path.join(d, "g.tsv")
        with open(greedy, "w") as f:
            f.write("# argmax_ref\tik_llama.cpp\tmodel=/m.gguf\tgen=3\n"
                    "#id\tn_tokens\targmax\ttop5_ids\ttop5_logits\tgen_ids\tgen_margins\n"
                    "0\t4\t3\t3,4,5,6,7\t5,4,3,2,1\t3,3,2\t1,1,1\n")
        corpus = os.path.join(d, "c.ids")
        with open(corpus, "w") as f:
            f.write("1\n1\n1\n2\n9\n")
        out = os.path.join(d, "rows.txt")
        line = run(target, 5, out, [greedy], [corpus], [])
        text = open(out).read().splitlines()
        # greedy ids: 3,4,5,6,7,3,3,2 -> 3: 3/8, 2..7 others 1/8; corpus: 1: 3/5, 2: 1/5, 9: 1/5.
        # scores: 1 .6, 3 .375, 2 .325, 9 .2, 4..7 .125; kept 11 and 10, then 1, 3, 2.
        expect("the list", text == [f"{FORMAT} vocab=12 rows=5 vocab_sha256={vocab_sha256(tokens)}",
                                    "1", "2", "3", "10", "11"])
        expect(f"the summary line: {line}", line.startswith("rows=5 vocab=12 greedy_ids=8 greedy_cover=0.5000")
               and "corpus_cover=0.8000" in line)
        run(target, 5, out + "2", [greedy], [corpus], [])
        expect("deterministic", open(out).read() == open(out + "2").read())
        refused("rows past the vocabulary", 64, lambda: run(target, 12, out, [greedy], [corpus], []))
        refused("no rows", 64, lambda: run(target, 0, out, [greedy], [corpus], []))
        # 1..7, 9 and the kept 10, 11: ten distinct ids.
        refused("more rows than the inputs name", 65, lambda: run(target, 11, out, [greedy], [corpus], []))
        bad = os.path.join(d, "bad.ids")
        with open(bad, "w") as f:
            f.write("1\n12\n")
        refused("a corpus id past the vocabulary", 65, lambda: run(target, 3, out, [], [bad], []))
        with open(bad, "w") as f:
            f.write("1\n\n")
        refused("a blank corpus line", 65, lambda: run(target, 3, out, [], [bad], []))
        g2 = os.path.join(d, "g2.tsv")
        with open(g2, "w") as f:
            f.write("0\t4\t3\t3,4,5,6,7\t5,4,3,2,1\t3,3,2\t1,1,1\n")
        refused("a greedy file with no column line", 65, lambda: run(target, 3, out, [g2], [], []))
        with open(g2, "w") as f:
            f.write("#id\tn_tokens\targmax\ttop5_ids\n0\t4\t3\t3,4\n")
        refused("a greedy file without gen_ids", 65, lambda: run(target, 3, out, [g2], [], []))
        bt = os.path.join(d, "bad.gguf")
        _gguf(bt, [b"a", b"\xff"], eos=0, pad=1)
        refused("a token that is not UTF-8", 65, lambda: header_tokens(bt))
        _gguf(bt, ["a�".encode()], eos=0, pad=0)
        refused("a token holding U+FFFD", 65, lambda: header_tokens(bt))
        _gguf(bt, [b"a", b"b"], eos=5, pad=0)
        refused("an end id past the vocabulary", 65, lambda: header_tokens(bt))
    print("draft-vocab self-test:", "ok" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
