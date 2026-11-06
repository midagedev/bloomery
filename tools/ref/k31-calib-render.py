#!/usr/bin/env python3
"""k31-calib-render.py — the conversations of k31-calib-gen.py as the imatrix's calibration text, in K3.1's own encoding.

    uv run --with tiktoken python -I tools/ref/k31-calib-render.py --conversations conversations.jsonl \
        --k31-folder DIR --out calib-chat-c8192.txt [--ctx 8192] [--chunks N] [--vocab-adds-bos 1]
    python3 -I tools/ref/k31-calib-render.py --self-test [--k31-folder DIR]

The rendering is the folder's own `encoding_k3.py` (loaded from --k31-folder, not in the tree): `build_chat_segments` gives each
conversation as segments, a control segment (`<|open|>`, `<|sep|>`, `<|close|>`, `<|end_of_msg|>`) or ordinary text; the text is their
concatenation, the special tokens as their literal strings, so llama-imatrix reads it with --parse-special (k31-quant.sh
--corpus-special). A conversation with reasoning is rendered with the think channel, one without it with the channel dropped
(thinking=False), and none ends in a generation prompt. A conversation whose text contains a special token's literal string is skipped
by name: the imatrix would parse it as a control token.

Token counts come from the folder's tiktoken vocabulary (tiktoken.model, tokenizer_config.json, the split pattern of
tokenization_kimi.py), a run at a time: a control segment as one token, a run of adjacent text segments with the specials disallowed. Needs `tiktoken`.

Packing: the imatrix reads the text as one token stream cut every --ctx tokens, and puts a BOS over the first token of each chunk.
Each chunk is therefore one bin of whole conversations (best fit, in a fixed shuffled order): it opens with a literal `[BOS]` (the
stream's own BOS stands in for it in the first chunk when the vocabulary adds one: --vocab-adds-bos, default 1), holds at most
ctx-1 tokens of conversations, and its slack is filled with prefixes (cut at run boundaries) of other conversations, the last
tokens with `<|end_of_msg|>`, so the next chunk starts on its own `[BOS]`. The count is tiktoken's: confirm the chunk starts with
llama-tokenize on the real vocabulary before a long run. A manifest `<out>.manifest.tsv` lists chunk, conversation id, language, kind
and tokens (pad rows are `pad`).
"""
import argparse
import ast
import importlib.util
import json
import os
import random
import re
import sys
import tempfile

BOS = "[BOS]"
END_OF_MSG = "<|end_of_msg|>"


def load_encoder(folder):
    path = os.path.join(folder, "encoding_k3.py")
    if not os.path.isfile(path):
        raise SystemExit(f"k31-calib-render: no encoding_k3.py in {folder}")
    spec = importlib.util.spec_from_file_location("encoding_k3", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["encoding_k3"] = mod
    spec.loader.exec_module(mod)
    return mod


class Counter:
    """Token counts from the folder's tiktoken vocabulary."""

    def __init__(self, folder):
        try:
            import tiktoken
            from tiktoken.load import load_tiktoken_bpe
        except ImportError:
            raise SystemExit("k31-calib-render: needs the tiktoken package (uv run --with tiktoken python -I ...)")
        cfg = json.load(open(os.path.join(folder, "tokenizer_config.json"), encoding="utf-8"))
        ranks = load_tiktoken_bpe(os.path.join(folder, "tiktoken.model"))
        base = len(ranks)
        names = {int(i): v["content"] for i, v in cfg["added_tokens_decoder"].items()}
        self.special = {names.get(i, f"<|reserved_token_{i}|>"): i for i in range(base, base + 256)}
        self.enc = tiktoken.Encoding(name="k31", pat_str=pat_str_of(os.path.join(folder, "tokenization_kimi.py")),
                                     mergeable_ranks=ranks, special_tokens=self.special)

    def count(self, text, control):
        if control:
            return len(self.enc.encode(text, allowed_special="all"))
        return len(self.enc.encode(text, disallowed_special=()))


def pat_str_of(path):
    """The split pattern of tokenization_kimi.py: its `pat_str = "|".join([...])`, evaluated as a join of string literals only."""
    for node in ast.walk(ast.parse(open(path, encoding="utf-8").read())):
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "pat_str" for t in node.targets):
            call = node.value
            if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == "join"
                    and isinstance(call.func.value, ast.Constant)):
                return call.func.value.value.join(ast.literal_eval(call.args[0]))
    raise SystemExit(f"k31-calib-render: no pat_str = \"|\".join([...]) in {path}")


def render_conversation(enc, counter, row):
    """(segments as (text, tokens), error or None) for one stored conversation."""
    msgs = [dict(m) for m in row["messages"]]
    segs = enc.build_chat_segments(msgs, row.get("tools") or None, add_generation_prompt=False,
                                   thinking=bool(row.get("thinking")))
    runs = []  # adjacent text segments are one run: llama-imatrix tokenizes them jointly
    for s in segs:
        if not s.allow_special and any(name in s.text for name in counter.special):
            return None, "its text contains a special token's literal string"
        if runs and not s.allow_special and not runs[-1][1]:
            runs[-1] = (runs[-1][0] + s.text, False)
        else:
            runs.append((s.text, s.allow_special))
    return [(text, counter.count(text, control)) for text, control in runs], None


def pack(convs, cap, seed=31):
    """Best fit, in a fixed shuffled order: bins (lists of conversation indexes) of at most cap tokens."""
    order = list(range(len(convs)))
    random.Random(seed).shuffle(order)
    bins, loads = [], []
    for i in order:
        n = convs[i]["tokens"]
        best = None
        for b, load in enumerate(loads):
            if load + n <= cap and (best is None or load > loads[best]):
                best = b
        if best is None:
            bins.append([i])
            loads.append(n)
        else:
            bins[best].append(i)
            loads[best] += n
    return bins, loads


def prefix_fill(convs, bin_ids, slack, rnd):
    """Segments (text, tokens) that fill slack: prefixes of conversations not in the bin, cut at run boundaries."""
    pad, parts = [], []
    others = [i for i in range(len(convs)) if i not in bin_ids]
    rnd.shuffle(others)
    for i in others:
        if slack <= 0:
            break
        taken, text = 0, []
        for t, n in convs[i]["segs"]:
            if taken + n > slack:
                break
            text.append(t)
            taken += n
        if taken >= 32:
            pad.append("".join(text))
            parts.append((convs[i]["id"], taken))
            slack -= taken
    return pad, parts, slack


def render(rows, counter, enc, ctx, chunks=0, adds_bos=True, say=print):
    convs, skipped = [], []
    for row in rows:
        segs, err = render_conversation(enc, counter, row)
        if err:
            skipped.append((row["id"], err))
            continue
        n = sum(c for _, c in segs)
        if n > ctx - 1:
            skipped.append((row["id"], f"{n} tokens do not fit a chunk of {ctx}"))
            continue
        convs.append({"id": row["id"], "lang": row["lang"], "kind": row["kind"], "thinking": bool(row.get("thinking")),
                      "segs": segs, "text": "".join(t for t, _ in segs), "tokens": n})
    cap = ctx - 1
    bins, loads = pack(convs, cap)
    # whole chunks only: a bin under half full is the leftover of the packing and goes unless it is the only one
    keep = [b for b, load in zip(bins, loads) if load >= cap // 2] or bins[:1]
    if chunks:
        keep = keep[:chunks]
    rnd = random.Random(7)
    text, manifest, pad_total = [], [], 0
    for k, ids in enumerate(keep):
        body = [convs[i]["text"] for i in ids]
        load = sum(convs[i]["tokens"] for i in ids)
        pad, parts, slack = prefix_fill(convs, set(ids), cap - load, rnd)
        body.extend(pad)
        for cid, n in parts:
            manifest.append((k, f"pad:{cid}", "", "", n))
        if slack:
            body.append(END_OF_MSG * slack)
            manifest.append((k, "pad:end_of_msg", "", "", slack))
        pad_total += cap - load
        head = "" if (k == 0 and adds_bos) else BOS
        text.append(head + "".join(body))
        for i in ids:
            manifest.append((k, convs[i]["id"], convs[i]["lang"], convs[i]["kind"], convs[i]["tokens"]))
    used = {i for ids in keep for i in ids}
    kept = [convs[i] for i in sorted(used)]
    total = sum(c["tokens"] for c in kept)
    share = lambda f: 100.0 * sum(c["tokens"] for c in kept if f(c)) / max(total, 1)
    say(f"k31-calib-render: {len(rows)} conversations in, {len(kept)} rendered, {len(skipped)} skipped"
        + "".join(f"; {i}: {e}" for i, e in skipped))
    say(f"k31-calib-render: {total} conversation tokens + {pad_total} pad = {len(keep) * ctx - (1 if adds_bos else 0)} text tokens (tiktoken count)"
        f" in {len(keep)} chunk(s) of {ctx}")
    say("k31-calib-render: share by tokens: " + " ".join(f"{n} {share(f):.1f}%" for n, f in [
        ("EN", lambda c: c["lang"] == "en"), ("KO", lambda c: c["lang"] == "ko"), ("code", lambda c: c["lang"] == "code"),
        ("tool-use", lambda c: c["kind"] == "tool"), ("reasoning", lambda c: c["thinking"])]) + " (the last two overlap the first three)")
    return "".join(text), manifest, len(keep), total


def load_rows(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


# ---------------------------------------------------------------- self-test

STUB_ENCODER = '''
from dataclasses import dataclass
@dataclass(frozen=True)
class EncodeSegment:
    text: str
    allow_special: bool = False
def build_chat_segments(messages, tools=None, *, add_generation_prompt=True, thinking=True, **kw):
    out = []
    if tools:
        out += [EncodeSegment("<|open|>", True), EncodeSegment("message tools", False), EncodeSegment("<|sep|>", True)]
    for m in messages:
        out += [EncodeSegment("<|open|>", True), EncodeSegment("message " + m["role"], False), EncodeSegment("<|sep|>", True)]
        if m["role"] == "assistant" and thinking:
            out += [EncodeSegment("<|open|>", True), EncodeSegment("think", False), EncodeSegment("<|sep|>", True),
                    EncodeSegment(m.get("reasoning", ""), False)]
        out += [EncodeSegment(m["content"], False), EncodeSegment("<|end_of_msg|>", True)]
    return out
'''


class FakeCounter:
    """One token per word, symbol or control string, specials named like the real vocabulary's."""
    special = {"<|open|>": 1, "<|close|>": 2, "<|sep|>": 3, "<|end_of_msg|>": 4, "[BOS]": 5, "[EOS]": 6}

    def count(self, text, control):
        return 1 if control else len(re.findall(r"\w+|[^\w\s]", text))


def self_test(folder=None):
    d = tempfile.mkdtemp(prefix="k31render-")
    open(d + "/encoding_k3.py", "w").write(STUB_ENCODER)
    enc, counter = load_encoder(d), FakeCounter()

    def conv(i, n, lang="en", kind="qa", thinking=False, content=None):
        return {"id": f"c{i}", "lang": lang, "kind": kind, "thinking": thinking, "tools": None,
                "messages": [{"role": "user", "content": "q " * 2},
                             {"role": "assistant", "content": content or " ".join(["w"] * n), "reasoning": "r r" if thinking else None}]}
    rows = [conv(i, 40 + 7 * (i % 5), ["en", "ko", "code"][i % 3], "tool" if i % 4 == 0 else "qa", i % 2 == 0) for i in range(30)]
    rows.append(conv(98, 5, content="has <|open|> inside"))
    rows.append(conv(99, 500))
    lines = []
    text, manifest, nchunks, total = render(rows, counter, enc, 100, say=lines.append)
    assert "c98: its text contains" in lines[0] and "c99: " in lines[0], lines[0]
    assert nchunks >= 2 and text.count(BOS) == nchunks - 1, (nchunks, text.count(BOS))
    # every chunk is exactly ctx tokens in the fake count (the stream's BOS stands over the first chunk's first token)
    chunks = text.split(BOS)
    sizes = []
    for k, part in enumerate(chunks):
        n = 0
        for piece in re.split(r"(<\|[a-z_]+\|>)", part):
            if piece in counter.special:
                n += 1
            elif piece:
                n += counter.count(piece, False)
        sizes.append(n + 1)
    assert all(s == 100 for s in sizes), sizes
    # whole conversations: every conversation id appears in one chunk of the manifest, none twice
    ids = [m[1] for m in manifest if not m[1].startswith("pad:")]
    assert len(ids) == len(set(ids)) and "c98" not in ids and "c99" not in ids, ids
    # the think channel follows the conversation's own flag: only the rendered thinking conversations carry it
    assert text.count("message assistant") == len(ids)
    on, _ = render_conversation(enc, counter, conv(0, 5, thinking=True))
    off, _ = render_conversation(enc, counter, conv(1, 5, thinking=False))
    assert "think" in "".join(x for x, _ in on) and "think" not in "".join(x for x, _ in off)
    # --chunks cuts to whole chunks
    _, _, n2, _ = render(rows, counter, enc, 100, chunks=1, say=lambda s: None)
    assert n2 == 1
    # with --vocab-adds-bos 0 the first chunk opens with its own literal BOS
    t0, _, _, _ = render(rows, counter, enc, 100, adds_bos=False, say=lambda s: None)
    assert t0.startswith(BOS)
    if folder:
        import_check(folder)
    print("k31-calib-render self-test: ok")
    return 0


def import_check(folder):
    """Against the real folder: the encoder's own markers open the text and the count agrees with a hand split."""
    enc, counter = load_encoder(folder), Counter(folder)
    row = {"id": "x", "lang": "ko", "kind": "qa", "thinking": True, "tools": None,
           "messages": [{"role": "user", "content": "안녕"}, {"role": "assistant", "content": "안녕하세요", "reasoning": "인사"}]}
    segs, err = render_conversation(enc, counter, row)
    text = "".join(t for t, _ in segs)
    assert err is None and text.startswith("<|open|>message") and text.endswith("<|end_of_msg|>"), text
    assert "<|open|>think<|sep|>인사<|close|>think<|sep|>" in text, text
    assert sum(n for _, n in segs) > 10


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--conversations")
    ap.add_argument("--k31-folder")
    ap.add_argument("--out")
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--chunks", type=int, default=0, help="at most this many chunks (default: all that fill)")
    ap.add_argument("--vocab-adds-bos", type=int, default=1, choices=(0, 1))
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test(a.k31_folder)
    if not (a.conversations and a.k31_folder and a.out):
        ap.error("--conversations, --k31-folder and --out are required")
    enc, counter = load_encoder(a.k31_folder), Counter(a.k31_folder)
    text, manifest, nchunks, _ = render(load_rows(a.conversations), counter, enc, a.ctx, a.chunks, bool(a.vocab_adds_bos))
    with open(a.out, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    with open(a.out + ".manifest.tsv", "w", encoding="utf-8") as f:
        for row in manifest:
            f.write("\t".join(str(x) for x in row) + "\n")
    print(f"k31-calib-render: wrote {a.out}: {nchunks} chunk(s) of {a.ctx}, {len(text.encode())} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
