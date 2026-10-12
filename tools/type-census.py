#!/usr/bin/env python3
"""Census of the GGUF files users download, by header read only: which ggml type each tensor of a set holds.

    tools/type-census.py OUT_DIR [--stubs DIR] REPO...
    tools/type-census.py --self-test    offline: the parse, the grouping, the refusals and the stubs on synthetic headers

For each REPO (`owner/name`) it lists the files of the main branch, groups them into sets (the hf crate's split
shards together; `mmproj` files left out, every other .gguf kept: a draft head's file is a file users download), and
reads each shard's header by ranged fetches (curl, tools/bloomery/hf.py); no tensor byte is read. A listing with a
next page is refused by name. It writes OUT_DIR/<owner>__<name>__<set>.json for each set:

    repo, set, shards (the files, in order), sizes (file -> bytes), kv (the `general.*` keys and the layer and expert
    counts), top (a tensor outside the blocks -> [type, dims]), blk (the tensor name without `blk.<i>.` -> type ->
    the sorted layers)

and prints one line per set:

    <repo> <set> arch=<general.architecture> tensors=<n> shards=<n>

Type names are tools/bloomery/hf.py's ggml names. A tensor type id the table does not name fails its set
(`ggml type id <n> in <file>: not in the type table`), never a placeholder name.

--stubs DIR writes, for each shard of a set that read whole, its header bytes up to the tensor data (padded to
`general.alignment`) to DIR/<repo>/<file>, then truncates the file to the shard's real size: a sparse file the GGUF
reader opens with every tensor in place and every tensor byte zero, for plan-only probes.

A set that fails prints `<repo> <set>: FAILED <reason>` and writes nothing; a repo that cannot be listed prints
`<repo>: listing failed: <reason>`. Exit status: 0 every set read; 1 a set, a listing or an output name failed; 64 a
usage error.
"""
import collections
import concurrent.futures
import contextlib
import importlib.util
import io
import json
import os
import re
import struct
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_by_path(name, *rel):
    """A module of this tree, loaded by path (tools/ has no packages), leaving no bytecode behind."""
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, *rel))
    mod = importlib.util.module_from_spec(spec)
    keep, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = keep
    return mod


hf = load_by_path("bloomery_hf", "tools", "bloomery", "hf.py")

POOL = 8
HEADER_BUDGET = 64 * hf.MIB  # a header past this is an error, not a longer wait
KV_SUFFIXES = ("expert_count", "expert_used_count", "block_count", "nextn_predict_layers", "leading_dense_block_count")
BLOCK = re.compile(r"blk\.(\d+)\.(.*)")
REPO = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*")
USAGE = "usage: type-census.py OUT_DIR [--stubs DIR] REPO... | --self-test"


def census_file(name):
    """The files a census reads: every .gguf but a multimodal projector (the hf crate's `is_model` also drops the
    draft and importance-matrix files, which are files users download)."""
    return name.endswith(".gguf") and "mmproj" not in name.lower()


def kv_of(meta):
    """The metadata a census keeps: `general.*` and the layer and expert counts, a string under 200 characters or a
    scalar; an array is left out."""
    kv = {}
    for k, v in meta.items():
        if not (k.startswith("general.") or k.endswith(KV_SUFFIXES)):
            continue
        if isinstance(v, bytes):
            v = v.decode("utf-8", "replace")
            if len(v) >= 200:
                continue
        elif v is None:
            continue
        kv[k] = v
    return kv


def grouped(tensors):
    """(top, blk) of [(name, type name, dims)]: a tensor outside the blocks -> [type, dims]; the rest keyed by the
    name without `blk.<i>.`, then by type, then the sorted layers."""
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    top = {}
    for name, ty, dims in tensors:
        m = BLOCK.match(name)
        if m:
            by[m.group(2)][ty].append(int(m.group(1)))
        else:
            top[name] = [ty, dims]
    return top, {k: {ty: sorted(ls) for ty, ls in v.items()} for k, v in sorted(by.items())}


def stub_head(local, size):
    """The header bytes of the prefix file `local` (of a file of `size` bytes) up to the tensor data, zero-padded to
    `general.alignment`."""
    start = hf.gguf_ranges().header(local, size=size)[2]
    with open(local, "rb") as f:
        head = f.read(start)
    return head + bytes(start - len(head))


def stub_path(stubs, repo, path):
    """DIR/<repo>/<file>; a hub path that is not a plain relative one is a Red."""
    parts = path.split("/")
    if path.startswith("/") or any(p in ("", ".", "..") for p in parts):
        raise hf.Red(f"{repo}: the file path {path!r} is not a plain relative path, so no stub is written for it")
    return os.path.join(stubs, repo, *parts)


def write_stub(fn, head, size):
    """`head` at the start of `fn`, the file extended to `size` bytes (sparse, every other byte zero)."""
    if len(head) > size:
        raise hf.Red(f"{fn}: the header and its padding take {len(head)} bytes, more than the file's {size}")
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    with open(fn, "wb") as w:
        w.write(head)
        w.truncate(size)


def out_name(repo, name):
    return f"{repo.replace('/', '__')}__{os.path.basename(name)}.json"


def read_shard(hub, repo, entry, scratch, want_head):
    """(kv, [(tensor name, type name, dims)], stub head or None) of one shard, its header read through `hub`."""
    path, size = entry.path, entry.size
    local = os.path.join(scratch, "header.bin")
    try:
        meta, tensors, got = hf.read_header(lambda a, b: hub.range(repo, path, a, b), local, size, HEADER_BUDGET)
        if meta is None:
            raise hf.Red(f"the header is unfinished at {got} bytes, the most this reads")
        named = [(name, hf.ggml_type_name(ty, path), dims) for name, ty, dims, _ in tensors]
        return kv_of(meta), named, (stub_head(local, size) if want_head else None)
    except hf.Red as e:
        if path in str(e):
            raise
        raise hf.Red(f"{path}: {e}") from e


def census_set(make_hub, repo, s, out, stubs):
    """Read every shard of the set `s` (hf.gguf_sets' dict), write its JSON (and stubs), return its printed line."""
    files = s["files"]
    targets = [stub_path(stubs, repo, e.path) for e in files] if stubs is not None else []
    kv, tensors, heads = {}, [], []
    with tempfile.TemporaryDirectory() as scratch:
        hub = make_hub(scratch)
        for e in files:
            k, t, head = read_shard(hub, repo, e, scratch, stubs is not None)
            kv.update(k)
            tensors += t
            heads.append(head)
    for fn, head, e in zip(targets, heads, files):
        write_stub(fn, head, e.size)
    top, blk = grouped(tensors)
    rec = {"repo": repo, "set": s["name"], "shards": [e.path for e in files], "sizes": {e.path: e.size for e in files},
           "kv": kv, "top": top, "blk": blk}
    with open(os.path.join(out, out_name(repo, s["name"])), "w") as f:
        json.dump(rec, f, indent=1)
    return (f"{repo} {os.path.basename(s['name'])} arch={kv.get('general.architecture')} tensors={len(tensors)} "
            f"shards={len(files)}")


def one_line(e):
    return " ".join(str(e).split())


def run(out, stubs, repos, make_hub=hf.Hub):
    """Census `repos` into `out`; the exit status."""
    os.makedirs(out, exist_ok=True)
    hf.gguf_ranges()  # loaded here, before the pool's threads ask
    failed, jobs, claimed = 0, [], {}
    with tempfile.TemporaryDirectory() as scratch:
        hub = make_hub(scratch)
        for repo in repos:
            try:
                sets = hf.gguf_sets(repo, hub.listing(repo, follow=False), census_file)
                if not sets:
                    raise hf.Red(f"{repo} holds no GGUF file this reads")
            except (hf.Red, hf.Refuse) as e:
                print(f"{repo}: listing failed: {one_line(e)}", flush=True)
                failed += 1
                continue
            for s in sets:
                fn = out_name(repo, s["name"])
                if fn in claimed:
                    print(f"{repo} {s['name']}: FAILED its output file {fn} is the one of the set {claimed[fn]}",
                          flush=True)
                    failed += 1
                    continue
                claimed[fn] = s["name"]
                jobs.append((repo, s))
    with concurrent.futures.ThreadPoolExecutor(POOL) as ex:
        futs = {ex.submit(census_set, make_hub, repo, s, out, stubs): (repo, s["name"]) for repo, s in jobs}
        for fu in concurrent.futures.as_completed(futs):
            try:
                print(fu.result(), flush=True)
            except (hf.Red, hf.Refuse, OSError) as e:
                print(f"{futs[fu][0]} {futs[fu][1]}: FAILED {one_line(e)}", flush=True)
                failed += 1
            except Exception as e:
                print(f"{futs[fu][0]} {futs[fu][1]}: FAILED {type(e).__name__}: {one_line(e)}", flush=True)
                failed += 1
    return 1 if failed else 0


# ------------------------------------------------------------------------------------------------
# The offline test: synthetic headers, an in-memory hub, a stub curl.

def kv_string(key, value):
    k, v = key.encode(), value.encode()
    return struct.pack("<Q", len(k)) + k + struct.pack("<I", 8) + struct.pack("<Q", len(v)) + v


def kv_u32(key, value):
    k = key.encode()
    return struct.pack("<Q", len(k)) + k + struct.pack("<I", 4) + struct.pack("<I", value)


def kv_strings(key, values):
    k = key.encode()
    body = b"".join(struct.pack("<Q", len(v)) + v for v in (s.encode() for s in values))
    return struct.pack("<Q", len(k)) + k + struct.pack("<I", 9) + struct.pack("<I", 8) + struct.pack("<Q", len(values)) + body


def tensor_infos_len(tensors):
    return sum(8 + len(name.encode()) + 4 + 8 * len(dims) + 4 + 8 for name, _, dims, _ in tensors)


class FakeHub:
    """The hub's two reads, over files held in memory: {repo: {path: bytes}}; a repo in `paged` has a next page."""

    def __init__(self, repos, paged=()):
        self.repos, self.paged = repos, set(paged)

    def listing(self, repo, follow=True):
        if repo in self.paged and not follow:
            raise hf.Red(f"{repo}: the listing has a next page; this reader takes one")
        return [hf.Entry(p, len(b), "sha256", "0" * 64) for p, b in self.repos[repo].items()]

    def range(self, repo, path, start, end):
        data = self.repos[repo][path][start:end]
        if len(data) != end - start:
            raise hf.Red(f"{repo}: the range {start}-{end - 1} of {path} is past its {len(self.repos[repo][path])} bytes")
        return data


def self_test():
    ok = True

    def check(name, cond, detail=""):
        """`cond` is a value, or a function returning one that may raise: an error is a FAIL naming it."""
        nonlocal ok
        if callable(cond):
            try:
                cond = cond()
            except Exception as e:
                cond, detail = False, f"{type(e).__name__}: {e}"
        print(("ok " if cond else "FAIL ") + name + ("" if cond else f": {detail}"))
        ok = ok and bool(cond)

    g = hf.gguf_ranges()
    align = 64
    kvs = [kv_string("general.architecture", "tst"), kv_u32("general.alignment", align), kv_u32("tst.block_count", 12),
           kv_u32("tst.expert_count", 8), kv_u32("tst.nextn_predict_layers", 1), kv_u32("tst.context_length", 1024),
           kv_string("general.name", "x" * 200), kv_string("tokenizer.ggml.model", "gpt2"),
           kv_strings("general.tags", ["a", "b"])]
    blob = b"\7" * 8
    layers = [("blk.10.attn_q.weight", 12), ("blk.2.attn_q.weight", 12), ("blk.2.ffn_down.weight", 14),
              ("blk.10.ffn_down.weight", 14), ("blk.2.ffn_up.weight", 12), ("blk.10.ffn_up.weight", 14)]
    tensors = [("token_embd.weight", 0, [8, 3], blob)] + [(n, ty, [256, 2], blob) for n, ty in layers]
    want_kv = {"general.architecture": "tst", "general.alignment": align, "tst.block_count": 12,
               "tst.expert_count": 8, "tst.nextn_predict_layers": 1}
    want_top = {"token_embd.weight": ["F32", [8, 3]]}
    want_blk = {"attn_q.weight": {"Q4_K": [2, 10]}, "ffn_down.weight": {"Q6_K": [2, 10]},
                "ffn_up.weight": {"Q4_K": [2], "Q6_K": [10]}}

    with tempfile.TemporaryDirectory() as tmp:
        def build(name, kv, ts):
            path = os.path.join(tmp, name)
            g.write_gguf(path, kv, ts, align=align)
            with open(path, "rb") as f:
                return f.read()

        data = build("good.gguf", kvs, tensors)
        tell = 4 + 4 + 8 + 8 + sum(len(k) for k in kvs) + tensor_infos_len(tensors)
        entry = hf.Entry("d/good.gguf", len(data), "sha256", "0" * 64)
        hub = FakeHub({"o/r": {"d/good.gguf": data}})

        def shard(e=entry, want_head=True, h=hub):
            return read_shard(h, "o/r", e, tmp, want_head)

        got_kv, got_tensors, head = shard()
        check("the kv keeps general.*, the layer and expert counts and a short string, and drops the rest",
              got_kv == want_kv and list(got_kv) == list(want_kv), str(got_kv))
        check("a long string, an array and a key outside the filter are not kept",
              "general.name" not in got_kv and "general.tags" not in got_kv and "tst.context_length" not in got_kv)
        check("each tensor is read with its ggml type name, in file order",
              [(n, t) for n, t, _ in got_tensors] == [("token_embd.weight", "F32")] + [
                  (n, {12: "Q4_K", 14: "Q6_K"}[ty]) for n, ty in layers], str(got_tensors))
        top, blk = grouped(got_tensors)
        check("a tensor outside the blocks is top, keyed by its name with [type, dims]", top == want_top, str(top))
        check("blk is keyed by the name without blk.<i>., then type, then the layers sorted as numbers, keys sorted",
              blk == want_blk and list(blk) == sorted(blk) and blk["attn_q.weight"]["Q4_K"] == [2, 10], str(blk))

        bad = build("bad.gguf", kvs, tensors + [("blk.3.attn_k.weight", 77, [256, 2], blob)])
        bad_entry = hf.Entry("d/bad.gguf", len(bad), "sha256", "0" * 64)
        check("a type id the table does not name is a named error, never a placeholder",
              lambda: _raises(lambda: shard(bad_entry, h=FakeHub({"o/r": {"d/bad.gguf": bad}})), hf.Red,
                              "ggml type id 77 in d/bad.gguf: not in the type table"))
        check("the type table names every id gguf-ranges.py sizes, one name each",
              set(g.QUANT) <= set(hf.TYPE_NAMES) and len(set(hf.TYPE_NAMES.values())) == len(hf.TYPE_NAMES),
              str(sorted(set(g.QUANT) - set(hf.TYPE_NAMES))))

        # -- the stub
        check("the stub head is the header up to the aligned tensor data, GGUF first",
              head[:4] == b"GGUF" and len(head) % align == 0 and tell <= len(head) < tell + align
              and head == data[:len(head)] and len(head) < len(data), f"{len(head)} {tell} {len(data)}")
        short = os.path.join(tmp, "prefix.bin")
        with open(short, "wb") as f:
            f.write(data[:tell])
        check("a prefix that ends inside the padding is padded with zeros to the tensor data",
              lambda: stub_head(short, len(data)) == head, str(len(head)))
        stub = os.path.join(tmp, "stubs", "o", "r", "d", "good.gguf")
        write_stub(stub, head, len(data))
        with open(stub, "rb") as f:
            raw = f.read()
        check("the stub has the shard's size, the head, then zeros", len(raw) == len(data) and raw[:len(head)] == head
              and not any(raw[len(head):]) and os.path.getsize(stub) == len(data), f"{len(raw)} {len(data)}")
        m2, t2, start, size2, _ = g.header(stub)
        check("the GGUF reader opens the stub with every tensor in place", (m2, t2) == g.header(os.path.join(tmp, "good.gguf"))[:2]
              and start == len(head) and size2 == len(data))
        check("a head longer than the file is refused", lambda: _raises(lambda: write_stub(stub, head, len(head) - 1),
                                                                         hf.Red, "more than the file's"))
        check("a hub path that climbs out of the stub directory is refused",
              _raises(lambda: stub_path(tmp, "o/r", "../x.gguf"), hf.Red, "not a plain relative path")
              and _raises(lambda: stub_path(tmp, "o/r", "/x.gguf"), hf.Red, "not a plain relative path")
              and stub_path(tmp, "o/r", "d/x.gguf") == os.path.join(tmp, "o/r/d/x.gguf"))

        # -- the sets and the whole run, over an in-memory hub
        second = build("two.gguf", [kv_string("general.architecture", "tst"), kv_u32("tst.block_count", 12)],
                       [("blk.11.attn_q.weight", 12, [256, 2], blob), ("output.weight", 8, [8, 3], blob)])
        repos = {
            "o/r": {"m-00001-of-00002.gguf": data, "m-00002-of-00002.gguf": second, "solo.gguf": data,
                    "mtp-draft.gguf": data, "mmproj-m.gguf": data, "README.md": b"x"},
            "o/bad": {"bad.gguf": bad, "ok.gguf": data},
            "o/dup": {"a/x.gguf": data, "b/x.gguf": data},
            "o/none": {"README.md": b"x"},
            "o/paged": {"p.gguf": data},
        }
        fake = FakeHub(repos, paged=["o/paged"])
        check("the census keeps a draft head's file and leaves an mmproj file out",
              census_file("mtp-x.gguf") and census_file("dflash-x.gguf") and census_file("imatrix.gguf")
              and not census_file("mmproj-x.gguf") and not census_file("MMPROJ-x.gguf") and not census_file("x.bin"))

        def census(repo_list, stubs=None):
            out = os.path.join(tmp, "out")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = run(out, stubs, repo_list, lambda scratch: fake)
            return rc, sorted(buf.getvalue().splitlines()), out

        rc, lines, out = census(["o/r"], os.path.join(tmp, "st"))
        check("a run prints one line per set: the split set once, the draft head's file kept, mmproj and README left out",
              rc == 0 and lines == sorted([
                  "o/r m arch=tst tensors=9 shards=2", "o/r solo arch=tst tensors=7 shards=1",
                  "o/r mtp-draft arch=tst tensors=7 shards=1"]), f"{rc} {lines}")
        rec = json.load(open(os.path.join(out, "o__r__m.json")))
        check("its JSON has census's keys, the shards in order, the sizes, and the shards' tensors grouped together",
              list(rec) == ["repo", "set", "shards", "sizes", "kv", "top", "blk"] and rec["set"] == "m"
              and rec["shards"] == ["m-00001-of-00002.gguf", "m-00002-of-00002.gguf"]
              and rec["sizes"] == {"m-00001-of-00002.gguf": len(data), "m-00002-of-00002.gguf": len(second)}
              and rec["top"] == {"token_embd.weight": ["F32", [8, 3]], "output.weight": ["Q8_0", [8, 3]]}
              and rec["blk"]["attn_q.weight"] == {"Q4_K": [2, 10, 11]} and rec["kv"]["tst.block_count"] == 12,
              json.dumps(rec)[:300])
        check("--stubs wrote a sparse-size stub of every shard at DIR/<repo>/<file>", all(
            os.path.getsize(os.path.join(tmp, "st", "o", "r", f)) == n for f, n in (
                ("m-00001-of-00002.gguf", len(data)), ("m-00002-of-00002.gguf", len(second)), ("solo.gguf", len(data)))))
        rc, lines, out = census(["o/bad"], os.path.join(tmp, "st2"))
        check("a set with an unnamed type id fails by name, ends the run with 1, and writes no JSON and no stub",
              rc == 1 and "o/bad bad: FAILED ggml type id 77 in bad.gguf: not in the type table" in lines
              and "o/bad ok arch=tst tensors=7 shards=1" in lines and not os.path.exists(os.path.join(out, "o__bad__bad.json"))
              and not os.path.exists(os.path.join(tmp, "st2", "o", "bad", "bad.gguf")), f"{rc} {lines}")
        rc, lines, _ = census(["o/dup"])
        check("two sets that would write one output file fail the second by name",
              rc == 1 and len(lines) == 2 and any("FAILED its output file o__dup__x.json is the one of the set" in l
                                                  for l in lines), f"{rc} {lines}")
        rc, lines, _ = census(["o/none"])
        check("a repo with no GGUF file, and a listing with a next page, are named failures",
              rc == 1 and lines == ["o/none: listing failed: o/none holds no GGUF file this reads"]
              and census(["o/paged"])[1] == ["o/paged: listing failed: o/paged: the listing has a next page; this "
                                             "reader takes one"], f"{rc} {lines}")

        # -- Hub.listing(follow=False), through a stub curl whose answer carries a Link header or not
        stubdir = os.path.join(tmp, "bin")
        os.makedirs(stubdir)
        with open(os.path.join(stubdir, "curl"), "w") as f:
            f.write('#!/bin/sh\nwhile [ $# -gt 0 ]; do\n  if [ "$1" = "--dump-header" ]; then\n'
                    '    if [ -n "$STUB_NEXT" ]; then printf \'Link: <https://h/next>; rel="next"\\r\\n\' > "$2"; '
                    'else : > "$2"; fi\n  fi\n  shift\ndone\necho \'[]\'\n')
        os.chmod(os.path.join(stubdir, "curl"), 0o755)
        saved = dict(os.environ)
        try:
            os.environ.update({"PATH": stubdir, "STUB_NEXT": "1"})
            paged = _raises(lambda: hf.Hub(tmp).listing("o/r", follow=False), hf.Red, "has a next page")
            os.environ["STUB_NEXT"] = ""
            whole = hf.Hub(tmp).listing("o/r", follow=False)
        finally:
            os.environ.clear()
            os.environ.update(saved)
        check("a reader that takes one page refuses a next page by name, and takes a listing that has none",
              paged and whole == [])
    return ok


def _raises(fn, exc, text):
    try:
        fn()
    except exc as e:
        return text in str(e)
    return False


def main(argv):
    if argv == ["--self-test"]:
        try:
            return 0 if self_test() else 1
        except Exception as e:
            print(f"FAIL the self-test stopped on {type(e).__name__}: {e}")
            return 1
    args, stubs = list(argv), None
    if "--stubs" in args:
        at = args.index("--stubs")
        if at + 1 >= len(args):
            print(USAGE, file=sys.stderr)
            return 64
        stubs = args[at + 1]
        del args[at:at + 2]
    if len(args) < 2 or any(a.startswith("-") for a in args):
        print(USAGE, file=sys.stderr)
        return 64
    out, repos = args[0], args[1:]
    for r in repos:
        if not REPO.fullmatch(r) or ".." in r.split("/"):
            print(f"type-census.py: {r!r} is not <owner>/<name>", file=sys.stderr)
            return 64
    return run(out, stubs, repos)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
