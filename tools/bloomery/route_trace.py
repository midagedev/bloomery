"""The one reader of a route trace's sidecars (crates/gpu/src/host/route_trace.rs writes the set).

A route trace is a router set — MANIFEST.tsv and topk-<layer>.u16, which tools/ref/router-coverage.py's
read_manifest and read_topk open as they open router_trace's — with what the replay also needs:

    call rows    in MANIFEST.tsv, one per prompt call the engine ran: index, first (its first position),
                 prompt_call (positions its ids ran, one step each), end (the next call's first, or the
                 set's positions), pos0 (the cache position of its first id)
    slots-<layer>.u8   the slot each routed id ran in, the topk file's shape: C the stage card, T the tier
                 card, H the host
    contexts.tsv one row per request, written after the run by the driver (tools/ref/route-trace-chat.py)
                 from its own requests and the call rows: request, first, end, prompt (the positions whose
                 input is a prompt id), prompt_call, prompt_ids, cache, generated, stop, prompt_id, genre,
                 split; header lines `# key<TAB>value`, then `# columns<TAB>name...`, then the rows

Every read refuses by name, never defaults: a call row whose columns do not parse, calls that do not start
at position 0 or leave positions before the first, a slots file of another length than the set's positions
x n_expert_used or holding a byte other than C, T, H, an id at or past n_expert, a contexts row that
disagrees with its call row (first, prompt_call, end), a prompt outside prompt_call..end - first, a request
index out of order, a contexts file with another count of rows than the call rows, or a split other than
learn and held.

The writer rewrites MANIFEST.tsv after every position with `# positions` and writes `# complete` only when
it finishes; a set without `# complete` is refused by every router set reader. Two commands mend a set
whose writer's process is gone:

    trim DIR   cut every layer file to the manifest's positions and say what was cut: a process killed
               between a position's appends and its manifest leaves at most one position more on some
               files. A file shorter than the manifest, one longer by more than a position, or a
               mismatch in a complete set is refused.
    seal DIR   trim, then write `# complete` (and `# sealed`, naming this tool) when every call ran its
               prompt ids: what a driver does once the server it stopped with a signal has exited and
               every request it sent was answered. A set already complete is refused.

    python3 tools/bloomery/route_trace.py check <set dir>   what router-coverage's reader and these read
    python3 tools/bloomery/route_trace.py trim|seal <set dir>
    python3 tools/bloomery/route_trace.py --self-test

tools/ has no packages: a script imports this file by path, as it imports manifest.py.
"""
import contextlib
import importlib.util
import io
import os
import sys
import tempfile
from array import array

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("manifest", os.path.join(_here, "manifest.py"))
manifest = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(manifest)

KINDS = b"CTH"
CALL_COLUMNS = ("first", "prompt_call", "end", "pos0")
CONTEXT_COLUMNS = ("request", "first", "end", "prompt", "prompt_call", "prompt_ids", "cache", "generated",
                   "stop", "prompt_id", "genre", "split")
CONTEXT_INTS = ("request", "first", "end", "prompt", "prompt_call", "prompt_ids", "cache", "generated")
SPLITS = ("learn", "held")


class TraceError(ValueError):
    pass


def read_calls(set_dir):
    """The manifest's call rows in file order, each a dict of CALL_COLUMNS, and its positions (`# tokens`)."""
    try:
        m = manifest.read(set_dir)
        calls = [{k: r.int(k) for k in CALL_COLUMNS} for r in m.rows("call")]
        tokens = int(m.header["tokens"])
    except (manifest.ManifestError, KeyError, ValueError) as e:
        raise TraceError(f"{set_dir}: {e}") from None
    if tokens and not calls:
        raise TraceError(f"{set_dir}: {tokens} positions and no call row: every position belongs to a call")
    if calls and calls[0]["first"] != 0:
        raise TraceError(f"{set_dir}: the first call starts at position {calls[0]['first']}, not 0")
    for i, c in enumerate(calls):
        want_end = calls[i + 1]["first"] if i + 1 < len(calls) else tokens
        if c["end"] != want_end or not c["first"] + c["prompt_call"] <= c["end"]:
            raise TraceError(f"{set_dir}: call {i} {c} is not one of consecutive calls over {tokens} positions")
    return calls, tokens


def read_slots(set_dir, layer, tokens, n_used):
    """Layer `layer`'s slot kinds, tokens x n_used bytes, each one of C, T, H."""
    path = os.path.join(set_dir, f"slots-{layer}.u8")
    with open(path, "rb") as f:
        b = f.read()
    if len(b) != tokens * n_used:
        raise TraceError(f"{path} holds {len(b)} kinds, not tokens x n_expert_used = {tokens * n_used}")
    bad = set(b) - set(KINDS)
    if bad:
        raise TraceError(f"{path} holds {sorted(bad)}, not only C, T, H")
    return b


def read_contexts(set_dir):
    """contexts.tsv's rows as dicts (CONTEXT_INTS as ints), checked against the call rows; its header
    lines as a dict; None and {} when the set has no contexts.tsv."""
    path = os.path.join(set_dir, "contexts.tsv")
    if not os.path.isfile(path):
        return None, {}
    calls, tokens = read_calls(set_dir)
    header, names, rows = {}, None, []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.rstrip("\n")
            at = f"{path}:{n}"
            if line.startswith("# "):
                key, tab, value = line[2:].partition("\t")
                if key == "columns":
                    names = value.split("\t")
                    missing = [c for c in CONTEXT_COLUMNS if c not in names]
                    if missing:
                        raise TraceError(f"{at}: the column line names no {missing}")
                elif tab:
                    header.setdefault(key, value)
                continue
            if not line:
                continue
            if names is None:
                raise TraceError(f"{at}: a row before the # columns line")
            fields = line.split("\t")
            if len(fields) != len(names):
                raise TraceError(f"{at}: {len(fields)} fields, the column line names {len(names)}")
            r = dict(zip(names, fields))
            for k in CONTEXT_INTS:
                try:
                    r[k] = int(r[k])
                except ValueError:
                    raise TraceError(f"{at}: {k} {r[k]!r} is not an integer") from None
            if r["split"] not in SPLITS:
                raise TraceError(f"{at}: split {r['split']!r} is not learn or held")
            rows.append(r)
    if len(rows) != len(calls):
        raise TraceError(f"{path}: {len(rows)} requests, the manifest has {len(calls)} call rows")
    for i, (r, c) in enumerate(zip(rows, calls)):
        if r["request"] != i:
            raise TraceError(f"{path}: row {i} is request {r['request']}")
        for k in ("first", "prompt_call", "end"):
            if r[k] != c[k]:
                raise TraceError(f"{path}: request {i} {k} {r[k]}, the call row's {c[k]}")
        if not r["prompt_call"] <= r["prompt"] <= r["end"] - r["first"]:
            raise TraceError(f"{path}: request {i} prompt {r['prompt']} is outside "
                             f"{r['prompt_call']}..{r['end'] - r['first']}")
    return rows, header


def split_spans(set_dir, split):
    """The [first, end) of every request of `split` in contexts.tsv, in order; refused by name when the set
    has no contexts.tsv (no split to read) or `split` is not learn or held."""
    if split not in SPLITS:
        raise TraceError(f"split {split!r} is not learn or held")
    rows, _ = read_contexts(set_dir)
    if rows is None:
        raise TraceError(f"{set_dir}: no contexts.tsv, so no learn/held split to read")
    return [(r["first"], r["end"]) for r in rows if r["split"] == split]


def _layout(set_dir):
    """The manifest, its positions, n_expert_used and every layer's two files with their lengths in bytes a
    position."""
    try:
        m = manifest.read(set_dir)
        tokens = int(m.header["tokens"])
        k = int(m.header["n_expert_used"])
        layers = [r.int("layer") for r in m.rows("layer")]
    except (manifest.ManifestError, KeyError, ValueError) as e:
        raise TraceError(f"{set_dir}: {e}") from None
    if "positions" in m.header and int(m.header["positions"]) != tokens:
        raise TraceError(f"{set_dir}: # positions {m.header['positions']}, # tokens {tokens}")
    files = [(os.path.join(set_dir, f"topk-{l}.u16"), 2 * k) for l in layers]
    files += [(os.path.join(set_dir, f"slots-{l}.u8"), k) for l in layers]
    return m, tokens, layers, files


def trim(set_dir):
    """Cut every layer file to the manifest's positions (the module doc); returns the line that says what
    was cut."""
    m, tokens, _, files = _layout(set_dir)
    cut = []
    for path, row in files:
        size = os.path.getsize(path)
        want = tokens * row
        if size == want:
            continue
        if m.complete is not None:
            raise TraceError(f"{path}: {size} bytes in a complete set of {tokens} positions ({want} bytes)")
        if size < want or size - want > row:
            raise TraceError(f"{path}: {size} bytes, the manifest's {tokens} positions are {want} bytes and a "
                             f"killed writer leaves at most one position ({row} bytes) more")
        cut.append((path, size - want))
        os.truncate(path, want)
    if not cut:
        return f"trim {set_dir}: every file holds the manifest's {tokens} positions"
    return (f"trim {set_dir}: cut to {tokens} positions: "
            + ", ".join(f"{os.path.basename(p)} -{n} B" for p, n in cut))


def seal(set_dir):
    """trim, then `# complete` for a set whose writer's process is gone (the module doc); returns the lines
    that say so."""
    m, tokens, layers, _ = _layout(set_dir)
    if m.complete is not None:
        raise TraceError(f"{set_dir}: complete already ({m.complete!r})")
    said = trim(set_dir)
    calls, _ = read_calls(set_dir)
    path = os.path.join(set_dir, "MANIFEST.tsv")
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if not text.endswith("\n"):
        raise TraceError(f"{path}: its last line is cut off")
    text += ("# sealed\ttools/bloomery/route_trace.py seal: the writer's process ended without finishing the "
             f"set\n# complete\t{tokens}\t{len(layers)}\n")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)
    return f"{said}; sealed: {tokens} positions in {len(calls)} calls, # complete written"


def check(set_dir):
    """Open the set through router-coverage's reader and these; one line a part."""
    spec = importlib.util.spec_from_file_location(
        "router_coverage", os.path.join(_here, "..", "ref", "router-coverage.py"))
    rc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rc)
    try:
        s = rc.read_manifest(set_dir)
        ids = [rc.read_topk(s, l) for l in s["layers"]]
    except rc.SetError as e:
        raise TraceError(str(e)) from None
    for l, a in zip(s["layers"], ids):
        if len(a) and max(a) >= s["n_expert"]:
            raise TraceError(f"{set_dir}: layer {l} holds id {max(a)}, not below n_expert {s['n_expert']}")
    print(f"router-coverage read_manifest: {s['tokens']} tokens, {len(s['layers'])} layers "
          f"{s['layers'][0]}..{s['layers'][-1]}, {s['n_expert']} experts, top-{s['n_used']}; read_topk: "
          f"{sum(len(a) for a in ids)} ids, max {max((max(a) for a in ids if len(a)), default=0)}")
    kinds = [read_slots(set_dir, l, s["tokens"], s["n_used"]) for l in s["layers"]]
    total = sum(len(k) for k in kinds)
    counts = {chr(c): sum(k.count(c) for k in kinds) for c in KINDS}
    print(f"slots: {total} kinds, " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    calls, _ = read_calls(set_dir)
    print(f"calls: {len(calls)} " + " ".join(f"[{c['first']},{c['end']}) prompt_call {c['prompt_call']} "
                                               f"pos0 {c['pos0']}" for c in calls[:4])
          + (" ..." if len(calls) > 4 else ""))
    rows, header = read_contexts(set_dir)
    if rows is None:
        print("contexts: none (no contexts.tsv)")
    else:
        print(f"contexts: {len(rows)} requests, {sum(r['end'] - r['first'] for r in rows)} positions, "
              f"prompt {sum(r['prompt'] for r in rows)}; build {header.get('build', '?')}")
    return 0


def _write_set(d, calls, tokens, n_layer=2, n_used=2, kinds=b"CH", complete=True, n_expert=8):
    os.makedirs(d)
    lines = ["# router_trace — self-test", f"# tokens\t{tokens}", f"# n_expert\t{n_expert}",
             f"# n_expert_used\t{n_used}", "# layer\tlayer\ttokens\tfile\tslots"]
    lines += [f"layer\t{l}\t{tokens}\ttopk-{l}.u16\tslots-{l}.u8" for l in range(n_layer)]
    lines.append("# call\tindex\tfirst\tprompt_call\tend\tpos0")
    lines += [f"call\t{i}\t{a}\t{p}\t{e}\t0" for i, (a, p, e) in enumerate(calls)]
    lines.append(f"# positions\t{tokens}")
    if complete:
        lines.append(f"# complete\t{tokens}\t{n_layer}")
    with open(os.path.join(d, "MANIFEST.tsv"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    for l in range(n_layer):
        with open(os.path.join(d, f"topk-{l}.u16"), "wb") as f:
            f.write(array("H", [(t + s) % 8 for t in range(tokens) for s in range(n_used)]).tobytes())
        with open(os.path.join(d, f"slots-{l}.u8"), "wb") as f:
            f.write(bytes(kinds[(t + s) % len(kinds)] for t in range(tokens) for s in range(n_used)))


def _write_contexts(d, rows):
    with open(os.path.join(d, "contexts.tsv"), "w", encoding="utf-8") as f:
        f.write("# route-trace contexts — self-test\n# build\tb0\n# columns\t" + "\t".join(CONTEXT_COLUMNS) + "\n")
        for r in rows:
            f.write("\t".join(str(r[c]) for c in CONTEXT_COLUMNS) + "\n")


def _refused(want, fn, *args):
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            fn(*args)
    except TraceError as e:
        assert want in str(e), (want, str(e))
        return
    raise AssertionError(f"accepted: {want}")


def _self_test_mend(root):
    """The call rows' start, the split, the ids' range, trim and seal."""
    d = os.path.join(root, "late")
    _write_set(d, [(1, 3, 9)], 9)
    _refused("the first call starts at position 1, not 0", read_calls, d)
    d = os.path.join(root, "nocall")
    _write_set(d, [], 9)
    _refused("9 positions and no call row", read_calls, d)
    d = os.path.join(root, "wide")
    _write_set(d, [(0, 3, 9)], 9, n_expert=4)
    _refused("layer 0 holds id 7, not below n_expert 4", check, d)

    d = os.path.join(root, "split")
    _write_set(d, [(0, 3, 9), (9, 2, 12)], 12)
    rows = [dict(request=0, first=0, end=9, prompt=4, prompt_call=3, prompt_ids=4, cache=0, generated=6,
                 stop="eos", prompt_id="a", genre="ko", split="learn"),
            dict(request=1, first=9, end=12, prompt=3, prompt_call=2, prompt_ids=3, cache=0, generated=1,
                 stop="eos", prompt_id="b", genre="en", split="held")]
    _refused("no contexts.tsv, so no learn/held split", split_spans, d, "learn")
    _write_contexts(d, rows)
    assert split_spans(d, "learn") == [(0, 9)] and split_spans(d, "held") == [(9, 12)]
    _refused("split 'both' is not learn or held", split_spans, d, "both")
    _write_contexts(d, [rows[0], dict(rows[1], split="test")])
    _refused("split 'test' is not learn or held", read_contexts, d)

    # A killed writer: layer 0's files one position past the manifest, layer 1's not.
    d = os.path.join(root, "killed")
    _write_set(d, [(0, 3, 9), (9, 2, 12)], 12, complete=False)
    for name, extra in (("topk-0.u16", b"\x01\x00\x02\x00"), ("slots-0.u8", b"CH")):
        with open(os.path.join(d, name), "ab") as f:
            f.write(extra)
    said = seal(d)
    assert "topk-0.u16 -4 B, slots-0.u8 -2 B" in said and "12 positions in 2 calls" in said, said
    m = manifest.read(d)
    assert m.complete == "12\t2" and m.header["sealed"].startswith("tools/bloomery/route_trace.py seal"), m.header
    assert os.path.getsize(os.path.join(d, "topk-0.u16")) == 48 and "every file holds" in trim(d)
    _refused("complete already", seal, d)
    with open(os.path.join(d, "slots-1.u8"), "ab") as f:
        f.write(b"C")
    _refused("slots-1.u8: 25 bytes in a complete set of 12 positions", trim, d)
    d = os.path.join(root, "far")
    _write_set(d, [(0, 3, 12)], 12, complete=False)
    with open(os.path.join(d, "topk-1.u16"), "ab") as f:
        f.write(bytes(8))
    _refused("at most one position (4 bytes) more", trim, d)
    d = os.path.join(root, "short")
    _write_set(d, [(0, 3, 12)], 12, complete=False)
    with open(os.path.join(d, "slots-0.u8"), "r+b") as f:
        f.truncate(23)
    _refused("slots-0.u8: 23 bytes", trim, d)
    d = os.path.join(root, "cutcall")
    _write_set(d, [(0, 3, 9), (9, 5, 12)], 12, complete=False)
    _refused("call 1", seal, d)
    assert manifest.read(d).complete is None


def self_test():
    """Two requests of unequal lengths read back; each refusal by name."""
    with tempfile.TemporaryDirectory() as root:
        d = os.path.join(root, "t")
        _write_set(d, [(0, 3, 9), (9, 5, 12)], 12)
        ctx = [dict(request=0, first=0, end=9, prompt=4, prompt_call=3, prompt_ids=4, cache=0, generated=6,
                    stop="eos", prompt_id="ko-01", genre="ko", split="learn"),
               dict(request=1, first=9, end=12, prompt=6, prompt_call=5, prompt_ids=6, cache=0, generated=0,
                    stop="limit", prompt_id="en-01", genre="en", split="held")]
        assert read_contexts(d) == (None, {})
        # request 1's prompt call ran 5 of its positions but the set holds 3 past its first: refused
        # by read_calls before the contexts are read.
        try:
            read_calls(d)
            raise AssertionError("a call longer than its positions was read")
        except TraceError as e:
            assert "is not one of consecutive calls" in str(e), e
        d = os.path.join(root, "u")
        _write_set(d, [(0, 3, 9), (9, 2, 12)], 12)
        ctx[1].update(prompt=3, prompt_call=2, prompt_ids=3, generated=1)
        _write_contexts(d, ctx)
        rows, header = read_contexts(d)
        assert [(r["first"], r["end"], r["prompt"]) for r in rows] == [(0, 9, 4), (9, 12, 3)], rows
        assert header == {"build": "b0"} and rows[1]["genre"] == "en", header
        assert read_slots(d, 1, 12, 2)[:4] == b"CHHC"

        def refused(rows_, want, **edit):
            e_dir = os.path.join(root, f"r{len(os.listdir(root))}")
            _write_set(e_dir, [(0, 3, 9), (9, 2, 12)], 12, **edit)
            _write_contexts(e_dir, rows_)
            try:
                read_contexts(e_dir)
                if "kinds" in edit:
                    read_slots(e_dir, 0, 12, 2)
            except TraceError as e:
                assert want in str(e), (want, str(e))
                return
            raise AssertionError(f"accepted: {want}")

        refused([dict(ctx[0], first=1), ctx[1]], "request 0 first 1, the call row's 0")
        refused([dict(ctx[0], prompt=2), ctx[1]], "prompt 2 is outside 3..9")
        refused([ctx[1], ctx[0]], "row 0 is request 1")
        refused([ctx[0]], "1 requests, the manifest has 2 call rows")
        refused(ctx, "not only C, T, H", kinds=b"CX")
        with open(os.path.join(d, "slots-0.u8"), "ab") as f:
            f.write(b"C")
        try:
            read_slots(d, 0, 12, 2)
            raise AssertionError("a slots file longer than the set was read")
        except TraceError as e:
            assert "holds 25 kinds, not tokens x n_expert_used = 24" in str(e), e
        _self_test_mend(root)
    print("route_trace: self-test ok")
    return 0


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-test"]:
        sys.exit(self_test())
    if len(sys.argv) == 3 and sys.argv[1] in ("check", "trim", "seal"):
        try:
            if sys.argv[1] == "check":
                sys.exit(check(sys.argv[2]))
            print(trim(sys.argv[2]) if sys.argv[1] == "trim" else seal(sys.argv[2]))
            sys.exit(0)
        except (TraceError, OSError, ValueError) as e:
            print(f"route_trace: {e}", file=sys.stderr)
            sys.exit(1)
    print(__doc__, file=sys.stderr)
    sys.exit(2)
