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

Every read refuses by name, never defaults: a call row whose columns do not parse, a slots file of another
length than the set's positions x n_expert_used or holding a byte other than C, T, H, a contexts row that
disagrees with its call row (first, prompt_call, end), a prompt outside prompt_call..end - first, a request
index out of order, or a contexts file with another count of rows than the call rows.

    python3 tools/bloomery/route_trace.py check <set dir>   what router-coverage's reader and these read
    python3 tools/bloomery/route_trace.py --self-test

tools/ has no packages: a script imports this file by path, as it imports manifest.py.
"""
import importlib.util
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


def check(set_dir):
    """Open the set through router-coverage's reader and these; one line a part."""
    spec = importlib.util.spec_from_file_location(
        "router_coverage", os.path.join(_here, "..", "ref", "router-coverage.py"))
    rc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rc)
    s = rc.read_manifest(set_dir)
    ids = [rc.read_topk(s, l) for l in s["layers"]]
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


def _write_set(d, calls, tokens, n_layer=2, n_used=2, kinds=b"CH"):
    os.makedirs(d)
    lines = ["# router_trace — self-test", f"# tokens\t{tokens}", "# n_expert\t8", f"# n_expert_used\t{n_used}",
             "# layer\tlayer\ttokens\tfile\tslots"]
    lines += [f"layer\t{l}\t{tokens}\ttopk-{l}.u16\tslots-{l}.u8" for l in range(n_layer)]
    lines.append("# call\tindex\tfirst\tprompt_call\tend\tpos0")
    lines += [f"call\t{i}\t{a}\t{p}\t{e}\t0" for i, (a, p, e) in enumerate(calls)]
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
    print("route_trace: self-test ok")
    return 0


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-test"]:
        sys.exit(self_test())
    if len(sys.argv) == 3 and sys.argv[1] == "check":
        try:
            sys.exit(check(sys.argv[2]))
        except (TraceError, OSError, ValueError) as e:
            print(f"route_trace: {e}", file=sys.stderr)
            sys.exit(1)
    print(__doc__, file=sys.stderr)
    sys.exit(2)
