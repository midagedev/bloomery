#!/usr/bin/env python3
"""A router rank file for the expert planner, from router sets (tools/ref/router_trace.cpp).

    tools/ref/router-rank.py rank <set> [<set>...] --out FILE [--data DIR] [--layers 2-39]
    tools/ref/router-rank.py --self-test

A set is a directory router_trace wrote (counts.tsv, topk-<layer>.u16, MANIFEST.tsv), read through
tools/ref/router-coverage.py: a set whose manifest has no `# complete` trailer is refused. With
--data DIR each <set> is the name of a directory under DIR; without it, a path.

rank  Writes FILE, the input of BLOOMERY_EXPERT_RANK (crates/placement/src/placement/rank.rs): one
      TSV, a header line

          # router-rank<TAB>n_expert=<N><TAB>n_used=<U><TAB>sets=<a,b><TAB>positions=<total>

      then one row `<layer><TAB><id>,<id>,...` per layer, ascending: every expert id of the layer
      exactly once, the hottest first, ties to the lower id. The sets' counts are pooled (a layer's
      count of an expert is the sum over the sets of the tokens that selected it), `positions` is
      their tokens in all. Every set must hold the same number of experts and of experts a token
      selects. The layers are --layers (ranges `a-b` inclusive), else every layer the sets have,
      which then must be the same list in every set. Refused by name, nothing written: a set
      without its trailer, a set twice, a layer a set lacks, sets of other shapes, a topk file with
      an id past n_expert, counts.tsv that disagree with topk. FILE is written whole or not at all.
"""
import importlib.util
import io
import os
import sys
import tempfile
from array import array
from contextlib import redirect_stderr, redirect_stdout

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("router_coverage", os.path.join(_here, "router-coverage.py"))
rc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rc)

SetError = rc.SetError
HEADER = "# router-rank"


def set_name(arg):
    return os.path.basename(os.path.normpath(arg))


def read_sets(args, data):
    """The sets named by `args`, read through router-coverage's reader; a set twice is refused."""
    if not args:
        raise SetError("rank: at least one set")
    sets, names = [], []
    for a in args:
        d = os.path.join(data, a) if data is not None else a
        name = set_name(d)
        if name in names:
            raise SetError(f"rank: set {name} is given twice: its counts would be pooled twice")
        names.append(name)
        sets.append(rc.read_manifest(d))
    first = sets[0]
    for name, s in zip(names, sets):
        if (s["n_expert"], s["n_used"]) != (first["n_expert"], first["n_used"]):
            raise SetError(
                f"rank: set {name} has {s['n_expert']} experts and selects {s['n_used']}, "
                f"set {names[0]} has {first['n_expert']} and {first['n_used']}")
    return names, sets


def rank_layers(names, sets, keep):
    """The layers to rank: `keep` when given, else the one list every set has; a layer a set lacks is
    refused."""
    if keep is None:
        for name, s in zip(names, sets):
            if s["layers"] != sets[0]["layers"]:
                raise SetError(
                    f"rank: set {name} holds layers {s['layers']} and set {names[0]} {sets[0]['layers']}: "
                    f"name the layers to rank with --layers")
        keep = list(sets[0]["layers"])
    if not keep or len(set(keep)) != len(keep):
        raise SetError("rank: --layers names no layer, or a layer twice")
    for name, s in zip(names, sets):
        for layer in keep:
            if layer not in s["layers"]:
                raise SetError(f"rank: set {name} has no layer {layer}")
    return sorted(keep)


def layer_counts(s, layer):
    """The layer's selections per expert, counted from its topk file and held to counts.tsv."""
    E, T, K = s["n_expert"], s["tokens"], s["n_used"]
    ids = rc.read_topk(s, layer)
    over = [e for e in ids if e >= E]
    if over:
        raise SetError(f"{s['dir']}: layer {layer} has id {over[0]}, not below n_expert {E}")
    counted = rc.count(ids, E, 0, T, K)
    if counted != s["counts"][layer]:
        raise SetError(f"{s['dir']}: layer {layer} counts.tsv does not match topk-{layer}.u16")
    return counted


def rank(args, data, keep, out):
    """Write the rank file of the sets `args` to `out`; returns its header line."""
    names, sets = read_sets(args, data)
    layers = rank_layers(names, sets, keep)
    for s in sets:
        s["counts"] = rc.read_counts(s)
    E = sets[0]["n_expert"]
    rows = []
    for layer in layers:
        pooled = [0] * E
        for s in sets:
            for e, c in enumerate(layer_counts(s, layer)):
                pooled[e] += c
        order = rc.hot(pooled, E)
        assert sorted(order) == list(range(E))
        rows.append(f"{layer}\t{','.join(str(e) for e in order)}\n")
    head = (f"{HEADER}\tn_expert={E}\tn_used={sets[0]['n_used']}\tsets={','.join(names)}"
            f"\tpositions={sum(s['tokens'] for s in sets)}")
    tmp = f"{out}.tmp.{os.getpid()}"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(head + "\n")
            f.writelines(rows)
        os.replace(tmp, out)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return head


def read_rank(path):
    """The header fields and the layers' id lists of a rank file, as the planner's reader takes it."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    assert text.endswith("\n"), "a rank file ends with a newline"
    head, *rows = text[:-1].split("\n")
    first, *fields = head.split("\t")
    assert first == HEADER, head
    meta = dict(f.split("=", 1) for f in fields)
    layers = {}
    for row in rows:
        layer, ids = row.split("\t")
        assert int(layer) not in layers
        layers[int(layer)] = [int(e) for e in ids.split(",")]
    return meta, layers


def parse_args(argv):
    sets, opts = [], {}
    it = iter(argv)
    for a in it:
        if a in ("--out", "--data", "--layers"):
            if a in opts:
                raise SetError(f"rank: {a} supplied more than once")
            v = next(it, None)
            if v is None:
                raise SetError(f"rank: {a} takes a value")
            opts[a] = v
        elif a.startswith("-"):
            raise SetError(f"rank: unknown flag {a}")
        else:
            sets.append(a)
    if "--out" not in opts:
        raise SetError("rank: --out FILE names where the rank is written")
    keep = rc.layer_list(opts["--layers"]) if "--layers" in opts else None
    return sets, opts.get("--data"), keep, opts["--out"]


def self_test():
    """Synthetic sets with known orders: the pooled order, ties, the file's shape and each refusal."""
    E, K = 8, 2

    def make(tmp, name, rows_by_layer, complete=True, n_expert=E):
        d = os.path.join(tmp, name)
        os.mkdir(d)
        T = len(next(iter(rows_by_layer.values())))
        with open(os.path.join(d, "counts.tsv"), "w") as f:
            f.write("# layer\texpert\tcount\n")
            for l, rows in rows_by_layer.items():
                c = [0] * n_expert
                for r in rows:
                    for e in r:
                        c[e] += 1
                for e in range(n_expert):
                    f.write(f"{l}\t{e}\t{c[e]}\n")
                with open(os.path.join(d, f"topk-{l}.u16"), "wb") as g:
                    g.write(array("H", [e for r in rows for e in r]).tobytes())
        with open(os.path.join(d, "MANIFEST.tsv"), "w") as f:
            f.write("# router_trace — self-test\n# tokens\t%d\n# n_expert\t%d\n# n_expert_used\t%d\n"
                    % (T, n_expert, len(next(iter(rows_by_layer.values()))[0])))
            f.write(f"# model\t/models/q3/m-00001-of-00002.gguf\n# model_file\tm.gguf\n# build\tb\n# ids\t{d}/ids\n")
            f.write("# layer\tlayer\tsource\tproducer\ttokens\tid_sum\tignored\tfile\n")
            for l in rows_by_layer:
                f.write(f"layer\t{l}\tVIEW-of-ARGSORT\tp\t{T}\t0\t0\ttopk-{l}.u16\n")
            if complete:
                f.write(f"# complete\t{T}\t{len(rows_by_layer)}\n")
        with open(os.path.join(d, "ids"), "w") as f:
            f.write("\n".join(str(t) for t in range(T)) + "\n")
        return d

    def refused(why, args, data, keep):
        never = os.path.join(tmp, f"never-{why.replace(' ', '-')}.tsv")
        try:
            rank(args, data, keep, never)
        except SetError as e:
            assert why in str(e), (why, str(e))
            assert not os.path.exists(never), "a refused rank wrote a file"
            assert [f for f in os.listdir(tmp) if ".tmp." in f] == [], "a refused rank left a temp file"
            return
        raise AssertionError(f"expected a refusal naming {why!r}")

    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "rank.tsv")
        # layer 2: expert 5 takes 6 selections, 1 takes 4, 3 and 7 one each (a tie: 3 first), the rest
        # none; layer 3 selects experts 0 and 1 alone, six each.
        a2 = [[5, 1]] * 4 + [[5, 3], [5, 7]]
        a3 = [[0, 1]] * 6
        a = make(tmp, "a", {2: a2, 3: a3})
        head = rank([a], None, None, out)
        assert head == "# router-rank\tn_expert=8\tn_used=2\tsets=a\tpositions=6", head
        meta, layers = read_rank(out)
        assert meta == {"n_expert": "8", "n_used": "2", "sets": "a", "positions": "6"}, meta
        assert sorted(layers) == [2, 3]
        for l, ids in layers.items():
            assert sorted(ids) == list(range(E)), (l, ids)
        assert layers[2] == [5, 1, 3, 7, 0, 2, 4, 6], layers[2]  # hottest first, ties and the idle to the lower id
        assert layers[3] == [0, 1, 2, 3, 4, 5, 6, 7], layers[3]
        assert open(out).readline().rstrip("\n") == head
        assert [f for f in os.listdir(tmp) if ".tmp." in f] == [], "the temp file is gone"
        # two sets pool: b alone puts 7 and 6 first; pooled the counts add (7: 1 + 6, 5: 6 + 0, 6: 0 + 6)
        b = make(tmp, "b", {2: [[7, 6]] * 6, 3: a3})
        head = rank([a, b], None, [2], out)
        assert head == "# router-rank\tn_expert=8\tn_used=2\tsets=a,b\tpositions=12", head
        _, layers = read_rank(out)
        assert list(layers) == [2] and layers[2][:4] == [7, 5, 6, 1], layers
        # --data joins names to its directory, and a set name is its directory's last part
        head = rank(["a", "b"], tmp, [2], out)
        assert "sets=a,b" in head
        # refusals, each naming its cause, none leaving a file
        cut = make(tmp, "cut", {2: a2}, complete=False)
        refused("complete", [cut], None, None)
        refused("given twice", [a, a], None, None)
        refused("no layer 4", [a, b], None, [2, 3, 4])
        refused("name the layers", [a, make(tmp, "c", {2: a2})], None, None)
        refused("at least one set", [], None, None)
        wide = make(tmp, "wide", {2: [[9, 1]] * 2}, n_expert=16)
        refused("16 experts", [a, wide], None, None)
        refused("names no layer", [a], None, [])
        # topk and counts.tsv must agree, and no id may pass n_expert
        skew = make(tmp, "skew", {2: [[0, 1]] * 2})
        with open(os.path.join(skew, "topk-2.u16"), "wb") as f:
            f.write(array("H", [2, 3, 2, 3]).tobytes())
        refused("does not match", [skew], None, None)
        past = make(tmp, "past", {2: [[0, 1]] * 2})
        with open(os.path.join(past, "topk-2.u16"), "wb") as f:
            f.write(array("H", [0, 9, 0, 9]).tobytes())
        refused("not below n_expert", [past], None, None)
        # the command line: flags, a missing --out, a missing value
        assert parse_args(["a", "b", "--out", "o", "--layers", "2-4,7"]) == (["a", "b"], None, [2, 3, 4, 7], "o")
        for bad in (["a"], ["a", "--out"], ["a", "--out", "o", "--out", "p"], ["a", "--out", "o", "--x"]):
            try:
                parse_args(bad)
            except SetError:
                continue
            raise AssertionError(f"{bad} was taken")
        with redirect_stdout(io.StringIO()) as shown, redirect_stderr(io.StringIO()):
            assert main(["rank", a, "--out", out]) == 0
            assert main(["rank", cut, "--out", out]) == 1
            assert main(["nonsense"]) == 64
        assert shown.getvalue().splitlines()[-1].startswith("# router-rank\tn_expert=8"), shown.getvalue()
    print("router-rank --self-test: ok")
    return 0


def main(argv):
    if argv[:1] == ["--self-test"]:
        return self_test()
    if argv[:1] == ["rank"]:
        try:
            sets, data, keep, out = parse_args(argv[1:])
            head = rank(sets, data, keep, out)
        except SetError as e:
            print(f"router-rank: {e}", file=sys.stderr)
            return 1
        print(f"router-rank: wrote {out}")
        print(head)
        return 0
    print(__doc__, file=sys.stderr)
    return 64


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
