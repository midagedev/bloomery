#!/usr/bin/env python3
"""A hot list for the placement lever BLOOMERY_HOT_LIST, learned from router sets.

    tools/ref/router-hotlist.py <set> [<set>...] --n 64|all --out <file>
    tools/ref/router-hotlist.py --self-test

Each <set> is a router_trace directory (tools/ref/router-coverage.py's docstring has the format); a set
whose manifest has no `# complete` trailer is refused, and so are sets that disagree on n_expert,
n_expert_used, the layer list, the model file or the build. The model file is the `# model` line, the
first shard's full path: two quantizations of one model share their shard names, so the basename
(`# model_file`) cannot tell them apart; a set without `# model` or `# build` is refused by name. Per layer, every selection of every token of every
set is counted together (the union of the sets' traffic), and the layer's experts are ranked hottest
first, ties to the lower id (router-coverage.py's `hot`). The file keeps the first --n of each layer
in that rank order:

    # router-hotlist
    # sets      <dir>:<tokens>,...
    # n         <per-layer count>
    # n_expert  <experts per routed stack>
    # order     rank
    # date      <UTC date>
    # model     the sets' first shard, full path
    # model_file / # build   the sets' own manifest values
    <layer>\t<id>,<id>,...

The placement (crates/model/src/placement/hot_list.rs) takes each layer's first n_l ids, where n_l is
the plan's count for the layer — the plan decides how many, this file decides which — and refuses a
layer that lists fewer than n_l. --n is therefore the largest count any plan may ask for; `all` is
the sets' own n_expert, every expert of a layer ranked.
"""
import datetime
import importlib.util
import os
import sys
import tempfile
from array import array
from collections import Counter

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("router_coverage", os.path.join(_here, "router-coverage.py"))
rc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rc)


def learn(set_dirs, n):
    sets = [rc.read_manifest(d) for d in set_dirs]
    first = sets[0]
    for s in sets[1:]:
        for key in ("n_expert", "n_used", "layers"):
            if s[key] != first[key]:
                raise rc.SetError(f"{s['dir']}: {key} {s[key]} is not {first['dir']}'s {first[key]}")
    for s in sets:
        for key in ("model", "build"):
            if s["header"].get(key) is None:
                raise rc.SetError(f"{s['dir']}: its manifest has no `# {key}` line, so what it was traced from is unknown")
            if s["header"][key] != first["header"][key]:
                raise rc.SetError(f"{s['dir']}: `# {key}` {s['header'][key]!r} is not {first['dir']}'s {first['header'][key]!r}")
    E = first["n_expert"]
    if n == "all":
        n = E
    if not 0 < n <= E:
        raise rc.SetError(f"--n {n} is not in 1..{E}")
    lists = {}
    for layer in first["layers"]:
        c = Counter()
        for s in sets:
            c.update(rc.read_topk(s, layer))
        counts = [c.get(e, 0) for e in range(E)]
        if max(c, default=0) >= E:
            raise rc.SetError(f"layer {layer}: an id {max(c)} is not below n_expert {E}")
        lists[layer] = rc.hot(counts, n)
    return sets, lists, n


def write(path, sets, lists, n):
    first = sets[0]
    lines = [
        "# router-hotlist",
        "# sets\t" + ",".join(f"{os.path.basename(os.path.normpath(s['dir']))}:{s['tokens']}" for s in sets),
        f"# n\t{n}",
        f"# n_expert\t{first['n_expert']}",
        "# order\trank",
        "# date\t" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d"),
    ]
    for key in ("model", "model_file", "build"):
        vals = sorted({s["header"].get(key, "?") for s in sets})
        lines.append(f"# {key}\t{','.join(vals)}")
    for layer in sorted(lists):
        lines.append(f"{layer}\t{','.join(str(e) for e in lists[layer])}")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(tmp, path)


def main(argv):
    if argv == ["--self-test"]:
        return self_test()
    sets, n, out = [], None, None
    it = iter(argv)
    for a in it:
        if a == "--n":
            n = next(it)
            n = n if n == "all" else int(n)
        elif a == "--out":
            out = next(it)
        elif a.startswith("-"):
            sys.exit(f"unknown flag {a}")
        else:
            sets.append(a)
    if not sets or n is None or out is None:
        sys.exit(__doc__)
    try:
        s, lists, n = learn(sets, n)
    except rc.SetError as e:
        sys.exit(f"router-hotlist: {e}")
    write(out, s, lists, n)
    print(f"router-hotlist: {len(lists)} layers x {n} ids from {len(s)} sets "
          f"({sum(x['tokens'] for x in s)} tokens) -> {out}")
    return 0


MODEL = "/models/q3/m-00001-of-00002.gguf"


def _fake_set(root, name, layers, rows, n_expert=8, n_used=2, complete=True, model=MODEL):
    d = os.path.join(root, name)
    os.makedirs(d)
    with open(os.path.join(d, "MANIFEST.tsv"), "w", encoding="utf-8") as f:
        f.write("# router_trace — test\n")
        if model is not None:
            f.write(f"# model\t{model}\n")
        f.write("# model_file\tm-00001-of-00002.gguf\n# build\tb0\n")
        f.write(f"# tokens\t{len(rows)}\n# n_expert\t{n_expert}\n# n_expert_used\t{n_used}\n")
        f.write("# layer\tlayer\tsource\tproducer\ttokens\tid_sum\tignored\tfile\n")
        for l in layers:
            f.write(f"layer\t{l}\tx\tx\t{len(rows)}\t0\t0\ttopk-{l}.u16\n")
        if complete:
            f.write(f"# complete\t{len(rows)}\t{len(layers)}\n")
    for l in layers:
        a = array("H", [e for r in rows for e in r])
        if sys.byteorder != "little":
            a.byteswap()
        with open(os.path.join(d, f"topk-{l}.u16"), "wb") as f:
            f.write(a.tobytes())
    return d


def self_test():
    with tempfile.TemporaryDirectory() as root:
        # Set A favours 5 then 3; set B favours 3 then 1: the union ranks 3 (4), 5 (3), 1 (2), then
        # the ties at 1 selection broken by the lower id.
        a = _fake_set(root, "a", [0, 1], [(5, 3), (5, 3), (5, 0)])
        b = _fake_set(root, "b", [0, 1], [(3, 1), (3, 1), (7, 2)])
        sets, lists, n = learn([a, b], 4)
        assert lists[0] == [3, 5, 1, 0] and n == 4, lists[0]
        _, whole, n_all = learn([a, b], "all")
        assert n_all == 8 and len(whole[0]) == 8 and whole[0][:4] == [3, 5, 1, 0], whole[0]
        out = os.path.join(root, "hot.txt")
        write(out, sets, lists, 4)
        text = open(out, encoding="utf-8").read()
        assert "# order\trank\n" in text and "# n_expert\t8\n" in text and "\n0\t3,5,1,0\n" in text, text
        assert f"# model\t{MODEL}\n" in text, text
        # Two quantizations of one model share their shard names: the full path tells them apart, and a
        # set that states no path is refused, never compared as equal.
        other = _fake_set(root, "other", [0, 1], [(3, 1)], model="/models/q3-requant/m-00001-of-00002.gguf")
        for bad, what in ((other, "another file of the same basename"),
                          (_fake_set(root, "nopath", [0, 1], [(3, 1)], model=None), "a set without # model")):
            try:
                learn([a, bad], 2)
                raise AssertionError(f"{what} was accepted")
            except rc.SetError:
                pass
        c = _fake_set(root, "c", [0, 1], [(1, 2)], complete=False)
        try:
            learn([a, c], 2)
            raise AssertionError("an incomplete set was accepted")
        except rc.SetError:
            pass
        d = _fake_set(root, "d", [0, 1], [(1, 2)], n_expert=16)
        try:
            learn([a, d], 2)
            raise AssertionError("sets of another n_expert were accepted")
        except rc.SetError:
            pass
    print("router-hotlist self-test: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
