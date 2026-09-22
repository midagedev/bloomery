#!/usr/bin/env python3
"""How concentrated MoE routing is, per layer, from router sets (tools/ref/router_trace.cpp).

    tools/ref/router-coverage.py coverage <set> [<set>...]
    tools/ref/router-coverage.py transfer <set A> <set B> [--n 64]
    tools/ref/router-coverage.py compare <set A> <set B>
    tools/ref/router-coverage.py oracle <set> <oracle set dir>
    tools/ref/router-coverage.py --self-test

A set is a directory router_trace wrote under $BLOOMERY_DATA/router/: counts.tsv, topk-<layer>.u16
(tokens x n_expert_used ids, little-endian u16, token-major) and MANIFEST.tsv. Every command
refuses a set whose manifest has no `# complete` trailer.

coverage  Per layer: the share of the layer's selections (tokens x n_expert_used) that its n hottest
          experts receive, for n in N, under the uniform share n/n_expert; the same over all layers
          (the mean of the layer shares — every layer has the same number of selections); the top-1
          expert's share of tokens (at most 100 %: an expert is picked once per token at most) and
          the Gini coefficient of the counts (0 = uniform). Two tables:
            in    the hot list and the share come from the same tokens. Biased up: the top n of noisy
                  counts are partly the lucky ones.
            held  the hot list from the first half of the tokens, the share on the second half — what
                  a placement fixed before the traffic would serve, and the noise floor for `transfer`.
transfer  Per layer, with each set's hot-n list learned on all its tokens: the overlap |hot(A) & hot(B)|,
          A's list measured on B's selections beside B's own list on B (in-sample) and B's held-out
          share, and the same the other way round. Does a list learned on one stream serve another?
compare   Two traces of the same tokens (another schedule, ubatch or backend), token by token, per
          layer: the ids equal in rank order, the ids shared as sets, the tokens whose selection is the
          same set, and the first layer where any token's set differs.
oracle    Per layer, the set's ids against the oracle set's integer twin of ffn_moe_topk-<layer>,
          integer for integer in rank order, and as sets. The twin is the one the oracle manifest's
          `tensor` row points at: the logical twin when the row marks a view (its flat file is the
          argsort's head, the first n_expert_used x n_tokens values of token 0's sorted row, not each
          token's top-k), else the flat one. The set's tokens, model file and ik build must be the
          oracle's. Exit 1 on any difference.
"""
import os
import struct
import sys
import tempfile
from array import array

N_LIST = (16, 32, 48, 64, 96, 128, 192)


class SetError(Exception):
    pass


def read_manifest(set_dir):
    path = os.path.join(set_dir, "MANIFEST.tsv")
    header, layers, complete = {}, [], False
    try:
        with open(path, encoding="utf-8") as f:
            first = f.readline()
            if not first.startswith("# router_trace"):
                raise SetError(f"{path} is not a router set manifest")
            for line in f:
                line = line.rstrip("\n")
                if line.startswith("# complete"):
                    complete = True
                elif line.startswith("# "):
                    key, _, val = line[2:].partition("\t")
                    header.setdefault(key, val)
                elif line.startswith("layer\t"):
                    layers.append(int(line.split("\t")[1]))
    except FileNotFoundError:
        raise SetError(f"no manifest at {path}") from None
    if not complete:
        raise SetError(f"{path} has no `# complete` trailer: the run that wrote it did not finish")
    return {
        "dir": set_dir,
        "header": header,
        "layers": layers,
        "tokens": int(header["tokens"]),
        "n_expert": int(header["n_expert"]),
        "n_used": int(header["n_expert_used"]),
    }


def read_counts(s):
    counts = {l: [0] * s["n_expert"] for l in s["layers"]}
    with open(os.path.join(s["dir"], "counts.tsv"), encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            l, e, c = (int(x) for x in line.split("\t"))
            counts[l][e] = c
    for l, c in counts.items():
        if sum(c) != s["tokens"] * s["n_used"]:
            raise SetError(f"{s['dir']}: layer {l} counts sum to {sum(c)}, not tokens x n_expert_used")
    return counts


def read_topk(s, layer):
    a = array("H")
    path = os.path.join(s["dir"], f"topk-{layer}.u16")
    with open(path, "rb") as f:
        a.frombytes(f.read())
    if sys.byteorder != "little":
        a.byteswap()
    if len(a) != s["tokens"] * s["n_used"]:
        raise SetError(f"{path} holds {len(a)} ids, not tokens x n_expert_used = {s['tokens'] * s['n_used']}")
    return a


def count(ids, n_expert, t0, t1, n_used):
    c = [0] * n_expert
    for e in ids[t0 * n_used:t1 * n_used]:
        c[e] += 1
    return c


def hot(counts, n):
    """The n hottest experts, ties broken by the lower id so the list is reproducible."""
    return sorted(range(len(counts)), key=lambda e: (-counts[e], e))[:n]


def share(counts, experts):
    total = sum(counts)
    return sum(counts[e] for e in experts) / total if total else 0.0


def gini(counts):
    x = sorted(counts)
    n, total = len(x), sum(x)
    if total == 0:
        return 0.0
    return 2.0 * sum((i + 1) * v for i, v in enumerate(x)) / (n * total) - (n + 1) / n


def pct(v):
    return f"{100.0 * v:.1f}"


def coverage(set_dirs):
    for d in set_dirs:
        s = read_manifest(d)
        counts = read_counts(s)
        E, T, K = s["n_expert"], s["tokens"], s["n_used"]
        h = s["header"]
        half = T // 2
        print(f"## {d}\n")
        print(f"{T} tokens of {h.get('ids', '?')} (md5 {h.get('ids_md5', '?')}), chunk {h.get('chunk', '?')}, "
              f"{len(s['layers'])} layers, {E} experts, top-{K}. held = hot list from tokens [0, {half}), "
              f"share on [{half}, {T}).\n")
        rows_in, rows_held = [], []
        for l in s["layers"]:
            c = counts[l]
            ids = read_topk(s, l)
            first, second = count(ids, E, 0, half, K), count(ids, E, half, T, K)
            if [x + y for x, y in zip(first, second)] != c:
                raise SetError(f"{d}: layer {l} counts.tsv does not match topk-{l}.u16")
            order = hot(c, E)
            order_first = hot(first, E)
            # in-sample rows carry the top-1 share of tokens and the Gini ahead of the hot-n shares
            rows_in.append((l, [max(c) / T, gini(c)] + [share(c, order[:n]) for n in N_LIST]))
            rows_held.append((l, [share(second, order_first[:n]) for n in N_LIST]))
        n_cols = " | ".join(f"n={n}" for n in N_LIST)
        uniform = " | ".join(pct(n / E) for n in N_LIST)
        tables = (("in-sample", "top-1 % of tokens | Gini | ", f"{pct(K / E)} | 0.000 | ", 2, rows_in),
                  ("held-out", "", "", 0, rows_held))
        for title, extra_head, extra_uniform, n_extra, rows in tables:
            def cells(v):
                return " | ".join(f"{x:.3f}" if i == 1 and n_extra else pct(x) for i, x in enumerate(v))
            print(f"### hot-n share of selections, {title} (%)\n")
            print(f"| layer | {extra_head}{n_cols} |")
            print("|---:|" + "---:|" * (n_extra + len(N_LIST)))
            print(f"| uniform | {extra_uniform}{uniform} |")
            for l, v in rows:
                print(f"| {l} | {cells(v)} |")
            mean = [sum(r[1][i] for r in rows) / len(rows) for i in range(len(rows[0][1]))]
            print(f"| all | {cells(mean)} |\n")


def transfer(dir_a, dir_b, n):
    a, b = read_manifest(dir_a), read_manifest(dir_b)
    if (a["n_expert"], a["n_used"]) != (b["n_expert"], b["n_used"]) or a["layers"] != b["layers"]:
        raise SetError("the two sets route over different experts or layers")
    ca, cb = read_counts(a), read_counts(b)
    E, K = a["n_expert"], a["n_used"]

    def held(s, l):
        ids, T = read_topk(s, l), s["tokens"]
        return share(count(ids, E, T // 2, T, K), hot(count(ids, E, 0, T // 2, K), n))

    print(f"## hot-{n} transfer: A = {dir_a} ({a['tokens']} tokens), B = {dir_b} ({b['tokens']} tokens)\n")
    print(f"Shares are of the evaluated set's selections, %. Uniform: {pct(n / E)}.\n")
    print(f"| layer | overlap of hot-{n} | A-list on B | B-list on B (in) | B held-out | "
          "B-list on A | A-list on A (in) | A held-out |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|")
    sums = [0.0] * 7
    for l in a["layers"]:
        ha, hb = hot(ca[l], n), hot(cb[l], n)
        row = [len(set(ha) & set(hb)), share(cb[l], ha), share(cb[l], hb), held(b, l),
               share(ca[l], hb), share(ca[l], ha), held(a, l)]
        sums = [x + y for x, y in zip(sums, row)]
        print(f"| {l} | {row[0]} | " + " | ".join(pct(v) for v in row[1:]) + " |")
    k = len(a["layers"])
    print(f"| all | {sums[0] / k:.1f} | " + " | ".join(pct(v / k) for v in sums[1:]) + " |\n")


def agreement(a, b, K):
    """(ids equal in rank order, ids shared as sets, tokens whose selection is the same set)."""
    ranked = sum(1 for x, y in zip(a, b) if x == y)
    shared = same = 0
    for t in range(len(a) // K):
        sa, sb = set(a[t * K:(t + 1) * K]), set(b[t * K:(t + 1) * K])
        shared += len(sa & sb)
        same += sa == sb
    return ranked, shared, same


def compare(dir_a, dir_b):
    a, b = read_manifest(dir_a), read_manifest(dir_b)
    if (a["tokens"], a["n_used"], a["layers"]) != (b["tokens"], b["n_used"], b["layers"]):
        raise SetError("the two sets differ in token count, top-k or layers")
    if a["header"].get("ids_md5") != b["header"].get("ids_md5"):
        raise SetError("the two sets were traced from different ids files")
    T, K = a["tokens"], a["n_used"]
    print(f"## compare: A = {dir_a}, B = {dir_b} ({T} tokens, top-{K})\n")
    for key in ("schedule", "n_ubatch", "flags"):
        print(f"- {key}: A `{a['header'].get(key)}`, B `{b['header'].get(key)}`")
    print("\n| layer | ids equal, rank order % | ids shared as sets % | tokens with the same set % |")
    print("|---:|---:|---:|---:|")
    first_diff = None
    for l in a["layers"]:
        ranked, shared, same = agreement(read_topk(a, l), read_topk(b, l), K)
        if same < T and first_diff is None:
            first_diff = l
        print(f"| {l} | {pct(ranked / (T * K))} | {pct(shared / (T * K))} | {pct(same / T)} |")
    print(f"\nfirst layer with a differing set: {first_diff if first_diff is not None else 'none'}\n")


def read_oracle(oracle_dir):
    path = os.path.join(oracle_dir, "MANIFEST.tsv")
    header, tensors, ints, complete = {}, {}, {}, False
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#"):
                key, _, val = line[2:].partition("\t")
                complete |= key == "complete"
                header.setdefault(key, val)
                continue
            p = line.split("\t")
            if p[0] == "tensor":
                tensors[(p[1], int(p[2]))] = p
            elif p[0] == "int" and p[3] == "tensor":
                ints[(p[1], int(p[2]), p[6])] = p
    if not complete:
        raise SetError(f"{path} has no `# complete` trailer")
    return header, tensors, ints


def oracle(set_dir, oracle_dir):
    s = read_manifest(set_dir)
    header, tensors, ints = read_oracle(oracle_dir)
    K, T = s["n_used"], s["tokens"]
    problems = []
    for key in ("model_file", "build"):
        if s["header"].get(key) != header.get(key):
            problems.append(f"{key}: set {s['header'].get(key)!r}, oracle {header.get(key)!r}")
    want = [int(x) for x in header.get("tokens", "").split(",") if x]
    try:
        with open(s["header"]["ids"], encoding="utf-8") as f:
            got = [int(x) for x in f.read().split()][:T]
    except OSError as e:
        got = None
        problems.append(f"cannot read the set's ids file: {e}")
    if got is not None and got != want:
        problems.append(f"tokens: set {got}, oracle {want}")
    if problems:
        for p in problems:
            print(f"oracle: {p}")
        print("oracle: FAIL — the set and the oracle are not the same run's inputs")
        return 1

    total = equal = 0
    bad_layers = []
    for l in s["layers"]:
        name = f"ffn_moe_topk-{l}"
        row = tensors.get((name, 0))
        if row is None:
            print(f"layer {l:2d}: the oracle has no {name}")
            bad_layers.append(l)
            continue
        # tensor row: kind name occurrence type ne0 ne1 ne2 ne3 bytes sum op contig logical src0 src1
        # int row:    int name occurrence of type twin layout count bytes sum absmax file
        ne0, ne1 = int(row[4]), int(row[5])
        layout = "logical" if row[12] == "1" else "flat"
        twin = ints.get((name, 0, layout))
        if twin is None or (ne0, ne1) != (K, T):
            print(f"layer {l:2d}: oracle {name} is [{ne0}, {ne1}] with no {layout} twin; the set is [{K}, {T}]")
            bad_layers.append(l)
            continue
        with open(os.path.join(oracle_dir, twin[11]), "rb") as f:
            ref = list(struct.unpack(f"<{ne0 * ne1}i", f.read()))
        ours = list(read_topk(s, l))
        diff = [t for t in range(T) if ours[t * K:(t + 1) * K] != ref[t * K:(t + 1) * K]]
        n_eq, n_shared, _ = agreement(ours, ref, K)
        total += len(ref)
        equal += n_eq
        line = f"layer {l:2d}: {n_eq}/{len(ref)} ids equal, {n_shared}/{len(ref)} shared as sets  " \
               f"(oracle: {row[10]} of {row[13]}, contig {row[11]}, {twin[11]})"
        if diff:
            t = diff[0]
            line += f"  DIFF at token {t}: ours {ours[t * K:(t + 1) * K]} oracle {ref[t * K:(t + 1) * K]}"
            bad_layers.append(l)
        print(line)
    ok = not bad_layers and total == equal
    print(f"total: {len(s['layers'])} layers, {equal}/{total} ids equal, "
          f"{len(bad_layers)} layer(s) differ — {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def self_test():
    """Synthetic sets with known answers: the math, the refusals and the oracle comparison."""
    E, K = 8, 2
    with tempfile.TemporaryDirectory() as tmp:
        def make(name, rows_by_layer, complete=True):
            d = os.path.join(tmp, name)
            os.mkdir(d)
            T = len(next(iter(rows_by_layer.values())))
            with open(os.path.join(d, "counts.tsv"), "w") as f:
                f.write("# layer\texpert\tcount\n")
                for l, rows in rows_by_layer.items():
                    c = [0] * E
                    for r in rows:
                        for e in r:
                            c[e] += 1
                    for e in range(E):
                        f.write(f"{l}\t{e}\t{c[e]}\n")
                    with open(os.path.join(d, f"topk-{l}.u16"), "wb") as g:
                        g.write(array("H", [e for r in rows for e in r]).tobytes())
            with open(os.path.join(d, "MANIFEST.tsv"), "w") as f:
                f.write("# router_trace — self-test\n# tokens\t%d\n# n_expert\t%d\n# n_expert_used\t%d\n" % (T, E, K))
                f.write("# model_file\tm.gguf\n# build\tb\n# ids\t%s\n" % os.path.join(d, "ids"))
                for l in rows_by_layer:
                    f.write(f"layer\t{l}\tVIEW-of-ARGSORT\tp\t{T}\t0\t0\ttopk-{l}.u16\n")
                if complete:
                    f.write(f"# complete\t{T}\t{len(rows_by_layer)}\n")
            with open(os.path.join(d, "ids"), "w") as f:
                f.write("\n".join(str(t) for t in range(T)) + "\n")
            return d

        uniform = [[(2 * t) % E, (2 * t + 1) % E] for t in range(8)]  # every expert twice
        s = read_manifest(make("u", {0: uniform}))
        c = read_counts(s)[0]
        assert c == [2] * E and gini(c) == 0.0, c
        assert share(c, hot(c, 2)) == 2 / E
        skew = [[0, 1]] * 6 + [[2, 3], [4, 5]]  # experts 0, 1 take 12 of 16 selections
        c = read_counts(read_manifest(make("s", {0: skew})))[0]
        assert share(c, hot(c, 2)) == 12 / 16 and hot(c, 2) == [0, 1], (c, hot(c, 2))
        assert abs(gini([0] * 7 + [1]) - 7 / 8) < 1e-12
        assert agreement([1, 2, 3, 4], [2, 1, 3, 5], K) == (1, 3, 1)
        try:
            read_manifest(make("x", {0: uniform}, complete=False))
            raise AssertionError("an incomplete set was accepted")
        except SetError:
            pass

        # the oracle comparison: the logical twin is the right one, and one flipped id is a FAIL
        o = os.path.join(tmp, "oracle")
        os.mkdir(o)
        rows = [[3, 5], [1, 0], [7, 2]]
        T = len(rows)
        flat = [3, 5, 6, 4, 1, 0]  # a view's flat head: token 0's sorted row, not each token's top-k
        with open(os.path.join(o, "ffn_moe_topk-0.0.logical.i32"), "wb") as f:
            f.write(struct.pack(f"<{T * K}i", *[e for r in rows for e in r]))
        with open(os.path.join(o, "ffn_moe_topk-0.0.i32"), "wb") as f:
            f.write(struct.pack(f"<{T * K}i", *flat))
        with open(os.path.join(o, "MANIFEST.tsv"), "w") as f:
            f.write("# model_file\tm.gguf\n# build\tb\n# tokens\t0,1,2\n")
            f.write(f"tensor\tffn_moe_topk-0\t0\ti32\t{K}\t{T}\t1\t1\t24\t0\tVIEW\t0\t1\tp (sort)\t-\n")
            f.write("int\tffn_moe_topk-0\t0\ttensor\ti32\ti32\tflat\t6\t24\t0\t6\tffn_moe_topk-0.0.i32\n")
            f.write("int\tffn_moe_topk-0\t0\ttensor\ti32\ti32\tlogical\t6\t24\t0\t7\tffn_moe_topk-0.0.logical.i32\n")
            f.write("# complete\t1\t0\n")
        devnull = open(os.devnull, "w")
        saved, sys.stdout = sys.stdout, devnull
        try:
            good = oracle(make("og", {0: rows}), o)
            bad = oracle(make("ob", {0: [[3, 5], [0, 1], [7, 2]]}), o)
            # the report commands run end to end on a two-layer set
            two = make("two", {0: uniform, 3: skew})
            coverage([two])
            transfer(two, two, 2)
            compare(two, two)
        finally:
            sys.stdout = saved
            devnull.close()
        assert good == 0 and bad == 1, (good, bad)
    print("router-coverage: self-test ok")
    return 0


def main(argv):
    if argv[:1] == ["--self-test"]:
        return self_test()
    try:
        if len(argv) >= 2 and argv[0] == "coverage":
            coverage(argv[1:])
            return 0
        if len(argv) in (3, 5) and argv[0] == "transfer":
            n = int(argv[4]) if len(argv) == 5 and argv[3] == "--n" else 64
            transfer(argv[1], argv[2], n)
            return 0
        if len(argv) == 3 and argv[0] == "compare":
            compare(argv[1], argv[2])
            return 0
        if len(argv) == 3 and argv[0] == "oracle":
            return oracle(argv[1], argv[2])
    except SetError as e:
        print(f"router-coverage: {e}", file=sys.stderr)
        return 1
    print(__doc__, file=sys.stderr)
    return 64


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
