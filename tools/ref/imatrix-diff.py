#!/usr/bin/env python3
"""Compare two llama-imatrix GGUF files tensor by tensor.

  imatrix-diff.py A.gguf B.gguf [--rtol 1e-3]   one line per tensor, then the verdict
  imatrix-diff.py --noise-floor A B [C D]       statistics of the pair A,B (and of C,D beside it): no verdict
  imatrix-diff.py --self-test

llama-imatrix (--output-format gguf) writes, per weight tensor NAME, two F32 tensors: `NAME.in_sum2` (the sum over
every token of the squared input activation, one value per input column and expert) and `NAME.counts` (how many
tokens fed it). Per tensor this prints the maximum relative difference of in_sum2, `counts` equal or not, and the
verdict: OK when every tensor's difference is within --rtol and every counts vector is equal.

The relative difference of one value is |a - b| / max(|a|, |b|, floor), floor = 1e-6 x the largest |value| of that
tensor in either file: a column whose sum of squares is a rounding residue of the tensor does not decide the tensor.

The default --rtol 1e-3 is the band for the same activations summed in another order: the sums are f32, so two runs that
add the same squares in another order agree to a few 1e-5 (a different text, chunk count or context moves in_sum2 by whole
percent and `counts` by whole chunks, far outside it). `counts` is an integer count and is compared exactly. The band does
not cover a run whose activations differ: a quantized matmul of another kernel (CPU against CUDA) or another batch width
(n_seq 1 against 2) moves the activations by ~1e-3 to 1e-2 per layer, and on a model with near-tied router scores the top-k
choice flips and the routed tensors' `counts` and sums diverge by whole percent. Measured on the K3 fixture (random
weights, ctx 128, 4 chunks): CPU against CPU at n_seq 1 and 2 already differs (worst 4.6e-1, counts on 3 tensors), the
dense layer-0 attention inputs agree to 3e-7 and everything after the first quantized matmul does not.

A file written at the pinned llama.cpp also holds `NAME.in_sum` (the plain activation sums) and `NAME.stats`; this compares
in_sum2 and counts only, the two the quantizer's importance weights are made from.

--noise-floor. The imatrix is a statistic of a routed model, not an exact value: another kernel or batch width flips router
choices and moves the routed tensors by whole percent (above), so max-relative-difference is the wrong ruler between such
runs. This mode compares the per-channel mean squared activation, in_sum2 / counts (per expert column, an expert with no
token counts as 0), and prints per tensor and as median, p95 and max over the tensors:
  l2   the relative L2 distance of that mean, ||a - b|| / sqrt((||a||^2 + ||b||^2) / 2) (symmetric in the two files)
  cnt  the relative difference of counts, sum |ca - cb| / max(sum ca, sum cb)
With two files that is one pair, e.g. CPU n_seq 1 against CPU n_seq 2: the noise floor of the path itself. With four, the
second pair (e.g. CUDA against CPU at one n_seq) is printed beside it with the ratio of the two pairs' medians, p95s and maxes.
It judges nothing: the reader compares the pair under test with the floor. Exit 0, or 65 on a refusal as below.

Refused by name (exit 65): a tensor present in one file only, an in_sum2 without its counts or the reverse, vectors of
different lengths, a non-F32 tensor, a NaN or infinity, a file the GGUF reader does not take. Exit 1: a verdict FAIL.
Exit 64: usage. The GGUF reader is tools/ref/gguf-splice.py's (load_set), the one place that walks a header.
"""

import array
import importlib.util
import math
import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
F32 = 0  # GGUF tensor type id of F32
IGNORED = (".in_sum", ".stats")


def _load(file, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S = _load("gguf-splice.py", "gguf_splice")
R = S.R


class Refused(Exception):
    pass


def read_imatrix(path):
    """{base name: {"in_sum2": array('f'), "counts": array('f')}} of one imatrix GGUF."""
    try:
        shards = S.load_set(path)
    except (S.Refusal, OSError) as e:
        raise Refused(f"{path}: not a GGUF file the reader takes: {e}")
    out = {}
    for s in shards:
        with open(s.path, "rb") as f:
            for t in s.tensors:
                if t.name.endswith(IGNORED):
                    continue  # the pin's file also holds NAME.in_sum (activation sums) and NAME.stats: not compared
                kind = None
                for k in ("in_sum2", "counts"):
                    if t.name.endswith("." + k):
                        kind, base = k, t.name[:-len(k) - 1]
                if kind is None:
                    raise Refused(f"{path}: tensor {t.name} is neither NAME.in_sum2 nor NAME.counts: not an imatrix file")
                if t.ty != F32:
                    raise Refused(f"{path}: tensor {t.name} has type id {t.ty}, not F32")
                f.seek(s.start + t.off)
                raw = f.read(t.nb)
                if len(raw) != t.nb:
                    raise Refused(f"{path}: tensor {t.name} is cut short")
                v = array.array("f")
                v.frombytes(raw)
                if sys.byteorder != "little":
                    v.byteswap()
                if any(not math.isfinite(x) for x in v):
                    raise Refused(f"{path}: tensor {t.name} holds a NaN or an infinity")
                out.setdefault(base, {})[kind] = v
    for base, d in out.items():
        if set(d) != {"in_sum2", "counts"}:
            have = next(iter(d))
            raise Refused(f"{path}: {base} has {have} but not its {'counts' if have == 'in_sum2' else 'in_sum2'}")
    return out


def rel_max(a, b):
    """The largest relative difference of two equal-length vectors (see the module docstring for the floor)."""
    floor = 1e-6 * max([abs(x) for x in a] + [abs(x) for x in b] + [0.0])
    worst = 0.0
    for x, y in zip(a, b):
        den = max(abs(x), abs(y), floor)
        if den > 0:
            worst = max(worst, abs(x - y) / den)
    return worst


def compare(a, b, a_name="A", b_name="B"):
    """[(name, max rel diff of in_sum2, counts equal)] over the tensors of both; refuses a tensor in one only."""
    only_a, only_b = sorted(set(a) - set(b)), sorted(set(b) - set(a))
    if only_a or only_b:
        raise Refused(f"tensors in one file only: {len(only_a)} only in {a_name} (first {only_a[:1]}), "
                      f"{len(only_b)} only in {b_name} (first {only_b[:1]})")
    rows = []
    for name in sorted(a):
        for kind in ("in_sum2", "counts"):
            if len(a[name][kind]) != len(b[name][kind]):
                raise Refused(f"{name}.{kind}: {len(a[name][kind])} values in {a_name}, {len(b[name][kind])} in {b_name}")
        rows.append((name, rel_max(a[name]["in_sum2"], b[name]["in_sum2"]), a[name]["counts"] == b[name]["counts"]))
    return rows


def mean_sq(entry):
    """in_sum2 / counts per channel: the vector (expert-major) and counts; an expert with no token counts as 0."""
    s, c = entry["in_sum2"], entry["counts"]
    n = len(c)
    if n == 0 or len(s) % n:
        raise Refused(f"{len(s)} in_sum2 values do not split into {n} counts")
    per = len(s) // n
    return [(s[e * per + i] / c[e] if c[e] > 0 else 0.0) for e in range(n) for i in range(per)], c


def stat_rows(a, b, a_name="A", b_name="B"):
    """[(name, relative L2 of the mean squared activation, relative difference of counts)]; refuses a tensor in one file only."""
    compare(a, b, a_name, b_name)  # the same refusals as the strict mode (tensor sets, lengths)
    rows = []
    for name in sorted(a):
        ma, ca = mean_sq(a[name])
        mb, cb = mean_sq(b[name])
        d2 = sum((x - y) ** 2 for x, y in zip(ma, mb))
        n2 = (sum(x * x for x in ma) + sum(y * y for y in mb)) / 2
        l2 = 0.0 if d2 == 0 else math.sqrt(d2 / n2)
        tot = max(sum(ca), sum(cb))
        cnt = 0.0 if tot == 0 else sum(abs(x - y) for x, y in zip(ca, cb)) / tot
        rows.append((name, l2, cnt))
    return rows


def quantile(vals, q):
    v = sorted(vals)
    if not v:
        return 0.0
    k = (len(v) - 1) * q
    lo = int(math.floor(k))
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def aggregate(rows):
    """{metric: (median, p95, max)} over the tensors."""
    out = {}
    for j, m in ((1, "l2"), (2, "cnt")):
        vals = [r[j] for r in rows]
        out[m] = (quantile(vals, 0.5), quantile(vals, 0.95), max(vals, default=0.0))
    return out


def run_noise(files, out=sys.stdout):
    pairs = []
    for i in range(0, len(files), 2):
        rows = stat_rows(read_imatrix(files[i]), read_imatrix(files[i + 1]), files[i], files[i + 1])
        pairs.append((files[i], files[i + 1], rows, aggregate(rows)))
    for k, (fa, fb, rows, agg) in enumerate(pairs):
        tag = "floor" if k == 0 else "test"
        for name, l2, cnt in rows:
            print(f"{tag}\t{name}\tl2={l2:.3e}\tcnt={cnt:.3e}", file=out)
        print(f"imatrix-diff: {tag} pair {fa} {fb}: {len(rows)} tensors", file=out)
        for m in ("l2", "cnt"):
            med, p95, mx = agg[m]
            print(f"imatrix-diff: {tag} {m}: median {med:.3e} p95 {p95:.3e} max {mx:.3e}", file=out)
    if len(pairs) == 2:
        for m in ("l2", "cnt"):
            r = [(t / f if f > 0 else float("inf")) for f, t in zip(pairs[0][3][m], pairs[1][3][m])]
            print(f"imatrix-diff: test/floor {m}: median x{r[0]:.2f} p95 x{r[1]:.2f} max x{r[2]:.2f}", file=out)
    return 0


def verdict(rows, rtol):
    bad = [r for r in rows if r[1] > rtol or not r[2]]
    return not bad, bad


def run(pa, pb, rtol, out=sys.stdout):
    rows = compare(read_imatrix(pa), read_imatrix(pb), pa, pb)
    for name, d, same in rows:
        print(f"{name}\tmax_rel_in_sum2={d:.3e}\tcounts={'equal' if same else 'DIFFER'}", file=out)
    ok, bad = verdict(rows, rtol)
    worst = max((r[1] for r in rows), default=0.0)
    n_cnt = sum(1 for r in rows if not r[2])
    print(f"imatrix-diff: {len(rows)} tensors, worst max_rel_in_sum2 {worst:.3e} (rtol {rtol:g}), counts differ on {n_cnt}; "
          f"verdict {'OK' if ok else 'FAIL'} ({len(bad)} tensor(s) outside)", file=out)
    return 0 if ok else 1


def _write(path, entries):
    """entries {base: (in_sum2 list, counts list)}: a one-file imatrix GGUF."""
    tensors = []
    for base, (s, c) in entries.items():
        for kind, vals in (("in_sum2", s), ("counts", c)):
            tensors.append((f"{base}.{kind}", F32, [len(vals)], array.array("f", vals).tobytes()))
    R.write_gguf(path, [R._kv_string("general.architecture", "imatrix"), R._kv_scalar("general.alignment", 4, 32)], tensors)


def self_test():
    ok = True

    def check(label, cond):
        nonlocal ok
        print(f"  {'ok  ' if cond else 'FAIL'} {label}")
        ok = ok and cond

    def refuses(pa, pb, needle):
        try:
            run(pa, pb, 1e-3, out=open(os.devnull, "w"))
        except Refused as e:
            return needle in str(e)
        return False

    base = {"blk.0.ffn_up_exps.weight": ([1.0, 2.0, 4.0, 8.0], [10.0, 10.0, 10.0, 10.0]),
            "blk.0.attn_q.weight": ([0.5, 0.25], [10.0, 10.0])}
    with tempfile.TemporaryDirectory() as d:
        p = lambda n: os.path.join(d, n)
        _write(p("a"), base)
        _write(p("same"), base)
        near = {k: ([x * (1 + 2e-4) for x in s], c) for k, (s, c) in base.items()}
        _write(p("near"), near)
        far = {k: ([x * 1.02 for x in s], c) for k, (s, c) in base.items()}
        _write(p("far"), far)
        fewer = {k: (s, [c[0] - 5] * len(c)) for k, (s, c) in base.items()}  # one chunk less: counts and sums move
        _write(p("fewer"), fewer)
        extra = dict(base)
        extra["blk.1.attn_q.weight"] = ([1.0], [10.0])
        _write(p("extra"), extra)
        short = dict(base)
        short["blk.0.attn_q.weight"] = ([0.5], [10.0])
        _write(p("short"), short)
        _write(p("nan"), {"blk.0.attn_q.weight": ([float("nan"), 1.0], [10.0, 10.0])})
        null = open(os.devnull, "w")
        # noise-floor mode: scaled means, shifted counts, the aggregate and the refusals
        mean_up = {k: ([x * 1.1 for x in s], c) for k, (s, c) in base.items()}  # same counts, mean x1.1
        _write(p("mean_up"), mean_up)
        import io
        buf = io.StringIO()
        run_noise([p("a"), p("same"), p("a"), p("mean_up")], buf)
        text = buf.getvalue()
        want = 0.1 / math.sqrt((1 + 1.21) / 2)
        check("noise-floor: identical pair has l2 0 and cnt 0", "floor l2: median 0.000e+00 p95 0.000e+00 max 0.000e+00" in text
              and "floor cnt: median 0.000e+00" in text)
        check(f"noise-floor: means x1.1 give l2 {want:.4f}", f"test l2: median {want:.3e}" in text)
        check("noise-floor: the ratio line is printed", "test/floor l2:" in text)
        cnt_half = {k: (s, [c[0] / 2] * len(c)) for k, (s, c) in base.items()}
        _write(p("cnt_half"), cnt_half)
        buf = io.StringIO()
        run_noise([p("a"), p("cnt_half")], buf)
        check("noise-floor: counts halved give cnt 0.5", "floor cnt: median 5.000e-01" in buf.getvalue())
        try:
            run_noise([p("a"), p("extra")], io.StringIO())
            ok_ref = False
        except Refused as e:
            ok_ref = "in one file only" in str(e)
        check("noise-floor: a tensor in one file only: refused by name", ok_ref)
        check("identical files: OK, rc 0", run(p("a"), p("same"), 1e-3, null) == 0)
        check("2e-4 apart (sum-order scale): OK within 1e-3", run(p("a"), p("near"), 1e-3, null) == 0)
        check("2e-4 apart fails a 1e-5 band", run(p("a"), p("near"), 1e-5, null) == 1)
        check("2 % apart: FAIL, rc 1", run(p("a"), p("far"), 1e-3, null) == 1)
        check("different chunk count (counts and sums differ): FAIL, rc 1", run(p("a"), p("fewer"), 1e-3, null) == 1)
        check("counts differ with sums equal still FAIL",
              run(p("a"), _w(p, _write, {k: (s, [c[0] + 1] * len(c)) for k, (s, c) in base.items()}, "cnt"), 1e-3, null) == 1)
        check("a tensor in one file only: refused by name", refuses(p("a"), p("extra"), "in one file only"))
        check("vectors of different length: refused by name", refuses(p("a"), p("short"), "values in"))
        check("a NaN: refused by name", refuses(p("nan"), p("nan"), "NaN"))
        check("not a GGUF: refused by name", refuses(p("a"), os.path.abspath(__file__), "not a GGUF"))
    print("imatrix-diff self-test: " + ("ok" if ok else "FAILED"))
    return 0 if ok else 1


def _w(p, write, entries, name):
    write(p(name), entries)
    return p(name)


def main(argv):
    if argv == ["--self-test"]:
        return self_test()
    if argv and argv[0] == "--noise-floor":
        if len(argv) not in (3, 5) or any(a.startswith("-") for a in argv[1:]):
            print("imatrix-diff.py: --noise-floor takes two files (A B) or four (A B C D)", file=sys.stderr)
            return 64
        try:
            return run_noise(argv[1:])
        except Refused as e:
            print(f"imatrix-diff.py: {e}", file=sys.stderr)
            return 65
    rtol, files, i = 1e-3, [], 0
    while i < len(argv):
        if argv[i] == "--rtol":
            if i + 1 >= len(argv):
                print("imatrix-diff.py: --rtol takes a value", file=sys.stderr)
                return 64
            try:
                rtol = float(argv[i + 1])
            except ValueError:
                print(f"imatrix-diff.py: --rtol {argv[i + 1]} is not a number", file=sys.stderr)
                return 64
            i += 2
        elif argv[i].startswith("-"):
            print(f"imatrix-diff.py: unknown option {argv[i]}", file=sys.stderr)
            return 64
        else:
            files.append(argv[i])
            i += 1
    if len(files) != 2 or not rtol > 0:
        print(__doc__.split("\n\n")[0] if __doc__ else "usage", file=sys.stderr)
        return 64
    try:
        return run(files[0], files[1], rtol)
    except Refused as e:
        print(f"imatrix-diff.py: {e}", file=sys.stderr)
        return 65


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
