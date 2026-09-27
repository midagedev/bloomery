#!/usr/bin/env python3
"""The reading side of tools/ref/dma-dram-share.sh (its header has the run): the rows, the close, the copy's
rate alone and the box's core layout. Reads logs only; takes no lease and starts no process.

    dma-dram-share.py row <round> <cond> <bench log> <copy log|-> <copy cpus>
    dma-dram-share.py summary <rows file> <alone file>
    dma-dram-share.py alone <copy log>
    dma-dram-share.py topology
    dma-dram-share.py --self-test

row       One bench run's rows. The bench log is the runner's: every line `<epoch ms>\t<line>`. Per `time`
          line (one arm-round) the span runs from the line before it (the previous `time` line, or the
          last line before the first) to it; the copy's rate for the span is the bytes of its interval
          lines (`h2d loop … t_ms=<end> interval_ms=<length> bytes=<b>`) that lie inside the span over
          their summed length. Tags: [overlap cpus …] when a `v41host thread … cpu=<c>` line names a copy
          cpu, [unpinned] when a thread line says `pinned=no`, the pool line `pin_failed=true`, or no
          thread line names a cpu (the overlap check would see nothing), [copy-gap] when the copy's intervals do not cover the span from end to end or none lies
          inside it, [no-time-line] when the bench printed none.
            P2 round=<r> cond=<c> arm=<a> union_gbps=<U> copy_gbps=<C|-> admissible=<yes|no>[ tags]
summary   The medians over rows with admissible=yes and no tag, and per round k = (U_alone - U_with) /
          C_with against that round's alone row:
            P2 summary arm=<a> U_alone=<> | pageable U=<> C=<> k=<> n=<> | staged U=<> C=<> k=<> n=<> |
            copy alone pageable=<> staged=<>
alone     The mean GB/s of a copy log's intervals after its first (the first holds the loop's start).
topology  `<threads> <cpus>`: the physical cores (the first cpu of each thread_siblings_list) less one per
          L3 group, and each group's last physical core, groups in cpu order — the bench's default thread
          count and the cpus a pool of that count leaves free (crates/threads spreads its workers over the
          groups, physical cores first).
"""
import glob
import os
import re
import statistics
import sys
import tempfile


def fields(line):
    return dict(m.groups() for m in re.finditer(r"(\S+?)=(\S+)", line))


def stamped(path):
    out = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            ts, _, rest = line.rstrip("\n").partition("\t")
            out.append((int(ts), rest))
    return out


def intervals(path):
    iv = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("h2d loop ") and " interval_ms=" in line and " summary " not in line:
                d = fields(line)
                end = float(d["t_ms"])
                iv.append((end - float(d["interval_ms"]), end, int(d["bytes"])))
    return iv


def rows(rnd, cond, bench, copy, cpus):
    lines = stamped(bench)
    copy_cpus = {int(c) for c in cpus.split(",") if c}
    bench_cpus = {int(fields(l)["cpu"]) for _, l in lines if l.startswith("v41host thread ") and " cpu=" in l}
    overlap = sorted(bench_cpus & copy_cpus)
    unpinned = not bench_cpus or any(
        (l.startswith("v41host thread ") and " pinned=no" in l)
        or (l.startswith("v41host pool ") and " pin_failed=true" in l) for _, l in lines)
    iv = intervals(copy) if copy != "-" else None
    times = [i for i, (_, l) in enumerate(lines) if l.startswith("time ")]
    if not times:
        return [f"P2 round={rnd} cond={cond} arm=- union_gbps=- copy_gbps=- admissible=no [no-time-line]"]
    out = []
    prev = lines[times[0] - 1][0] if times[0] > 0 else lines[0][0]
    for i in times:
        ts, l = lines[i]
        d = fields(l)
        tags = [f"[overlap cpus {','.join(map(str, overlap))}]"] if overlap else []
        if unpinned:
            tags.append("[unpinned]")
        rate = "-"
        if iv is not None:
            inside = [(a, b, n) for a, b, n in iv if a >= prev and b <= ts]
            covered = bool(iv) and iv[0][0] <= prev and iv[-1][1] >= ts
            if not inside or not covered:
                tags.append("[copy-gap]")
            if inside:
                rate = f"{sum(n for _, _, n in inside) / (sum(b - a for a, b, _ in inside) * 1e6):.2f}"
        out.append(f"P2 round={rnd} cond={cond} arm={d['arm']} union_gbps={d['gbps_mean']} copy_gbps={rate} "
                   f"admissible={d['admissible']}{''.join(' ' + t for t in tags)}")
        prev = ts
    return out


def summary(rows_path, alone_path):
    recs = []
    with open(rows_path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("P2 round="):
                d = fields(line)
                d["clean"] = d["admissible"] == "yes" and "[" not in line
                recs.append(d)
    alone = {}
    with open(alone_path, encoding="utf-8") as f:
        for line in f:
            arm, rate = line.split()
            alone[arm] = rate

    def med(v):
        return f"{statistics.median(v):.2f}" if v else "-"
    out = []
    for arm in sorted({r["arm"] for r in recs if r["arm"] != "-"}):
        mine = [r for r in recs if r["arm"] == arm and r["clean"]]
        base = {r["round"]: float(r["union_gbps"]) for r in mine if r["cond"] == "alone"}
        cells = [f"P2 summary arm={arm} U_alone={med(list(base.values()))}"]
        for cond in ("pageable", "staged"):
            u, c, k = [], [], []
            for r in mine:
                if r["cond"] != cond or r["copy_gbps"] == "-":
                    continue
                u.append(float(r["union_gbps"]))
                c.append(float(r["copy_gbps"]))
                if r["round"] in base:
                    k.append((base[r["round"]] - u[-1]) / c[-1])
            cells.append(f"{cond} U={med(u)} C={med(c)} k={med(k)} n={len(u)}")
        cells.append(f"copy alone pageable={alone.get('pageable-loop', '-')} staged={alone.get('staged-loop', '-')}")
        out.append(" | ".join(cells))
    return out


def alone_rate(path):
    iv = intervals(path)[1:]
    t = sum(b - a for a, b, _ in iv)
    return f"{sum(n for _, _, n in iv) / (t * 1e6):.2f}" if t > 0 else "-"


def cpu_list(text):
    out = []
    for part in text.strip().split(","):
        a, _, b = part.partition("-")
        out.extend(range(int(a), int(b or a) + 1))
    return out


def topology(root="/sys/devices/system/cpu"):
    groups = {}
    for d in sorted(glob.glob(f"{root}/cpu[0-9]*"), key=lambda p: int(p.rsplit("cpu", 1)[1])):
        c = int(d.rsplit("cpu", 1)[1])
        sib = f"{d}/topology/thread_siblings_list"
        if not os.path.exists(sib) or cpu_list(open(sib).read())[0] != c:
            continue
        groups.setdefault(open(f"{d}/cache/index3/shared_cpu_list").read().strip(), []).append(c)
    if not groups:
        raise SystemExit(f"dma-dram-share: no cpu topology under {root}")
    return f"{sum(len(v) for v in groups.values()) - len(groups)} {','.join(str(v[-1]) for v in groups.values())}"


def self_test():
    with tempfile.TemporaryDirectory() as d:
        def write(name, text):
            p = os.path.join(d, name)
            with open(p, "w", encoding="utf-8") as f:
                f.write(text)
            return p
        # a bench whose threads sit on cpus 0 and 2, one arm-round from 1000 to 11000 ms; a copy of 2 GB a
        # second inside that span and 9 GB a second in the intervals that reach outside it
        bench = write("b", "".join(f"{t}\t{l}\n" for t, l in (
            (900, "v41host thread tid=1 name=w cpu=0 core=0"), (900, "v41host thread tid=2 name=w cpu=2 core=2"),
            (1000, "v41host working_set experts_per_layer=32"),
            (11000, "time threads=28 round=1/1 arm=engine:6 tokens=9 gbps_mean=90.00 admissible=yes"),
            (11000, "dispatch threads=28 round=1/1 arm=engine:6 kind=g bytes=1 count=1 us_mean=1 gbps=1"))))
        lines = [f"h2d loop arm=staged-loop t_ms={t} interval_ms=1000.0 bytes={9 if t in (1000, 12000) else 2}000000000 GB/s=0\n"
                 for t in range(1000, 13000, 1000)]
        copy = write("c", "h2d loop arm=staged-loop card=\"x\" bdf=0 start t_ms=0 window=1\n" + "".join(lines))
        gap = write("g", "".join(lines[:3]))
        alone = write("a", "900\tv41host thread tid=1 name=w cpu=0\n1000\tx\n"
                           "11000\ttime threads=28 round=1/1 arm=engine:6 tokens=9 gbps_mean=120.00 admissible=yes\n")
        got = (rows(1, "alone", alone, "-", "3,7") + rows(1, "staged", bench, copy, "3,7")
               + rows(2, "staged", bench, copy, "2,7") + rows(3, "staged", bench, gap, "3,7"))
        want = [
            "P2 round=1 cond=alone arm=engine:6 union_gbps=120.00 copy_gbps=- admissible=yes",
            "P2 round=1 cond=staged arm=engine:6 union_gbps=90.00 copy_gbps=2.00 admissible=yes",
            "P2 round=2 cond=staged arm=engine:6 union_gbps=90.00 copy_gbps=2.00 admissible=yes [overlap cpus 2]",
            "P2 round=3 cond=staged arm=engine:6 union_gbps=90.00 copy_gbps=2.00 admissible=yes [copy-gap]",
        ]
        assert got == want, got
        # k = (120 - 90) / 2 = 15 from round 1 against its alone row; rounds 2 and 3 carry tags and stay out
        s = summary(write("rows", "\n".join(got) + "\n"), write("alone", "staged-loop 25.00\n"))
        assert s == ["P2 summary arm=engine:6 U_alone=120.00 | pageable U=- C=- k=- n=0 | staged U=90.00 C=2.00 "
                     "k=15.00 n=1 | copy alone pageable=- staged=25.00"], s
        assert rows(1, "x", write("n", "1\tv41host model=m\n"), "-", "") == [
            "P2 round=1 cond=x arm=- union_gbps=- copy_gbps=- admissible=no [no-time-line]"]
        # a thread the pool could not pin, a failed pin, or no thread line at all: the cores are not known
        tline = "11000\ttime threads=28 round=1/1 arm=engine:6 tokens=9 gbps_mean=1.00 admissible=yes\n"
        for head in ("900\tv41host thread tid=1 name=w cpu=0\n900\tv41host thread tid=2 name=w cpus=0-63 pinned=no\n",
                     "900\tv41host pool threads=28 pinned_caller=true pin_failed=true\n900\tv41host thread tid=1 name=w cpu=0\n",
                     "900\tv41host pool threads=28 pinned_caller=true pin_failed=false\n"):
            got = rows(1, "alone", write("u", head + tline), "-", "3")
            assert got == ["P2 round=1 cond=alone arm=engine:6 union_gbps=1.00 copy_gbps=- admissible=yes [unpinned]"], got
        # alone: the first interval is the loop's start and is left out: (20 + 30) GB over 2 s = 25
        one = write("o", "".join(f"h2d loop arm=x t_ms=1 interval_ms=1000.0 bytes={b} GB/s=1\n"
                                 for b in (9, 20000000000, 30000000000)))
        assert alone_rate(one) == "25.00", alone_rate(one)
        # topology: two L3 groups of two cores with SMT siblings 4-7 — 2 threads, cpus 1 and 3 free
        for c in range(8):
            os.makedirs(f"{d}/cpu/cpu{c}/topology")
            os.makedirs(f"{d}/cpu/cpu{c}/cache/index3")
            write(f"cpu/cpu{c}/topology/thread_siblings_list", f"{c % 4},{c % 4 + 4}\n")
            write(f"cpu/cpu{c}/cache/index3/shared_cpu_list", "0-1,4-5\n" if c % 4 < 2 else "2-3,6-7\n")
        assert topology(f"{d}/cpu") == "2 1,3", topology(f"{d}/cpu")
    print("dma-dram-share: self-test ok")
    return 0


def main(argv):
    if argv == ["--self-test"]:
        return self_test()
    if len(argv) == 6 and argv[0] == "row":
        print("\n".join(rows(*argv[1:])))
    elif len(argv) == 3 and argv[0] == "summary":
        print("\n".join(summary(argv[1], argv[2])))
    elif len(argv) == 2 and argv[0] == "alone":
        print(alone_rate(argv[1]))
    elif argv == ["topology"]:
        print(topology())
    else:
        print(__doc__, file=sys.stderr)
        return 64
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
