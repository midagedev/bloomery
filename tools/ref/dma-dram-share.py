#!/usr/bin/env python3
"""The reading side of tools/ref/dma-dram-share.sh (its header has the run): the rows, the close, the copy's
rate alone and the box's core layout. Reads logs only; takes no lease and starts no process.

    dma-dram-share.py plan <conds> <token ms> <chunk> <ring> <fill threads> <block> <anon bytes>
    dma-dram-share.py row <round> <cond> <bench log> <copy log|-> <copy cpus>
    dma-dram-share.py summary <rows file> <alone file> [<conds> [<block>]]
    dma-dram-share.py alone <copy log>
    dma-dram-share.py topology
    dma-dram-share.py --self-test

plan      The conditions' grammar, its one parser: a comma list, each once, `alone` among them, of alone,
          pageable, staged, staging:<n|max>, direct:<n|max> and odirect:<n|max> (n copies a token, a
          positive number). One line a condition, `cond=<c> <h2d_probe arguments>`: pageable and
          staged copy the shard's map in <chunk>s with <ring> and <fill threads>; staging is the
          residency machine's copy path, a staged-loop over an anonymous window of <anon bytes> in
          <block>s, a ring of MACHINE_RING and one fill thread; direct a direct-loop over the same
          window, MACHINE_RING in flight; odirect an odirect-loop — each <block> read with O_DIRECT
          from the model file (h2d_probe's --od-file default) into a pinned ring slot, blocks at a
          fixed stride over the file, MACHINE_RING deep, <anon bytes> a round — so it reads the model
          file, unlike the anonymous conditions; n paces them at n x 1000 / <token ms> blocks a
          second (--rate), max leaves them unpaced.

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
          C_with against that round's alone row, a cell per condition of <conds> (default
          alone,pageable,staged) but alone:
            P2 summary arm=<a> U_alone=<> | pageable U=<> C=<> k=<> n=<> | staged U=<> C=<> k=<> n=<> |
            copy alone pageable=<> staged=<>
          and per staging, direct or odirect condition c, the mean of its per-round k, with the 95 % t
          interval of that mean (tools/ref/tdist.py; `-` under two rounds) and its copies a second at
          <block> bytes (default 16773120, one V4.1 expert):
            P2 c arm=<a> cond=<c> C=<median> copies_per_s=<> c=<mean> lo=<> hi=<> n=<>
          Beside a DRAM-bound leg k is c, the host-slot equivalents one copy costs (dma-dram-share.sh).
alone     The mean GB/s of a copy log's intervals after its first (the first holds the loop's start).
topology  `<threads> <cpus>`: the physical cores (the first cpu of each thread_siblings_list) less one per
          L3 group, and each group's last physical core, groups in cpu order — the bench's default thread
          count and the cpus a pool of that count leaves free (crates/threads spreads its workers over the
          groups, physical cores first).
"""
import glob
import math
import os
import re
import statistics
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tdist import t975  # noqa: E402  (tools/ref/tdist.py, beside this file)

# The residency machine's staging ring (crates/gpu/src/host/swap.rs RING_SLOTS), and one V4.1 routed
# expert's bytes, the block a staging or direct copy moves.
MACHINE_RING = 4
EXPERT_BYTES = 16773120
DEFAULT_CONDS = "alone,pageable,staged"


def plan(conds, token_ms, chunk, ring, fill, block, anon):
    """`cond=<c> <h2d_probe arguments>` per condition, or SystemExit naming what the grammar refuses."""
    def refuse(why):
        raise SystemExit(f"dma-dram-share: --conds {conds}: {why}")
    try:
        tms = float(token_ms)
    except ValueError:
        tms = float("nan")
    if not (math.isfinite(tms) and tms > 0):
        raise SystemExit(f"dma-dram-share: --token-ms {token_ms}: a positive number of milliseconds")
    chunk, ring, fill, block, anon = (int(v) for v in (chunk, ring, fill, block, anon))
    if block % 4096 or anon % 4096 or anon < block:
        raise SystemExit(f"dma-dram-share: --block {block} and --anon-bytes {anon}: multiples of 4096, "
                         "the window at least one block")
    names = conds.split(",")
    if len(set(names)) != len(names):
        refuse("a condition twice")
    if "alone" not in names:
        refuse("no alone condition: k is read against each round's alone row")
    out = []
    for c in names:
        if c == "alone":
            out.append("cond=alone")
        elif c in ("pageable", "staged"):
            out.append(f"cond={c} --arm {c}-loop --chunk {chunk} --ring {ring} --fill-threads {fill}")
        else:
            path, _, n = c.partition(":")
            if path not in ("staging", "direct", "odirect"):
                refuse(f"unknown condition {c!r}; the conditions are alone, pageable, staged, "
                       "staging:<n|max>, direct:<n|max>, odirect:<n|max>")
            if n == "max":
                rate = ""
            else:
                try:
                    per_token = float(n)
                except ValueError:
                    per_token = float("nan")
                if not (math.isfinite(per_token) and per_token > 0):
                    refuse(f"{c}: n is a positive number of copies a token, or max")
                rate = f" --rate {per_token * 1000 / tms:.6g}"
            arm = {"staging": "--arm staged-loop", "direct": "--arm direct-loop",
                   "odirect": "--arm odirect-loop"}[path]
            source = "" if path == "odirect" else " --source anon"
            fills = " --fill-threads 1" if path == "staging" else ""
            out.append(f"cond={c} {arm}{source} --bytes {anon} --chunk {block} "
                       f"--ring {MACHINE_RING}{fills}{rate}")
    return out


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


def summary(rows_path, alone_path, conds=DEFAULT_CONDS, block=EXPERT_BYTES):
    plan(conds, "1", 4096, 2, 1, EXPERT_BYTES, EXPERT_BYTES)  # refuses a <conds> outside the grammar by name
    names = [c for c in conds.split(",") if c != "alone"]
    block = int(block)
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
        cs = []
        for cond in names:
            u, c, k = [], [], []
            for r in mine:
                if r["cond"] != cond or r["copy_gbps"] == "-":
                    continue
                u.append(float(r["union_gbps"]))
                c.append(float(r["copy_gbps"]))
                if r["round"] in base:
                    k.append((base[r["round"]] - u[-1]) / c[-1])
            cells.append(f"{cond} U={med(u)} C={med(c)} k={med(k)} n={len(u)}")
            if cond.startswith(("staging:", "direct:", "odirect:")):
                cs.append(c_line(arm, cond, c, k, block))
        cells.append("copy alone " + " ".join(f"{n}={alone.get(n, '-')}" for n in names))
        out.append(" | ".join(cells))
        out.extend(cs)
    return out


def c_line(arm, cond, c, k, block):
    """The P2 c line: the mean of the per-round k with its 95 % t interval, and the copy's rate."""
    rate = f"{statistics.median(c):.2f}" if c else "-"
    per_s = f"{statistics.median(c) * 1e9 / block:.1f}" if c else "-"
    if not k:
        mean = lo = hi = "-"
    else:
        m = statistics.fmean(k)
        mean = f"{m:.3f}"
        if len(k) < 2:
            lo = hi = "-"
        else:
            h = t975(len(k) - 1) * statistics.stdev(k) / math.sqrt(len(k))
            lo, hi = f"{m - h:.3f}", f"{m + h:.3f}"
    return f"P2 c arm={arm} cond={cond} C={rate} copies_per_s={per_s} c={mean} lo={lo} hi={hi} n={len(k)}"


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
        s = summary(write("rows", "\n".join(got) + "\n"), write("alone", "staged 25.00\n"))
        assert s == ["P2 summary arm=engine:6 U_alone=120.00 | pageable U=- C=- k=- n=0 | staged U=90.00 C=2.00 "
                     "k=15.00 n=1 | copy alone pageable=- staged=25.00"], s
        # a block size passed where the conditions go is refused, not read as a condition named 16773120
        try:
            summary(write("rows2", "\n".join(got) + "\n"), write("alone2", "staged 25.00\n"), "16773120")
        except SystemExit:
            pass
        else:
            raise AssertionError("summary took 16773120 for its conditions")
        # the default conditions keep the probe's arguments the runner passed before --conds existed
        assert plan(DEFAULT_CONDS, "37.5", 67108864, 4, 4, EXPERT_BYTES, 1006387200) == [
            "cond=alone",
            "cond=pageable --arm pageable-loop --chunk 67108864 --ring 4 --fill-threads 4",
            "cond=staged --arm staged-loop --chunk 67108864 --ring 4 --fill-threads 4"], "default plan"
        # 6 copies a token at 37.5 ms a token is 160 a second; 3 is 80; max is unpaced
        got = plan("alone,staging:6,direct:3,staging:max,direct:max", "37.5", 67108864, 4, 4, EXPERT_BYTES,
                   1006387200)
        anon = "--source anon --bytes 1006387200 --chunk 16773120 --ring 4"
        assert got == ["cond=alone",
                       f"cond=staging:6 --arm staged-loop {anon} --fill-threads 1 --rate 160",
                       f"cond=direct:3 --arm direct-loop {anon} --rate 80",
                       f"cond=staging:max --arm staged-loop {anon} --fill-threads 1",
                       f"cond=direct:max --arm direct-loop {anon}"], got
        # odirect: the same window, pacing and ring, the model file itself the source (no --source
        # anon, no fill threads)
        got = plan("alone,odirect:3,odirect:max", "37.5", 67108864, 4, 4, EXPERT_BYTES, 1006387200)
        assert got == ["cond=alone",
                       "cond=odirect:3 --arm odirect-loop --bytes 1006387200 --chunk 16773120 "
                       "--ring 4 --rate 80",
                       "cond=odirect:max --arm odirect-loop --bytes 1006387200 --chunk 16773120 "
                       "--ring 4"], got
        for bad in ("staging:6", "alone,alone", "alone,copy:3", "alone,staging:0", "alone,direct:x",
                    "alone,staging:", "alone,odirect:0", "alone,odirect:x", "alone,odirect:"):
            try:
                plan(bad, "37.5", 67108864, 4, 4, EXPERT_BYTES, 1006387200)
            except SystemExit:
                continue
            raise AssertionError(f"plan accepted --conds {bad}")
        # c: per round k against that round's alone row, then the mean and its t interval. Staging: round 1
        # (120 - 114) / 2 = 3, round 2 (118 - 113) / 2.5 = 2, round 3's row is tagged and stays out; direct:
        # 0.5 and 1.0, and round 3's row counts in U and C but has no clean alone row to give a k: n = 2
        # each. t975(1) = 12.706: staging h = 12.706 * 0.7071 / sqrt(2) = 6.353, direct 3.176.
        rows_c = [
            "P2 round=1 cond=alone arm=engine:5 union_gbps=120.00 copy_gbps=- admissible=yes",
            "P2 round=1 cond=staging:6 arm=engine:5 union_gbps=114.00 copy_gbps=2.00 admissible=yes",
            "P2 round=1 cond=direct:6 arm=engine:5 union_gbps=119.00 copy_gbps=2.00 admissible=yes",
            "P2 round=2 cond=alone arm=engine:5 union_gbps=118.00 copy_gbps=- admissible=yes",
            "P2 round=2 cond=staging:6 arm=engine:5 union_gbps=113.00 copy_gbps=2.50 admissible=yes",
            "P2 round=2 cond=direct:6 arm=engine:5 union_gbps=116.00 copy_gbps=2.00 admissible=yes",
            "P2 round=3 cond=alone arm=engine:5 union_gbps=90.00 copy_gbps=- admissible=yes [unpinned]",
            "P2 round=3 cond=staging:6 arm=engine:5 union_gbps=60.00 copy_gbps=2.00 admissible=yes [copy-gap]",
            "P2 round=3 cond=direct:6 arm=engine:5 union_gbps=80.00 copy_gbps=2.00 admissible=yes",
        ]
        s = summary(write("rc", "\n".join(rows_c) + "\n"), write("ac", "staging:6 2.70\ndirect:6 2.68\n"),
                    "alone,staging:6,direct:6", "16773120")
        assert s == [
            "P2 summary arm=engine:5 U_alone=119.00 | staging:6 U=113.50 C=2.25 k=2.50 n=2 | direct:6 U=116.00 "
            "C=2.00 k=0.75 n=3 | copy alone staging:6=2.70 direct:6=2.68",
            "P2 c arm=engine:5 cond=staging:6 C=2.25 copies_per_s=134.1 c=2.500 lo=-3.853 hi=8.853 n=2",
            "P2 c arm=engine:5 cond=direct:6 C=2.00 copies_per_s=119.2 c=0.750 lo=-2.426 hi=3.926 n=2",
        ], s
        # odirect's c line, the same reading: round 1 (120 - 118.4) / 2 = 0.8, round 2
        # (118 - 117) / 2 = 0.5; mean 0.65, h = 12.706 * 0.2121 / sqrt(2) = 1.906
        rows_od = [
            "P2 round=1 cond=alone arm=engine:5 union_gbps=120.00 copy_gbps=- admissible=yes",
            "P2 round=1 cond=odirect:6 arm=engine:5 union_gbps=118.40 copy_gbps=2.00 admissible=yes",
            "P2 round=2 cond=alone arm=engine:5 union_gbps=118.00 copy_gbps=- admissible=yes",
            "P2 round=2 cond=odirect:6 arm=engine:5 union_gbps=117.00 copy_gbps=2.00 admissible=yes",
        ]
        s = summary(write("ro", "\n".join(rows_od) + "\n"), write("ao", "odirect:6 2.71\n"),
                    "alone,odirect:6", "16773120")
        assert s == [
            "P2 summary arm=engine:5 U_alone=119.00 | odirect:6 U=117.70 C=2.00 k=0.65 n=2 | "
            "copy alone odirect:6=2.71",
            "P2 c arm=engine:5 cond=odirect:6 C=2.00 copies_per_s=119.2 c=0.650 lo=-1.256 hi=2.556 n=2",
        ], s
        # the probe's two shapes around a stop-file written after the bench's time line (11000): the copy's
        # full intervals end at 10600 and it stops at 11300. The old probe printed nothing for the last 700 ms,
        # so the span was never covered to its end; the probe now flushes that partial interval at the stop.
        full = [f"h2d loop arm=staged-loop t_ms={t} interval_ms=1000.0 bytes=2000000000 GB/s=2\n"
                for t in range(1600, 11000, 1000)]
        tail = "h2d loop arm=staged-loop t_ms=11300 interval_ms=700.0 bytes=1400000000 GB/s=2\n"
        old_probe = rows(1, "staged", bench, write("op", "".join(full)), "3,7")
        new_probe = rows(1, "staged", bench, write("np", "".join(full) + tail), "3,7")
        assert old_probe == ["P2 round=1 cond=staged arm=engine:6 union_gbps=90.00 copy_gbps=2.00 admissible=yes "
                             "[copy-gap]"], old_probe
        assert new_probe == ["P2 round=1 cond=staged arm=engine:6 union_gbps=90.00 copy_gbps=2.00 admissible=yes"], \
            new_probe
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
    if len(argv) == 8 and argv[0] == "plan":
        print("\n".join(plan(*argv[1:])))
    elif len(argv) == 6 and argv[0] == "row":
        print("\n".join(rows(*argv[1:])))
    elif 3 <= len(argv) <= 5 and argv[0] == "summary":
        print("\n".join(summary(*argv[1:])))
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
