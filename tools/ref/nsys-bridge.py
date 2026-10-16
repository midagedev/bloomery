#!/usr/bin/env python3
"""Per replay of a graph-mode trace of a hybrid chain (V4.1, Qwen3.8): the card's critical segments and the
kernel time inside them, the host bridges, the launch call, and the idle between replays — read from an
Nsight Systems sqlite export.

  nsys-bridge.py <run>.sqlite [--handoff NAME] [--post PREFIX[,PREFIX]] [--replay R] [--from R]
                              [--joins N] [--kernel PATTERN]
  nsys-bridge.py --self-test

A replay is the kernels that start between one `cuGraphLaunch` call and the next. Per hybrid layer
the stream runs handoff -> go -> [card work] -> wait -> post (chain/ffn.rs); the go and the wait are
stream memory operations, which the trace records as nothing. So, per replay:

  critical segments  first kernel start -> handoff 0 end, then each post start -> the next handoff
                     end, then the last post start -> the last kernel end (the tail): the card's own
                     chain with the host bridges cut out. Their sum is `crit`. A replay with J joins
                     has J + 1 segments.
  busy               the union of the replay's kernel intervals (any stream) clipped to the critical
                     segments: the time inside them with a kernel running. `gaps` is `crit - busy`,
                     the time inside them with none: the in-chain latency between kernels.
  bridge k           handoff k end -> post k start: go to join as the card sees it (the host's leg
                     plus both signalling latencies). k counts the joins in chain order, from 0 —
                     the k-th hybrid layer the replay serves, not the model's layer number.
  launch             the `cuGraphLaunch` call's own span, where its first kernel started relative to
                     the call, when the call returned relative to that kernel, and relative to the
                     end of handoff 0 (positive: layer 0's go was issued before the host could
                     serve it, which it does only after the call returns).
  idle               the last kernel end -> the next replay's first kernel start, and -> the next
                     launch call's start (the host turnaround between steps).

--handoff (default `ds41_ffn_handoff`) is the handoff kernel's exact name. --post (default the prefix
`ds41_ffn_post`) is the post kernels' name prefixes, comma-separated: a trace whose layers end in
different kernels (Qwen3.8's `q38_card_shared_add` on a layer with card experts, `q38_shared_add` on
one without) names both. The names the trace holds are printed so a rename shows. Qwen3.8's step is
`--handoff ds41_ffn_handoff_10`, its m-column verify `--handoff ds41_ffn_handoff_10_cols`, both
`--post q38_card_shared_add,q38_shared_add`.

--replay R prints that replay's every segment (with its busy and gap) and bridge. The per-join table and
the summary line are over the replays from --from (default: the first replay with the chosen join
count) to the last that have it; the join count is the modal one, or --joins N: a trace in which short
walks (a draft's, 0-1 joins) outnumber the full passes needs it to read the passes. An empty selection
is an error. The per-join table's `gap_mean` is the segment ending at handoff k's gap.

--kernel PATTERN (comma-separated substrings of kernel names; the names matched are printed) adds to the
summary the mean µs of one matching launch, the µs and the count of them a replay.

The summary line, last, is the one to quote (means over the selected replays, µs and ms are the
tracer's; the tracer stretches the launch call itself):
  summary replays=<n> joins=<J> crit_ms busy_ms gaps_ms gap_us_per_join idle_ms_p50 [kernel[<pat>]_us
  kernel[<pat>]_us_per_replay n/replay]
`gap_us_per_join` is gaps / (J + 1), one per critical segment; `idle_ms_p50` is the median idle of the
selected replays that have a next replay. Read-only.
"""
import argparse
import bisect
import contextlib
import io
import os
import sqlite3
import statistics
import sys
import tempfile
from collections import Counter


def open_ro(path):
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def load(db, handoff, post):
    names = dict(db.execute("SELECT id, value FROM StringIds"))
    ks = [(s, e, names.get(n, str(n)).split("(")[0])
          for s, e, n in db.execute("SELECT start, end, shortName FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start")]
    gl = [(s, e, tid) for s, e, n, tid in
          db.execute("SELECT start, end, nameId, globalTid FROM CUPTI_ACTIVITY_KIND_RUNTIME ORDER BY start")
          if names.get(n, "").startswith("cuGraphLaunch")]
    seen = sorted({k[2] for k in ks if k[2] == handoff or k[2].startswith(post)})
    return ks, gl, seen


def union_len(iv):
    """The length of the union of the intervals (sorted in place)."""
    iv.sort()
    total, cs, ce = 0, None, None
    for s, e in iv:
        if ce is None or s > ce:
            if ce is not None:
                total += ce - cs
            cs, ce = s, e
        elif e > ce:
            ce = e
    return total + (ce - cs if ce is not None else 0)


def seg_busy(w, segs):
    """Per critical segment, the union length of the kernels of `w` clipped to it. The segments are
    ordered and disjoint, so each kernel meets a contiguous run of them, found by bisection."""
    ends = [e for _, e in segs]
    clipped = [[] for _ in segs]
    for s, e, _ in w:
        j = bisect.bisect_right(ends, s)
        while j < len(segs) and segs[j][0] < e:
            lo, hi = max(s, segs[j][0]), min(e, segs[j][1])
            if lo < hi:
                clipped[j].append((lo, hi))
            j += 1
    return [union_len(c) for c in clipped]


def replays(ks, gl, handoff, post):
    out = []
    j = 0
    for i, (g0, g1, tid) in enumerate(gl):
        nxt = gl[i + 1][0] if i + 1 < len(gl) else float("inf")
        while j < len(ks) and ks[j][0] < g0:
            j += 1
        w = []
        while j < len(ks) and ks[j][0] < nxt:
            w.append(ks[j])
            j += 1
        if not w:
            continue
        segs, bridges = [], []
        cur, h_end = w[0][0], None
        for s, e, n in w:
            if n == handoff and h_end is None:
                segs.append((cur, e))
                h_end = e
            elif n.startswith(post) and h_end is not None:
                bridges.append((h_end, s))
                cur, h_end = s, None
        last_end = max(e for _, e, _ in w)
        tail = last_end - cur if h_end is None else 0
        if h_end is None:
            segs.append((cur, last_end))
        sb = seg_busy(w, segs)
        out.append(dict(i=i, g0=g0, g1=g1, tid=tid % (1 << 24), w=w, segs=segs, bridges=bridges,
                        first=w[0][0], last=last_end, tail=tail, next_call=nxt,
                        crit=sum(e - s for s, e in segs), seg_busy=sb, busy=sum(sb)))
    for a, b in zip(out, out[1:]):
        a["idle"] = b["first"] - a["last"]
        a["to_call"] = b["g0"] - a["last"]
    return out


def us(x):
    return x / 1e3


def summary(use, joins, kernel, pats):
    """The summary line over the selected replays."""
    mean = statistics.fmean
    gaps = mean([r["crit"] - r["busy"] for r in use])
    idle = [r["idle"] for r in use if "idle" in r]
    f = [f"replays={len(use)}", f"joins={joins}", f"crit_ms={mean([r['crit'] for r in use]) / 1e6:.3f}",
         f"busy_ms={mean([r['busy'] for r in use]) / 1e6:.3f}", f"gaps_ms={gaps / 1e6:.3f}",
         f"gap_us_per_join={us(gaps / (joins + 1)):.1f}",
         f"idle_ms_p50={statistics.median(idle) / 1e6:.3f}" if idle else "idle_ms_p50=-"]
    if pats:
        d = [[e - s for s, e, n in r["w"] if any(p in n for p in pats)] for r in use]
        every = [x for r in d for x in r]
        f += [f"kernel[{kernel}]_us={us(mean(every)):.2f}" if every else f"kernel[{kernel}]_us=-",
              f"kernel[{kernel}]_us_per_replay={us(mean([sum(r) for r in d])):.1f}",
              f"n/replay={mean([len(r) for r in d]):.2f}"]
    return "summary " + " ".join(f)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("sqlite", nargs="?", help="an `nsys export --type sqlite` file of a graph-mode run")
    ap.add_argument("--handoff", default="ds41_ffn_handoff", help="the handoff kernel's name")
    ap.add_argument("--post", default="ds41_ffn_post",
                    help="the prefixes of the post kernels' names, comma-separated")
    ap.add_argument("--replay", type=int, help="print this replay's every segment and bridge")
    ap.add_argument("--from", dest="start", type=int, help="first replay of the per-join table and the summary")
    ap.add_argument("--joins", type=int, help="use the replays with exactly this many joins (default: the modal count)")
    ap.add_argument("--kernel", help="comma-separated substrings of kernel names: their mean µs and count a replay")
    ap.add_argument("--self-test", action="store_true", help="run the reader's own tests on a synthetic trace")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if a.sqlite is None:
        ap.error("a sqlite file (or --self-test) is required")
    post = tuple(p for p in a.post.split(",") if p)
    if not post:
        sys.exit("nsys-bridge.py: --post names no prefix")
    pats = [p for p in (a.kernel or "").split(",") if p]
    if a.kernel is not None and not pats:
        sys.exit("nsys-bridge.py: --kernel names no pattern")

    ks, gl, seen = load(open_ro(a.sqlite), a.handoff, post)
    if not gl:
        sys.exit("nsys-bridge.py: no cuGraphLaunch call in the trace (a graph-mode run with CUDA API tracing)")
    R = replays(ks, gl, a.handoff, post)
    if not R:
        sys.exit("nsys-bridge.py: no kernel starts after a cuGraphLaunch call in the trace")
    print(f"# {a.sqlite}: {len(gl)} cuGraphLaunch calls, {len(ks)} kernels, boundary kernels {seen}")
    if pats:
        hit = Counter(k[2] for k in ks if any(p in k[2] for p in pats))
        if not hit:
            sys.exit(f"nsys-bridge.py: --kernel {a.kernel} matches no kernel in the trace")
        print(f"# --kernel {a.kernel} matches " + ", ".join(f"{n} x{c}" for n, c in sorted(hit.items())))
    print(f"{'replay':>6s} {'tid':>8s} {'launch':>8s} {'call>k0':>8s} {'ret-k0':>8s} {'ret-h0':>8s} {'crit':>8s} "
          f"{'joins':>5s} {'bridges':>8s} {'bridge0':>8s} {'bridge1':>8s} {'tail':>7s} {'idle':>7s} {'>call':>7s} "
          f"{'busy':>8s} {'gaps':>8s}")
    for r in R:
        h0 = r["segs"][0][1] if r["bridges"] else None
        b = [us(p - h) for h, p in r["bridges"]]
        f = lambda v: f"{v:8.1f}" if v is not None else f"{'-':>8s}"
        print(f"{r['i']:6d} {r['tid']:8d} {f(us(r['g1'] - r['g0']))} {f(us(r['first'] - r['g0']))} "
              f"{f(us(r['g1'] - r['first']))} {f(us(r['g1'] - h0) if h0 else None)} {f(us(r['crit']))} {len(b):5d} "
              f"{f(sum(b))} {f(b[0] if b else None)} {f(b[1] if len(b) > 1 else None)} {us(r['tail']):7.1f} "
              f"{us(r['idle']) if 'idle' in r else float('nan'):7.1f} {us(r['to_call']) if 'to_call' in r else float('nan'):7.1f} "
              f"{f(us(r['busy']))} {f(us(r['crit'] - r['busy']))}")
    print("# µs. launch = the call's span; call>k0 = call start -> first kernel; ret-k0 / ret-h0 = the call's return "
          "after the first kernel's start / after handoff 0's end; crit = the critical segments' sum; idle = last "
          "kernel end -> next replay's first kernel; >call = last kernel end -> next launch call; busy = the "
          "kernel time (union) inside the critical segments; gaps = crit - busy")

    counts = Counter(len(r["bridges"]) for r in R)
    want = counts.most_common(1)[0][0] if a.joins is None else a.joins
    first = next((r["i"] for r in R if len(r["bridges"]) == want), None)
    if first is None:
        sys.exit(f"nsys-bridge.py: no replay with {want} joins (join counts in the trace: "
                 + ", ".join(f"{j} x{n}" for j, n in sorted(counts.items())) + ")")
    start = a.start if a.start is not None else first
    use = [r for r in R if r["i"] >= start and len(r["bridges"]) == want]
    if not use:
        sys.exit(f"nsys-bridge.py: no replay with {want} joins from replay {start}")
    if want:
        print()
        print(f"# per join over replays {use[0]['i']}..{use[-1]['i']} ({len(use)} replays with {want} joins): "
              f"bridge = handoff k end -> post k start, seg = the critical segment ending at handoff k, "
              f"gap = its time with no kernel running")
        print(f"{'join':>4s} {'bridge_mean':>11s} {'min':>7s} {'max':>7s} {'seg_mean':>9s} {'gap_mean':>9s}")
        for k in range(want):
            bs = [us(r["bridges"][k][1] - r["bridges"][k][0]) for r in use]
            ss = [us(r["segs"][k][1] - r["segs"][k][0]) for r in use]
            gs = [us(r["segs"][k][1] - r["segs"][k][0] - r["seg_busy"][k]) for r in use]
            print(f"{k:4d} {statistics.fmean(bs):11.1f} {min(bs):7.1f} {max(bs):7.1f} {statistics.fmean(ss):9.1f} "
                  f"{statistics.fmean(gs):9.1f}")
    if a.replay is not None:
        r = next((x for x in R if x["i"] == a.replay), None)
        if r is None:
            sys.exit(f"nsys-bridge.py: no replay {a.replay}")
        print()
        print(f"# replay {a.replay}: times from its first kernel's start (µs); busy = the kernel time inside the "
              f"segment, gap = its length less busy")
        t0 = r["first"]
        for k, (s, e) in enumerate(r["segs"]):
            print(f"  seg {k:3d} {us(s - t0):9.1f} -> {us(e - t0):9.1f}  {us(e - s):8.1f}  "
                  f"busy {us(r['seg_busy'][k]):8.1f}  gap {us(e - s - r['seg_busy'][k]):8.1f}")
            if k < len(r["bridges"]):
                h, p = r["bridges"][k]
                print(f"  bridge {k:3d}              {us(p - h):8.1f}")
    print()
    print(summary(use, want, a.kernel, pats))


# ---------------------------------------------------------------------------------------------------
# The self-test: a synthetic trace with the tables and columns the reader queries, every printed number
# derived on paper below (a fixture unit is U ns = 100 µs, so the times in the tables are in 100 µs).

U = 100_000
H, PC, PS, RT, G, ACC = ("ds41_ffn_handoff_10", "q38_card_shared_add", "q38_shared_add",
                         "qwen35moe_router_fused_512", "q38_gemv", "q38_card_acc")
# (base in units, kernels as (name, start, end) in units after the base). Each replay's launch call runs from 4 µs
# before its first kernel to 2 µs after it.
#   P  2 joins, one layer ending in each post prefix; a shadow kernel (ACC) wholly in bridge 0.
#        segments [0,25] [60,90] [130,145] = 70; busy 21 + 25 + 13 = 59; gaps 11; bridges 35, 40; tail 15
#   B  2 joins, only PC; G at [0,10] and [5,15] overlap (union 15, sum 20), ACC [50,70] starts in bridge 0 and
#        covers the gap [64,66] of segment 1, clipped to [60,70] there.
#        segments [0,23] [60,85] [120,140] = 68; busy 18 + 23 + 19 = 60 (summed: 23 + 31 + 19 = 73); gaps 8;
#        bridges 37, 35; tail 20
#   M1, M2  2 joins, only PC, a router kernel (RT) in each of the two chain segments (8, 8 and 6, 8 units).
#        M1 segments [0,24] [50,82] [100,117] = 73; busy 21 + 27 + 15 = 63; gaps 3 + 5 + 2 = 10; bridges 26, 18
#        M2 segments [0,26] [52,86] [108,127] = 79; busy 22 + 27 + 17 = 66; gaps 4 + 7 + 2 = 13; bridges 26, 22
#   C  0 joins (a draft's walk): one segment [0,30]; busy 12 + 8 + 6 = 26; gaps 4; one RT of 8
# Bases 1000, 1150, 1300, 1420, 1560: idle (last kernel end -> next first) 145 -> 5, B 140 -> 10, M1 117 -> 3,
# M2 127 -> 13 units. The modal join count is 2 (P, B, M1, M2 against C).
FIX = [
    (1000, [(G, 0, 10), (RT, 12, 20), (H, 22, 25), (ACC, 30, 50), (PC, 60, 65), (G, 66, 76), (RT, 77, 83),
            (H, 86, 90), (PS, 130, 135), (G, 137, 145)]),
    (1150, [(G, 0, 10), (G, 5, 15), (H, 20, 23), (ACC, 50, 70), (PC, 60, 64), (G, 66, 80), (H, 82, 85),
            (PC, 120, 124), (G, 125, 140)]),
    (1300, [(G, 0, 10), (RT, 11, 19), (H, 21, 24), (PC, 50, 55), (G, 56, 66), (RT, 68, 76), (H, 78, 82),
            (PC, 100, 105), (G, 107, 117)]),
    (1420, [(G, 0, 12), (RT, 13, 19), (H, 22, 26), (PC, 52, 58), (G, 60, 70), (RT, 72, 80), (H, 83, 86),
            (PC, 108, 113), (G, 115, 127)]),
    (1560, [(G, 0, 12), (RT, 14, 22), (G, 24, 30)]),
]
# V4.1's names, the defaults' (two identical replays 10 units apart). Segments [0,15] [40,61] [80,90] = 46; busy
# 13 + 17 + 8 = 38; gaps 2 + 4 + 2 = 8; bridges 25, 19; tail 10; idle 190 -> 200 base = 10 units.
V41 = [(100, [(G, 0, 10), ("ds41_ffn_handoff", 12, 15), ("ds41_ffn_post_a", 40, 44), (G, 46, 56),
              ("ds41_ffn_handoff", 58, 61), ("ds41_ffn_post_b", 80, 84), (G, 86, 90)])]
V41.append((200, V41[0][1]))

# every column of the per-replay table, as printed, per fixture replay
ROWS = {
    0: {"tid": "77", "launch": "6.0", "call>k0": "4.0", "ret-k0": "2.0", "ret-h0": "-2498.0", "crit": "7000.0",
        "joins": "2", "bridges": "7500.0", "bridge0": "3500.0", "bridge1": "4000.0", "tail": "1500.0",
        "idle": "500.0", ">call": "496.0", "busy": "5900.0", "gaps": "1100.0"},
    1: {"tid": "77", "launch": "6.0", "call>k0": "4.0", "ret-k0": "2.0", "ret-h0": "-2298.0", "crit": "6800.0",
        "joins": "2", "bridges": "7200.0", "bridge0": "3700.0", "bridge1": "3500.0", "tail": "2000.0",
        "idle": "1000.0", ">call": "996.0", "busy": "6000.0", "gaps": "800.0"},
    2: {"tid": "77", "launch": "6.0", "call>k0": "4.0", "ret-k0": "2.0", "ret-h0": "-2398.0", "crit": "7300.0",
        "joins": "2", "bridges": "4400.0", "bridge0": "2600.0", "bridge1": "1800.0", "tail": "1700.0",
        "idle": "300.0", ">call": "296.0", "busy": "6300.0", "gaps": "1000.0"},
    3: {"tid": "77", "launch": "6.0", "call>k0": "4.0", "ret-k0": "2.0", "ret-h0": "-2598.0", "crit": "7900.0",
        "joins": "2", "bridges": "4800.0", "bridge0": "2600.0", "bridge1": "2200.0", "tail": "1900.0",
        "idle": "1300.0", ">call": "1296.0", "busy": "6600.0", "gaps": "1300.0"},
    4: {"tid": "77", "launch": "6.0", "call>k0": "4.0", "ret-k0": "2.0", "ret-h0": "-", "crit": "3000.0",
        "joins": "0", "bridges": "0.0", "bridge0": "-", "bridge1": "-", "tail": "3000.0",
        "idle": "nan", ">call": "nan", "busy": "2600.0", "gaps": "400.0"},
}
BUSY_GAPS = ("busy", "gaps")


def build(path, reps):
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT NOT NULL)")
    db.execute('CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, "end" INTEGER, shortName INTEGER)')
    db.execute('CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (start INTEGER, "end" INTEGER, nameId INTEGER, '
               'globalTid INTEGER)')
    ids = {}

    def sid(name):
        if name not in ids:
            ids[name] = len(ids) + 1
            db.execute("INSERT INTO StringIds VALUES (?, ?)", (ids[name], name))
        return ids[name]

    tid = (5 << 24) | 77
    db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (1, 2, ?, ?)", (sid("cuLaunchKernel"), tid))
    for base, kernels in reps:
        db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?, ?)",
                   (base * U - 4000, base * U + 2000, sid("cuGraphLaunch"), tid))
        for name, s, e in kernels:
            db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?)",
                       ((base + s) * U, (base + e) * U, sid(name)))
    db.commit()
    db.close()


def run(argv):
    """main(argv) in process: (exit code, its stdout and stderr)."""
    out = io.StringIO()
    code = 0
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        try:
            main(argv)
        except SystemExit as e:
            code = e.code if e.code is not None else 0
    return code, out.getvalue()


def table(out, first):
    """The rows of the table whose header line starts with `first`, as {column: text} (the first column is
    each row's key)."""
    lines = out.splitlines()
    at = next(i for i, l in enumerate(lines) if l.split()[:1] == [first] and not l.startswith("#"))
    head = lines[at].split()
    rows = {}
    for l in lines[at + 1:]:
        if not l.strip() or l.startswith("#"):
            break
        t = l.split()
        rows[int(t[0])] = dict(zip(head[1:], t[1:]))
    return rows


def summary_of(out):
    """The summary line as {key: text}."""
    last = out.rstrip("\n").splitlines()[-1]
    if not last.startswith("summary "):
        return {}
    return dict(t.split("=", 1) for t in last.split()[1:])


class Checks:
    """Every check runs; a failed id prints one line, so a mutant's red lines say which assertion it hit."""

    def __init__(self):
        self.n = 0
        self.red = {}

    def expect(self, cid, why, got, want):
        self.n += 1
        if isinstance(want, dict):
            bad = [f"{k}: got {got.get(k)!r} want {want[k]!r}" for k in want if got.get(k) != want[k]]
        else:
            bad = [] if got == want else [f"got {got!r} want {want!r}"]
        if bad:
            more = f" (+{len(bad) - 4} more)" if len(bad) > 4 else ""
            self.red.setdefault(cid, []).append(f"{why} -- " + "; ".join(bad[:4]) + more)

    def finish(self):
        for cid, reds in self.red.items():
            print(f"FAIL {cid}: " + " | ".join(reds))
        if self.red:
            print(f"nsys-bridge: self-test FAILED, {len(self.red)} of the checks red ({self.n} assertions run)")
            return 1
        print(f"nsys-bridge: self-test ok ({self.n} assertions)")
        return 0


def self_test():
    c = Checks()
    with tempfile.TemporaryDirectory() as d:
        main_db, v41_db, bare_db = (os.path.join(d, n) for n in ("main.sqlite", "v41.sqlite", "bare.sqlite"))
        build(main_db, FIX)
        build(v41_db, V41)
        build(bare_db, [])
        both = [main_db, "--handoff", H, "--post", f"{PC},{PS}"]

        code, out = run(both)
        rows = table(out, "replay")
        no_bg = [k for k in ROWS[0] if k not in BUSY_GAPS]
        c.expect("rows-cut", "the per-replay columns of the segments, bridges, launch and idle (the V4.1 reader's "
                 "own arithmetic) moved", {i: {k: rows.get(i, {}).get(k) for k in no_bg} for i in (1, 2, 3, 4)},
                 {i: {k: ROWS[i][k] for k in no_bg} for i in (1, 2, 3, 4)})
        c.expect("rows-cut", "the trace header's call and kernel counts moved",
                 "5 cuGraphLaunch calls, 40 kernels" in out, True)
        c.expect("busy-gaps-plain", "busy must be the kernel time inside the critical segments and gaps crit - busy "
                 "(replays with no overlapping kernels)",
                 {i: {k: rows.get(i, {}).get(k) for k in BUSY_GAPS} for i in (2, 3, 4)},
                 {i: {k: ROWS[i][k] for k in BUSY_GAPS} for i in (2, 3, 4)})
        c.expect("busy-union", "busy must be the UNION of the kernels' intervals inside the segments: two kernels "
                 "running at once, and a shadow kernel running over an in-chain gap, count their covered time once, "
                 "and a kernel from the bridge counts from the segment's start",
                 {k: rows.get(1, {}).get(k) for k in BUSY_GAPS}, {k: ROWS[1][k] for k in BUSY_GAPS})
        c.expect("post-prefixes", "--post a,b must cut a join at either prefix: a layer without card experts ends in "
                 "q38_shared_add, and a reader that keeps only the first prefix loses its join (and the name from the "
                 "boundary kernels printed)",
                 {**rows.get(0, {}), "boundary": f"boundary kernels {sorted([H, PC, PS])}" in out},
                 {**ROWS[0], "boundary": True})

        # the per-join table and the summary of the replays 2 and 3 (M1 and M2: 2 joins, a router in each of two layers)
        code, out = run(both + ["--joins", "2", "--from", "2", "--kernel", "qwen35moe_router"])
        c.expect("summary-means", "the summary's means and the idle's median over the selected replays",
                 summary_of(out),
                 {"replays": "2", "joins": "2", "crit_ms": "7.600", "busy_ms": "6.450", "gaps_ms": "1.150",
                  "gap_us_per_join": "383.3", "idle_ms_p50": "0.800"})
        c.expect("summary-means", "the run exits 0", code, 0)
        c.expect("per-join-table", "the per-join table's means, min, max and the new gap mean over the selected "
                 "replays", table(out, "join"),
                 {0: {"bridge_mean": "2600.0", "min": "2600.0", "max": "2600.0", "seg_mean": "2500.0",
                      "gap_mean": "350.0"},
                  1: {"bridge_mean": "2000.0", "min": "1800.0", "max": "2200.0", "seg_mean": "3300.0",
                      "gap_mean": "600.0"}})
        c.expect("per-join-table", "the table's heading names the replays and the join count",
                 "# per join over replays 2..3 (2 replays with 2 joins)" in out, True)
        key = "kernel[qwen35moe_router]"
        c.expect("kernel-stat", "--kernel's mean µs of one launch, the µs and the count a replay (the router: "
                 "8, 8, 6, 8 units over two replays of two)", summary_of(out),
                 {f"{key}_us": "750.00", f"{key}_us_per_replay": "1500.0", "n/replay": "2.00"})
        c.expect("kernel-stat", "the kernel names matched are printed, with their count over the trace "
                 "(7: 2 + 2 + 2 + 1)", "# --kernel qwen35moe_router matches qwen35moe_router_fused_512 x7" in out, True)
        # a pattern is a substring of the name, not its start (M2 alone: its routers are 6 and 8 units)
        code, out = run(both + ["--joins", "2", "--from", "3", "--kernel", "router_fused_512"])
        c.expect("kernel-stat", "--kernel PATTERN is a substring of the kernel's name, so `router_fused_512` finds "
                 "qwen35moe_router_fused_512", summary_of(out),
                 {"replays": "1", "kernel[router_fused_512]_us": "700.00",
                  "kernel[router_fused_512]_us_per_replay": "1400.0", "n/replay": "2.00"})

        # --joins, on a join count that is not the modal one
        code, out = run(both + ["--joins", "0", "--kernel", "qwen35moe_router"])
        c.expect("joins-select", "--joins N must select the replays with exactly N joins, not the modal count: the "
                 "draft's walks (0-1 joins) are not the verifies' (many joins) and the table must read the one "
                 "asked for",
                 summary_of(out),
                 {"replays": "1", "joins": "0", "crit_ms": "3.000", "busy_ms": "2.600", "gaps_ms": "0.400",
                  "gap_us_per_join": "400.0", "idle_ms_p50": "-", f"{key}_us": "800.00",
                  f"{key}_us_per_replay": "800.0", "n/replay": "1.00"})
        # (one prefix: P then has one join, whatever --post reads, so the counts are the same in every build)
        code, out = run([main_db, "--handoff", H, "--post", PC, "--joins", "9"])
        c.expect("joins-select", "a join count no replay has must end by name with the counts the trace holds, not "
                 "print an empty summary", code,
                 "nsys-bridge.py: no replay with 9 joins (join counts in the trace: 0 x1, 1 x1, 2 x3)")

        # --replay: B's segments and bridges, and each segment's busy and gap
        code, out = run(both + ["--replay", "1"])
        seg = [l.split() for l in out.splitlines() if l.split()[:1] in (["seg"], ["bridge"])]
        c.expect("replay-detail", "--replay prints each segment's start, end and length and each bridge, in µs from "
                 "the replay's first kernel",
                 [t[:6] if t[0] == "seg" else t for t in seg],
                 [["seg", "0", "0.0", "->", "2300.0", "2300.0"], ["bridge", "0", "3700.0"],
                  ["seg", "1", "6000.0", "->", "8500.0", "2500.0"], ["bridge", "1", "3500.0"],
                  ["seg", "2", "12000.0", "->", "14000.0", "2000.0"]])
        c.expect("busy-union", "--replay's per-segment busy and gap are the union's",
                 [t[6:] for t in seg if t[0] == "seg"],
                 [["busy", "1800.0", "gap", "500.0"], ["busy", "2300.0", "gap", "200.0"],
                  ["busy", "1900.0", "gap", "100.0"]])

        # V4.1's names are the defaults: no flag but the file
        code, out = run([v41_db])
        c.expect("v41-defaults", "the defaults (--handoff ds41_ffn_handoff, --post ds41_ffn_post, the modal join "
                 "count) must read a V4.1 trace as before", (code, summary_of(out)),
                 (0, {"replays": "2", "joins": "2", "crit_ms": "4.600", "busy_ms": "3.800", "gaps_ms": "0.800",
                      "gap_us_per_join": "266.7", "idle_ms_p50": "1.000"}))
        v41_row = {"tid": "77", "launch": "6.0", "call>k0": "4.0", "ret-k0": "2.0", "ret-h0": "-1498.0",
                   "crit": "4600.0", "joins": "2", "bridges": "4400.0", "bridge0": "2500.0", "bridge1": "1900.0",
                   "tail": "1000.0", "busy": "3800.0", "gaps": "800.0"}
        c.expect("v41-defaults", "the defaults' per-replay rows", table(out, "replay"),
                 {0: {**v41_row, "idle": "1000.0", ">call": "996.0"}, 1: {**v41_row, "idle": "nan", ">call": "nan"}})
        c.expect("v41-defaults", "the defaults' per-join table", table(out, "join"),
                 {0: {"bridge_mean": "2500.0", "min": "2500.0", "max": "2500.0", "seg_mean": "1500.0",
                      "gap_mean": "200.0"},
                  1: {"bridge_mean": "1900.0", "min": "1900.0", "max": "1900.0", "seg_mean": "2100.0",
                      "gap_mean": "400.0"}})
        c.expect("v41-defaults", "the defaults' boundary kernels", "boundary kernels ['ds41_ffn_handoff', "
                 "'ds41_ffn_post_a', 'ds41_ffn_post_b']" in out, True)

        # an empty selection ends by name
        refusals = [
            (run([bare_db]), "nsys-bridge.py: no cuGraphLaunch call in the trace (a graph-mode run with CUDA API "
                             "tracing)"),
            (run(both + ["--from", "99"]), "nsys-bridge.py: no replay with 2 joins from replay 99"),
            (run(both + ["--kernel", "no_such_kernel"]), "nsys-bridge.py: --kernel no_such_kernel matches no kernel "
                                                         "in the trace"),
            (run(both + ["--replay", "99"]), "nsys-bridge.py: no replay 99"),
        ]
        c.expect("refusals", "an empty trace, selection, pattern or replay must end by name (the message is the exit "
                 "code), never a traceback or an empty summary",
                 [r[0][0] for r in refusals], [r[1] for r in refusals])
        c.expect("refusals", "a run with neither a file nor --self-test is a usage error", run([])[0], 2)
    return c.finish()


if __name__ == "__main__":
    sys.exit(main())
