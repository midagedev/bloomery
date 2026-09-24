#!/usr/bin/env python3
"""Per replay of a V4.1 graph-mode trace: the card's critical segments, the host bridges, the launch
call, and the idle between replays — read from an Nsight Systems sqlite export.

  nsys-bridge.py <run>.sqlite [--handoff NAME] [--post PREFIX] [--replay R] [--from R]

A replay is the kernels that start between one `cuGraphLaunch` call and the next. Per hybrid layer
the stream runs handoff -> go -> [card work] -> wait -> post (chain/ffn.rs); the go and the wait are
stream memory operations, which the trace records as nothing. So, per replay:

  critical segments  first kernel start -> handoff 0 end, then each post start -> the next handoff
                     end, then the last post start -> the last kernel end (the tail): the card's own
                     chain with the host bridges cut out. Their sum is `crit`.
  bridge k           handoff k end -> post k start: go to join as the card sees it (the host's leg
                     plus both signalling latencies). k counts the joins in chain order, from 0 —
                     the k-th hybrid layer the replay serves, not the model's layer number.
  launch             the `cuGraphLaunch` call's own span, where its first kernel started relative to
                     the call, when the call returned relative to that kernel, and relative to the
                     end of handoff 0 (positive: layer 0's go was issued before the host could
                     serve it, which it does only after the call returns — unless the launch runs
                     on its own thread, whose id the `tid` column shows).
  idle               the last kernel end -> the next replay's first kernel start, and -> the next
                     launch call's start (the host turnaround between steps).

--handoff (default `ds41_ffn_handoff`) and --post (default the prefix `ds41_ffn_post`) name the
boundary kernels; the names the trace holds are printed so a rename shows. --replay R prints that
replay's every segment and bridge. The per-join table is over the replays from --from (default: the
first replay with the modal join count) to the last. µs are the tracer's; the tracer stretches the
launch call itself (see ds41join's launch_us for the untraced value). Read-only.
"""
import argparse
import sqlite3
import statistics
import sys
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
        out.append(dict(i=i, g0=g0, g1=g1, tid=tid % (1 << 24), w=w, segs=segs, bridges=bridges,
                        first=w[0][0], last=last_end, tail=tail, next_call=nxt))
    for a, b in zip(out, out[1:]):
        a["idle"] = b["first"] - a["last"]
        a["to_call"] = b["g0"] - a["last"]
    return out


def us(x):
    return x / 1e3


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("sqlite", help="an `nsys export --type sqlite` file of a graph-mode run")
    ap.add_argument("--handoff", default="ds41_ffn_handoff", help="the handoff kernel's name")
    ap.add_argument("--post", default="ds41_ffn_post", help="the prefix of the post kernels' names")
    ap.add_argument("--replay", type=int, help="print this replay's every segment and bridge")
    ap.add_argument("--from", dest="start", type=int, help="first replay of the per-join table")
    a = ap.parse_args()

    ks, gl, seen = load(open_ro(a.sqlite), a.handoff, a.post)
    if not gl:
        sys.exit("nsys-bridge.py: no cuGraphLaunch call in the trace (a graph-mode run with CUDA API tracing)")
    R = replays(ks, gl, a.handoff, a.post)
    print(f"# {a.sqlite}: {len(gl)} cuGraphLaunch calls, {len(ks)} kernels, boundary kernels {seen}")
    print(f"{'replay':>6s} {'tid':>8s} {'launch':>8s} {'call>k0':>8s} {'ret-k0':>8s} {'ret-h0':>8s} {'crit':>8s} "
          f"{'joins':>5s} {'bridges':>8s} {'bridge0':>8s} {'bridge1':>8s} {'tail':>7s} {'idle':>7s} {'>call':>7s}")
    for r in R:
        h0 = r["segs"][0][1] if r["bridges"] else None
        b = [us(p - h) for h, p in r["bridges"]]
        crit = sum(e - s for s, e in r["segs"])
        f = lambda v: f"{v:8.1f}" if v is not None else f"{'-':>8s}"
        print(f"{r['i']:6d} {r['tid']:8d} {f(us(r['g1'] - r['g0']))} {f(us(r['first'] - r['g0']))} "
              f"{f(us(r['g1'] - r['first']))} {f(us(r['g1'] - h0) if h0 else None)} {f(us(crit))} {len(b):5d} "
              f"{f(sum(b))} {f(b[0] if b else None)} {f(b[1] if len(b) > 1 else None)} {us(r['tail']):7.1f} "
              f"{us(r['idle']) if 'idle' in r else float('nan'):7.1f} {us(r['to_call']) if 'to_call' in r else float('nan'):7.1f}")
    print("# µs. launch = the call's span; call>k0 = call start -> first kernel; ret-k0 / ret-h0 = the call's return "
          "after the first kernel's start / after handoff 0's end; crit = the critical segments' sum; idle = last "
          "kernel end -> next replay's first kernel; >call = last kernel end -> next launch call")

    modal = Counter(len(r["bridges"]) for r in R).most_common(1)[0][0]
    start = a.start if a.start is not None else next(r["i"] for r in R if len(r["bridges"]) == modal)
    use = [r for r in R if r["i"] >= start and len(r["bridges"]) == modal]
    if modal and use:
        print()
        print(f"# per join over replays {use[0]['i']}..{use[-1]['i']} ({len(use)} replays with {modal} joins): "
              f"bridge = handoff k end -> post k start, seg = the critical segment ending at handoff k")
        print(f"{'join':>4s} {'bridge_mean':>11s} {'min':>7s} {'max':>7s} {'seg_mean':>9s}")
        for k in range(modal):
            bs = [us(r["bridges"][k][1] - r["bridges"][k][0]) for r in use]
            ss = [us(r["segs"][k][1] - r["segs"][k][0]) for r in use]
            print(f"{k:4d} {statistics.fmean(bs):11.1f} {min(bs):7.1f} {max(bs):7.1f} {statistics.fmean(ss):9.1f}")
    if a.replay is not None:
        r = next((x for x in R if x["i"] == a.replay), None)
        if r is None:
            sys.exit(f"nsys-bridge.py: no replay {a.replay}")
        print()
        print(f"# replay {a.replay}: times from its first kernel's start (µs)")
        t0 = r["first"]
        for k, (s, e) in enumerate(r["segs"]):
            print(f"  seg {k:3d} {us(s - t0):9.1f} -> {us(e - t0):9.1f}  {us(e - s):8.1f}")
            if k < len(r["bridges"]):
                h, p = r["bridges"][k]
                print(f"  bridge {k:3d}              {us(p - h):8.1f}")


if __name__ == "__main__":
    main()
