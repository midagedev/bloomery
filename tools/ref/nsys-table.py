#!/usr/bin/env python3
"""Kernel table of an Nsight Systems sqlite export: one row per (kernel, gridX, gridY, blockX).

  nsys-table.py <run>.sqlite [--steps N] [--pattern a,b] [--sort sum|name|count]

Per row: launches, mean / min / max µs, the summed µs, and per step the summed µs and the launch
count (sum / N, count / N). N is --steps, or else the number of `cuGraphLaunch` calls the trace
recorded (one replay per decode step in generate_ds41's graph mode), or 1 when there is none — the
header says which. --pattern keeps the rows whose kernel name contains any of the comma-separated
substrings. A kernel launched at several geometries gets one row per geometry, which is what tells
a kernel whose grid follows the depth from one that does not.

The trace's µs are the tracer's: kernel durations are close to an untraced run's, the gaps between
them and the host calls are not (nsys-bridge.py reads those). The database is opened read-only.
"""
import argparse
import sqlite3
import sys


def open_ro(path):
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def graph_launches(db):
    names = dict(db.execute("SELECT id, value FROM StringIds"))
    try:
        rows = db.execute("SELECT nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME").fetchall()
    except sqlite3.OperationalError:
        return 0
    return sum(1 for (n,) in rows if names.get(n, "").startswith("cuGraphLaunch"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("sqlite", help="an `nsys export --type sqlite` file")
    ap.add_argument("--steps", type=int, help="divide per-step columns by this (default: cuGraphLaunch calls)")
    ap.add_argument("--pattern", default="", help="comma-separated substrings of kernel names to keep")
    ap.add_argument("--sort", choices=("sum", "name", "count"), default="sum", help="row order (default sum, largest first)")
    a = ap.parse_args()

    db = open_ro(a.sqlite)
    if a.steps is not None:
        steps, why = a.steps, "--steps"
    else:
        n = graph_launches(db)
        steps, why = (n, "cuGraphLaunch calls") if n else (1, "no cuGraphLaunch in the trace")
    if steps < 1:
        sys.exit("nsys-table.py: --steps must be at least 1")
    pats = [p for p in a.pattern.split(",") if p]
    q = """SELECT s.value, k.gridX, k.gridY, k.blockX, count(*), avg(k.end - k.start), min(k.end - k.start),
                  max(k.end - k.start), sum(k.end - k.start)
           FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id = k.shortName
           GROUP BY s.value, k.gridX, k.gridY, k.blockX"""
    rows = [r for r in db.execute(q) if not pats or any(p in r[0] for p in pats)]
    key = {"sum": lambda r: -r[8], "name": lambda r: (r[0], r[1], r[2], r[3]), "count": lambda r: -r[4]}[a.sort]
    rows.sort(key=key)
    total = sum(r[8] for r in rows)
    print(f"# {a.sqlite}: {len(rows)} rows, steps N={steps} ({why}), kernel time {total / 1e3 / steps:.1f} µs/step"
          f"{' pattern=' + a.pattern if pats else ''}")
    print(f"{'kernel':34s} {'grid':>11s} {'block':>5s} {'n':>6s} {'n/step':>7s} {'mean_us':>8s} {'min_us':>8s} "
          f"{'max_us':>8s} {'sum_us':>10s} {'us/step':>9s} {'share':>6s}")
    for name, gx, gy, bx, n, avg, mn, mx, sm in rows:
        print(f"{name[:34]:34s} {f'{gx}x{gy}':>11s} {bx:5d} {n:6d} {n / steps:7.1f} {avg / 1e3:8.2f} {mn / 1e3:8.2f} "
              f"{mx / 1e3:8.2f} {sm / 1e3:10.1f} {sm / 1e3 / steps:9.1f} {100 * sm / total if total else 0:5.1f}%")


if __name__ == "__main__":
    main()
