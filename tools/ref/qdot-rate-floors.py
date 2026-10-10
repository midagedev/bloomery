#!/usr/bin/env python3
"""The qdot-rate floor: every Rust row that has an ik twin holds its same-round rate ratio ours/ik at or
above the floor tools/ref/qdot-rate-floors.tsv gives it.

A `just measure-qdot-rate` log (tools/ref/qdot-rate.sh) is rounds of two arms: the Rust `qdot-rate`
lines (`<TYPE>  k=<K> rows <N> x <B> B x <P> passes in <duration> = … GB/s`) and ik's harness lines
(`ik <TYPE> [<form>] rows <N> x <B> B x <P> passes in <S> s = … GB/s`). A rate is read from the line's
bytes and time, not its rounded GB/s. A floors row `type k ik_harness floor` pairs the Rust row of
that type and k with the ik line of that type and the same row bytes in the same round; the harness
name is the runner's (qdot-rate.sh runs the harnesses the table names).

    qdot-rate-floors.py check LOG [--floors TSV]   rc 1 naming each row: a ratio under its floor, a
                                                    Rust row or its ik twin missing from a round, a
                                                    log with no round
    qdot-rate-floors.py derive LOG [--floors TSV]  per row: each round's ratio, the mean, the SD and
                                                    the floor the table's rule gives (below)
    qdot-rate-floors.py harnesses [--floors TSV]   the table's ik harness names, one line
    qdot-rate-floors.py --self-test                 canned logs: green, a row under its floor, a row
                                                    with no ik twin, a missing row, no round

The floor rule (derive): each round's ratio of each row is checked, CHECKS = rows x 6 rounds of them
in a sitting at most, so the floor is the lower end of the one-sided prediction interval of one new
round at a family-wise 5 %: mean - t(1 - 0.05 / CHECKS, n - 1) * sd * sqrt(1 + 1/n) from the log's n
rounds, less the print resolution of the two times (half a unit of the last digit of each, as a
fraction of the mean).
"""

import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FLOORS = os.path.join(HERE, "qdot-rate-floors.tsv")
ROUND = re.compile(r"^--- round (\d+):")
RUST = re.compile(r"^(\S+)\s+k=(\d+) rows (\d+) x (\d+) B x (\d+) passes in ([0-9.]+)(ns|µs|us|ms|s) = ")
IK = re.compile(r"^ik (\S+)(?: \S+)? rows (\d+) x (\d+) B x (\d+) passes in ([0-9.]+) s = ")
UNIT = {"ns": 1e-9, "µs": 1e-6, "us": 1e-6, "ms": 1e-3, "s": 1.0}
# The most rounds a sitting runs (docs/cards/iqhost-rate.card's 6): the checks one sitting makes are
# the table's rows times this.
SITTING_ROUNDS = 6


def t_cdf(t, df):
    """Student's t CDF by Simpson's rule over the density (df >= 1, t >= 0)."""
    c = math.gamma((df + 1) / 2) / (math.sqrt(df * math.pi) * math.gamma(df / 2))
    n = 20000
    h = t / n
    f = [c * (1 + (i * h) ** 2 / df) ** (-(df + 1) / 2) for i in range(n + 1)]
    return 0.5 + h / 3 * (f[0] + f[-1] + 4 * sum(f[1:-1:2]) + 2 * sum(f[2:-1:2]))


def t_quantile(p, df):
    """The t with t_cdf(t, df) = p, by bisection (0.5 < p < 1)."""
    lo, hi = 0.0, 1000.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


class FloorError(Exception):
    pass


def load_floors(path):
    rows = []
    for n, line in enumerate(open(path), 1):
        s = line.split("#", 1)[0].strip()
        if not s:
            continue
        f = s.split()
        if len(f) != 4:
            raise FloorError(f"{path}:{n}: a row is `type k ik_harness floor`, got {s!r}")
        try:
            rows.append((f[0], int(f[1]), f[2], float(f[3])))
        except ValueError:
            raise FloorError(f"{path}:{n}: k is an integer and floor a number, got {s!r}") from None
    if not rows:
        raise FloorError(f"{path}: no floor row")
    return rows


def resolution(text):
    """Half a unit of the time's last printed digit, as a fraction of the time."""
    digits = text.split(".", 1)[1] if "." in text else ""
    return 0.5 * 10 ** -len(digits) / float(text)


def parse_log(path):
    """[(round number, {(type, k): (GB/s, row bytes, resolution)}, {(type, row bytes): [(GB/s, resolution)]})]"""
    rounds = []
    for line in open(path, errors="replace"):
        line = line.rstrip("\n")
        m = ROUND.match(line)
        if m:
            rounds.append((int(m.group(1)), {}, {}))
            continue
        if not rounds:
            continue
        _, ours, ik = rounds[-1]
        m = IK.match(line)
        if m:
            ty, rows, rb, passes, t = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)), m.group(5)
            ik.setdefault((ty, rb), []).append((rows * rb * passes / float(t) / 1e9, resolution(t)))
            continue
        m = RUST.match(line)
        if m:
            ty, k, rows, rb, passes = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5))
            secs = float(m.group(6)) * UNIT[m.group(7)]
            ours[(ty, k)] = (rows * rb * passes / secs / 1e9, rb, resolution(m.group(6)))
    return rounds


def ratios(rounds, row):
    """[(round, ratio, resolution)] for one floors row, and the problems that row has, by name."""
    ty, k, harness, _ = row
    out, problems = [], []
    for r, ours, ik in rounds:
        if (ty, k) not in ours:
            problems.append(f"{ty}@{k}: round {r} has no qdot-rate row {ty} k={k}")
            continue
        g, rb, res = ours[(ty, k)]
        twins = ik.get((ty, rb), [])
        if not twins:
            problems.append(f"{ty}@{k}: round {r} has no ik twin — no `ik {ty} … x {rb} B` line ({harness})")
            continue
        if len(twins) > 1:
            problems.append(f"{ty}@{k}: round {r} has {len(twins)} ik lines of {ty} at {rb} B")
            continue
        gi, resi = twins[0]
        out.append((r, g / gi, res + resi))
    return out, problems


def check(log, floors):
    rows = load_floors(floors)
    rounds = parse_log(log)
    if not rounds:
        print(f"qdot-rate-floors: {log} has no `--- round N:` line — not a qdot-rate.sh log")
        return 1
    bad = 0
    for row in rows:
        ty, k, harness, floor = row
        rs, problems = ratios(rounds, row)
        for p in problems:
            print(f"qdot-rate-floors: RED {p}")
            bad += 1
        for r, q, _ in rs:
            mark = "ok " if q >= floor else "RED"
            print(f"qdot-rate-floors: {mark} {ty}@{k} round {r}: ours/ik {q:.3f} (floor {floor:.2f}, {harness})")
            if q < floor:
                bad += 1
    print(f"qdot-rate-floors: {'ok' if bad == 0 else f'{bad} red'} — {len(rows)} rows x {len(rounds)} rounds")
    return 1 if bad else 0


def derive(log, floors):
    rows = load_floors(floors)
    rounds = parse_log(log)
    checks = len(rows) * SITTING_ROUNDS
    print(f"rule: {len(rows)} rows x {SITTING_ROUNDS} rounds = {checks} checks, one-sided p = 1 - 0.05/{checks}")
    for row in rows:
        ty, k, harness, floor = row
        rs, problems = ratios(rounds, row)
        for p in problems:
            print(f"{ty}@{k}\t{p}")
        n = len(rs)
        if n < 2:
            print(f"{ty}@{k}\t{n} round(s): no SD")
            continue
        qs = [q for _, q, _ in rs]
        mean = sum(qs) / n
        sd = math.sqrt(sum((q - mean) ** 2 for q in qs) / (n - 1))
        res = max(x for _, _, x in rs)
        t = t_quantile(1 - 0.05 / checks, n - 1)
        lo = mean - t * sd * math.sqrt(1 + 1 / n) - res * mean
        per = " ".join(f"{q:.3f}" for q in qs)
        print(f"{ty}@{k}\t{harness}\trounds {per}\tmean {mean:.3f}\tsd {sd:.4f}\tt({n - 1}) {t:.3f}"
              f"\tres {res:.4f}\tfloor {lo:.3f}\t(table {floor:.2f})")
    return 0


def self_test():
    import tempfile

    tsv = "Q4_K 2048 q4k_x4_rate 0.85\nIQ3_S 4096 iq3s_rate 0.90  # PIN(2026-10-10): spec\n"

    def rnd(n, ours_s, ik_s, drop_ik=False, drop_rust=False):
        lines = [f"--- round {n}: ik then rust"]
        lines.append("ik Q4_K x4 rows 360448 x 1152 B x 6 passes in 0.148 s = 16.9 GB/s (dst[0] 0)")
        if not drop_ik:
            lines.append(f"ik IQ3_S rows 360448 x 1760 B x 6 passes in {ik_s} s = 4.4 GB/s (dst[0] 0)")
        lines.append("Q4_K  k=2048 rows 360448 x 1152 B x 6 passes in 153.770ms = 16.2 GB/s (sum NaN)")
        if not drop_rust:
            lines.append(f"IQ3_S  k=4096 rows 360448 x 1760 B x 6 passes in {ours_s} = 0.6 GB/s (sum 1)")
        return lines

    cases = [
        ("green", rnd(1, "720.000ms", "0.865") + rnd(2, "721.000ms", "0.864"), 0, "ok — 2 rows x 2 rounds"),
        ("under", rnd(1, "6.503s", "0.865") + rnd(2, "720.000ms", "0.864"), 1,
         "RED IQ3_S@4096 round 1: ours/ik 0.133"),
        ("no twin", rnd(1, "720.000ms", "0.865", drop_ik=True), 1,
         "RED IQ3_S@4096: round 1 has no ik twin — no `ik IQ3_S … x 1760 B` line (iq3s_rate)"),
        ("missing", rnd(1, "720.000ms", "0.865", drop_rust=True), 1,
         "RED IQ3_S@4096: round 1 has no qdot-rate row IQ3_S k=4096"),
        ("no round", ["Q4_K  k=2048 rows 360448 x 1152 B x 6 passes in 153.770ms = 16.2 GB/s"], 1,
         "has no `--- round N:` line"),
    ]
    with tempfile.TemporaryDirectory() as d:
        fl = os.path.join(d, "floors.tsv")
        open(fl, "w").write(tsv)
        for name, lines, want_rc, want in cases:
            log = os.path.join(d, "log")
            open(log, "w").write("\n".join(lines) + "\n")
            import contextlib
            import io

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = check(log, fl)
            out = buf.getvalue()
            if rc != want_rc or want not in out:
                print(out)
                print(f"qdot-rate-floors self-test: case {name!r}: rc {rc} (want {want_rc}), missing {want!r}")
                return 1
        bad = os.path.join(d, "bad.tsv")
        open(bad, "w").write("IQ3_S 4096 iq3s_rate\n")
        try:
            load_floors(bad)
            print("qdot-rate-floors self-test: a three-column row was accepted")
            return 1
        except FloorError:
            pass
    load_floors(FLOORS)
    print("qdot-rate-floors self-test: ok (green, under, no twin, missing, no round, malformed row, the table loads)")
    return 0


def main(argv):
    if argv[1:] == ["--self-test"]:
        return self_test()
    args = argv[1:]
    floors = FLOORS
    if "--floors" in args:
        i = args.index("--floors")
        floors = args[i + 1]
        del args[i : i + 2]
    try:
        if args[:1] == ["harnesses"] and len(args) == 1:
            print(" ".join(dict.fromkeys(h for _, _, h, _ in load_floors(floors))))
            return 0
        if len(args) == 2 and args[0] == "check":
            return check(args[1], floors)
        if len(args) == 2 and args[0] == "derive":
            return derive(args[1], floors)
    except (FloorError, OSError) as e:
        print(f"qdot-rate-floors: {e}")
        return 64
    print(__doc__.split("\n\n")[2] if len(__doc__.split("\n\n")) > 2 else __doc__, file=sys.stderr)
    return 64


if __name__ == "__main__":
    sys.exit(main(sys.argv))
