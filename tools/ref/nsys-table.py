#!/usr/bin/env python3
"""Kernel table of an Nsight Systems sqlite export: one row per (kernel, gridX, gridY, blockX).

  nsys-table.py <run>.sqlite [--steps N] [--pattern a,b] [--sort sum|name|count]
                              [--kernel REGEX]... [--min-per-token X] [--tsv]

Per row: launches, mean / min / max µs, the summed µs, and per step the summed µs and the launch
count (sum / N, count / N). N is --steps, or else the number of `cuGraphLaunch` calls the trace
recorded (one replay per decode step in generate_ds41's graph mode), or 1 when there is none — the
header says which. --pattern keeps the rows whose kernel name contains any of the comma-separated
substrings, --kernel the rows whose name matches REGEX (searched; repeatable: a name any one pattern
matches stays), and --min-per-token the rows whose launch count per step is X or more. A kernel
launched at several geometries gets one row per geometry, which is what tells a kernel whose grid
follows the depth from one that does not.

--tsv prints one row a launch shape instead, every kernel of the trace grouped by its demangled
name and its full grid and block, the per-token columns divided by the same N:

  kernel  template  grid  block  launches  per_token  us_mean  us_p50  ms_per_token

kernel is the demangled name before the outer '<' (before the '(' of the parameter list when the
kernel is no template), without its 'void ' return type; template is the text between that '<' and
its matching '>' with spaces removed (empty when the kernel is no template). grid is x when y = z =
1, else x*y or x*y*z; block is written the same way. per_token is launches / N (2 decimals); us_mean
and us_p50 are the mean and the median of a launch's end - start in ns / 1000 (2 decimals);
ms_per_token is the group's summed ns / N / 1e6 (3 decimals). Rows sort by ms_per_token, largest
first (ties by kernel, template, grid, block). Here --pattern and --kernel match the demangled name.

The trace's µs are the tracer's: kernel durations are close to an untraced run's, the gaps between
them and the host calls are not (nsys-bridge.py reads those). The database is opened read-only. A
refusal — a file that is not there, a database without the kernel or string table, a --steps below
1, a --min-per-token that is not a finite number 0 or more, a --kernel that is not a regex, or no
row left after the filters in --tsv mode — is one `nsys-table.py: …` line on stderr, exit 64.
"""
import argparse
import math
import os
import re
import sqlite3
import statistics
import subprocess
import sys
import tempfile

KERNEL_TABLE = "CUPTI_ACTIVITY_KIND_KERNEL"
TSV_HEADER = "kernel\ttemplate\tgrid\tblock\tlaunches\tper_token\tus_mean\tus_p50\tms_per_token"


class Refusal(Exception):
    """A named refusal: one `nsys-table.py: …` line on stderr, exit status 64."""


def dims(x, y, z):
    """A launch dimension as written: x when y = z = 1, else x*y or x*y*z."""
    if y == 1 and z == 1:
        return str(x)
    if z == 1:
        return f"{x}*{y}"
    return f"{x}*{y}*{z}"


def split_name(demangled):
    """(kernel, template) of a demangled kernel name: the part before the outer '<' — before the '('
    of the parameter list when the name has no template — without its 'void ' return type, and the
    text between that '<' and its matching '>' with spaces removed ("" when there is no template).
    A '(' inside the template (as '(ggml_type)40') sits at angle depth 1 and never ends the name."""
    name = demangled[5:] if demangled.startswith("void ") else demangled
    angle, lt = 0, -1
    for j, c in enumerate(name):
        if c == "<":
            if angle == 0:
                lt = j
            angle += 1
        elif c == ">" and angle > 0:
            angle -= 1
            if angle == 0:
                return name[:lt], name[lt + 1:j].replace(" ", "")
        elif c == "(" and angle == 0:
            return name[:j], ""
    if lt >= 0:
        raise Refusal(f"a demangled name whose template never closes: {demangled!r}")
    return name, ""


def open_ro(path):
    """The export's sqlite, read-only, with the kernel and string tables an export holds checked."""
    if not os.path.isfile(path):
        raise Refusal(f"{path} is not a file")
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        tables = {t for (t,) in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    except sqlite3.DatabaseError as e:
        db.close()
        raise Refusal(f"{path} is not an sqlite database: {e}") from e
    for t in (KERNEL_TABLE, "StringIds"):
        if t not in tables:
            db.close()
            raise Refusal(f"{path} has no {t} table, not an Nsight Systems kernel export")
    return db


def graph_launches(db):
    names = dict(db.execute("SELECT id, value FROM StringIds"))
    try:
        rows = db.execute("SELECT nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME").fetchall()
    except sqlite3.OperationalError:
        return 0
    return sum(1 for (n,) in rows if names.get(n, "").startswith("cuGraphLaunch"))


def row_filter(pats, regexes, min_per_token, steps):
    """keep(name, launches): the --pattern, --kernel and --min-per-token verdict on one group of the
    table's rows — the name filters on the name the mode groups by (the short name by default, the
    demangled one with --tsv), the bound on the group's launch count per step."""
    def keep(name, launches):
        if pats and not any(p in name for p in pats):
            return False
        if regexes and not any(p.search(name) for p in regexes):
            return False
        return launches / steps >= min_per_token

    return keep


def tsv_table(db, a, keep, steps):
    """The TSV lines: the header, then one row a launch shape, ms_per_token largest first."""
    q = (f'SELECT s.value, k.gridX, k.gridY, k.gridZ, k.blockX, k.blockY, k.blockZ, k."end" - k."start" '
         f"FROM {KERNEL_TABLE} k JOIN StringIds s ON s.id = k.demangledName")
    groups = {}
    try:
        for row in db.execute(q):
            if None in row:
                raise Refusal(f"a kernel row with a NULL column: {row!r}")
            groups.setdefault(row[:7], []).append(row[7])
    except sqlite3.OperationalError as e:
        raise Refusal(f"cannot read the kernel rows: {e}") from e
    rows = []
    for (name, gx, gy, gz, bx, by, bz), durations in groups.items():
        if not keep(name, len(durations)):
            continue
        kernel, template = split_name(name)
        rows.append((sum(durations), kernel, template, dims(gx, gy, gz), dims(bx, by, bz), len(durations),
                     len(durations) / steps, statistics.mean(durations) / 1000, statistics.median(durations) / 1000))
    if not rows:
        why = []
        if [p for p in a.pattern.split(",") if p]:
            why.append(f"pattern={a.pattern}")
        if a.kernel:
            why.append("kernel=" + "|".join(a.kernel))
        if a.min_per_token:
            why.append(f"min-per-token={a.min_per_token:g}")
        raise Refusal(f"no kernel rows{' after the filters (' + ', '.join(why) + ')' if why else ' in the trace'}")
    rows.sort(key=lambda r: (-r[0], r[1], r[2], r[3], r[4]))
    out = [TSV_HEADER]
    for total, kernel, template, grid, block, launches, per_token, mean_us, p50_us in rows:
        out.append(f"{kernel}\t{template}\t{grid}\t{block}\t{launches}\t{per_token:.2f}\t{mean_us:.2f}"
                   f"\t{p50_us:.2f}\t{total / steps / 1e6:.3f}")
    return out


def main(argv):
    if argv == ["--self-test"]:
        return self_test()
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("sqlite", help="an `nsys export --type sqlite` file")
    ap.add_argument("--steps", type=int, help="divide per-step columns by this (default: cuGraphLaunch calls)")
    ap.add_argument("--pattern", default="", help="comma-separated substrings of kernel names to keep")
    ap.add_argument("--sort", choices=("sum", "name", "count"), default="sum", help="row order (default sum, largest first)")
    ap.add_argument("--kernel", action="append", default=[], metavar="REGEX",
                    help="a regex a kept row's kernel name matches (searched; the demangled name with --tsv); repeatable")
    ap.add_argument("--min-per-token", type=float, default=0.0,
                    help="drop the rows whose launch count per step is below this (a finite number, 0 or more)")
    ap.add_argument("--tsv", action="store_true",
                    help="one row a launch shape, grouped by the demangled name and the full grid and block, as a TSV")
    a = ap.parse_args(argv)
    if a.steps is not None and a.steps < 1:
        raise Refusal("--steps must be at least 1")
    regexes = []
    for raw in a.kernel:
        try:
            regexes.append(re.compile(raw))
        except re.error as e:
            raise Refusal(f"--kernel {raw!r} is not a regex: {e}") from e
    if not math.isfinite(a.min_per_token) or a.min_per_token < 0:
        raise Refusal(f"--min-per-token must be a finite number 0 or more, got {a.min_per_token}")
    db = open_ro(a.sqlite)
    if a.steps is not None:
        steps, why = a.steps, "--steps"
    else:
        n = graph_launches(db)
        steps, why = (n, "cuGraphLaunch calls") if n else (1, "no cuGraphLaunch in the trace")
    pats = [p for p in a.pattern.split(",") if p]
    keep = row_filter(pats, regexes, a.min_per_token, steps)
    if a.tsv:
        print("\n".join(tsv_table(db, a, keep, steps)))
        return 0
    q = """SELECT s.value, k.gridX, k.gridY, k.blockX, count(*), avg(k.end - k.start), min(k.end - k.start),
                  max(k.end - k.start), sum(k.end - k.start)
           FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id = k.shortName
           GROUP BY s.value, k.gridX, k.gridY, k.blockX"""
    rows = [r for r in db.execute(q) if keep(r[0], r[4])]
    key = {"sum": lambda r: -r[8], "name": lambda r: (r[0], r[1], r[2], r[3]), "count": lambda r: -r[4]}[a.sort]
    rows.sort(key=key)
    total = sum(r[8] for r in rows)
    extra = ((f" pattern={a.pattern}" if pats else "") + (f" kernel={'|'.join(a.kernel)}" if a.kernel else "")
             + (f" min-per-token={a.min_per_token:g}" if a.min_per_token else ""))
    print(f"# {a.sqlite}: {len(rows)} rows, steps N={steps} ({why}), kernel time {total / 1e3 / steps:.1f} µs/step{extra}")
    print(f"{'kernel':34s} {'grid':>11s} {'block':>5s} {'n':>6s} {'n/step':>7s} {'mean_us':>8s} {'min_us':>8s} "
          f"{'max_us':>8s} {'sum_us':>10s} {'us/step':>9s} {'share':>6s}")
    for name, gx, gy, bx, n, avg, mn, mx, sm in rows:
        print(f"{name[:34]:34s} {f'{gx}x{gy}':>11s} {bx:5d} {n:6d} {n / steps:7.1f} {avg / 1e3:8.2f} {mn / 1e3:8.2f} "
              f"{mx / 1e3:8.2f} {sm / 1e3:10.1f} {sm / 1e3 / steps:9.1f} {100 * sm / total if total else 0:5.1f}%")
    return 0


def build_fixture(path, with_kernel_table=True):
    """The self-test's nsys sqlite: three kernels at two launch shapes each. An nsys export holds
    more columns than these; the reader names the ones it selects, these are they."""
    names = [
        "void mul_mat_vec_q<(ggml_type)40>(__grid_constant__ const mul_mat_vec_params, const void *, "
        "const void *, float *, int, int, long)",
        "void ssm_scan_f32_group<4, 128>(const ssm_scan_params &, float *)",
        "void k_argsort_f32_i32(const float *, int *, int)",
    ]
    shorts = ["mul_mat_vec_q", "ssm_scan_f32_group", "k_argsort_f32_i32"]
    # (kernel, grid, block, [a launch's duration in ns])
    launches = [
        (0, (18560, 1, 1), (32, 4, 1), [62000, 64000, 66000]),
        (0, (4096, 1, 1), (32, 4, 1), [33000, 35000]),
        (1, (2048, 2, 1), (128, 1, 1), [10000, 12000, 14000, 20000]),
        (1, (2048, 1, 1), (128, 1, 1), [5000, 9000]),
        (2, (1, 1, 1), (512, 1, 1), [8000, 12000]),
        (2, (1, 2, 3), (256, 1, 1), [4000, 4000]),
    ]
    db = sqlite3.connect(path)
    with db:
        db.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT NOT NULL)")
        for i, name in enumerate(names):
            db.execute("INSERT INTO StringIds (id, value) VALUES (?, ?)", (i, name))
        for i, name in enumerate(shorts):
            db.execute("INSERT INTO StringIds (id, value) VALUES (?, ?)", (i + 10, name))
        if with_kernel_table:
            db.execute(f"CREATE TABLE {KERNEL_TABLE} (start INTEGER NOT NULL, end INTEGER NOT NULL, "
                       "demangledName INTEGER NOT NULL, shortName INTEGER NOT NULL, gridX INTEGER NOT NULL, "
                       "gridY INTEGER NOT NULL, gridZ INTEGER NOT NULL, blockX INTEGER NOT NULL, "
                       "blockY INTEGER NOT NULL, blockZ INTEGER NOT NULL)")
            t = 1_000_000
            for ki, grid, block, durations in launches:
                for ns in durations:
                    gx, gy, gz = grid
                    bx, by, bz = block
                    db.execute(f"INSERT INTO {KERNEL_TABLE} VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                               (t, t + ns, ki, ki + 10, gx, gy, gz, bx, by, bz))
                    t += 1_000_000
    db.close()


# The default table of the self-test's fixture at --steps 2, byte for byte the lines the tool printed
# before the --tsv columns were folded in (the header's path stands in for the temp fixture's).
DEFAULT_TABLE = """\
# <fixture>: 6 rows, steps N=2 (--steps), kernel time 179.0 µs/step
kernel                                    grid block      n  n/step  mean_us   min_us   max_us     sum_us   us/step  share
mul_mat_vec_q                          18560x1    32      3     1.5    64.00    62.00    66.00      192.0      96.0  53.6%
mul_mat_vec_q                           4096x1    32      2     1.0    34.00    33.00    35.00       68.0      34.0  19.0%
ssm_scan_f32_group                      2048x2   128      4     2.0    14.00    10.00    20.00       56.0      28.0  15.6%
k_argsort_f32_i32                          1x1   512      2     1.0    10.00     8.00    12.00       20.0      10.0   5.6%
ssm_scan_f32_group                      2048x1   128      2     1.0     7.00     5.00     9.00       14.0       7.0   3.9%
k_argsort_f32_i32                          1x2   256      2     1.0     4.00     4.00     4.00        8.0       4.0   2.2%
"""


def run(args):
    """(rc, stdout, stderr) of this tool run as a process with args."""
    p = subprocess.run([sys.executable, os.path.abspath(__file__), *args], capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


def check_output(name, args, want):
    """A run that must print want (its lines) and nothing on stderr."""
    rc, out, err = run(args)
    got = out.splitlines()
    assert rc == 0 and err == "" and got == want, f"{name}: rc={rc} err={err!r}\n  got  {got!r}\n  want {want!r}"


def check_refusal(name, args, message):
    """A run that must refuse: exit 64, one `nsys-table.py: …` line on stderr holding message, no stdout."""
    rc, out, err = run(args)
    lines = err.splitlines()
    assert rc == 64 and len(lines) == 1 and lines[0].startswith("nsys-table.py: ") and message in lines[0] \
        and out == "", f"{name}: rc={rc} out={out!r} err={err!r}"


def self_test():
    n = 0
    with tempfile.TemporaryDirectory() as d:
        fx = os.path.join(d, "trace.sqlite")
        build_fixture(fx)
        # the --tsv table at --steps 2: template split (with 'void ' and spaces removed), the
        # x*y and x*y*z dimension spellings, per-token division and the ms_per_token order
        check_output("the --tsv table", [fx, "--steps", "2", "--tsv"], [TSV_HEADER] + [
            "mul_mat_vec_q\t(ggml_type)40\t18560\t32*4\t3\t1.50\t64.00\t64.00\t0.096",
            "mul_mat_vec_q\t(ggml_type)40\t4096\t32*4\t2\t1.00\t34.00\t34.00\t0.034",
            "ssm_scan_f32_group\t4,128\t2048*2\t128\t4\t2.00\t14.00\t13.00\t0.028",
            "k_argsort_f32_i32\t\t1\t512\t2\t1.00\t10.00\t10.00\t0.010",
            "ssm_scan_f32_group\t4,128\t2048\t128\t2\t1.00\t7.00\t7.00\t0.007",
            "k_argsort_f32_i32\t\t1*2*3\t256\t2\t1.00\t4.00\t4.00\t0.004",
        ])
        n += 1
        # --kernel is a regex on the demangled name (the template text is part of it), repeatable
        check_output("a --kernel regex", [fx, "--steps", "2", "--tsv", "--kernel", r"mul_mat_vec_q<\(ggml_type\)4[09]>"],
                     [TSV_HEADER] + [
                         "mul_mat_vec_q\t(ggml_type)40\t18560\t32*4\t3\t1.50\t64.00\t64.00\t0.096",
                         "mul_mat_vec_q\t(ggml_type)40\t4096\t32*4\t2\t1.00\t34.00\t34.00\t0.034",
                     ])
        n += 1
        check_output("two --kernel flags", [fx, "--steps", "2", "--tsv", "--kernel", "argsort", "--kernel", "ssm_scan"],
                     [TSV_HEADER] + [
                         "ssm_scan_f32_group\t4,128\t2048*2\t128\t4\t2.00\t14.00\t13.00\t0.028",
                         "k_argsort_f32_i32\t\t1\t512\t2\t1.00\t10.00\t10.00\t0.010",
                         "ssm_scan_f32_group\t4,128\t2048\t128\t2\t1.00\t7.00\t7.00\t0.007",
                         "k_argsort_f32_i32\t\t1*2*3\t256\t2\t1.00\t4.00\t4.00\t0.004",
                     ])
        n += 1
        # --min-per-token keeps a row at the bound and drops every row below it
        check_output("--min-per-token", [fx, "--steps", "2", "--tsv", "--min-per-token", "2"], [TSV_HEADER] + [
            "ssm_scan_f32_group\t4,128\t2048*2\t128\t4\t2.00\t14.00\t13.00\t0.028",
        ])
        n += 1
        # the default table of the same database is unchanged (the header's fixture path is the one
        # line that names the temp directory)
        rc, out, err = run([fx, "--steps", "2"])
        got = out.replace(fx, "<fixture>")
        assert rc == 0 and err == "" and got == DEFAULT_TABLE, \
            f"the default table: rc={rc} err={err!r}\n  got  {got!r}\n  want {DEFAULT_TABLE!r}"
        n += 1
        # the refusals: exit 64, one `nsys-table.py: …` line on stderr
        check_refusal("a missing file", [os.path.join(d, "nothere.sqlite"), "--steps", "2"], "is not a file")
        n += 1
        noku = os.path.join(d, "noku.sqlite")
        build_fixture(noku, with_kernel_table=False)
        check_refusal("no kernel table", [noku, "--steps", "2"], f"has no {KERNEL_TABLE} table")
        n += 1
        check_refusal("--steps 0", [fx, "--steps", "0"], "--steps must be at least 1")
        n += 1
        check_refusal("--min-per-token nan", [fx, "--steps", "2", "--tsv", "--min-per-token", "nan"],
                      "must be a finite number 0 or more")
        n += 1
        check_refusal("a bad --kernel regex", [fx, "--steps", "2", "--tsv", "--kernel", "("], "is not a regex")
        n += 1
        check_refusal("no row left", [fx, "--steps", "2", "--tsv", "--kernel", "nomatch"],
                      "no kernel rows after the filters")
        n += 1
    print(f"nsys-table: self-test ok ({n} cases)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except Refusal as e:
        print(f"nsys-table.py: {e}", file=sys.stderr)
        sys.exit(64)
