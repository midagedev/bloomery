#!/usr/bin/env python3
"""The review bot's deterministic diagnostics (.github/workflows/review.yml, job `diagnostics`).

    tools/review/diagnostics.py --base DIR --pr DIR --changed FILE --diff FILE --out FILE --dryer BIN --crapper BIN
    tools/review/diagnostics.py --self-test

Runs two of Uncle Bob's tools on the changed Rust code and writes one Markdown file the reviewer reads:
- dryer (structural duplicate candidates) over the crates that hold a changed .rs file, plus `crates/gpu` when a
  `crates/gpu-*` family crate changed (a per-model copy of shared code is the duplicate this project cares about).
  `crates/gpu-gates` never joins: its gate bins are alike by design (a boilerplate `main` each) and it alone costs
  minutes. The roots stay under MAX_LINES in all (dryer compares every pair, so time grows with the square); over it,
  the largest roots are dropped and named. Kept are
  pairs where one side overlaps lines the diff adds (an untouched boilerplate `main` in a changed file is not this
  PR's duplicate), at most PER_RANGE partners for one range, at most MAX_PAIRS, closest first.
- crapper, complexity only (coverage needs a build this job must not run), on the changed files at the base and at
  the head. Kept are functions whose complexity rose by RISE or more, or that are new or grew and sit above HIGH.
Both tools only parse source; nothing from the PR is executed. A tool that fails or is missing is named in the output
(the reviewer says so in its Notes), never skipped in silence. Exit 0 whenever the output file was written.
"""

import os
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from post import added_lines  # noqa: E402

MAX_PAIRS = 15
PER_RANGE = 2
MAX_LINES = 300_000
SKIP_CRATES = {"gpu-gates"}
MIN_SCORE = 0.85
RISE = 5
HIGH = 15
MAX_FUNCS = 15
TOOL_TIMEOUT = 600


def rust_changed(changed_path, pr):
    files = [l.strip() for l in open(changed_path, encoding="utf-8") if l.strip()]
    return [f for f in files if f.endswith(".rs") and os.path.isfile(os.path.join(pr, f))]


def dryer_roots(changed, size=None):
    """(roots, dropped): the crates to search and the ones left out for SKIP_CRATES or the line budget."""
    crates = sorted({f.split("/")[1] for f in changed if f.startswith("crates/") and f.count("/") >= 2})
    roots = {c for c in crates if c not in SKIP_CRATES}
    dropped = sorted(c for c in crates if c in SKIP_CRATES)
    if any(c.startswith("gpu-") and c not in SKIP_CRATES for c in crates):
        roots.add("gpu")
    if size is not None:
        while roots and sum(size(c) for c in roots) > MAX_LINES:
            big = max(roots, key=size)
            roots.discard(big)
            dropped.append(big)
    return sorted("crates/" + c for c in roots), dropped


def rust_lines(tree):
    def size(crate):
        n = 0
        for root, _, files in os.walk(os.path.join(tree, "crates", crate)):
            for f in files:
                if f.endswith(".rs"):
                    with open(os.path.join(root, f), "rb") as fh:
                        n += fh.read().count(b"\n")
        return n
    return size


def parse_dryer(text, tree):
    """[(score, (path, a, b), (path, a, b))] from dryer's text report, paths relative to tree."""
    pairs, cur = [], None
    prefix = tree.rstrip("/") + "/"
    for line in text.splitlines():
        m = re.match(r"^DUPLICATE score=([0-9.]+)", line)
        if m:
            cur = [float(m.group(1))]
            pairs.append(cur)
            continue
        m = re.match(r"^\s+(\S+):(\d+)-(\d+)\s*$", line)
        if m and cur is not None and len(cur) < 3:
            path = m.group(1)
            path = path[len(prefix):] if path.startswith(prefix) else path
            cur.append((path, int(m.group(2)), int(m.group(3))))
    return [tuple(p) for p in pairs if len(p) == 3]


def keep_pairs(pairs, added):
    """Pairs with a side over added lines ({path: set of lines}), PER_RANGE partners a range, and the count."""
    def touches(side):
        lines = added.get(side[0], ())
        return any(side[1] <= n <= side[2] for n in lines)

    kept, per = [], {}
    for p in sorted(pairs, key=lambda p: -p[0]):
        if p[0] < MIN_SCORE:
            continue
        side = p[1] if touches(p[1]) else p[2] if touches(p[2]) else None
        if side is None:
            continue
        if per.get(side, 0) >= PER_RANGE:
            continue
        per[side] = per.get(side, 0) + 1
        kept.append(p)
    return kept[:MAX_PAIRS], len(kept)


def parse_crap(edn):
    """{(namespace, name): complexity} from crapper's .metrics/crap.edn (the highest when a name repeats)."""
    out = {}
    for m in re.finditer(r'\{:name "([^"]*)", :namespace "([^"]*)", :complexity (\d+)', edn):
        key = (m.group(2), m.group(1))
        out[key] = max(out.get(key, 0), int(m.group(3)))
    return out


def rising(base, head):
    rows = []
    for key, cc in head.items():
        old = base.get(key)
        if old is None:
            if cc > HIGH:
                rows.append((key, None, cc))
        elif cc - old >= RISE or (cc > HIGH and cc > old):
            rows.append((key, old, cc))
    rows.sort(key=lambda r: -(r[2] - (r[1] or 0)))
    return rows[:MAX_FUNCS]


def run(cmd, cwd):
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=TOOL_TIMEOUT)
        return p.returncode, p.stdout, p.stderr
    except (OSError, subprocess.TimeoutExpired) as e:
        return None, "", str(e)


def crap_of(crapper, tree, files, work):
    """{(file, namespace, name): complexity} over the files that exist in tree, one crapper run a file so each row
    names its file (crapper's snapshot carries no path)."""
    out = {}
    for i, f in enumerate(x for x in files if os.path.isfile(os.path.join(tree, x))):
        w = os.path.join(work, str(i))
        os.makedirs(w, exist_ok=True)
        rc, so, err = run([crapper, "--root", w, "-s", tree, "--no-coverage", f], w)
        edn = os.path.join(w, ".metrics", "crap.edn")
        if rc == 0 and "No source files to analyze" in so + err:
            continue  # crapper leaves test files (tests/, *_test.rs) out of its analysis by design
        if rc != 0 or not os.path.isfile(edn):
            return None, f"crapper exited {rc} on {f}: {err.strip()[-300:]}"
        for (ns, name), cc in parse_crap(open(edn, encoding="utf-8").read()).items():
            out[(f, ns, name)] = cc
    return out, None


def report(base, pr, changed_path, diff_path, dryer, crapper):
    changed = rust_changed(changed_path, pr)
    lines = ["# Diagnostics", ""]
    if not changed:
        return "\n".join(lines + ["No Rust file changed; dryer and crapper did not run.", ""])
    with tempfile.TemporaryDirectory() as work:
        roots, left_out = dryer_roots(changed, rust_lines(pr))
        lines += [f"## dryer: duplicate candidates (score ≥ {MIN_SCORE}, one side over lines this diff adds)", ""]
        if roots:
            args = [dryer, "--root", os.path.join(work, "dry")]
            for r in roots:
                args += ["-s", os.path.join(pr, r)]
            os.makedirs(os.path.join(work, "dry"), exist_ok=True)
            rc, out, err = run(args, work)
            if rc != 0:
                lines.append(f"dryer did not run: exit {rc}: {err.strip()[-300:]}")
            else:
                added = added_lines(open(diff_path, encoding='utf-8', errors='replace').read())
                kept, total = keep_pairs(parse_dryer(out, pr), added)
                lines.append(f"Searched: {', '.join(roots)}. {total} pair(s) qualify; the closest {len(kept)} follow.")
                if left_out:
                    lines.append(f"Not searched: {', '.join('crates/' + c for c in left_out)} (alike by design, or over "
                                 f"the {MAX_LINES:,}-line budget). Grep those by hand when the diff touches them.")
                for score, a, b in kept:
                    lines.append(f"- {score:.2f} `{a[0]}:{a[1]}-{a[2]}` ~ `{b[0]}:{b[1]}-{b[2]}`")
        else:
            lines.append("No changed .rs file is under crates/; dryer did not run.")
        lines += ["", f"## crapper: complexity of changed functions (rise ≥ {RISE}, or above {HIGH} and growing)", ""]
        head, e1 = crap_of(crapper, pr, changed, os.path.join(work, "crap-head"))
        basec, e2 = crap_of(crapper, base, changed, os.path.join(work, "crap-base"))
        if e1 or e2:
            lines.append("crapper did not run: " + "; ".join(e for e in (e1, e2) if e))
        else:
            rows = rising(basec, head)
            if not rows:
                lines.append("No changed function crosses the lines.")
            for (f, ns, name), old, cc in rows:
                lines.append(f"- `{ns}::{name}` in `{f}`: {'new' if old is None else old} → {cc}")
    return "\n".join(lines + [""])


def self_test():
    assert dryer_roots(["crates/gpu-glm5next/src/a.rs", "crates/model/src/b.rs", "tools/x.rs"]) == (
        ["crates/gpu", "crates/gpu-glm5next", "crates/model"], [])
    assert dryer_roots(["crates/gpu-gates/src/bin/g.rs"]) == ([], ["gpu-gates"])
    sizes = {"gpu": 200_000, "model": 80_000, "gpu-mimo2": 40_000}
    assert dryer_roots(["crates/gpu-mimo2/src/a.rs", "crates/model/src/b.rs"], sizes.get) == (
        ["crates/gpu-mimo2", "crates/model"], ["gpu"])

    text = (
        "DUPLICATE score=1.00\n  /t/crates/p/src/a.rs:153-158\n  /t/crates/p/src/b.rs:59-64\n\n"
        "DUPLICATE score=0.83\n  /t/crates/p/src/a.rs:1-9\n  /t/crates/p/src/c.rs:1-9\n\n"
        "DUPLICATE score=0.90\n  /t/crates/p/src/c.rs:5-9\n  /t/crates/p/src/d.rs:5-9\n"
    )
    pairs = parse_dryer(text, "/t")
    assert pairs[0] == (1.0, ("crates/p/src/a.rs", 153, 158), ("crates/p/src/b.rs", 59, 64)), pairs
    kept, total = keep_pairs(pairs, {"crates/p/src/a.rs": {155}})
    assert total == 1 and kept[0][0] == 1.0, kept  # 0.83 is under the line; c~d touches no added line
    assert keep_pairs(pairs, {"crates/p/src/a.rs": {170}}) == ([], 0)  # a changed file, but not those lines
    many = [(1.0, ("x.rs", 1, 9), (f"y{i}.rs", 1, 9)) for i in range(5)]
    assert keep_pairs(many, {"x.rs": {3}})[1] == PER_RANGE

    edn = ('{:entries [\n  {:name "f", :namespace "m::a", :complexity 32, :coverage nil, :crap nil}\n'
           '  {:name "g", :namespace "m::a", :complexity 4, :coverage nil, :crap nil}\n'
           '  {:name "h", :namespace "m::a", :complexity 18, :coverage nil, :crap nil}\n'
           '  {:name "k", :namespace "m::a", :complexity 16, :coverage nil, :crap nil}]}\n')
    head = parse_crap(edn)
    assert head[("m::a", "f")] == 32 and len(head) == 4
    base = {("m::a", "f"): 20, ("m::a", "g"): 3, ("m::a", "h"): 18}
    got = rising(base, head)
    # f rose by 12; k is new above HIGH; h stayed at 18 (not growing); g rose by 1.
    assert [(k[1], o, c) for k, o, c in got] == [("k", None, 16), ("f", 20, 32)], got

    with tempfile.TemporaryDirectory() as d:
        pr = os.path.join(d, "pr")
        os.makedirs(os.path.join(pr, "docs"))
        open(os.path.join(pr, "docs", "a.md"), "w").write("x")
        ch = os.path.join(d, "changed.txt")
        open(ch, "w").write("docs/a.md\n")
        dp = os.path.join(d, "diff.patch")
        open(dp, "w").write("")
        assert "No Rust file changed" in report(d, pr, ch, dp, "dryer", "crapper")
        os.makedirs(os.path.join(pr, "crates", "p", "src"))
        open(os.path.join(pr, "crates", "p", "src", "a.rs"), "w").write("fn f() {}\n")
        open(ch, "w").write("crates/p/src/a.rs\n")
        out = report(d, pr, ch, dp, "/nonexistent/dryer", "/nonexistent/crapper")
        assert "dryer did not run" in out and "crapper did not run" in out, out
        # A file crapper does not analyze (a test file) is skipped, not a failed run.
        fake = os.path.join(d, "crapper-skips")
        open(fake, "w").write("#!/bin/sh\necho 'No source files to analyze.'\n")
        os.chmod(fake, 0o755)
        out = report(d, pr, ch, dp, "/nonexistent/dryer", fake)
        assert "crapper did not run" not in out and "No changed function" in out, out
    print("diagnostics.py self-test: ok")


def main(argv):
    if argv[1:] == ["--self-test"]:
        self_test()
        return 0
    opts = dict(zip(argv[1::2], argv[2::2]))
    need = ["--base", "--pr", "--changed", "--diff", "--out", "--dryer", "--crapper"]
    missing = [k for k in need if k not in opts]
    if missing or len(argv) % 2 == 0:
        print(__doc__, file=sys.stderr)
        return 64
    text = report(opts["--base"], opts["--pr"], opts["--changed"], opts["--diff"], opts["--dryer"], opts["--crapper"])
    with open(opts["--out"], "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
