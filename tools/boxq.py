#!/usr/bin/env python3
"""boxq.py — the idle-fill list's reader and the guard on the box queue client's exit codes.

  boxq.py fill-list   [--file F] [--justfile J]   print tools/box-fill.tsv's rows, each checked against the justfile
  boxq.py codes-check [--root DIR]                 no other tool or recipe produces exit codes 80, 81, 83, 84, 85
  boxq.py --self-test [CASE …]                     --list-cases names the cases

fill-list: a row names a justfile recipe (or `cmd:<command>`), the card lane it holds, where its predicted minutes come
from (`times:<item>` or `given:<minutes>`) and a reason. A row whose recipe the justfile does not hold, an unknown lane,
a malformed minutes source, a duplicate or a row with no reason is refused by name (rc 65); the recipe's justfile
groups are printed beside it, read from the justfile and kept nowhere else.

codes-check: the box queue's client keeps 80 (stale tree), 81 (lost), 83 (dir owned by another tree), 84 (consumer not
alive) and 85 (a job's dir lock not taken). 82 is retired and never reused. The check refuses any script under tools/
or the justfile that produces one of these codes (`exit N`, `sys.exit(N)`, `return N`, `flock -E N`, `SystemExit(N)`);
a script that only tests a code, or names it in a comment, is no collision. The codes also stay clear of those tools/
and the justfile already name.

Exit: 0 ok; 64 usage; 65 a fill row or a code collision, named; 69 an unreadable file.
"""

import argparse
import contextlib
import io
import os
import re
import sys
import tempfile

USAGE, DATA, ROOT_BAD = 64, 65, 69
OWN_CODES = (80, 81, 83, 84, 85)
# What tools/ and the justfile already name; the collision check holds the queue's codes apart from these.
OTHER_CODES = frozenset(range(64, 76)) | {77, 78, 97, 124, 137}
LANES = ("3090", "a6000", "both", "any", "none")


class Refusal(Exception):
    def __init__(self, code, name, msg):
        super().__init__(msg)
        self.code, self.name, self.msg = code, name, msg


class Parser(argparse.ArgumentParser):
    def error(self, message):
        raise Refusal(USAGE, "usage", message)


def here():
    return os.path.dirname(os.path.abspath(__file__))


def justfile_recipes(path):
    """{recipe: [groups]} from a justfile's text: a recipe is a column-0 name whose first unquoted ':' is not ':='."""
    try:
        with open(path) as f:
            lines = f.read().splitlines()
    except OSError as e:
        raise Refusal(ROOT_BAD, "no-justfile", f"{path}: {e}")
    out, attrs = {}, []
    for line in lines:
        if not line or line[0] in " \t":
            continue
        s = line.rstrip()
        if s.startswith("#"):
            continue
        m = re.match(r"^\[(.*)\]$", s)
        if m:
            attrs += re.findall(r"group\s*[(:]\s*['\"]([^'\"]+)['\"]", m.group(1))
            continue
        quote, colon = None, -1
        for i, ch in enumerate(s):
            if quote:
                quote = None if ch == quote else quote
            elif ch in "'\"`":
                quote = ch
            elif ch == ":":
                colon = i
                break
        name = re.match(r"^@?([A-Za-z_][A-Za-z0-9_-]*)", s)
        if (name and colon > 0 and s[colon:colon + 2] != ":="
                and name.group(1) not in ("set", "alias", "export", "import", "mod", "unexport")):
            out[name.group(1)] = attrs
        attrs = []
    return out


def read_fill(path, jf):
    recipes = justfile_recipes(jf)
    try:
        with open(path) as f:
            text = f.read().splitlines()
    except OSError as e:
        raise Refusal(ROOT_BAD, "no-fill-file", f"{path}: {e}")
    rows, seen = [], set()
    for n, line in enumerate(text, 1):
        if not line.strip() or line.startswith("#"):
            continue
        cols = line.split("\t")
        where = f"{path}:{n}"
        if len(cols) != 4:
            raise Refusal(DATA, "bad-fill-row", f"{where}: {len(cols)} columns, the file has 4 (item, lane, minutes, why)")
        item, lane, src, why = cols
        if item in seen:
            raise Refusal(DATA, "bad-fill-row", f"{where}: {item} appears twice")
        seen.add(item)
        if not item.startswith("cmd:") and item not in recipes:
            raise Refusal(DATA, "bad-fill-row", f"{where}: {item} is not a recipe of {jf}")
        if lane not in LANES:
            raise Refusal(DATA, "bad-fill-row", f"{where}: lane {lane!r} is not one of {', '.join(LANES)}")
        m = re.match(r"^(times:\S+|given:([0-9]+(?:\.[0-9]+)?))$", src)
        if not m:
            raise Refusal(DATA, "bad-fill-row", f"{where}: minutes source {src!r} is not times:<item> or given:<minutes>")
        if item.startswith("cmd:") and not m.group(2):
            raise Refusal(DATA, "bad-fill-row", f"{where}: a cmd: row has no recipe to take times from: use given:<minutes>")
        if not why.strip():
            raise Refusal(DATA, "bad-fill-row", f"{where}: no reason")
        rows.append({"item": item, "lane": lane, "src": src, "why": why,
                     "groups": recipes.get(item, [])})
    return rows


def cmd_fill_list(a):
    for r in read_fill(a.file or os.path.join(here(), "box-fill.tsv"), a.justfile or os.path.join(here(), "..", "justfile")):
        print(f"{r['item']}\tlane={r['lane']}\tgroups={','.join(r['groups']) or '-'}\t{r['src']}")

PRODUCE_RES = [re.compile(p) for p in (
    r"(?<![\w$.-])exit\s+(8[01345])\b", r"\bexit\(\s*(8[01345])\s*\)", r"(?<![\w$-])return\s+(8[01345])\b",
    r"\s-E\s*(8[01345])\b", r"\bSystemExit\(\s*(8[01345])\b", r"\bos\._exit\(\s*(8[01345])\b",
)]


def cmd_codes_check(a):
    base = a.root or os.path.join(here(), "..")
    bad = []
    files = [os.path.join(base, "justfile")]
    for dp, _, fns in os.walk(os.path.join(base, "tools")):
        files += [os.path.join(dp, f) for f in fns if f.endswith((".sh", ".py")) or "." not in f]
    for p in sorted(files):
        if os.path.abspath(p) == os.path.abspath(__file__) or not os.path.isfile(p):
            continue
        try:
            with open(p, errors="replace") as f:
                for n, line in enumerate(f, 1):
                    if line.lstrip().startswith("#"):
                        continue
                    for rx in PRODUCE_RES:
                        for m in rx.finditer(line):
                            bad.append(f"{os.path.relpath(p, base)}:{n}: produces {m.group(1)}")
        except OSError as e:
            raise Refusal(ROOT_BAD, "unreadable", f"{p}: {e}")
    if bad or set(OWN_CODES) & OTHER_CODES or len(set(OWN_CODES)) != len(OWN_CODES):
        raise Refusal(DATA, "exit-code-collision", "\n".join(bad) or "the tool's own codes overlap the reserved set")
    print("boxq codes 80, 81, 83-85: no other tool or recipe produces them")


def parser():
    p = Parser(prog="boxq.py", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd")
    s = sub.add_parser("fill-list")
    s.add_argument("--file")
    s.add_argument("--justfile")
    sub.add_parser("codes-check").add_argument("--root")
    return p


HANDLERS = {"fill-list": cmd_fill_list, "codes-check": cmd_codes_check}


def main(argv):
    try:
        if argv and argv[0] == "--self-test":
            return self_test(argv[1:])
        if argv and argv[0] == "--list-cases":
            print("\n".join(c.__name__[5:].replace("_", "-") for c in CASES))
            return 0
        a = parser().parse_args(argv)
        if not a.cmd:
            raise Refusal(USAGE, "usage", "a subcommand is needed: fill-list or codes-check")
        HANDLERS[a.cmd](a)
        return 0
    except Refusal as r:
        print(f"boxq: {r.name}: {r.msg}", file=sys.stderr)
        return r.code


# ---- self-test ----------------------------------------------------------------------------------------------------

class Fix:
    """One case's world: a temp dir, and a helper that runs a command in this process."""

    def __init__(self, tmp):
        self.tmp = tmp

    def call(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = main(list(argv))
        return rc, out.getvalue(), err.getvalue()


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def case_fill_list(fx):
    jf = os.path.join(fx.tmp, "justfile")
    with open(jf, "w") as f:
        f.write("set shell := ['bash', '-c']\nX := 'a:b'\nalias w := weekly-a\n\n[group('solo')]\n# c\n[group('v41-load')]\n"
                "weekly-a:\n    echo hi:there\n\nweekly-b *ARGS:\n    true\nname arg='x:y' *R: dep\n    true\nexport Y := 'z'\n")
    check(justfile_recipes(jf) == {"weekly-a": ["solo", "v41-load"], "weekly-b": [], "name": []},
          f"the justfile parse read {justfile_recipes(jf)}")
    ff = os.path.join(fx.tmp, "fill.tsv")

    def run(*rows):
        with open(ff, "w") as f:
            f.write("# header\n" + "\n".join("\t".join(r) for r in rows) + "\n")
        return fx.call("fill-list", "--file", ff, "--justfile", jf)

    good = ("weekly-a", "both", "times:weekly-a", "why")
    rc, out, err = run(good, ("weekly-b", "any", "given:10", "why"))
    check(rc == 0 and "groups=solo,v41-load" in out and out.count("\n") == 2, f"good rows: {rc} {out!r} {err!r}")
    for bad, name in ((("nope", "any", "given:5", "w"), "a recipe the justfile lacks"),
                      (("weekly-a", "moon", "given:5", "w"), "an unknown lane"),
                      (("weekly-a", "any", "five", "w"), "a bad minutes source"),
                      (("cmd:true", "any", "times:x", "w"), "a cmd: row on times"),
                      (("weekly-a", "any", "given:5", ""), "no reason"),
                      (("weekly-a", "any", "given:5"), "three columns")):
        rc, _, err = run(bad)
        check(rc == DATA and "bad-fill-row" in err, f"{name} must be 65 by name, got {rc} {err!r}")
    check(run(good, good)[0] == DATA, "a duplicate row is refused")
    check(run(("weekly-a", "any", "given:45", "w"))[0] == 0, "a row over 30 min parses: the minutes bound nothing")


def case_codes(fx):
    check(not set(OWN_CODES) & OTHER_CODES and len(set(OWN_CODES)) == 5 and OWN_CODES == (80, 81, 83, 84, 85),
          "the codes 80, 81, 83-85 overlap the reserved set or each other")
    d = os.path.join(fx.tmp, "repo")
    os.makedirs(os.path.join(d, "tools"))
    for n, text in (("justfile", "r:\n    exit 64\n"), ("tools/a.sh", "# exit 80 in a comment\nexit 75\n"),
                    ("tools/b.py", "sys.exit(97)\n")):
        with open(os.path.join(d, n), "w") as f:
            f.write(text)
    rc, out, err = fx.call("codes-check", "--root", d)
    check(rc == 0, f"a tree producing 64, 75 and 97 is clean: {rc} {err!r}")
    for bad in ("exit 80", "sys.exit(83)", "return 84", "flock -s -n -E 85 f true", "(exit 81)", "raise SystemExit(80)"):
        with open(os.path.join(d, "tools", "c.sh"), "w") as f:
            f.write(f"x\n{bad}\n")
        rc, out, err = fx.call("codes-check", "--root", d)
        check(rc == DATA and "tools/c.sh:2" in err, f"a tool producing '{bad}' must be 65: {rc} {err!r}")
    with open(os.path.join(d, "tools", "c.sh"), "w") as f:
        f.write("exit 82\n")
    check(fx.call("codes-check", "--root", d)[0] == 0, "82 is no longer the queue's: another tool may produce it")
    with open(os.path.join(d, "tools", "c.sh"), "w") as f:
        f.write('[ "$rc" = 80 ] && case $rc in 83) echo x;; esac\n')
    check(fx.call("codes-check", "--root", d)[0] == 0, "a tool that tests a code, not produces it, is not a collision")

CASES = [case_fill_list, case_codes]


def self_test(names):
    known = {c.__name__[5:].replace("_", "-"): c for c in CASES}
    unknown = [n for n in names if n not in known]
    if unknown:
        raise Refusal(USAGE, "usage", f"no such case: {', '.join(unknown)}")
    failed = []
    for name, fn in known.items():
        if names and name not in names:
            continue
        with tempfile.TemporaryDirectory(prefix="boxq-selftest-") as tmp:
            try:
                fn(Fix(tmp))
                print(f"ok   {name}")
            except Exception as e:  # a case's red line is its message; a traceback only for a non-assertion
                failed.append(name)
                print(f"FAIL {name}: {e if isinstance(e, AssertionError) else repr(e)}")
    if failed:
        print(f"boxq self-test: {len(failed)} red: {', '.join(failed)}")
        return 1
    print(f"boxq self-test: {len(names) or len(known)} cases ok")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
