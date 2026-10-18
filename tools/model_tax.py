#!/usr/bin/env python3
"""The model-name ratchet (Mac, text only, no build): how many code lines in common code name a model.

    tools/model_tax.py check  [--base REV] [--root DIR]
    tools/model_tax.py rows   [--family F] [--root DIR]
    tools/model_tax.py lower  [--root DIR]
    tools/model_tax.py --self-test

A model addition pays a tax in common code: an `Arch::Foo =>` arm, a `Program::FooBody` key, a seat word, a shape
constant. The next addition repeats it. This check counts those lines and holds the count down, two-sided, like
tools/spawn-allow.txt: a model name that enters common code must be argued for, and a lift that removes names must
lower the entry in the same change.

What is counted, per (file, family):

  file   crates/*/src/**/*.rs that is outside every model's own paths and outside the registry files.
  line   a code line that matches the family's token (tools/model-tokens.tsv), read after the Rust lexer
         (tools/recipes.py rust_lex) has blanked every comment: a code line that starts with `*` counts, a comment
         after code does not. Skipped: blank lines, string continuation lines (a doc string), every line from the
         first `#[cfg(test)]` of a file on, and every line of a tests file (`tests.rs`, `*_tests.rs`, a `tests/`
         directory).
  own    a path that matches any family's token, or starts with a family's extra own path. A model's own code may
         name its model. Another model's name inside it is rule 1 of tools/check-arch.sh.
  registry  a file in tools/model-registries.txt: a table keyed by a name (an arch string, a family, a projector, a
         pre-tokenizer). A new model adds its rows there. Its lines are printed, not ratcheted.

The tables:

  tools/model-tokens.tsv     family <TAB> token (Python regex, case-insensitive) <TAB> extra own paths (comma
                             separated prefixes, `-` none) <TAB> since (`baseline` or the date the family joined).
                             A family `-` is not a model: its token names a crate (`gpu-<x>`) that looks like a
                             model directory and is common code.
  tools/model-registries.txt one file a line.
  tools/model-tax.tsv        path <TAB> family <TAB> count <TAB> why. `why` is `baseline <fate> <count at the
                             baseline>: <where it goes>` for a family that joined at the baseline (the count may fall
                             below that number and never rise above it), or `PIN(YYYY-MM-DD) <family>: <the missing
                             description field or program>`: the fact the model needs that ModelSpec or the model's
                             own code does not yet carry. A family that joined after the baseline has PIN rows only.

`check` fails (exit 1) on:

  - a (file, family) that holds more lines than its row ("the row allows"), fewer ("lower the entry"), or has no row;
  - a `why` that is not one of the two forms, a PIN row without a date, with another family or without a field;
  - a `baseline` row of a family that joined after the baseline, or above its baseline count;
  - a model directory (`crates/*/src/arch/<d>/`, `crates/gpu-<x>/`, `tools/ref/models/<m>.sh`) that no tokens row
    names ("unknown model dir"): a new model starts with its tokens row.
With `--base REV` it also holds the tables to what REV had, so a count cannot be raised by editing the table (the
tables are the argument, and an edit of them is what the check must not take on trust):
  - a row REV lacks must be a PIN, a baseline row never appears after REV, a PIN row REV had goes back to no baseline;
  - a baseline row keeps the baseline count it had at REV and does not rise above its count at REV;
  - a PIN row that rises needs a PIN of another date or field than REV's row (whitespace is no difference);
  - a family REV lacks joins with its date, and a family's `since` is not changed.
A REV whose tree lacks tools/model-tax.tsv has nothing to compare (the table starts there). It also lists the registry
files the tables gained since REV, because a file moved into the registries leaves the ratchet.
A table that does not parse, a registry file that is not there, a source file it cannot read, a REV that is not a
commit and a table REV's tree lists but git cannot read give no verdict (exit 2). It also prints, and never fails on: the registries' lines, per family the files outside its own paths and the
registries that name it (the touch count), the gate bins per family, and the 8-line windows copied between families'
own files (identifiers that name a model collapse to one word).

`rows` prints a row stub for every (file, family) that has no row, to be filled with a PIN why; `lower` rewrites
model-tax.tsv with every count that fell, and drops a row that reached 0; it never raises a count or adds a row.

Exit: 0 clean; 1 a violation; 2 no verdict (an argument, a table or a file this tool cannot read).
"""
import collections
import datetime
import hashlib
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from recipes import rust_lex  # noqa: E402  (the one Rust lexer: comments blanked, strings kept)

TOKENS, REGISTRIES, TAX = "tools/model-tokens.tsv", "tools/model-registries.txt", "tools/model-tax.tsv"
FATES = "APDSOTKR"
WINDOW = 8
TESTS_FILE = re.compile(r"(^|/)tests?\.rs$|_tests\.rs$|/tests/")
DISPATCH = re.compile(r"\b(Arch|Program|Family|Model|Variant|Seat|Projector)::[A-Z]\w*")
PIN = re.compile(r"^PIN\(([^)]*)\)\s*(.*)$")
BASELINE = re.compile(r"^baseline ([A-Z]) (\d+): (\S.*)$")
STUB = "PIN(YYYY-MM-DD) {family}: <missing field>"


class NoVerdict(Exception):
    """An input this tool cannot read; the message names it."""


class Family:
    def __init__(self, name, token, own, since):
        self.name, self.token, self.own, self.since = name, token, own, since
        self.regex = re.compile(token, re.I)

    def owns(self, path):
        return bool(self.regex.search(path)) or any(path.startswith(p) for p in self.own)


def read(root, rel):
    try:
        with open(os.path.join(root, rel), encoding="utf-8") as f:
            return f.read()
    except OSError as e:
        raise NoVerdict(f"{rel}: cannot read ({e.strerror})")
    except UnicodeDecodeError:
        raise NoVerdict(f"{rel}: cannot read (not UTF-8)")


def parse_table(text, rel, cols):
    """The rows of a tab-separated table: (line number, columns). A comment or blank line is skipped."""
    rows = []
    for n, line in enumerate(text.split("\n"), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        c = line.split("\t")
        if len(c) != cols:
            raise NoVerdict(f"{rel}:{n}: {len(c)} columns, want {cols}")
        rows.append((n, c))
    return rows


def table(root, rel, cols):
    return parse_table(read(root, rel), rel, cols)


def load_tokens(root):
    families, commons = [], []
    for n, (name, token, own, since) in table(root, TOKENS, 4):
        where = f"{TOKENS}:{n}"
        try:
            re.compile(token, re.I)
        except re.error as e:
            raise NoVerdict(f"{where}: the token {token!r} is not a regex ({e})")
        if name == "-":
            commons.append(re.compile(f"^(?:{token})$", re.I))
            continue
        if not re.fullmatch(r"[a-z0-9_]+", name):
            raise NoVerdict(f"{where}: the family {name!r} is not lower-case letters, digits and underscores")
        if any(f.name == name for f in families):
            raise NoVerdict(f"{where}: the family {name} is named twice")
        if since != "baseline":
            try:
                datetime.date.fromisoformat(since)
            except ValueError:
                raise NoVerdict(f"{where}: since {since!r} is neither `baseline` nor a date YYYY-MM-DD")
        paths = [] if own == "-" else own.split(",")
        for p in paths:
            if not p.endswith("/") or p.startswith("/"):
                raise NoVerdict(f"{where}: the own path {p!r} is not a relative directory prefix ending in /")
        families.append(Family(name, token, paths, since))
    if not families:
        raise NoVerdict(f"{TOKENS}: no family")
    return families, commons


def load_registries(root):
    return load_registries_text(read(root, REGISTRIES))


def source_files(root):
    """crates/*/src/**/*.rs, sorted, relative with forward slashes."""
    crates = os.path.join(root, "crates")
    try:
        names = sorted(os.listdir(crates))
    except OSError as e:
        raise NoVerdict(f"crates: cannot list ({e.strerror})")
    out = []
    for name in names:
        src = os.path.join(crates, name, "src")
        if not os.path.isdir(src):
            continue
        for dp, dns, fns in os.walk(src, onerror=_walk_error):
            dns.sort()
            out += [os.path.relpath(os.path.join(dp, f), root).replace(os.sep, "/") for f in sorted(fns) if f.endswith(".rs")]
    return out


def _walk_error(e):
    raise NoVerdict(f"{e.filename}: cannot list ({e.strerror})")


def classify(text, rel, union, quoted):
    """The code lines of a file that match a token: (line number, class, stripped text). The lexer blanks every comment
    (line, block, doc), so a line that starts with `*` is code. The class is test, doc, dispatch, literal or ident; only
    the last three are code."""
    out = []
    test_from = 0 if TESTS_FILE.search(rel) else None
    for i, line in enumerate(rust_lex(text)[0].split("\n"), 1):
        s = line.strip()
        if test_from is None and s.startswith("#[cfg(test)]"):
            test_from = i
        if not s or not union.search(s):
            continue
        if test_from is not None and i >= test_from:
            cls = "test"
        elif s.startswith("doc:") or s.endswith("\\") or (
                re.match(r"^[A-Za-z`(].*\",?$", s) and "=" not in s and "::" not in s):
            cls = "doc"
        elif DISPATCH.search(s) and union.search(re.sub(r'"[^"]*"', "", s)):
            cls = "dispatch"
        elif quoted.search(s):
            cls = "literal"
        else:
            cls = "ident"
        out.append((i, cls, s))
    return out


class Scan:
    """One walk of the tree: every code line that names a family, by file."""

    def __init__(self, root):
        self.root = root
        self.families, self.commons = load_tokens(root)
        self.registries = load_registries(root)
        for r in self.registries:
            if not os.path.isfile(os.path.join(root, r)):
                raise NoVerdict(f"{REGISTRIES}: {r} is not a file")
        union = "|".join(f"(?:{f.token})" for f in self.families)
        self.union = re.compile(union, re.I)
        quoted = re.compile(r'"[^"]*(?:' + union + r')[^"]*"', re.I)
        self.files = source_files(root)
        self.own = {}                                  # file -> True when any family owns the path
        self.hits = {}                                 # file -> [(line, class, family names)] of code lines
        for rel in self.files:
            self.own[rel] = any(f.owns(rel) for f in self.families)
            text = read(root, rel)
            if not self.union.search(text):
                continue
            found = []
            for i, cls, s in classify(text, rel, self.union, quoted):
                if cls in ("test", "doc"):
                    continue
                found.append((i, cls, [f.name for f in self.families if f.regex.search(s)]))
            if found:
                self.hits[rel] = found

    def common(self):
        """The ratcheted lines: (file, line, class, families) outside every own path and the registries."""
        return [(rel, i, c, fs) for rel, found in self.hits.items()
                if not self.own[rel] and rel not in self.registries for i, c, fs in found]

    def counts(self):
        n = collections.Counter()
        for rel, _, _, fs in self.common():
            for f in fs:
                n[(rel, f)] += 1
        return n

    def registry_lines(self):
        return [(rel, i) for rel, found in self.hits.items() if rel in self.registries for i, _, _ in found]

    def touch(self):
        """Per family: the files outside its own paths and the registries that name it."""
        fam = {f.name: f for f in self.families}
        out = collections.defaultdict(set)
        for rel, found in self.hits.items():
            if rel in self.registries:
                continue
            for _, _, fs in found:
                for f in fs:
                    if not fam[f].owns(rel):
                        out[f].add(rel)
        return out


def model_dirs(root):
    """The directories and files that introduce a model: (kind, name to match, path)."""
    out = []
    crates = os.path.join(root, "crates")
    try:
        names = sorted(os.listdir(crates))
        for name in names:
            if name.startswith("gpu-") and os.path.isdir(os.path.join(crates, name)):
                out.append(("crate", name[4:], f"crates/{name}"))
            arch = os.path.join(crates, name, "src", "arch")
            if os.path.isdir(arch):
                for d in sorted(os.listdir(arch)):
                    if os.path.isdir(os.path.join(arch, d)):
                        out.append(("arch dir", d, f"crates/{name}/src/arch/{d}"))
        prof = os.path.join(root, "tools", "ref", "models")
        if os.path.isdir(prof):
            for f in sorted(os.listdir(prof)):
                if f.endswith(".sh"):
                    out.append(("profile", f[:-3], f"tools/ref/models/{f}"))
    except OSError as e:
        raise NoVerdict(f"{e.filename}: cannot list ({e.strerror})")
    return out


def tax_rows(root):
    return parse_tax(read(root, TAX), TAX)


def parse_tax(text, rel):
    rows, seen = [], {}
    for n, (path, family, count, why) in parse_table(text, rel, 4):
        where = f"{rel}:{n}"
        if not count.isdigit() or int(count) < 1:
            raise NoVerdict(f"{where}: the count {count!r} is not a positive integer")
        if (path, family) in seen:
            raise NoVerdict(f"{where}: ({path}, {family}) is a row of line {seen[(path, family)]} already")
        seen[(path, family)] = n
        rows.append((n, path, family, int(count), why))
    return rows


def why_error(family, since, count, why):
    """None when the why is well formed for the family and the count; else what is wrong."""
    m = PIN.match(why)
    if m:
        date, rest = m.groups()
        try:
            datetime.date.fromisoformat(date)
        except ValueError:
            return f"the PIN needs a date, PIN(YYYY-MM-DD) {family}: <the missing description field or program>"
        head, sep, field = rest.partition(":")
        if head.strip() != family:
            return f"the PIN names {head.strip() or 'no family'}, the row is {family}"
        if not sep or not field.strip() or field.strip().startswith("<"):
            return f"the PIN names no missing field, PIN({date}) {family}: <the missing description field or program>"
        return None
    m = BASELINE.match(why)
    if m:
        if m.group(1) not in FATES:
            return f"the baseline fate {m.group(1)} is not one of {' '.join(FATES)}"
        if since != "baseline":
            return (f"{family} joined on {since}: a baseline row is only for a family that joined at the baseline, "
                    f"use PIN(YYYY-MM-DD) {family}: <the missing field>")
        if count > int(m.group(2)):
            return (f"the row {count} is above its baseline {m.group(2)}: a count above the baseline is argued with "
                    f"PIN(YYYY-MM-DD) {family}: <the missing description field or program>")
        return None
    return (f"the why {why!r} is neither `baseline <fate> <count at the baseline>: <where>` nor "
            f"PIN(YYYY-MM-DD) {family}: <the missing field>")


def git(root, *args):
    try:
        env = dict(os.environ, GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_SYSTEM=os.devnull, LC_ALL="C")
        return subprocess.run(["git", "-C", root, *args], capture_output=True, text=True, env=env)
    except OSError as e:
        raise NoVerdict(f"git: cannot run ({e.strerror})")


def base_files(root, rev):
    """The commit REV names and the text of the three tables at it. None is a table the commit's tree does not list; a
    table it lists and git cannot read is no verdict."""
    if rev.startswith("-"):
        raise NoVerdict(f"--base {rev}: not a revision")
    p = git(root, "rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}")
    if p.returncode != 0:
        raise NoVerdict(f"--base {rev}: not a commit of {root} (git rev-parse exit {p.returncode}: "
                        f"{p.stderr.strip() or 'no such revision'})")
    sha = p.stdout.strip()
    texts = {}
    for rel in (TAX, TOKENS, REGISTRIES):
        q = git(root, "ls-tree", "--name-only", sha, "--", rel)
        if q.returncode != 0:
            raise NoVerdict(f"--base {rev}: cannot list {rel} at {sha[:9]} ({q.stderr.strip()})")
        if not q.stdout.strip():
            texts[rel] = None
            continue
        q = git(root, "show", f"{sha}:{rel}")
        if q.returncode != 0:
            raise NoVerdict(f"--base {rev}: cannot read {rel} at {sha[:9]} ({q.stderr.strip()})")
        texts[rel] = q.stdout
    return sha, texts


def base_violations(rows, since, old_rows, old_since, rev):
    """What the tables gained or raised since REV that no PIN argues. A row REV lacks is a PIN. A baseline row is the
    cut's: the same baseline count it had, never higher than it was. A PIN row that rises needs a PIN that differs from the
    one REV had. A family REV lacks has the date it joined."""
    old = {(p, f): (c, w) for _, p, f, c, w in old_rows}
    bad = []
    for n, path, family, count, why in rows:
        now = BASELINE.match(why)
        was = old.get((path, family))
        where = f"{TAX}:{n}: {path}: {family} row"
        if was is None:
            if now:
                bad.append(f"{where} is new since {rev} and a baseline row: a row that {rev} lacks is a "
                           f"PIN(YYYY-MM-DD) {family}: <the missing description field or program>")
            continue
        was_count, was_why = was
        before = BASELINE.match(was_why)
        if now:
            if not before:
                bad.append(f"{where} was a PIN at {rev} and is a baseline row: a row does not go back to the baseline")
            elif now.group(2) != before.group(2):
                bad.append(f"{where}: the baseline count changed from {before.group(2)} at {rev} to {now.group(2)}")
            elif count > was_count:
                bad.append(f"{where} is raised from {was_count} at {rev} to {count}: a count above {rev}'s is argued with "
                           f"PIN(YYYY-MM-DD) {family}: <the missing description field or program>")
        elif count > was_count and re.sub(r"\s+", "", why) == re.sub(r"\s+", "", was_why):
            bad.append(f"{where} is raised from {was_count} at {rev} to {count} under the same PIN: argue the rise with a "
                       "PIN of its own date or field")
    for name, joined in sorted(since.items()):
        if name in old_since:
            if joined != old_since[name]:
                bad.append(f"{TOKENS}: {name} joined {old_since[name]} at {rev}, the row now says {joined}")
        elif joined == "baseline":
            bad.append(f"{TOKENS}: {name} is a new family since {rev} and joined at `baseline`: a family that {rev} lacks "
                       "has the date it joined")
    return bad


def tokens_since(text, rel):
    return {c[0]: c[3] for _, c in parse_table(text, rel, 4) if c[0] != "-"}


def against_base(root, rev, rows, families, out):
    """The violations against REV, and the lines that say what changed. A REV without the table has nothing to compare."""
    sha, texts = base_files(root, rev)
    if texts[TAX] is None:
        print(f"check-model-tax: base {rev} ({sha[:9]}) has no {TAX}: the table starts here, nothing to compare", file=out)
        return []
    if texts[TOKENS] is None:
        raise NoVerdict(f"--base {rev}: {rev} has {TAX} but not {TOKENS}")
    old_rows = parse_tax(texts[TAX], f"{rev}:{TAX}")
    since = {f.name: f.since for f in families}
    bad = base_violations(rows, since, old_rows, tokens_since(texts[TOKENS], f"{rev}:{TOKENS}"), rev)
    old = {(p, f): c for _, p, f, c, _ in old_rows}
    now = {(p, f): c for _, p, f, c, _ in rows}
    grown = [r for r in rows if (r[1], r[2]) not in old or r[3] > old[(r[1], r[2])]]
    print(f"check-model-tax: rows against {rev} ({sha[:9]}): {len(old)} there, {len(now)} here, {len(grown)} new or raised, "
          f"{sum(1 for k in old if k not in now)} gone, {sum(1 for k in now if k in old and now[k] < old[k])} lowered", file=out)
    for n, path, family, count, why in grown:
        print(f"check-model-tax:   {path} {family} {count}: {why}", file=out)
    if texts[REGISTRIES] is not None:
        added = sorted(set(load_registries_text(read(root, REGISTRIES))) - set(load_registries_text(texts[REGISTRIES])))
        if added:
            print(f"check-model-tax: registry files added since {rev}, their lines leave the ratchet: {' '.join(added)}", file=out)
    return bad


def load_registries_text(text):
    return [x.strip() for x in text.split("\n") if x.strip() and not x.lstrip().startswith("#")]


def check(root, out, err, base=None):
    """Print the verdict. Returns 0 clean or 1 violation; raises NoVerdict. With BASE, the rows are also held to what that
    revision had."""
    scan = Scan(root)
    fams = {f.name: f for f in scan.families}
    bad = []
    for kind, name, path in model_dirs(root):
        if any(f.regex.search(name) for f in scan.families) or any(c.match(name) for c in scan.commons):
            continue
        bad.append(f"{path}: unknown model dir ({kind} {name}): add a tokens row to {TOKENS}")
    counts = scan.counts()
    rows = tax_rows(root)
    have = {}
    for n, path, family, count, why in rows:
        have[(path, family)] = count
        if family not in fams:
            bad.append(f"{TAX}:{n}: {path}: the family {family} has no tokens row")
            continue
        e = why_error(family, fams[family].since, count, why)
        if e:
            bad.append(f"{TAX}:{n}: {path}: {family} row: {e}")
        held = counts.get((path, family), 0)
        if held > count:
            bad.append(f"{path}: {family} {held} lines, the row allows {count} (a model name in common code: put the fact "
                       "in ModelSpec or the model's code)")
        elif held < count:
            bad.append(f"{path}: {family} row {count}, the file holds {held} (lower the entry)")
    missing = collections.defaultdict(list)
    for (path, family), held in sorted(counts.items()):
        if (path, family) not in have:
            bad.append(f"{path}: {family} has no row (adding one needs PIN(YYYY-MM-DD) {family}: <the missing description "
                       "field or program>)")
            missing[family].append((path, held))
    for family, found in sorted(missing.items()):
        if fams[family].since != "baseline":
            bad.append(f"new model {family}: {sum(h for _, h in found)} common lines outside registries, in {len(found)} files "
                       "without a PIN row")
    common = scan.common()
    reg = scan.registry_lines()
    cls = collections.Counter(c for _, _, c, _ in common)
    cfiles = {r for r, _, _, _ in common}
    rfiles = {r for r, _ in reg}
    print(f"check-model-tax: {len(common)} common lines in {len(cfiles)} files (dispatch {cls['dispatch']}, literal "
          f"{cls['literal']}, ident {cls['ident']}); {len(rows)} rows allow {sum(r[3] for r in rows)} (file, family) lines; "
          f"registries {len(reg)} lines in {len(rfiles)} files, not ratcheted", file=out)
    if base is not None:
        bad += against_base(root, base, rows, scan.families, out)
    report(scan, root, out)
    if bad:
        for b in bad:
            print(f"check-model-tax: {b}", file=err)
        print(f"check-model-tax: failed, {len(bad)} violation(s)", file=err)
        return 1
    print("check-model-tax: ok", file=out)
    return 0


def own_family(scan, rel):
    return next((f.name for f in scan.families if f.owns(rel)), None)


def report(scan, root, out):
    touch = scan.touch()
    print("check-model-tax: touch, files outside the family's own paths and the registries that name it:", file=out)
    print("  " + "  ".join(f"{f.name} {len(touch.get(f.name, ()))}" for f in scan.families), file=out)
    bins = collections.Counter()
    manifest = os.path.join(root, "crates", "gpu-gates", "Cargo.toml")
    if os.path.isfile(manifest):
        for name in re.findall(r'^\[\[bin\]\]\s*\nname\s*=\s*"([^"]+)"', read(root, "crates/gpu-gates/Cargo.toml"), re.M):
            bins[next((f.name for f in scan.families if f.regex.search(name)), "common")] += 1
        print("check-model-tax: gate bins per family: " + "  ".join(f"{k} {v}" for k, v in sorted(bins.items())), file=out)
    else:
        print("check-model-tax: gate bins per family: crates/gpu-gates/Cargo.toml is not there", file=out)
    pairs = copied_windows(scan, root)
    print(f"check-model-tax: {WINDOW}-line windows copied between families' own files, top 10 pairs of {len(pairs)}:", file=out)
    for (a, b), n in pairs.most_common(10):
        print(f"  {n:5d} {a}  ~  {b}", file=out)


def copied_windows(scan, root):
    """File pairs of different families that share normalized WINDOW-line windows."""
    name = re.compile(r"[A-Za-z0-9_]*(?:" + "|".join(f"(?:{f.token})" for f in scan.families) + r")[A-Za-z0-9_]*", re.I)
    skip = ("use ", "pub use ", "#[", "#![", "reason =", "clippy::", "domain =")
    lone = {"{", "}", "};", "},", ")", ");", "),", "]", "];", ")]"}
    where = collections.defaultdict(list)
    files = [r for r in source_files(root) + _tests_files(root) if own_family(scan, r)]
    for rel in sorted(set(files)):
        norm, in_test = [], False
        for raw in rust_lex(read(root, rel))[0].split("\n"):
            s = raw.strip()
            if s.startswith("#[cfg(test)]"):
                in_test = True
            if in_test:
                break
            if not s or s in lone or s.startswith(skip):
                continue
            norm.append(re.sub(r"\s+", " ", name.sub("M", s)))
        fam = own_family(scan, rel)
        for k in range(len(norm) - WINDOW + 1):
            h = hashlib.md5("\n".join(norm[k:k + WINDOW]).encode()).hexdigest()
            where[h].append((rel, fam))
    pairs = collections.Counter()
    for occ in where.values():
        if len({f for _, f in occ}) < 2:
            continue
        for x in occ:
            for y in occ:
                if x[1] < y[1]:
                    pairs[(x[0], y[0])] += 1
    return pairs


def _tests_files(root):
    out = []
    crates = os.path.join(root, "crates")
    for name in sorted(os.listdir(crates)):
        t = os.path.join(crates, name, "tests")
        if os.path.isdir(t):
            for dp, dns, fns in os.walk(t):
                dns.sort()
                out += [os.path.relpath(os.path.join(dp, f), root).replace(os.sep, "/") for f in sorted(fns) if f.endswith(".rs")]
    return out


def rows_cmd(root, family, out):
    scan = Scan(root)
    if family is not None and family not in {f.name for f in scan.families}:
        raise NoVerdict(f"{TOKENS}: no family {family}")
    have = {(p, f) for _, p, f, _, _ in tax_rows(root)}
    for (path, fam), n in sorted(scan.counts().items()):
        if (path, fam) not in have and family in (None, fam):
            print(f"{path}\t{fam}\t{n}\t{STUB.format(family=fam)}", file=out)
    return 0


def lower_cmd(root, out):
    scan = Scan(root)
    counts = scan.counts()
    path = os.path.join(root, TAX)
    text = read(root, TAX)
    keep, changed = [], 0
    for line in text.split("\n"):
        c = line.split("\t")
        if line.strip() and not line.lstrip().startswith("#") and len(c) == 4 and c[2].isdigit():
            held = counts.get((c[0], c[1]), 0)
            if held < int(c[2]):
                changed += 1
                print(f"check-model-tax: {c[0]}: {c[1]} {c[2]} -> {held}", file=out)
                if held == 0:
                    continue
                c[2] = str(held)
                line = "\t".join(c)
        keep.append(line)
    if changed:
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(keep))
    print(f"check-model-tax: lowered {changed} row(s)", file=out)
    return 0


# ---------------------------------------------------------------- self-test

TOKENS_FIXTURE = """# family\ttoken\textra own paths\tsince
qwen\tqwen\t-\tbaseline
glm\tglm\tcrates/glmdir/\tbaseline
-\tgates\t-\t-
"""
FIXTURE = {
    "tools/model-tokens.tsv": TOKENS_FIXTURE,
    "tools/model-registries.txt": "# the tables\ncrates/gpu/src/lib.rs\n",
    "tools/model-tax.tsv": "crates/model/src/arch/mod.rs\tqwen\t2\tbaseline A 2: the registry\n",
    "crates/model/src/arch/mod.rs": "pub fn spec(a: Arch) -> Spec {\n    match a {\n        Arch::Qwen => qwen_spec(),\n"
                                    "        Arch::QwenMoe => qwen_moe_spec(),\n        _ => other(),\n    }\n}\n"
                                    "// qwen in a comment\n",
    "crates/model/src/arch/qwen/mod.rs": "pub fn qwen_spec() {}\n",
    "crates/gpu/src/lib.rs": "pub type QwenModel = Model<Qwen>;\n",
    "crates/gpu/src/other.rs": "pub fn other() {}\n",
    "crates/gpu-gates/Cargo.toml": '[[bin]]\nname = "gate_qwen_e2e"\npath = "src/bin/a.rs"\n[[bin]]\nname = "gate_common"\npath = "src/bin/b.rs"\n',
    "crates/gpu-gates/src/lib.rs": "pub fn gates() {}\n",
    "crates/glmdir/src/lib.rs": "pub fn glm() {}\n",
    "tools/ref/models/qwen.sh": "#!/bin/sh\n",
}


def fixture(root, edits=None):
    files = dict(FIXTURE)
    files.update(edits or {})
    for rel, text in files.items():
        if text is None:
            continue
        path = os.path.join(root, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)


def run(root, base=None):
    out, err = io.StringIO(), io.StringIO()
    try:
        rc = check(root, out, err, base)
    except NoVerdict as e:
        return 2, out.getvalue(), f"check-model-tax: {e}, no verdict\n"
    return rc, out.getvalue(), err.getvalue()


def self_test():
    fails = []
    base = tempfile.mkdtemp(prefix="model-tax-selftest-")

    def case(name, edits, want_rc, want_text=None, where="err", remove=()):
        d = tempfile.mkdtemp(dir=base)
        fixture(d, edits)
        for rel in remove:
            os.remove(os.path.join(d, rel))
        rc, out, err = run(d)
        text = err if where == "err" else out
        if rc != want_rc or (want_text is not None and want_text not in text):
            fails.append(f"case {name}: rc {rc} (want {want_rc}), want {want_text!r} in the {where} text:\n{err}{out}")

    try:
        tax = FIXTURE["tools/model-tax.tsv"]
        arch = FIXTURE["crates/model/src/arch/mod.rs"]
        case("clean fixture passes", {}, 0, "check-model-tax: 2 common lines in 1 files", "out")
        case("a: a new Arch::Qwen arm", {"crates/model/src/arch/mod.rs": arch.replace(
            "        _ => other(),", "        Arch::QwenB => qwen_b(),\n        _ => other(),")}, 1, "qwen 3 lines, the row allows 2")
        case("b: a counted line removed", {"crates/model/src/arch/mod.rs": arch.replace("        Arch::Qwen => qwen_spec(),\n", "")},
             1, "qwen row 2, the file holds 1 (lower the entry)")
        new_family = TOKENS_FIXTURE + "foo\tfoo\t-\t2026-10-18\n"
        foo_line = "pub fn foo_spec() {}\n"
        case("c: a new family's name in common code has no row",
             {"tools/model-tokens.tsv": new_family, "crates/gpu/src/other.rs": foo_line}, 1, "other.rs: foo has no row")
        case("c: the new family is a new model", {"tools/model-tokens.tsv": new_family, "crates/gpu/src/other.rs": foo_line},
             1, "new model foo: 1 common lines outside registries, in 1 files without a PIN row")
        pin = "crates/gpu/src/other.rs\tfoo\t1\t%s\n"
        case("d: a PIN row without a date", {"tools/model-tokens.tsv": new_family, "crates/gpu/src/other.rs": foo_line,
                                           "tools/model-tax.tsv": tax + pin % "PIN() foo: the foo shape"},
             1, "the PIN needs a date")
        case("d: a PIN row without a field", {"tools/model-tokens.tsv": new_family, "crates/gpu/src/other.rs": foo_line,
                                            "tools/model-tax.tsv": tax + pin % "PIN(2026-10-18) foo:"}, 1, "the PIN names no missing field")
        case("d: a PIN row of another family", {"tools/model-tokens.tsv": new_family, "crates/gpu/src/other.rs": foo_line,
                                              "tools/model-tax.tsv": tax + pin % "PIN(2026-10-18) qwen: a field"},
             1, "the PIN names qwen, the row is foo")
        case("d: the stub is refused", {"tools/model-tokens.tsv": new_family, "crates/gpu/src/other.rs": foo_line,
                                        "tools/model-tax.tsv": tax + pin % STUB.format(family="foo")}, 1, "the PIN needs a date")
        case("d: a baseline row of a new family", {"tools/model-tokens.tsv": new_family, "crates/gpu/src/other.rs": foo_line,
                                                  "tools/model-tax.tsv": tax + pin % "baseline A 1: the registry"},
             1, "joined on 2026-10-18: a baseline row is only for a family that joined at the baseline")
        case("d: a dated PIN row passes", {"tools/model-tokens.tsv": new_family, "crates/gpu/src/other.rs": foo_line,
                                           "tools/model-tax.tsv": tax + pin % "PIN(2026-10-18) foo: the foo shape field"}, 0)
        case("e: comment, cfg(test) tail and tests file are not counted", {
            "crates/gpu/src/other.rs": "// qwen here\n/// qwen doc\npub fn other() {}\n#[cfg(test)]\nmod t { fn qwen_t() {} }\n",
            "crates/gpu/src/tests.rs": "fn qwen_case() {}\n", "crates/gpu/src/b_tests.rs": "fn qwen_case() {}\n",
            "crates/gpu/tests/x.rs": "fn qwen_case() {}\n", "crates/gpu/src/tests/x.rs": "fn qwen_case() {}\n"}, 0)
        other = "crates/gpu/src/other.rs"
        for name, text, want in (
            ("a block comment line with no star is not code", "/*\n qwen here\n*/\npub fn other() {}\n", 0),
            ("a starred block comment line is not code", "/*\n * qwen here\n */\npub fn other() {}\n", 0),
            ("a nested block comment is not code", "/* a /* b */\n qwen here\n*/\npub fn other() {}\n", 0),
            ("a block comment after code is not code", "pub fn other() {} /* qwen */\n", 0),
            ("a trailing line comment is not code", "pub fn other() {} // qwen\n", 0),
            ("a double slash inside a string is code", 'pub fn other() { let u = "http://qwen/x"; }\n', 1),
            ("a dereference assignment is code", "pub fn other(a: &mut Arch) {\n    *a = Arch::Qwen;\n}\n", 1),
            ("a continued expression starting with a star is code", "pub fn other() -> u32 {\n    a\n        * qwen_scale\n}\n", 1),
        ):
            case("e: " + name, {other: text}, want, None if want == 0 else "other.rs: qwen has no row")
        case("f: a hit in a family's own path passes", {"crates/model/src/arch/qwen/mod.rs": "fn qwen_spec() { Arch::Qwen; }\n"}, 0)
        case("f: an extra own path passes", {"crates/glmdir/src/lib.rs": "fn x() { qwen_thing(); }\n"}, 0)
        case("g: a gpu-foo crate with no tokens row", {"crates/gpu-foo/src/lib.rs": "pub fn x() {}\n"}, 1,
             "crates/gpu-foo: unknown model dir (crate foo)")
        case("g: an arch dir with no tokens row", {"crates/model/src/arch/bar/mod.rs": "pub fn x() {}\n"}, 1,
             "unknown model dir (arch dir bar)")
        case("g: a profile with no tokens row", {"tools/ref/models/baz.sh": "#!/bin/sh\n"}, 1, "unknown model dir (profile baz)")
        case("g: the common crates pass", {"crates/gpu-gates/src/x.rs": "pub fn x() {}\n"}, 0)
        case("h: a registry row passes and is reported", {"crates/gpu/src/lib.rs": FIXTURE["crates/gpu/src/lib.rs"]
                                                           + "pub type Foo = Model<Qwen>;\npub type Glm = Model<Glm>;\n"}, 0,
             "registries 3 lines in 1 files, not ratcheted", "out")
        case("h: a new family's row in a registry passes", {"tools/model-tokens.tsv": new_family,
                                                           "crates/gpu/src/lib.rs": "pub type F = Model<Foo>;\n"}, 0)
        case("a: a row raised above its baseline", {"tools/model-tax.tsv": tax.replace("\tqwen\t2\tbaseline A 2", "\tqwen\t2\tbaseline A 1"),
                                                   }, 1, "the row 2 is above its baseline 1")
        case("a: a PIN row argues a count above the baseline", {"tools/model-tax.tsv": tax.replace(
            "\tqwen\t2\tbaseline A 2: the registry", "\tqwen\t2\tPIN(2026-10-18) qwen: a spec field")}, 0)
        case("a baseline row with no count", {"tools/model-tax.tsv": tax.replace("baseline A 2:", "baseline A:")}, 1,
             "is neither `baseline <fate> <count at the baseline>: <where>`")
        case("a baseline row with a fate that is none", {"tools/model-tax.tsv": tax.replace("baseline A 2:", "baseline Z 2:")}, 1,
             "the baseline fate Z is not one of")
        case("a row for a file that is gone", {"tools/model-tax.tsv": tax + "crates/gpu/src/gone.rs\tqwen\t1\tbaseline A 1: gone\n"},
             1, "gone.rs: qwen row 1, the file holds 0 (lower the entry)")
        case("a row twice", {"tools/model-tax.tsv": tax + tax}, 2, "is a row of line 1 already")
        case("a row of an unknown family", {"tools/model-tax.tsv": tax + "crates/gpu/src/other.rs\tzzz\t1\tbaseline A 1: x\n"},
             1, "the family zzz has no tokens row")
        case("a registry file that is not there", {"tools/model-registries.txt": "crates/gpu/src/nope.rs\n"}, 2,
             "crates/gpu/src/nope.rs is not a file")
        case("a tokens row that is no regex", {"tools/model-tokens.tsv": TOKENS_FIXTURE + "bad\t(\t-\tbaseline\n"}, 2, "is not a regex")
        case("a table with the wrong columns", {"tools/model-tax.tsv": "crates/a.rs\tqwen\t1\n"}, 2, "3 columns, want 4")
        case("no tax table", {}, 2, "tools/model-tax.tsv: cannot read", remove=("tools/model-tax.tsv",))
        # --base: the tables held to what a revision had
        def rows_of(*lines):
            return parse_tax("\n".join(lines) + "\n", "t")

        def vs(now, was, since=None, was_since=None):
            return "\n".join(base_violations(rows_of(*now), since or {"qwen": "baseline"}, rows_of(*was),
                                              was_since or {"qwen": "baseline"}, "REV"))

        row = "a.rs\tqwen\t%d\t%s"
        cut = "baseline A 4: r"
        for name, now, was, want in (
            ("a new baseline row", [row % (1, "baseline A 1: r")], [], "is new since REV and a baseline row"),
            ("a new PIN row", [row % (1, "PIN(2026-10-18) qwen: a field")], [], None),
            ("a baseline row raised", [row % (3, cut)], [row % (2, cut)], "raised from 2 at REV to 3"),
            ("a baseline ceiling raised", [row % (2, "baseline A 5: r")], [row % (2, cut)], "baseline count changed from 4 at REV to 5"),
            ("a baseline row lowered", [row % (1, cut)], [row % (2, cut)], None),
            ("a baseline row gone", [], [row % (2, cut)], None),
            ("a PIN raised under the same PIN", [row % (3, "PIN(2026-10-18) qwen: a field")],
             [row % (2, "PIN(2026-10-18) qwen: a field")], "under the same PIN"),
            ("a PIN raised with a space added to its why", [row % (3, "PIN(2026-10-18) qwen: a field ")],
             [row % (2, "PIN(2026-10-18) qwen: a field")], "under the same PIN"),
            ("a PIN raised with its spaces moved", [row % (3, "PIN(2026-10-18)  qwen:  a  field")],
             [row % (2, "PIN(2026-10-18) qwen: a field")], "under the same PIN"),
            ("a PIN raised under a new date", [row % (3, "PIN(2026-10-19) qwen: a field")],
             [row % (2, "PIN(2026-10-18) qwen: a field")], None),
            ("a baseline row argued up by a PIN", [row % (6, "PIN(2026-10-18) qwen: a field")], [row % (2, cut)], None),
            ("a PIN back to the baseline", [row % (1, cut)], [row % (2, "PIN(2026-10-18) qwen: a field")], "does not go back"),
        ):
            got = vs(now, was)
            if (want is None and got) or (want is not None and want not in got):
                fails.append(f"case base: {name}: want {want!r}, got {got!r}")
        for name, since, was_since, want in (
            ("a family's date changed", {"qwen": "baseline"}, {"qwen": "2026-10-18"}, "joined 2026-10-18 at REV"),
            ("a new family at the baseline", {"qwen": "baseline", "foo": "baseline"}, {"qwen": "baseline"}, "foo is a new family"),
            ("a new family with its date", {"qwen": "baseline", "foo": "2026-10-18"}, {"qwen": "baseline"}, None),
        ):
            got = vs([], [], since, was_since)
            if (want is None and got) or (want is not None and want not in got):
                fails.append(f"case base: {name}: want {want!r}, got {got!r}")
        # the same, against a revision of a repository
        def commit(d, msg):
            for a in (("init", "-q"), ("add", "-A"), ("-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false",
                                                     "-c", "gc.auto=0", "-c", "maintenance.auto=false", "commit", "-q", "-m", msg)):
                r = git(d, *a)
                if r.returncode != 0:
                    fails.append(f"case base: git {' '.join(a)}: {r.stderr}")

        g = tempfile.mkdtemp(dir=base)
        fixture(g)
        commit(g, "table")
        rc, out, err = run(g, "HEAD")
        if rc != 0 or "rows against HEAD" not in out or "0 new or raised" not in out:
            fails.append(f"case base: an unchanged table against its own commit: {rc} {out}{err}")
        fixture(g, {"crates/gpu/src/other.rs": "pub fn qwen_x() {}\n",
                    "tools/model-tax.tsv": FIXTURE["tools/model-tax.tsv"] + "crates/gpu/src/other.rs\tqwen\t1\tbaseline A 1: forged\n"})
        rc, out, err = run(g)
        if rc != 0:
            fails.append(f"case base: the forged row passes without a base: {rc} {err}")
        rc, out, err = run(g, "HEAD")
        if rc != 1 or "other.rs: qwen row is new since HEAD and a baseline row" not in err:
            fails.append(f"case base: a forged baseline row is refused against HEAD: {rc} {err}")
        fixture(g, {"crates/gpu/src/other.rs": "pub fn qwen_x() {}\n",
                    "tools/model-tax.tsv": FIXTURE["tools/model-tax.tsv"] + "crates/gpu/src/other.rs\tqwen\t1\tPIN(2026-10-18) qwen: a spec field\n"})
        rc, out, err = run(g, "HEAD")
        if rc != 0 or "1 new or raised" not in out or "other.rs qwen 1: PIN(2026-10-18)" not in out:
            fails.append(f"case base: a new PIN row passes against HEAD and is listed: {rc} {out}{err}")
        fixture(g, {"tools/model-registries.txt": FIXTURE["tools/model-registries.txt"] + "crates/gpu/src/other.rs\n"})
        rc, out, err = run(g, "HEAD")
        if "registry files added since HEAD, their lines leave the ratchet: crates/gpu/src/other.rs" not in out:
            fails.append(f"case base: a file added to the registries is listed: {rc} {out}{err}")
        fixture(g)
        rc, out, err = run(g, "nosuchrev")
        if rc != 2 or "not a commit" not in err:
            fails.append(f"case base: a revision that is not a commit is no verdict: {rc} {err}")
        rc, out, err = run(g, "--output=x")
        if rc != 2 or "not a revision" not in err:
            fails.append(f"case base: an option for a revision is refused: {rc} {err}")
        g2 = tempfile.mkdtemp(dir=base)
        fixture(g2, {"tools/model-tax.tsv": None, "tools/model-registries.txt": None, "tools/model-tokens.tsv": None})
        commit(g2, "no table")
        for rel in (TAX, TOKENS, REGISTRIES):
            os.makedirs(os.path.join(g2, "tools"), exist_ok=True)
            with open(os.path.join(g2, rel), "w") as f:
                f.write(FIXTURE[rel])
        rc, out, err = run(g2, "HEAD")
        if rc != 0 or "has no tools/model-tax.tsv: the table starts here" not in out:
            fails.append(f"case base: a revision with no table has nothing to compare: {rc} {out}{err}")
        # a table the base lists and git cannot read is no verdict, never "the table starts here"
        for rel in (TAX, REGISTRIES):
            g3 = tempfile.mkdtemp(dir=base)
            fixture(g3)
            commit(g3, "table")
            blob = git(g3, "rev-parse", f"HEAD:{rel}").stdout.strip()
            os.remove(os.path.join(g3, ".git", "objects", blob[:2], blob[2:]))
            rc, out, err = run(g3, "HEAD")
            if rc != 2 or f"cannot read {rel}" not in err or "starts here" in out:
                fails.append(f"case base: the blob of {rel} gone from the base is no verdict: {rc} {out}{err}")
        # lower: counts fall, never rise, a row that reached 0 goes, and a second pass changes nothing
        d = tempfile.mkdtemp(dir=base)
        fixture(d, {"tools/model-tax.tsv": "# kept comment\ncrates/model/src/arch/mod.rs\tqwen\t5\tbaseline A 5: r\n"
                                           "crates/gpu/src/gone.rs\tqwen\t1\tbaseline A 1: gone\n"
                                           "crates/gpu/src/more.rs\tqwen\t1\tbaseline A 1: more\n",
                    "crates/gpu/src/more.rs": "fn qwen_a() {}\nfn qwen_b() {}\n"})
        buf = io.StringIO()
        lower_cmd(d, buf)
        got = read(d, TAX)
        if got != ("# kept comment\ncrates/model/src/arch/mod.rs\tqwen\t2\tbaseline A 5: r\n"
                   "crates/gpu/src/more.rs\tqwen\t1\tbaseline A 1: more\n"):
            fails.append(f"case lower: {got!r}")
        rc, _, err = run(d)
        if rc != 1 or "more.rs: qwen 2 lines, the row allows 1" not in err:
            fails.append(f"case lower: a lowered table must still refuse the file that grew: {rc} {err}")
        # rows: a stub for a (file, family) with no row, none for one that has it
        d = tempfile.mkdtemp(dir=base)
        fixture(d, {"tools/model-tax.tsv": "", "crates/gpu/src/other.rs": "pub fn qwen_x() {}\n"})
        buf = io.StringIO()
        rows_cmd(d, "qwen", buf)
        if buf.getvalue().splitlines() != [
                "crates/gpu/src/other.rs\tqwen\t1\tPIN(YYYY-MM-DD) qwen: <missing field>",
                "crates/model/src/arch/mod.rs\tqwen\t2\tPIN(YYYY-MM-DD) qwen: <missing field>"]:
            fails.append(f"case rows: {buf.getvalue()!r}")
        # copied windows between two families' own files
        d = tempfile.mkdtemp(dir=base)
        body = "".join(f"    let x{i} = step_{i}(qwen_state);\n" for i in range(10))
        fixture(d, {"crates/model/src/arch/qwen/a.rs": "fn a() {\n" + body + "}\n",
                    "crates/glmdir/src/b.rs": "fn b() {\n" + body.replace("qwen_state", "glm_state") + "}\n"})
        pairs = copied_windows(Scan(d), d)
        if sum(pairs.values()) < 1:
            fails.append(f"case copies: no shared window in {dict(pairs)}")
        # only code is windowed: a copied block comment and a line cut at a `//` inside a string are no copy
        d = tempfile.mkdtemp(dir=base)
        fixture(d, {"crates/model/src/arch/qwen/a.rs": "/*\n" + body + "*/\n",
                    "crates/glmdir/src/b.rs": "/*\n" + body.replace("qwen_state", "glm_state") + "*/\n"})
        pairs = copied_windows(Scan(d), d)
        if sum(pairs.values()) != 0:
            fails.append(f"case copies: a block comment is no window: {dict(pairs)}")
        d = tempfile.mkdtemp(dir=base)
        fixture(d, {"crates/model/src/arch/qwen/a.rs": "".join(f'    let u{i} = "a//b{i}";\n' for i in range(10)),
                    "crates/glmdir/src/b.rs": "".join(f'    let u{i} = "a//c{i}";\n' for i in range(10))})
        pairs = copied_windows(Scan(d), d)
        if sum(pairs.values()) != 0:
            fails.append(f"case copies: a string with `//` is not cut: {dict(pairs)}")
        # a file that is not UTF-8 is no verdict, whichever table or source it is
        for rel in (TAX, "crates/gpu/src/other.rs"):
            d = tempfile.mkdtemp(dir=base)
            fixture(d)
            with open(os.path.join(d, rel), "wb") as f:
                f.write(b"pub fn x() { let _ = \"\xff\"; }\n")
            try:
                rc, out, err = run(d)
            except UnicodeDecodeError as e:
                rc, err = "crash", str(e)
            if rc != 2 or f"{rel}: cannot read" not in err:
                fails.append(f"case read: {rel} that is not UTF-8 is no verdict: {rc} {err}")
    finally:
        shutil.rmtree(base, ignore_errors=True)
    if fails:
        for f in fails:
            print(f"model-tax self-test FAIL: {f}", file=sys.stderr)
        print(f"check-model-tax: self-test {len(fails)} failed", file=sys.stderr)
        return 1
    print("check-model-tax: self-test ok")
    return 0


def main(argv):
    root, family, base, args = REPO, None, None, []
    it = iter(argv)
    for a in it:
        if a == "--root":
            root = next(it, None)
        elif a == "--family":
            family = next(it, None)
        elif a == "--base":
            base = next(it, None)
            if base is None:
                print("check-model-tax: --base takes a revision", file=sys.stderr)
                return 2
        else:
            args.append(a)
    if root is None or (family is None and "--family" in argv):
        print("check-model-tax: --root and --family take a value", file=sys.stderr)
        return 2
    if args == ["--self-test"]:
        return self_test()
    cmd = args[0] if args else "check"
    if len(args) > 1 or cmd not in ("check", "rows", "lower") or (base is not None and cmd != "check"):
        print("check-model-tax: usage: check [--base REV] | rows [--family F] | lower | --self-test, each with --root DIR",
              file=sys.stderr)
        return 2
    try:
        if cmd == "check":
            return check(root, sys.stdout, sys.stderr, base)
        if cmd == "rows":
            return rows_cmd(root, family, sys.stdout)
        return lower_cmd(root, sys.stdout)
    except NoVerdict as e:
        print(f"check-model-tax: {e}, no verdict", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
