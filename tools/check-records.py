#!/usr/bin/env python3
"""The record-kind registration rule as a static check (Mac, text only, no build).

    tools/check-records.py check [--root DIR]
    tools/check-records.py schema [--root DIR] [--strict]
    tools/check-records.py --self-test

`record::at_main` and `record::at_main_with_flags` register what a process may print,
and a line of any other kind panics by name. A seat's list is written by hand, but shared modules
print too, after the seat registered. Twice a seat reached a shared module's print without
registering its kind (the GLM seat and `vision`, the GLM and V4.1 seats and the `hf` fetch lines),
and both shipped, because the flag that reaches the print was never started by a gate.

The rule this check holds, with `record::require` (the runtime half) beside it:

  * A *group* is what one entry's path of a shared module prints: a `pub const` slice of kinds in
    record.rs's `groups::<module>` (the module's file stem). A list is its own kinds plus the
    groups it uses (`kinds!`), so a kind added to a group reaches every list that uses it.
  * A shared module calls `record::require(record::groups::<module>::GROUP, "<entry>")` at the
    entry its callers pass, and the process panics by name when the list it registered lacks a kind
    of the group (before `at_main` the requirement waits for it).
  * This check holds the groups equal to what their modules print, and every shared printer a
    group. It reads the sources as text:
      - a *printing file* constructs kinds: `Record::new(&KIND)`, or a call of a `record.rs`
        top-level `pub fn` builder (`record::name(`, a bare name imported from `record`),
        resolved to the kinds the builder constructs, the builders it calls included; the file's
        `#[cfg(test)] mod tests` block is left out. A `Record::new(` whose argument is not `&KIND`
        is a violation (the check could not read it).
      - a *root* is a file that calls either record startup entry; its list is the static it names, composed
        by `kinds!`.
      - the files a root *reaches* are the files its module-name references lead to, transitively
        (`X::` in the text, `X` in a `use` list, a module declared by `mod`/`#[path]` through
        tools/recipes.py's `Tree` walk, an item lib.rs re-exports with `pub use` named through
        `bloomery_gpu_gates::` or `crate::`). The reach over-approximates: it cannot tell a
        function that prints from one that does not, which is why a shared printer is a group
        (checked at run time, exactly) and not a registration rule.
      - a file with `record::require` calls is a *group file*: every kind it constructs is in a
        group of its module, every group of its module is required by a call in it, and a require
        names a group record.rs declares.
      - a file reached by exactly one root: its kinds are in that root's list.
      - a file reached by two or more roots and no group: red, "make it a group".
      - EXCEPTIONS holds the one file kept outside the rule, with its reason.
  * A list that holds a plan record holds the `need` record beside it (PLAN_KINDS, NEED_KIND), and a
    file that prints a plan record calls the need owner (NEED_CALL): the need hook reads that line, so
    a binary that loads placed and prints a plan alone fails every start that sets the hook. Every
    root and every printing file is read, so a new binary or seat is held without being listed.
  * `schema` prints, for each checked-in tools/bloomery/schema/*.jsonl, whether the kind names
    of the composed list are the file's, in its order. `--strict` makes a difference an exit 1;
    tools/check-records.sh runs it so. The kinds' fields are the box test's
    (`checked_in_schemas_are_current`).

Blind spots, stated: the reach is by module name, so a module reached only through a trait object
or a closure passed across files is found when its name is written in the reaching file; a
`Record::new` through a variable is a violation, not a read; a kind printed by `println!` text is
no record.

Exit: 0 ok, 1 violations (one line each, `check-records: <file>: <why>`), 2 usage or an unreadable
tree.
"""

import json
import os
import re
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import recipes  # noqa: E402  (the Tree walk: one reader of #[path]/mod declarations)

PKG = "bloomery-gpu-gates"
PKG_DIR = "crates/gpu-gates"
SRC = PKG_DIR + "/src"
RECORD = SRC + "/record.rs"
LIB = SRC + "/lib.rs"
SCHEMA_DIR = "tools/bloomery/schema"

# Files outside the rule, each with its reason. generate.rs's printers run on every generate start,
# which the gates exercise; the generate/chat lists are read as exact print sets by
# tools/ref/cold-blocks.sh and tools/ref/depth-glm5next.sh, so its kinds stay in those lists by hand.
EXCEPTIONS = {
    SRC + "/generate.rs": "its kinds stay in the generate/chat lists by hand: tools/ref/cold-blocks.sh:535-542 and "
    "tools/ref/depth-glm5next.sh:63-68 read those schemas as exact print sets, and its printers run on every generate start",
}

# The plan records and the record that goes beside each of them: a placed load's need, which the
# owner `need::print` (or `need::staged`, for a plan listed apart from its print) renders.
# PIN(2026-10-21): replaces record.rs's a_binary_that_prints_a_plan_prints_its_need; these rules cover every root and plan printer.
PLAN_KINDS = {"PLAN", "PLAN38"}
NEED_KIND = "NEED"
NEED_CALL = re.compile(r"\bneed::(?:print|staged)\(")

# A module name no reference leads to a file by.
SKIP = {"record", "lib", "main", "mod", "tests", "common"}


class Refuse(Exception):
    """The tree is unreadable for this check (exit 2)."""


# ------------------------------------------------------------------ the source as text


def scan(text):
    """(code, skel): `text` with comments blanked, and the same with string and char literal
    contents blanked too, line structure and offsets kept. Handles `//`, nested `/* */`, `"…"`
    with escapes, `r#"…"#` and char literals (a lifetime is not one)."""
    n = len(text)
    code = list(text)
    skel = list(text)
    i = 0

    def blank(buf, a, b):
        for k in range(a, b):
            if buf[k] != "\n":
                buf[k] = " "

    while i < n:
        c = text[i]
        if text.startswith("//", i):
            j = text.find("\n", i)
            j = n if j < 0 else j
            blank(code, i, j)
            blank(skel, i, j)
            i = j
        elif text.startswith("/*", i):
            depth, j = 1, i + 2
            while j < n and depth:
                if text.startswith("/*", j):
                    depth += 1
                    j += 2
                elif text.startswith("*/", j):
                    depth -= 1
                    j += 2
                else:
                    j += 1
            blank(code, i, j)
            blank(skel, i, j)
            i = j
        elif c == "r" and re.match(r'r#*"', text[i : i + 40]) and (i == 0 or not (text[i - 1].isalnum() or text[i - 1] == "_")):
            m = re.match(r'r(#*)"', text[i:])
            close = '"' + m.group(1)
            j = text.find(close, i + len(m.group(0)))
            j = n if j < 0 else j + len(close)
            blank(skel, i + len(m.group(0)), j - len(close))
            i = j
        elif c == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            blank(skel, i + 1, min(j, n))
            i = j + 1
        elif c == "'":
            m = re.match(r"'(?:\\.[^']*|[^\\'])'", text[i : i + 12])
            if m:
                blank(skel, i + 1, i + len(m.group(0)) - 1)
                i += len(m.group(0))
            else:
                i += 1
        else:
            i += 1
    return "".join(code), "".join(skel)


def cut_tests(code, skel):
    """`code` with each `#[cfg(test)] mod tests { … }` blanked (braces matched on `skel`)."""
    out = list(code)
    for m in re.finditer(r"#\[cfg\(test\)\]\s*(?:#\[[^\]]*\]\s*)*mod\s+tests\s*\{", skel):
        depth, j = 1, m.end()
        while j < len(skel) and depth:
            depth += (skel[j] == "{") - (skel[j] == "}")
            j += 1
        for k in range(m.start(), j):
            if out[k] != "\n":
                out[k] = " "
    return "".join(out)


class Source:
    """The text of a tree's files, comment-free, test blocks cut."""

    def __init__(self, tree):
        self.tree = tree
        self._code = {}

    def code(self, rel):
        if rel not in self._code:
            text = self.tree.read(rel)
            c, s = scan(text)
            self._code[rel] = cut_tests(c, s)
        return self._code[rel]


# ------------------------------------------------------------------ record.rs


def bracket_end(text, start, open_ch="[", close_ch="]"):
    """Index just past the bracket that closes the one at `start`."""
    depth = 0
    for j in range(start, len(text)):
        depth += text[j] == open_ch
        depth -= text[j] == close_ch
        if depth == 0:
            return j + 1
    raise Refuse("an unclosed bracket in record.rs")


class RecordRs:
    """What record.rs declares: the kinds, the builders, the groups and the lists."""

    def __init__(self, code):
        self.code = code
        self.kinds = set(re.findall(r"^pub static ([A-Z][A-Z_0-9]*): Kind\b", code, re.M))
        self.builders = {}
        for m in re.finditer(r"^pub fn ([a-z_0-9]+)\b.*?^}", code, re.M | re.S):
            body = m.group(0)
            self.builders[m.group(1)] = (
                set(re.findall(r"Record::new\(&([A-Z][A-Z_0-9]*)\)", body)),
                {c for c in re.findall(r"(?<![\w:.])([a-z_0-9]+)\(", body) if c != m.group(1)},
            )
        self.groups = self._groups()
        self.lists = self._lists()

    def builder_kinds(self, name, seen=None):
        seen = set() if seen is None else seen
        if name in seen or name not in self.builders:
            return set()
        seen.add(name)
        ks, calls = self.builders[name]
        out = set(ks)
        for c in calls:
            if c in self.builders:
                out |= self.builder_kinds(c, seen)
        return out

    def _groups(self):
        """{module stem: {GROUP: [kind, …]}} from `pub mod groups { pub mod <stem> { pub const GROUP: … } }`."""
        out = {}
        m = re.search(r"^pub mod groups\s*\{", self.code, re.M)
        if not m:
            return out
        end = bracket_end(self.code, m.end() - 1, "{", "}")
        body = self.code[m.end() : end - 1]
        for sm in re.finditer(r"pub mod ([a-z_0-9]+)\s*\{", body):
            send = bracket_end(body, sm.end() - 1, "{", "}")
            for gm in re.finditer(r"pub const ([A-Z][A-Z_0-9]*): &\[&Kind\]\s*=\s*&\[", body[sm.end() : send]):
                a = sm.end() + gm.end() - 1
                b = bracket_end(body, a)
                out.setdefault(sm.group(1), {})[gm.group(1)] = re.findall(r"&([A-Z][A-Z_0-9]*)", body[a:b])
        return out

    def _lists(self):
        """{LIST: [part, …]}: each `pub static LIST: &[&Kind] = …;` as its parts in order, a part
        a kind name or a `(stem, GROUP)` pair."""
        out = {}
        for m in re.finditer(r"^pub static ([A-Z][A-Z_0-9]*): &\[&Kind\]\s*=\s*", self.code, re.M):
            out[m.group(1)] = self.parts(self.code, m.end())
        return out

    def parts(self, text, at):
        """The parts of the list expression that starts at `at` (up to its `;`)."""
        j = at
        depth = 0
        while j < len(text) and not (text[j] == ";" and depth == 0):
            depth += text[j] in "[({"
            depth -= text[j] in "])}"
            j += 1
        expr = text[at:j]
        parts = []
        for tok in re.finditer(r"(?:[a-z_0-9]+::)*groups::([a-z_0-9]+)::([A-Z][A-Z_0-9]*)|&((?:[a-z_0-9]+::)*[A-Z][A-Z_0-9]*)", expr):
            if tok.group(1):
                parts.append((tok.group(1), tok.group(2)))
            else:
                parts.append(tok.group(3).split("::")[-1])
        return parts

    def resolve(self, name, problems, where, text=None):
        """The kind names of list `name` in order (groups expanded); problems get a line per defect.
        `text` is a file-local list's source (a `static NAME: &[&Kind] = …` outside record.rs)."""
        if text is None:
            parts = self.lists.get(name)
        else:
            parts = self.parts(text[0], text[1])
        if parts is None:
            problems.append(f"{where}: no list {name} (a `pub static {name}: &[&Kind]` in record.rs)")
            return None
        out = []
        for p in parts:
            if isinstance(p, tuple):
                kinds = self.groups.get(p[0], {}).get(p[1])
                if kinds is None:
                    problems.append(f"{where}: list {name} names groups::{p[0]}::{p[1]}, which record.rs does not declare")
                    continue
                out.extend(kinds)
            elif p in self.kinds:
                out.append(p)
            else:
                problems.append(f"{where}: list {name} names {p}, which is no kind of record.rs")
        return out


# ------------------------------------------------------------------ the tree


class Gates:
    """The gpu-gates package as the check reads it: its files, the printing ones, the roots, the reach.

    A module name leads to a file within one binary's module tree (and the library's, which every
    binary links), never across binaries: two binaries each declare their own `mod drive`."""

    def __init__(self, tree):
        self.tree = tree
        self.src = Source(tree)
        pkg = tree.packages.get(PKG)
        if pkg is None or pkg.dir != PKG_DIR:
            raise Refuse(f"the workspace has no package {PKG} at {PKG_DIR}")
        self.files = set()
        self.scopes = {}  # file -> [{module name: files}] of every target compiling it
        lib_names = {}
        built = []
        for t in pkg.targets:
            if t.kind not in ("lib", "bin"):
                continue
            w = tree.walk(t.src)
            names = {}
            for path, f in w.mods.items():
                if path and f not in (RECORD, LIB):
                    names.setdefault(path[-1], set()).add(f)
            built.append((t.kind, w.files, names))
            if t.kind == "lib":
                lib_names = names
        for kind, files, names in built:
            for f in files:
                if not f.startswith(SRC + "/"):
                    continue
                self.files.add(f)
                scope = lib_names if kind == "lib" else {**lib_names, **{k: v | lib_names.get(k, set()) for k, v in names.items()}}
                self.scopes.setdefault(f, []).append(scope)
        if RECORD not in self.files:
            raise Refuse(f"{RECORD} is not in the package's module tree")
        self.lib_names = lib_names
        self.rec = RecordRs(self.src.code(RECORD))
        self.exports = self._exports()
        self._idents = {}

    def _exports(self):
        """{leaf: {module name}}: the items lib.rs re-exports from its own modules."""
        out = {}
        if LIB not in self.files:
            return out
        for m in re.finditer(r"\bpub use ([^;]*);", self.src.code(LIB), re.S):
            stmt = m.group(1)
            head = re.match(r"\s*(?:crate::)?([a-z_0-9]+)", stmt)
            if not head or head.group(1) not in self.lib_names:
                continue
            if re.search(r"::\s*\*", stmt):
                raise Refuse(f"{LIB}: `pub use {head.group(1)}::*` re-exports names this check cannot list")
            for item in re.findall(r"(?:\bas\s+)?([A-Za-z_][A-Za-z_0-9]*)\s*(?=[,}]|$)", stmt):
                if item != "self":
                    out.setdefault(item, set()).add(head.group(1))
        return out

    # ---- what a file prints

    def imported_builders(self, t):
        names = set()
        for u in re.findall(r"\buse\s[^;]*\brecord::[^;]*;", t, re.S):
            m = re.search(r"\brecord::\{([^}]*)\}", u)
            if m:
                names |= set(re.findall(r"[a-z_0-9]+", m.group(1)))
            else:
                m = re.search(r"\brecord::([a-z_0-9]+)\s*;", u)
                if m:
                    names.add(m.group(1))
        return names & set(self.rec.builders)

    def prints(self, rel):
        """(kinds the file constructs, [unreadable `Record::new(` arguments])."""
        t = self.src.code(rel)
        ks, bad = set(), []
        for m in re.finditer(r"Record::new\(([^)]*)\)", t):
            arg = m.group(1).strip()
            mm = re.fullmatch(r"&(?:[a-z_0-9]+::)*([A-Z][A-Z_0-9]*)", arg)
            if mm and mm.group(1) in self.rec.kinds:
                ks.add(mm.group(1))
            else:
                bad.append(arg)
        called = set(re.findall(r"\brecord::([a-z_0-9]+)\(", t))
        for b in self.imported_builders(t):
            if re.search(r"(?<![\w:.])" + b + r"\(", t):
                called.add(b)
        for b in called:
            ks |= self.rec.builder_kinds(b)
        return ks, bad

    def requires(self, rel):
        """[(stem, GROUP or None, who, text)] for each `record::require(` call."""
        t = self.src.code(rel)
        out = []
        for m in re.finditer(r"\brecord::require\(", t):
            end = bracket_end(t, m.end() - 1, "(", ")")
            args = t[m.end() : end - 1]
            gm = re.match(r"\s*(?:[a-z_0-9]+::)*groups::([a-z_0-9]+)::([A-Z][A-Z_0-9]*)\s*,\s*\"([^\"]*)\"\s*,?\s*$", args, re.S)
            out.append((gm.group(1), gm.group(2), gm.group(3), args.strip()) if gm else (None, None, None, args.strip()))
        return out

    def roots(self):
        """{root file: [(list name, local list source or None)]}."""
        out = {}
        for f in sorted(self.files - {RECORD}):
            t = self.src.code(f)
            for m in re.finditer(r"\brecord::at_main(?:_with_flags)?\(", t):
                end = bracket_end(t, m.end() - 1, "(", ")")
                args = t[m.end() : end - 1]
                am = re.fullmatch(r"\s*[^,]+,\s*((?:[A-Za-z_0-9]+::)*)([A-Z][A-Z_0-9]*)\s*(?:,\s*.*)?\s*", args, re.S)
                if not am:
                    out.setdefault(f, []).append((None, args.strip()))
                    continue
                name = am.group(2)
                local = None
                lm = re.search(r"^\s*(?:pub(?:\([a-z]+\))? )?static " + name + r":\s*&\[&Kind\]\s*=\s*", t, re.M)
                if lm and name not in self.rec.lists:
                    local = (t, lm.end())
                out.setdefault(f, []).append((name, local))
        return out

    # ---- the reach

    def idents(self, rel):
        """(identifiers of the file's `use` lists, identifiers followed by `::`, the `use` statements)."""
        if rel not in self._idents:
            t = self.src.code(rel)
            uses = re.findall(r"\buse\s[^;]*;", t, re.S)
            use_ids = set(re.findall(r"[A-Za-z_][A-Za-z_0-9]*", " ".join(uses)))
            qual = set(re.findall(r"(?<![A-Za-z0-9_])([A-Za-z_][A-Za-z_0-9]*)::", t))
            self._idents[rel] = (use_ids, qual, uses)
        return self._idents[rel]

    def refs(self, rel):
        use_ids, qual, uses = self.idents(rel)
        t = self.src.code(rel)
        out = set()
        for scope in self.scopes.get(rel, []):
            for n, files in scope.items():
                if n not in SKIP and (n in use_ids or n in qual):
                    out |= files
        top = set()
        for u in uses:
            if re.match(r"use\s+(?:bloomery_gpu_gates|crate)::", u):
                top |= set(re.findall(r"[A-Za-z_][A-Za-z_0-9]*", u))
        for leaf, mods in self.exports.items():
            if leaf in top or re.search(r"\b(?:bloomery_gpu_gates|crate)::" + leaf + r"\b", t):
                for m in mods:
                    out |= self.lib_names.get(m, set())
        return {f for f in out if f in self.files and f != RECORD}

    def reach(self, root):
        seen, todo = {root}, [root]
        while todo:
            f = todo.pop()
            for g in self.refs(f):
                if g not in seen:
                    seen.add(g)
                    todo.append(g)
        return seen


# ------------------------------------------------------------------ the check


def check_tree(tree):
    """(problems, summary) for the tree."""
    g = Gates(tree)
    problems = []
    rec = g.rec
    roots = g.roots()
    lists = {}  # root file -> [(name, kinds)]
    for f, calls in roots.items():
        for name, local in calls:
            if name is None:
                problems.append(f"{f}: record::at_main({local}) names no list this check reads (the second argument is a bare LIST)")
                continue
            kinds = rec.resolve(name, problems, f, local)
            if kinds is not None:
                lists.setdefault(f, []).append((name, set(kinds)))
    reach = {f: g.reach(f) for f in roots}

    for f, ls in sorted(lists.items()):
        for lname, have in ls:
            if have & PLAN_KINDS and NEED_KIND not in have:
                problems.append(f"{f}: list {lname} holds a plan record and not {NEED_KIND}: a placed load prints its need beside its plan, which the need hook reads")

    printing = {}
    for f in sorted(g.files - {RECORD}):
        ks, bad = g.prints(f)
        for b in bad:
            problems.append(f"{f}: `Record::new({b})` is not `Record::new(&KIND)` of a record.rs kind, which this check cannot read")
        if ks:
            printing[f] = ks

    for f, ks in sorted(printing.items()):
        if ks & PLAN_KINDS and not NEED_CALL.search(g.src.code(f)):
            problems.append(f"{f}: prints a plan record and calls neither need::print nor need::staged: a placed load's need goes beside its plan, which the need hook reads")

    by_stem = {}
    for f in g.files:
        by_stem.setdefault(os.path.basename(f)[:-3], []).append(f)
    required = {}  # file -> [(stem, GROUP)]
    for f in sorted(g.files - {RECORD}):
        calls = g.requires(f)
        for stem, grp, who, text in calls:
            if stem is None:
                problems.append(f"{f}: record::require({text}) does not name a `record::groups::<module>::<GROUP>` and a quoted entry")
            elif grp not in rec.groups.get(stem, {}):
                problems.append(f"{f}: record::require names groups::{stem}::{grp}, which record.rs does not declare")
            elif not who:
                problems.append(f"{f}: record::require(groups::{stem}::{grp}, \"\") names no entry")
            else:
                required.setdefault(f, []).append((stem, grp))

    # group files
    group_files = {}
    for stem, groups in rec.groups.items():
        files = [f for f in by_stem.get(stem, []) if f != RECORD]
        if len(files) != 1:
            problems.append(f"{RECORD}: groups::{stem} has {len(files)} files named {stem}.rs in the package, want exactly 1")
            continue
        group_files[files[0]] = stem
    for f in sorted(set(required) | set(group_files)):
        stem = os.path.basename(f)[:-3]
        mine = rec.groups.get(stem)
        if mine is None or group_files.get(f) != stem:
            problems.append(f"{f}: calls record::require but record.rs declares no groups::{stem} for this file")
            continue
        allowed = {k for ks in mine.values() for k in ks}
        for k in sorted(printing.get(f, set()) - allowed):
            problems.append(f"{f}: constructs {k}, which is in no group of groups::{stem} ({', '.join(sorted(mine))})")
        named = {grp for s, grp in required.get(f, []) if s == stem}
        for grp in sorted(set(mine) - named):
            problems.append(f"{f}: groups::{stem}::{grp} is required by no record::require call in this file")

    # every other printing file
    n_checked = 0
    for f, ks in sorted(printing.items()):
        if f in group_files or f in required:
            continue
        if f in EXCEPTIONS:
            continue
        n_checked += 1
        owners = sorted(r for r in roots if f in reach[r])
        if len(owners) >= 2:
            names = ", ".join(os.path.basename(r) for r in owners)
            problems.append(f"{f}: prints {', '.join(sorted(ks))} and is reached by {len(owners)} roots ({names}) without a group: make it a group")
        elif len(owners) == 1:
            r = owners[0]
            for lname, have in lists.get(r, []):
                for k in sorted(ks - have):
                    problems.append(f"{f}: prints {k}, which {os.path.basename(r)}'s list {lname} lacks")
    summary = (
        f"{len(g.files)} files read, {len(rec.kinds)} kinds, {len(roots)} roots, "
        f"{sum(len(v) for v in rec.groups.values())} groups in {len(rec.groups)} modules, "
        f"{len(printing)} printing files, {len(group_files)} group files, {n_checked} single-root files held to a list"
    )
    return problems, summary, g


# ------------------------------------------------------------------ the schemas


def kind_names(g):
    """{KIND: schema name} from record.rs (`name: "…"` of each `pub static KIND: Kind`)."""
    out = {}
    for m in re.finditer(r"^pub static ([A-Z][A-Z_0-9]*): Kind = Kind \{\s*name: \"([^\"]*)\"", g.src.code(RECORD), re.M):
        out[m.group(1)] = m.group(2)
    return out


def list_of_bin(g, bin_):
    """The static list registered for the literal binary name, including parsed-flag startup."""
    for f in sorted(g.files - {RECORD}):
        for m in re.finditer(r"\brecord::at_main(?:_with_flags)?\(\s*\"([^\"]*)\"\s*,\s*(?:[A-Za-z_0-9]+::)*([A-Z][A-Z_0-9]*)\s*(?:,|\))", g.src.code(f)):
            if m.group(1) == bin_:
                return m.group(2)
    return None


def cmd_schema(root, strict):
    """Per checked-in schema file: the composed list's kind names against the file's, in order."""
    tree = recipes.Tree(root)
    g = Gates(tree)
    names = kind_names(g)
    schema_dir = os.path.join(root, SCHEMA_DIR)
    differ = 0
    for fn in sorted(os.listdir(schema_dir)):
        if not fn.endswith(".jsonl"):
            continue
        bin_ = fn[: -len(".jsonl")]
        with open(os.path.join(schema_dir, fn), encoding="utf-8") as fh:
            have = [json.loads(line)["kind"] for i, line in enumerate(fh) if i and line.strip()]
        lst = list_of_bin(g, bin_)
        problems = []
        kinds = g.rec.resolve(lst, problems, bin_) if lst else None
        if kinds is None:
            print(f"schema {bin_}: no record startup registers a list under that name {problems}")
            differ += 1
            continue
        want = [names[k] for k in kinds]
        if want == have:
            print(f"schema {bin_}: equal ({len(have)} kinds, list {lst})")
            continue
        differ += 1
        add = [k for k in want if k not in have]
        drop = [k for k in have if k not in want]
        same = [k for k in want if k in have] == [k for k in have if k in want]
        print(
            f"schema {bin_}: DIFFERS (list {lst}): +{add or '-'} -{drop or '-'}; "
            + ("the kinds both hold keep their order" if same else "the order of the kinds both hold moved")
        )
    print(f"check-records: schema: {differ} of the checked-in files differ from the composed lists (just records-refresh)" if differ else "check-records: schema: every checked-in file is its composed list")
    return 1 if (differ and strict) else 0


def cmd_check(root):
    tree = recipes.Tree(root)
    problems, summary, _ = check_tree(tree)
    for p in problems:
        print(f"check-records: {p}")
    if problems:
        print(f"check-records: {len(problems)} violation(s)", file=sys.stderr)
        return 1
    print(f"check-records: ok ({summary})")
    return 0


# ------------------------------------------------------------------ the self-test

KINDS_FIX = """\
pub static A: Kind = Kind { name: "a", head: "a", parts: &[], doc: "" };
pub static B: Kind = Kind { name: "b", head: "b", parts: &[], doc: "" };
pub static C: Kind = Kind { name: "c", head: "c", parts: &[], doc: "" };

pub fn made_a() -> Record {
    Record::new(&A)
}

pub fn made_ab() -> Record {
    made_a();
    Record::new(&B)
}
"""

GROUPS_FIX = """
pub mod groups {
    use crate::record::{A, B, Kind};

    pub mod shared {
        use super::*;
        pub const ONE: &[&Kind] = &[&A, &B];
    }
}
"""

LISTS_GROUPED = """
pub static ROOT1: &[&Kind] = kinds![&[&C], groups::shared::ONE];
pub static ROOT2: &[&Kind] = kinds![&[&C], groups::shared::ONE];
"""

LISTS_PLAIN = """
pub static ROOT1: &[&Kind] = &[&A, &C];
pub static ROOT2: &[&Kind] = &[&B, &C];
"""

LIB_FIX = "pub mod record;\npub mod shared;\n{reexport}\n"

SHARED_GROUP = """\
use crate::record::{self, Record};

pub struct Item;

pub fn enter() {
    record::require(record::groups::shared::ONE, "shared::enter");
    Record::new(&crate::record::A).print();
    record::made_ab().print();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn t() {
        Record::new(&crate::record::C).print();
    }
}
"""

ROOT_FIX = """\
use bloomery_gpu_gates::record;

fn main() {{
    record::at_main("{name}", record::{lst});
    {body}
}}
"""


def tree_in(top, files):
    """The Tree over `files` (name under src → text) written below `top`: a lib if `lib.rs` is
    among them, a bin for each `bin/<name>.rs`."""
    targets = []
    for name, text in files.items():
        p = os.path.join(top, SRC, name)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(text)
        if name == "lib.rs":
            targets.append({"kind": ["lib"], "name": "bloomery_gpu_gates", "src_path": p})
        elif name.startswith("bin/") and name.count("/") == 1:
            targets.append({"kind": ["bin"], "name": name[4:-3], "src_path": p})
    meta = {
        "workspace_root": top,
        "workspace_members": ["gates"],
        "packages": [
            {
                "id": "gates",
                "name": PKG,
                "manifest_path": os.path.join(top, PKG_DIR, "Cargo.toml"),
                "dependencies": [],
                "features": {},
                "targets": targets,
            }
        ],
    }
    return recipes.Tree(top, meta=meta)


def fixture(grouped=True, shared=SHARED_GROUP, reexport="pub use shared::Item;", one="bloomery_gpu_gates::Item; shared::enter();", two="shared::enter();", record_text=None):
    """A fixture's files: `grouped` has the group module and the lists that use it."""
    rec = record_text or (KINDS_FIX + (GROUPS_FIX + LISTS_GROUPED if grouped else LISTS_PLAIN))
    return {
        "lib.rs": LIB_FIX.format(reexport=reexport),
        "record.rs": rec,
        "shared.rs": shared,
        "bin/one.rs": ROOT_FIX.format(name="one", lst="ROOT1", body=one),
        "bin/two.rs": ROOT_FIX.format(name="two", lst="ROOT2", body=two),
    }


def run_fixture(files):
    top = tempfile.mkdtemp(prefix="check-records-")
    try:
        problems, _, _ = check_tree(tree_in(top, files))
        return problems
    finally:
        shutil.rmtree(top, ignore_errors=True)


def self_test():
    fails = []

    def expect(what, files, want):
        """want None: green; else a substring one problem line holds."""
        try:
            problems = run_fixture(files)
        except Exception as e:  # noqa: BLE001
            fails.append(f"{what}: raised {type(e).__name__}: {e}")
            return
        if want is None:
            if problems:
                fails.append(f"{what}: wanted green, got {problems}")
        elif not any(want in p for p in problems):
            fails.append(f"{what}: wanted a problem holding {want!r}, got {problems}")

    # a clean tree: two roots share a group file that requires its group; its `mod tests` prints
    # a kind outside the group, which the check leaves out
    expect("clean tree", fixture(), None)
    # control: the same block without its cfg(test) is the file's code, and the kind is outside the group
    expect("mod tests control", fixture(shared=SHARED_GROUP.replace("#[cfg(test)]\n", "")), "constructs C, which is in no group")

    # a group file constructing a kind outside its group
    outside = SHARED_GROUP.replace("record::made_ab().print();", "record::made_ab().print();\n    Record::new(&crate::record::C).print();")
    expect("kind outside its group", fixture(shared=outside), "constructs C, which is in no group")

    # a group with no require
    expect("group never required", fixture(shared=SHARED_GROUP.replace('    record::require(record::groups::shared::ONE, "shared::enter");\n', "")), "is required by no record::require call")

    # a printing file reached by two roots without groups
    plain = "use crate::record::{self, Record};\npub struct Item;\npub fn enter() {\n    Record::new(&crate::record::A).print();\n}\n"
    expect("two roots, no group", fixture(grouped=False, shared=plain), "reached by 2 roots")

    # a builder call resolved to its kind: made_ab constructs B (and A through made_a); the only
    # root that reaches the file lists A and C
    via_builder = "use crate::record::{self, Record};\npub struct Item;\npub fn enter() {\n    record::made_ab().print();\n}\n"
    expect("a builder call resolves to the kinds it constructs", fixture(grouped=False, shared=via_builder, two="let _ = 1;"), "prints B, which one.rs's list ROOT1 lacks")
    parsed_startup = fixture(grouped=False, shared=via_builder, two="let _ = 1;")
    parsed_startup["bin/one.rs"] = parsed_startup["bin/one.rs"].replace(
        'record::at_main("one", record::ROOT1)',
        'record::at_main_with_flags("one", record::ROOT1, |flag| matches!(flag, "--records-schema"))',
    )
    expect("parsed-flag startup still rejects an unregistered printed kind", parsed_startup,
           "prints B, which one.rs's list ROOT1 lacks")

    # a `pub use` re-export edge: the root names the item only through the crate path
    only_a = "use crate::record::{self, Record};\npub struct Item;\npub fn enter() {\n    Record::new(&crate::record::B).print();\n}\n"
    via_reexport = fixture(grouped=False, shared=only_a, one="let _x: Option<bloomery_gpu_gates::Item> = None;", two="let _ = 1;")
    expect("pub use edge reaches the printer", via_reexport, "prints B, which one.rs's list ROOT1 lacks")
    without = dict(via_reexport)
    without["lib.rs"] = LIB_FIX.format(reexport="")
    expect("without the re-export nothing is reached", without, None)

    # a module name leads to a file within one binary's tree: two bins each with an inline `mod drive`
    drive = "mod drive {{ pub fn go() {{ record::Record::new(&record::{k}).print(); }} }}\n"
    two_drives = fixture(grouped=False, shared=plain, one="drive::go();", two="drive::go();")
    two_drives["bin/one.rs"] = drive.format(k="A") + two_drives["bin/one.rs"]
    two_drives["bin/two.rs"] = drive.format(k="B") + two_drives["bin/two.rs"]
    two_drives["lib.rs"] = LIB_FIX.format(reexport="")
    two_drives["shared.rs"] = "pub struct Item;\n"
    expect("one binary's module is not another's", two_drives, None)

    # a Record::new through a variable cannot be read
    expect("Record::new through a variable", fixture(shared=SHARED_GROUP.replace("Record::new(&crate::record::A).print();", "let k = &crate::record::A;\n    Record::new(k).print();")), "cannot read")

    # a require naming a group record.rs does not declare
    expect("require of an undeclared group", fixture(shared=SHARED_GROUP.replace("groups::shared::ONE", "groups::shared::TWO")), "does not declare")

    # a group module whose file does not require it nor exist
    expect("group module without a file", fixture(record_text=KINDS_FIX + GROUPS_FIX.replace("pub mod shared", "pub mod nowhere") + LISTS_GROUPED.replace("groups::shared", "groups::nowhere")), "files named nowhere.rs")

    # a list with a plan record and no need record beside it is named; with it, or with the other
    # plan kind and its need, it is green
    plan_kinds = KINDS_FIX + "pub static PLAN: Kind = Kind { name: \"plan\", head: \"plan\", parts: &[], doc: \"\" };\n" \
        "pub static PLAN38: Kind = Kind { name: \"plan38\", head: \"plan38\", parts: &[], doc: \"\" };\n" \
        "pub static NEED: Kind = Kind { name: \"need\", head: \"need\", parts: &[], doc: \"\" };\n"
    plan_lists = "pub static ROOT1: &[&Kind] = &[&PLAN, &NEED, &C];\npub static ROOT2: &[&Kind] = &[&C];\n"
    no_items = "pub struct Item;\n"
    expect("a plan list holding its need", fixture(grouped=False, shared=no_items, record_text=plan_kinds + plan_lists), None)
    expect("a plan list without its need", fixture(grouped=False, shared=no_items, record_text=plan_kinds + plan_lists.replace("&PLAN, &NEED", "&PLAN")), "list ROOT1 holds a plan record and not NEED")
    expect("a plan38 list without its need", fixture(grouped=False, shared=no_items, record_text=plan_kinds + plan_lists.replace("&PLAN, &NEED", "&PLAN38")), "list ROOT1 holds a plan record and not NEED")

    # a file that prints a plan record calls the need owner: without the call it is named, with it
    # (print, or staged for the plan that is listed apart) it is green
    plan_file = "use crate::record::{{self, Record}};\npub struct Item;\npub fn enter() {{\n    Record::new(&crate::record::PLAN).print();\n{call}}}\n"
    plan_tree = dict(record_text=plan_kinds + plan_lists, two="let _ = 1;", grouped=False)
    expect("a plan printer without a need call", fixture(shared=plan_file.format(call=""), **plan_tree), "calls neither need::print nor need::staged")
    expect("a plan printer that prints its need", fixture(shared=plan_file.format(call="    bloomery_gpu_gates::need::print(&plan, 0, 0, true);\n"), **plan_tree), None)
    expect("a plan printer that stages its need", fixture(shared=plan_file.format(call="    let n = bloomery_gpu_gates::need::staged(&plan, 0, 0);\n"), **plan_tree), None)

    # the real tree: green, then red with a kind added to a group file
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.path.isfile(os.path.join(root, RECORD)):
        real = recipes.Tree(root)
        problems, summary, _ = check_tree(real)
        if problems:
            fails.append(f"the real tree is not green: {problems[:3]}")
        top = tempfile.mkdtemp(prefix="check-records-real-")
        try:
            shutil.copytree(os.path.join(root, SRC), os.path.join(top, SRC), ignore=shutil.ignore_patterns("target"))
            meta = recipes.cargo_metadata(root)
            r = os.path.realpath(root)
            blob = json.dumps(meta).replace(r, os.path.realpath(top))
            meta = json.loads(blob)
            meta["packages"] = [p for p in meta["packages"] if p["name"] == PKG]
            meta["workspace_members"] = [p["id"] for p in meta["packages"]]
            vs = os.path.join(top, SRC, "vision_seat.rs")
            with open(vs, encoding="utf-8") as fh:
                text = fh.read()
            with open(vs, "w", encoding="utf-8") as fh:
                fh.write(text.replace("pub fn seat_record(at: &Opened) -> Record {", "pub fn seat_record(at: &Opened) -> Record {\n    let _ = Record::new(&record::TOP2);", 1))
            problems, _, _ = check_tree(recipes.Tree(top, meta=meta))
            hit = [p for p in problems if "vision_seat.rs" in p and "TOP2" in p]
            if not hit:
                fails.append(f"a TOP2 line in vision_seat.rs is not named: {problems[:3]}")
            with open(vs, "w", encoding="utf-8") as fh:
                fh.write(text)
            host_tier = os.path.join(top, SRC, "record/host_tier.rs")
            record = os.path.join(top, RECORD)
            for path, old, new, want in [
                (record, "pub const ARENA: &[&Kind] = &[&NVTIER];", "pub const ARENA: &[&Kind] = &[];", "constructs NVTIER"),
                (record, "pub const CENSUS: &[&Kind] = &[&TIER_CENSUS];", "pub const CENSUS: &[&Kind] = &[];", "constructs TIER_CENSUS"),
                (host_tier, 'record::require(record::groups::host_tier::ARENA, "host_tier::eprint_nvtier");', "", "groups::host_tier::ARENA is required by no"),
                (host_tier, 'record::require(record::groups::host_tier::CENSUS, "CensusSpan::mark");', "", "groups::host_tier::CENSUS is required by no"),
            ]:
                with open(path, encoding="utf-8") as fh:
                    original = fh.read()
                if original.count(old) != 1:
                    fails.append(f"host-tier mutant {want}: expected one mutation site")
                    continue
                try:
                    with open(path, "w", encoding="utf-8") as fh:
                        fh.write(original.replace(old, new, 1))
                    problems, _, _ = check_tree(recipes.Tree(top, meta=meta))
                    if not any("host_tier.rs" in p and want in p for p in problems):
                        fails.append(f"host-tier mutant {want}: {problems}")
                    else:
                        print(f"check-records: FAIL-first host-tier {want}: ok")
                finally:
                    with open(path, "w", encoding="utf-8") as fh:
                        fh.write(original)
        finally:
            shutil.rmtree(top, ignore_errors=True)
    else:
        fails.append(f"{RECORD} is not under {root}: the real-tree fixtures could not run")

    for f in fails:
        print(f"check-records self-test FAIL: {f}", file=sys.stderr)
    if fails:
        return 1
    print("check-records: self-test ok")
    return 0


# ------------------------------------------------------------------ main


def main(argv):
    if argv == ["--self-test"]:
        return self_test()
    args = list(argv)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if "--root" in args:
        i = args.index("--root")
        if i + 1 >= len(args):
            print("check-records: --root takes a directory", file=sys.stderr)
            return 2
        root = args[i + 1]
        del args[i : i + 2]
    strict = "--strict" in args
    if strict:
        args.remove("--strict")
    try:
        if args == ["check"]:
            return cmd_check(root)
        if args == ["schema"]:
            return cmd_schema(root, strict)
    except (Refuse, recipes.RecipeError) as e:
        print(f"check-records: refused: {e}", file=sys.stderr)
        return 2
    print("usage: tools/check-records.py check [--root DIR] | schema [--root DIR] [--strict] | --self-test", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
