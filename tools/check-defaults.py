#!/usr/bin/env python3
"""The defaults-twins rule as a static check (Mac, text only, no build).

    tools/check-defaults.py check [--root DIR]
    tools/check-defaults.py list  [--root DIR]
    tools/check-defaults.py --self-test

GitHub #3 shipped because every gate that ran the MTP draft pinned
`BLOOMERY_MTP_WIDTH=fixed`, so no gate ran the default width chooser across a
server's request sequence. The rule this check holds: a lever a gate pins off
its default needs a defaults twin — a clause that runs the same model path at
the default — or a named, dated reason why not, as a row of
tools/defaults-twins.tsv.

`check` scans every off-default lever pin the tree's gate surface sets and
holds it to the TSV; `list` prints the pins the scanner sees (the census's one
reader — the TSV is seeded from it, so the two cannot drift apart in shape)
with each TSV row's twin resolved to its `path:line`, so a reader can still
jump to the clause.

What a pin is, and where the scanner looks:

  justfile   every recipe (any name — the census covers gate-* and weekly-*,
             the check guards the rest too): a `BLOOMERY_<lever>=<value>`
             assignment on a non-comment body line. The value is the text as
             written (a `${NAME:-default}` escape or a `$c` loop variable
             stays as it is); a lever mentioned only inside a printed string
             that also has the real assignment on the same line collapses
             into that one pin.
  .rs        every .rs under crates/gpu-gates/src (the gate bins and their
             shared harness) and every crates/*/tests/*.rs (the gate-* recipes
             that run crate tests): a child's `.env(<lever>, <value>)`, an
             in-process `set_var(<lever>, <value>)`, and a `(lever, value)`
             pair literal anywhere in the file — the shape every lever table
             feeds an `.envs(..)` or a `for (k, v) in .. cmd.env(k, v)` loop
             with, be it a `const` array, a local `let` array or an arms
             table. A name given as `bloomery_levers::<CONST>`, a same-file
             `const` or a one-hop `let` alias resolves to its value.
  flags      a CLI flag that overrides a lever, listed in FLAGS below: today
             `--cache-type-k` (the qwen3 seat's spelling of BLOOMERY_QWEN3_KV,
             shared/serve_seats/qwen3.rs). A new flag-to-lever coupling is a
             row of FLAGS the day it is born, with its parser's path.

A lever is a live row of crates/levers/src/registry.rs — `Site::Parsed` or
`Site::Direct`. A retired name set by a gate is a named refusal, not a path,
so it is no pin; the P and R rows are no lever at all.

The TSV, one row a pin identity (lever, where, value — sites collapse):

    lever <TAB> where <TAB> value <TAB> twin <TAB> reason

`where` is the justfile recipe for a recipe pin, and the recipe that runs the
file for a Rust pin (several, comma-joined, when several run it). `twin` is
`path::needle` — a clause that runs the same model path at the default, named
by a literal substring of one of its lines (a Rust clause's name or a unique
string literal in it; a justfile recipe's `name:` header or the arm's log
name); the needle must occur exactly once in `path`, so a twin whose clause
moved or vanished is red instead of a silent pass, and a pin whose value
already is the default's word may name its own clause. `twin` `none` needs a
reason opening with a date (`YYYY-MM-DD: `); every run prints the count of
those rows so the open gaps stay visible. Red (exit 1): a pin with no row, a
row whose pin no longer exists, a twin whose path is not in the tree or whose
needle occurs zero times or more than once in it, an undated `none`. Named
error (exit 65): a registry row, a recipe or a TSV line the reader cannot
parse, a lever the registry does not know, a gate file no recipe names.

Blind spots, each named: lever values built in code as typed values the
environment never carries (`Residency::Mid` consts, `set_hoststream`,
`KvQ8::Q8`, `GLM_RESIDENCY_UNSET`, gate_hybrid's n_l, `--slots`) — the census
in the round report lists them by hand; a lever a runner or a caller's
environment carries (`BLOOMERY_BOX_ENV`, tools/ref/*) — measurement, not
gates; a twin whose clause is rewritten around an unchanged needle line — the
needle pins the line's text, not the clause's behaviour.
"""
import os
import re
import shutil
import sys
import tempfile

TSV = "tools/defaults-twins.tsv"
REGISTRY = "crates/levers/src/registry.rs"
JUSTFILE = "justfile"

# A CLI flag that overrides a lever, with the parser that owns the coupling.
FLAGS = {
    "--cache-type-k": ("BLOOMERY_QWEN3_KV", "crates/gpu-gates/src/bin/shared/serve_seats/qwen3.rs"),
}

NAME_TAIL = r"[^\s\"';&)]+"


class InputError(Exception):
    """Input this tool cannot read; the message names the file and the line."""


def balanced(text, open_at):
    """The text between the bracket at open_at and the one that closes it; string
    literals are skipped whole."""
    depth, i = 0, open_at
    while i < len(text):
        c = text[i]
        if c == '"':
            i += 1
            while i < len(text) and text[i] != '"':
                i += 2 if text[i] == "\\" else 1
        elif c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
            if depth == 0:
                return text[open_at + 1:i]
        i += 1
    return None


def split_args(inner):
    """Top-level comma-separated items of an argument or parameter list."""
    out, depth, cur, i = [], 0, "", 0
    while i < len(inner):
        c = inner[i]
        if c == '"':
            j = i + 1
            while j < len(inner) and inner[j] != '"':
                j += 2 if inner[j] == "\\" else 1
            cur += inner[i:j + 1]
            i = j + 1
            continue
        if c in "([{<":
            depth += 1
        elif c in ")]}>":
            depth -= 1
        elif c == "," and depth == 0:
            out.append(cur.strip())
            cur, i = "", i + 1
            continue
        cur += c
        i += 1
    if cur.strip():
        out.append(cur.strip())
    return out


def line_of(text, at):
    return text.count("\n", 0, at) + 1


# ---------------------------------------------------------------- the registry


def live_levers(root):
    """({name: site}, {const: name}) of the registry's live lever rows (Parsed,
    Direct). A row the reader cannot hold is a named error, never a skip."""
    path = os.path.join(root, REGISTRY)
    try:
        text = open(path, encoding="utf-8").read()
    except OSError as e:
        raise InputError(f"{REGISTRY}: {e}") from None
    consts = dict(re.findall(r'\b(?:pub )?const ([A-Z_][A-Z0-9_]*)\s*:\s*&(?:\'static )?str'
                             r'\s*=\s*"([^"]*)"', text))
    out = {}
    for m in re.finditer(r"LeverSpec\s*\{", text):
        if text[:m.start()].rstrip().endswith(">"):
            continue  # a helper's `-> LeverSpec {` signature brace, no row
        block = balanced(text, m.end() - 1)
        if block is None:
            raise InputError(f"{REGISTRY}:{line_of(text, m.start())}: a LeverSpec block does not close")
        if re.search(r"^\s*name\s*,", block, re.M):
            continue  # the `path`/`runner` helpers' own template (name shorthand), no row
        n = re.search(r"\bname:\s*([A-Za-z_][A-Za-z0-9_]*|\"[^\"]*\")", block)
        s = re.search(r"\bsite:\s*Site::(Parsed|Direct|Retired|Env)", block)
        if not n or not s:
            raise InputError(f"{REGISTRY}:{line_of(text, m.start())}: a LeverSpec row without a "
                             "readable name or site")
        name = n.group(1)
        if name.startswith('"'):
            name = name[1:-1]  # a string literal names itself
        elif name in consts:
            name = consts[name]
        else:
            raise InputError(f"{REGISTRY}:{line_of(text, m.start())}: a row names {n.group(1)}, "
                             "which no const of the file defines")
        if name in out:
            raise InputError(f"{REGISTRY}: {name} has two LeverSpec rows")
        if s.group(1) in ("Parsed", "Direct"):
            out[name] = s.group(1)
    if not out:
        raise InputError(f"{REGISTRY}: no live lever row found — the reader is lost")
    return out, consts


# ---------------------------------------------------------------- the justfile


def recipes_of(root):
    """{name: body} of the justfile's recipes; a header the reader cannot hold
    ends the read by name (an assignment and a comment are no recipe)."""
    try:
        lines = open(os.path.join(root, JUSTFILE), encoding="utf-8").read().split("\n")
    except OSError as e:
        raise InputError(f"{JUSTFILE}: {e}") from None
    out, name, body = {}, None, []
    header = re.compile(r"^([a-z0-9][a-z0-9_-]*)(?:[ \t]*[a-zA-Z0-9_*'\",=\[\].() -]*):")
    for line in lines:
        if line[:1] not in (" ", "\t"):
            if name is not None:
                out[name] = "\n".join(body)
            name, body = None, []
            if not line.strip() or line.startswith(("#", "[")):
                continue
            if re.match(r"^[a-z0-9][a-z0-9_-]*\s*:=", line):
                continue  # an assignment
            m = header.match(line)
            if m:
                name, body = m.group(1), []
            continue
        if name is not None:
            body.append(line)
    if name is not None:
        out[name] = "\n".join(body)
    return out


def scan_justfile(root, live):
    """Pins of the recipes: (lever, recipe, value, 'recipe', file:line)."""
    try:
        lines = open(os.path.join(root, JUSTFILE), encoding="utf-8").read().split("\n")
    except OSError as e:
        raise InputError(f"{JUSTFILE}: {e}") from None
    pins, header = [], re.compile(r"^([a-z0-9][a-z0-9_-]*)(?:[ \t]*[a-zA-Z0-9_*'\",=\[\].() -]*):")
    assign = re.compile(r"(BLOOMERY_[A-Z0-9_]+)=" + f"({NAME_TAIL})")
    name = None
    for n, line in enumerate(lines, 1):
        if line[:1] not in (" ", "\t"):
            name = None
            if not line.strip() or line.startswith(("#", "[")):
                continue
            if re.match(r"^[a-z0-9][a-z0-9_-]*\s*:=", line):
                continue
            m = header.match(line)
            if m:
                name = m.group(1)
            continue
        if name is None or line.lstrip().startswith("#"):
            continue
        for m in assign.finditer(line):
            if m.group(1) in live:
                pins.append((m.group(1), name, m.group(2), "recipe", f"{JUSTFILE}:{n}"))
    return pins


# ---------------------------------------------------------------- the rust side


def scan_rust(rel, text, live, reg_consts):
    """Pins of one Rust file: (lever, where-label-later, value, kind, rel:line). A
    lever's const resolves through the file's own consts, then the registry's
    (`bloomery_levers::<CONST>`), so a pin never depends on its spelling."""
    consts = dict(re.findall(r'\b(?:pub )?const ([A-Z_][A-Z0-9_]*)\s*:\s*&(?:\'static )?str'
                             r'\s*=\s*"([^"]*)"', text))
    names = dict(reg_consts)
    names.update(consts)
    locals_ = {}
    for m in re.finditer(r"\blet ([a-z_][a-z0-9_]*)\s*=\s*([A-Z_][A-Z0-9_]*)\s*;", text):
        if m.group(2) in names and m.group(1) not in locals_:
            locals_[m.group(1)] = names[m.group(2)]

    def lever_of(arg):
        arg = arg.strip()
        if arg.startswith('"') and arg.endswith('"') and len(arg) >= 2:
            return arg[1:-1]
        arg = re.sub(r"^.*::", "", arg)
        return names.get(arg)

    def value_of(arg):
        arg = arg.strip()
        if arg.startswith('"') and arg.endswith('"') and len(arg) >= 2:
            return arg[1:-1]
        if re.fullmatch(r"&?'?[A-Za-z_][A-Za-z0-9_]*", arg):
            bare = arg.lstrip("&'")
            if bare in consts:
                return consts[bare]
            if bare in locals_:
                return locals_[bare]
            return bare
        return re.sub(r"\s+", " ", arg)

    pins = []
    taken = []  # the '(' offsets an .env( or set_var( call already read

    def add(lever, value, kind, at):
        if lever in live:
            pins.append((lever, None, value, kind, f"{rel}:{line_of(text, at)}"))

    for m in re.finditer(r"\.env\s*\(", text):
        inner = balanced(text, m.end() - 1)
        if inner is None:
            raise InputError(f"{rel}:{line_of(text, m.start())}: an .env( call does not close")
        args = split_args(inner)
        if len(args) < 2:
            continue
        taken.append(m.end() - 1)
        add(lever_of(args[0]), value_of(args[1]), "env", m.start())
    for m in re.finditer(r"\bset_var\s*\(", text):
        inner = balanced(text, m.end() - 1)
        if inner is None:
            raise InputError(f"{rel}:{line_of(text, m.start())}: a set_var call does not close")
        args = split_args(inner)
        if len(args) >= 2:
            taken.append(m.end() - 1)
            add(lever_of(args[0]), value_of(args[1]), "set", m.start())
    pair = re.compile(r"\(\s*(\"BLOOMERY_[A-Z0-9_]+\"|bloomery_levers::[A-Z_][A-Z0-9_]*|[A-Z_][A-Z0-9_]*)"
                      r"\s*,\s*([^()]+?)\s*\)")
    plain = re.compile(r"&?(?:[A-Za-z_][A-Za-z0-9_]*::)*[A-Za-z_][A-Za-z0-9_]*")
    for m in pair.finditer(text):
        if m.start() in taken:  # the .env( or set_var( call the pair sits in, already read
            continue
        v = m.group(2).strip()
        quoted = v.startswith('"') and v.endswith('"') and len(v) >= 2
        if not quoted and not plain.fullmatch(v):
            continue  # an expression value is no table arm: a computed value flows through .env(
        add(lever_of(m.group(1)), value_of(m.group(2)), "table", m.start())
    for flag, (lever, _) in FLAGS.items():
        for m in re.finditer(re.escape(f'"{flag}"') + r'\s*,\s*"([^"]*)"', text):
            add(lever, m.group(1), "flag", m.start())
    return pins


def rust_files(root):
    out = []
    base = os.path.join(root, "crates", "gpu-gates", "src")
    for dirpath, _, files in os.walk(base):
        out += [os.path.join(dirpath, f) for f in sorted(files) if f.endswith(".rs")]
    crates = os.path.join(root, "crates")
    for crate in sorted(os.listdir(crates)):
        tests = os.path.join(crates, crate, "tests")
        if not os.path.isdir(tests):
            continue
        for dirpath, _, files in os.walk(tests):
            out += [os.path.join(dirpath, f) for f in sorted(files) if f.endswith(".rs")]
    return out


def scan_tree(root):
    """Every pin of the tree's gate surface, with the recipe label a Rust pin's
    file runs under (several, comma-joined)."""
    live, reg_consts = live_levers(root)
    recipes = recipes_of(root)
    pins = scan_justfile(root, live)
    for path in rust_files(root):
        rel = os.path.relpath(path, root)
        try:
            text = open(path, encoding="utf-8").read()
        except OSError as e:
            raise InputError(f"{rel}: {e}") from None
        for lever, _, value, kind, site in scan_rust(rel, text, live, reg_consts):
            stem = os.path.splitext(os.path.basename(rel))[0]
            # a crate test runs by its recipe's `--test <stem>`; a gate bin by its
            # name as a word (a substring would take `mt` for `glm5next`)
            if rel.startswith("crates/") and "/tests/" in rel:
                who = sorted(n for n, body in recipes.items()
                             if re.search(rf"--test {stem}\b", body))
            else:
                who = sorted(n for n, body in recipes.items() if re.search(rf"\b{stem}\b", body))
            if not who:
                raise InputError(f"{site}: a pin in {rel}, which no justfile recipe names — "
                                 "give its row a recipe or teach the scanner")
            pins.append((lever, ",".join(who), value, kind, site))
    return pins, live


# ---------------------------------------------------------------- the TSV


def read_tsv(root, live):
    """[(lever, where, value, twin, reason, line)] with every field checked."""
    path = os.path.join(root, TSV)
    try:
        lines = open(path, encoding="utf-8").read().split("\n")
    except OSError as e:
        raise InputError(f"{TSV}: {e}") from None
    rows = []
    for n, line in enumerate(lines, 1):
        if not line.strip() or line.startswith("#"):
            continue
        cols = line.split("\t")
        if len(cols) != 5 or not all(c.strip() for c in cols):
            raise InputError(f"{TSV}:{n}: a row is lever<TAB>where<TAB>value<TAB>twin<TAB>reason: {line!r}")
        lever, where, value, twin, reason = (c.strip() for c in cols)
        if lever not in live:
            raise InputError(f"{TSV}:{n}: {lever} is no live lever of the registry "
                             f"(live rows only; a retired name is a named refusal, P and R rows are no lever)")
        rows.append((lever, where, value, twin, reason, n))
    return rows


def resolve_twin(root, p, needle):
    """(occurrences, line) of the needle in root/p — the line set only when it
    occurs exactly once; occurrences None when the file is not in the tree."""
    f = os.path.join(root, p)
    if not os.path.isfile(f):
        return None, None
    text = open(f, encoding="utf-8", errors="replace").read()
    n = text.count(needle)
    if n != 1:
        return n, None
    at = text.find(needle)
    return 1, text.count("\n", 0, at) + 1


def split_twin(twin):
    """(path, needle) of a `path::needle` twin; (None, None) when malformed."""
    if "::" not in twin:
        return None, None
    p, needle = twin.split("::", 1)
    if not p.strip() or not needle.strip():
        return None, None
    return p, needle


def check(root):
    pins, live = scan_tree(root)
    rows = read_tsv(root, live)
    bad = []
    seen = {(lever, where, value) for lever, where, value, _, _, _ in rows}
    for lever, where, value, _, site in pins:
        if (lever, where, value) not in seen:
            bad.append(f"{site}: {lever}={value} pinned in {where} and no row of {TSV} holds it — "
                       "a lever a gate pins off its default needs a defaults twin or a dated reason")
    pin_keys = {(lever, where, value) for lever, where, value, _, _ in pins}
    for lever, where, value, twin, reason, n in rows:
        if (lever, where, value) not in pin_keys:
            bad.append(f"{TSV}:{n}: the pin {lever}={value} in {where} no longer exists — take the row with it")
        if twin == "none":
            if not re.match(r"^\d{4}-\d{2}-\d{2}: ", reason):
                bad.append(f"{TSV}:{n}: a `none` twin needs a reason opening with a date "
                           f"(YYYY-MM-DD: …): {reason!r}")
            continue
        p, needle = split_twin(twin)
        if p is None:
            bad.append(f"{TSV}:{n}: a twin is `path::needle` or `none`: {twin!r}")
            continue
        hits, _ = resolve_twin(root, p, needle)
        if hits is None:
            bad.append(f"{TSV}:{n}: the twin {twin} is not found in the tree")
        elif hits == 0:
            bad.append(f"{TSV}:{n}: the twin needle of {twin} occurs zero times in {p} "
                       "— the clause moved or is gone; re-anchor the row")
        elif hits > 1:
            bad.append(f"{TSV}:{n}: the twin needle of {twin} occurs {hits} times in "
                       f"{p}, not exactly once — the anchor no longer names one clause")
    gaps = sum(1 for r in rows if r[3] == "none")
    pins.sort(key=lambda p: (p[0], p[1], p[2]))
    if bad:
        for b in bad:
            print(b, file=sys.stderr)
        print(f"check-defaults: {len(bad)} problem(s) — every off-default lever pin needs a "
              f"defaults twin or a dated reason in {TSV}", file=sys.stderr)
        return 1
    print(f"check-defaults: ok ({len(pin_keys)} pinned lever values, each with a twin; "
          f"{gaps} `none` row(s) — open gaps and no-twin pins, each row's reason says which)")
    return 0


def run_list(root):
    pins, live = scan_tree(root)
    for lever, where, value, kind, site in sorted(pins):
        print(f"{lever}\t{where}\t{value}\t{kind}\t{site}")
    for lever, where, value, twin, reason, n in read_tsv(root, live):
        if twin == "none":
            print(f"{lever}\t{where}\t{value}\tnone\t-")
            continue
        p, needle = split_twin(twin)
        at = twin
        if p is not None:
            hits, line = resolve_twin(root, p, needle)
            if hits == 1:
                at = f"{p}:{line}"
            elif hits is None:
                at = f"{twin} (path gone)"
            else:
                at = f"{twin} ({hits} hits)"
        print(f"{lever}\t{where}\t{value}\ttwin\t{at}")
    ident = len({(p[0], p[1], p[2]) for p in pins})
    print(f"check-defaults: {len(pins)} pin sites, {ident} pin identities, {len(live)} live levers")
    return 0


# ---------------------------------------------------------------- self-test


_MINI_JUST = """\
# a comment mentioning BLOOMERY_MTP_WIDTH=fixed in prose
x-assign := 'BLOOMERY_MTP_WIDTH=fixed'

# the doc
gate-mini:
    BLOOMERY_MTP_WIDTH=fixed run --bin gate_mini --test gate_mini

weekly-mini *ARGS:
    # BLOOMERY_POISON=1 in a body comment
    BLOOMERY_MTP_WIDTH=fixed BLOOMERY_HOST_LOCK=${BLOOMERY_HOST_LOCK:-1} run {{ARGS}}
"""

# The fixture's made-up names, spelled in two pieces: tools/check-levers.sh holds
# every BLOOMERY_* name under tools/ to a registry row, and a self-test's made-up
# name is no row (its own fixtures spell theirs the same way).
_RETIRED = "BLOOMERY" + "_OLD"
_TYPO = "BLOOMERY" + "_WIDTH"

_MINI_REGISTRY = '''\
pub const MTP_WIDTH: &str = "BLOOMERY_MTP_WIDTH";
pub const HOST_LOCK: &str = "BLOOMERY_HOST_LOCK";
pub const QWEN3_KV: &str = "BLOOMERY_QWEN3_KV";

pub(crate) static REGISTRY: &[LeverSpec] = &[
    LeverSpec {
        name: MTP_WIDTH,
        class: Class::A,
        kind: Kind::Words(&["cost", "fixed"]),
        default: Unset::Is("cost"),
        doc: "mini",
        site: Site::Parsed { left: &[] },
    },
    LeverSpec {
        name: HOST_LOCK,
        class: Class::C,
        kind: Kind::Flag,
        default: Unset::Is("0"),
        doc: "mini",
        site: Site::Direct { at: &[] },
    },
    LeverSpec {
        name: QWEN3_KV,
        class: Class::A,
        kind: Kind::Words(&["f16", "q8_0"]),
        default: Unset::Is("f16"),
        doc: "mini",
        site: Site::Parsed { left: &[] },
    },
    LeverSpec {
        name: "@RETIRED@",
        class: Class::A,
        kind: Kind::Flag,
        default: Unset::Means("gone"),
        doc: "mini",
        site: Site::Retired { why: "gone", left: &[] },
    },
];
'''

_MINI_BIN = '''\
//! A mini gate bin.
const WORD: &str = "mid-p0-s1";
const TABLE: &[(&str, &str)] = &[
    ("BLOOMERY_HOST_LOCK", "1"),
    (bloomery_levers::MTP_WIDTH, "fixed"),
];
fn main() {
    let word = WORD;
    let mut cmd = std::process::Command::new("x");
    cmd.env(bloomery_levers::MTP_WIDTH, "fixed")
        .env(bloomery_levers::MTP_WIDTH, word)
        .env("@RETIRED@", "1")
        .envs(TABLE.iter().copied());
    unsafe { std::env::set_var("BLOOMERY_MTP_WIDTH", "cost"); }
    let a = ["--cache-type-k", "q8_0"];
}
'''

_MINI_TEST = '''\
//! A mini test gate.
#[test]
fn t() {
    let out = std::process::Command::new("self")
        .env("BLOOMERY_MTP_WIDTH", "fixed")
        .output();
}
'''

_MINI_TSV = "\n".join([
    "# lever\twhere\tvalue\ttwin\treason",
    "BLOOMERY_HOST_LOCK\tweekly-mini\t${BLOOMERY_HOST_LOCK:-1}\tmini/justfile::eleven\tthe recipe's own escape arm",
    "BLOOMERY_HOST_LOCK\tgate-mini\t1\tmini/bin.rs::three\tthe table's arm",
    "BLOOMERY_MTP_WIDTH\tgate-mini\tcost\tmini/bin.rs::fourteen\tset_var names the default value",
    "BLOOMERY_MTP_WIDTH\tgate-mini\tfixed\tmini/bin.rs::eleven\tthe pinned arm",
    "BLOOMERY_MTP_WIDTH\tgate-mini\tmid-p0-s1\tmini/bin.rs::three\tthe alias arm",
    "BLOOMERY_QWEN3_KV\tgate-mini\tq8_0\tmini/bin.rs::fifteen\tthe flag arm",
    "BLOOMERY_MTP_WIDTH\tweekly-mini\tfixed\tnone\t2026-10-08: fixture row for the dated-reason rule",
    "BLOOMERY_MTP_WIDTH\tgate-mini\tfixed\tmini/test.rs::five\tthe test's pinned arm",
]) + "\n"


def _fixture():
    root = tempfile.mkdtemp(prefix="check-defaults-")
    os.makedirs(os.path.join(root, "crates", "levers", "src"))
    os.makedirs(os.path.join(root, "crates", "gpu-gates", "src", "bin"))
    os.makedirs(os.path.join(root, "crates", "model", "tests"))
    os.makedirs(os.path.join(root, "tools"))
    os.makedirs(os.path.join(root, "mini"))
    just = os.path.join(root, "justfile")
    open(just, "w").write(_MINI_JUST)
    open(os.path.join(root, "crates", "levers", "src", "registry.rs"), "w").write(
        _MINI_REGISTRY.replace("@RETIRED@", _RETIRED))
    open(os.path.join(root, "crates", "gpu-gates", "src", "bin", "gate_mini.rs"), "w").write(
        _MINI_BIN.replace("@RETIRED@", _RETIRED))
    open(os.path.join(root, "crates", "model", "tests", "gate_mini.rs"), "w").write(_MINI_TEST)
    # the fixture names its twins at mini/*: numbered-word files stand in for the
    # clause files — a green twin's word is its line's own ("eleven" is line 11,
    # and a word is never a substring of another, as "four" is of "fourteen");
    # bin.rs carries "dup" on two lines (the more-than-once red) and nothing
    # holds "seventeen" (the zero-times red)
    open(os.path.join(root, "mini", "justfile"), "w").write("one\ntwo\nthree\nfour\nfive\nsix\nseven\n"
                                                            "eight\nnine\nten\neleven\ntwelve\n"
                                                            "thirteen\nfourteen\nfifteen\nsixteen\n")
    open(os.path.join(root, "mini", "bin.rs"), "w").write("one\ntwo\nthree\nfour\nfive\nsix\nseven\n"
                                                          "eight\nnine\nten\neleven\ntwelve\n"
                                                          "thirteen\nfourteen\nfifteen\nsixteen\ndup\ndup\n")
    open(os.path.join(root, "mini", "test.rs"), "w").write("one\ntwo\nthree\nfour\nfive\nsix\n")
    tsv = os.path.join(root, TSV)
    open(tsv, "w").write(_MINI_TSV)
    return root, just, tsv


def self_test():
    fails = []
    root, just, tsv = _fixture()
    saved_just, saved_tsv = open(just).read(), open(tsv).read()

    def rerun():
        try:
            return check(root)
        except InputError as e:
            fails.append(f"refused: {e}")
            return 65

    if rerun() != 0:
        fails.append("the green fixture is red")
    open(just, "w").write(saved_just + "\nred-mid:\n    BLOOMERY_MTP_WIDTH=fixed run\n")
    out = rerun()
    if out != 1:
        fails.append(f"a pin with no row ends {out}, not 1")
    open(just, "w").write(saved_just)
    rows = saved_tsv.rstrip("\n").split("\n")
    open(tsv, "w").write("\n".join(r for r in rows if "fixture row" not in r) + "\n")
    if rerun() != 1:
        fails.append("a deleted row (its pin stands) is not red")
    # the needle rule's three cases: one occurrence green (the fixture above),
    # zero red (the clause moved or is gone), more than once red (no one clause)
    open(tsv, "w").write(saved_tsv.replace("mini/bin.rs::three\tthe table's arm",
                                           "mini/bin.rs::seventeen\tthe table's arm"))
    if rerun() != 1:
        fails.append("a twin whose needle occurs zero times in its file is not red")
    open(tsv, "w").write(saved_tsv.replace("mini/bin.rs::three\tthe table's arm",
                                           "mini/bin.rs::dup\tthe table's arm"))
    if rerun() != 1:
        fails.append("a twin whose needle occurs more than once in its file is not red")
    open(tsv, "w").write(saved_tsv.replace("mini/bin.rs::three\tthe table's arm",
                                           "mini/bin.rs:3\tthe table's arm"))
    if rerun() != 1:
        fails.append("a `path:line` twin (the old line-number form) is not red")
    open(tsv, "w").write(saved_tsv.replace("mini/bin.rs::three\tthe table's arm",
                                           "mini/gone.rs::one\tthe table's arm"))
    if rerun() != 1:
        fails.append("a twin whose path is gone is not red")
    open(tsv, "w").write(saved_tsv.replace("2026-10-08: fixture row for the dated-reason rule",
                                           "no date here"))
    if rerun() != 1:
        fails.append("an undated none reason is not red")
    open(tsv, "w").write(f"{saved_tsv}{_RETIRED}\tgate-mini\t1\tnone\t2026-10-08: retired\n")
    try:
        check(root)
        fails.append("a retired name as a row is not refused")
    except InputError:
        pass
    open(tsv, "w").write(f"{saved_tsv}{_TYPO}\tgate-mini\t1\tnone\t2026-10-08: typo\n")
    try:
        check(root)
        fails.append("an unknown lever name is not refused")
    except InputError:
        pass
    open(just, "w").write(saved_just)
    open(tsv, "w").write(saved_tsv)

    # the readers: a recipe header with a `:=` is no recipe; comments carry no pin
    live, reg_consts = live_levers(root)
    pins = scan_justfile(root, live)
    got = {(p[0], p[1], p[2]) for p in pins}
    want = {("BLOOMERY_MTP_WIDTH", "gate-mini", "fixed"),
            ("BLOOMERY_MTP_WIDTH", "weekly-mini", "fixed"),
            ("BLOOMERY_HOST_LOCK", "weekly-mini", "${BLOOMERY_HOST_LOCK:-1}")}
    if got != want:
        fails.append(f"justfile pins: {sorted(got)}, expected {sorted(want)}")
    # the rust side: env, set_var, table pairs, aliases, flags; a retired name is no pin
    rel = "crates/gpu-gates/src/bin/gate_mini.rs"
    got = {(p[0], p[2], p[3]) for p in scan_rust(rel, open(os.path.join(root, rel)).read(),
                                                 live, reg_consts)}
    want = {("BLOOMERY_MTP_WIDTH", "fixed", "env"), ("BLOOMERY_MTP_WIDTH", "mid-p0-s1", "env"),
            ("BLOOMERY_MTP_WIDTH", "cost", "set"), ("BLOOMERY_HOST_LOCK", "1", "table"),
            ("BLOOMERY_MTP_WIDTH", "fixed", "table"), ("BLOOMERY_QWEN3_KV", "q8_0", "flag")}
    if got != want:
        fails.append(f"rust pins: {sorted(got)}, expected {sorted(want)}")
    # the twin resolver: one hit gives the line, zero and more-than-one and a
    # missing file do not
    got = (resolve_twin(root, "mini/bin.rs", "three"), resolve_twin(root, "mini/bin.rs", "seventeen"),
           resolve_twin(root, "mini/bin.rs", "dup"), resolve_twin(root, "mini/gone.rs", "one"))
    if got != ((1, 3), (0, None), (2, None), (None, None)):
        fails.append(f"resolve_twin: {got}")
    # a gate file no recipe names is a named error
    open(os.path.join(root, "crates", "model", "tests", "orphan.rs"), "w").write(_MINI_TEST)
    try:
        scan_tree(root)
        fails.append("a pin in a file no recipe names is not refused")
    except InputError:
        pass
    shutil.rmtree(root)

    for f in fails:
        print(f"check-defaults self-test FAIL: {f}", file=sys.stderr)
    if fails:
        print(f"check-defaults: self-test {len(fails)} failed", file=sys.stderr)
        return 1
    print("check-defaults: self-test ok")
    return 0


def main(argv):
    root = os.getcwd()
    try:
        if argv[:1] == ["--self-test"]:
            if len(argv) != 1:
                print("check-defaults: --self-test takes nothing", file=sys.stderr)
                return 64
            return self_test()
        if argv and argv[0] in ("check", "list"):
            rest = argv[1:]
            if rest[:1] == ["--root"]:
                if len(rest) < 2:
                    print("check-defaults: --root takes a directory", file=sys.stderr)
                    return 64
                root = rest[1]
                rest = rest[2:]
            if rest:
                print(__doc__, file=sys.stderr)
                return 64
            return check(root) if argv[0] == "check" else run_list(root)
        print(__doc__, file=sys.stderr)
        return 64
    except InputError as e:
        print(f"check-defaults: {e}", file=sys.stderr)
        return 65


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
