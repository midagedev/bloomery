#!/usr/bin/env python3
"""The V4.1 binaries' record lines, read by kind and field name.

Every line generate_ds41, bloomery-chat, bloomery-serve-ds41 and the prefill gate write is a record of
a kind crates/gpu-gates/src/record.rs declares: a head (fixed words), then its parts in order —
` name=value`, a bare value, a flag word, literal text. `<bin> --records-schema` prints the kinds with
their fields, types and units; tools/bloomery/schema/<bin>.jsonl is that output, checked in, and
record.rs's `checked_in_schemas_are_current` holds it to the binary's (`just records-refresh` rewrites
it). A reader names a kind and a field; the syntax is this file's and record.rs's alone.

  records.py parse [--bin B] FILE...             each record as `<kind>\\t<its fields as JSON>`, in order
  records.py lines [--bin B] FILE KIND...        the lines of those kinds, as printed, in order
  records.py sh [--bin B] FILE VAR=KIND.FIELD... shell assignments, quoted: the field as printed, of the
                                                 first record of KIND (`a|b` takes either kind); with
                                                 `.FIELD*` every record's, one a line
  records.py tail [--bin B] FILE KIND...         the lines of those kinds after the file's last record of any
                                                 other kind, as printed, in order: what a process prints
                                                 between one unit of its output and the next
  records.py check [--bin B] FILE...             the lines a kind's head opens that no kind reads whole:
                                                 read loose, or not at all (exit 1 when there is one)
  records.py --self-test                         every kind's line read whole, and `check`'s passes
  records.py refresh                             `just records-refresh`'s stdin into the tree: each
                                                 `--records-schema` output into schema/<bin>.jsonl, and
                                                 after a `#> <path> <what>` line the generate_ds41
                                                 records that follow into <path> (from the repository
                                                 root) under a `# just records-refresh: <what>` line;
                                                 every other line skipped; a marker no record follows
                                                 (or a schema that miscounts its kinds) refuses the
                                                 whole stdin before any file is written

A line is matched by its head, the longest first (`stat prefill split` before `stat prefill`), then by
its kind's whole pattern; kinds that share a head are told apart by the pattern (`load` and `load
draft=`). A line of an older text that its kind's pattern no longer matches — a field added or
dropped since — is read loose when it opens with the kind's first field, or with any of its fields
when no other kind shares its head: its `name=value` pairs by name, `loose` set, the bare values left
out, a name the kind does not have kept in `extra`. A line of no kind (a library's `load host_tier`,
a runner's own) is not a record; `check` passes over the ones that open a kind's head (NOT_RECORDS).

Exit status: 0; 1 `check` found a line; 2 a named refusal (a schema file missing or of another
version); 64 a usage error.
"""
import contextlib
import io
import json
import re
import shlex
import sys
from pathlib import Path

SCHEMA_DIR = Path(__file__).resolve().parent / "schema"
VERSION = 1
DEFAULT_BIN = "generate_ds41"

# A value as the binaries write it, by the schema's type.
VALUE = {
    "u64": r"\d+",
    "i64": r"-?\d+",
    "bool": r"true|false",
    "word": r"\S+",
    "text": r".+?",
    "list": r"\[[^\]]*\]",
    "csv": r"\[[^\]]*\]",
}
FLOAT = r"-?(?:\d+(?:\.\d+)?|inf|NaN)"
# Lines the binaries print under a kind's head that are no record: crates/model's host tier says how it
# reads its experts (r8file.rs, moe.rs) with `load host_tier …` on stderr, outside record.rs.
NOT_RECORDS = ("load host_tier ",)
# A `name=` in a loose read: a field name opens with a letter or `_` (`tok/s(p50)` is one), so the
# `(BLOOMERY_CED=` inside a text value is not.
LOOSE_KEY = re.compile(r" ([A-Za-z_][\w/()]*)=")
LIST_AT = re.compile(r"\[[^\]]*\]")


def refuse(msg):
    print(f"records.py: {msg}", file=sys.stderr)
    raise SystemExit(2)


def value_re(ty):
    return FLOAT if ty.startswith("f64") else VALUE[ty]


def convert(ty, text):
    """`text` as the schema's type: an int, a float, a bool, a list (of ints when every item is
    one), or the text itself."""
    if ty in ("u64", "i64"):
        return int(text)
    if ty.startswith("f64"):
        return float(text)
    if ty == "bool":
        if text not in ("true", "false"):
            raise ValueError(f"{text!r} is not a bool")
        return text == "true"
    if ty in ("list", "csv"):
        inner = text.strip()[1:-1].strip()
        items = [x.strip() for x in inner.split(",")] if inner else []
        try:
            return [int(x) for x in items]
        except ValueError:
            return items
    return text


class Rec:
    """One record: its kind's name, its fields (typed) and as printed (`raw`), the line, whether it was
    read loose, and in a loose read the names its kind does not have (`extra`, as printed)."""
    __slots__ = ("kind", "fields", "raw", "line", "loose", "extra")

    def __init__(self, kind, fields, raw, line, loose=False, extra=None):
        self.kind, self.fields, self.raw, self.line = kind, fields, raw, line
        self.loose, self.extra = loose, extra or {}

    def __getitem__(self, name):
        return self.fields[name]

    def __contains__(self, name):
        return name in self.fields

    def get(self, name, default=None):
        return self.fields.get(name, default)


class Kind:
    def __init__(self, obj):
        self.name, self.head, self.doc = obj["kind"], obj["head"], obj.get("doc", "")
        self.parts = obj["parts"]
        self.values = {}           # value name -> its part
        self.groups = []           # (group, value name, type or "flag")
        pat = [re.escape(self.head)]
        for i, p in enumerate(self.parts):
            g = f"g{i}"
            if "key" in p or "pos" in p:
                name = p.get("key", p.get("pos"))
                v = f"(?P<{g}>{value_re(p['ty'])})"
                s = f" {re.escape(name)}={v}" if "key" in p else v
                if p.get("opt"):
                    s = f"(?:{s})?"
                self.groups.append((g, name, p["ty"]))
                self.values[name] = p
            elif "flag" in p:
                s = f"(?P<{g}> {re.escape(p['flag'])})?"
                self.groups.append((g, p["flag"], "flag"))
                self.values[p["flag"]] = p
            else:
                s = re.escape(p["lit"])
            pat.append(s)
        self.regex = re.compile("".join(pat) + "$")
        first = next((p for p in self.parts if "lit" not in p), None)
        self.first_key = first["key"] if first is not None and "key" in first else None

    def opens(self, line):
        """Whether `line` opens with this kind's head, as a word: followed by a space or the line's
        end, unless the head ends in `=` or `:`."""
        if not line.startswith(self.head):
            return False
        rest = line[len(self.head):]
        return rest == "" or rest[0] == " " or self.head[-1] in "=:"

    def read(self, line):
        m = self.regex.match(line)
        if not m:
            return None
        fields, raw = {}, {}
        for g, name, ty in self.groups:
            text = m.group(g)
            if ty == "flag":
                fields[name] = text is not None
            elif text is not None:
                raw[name] = text
                fields[name] = convert(ty, text)
        return Rec(self.name, fields, raw, line)

    def read_loose(self, line, alone):
        """`line` by its `name=value` pairs, when it opens with this kind's first field — or, `alone`
        (no other kind has this head), with any of its named fields."""
        rest = line[len(self.head):]
        at = LOOSE_KEY.match(rest)
        if self.first_key is None or at is None:
            return None
        if at.group(1) != self.first_key and not (alone and "key" in self.values.get(at.group(1), {})):
            return None
        keys = list(LOOSE_KEY.finditer(rest))
        fields, raw, extra = {}, {}, {}
        for j, m in enumerate(keys):
            end = keys[j + 1].start() if j + 1 < len(keys) else len(rest)
            span = rest[m.end():end]
            p = self.values.get(m.group(1))
            ty = p["ty"] if p is not None and "key" in p else None
            if ty == "text":
                text = span.strip()
            elif ty in ("list", "csv"):
                at = LIST_AT.match(span)
                text = at.group(0) if at else ""
            else:
                text = span.split()[0] if span.split() else ""
            if ty is None:
                extra[m.group(1)] = text
                continue
            raw[m.group(1)] = text
            try:
                fields[m.group(1)] = convert(ty, text)
            except ValueError:
                fields[m.group(1)] = text
        return Rec(self.name, fields, raw, line, loose=True, extra=extra)


class Schema:
    def __init__(self, bin_name, kinds):
        self.bin, self.kinds = bin_name, kinds
        self.by_name = {k.name: k for k in kinds}
        self.order = sorted(kinds, key=lambda k: -len(k.head))

    def candidates(self, line):
        return [k for k in self.order if k.opens(line)]

    def read_line(self, line):
        line = line.rstrip("\n")
        cands = self.candidates(line)
        for k in cands:
            r = k.read(line)
            if r is not None:
                return r
        for k in cands:
            r = k.read_loose(line, sum(1 for c in cands if c.head == k.head) == 1)
            if r is not None:
                return r
        return None


def load(bin_name=DEFAULT_BIN, path=None):
    """The schema `bin_name` prints, from its checked-in file (or `path`)."""
    path = Path(path) if path else SCHEMA_DIR / f"{bin_name}.jsonl"
    try:
        lines = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    except OSError as e:
        refuse(f"no schema for {bin_name}: {e} (just records-refresh writes it)")
    head, kinds = lines[0], lines[1:]
    if head.get("records") != VERSION:
        refuse(f"{path}: schema version {head.get('records')}, this reader reads {VERSION}")
    if head.get("kinds") != len(kinds):
        refuse(f"{path}: the header counts {head.get('kinds')} kinds, the file holds {len(kinds)}")
    return Schema(head.get("bin"), [Kind(k) for k in kinds])


def read(source, bin_name=DEFAULT_BIN, schema=None):
    """Every record of `source` — a path, `-` for stdin, or the text's lines — in order."""
    schema = schema or load(bin_name)
    if isinstance(source, (str, Path)):
        text = sys.stdin.read() if str(source) == "-" else Path(source).read_text(errors="replace")
        source = text.splitlines()
    out = []
    for line in source:
        r = schema.read_line(line)
        if r is not None:
            out.append(r)
    return out


def of_kind(recs, *kinds):
    return [r for r in recs if r.kind in kinds]


def first(recs, *kinds):
    return next((r for r in recs if r.kind in kinds), None)


def tail(recs, *kinds):
    """The records of `kinds` after the last record of any other kind, in order."""
    out = []
    for r in recs:
        out = out + [r] if r.kind in kinds else []
    return out


# ---- the command line ----

def take_bin(argv):
    if len(argv) >= 2 and argv[0] == "--bin":
        return argv[1], argv[2:]
    return DEFAULT_BIN, argv


def cmd_parse(argv):
    bin_name, files = take_bin(argv)
    if not files:
        usage()
    schema = load(bin_name)
    for f in files:
        for r in read(f, schema=schema):
            obj = dict(r.fields)
            if r.loose:
                obj["_loose"] = True
            if r.extra:
                obj["_extra"] = r.extra
            print(f"{r.kind}\t{json.dumps(obj, sort_keys=False)}")
    return 0


def cmd_lines(argv):
    bin_name, rest = take_bin(argv)
    if len(rest) < 2:
        usage()
    kinds = set(rest[1:])
    schema = load(bin_name)
    unknown = kinds - set(schema.by_name)
    if unknown:
        refuse(f"{bin_name} prints no kind {', '.join(sorted(unknown))}")
    for r in read(rest[0], schema=schema):
        if r.kind in kinds:
            print(r.line)
    return 0


def cmd_tail(argv):
    bin_name, rest = take_bin(argv)
    if len(rest) < 2:
        usage()
    kinds = rest[1:]
    schema = load(bin_name)
    unknown = set(kinds) - set(schema.by_name)
    if unknown:
        refuse(f"{bin_name} prints no kind {', '.join(sorted(unknown))}")
    for r in tail(read(rest[0], schema=schema), *kinds):
        print(r.line)
    return 0


def cmd_sh(argv):
    bin_name, rest = take_bin(argv)
    if len(rest) < 2:
        usage()
    schema = load(bin_name)
    recs = read(rest[0], schema=schema)
    for spec in rest[1:]:
        var, _, want = spec.partition("=")
        kinds, _, field = want.rpartition(".")
        every = field.endswith("*")
        field = field.rstrip("*")
        names = kinds.split("|")
        for k in names:
            if k not in schema.by_name:
                refuse(f"{bin_name} prints no kind {k}")
            if field not in schema.by_name[k].values:
                refuse(f"kind {k} has no field {field}")
        if not re.fullmatch(r"[A-Za-z_]\w*", var):
            refuse(f"{var!r} is not a shell variable name")
        vals = [r.raw[field] for r in recs if r.kind in names and field in r.raw]
        if schema.by_name[names[0]].values[field].get("flag"):
            vals = ["1" if r.fields.get(field) else "" for r in recs if r.kind in names]
        text = "\n".join(vals) if every else (vals[0] if vals else "")
        print(f"{var}={shlex.quote(text)}")
    return 0


def check_lines(schema, lines):
    """(line number, what, line) of each line a kind's head opens that no kind reads whole; a line of
    NOT_RECORDS is no record and is passed over."""
    out = []
    for n, line in enumerate(lines, 1):
        if not schema.candidates(line) or line.startswith(NOT_RECORDS):
            continue
        r = schema.read_line(line)
        if r is None or r.loose:
            out.append((n, "no kind reads it" if r is None else f"read loose as {r.kind}", line))
    return out


def cmd_check(argv):
    bin_name, files = take_bin(argv)
    if not files:
        usage()
    schema = load(bin_name)
    found = 0
    for f in files:
        for n, what, line in check_lines(schema, Path(f).read_text(errors="replace").splitlines()):
            found += 1
            print(f"{f}:{n}: {what}: {line}")
    return 1 if found else 0


def refresh_writes(lines, gen, root):
    """`refresh`'s files from `lines` as (path, text, note), every one checked before any is written:
    stdin with no schema, a schema whose header miscounts its kinds and a marker no record follows
    refuse the whole refresh, each named, so a red refresh leaves the tree as it was."""
    schemas, files, cur = {}, {}, None
    for line in lines:
        line = line.rstrip("\n")
        if line.startswith('{"records":'):
            cur = ("schema", json.loads(line)["bin"])
            schemas[cur[1]] = [line]
        elif line.startswith('{"kind":') and cur is not None and cur[0] == "schema":
            schemas[cur[1]].append(line)
        elif line.startswith("#> "):
            path, _, what = line[3:].partition(" ")
            cur = ("file", path)
            files[path] = [f"# just records-refresh: {what}"]
        elif cur is not None and cur[0] == "file" and gen.read_line(line) is not None:
            files[cur[1]].append(line)
    problems = [] if schemas else ["no --records-schema output on stdin"]
    for name, got in schemas.items():
        head = json.loads(got[0])
        if head["kinds"] != len(got) - 1:
            problems.append(f"{name}: the header counts {head['kinds']} kinds, stdin held {len(got) - 1}")
    problems += [f"{rel}: no record followed its marker" for rel, got in files.items() if len(got) < 2]
    if problems:
        refuse(f"nothing written: {'; '.join(problems)}")
    writes = []
    for name, got in schemas.items():
        path = SCHEMA_DIR / f"{name}.jsonl"
        writes.append((path, "\n".join(got) + "\n", f"{path.relative_to(root)}: {len(got) - 1} kinds"))
    for rel, got in files.items():
        writes.append((root / rel, "\n".join(got) + "\n", f"{rel}: {len(got) - 1} records"))
    return writes


def cmd_refresh(argv):
    if argv:
        usage()
    root = SCHEMA_DIR.parent.parent.parent
    for path, text, note in refresh_writes(sys.stdin, load(DEFAULT_BIN), root):
        path.write_text(text)
        print(note)
    return 0


def usage():
    print(__doc__.split("\n\n")[1], file=sys.stderr)
    raise SystemExit(64)


SAMPLE = {"u64": "1", "i64": "-1", "bool": "true", "word": "w", "text": "t", "list": "[1,2]", "csv": "[a,b]"}


def sample_line(kind):
    """A line of `kind` with every part present: a value of each field's type, each flag set."""
    out = [kind.head]
    for p in kind.parts:
        if "key" in p:
            out.append(f" {p['key']}={'1.5' if p['ty'].startswith('f64') else SAMPLE[p['ty']]}")
        elif "pos" in p:
            out.append("1.5" if p["ty"].startswith("f64") else SAMPLE[p["ty"]])
        elif "flag" in p:
            out.append(f" {p['flag']}")
        else:
            out.append(p["lit"])
    return "".join(out)


def self_test():
    schema = load(DEFAULT_BIN)
    for k in schema.kinds:
        line = sample_line(k)
        r = schema.read_line(line)
        assert r is not None and r.kind == k.name and not r.loose, (k.name, line, r and r.kind)
        assert check_lines(schema, [line]) == [], (k.name, line)
    # the host tier's own lines open the `load` head and are no record: check passes them over
    tier = ["load host_tier r8=on (/models/m-r8/m-r8.gguf)", "load host_tier r8=off (BLOOMERY_R8=off)",
            "load host_tier type=iq3_xxs k=4096 path=fused"]
    assert all(schema.read_line(x) is None for x in tier), tier
    assert check_lines(schema, tier) == [], check_lines(schema, tier)
    # a `load` record missing its fields is still found, and so is a head no kind reads whole
    found = check_lines(schema, ["load resident_bytes=1", "load host_tierx=1"])
    assert [w for _, w, _ in found] == ["read loose as load", "no kind reads it"], found
    # tail: the run of a kind after the last record of another, not the same kind earlier in the text
    reset = sample_line(schema.by_name["residency_reset"])
    step = sample_line(schema.by_name["time_step"])
    recs = read([reset, step, "a line of no kind", reset, reset], schema=schema)
    assert [r.line for r in tail(recs, "residency_reset")] == [reset, reset], recs
    assert tail(read([reset, step], schema=schema), "residency_reset") == []
    # refresh checks every file before it writes one: a marker no record follows refuses the whole
    # stdin by name, the schema and the earlier marker's file included, and nothing is returned
    root = SCHEMA_DIR.parent.parent.parent
    plan = ["#> tools/flow/plans/a.rec generate_ds41 --plan --depth 1", reset, "a line of no kind"]
    good = ['{"records":1,"bin":"generate_ds41","kinds":1}', '{"kind":"residency_reset"}'] + plan
    writes = refresh_writes(good, schema, root)
    assert [(str(p.relative_to(root)), n) for p, _, n in writes] == [
        ("tools/bloomery/schema/generate_ds41.jsonl", "tools/bloomery/schema/generate_ds41.jsonl: 1 kinds"),
        ("tools/flow/plans/a.rec", "tools/flow/plans/a.rec: 1 records"),
    ], writes
    assert writes[1][1] == f"# just records-refresh: generate_ds41 --plan --depth 1\n{reset}\n", writes[1][1]
    bad = good + ["#> tools/flow/plans/b.rec generate_ds41 --plan --depth 2", "a line of no kind"]
    said = io.StringIO()
    try:
        with contextlib.redirect_stderr(said):
            refresh_writes(bad, schema, root)
        raise AssertionError("a marker with no record was written")
    except SystemExit as e:
        assert e.code == 2, e.code
    assert said.getvalue() == (
        "records.py: nothing written: tools/flow/plans/b.rec: no record followed its marker\n"
    ), said.getvalue()
    print(f"records: self-test ok ({len(schema.kinds)} kinds)")
    return 0


def main(argv):
    if argv == ["--self-test"]:
        return self_test()
    cmds = {"parse": cmd_parse, "lines": cmd_lines, "tail": cmd_tail, "sh": cmd_sh, "check": cmd_check,
            "refresh": cmd_refresh}
    if not argv or argv[0] not in cmds:
        usage()
    return cmds[argv[0]](argv[1:])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
