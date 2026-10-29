#!/usr/bin/env python3
"""The model catalog's committed print, read: `tools/catalog.tsv`, the rows of `hf::catalog` (crates/hf/src/catalog.rs,
`print`; a test of that crate holds the file equal to the print). One reader for the Python tools that look at the
catalog, as tools/bloomery/manifest.py is for the reference sets: tools/check-catalog-docs.py and
tools/hf-arch-check.py take their rows from here, never from a copy of their own.

tools/ has no packages: a script imports this file by path, as tools/hf-arch-check.py does tools/bloomery/hf.py.

    read(text, where)       the rows, as Row, in the file's order; an unreadable file is an Invalid naming file:line
    models(rows)            {model name: [its rows]}, in the file's order
    default_model(rows, seat) the one model a seat word names (its alias); anything else is an Invalid

A Row is one file a model can start from: `name` (the catalog key), `aliases`, `seat` (the `--model` word of the seat
that serves it), `arch`, `variant`, `format`, `hf` (the `--hf` value: `owner/name:variant` for a GGUF set, `owner/name`
for a checkpoint folder), `revision`, `bytes` (an int, or None where the catalog records none), `verified`, `note`, and
`line` (its line in the file). `repo` and `quant` split `hf`; `quant` is None for a folder.

    tools/bloomery/catalog.py --self-test
"""
import collections
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CATALOG_TSV = "tools/catalog.tsv"
HEADER = ["name", "aliases", "seat", "arch", "variant", "format", "hf", "revision", "bytes", "verified", "note"]


class Invalid(Exception):
    """A catalog file this reader cannot take: the message names the file and the line."""


class Row(collections.namedtuple("Row", HEADER + ["line"])):
    __slots__ = ()

    @property
    def repo(self):
        return self.hf.partition(":")[0]

    @property
    def quant(self):
        return self.hf.partition(":")[2] or None


def read(text, where=CATALOG_TSV):
    """The rows of the catalog print `text`; `where` names it in a refusal."""
    rows, header_seen, seen = [], False, set()
    for no, line in enumerate(text.splitlines(), 1):
        if not line or line.startswith("#"):
            continue
        cols = line.split("\t")
        if not header_seen:
            if cols != HEADER:
                raise Invalid(f"{where}:{no}: the header is {cols}, not {HEADER}")
            header_seen = True
            continue
        if len(cols) != len(HEADER):
            raise Invalid(f"{where}:{no}: {len(cols)} columns, want {len(HEADER)} ({' '.join(HEADER)})")
        c = dict(zip(HEADER, cols))
        if c["verified"] not in ("true", "false"):
            raise Invalid(f"{where}:{no}: verified is {c['verified']!r}, not true or false")
        if c["bytes"] != "-" and not c["bytes"].isdigit():
            raise Invalid(f"{where}:{no}: bytes is {c['bytes']!r}, not a decimal count or `-`")
        for key in ("name", "seat", "arch", "variant", "format", "hf", "revision"):
            if not c[key] or c[key] == "-":
                raise Invalid(f"{where}:{no}: {key} is empty")
        if (c["name"], c["variant"].lower()) in seen:
            raise Invalid(f"{where}:{no}: {c['name']} lists the variant {c['variant']} twice")
        seen.add((c["name"], c["variant"].lower()))
        rows.append(Row(
            name=c["name"], aliases=tuple(a for a in c["aliases"].split(",") if a and a != "-"), seat=c["seat"],
            arch=c["arch"], variant=c["variant"], format=c["format"], hf=c["hf"], revision=c["revision"],
            bytes=None if c["bytes"] == "-" else int(c["bytes"]), verified=c["verified"] == "true", note=c["note"],
            line=no))
    if not header_seen:
        raise Invalid(f"{where}: no header line")
    if not rows:
        raise Invalid(f"{where}: the catalog holds no row")
    return rows


def read_file(root=ROOT):
    """The rows of `tools/catalog.tsv` under `root`."""
    path = os.path.join(root, CATALOG_TSV)
    try:
        with open(path, encoding="utf-8") as f:
            return read(f.read(), CATALOG_TSV)
    except OSError as e:
        raise Invalid(f"{CATALOG_TSV}: {e.strerror}")


def models(rows):
    """{model name: [its rows]}, in the file's order; a model's rows share a seat and an architecture."""
    out = collections.OrderedDict()
    for r in rows:
        out.setdefault(r.name, []).append(r)
    for name, rs in out.items():
        if len({(r.seat, r.arch, r.aliases) for r in rs}) != 1:
            raise Invalid(f"{CATALOG_TSV}: the rows of {name} disagree on seat, architecture or aliases")
    return out


def default_model(rows, seat):
    """The name of the model `seat` (a seat word) names: the one whose aliases hold it. A seat with none, or two,
    is an Invalid."""
    names = [n for n, rs in models(rows).items() if seat in rs[0].aliases]
    if len(names) != 1:
        raise Invalid(f"{CATALOG_TSV}: the seat word {seat} is the alias of {len(names)} models {names}, not one")
    return names[0]


def self_test():
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        print(("ok " if cond else "FAIL ") + name + ("" if cond else f": {detail}"))
        ok = ok and bool(cond)

    def refused(text, needle):
        try:
            read(text)
        except Invalid as e:
            return needle in str(e), str(e)
        return False, "no refusal"

    head = "\t".join(HEADER)
    good_row = "m\ta,b\ts\tarch\tQ4_K_M\tgguf4\to/r:Q4_K_M\t" + "0" * 40 + "\t12\ttrue\tnote"
    rows = read(f"# print\n{head}\n{good_row}\n")
    r = rows[0]
    check("a row reads its columns", (r.name, r.aliases, r.seat, r.variant, r.bytes, r.verified, r.line) == (
        "m", ("a", "b"), "s", "Q4_K_M", 12, True, 3), str(r))
    check("a GGUF row splits its repo and quant", (r.repo, r.quant) == ("o/r", "Q4_K_M"), str((r.repo, r.quant)))
    folder = read(f"{head}\n" + good_row.replace("o/r:Q4_K_M", "o/f").replace("\ta,b\t", "\t-\t").replace("\t12\t", "\t-\t") + "\n")[0]
    check("a folder row has no quant, no aliases and no bytes", (folder.repo, folder.quant, folder.aliases, folder.bytes) == (
        "o/f", None, (), None), str(folder))
    for name, text, needle in (
        ("a wrong header is refused", good_row + "\n", "the header is"),
        ("a row of ten columns is refused", f"{head}\n" + "\t".join(good_row.split("\t")[:10]) + "\n", "10 columns"),
        ("verified other than true or false is refused", f"{head}\n" + good_row.replace("true", "yes") + "\n", "not true or false"),
        ("bytes that are no count are refused", f"{head}\n" + good_row.replace("\t12\t", "\t1.5\t") + "\n", "not a decimal count"),
        ("an empty variant is refused", f"{head}\n" + good_row.replace("\tQ4_K_M\tgguf4", "\t\tgguf4") + "\n", "variant is empty"),
        ("a variant listed twice is refused", f"{head}\n{good_row}\n{good_row}\n", "twice"),
        ("a file with no row is refused", f"{head}\n", "holds no row"),
        ("a file with no header is refused", "# nothing\n", "no header"),
    ):
        got = refused(text, needle)
        check(name, got[0], got[1])
    two = read(f"{head}\n{good_row}\n" + good_row.replace("Q4_K_M", "Q3_K_M").replace("m\ta,b", "n\ta,b") + "\n")
    try:
        default_model(two, "a")
        check("a seat word that names two models is refused", False, "no refusal")
    except Invalid as e:
        check("a seat word that names two models is refused", "alias of 2 models" in str(e), str(e))
    check("a seat word that is one model's alias names it", default_model(rows, "b") == "m")
    try:
        models(read(f"{head}\n{good_row}\n" + good_row.replace("Q4_K_M", "Q3_K_M").replace("\ts\t", "\tt\t") + "\n"))
        check("a model whose rows disagree on the seat is refused", False, "no refusal")
    except Invalid as e:
        check("a model whose rows disagree on the seat is refused", "disagree" in str(e), str(e))
    try:
        real = read_file()
        names = models(real)
        check("the committed catalog reads and every model has rows", len(real) >= 1 and all(names.values()), str(len(real)))
        check("every seat of the catalog has one default model",
              all(default_model(real, s) for s in {r.seat for r in real}))
    except Invalid as e:
        check("the committed catalog reads", False, str(e))
    print("catalog.py self-test: " + ("PASS" if ok else "FAIL"))
    return ok


def main(argv):
    if argv == ["--self-test"]:
        return 0 if self_test() else 1
    print("usage: catalog.py --self-test", file=sys.stderr)
    return 64


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
