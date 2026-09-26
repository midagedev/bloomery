"""The one reader of a reference set's MANIFEST.tsv for the Python tools; the Rust owner of the
same sets is crates/refset.

A manifest holds `# key<TAB>value` header lines and data rows, the row's kind first. Every data row
is read by the column names its kind's column line gives — `# kind<TAB>name<TAB>...` for `tensor`
rows, `# input<TAB>...`, `# int<TAB>...` (tools/ref/dump_ref.cpp, dump_draft.cpp), `# draft`,
`# verify`, `# plain` (dump_draft.cpp), `# layer<TAB>...` and `# call<TAB>...`
(tools/ref/router_trace.cpp) — never by position. `skip` and `skip-input` rows (the dumpers'
nodes written to no file) have no column line: five fields, and a draft set's four more (block, row,
accepted, graph). A header key given twice keeps its
first value. A data row of a kind with no column line before it, a row of another width than its
line, a second column line of a kind or a second `# complete`, and a field a caller asks for that
the line does not name or that does not parse are a ManifestError — a ValueError — naming the file
and the line.

tools/ has no packages: a script imports this file by path,

    _spec = importlib.util.spec_from_file_location(
        "manifest", os.path.join(<the repository root>, "tools", "bloomery", "manifest.py"))
    manifest = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(manifest)

and `python3 tools/bloomery/manifest.py --self-test` checks the reader.
"""
import os
import sys
import tempfile

# A column line's key and the kind of the rows it names.
COLUMN_KEYS = {"kind": "tensor", "input": "input", "int": "int", "draft": "draft",
               "verify": "verify", "plain": "plain", "layer": "layer", "call": "call"}

# The rows with no column line: the dumpers' skipped nodes, and dump_draft's with the four columns that
# place every row of a draft set in a block.
SKIP_COLUMNS = ["kind", "name", "occurrence", "type", "reason"]
DRAFT_SKIP_COLUMNS = SKIP_COLUMNS + ["block", "row", "accepted", "graph"]


class ManifestError(ValueError):
    pass


class Row:
    """One data row, read by its kind's column names. Field 0 is the kind; a name is looked up
    past it, so a line that names field 1 like its kind (router_trace's `# layer<TAB>layer...`)
    reads the layer."""

    __slots__ = ("kind", "line", "_names", "_fields", "_at")

    def __init__(self, names, fields, at, line):
        self.kind = fields[0]
        self.line = line
        self._names = names
        self._fields = fields
        self._at = at

    def has(self, name):
        return name in self._names[1:]

    def __getitem__(self, name):
        try:
            i = self._names.index(name, 1)
        except ValueError:
            raise ManifestError(f"{self._at}: the {self.kind} column line names no {name} field") from None
        return self._fields[i]

    def get(self, name, default=None):
        return self[name] if self.has(name) else default

    def int(self, name):
        v = self[name]
        try:
            return int(v)
        except ValueError:
            raise ManifestError(f"{self._at}: {name} {v!r} is not an integer") from None


class Manifest:
    """A set's MANIFEST.tsv: its first line if it is a title, its header, the `# complete`
    trailer's value (None when the dump that wrote the set did not finish), and its rows by
    kind, in file order."""

    def __init__(self, path):
        self.path = path
        self.dir = os.path.dirname(path)
        self.title = None
        self.header = {}
        self.complete = None
        self.columns = {}
        self._rows = {}

    def rows(self, kind):
        return self._rows.get(kind, [])


def read(path):
    """The manifest at `path`, a set directory or its MANIFEST.tsv."""
    if os.path.isdir(path):
        path = os.path.join(path, "MANIFEST.tsv")
    m = Manifest(path)
    try:
        f = open(path, encoding="utf-8")
    except OSError as e:
        raise ManifestError(f"cannot read {path}: {e}") from None
    with f:
        for n, line in enumerate(f, 1):
            line = line.rstrip("\n")
            at = f"{path}:{n}"
            if line.startswith("#"):
                key, tab, value = line[2:].partition("\t")
                if not tab:
                    if n == 1:
                        m.title = line
                    continue
                if key in COLUMN_KEYS:
                    kind = COLUMN_KEYS[key]
                    if kind in m.columns:
                        raise ManifestError(f"{at}: a second # {key} column line")
                    m.columns[kind] = [key] + value.split("\t")
                elif key == "complete":
                    if m.complete is not None:
                        raise ManifestError(f"{at}: a second # complete line")
                    m.complete = value
                else:
                    m.header.setdefault(key, value)
                continue
            if not line:
                continue
            fields = line.split("\t")
            kind = fields[0]
            if kind in ("skip", "skip-input"):
                names = DRAFT_SKIP_COLUMNS if len(fields) == len(DRAFT_SKIP_COLUMNS) else SKIP_COLUMNS
            else:
                names = m.columns.get(kind)
            if names is None:
                raise ManifestError(f"{at}: a {kind} row before its column line")
            if len(fields) != len(names):
                raise ManifestError(f"{at}: a {kind} row of {len(fields)} fields, its column line "
                                    f"names {len(names)}")
            m._rows.setdefault(kind, []).append(Row(names, fields, at, n))
    return m


def self_test():
    """Rows read by name whatever their order, and each refusal by name."""
    kind = "# kind\tname\toccurrence\ttype\tne0\tne1\tne2\tne3\tbytes\tsum\top\tcontig\tlogical\tsrc0\tsrc1"
    tensor = "tensor\tx y\t0\tf32\t2\t3\t1\t1\t24\t0.5\tMUL_MAT\t1\t0\tw\t-"
    layer = "# layer\tlayer\tsource\tproducer\ttokens\tid_sum\tignored\tfile"
    with tempfile.TemporaryDirectory() as d:
        def manifest(lines):
            with open(os.path.join(d, "MANIFEST.tsv"), "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            return read(d)

        def refused(lines, what):
            try:
                manifest(lines)
            except ManifestError as e:
                assert what in str(e), (what, str(e))
                return
            raise AssertionError(f"accepted: {what}")

        m = manifest(["# dump_ref — self-test", "# model\t/m/a.gguf", "# n_threads\t32\t32", kind,
                      tensor, "skip\tq\t0\tq8_0\tquantized", "# complete\t1\t1"])
        (r,) = m.rows("tensor")
        assert (r["name"], r.int("ne1"), r["src0"], r.int("contig")) == ("x y", 3, "w", 1), r["name"]
        assert m.header == {"model": "/m/a.gguf", "n_threads": "32\t32"}, m.header
        assert m.complete == "1\t1" and m.title == "# dump_ref — self-test" and len(m.rows("skip")) == 1
        m = manifest(["skip-input\tleaf_3\t0\tf32\tgraph-scratch\t0\t-\t0\tblock"])
        assert (m.rows("skip-input")[0]["reason"], m.rows("skip-input")[0]["graph"]) == ("graph-scratch", "block")
        refused(["skip\tq\t0\tq8_0"], "a skip row of 4 fields")
        # A column line in another order reads the same row.
        names = kind.split("\t")[1:]
        order = list(reversed(range(len(names))))
        fields = tensor.split("\t")[1:]
        m = manifest(["# kind\t" + "\t".join(names[i] for i in order),
                      "tensor\t" + "\t".join(fields[i] for i in order)])
        (r,) = m.rows("tensor")
        assert (r["name"], r.int("ne1"), r["op"]) == ("x y", 3, "MUL_MAT"), r["name"]
        assert m.complete is None
        # router_trace's layer line names field 1 like its kind.
        m = manifest([layer, "layer\t7\tVIEW\tp\t5\t0\t0\ttopk-7.u16"])
        assert m.rows("layer")[0].int("layer") == 7 and m.rows("layer")[0]["file"] == "topk-7.u16"
        refused([tensor], "a tensor row before its column line")
        refused([kind, tensor + "\textra"], "a tensor row of 16 fields")
        refused([kind, kind], "a second # kind column line")
        assert manifest(["# model\ta", "# model\tb"]).header == {"model": "a"}
        refused(["# complete\t1\t0", "# complete\t1\t0"], "a second # complete line")
        try:
            manifest([kind, tensor.replace("\t3\t", "\tx\t", 1)]).rows("tensor")[0].int("ne1")
            raise AssertionError("a garbage ne1 was read")
        except ManifestError as e:
            assert "ne1 'x' is not an integer" in str(e), str(e)
        try:
            manifest([layer, "layer\t7\tVIEW\tp\t5\t0\t0\ttopk-7.u16"]).rows("layer")[0]["op"]
            raise AssertionError("a field the line does not name was read")
        except ManifestError as e:
            assert "names no op field" in str(e), str(e)
    print("manifest: self-test ok")
    return 0


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-test"]:
        sys.exit(self_test())
    print(__doc__, file=sys.stderr)
    sys.exit(2)
