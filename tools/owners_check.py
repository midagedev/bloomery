#!/usr/bin/env python3
"""The owners' declared cover: which common gate covers each common owner, and whether the declaration holds (Mac).

    tools/owners_check.py plan [--owner O]... [--owners FILE|all]
    tools/owners_check.py check [--owners FILE|all]
    tools/owners_check.py audit --out DIR [--owner O]... [--dry-run]
    tools/owners_check.py --self-test

The three tables of tools/gate_class.py name the common owners (tools/owners.tsv), the gates
(tools/gate-class.tsv) and the families (tools/families.tsv). A gate's `owners` column declares the
owners it tests. This tool answers three questions about those declarations:

  plan    one line per mutant: `owner<TAB>patch<TAB>gates`, the gates the owner is covered by. An
          owner with no such gate prints `owner<TAB>-<TAB>NO-COVER`; an owner with no mutant yet
          prints `owner<TAB>-<TAB>NO-MUTANT`.
  check   the mutants' own clauses: every owner has a mutant dir with a patch, every patch applies
          (`git apply --check`), edits only files of its owner, carries its `# mutant:` header,
          creates and deletes nothing, and has a cover gate. One line per finding, rc 1 when there
          is one.
  audit   the mutation audit (the box): per owner, its mutants through its cover gates with
          tools/mutant-run.sh, the FAIL-first loop (apply, run the gate, restore, check the md5s).
          It ends with a table `owner<TAB>patch<TAB>gate<TAB>killed|survived|error` and exits 1 when
          a mutant survives every cover gate. `--dry-run` prints the commands instead of running
          them.

**The crates of a run** are the crates the owner's mutants live in (the crates of the mutants' own
files): tools/mutant-run.sh requires every crate it is told to print cargo's `Compiling <crate> v…`
line in every run, clean and mutated, so a crate the mutation does not touch cannot be named. A
cover gate that does not build the mutated crate therefore ends NO-COMPILE, which is the truth about
that pair: the gate does not run the owner's code.

**The cover gate of an owner** is a `common` row of tools/gate-class.tsv whose `owners` list names
the owner and which the fixture tier runs. The fixture tier is the one a landing runs, so a gate the
fixture tier defers (tools/gate-batch.sh's classifier: a real-only gate, a real-file gate, a gate of
a family with no fixture, a needs-env gate) is no cover here. The classifier is the one owner of that
question (tools/rotation.py reads the same output); the classifier reads `python3 tools/recipes.py
opens RECIPE` for the fence a gate reaches, so a gate that opens no model file (opens: none) and a
gate that opens one through the tier (a fixture-capable fence) are both fixture-runnable, and a
gate the fixture tier defers is not. The classifier's output is cached under
~/.cache/bloomery/owners-check-classes.tsv (the key: a digest of the justfile and the tools the
classifier reads); `--classes FILE` reads a saved one instead.

**A mutant** is tools/mutants/<owner>/<nn>-<what>.patch: a `git apply`-able diff on the tree at the
base, editing only files of that owner and changing a behaviour a correct gate must catch. It must
compile (a build failure would read as a kill) and it must not create, delete or rename a file. Its
first line is `# mutant: <one line>; expected killer: <recipe>`.

`--owners FILE` reads a scope: a file of owner names, one a line (`#` comments skipped). `all` is
every owner of tools/owners.tsv. The scope is the default; `--owner O` narrows it further.

Exit codes: 0, 1 (a finding, a red check, a surviving mutant), 2 (a named input failure), 70 (the
classifier or a tool it runs failed, named).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # `python3 -I` drops the script's directory
import gate_class  # noqa: E402
import recipes  # noqa: E402

ROOT = recipes.ROOT
MUTANT_DIR = "tools/mutants"
HEADER_RE = re.compile(r"^#\s*mutant:\s*\S.*;\s*expected killer:\s*\S.*$", re.M)
CLASSES_CACHE = Path(os.path.expanduser("~/.cache/bloomery/owners-check-classes.tsv"))
# The tools the fixture classifier reads; a change in one of them moves the cache's key.
CLASSES_KEY_FILES = (
    "justfile",
    "tools/gate-batch.sh",
    "tools/recipes.py",
    "tools/fixture-dirs.tsv",
    "tools/fixture-variants.tsv",
    "tools/gate-paths.tsv",
    "tools/arch-registry.tsv",
    "tools/gate-batch-resources.tsv",
)
# The classes the fixture tier defers: a gate in one of them is no cover (the header above).
DEFERRED_CLASS = "D"
# The mutant-run's verdicts, as its output lines name them.
VERDICT_KILLED = "killed"
VERDICT_SURVIVED = "survived"
VERDICT_ERROR = "error"
MUT_LINE = re.compile(r"^mut-(?P<name>.+?) rc=(?P<rc>-?\d+) \S+ (?P<verdict>\S+)(?P<rest>.*)$")


class OwnersError(Exception):
    """A named input failure; main() prints it and exits 2 (input) or 70 (a tool)."""

    def __init__(self, msg: str, rc: int = 2):
        super().__init__(msg)
        self.rc = rc


@dataclasses.dataclass(frozen=True)
class Mutant:
    """One mutant of one owner: its path, the owner's it belongs to and the files it edits."""

    owner: str
    path: Path
    rel: str
    files: tuple[str, ...]
    creates: bool
    deletes: bool
    renames: bool
    has_header: bool


# ---------------------------------------------------------------------------------------------- the tree


def owner_files(root: Path, owner: gate_class.OwnerRow) -> list[str]:
    """The tree's files the owner owns, as paths relative to the root, sorted."""
    out: set[str] = set()
    for g in owner.globs:
        for p in gate_class.matching_files(root, g):
            out.add(str(p.relative_to(root)))
    return sorted(out)


def crate_of(root: Path, rel: str) -> str | None:
    """The package name of the crate the file belongs to, read from the nearest manifest."""
    p = (root / rel).parent
    while p != root and p != root.parent:
        manifest = p / "Cargo.toml"
        if manifest.is_file():
            for line in manifest.read_text(encoding="utf-8").splitlines():
                m = re.match(r"^name\s*=\s*\"([^\"]+)\"", line.strip())
                if m:
                    return m.group(1)
            return None
        p = p.parent
    return None


def crates_of(root: Path, files: list[str]) -> list[str]:
    """The crates the files belong to, sorted, one a name."""
    out = {c for c in (crate_of(root, f) for f in files) if c}
    return sorted(out)


# ---------------------------------------------------------------------------------------------- the classes


def run_classifier(root: Path) -> str:
    """`tools/gate-batch.sh --tier fixture --classes`: the fixture tier's class per recipe."""
    script = root / "tools" / "gate-batch.sh"
    proc = subprocess.run(
        ["bash", str(script), "--tier", "fixture", "--classes"],
        capture_output=True,
        text=True,
        cwd=str(root),
    )
    if proc.returncode != 0:
        raise OwnersError(
            f"`tools/gate-batch.sh --tier fixture --classes` failed (rc {proc.returncode}): "
            f"{proc.stderr.strip()[-400:]}",
            70,
        )
    return proc.stdout


def classes_key(root: Path) -> str:
    """A digest of the files the classifier reads; the cache's key."""
    h = hashlib.sha256()
    for rel in CLASSES_KEY_FILES:
        p = root / rel
        h.update(rel.encode())
        h.update(p.read_bytes() if p.is_file() else b"")
    return h.hexdigest()


def parse_classes(text: str) -> dict[str, tuple[str, str]]:
    """The classifier's output: `name<TAB>class<TAB>kind<TAB>cards<TAB>why`, one recipe a line."""
    out: dict[str, tuple[str, str]] = {}
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        f = line.split("\t")
        if len(f) != 5 or not f[0] or len(f[1]) != 1:
            raise OwnersError(f"classes line {n}: not `name, class, kind, cards, why`: {line[:100]!r}")
        if f[0] in out:
            raise OwnersError(f"classes line {n}: the recipe {f[0]} is listed twice")
        out[f[0]] = (f[1], f[2])
    if not out:
        raise OwnersError("the classes input names no recipe")
    return out


def fixture_classes(root: Path, classes_file: str | None = None, refresh: bool = False) -> dict[str, tuple[str, str]]:
    """The fixture tier's class per recipe: the saved file, else the cache, else the classifier."""
    if classes_file:
        p = Path(classes_file)
        if not p.is_file():
            raise OwnersError(f"the classes file {p} does not exist")
        return parse_classes(p.read_text(encoding="utf-8"))
    key = classes_key(root)
    if not refresh and CLASSES_CACHE.is_file():
        text = CLASSES_CACHE.read_text(encoding="utf-8")
        head, _, body = text.partition("\n")
        if head.strip() == key:
            return parse_classes(body)
    text = run_classifier(root)
    try:
        CLASSES_CACHE.parent.mkdir(parents=True, exist_ok=True)
        CLASSES_CACHE.write_text(key + "\n" + text, encoding="utf-8")
    except OSError:
        pass  # a cache that cannot be written is no failure; the run paid for it.
    return parse_classes(text)


def deferred(classes: dict[str, tuple[str, str]]) -> set[str]:
    """The recipes the fixture tier defers: the class `D` ones."""
    return {name for name, (cls, _kind) in classes.items() if cls == DEFERRED_CLASS}


# ---------------------------------------------------------------------------------------------- the cover


def cover_gates(
    tables: gate_class.Tables,
    owner: str,
    deferred_recipes: set[str],
) -> list[str]:
    """The gates that cover the owner: the `common` rows naming it that the fixture tier runs."""
    out = [
        row.recipe
        for row in tables.gates.values()
        if row.cls == "common" and owner in row.owners and row.recipe not in deferred_recipes
    ]
    return sorted(out)


def common_gates_naming(tables: gate_class.Tables, owner: str) -> list[str]:
    """Every `common` gate naming the owner, the ones the fixture tier defers included."""
    return sorted(
        row.recipe for row in tables.gates.values() if row.cls == "common" and owner in row.owners
    )


# ---------------------------------------------------------------------------------------------- the mutants


def patch_files(text: str) -> tuple[tuple[str, ...], bool, bool, bool]:
    """The files a patch edits (relative paths), and whether it creates, deletes or renames one."""
    files: list[str] = []
    creates = deletes = renames = False
    for line in text.splitlines():
        if line.startswith("new file mode"):
            creates = True
        elif line.startswith("deleted file mode"):
            deletes = True
        elif line.startswith("rename from") or line.startswith("rename to"):
            renames = True
        elif line.startswith("--- "):
            path = line[4:].strip()
            if path != "/dev/null":
                files.append(path[2:] if path.startswith(("a/", "b/")) else path)
        elif line.startswith("+++ "):
            path = line[4:].strip()
            if path != "/dev/null":
                p = path[2:] if path.startswith(("a/", "b/")) else path
                if p not in files:
                    files.append(p)
    return tuple(sorted(set(files))), creates, deletes, renames


def read_mutant(root: Path, owner: str, path: Path) -> Mutant:
    """One mutant: its owner, its path, the files it edits and its header's presence."""
    text = path.read_text(encoding="utf-8", errors="replace")
    files, creates, deletes, renames = patch_files(text)
    return Mutant(
        owner=owner,
        path=path,
        rel=str(path.relative_to(root)),
        files=files,
        creates=creates,
        deletes=deletes,
        renames=renames,
        has_header=bool(HEADER_RE.search(text)),
    )


def mutants_of(root: Path, owner: str) -> list[Mutant]:
    """The owner's mutants, in name order; the dir holds `<name>.patch` or `<name>.diff`."""
    d = root / MUTANT_DIR / owner
    if not d.is_dir():
        return []
    out = []
    for p in sorted(d.iterdir()):
        if p.is_file() and p.suffix in (".patch", ".diff"):
            out.append(read_mutant(root, owner, p))
    return out


# ---------------------------------------------------------------------------------------------- the scope


def read_scope(path: str) -> list[str]:
    """The owner names of a scope file, one a line, `#` comments skipped."""
    p = Path(path)
    if not p.is_file():
        raise OwnersError(f"the scope file {p} does not exist")
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line and line not in out:
            out.append(line)
    if not out:
        raise OwnersError(f"the scope file {p} names no owner")
    return out


def scope_owners(tables: gate_class.Tables, args: argparse.Namespace) -> list[str]:
    """The owners the run is about: the scope (`--owners`), narrowed by `--owner`."""
    if getattr(args, "owners", None) in (None, "", "all"):
        names = sorted(tables.owners)
    else:
        names = read_scope(args.owners)
    unknown = [n for n in names if n not in tables.owners]
    if unknown:
        raise OwnersError(f"the scope names an owner tools/owners.tsv does not hold: {', '.join(unknown)}")
    picked = getattr(args, "owner", None) or []
    if picked:
        unknown = [n for n in picked if n not in tables.owners]
        if unknown:
            raise OwnersError(f"--owner names an owner tools/owners.tsv does not hold: {', '.join(unknown)}")
        names = [n for n in names if n in picked]
    return names


# ---------------------------------------------------------------------------------------------- the check


def git_apply_check(root: Path, patch: Path) -> tuple[bool, str]:
    """Whether the patch applies on the tree at `root`, and git's own words when it does not."""
    proc = subprocess.run(
        ["git", "-C", str(root), "apply", "--check", str(patch)],
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0, proc.stderr.strip()


def check_scope(
    root: Path,
    tables: gate_class.Tables,
    classes: dict[str, tuple[str, str]],
    owners: list[str],
) -> list[str]:
    """The findings of the check, one line each, empty when the tree is honest."""
    findings: list[str] = []
    deferred_recipes = deferred(classes)
    for owner in owners:
        row = tables.owners[owner]
        own = owner_files(root, row)
        cover = cover_gates(tables, owner, deferred_recipes)
        muts = mutants_of(root, owner)
        if not muts:
            findings.append(f"the owner {owner} has no mutant in {MUTANT_DIR}/{owner}")
        for m in muts:
            if not m.has_header:
                findings.append(f"{m.rel}: no `# mutant: <one line>; expected killer: <recipe>` header")
            if m.creates or m.deletes or m.renames:
                findings.append(f"{m.rel}: a mutant creates, deletes or renames a file")
            for f in m.files:
                if f not in own:
                    findings.append(f"{m.rel}: edits {f}, which the owner {owner} does not own")
            ok, why = git_apply_check(root, m.path)
            if not ok:
                findings.append(f"{m.rel}: does not apply: {why.splitlines()[-1] if why else 'git said nothing'}")
        if not cover:
            named = common_gates_naming(tables, owner)
            detail = f" (the common gates naming it: {', '.join(named)})" if named else ""
            findings.append(f"the owner {owner} has no cover gate in the fixture tier{detail}")
    return findings


# ---------------------------------------------------------------------------------------------- the plan


def cmd_plan(args: argparse.Namespace) -> int:
    root = Path(ROOT)
    tables = gate_class.load(root)
    owners = scope_owners(tables, args)
    classes = fixture_classes(root, args.classes)
    deferred_recipes = deferred(classes)
    for owner in owners:
        cover = cover_gates(tables, owner, deferred_recipes)
        muts = mutants_of(root, owner)
        if not cover:
            print(f"{owner}\t-\tNO-COVER")
            continue
        if not muts:
            print(f"{owner}\t-\tNO-MUTANT")
            continue
        for m in muts:
            print(f"{owner}\t{m.rel}\t{','.join(cover)}")
    return 0


# ---------------------------------------------------------------------------------------------- the audit


def audit_command(
    root: Path,
    owner: str,
    gate: str,
    mutants: Path,
    crates: list[str],
    out: Path,
) -> list[str]:
    """The mutant-run.sh command for one owner and one gate, as the audit runs it."""
    copy = out / "copy" / f"{owner}__{gate}"
    rundir = out / "run" / f"{owner}__{gate}"
    cmd = [
        str(root / "tools" / "mutant-run.sh"),
        "--clean",
        "first",
        "--copy",
        str(copy),
        "--out",
        str(rundir),
    ]
    for c in crates:
        cmd += ["--crate", c]
    cmd += [str(mutants), "--", "just", gate]
    return cmd


def run_audit(
    args: argparse.Namespace,
    root: Path | None = None,
    classes: dict[str, tuple[str, str]] | None = None,
) -> int:
    """The audit: every owner's mutants through its cover gates, and the table of the verdicts."""
    root = Path(ROOT) if root is None else root
    tables = gate_class.load(root)
    owners = scope_owners(tables, args)
    if classes is None:
        classes = fixture_classes(root, args.classes)
    deferred_recipes = deferred(classes)
    out = Path(args.out)
    rows: list[tuple[str, str, str, str]] = []
    for owner in owners:
        cover = cover_gates(tables, owner, deferred_recipes)
        muts = mutants_of(root, owner)
        if not cover:
            # No cover gate: the mutant survives every cover gate, there being none.
            if args.dry_run:
                print(f"# {owner}: no cover gate in the fixture tier; its mutant survives every cover gate")
            for m in muts:
                rows.append((owner, m.rel, "NO-COVER", VERDICT_SURVIVED))
            if not muts:
                rows.append((owner, "-", "NO-COVER", VERDICT_SURVIVED))
            continue
        crates = crates_of(root, [f for m in muts for f in m.files]) or crates_of(
            root, owner_files(root, tables.owners[owner])
        )
        for gate in cover:
            cmd = audit_command(root, owner, gate, root / MUTANT_DIR / owner, crates, out)
            if args.dry_run:
                print(" ".join(shlex_quote(c) for c in cmd))
                continue
            out.mkdir(parents=True, exist_ok=True)
            proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(root))
            verdicts = parse_verdicts(proc.stdout)
            for m in muts:
                verdict = verdicts.get(m.path.stem, VERDICT_ERROR)
                rows.append((owner, m.rel, gate, verdict))
            for line in proc.stdout.splitlines():
                if line.startswith("mut-") or "summary" in line:
                    print(f"  {line}")
            if proc.returncode not in (0, 1):
                print(f"  {owner}/{gate}: the runner ended rc {proc.returncode}: {proc.stderr.strip()[-300:]}")
    if args.dry_run:
        return 0
    print()
    for owner, patch, gate, verdict in rows:
        print(f"{owner}\t{patch}\t{gate}\t{verdict}")
    survived = [r for r in rows if r[3] in (VERDICT_SURVIVED, VERDICT_ERROR)]
    print(f"owners_check audit: {len(rows)} mutant(s), {len(survived)} not killed")
    return 1 if survived else 0


def shlex_quote(s: str) -> str:
    """A shell word: quoted when it holds a space or a quote."""
    if s and all(c.isalnum() or c in "-_./=:" for c in s):
        return s
    return "'" + s.replace("'", "'\\''") + "'"


def parse_verdicts(stdout: str) -> dict[str, str]:
    """The mutant-run's verdicts, a mutant's name to `killed`, `survived` or `error`."""
    out: dict[str, str] = {}
    for line in stdout.splitlines():
        m = MUT_LINE.match(line.strip())
        if not m:
            continue
        v = m.group("verdict")
        if v == "killed":
            out[m.group("name")] = VERDICT_KILLED
        elif v == "SURVIVED":
            out[m.group("name")] = VERDICT_SURVIVED
        else:
            out[m.group("name")] = VERDICT_ERROR
    return out


# ---------------------------------------------------------------------------------------------- the self-test


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def fake_tree(tmp: Path) -> tuple[Path, dict[str, tuple[str, str]]]:
    """A small tree with two owners, two gates, one mutant each and the classes that defer one gate."""
    root = tmp / "tree"
    _write(root / "justfile", "gate-a:\n    echo a\n\ngate-b:\n    echo b\n\n")
    _write(root / "tools" / "arch-registry.tsv", "# the registry's rows\nnames\tvariant\tread\tdraft\tfixture\nfam-a\tFamA\tno\tno\t-\n")
    _write(
        root / "tools" / "owners.tsv",
        "# owner\tkind\tglobs\n"
        "own-a\tkernel\tcrates/a/a.rs crates/a/b.rs\n"
        "own-b\tkernel\tcrates/b/c.rs\n",
    )
    groups = "".join(f"{g}\tF\tother\tcrates/b/c.rs\t30\tno\n" for g in gate_class.DECLARED_GROUPS)
    _write(
        root / "tools" / "families.tsv",
        "# family\trep\ttier\tglobs\trotation_days\tfixture\nfam-a\tR\tmain\tcrates/a/**\t7\tno\n" + groups,
    )
    _write(
        root / "tools" / "gate-class.tsv",
        "# recipe\tclass\tfamily\ttopic\towners\n"
        "gate-a\tcommon\t\tkernel\town-a\n"
        "gate-b\tcommon\t\tkernel\town-b\n",
    )
    _write(root / "crates" / "a" / "a.rs", "fn a() {\n    one();\n}\n")
    _write(root / "crates" / "a" / "b.rs", "fn b() {\n    two();\n}\n")
    _write(root / "crates" / "b" / "c.rs", "fn c() {\n    three();\n}\n")
    _write(root / "crates" / "a" / "Cargo.toml", "[package]\nname = \"fake-a\"\nversion = \"0.1.0\"\n")
    _write(root / "crates" / "b" / "Cargo.toml", "[package]\nname = \"fake-b\"\nversion = \"0.1.0\"\n")
    # The git repo the apply check reads.
    git = ["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run(["git", "init", "-q", str(root)], capture_output=True, text=True)
    subprocess.run(git + ["add", "-A"], capture_output=True, text=True)
    subprocess.run(git + ["commit", "-qm", "base"], capture_output=True, text=True)
    # The mutants: own-a's edits a file of its own; own-b's edits a file of the other owner.
    _write(
        root / MUTANT_DIR / "own-a" / "01-one.patch",
        "# mutant: own-a's rule; expected killer: gate-a\n"
        "--- a/crates/a/a.rs\n+++ b/crates/a/a.rs\n@@ -1,3 +1,3 @@\n fn a() {\n-    one();\n+    two();\n }\n",
    )
    _write(
        root / MUTANT_DIR / "own-b" / "01-three.patch",
        "# mutant: own-b's rule; expected killer: gate-b\n"
        "--- a/crates/a/a.rs\n+++ b/crates/a/a.rs\n@@ -1,3 +1,3 @@\n fn a() {\n-    one();\n+    two();\n }\n",
    )
    # gate-b is deferred in the fixture tier: own-b has no cover.
    classes = {
        "gate-a": ("A", "fixed"),
        "gate-b": ("D", "real-only"),
        "other": ("F", "balanced"),
    }
    return root, classes


def self_test() -> int:
    fails: list[str] = []

    def expect(cond: bool, msg: str) -> None:
        if not cond:
            fails.append(msg)

    import tempfile

    with tempfile.TemporaryDirectory(prefix="owners-check-selftest-") as tmpdir:
        tmp = Path(tmpdir)
        root, classes = fake_tree(tmp)
        tables = gate_class.load(root)

        # The fake tree is green: two owners, a mutant each, a cover gate for own-a.
        found = check_scope(root, tables, classes, ["own-a"])
        expect(found == [], f"check: the fake tree's own-a is green: {found}")
        expect(cover_gates(tables, "own-a", deferred(classes)) == ["gate-a"], "cover: own-a's cover gate is gate-a")
        expect(cover_gates(tables, "own-b", deferred(classes)) == [], "cover: own-b's only gate is deferred, so it has no cover")

        # A patch outside its owner.
        found = check_scope(root, tables, classes, ["own-b"])
        expect(any("own-b" in f and "does not own" in f for f in found), f"check: a patch outside its owner is red: {found}")

        # A missing header.
        m = root / MUTANT_DIR / "own-a" / "01-one.patch"
        text = m.read_text(encoding="utf-8")
        m.write_text(text.split("\n", 1)[1], encoding="utf-8")
        found = check_scope(root, tables, classes, ["own-a"])
        expect(any("header" in f for f in found), f"check: a patch without the header is red: {found}")
        m.write_text(text, encoding="utf-8")

        # A patch that does not apply.
        m.write_text(text.replace("-    one();", "-    three();"), encoding="utf-8")
        found = check_scope(root, tables, classes, ["own-a"])
        expect(any("does not apply" in f for f in found), f"check: a patch that does not apply is red: {found}")
        m.write_text(text, encoding="utf-8")

        # An owner with no mutant dir.
        shutil.rmtree(root / MUTANT_DIR / "own-a")
        found = check_scope(root, tables, classes, ["own-a"])
        expect(any("no mutant" in f for f in found), f"check: an owner with no mutant is red: {found}")
        _write(
            root / MUTANT_DIR / "own-a" / "01-one.patch",
            text,
        )

        # An owner with no cover gate.
        found = check_scope(root, tables, classes, ["own-b"])
        expect(any("no cover gate" in f for f in found), f"check: an owner with no cover gate is red: {found}")

        # The audit's dry run prints the commands it would run.
        out = tmp / "out"
        cmd = audit_command(root, "own-a", "gate-a", root / MUTANT_DIR / "own-a", ["fake-a"], out)
        expect(cmd[:2] == [str(root / "tools" / "mutant-run.sh"), "--clean"], f"audit: the command starts with the runner: {cmd}")
        expect("--copy" in cmd and "first" in cmd, f"audit: the command copies the tree and cleans first: {cmd}")
        expect(cmd[-3:] == ["--", "just", "gate-a"], f"audit: the command runs the gate through just: {cmd}")

        # The audit's dry run on the fake tree prints the commands it would run there.
        import contextlib
        import io

        ns = argparse.Namespace(
            out=str(out), owner=["own-a"], owners="all", classes=None, dry_run=True
        )
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = run_audit(ns, root=root, classes=classes)
        printed = buf.getvalue()
        expect(rc == 0, f"audit: the dry run ends rc 0: {rc}")
        expect(
            "mutant-run.sh" in printed and "just gate-a" in printed,
            f"audit: the dry run prints the command it would run: {printed!r}",
        )

        # The tree's own tables load and its owners hold a mutant each (the tree this tool ships in).
        try:
            real = gate_class.load(Path(ROOT))
            expect(len(real.owners) > 0, "load: the tree's own tables hold owners")
        except (gate_class.GateClassError, recipes.RecipeError) as e:
            expect(False, f"the tree's own tables: {e}")

    if fails:
        for f in fails:
            print(f"owners_check self-test FAIL: {f}", file=sys.stderr)
        print(f"owners_check self-test: {len(fails)} failed", file=sys.stderr)
        return 1
    print("owners_check self-test: ok")
    return 0


# ---------------------------------------------------------------------------------------------- the main


def main(argv: list[str]) -> int:
    if argv == ["--self-test"]:
        return self_test()
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan", help="one line a mutant: its owner, its patch and its cover gates")
    p.add_argument("--owner", action="append", help="an owner to plan (repeatable)")
    p.add_argument("--owners", help="the scope: a file of owner names, or `all`")
    p.add_argument("--classes", help="a saved `tools/gate-batch.sh --tier fixture --classes`")
    p = sub.add_parser("check", help="the mutants' own clauses; rc 1 with one line per finding")
    p.add_argument("--owners", help="the scope: a file of owner names, or `all`")
    p.add_argument("--classes", help="a saved `tools/gate-batch.sh --tier fixture --classes`")
    p = sub.add_parser("audit", help="the mutation audit: the mutants through the cover gates")
    p.add_argument("--out", required=True, help="the work dir (the copies, the runner's logs)")
    p.add_argument("--owner", action="append", help="an owner to audit (repeatable)")
    p.add_argument("--owners", help="the scope: a file of owner names, or `all`")
    p.add_argument("--classes", help="a saved `tools/gate-batch.sh --tier fixture --classes`")
    p.add_argument("--dry-run", action="store_true", help="print the commands instead of running them")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "plan":
            return cmd_plan(args)
        if args.cmd == "check":
            root = Path(ROOT)
            tables = gate_class.load(root)
            owners = scope_owners(tables, args)
            classes = fixture_classes(root, args.classes)
            findings = check_scope(root, tables, classes, owners)
            for f in findings:
                print(f"RED  {f}")
            print(f"owners_check check: {len(findings)} finding(s) over {len(owners)} owner(s)")
            return 1 if findings else 0
        return run_audit(args)
    except OwnersError as e:
        print(f"owners_check.py: {e}", file=sys.stderr)
        return e.rc
    except gate_class.GateClassError as e:
        print(f"owners_check.py: {e}", file=sys.stderr)
        return 2
    except recipes.RecipeError as e:
        print(f"owners_check.py: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
