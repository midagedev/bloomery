#!/usr/bin/env python3
"""The class tables of the gate-* recipes: their loader and the checks that keep them honest (Mac).

    tools/gate_class.py check          # rc 1 with one line per finding
    tools/gate_class.py summary         # the counts per class, per family and per owner
    tools/gate_class.py budget [--shared]       # the family budget table as it stands; --shared: the families whose gates load one real file in several bins
    tools/gate_class.py budget --refresh [--times FILE] [--classes FILE]   # real_s and unrowed rewritten from gate-times.tsv
    tools/gate_class.py --self-test

Four tables, one row a thing:

  tools/gate-class.tsv     recipe<TAB>class<TAB>family<TAB>topic<TAB>owners
  tools/families.tsv       family<TAB>rep<TAB>tier<TAB>globs<TAB>rotation_days<TAB>fixture
  tools/owners.tsv         owner<TAB>kind<TAB>globs
  tools/family-budget.tsv  family<TAB>real_s<TAB>unrowed<TAB>budget_s<TAB>waivers

`class` is `common` (family-agnostic: a synthetic input or a shape-generic owner), `family` (one family's own weights
or code) or `tooling` (harness, static, fixture writer, PTX). `topic` is what the gate tests, or `by-hand`: a gate no
batch runs, because its recipe needs an environment only a person supplies (its box command opens with
tools/ref/needs-env.sh, which tools/gate-batch.sh reads to defer the item by name); its class and family stay its own.
`family` is the registry's canonical name (tools/arch-registry.tsv column 1, the first name of a comma list) or a
declared non-arch group; a `common` or a `tooling` row names none. `owners` is the `+` list of the owners the gate
tests, names from tools/owners.tsv, or `-`. `rep` is `R` (a representative family, run whole on a landing) or `F` (a
fixture family, run on its own change and in the rotation). `tier` is `main` (the real tier runs its gates daily and at a
release) or `other`: the fixture tier at its landing and in the rotation (30 days) when the family has a fixture
(`fixture` yes, the fact of tools/fixture-dirs.tsv), its real-tier gates owed only on its own change (its globs, its recipe's
text) or by name, never because a common file moved; a family with no fixture keeps its real file, which the rotation runs
once a month. A family new in a release (`new_families`) runs the real tier daily and at that release whatever its tier.
`globs` are the family's own files and the owner's files; a glob matches a file or the round is red.

`check` runs the checks of the design (§4.4): every `gate-*` recipe has a row and every row names a recipe that
exists; the families of both tables are the registry's canonical names plus the declared non-arch groups; every
glob of both tables matches a file in the tree; the `by-hand` rows are exactly the recipes that call
tools/ref/needs-env.sh, in their own text or through a dependency; the `fixture` column is tools/fixture-dirs.tsv's (yes exactly
when that table names a directory for the family, not `self`); and a change to a kernel file keeps a gate that launches it
(kernel_findings: an owner a kept gate names, a gate's own target, a representative family's gate, or the globs of every F family
whose gates launch it; read_kernel_reach reads the launches through tools/recipes.py's Reach, ~15 s). One line per finding, rc 1
when there is one. A table that does not load is a named error, rc 2: a family row with no `tier`, or one the tiers do not name, is
such an error.

The class stage of `just affected` and the release list read the tables (tools/recipes.py `class_tables`), and so does the rotation
(tools/rotation.py, for the F families' gates); this tool checks them. The justfile's recipe names are read through tools/recipes.py,
the one parser.

The fourth table holds the release budget of each family, the rule that a new model adds the least to every later release (AGENTS.md,
Conventions). `check` refuses, by family: real_s over budget_s (real_s is the sum of the medians of the family's real-tier gates in
gate-times.tsv, written by `budget --refresh` on the machine that has the file, since CI has none); a budget_s over the default 300 s with
no `budget` waiver; more than one `+ship` gate with no `ship` waiver; gates and no fixture (families.tsv `fixture` no) with no `fixture`
waiver, a waiver that grandfathers a family which predates the rule and so is not dated after RULE_DAY; tier `main` with no `main` waiver
that names the maintainer's call. A waiver is `rule=PIN(YYYY-MM-DD): reason`; one that is no longer needed is a note, never a finding.
Gates of one family that load the same real file in several bins are reported (`budget --shared`), not refused.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import importlib.util
import os
import re
import sys
from collections.abc import Callable, Iterable
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # `python3 -I` drops the script's directory
import recipes  # noqa: E402

ROOT = recipes.ROOT
CLASSES = ("common", "family", "tooling")
REPS = ("R", "F")
# The tiers of a family and the days its gates may go unrun before the rotation asks for them (the `rotation_days` column
# holds the same number; `check` holds them together). A gate no family owns (common, tooling) runs on the main tier's days.
# PIN(2026-11-03): other 30 d. The maintainer's risk-based scope: the real tier runs daily and at a release for the main
# families (deepseek41, glm5next, qwen4exp), the families new in the release and the common and tooling gates; every other
# family is fixture-tier at its landing and, with a fixture, in its 30-day rotation too (its real gates owed only on its own change
# or by name: on 2026-11-09 the maintainer kept only the three main families' real files on NVMe), with none real-tier once a month
# or on a user report. Derivation: of the 9,173 s of item time in
# lane X of the 0.3.12 release batch (rows of gate-times.tsv, 06:11-10:11), the other families held 6,702 s, Muse Spark 3,843 s and
# Mistral Large 4 2,469 s of it, and 13 of that batch's 14 reds were also red on main before it.
TIERS = ("main", "other")
TIER_DAYS = {"main": 7, "other": 30}
GATE_PREFIX = recipes.GATE_PREFIX
# The topic of a gate that only a person runs, and the script whose call in a recipe makes it one.
BY_HAND = "by-hand"
NEEDS_ENV_SCRIPT = "tools/ref/needs-env.sh"
# The families that are not architectures: a group of gates with no registry row of its own. A family that is not in
# the registry and not here has no owner, and the check is red.
DECLARED_GROUPS = ("flint", "gemma4a", "gemma4v", "minicpmv47", "qwen38d")


class GateClassError(Exception):
    """A named input failure; main() prints it and exits 2."""

    def __init__(self, msg: str, rc: int = 2):
        super().__init__(msg)
        self.rc = rc


@dataclasses.dataclass(frozen=True)
class GateRow:
    recipe: str
    cls: str
    family: str
    topic: str
    owners: tuple[str, ...]

    @property
    def by_hand(self) -> bool:
        return self.topic == BY_HAND


@dataclasses.dataclass(frozen=True)
class FamilyRow:
    family: str
    rep: str
    tier: str
    globs: tuple[str, ...]
    rotation_days: int
    fixture: bool


@dataclasses.dataclass(frozen=True)
class OwnerRow:
    owner: str
    kind: str
    globs: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class Tables:
    gates: dict[str, GateRow]
    families: dict[str, FamilyRow]
    owners: dict[str, OwnerRow]


# ---------------------------------------------------------------------------------------------- the tables


def _rows(path: Path) -> list[tuple[int, list[str]]]:
    """(line number, tab-separated fields) of every data row of a table; '#' comments and blank lines skipped."""
    out: list[tuple[int, list[str]]] = []
    with path.open(encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            line = line.rstrip("\n")
            if not line.strip() or line.startswith("#"):
                continue
            out.append((n, line.split("\t")))
    return out


def _where(path: Path, line: int) -> str:
    return f"{path.name}:{line}"


def load_owners(root: Path) -> dict[str, OwnerRow]:
    """tools/owners.tsv: the owner, its kind and the files it owns."""
    path = root / "tools" / "owners.tsv"
    if not path.exists():
        raise GateClassError(f"tools/owners.tsv does not exist under {root}")
    owners: dict[str, OwnerRow] = {}
    for n, f in _rows(path):
        if len(f) != 3:
            raise GateClassError(f"{_where(path, n)}: a row is not `owner, kind, globs` (tab-separated): {'\\t'.join(f)[:100]!r}")
        name, kind, globs = f[0].strip(), f[1].strip(), f[2].split()
        if not name or not kind:
            raise GateClassError(f"{_where(path, n)}: a row has no owner name or no kind")
        if name in owners:
            raise GateClassError(f"{_where(path, n)}: the owner {name} is listed twice")
        owners[name] = OwnerRow(owner=name, kind=kind, globs=tuple(globs))
    if not owners:
        raise GateClassError("tools/owners.tsv names no owner")
    return owners


def load_families(root: Path) -> dict[str, FamilyRow]:
    """tools/families.tsv: the family, its rep, its tier, its own files, its rotation period and whether it has a fixture."""
    path = root / "tools" / "families.tsv"
    if not path.exists():
        raise GateClassError(f"tools/families.tsv does not exist under {root}")
    families: dict[str, FamilyRow] = {}
    for n, f in _rows(path):
        if len(f) == 5:
            raise GateClassError(f"{_where(path, n)}: a row has no `tier` (main or other): a row is `family, rep, tier, globs, rotation_days, fixture`: {'\\t'.join(f)[:100]!r}")
        if len(f) != 6:
            raise GateClassError(f"{_where(path, n)}: a row is not `family, rep, tier, globs, rotation_days, fixture` (tab-separated): {'\\t'.join(f)[:100]!r}")
        name, rep, tier, globs, days, fixture = f[0].strip(), f[1].strip(), f[2].strip(), f[3].split(), f[4].strip(), f[5].strip()
        if not name:
            raise GateClassError(f"{_where(path, n)}: a row has no family name")
        if rep not in REPS:
            raise GateClassError(f"{_where(path, n)}: unknown rep {rep!r} (one of {', '.join(REPS)})")
        if tier == "":
            raise GateClassError(f"{_where(path, n)}: the family {name} has no `tier` (one of {', '.join(TIERS)})")
        if tier not in TIERS:
            raise GateClassError(f"{_where(path, n)}: unknown tier {tier!r} (one of {', '.join(TIERS)})")
        if days == "":
            days_s = TIER_DAYS[tier]
        elif days.isdigit() and int(days) > 0:
            days_s = int(days)
        else:
            raise GateClassError(f"{_where(path, n)}: `rotation_days` is not a number of days: {days!r}")
        if fixture not in ("yes", "no"):
            raise GateClassError(f"{_where(path, n)}: `fixture` is {fixture!r} (yes or no)")
        if name in families:
            raise GateClassError(f"{_where(path, n)}: the family {name} is listed twice")
        families[name] = FamilyRow(family=name, rep=rep, tier=tier, globs=tuple(globs), rotation_days=days_s, fixture=fixture == "yes")
    if not families:
        raise GateClassError("tools/families.tsv names no family")
    return families


def load_gates(root: Path, owners: dict[str, OwnerRow], families: dict[str, FamilyRow]) -> dict[str, GateRow]:
    """tools/gate-class.tsv: the recipe, its class, its family, its topic and the owners it tests."""
    path = root / "tools" / "gate-class.tsv"
    if not path.exists():
        raise GateClassError(f"tools/gate-class.tsv does not exist under {root}")
    gates: dict[str, GateRow] = {}
    for n, f in _rows(path):
        if len(f) != 5:
            raise GateClassError(f"{_where(path, n)}: a row is not `recipe, class, family, topic, owners` (tab-separated): {'\\t'.join(f)[:100]!r}")
        recipe, cls, family, topic, owners_s = (x.strip() for x in f)
        if not recipe:
            raise GateClassError(f"{_where(path, n)}: a row has no recipe name")
        if cls not in CLASSES:
            raise GateClassError(f"{_where(path, n)}: unknown class {cls!r} (one of {', '.join(CLASSES)})")
        if cls == "family" and not family:
            raise GateClassError(f"{_where(path, n)}: a `family` row names no family")
        if cls != "family" and family:
            raise GateClassError(f"{_where(path, n)}: a `{cls}` row names a family ({family}); only a `family` row does")
        if family and family not in families:
            raise GateClassError(f"{_where(path, n)}: the family {family} is not in families.tsv")
        if owners_s in ("", "-"):
            owner_names: tuple[str, ...] = ()
        else:
            owner_names = tuple(owners_s.split("+"))
            for o in owner_names:
                if not o:
                    raise GateClassError(f"{_where(path, n)}: the owner list has an empty name")
                if o not in owners:
                    raise GateClassError(f"{_where(path, n)}: the owner {o} is not in owners.tsv")
            if len(set(owner_names)) != len(owner_names):
                raise GateClassError(f"{_where(path, n)}: the owner list names an owner twice")
        if recipe in gates:
            raise GateClassError(f"{_where(path, n)}: the recipe {recipe} is listed twice")
        gates[recipe] = GateRow(recipe=recipe, cls=cls, family=family, topic=topic, owners=owner_names)
    if not gates:
        raise GateClassError("tools/gate-class.tsv names no recipe")
    return gates


def load(root: str | os.PathLike = ROOT) -> Tables:
    """The three tables of the tree at ROOT, loaded and cross-checked. A named error names the file and the line."""
    root = Path(root)
    owners = load_owners(root)
    families = load_families(root)
    gates = load_gates(root, owners, families)
    return Tables(gates=gates, families=families, owners=owners)


# ---------------------------------------------------------------------------------------------- the tree


def matching_files(root: Path, pattern: str) -> list[Path]:
    """The files of the tree at ROOT the glob matches; `*` stays in a path segment, `**` crosses one."""
    try:
        if pattern.endswith("/**"):
            d = root / pattern[:-3]
            if not d.is_dir():
                return []
            return [p for p in d.rglob("*") if p.is_file()]
        return [p for p in root.glob(pattern) if p.is_file()]
    except (ValueError, OSError):
        return []


def registry_families(root: Path) -> set[str]:
    """The registry's canonical names: the first name of every row of tools/arch-registry.tsv."""
    path = root / "tools" / "arch-registry.tsv"
    if not path.exists():
        raise GateClassError(f"tools/arch-registry.tsv does not exist under {root}")
    out: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        names = line.split("\t")[0].strip()
        if not names or names == "names":
            continue
        out.add(names.split(",")[0].strip())
    if not out:
        raise GateClassError("tools/arch-registry.tsv names no architecture")
    return out


# ---------------------------------------------------------------------------------------------- the scope


OTHER_REASON = "other family: monthly rotation"


@dataclasses.dataclass(frozen=True)
class GateScope:
    """What the real tier owes a gate: the class-table family it belongs to (empty for a common or a tooling gate), its tier
    (`main`, `other`, or the gate's class where it has no family), the days it may go unrun and whether its family has a fixture
    (families.tsv `fixture`). An `other` family with a fixture is owed on the fixture tier (`fixture_tier`)."""

    family: str
    tier: str
    days: int
    fixture: bool = False

    @property
    def fixture_tier(self) -> bool:
        """The rotation holds this gate's family to the fixture tier: an `other` family with a fixture file. Its real-tier gates are
        owed only on its own change or by name (tools/rotation.py)."""
        return self.tier == "other" and self.fixture


def gate_scope(tables: Tables, name: str, profile_family: str = "") -> GateScope:
    """The scope of a recipe: its class-table row names the family. A recipe with no row (a `weekly-*` recipe, `refset-check`)
    belongs to the family of the profile its box call sets (`profile_family`) when families.tsv names that family, else it is a
    common gate. A common or a tooling gate runs on the main tier's days."""
    row = tables.gates.get(name)
    if row is not None and row.cls == "family":
        fam = tables.families[row.family]
        return GateScope(row.family, fam.tier, fam.rotation_days, fam.fixture)
    if row is not None:
        return GateScope("", row.cls, TIER_DAYS["main"])
    fam = tables.families.get(profile_family)
    if fam is not None:
        return GateScope(profile_family, fam.tier, fam.rotation_days, fam.fixture)
    return GateScope("", "common", TIER_DAYS["main"])


def left_out(scope: GateScope, new: set[str] | frozenset[str]) -> str:
    """Why the real tier of a daily pass or a release leaves a gate out: its family is not main and is not new in the
    release. Empty when the gate is in scope."""
    if scope.tier == "other" and scope.family not in new:
        return OTHER_REASON
    return ""


def partition(names: list[str], scope_of: Callable[[str], GateScope], new: set[str] | frozenset[str]) -> tuple[list[str], dict[str, tuple[str, GateScope]]]:
    """(the names in scope in their order, the names left out as name -> (reason, scope))."""
    kept: list[str] = []
    out: dict[str, tuple[str, GateScope]] = {}
    for n in names:
        sc = scope_of(n)
        why = left_out(sc, new)
        if why:
            out[n] = (why, sc)
        else:
            kept.append(n)
    return kept, out


def own_files(families: dict[str, FamilyRow], files: Iterable[str]) -> dict[str, list[str]]:
    """The files each family owns: matched by one of its globs and by no other family's glob (a glob may serve several
    families, and a file of such a glob tells nothing of any one of them)."""
    pats = {name: [(recipes.glob_prefix(g), recipes.class_glob(g)) for g in fam.globs] for name, fam in families.items()}
    own: dict[str, list[str]] = {name: [] for name in families}
    for f in files:
        hits = [name for name, ps in pats.items() if any(f.startswith(pre) and pat.match(f) for pre, pat in ps)]
        if len(hits) == 1:
            own[hits[0]].append(f)
    return own


def new_families(families: dict[str, FamilyRow], base_files: Iterable[str], head_files: Iterable[str]) -> set[str]:
    """The families new in a release: one with a file of its own (own_files) in the tree at the head and none in the tree at
    the base, the last release tag's. Pure: the two trees come in as file lists."""
    at_base = own_files(families, base_files)
    at_head = own_files(families, head_files)
    return {name for name in families if at_head[name] and not at_base[name]}


# ---------------------------------------------------------------------------------------------- the kernel files


PTX_NAMES = "tools/check-ptx-names.py"  # the one reader of a `#[kernel]` declaration (its scan_file)
OWN, DEP = "own", "dep"  # how a gate launches a kernel file: its own bin or test target's file, or a dependency chain
OWN_KINDS = ("bin", "test")  # the targets whose own module tree is a launch: a lib's own tests or doctests read every file of the lib


@dataclasses.dataclass(frozen=True)
class KernelReach:
    """Who launches each kernel file of the workspace (a file that declares a `#[kernel]`, as tools/check-ptx-names.py reads
    one): file -> gate -> OWN (the file is in the gate's own bin or test target; a lib's own tests are not a launch: a device
    crate's lib tests read every file of it) or DEP (a family gate whose reach holds a unit of the file through a chain that
    enters no other family's own file, recipes.Reach.gate_files_kept_out). `skipped` holds the kernel files of a bin
    recipes.CARRIER_SKIP names, with its reason: no gate links it."""

    launchers: dict[str, dict[str, str]]
    skipped: dict[str, str]


def _ptx_names():
    spec = importlib.util.spec_from_file_location("check_ptx_names", os.path.join(ROOT, PTX_NAMES))
    if spec is None or spec.loader is None:
        raise GateClassError(f"{PTX_NAMES} is not readable")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def family_files(root: Path, families: dict[str, FamilyRow]) -> dict[str, set[str]]:
    """The files of the tree each family's globs match."""
    return {name: {str(p.relative_to(root)) for g in fam.globs for p in matching_files(root, g)} for name, fam in families.items()}


def read_kernel_reach(root: Path, tables: Tables) -> KernelReach:
    """The kernel files of the workspace members and their launchers (KernelReach), read through tools/recipes.py's tree, graph
    and Reach. A family gate's chain is kept out of every file another family's globs match and its own do not (a seat of the
    one-binary server, a tower the image crate's lib dispatches to: that family's launch, not this gate's); a kernel file is
    never kept out, so a gate that reaches one through its own code stands."""
    ptx = _ptx_names()
    try:
        side = recipes.make_side(str(root))
    except recipes.RecipeError as e:
        raise GateClassError(f"the tree's recipes and crates do not load: {e}") from e
    tree, graph = side.tree, side.graph
    kernels: dict[str, dict[str, str]] = {}
    for pkg in sorted(tree.packages.values(), key=lambda p: p.dir):
        for d, _, fs in os.walk(root / pkg.dir):
            for f in sorted(fs):
                if not f.endswith(".rs"):
                    continue
                rel = os.path.relpath(os.path.join(d, f), root)
                text = (root / rel).read_text(encoding="utf-8")
                if "#[kernel" not in text:
                    continue
                try:
                    found, _ = ptx.scan_file(text, rel)
                except ptx.Refuse as e:
                    raise GateClassError(f"{PTX_NAMES} refuses {rel}: {e}") from e
                if found:
                    kernels[rel] = {}
    skipped: dict[str, str] = {}
    scan_pkg = tree.packages.get(recipes.SCAN_PACKAGE)
    for name, why in recipes.CARRIER_SKIP.items():
        t = scan_pkg.find("bin", name) if scan_pkg is not None else None
        if t is None:
            raise GateClassError(f"recipes.CARRIER_SKIP names the bin {name}, which {recipes.SCAN_PACKAGE} does not build")
        for f in tree.tree_of(t.src, True)[0]:
            if f in kernels:
                skipped[f] = f"{name}: {why}"
    gates = sorted(n for n in side.recipes if n.startswith(GATE_PREFIX))
    for n in gates:
        for pname, kind, tname, _ in graph.inputs(n).targets:
            if kind not in OWN_KINDS:
                continue
            t = tree.packages[pname].find(kind, tname or "")
            if t is None:
                continue
            for f in tree.tree_of(t.src, True)[0]:
                if f in kernels:
                    kernels[f][n] = OWN
    files = family_files(root, tables.families)
    every = set().union(*files.values())
    reach = recipes.reach_of(tree)
    for n in gates:
        row = tables.gates.get(n)
        if row is None or row.cls != "family":
            continue
        kept_out = frozenset(every - files[row.family] - set(kernels))
        held = reach.gate_files_kept_out(graph, n, kept_out)
        for f in kernels:
            if f in held:
                kernels[f].setdefault(n, DEP)
    return KernelReach(kernels, skipped)


def _matches(globs: Iterable[str], f: str) -> bool:
    return any(f.startswith(recipes.glob_prefix(g)) and recipes.class_glob(g).match(f) for g in globs)


def kernel_findings(tables: Tables, kr: KernelReach) -> list[str]:
    """The kernel-file ownership rule: a change to a kernel file keeps a gate that launches it. A file passes when an owner a gate's
    class row names owns it and that gate is kept on the file's change (a common gate always is; a family gate when its family's
    globs match the file), when it is a gate's own bin or test target's file (the class stage keeps that gate by its own target),
    when a representative family's gate launches it (the class stage keeps an R family's gate by reach), or when every F family
    whose gates launch it names it in its globs (an F family's gate is kept only on its own files). A file no gate launches is a
    finding unless a bin recipes.CARRIER_SKIP names holds it."""
    out: list[str] = []
    namers: dict[str, list[GateRow]] = {}
    for row in tables.gates.values():
        for o in row.owners:
            namers.setdefault(o, []).append(row)
    for f in sorted(kr.launchers):
        if f in kr.skipped:
            continue
        owned = any(row.cls != "family" or _matches(tables.families[row.family].globs, f)
                    for o, orow in tables.owners.items() if _matches(orow.globs, f) for row in namers.get(o, ()))
        launch = kr.launchers[f]
        if owned or OWN in launch.values():
            continue
        fams: dict[str, list[str]] = {}
        for n in sorted(launch):
            row = tables.gates.get(n)
            if row is not None and row.cls == "family":
                fams.setdefault(row.family, []).append(n)
        if any(tables.families[x].rep == "R" for x in fams):
            continue
        if not fams:
            out.append(f"kernel file {f}: no gate launches it (no owner a kept gate names, no gate's own target, no family gate's reach): "
                       f"gate it, or name its bin in tools/recipes.py CARRIER_SKIP with the reason")
            continue
        missing = [x for x in sorted(fams) if not _matches(tables.families[x].globs, f)]
        if missing:
            who = "; ".join(f"{x} ({' '.join(fams[x][:3])}{' …' if len(fams[x]) > 3 else ''})" for x in missing)
            out.append(f"kernel file {f}: launched by the F family gates of {who}, and those families' globs do not name it: a change to "
                       f"it keeps none of them (add it to their families.tsv globs, or to an owner a gate that launches it names)")
    return out


def fixture_findings(root: Path, tables: Tables) -> list[str]:
    """families.tsv's `fixture` column against tools/fixture-dirs.tsv, the one table of a family's fixture: yes exactly when the table
    has a row for the family whose directory is not `self`. The table's other rows (a profile's cut, a family families.tsv does not
    name) are not this column's."""
    try:
        rows, _ = recipes.fixture_table((root / recipes.FIXTURE_DIRS).read_text(encoding="utf-8"),
                                        (root / recipes.REF_PATHS).read_text(encoding="utf-8"))
    except OSError as e:
        raise GateClassError(f"the fixture table is not readable under {root}: {e}") from e
    except recipes.RecipeError as e:
        raise GateClassError(f"the fixture table does not load: {e}") from e
    out = []
    for fam in sorted(tables.families):
        want = rows.get(fam, "self") != "self"
        if tables.families[fam].fixture != want:
            got = "yes" if tables.families[fam].fixture else "no"
            where = f"its row there is {rows[fam]}" if fam in rows else "it has no row there"
            out.append(f"families.tsv: the family {fam} says fixture {got}, and {recipes.FIXTURE_DIRS} says {'yes' if want else 'no'} ({where})")
    return out


# ---------------------------------------------------------------------------------------------- the checks


def calls_needs_env(recs: dict[str, recipes.Recipe], name: str, seen: frozenset[str] = frozenset()) -> bool:
    """Whether recipe `name` calls tools/ref/needs-env.sh in its own text or through a dependency."""
    rec = recs.get(name)
    if rec is None or name in seen:
        return False
    return NEEDS_ENV_SCRIPT in rec.text or any(calls_needs_env(recs, d, seen | {name}) for d in rec.deps)


def check(root: Path, tables: Tables, kernels: Callable[[Path, Tables], KernelReach] = read_kernel_reach) -> list[str]:
    """The findings of the design's checks: one line each, empty when the tables are honest. `kernels` reads who launches each
    kernel file (read_kernel_reach on a tree; the self-test hands a table of its own)."""
    findings: list[str] = []
    justfile = root / "justfile"
    if not justfile.exists():
        raise GateClassError(f"there is no justfile under {root}")
    try:
        recs = recipes.load_justfile(str(justfile))
    except recipes.RecipeError as e:
        raise GateClassError(f"the justfile does not load: {e}") from e
    gate_names = {n for n in recs if n.startswith(GATE_PREFIX)}

    # Every gate has a row, and every row names a gate.
    for name in sorted(gate_names - set(tables.gates)):
        findings.append(f"no gate-class.tsv row for the recipe {name}")
    for name in sorted(set(tables.gates) - gate_names):
        if name in recs:
            findings.append(f"gate-class.tsv names {name}, which is not a {GATE_PREFIX} recipe of the justfile")
        else:
            findings.append(f"gate-class.tsv names {name}, which is not a recipe of the justfile")

    # A by-hand gate is a recipe that calls tools/ref/needs-env.sh and the other way round: a batch defers such a recipe
    # by name and nothing else runs it, so one the table did not declare would drop out of every list without a row.
    for name in sorted(gate_names & set(tables.gates)):
        row, needs = tables.gates[name], calls_needs_env(recs, name)
        if needs and not row.by_hand:
            findings.append(f"{name} calls {NEEDS_ENV_SCRIPT} and its topic is {row.topic!r}, not {BY_HAND!r}: nothing runs it unattended")
        if row.by_hand and not needs:
            findings.append(f"{name} is declared {BY_HAND} and its recipe calls no {NEEDS_ENV_SCRIPT}")

    # The families of both tables are the registry's canonical names plus the declared groups.
    want = registry_families(root) | set(DECLARED_GROUPS)
    for fam in sorted(want - set(tables.families)):
        findings.append(f"families.tsv has no row for the family {fam}")
    for fam in sorted(set(tables.families) - want):
        findings.append(f"families.tsv names the family {fam}, which the registry does not name and no group is declared")
    # A family with no gate row of its own is not a finding (a registry row may have no gate yet); the summary names
    # it, and the loader refuses a gate row that names a family families.tsv does not hold.

    # The rotation period is the tier's, and a family owns a file no other family's glob matches (a release tells a new
    # family by its own files).
    for fam in sorted(tables.families):
        row = tables.families[fam]
        if row.rotation_days != TIER_DAYS[row.tier]:
            findings.append(f"families.tsv: the family {fam} is tier {row.tier}, whose rotation is {TIER_DAYS[row.tier]} d, and its rotation_days is {row.rotation_days}")
    listing = sorted({str(p.relative_to(root)) for fam in tables.families.values() for g in fam.globs for p in matching_files(root, g)})
    owned = own_files(tables.families, listing)
    for fam in sorted(tables.families):
        if not owned[fam]:
            findings.append(f"families.tsv: the family {fam} owns no file that no other family's glob matches, so a release cannot tell when it is new")

    # Every glob matches a file.
    for owner in sorted(tables.owners):
        for g in tables.owners[owner].globs:
            if not matching_files(root, g):
                findings.append(f"owners.tsv: the owner {owner}'s glob {g} matches no file")
    for fam in sorted(tables.families):
        for g in tables.families[fam].globs:
            if not matching_files(root, g):
                findings.append(f"families.tsv: the family {fam}'s glob {g} matches no file")

    # The fixture column is the fixture table's fact, and a change to a kernel file keeps a gate that launches it.
    findings += fixture_findings(root, tables)
    findings += kernel_findings(tables, kernels(root, tables))
    # The release budget of each family.
    findings += budget_findings(tables, load_budget(root))[0]
    return findings


# ---------------------------------------------------------------------------------------------- the family budget


BUDGET_TABLE = Path("tools") / "family-budget.tsv"
# What a new family may cost a release: the sum of its real-tier gates' medians, 300 s unless its row carries a `budget` waiver.
DEFAULT_BUDGET_S = 300
# recipes.predicted_wall's price of a gate with no real-tier row (release-list --wall); the self-test holds the two together.
UNROWED_S = 45
# PIN(2026-11-14): the day the budget rule landed. A `fixture` waiver grandfathers a family that predates the rule, so none is dated
# after this day: a newer family has a fixture row, and moving the day is the maintainer's call.
RULE_DAY = datetime.date(2026, 11, 14)
WAIVER_RULES = ("budget", "ship", "main", "fixture")
WAIVER = re.compile(rf"({'|'.join(WAIVER_RULES)})=PIN\((\d{{4}}-\d{{2}}-\d{{2}})\): (\S.*)")
WAIVER_SEP = " | "
# The bins a card gate runs are named gate_*; the other bins a recipe builds (the server, the generator) are tools it drives.
GATE_BIN = "gate_"
# A `main` waiver names the call it records.
MAIN_CALL = "maintainer"
# release-list --wall flags a family whose table real_s differs from the live medians by more than STALE_FRAC of the larger and STALE_MIN_S.
STALE_FRAC, STALE_MIN_S = 0.1, 20


@dataclasses.dataclass(frozen=True)
class Waiver:
    day: datetime.date
    reason: str


@dataclasses.dataclass(frozen=True)
class BudgetRow:
    family: str
    real_s: int
    unrowed: int
    budget_s: int
    waivers: dict[str, Waiver]


@dataclasses.dataclass(frozen=True)
class FamilyCost:
    """What a family's real-tier gates cost: the gates themselves (the recipes a real-tier batch runs unattended, whatever the family's
    tier), the sum of the medians of those with a real-tier row and how many have none."""

    gates: tuple[str, ...]
    real_s: int
    unrowed: int

    @property
    def priced_s(self) -> int:
        """real_s with every gate that has no row at the price release-list --wall gives it: the number a release would spend."""
        return self.real_s + self.unrowed * UNROWED_S


def parse_waivers(text: str, where: str) -> dict[str, Waiver]:
    """The `waivers` column: `-`, or entries `rule=PIN(YYYY-MM-DD): reason` joined by ' | '."""
    if text in ("", "-"):
        return {}
    out: dict[str, Waiver] = {}
    for entry in text.split(WAIVER_SEP):
        m = WAIVER.fullmatch(entry.strip())
        if m is None:
            raise GateClassError(f"{where}: a waiver is `rule=PIN(YYYY-MM-DD): reason` with the rule one of {', '.join(WAIVER_RULES)}, joined by "
                                 f"{WAIVER_SEP!r}: {entry[:80]!r}")
        rule, day_s, reason = m.groups()
        try:
            day = datetime.date.fromisoformat(day_s)
        except ValueError as e:
            raise GateClassError(f"{where}: the {rule} waiver is dated {day_s}, which is no day") from e
        if rule in out:
            raise GateClassError(f"{where}: the {rule} waiver is listed twice")
        out[rule] = Waiver(day, reason.strip())
    return out


def load_budget(root: str | os.PathLike) -> dict[str, BudgetRow]:
    """tools/family-budget.tsv: the family, its real_s and unrowed (written by `budget --refresh`), its budget_s and its waivers. Which
    families have a row is `check`'s, not the loader's."""
    root = Path(root)
    path = root / BUDGET_TABLE
    if not path.exists():
        raise GateClassError(f"{BUDGET_TABLE} does not exist under {root}")
    rows: dict[str, BudgetRow] = {}
    for n, f in _rows(path):
        where = _where(path, n)
        if len(f) != 5:
            raise GateClassError(f"{where}: a row is not `family, real_s, unrowed, budget_s, waivers` (tab-separated): {'\\t'.join(f)[:100]!r}")
        name, real, unrowed, budget, waivers = (x.strip() for x in f)
        if not name:
            raise GateClassError(f"{where}: a row has no family name")
        nums: dict[str, int] = {}
        for label, v in (("real_s", real), ("unrowed", unrowed), ("budget_s", budget)):
            if not v.isdigit():
                raise GateClassError(f"{where}: `{label}` is not a whole number: {v!r}")
            nums[label] = int(v)
        if nums["budget_s"] == 0:
            raise GateClassError(f"{where}: the family {name} has a budget_s of 0")
        if name in rows:
            raise GateClassError(f"{where}: the family {name} is listed twice")
        rows[name] = BudgetRow(name, nums["real_s"], nums["unrowed"], nums["budget_s"], parse_waivers(waivers, where))
    return rows


def family_costs(tables: Tables, classes: dict[str, tuple[str, str]], scope_of: Callable[[str], GateScope],
                 plain: dict[str, list[tuple[str, int]]]) -> dict[str, FamilyCost]:
    """The real-tier cost of every family of families.tsv, and under the key '' of the gates no family owns: the recipes a real-tier
    batch runs (recipes.real_tier_candidates) minus the by-hand ones, which no batch runs unattended, each by the family its scope names;
    a gate costs the median of the last five real-tier rows of `plain` (recipes.times_rows) and one with none counts in `unrowed`.
    Every family is in the result, a family with no real-tier gate at zero."""
    owned: dict[str, list[str]] = {fam: [] for fam in tables.families}
    owned[""] = []
    for n in recipes.real_tier_candidates(classes):
        if classes[n][1] == "needs-env":
            continue
        fam = scope_of(n).family
        owned[fam if fam in tables.families else ""].append(n)
    out: dict[str, FamilyCost] = {}
    for fam, names in owned.items():
        w = recipes.predicted_wall(names, {}, plain, default=0)
        out[fam] = FamilyCost(tuple(names), w["sum"], w["unrowed"])
    return out


def measured_costs(root: Path, tables: Tables, times: str, classes_file: str | None) -> dict[str, FamilyCost]:
    """family_costs of the tree at ROOT from the times file; a missing or empty file is a named error, never a table of zeros."""
    if not os.path.exists(times):
        raise GateClassError(f"the times file {times} does not exist (--times FILE; the lead's machine holds gate-times.tsv)")
    _fixture, plain = recipes.times_rows(times)
    if not plain:
        raise GateClassError(f"the times file {times} holds no real-tier row")
    classes = recipes.read_class_file(classes_file) if classes_file else recipes.fixture_class_rows(str(root))
    rs = recipes.load_justfile(str(root / "justfile"))
    return family_costs(tables, classes, recipes.gate_scope_of(tables, rs), plain)


def refresh_text(text: str, families: Iterable[str], costs: dict[str, FamilyCost]) -> tuple[str, list[str]]:
    """(the new text of the budget table, one line per row it changed or added): each data row's real_s and unrowed from `costs`, the rest
    of the row and every comment as they were, and a row at the end (the default budget, no waiver) for each of `families` that has none."""
    out: list[str] = []
    seen: set[str] = set()
    changes: list[str] = []
    for line in text.splitlines():
        f = line.split("\t")
        if not line.strip() or line.startswith("#") or len(f) != 5 or f[0].strip() not in costs:
            out.append(line)
            continue
        name = f[0].strip()
        seen.add(name)
        c = costs[name]
        if (f[1].strip(), f[2].strip()) != (str(c.real_s), str(c.unrowed)):
            changes.append(f"{name}: real_s {f[1].strip()} -> {c.real_s}, unrowed {f[2].strip()} -> {c.unrowed}")
        out.append("\t".join([f[0], str(c.real_s), str(c.unrowed), f[3], f[4]]))
    for fam in families:
        if fam not in seen and fam in costs:
            c = costs[fam]
            out.append(f"{fam}\t{c.real_s}\t{c.unrowed}\t{DEFAULT_BUDGET_S}\t-")
            changes.append(f"{fam}: a new row, real_s {c.real_s}, unrowed {c.unrowed}, budget_s {DEFAULT_BUDGET_S}")
    return "\n".join(out) + "\n", changes


def budget_findings(tables: Tables, budgets: dict[str, BudgetRow]) -> tuple[list[str], list[str]]:
    """(the findings of the budget rule, its notes): one line each. A finding refuses; a note names a waiver nothing needs any more."""
    findings: list[str] = []
    notes: list[str] = []
    t = BUDGET_TABLE.name
    gates: dict[str, list[str]] = {}
    ships: dict[str, list[str]] = {}
    for n, row in sorted(tables.gates.items()):
        if row.cls == "family":
            gates.setdefault(row.family, []).append(n)
        try:
            mark = recipes.release_mark(row)
        except recipes.RecipeError as e:
            findings.append(str(e))
            continue
        if mark == "ship":
            ships.setdefault(row.family, []).append(n)
    for fam in sorted(set(budgets) - set(tables.families)):
        findings.append(f"{t} names the family {fam}, which families.tsv does not hold")
    for fam in sorted(tables.families):
        row = budgets.get(fam)
        if row is None:
            findings.append(f"{t} has no row for the family {fam}: `python3 tools/gate_class.py budget --refresh` adds it with the default "
                            f"budget of {DEFAULT_BUDGET_S} s")
            continue
        fr, w = tables.families[fam], row.waivers
        # The budget: a family's real-tier gates sum to at most budget_s, and a budget over the default has its dated reason.
        if row.real_s > row.budget_s:
            findings.append(f"{t}: the family {fam}'s real-tier gates sum to {row.real_s} s, over its budget_s of {row.budget_s} s: cut them to one "
                            "load of the file with the clauses in it (and none a common gate pins), or raise budget_s with a dated `budget` waiver")
        if row.budget_s > DEFAULT_BUDGET_S and "budget" not in w:
            findings.append(f"{t}: the family {fam}'s budget_s is {row.budget_s} s, over the default {DEFAULT_BUDGET_S} s, with no `budget` "
                            "waiver (`budget=PIN(YYYY-MM-DD): reason`)")
        if "budget" in w and row.budget_s <= DEFAULT_BUDGET_S:
            notes.append(f"{t}: the family {fam}'s `budget` waiver is not needed: its budget_s is {row.budget_s} s; drop it")
        # One +ship gate: the family's serve gate.
        mine = ships.get(fam, [])
        if len(mine) > 1 and "ship" not in w:
            findings.append(f"{t}: the family {fam} has {len(mine)} +ship gates ({' '.join(mine)}): a family blocks a release with one, its serve "
                            "gate, unless a dated `ship` waiver says why")
        if "ship" in w and len(mine) <= 1:
            notes.append(f"{t}: the family {fam}'s `ship` waiver is not needed: it has {len(mine)} +ship gate(s); drop it")
        # A fixture row: a landing runs the family on the fixture tier. A family that predates the rule is grandfathered by name.
        if not fr.fixture and gates.get(fam) and "fixture" not in w:
            findings.append(f"{t}: the family {fam} has {len(gates[fam])} gate(s) and no fixture (families.tsv fixture no): a new family has a "
                            f"fixture row (tools/fixture-dirs.tsv), since a landing runs it on the fixture tier; only a family that predates "
                            f"{RULE_DAY} is grandfathered, by a dated `fixture` waiver")
        if "fixture" in w and w["fixture"].day > RULE_DAY:
            findings.append(f"{t}: the family {fam}'s `fixture` waiver is dated {w['fixture'].day}, after the rule's day {RULE_DAY}: a waiver "
                            "grandfathers a family that predates the rule, and a newer family has a fixture row")
        if "fixture" in w and fr.fixture:
            notes.append(f"{t}: the family {fam}'s `fixture` waiver is not needed: families.tsv says fixture yes; drop it")
        # Tier main is the maintainer's call, recorded.
        if fr.tier == "main" and "main" not in w:
            findings.append(f"{t}: the family {fam} is tier main with no `main` waiver: main is the maintainer's call, recorded as "
                            f"`main=PIN(YYYY-MM-DD): main by the {MAIN_CALL}`")
        if "main" in w and MAIN_CALL not in w["main"].reason:
            findings.append(f"{t}: the family {fam}'s `main` waiver does not name the {MAIN_CALL}'s call: {w['main'].reason!r}")
        if "main" in w and fr.tier != "main":
            notes.append(f"{t}: the family {fam}'s `main` waiver is not needed: its tier is {fr.tier}; drop it")
    return findings, notes


def shared_loads(recs: dict[str, recipes.Recipe], tables: Tables) -> dict[str, list[tuple[str, list[str], list[str]]]]:
    """family -> [(profile, bins, gates)] for every real file (a BLOOMERY_MODEL profile a recipe's box call names) that the card gates of a
    family run in more than one gate_* bin: gates that could be one bin with the clauses in it, each loading the file again. Report only."""
    seen: dict[str, dict[str, tuple[set[str], list[str]]]] = {}
    for n, row in sorted(tables.gates.items()):
        rec = recs.get(n)
        if row.cls != "family" or rec is None:
            continue
        try:
            rc = recipes.recipe_commands(rec)
            profiles = recipes.RecipeCommands(models=[m for m in rc.models if m]).profiles()
        except recipes.RecipeError:
            continue
        bins = {b for inv in rc.invocations if inv.oxide for kind, b in inv.selectors if kind == "bin" and b and b.startswith(GATE_BIN)}
        if not bins:
            continue
        for p in profiles:
            have_bins, have_gates = seen.setdefault(row.family, {}).setdefault(p, (set(), []))
            have_bins |= bins
            have_gates.append(n)
    return {fam: [(p, sorted(bs), gs) for p, (bs, gs) in sorted(by.items()) if len(bs) > 1] for fam, by in sorted(seen.items())
            if any(len(bs) > 1 for bs, _ in by.values())}


def budget_lines(tables: Tables, budgets: dict[str, BudgetRow]) -> list[str]:
    """The budget table as it stands: per family (the larger real_s first) its tier, gate rows, real_s, gates with no row, budget_s and waivers."""
    cnt: dict[str, int] = {}
    for row in tables.gates.values():
        if row.family:
            cnt[row.family] = cnt.get(row.family, 0) + 1
    out = [f"{'family':<18} {'tier':<5} {'gates':>5} {'real_s':>7} {'unrowed':>7} {'budget_s':>8}  waivers"]
    for fam in sorted(budgets, key=lambda f: (-budgets[f].real_s, f)):
        b = budgets[fam]
        tier = tables.families[fam].tier if fam in tables.families else "?"
        over = " OVER" if b.real_s > b.budget_s else ""
        out.append(f"{fam:<18} {tier:<5} {cnt.get(fam, 0):>5} {b.real_s:>7} {b.unrowed:>7} {b.budget_s:>8}{over}  {' '.join(sorted(b.waivers)) or '-'}")
    over_default = sorted(f for f, b in budgets.items() if b.real_s > DEFAULT_BUDGET_S)
    out.append(f"{len(over_default)} family(ies) over the default {DEFAULT_BUDGET_S} s: {' '.join(over_default) or 'none'}")
    return out


def release_budget_lines(tables: Tables, budgets: dict[str, BudgetRow], costs: dict[str, FamilyCost], new: set[str]) -> list[str]:
    """`release-list --wall`'s per-family table: the table's real_s and budget_s beside this run's medians (`live`), the families the real
    tier holds first (`in`: main, or new since the base), a family over its budget or whose table has drifted flagged. The key '' of
    `costs` is the gates no family owns."""
    rows: list[tuple[int, str, str, int, int, int, int, int, str]] = []
    for fam, c in costs.items():
        b = budgets.get(fam)
        tier = tables.families[fam].tier if fam in tables.families else ""
        where = "always" if not fam else ("main" if tier == "main" else ("new" if fam in new else "-"))
        real_s, budget_s = (b.real_s, b.budget_s) if b is not None else (0, 0)
        flags = []
        if fam and b is None:
            flags.append("no budget row")
        elif b is not None and c.real_s > b.budget_s:
            flags.append("OVER budget")
        if b is not None and abs(c.real_s - b.real_s) > max(STALE_MIN_S, STALE_FRAC * max(c.real_s, b.real_s)):
            flags.append("table stale: budget --refresh")
        rows.append((0 if where != "-" else 1, fam or "(common, tooling)", where, len(c.gates), real_s, c.real_s, c.unrowed, budget_s, ", ".join(flags)))
    rows.sort(key=lambda r: (r[0], -r[5], r[1]))
    out = [f"  {'family':<18} {'in':<6} {'gates':>5} {'real_s':>7} {'live':>6} {'unrowed':>7} {'budget_s':>8}"]
    for _, name, where, n, real_s, live, unrowed, budget_s, flags in rows:
        out.append(f"  {name:<18} {where:<6} {n:>5} {real_s if budget_s else '-':>7} {live:>6} {unrowed:>7} {budget_s or '-':>8}{('  ' + flags) if flags else ''}")
    inside = [r for r in rows if r[0] == 0 and r[1] != "(common, tooling)"]
    out.append(f"  the real tier holds {len(inside)} famil(ies) ({', '.join(r[1] + ' ' + r[2] for r in inside) or 'none'}): {sum(r[5] for r in inside)} s "
               f"of live medians; a family new in a release adds at most its budget_s ({DEFAULT_BUDGET_S} s unless its row says more) to this list "
               "and one +ship gate to the blocking set, and leaves the real tier at the release after")
    return out


# ---------------------------------------------------------------------------------------------- the summary


def summary(tables: Tables) -> str:
    """The counts per class, per family and rep, and per owner."""
    per_class: dict[str, int] = {c: 0 for c in CLASSES}
    per_family: dict[str, int] = {f: 0 for f in tables.families}
    per_owner: dict[str, int] = {o: 0 for o in tables.owners}
    for row in tables.gates.values():
        per_class[row.cls] += 1
        if row.family:
            per_family[row.family] += 1
        for o in row.owners:
            per_owner[o] += 1
    out = ["per class:"]
    for c in CLASSES:
        out.append(f"  {c:<8} {per_class[c]:>4}")
    out.append("per family (rep, tier, gates):")
    for f in sorted(tables.families):
        out.append(f"  {f:<16} {tables.families[f].rep}  {tables.families[f].tier:<5} {per_family[f]:>4}")
    out.append("per owner (gates):")
    for o in sorted(per_owner):
        out.append(f"  {o:<20} {per_owner[o]:>4}")
    without = [f for f in sorted(tables.families) if not per_family[f]]
    if without:
        out.append("families with no gate of their own: " + " ".join(without))
    hand = sorted(n for n, row in tables.gates.items() if row.by_hand)
    if hand:
        out.append(f"{BY_HAND} gates ({len(hand)}): " + " ".join(hand))
    return "\n".join(out)


# ---------------------------------------------------------------------------------------------- the self-test


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _tree(tmp: Path) -> Path:
    """A small tree with two gate recipes, two families, one owner and the files the globs name."""
    root = tmp / "tree"
    _write(root / "justfile", "gate-a:\n    echo a\n\ngate-b:\n    echo b\n\nother:\n    echo o\n")
    _write(root / "tools" / "arch-registry.tsv", "# the registry's rows\nnames\tvariant\tread\tdraft\tfixture\nfam-a\tFamA\tno\tno\t-\nfam-b,FamB\tFamB\tyes\tno\t-\n")
    _write(root / "crates" / "fam-a" / "a.rs", "// a\n")
    _write(root / "crates" / "fam-b" / "b.rs", "// b\n")
    for g in DECLARED_GROUPS:
        _write(root / "crates" / g / "g.rs", "// g\n")
    _write(root / "tools" / "owners.tsv", "# owner\tkind\tglobs\nown-a\tkernel\tcrates/fam-a/a.rs\n")
    groups = "".join(f"{g}\tF\tother\tcrates/{g}/**\t30\tno\n" for g in DECLARED_GROUPS)
    _write(root / "tools" / "families.tsv",
           "# family\trep\ttier\tglobs\trotation_days\tfixture\n"
           f"fam-a\tR\tmain\tcrates/fam-a/**\t7\tno\n"
           f"fam-b\tF\tother\tcrates/fam-b/**\t30\tyes\n" + groups)
    _write(root / "tools" / "gate-class.tsv",
           "# recipe\tclass\tfamily\ttopic\towners\n"
           "gate-a\tfamily\tfam-a\tkernel\town-a\n"
           "gate-b\tcommon\t\thost\t-\n")
    _write(root / BUDGET_TABLE, _budget_table(f"fam-a\t10\t0\t300\t{_waiver('main', 'main by the maintainer')}{WAIVER_SEP}{_waiver('fixture')}", "fam-b\t20\t1\t300\t-",
                                                *(f"{g}\t0\t0\t300\t-" for g in DECLARED_GROUPS)))
    _write(root / recipes.FIXTURE_DIRS, "# arch\tdir\nfam-b\tfamb\nqwen38d\tself\nprof-cut\tcutdir\n")
    _write(root / recipes.REF_PATHS, ': "${BLOOMERY_MODEL:=${BLOOMERY_REF_MODEL_PROFILE:-deflt}}"\n')
    return root


def no_kernels(root: Path, tables: Tables) -> KernelReach:
    """A tree with no kernel file: the synthetic trees of the self-test have no crates for the reader to walk."""
    return KernelReach({}, {})


def kernel_self_test(expect: Callable[[bool, str], None]) -> None:
    """The kernel-file rule on synthetic tables: each clause that passes a file, and the two findings."""
    fams = {"fa": FamilyRow("fa", "R", "main", ("crates/fa/**",), 7, True), "fb": FamilyRow("fb", "F", "other", ("crates/fb/**",), 30, True),
            "fc": FamilyRow("fc", "F", "other", ("crates/fc/**", "crates/shared/c.rs"), 30, False)}
    gates = {"gate-a": GateRow("gate-a", "family", "fa", "e2e", ()), "gate-b": GateRow("gate-b", "family", "fb", "e2e", ()),
             "gate-b2": GateRow("gate-b2", "family", "fb", "e2e", ("own-f",)), "gate-c": GateRow("gate-c", "family", "fc", "e2e", ()),
             "gate-k": GateRow("gate-k", "common", "", "kernel", ("own-k",))}
    owners = {"own-k": OwnerRow("own-k", "kernel", ("crates/shared/k.rs",)), "own-f": OwnerRow("own-f", "kernel", ("crates/shared/f.rs",))}
    t = Tables(gates, fams, owners)

    def red(launchers: dict[str, dict[str, str]], skipped: dict[str, str] | None = None) -> list[str]:
        return kernel_findings(t, KernelReach(launchers, skipped or {}))

    # The q6k_heads shape: a shared file only F family gates launch, named by one family's globs and not the other's.
    found = red({"crates/shared/c.rs": {"gate-b": DEP, "gate-c": DEP}})
    expect(len(found) == 1 and "crates/shared/c.rs" in found[0] and "fb (gate-b)" in found[0] and "fc" not in found[0].split("launched by")[1].split("and those")[0],
           f"kernel rule: a file F family gates launch and one of their families' globs omits is red, naming that family alone: {found}")
    expect(red({"crates/shared/c.rs": {"gate-c": DEP}}) == [], "kernel rule: every launching F family names the file: green")
    expect(red({"crates/fb/x.rs": {"gate-b": DEP, "gate-c": DEP}}) and red({"crates/fb/x.rs": {"gate-b": DEP}}) == [],
           "kernel rule: a family's own file another F family launches is red; launched by its own family alone it is green")
    expect(red({"crates/shared/c.rs": {"gate-b": DEP, "gate-a": DEP}}) == [], "kernel rule: an R family gate launching it passes the file")
    expect(red({"crates/shared/c.rs": {"gate-b": DEP, "gate-x": OWN}}) == [], "kernel rule: a file of a gate's own target passes")
    expect(red({"crates/shared/k.rs": {"gate-b": DEP}}) == [], "kernel rule: an owner a common gate names passes the file")
    found = red({"crates/shared/f.rs": {"gate-c": DEP}})
    expect(found and "fc" in found[0], f"kernel rule: an owner only a family gate names, whose family's globs omit the file, does not pass it: {found}")
    t2 = Tables(gates, {**fams, "fb": FamilyRow("fb", "F", "other", ("crates/fb/**", "crates/shared/f.rs"), 30, True)}, owners)
    expect(kernel_findings(t2, KernelReach({"crates/shared/f.rs": {"gate-c": DEP}}, {})) == [],
           "kernel rule: an owner a family gate names passes the file when that family's globs name it (its gate is kept)")
    found = red({"crates/shared/z.rs": {}})
    expect(len(found) == 1 and "no gate launches it" in found[0], f"kernel rule: a kernel file no gate launches is red: {found}")
    expect(red({"crates/shared/z.rs": {}}, {"crates/shared/z.rs": "probe: no gate links it"}) == [], "kernel rule: a CARRIER_SKIP bin's file is not")
    expect(red({"crates/shared/z.rs": {"gate-k": DEP}}) != [], "kernel rule: a common gate's reach is no launch (a lib's tests reach every file)")


def _budget_table(*rows: str) -> str:
    return "# family\treal_s\tunrowed\tbudget_s\twaivers\n" + "".join(r + "\n" for r in rows)


def _waiver(rule: str, reason: str = "why", day: str | datetime.date = RULE_DAY) -> str:
    return f"{rule}=PIN({day}): {reason}"


def budget_self_test(expect: Callable[[bool, str], None], tmp: Path) -> None:
    """The budget rule on synthetic tables: each refusal on a table that breaks it (and the table that mends it green), the notes, the loader's
    named errors, the refresh and the two reports."""
    import contextlib
    import inspect
    import io

    def raises(fn, text: str) -> bool:
        try:
            fn()
        except GateClassError as e:
            return text in str(e)
        return False

    expect(UNROWED_S == inspect.signature(recipes.predicted_wall).parameters["default"].default,
           "budget: UNROWED_S is the price recipes.predicted_wall gives a gate with no real-tier row")

    # Three families: fm (main, two +ship gates), fo (an ordinary one) and fn (no fixture, one gate).
    fams = {"fm": FamilyRow("fm", "R", "main", ("crates/fm/**",), 7, True), "fo": FamilyRow("fo", "F", "other", ("crates/fo/**",), 30, True),
            "fn": FamilyRow("fn", "F", "other", ("crates/fn/**",), 30, False)}
    gates = {"gate-m-serve": GateRow("gate-m-serve", "family", "fm", "serve+ship", ()), "gate-m-two": GateRow("gate-m-two", "family", "fm", "twocard+ship", ()),
             "gate-o": GateRow("gate-o", "family", "fo", "e2e", ()), "gate-n": GateRow("gate-n", "family", "fn", "e2e", ()),
             "gate-c": GateRow("gate-c", "common", "", "kernel+kernel", ())}
    t = Tables(gates, fams, {})
    tier_main = _waiver("main", "main by the maintainer")

    def good() -> dict[str, BudgetRow]:
        return {"fm": BudgetRow("fm", 100, 0, 300, parse_waivers(WAIVER_SEP.join([tier_main, _waiver("ship")]), "t")),
                "fo": BudgetRow("fo", 100, 0, 300, {}),
                "fn": BudgetRow("fn", 100, 0, 300, parse_waivers(_waiver("fixture"), "t"))}

    def with_row(fam: str, real: int | None = None, budget: int | None = None, waivers: str | None = None, tables: Tables = t,
                 base: dict[str, BudgetRow] | None = None) -> tuple[list[str], list[str]]:
        rows = dict(base if base is not None else good())
        r = rows[fam]
        rows[fam] = BudgetRow(fam, r.real_s if real is None else real, r.unrowed, r.budget_s if budget is None else budget,
                              r.waivers if waivers is None else parse_waivers(waivers, "t"))
        return budget_findings(tables, rows)

    def names(found: list[str], *words: str) -> bool:
        return len(found) == 1 and all(w in found[0] for w in words)

    expect(budget_findings(t, good()) == ([], []), f"budget: the three families with their waivers are green with no note: {budget_findings(t, good())}")

    # The budget: real_s over budget_s is red at 301 and green at 300; a budget over 300 carries its waiver, and real_s stays under it.
    f, _ = with_row("fo", real=301)
    expect(names(f, "family fo", "301 s", "budget_s of 300 s"), f"budget: real_s over budget_s is red, naming the family: {f}")
    expect(with_row("fo", real=300)[0] == [], "budget: real_s equal to budget_s is green")
    f, _ = with_row("fo", budget=500)
    expect(names(f, "family fo", "no `budget` waiver"), f"budget: a budget_s over the default with no waiver is red: {f}")
    expect(with_row("fo", real=500, budget=500, waivers=_waiver("budget"))[0] == [], "budget: a budget over the default with its dated waiver is green")
    f, _ = with_row("fo", real=501, budget=500, waivers=_waiver("budget"))
    expect(names(f, "family fo", "501 s"), f"budget: a family over a waived budget is red all the same: {f}")
    f, n = with_row("fo", waivers=_waiver("budget"))
    expect(f == [] and len(n) == 1 and "`budget` waiver is not needed" in n[0], f"budget: a budget waiver on a budget at the default is a note, not a finding: {f} {n}")

    # One +ship gate: two are red with no waiver, naming them; one is green, and a waiver then is a note.
    f, _ = with_row("fm", waivers=tier_main)
    expect(names(f, "family fm", "2 +ship gates", "gate-m-serve gate-m-two"), f"budget: a second +ship gate with no waiver is red, naming both: {f}")
    one = Tables({k: v for k, v in gates.items() if k != "gate-m-two"}, fams, {})
    f, n = with_row("fm", tables=one)
    expect(f == [] and len(n) == 1 and "`ship` waiver is not needed" in n[0], f"budget: one +ship gate is green and its waiver a note: {f} {n}")
    bad_mark = Tables({**gates, "gate-m-two": GateRow("gate-m-two", "family", "fm", "twocard+sideways", ())}, fams, {})
    f, _ = with_row("fm", tables=bad_mark)
    expect(any("sideways" in x for x in f), f"budget: a release mark the table does not take is a finding, not a crash: {f}")
    ship_common = Tables({**gates, "gate-c": GateRow("gate-c", "common", "", "kernel+ship", ())}, fams, {})
    f, _ = with_row("fm", tables=ship_common)
    expect(len(f) == 1 and "gate-c" in f[0] and "+ship marks one family's" in f[0], f"budget: a +ship on a row that names no family is a finding: {f}")

    # A fixture: a family with gates and fixture no is red with no waiver and green with one dated on or before the rule's day, red after it;
    # a family with no gate needs none; a waiver on a family that has a fixture is a note.
    f, _ = with_row("fn", waivers="-")
    expect(names(f, "family fn", "1 gate(s) and no fixture", "families.tsv fixture no"), f"budget: gates and no fixture, no waiver, is red: {f}")
    nogate = Tables({k: v for k, v in gates.items() if k != "gate-n"}, fams, {})
    expect(with_row("fn", waivers="-", tables=nogate)[0] == [], "budget: a family with no gate needs no fixture waiver")
    expect(with_row("fn", waivers=_waiver("fixture", day=RULE_DAY - datetime.timedelta(days=30)))[0] == [], "budget: a fixture waiver dated before the rule's day is green")
    f, _ = with_row("fn", waivers=_waiver("fixture", day=RULE_DAY + datetime.timedelta(days=1)))
    expect(names(f, "family fn", "after the rule's day", str(RULE_DAY)), f"budget: a fixture waiver dated after the rule's day is red (a new family has a fixture): {f}")
    f, n = with_row("fo", waivers=_waiver("fixture"))
    expect(f == [] and len(n) == 1 and "`fixture` waiver is not needed" in n[0], f"budget: a fixture waiver on a family with a fixture is a note: {f} {n}")

    # Tier main: red with no waiver, red when the waiver does not name the maintainer's call; a waiver on another tier is a note.
    f, _ = with_row("fm", waivers=_waiver("ship"))
    expect(names(f, "family fm", "tier main", "no `main` waiver"), f"budget: tier main with no waiver is red: {f}")
    f, _ = with_row("fm", waivers=WAIVER_SEP.join([_waiver("main", "it is big"), _waiver("ship")]))
    expect(names(f, "family fm", f"does not name the {MAIN_CALL}"), f"budget: a main waiver that names no maintainer's call is red: {f}")
    f, n = with_row("fo", waivers=tier_main)
    expect(f == [] and len(n) == 1 and "`main` waiver is not needed" in n[0], f"budget: a main waiver on an other family is a note: {f} {n}")

    # The rows against families.tsv: a family with no row and a row with no family are red.
    rows = good()
    del rows["fo"]
    f, _ = budget_findings(t, rows)
    expect(names(f, "no row for the family fo", "budget --refresh"), f"budget: a family with no row is red, naming the fix: {f}")
    rows = good()
    rows["fz"] = BudgetRow("fz", 1, 0, 300, {})
    f, _ = budget_findings(t, rows)
    expect(names(f, "names the family fz", "families.tsv does not hold"), f"budget: a row of no family is red: {f}")

    # The loader: its named errors, with the file and the line.
    root = _tree(tmp / "budget-load")
    path = root / BUDGET_TABLE

    def loaded(*rows: str):
        _write(path, _budget_table(*rows))
        return lambda: load_budget(root)

    expect(raises(loaded("fam-a\t1\t0\t300"), "family-budget.tsv:2: a row is not"), "budget load: a short row is a named error with the file and the line")
    expect(raises(loaded("fam-a\tx\t0\t300\t-"), "`real_s` is not a whole number"), "budget load: a real_s that is no number is a named error")
    expect(raises(loaded("fam-a\t1\t-1\t300\t-"), "`unrowed` is not a whole number"), "budget load: a negative unrowed is a named error")
    expect(raises(loaded("fam-a\t1\t0\t0\t-"), "budget_s of 0"), "budget load: a budget of 0 is a named error")
    expect(raises(loaded("fam-a\t1\t0\t300\t-", "fam-a\t1\t0\t300\t-"), "the family fam-a is listed twice"), "budget load: a family twice is a named error")
    expect(raises(loaded("fam-a\t1\t0\t300\tmain by the maintainer"), "a waiver is `rule=PIN(YYYY-MM-DD): reason`"), "budget load: a waiver with no rule and no PIN is a named error")
    expect(raises(loaded("fam-a\t1\t0\t300\tgate=PIN(2026-11-14): why"), "a waiver is `rule=PIN(YYYY-MM-DD): reason`"), "budget load: a rule the table does not take is a named error")
    expect(raises(loaded("fam-a\t1\t0\t300\tmain=PIN(2026-13-45): why"), "dated 2026-13-45, which is no day"), "budget load: a waiver dated on no day is a named error")
    expect(raises(loaded("fam-a\t1\t0\t300\tmain=PIN(2026-11-14): a | main=PIN(2026-11-14): b"), "the main waiver is listed twice"), "budget load: a rule twice in one row is a named error")
    got = loaded("fam-a\t1\t2\t300\tmain=PIN(2026-11-14): by the maintainer | fixture=PIN(2026-11-01): predates")()
    expect(got["fam-a"] == BudgetRow("fam-a", 1, 2, 300, {"main": Waiver(RULE_DAY, "by the maintainer"), "fixture": Waiver(datetime.date(2026, 11, 1), "predates")}),
           f"budget load: a row and its two waivers parse: {got}")
    path.unlink()
    expect(raises(lambda: load_budget(root), "family-budget.tsv does not exist"), "budget load: a missing table is a named error")

    # The costs: a gate costs the median of its last five real-tier rows, a gate with none is unrowed and costs nothing, a by-hand gate and a
    # recipe the real tier does not run are no cost, a gate no family owns is the key '' and a family with no gate is zero.
    classes = {"gate-o": ("A", "plain"), "gate-o2": ("B", "plain"), "gate-o3": ("A", "plain"), "gate-env": ("D", "needs-env"), "gate-m-serve": ("X", "plain"),
               "gate-fixture": ("D", "fixture-only"), "gate-c": ("A", "plain"), "weekly-w": ("A", "plain")}
    scope = {"gate-o": "fo", "gate-o2": "fo", "gate-o3": "fo", "gate-env": "fo", "gate-m-serve": "fm", "gate-fixture": "fo", "gate-c": "", "weekly-w": "fn"}
    plain = {"gate-o": [("A", 900)] + [("A", s) for s in (10, 50, 30, 20, 40)], "gate-o2": [("B", 7)], "gate-m-serve": [("X", 11), ("X", 13)],
             "gate-c": [("A", 5)], "gate-env": [("A", 999)], "weekly-w": [("A", 60)]}
    costs = family_costs(t, classes, lambda n: GateScope(scope[n], "other", 30), plain)
    expect(costs["fo"] == FamilyCost(("gate-o", "gate-o2", "gate-o3"), 37, 1),
           f"costs: fo is the median of the last five rows of gate-o (30), gate-o2's one row (7) and one gate with none: {costs['fo']}")
    expect(costs["fm"] == FamilyCost(("gate-m-serve",), 12, 0) and costs["fn"] == FamilyCost(("weekly-w",), 60, 0),
           f"costs: a gate's median rounds half up, and a weekly recipe belongs to its profile's family: {costs['fm']} {costs['fn']}")
    expect(costs[""] == FamilyCost(("gate-c",), 5, 0), f"costs: a gate no family owns is the key '': {costs['']}")
    expect(costs["fo"].priced_s == 37 + UNROWED_S, "costs: the priced cost is real_s and each unrowed gate at the price")
    expect("gate-env" not in costs["fo"].gates and "gate-fixture" not in costs["fo"].gates, "costs: a by-hand gate and a fixture-only one are no real-tier cost")
    expect(set(costs) == {*fams, ""} and all(isinstance(c, FamilyCost) for c in costs.values()), f"costs: every family is in the result: {sorted(costs)}")

    # The refresh: real_s and unrowed rewritten, every comment and every other field kept, a family with no row appended; the second run changes nothing.
    old = _budget_table("# a note that stays", "fo\t1\t9\t450\tbudget=PIN(2026-11-14): kept", "fm\t12\t0\t300\t-", "fz\t3\t0\t300\t-")
    text, changes = refresh_text(old, list(fams), costs)
    lines = text.splitlines()
    expect("# a note that stays" in lines and "fo\t37\t1\t450\tbudget=PIN(2026-11-14): kept" in lines, f"refresh: a row's real_s and unrowed move, its budget_s and waivers stay: {lines}")
    expect("fm\t12\t0\t300\t-" in lines and "fz\t3\t0\t300\t-" in lines, "refresh: a row already right and a row of no family are kept as they were")
    expect(f"fn\t60\t0\t{DEFAULT_BUDGET_S}\t-" in lines and len(changes) == 2, f"refresh: a family with no row gets one at the default budget: {changes}")
    again, again_changes = refresh_text(text, list(fams), costs)
    expect(again == text and again_changes == [], "refresh: a second refresh from the same times changes nothing")

    # The command, end to end on a tree: --refresh writes the table, prints what moved and the gates with no row, and refuses a times file that
    # is missing or holds no real-tier row (never a table of zeros).
    root = _tree(tmp / "budget-cmd")
    _write(root / "tools" / "gate-class.tsv", "# recipe\tclass\tfamily\ttopic\towners\ngate-a\tfamily\tfam-a\tkernel\town-a\ngate-b\tcommon\t\thost\t-\n")
    cls_file, times_file = tmp / "budget-cmd.classes", tmp / "budget-cmd.times"
    _write(cls_file, "gate-a\tA\tplain\tx\ngate-b\tA\tplain\tx\n")
    _write(times_file, "".join(f"gate-a\tA\tnone\t{s}\t2026-11-14T00:00:0{i}+0900\n" for i, s in enumerate((100, 300, 200))) + "gate-b\tB\tnone\t8\t2026-11-14T00:00:09+0900\n"
           + "gate-a@BLOOMERY_TIER=fixture\tA\tnone\t1\t2026-11-14T00:00:10+0900\n")
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rc = main(["budget", "--refresh", "--root", str(root), "--times", str(times_file), "--classes", str(cls_file)])
    shown = out.getvalue()
    refreshed = load_budget(root)
    expect(rc == 0 and refreshed["fam-a"].real_s == 200 and refreshed["fam-a"].budget_s == 300 and "main" in refreshed["fam-a"].waivers,
           f"budget --refresh: fam-a's real_s is the median of its three real-tier rows, its fixture row not read, its waivers kept: {rc} {refreshed['fam-a']}")
    expect("fam-a: real_s 10 -> 200" in shown and "fam-b: a new row" not in shown and "fam-a  " in shown,
           f"budget --refresh: the moved row and the table are printed: {shown}")
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        main(["budget", "--refresh", "--root", str(root), "--times", str(times_file), "--classes", str(cls_file)])
    expect("0 row(s)" in out.getvalue(), f"budget --refresh: a second run changes no row: {out.getvalue().splitlines()[:1]}")
    expect(raises(lambda: measured_costs(root, load(root), str(tmp / "no-such-times"), str(cls_file)), "does not exist"),
           "budget --refresh: a missing times file is a named error")
    _write(times_file, "gate-a@BLOOMERY_TIER=fixture\tA\tnone\t1\t2026-11-14T00:00:10+0900\n")
    expect(raises(lambda: measured_costs(root, load(root), str(times_file), str(cls_file)), "holds no real-tier row"),
           "budget --refresh: a times file with no real-tier row is a named error")

    # `check` holds the table: a tree whose table is right is green, one whose family is over its budget is red, one with no table is a named error.
    root = _tree(tmp / "budget-check")
    expect(check(root, load(root), no_kernels) == [], f"budget check: the tree's table is green: {check(root, load(root), no_kernels)}")
    _write(root / BUDGET_TABLE, (root / BUDGET_TABLE).read_text(encoding="utf-8").replace("fam-b\t20\t1\t300", "fam-b\t301\t1\t300"))
    found = check(root, load(root), no_kernels)
    expect(len(found) == 1 and "family fam-b" in found[0] and "301 s" in found[0], f"budget check: a family over its budget turns `check` red: {found}")
    (root / BUDGET_TABLE).unlink()
    expect(raises(lambda: check(root, load(root), no_kernels), "family-budget.tsv does not exist"), "budget check: a tree with no budget table is a named error")

    # The gates that load one real file in several bins: report only.
    _write(root / "justfile", "".join(
        f"gate-{n}:\n    {env}./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --release --bin {b} && bash tools/gpu-gate.sh {b}'\n\n"
        for n, env, b in (("x1", "BLOOMERY_MODEL=m1 ", "gate_one"), ("x2", "BLOOMERY_MODEL=m1 ", "gate_two"), ("x3", "BLOOMERY_MODEL=m2 ", "gate_one"),
                          ("x4", "", "gate_three"), ("x5", "BLOOMERY_MODEL=m1 ", "gate_one"), ("x6", "BLOOMERY_MODEL=m1 ", "bloomery-serve"))))
    sg = Tables({n: GateRow(n, "family", f, "e2e", ()) for n, f in (("gate-x1", "fx"), ("gate-x2", "fx"), ("gate-x3", "fx"), ("gate-x4", "fx"),
                                                                   ("gate-x5", "fy"), ("gate-x6", "fy"))}, {}, {})
    shared = shared_loads(recipes.load_justfile(str(root / "justfile")), sg)
    expect(shared == {"fx": [("m1", ["gate_one", "gate_two"], ["gate-x1", "gate-x2"])]},
           f"shared loads: a family's two gate bins on one profile are reported; one bin on a profile, a gate with no profile and a tool bin are not: {shared}")

    # The two reports.
    budgets = {"fm": BudgetRow("fm", 100, 0, 300, {}), "fo": BudgetRow("fo", 100, 0, 300, {}), "fn": BudgetRow("fn", 10, 0, 300, {})}
    live = {"fm": FamilyCost(("g1",), 100, 0), "fo": FamilyCost(("g2", "g3"), 350, 1), "fn": FamilyCost(("g4",), 500, 0), "": FamilyCost(("g5",), 70, 0)}
    text = "\n".join(release_budget_lines(t, budgets, live, {"fo"}))
    expect("OVER budget" in text and "table stale" in text and "(common, tooling)" in text, f"release table: a family over its budget and a stale table are flagged: {text}")
    row = {ln.split()[0]: ln for ln in text.splitlines()[1:-1]}
    expect(row["fm"].split()[1] == "main" and row["fo"].split()[1] == "new" and row["fn"].split()[1] == "-" and text.index("  fo ") < text.index("  fm ") < text.index("  fn "),
           f"release table: the families the real tier holds come first (the larger live cost first), tagged main or new: {text}")
    expect("OVER budget" in row["fo"] and "OVER budget" in row["fn"] and "OVER" not in row["fm"], f"release table: only the families whose live cost is over are flagged: {text}")
    lines = budget_lines(t, {**budgets, "fn": BudgetRow("fn", 400, 0, 450, {})})
    expect(lines[-1] == f"1 family(ies) over the default {DEFAULT_BUDGET_S} s: fn" and lines[1].startswith("fn "),
           f"budget table: the larger real_s first, and the last line counts the families over the default: {lines}")


def self_test() -> int:
    fails: list[str] = []

    def expect(cond: bool, msg: str) -> None:
        if not cond:
            fails.append(msg)

    def raises(fn, text: str) -> bool:
        try:
            fn()
        except (GateClassError, recipes.RecipeError) as e:
            return text in str(e)
        return False

    import tempfile

    with tempfile.TemporaryDirectory(prefix="gate-class-selftest-") as tmpdir:
        tmp = Path(tmpdir)
        root = _tree(tmp)
        tables = load(root)
        expect(sorted(tables.gates) == ["gate-a", "gate-b"], f"load: the tree's two gate rows: {sorted(tables.gates)}")
        expect(tables.gates["gate-a"].owners == ("own-a",), f"load: the owner of gate-a: {tables.gates['gate-a'].owners}")
        expect(tables.families["fam-b"].rotation_days == 30 and tables.families["fam-b"].tier == "other" and tables.families["fam-b"].fixture,
               "load: the family's tier, rotation days and fixture flag")
        expect(check(root, tables, no_kernels) == [], f"check: the tree's tables are green: {check(root, tables, no_kernels)}")

        # A malformed row, an unknown class, an unknown rep, an unknown owner, an unknown family, a duplicate row.
        gate_rows = (root / "tools" / "gate-class.tsv").read_text(encoding="utf-8").splitlines()
        fam_rows = (root / "tools" / "families.tsv").read_text(encoding="utf-8").splitlines()
        owner_rows = (root / "tools" / "owners.tsv").read_text(encoding="utf-8").splitlines()

        def with_gate(text: str):
            _write(root / "tools" / "gate-class.tsv", text)
            return lambda: load(root)

        def with_fam(text: str):
            _write(root / "tools" / "families.tsv", text)
            return lambda: load(root)

        def with_owner(text: str):
            _write(root / "tools" / "owners.tsv", text)
            return lambda: load(root)

        def fam_text(*rows: str) -> str:
            """The tree's families.tsv with these rows and the declared groups, which every tree must hold."""
            return "\n".join(["# h", *rows, *(f"{g}\tF\tother\tcrates/{g}/**\t30\tno" for g in DECLARED_GROUPS)]) + "\n"

        bad = "\n".join(["# h", "gate-a\tfamily\tfam-a\tkernel\town-a", "gate-b\tcommon\thost"]) + "\n"
        expect(raises(with_gate(bad), "gate-class.tsv:3: a row is not"), "load: a short row is a named error with the file and the line")
        bad = "\n".join(["# h", "gate-a\tfamily\tfam-a\tkernel\town-a", "gate-b\tsideways\t\thost\t-"]) + "\n"
        expect(raises(with_gate(bad), "unknown class 'sideways'"), "load: an unknown class is a named error")
        bad = "\n".join(["# h", "gate-a\tfamily\tfam-a\tkernel\town-a", "gate-b\tfamily\tfam-b\tkernel\town-z"]) + "\n"
        expect(raises(with_gate(bad), "the owner own-z is not in owners.tsv"), "load: an owner that is not in owners.tsv is a named error")
        bad = "\n".join(["# h", "gate-a\tfamily\tfam-z\tkernel\town-a"]) + "\n"
        expect(raises(with_gate(bad), "the family fam-z is not in families.tsv"), "load: a family that is not in families.tsv is a named error")
        bad = "\n".join(["# h", "gate-a\tfamily\tfam-a\tkernel\town-a", "gate-a\tcommon\t\thost\t-"]) + "\n"
        expect(raises(with_gate(bad), "the recipe gate-a is listed twice"), "load: a recipe twice is a named error")
        bad = "\n".join(["# h", "gate-a\tfamily\tfam-a\tkernel\town-a", "gate-b\tcommon\tfam-b\thost\t-"]) + "\n"
        expect(raises(with_gate(bad), "a `common` row names a family"), "load: a common row with a family is a named error")
        bad = "\n".join(["# h", "gate-a\tfamily\t\tkernel\town-a"]) + "\n"
        expect(raises(with_gate(bad), "a `family` row names no family"), "load: a family row with no family is a named error")
        bad = fam_text("fam-a\tR\tmain\tcrates/fam-a/**\t7\tno", "fam-b\tX\tother\tcrates/fam-b/**\t30\tyes")
        expect(raises(with_fam(bad), "unknown rep 'X'"), "load: an unknown rep is a named error")
        bad = fam_text("fam-a\tR\tmain\tcrates/fam-a/**\t7\tno", "fam-b\tF\tother\tcrates/fam-b/**\t0\tyes")
        expect(raises(with_fam(bad), "`rotation_days` is not a number of days"), "load: a bad rotation period is a named error")
        bad = fam_text("fam-a\tR\tmain\tcrates/fam-a/**\t7\tno", "fam-b\tF\tother\tcrates/fam-b/**\t30\tmaybe")
        expect(raises(with_fam(bad), "`fixture` is 'maybe'"), "load: a bad fixture flag is a named error")
        bad = "\n".join(["# h", "own-a\tkernel\tcrates/fam-a/a.rs", "own-a\tkernel\tcrates/fam-a/a.rs"]) + "\n"
        expect(raises(with_owner(bad), "the owner own-a is listed twice"), "load: an owner twice is a named error")
        bad = "\n".join(["# h", "own-a\tkernel"]) + "\n"
        expect(raises(with_owner(bad), "a row is not `owner, kind, globs`"), "load: a short owner row is a named error")

        # The checks: a gate with no row, a row that names no gate, a family the tables disagree on, a glob that
        # matches no file.
        _write(root / "tools" / "owners.tsv", "\n".join(owner_rows) + "\n")
        _write(root / "tools" / "families.tsv", "\n".join(fam_rows) + "\n")
        _write(root / "tools" / "gate-class.tsv", "\n".join(gate_rows) + "\n")
        good = check(root, load(root), no_kernels)
        expect(good == [], f"check: the restored tables are green: {good}")

        one = "\n".join(["# h", "gate-a\tfamily\tfam-a\tkernel\town-a"]) + "\n"
        _write(root / "tools" / "gate-class.tsv", one)
        found = check(root, load(root), no_kernels)
        expect(any("gate-b" in f for f in found), f"check: a gate with no row is red, naming it: {found}")
        two = "\n".join(["# h", "gate-a\tfamily\tfam-a\tkernel\town-a", "gate-b\tcommon\t\thost\t-", "gate-z\tcommon\t\thost\t-"]) + "\n"
        _write(root / "tools" / "gate-class.tsv", two)
        found = check(root, load(root), no_kernels)
        expect(any("gate-z" in f for f in found), f"check: a row that names no gate is red: {found}")
        three = "\n".join(["# h", "gate-a\tfamily\tfam-a\tkernel\town-a", "gate-b\tcommon\t\thost\t-",
                           "gate-c\tfamily\tfam-b\tkernel\t-"]) + "\n"
        _write(root / "tools" / "gate-class.tsv", three)
        found = check(root, load(root), no_kernels)
        expect(any("gate-c" in f for f in found), f"check: a row that names no gate is red: {found}")
        _write(root / "tools" / "gate-class.tsv", "\n".join(gate_rows) + "\n")

        # A family the two tables do not agree on, and a family the registry does not name.
        _write(root / "tools" / "families.tsv", "\n".join(fam_rows) + "\n")
        only_gate = "\n".join(["# h", "gate-a\tfamily\tfam-a\tkernel\town-a", "gate-b\tfamily\tfam-z\tkernel\t-"]) + "\n"
        _write(root / "tools" / "gate-class.tsv", only_gate)
        expect(raises(lambda: load(root), "the family fam-z is not in families.tsv"),
               "load: a family the gate table names and families.tsv does not is a named error")
        _write(root / "tools" / "gate-class.tsv", "\n".join(gate_rows) + "\n")
        no_fam = fam_text("fam-a\tR\tmain\tcrates/fam-a/**\t7\tno")
        _write(root / "tools" / "families.tsv", no_fam)
        found = check(root, load(root), no_kernels)
        expect(any("fam-b" in f for f in found), f"check: a family of the registry with no row is red: {found}")
        extra = fam_text("fam-a\tR\tmain\tcrates/fam-a/**\t7\tno", "fam-b\tF\tother\tcrates/fam-b/**\t30\tyes", "fam-z\tF\tother\tcrates/fam-z/**\t30\tno")
        _write(root / "tools" / "families.tsv", extra)
        found = check(root, load(root), no_kernels)
        expect(any("fam-z" in f for f in found), f"check: a family the registry does not name is red: {found}")
        _write(root / "tools" / "families.tsv", "\n".join(fam_rows) + "\n")

        # A glob that matches no file.
        nofile = "\n".join(["# h", "own-a\tkernel\tcrates/fam-a/nothing.rs"]) + "\n"
        _write(root / "tools" / "owners.tsv", nofile)
        found = check(root, load(root), no_kernels)
        expect(any("nothing.rs" in f for f in found), f"check: an owner glob that matches no file is red: {found}")
        _write(root / "tools" / "owners.tsv", "\n".join(owner_rows) + "\n")
        nofile = fam_text("fam-a\tR\tmain\tcrates/fam-a/**\t7\tno", "fam-b\tF\tother\tcrates/fam-b/nothing/**\t30\tyes")
        _write(root / "tools" / "families.tsv", nofile)
        found = check(root, load(root), no_kernels)
        expect(any("nothing" in f for f in found), f"check: a family glob that matches no file is red: {found}")
        _write(root / "tools" / "families.tsv", "\n".join(fam_rows) + "\n")
        expect(check(root, load(root), no_kernels) == [], "check: the restored tree is green again")

        # A by-hand gate is a recipe that calls tools/ref/needs-env.sh, in its text or through a dependency, and a row whose
        # topic is by-hand: the two sets are one.
        base_gates = "\n".join(gate_rows) + "\n"
        _write(root / "justfile", "gate-a:\n    echo a\n\ngate-b:\n    echo b\n\n"
                                  "gate-c:\n    ./tools/box.sh 'bash tools/ref/needs-env.sh gate-c NE_VAR && echo c'\n\n"
                                  "gate-d: gate-c\n    echo d\n\nother:\n    echo o\n")

        def hand_rows(c: str, d: str, b: str = "host") -> str:
            return base_gates.replace("gate-b\tcommon\t\thost", f"gate-b\tcommon\t\t{b}") + f"gate-c\tcommon\t\t{c}\t-\ngate-d\tcommon\t\t{d}\t-\n"

        _write(root / "tools" / "gate-class.tsv", hand_rows(BY_HAND, BY_HAND))
        found = check(root, load(root), no_kernels)
        expect(found == [], f"check: the two by-hand gates, one through a dependency, are green: {found}")
        expect(f"{BY_HAND} gates (2): gate-c gate-d" in summary(load(root)), "summary: the by-hand gates are listed by name")
        _write(root / "tools" / "gate-class.tsv", hand_rows("host", BY_HAND))
        found = check(root, load(root), no_kernels)
        expect(any("gate-c calls" in f and BY_HAND in f for f in found), f"check: a recipe that calls needs-env.sh and is not declared is red: {found}")
        _write(root / "tools" / "gate-class.tsv", hand_rows(BY_HAND, "host"))
        found = check(root, load(root), no_kernels)
        expect(any("gate-d calls" in f for f in found), f"check: a recipe that reaches the call through a dependency is red when not declared: {found}")
        _write(root / "tools" / "gate-class.tsv", hand_rows(BY_HAND, BY_HAND, BY_HAND))
        found = check(root, load(root), no_kernels)
        expect(any("gate-b is declared by-hand and its recipe calls no" in f for f in found), f"check: a by-hand row whose recipe calls no needs-env.sh is red: {found}")
        _write(root / "justfile", "gate-a:\n    echo a\n\ngate-b:\n    echo b\n\nother:\n    echo o\n")
        _write(root / "tools" / "gate-class.tsv", base_gates)

        # The tier column: a row without it (the five-column rows of the old table), with it empty or with a word the tiers do not
        # name is a named error; the days column left empty is the tier's.
        old_row = "fam-b\tF\tcrates/fam-b/**\t30\tyes"
        expect(raises(with_fam(fam_text("fam-a\tR\tmain\tcrates/fam-a/**\t7\tno", old_row)), "a row has no `tier`"),
               "load: a five-column family row (no tier) is a named error")
        bad = fam_text("fam-a\tR\tmain\tcrates/fam-a/**\t7\tno", "fam-b\tF\t\tcrates/fam-b/**\t30\tyes")
        expect(raises(with_fam(bad), "the family fam-b has no `tier`"), "load: an empty tier is a named error")
        bad = fam_text("fam-a\tR\tmain\tcrates/fam-a/**\t7\tno", "fam-b\tF\tweekly\tcrates/fam-b/**\t30\tyes")
        expect(raises(with_fam(bad), "unknown tier 'weekly'"), "load: a tier the tiers do not name is a named error")
        days_rows = with_fam(fam_text("fam-a\tR\tmain\tcrates/fam-a/**\t\tno", "fam-b\tF\tother\tcrates/fam-b/**\t\tyes"))
        days_tables = days_rows()
        expect(days_tables.families["fam-a"].rotation_days == 7 and days_tables.families["fam-b"].rotation_days == 30,
               f"load: an empty rotation_days is the tier's days: {days_tables.families['fam-a'].rotation_days} {days_tables.families['fam-b'].rotation_days}")
        # The rule's days: a main family at 30 d and an other family at 7 d are findings; so is a family whose every file another
        # family's glob also matches (a release cannot tell it is new).
        found = check(root, days_rows(), no_kernels)
        expect(found == [], f"check: the tier's days in the column are green: {found}")
        for fam_a_days, fam_b_days, who in (("30", "30", "fam-a"), ("7", "7", "fam-b")):
            found = check(root, with_fam(fam_text(f"fam-a\tR\tmain\tcrates/fam-a/**\t{fam_a_days}\tno", f"fam-b\tF\tother\tcrates/fam-b/**\t{fam_b_days}\tyes"))(), no_kernels)
            expect(any(who in f and "whose rotation is" in f for f in found), f"check: {who} off its tier's days is red: {found}")
        shared = fam_text("fam-a\tR\tmain\tcrates/fam-a/**\t7\tno", "fam-b\tF\tother\tcrates/fam-b/**\t30\tyes").replace("qwen38d\tF\tother\tcrates/qwen38d/**", "qwen38d\tF\tother\tcrates/fam-b/**")
        found = check(root, with_fam(shared)(), no_kernels)
        expect(any("fam-b owns no file" in f for f in found) and any("qwen38d owns no file" in f for f in found)
               and not any("fam-a owns no file" in f for f in found), f"check: two families on one glob own no file each: {found}")
        _write(root / "tools" / "families.tsv", "\n".join(fam_rows) + "\n")

        # The scope of a gate: its row's family and that family's tier, a recipe with no row by its profile's family, else common.
        sc = {n: gate_scope(tables, n, prof) for n, prof in (("gate-a", ""), ("gate-b", ""), ("weekly-x", "fam-b"), ("weekly-y", "fam-a"), ("weekly-z", "nobody"), ("refset-check", ""))}
        expect(sc["gate-a"] == GateScope("fam-a", "main", 7), f"gate_scope: a family row is the family's tier and days: {sc['gate-a']}")
        expect(sc["gate-b"] == GateScope("", "common", 7), f"gate_scope: a common row has no family and the main days: {sc['gate-b']}")
        expect(sc["weekly-x"] == GateScope("fam-b", "other", 30, True) and sc["weekly-y"] == GateScope("fam-a", "main", 7),
               f"gate_scope: a recipe with no row belongs to its profile's family: {sc['weekly-x']} {sc['weekly-y']}")
        expect(sc["weekly-z"] == GateScope("", "common", 7) and sc["refset-check"] == GateScope("", "common", 7),
               f"gate_scope: a recipe with no row and no family of its profile is common: {sc['weekly-z']} {sc['refset-check']}")
        expect(sc["weekly-x"].fixture_tier and not sc["weekly-y"].fixture_tier and not GateScope("fam-c", "other", 30, False).fixture_tier,
               "gate_scope: an other family with a fixture is held to the fixture tier; a main family, or an other one with none, is not")
        tooling = Tables(gates={"gate-t": GateRow("gate-t", "tooling", "", "host", ())}, families=tables.families, owners=tables.owners)
        expect(gate_scope(tooling, "gate-t") == GateScope("", "tooling", 7), "gate_scope: a tooling row keeps its class as the tier")
        # The real tier leaves an other family's gate out unless the family is new; a main family, a common and a tooling gate stay.
        expect(left_out(sc["weekly-x"], set()) == OTHER_REASON, "left_out: a gate of an other family is left out of the real tier")
        expect(left_out(sc["weekly-x"], {"fam-b"}) == "", "left_out: the same family new in the release is in")
        expect(left_out(sc["gate-a"], set()) == "" and left_out(sc["gate-b"], set()) == "" and left_out(GateScope("", "tooling", 7), set()) == "",
               "left_out: a main family's gate, a common and a tooling gate stay")
        kept, out = partition(["gate-a", "weekly-x", "gate-b"], lambda n: sc[n], set())
        expect(kept == ["gate-a", "gate-b"] and list(out) == ["weekly-x"] and out["weekly-x"][0] == OTHER_REASON and out["weekly-x"][1].family == "fam-b",
               f"partition: the names in scope in order, the others with their reason and family: {kept} {out}")

        # The families new in a release: a file of its own at the head and none at the base. A new file of an old family is not it;
        # a glob two families share tells nothing of either.
        fr = lambda n, *globs: FamilyRow(n, "F", "other", tuple(globs), 30, False)  # noqa: E731
        fams = {"x": fr("x", "shared/**", "x/**"), "y": fr("y", "shared/**", "y/**"), "z": fr("z", "shared/**"), "w": fr("w", "w/**")}
        base = ["shared/s.rs", "x/a.rs", "w/a.rs"]
        expect(own_files(fams, base) == {"x": ["x/a.rs"], "y": [], "z": [], "w": ["w/a.rs"]}, f"own_files: a file of a shared glob is nobody's: {own_files(fams, base)}")
        expect(new_families(fams, base, base + ["y/b.rs"]) == {"y"}, "new_families: the first file of a family is it")
        expect(new_families(fams, base, base + ["x/new.rs", "w/new.rs", "shared/new.rs"]) == set(),
               "new_families: a new file of a family that had one, or of a shared glob, is not")
        expect(new_families(fams, base, base) == set(), "new_families: nothing added is nothing new")
        expect(new_families(fams, base, base + ["x/b.rs", "y/b.rs"]) == {"y"}, "new_families: only the family with no file at the base")

        # The fixture column is the fixture table's: yes exactly when the table has a row for the family whose directory is not
        # `self`; a profile row of the table is not the column's.
        dirs = (root / recipes.FIXTURE_DIRS).read_text(encoding="utf-8")
        for text, who, says in ((dirs.replace("fam-b\tfamb", "fam-b\tself"), "fam-b", "fixture yes"), (dirs.replace("fam-b\tfamb\n", ""), "fam-b", "no row there"),
                                (dirs + "fam-a\tfama\n", "fam-a", "fixture no")):
            _write(root / recipes.FIXTURE_DIRS, text)
            found = check(root, load(root), no_kernels)
            expect(len(found) == 1 and who in found[0] and says in found[0], f"check: the fixture column against the table is red, naming {who}: {found}")
        _write(root / recipes.FIXTURE_DIRS, dirs)
        expect(check(root, load(root), no_kernels) == [], "check: the fixture column restored is green")
        kernel_self_test(expect)
        budget_self_test(expect, tmp)

        # The tree's own tables: they load and they are green.
        try:
            real = load(ROOT)
            kr = read_kernel_reach(Path(ROOT), real)
            findings = check(Path(ROOT), real, lambda r, t: kr)
            expect(findings == [], f"check: the tree's own tables are green: {findings[:6]}")
            # The reader against two files read by hand: the latent heads' Q6_K kernel runs in the gated-MLA bodies (kimi-k3, raon1),
            # the half-resolution position kernel in the RADIO tower (nemotron_h). A lib's own tests are no launch, and a seat or a tower
            # of another family is that family's.
            for f, want in (("crates/gpu/src/q6k_heads.rs", {"kimi-k3", "raon1"}), ("crates/gpu-vision/src/pos_half.rs", {"nemotron_h"})):
                got = kr.launchers.get(f, {})
                fams = {real.gates[n].family for n, how in got.items() if how == DEP and n in real.gates}
                expect(OWN not in got.values() and fams == want, f"read_kernel_reach: {f} is launched by the gates of {sorted(want)} alone: {sorted(fams)}, own {[n for n, h in got.items() if h == OWN]}")
            expect(len(real.gates) > 0 and len(real.families) > 0 and len(real.owners) > 0, "load: the tree's own tables hold rows")
            text = summary(real)
            expect("per class:" in text and "per family" in text and "per owner" in text, "summary: the three counts are printed")
        except (GateClassError, recipes.RecipeError) as e:
            expect(False, f"the tree's own tables: {e}")

    if fails:
        for f in fails:
            print(f"gate_class self-test FAIL: {f}", file=sys.stderr)
        print(f"gate_class self-test: {len(fails)} failed", file=sys.stderr)
        return 1
    print("gate_class self-test: ok")
    return 0


def cmd_budget(args: argparse.Namespace) -> int:
    """`budget`: the family budget table; --refresh first rewrites its real_s and unrowed from the times file; --shared adds the families whose
    gates load one real file in several bins."""
    root = Path(args.root)
    tables = load(root)
    if args.refresh:
        costs = measured_costs(root, tables, args.times, args.classes)
        path = root / BUDGET_TABLE
        text, changes = refresh_text(path.read_text(encoding="utf-8"), list(tables.families), costs)
        if changes:
            path.write_text(text, encoding="utf-8")
        print(f"budget --refresh: {len(changes)} row(s) of {BUDGET_TABLE} changed from {args.times}")
        for c in changes:
            print(f"  {c}")
        unrowed = {f: c.unrowed for f, c in costs.items() if c.unrowed}
        if unrowed:
            print("budget --refresh: gates with no real-tier row (in no real_s; release-list --wall prices each at "
                  f"{UNROWED_S} s): " + ", ".join(f"{f or '(common, tooling)'} {n}" for f, n in sorted(unrowed.items())))
    budgets = load_budget(root)
    print("\n".join(budget_lines(tables, budgets)))
    if args.shared:
        recs = recipes.load_justfile(str(root / "justfile"))
        shared = shared_loads(recs, tables)
        print(f"{len(shared)} family(ies) whose card gates load one real file in more than one bin (could be one bin, report only):")
        for fam, groups in shared.items():
            for profile, bins, gate_names in groups:
                print(f"  {fam}: profile {profile}: {len(bins)} bins in {len(gate_names)} gates: {' '.join(bins)}")
    return 0


def main(argv: list[str]) -> int:
    if argv == ["--self-test"]:
        return self_test()
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check", help="the findings of the design's checks; rc 1 when there is one")
    sub.add_parser("summary", help="the counts per class, per family and per owner")
    bp = sub.add_parser("budget", help="the family budget table (tools/family-budget.tsv) as it stands")
    bp.add_argument("--refresh", action="store_true", help="first rewrite real_s and unrowed of every row from the times file (a family with no row gets one)")
    bp.add_argument("--times", default=os.environ.get("BLOOMERY_GATE_TIMES", os.path.expanduser("~/.cache/bloomery/gate-times.tsv")),
                    help="the times file the medians come from (--refresh)")
    bp.add_argument("--classes", help="a saved `tools/gate-batch.sh --tier fixture --classes` (default: run it; --refresh)")
    bp.add_argument("--shared", action="store_true", help="also list the families whose card gates load one real file in more than one bin")
    bp.add_argument("--root", default=str(ROOT), help="the tree whose tables are read (default this one)")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "budget":
            return cmd_budget(args)
        tables = load(ROOT)
        if args.cmd == "summary":
            print(summary(tables))
            return 0
        findings = check(Path(ROOT), tables)
        notes = budget_findings(tables, load_budget(Path(ROOT)))[1]
    except (GateClassError, recipes.RecipeError) as e:
        print(f"gate_class.py: {e}", file=sys.stderr)
        return 2
    for f in findings:
        print(f"RED  {f}")
    for n in notes:
        print(f"note {n}")
    print(f"gate_class check: {len(findings)} finding(s)")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
