#!/usr/bin/env python3
"""The rotation: the gates no landing runs, scheduled (Mac, reads the lead's green ledger).

    tools/rotation.py due   [--days N] [--ledger PATH] [--tier real] [--classes FILE] [--with-weekly] [--out FILE]
    tools/rotation.py check [--days N] [--ledger PATH] [--classes FILE]
    tools/rotation.py --self-test

A landing runs the fixture tier. The gates it defers (class D: a real-only gate, a gate of a family with no fixture file, a real-file
gate, a needs-env gate; tools/gate-batch.sh's classifier, read from `--classes`, never re-implemented) and the `weekly-*` recipes run
only in a real-tier batch, and nothing else schedules them. A landing also runs a gate of an F family (tools/families.tsv rep F; the gates
are the class tables' family rows, through recipes.f_family_gates) only when its family's own files changed, and the release list leaves
those gates out. The rotation is the schedule for all of them: one list for one batch. A gate of an F family is grouped under its
class-table family (`<family> (F family)`), every other gate under the family of its recipe's BLOOMERY_MODEL.

  due    the rotation's gates with no green ledger row at their current input key (tools/recipes.py key, as gate-batch.sh --ledger
         computes it) and whose last green row, of any key, is N days old or older (default 7) or missing. Output: a `--list` file for
         tools/gate-batch.sh, grouped by family (the BLOOMERY_MODEL of the recipe's box line, else the default profile: gate-batch.sh's
         family()), the stalest family first, each gate with its predicted wall (the median of its last 5 rows of the times file, 45 s
         where it has none: gate-batch.sh's expected()) and the total. A needs-env gate is listed and the batch defers it, so the list
         never hides a gate that nothing can run unattended. --with-weekly puts every `weekly-*` recipe first, then every CUDA
         gate on a GGUF input (`mac_gguf_cuda`, the set a mac landing defers) that the list does not already name, then the due
         gates that are not weekly (`just weekly`): one batch, one ledger read, one sitting.
         The key status comes from one `tools/gate-batch.sh --ledger --dry-run` over the gates stale by date: it reads the box once
         (the manifest, through tools/box.sh) and only when some gate is stale by date. When it cannot, every gate stale by date is
         listed as due with `key unknown` and the reason on stderr: the batch's own ledger check skips any that is green at its key.
  check  rc 1 naming each family whose gates (the same set) have no green ledger row newer than N days. tools/release/publish.sh runs it
         before a release is published and stops on rc 1. A family of needs-env gates alone is printed `env` and not counted: it
         runs only on a machine that sets its variable, which this ledger's batches defer by name.

The ledger and the times file are the ones gate-batch.sh reads: BLOOMERY_GATE_LEDGER and BLOOMERY_GATE_TIMES, else
~/.cache/bloomery/. `--classes FILE` reads a saved `tools/gate-batch.sh --tier fixture --classes` (the classifier is slow: every
recipe is classified); without it the tool runs it. Exit codes: 0, 1 (check red), 2 (a bad argument or input, named), 70 (the classifier
or a tool it runs failed, named).
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime as dt
import io
import os
import re
import statistics
import subprocess
import sys
import tempfile
from collections.abc import Callable

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # `python3 -I` drops the script's directory
import recipes  # noqa: E402

ROOT = recipes.ROOT
DEFAULT_DAYS = 7
DEFAULT_S = 45  # tools/gate-batch.sh's DEFAULT_S
TIMES_ROWS = 5  # the rows its expected() takes the median of
DATE_FMT = "%Y-%m-%dT%H:%M:%S%z"
# The class D kinds a fixture-tier batch defers to the real tier (gate-batch.sh's header, «deferred»); the real tier's own D kinds
# (variant, fixture-only) run in a fixture batch, which is a landing, and are not the rotation's.
DEFERRED_KINDS = ("real-only", "real-file", "no-fixture", "needs-env")
F_KIND = "family"  # the kind of an F-family gate no landing defers: a landing runs it only on its family's own change
CLASS_LETTERS = "ABCDFRTX"
NEEDS_ENV = re.compile(r"needs-env call (\w+)")


class RotationError(Exception):
    """A named input or tool failure; main() prints it and exits 2 (input) or 70 (a tool)."""

    def __init__(self, msg: str, rc: int = 2):
        super().__init__(msg)
        self.rc = rc


@dataclasses.dataclass(frozen=True)
class Row:
    name: str
    cls: str
    kind: str
    cards: str
    why: str


@dataclasses.dataclass(frozen=True)
class Gate:
    name: str
    kind: str  # the deferred kind, `weekly` for a weekly-* recipe the fixture tier does not defer, or `family` for an F-family gate it does not defer
    weekly: bool
    family: str
    env: str  # the variable a needs-env gate waits for, else ""


@dataclasses.dataclass
class Entry:
    gate: Gate
    last: dt.datetime | None  # the newest green row of any key
    age: dt.timedelta | None
    wall: int
    wall_src: str
    status: str  # a status of the batch's ledger plan (run, never, error, deferred), or key-unknown
    detail: str


# ---------------------------------------------------------------------------------------------- the classifier


def parse_classes(text: str) -> dict[str, Row]:
    """`tools/gate-batch.sh --classes`: `name<TAB>class<TAB>kind<TAB>cards<TAB>why`, one recipe a line."""
    rows: dict[str, Row] = {}
    for n, ln in enumerate(text.splitlines(), 1):
        if not ln.strip():
            continue
        f = ln.split("\t")
        if len(f) != 5 or len(f[1]) != 1 or f[1] not in CLASS_LETTERS or not f[0]:
            raise RotationError(f"classes line {n} is not `name, class, kind, cards, why` (tab-separated, class one of {CLASS_LETTERS}): {ln[:100]!r}")
        if f[0] in rows:
            raise RotationError(f"classes line {n}: {f[0]} is listed twice")
        rows[f[0]] = Row(*f)
    if not rows:
        raise RotationError("the classes input names no recipe: it is the output of `tools/gate-batch.sh --tier fixture --classes`")
    return rows


def run_classes() -> str:
    proc = subprocess.run([os.path.join(ROOT, "tools/gate-batch.sh"), "--tier", "fixture", "--classes"], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RotationError(f"`tools/gate-batch.sh --tier fixture --classes` failed (rc {proc.returncode}): {proc.stderr.strip()[-400:]}", 70)
    return proc.stdout


def family_of(rs: dict[str, recipes.Recipe], name: str) -> str:
    """gate-batch.sh's family(): the BLOOMERY_MODEL of the first recipe in the dependency closure (the recipe, then its
    dependencies depth first) whose tools/box.sh call sets one, else the default profile."""
    seen: set[str] = set()

    def closure(n: str) -> list[str]:
        if n in seen or n not in rs:
            return []
        seen.add(n)
        return [n] + [m for d in rs[n].deps for m in closure(d)]

    for n in closure(name):
        for model in recipes.recipe_commands(rs[n]).models:
            if model:
                return model
    return recipes.DEFAULT_PROFILE


# Deferred recipes that are gates in all but name: refset-check holds every reference set to its family's check, the
# check a moved model file makes red first.
ROTATION_CHECKS = ("refset-check",)


def f_group(fam: str) -> str:
    """The group of an F family's gates in `due` and `check`, apart from a recipe-profile family of the same name: one group's fresh
    gates never stand for another's stale ones."""
    return f"{fam} (F family)"


def rotation_gates(rows: dict[str, Row], family: Callable[[str], str], f_gates: dict[str, str]) -> list[Gate]:
    """The rotation's gates: every gate-* or weekly-* recipe the fixture tier defers, every weekly-* recipe and every gate of an F
    family (`f_gates`: recipe -> class-table family, recipes.f_family_gates), which is the group it is listed under. Recipes the
    classifier defers that are not gates (the dump-ref-*, stage-*, route-trace-* tools) are not the rotation's. A gate the class
    tables name that the classifier does not is a named error: the tables or the justfile moved."""
    missing = sorted(n for n in f_gates if n not in rows)
    if missing:
        raise RotationError(f"the class tables name {len(missing)} F-family gate(s) the classifier does not list: {' '.join(missing[:6])}")
    out = []
    for name in sorted(rows):
        row = rows[name]
        weekly = name.startswith(recipes.WEEKLY_PREFIX)
        if not (weekly or name.startswith(recipes.GATE_PREFIX) or name in ROTATION_CHECKS):
            continue
        deferred = row.cls == "D" and row.kind in DEFERRED_KINDS
        if not (deferred or weekly or name in f_gates):
            continue
        m = NEEDS_ENV.search(row.why) if row.kind == "needs-env" else None
        kind = row.kind if deferred else "weekly" if weekly else F_KIND
        out.append(Gate(name, kind, weekly, f_group(f_gates[name]) if name in f_gates else family(name), m.group(1) if m else ""))
    return out


# ---------------------------------------------------------------------------------------------- the ledger and the times


def parse_date(s: str) -> dt.datetime:
    return dt.datetime.strptime(s, DATE_FMT)


def last_green(ledger: recipes.Ledger) -> tuple[dict[str, dt.datetime], int]:
    """The newest green row of each recipe, of any key (an item with env or ARGS counts for its recipe), and the rows whose date
    does not parse."""
    last: dict[str, dt.datetime] = {}
    bad = 0
    for rec in ledger.by_key.values():
        try:
            d = parse_date(rec.date)
        except ValueError:
            bad += 1
            continue
        if rec.recipe not in last or d > last[rec.recipe]:
            last[rec.recipe] = d
    return last, bad


def read_times(path: str) -> tuple[dict[str, list[int]], int]:
    """`item lane card secs date` rows in file order, by item; a malformed row is counted (gate-batch.sh refuses it by name)."""
    rows: dict[str, list[int]] = {}
    bad = 0
    if not os.path.exists(path):
        return rows, bad
    with open(path, encoding="utf-8", errors="replace") as fh:
        for ln in fh:
            f = ln.rstrip("\n").split("\t")
            if len(f) != 5 or not f[3].isdigit():
                bad += 1
                continue
            rows.setdefault(f[0], []).append(int(f[3]))
    return rows, bad


def expected(times: dict[str, list[int]], name: str) -> tuple[int, str]:
    """gate-batch.sh's expected(): the median of the item's last rows, rounded half up, else the default."""
    got = times.get(name, [])[-TIMES_ROWS:]
    if not got:
        return DEFAULT_S, "default"
    return int(statistics.median(got) + 0.5), f"median of {len(got)} row{'s' if len(got) > 1 else ''}"


# ---------------------------------------------------------------------------------------------- the key status


LANE_LINE = re.compile(r"^lane (\S+)(?: \(simulated\))?\s+(\S+)(?:\s|$)")  # `lane B@45s (simulated)  <stem>` when the plan simulates the lanes
LEDGER_LINE = re.compile(r"^\s+ledger: (\S+) — (.*)$")
DEFER_LINE = re.compile(r"^\s+deferred \((\S+?)\)")


def parse_dry_run(text: str, names: list[str]) -> dict[str, tuple[str, str]]:
    """The ledger status of each name in `tools/gate-batch.sh --ledger --dry-run`'s text: the `ledger: <status> — <detail>` line under
    each item's `lane <L>  <stem>` line (skip, run, never, error), `deferred` for an item under `lane -` that the batch defers
    (needs-env). A name with no status is a named error: the plan changed, or an item went missing."""
    got: dict[str, tuple[str, str]] = {}
    cur: str | None = None
    for ln in text.splitlines():
        m = LANE_LINE.match(ln)
        if m:
            cur = m.group(2)
            if m.group(1) == "-":
                got[cur] = ("deferred", "the batch defers it")
            continue
        m = LEDGER_LINE.match(ln)
        if m and cur is not None:
            got[cur] = (m.group(1), m.group(2))
            continue
        m = DEFER_LINE.match(ln)
        if m and cur is not None and got.get(cur, ("",))[0] == "deferred":
            got[cur] = ("deferred", m.group(1))
    missing = [n for n in names if n not in got]
    if missing:
        raise RotationError(f"the dry run printed no ledger status for {len(missing)} gate(s): {' '.join(missing[:6])}", 70)
    return got


def batch_keys(names: list[str], tier: str, ledger_path: str) -> dict[str, tuple[str, str]]:
    """One dry run of tools/gate-batch.sh over `names`, on the lead's ledger: its plan computes each item's key and ledger status
    (the box manifest is read once, through tools/box.sh). Nothing runs and nothing is written."""
    with tempfile.TemporaryDirectory(prefix="rotation-") as d:
        lst = os.path.join(d, "keys.list")
        with open(lst, "w", encoding="utf-8") as fh:
            fh.write("\n".join(names) + "\n")
        env = dict(os.environ, BLOOMERY_GATE_LEDGER=ledger_path)
        cmd = [os.path.join(ROOT, "tools/gate-batch.sh"), "--ledger", "--dry-run", "--tier", tier, "--list", lst]
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        raise RotationError(f"`gate-batch.sh --ledger --dry-run` failed (rc {proc.returncode}): {proc.stderr.strip()[-400:]}", 70)
    return parse_dry_run(proc.stdout, names)


# ---------------------------------------------------------------------------------------------- due


def due_entries(
    gates: list[Gate],
    last: dict[str, dt.datetime],
    times: dict[str, list[int]],
    now: dt.datetime,
    days: float,
    key_of: Callable[[list[str]], dict[str, tuple[str, str]]],
) -> tuple[list[Entry], str]:
    """The gates due, and the reason their key status is unknown ('' when it is known): stale by date first (no green row of any key,
    or the newest N days old or older), then the key status of those alone; a gate green at its current key is not due."""
    limit = dt.timedelta(days=days)
    stale = []
    for g in gates:
        lg = last.get(g.name)
        age = None if lg is None else now - lg
        if age is None or age >= limit:
            stale.append((g, lg, age))
    keys: dict[str, tuple[str, str]] = {}
    unknown = ""
    if stale:
        try:
            keys = key_of([g.name for g, _, _ in stale])
        except RotationError as e:
            unknown = str(e)
    out = []
    for g, lg, age in stale:
        if unknown:
            status, detail = "key-unknown", unknown
        else:
            status, detail = keys[g.name]
        if status == "skip":
            continue
        wall, src = expected(times, g.name)
        out.append(Entry(g, lg, age, wall, src, status, detail))
    return out, unknown


def staleness(e: Entry) -> float:
    return float("inf") if e.age is None else e.age.total_seconds()


def by_family(entries: list[Entry]) -> list[tuple[str, list[Entry]]]:
    """Families in order of staleness (the one holding the stalest gate first, a gate no row names being the stalest), each family's
    gates stalest first, ties by name."""
    fams: dict[str, list[Entry]] = {}
    for e in entries:
        fams.setdefault(e.gate.family, []).append(e)
    for es in fams.values():
        es.sort(key=lambda e: (-staleness(e), e.gate.name))
    return sorted(fams.items(), key=lambda kv: (-staleness(kv[1][0]), kv[0]))


def age_text(e: Entry) -> str:
    return "never green" if e.age is None else f"last green {int(e.age.total_seconds() // 86400)} d ago"


def reason(e: Entry) -> str:
    """Why the gate is due, from the batch's ledger status: run (no green row at the key), never (a recipe the batch never skips),
    error (no key), deferred (the batch defers it), or key-unknown (ours: the status could not be read)."""
    if e.status == "deferred":
        return f"needs {e.gate.env}: the batch defers it unless the variable is set" if e.gate.env else f"the batch defers it ({e.detail})"
    if e.status == "key-unknown":
        return "key unknown: the batch's ledger check decides"
    if e.status == "never":
        return "the batch never skips it"
    if e.status == "error":
        return f"key not computed ({e.detail})"
    return "no green row at its current key" if e.last is None else "key moved"


def format_list(
    entries: list[Entry],
    weekly: list[Gate],
    days: float,
    tier: str,
    now: dt.datetime,
    ledger_path: str,
    times_path: str,
    unknown: str,
    mac: list[str] | None = None,
) -> str:
    runs = [e for e in entries if e.status != "deferred"]
    total = sum(e.wall for e in runs)
    n_def = len(entries) - len(runs)
    defaults = [e.gate.name for e in runs if e.wall_src == "default"]
    out = [
        f"# rotation due: tier {tier}, no green row at the current key and none newer than {days:g} d, {now.strftime(DATE_FMT)}",
        f"# ledger {ledger_path}; times {times_path}",
        f"# {len(entries)} gate(s) due: {len(runs)} run, predicted {total} s (gate-times medians, {DEFAULT_S} s where a gate has no row"
        + (f": {len(defaults)} so)" if defaults else ")")
        + (f"; {n_def} wait(s) for an environment variable, which the batch defers" if n_def else ""),
    ]
    if unknown:
        out.append(f"# key status unknown ({unknown}): every gate stale by date is listed; the batch's ledger check skips any green at its key")
    mac = list(mac or [])
    if weekly or mac:
        note = f"# the {len(weekly)} weekly-* recipes"
        if mac:
            note += f" and {len(mac)} CUDA GGUF-input gate(s)"
        note += " first, then the due gates"
        out.append(note)
        out += [g.name for g in weekly]
        out += mac
    for fam, es in by_family([e for e in entries if not (weekly and e.gate.weekly)]):
        wall = sum(e.wall for e in es if e.status != "deferred")
        out.append(f"# family {fam}: {len(es)} gate(s), {wall} s, stalest {'never' if es[0].age is None else age_text(es[0])[len('last green '):]}")
        for e in es:
            sec = f"{e.wall} s ({e.wall_src})" if e.status != "deferred" else "not run unattended"
            out.append(f"# {e.gate.name}: {e.gate.kind}, {sec}, {age_text(e)}, {reason(e)}")
            out.append(e.gate.name)
    return "\n".join(out) + "\n"


def ledger_path_of(arg: str | None) -> str:
    return arg or os.environ.get("BLOOMERY_GATE_LEDGER") or os.path.expanduser("~/.cache/bloomery/gate-ledger.tsv")


def times_path_of() -> str:
    return os.environ.get("BLOOMERY_GATE_TIMES") or os.path.expanduser("~/.cache/bloomery/gate-times.tsv")


def load_gates(args: argparse.Namespace, rs: dict[str, recipes.Recipe] | None = None, f_gates: dict[str, str] | None = None) -> list[Gate]:
    if args.classes:
        try:
            with open(args.classes, encoding="utf-8") as fh:
                text = fh.read()
        except OSError as e:
            raise RotationError(f"--classes {args.classes}: {e.strerror}")
    else:
        text = run_classes()
    rows = parse_classes(text)
    if rs is None:
        rs = recipes.load_justfile(os.path.join(ROOT, "justfile"))
    if f_gates is None:
        f_gates = recipes.f_family_gates(recipes.class_tables(ROOT))
    return rotation_gates(rows, lambda n: family_of(rs, n), f_gates)


def load_ledger(path: str) -> tuple[dict[str, dt.datetime], int]:
    if not os.path.exists(path):
        print(f"rotation.py: {path} does not exist: no gate has a green row", file=sys.stderr)
    led = recipes.Ledger(path)
    if led.bad:
        print(f"rotation.py: {path}: {led.bad} malformed line(s) ignored", file=sys.stderr)
    last, bad = last_green(led)
    if bad:
        print(f"rotation.py: {path}: {bad} row(s) with a date that does not parse ignored", file=sys.stderr)
    return last, bad


def cmd_due(args: argparse.Namespace, now: dt.datetime | None = None, key_of: Callable | None = None, rs: dict | None = None,
            f_gates: dict[str, str] | None = None) -> int:
    now = now or dt.datetime.now().astimezone()
    if rs is None:
        rs = recipes.load_justfile(os.path.join(recipes.ROOT, "justfile"))
    gates = load_gates(args, rs, f_gates)
    ledger_path = ledger_path_of(args.ledger)
    times_path = times_path_of()
    last, _ = load_ledger(ledger_path)
    times, bad = read_times(times_path)
    if bad:
        print(f"rotation.py: {times_path}: {bad} malformed row(s) ignored (tools/gate-batch.sh refuses them by name)", file=sys.stderr)
    if not os.path.exists(times_path):
        print(f"rotation.py: {times_path} does not exist: every gate is predicted at {DEFAULT_S} s", file=sys.stderr)
    key_of = key_of or (lambda names: batch_keys(names, args.tier, ledger_path))
    entries, unknown = due_entries(gates, last, times, now, args.days, key_of)
    if unknown:
        print(f"rotation.py: key status unknown, every gate stale by date is listed: {unknown}", file=sys.stderr)
    weekly = [g for g in gates if g.weekly] if args.with_weekly else []
    # A mac landing defers every CUDA gate on a GGUF input. The weekly list carries that set, and a name already
    # on it (a weekly recipe, or a gate the rotation already lists) is not written twice.
    present = {g.name for g in weekly} | {e.gate.name for e in entries}
    mac = [n for n, _ in recipes.mac_gguf_cuda(rs, recipes.ROOT) if n not in present] if args.with_weekly else []
    text = format_list(entries, weekly, args.days, args.tier, now, ledger_path, times_path, unknown, mac)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
        sys.stderr.write("".join(ln + "\n" for ln in text.splitlines()[:3]))
    else:
        sys.stdout.write(text)
    return 0


def check_families(gates: list[Gate], last: dict[str, dt.datetime], now: dt.datetime, days: float) -> list[tuple[str, list[Gate], dt.datetime | None, bool]]:
    """Per family (sorted by name): its gates, the newest green row among them, and whether that row is newer than N days."""
    limit = dt.timedelta(days=days)
    fams: dict[str, list[Gate]] = {}
    for g in gates:
        fams.setdefault(g.family, []).append(g)
    out = []
    for fam in sorted(fams):
        seen = [last[g.name] for g in fams[fam] if g.name in last]
        newest = max(seen) if seen else None
        out.append((fam, fams[fam], newest, newest is not None and now - newest < limit))
    return out


def env_only(gates: list[Gate]) -> bool:
    """A family whose every gate is a needs-env gate: it runs only on a machine that sets its variable, which the lead's batches
    defer by name, so no row of this ledger can make it green; `check` names it apart instead of counting it red."""
    return bool(gates) and all(g.kind == "needs-env" for g in gates)


def cmd_check(args: argparse.Namespace, now: dt.datetime | None = None, rs: dict | None = None, f_gates: dict[str, str] | None = None) -> int:
    now = now or dt.datetime.now().astimezone()
    gates = load_gates(args, rs, f_gates)
    last, _ = load_ledger(ledger_path_of(args.ledger))
    red = 0
    for fam, gs, newest, ok in check_families(gates, last, now, args.days):
        when = "no green row" if newest is None else f"newest green row {int((now - newest).total_seconds() // 86400)} d ago"
        if ok:
            print(f"ok   {fam}: {len(gs)} gate(s), {when}")
        elif env_only(gs):
            print(f"env  {fam}: {len(gs)} gate(s), {when}; each waits for a machine that sets "
                  f"{' '.join(sorted({g.env for g in gs}))} (a batch defers it by name elsewhere), so this ledger cannot hold it: "
                  f"{' '.join(g.name for g in gs[:8])}" + (" …" if len(gs) > 8 else ""))
        else:
            red += 1
            print(f"RED  {fam}: {len(gs)} gate(s), {when}; none is newer than {args.days:g} d: {' '.join(g.name for g in gs[:8])}" + (" …" if len(gs) > 8 else ""))
    print(f"rotation check: {red} family(ies) with no green row newer than {args.days:g} d")
    return 1 if red else 0


# ---------------------------------------------------------------------------------------------- the self-test


def self_test() -> int:
    fails: list[str] = []

    def expect(cond: bool, msg: str) -> None:
        if not cond:
            fails.append(msg)

    def raises(fn: Callable, text: str) -> bool:
        try:
            fn()
        except RotationError as e:
            return text in str(e)
        return False

    now = dt.datetime(2026, 10, 21, 12, 0, 0, tzinfo=dt.timezone(dt.timedelta(hours=9)))

    def ago(d: float) -> str:
        return (now - dt.timedelta(days=d)).strftime(DATE_FMT)

    with tempfile.TemporaryDirectory(prefix="rotation-selftest-") as tmp:
        # The classes: five rotation gates of two families, a weekly that runs in a landing, and recipes that are not the rotation's.
        classes = "\n".join("\t".join(r) for r in [
            ("gate-fresh", "D", "real-only", "-", "gate-fresh: real-only call"),
            ("gate-oldkey", "D", "real-file", "-", "family a: its fixture-table row is self"),
            ("gate-never", "D", "no-fixture", "-", "family b: no row in the fixture table"),
            ("gate-recent", "D", "real-only", "-", "gate-recent: real-only call"),
            ("gate-edge", "D", "real-only", "-", "gate-edge: real-only call"),
            ("gate-env", "D", "needs-env", "-", "gate-env: needs-env call NEEDS_VAR; no device code"),
            ("weekly-w", "X", "solo", "3090,a6000", "weekly-w: [group('solo')]"),
            ("gate-landing", "F", "balanced", "3090,a6000", "gate-landing: gpu-gate.sh any"),
            ("gate-fixonly", "D", "fixture-only", "-", "gate-fixonly: fixture-only call"),
            ("gate-variant", "D", "variant", "-", "variant"),
            ("dump-ref-x", "D", "real-only", "-", "dump-ref-x: real-only call"),
            ("stage-x", "D", "real-only", "-", "stage-x: real-only call"),
        ]) + "\n"
        rows = parse_classes(classes)
        fam = {"gate-fresh": "fa", "gate-oldkey": "fa", "gate-never": "fb", "gate-recent": "fb", "gate-edge": "fc", "gate-env": "fc", "weekly-w": "fa"}
        gates = rotation_gates(rows, lambda n: fam[n], {})
        names = [g.name for g in gates]
        expect(names == ["gate-edge", "gate-env", "gate-fresh", "gate-never", "gate-oldkey", "gate-recent", "weekly-w"],
               f"rotation_gates: a deferred kind or a weekly, and no landing gate, fixture-only or variant gate, or tool recipe: {names}")
        expect([g.kind for g in gates if g.name == "weekly-w"] == ["weekly"], "rotation_gates: a weekly the fixture tier does not defer is kind weekly")
        expect([g.env for g in gates if g.name == "gate-env"] == ["NEEDS_VAR"], "rotation_gates: the variable a needs-env gate waits for")
        expect(raises(lambda: parse_classes("gate-x\tD\treal-only\n"), "classes line 1"), "parse_classes: a short row is a named error")
        expect(raises(lambda: parse_classes("gate-x\tQ\ta\tb\tc\n"), "classes line 1"), "parse_classes: an unknown class letter is a named error")
        expect(raises(lambda: parse_classes("gate-x\tD\ta\tb\tc\ngate-x\tD\ta\tb\tc\n"), "listed twice"), "parse_classes: a recipe twice is a named error")
        expect(raises(lambda: parse_classes(""), "names no recipe"), "parse_classes: an empty input is a named error")

        # The ledger: gate-fresh green at the current key 20 d ago; gate-oldkey green at an older key 8 d ago; gate-recent green 3 d ago
        # at an older key; gate-edge exactly 7 d; gate-never and gate-env no row; weekly-w 30 d ago.
        led = os.path.join(tmp, "ledger.tsv")
        k = lambda c: c * 64  # noqa: E731
        with open(led, "w", encoding="utf-8") as fh:
            for key, rec, d in [("a", "gate-fresh", 20), ("b", "gate-oldkey", 8), ("c", "gate-recent", 3), ("d", "gate-edge", 7), ("e", "weekly-w", 30),
                                ("f", "gate-oldkey", 40)]:
                fh.write(f"{k(key)}\t{rec}\t{rec}\tabc123456789\t{ago(d)}\t/x\n")
            fh.write("not a row\n")
            fh.write(f"{k('0')}\tgate-baddate\tgate-baddate\tabc\t2026-13-45\t/x\n")
        ledger = recipes.Ledger(led)
        last, bad = last_green(ledger)
        expect(bad == 1 and ledger.bad == 1, f"last_green: a malformed line and a bad date are counted, not read (bad dates {bad}, malformed {ledger.bad})")
        expect(last["gate-oldkey"] == now - dt.timedelta(days=8), "last_green: the newest row of any key stands (8 d, not the 40 d row)")
        times_path = os.path.join(tmp, "times.tsv")
        with open(times_path, "w", encoding="utf-8") as fh:
            for item, secs in [("gate-oldkey", 10), ("gate-oldkey", 20), ("gate-oldkey", 90), ("gate-oldkey", 30), ("gate-oldkey", 40), ("gate-oldkey", 50),
                               ("gate-edge", 100), ("gate-fresh", 7), ("gate-fresh", 8)]:
                fh.write(f"{item}\tA\t3090\t{secs}\t{ago(1)}\n")
            fh.write("a malformed row\n")
        times, tbad = read_times(times_path)
        expect(tbad == 1, f"read_times: a malformed row is counted ({tbad})")
        expect(expected(times, "gate-oldkey") == (40, "median of 5 rows"), f"expected: the median of the last 5 rows, not of all six: {expected(times, 'gate-oldkey')}")
        expect(expected(times, "gate-fresh") == (8, "median of 2 rows"), f"expected: a median of two rounds half up: {expected(times, 'gate-fresh')}")
        expect(expected(times, "gate-never") == (DEFAULT_S, "default"), "expected: a gate with no row is the default")

        keyed: list[list[str]] = []

        def keys(status: dict[str, str]) -> Callable:
            def f(names: list[str]) -> dict[str, tuple[str, str]]:
                keyed.append(list(names))
                return {n: (status.get(n, "run"), f"detail {n}") for n in names}
            return f

        # gate-fresh is green at its current key: the key table says skip.
        entries, unknown = due_entries(gates, last, times, now, 7, keys({"gate-fresh": "skip", "gate-env": "deferred"}))
        due = {e.gate.name: e for e in entries}
        expect(not unknown, "due: a known key status has no unknown reason")
        expect("gate-fresh" not in due, "due: a gate green at its current key is not due, though its last run is 20 d old")
        status_of = lambda n: due[n].status if n in due else None  # noqa: E731
        expect(status_of("gate-oldkey") == "run", "due: a gate green at an old key and last run 8 d ago is due")
        expect(status_of("gate-never") == "run" and due["gate-never"].last is None, "due: a gate that never ran is due, with no last green row")
        expect("gate-recent" not in due, "due: a gate whose last run, of any key, is 3 d old is not due though its key moved")
        expect("gate-edge" in due, "due: a gate whose last run is exactly N d old is due")
        expect("weekly-w" in due, "due: a weekly last run 30 d ago at a moved key is due")
        expect("gate-recent" not in keyed[0] and "gate-fresh" in keyed[0], f"due: the key status is asked of the gates stale by date alone: {keyed[0]}")
        expect(status_of("gate-env") == "deferred" and due["gate-env"].last is None if "gate-env" in due else False, "due: a needs-env gate that never ran is listed, deferred")
        entries1, _ = due_entries(gates, last, times, now, 1, keys({"gate-fresh": "skip"}))
        expect({e.gate.name for e in entries1} >= {"gate-recent", "gate-edge"} and "gate-fresh" not in {e.gate.name for e in entries1},
               "due: --days 1 makes the 3 d gate due and still not the one green at its key")
        keyed.clear()
        due_entries(gates, {g.name: now for g in gates}, times, now, 7, keys({}))
        expect(keyed == [], "due: no gate stale by date asks nothing of the box")
        # A key status that cannot be read lists every gate stale by date, with the reason.

        def broken(names: list[str]) -> dict[str, tuple[str, str]]:
            raise RotationError("box down", 70)

        entries_u, unknown = due_entries(gates, last, times, now, 7, broken)
        expect(unknown == "box down" and {e.gate.name for e in entries_u} == {"gate-oldkey", "gate-never", "gate-edge", "gate-env", "gate-fresh", "weekly-w"}
               and all(e.status == "key-unknown" for e in entries_u), "due: an unreadable key status lists every gate stale by date as key-unknown")

        g0 = gates[0]
        reasons = [reason(Entry(g0, lg, None if lg is None else dt.timedelta(days=9), 1, "default", st, "d")) for st, lg in
                   (("run", None), ("run", now), ("never", now), ("error", now), ("key-unknown", None), ("deferred", None))]
        expect(reasons == ["no green row at its current key", "key moved", "the batch never skips it", "key not computed (d)",
                           "key unknown: the batch's ledger check decides", "the batch defers it (d)"], f"reason: one line for each ledger status: {reasons}")

        # The list: families by staleness, each gate's wall, the total; a gate line carries nothing but its name (gate-batch.sh --list).
        text = format_list(entries, [], 7, "real", now, led, times_path, "")
        items = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
        expect(all(re.fullmatch(r"(gate|weekly)-[a-z]+", ln) for ln in items), f"format_list: every non-comment line is a bare recipe name: {items}")
        expect(items == ["gate-never", "gate-env", "gate-edge", "weekly-w", "gate-oldkey"], f"format_list: families stalest first, a gate no row names first: {items}")
        expect("# family fb: 1 gate(s), 45 s, stalest never" in text, "format_list: a family line with its wall and its stalest gate")
        # predicted total: oldkey 40 + never 45 + edge 100 + weekly 45 = 230 (gate-env is deferred, gate-fresh is skipped)
        expect("# 5 gate(s) due: 4 run, predicted 230 s" in text, f"format_list: the total counts what runs: {text.splitlines()[2]}")
        expect("1 wait(s) for an environment variable" in text, "format_list: the needs-env gate is counted apart")
        fams = [f for f, _ in by_family(entries)]
        expect(fams == ["fb", "fc", "fa"], f"by_family: never green first, then the stalest: {fams}")
        expect(" ".join(e.gate.name for e in dict(by_family(entries))["fa"]) == "weekly-w gate-oldkey", "by_family: within a family the stalest first")
        weekly = [g for g in gates if g.weekly]
        text_w = format_list(entries, weekly, 7, "real", now, led, times_path, "")
        items_w = [ln for ln in text_w.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
        expect(items_w[0] == "weekly-w" and items_w.count("weekly-w") == 1, f"format_list: with the weeklies, they come first and once: {items_w}")
        expect(format_list([], [], 7, "real", now, led, times_path, "").count("\n") == 3, "format_list: nothing due is three comment lines")
        expect("key status unknown (box down)" in format_list(entries_u, [], 7, "real", now, led, times_path, "box down"), "format_list: an unknown key status is said in the list")

        # check: a family with nothing green in N days is red; one with a recent row is not; a family with no row at all is red.
        res = {f: (n, ok) for f, _, n, ok in check_families(gates, last, now, 7)}
        expect(res["fa"][1] is False and res["fb"][1] is True and res["fc"][1] is False,
               f"check_families: fa (newest 8 d) red, fb (3 d) ok, fc (7 d and none) red: {res}")
        expect(res["fc"][0] == now - dt.timedelta(days=7), "check_families: the newest row of the family")
        expect({f: ok for f, _, _, ok in check_families(gates, last, now, 9)}["fa"] is True, "check_families: --days 9 keeps the 8 d family ok")

        # The dry run's text, as tools/gate-batch.sh prints it.
        dry = ("gate-batch: dry run — 5 items (and 1 deferred, the variable they need is not set: needs-env), lanes 2, logs would go to /x\n"
               "lane A  gate-oldkey                  just gate-oldkey\n        fixed, 40s — gpu-gate.sh 3090\n"
               "        ledger: run — no green record at this key; gate-oldkey was green at abc 2026-10-13\n"
               "lane B@45s (simulated)  gate-fresh   BLOOMERY_BOX_ENV='BLOOMERY_GATE_CARD=3090' just gate-fresh\n"
               "        balanced, expected 8 s (median of 2 rows) — x\n        ledger: skip — green-at=abc 2026-10-01 src=lead tree=/x\n"
               "        movable — a free lane takes it on its card, longest-est first (never a fixed, solo or v41-load item)\n"
               "lane X  weekly-w                     just weekly-w\n        solo, expected 79 s — why\n        ledger: error — the key tool failed (rc=1)\n"
               "lane C  weekly-w-with-a-name-longer-than-twenty-eight-chars just x\n        plan — why\n        ledger: never — a reference tree\n"
               "gate-batch: the chain is fed by lane A (3090): a lane takes a member\n"
               "lane -  gate-env                     just gate-env\n        deferred (needs-env), never run in this batch: a variable the recipe needs is not set — why\n"
               "gate-batch: predicted laneA=40s\n")
        long_name = "weekly-w-with-a-name-longer-than-twenty-eight-chars"
        want_names = ["gate-oldkey", "gate-fresh", "weekly-w", long_name, "gate-env"]
        try:
            got = parse_dry_run(dry, want_names)
        except RotationError as e:
            got = {}
            expect(False, f"parse_dry_run: a simulated lane label names no item, and every item has its status: {e}")
        expect(got == {"gate-oldkey": ("run", "no green record at this key; gate-oldkey was green at abc 2026-10-13"),
                       "gate-fresh": ("skip", "green-at=abc 2026-10-01 src=lead tree=/x"), "weekly-w": ("error", "the key tool failed (rc=1)"),
                       long_name: ("never", "a reference tree"), "gate-env": ("deferred", "needs-env")} or not got,
               f"parse_dry_run: skip, run, never, error and deferred read from the plan: {got}")
        expect(raises(lambda: parse_dry_run(dry, ["gate-oldkey", "gate-missing"]), "gate-missing"), "parse_dry_run: a gate with no status is a named error")

        # family_of: the BLOOMERY_MODEL of the first box call in the closure, else the default profile.
        def rec(name: str, line: str, deps: list[str] | None = None) -> recipes.Recipe:
            return recipes.Recipe(name, [line], deps or [], "")

        rs = {"g-own": rec("g-own", "BLOOMERY_MODEL=glm5next ./tools/box.sh 'x'"), "g-dep": rec("g-dep", "true", ["d1"]),
              "d1": rec("d1", "BLOOMERY_MODEL=qwen3moe ./tools/box.sh 'x'"), "g-none": rec("g-none", "./tools/box.sh 'x'"),
              "g-both": rec("g-both", "BLOOMERY_MODEL=a ./tools/box.sh 'x'", ["d1"])}
        expect([family_of(rs, n) for n in ("g-own", "g-dep", "g-none", "g-both")] == ["glm5next", "qwen3moe", recipes.DEFAULT_PROFILE, "a"],
               f"family_of: own call, dependency, default, own before dependency: {[family_of(rs, n) for n in ('g-own', 'g-dep', 'g-none', 'g-both')]}")

        # The commands, end to end on the synthetic inputs: no classifier, no box.
        cls_path = os.path.join(tmp, "classes.tsv")
        with open(cls_path, "w", encoding="utf-8") as fh:
            fh.write(classes)
        rs2 = {n: rec(n, f"BLOOMERY_MODEL={fam[n]} ./tools/box.sh 'x'") for n in fam}
        old_times = os.environ.get("BLOOMERY_GATE_TIMES")
        os.environ["BLOOMERY_GATE_TIMES"] = times_path
        try:
            out_path = os.path.join(tmp, "due.list")
            args = argparse.Namespace(days=7.0, ledger=led, tier="real", classes=cls_path, with_weekly=True, out=out_path)
            with contextlib.redirect_stderr(io.StringIO()):
                rc = cmd_due(args, now, keys({"gate-fresh": "skip"}), rs2, {})
            written = open(out_path, encoding="utf-8").read()
            expect(rc == 0 and written.splitlines()[3] == "# the 1 weekly-* recipes first, then the due gates" and "weekly-w" in written
                   and "gate-oldkey" in written and "gate-fresh" not in written, f"cmd_due: the list file written (rc {rc}): {written}")
            # A CUDA gate on a GGUF fixture is not a rotation gate. The weekly list without the mac union omits it;
            # due --with-weekly names it once. A checkpoint-folder CUDA gate stays off. A gate already due is not repeated.
            def bare(text: str) -> list[str]:
                return [ln for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]

            gguf_line = "BLOOMERY_MODEL=qwen3moe ./tools/box.sh 'bash tools/gpu-gate.sh x'"
            fold_line = "BLOOMERY_MODEL=minimax-m3 ./tools/box.sh 'bash tools/gpu-gate.sh x'"
            rs_mac = dict(rs2)
            rs_mac["gate-macgguf"] = rec("gate-macgguf", gguf_line)
            rs_mac["gate-foldkeep"] = rec("gate-foldkeep", fold_line)
            rs_mac["gate-never"] = rec("gate-never", gguf_line)
            plain = format_list(entries, [g for g in gates if g.weekly], 7, "real", now, led, times_path, "")
            expect("gate-macgguf" not in bare(plain), f"weekly union: without it a CUDA GGUF gate is absent: {bare(plain)}")
            out_mac = os.path.join(tmp, "due-mac.list")
            args_mac = argparse.Namespace(days=7.0, ledger=led, tier="real", classes=cls_path, with_weekly=True, out=out_mac)
            with contextlib.redirect_stderr(io.StringIO()):
                rc = cmd_due(args_mac, now, keys({"gate-fresh": "skip"}), rs_mac, {})
            items_mac = bare(open(out_mac, encoding="utf-8").read())
            expect(rc == 0 and items_mac.count("gate-macgguf") == 1 and items_mac[0] == "weekly-w",
                   f"weekly union: due --with-weekly names the CUDA GGUF gate once, after the weekly recipes: {items_mac}")
            expect("gate-foldkeep" not in items_mac, f"weekly union: a checkpoint-folder CUDA gate stays off: {items_mac}")
            expect(items_mac.count("gate-never") == 1, f"weekly union: a gate already due is not listed twice: {items_mac}")
            out_off = os.path.join(tmp, "due-off.list")
            args_off = argparse.Namespace(days=7.0, ledger=led, tier="real", classes=cls_path, with_weekly=False, out=out_off)
            with contextlib.redirect_stderr(io.StringIO()):
                rc = cmd_due(args_off, now, keys({"gate-fresh": "skip"}), rs_mac, {})
            expect(rc == 0 and "gate-macgguf" not in open(out_off, encoding="utf-8").read(),
                   "weekly union: without --with-weekly the CUDA GGUF gate stays off the due list")
            argsc = argparse.Namespace(days=7.0, ledger=led, classes=cls_path)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
                rc = cmd_check(argsc, now, rs2, {})
            expect(rc == 1 and "RED  fa:" in buf.getvalue() and "ok   fb:" in buf.getvalue() and "RED  fc:" in buf.getvalue(),
                   f"cmd_check: rc 1 naming the families with no row newer than 7 d: {rc} {buf.getvalue()}")
            argsc.days = 9.0
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
                rc = cmd_check(argsc, now, rs2, {})
            expect(rc == 0 and "RED" not in buf.getvalue(), f"cmd_check: rc 0 when every family has a row newer than N days: {rc} {buf.getvalue()}")
            env_g = Gate("gate-e", "needs-env", False, "fe", "NEEDS_VAR")
            real_g = Gate("gate-r", "real-only", False, "fe", "")
            expect(env_only([env_g]) and not env_only([env_g, real_g]) and not env_only([]),
                   "env_only: a family of needs-env gates alone is named apart; one other gate keeps it countable")

            # The F families. A landing runs an F family's gate only on its family's change, so the rotation schedules it, from the class
            # tables' set (`f_gates` stands for recipes.f_family_gates): gate-fx-old green 8 d ago and gate-fx-new never (family fx), gate-fz
            # green 6 d ago (fz), gate-fy-real an F gate the landing defers too (fy), gate-other a landing gate of no F family and gate-prof a deferred
            # gate of the recipe-profile family fx, green 1 d ago.
            f_classes = "\n".join("\t".join(r) for r in [
                ("gate-fx-old", "F", "balanced", "3090,a6000", "gate-fx-old: gpu-gate.sh any"),
                ("gate-fx-new", "F", "balanced", "3090,a6000", "gate-fx-new: gpu-gate.sh any"),
                ("gate-fy-real", "D", "real-only", "-", "gate-fy-real: real-only call"),
                ("gate-fz", "X", "solo", "3090,a6000", "gate-fz: [group('solo')]"),
                ("gate-other", "F", "balanced", "3090,a6000", "gate-other: gpu-gate.sh any"),
                ("gate-prof", "D", "real-only", "-", "gate-prof: real-only call"),
            ]) + "\n"
            f_gates = {"gate-fx-old": "fx", "gate-fx-new": "fx", "gate-fy-real": "fy", "gate-fz": "fz"}
            f_rows = parse_classes(f_classes)
            fg = rotation_gates(f_rows, lambda n: "profile", f_gates)
            expect([g.name for g in fg] == sorted([*f_gates, "gate-prof"]) and "gate-other" not in [g.name for g in fg],
                   f"rotation_gates: every F-family gate of the class tables, the deferred gate and no other landing gate: {[g.name for g in fg]}")
            expect({g.name: (g.kind, g.family) for g in fg} == {"gate-fx-old": ("family", "fx (F family)"), "gate-fx-new": ("family", "fx (F family)"),
                                                                 "gate-fy-real": ("real-only", "fy (F family)"), "gate-fz": ("family", "fz (F family)"),
                                                                 "gate-prof": ("real-only", "profile")},
                   f"rotation_gates: an F gate the landing runs is kind family, a deferred one keeps its kind, both grouped by the class-table family: {[(g.name, g.kind, g.family) for g in fg]}")
            expect(sorted(g.name for g in rotation_gates(f_rows, lambda n: "profile", {})) == ["gate-fy-real", "gate-prof"],
                   "rotation_gates: with no F-family set only the deferred gates are the rotation's (what a landing's F gates were before)")
            expect(raises(lambda: rotation_gates(f_rows, lambda n: "profile", {**f_gates, "gate-gone": "fx"}), "gate-gone"),
                   "rotation_gates: an F-family gate the classifier does not list is a named error")

            def f_ledger(name: str, rows_: list[tuple[str, str, float]]) -> str:
                path_ = os.path.join(tmp, name)
                with open(path_, "w", encoding="utf-8") as fh_:
                    for key_, rec_, d_ in rows_:
                        fh_.write(f"{k(key_)}\t{rec_}\t{rec_}\tabc123456789\t{ago(d_)}\t/x\n")
                return path_

            f_cls_path = os.path.join(tmp, "f-classes.tsv")
            with open(f_cls_path, "w", encoding="utf-8") as fh:
                fh.write(f_classes)
            rs_f = {n: rec(n, f"BLOOMERY_MODEL={'fx' if n == 'gate-prof' else 'profile'} ./tools/box.sh 'x'") for n in f_rows}
            led_old = f_ledger("f-old.tsv", [("1", "gate-fx-old", 8), ("2", "gate-fz", 6), ("4", "gate-prof", 1)])
            out_f = os.path.join(tmp, "f-due.list")
            args_f = argparse.Namespace(days=7.0, ledger=led_old, tier="real", classes=f_cls_path, with_weekly=False, out=out_f)
            with contextlib.redirect_stderr(io.StringIO()):
                rc = cmd_due(args_f, now, keys({}), rs_f, f_gates)
            items_f = bare(open(out_f, encoding="utf-8").read())
            expect(rc == 0 and sorted(items_f) == ["gate-fx-new", "gate-fx-old", "gate-fy-real"],
                   f"cmd_due: an F gate last green 8 d ago is due, one 6 d ago is not, a landing gate of no F family never is: {items_f}")
            argsf = argparse.Namespace(days=7.0, ledger=led_old, classes=f_cls_path)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
                rc = cmd_check(argsf, now, rs_f, f_gates)
            expect(rc == 1 and "RED  fx (F family):" in buf.getvalue() and "RED  fy (F family):" in buf.getvalue()
                   and "ok   fz (F family):" in buf.getvalue() and "ok   fx: 1 gate(s)" in buf.getvalue(),
                   f"cmd_check: rc 1 for an F family whose newest row is 8 d old or missing though a profile family of its name is fresh, ok for the 6 d one: {rc} {buf.getvalue()}")
            argsf.ledger = f_ledger("f-fresh.tsv", [("1", "gate-fx-old", 6), ("2", "gate-fz", 6), ("3", "gate-fy-real", 3), ("4", "gate-prof", 1)])
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
                rc = cmd_check(argsf, now, rs_f, f_gates)
            expect(rc == 0 and "RED" not in buf.getvalue(), f"cmd_check: rc 0 once every F family has a row newer than 7 d: {rc} {buf.getvalue()}")
        finally:
            if old_times is None:
                os.environ.pop("BLOOMERY_GATE_TIMES", None)
            else:
                os.environ["BLOOMERY_GATE_TIMES"] = old_times

    if fails:
        for f in fails:
            print(f"rotation self-test FAIL: {f}", file=sys.stderr)
        print(f"rotation self-test: {len(fails)} failed", file=sys.stderr)
        return 1
    print("rotation self-test: ok")
    return 0


def main(argv: list[str]) -> int:
    if argv == ["--self-test"]:
        return self_test()
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, helptext in (("due", "the list for one real-tier batch"), ("check", "rc 1 naming each family with no green row newer than N days")):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("--days", type=float, default=DEFAULT_DAYS, help=f"the age at which a last run is stale (default {DEFAULT_DAYS})")
        p.add_argument("--ledger", help="the green ledger (default BLOOMERY_GATE_LEDGER, else ~/.cache/bloomery/gate-ledger.tsv)")
        p.add_argument("--classes", help="a saved `tools/gate-batch.sh --tier fixture --classes` (default: run it)")
        if name == "due":
            p.add_argument("--tier", choices=["real"], default="real", help="the tier the list runs in: the real tier, which runs what the fixture tier defers")
            p.add_argument("--with-weekly", action="store_true", help="every weekly-* recipe first, then every CUDA gate on a GGUF input the list does not already name, then the due gates that are not weekly (`just weekly`)")
            p.add_argument("--out", help="write the list to FILE (its first lines to stderr) instead of stdout")
    args = ap.parse_args(argv)
    if args.days < 0:
        print("rotation.py: --days takes a number of days, 0 or more", file=sys.stderr)
        return 2
    try:
        return cmd_due(args) if args.cmd == "due" else cmd_check(args)
    except (RotationError, recipes.RecipeError) as e:
        print(f"rotation.py: {e}", file=sys.stderr)
        return e.rc if isinstance(e, RotationError) else 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
