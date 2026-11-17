#!/usr/bin/env python3
"""`just affected` with the cards in view: a recipe that needs two cards is marked deferred on a one-card machine.

    tools/affected-gates.sh [arguments of tools/recipes.py affected]     # runs this
    tools/affected-cards.py --self-test

tools/recipes.py `affected` prints the list and is run unchanged: same arguments, same stderr, same exit code. This adds only
what the machine the gates run on decides, read from the same places tools/gate-batch.sh reads it:

  the view     the cards in view: tools/ref/cards.sh's cards_view, through the read-only box.sh call gate-batch.sh's VIEW_PROBE is
               (that line is read from tools/gate-batch.sh, so there is one probe). It asks the box over ssh, and this machine's own
               cards under BLOOMERY_BOX_LOCAL=1; `--no-box` asks no ssh, so it reads the view only under BLOOMERY_BOX_LOCAL=1. A
               read-only call syncs nothing: a box with no directory for this tree yet (box.sh ends 66) leaves the view unread.
  the needs    the recipe's text, as tools/gate-batch.sh --classes reads it: its row's reason column names `BLOOMERY_CARD=both` for
               every recipe whose closure names it, so a tag cannot disagree with the batch's own deferral (two-card).
               tools/gate-class.tsv and tools/families.tsv carry no card column, and tools/recipes.py reads no card word.

On a view of two or more cards the output is the list byte for byte. On a view of one card every selected recipe whose closure
names BLOOMERY_CARD=both stays in the list, its line gets a `[two-card: deferred, 1 card in view]` suffix (the first word is
still the recipe, which is all a batch `--list` reads), and a closing `two-card (N), deferred on this machine` block lists them
(`  - name  (reason)` lines, no item to a batch). Recipes that name BLOOMERY_CARD=a6000 are not tagged: the view does not prove
them deferred (box.sh runs that pick on the card in view when the machine declares its cards, BLOOMERY_BOX_CARD=local, and looks
the A6000 up by name otherwise, which the view's count does not show); one closing line names them and which of the two holds.
A view or a classifier that cannot be read leaves the list as it is and says so in one line under the final `notes:`.

The probe and the classifier run beside tools/recipes.py (which runs a classifier of its own for the owed block), so the added wall
is the output's post-processing: the classifier is the shorter of the two.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECIPES = os.path.join(ROOT, "tools", "recipes.py")
GATE_BATCH = os.path.join(ROOT, "tools", "gate-batch.sh")
BOX = os.path.join(ROOT, "tools", "box.sh")
PROBE_BOUND = 120  # seconds: one read-only box.sh call (the shared ssh master; local mode's nvidia-smi is cut at 30 s)
CLASSES_BOUND = 600  # seconds: `gate-batch.sh --classes` (about 12 s of one core on the Mac)

TWO_CARD = re.compile(r"\bBLOOMERY_CARD=both\b")
A6000_NAMED = re.compile(r"\bBLOOMERY_CARD=a6000\b")
VIEW_LINE = re.compile(r"^cards-view rc=0 n=([0-9]+) local=([01]) ")
PICK_LINE = re.compile(r"^  (\S+)(?:\s|$)")
VIEW_PROBE_LINE = re.compile(r"^VIEW_PROBE='([^']*)'$", re.M)


class ViewError(Exception):
    """The cards in view cannot be asked: the message says why and what to do."""


@dataclass(frozen=True)
class View:
    n: int | None = None  # the cards in view; None when not read
    local: bool = False  # the machine's environment file declares its cards as the cards in view (BLOOMERY_BOX_CARD=local)
    why: str = ""  # when n is None


@dataclass(frozen=True)
class Rows:
    known: frozenset[str] = frozenset()
    two: frozenset[str] = frozenset()
    a6000: frozenset[str] = frozenset()
    why: str = ""  # not empty: the rows were not read


def read_probe(root: str) -> str:
    """The probe tools/gate-batch.sh runs (its VIEW_PROBE line): one source, so the two tools read the same view."""
    path = os.path.join(root, "tools", "gate-batch.sh")
    try:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
    except OSError as e:
        raise ViewError(f"{path}: {e.strerror}") from e
    m = VIEW_PROBE_LINE.search(text)
    if m is None:
        raise ViewError("tools/gate-batch.sh has no `VIEW_PROBE='…'` line: the probe that reads the cards in view moved, and this reads it from there")
    return m.group(1)


def parse_view(rc: int | None, out: str) -> View:
    """The probe's answer as gate-batch.sh's view_read reads it: a `cards-view rc=0 n= local=` line, or the reason there is none."""
    lines = [ln for ln in out.splitlines() if ln.strip()]
    line = next((ln for ln in reversed(lines) if ln.startswith("cards-view rc=")), "")
    last = lines[-1] if lines else ""
    if rc is None:
        return View(why=f"the read-only box.sh call that reads the cards in view did not end within {PROBE_BOUND} s")
    if rc == 66:
        return View(why=f"the box has no directory for this tree yet (box.sh ended 66, and `just affected` syncs nothing; any other box command makes it): {last}")
    if rc != 0 or not line:
        return View(why=f"the read-only box.sh call that reads the cards in view ended {rc} and printed {line or 'no cards-view line'}: {last}")
    m = VIEW_LINE.match(line)
    if m is None:
        return View(why=f"cards_view did not answer: {line[len('cards-view '):]}")
    return View(n=int(m.group(1)), local=m.group(2) == "1")


def parse_rows(out: str) -> Rows:
    """`gate-batch.sh --classes` rows (name, class, kind, cards, reason): the recipes, and which of them name a card pick."""
    known: set[str] = set()
    two: set[str] = set()
    a6000: set[str] = set()
    for line in out.split("\n"):
        if not line.strip():
            continue
        f = line.split("\t", 4)
        if len(f) != 5:
            return Rows(why=f"a classifier row of {len(f)} tab-separated fields, not 5 (name, class, kind, cards, reason): {line[:120]!r}")
        known.add(f[0])
        if TWO_CARD.search(f[4]):
            two.add(f[0])
        if A6000_NAMED.search(f[4]):
            a6000.add(f[0])
    if not known:
        return Rows(why="`tools/gate-batch.sh --classes` printed no row")
    return Rows(frozenset(known), frozenset(two), frozenset(a6000))


def base_names(recipes_line: str) -> list[str]:
    """The recipes of a `recipes:` line: a variant item `name@VAR=value` is its recipe, once."""
    seen: dict[str, None] = {}
    for w in recipes_line.split()[1:]:
        seen[w.split("@", 1)[0]] = None
    return list(seen)


def with_notes(lines: list[str], notes: list[str], end: str) -> list[str]:
    """The output's lines with `  note` lines under the final `notes:` block (made when there is none)."""
    body = list(lines)
    while body and body[-1] == "":
        body.pop()
    if "notes:" not in body:
        body.append("notes:")
    return body + [f"  {n}" for n in notes] + ([""] if end else [])


def annotate(text: str, view: View, rows: Rows | None) -> str:
    """The list `recipes.py affected` printed, with the cards in view applied. Two cards or more: the text itself."""
    if view.n is not None and view.n >= 2:
        return text
    end = "\n" if text.endswith("\n") else ""
    lines = text.rstrip("\n").split("\n")
    at = next((i for i, ln in enumerate(lines) if ln.startswith("recipes: ")), None)
    if at is None:
        return "\n".join(with_notes(lines, ["no `recipes:` line in the output above: no recipe is tagged for the cards in view"], end))
    picks = base_names(lines[at])
    if not picks:
        return text
    if view.n is None:
        return "\n".join(with_notes(lines, [f"cards in view not read ({view.why}): no recipe is tagged two-card"], end))
    if rows is None or rows.why:
        why = rows.why if rows is not None else "the classifier was not run"
        return "\n".join(with_notes(lines, [f"two-card tags not added ({view.n} card in view): {why}"], end))
    two = [p for p in picks if p in rows.two]
    a6000 = [p for p in picks if p in rows.a6000]
    unknown = [p for p in picks if p not in rows.known]
    for i in range(at):
        m = PICK_LINE.match(lines[i])
        if m and m.group(1).split("@", 1)[0] in rows.two:
            lines[i] += f"  [two-card: deferred, {view.n} card in view]"
    notes = []
    if unknown:
        notes.append(f"{len(unknown)} selected recipe(s) have no row in `tools/gate-batch.sh --classes`, so none of them is tagged: {' '.join(unknown)}")
    out = with_notes(lines, notes, "") if notes else lines[:]
    if two:
        out.append(f"two-card ({len(two)}), deferred on this machine ({view.n} card in view, read by tools/ref/cards.sh's cards_view): BLOOMERY_CARD=both "
                   "runs on two or more cards in view and tools/box.sh ends 64 on fewer. Each stays in the list above; tools/gate-batch.sh defers it by name "
                   "(two_card=) and a machine with two cards runs it:")
        out += [f"  - {n}  (BLOOMERY_CARD=both in its recipe)" for n in two]
    if a6000:
        how = ("this machine declares its cards (BLOOMERY_BOX_CARD=local), so the A6000 pick runs on the card in view" if view.local else
               "this machine does not declare its cards (BLOOMERY_BOX_CARD=local), so box.sh looks the A6000 up by name and ends 64 unless the one card in view is an A6000")
        out.append(f"a6000 ({len(a6000)}), named by the recipe and not tagged: {how}: {' '.join(a6000)}")
    return "\n".join(out) + end


class Child:
    """A started command whose output goes to temporary files. A background one (probe, classifier) runs in a session of its own, so
    killing it takes its children too; the run of recipes.py keeps this process's session, terminal and stdin."""

    def __init__(self, argv: list[str], env: dict[str, str] | None = None, inherit: bool = False, merge: bool = False):
        self.out = tempfile.TemporaryFile()
        self.err: object = self.out if merge else (None if inherit else tempfile.TemporaryFile())
        self.own_session = not inherit
        self.proc = subprocess.Popen(argv, cwd=ROOT, env=env, stdin=None if inherit else subprocess.DEVNULL, stdout=self.out,
                                     stderr=self.err, start_new_session=self.own_session)

    def finish(self, bound: float | None) -> tuple[int | None, str, str]:
        """(exit code or None past the bound, stdout, stderr)."""
        try:
            rc: int | None = self.proc.wait(timeout=bound)
        except subprocess.TimeoutExpired:
            self.kill()
            rc = None
        return rc, self.text(self.out), self.text(self.err) if self.err is not None and self.err is not self.out else ""

    @staticmethod
    def text(f: object) -> str:
        f.seek(0)  # type: ignore[attr-defined]
        return f.read().decode("utf-8", "replace")  # type: ignore[attr-defined]

    def raw(self) -> bytes:
        self.out.seek(0)
        return self.out.read()

    def kill(self) -> None:
        if self.proc.poll() is not None:
            return
        try:
            if self.own_session:
                os.killpg(self.proc.pid, signal.SIGTERM)
            else:
                self.proc.terminate()
        except ProcessLookupError:
            pass
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()


def start_probe(children: list[Child]) -> Child:
    env = dict(os.environ)
    env["BLOOMERY_BOX_READONLY"] = "1"
    env["BLOOMERY_BOX_WAIT"] = "0"
    c = Child([BOX, read_probe(ROOT)], env=env, merge=True)
    children.append(c)
    return c


def exit_code(rc: int) -> int:
    return rc if rc >= 0 else 128 - rc


def run(argv: list[str], children: list[Child]) -> int:
    aff = Child([sys.executable, RECIPES, "affected", *argv], inherit=True)
    children.append(aff)
    probe: Child | None = None
    skip = ""
    try:
        if "--no-box" in argv and os.environ.get("BLOOMERY_BOX_LOCAL") != "1":
            skip = "--no-box asks no ssh, and the cards in view are the box's; BLOOMERY_BOX_LOCAL=1 reads this machine's own"
        else:
            probe = start_probe(children)
    except (ViewError, OSError) as e:
        skip = str(e)
    if probe is None:
        view = View(why=skip)
    else:
        rc, out, _ = probe.finish(PROBE_BOUND)
        view = parse_view(rc, out)
    classes: Child | None = None
    classes_why = ""
    if view.n is not None and view.n < 2:
        try:
            classes = Child([GATE_BATCH, "--classes"])
            children.append(classes)
        except OSError as e:
            classes_why = f"{GATE_BATCH}: {e.strerror}"
    rc, _, _ = aff.finish(None)
    raw = aff.raw()
    if rc is None or rc != 0:
        emit(raw)
        return exit_code(rc if rc is not None else 1)
    text = raw.decode("utf-8", "surrogateescape")
    rows: Rows | None = Rows(why=classes_why) if classes_why else None
    if classes is not None and any(ln.startswith("recipes: ") and len(ln.split()) > 1 for ln in text.split("\n")):
        crc, cout, cerr = classes.finish(CLASSES_BOUND)
        if crc == 0:
            rows = parse_rows(cout)
        elif crc is None:
            rows = Rows(why=f"`tools/gate-batch.sh --classes` did not end within {CLASSES_BOUND} s")
        else:
            tail = (cerr.strip().splitlines() or ["no message"])[-1]
            rows = Rows(why=f"`tools/gate-batch.sh --classes` ended {crc}: {tail}")
    emit(annotate(text, view, rows).encode("utf-8", "surrogateescape"))
    return 0


def emit(data: bytes) -> None:
    sys.stdout.flush()
    sys.stdout.buffer.write(data)
    sys.stdout.buffer.flush()


def main(argv: list[str]) -> int:
    if argv == ["--self-test"]:
        return self_test()
    if any(a in ("-h", "--help") for a in argv):
        os.execv(sys.executable, [sys.executable, RECIPES, "affected", *argv])
    children: list[Child] = []
    try:
        return run(argv, children)
    except BrokenPipeError:
        return 141
    finally:
        for c in children:
            c.kill()


# ----------------------------------------------------------------------------------------------
# self-test
# ----------------------------------------------------------------------------------------------

FIXTURE_PLAIN = """\
affected: aaaaaaaaa..(working tree) — 3 changed files, 5 of 9 gate-recipes selected (class-staged: 5 kept, 4 left out), 0 of 1 weekly-recipes by a trigger
  gate-one                        tools/gpu-gate.sh (+3)  [script]
  gate-two                        tools/gpu-gate.sh (+3)  [script]  (class: the recipe's own target file tools/gpu-gate.sh changed)
  gate-three                      tools/gpu-gate.sh (+3)  [script]
  gate-six@BLOOMERY_FIXTURE_VARIANT=v1  [variant v1: its row lists gate-six, which is selected]
  weekly-two                      crates/x/src/lib.rs  [trigger row]
recipes: gate-one gate-two gate-three gate-six@BLOOMERY_FIXTURE_VARIANT=v1 weekly-two
always: check-recipes check
left out (1), each by its reason (--no-class keeps the class stage's gates):
  - gate-two-left  (class: family gate of f, no file of f changed (rep F))
real-tier owed (1): gate-two
  - gate-two  (real-only)
unmapped (1):
  gate-two.txt
notes:
  dep-info not read (--no-box): static walk only
"""

FIXTURE_NARROW = """\
affected: aaaaaaaaa..(working tree) — 3 changed files, 2 of 9 gate-recipes selected (narrowed from 6 by ptx-scan; class-staged: 2 kept, 0 left out)
narrow: ptx-scan gate_x identical
narrow: file crates/x/src/lib.rs: derivation
  gate-one       [script]
  gate-two       [script]
recipes: gate-one gate-two
always: check-recipes check
left out (0), each by its reason (--no-class keeps the class stage's gates):
unmapped (0):
notes:
  dep-info not read (--no-box): static walk only
"""

ROWS_TEXT = "\n".join("\t".join(r) for r in [
    ("gate-one", "F", "balanced", "3090,a6000", "gate-one: gpu-gate.sh any"),
    ("gate-two", "X", "solo", "-", "gate-two: BLOOMERY_CARD=both (box.sh picks the cards, and the gate runner takes their locks)"),
    ("gate-three", "X", "solo", "-", "gate-three: BLOOMERY_CARD=a6000 (box.sh picks the cards, and the gate runner takes their locks)"),
    ("gate-six", "X", "solo", "-", "gate-six: BLOOMERY_CARD=both (box.sh picks the cards, and the gate runner takes their locks)"),
    ("weekly-two", "X", "solo", "-", "weekly-two: BLOOMERY_CARD=both (box.sh picks the cards, and the gate runner takes their locks)"),
    ("gate-two-left", "F", "balanced", "-", "no device code"),
]) + "\n"

# The plain fixture with the cards of one machine that declares them (BLOOMERY_BOX_CARD=local) applied.
TAG = "  [two-card: deferred, 1 card in view]"
BLOCK_TWO = ("two-card (3), deferred on this machine (1 card in view, read by tools/ref/cards.sh's cards_view): BLOOMERY_CARD=both runs on two or more cards "
             "in view and tools/box.sh ends 64 on fewer. Each stays in the list above; tools/gate-batch.sh defers it by name (two_card=) and a machine with "
             "two cards runs it:")
A6000_LOCAL = ("a6000 (1), named by the recipe and not tagged: this machine declares its cards (BLOOMERY_BOX_CARD=local), so the A6000 pick runs on the card "
               "in view: gate-three")
A6000_UNDECLARED = ("a6000 (1), named by the recipe and not tagged: this machine does not declare its cards (BLOOMERY_BOX_CARD=local), so box.sh looks the "
                    "A6000 up by name and ends 64 unless the one card in view is an A6000: gate-three")


def batch_items(text: str) -> list[str]:
    """What tools/gate-batch.sh --list reads of an `affected` output (its PYPLAN's rule): the first word of each `  gate-`/`  weekly-` line."""
    lines = text.split("\n")
    head = [ln for ln in lines if ln.strip()][:2]
    if head and head[0].startswith("./tools/affected-gates.sh ") and len(head) > 1:
        head = head[1:]
    if not (head and head[0].startswith(("affected:", "affected (fast):"))):
        return ["<not an affected output>"]
    return [ln.split()[0] for ln in lines if ln.startswith(("  gate-", "  weekly-"))]


PROBE_LINE = ("VIEW_PROBE='l=0; if [ \"${BLOOMERY_BOX_CARD:-}\" = local ]; then l=1; else unset CUDA_VISIBLE_DEVICES; fi; BLOOMERY_BOX_CARD=local . "
              "tools/ref/cards.sh; if type cards_view > /dev/null 2>&1; then cards_view; v=$?; else v=127 CARDS_VIEW_WHY=\"x\"; fi; echo \"cards-view rc=$v "
              "n=${CARDS_VIEW_N:-0} local=$l kind=${CARDS_VIEW_KIND:-} why=${CARDS_VIEW_WHY:-}\"'\n")

FAKE_BOX = """#!/bin/sh
here=$(dirname "$0")/..
echo "box ro=${BLOOMERY_BOX_READONLY:-} wait=${BLOOMERY_BOX_WAIT:-} probe=$(printf %s "$1" | cut -c1-8)" >> "$here/calls.log"
[ -z "${FAKE_VIEW_RC:-}" ] || { echo 'box.sh: the fake cannot read the cards in view' >&2; exit "$FAKE_VIEW_RC"; }
echo "${FAKE_VIEW_LINE:-cards-view rc=0 n=${FAKE_VIEW_N:-2} local=${FAKE_VIEW_LOCAL:-0} kind= why=}"
"""

FAKE_BATCH = """#!/bin/sh
here=$(dirname "$0")/..
echo "gate-batch $*" >> "$here/calls.log"
[ -z "${FAKE_CLASSES_RC:-}" ] || { echo 'gate-batch: the fake classifier failed' >&2; exit "$FAKE_CLASSES_RC"; }
cat "$here/tools/rows.tsv"
"""

FAKE_RECIPES = """import os, sys
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "calls.log"), "a") as fh:
    fh.write("recipes " + " ".join(sys.argv[1:]) + "\\n")
sys.stderr.write("fake recipes.py: a line on stderr\\n")
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.environ.get("FAKE_AFF", "fixture.txt")), "rb") as fh:
    sys.stdout.buffer.write(fh.read())
sys.exit(int(os.environ.get("FAKE_AFF_RC", "0")))
"""

REAL_WRAPPER_LAST = 'exec python3 "$(dirname "$0")/affected-cards.py" "$@"'
BEFORE_WRAPPER = '#!/usr/bin/env bash\nset -euo pipefail\nexec python3 "$(dirname "$0")/recipes.py" affected "$@"\n'


def mutated(src: str, old: str, new: str) -> str:
    if src.count(old) != 1:
        raise AssertionError(f"the self-test's mutant needs exactly one `{old}` in this file, found {src.count(old)}")
    return src.replace(old, new)


def make_tree(root: str, helper_src: str, wrapper_src: str) -> None:
    """A tree whose tools are the fakes above and this tool (and its wrapper) as the sources given."""
    tools = os.path.join(root, "tools")
    os.makedirs(tools, exist_ok=True)
    with open(os.path.join(tools, "affected-cards.py"), "w", encoding="utf-8") as fh:
        fh.write(helper_src)
    with open(os.path.join(tools, "affected-gates.sh"), "w", encoding="utf-8") as fh:
        fh.write(wrapper_src)
    for name, body in (("box.sh", FAKE_BOX), ("gate-batch.sh", "#!/bin/sh\n" + PROBE_LINE + FAKE_BATCH.split("\n", 1)[1]), ("recipes.py", FAKE_RECIPES)):
        with open(os.path.join(tools, name), "w", encoding="utf-8") as fh:
            fh.write(body)
        os.chmod(os.path.join(tools, name), 0o755)
    for name, body in (("fixture.txt", FIXTURE_PLAIN), ("narrow.txt", FIXTURE_NARROW), ("rows.tsv", ROWS_TEXT)):
        with open(os.path.join(tools, name), "w", encoding="utf-8") as fh:
            fh.write(body)


def run_tree(root: str, args: list[str], **env: str) -> tuple[int, str, str, str]:
    """(rc, stdout, stderr, the calls the fakes logged) of `tools/affected-gates.sh args` in a tree made by make_tree."""
    log = os.path.join(root, "calls.log")
    if os.path.exists(log):
        os.remove(log)
    e = {k: v for k, v in os.environ.items() if not k.startswith(("BLOOMERY_", "FAKE_"))}
    e.update(env)
    p = subprocess.run(["bash", os.path.join(root, "tools", "affected-gates.sh"), *args], capture_output=True, text=True, env=e, cwd=root, timeout=60)
    calls = open(log, encoding="utf-8").read() if os.path.exists(log) else ""
    return p.returncode, p.stdout, p.stderr, calls


def closure_two_card(root: str) -> set[str]:
    """The recipes whose closure (the recipe and its dependencies) names BLOOMERY_CARD=both, from `just --dump` and nothing else."""
    dump = json.loads(subprocess.run(["just", "--justfile", os.path.join(root, "justfile"), "--dump", "--dump-format", "json"], check=True,
                                     capture_output=True, text=True, timeout=120).stdout)
    recipes, assigns = dump["recipes"], dump["assignments"]

    def body(name: str) -> str:
        out = []
        for frags in recipes[name]["body"]:
            line = ""
            for f in frags:
                if isinstance(f, str):
                    line += f
                elif len(f) == 1 and f[0][0] == "variable" and f[0][1] in assigns and isinstance(assigns[f[0][1]]["value"], str):
                    line += assigns[f[0][1]]["value"]
                else:
                    line += "{{}}"
            out.append(line)
        return "\n".join(out)

    def closure(name: str, seen: set[str]) -> list[str]:
        if name in seen:
            return []
        seen.add(name)
        found = [name]
        for dep in recipes[name]["dependencies"]:
            found += closure(dep["recipe"], seen)
        return found

    return {n for n in recipes if any(TWO_CARD.search(body(m)) for m in closure(n, set()))}


def self_test() -> int:
    fails: list[str] = []

    def expect(cond: bool, msg: str) -> None:
        if not cond:
            fails.append(msg)

    rows = parse_rows(ROWS_TEXT)
    expect(rows.why == "" and rows.two == {"gate-two", "gate-six", "weekly-two"} and rows.a6000 == {"gate-three"} and "gate-one" in rows.known,
           f"rows: the recipes that name each card pick are read from the reason column: {rows}")
    expect(parse_rows("gate-x\tF\tbalanced\n").why.startswith("a classifier row of 3 tab-separated fields") and parse_rows("\n").why != "",
           "rows: a short row and no row are named errors, not an empty set")
    one, one_decl, two_cards = View(1, False), View(1, True), View(2, False)

    # the view's reading, as gate-batch.sh's view_read reads the probe
    expect(parse_view(0, "x\ncards-view rc=0 n=1 local=1 kind= why=\n") == View(1, True), "view: a one-card machine that declares its cards")
    expect(parse_view(0, "cards-view rc=0 n=2 local=0 kind= why=\n") == View(2, False), "view: two cards, none declared")
    expect(parse_view(66, "box.sh: ws has no ~/repo/x: nothing was synced there yet\n").why.startswith("the box has no directory for this tree yet (box.sh ended 66"),
           "view: rc 66 says there is no remote directory and that nothing is synced")
    expect("ended 70 and printed no cards-view line: box.sh: boom" in parse_view(70, "box.sh: boom\n").why, "view: a failed call is named with its last line")
    expect(parse_view(0, "cards-view rc=1 n=0 local=1 kind=gone why=CUDA_VISIBLE_DEVICES=GPU-x\n").why.startswith("cards_view did not answer: rc=1 n=0 local=1 kind=gone"),
           "view: a pin that names a lost card is cards_view's own message")
    expect(parse_view(None, "").why.endswith(f"did not end within {PROBE_BOUND} s"), "view: a call past its bound is named")
    probe = read_probe(ROOT)
    expect(probe.startswith("l=0; if [ \"${BLOOMERY_BOX_CARD:-}\" = local ]") and "cards-view rc=" in probe,
           "view: the probe is gate-batch.sh's own VIEW_PROBE line, read from there")

    # the list, a one-card view: tagged, still the list a batch reads
    out = annotate(FIXTURE_PLAIN, one_decl, rows)
    lines = out.split("\n")
    tagged = [ln for ln in lines if ln.endswith(TAG)]
    expect([ln.split()[0] for ln in tagged] == ["gate-two", "gate-six@BLOOMERY_FIXTURE_VARIANT=v1", "weekly-two"],
           f"one card: the two-card pick lines, the variant and the weekly one included, carry the suffix, and no other line does: {tagged}")
    expect(batch_items(out) == batch_items(FIXTURE_PLAIN) and out.startswith("affected: "),
           f"one card: the head and the items a batch --list reads are the list's own: {batch_items(out)}")
    expect(lines.count("recipes: gate-one gate-two gate-three gate-six@BLOOMERY_FIXTURE_VARIANT=v1 weekly-two") == 1,
           "one card: the `recipes:` line is not changed, no recipe is dropped from it")
    expect("  - gate-two  (real-only)" in lines and "  gate-two.txt" in lines and "  - gate-two-left  (class: family gate of f, no file of f changed (rep F))" in lines,
           "one card: lines after `recipes:` (left out, owed, unmapped) are not tagged, even when they name a tagged recipe")
    i = lines.index(BLOCK_TWO)
    expect(lines[i + 1:i + 4] == ["  - gate-two  (BLOOMERY_CARD=both in its recipe)", "  - gate-six  (BLOOMERY_CARD=both in its recipe)", "  - weekly-two  (BLOOMERY_CARD=both in its recipe)"],
           f"one card: the closing block names each recipe once, in list order, as `  - name  (reason)` lines: {lines[i:i + 5]}")
    expect(A6000_LOCAL in lines and lines.index(A6000_LOCAL) > i and out.endswith("\n"), "one card: the a6000 line follows, naming the pick and why it is not deferred (declared)")
    expect(A6000_UNDECLARED in annotate(FIXTURE_PLAIN, one, rows).split("\n") and TAG in annotate(FIXTURE_PLAIN, one, rows),
           "one card, no declaration: the a6000 line says box.sh looks the A6000 up by name")
    nar = annotate(FIXTURE_NARROW, one_decl, rows)
    expect([ln for ln in nar.split("\n") if ln.endswith(TAG)] == [f"  gate-two       [script]{TAG}"] and "narrow: ptx-scan gate_x identical" in nar.split("\n")
           and batch_items(nar) == ["gate-one", "gate-two"], f"one card, narrowed shape: the pick line without a file is tagged, the narrow: lines are not: {nar}")

    # two cards: the list byte for byte
    expect(annotate(FIXTURE_PLAIN, two_cards, rows) == FIXTURE_PLAIN and annotate(FIXTURE_PLAIN, View(8, True), None) == FIXTURE_PLAIN,
           "two cards in view: the output is the list itself, whatever the rows")

    # what cannot be read keeps today's lines and says so under notes:
    unread = annotate(FIXTURE_PLAIN, View(why="--no-box asks no ssh"), None)
    expect(unread.startswith(FIXTURE_PLAIN) and unread.endswith("  cards in view not read (--no-box asks no ssh): no recipe is tagged two-card\n")
           and unread.count("\n") == FIXTURE_PLAIN.count("\n") + 1, f"unread view: the list as it was plus one line under notes: {unread[len(FIXTURE_PLAIN):]!r}")
    norows = annotate(FIXTURE_PLAIN, one, Rows(why="`tools/gate-batch.sh --classes` ended 70: boom"))
    expect(norows.startswith(FIXTURE_PLAIN) and norows.endswith("  two-card tags not added (1 card in view): `tools/gate-batch.sh --classes` ended 70: boom\n")
           and TAG not in norows, "unread classifier: the list as it was plus one line naming the failure")
    unk = annotate(FIXTURE_PLAIN, one, parse_rows("gate-one\tF\tbalanced\t-\tno device code\n"))
    expect("  4 selected recipe(s) have no row in `tools/gate-batch.sh --classes`, so none of them is tagged: gate-two gate-three gate-six weekly-two" in unk.split("\n")
           and TAG not in unk, "a selected recipe the classifier does not name is named, not silently left untagged")
    nolist = annotate("hello\n", one, rows)
    expect(nolist == "hello\nnotes:\n  no `recipes:` line in the output above: no recipe is tagged for the cards in view\n", f"no `recipes:` line is named: {nolist!r}")
    empty = FIXTURE_PLAIN.replace("recipes: gate-one gate-two gate-three gate-six@BLOOMERY_FIXTURE_VARIANT=v1 weekly-two", "recipes: ")
    expect(annotate(empty, one, rows) == empty, "no recipe selected: nothing to tag, the output is the list itself")
    quiet = parse_rows("".join(f"{n}\tF\tbalanced\t-\tno device code\n" for n in ("gate-one", "gate-two", "gate-three", "gate-six", "weekly-two")))
    plain_one = annotate(FIXTURE_PLAIN, one, quiet)
    expect(plain_one == FIXTURE_PLAIN, "one card, no selected recipe names a card pick: the output is the list itself")

    # the whole wrapper, on fakes: the tree's box.sh, gate-batch.sh and recipes.py are stubs, so nothing real is asked
    me = os.path.abspath(__file__)
    helper = open(me, encoding="utf-8").read()
    wrapper = open(os.path.join(ROOT, "tools", "affected-gates.sh"), encoding="utf-8").read()
    expect(REAL_WRAPPER_LAST in wrapper, "wrapper: tools/affected-gates.sh runs this tool")
    work = tempfile.mkdtemp(prefix="affected-cards-test.")
    try:
        t = os.path.join(work, "tree")
        make_tree(t, helper, wrapper)
        rc, so, se, calls = run_tree(t, ["main", "--no-box", "--fast"], FAKE_VIEW_N="2", BLOOMERY_BOX_LOCAL="1")
        expect(rc == 0 and so == FIXTURE_PLAIN and "fake recipes.py: a line on stderr" in se and "gate-batch" not in calls,
               f"tree, two cards: stdout is recipes.py's byte for byte, its stderr passes, the classifier is not run: rc={rc} calls={calls!r}")
        expect("recipes affected main --no-box --fast\n" in calls and "box ro=1 wait=0 probe=l=0; if" in calls,
               f"tree: the arguments reach recipes.py as given, and the probe is a read-only box.sh call that waits for nothing: {calls!r}")
        rc, so, se, calls = run_tree(t, ["main"], FAKE_VIEW_N="1", FAKE_VIEW_LOCAL="1")
        expect(rc == 0 and so == annotate(FIXTURE_PLAIN, one_decl, rows) and TAG in so and "gate-batch --classes\n" in calls and calls.count("box ro=1") == 1,
               f"tree, one card: the two-card picks are tagged, the probe is asked once, the classifier runs: rc={rc} {so[-300:]!r}")
        rc, so, se, calls = run_tree(t, ["main"], FAKE_VIEW_N="1", FAKE_AFF_RC="3")
        expect(rc == 3 and so == FIXTURE_PLAIN, f"tree: recipes.py's exit code and output are kept as they are when it fails: rc={rc}")
        rc, so, se, calls = run_tree(t, ["main"], FAKE_VIEW_RC="66")
        expect(rc == 0 and so.startswith(FIXTURE_PLAIN) and "the box has no directory for this tree yet (box.sh ended 66" in so and calls.count("box ") == 1,
               f"tree, a box with no directory for this tree: one line says so, and no syncing call follows: {so[len(FIXTURE_PLAIN):]!r} {calls!r}")
        rc, so, se, calls = run_tree(t, ["main", "--no-box"], FAKE_VIEW_N="1")
        expect(rc == 0 and "box " not in calls and "--no-box asks no ssh" in so and TAG not in so, f"tree, --no-box on a box: no ssh, and the output says why: {so[-200:]!r}")
        rc, so, se, calls = run_tree(t, ["main", "--no-box"], FAKE_VIEW_N="1", BLOOMERY_BOX_LOCAL="1")
        expect(rc == 0 and "box ro=1" in calls and TAG in so, "tree, --no-box under BLOOMERY_BOX_LOCAL=1: this machine's cards are read, with no ssh")
        rc, so, se, calls = run_tree(t, ["main"], FAKE_VIEW_N="1", FAKE_CLASSES_RC="70")
        expect(rc == 0 and so.startswith(FIXTURE_PLAIN) and "`tools/gate-batch.sh --classes` ended 70: gate-batch: the fake classifier failed" in so and TAG not in so,
               f"tree, a classifier that fails: today's list and one line naming it: {so[len(FIXTURE_PLAIN):]!r}")
        rc, so, se, calls = run_tree(t, ["main", "--narrow"], FAKE_VIEW_N="1", FAKE_AFF="narrow.txt")
        expect(rc == 0 and so == annotate(FIXTURE_NARROW, one, rows), "tree, one card, the narrowed shape: tagged by the same rule")

        # FAIL-first: today's wrapper (recipes.py run bare), and the tool with each rule removed, must each turn the one-card or two-card case red
        def one_card_tags(tree: str) -> bool:
            r, o, _, _ = run_tree(tree, ["main"], FAKE_VIEW_N="1", FAKE_VIEW_LOCAL="1")
            return r == 0 and o == annotate(FIXTURE_PLAIN, one_decl, rows)

        def two_card_untouched(tree: str) -> bool:
            r, o, _, c = run_tree(tree, ["main"], FAKE_VIEW_N="2")
            return r == 0 and o == FIXTURE_PLAIN and "gate-batch" not in c

        for label, hsrc, wsrc, case in (
            ("today's wrapper (recipes.py run bare)", helper, BEFORE_WRAPPER, one_card_tags),
            ("no tag on a pick line", mutated(helper, "lines[i] += f\"  [two-card: deferred, " "{view.n} card in view]\"", "pass"), wrapper, one_card_tags),
            ("a one-card view read as two cards", mutated(helper, "if view.n is not None and view.n " ">= 2:", "if view.n is not None and view.n >= 1:"), wrapper, one_card_tags),
            ("the two-card view annotated like the one-card one", mutated(helper, "if view.n is not None and view.n " ">= 2:\n        return text\n", ""), wrapper, two_card_untouched),
            ("the classifier run on every view", mutated(helper, "if view.n is not None and view.n " "< 2:\n        try:", "if True:\n        try:"), wrapper, two_card_untouched),
        ):
            m = os.path.join(work, "mutant")
            shutil.rmtree(m, ignore_errors=True)
            make_tree(m, hsrc, wsrc)
            expect(not case(m), f"FAIL-first: the case must be red on {label}")
    finally:
        shutil.rmtree(work, ignore_errors=True)

    # the contract with the classifier, on the real justfile: its rows name BLOOMERY_CARD=both for exactly the recipes whose closure does
    proc = subprocess.run([GATE_BATCH, "--classes"], capture_output=True, text=True, cwd=ROOT, timeout=CLASSES_BOUND)
    real = parse_rows(proc.stdout) if proc.returncode == 0 else Rows(why=proc.stderr[-200:])
    expect(real.why == "", f"classifier: `tools/gate-batch.sh --classes` ran: {real.why}")
    if real.why == "":
        want = closure_two_card(ROOT)
        expect(set(real.two) == want and len(want) > 0,
               f"classifier: the rows name BLOOMERY_CARD=both for the recipes whose closure does (rows only: {sorted(set(real.two) - want)}; text only: {sorted(want - set(real.two))})")

    if fails:
        for f in fails:
            print(f"affected-cards self-test FAIL: {f}", file=sys.stderr)
        print(f"affected-cards self-test: {len(fails)} failed", file=sys.stderr)
        return 1
    print("affected-cards self-test: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
