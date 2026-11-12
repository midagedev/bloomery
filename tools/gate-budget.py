#!/usr/bin/env python3
"""The verification budget, as a ratchet: what a landing and a release may cost in wall time (Mac, no box).

    tools/gate-budget.py check [--times FILE] [--classes FILE] [--only landing|blocking]
    tools/gate-budget.py --self-test

`check` holds two kinds of pin:
  landing   for each canonical change set (CANON: a real commit range of each kind a landing sees, a common crate's change, one
            family's own change, a gate binary's change alone), the predicted wall of its landing list: `tools/recipes.py
            affected RANGE`, then the batch's own plan of it on the fixture tier (`tools/gate-batch.sh --tier fixture --dry-run
            --list`, the `wall=` of its `gate-batch: predicted` line; no ledger, so no box is read). It must stay at or under the
            set's PIN.
  blocking  for each gate of the release's blocking set (`tools/recipes.py release-list --blocking`), the median of its last 5
            real-tier rows in the times file (gate-batch.sh's expected()) must stay at or under its PIN in BLOCKING_PINS; a
            blocking gate with no PIN row is red by name, so a gate that joins the set brings its budget with it.
A pin moves down when a change makes it reachable and up only with a dated reason beside it. Exit codes: 0 every pin holds,
1 a pin broken (each named, with its number and the pin), 2 a bad argument or input, 70 a tool it runs failed (named).
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import statistics
import subprocess
import sys
from collections.abc import Callable

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # `python3 -I` drops the script's directory
import recipes  # noqa: E402

ROOT = recipes.ROOT
TIMES_ROWS = 5  # the rows gate-batch.sh's expected() takes the median of


class BudgetError(Exception):
    def __init__(self, msg: str, rc: int = 2):
        super().__init__(msg)
        self.rc = rc


@dataclasses.dataclass(frozen=True)
class Canon:
    kind: str    # the kind of change
    rng: str     # the commit range that is one
    what: str    # what it touched
    pin_s: int   # the landing wall its list may predict, seconds


# PIN(2026-11-12): the landing walls the express rule holds (a landing of ≤ 15 min lands alone; a common crate's change runs the
# common gates on both cards in half an hour). The ranges are real one-commit changes of main.
CANON = (
    Canon("common crate", "18a085d38~1..18a085d38", "crates/gpu/src/host (a node tier's layer is skipped only when its picks name no expert)", 1800),
    Canon("one family", "928a94a7d~1..928a94a7d", "crates/gpu-deepseek41 (V4.1's prompt walk runs its groups by call_groups)", 900),
    Canon("gate bin", "392fef051~1..392fef051", "crates/gpu-gates/src/bin (the qwen38 serve gate's small-card clauses)", 900),
)

# PIN(2026-11-12): each blocking gate's real-tier median, today's median plus a fifth for the run-to-run spread, to the next 10 s.
BLOCKING_PINS = {
    "gate-gpu-ds41-twocard": 90,
    "gate-gpu-e2e": 50,
    "gate-gpu-glm5next-serve": 310,
    "gate-gpu-glm5next-twocard": 320,
    "gate-gpu-qwen38-serve": 750,
    "gate-gpu-qwen38-twocard": 300,
    "gate-ptx-spill": 400,
}


def times_path_of(arg: str | None) -> str:
    return arg or os.environ.get("BLOOMERY_GATE_TIMES") or os.path.expanduser("~/.cache/bloomery/gate-times.tsv")


def affected(rng: str) -> list[str]:
    """`tools/recipes.py affected RANGE`'s raw lines, which the batch reads as a --list (its header and reasons are no items)."""
    proc = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "recipes.py"), "affected", rng], capture_output=True, text=True, cwd=ROOT)
    if proc.returncode != 0:
        raise BudgetError(f"`tools/recipes.py affected {rng}` failed (exit {proc.returncode}): {proc.stderr.strip()[-300:]}", 70)
    return proc.stdout.splitlines()


def plan(names: list[str]) -> tuple[int, str, int]:
    """The batch's own plan of `names` on the fixture tier: (wall seconds, its terms, the items it runs)."""
    try:
        return recipes.batch_plan(names, recipes.FIXTURE_TIER)
    except recipes.RecipeError as e:
        raise BudgetError(str(e), 70) from e


def blocking(classes: str | None) -> list[str]:
    """The release's blocking set of this tree (recipes.blocking_set), told against the newest release tag."""
    tables = recipes.class_tables(ROOT)
    rows = recipes.read_class_file(classes) if classes else recipes.fixture_class_rows(ROOT)
    rs = recipes.load_justfile(os.path.join(ROOT, "justfile"))
    tags = recipes.release_tags(ROOT)
    new = recipes.families_new_between(ROOT, tables, tags[0]) if tags else set()
    return recipes.blocking_set(tables, rows, rs, new).names


def medians(path: str) -> dict[str, int]:
    """Each recipe's median of its last TIMES_ROWS real-tier rows (recipes.times_rows: the plain rows, never a fixture item's)."""
    _fixture, plain = recipes.times_rows(path)
    return {n: int(statistics.median([s for _, s in rows[-TIMES_ROWS:]]) + 0.5) for n, rows in plain.items() if rows}


@dataclasses.dataclass
class Verdict:
    lines: list[str]
    red: int


def judge(canon: tuple[Canon, ...], pins: dict[str, int], blocking_names: list[str], med: dict[str, int],
          affected_of: Callable[[str], list[str]], plan_of: Callable[[list[str]], tuple[int, str, int]], only: str = "") -> Verdict:
    lines: list[str] = []
    red = 0
    if only != "blocking":
        for c in canon:
            wall, terms, n_items = plan_of(affected_of(c.rng))
            ok = wall <= c.pin_s
            red += not ok
            lines.append(f"{'ok ' if ok else 'RED'}  landing {c.kind} ({c.rng}: {c.what}): {n_items} item(s) to run, predicted wall {wall} s, "
                         f"pin {c.pin_s} s{'' if ok else f', over by {wall - c.pin_s} s'} [{terms}]")
    if only != "landing":
        for n in blocking_names:
            if n not in pins:
                red += 1
                lines.append(f"RED  blocking {n}: no PIN row in BLOCKING_PINS: a gate that joins the blocking set brings its budget")
                continue
            m = med.get(n)
            if m is None:
                lines.append(f"ok   blocking {n}: no real-tier row in the times file yet, pin {pins[n]} s")
                continue
            ok = m <= pins[n]
            red += not ok
            lines.append(f"{'ok ' if ok else 'RED'}  blocking {n}: median {m} s, pin {pins[n]} s{'' if ok else f', over by {m - pins[n]} s'}")
        gone = sorted(set(pins) - set(blocking_names))
        if gone:
            red += len(gone)
            lines.append(f"RED  blocking: PIN rows for gates outside the blocking set ({' '.join(gone)}): drop them with the change that moved them")
        if blocking_names:
            total = sum(med.get(n, 0) for n in blocking_names)
            lines.append(f"     blocking: {len(blocking_names)} gate(s), the medians sum {total} s (serial; the batch's plan packs them)")
    return Verdict(lines, red)


def cmd_check(args: argparse.Namespace) -> int:
    names = [] if args.only == "landing" else blocking(args.classes)
    v = judge(CANON, BLOCKING_PINS, names, medians(times_path_of(args.times)), affected, plan, args.only or "")
    for ln in v.lines:
        print(ln)
    print(f"gate-budget: {v.red} pin(s) broken")
    return 1 if v.red else 0


def self_test() -> int:
    fails: list[str] = []

    def expect(cond: bool, msg: str) -> None:
        if not cond:
            fails.append(msg)

    canon = (Canon("common crate", "a~1..a", "crates/x", 1000), Canon("gate bin", "b~1..b", "crates/gpu-gates/src/bin", 300))
    lists = {"a~1..a": ["g1", "g2", "g3"], "b~1..b": ["g4"]}
    walls = {("g1", "g2", "g3"): 900, ("g4",): 200}
    plan_of = lambda ns: (walls[tuple(ns)], "laneA=1s", len(ns))  # noqa: E731
    pins = {"blk-a": 100, "blk-b": 200}
    med = {"blk-a": 90, "blk-b": 150}
    v = judge(canon, pins, ["blk-a", "blk-b"], med, lambda r: lists[r], plan_of)
    expect(v.red == 0 and all(ln.startswith(("ok ", "     ")) for ln in v.lines), f"green: every pin holds: {v.lines}")
    expect(any("landing common crate (a~1..a: crates/x): 3 item(s) to run, predicted wall 900 s, pin 1000 s" in ln for ln in v.lines),
           f"green: the landing line names the set, its range, its items, its wall and its pin: {v.lines}")
    walls[("g1", "g2", "g3")] = 1001
    v = judge(canon, pins, ["blk-a", "blk-b"], med, lambda r: lists[r], plan_of)
    expect(v.red == 1 and any(ln.startswith("RED  landing common crate") and "over by 1 s" in ln for ln in v.lines),
           f"red: a landing wall past its pin is named with the excess: {v.lines}")
    walls[("g1", "g2", "g3")] = 1000
    v = judge(canon, pins, ["blk-a", "blk-b"], med, lambda r: lists[r], plan_of)
    expect(v.red == 0, f"green: a wall at its pin holds: {v.lines}")
    v = judge(canon, pins, ["blk-a", "blk-b"], {**med, "blk-b": 260}, lambda r: lists[r], plan_of)
    expect(v.red == 1 and any(ln == "RED  blocking blk-b: median 260 s, pin 200 s, over by 60 s" for ln in v.lines),
           f"red: a blocking median past its pin: {v.lines}")
    v = judge(canon, pins, ["blk-a", "blk-b", "blk-new"], med, lambda r: lists[r], plan_of)
    expect(v.red == 1 and any("RED  blocking blk-new: no PIN row" in ln for ln in v.lines), f"red: a blocking gate with no pin: {v.lines}")
    v = judge(canon, pins, ["blk-a"], med, lambda r: lists[r], plan_of)
    expect(v.red == 1 and any("PIN rows for gates outside the blocking set (blk-b)" in ln for ln in v.lines), f"red: a stale pin row: {v.lines}")
    v = judge(canon, pins, ["blk-a", "blk-b"], {"blk-a": 90}, lambda r: lists[r], plan_of)
    expect(v.red == 0 and any("blk-b: no real-tier row in the times file yet" in ln for ln in v.lines), f"a gate with no row yet is named, not red: {v.lines}")
    v = judge(canon, pins, ["blk-a", "blk-b"], {**med, "blk-b": 999}, lambda r: lists[r], plan_of, "landing")
    expect(v.red == 0 and not any("blocking" in ln for ln in v.lines), f"--only landing reads no blocking pin: {v.lines}")
    # the medians: the last five real-tier rows, never a fixture item's
    import tempfile
    with tempfile.TemporaryDirectory(prefix="gate-budget-") as tmp:
        tp = os.path.join(tmp, "times.tsv")
        with open(tp, "w", encoding="utf-8") as fh:
            for s in (10, 500, 20, 30, 40, 50):
                fh.write(f"blk-a\tA\t3090\t{s}\t2026-11-12T00:00:00+0900\n")
            fh.write("blk-a@BLOOMERY_TIER=fixture\tA\t3090\t1\t2026-11-12T00:00:00+0900\n")
        expect(medians(tp) == {"blk-a": 40}, f"medians: the last 5 real-tier rows (500 20 30 40 50 -> 40): {medians(tp)}")
    # the tree's own pins: a canonical range each kind, one pin a blocking gate of the tree's table
    expect([c.kind for c in CANON] == ["common crate", "one family", "gate bin"] and all(c.pin_s > 0 and ".." in c.rng for c in CANON),
           f"the canonical sets: one of each kind, each a range with a pin: {CANON}")
    if fails:
        for f in fails:
            print(f"gate-budget self-test FAIL: {f}", file=sys.stderr)
        print(f"gate-budget self-test: {len(fails)} failed", file=sys.stderr)
        return 1
    print("gate-budget self-test: ok")
    return 0


def main(argv: list[str]) -> int:
    if argv == ["--self-test"]:
        return self_test()
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="every landing and blocking pin; rc 1 names each one broken")
    c.add_argument("--times", help="the times file (default BLOOMERY_GATE_TIMES, else ~/.cache/bloomery/gate-times.tsv)")
    c.add_argument("--classes", help="a saved `tools/gate-batch.sh --tier fixture --classes` (default: run it)")
    c.add_argument("--only", choices=["landing", "blocking"], help="one kind of pin")
    args = ap.parse_args(argv)
    try:
        return cmd_check(args)
    except (BudgetError, recipes.RecipeError) as e:
        print(f"gate-budget.py: {e}", file=sys.stderr)
        return e.rc if isinstance(e, BudgetError) else 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
