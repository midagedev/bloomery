#!/usr/bin/env python3
"""pr-gate-lines.py — a gate-batch log as the pull request body's gate lines.

  tools/pr-gate-lines.py LOG [--attach] [--logs DIR]   one markdown line per item, on stdout
  tools/pr-gate-lines.py --self-test

LOG is a tools/gate-batch.sh run.log (or the batch's own stdout, which carries the same item lines). Lines it reads
(the formats are printed by tools/gate-batch.sh's per-item and final steps):
  plan …                                    the predicted sums, skipped
  <stem> rc=<n> <s>s try=<t> lane=<L> tier=<T> […] [item=<recipe[@ENV][:ARGS]>]
  <stem> rc=skip|deferred|yield|unclaimed …  a ledger skip, a deferral, a yield to a hold, a pack item no lane ran
  DONE total=<n> red=<n> tier=<T> …         the batch's closing line
  gate-batch: …, logs in DIR   red: …       the stdout's header and red-list lines
Any other line is a named error. Each item prints as
  - `just <recipe>` (lane <x>): ok|FAIL rc <n>|skip|deferred|yield|unclaimed, <wall> s
and the first `gate-batch: … logs in DIR` line (the stdout's) or the DONE line is printed above them as the batch's identity.

rc 84 is tools/boxq.py's: the Mac lost the daemon for three polls and the job is left running on the box, so the item's
result is not known here. The job id is the last `boxq: <name> is job <id>` line of the item's own log (DIR/g-<stem>.log,
next to LOG unless --logs names another directory). With --attach the id is followed with `boxq.py attach <id>` (its
output goes to stderr) and the line takes the job's rc from the box: `ok (rc 84 on the Mac, attached: box rc 0)` or
`FAIL box rc <n>`. Without --attach, with no id in the log, or when the attach itself ends 84 again, the line says the job
is not attached and the exit is 3; such an item is never printed as ok or as a plain FAIL.
PR_GATE_LINES_BOXQ names another boxq.py (the self-test's stub); it runs under python3.

Exit: 0 every item ok (a ledger skip, a deferral and a yield are not failures, as in gate-batch's own exit); 1 some item
FAIL; 3 an rc 84 not resolved (a FAIL elsewhere in the log takes the 1); 64 usage; 65 an unreadable log or a line this
tool cannot parse.
"""

import contextlib
import io
import os
import re
import subprocess
import sys
import tempfile

OK, FAIL, UNRESOLVED, USAGE, BAD_LOG = 0, 1, 3, 64, 65
DOWN = 84  # tools/boxq.py: the daemon is not reachable, the job is left as it is

ITEM_RE = re.compile(r"^(?P<stem>\S+) rc=(?P<rc>\S+)(?P<rest> .*)?$")
WALL_RE = re.compile(r" (\d+)s(?: |$)")
LANE_RE = re.compile(r" lane=(\S+)")
ITEM_FIELD_RE = re.compile(r" item=(.*)$")
JOB_RE = re.compile(r"^boxq: .* is job ([A-Za-z0-9._-]+)\s*$")
LOGS_IN_RE = re.compile(r"^gate-batch: .*logs in \S+")
PASSTHROUGH = ("plan ", "gate-batch: ", "red: ", "STALL ")
SPECIAL_RC = ("skip", "deferred", "yield", "unclaimed")


class LogError(Exception):
    """A log this tool cannot read, or a line of it outside the formats above: exit 65."""


class Item:
    def __init__(self, stem, rc, rest, lineno):
        self.stem, self.rc, self.lineno = stem, rc, lineno
        self.lane = (LANE_RE.search(rest) or [None, "?"])[1]
        wall = WALL_RE.search(rest)
        self.wall = wall.group(1) if wall else None
        field = ITEM_FIELD_RE.search(rest)
        self.recipe = field.group(1).strip() if field else stem
        self.rest = rest


def parse_log(path):
    """(identity lines, items) of one log; a line outside the formats raises LogError naming it."""
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.read().splitlines()
    except (OSError, UnicodeDecodeError) as e:
        raise LogError(f"{path}: cannot read the log: {e}") from e
    ident, items, done = [], [], None
    for n, line in enumerate(lines, 1):
        if not line.strip():
            continue
        if line.startswith("DONE "):
            done = line
            continue
        if line.startswith(PASSTHROUGH):
            if LOGS_IN_RE.match(line) and not ident:
                ident.append(line[len("gate-batch: "):])
            continue
        m = ITEM_RE.match(line)
        if not m:
            raise LogError(f"{path}:{n}: not a gate-batch log line: {line!r}")
        rc, rest = m["rc"], m["rest"] or ""
        if not rc.isdigit() and rc not in SPECIAL_RC:
            raise LogError(f"{path}:{n}: rc={rc} is neither a number nor one of {', '.join(SPECIAL_RC)}: {line!r}")
        if rc.isdigit() and not WALL_RE.search(rest):
            raise LogError(f"{path}:{n}: an item with rc={rc} carries no `<s>s` wall: {line!r}")
        items.append(Item(m["stem"], rc, rest, n))
    if not items:
        raise LogError(f"{path}: no item line in the log")
    if done:
        ident.append(done)
    return ident, items


def job_id(logs_dir, stem):
    """The last `boxq: <name> is job <id>` of DIR/g-<stem>.log, or None (a missing log or no such line)."""
    try:
        with open(os.path.join(logs_dir, f"g-{stem}.log"), encoding="utf-8", errors="replace") as f:
            ids = [m[1] for m in map(JOB_RE.match, f.read().splitlines()) if m]
    except OSError:
        return None
    return ids[-1] if ids else None


def attach(jid):
    """boxq.py attach <jid>: the job's rc from the box, its output on our stderr."""
    boxq = os.environ.get("PR_GATE_LINES_BOXQ") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "boxq.py")
    sys.stdout.flush()
    sys.stderr.flush()
    return subprocess.run([sys.executable, boxq, "attach", jid], stdin=subprocess.DEVNULL, stdout=2, check=False).returncode


def render(item, logs_dir, do_attach):
    """(markdown line, status) with status OK, FAIL or UNRESOLVED."""
    head = f"- `just {item.recipe}` (lane {item.lane}): "
    wall = f", {item.wall} s" if item.wall is not None else ""
    if item.rc == "0":
        return head + "ok" + wall, OK
    if item.rc == "skip":
        return head + "skip (a green run of the same inputs is in the ledger)", OK
    if item.rc == "deferred":
        return head + "deferred (runs in the other tier)", OK
    if item.rc == "yield":
        return head + "yield (a hold of another batch was up, not run)", OK
    if item.rc == "unclaimed":
        return head + "FAIL unclaimed (no lane ran it)", FAIL
    if int(item.rc) != DOWN:
        return head + f"FAIL rc {item.rc}" + wall, FAIL
    jid = job_id(logs_dir, item.stem)
    shown = jid or "unknown"
    unattached = head + f"rc 84: daemon unreachable, job {shown} not attached — rerun with --attach"
    if not do_attach:
        return unattached + wall, UNRESOLVED
    if jid is None:
        return unattached + " (the item's log holds no job id)" + wall, UNRESOLVED
    rc = attach(jid)
    if rc == DOWN:
        return head + f"rc 84: daemon unreachable, job {jid} not attached (the attach ended 84 too)" + wall, UNRESOLVED
    if rc == 0:
        return head + "ok (rc 84 on the Mac, attached: box rc 0)" + wall, OK
    return head + f"FAIL box rc {rc} (rc 84 on the Mac, attached)" + wall, FAIL


def run(log, do_attach, logs_dir):
    try:
        ident, items = parse_log(log)
    except LogError as e:
        print(f"pr-gate-lines: {e}", file=sys.stderr)
        return BAD_LOG
    logs_dir = logs_dir or os.path.dirname(os.path.abspath(log))
    for line in ident:
        print(f"Gate batch: {line}")
    statuses = []
    for item in items:
        line, status = render(item, logs_dir, do_attach)
        print(line, flush=True)
        statuses.append(status)
    if FAIL in statuses:
        return FAIL
    return UNRESOLVED if UNRESOLVED in statuses else OK


def main(argv):
    if argv == ["--self-test"]:
        return self_test()
    do_attach, logs_dir, paths, it = False, None, [], iter(argv)
    for a in it:
        if a == "--attach":
            do_attach = True
        elif a == "--logs":
            logs_dir = next(it, None)
        elif a.startswith("-"):
            print(f"pr-gate-lines: unknown option {a}\n{__doc__.split(chr(10) + chr(10))[0]}", file=sys.stderr)
            return USAGE
        else:
            paths.append(a)
    if len(paths) != 1 or (logs_dir is None and "--logs" in argv):
        print("usage: pr-gate-lines.py LOG [--attach] [--logs DIR] | --self-test", file=sys.stderr)
        return USAGE
    return run(paths[0], do_attach, logs_dir)


# --self-test. The fixture lines copy the shapes tools/gate-batch.sh prints: the item line is its `line=` at
# `"${P_STEM[$i]} rc=$rc ${s}s try=$try lane=$lane tier=$TIER"` plus the `waited=`, `cold=`, `item=` suffixes (the
# per-item function, `line=` right before `echo "$line" >> "$RUNLOG"`); the skip line `"${P_STEM[$i]} rc=skip … lane=$2
# tier=$TIER"`, the deferred line `"${DP_STEM[$i]} rc=deferred ${DP_KIND[$i]} lane=- tier=$TIER deferred=1"`, the
# yield line `"${P_STEM[$i]} rc=yield held by $yowner: not retried (…) ${s}s try=$try lane=$lane tier=$TIER"`, the
# `plan …` line and `done_line="DONE total=$NT red=$red tier=$TIER …"` are its `echo … >> "$RUNLOG"` calls, and the
# `boxq: <name> is job <id>` line is tools/boxq.py's cmd_run `say(...)`, which box.sh's call leaves in the item's log.

STUB = """import os, sys
print("stub out", sys.argv[1:])
open(os.environ["STUB_CALLS"], "a").write(" ".join(sys.argv[1:]) + "\\n")
want = {"j-ok": 0, "j-red": 3, "j-down": 84}
sys.exit(want[sys.argv[2]])
"""

GREEN = [
    "plan laneA=60s laneB=40s laneX=0s wall=60s defaults=0 (predicted, derived from /x/times.tsv)",
    "gate-fmt rc=0 4s try=1 lane=B tier=real",
    "gate-qdot rc=0 61s try=1 lane=A tier=real waited=3s item=gate-qdot:--cases 4",
    "gate-gpu-x rc=skip green-at=abc1234 2026-10-14T10:00:00+0900 lane=A tier=real",
    "gate-gpu-y rc=deferred real-only lane=- tier=fixture deferred=1",
    "DONE total=4 red=0 tier=real wall=63s laneA=61s laneB=4s",
]


def self_test():
    fails = []

    def check(ok, what):
        print(("ok " if ok else "FAIL ") + what)
        if not ok:
            fails.append(what)

    def call(args, **env):
        out, err = io.StringIO(), io.StringIO()
        old = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                rc = main(args)
        finally:
            for k, v in old.items():
                os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
        return rc, out.getvalue(), err.getvalue()

    with tempfile.TemporaryDirectory(prefix="pr-gate-lines-") as t:
        def write(name, lines):
            p = os.path.join(t, name)
            with open(p, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            return p

        stub = write("stub-boxq.py", STUB.rstrip("\n").split("\n"))
        calls = os.path.join(t, "stub-calls.txt")
        env = {"PR_GATE_LINES_BOXQ": stub, "STUB_CALLS": calls}

        rc, out, _ = call([write("green.log", GREEN)])
        check(rc == 0, f"all ok exits 0: {rc}")
        check("- `just gate-fmt` (lane B): ok, 4 s" in out, f"an ok line: {out!r}")
        check("- `just gate-qdot:--cases 4` (lane A): ok, 61 s" in out, "the item= field names the recipe with its ARGS")
        check("skip" in out and "deferred" in out and "FAIL" not in out, "a skip and a deferral are no failure")
        check("Gate batch: DONE total=4" in out, "the DONE line is the identity when the log has no header line")

        stdout_style = ["gate-batch: 2 items, lanes 2, logs in /h/batches/tree-a/1"] + GREEN[1:3]
        rc, out, _ = call([write("stdout.log", stdout_style)])
        check(rc == 0 and out.startswith("Gate batch: 2 items, lanes 2, logs in /h/batches/tree-a/1\n"),
              f"the stdout's header line is the identity: {rc} {out!r}")

        red = GREEN[:2] + ["gate-qdot rc=1 61s try=1 lane=A tier=real"] + GREEN[2:]
        rc, out, _ = call([write("red.log", red)])
        check(rc == 1 and "- `just gate-qdot` (lane A): FAIL rc 1, 61 s" in out, f"one FAIL exits 1: {rc} {out!r}")

        down = ["gate-gpu-z rc=84 30s try=1 lane=A tier=real"]
        item_log = ["=== try 1 2026-10-14 10:00:00 lane=A: just gate-gpu-z", "boxq: gate-gpu-z is job j-ok"]
        write("g-gate-gpu-z.log", item_log)
        rc, out, _ = call([write("down.log", down)])
        check(rc == 3 and "rc 84: daemon unreachable, job j-ok not attached — rerun with --attach" in out
              and " ok" not in out and "FAIL" not in out, f"rc 84 without --attach exits 3, never ok or FAIL: {rc} {out!r}")

        rc, out, _ = call([os.path.join(t, "down.log"), "--attach"], **env)
        got = open(calls, encoding="utf-8").read() if os.path.exists(calls) else "(the stub never ran)"
        check(rc == 0 and "ok (rc 84 on the Mac, attached: box rc 0)" in out and got == "attach j-ok\n",
              f"rc 84 with --attach takes the box's rc 0 from `attach j-ok`: {rc} {out!r} {got!r}")

        write("g-gate-gpu-z.log", item_log + ["boxq: gate-gpu-z is job j-red"])
        rc, out, _ = call([os.path.join(t, "down.log"), "--attach"], **env)
        check(rc == 1 and "FAIL box rc 3" in out and "ok" not in out.split(":", 1)[1],
              f"the last job id of the log is attached, box rc 3 is a FAIL: {rc} {out!r}")

        write("g-gate-gpu-z.log", ["boxq: gate-gpu-z is job j-down"])
        rc, out, _ = call([os.path.join(t, "down.log"), "--attach"], **env)
        check(rc == 3 and "job j-down not attached" in out, f"an attach that ends 84 stays unresolved: {rc} {out!r}")

        os.remove(os.path.join(t, "g-gate-gpu-z.log"))
        rc, out, _ = call([os.path.join(t, "down.log"), "--attach"], **env)
        check(rc == 3 and "job unknown not attached" in out, f"no job id in the log: {rc} {out!r}")

        mixed = red[:3] + down
        rc, _, _ = call([write("mixed.log", mixed)])
        check(rc == 1, f"a FAIL beside an unresolved rc 84 exits 1: {rc}")

        for name, lines, want in (
            ("garbage", GREEN[:2] + ["this is not a gate-batch line"], "not a gate-batch log line"),
            ("badrc", ["gate-fmt rc=abc 4s try=1 lane=A tier=real"], "neither a number"),
            ("nowall", ["gate-fmt rc=0 try=1 lane=A tier=real"], "no `<s>s` wall"),
            ("empty", ["plan laneA=0s"], "no item line"),
        ):
            rc, _, err = call([write(f"{name}.log", lines)])
            check(rc == 65 and want in err, f"{name} log exits 65 and names the line: {rc} {err!r}")
        rc, _, err = call([os.path.join(t, "absent.log")])
        check(rc == 65 and "cannot read the log" in err, f"a missing log exits 65: {rc} {err!r}")

    for args in ([], ["--nope", "x"], ["a", "b"], ["--logs"]):
        rc, _, _ = call(args)
        check(rc == 64, f"usage {args} exits 64: {rc}")

    if fails:
        print(f"pr-gate-lines: self-test {len(fails)} failed", file=sys.stderr)
        return 1
    print("pr-gate-lines: self-test ok")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
