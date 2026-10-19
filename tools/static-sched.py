#!/usr/bin/env python3
"""static-sched.py — the one scheduler of the Mac static tier's jobs (tools/mac-static.sh's steps and check-recipes' blocks).

  static-sched.py run --out DIR [--max N] [--times FILE] [--keys FILE --ledger FILE --tree NAME [--no-skip]] JOBS
  static-sched.py --self-test

JOBS is a file of one job a line, four tab-separated fields: NAME, ATTRS, KEYSPEC, COMMAND (tools/recipes.py read_jobs reads
it, here and for static-key). COMMAND runs under
`bash -c` in the current directory, in a process group of its own, its stdout and stderr to DIR/NAME.log. ATTRS is `-` or
a comma list of:
  after=NAME   the job starts only after NAME ended (whatever its rc)
  alone        the job runs with no other job running (the first job to start when it has no `after`)
  cpus=K       the job counts K of the --max slots while it runs (a cargo step runs K compilers at once)
KEYSPEC is what tools/recipes.py static-key reads (`-`: never skipped); this file never reads it.

At most --max (default: the cpu count) slots are taken at once. Of the ready jobs, the one with the longest recorded
wall starts first: the median of its last three rows in --times (a job with no row counts as the longest, so a new job
starts early). With --keys (NAME<TAB>KEY<TAB>NOTE, the static-key output) and --ledger, a job whose key is green in
the ledger is not run: it writes rc 0, wall 0 and DIR/NAME.skip naming the green row. A keyed job that ends rc 0
appends `KEY<TAB>green<TAB>DATE<TAB>TREE` to the ledger (the format tools/recipes.py read_shape_ledger reads; a key never
equals a shape key, the kinds hash different headers); --no-skip runs every job and still appends. A job that ran
appends `NAME<TAB>WALL<TAB>DATE` to --times.

Per job DIR gets NAME.log, NAME.rc, NAME.wall (whole seconds) and, when skipped, NAME.skip; DIR/pids holds each started
job's pid (its process group) at spawn; DIR/sched.tsv one row a job: name, start and end offsets in seconds, rc, how it
ran (run or skip). A TERM, INT or HUP ends every running job's process group and the groups of the job's descendants (a job
may run a scheduler of its own, whose jobs are in groups of their own); a member that ignores the TERM gets a KILL once the
jobs' shells are gone, and the scheduler then exits 128 + the signal. An error that is no signal (a file it cannot write)
ends the running jobs the same way before it propagates. A signal that arrives while a job is being started waits until the
job is registered; one that arrives during the cleanup is ignored. Nothing is printed on stdout: the callers report from DIR.

Exit: 0 every job ran (or skipped) and wrote its rc, whatever the rcs; 64 usage or a malformed JOBS, keys or times line
(named), or a negative --max; 128+n a signal.
"""

import argparse
import datetime
import os
import shlex
import signal
import statistics
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recipes import Job, read_jobs, read_shape_ledger  # noqa: E402  (the one JOBS reader and the one ledger reader)

USAGE_RC = 64
STOP_SIGNALS = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)


def read_times(path: str | None) -> dict[str, float]:
    """name -> the median of its last three recorded walls. A row that is not NAME, a finite nonnegative WALL and DATE is
    refused by its line: the order of the jobs reads these walls."""
    rows: dict[str, list[float]] = {}
    if not path:
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            for n, ln in enumerate(fh, 1):
                f = ln.rstrip("\n").split("\t")
                try:
                    if len(f) != 3 or not f[0]:
                        raise ValueError("want NAME WALL DATE")
                    wall = float(f[1])
                    if not 0 <= wall < float("inf"):
                        raise ValueError("wall must be finite and nonnegative")
                except ValueError as err:
                    raise ValueError(f"{path}:{n}: {err}") from err
                rows.setdefault(f[0], []).append(wall)
    except FileNotFoundError:
        return {}
    return {k: statistics.median(v[-3:]) for k, v in rows.items()}


def read_keys(path: str | None) -> dict[str, tuple[str, str]]:
    """name -> (key, note); a key of `-` is a job that is never skipped."""
    keys: dict[str, tuple[str, str]] = {}
    if not path:
        return keys
    with open(path, encoding="utf-8") as fh:
        for n, ln in enumerate(fh, 1):
            f = ln.rstrip("\n").split("\t")
            if len(f) != 3:
                raise ValueError(f"{path}:{n}: {len(f)} fields, want 3 (NAME KEY NOTE)")
            keys[f[0]] = (f[1], f[2])
    return keys


def append_line(path: str, line: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(fd, line.encode())
    finally:
        os.close(fd)


def write(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


class Stop(Exception):
    def __init__(self, sig: int):
        self.sig = sig


def signal_group(pgid: int, sig: int) -> None:
    """Signal a job's process group. A group with no live member is gone: ESRCH, and EPERM on Darwin while its members are
    zombies."""
    try:
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def process_table() -> list[tuple[int, int, int]]:
    """(pid, ppid, pgid) of every process."""
    out = subprocess.check_output(["ps", "-eo", "pid=,ppid=,pgid="], text=True)
    return [(int(a), int(b), int(c)) for a, b, c in (ln.split() for ln in out.splitlines())]


def descendant_groups(roots: list[int], table: list[tuple[int, int, int]]) -> set[int]:
    """The process groups of the roots (each leads a group of its own) and of every process below them: a child of a member, or a
    member of a group found so far. A job that runs a scheduler of its own has jobs in groups its own group does not hold."""
    groups = set(roots)
    reach = set(groups)
    grew = True
    while grew:
        grew = False
        for pid, parent, group in table:
            if pid not in reach and (parent in reach or group in groups):
                reach.add(pid)
                groups.add(group)
                grew = True
    return groups


def end_jobs(running: dict[int, tuple[Job, subprocess.Popen, float]]) -> None:
    """End the running jobs: a TERM to each job's group and to the groups of its descendants, read before the first signal. The
    job's shell ends on the TERM at once while a member may ignore it, so once every leader has gone (or the 10 s are up) each
    group gets a KILL: nothing of a cancelled job outlives the scheduler, a nested scheduler's jobs included."""
    for s in STOP_SIGNALS:  # a second signal must not cut the cleanup short
        signal.signal(s, signal.SIG_IGN)
    try:
        groups = descendant_groups(list(running), process_table())
    except (OSError, ValueError, subprocess.CalledProcessError) as err:
        print(f"static-sched: cannot read the process table ({err}): only the jobs' own groups are ended", file=sys.stderr)
        groups = set(running)
    for group in groups:
        signal_group(group, signal.SIGTERM)
    deadline = time.monotonic() + 10
    for _, p, _ in running.values():
        try:
            p.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass
    for group in groups:
        signal_group(group, signal.SIGKILL)
    for _, p, _ in running.values():
        p.wait()


def run(args: argparse.Namespace) -> int:
    try:
        jobs = read_jobs(args.jobs)
        keys = read_keys(args.keys)
        times = read_times(args.times)
    except (OSError, ValueError) as err:
        print(f"static-sched: {err}", file=sys.stderr)
        return USAGE_RC
    if args.ledger and not args.keys:
        print("static-sched: --ledger reads the keys --keys names; give --keys too", file=sys.stderr)
        return USAGE_RC
    if args.max < 0:
        print("static-sched: --max must be nonnegative", file=sys.stderr)
        return USAGE_RC
    slots = args.max or os.cpu_count() or 1
    for j in jobs:
        j.cpus = min(j.cpus, slots)
    green = read_shape_ledger(args.ledger)[0] if args.ledger else {}
    out = args.out
    os.makedirs(out, exist_ok=True)
    for j in jobs:
        for ext in ("rc", "wall", "skip"):
            try:
                os.remove(os.path.join(out, f"{j.name}.{ext}"))
            except FileNotFoundError:
                pass
    write(os.path.join(out, "pids"), "")
    sched_rows: list[str] = []
    t0 = time.monotonic()
    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    done: set[str] = set()
    running: dict[int, tuple[Job, subprocess.Popen, float]] = {}

    def finish(j: Job, rc: int, wall: float, how: str, start: float) -> None:
        write(os.path.join(out, f"{j.name}.rc"), f"{rc}\n")
        write(os.path.join(out, f"{j.name}.wall"), f"{int(wall)}\n")
        sched_rows.append(f"{j.name}\t{start - t0:.1f}\t{time.monotonic() - t0:.1f}\t{rc}\t{how}\n")
        done.add(j.name)

    # The skips first: a green key costs no slot.
    pending: list[Job] = []
    for j in jobs:
        key, note = keys.get(j.name, ("-", ""))
        if args.ledger and not args.no_skip and key != "-" and key in green:
            d, tree = green[key]
            write(os.path.join(out, f"{j.name}.skip"), f"green at {d} in {tree}, key {key[:12]}{f' ({note})' if note else ''}\n")
            write(os.path.join(out, f"{j.name}.log"), f"static-sched: skipped, its input key is green at {d} in {tree} ({key[:12]})\n")
            finish(j, 0, 0, "skip", time.monotonic())
        else:
            pending.append(j)
    # Longest first; a job never timed counts as the longest. The sort is stable, so ties keep the file's order.
    pending.sort(key=lambda j: -times.get(j.name, float("inf")))

    def on_signal(sig, _frame):
        raise Stop(sig)

    for s in STOP_SIGNALS:
        signal.signal(s, on_signal)

    def used() -> int:
        return sum(j.cpus for j, _, _ in running.values())

    def ready(j: Job) -> bool:
        return j.after is None or j.after in done

    def start(j: Job) -> None:
        # A signal that arrives from the spawn to the registration waits until the job is in `running`, so the cleanup below
        # reaches its group.
        deferred: list[int] = []
        handlers = {s: signal.signal(s, lambda sig, _frame: deferred.append(sig)) for s in STOP_SIGNALS}
        try:
            with open(os.path.join(out, f"{j.name}.log"), "wb") as log:
                p = subprocess.Popen(["bash", "-c", j.command], stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                     start_new_session=True)
            running[p.pid] = (j, p, time.monotonic())
            append_line(os.path.join(out, "pids"), f"{j.name} {p.pid}\n")
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
        if deferred:
            raise Stop(deferred[0])

    stopped = 0
    try:
        while pending or running:
            alone_running = any(j.alone for j, _, _ in running.values())
            started = True
            while started and not alone_running:
                started = False
                for j in pending:
                    if not ready(j):
                        continue
                    if j.alone:
                        if running:
                            break  # the alone job waits for the slots to drain; nothing starts past it
                        start(j)
                        pending.remove(j)
                        alone_running = True
                        started = True
                        break
                    if used() + j.cpus <= slots or not running:
                        start(j)
                        pending.remove(j)
                        started = True
                        break
            if not running:
                if pending:  # every pending job waits on one that never ends: a cycle of after= names
                    names = " ".join(j.name for j in pending)
                    print(f"static-sched: jobs that wait on each other through after=: {names}", file=sys.stderr)
                    return USAGE_RC
                break
            pid, status = os.wait()
            if pid not in running:
                continue
            j, p, st = running.pop(pid)
            p.returncode = os.waitstatus_to_exitcode(status)
            rc = p.returncode if p.returncode >= 0 else 128 - p.returncode
            wall = time.monotonic() - st
            finish(j, rc, wall, "run", st)
            if args.times:
                append_line(args.times, f"{j.name}\t{wall:.1f}\t{date}\n")
            key, _ = keys.get(j.name, ("-", ""))
            if rc == 0 and args.ledger and key != "-":
                append_line(args.ledger, f"{key}\tgreen\t{date}\t{args.tree}\n")
    except Stop as stop:
        stopped = stop.sig
    finally:
        if running:  # a signal, or an error of ours (a file it cannot write): no registered job outlives the scheduler
            end_jobs(running)
    write(os.path.join(out, "sched.tsv"), "".join(sched_rows))
    return 128 + stopped if stopped else 0


def self_test() -> int:
    fails: list[str] = []
    me = os.path.abspath(__file__)

    def expect(ok: bool, what: str) -> None:
        print(f"{'ok  ' if ok else 'FAIL'} {what}")
        if not ok:
            fails.append(what)

    with tempfile.TemporaryDirectory(prefix="static-sched-test.") as d:
        live = os.path.join(d, "live")
        os.mkdir(live)
        # A stub job: its file in `live` while it runs; the count it saw at its start (it and those already there) to
        # seen.<name>. A count is never more than were running: a job's file goes before its process ends, and a slot
        # is given again only after the scheduler reaped it.
        stub = os.path.join(d, "stub.sh")
        write(stub, f'touch "{live}/$1"; ls "{live}" | wc -l | tr -d " " > "{d}/seen.$1"; '
                    f'echo "$1 $(date +%s)" >> "{d}/order"; sleep "$2"; rm -f "{live}/$1"; exit "${{3:-0}}"\n')

        def jobs_file(lines: list[str]) -> str:
            p = os.path.join(d, "jobs.tsv")
            write(p, "".join(ln + "\n" for ln in lines))
            return p

        def sched(argv: list[str]) -> int:
            return subprocess.run([sys.executable, me, "run", *argv], capture_output=True, text=True).returncode

        def sched_err(argv: list[str]) -> tuple[int, str]:
            r = subprocess.run([sys.executable, me, "run", *argv], capture_output=True, text=True)
            return r.returncode, r.stderr

        def seen(name: str) -> int:
            with open(os.path.join(d, f"seen.{name}")) as fh:
                return int(fh.read())

        def reset() -> None:
            for f in os.listdir(d):
                if f.startswith("seen.") or f == "order":
                    os.remove(os.path.join(d, f))

        def rd(name: str) -> str:
            with open(os.path.join(d, name)) as fh:
                return fh.read().strip()

        # bounded: six 0.5 s jobs on two slots never run three at once, and all of them run
        names = [f"j{i}" for i in range(6)]
        jf = jobs_file([f"{n}\t-\t-\tbash {stub} {n} 0.5" for n in names])
        o = os.path.join(d, "o1")
        rc = sched(["--out", o, "--max", "2", jf])
        expect(rc == 0, f"a run of six jobs ends 0 (got {rc})")
        expect(all(rd(f"o1/{n}.rc") == "0" for n in names), "every job wrote rc 0")
        expect(max(seen(n) for n in names) <= 2, f"two slots: a job saw at most 2 running (saw {max(seen(n) for n in names)})")
        expect(len(rd("o1/pids").split("\n")) == 6, "every started job's pid is in DIR/pids")
        # the same jobs on six slots overlap (the witness sees more than one): the bound above is the scheduler's
        reset()
        rc = sched(["--out", os.path.join(d, "o2"), "--max", "6", jf])
        expect(max(seen(n) for n in names) >= 2, "six slots: the jobs overlap")
        # longest first: on one slot the recorded times give the order, a job never timed first
        reset()
        times = os.path.join(d, "times.tsv")
        write(times, "j0\t1\td\nj1\t30\td\nj2\t5\td\nj3\t2\td\nj4\t9\td\n")
        jf = jobs_file([f"{n}\t-\t-\tbash {stub} {n} 0" for n in names])
        rc = sched(["--out", os.path.join(d, "o3"), "--max", "1", "--times", times, jf])
        order = [ln.split()[0] for ln in rd("order").split("\n")]
        expect(order == ["j5", "j1", "j4", "j2", "j3", "j0"], f"one slot runs the never-timed job first, then the longest recorded (got {order})")
        with open(times) as fh:
            rows = [ln.split("\t")[0] for ln in fh.read().split("\n") if ln]
        expect(rows.count("j5") == 1, "a job that ran appends its wall to --times")
        # after= and alone: b waits for a; the alone job runs with nothing beside it
        reset()
        jf = jobs_file([f"a\t-\t-\tbash {stub} a 0.4", f"b\tafter=a\t-\tbash {stub} b 0", f"c\t-\t-\tbash {stub} c 0.4",
                        f"solo\talone\t-\tbash {stub} solo 0.3", f"d\t-\t-\tbash {stub} d 0.4"])
        rc = sched(["--out", os.path.join(d, "o4"), "--max", "4", jf])
        spans = {f[0]: (float(f[1]), float(f[2])) for f in (ln.split("\t") for ln in rd("o4/sched.tsv").split("\n"))}
        expect(spans["b"][0] >= spans["a"][1], f"after=a starts once a ended (a {spans['a']}, b {spans['b']})")
        expect(seen("solo") == 1, f"the alone job saw only itself running (saw {seen('solo')})")
        # cpus=K takes K slots: while a cpus=3 job runs on four slots, one other job runs beside it, never two
        reset()
        jf = jobs_file([f"big\tcpus=3\t-\tbash {stub} big 0.6"] + [f"s{i}\t-\t-\tbash {stub} s{i} 0.3" for i in range(4)])
        rc = sched(["--out", os.path.join(d, "o5"), "--max", "4", jf])
        spans = {f[0]: (float(f[1]), float(f[2])) for f in (ln.split("\t") for ln in rd("o5/sched.tsv").split("\n"))}
        b0, b1 = spans["big"]
        beside = max(sum(1 for n, (s0, s1) in spans.items() if n != "big" and s0 <= t < s1) for t in [b0] + [s0 for s0, _ in spans.values() if b0 <= s0 < b1])
        expect(beside == 1, f"a cpus=3 job on four slots runs beside one other job (beside {beside})")
        # rcs, the skip by a green key, the ledger row of a green keyed job, and no row for a red one
        reset()
        k_green, k_new, k_red = "a" * 64, "b" * 64, "c" * 64
        ledger = os.path.join(d, "ledger.tsv")
        write(ledger, f"{k_green}\tgreen\t2026-10-19 10:00\ttreeX\nmalformed line\n")
        keys = os.path.join(d, "keys.tsv")
        write(keys, f"g\t{k_green}\ttools/x.sh closure\nn\t{k_new}\t\nr\t{k_red}\t\nu\t-\ttree read whole\n")
        jf = jobs_file([f"g\t-\tx\tbash {stub} g 0", f"n\t-\tx\tbash {stub} n 0", f"r\t-\tx\tbash {stub} r 0 3", f"u\t-\t-\tbash {stub} u 0"])
        o = os.path.join(d, "o6")
        rc = sched(["--out", o, "--keys", keys, "--ledger", ledger, "--tree", "treeY", jf])
        expect(not os.path.exists(os.path.join(d, "seen.g")), "a job whose key is green does not run")
        expect(rd("o6/g.rc") == "0" and rd("o6/g.wall") == "0", "a skipped job writes rc 0 and wall 0")
        expect("green at 2026-10-19 10:00 in treeX" in rd("o6/g.skip"), f"the skip names the green row: {rd('o6/g.skip')}")
        expect(rd("o6/r.rc") == "3", "a red job's rc is its own")
        with open(ledger) as fh:
            led = fh.read()
        expect(f"{k_new}\tgreen\t" in led and led.rstrip().endswith("treeY"), "a green keyed job appends its key under --tree")
        expect(k_red not in led, "a red job appends nothing")
        expect(led.count("\n") == 3, "an unkeyed job appends nothing")
        # --no-skip runs the green job and appends it again
        reset()
        rc = sched(["--out", os.path.join(d, "o6b"), "--keys", keys, "--ledger", ledger, "--tree", "treeZ", "--no-skip", jobs_file([f"g\t-\tx\tbash {stub} g 0"])])
        with open(ledger) as fh:
            led = fh.read()
        expect(os.path.exists(os.path.join(d, "seen.g")) and not os.path.exists(os.path.join(d, "o6b", "g.skip")), "--no-skip runs a job whose key is green")
        expect(led.rstrip().endswith("treeZ") and led.count(k_green) == 2, "--no-skip still appends the green key")
        # usage: a malformed line, a duplicate, an after= that names nothing, a ledger without keys
        for bad, why in ((["x\t-\tbash"], "three fields"), (["x\t-\t-\ttrue", "x\t-\t-\ttrue"], "a name twice"),
                         (["x\tafter=y\t-\ttrue"], "after= of no job"), (["x\tcpus=0\t-\ttrue"], "cpus=0"),
                         (["x\tbogus\t-\ttrue"], "an unknown attribute")):
            jf = jobs_file(bad)
            expect(sched(["--out", os.path.join(d, "o7"), jf]) == USAGE_RC, f"{why} is refused, 64")
        jf = jobs_file(["x\t-\t-\ttrue"])
        expect(sched(["--out", os.path.join(d, "o7"), "--ledger", ledger, jf]) == USAGE_RC, "--ledger without --keys is refused, 64")
        expect(sched(["--out", os.path.join(d, "o7"), "--max", "-1", jf]) == USAGE_RC, "a negative --max is refused, 64")
        # a malformed timing row is named by its line and refused, 64: a good row and then a bad one for a job must not keep the
        # old estimate, and a short row must not drop out of the history
        tbad = os.path.join(d, "times-bad.tsv")
        for rows, line, why in (("j0\t3\td\nj0\tfast\td\n", 2, "a wall that is not a number"), ("j0\t3\td\nj1\t4\n", 2, "a row of two fields"),
                                ("j0\t3\td\nj1\t4\td\tx\n", 2, "a row of four fields"), ("j0\t3\td\n\t4\td\n", 2, "a row with no name"),
                                ("j0\t3\td\n\n", 2, "a blank line"), ("j0\t-1\td\n", 1, "a negative wall"),
                                ("j0\tnan\td\n", 1, "a wall of nan"), ("j0\tinf\td\n", 1, "an infinite wall")):
            write(tbad, rows)
            rc, err = sched_err(["--out", os.path.join(d, "o7"), "--times", tbad, jf])
            expect(rc == USAGE_RC and f"times-bad.tsv:{line}:" in err, f"{why} is refused naming its line, 64 (got {rc}: {err.strip()})")
        # a TERM ends the running jobs' process groups, and their children, before the scheduler exits 143
        jf = jobs_file([f"long\t-\t-\tsleep 300 & echo $! > {d}/child; wait"])
        p = subprocess.Popen([sys.executable, me, "run", "--out", os.path.join(d, "o8"), jf])
        for _ in range(100):
            if os.path.exists(os.path.join(d, "child")) and os.path.getsize(os.path.join(d, "child")):
                break
            time.sleep(0.05)
        p.send_signal(signal.SIGTERM)
        rc = p.wait(timeout=30)
        child = int(rd("child"))
        try:
            os.kill(child, 0)
            alive = True
        except (ProcessLookupError, PermissionError):  # a zombie answers EPERM on Darwin
            alive = False
        expect(rc == 143, f"a TERM ends the scheduler 143 (got {rc})")
        expect(not alive, "a TERM ends a job's own children")
        # a child that ignores TERM and outlives its group leader: the job's shell ends on the TERM at once, so the group's
        # survivor must be ended by the scheduler (KILL) and not left behind a 143. The child writes its pid once it ignores TERM.
        stubborn = f"sh -c 'trap \"\" TERM; echo $$ > \"$1\"; exec sleep 300' _ {shlex.quote(d)}/child2 & wait"
        jf = jobs_file([f"stubborn\t-\t-\t{stubborn}"])
        p = subprocess.Popen([sys.executable, me, "run", "--out", os.path.join(d, "o9"), jf])
        for _ in range(200):
            if os.path.exists(os.path.join(d, "child2")) and os.path.getsize(os.path.join(d, "child2")):
                break
            time.sleep(0.05)
        p.send_signal(signal.SIGTERM)
        rc = p.wait(timeout=60)
        child2 = int(rd("child2"))
        for _ in range(100):  # init reaps the killed child a moment after the scheduler is gone
            try:
                os.kill(child2, 0)
            except (ProcessLookupError, PermissionError):
                alive = False
                break
            alive = True
            time.sleep(0.05)
        if alive:
            os.kill(child2, signal.SIGKILL)  # the pid the job wrote: never leave the survivor behind a red run
        expect(rc == 143, f"a TERM with a stubborn child ends the scheduler 143 (got {rc})")
        expect(not alive, "a child that ignores TERM and outlives its group leader is ended by the scheduler")

        def gone(pid: int) -> bool:
            for _ in range(100):  # init reaps what the scheduler's end orphaned a moment later
                try:
                    os.kill(pid, 0)
                except (ProcessLookupError, PermissionError):  # a zombie answers EPERM on Darwin
                    return True
                time.sleep(0.05)
            return False

        def wait_written(path: str) -> None:
            for _ in range(200):
                if os.path.exists(path) and os.path.getsize(path):
                    return
                time.sleep(0.05)

        # a TERM while the job's pid line is being written: DIR/pids is a FIFO held full by its reader, so the write blocks with
        # the job already spawned. The scheduler acts on the TERM once the job is registered: the job's pid line is out and its
        # group is ended before the exit 143 (the 0.5 s is only the TERM's delivery margin)
        o10 = os.path.join(d, "o10")
        os.mkdir(o10)
        fifo = os.path.join(o10, "pids")
        os.mkfifo(fifo)
        sink = os.open(fifo, os.O_RDONLY | os.O_NONBLOCK)
        filler = os.open(fifo, os.O_WRONLY | os.O_NONBLOCK)
        for size in (4096, 1):
            try:
                while True:
                    os.write(filler, b"x" * size)
            except BlockingIOError:
                pass
        os.close(filler)
        jf = jobs_file([f"reg\t-\t-\tsleep 300 & echo $$ $! > {d}/child3; wait"])
        p = subprocess.Popen([sys.executable, me, "run", "--out", o10, jf])
        wait_written(os.path.join(d, "child3"))
        p.send_signal(signal.SIGTERM)
        time.sleep(0.5)
        got = b""
        stop_at = time.monotonic() + 30
        while p.poll() is None and time.monotonic() < stop_at:  # reading the pipe frees the scheduler's write
            try:
                got += os.read(sink, 65536)
            except BlockingIOError:
                pass
            time.sleep(0.02)
        if p.poll() is None:
            p.kill()
        rc = p.wait()
        while chunk := os.read(sink, 65536):
            got += chunk
        os.close(sink)
        leader, child3 = (int(x) for x in rd("child3").split())
        ended = gone(child3)
        if not ended:
            signal_group(leader, signal.SIGKILL)  # the group the job wrote: never leave a survivor behind a red run
        expect(rc == 143, f"a TERM while a job's pid line is written ends the scheduler 143 (got {rc})")
        expect(f"reg {leader}\n" in got.decode(errors="replace"), "a TERM while a job's pid line is written leaves the line out")
        expect(ended, "a TERM while a job's pid line is written ends that job's group")
        # a second TERM while the scheduler waits for a job's shell to end: the shell ignores TERM for 2 s and a child of its
        # group outlives it, so the KILL after the wait is what ends the child, and the second TERM must not cut that short
        jf = jobs_file([f'twice\t-\t-\ttrap "" TERM; sleep 300 & echo $$ $! > {d}/child4; sleep 2'])
        p = subprocess.Popen([sys.executable, me, "run", "--out", os.path.join(d, "o11"), jf])
        wait_written(os.path.join(d, "child4"))
        p.send_signal(signal.SIGTERM)
        time.sleep(0.5)
        p.send_signal(signal.SIGTERM)
        try:
            rc = p.wait(timeout=60)
        except subprocess.TimeoutExpired:
            p.kill()
            rc = p.wait()
        leader, child4 = (int(x) for x in rd("child4").split())
        ended = gone(child4)
        if not ended:
            signal_group(leader, signal.SIGKILL)
        expect(rc == 143, f"a second TERM during the cleanup leaves the scheduler's exit 143 (got {rc})")
        expect(ended, "a second TERM during the cleanup leaves the KILL of the job's group to happen")
        # a job that runs a scheduler of its own (check-recipes' pytools block runs this self-test, which does): the TERM ends the
        # job's shell at once, and the nested scheduler's jobs are in groups of their own, so a stubborn one of them is ended only
        # when the cleanup reads the job's descendants and signals their groups too
        write(os.path.join(d, "nested.tsv"), f"inner\t-\t-\tsh -c 'trap \"\" TERM; echo $$ > \"$1\"; exec sleep 300' _ {shlex.quote(d)}/child5\n")
        nest = f"{shlex.quote(sys.executable)} {shlex.quote(me)} run --out {shlex.quote(d)}/o12in {shlex.quote(d)}/nested.tsv & wait"
        jf = jobs_file([f"outer\t-\t-\t{nest}"])
        p = subprocess.Popen([sys.executable, me, "run", "--out", os.path.join(d, "o12"), jf])
        wait_written(os.path.join(d, "child5"))
        p.send_signal(signal.SIGTERM)
        try:
            rc = p.wait(timeout=60)
        except subprocess.TimeoutExpired:
            p.kill()
            rc = p.wait()
        child5 = int(rd("child5"))
        ended = gone(child5)
        if not ended:
            os.kill(child5, signal.SIGKILL)  # the pid the job wrote: never leave the survivor behind a red run
        expect(rc == 143, f"a TERM with a nested scheduler ends the scheduler 143 (got {rc})")
        expect(ended, "a TERM ends the jobs of a nested scheduler, a stubborn one included")
        # the groups the cleanup signals, from a table of (pid, ppid, pgid): the roots', every child's below them (its own group
        # included), and the members of a group found, whose parent the table no longer shows below the root
        table = [(10, 1, 10), (11, 10, 10), (12, 11, 12), (13, 12, 12), (14, 13, 14), (15, 1, 10), (16, 1, 20), (17, 15, 17), (20, 1, 20)]
        got = descendant_groups([10], table)
        expect(got == {10, 12, 14, 17}, f"the groups of a job and of its descendants, and no other (got {sorted(got)})")
        expect(descendant_groups([10, 20], table) == {10, 12, 14, 17, 20}, "two roots: the groups of both")
        # an error that is not a signal (the out directory gone as a job ends: it removes it once its own pid line is out, after
        # which the scheduler writes nothing that would make it again) ends the running jobs' groups before it propagates: the
        # scheduler exits 1 with the traceback, and no registered job outlives it
        out13 = os.path.join(d, "o13")
        jf = jobs_file([f"hold\t-\t-\tsh -c 'echo $$ > \"$1\"; exec sleep 300' _ {shlex.quote(d)}/child6",
                        f"vanish\t-\t-\tuntil [ -s {shlex.quote(d)}/child6 ] && grep -q '^vanish ' {shlex.quote(out13)}/pids; do sleep 0.05; done; "
                        f"rm -rf {shlex.quote(out13)}"])
        p = subprocess.Popen([sys.executable, me, "run", "--out", out13, "--max", "4", jf], stderr=subprocess.DEVNULL)
        try:
            rc = p.wait(timeout=60)
        except subprocess.TimeoutExpired:
            p.kill()
            rc = p.wait()
        ended = gone(int(rd("child6")))
        if not ended:
            os.kill(int(rd("child6")), signal.SIGKILL)
        expect(rc == 1, f"an error that is not a signal ends the scheduler 1 (got {rc})")
        expect(ended, "an error that is not a signal ends the running jobs' groups")
        # a group holding only zombies (its leader ended and not yet reaped) is signalled without an error
        z = subprocess.Popen(["sh", "-c", "exit 0"], start_new_session=True)
        for _ in range(200):
            if subprocess.run(["ps", "-o", "stat=", "-p", str(z.pid)], capture_output=True, text=True).stdout.strip().startswith("Z"):
                break
            time.sleep(0.05)
        try:
            signal_group(z.pid, signal.SIGKILL)
            quiet = True
        except OSError:
            quiet = False
        z.wait()
        expect(quiet, "signalling a process group that holds only zombies is not an error")
    print(f"static-sched: self-test {'ok' if not fails else f'{len(fails)} failed'}")
    return 0 if not fails else 1


def main(argv: list[str]) -> int:
    if argv == ["--self-test"]:
        return self_test()
    ap = argparse.ArgumentParser(prog="static-sched.py", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("jobs", metavar="JOBS")
    r.add_argument("--out", required=True)
    r.add_argument("--max", type=int, default=0, help="slots at once (default: the cpu count)")
    r.add_argument("--times")
    r.add_argument("--keys")
    r.add_argument("--ledger")
    r.add_argument("--tree", default=os.path.basename(os.getcwd()))
    r.add_argument("--no-skip", action="store_true", help="with --ledger: run a job whose key is green too (and append it)")
    try:
        args = ap.parse_args(argv)
    except SystemExit as e:
        return USAGE_RC if e.code else 0
    return run(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
