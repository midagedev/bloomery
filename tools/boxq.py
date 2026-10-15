#!/usr/bin/env python3
"""boxq.py — the box queue's client: it posts a job to signalbox, follows it, lists jobs and cancels them.

  boxq.py submit --card C --class K [--expected-s N] --name TAG [--owner O] [--env K=V …] -- CMD…
                                                   post a job to signalbox; its id on stdout
  boxq.py attach ID [--rc-dir DIR] [--leave]       follow its out and err, mirror its rc, exit with it; --leave: the
                                                   job is another client's — a signal leaves it as it is, a cancelled
                                                   or lost job ends 81 whatever rc it kept, and nothing is mirrored
  boxq.py list [--match TEXT …]                    the queued and running jobs whose command holds every TEXT, one
                                                   id a line (none: no output)
  boxq.py run …                                    submit, then attach (the id goes to stderr: stdout is the job's out)
  boxq.py cancel ID                                post the cancel, print the job's status
  boxq.py --self-test [CASE …]                     --list-cases names the cases

Client. Every request is one `ssh <host> curl …` to the daemon on the box's loopback; a POST's JSON body goes on ssh's
stdin (no job text ever lands on a command line), a GET has no body and no stdin. CMD… is the job's `bash -c` text:
one word is the text as given, several words are shell-quoted and joined. A job's out and err follow as two
`ssh <host> stdbuf -o0 tail -c +N -F <path>` children (N is 1 until a tail is resumed); each ends when its ssh's
stdin closes, so a killed client leaves none on the box. The client counts the bytes it has passed on, and a tail
that starts again reads from the next byte.
The daemon is polled every BOXQ_POLL_S; a job seen terminal is given one more interval (at most 2 s) for the tails to
flush, then the tails are stopped (a job that was never seen running has its out and err read whole instead, no flush
to guess at). Every byte the tails did not pass on (a full pipe here, a tail short of the file's end) is read
again from the job's files before `<rc-dir>/<id>.rc` (default
$BOXQ_RC_DIR, else ~/.cache/bloomery/boxq) is written atomically with the exit code. A replay ssh that does not exit 0
is exit 84 with the resume line and no rc file. A queued job's wait is counted from the job's `submitted` (ISO; no zone means this machine's, which the
box shares), else from the client's first sight of it.
The box's rc is the authority: for done and failed the exit code is the job's rc, so
80, 83 and 85 from the job's command prefix pass through unchanged; a job cancelled by someone else exits with its rc,
else 81, with `boxq: <id> cancelled elsewhere` on stderr; 143 is only our own trapped TERM. A mirror that cannot be
written is a stderr line; the exit code stays the job's.
A job with no --expected-s is posted without one (the daemon reads it as an unknown wall) and its line says `predicted unknown`.
INT or TERM during run or attach cancels only a job that has not started: for a job last seen queued (or not seen yet) it
posts the cancel and keeps polling until the job is terminal, then exits 130 or 143. The cancel request, the status
requests after it and the sleeps between them all sit inside BOXQ_CANCEL_WAIT_S; a job not terminal by then, or a second
signal, exits 81 naming the job. A job last seen running is left running on the box: the
client stops following it, names `boxq.py attach <id>` and `boxq.py cancel <id>`, mirrors nothing and exits 130 or 143,
so an interrupted session does not throw away a job's card time. A job whose first answer after the signal is terminal
is not cancelled either: the exit is 130 or 143 and nothing is mirrored. A signal while unread bytes are replayed is
that same exit, and no rc file. A job that starts between the last poll and the
cancel is cancelled (the poll interval is the window).
After POLL_FAILS (3) polls in a row that cannot reach the daemon the client backs off exponentially, each pause at
most RECONNECT_STEP_S (60 s), for at most BOXQ_RECONNECT_S seconds (900), resumes each tail at the next byte, and only past that bound exits 84 with
`boxq: <id> left as it is on the box; resume with: tools/boxq.py attach <id>` and no rc file. A signal during a pause
is decided as after any poll.

  BOXQ_HOST   ssh host (default $BLOOMERY_BOX, else ws)      BOXQ_URL            the daemon (http://127.0.0.1:8030)
  BOXQ_SSH    the ssh command (default ssh)                  BOXQ_POLL_S         poll interval, seconds (5)
  BOXQ_NOTE_S seconds between queue-wait lines (60)          BOXQ_CANCEL_WAIT_S  wait after a cancel (150)
  BOXQ_RECONNECT_S seconds of backoff after those misses (900)

Exit: the job's rc (run, attach); 0 ok; 64 usage or no such job; 81 lost, or still running after the cancel; 84 the daemon
is not alive past the reconnect bound, answers outside the wire, or a replay ssh does not exit 0; 130/143 a trapped INT/TERM; a daemon refusal exits with the rc it names. 80, 83
and 85 are tools/box.sh's, and only pass through here. tools/boxq-check.py holds the guard that keeps the five codes apart
from every other tool, and the idle-fill list's reader.
"""

import argparse
import contextlib
import datetime
import fcntl
import http.server
import io
import json
import os
import re
import select
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time

from bloomery.selftest import load_cap

USAGE = 64
LANES = ("3090", "a6000", "both", "any", "none")
CLASSES = ("lead", "round", "sitting", "fill")
STATUSES = ("queued", "running", "done", "failed", "cancelled", "lost")
TERMINAL = ("done", "failed", "cancelled", "lost")
LOST, DOWN = 81, 84
INT_RC, TERM_RC = 130, 143
POLL_FAILS = 3  # polls in a row that cannot reach the daemon before the attach starts backing off
RECONNECT_S = 900.0  # BOXQ_RECONNECT_S default: the backoff budget, then 84 (a daemon blip has lasted ~12 min)
RECONNECT_STEP_S = 60.0  # the longest one backoff pause: a daemon back is seen within this
RESUME_LINE = "boxq: {jid} left as it is on the box; resume with: tools/boxq.py attach {jid}"
LINK_DOWN = ("daemon-down",)  # the Down names a blip can clear; any other (an ssh that cannot start) is not waited out
FLUSH_MAX_S = 2.0  # the longest the tails get, after a terminal status, to read what the job wrote last
NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")
ENV_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class Refusal(Exception):
    def __init__(self, code, name, msg):
        super().__init__(msg)
        self.code, self.name, self.msg = code, name, msg


class Parser(argparse.ArgumentParser):
    def error(self, message):
        raise Refusal(USAGE, "usage", message)


def here():
    return os.path.dirname(os.path.abspath(__file__))


# ---- the queue client ---------------------------------------------------------------------------------------------

class Down(Refusal):
    """The daemon cannot be reached, or answers outside the wire: client rc 84."""

    def __init__(self, name, msg, line=None):
        super().__init__(DOWN, name, msg)
        self.line = line


def say(msg):
    print(msg, file=sys.stderr, flush=True)


def env_secs(env, name, default, lo):
    raw = env.get(name)
    if raw in (None, ""):
        return default
    try:
        v = float(raw)
    except ValueError:
        v = float("nan")
    if not v >= lo:
        raise Refusal(USAGE, "usage", f"{name} is seconds >= {lo:g}, got {raw!r}")
    return v


def refusal(st, obj, path):
    """The exception for a reply that is not the success the caller wanted."""
    text = obj.get("error") if isinstance(obj.get("error"), str) else json.dumps(obj)[:200]
    rc = obj.get("rc")
    if st == 404 and path != "/jobs":
        return Refusal(USAGE, "no-such-job", f"{path}: {text}")
    if 400 <= st < 500 and isinstance(rc, int) and not isinstance(rc, bool) and 1 <= rc <= 255:
        return Refusal(rc, "refused", text)
    return Down("bad-reply", f"{path}: status {st}: {text}")


class Queue:
    """The daemon's loopback HTTP, one `ssh <host> curl` per request, and the tails' ssh children."""

    def __init__(self, env):
        self.host = env.get("BOXQ_HOST") or env.get("BLOOMERY_BOX") or "ws"
        self.ssh = shlex.split(env.get("BOXQ_SSH") or "ssh")
        self.url = (env.get("BOXQ_URL") or "http://127.0.0.1:8030").rstrip("/")
        self.poll_s = env_secs(env, "BOXQ_POLL_S", 5.0, 0.01)
        self.note_s = env_secs(env, "BOXQ_NOTE_S", 60.0, 0.0)
        self.cancel_wait_s = env_secs(env, "BOXQ_CANCEL_WAIT_S", 150.0, 0.0)
        self.reconnect_s = env_secs(env, "BOXQ_RECONNECT_S", RECONNECT_S, 0.0)

    def spawn(self, remote, **kw):
        # Own session: a Ctrl-C reaches this client only, which stops its children itself.
        try:
            return subprocess.Popen(self.ssh + [self.host, remote], start_new_session=True, **kw)
        except OSError as e:
            raise Down("no-ssh", f"{self.ssh[0]}: {e}")

    def request(self, method, path, body=None, timeout=30.0):
        """(status, JSON object) of one request."""
        words = ["curl", "-sS", "--max-time", "20", "-X", method]
        if body is not None:  # only a POST has a body, on stdin
            words += ["-H", "Content-Type: application/json", "-H", "Expect:", "--data-binary", "@-"]
        remote = " ".join(shlex.quote(w) for w in words + ["-w", r"\n%{http_code}", self.url + path])
        try:
            p = self.spawn(remote, stdin=subprocess.DEVNULL if body is None else subprocess.PIPE,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                out, err = p.communicate(None if body is None else body.encode(), timeout=timeout)
            except subprocess.TimeoutExpired:
                p.kill()
                p.communicate()
                raise Down("daemon-down", f"{method} {path}: no reply in {timeout:g} s")
        except OSError as e:
            raise Down("daemon-down", f"{method} {path}: {e}")
        if p.returncode != 0:
            why = (err.decode(errors="replace").strip().splitlines() or ["no message"])[0]
            raise Down("daemon-down", f"{method} {path}: ssh {self.host} rc {p.returncode}: {why}")
        raw, _, code = out.rpartition(b"\n")
        try:
            st, obj = int(code), json.loads(raw)
        except ValueError:
            raise Down("bad-reply", f"{method} {path}: not JSON and a status: {out[:120]!r}")
        if not isinstance(obj, dict) and not (method == "GET" and path == "/jobs" and isinstance(obj, list)):
            raise Down("bad-reply", f"{method} {path}: the reply is not a JSON object: {raw[:120]!r}")
        return st, obj

    def call(self, method, path, body=None, ok=(200,), timeout=30.0):
        st, obj = self.request(method, path, body, timeout=timeout)
        if st not in ok:
            raise refusal(st, obj, path)
        return obj

    def health(self):
        st, obj = self.request("GET", "/health")
        if st != 200 or obj.get("ok") is not True:
            raise Down("daemon-down", f"GET /health: status {st}, {json.dumps(obj)[:120]}")

    def job(self, jid, timeout=30.0):
        job = self.call("GET", f"/jobs/{jid}", timeout=timeout)
        if job.get("status") not in STATUSES:
            raise Down("bad-reply", f"job {jid}: status {job.get('status')!r} is not one of {', '.join(STATUSES)}")
        return job

    def cancel(self, jid, timeout=30.0):
        return self.call("POST", f"/jobs/{jid}/cancel", "{}", ok=(200, 202), timeout=timeout)

    def jobs(self):
        """The daemon's job list: one summary (id, status, name, …) per job, without the command."""
        st, obj = self.request("GET", "/jobs")
        if st != 200 or not isinstance(obj, list):
            raise Down("bad-reply", f"GET /jobs: status {st}, not a list: {json.dumps(obj)[:120]}")
        return obj


class Trap:
    """INT and TERM as a list and a wake-up byte: the wait between polls selects on the pipe, so a signal ends it."""

    def __init__(self):
        self.sigs = []
        self.r, self.w = os.pipe()
        os.set_blocking(self.r, False)
        os.set_blocking(self.w, False)
        self.old = {}

    def __enter__(self):
        for sig in (signal.SIGINT, signal.SIGTERM):
            self.old[sig] = signal.signal(sig, self.on_signal)
        return self

    def __exit__(self, *exc):
        for sig, old in self.old.items():
            signal.signal(sig, old)
        os.close(self.r)
        os.close(self.w)

    def on_signal(self, signum, _frame):
        self.sigs.append(signum)
        with contextlib.suppress(OSError):
            os.write(self.w, b"x")

    def wait(self, secs):
        select.select([self.r], [], [], secs)
        with contextlib.suppress(OSError):
            os.read(self.r, 64)

    def rc(self):
        return INT_RC if self.sigs and self.sigs[0] == signal.SIGINT else TERM_RC


class Tail:
    """One remote tail. `n` is the bytes of its file already written to `dst`."""

    def __init__(self, proc, src, dst):
        self.proc, self.src, self.dst, self.n = proc, src, dst, 0
        self.pending = b""  # read from src, not yet taken by dst; not in `n`


def drain(kids):
    """Bytes waiting on the tails' pipes, written on and counted once written. A write never blocks: what the reader has
    not taken stays in the tail's `pending` until its fd is writable. A closed reader's bytes are dropped and counted,
    the job goes on."""
    for t in kids:
        if not isinstance(t, Tail) or t.src is None and not t.pending:
            continue
        for _ in range(16):
            chunk = t.pending
            if not chunk:
                if t.src is None:
                    break
                try:
                    chunk = os.read(t.src, 1 << 16)
                except BlockingIOError:
                    break
                if not chunk:
                    os.close(t.src)
                    t.src = None
                    break
            blocking = os.get_blocking(t.dst)
            try:
                os.set_blocking(t.dst, False)
                try:
                    n = os.write(t.dst, chunk)
                except BlockingIOError:
                    n = 0
                except BrokenPipeError:
                    n = len(chunk)
            finally:
                os.set_blocking(t.dst, blocking)
            t.n += n
            t.pending = chunk[n:]
            if t.pending:
                break


def pause(trap, kids, secs):
    """The wait between polls, draining the tails. A trap with no pipe (the flush test) just records the wait. The wait
    also ends for a signal while a reader is slow: a tail with unwritten bytes is waited on at its dst, not its src."""
    if secs < 0:
        secs = 0.0
    r = getattr(trap, "r", None)
    tails = [t for t in kids if isinstance(t, Tail)]
    if r is None or not any(t.src is not None or t.pending for t in tails):
        trap.wait(secs)
        drain(kids)
        return
    end = time.monotonic() + secs
    while True:
        left = end - time.monotonic()
        if left <= 0:
            drain(kids)
            return
        ready, _, _ = select.select(
            [r, *(t.src for t in tails if t.src is not None and not t.pending)],
            [t.dst for t in tails if t.pending], [], left)
        if r in ready:
            with contextlib.suppress(OSError):
                os.read(r, 64)
            drain(kids)
            return
        drain(kids)
        if not any(t.src is not None or t.pending for t in tails):
            trap.wait(max(0.0, end - time.monotonic()))
            return


def tail_cmd(path, skip=0):
    # The remote tail ends when ssh's stdin does: killing the ssh client alone leaves a `tail -F` on the box for good.
    # -c +N is the next byte (1-based). stdbuf keeps a pipe from holding a block of the job's output.
    start = skip + 1
    return f"stdbuf -o0 tail -c +{start} -F {shlex.quote(path)} & t=$!; read x; kill $t"


def box_path(job, key):
    path = job.get(key)
    return path if isinstance(path, str) and path.startswith("/") else None


def spawn_all(q, cmds, fds):
    """One ssh child per remote command, its stdout on the matching fd and its stderr on ours."""
    kids = []
    sys.stdout.flush()
    sys.stderr.flush()
    try:
        for cmd, fd in zip(cmds, fds):
            kids.append(q.spawn(cmd, stdin=subprocess.PIPE, stdout=fd, stderr=2))
    except Down:
        unfollow(kids)
        raise
    return kids


def follow(q, job, fds=(1, 2), offsets=(0, 0)):
    """The job's out and err as two tail children writing to fds, each starting after `offsets` bytes already passed on."""
    paths = [box_path(job, key) for key in ("out", "err")]
    if None in paths:
        raise Down("bad-reply", f"job {job.get('id')}: out and err are {paths!r}, a running job has absolute box paths")
    sys.stdout.flush()
    sys.stderr.flush()
    tails = []
    try:
        for path, skip, fd in zip(paths, offsets, fds):
            r, w = os.pipe()
            os.set_blocking(r, False)
            try:
                proc = q.spawn(tail_cmd(path, skip), stdin=subprocess.PIPE, stdout=w, stderr=2)
            except BaseException:
                os.close(w)
                os.close(r)
                raise
            os.close(w)
            tails.append(Tail(proc, r, fd))
    except Down:
        unfollow(tails)
        raise
    return tails


def dump(q, job, trap, fds=(1, 2), offsets=(0, 0)):
    """A job whose tails are not running: each file is read from the next unread byte and the children end by
    themselves; a file the job never wrote, or a path it never got, is skipped. Each file is read with `tail -c` from
    the next unread byte."""
    cmds, out_fds = [], []
    for key, fd, skip in zip(("out", "err"), fds, offsets):
        path = box_path(job, key)
        if not path:
            continue
        qpath = shlex.quote(path)
        cmds.append(f"[ ! -e {qpath} ] || exec tail -c +{skip + 1} {qpath}")
        out_fds.append(fd)
    # The children write to pipes this process drains, so a closed local reader drops bytes (drain's rule) and is not
    # an ssh failure. A non-zero ssh is a missed read: 84 and no rc file, the same as a daemon the client cannot reach.
    kids = spawn_all(q, cmds, [subprocess.PIPE] * len(cmds))
    tails = []
    try:
        for proc, fd in zip(kids, out_fds):
            src = os.dup(proc.stdout.fileno())
            proc.stdout.close()
            os.set_blocking(src, False)
            tails.append(Tail(proc, src, fd))
        while not trap.sigs and any(t.src is not None or t.pending or t.proc.poll() is None for t in tails):
            pause(trap, tails, 0.1)
        if not trap.sigs:
            codes = [p.poll() for p in kids]
            if any(code != 0 for code in codes):
                say(f"boxq: {job['id']}: output read failed: ssh rc {codes}")
                raise Down("daemon-down", f"job {job['id']}: output read failed: ssh rc {codes}",
                           RESUME_LINE.format(jid=job["id"]))
    finally:
        unfollow(tails)
        unfollow(kids)


def _proc(k):
    return k.proc if isinstance(k, Tail) else k


def unfollow(kids, grace=3.0):
    """Stop the tails: close their stdin (the remote side ends), drain what they already sent, then terminate and
    kill a ssh that does not follow. Returns the bytes each tail delivered and empties kids."""
    procs = [_proc(k) for k in kids]
    for proc in procs:
        with contextlib.suppress(OSError):
            if proc.stdin:
                proc.stdin.close()
    end = time.monotonic() + grace
    while time.monotonic() < end and any(p.poll() is None for p in procs):
        drain(kids)
        srcs = [k.src for k in kids if isinstance(k, Tail) and k.src is not None]
        left = end - time.monotonic()
        if left <= 0:
            break
        if srcs:
            select.select(srcs, [], [], min(0.05, left))
        else:
            time.sleep(min(0.05, left))
    drain(kids)
    for proc in procs:
        if proc.poll() is None:
            proc.terminate()
    kill_end = time.monotonic() + 2
    for proc in procs:
        try:
            proc.wait(max(0.05, kill_end - time.monotonic()))
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    drain(kids)
    counts = []
    for k in kids:
        if isinstance(k, Tail):
            counts.append(k.n)
            if k.src is not None:
                os.close(k.src)
                k.src = None
        else:
            counts.append(0)
    kids.clear()
    return counts


def fold_offsets(offsets, kids):
    """Bytes passed on so far, after the running tails have been stopped and drained."""
    if not kids:
        return offsets
    extra = unfollow(kids)
    return [a + b for a, b in zip(offsets, extra)]


def eta_text(eta):
    if isinstance(eta, (int, float)) and not isinstance(eta, bool) and eta >= 0:
        return f"eta {round(eta)} s"
    return "eta unknown"


def submitted_age(job):
    """Seconds since the job's `submitted` (ISO; without a zone it is read in this machine's, which the box shares),
    or None when it is missing, not ISO, or in the future."""
    raw = job.get("submitted")
    if not isinstance(raw, str):
        return None
    try:
        t = datetime.datetime.fromisoformat(raw[:-1] + "+00:00" if raw.endswith("Z") else raw)
    except ValueError:
        return None
    age = (datetime.datetime.now(t.tzinfo) - t).total_seconds()
    return age if age >= 0 else None


def outcome(jid, job):
    """The exit code of a terminal job."""
    rc = job.get("rc")
    if job["status"] == "lost":
        return LOST
    if isinstance(rc, int) and not isinstance(rc, bool) and 0 <= rc <= 255:
        return rc
    if job["status"] == "cancelled":
        return LOST
    raise Down("bad-reply", f"job {jid} is {job['status']} with rc {rc!r}")


def mirror(rc_dir, jid, rc):
    """<rc-dir>/<id>.rc, written whole or not at all."""
    tmp = os.path.join(rc_dir, f".{jid}.rc.{os.getpid()}")
    try:
        os.makedirs(rc_dir, exist_ok=True)
        with open(tmp, "w") as f:
            f.write(f"{rc}\n")
        os.replace(tmp, os.path.join(rc_dir, f"{jid}.rc"))
    except OSError as e:
        say(f"boxq: {jid}: the rc mirror {rc_dir}/{jid}.rc was not written: {e}")
        with contextlib.suppress(OSError):
            os.unlink(tmp)


def attach_job(q, jid, rc_dir, trap, leave=False):
    """Follow the job to a terminal status, mirror and return its exit code. With leave the job is another client's: a
    signal posts no cancel and exits with the signal's code, a cancelled or lost job is no verdict of its command (81,
    whatever rc the daemon kept: 143 after its owner's stop), and no rc is mirrored — the owner's client mirrors its own."""
    kids, fails, last = [], 0, None
    queued_at, next_note, reason, cancel_at = None, 0.0, None, None
    offsets, reconnect, cancel_from = [0, 0], None, None
    try:
        while True:
            if cancel_from is not None and (len(trap.sigs) > 1 or time.monotonic() - cancel_from >= q.cancel_wait_s):
                st = last["status"] if last else "unknown"
                raise Refusal(LOST, "cancel-pending", f"job {jid} is still {st} after the cancel; check it on the box")
            job, wait, gone = None, q.poll_s, None
            try:
                timeout = 30.0 if reconnect is None else min(30.0, max(0.001, reconnect["until"] - time.monotonic()))
                if cancel_from is not None:
                    timeout = min(timeout, max(0.001, cancel_from + q.cancel_wait_s - time.monotonic()))
                job = q.job(jid, timeout=timeout)
                fails, reconnect = 0, None
            except Refusal as e:
                # Another client's job that the daemon no longer knows (a restart, a pruned history) is a miss like an
                # unreachable daemon: after POLL_FAILS the job is no verdict (84), never a usage error of this call.
                if not (leave or isinstance(e, Down)):
                    raise
                fails += 1
                if not (isinstance(e, Down) and e.name in LINK_DOWN):
                    if fails >= POLL_FAILS:
                        gone = Down(e.name, f"job {jid} is left as it is: {e.msg}")
                    else:
                        say(f"boxq: {jid}: poll {fails}/{POLL_FAILS} failed: {e.msg}")
                elif reconnect is None and fails < POLL_FAILS:
                    say(f"boxq: {jid}: poll {fails}/{POLL_FAILS} failed: {e.msg}")
                else:
                    if reconnect is None:
                        reconnect = {"until": time.monotonic() + q.reconnect_s, "shift": 0}
                        offsets = fold_offsets(offsets, kids)
                    remain = reconnect["until"] - time.monotonic()
                    if remain <= 0:
                        gone = Down("daemon-down", e.msg, RESUME_LINE.format(jid=jid))
                    else:
                        wait = min(q.poll_s * 2 ** min(reconnect["shift"], 20), RECONNECT_STEP_S, remain)
                        reconnect["shift"] += 1
                        say(f"boxq: {jid}: reconnecting, next poll in {wait:g} s")
            # a pending signal is decided here, before this poll's answer is taken: another client's job (leave) and a job
            # seen past the queue are not ours to cancel, and a terminal answer after the signal never stands in for the
            # signal's exit or writes a mirror. A miss does not pre-empt it: a signal during a reconnect pause is
            # decided by this block on the next poll, whether that poll answers or not.
            if trap.sigs and cancel_at is None:
                seen = [x["status"] for x in (last, job) if x is not None]
                if leave:
                    say(f"boxq: {jid}: signal, leaving the job as it is")
                    return trap.rc()
                if "running" in seen:
                    say(f"boxq: {jid}: signal while running: the job is left running on the box; "
                        f"follow it with `tools/boxq.py attach {jid}`, stop it with `tools/boxq.py cancel {jid}`")
                    return trap.rc()
                if seen and seen[-1] in TERMINAL:
                    say(f"boxq: {jid}: signal; the job had already ended {seen[-1]} on the box (no rc mirrored)")
                    return trap.rc()
                if cancel_from is None:
                    say(f"boxq: {jid}: signal, cancelling")
                    cancel_from = time.monotonic()
                try:
                    q.cancel(jid, timeout=min(30.0, max(0.001, cancel_from + q.cancel_wait_s - time.monotonic())))
                except Down:
                    # the daemon is not there to take it: ask again at the next poll, within the cancel wait (the loop's
                    # first check ends it past the wait or on a second signal)
                    pause(trap, kids, min(q.poll_s, max(0.0, cancel_from + q.cancel_wait_s - time.monotonic())))
                    continue
                cancel_at = cancel_from
                reconnect, fails, gone = None, 0, None
            if gone:
                raise gone
            if job is not None:
                if any(_proc(k).poll() is not None for k in kids):
                    offsets = fold_offsets(offsets, kids)
                last, now = job, time.monotonic()
                if job["status"] == "queued":
                    reason = job.get("wait_reason")
                    queued_at = now if queued_at is None else queued_at
                    if now >= next_note:
                        say(f"boxq: {jid} waiting: {reason or 'none'}, {eta_text(job.get('eta_s'))}")
                        next_note = now + q.note_s
                else:
                    if queued_at is not None:
                        age = submitted_age(job)
                        age = now - queued_at if age is None else age
                        say(f"boxq: {jid} queued {round(age)} s (wait_reason {reason or 'none'})")
                    if not kids and job["status"] not in TERMINAL:
                        kids = follow(q, job, offsets=tuple(offsets))
                    queued_at = None
                if job["status"] in TERMINAL:
                    break
            if cancel_from is not None:
                wait = min(wait, max(0.0, cancel_from + q.cancel_wait_s - time.monotonic()))
            pause(trap, kids, wait)
        if kids:
            pause(trap, kids, min(q.poll_s, FLUSH_MAX_S))  # the tails read what the job wrote last
        # whatever the tails did not pass on (a full local pipe, a tail that had not reached the end) is read from the
        # job's files, from the next byte, before the rc is mirrored
        offsets = fold_offsets(offsets, kids)
        dump(q, last, trap, offsets=tuple(offsets))
        if trap.sigs and cancel_at is None:
            say(f"boxq: {jid}: signal; the job had already ended {last['status']} on the box (no rc mirrored)")
            return trap.rc()
        if cancel_at is None and last["status"] == "cancelled":
            say(f"boxq: {jid} cancelled elsewhere")
        if leave:
            return LOST if last["status"] in ("cancelled", "lost") else outcome(jid, last)
        rc = trap.rc() if cancel_at is not None else outcome(jid, last)
        mirror(rc_dir, jid, rc)
        return rc
    finally:
        unfollow(kids)


def default_owner():
    return os.path.basename(os.path.dirname(here()))


def job_body(a):
    words = a.words
    if not words or not "".join(words).strip():
        raise Refusal(USAGE, "usage", "the job's command goes after --")
    owner = a.owner or default_owner()
    for flag, v in (("--name", a.name), ("--owner", owner)):
        if not NAME_RE.match(v):
            raise Refusal(USAGE, "usage", f"{flag} is [A-Za-z0-9._-]+, got {v!r}")
    if a.expected_s is not None and a.expected_s < 1:
        raise Refusal(USAGE, "usage", f"--expected-s is whole seconds >= 1, got {a.expected_s}")
    env = {}
    for kv in a.env:
        k, eq, v = kv.partition("=")
        if not eq or not ENV_RE.match(k):
            raise Refusal(USAGE, "usage", f"--env is NAME=VALUE, got {kv!r}")
        env[k] = v
    body = {"kind": "command", "argv": words[0] if len(words) == 1 else shlex.join(words), "env": env, "owner": owner,
            "name": a.name, "card": a.card, "job_class": a.job_class, "wait": 0}
    if a.expected_s is not None:
        body["expected_s"] = a.expected_s
    return body


def submit_job(q, body, trap=None):
    """Post the job; its id."""
    predicted = f"{body['expected_s']} s" if "expected_s" in body else "unknown"
    say(f"boxq: {body['name']} class={body['job_class']} card={body['card']} predicted {predicted}")
    q.health()
    if trap and trap.sigs:
        raise Refusal(trap.rc(), "interrupted", "before the job was posted")
    jid = q.call("POST", "/jobs", json.dumps(body), ok=(201,)).get("id")
    if not (isinstance(jid, str) and NAME_RE.match(jid)):
        raise Down("bad-reply", f"POST /jobs: the job has no usable id: {jid!r}")
    return jid


def job_id(a):
    if not NAME_RE.match(a.id):
        raise Refusal(USAGE, "usage", f"a job id is [A-Za-z0-9._-]+, got {a.id!r}")
    return a.id


def rc_dir_of(a):
    return a.rc_dir or os.environ.get("BOXQ_RC_DIR") or os.path.expanduser("~/.cache/bloomery/boxq")


def cmd_submit(a):
    body = job_body(a)
    print(submit_job(Queue(os.environ), body), flush=True)


def cmd_attach(a):
    q, jid = Queue(os.environ), job_id(a)
    with Trap() as trap:
        return attach_job(q, jid, rc_dir_of(a), trap, a.leave)


def cmd_list(a):
    """The ids of the queued and running jobs whose command holds every --match text (a job's command needs a request of
    its own, so with no --match none is read). A row off the wire (no id, an id not a name, a status the daemon does not
    have) is named on stderr and skipped. A job the daemon no longer knows by the time its command is read (404) is
    skipped; any other refusal, and a live job whose command is not a string, end the list (84 or the refusal's code):
    a reply the client cannot read is never an empty list."""
    q = Queue(os.environ)
    for summary in q.jobs():
        jid = summary.get("id") if isinstance(summary, dict) else None
        if not (isinstance(jid, str) and NAME_RE.fullmatch(jid)) or summary.get("status") not in STATUSES:
            say(f"boxq: list: a row off the wire is skipped: {json.dumps(summary)[:120]}")
            continue
        if summary["status"] not in ("queued", "running"):
            continue
        if a.match:
            try:
                job = q.job(jid)
            except Refusal as e:
                if isinstance(e, Down) or e.name != "no-such-job":
                    raise
                continue
            if job["status"] in TERMINAL:
                continue
            argv = job.get("argv")
            if not isinstance(argv, str):
                raise Down("bad-reply", f"job {jid}: argv is not a string: {argv!r}"[:200])
            if not all(m in argv for m in a.match):
                continue
        print(jid, flush=True)


def cmd_run(a):
    body, q = job_body(a), Queue(os.environ)
    with Trap() as trap:
        jid = submit_job(q, body, trap)
        say(f"boxq: {body['name']} is job {jid}")
        return attach_job(q, jid, rc_dir_of(a), trap)


def cmd_cancel(a):
    print(Queue(os.environ).cancel(job_id(a)).get("status"), flush=True)


def parser():
    p = Parser(prog="boxq.py", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd")
    for name in ("submit", "run"):
        s = sub.add_parser(name)
        s.add_argument("--card", required=True, choices=LANES)
        s.add_argument("--class", dest="job_class", required=True, choices=CLASSES)
        s.add_argument("--expected-s", dest="expected_s", type=int)
        s.add_argument("--name", required=True)
        s.add_argument("--owner")
        s.add_argument("--env", action="append", default=[])
        if name == "run":
            s.add_argument("--rc-dir")
    s = sub.add_parser("attach")
    s.add_argument("id")
    s.add_argument("--rc-dir")
    s.add_argument("--leave", action="store_true")
    sub.add_parser("list").add_argument("--match", action="append", default=[])
    sub.add_parser("cancel").add_argument("id")
    return p


HANDLERS = {"submit": cmd_submit, "attach": cmd_attach, "run": cmd_run, "cancel": cmd_cancel, "list": cmd_list}


def main(argv):
    try:
        if argv and argv[0] == "--self-test":
            return self_test(argv[1:])
        if argv and argv[0] == "--list-cases":
            print("\n".join(c.__name__[5:].replace("_", "-") for c in CASES))
            return 0
        head, words = argv, None
        if argv[:1] in (["submit"], ["run"]) and "--" in argv:
            head, words = argv[:argv.index("--")], argv[argv.index("--") + 1:]
        a = parser().parse_args(head)
        if not a.cmd:
            raise Refusal(USAGE, "usage", f"a subcommand is needed: {', '.join(HANDLERS)}")
        a.words = words
        return HANDLERS[a.cmd](a) or 0
    except Refusal as r:
        print(r.line if getattr(r, "line", None) else f"boxq: {r.name}: {r.msg}", file=sys.stderr)
        return r.code
    except KeyboardInterrupt:
        print("boxq: interrupted", file=sys.stderr)
        return INT_RC


# ---- self-test ----------------------------------------------------------------------------------------------------

class Fake:
    """The daemon's side of the wire in a thread: it records each request and serves job j1 from a script of poll
    states, the last state repeating. `post` and `health` are (status, object) replies that override the defaults;
    `cancel_to` is the status a cancelled job shows (None: the cancel is ignored); `slow` seconds pass before health
    answers; `gate` (a callable) holds the job running until it is true; `late` is a pair of byte strings appended to
    the out and err files 0.1 s after the first terminal poll was answered, the way a job's last output trails its
    status."""

    def __init__(self, files, states, health=(200, {"ok": True}), post=None, cancel_to="cancelled", slow=0.0, gate=None,
                 late=None, listing=None, job_reply=None):
        self.requests, self.polls, self.cancelled, self.slow = [], 0, False, slow
        self.files, self.states, self.health, self.post, self.cancel_to = files, states, health, post, cancel_to
        self.gate, self.late, self.listing, self.job_reply = gate, late, listing, job_reply
        self.down_until = 0.0  # until then every connection is closed unanswered
        fake = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def serve(self):
                if time.monotonic() < fake.down_until:
                    self.close_connection = True
                    with contextlib.suppress(OSError):
                        self.connection.shutdown(socket.SHUT_RDWR)
                    return
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                st, obj = fake.handle(self.command, self.path, body, self.headers.get("Content-Type"))
                data = json.dumps(obj).encode()
                self.send_response(st)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            do_GET = do_POST = serve

        self.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"
        threading.Thread(target=self.srv.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True).start()

    def job(self, **kw):
        job = {"id": "j1", "status": "queued", "rc": None, "wait_reason": None, "eta_s": None, "out": self.files[0],
               "err": self.files[1], "rc_file": "/box/j1/rc", "resolved_card": "a6000"}
        job.update(kw)
        return job

    def peek(self):
        if self.cancelled and self.cancel_to:
            return self.job(status=self.cancel_to)
        return self.job(**self.states[min(self.polls, len(self.states) - 1)])

    def poll(self):
        job = self.peek()
        if job["status"] in TERMINAL and not self.cancelled:
            if self.gate and not self.gate():
                return self.job(status="running")
            if self.late:
                timer = threading.Timer(0.1, self.append_late, [self.late])
                timer.daemon = True
                timer.start()
                self.late = None
        self.polls += 1
        return job

    def append_late(self, late):
        for path, data in zip(self.files, late):
            with contextlib.suppress(OSError), open(path, "ab") as f:
                f.write(data)

    def handle(self, method, path, body, ctype):
        self.requests.append((method, path, body, ctype))
        if (method, path) == ("GET", "/health"):
            time.sleep(self.slow)
            return self.health
        if (method, path) == ("POST", "/jobs"):
            return self.post or (201, self.job(**self.states[0]))
        if (method, path) == ("GET", "/jobs"):  # `listing` (summaries) replaces the default, j1 as it stands
            return 200, self.listing if self.listing is not None else [{"id": "j1", "status": self.peek()["status"]}]
        if path == "/jobs/j1" and method == "GET":
            return self.job_reply or (200, self.poll())  # `job_reply` (status, object) replaces j1's reply
        if path == "/jobs/j1/cancel" and method == "POST":
            self.cancelled = True
            return 200, self.peek()
        return 404, {"error": "no such job"}

    def cancels(self):
        return [r for r in self.requests if r[1] == "/jobs/j1/cancel"]

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


JOB_ARGS = ("--card", "a6000", "--class", "round", "--expected-s", "90", "--name", "tag-1")


class Fix:
    """One case's world: a temp dir and a helper that runs a command in this process; for the client cases also a fake
    daemon, an ssh stub that runs its last argument locally (so `curl` and `tail -F` meet the fake and its files), and
    the client as a child process. Every pid a case leaves behind is in a file here, and cleanup signals only those."""

    def __init__(self, tmp):
        self.tmp, self.fake, self.procs = tmp, None, []
        self.out, self.err = os.path.join(tmp, "job.out"), os.path.join(tmp, "job.err")
        self.rc_dir, self.ssh = os.path.join(tmp, "rc"), os.path.join(tmp, "ssh")
        self.ssh_log = os.path.join(tmp, "ssh.log")
        self.tail_pids, self.hung_pids = os.path.join(tmp, "tail.pids"), os.path.join(tmp, "hung.pid")
        self.follow_pids = os.path.join(tmp, "follow.pids")  # the tails started with -F (tail_pids has every tail)
        self.replay_fail = os.path.join(tmp, "replay-fail")
        self.tail_stall = os.path.join(tmp, "tail-stall")  # a -F tail passes 5 bytes on, then stalls until stopped

    def call(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = main(list(argv))
        return rc, out.getvalue(), err.getvalue()

    def script(self, path, body):
        with open(path, "w") as f:
            f.write("#!/bin/sh\n" + body)
        os.chmod(path, 0o755)

    def stubs(self):
        if os.path.exists(self.ssh):
            return
        for tool in ("curl", "tail"):
            check(shutil.which(tool), f"the client cases need {tool} on PATH")
        bindir, q = os.path.join(self.tmp, "bin"), shlex.quote
        os.makedirs(bindir)
        self.script(os.path.join(bindir, "tail"),
                    # A follow (not dump's read) records itself in follow_pids before tail_pids, so a wait on
                    # tail_pids never returns while a counted follow is still unrecorded.
                    f'case " $* " in *" -F "*) echo $$ >> {q(self.follow_pids)} ;; esac\n'
                    f'echo $$ >> {q(self.tail_pids)}\n'
                    f'case " $* " in *" -F "*)\n'
                    f'  if [ -f {q(self.tail_stall)} ]; then for f; do :; done; head -c 5 "$f"; exec sleep 600; fi ;;\n'
                    f'esac\n'
                    f'exec {q(shutil.which("tail"))} "$@"\n')
        if not shutil.which("stdbuf"):  # a Mac: the box's `stdbuf -o0 tail` runs here as plain tail
            self.script(os.path.join(bindir, "stdbuf"), 'shift\nexec "$@"\n')
        self.script(self.ssh, f'printf "%s\\n" "$*" >> {q(self.ssh_log)}\nfor last; do :; done\n'
                              f'if [ -f {q(self.replay_fail)} ]; then\n'
                              f'  case "$last" in\n'
                              f'    *"-F "*) ;;\n'
                              f'    *"exec tail -c "*) exit 255 ;;\n'
                              f'  esac\n'
                              f'fi\n'
                              f'PATH={q(bindir)}:$PATH\nexport PATH\nexec sh -c "$last"\n')
        os.makedirs(self.rc_dir, exist_ok=True)

    def files(self, out=b"", err=b""):
        for path, data in ((self.out, out), (self.err, err)):
            with open(path, "wb") as f:
                f.write(data)

    def serve(self, states, **kw):
        if self.fake:
            self.fake.close()
        if not os.path.exists(self.out):
            self.files()
        self.fake = Fake((self.out, self.err), states, **kw)
        return self.fake

    def env(self, **extra):
        self.stubs()
        env = {k: v for k, v in os.environ.items()
               if not k.startswith("BOXQ_") and k != "BLOOMERY_BOX"}
        env.update(BOXQ_SSH=self.ssh, BOXQ_HOST="testbox", BOXQ_POLL_S="0.2", BOXQ_CANCEL_WAIT_S="10",
                   BOXQ_RECONNECT_S="0")
        if self.fake:
            env["BOXQ_URL"] = self.fake.url
        for k, v in extra.items():
            env.pop(k, None) if v is None else env.__setitem__(k, v)
        return env

    def start(self, *argv, **extra):
        p = subprocess.Popen([sys.executable, os.path.abspath(__file__), *argv], env=self.env(**extra),
                             stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.procs.append(p)
        return p

    def finish(self, p, secs=40):
        try:
            out, err = p.communicate(timeout=secs)
        except subprocess.TimeoutExpired:
            p.kill()
            p.communicate()
            raise AssertionError(f"the client did not end in {secs} s")
        return p.returncode, out, err

    def client(self, *argv, **extra):
        return self.finish(self.start(*argv, **extra))

    def run_args(self, *extra, words=("true",)):
        return ("run", *JOB_ARGS, "--rc-dir", self.rc_dir, *extra, "--", *words)

    def rc_file(self, jid="j1"):
        try:
            with open(os.path.join(self.rc_dir, f"{jid}.rc")) as f:
                return f.read()
        except OSError:
            return None

    def rm_rc(self, jid="j1"):
        with contextlib.suppress(FileNotFoundError):
            os.unlink(os.path.join(self.rc_dir, f"{jid}.rc"))

    def pids(self, path):
        try:
            with open(path) as f:
                return [int(x) for x in f.read().split()]
        except OSError:
            return []

    def wait_pids(self, path, n, secs=10):
        secs = load_cap(secs)
        end = time.monotonic() + secs
        while len(self.pids(path)) < n:
            check(time.monotonic() < end,
                  f"{n} pids did not appear in {os.path.basename(path)} within {secs:.0f} s")
            time.sleep(0.02)

    def gone(self, path, n=None, secs=3):
        """The pids a stub recorded are all gone; the record restarts. For tail_pids, n counts the -F tails among them
        (each attach that ends also reads its files' rest with tail -c)."""
        pids = self.pids(path)
        counted = self.pids(self.follow_pids) if path == self.tail_pids else pids
        check(n is None or len(counted) == n, f"{os.path.basename(path)} holds {len(counted)} counted pids, {n} expected")
        end = time.monotonic() + secs
        while any(alive(p) for p in pids):
            check(time.monotonic() < end, f"pids still alive {secs} s on: {[p for p in pids if alive(p)]}")
            time.sleep(0.02)
        open(path, "w").close()
        if path == self.tail_pids:
            open(self.follow_pids, "w").close()

    def wait_requests(self, fake, n, which=None, secs=10):
        end = time.monotonic() + secs
        while len(which() if which else fake.requests) < n:
            check(time.monotonic() < end, f"the daemon saw fewer than {n} of the requests it waited for")
            time.sleep(0.02)

    def wait_polls(self, fake, n):
        self.wait_requests(fake, n, lambda: [r for r in fake.requests if r[:2] == ("GET", "/jobs/j1")])

    def cleanup(self):
        for p in self.procs:
            if p.poll() is None:
                p.kill()
                p.communicate()
        for path in (self.tail_pids, self.follow_pids, self.hung_pids):
            for pid in self.pids(path):
                with contextlib.suppress(ProcessLookupError):
                    os.kill(pid, signal.SIGKILL)
        if self.fake:
            self.fake.close()


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def case_submit_no_prediction(fx):
    fake = fx.serve([{"status": "queued"}])
    args = [a for a in JOB_ARGS if a not in ("--expected-s", "90")]
    rc, out, err = fx.client("submit", *args, "--", "true")
    check(rc == 0 and out == b"j1\n", f"a job with no prediction is posted: {rc} {out!r} {err!r}")
    check(err == b"boxq: tag-1 class=round card=a6000 predicted unknown\n", f"the line says unknown: {err!r}")
    body = json.loads(fake.requests[-1][2])
    check("expected_s" not in body and body["card"] == "a6000", f"the body leaves the field out: {body}")

def case_submit_body(fx):
    fake = fx.serve([{"status": "queued"}])
    long_cmd = "cd ~/repo/x && just gate-y " + "z" * 3000
    rc, out, err = fx.client("submit", *JOB_ARGS, "--owner", "w3b1", "--env", "A=1", "--env", "B=x=y", "--", long_cmd)
    check(rc == 0 and out == b"j1\n", f"submit prints the id and nothing else: {rc} {out!r} {err!r}")
    check(err == b"boxq: tag-1 class=round card=a6000 predicted 90 s\n", f"the predicted-wall line: {err!r}")
    check([r[:2] for r in fake.requests] == [("GET", "/health"), ("POST", "/jobs")],
          f"health, then the post: {fake.requests}")
    body = json.loads(fake.requests[1][2])
    want = {"kind": "command", "argv": long_cmd, "env": {"A": "1", "B": "x=y"}, "owner": "w3b1", "name": "tag-1",
            "card": "a6000", "job_class": "round", "expected_s": 90, "wait": 0}
    check(body == want and "priority" not in body, f"the body is the wire's fields and no priority: {body}")
    check(fake.requests[1][3] == "application/json", "the body is sent as application/json")
    log = open(fx.ssh_log).read().splitlines()
    check(log[1] == f"testbox curl -sS --max-time 20 -X POST -H 'Content-Type: application/json' -H Expect: "
                    f"--data-binary @- -w '\\n%{{http_code}}' {fake.url}/jobs", f"the ssh command of the post: {log[1]!r}")
    check(log[0] == f"testbox curl -sS --max-time 20 -X GET -w '\\n%{{http_code}}' {fake.url}/health",
          f"a GET has no body and no stdin: {log[0]!r}")
    check(fake.requests[0][2] == b"" and fake.requests[0][3] is None, f"the daemon saw a GET with a body: {fake.requests[0]}")
    check("just gate-y" not in "\n".join(log), "the job text is on an ssh command line")
    rc, out, err = fx.client("submit", *JOB_ARGS, "--", "just", "gate-y", "two words")
    body = json.loads(fake.requests[-1][2])
    tree = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    check(body["argv"] == "just gate-y 'two words'" and body["env"] == {} and body["owner"] == tree,
          f"several words are shell-quoted, the owner defaults to the tree {tree}: {body}")
    for env, host in (({"BOXQ_HOST": None, "BLOOMERY_BOX": "bx"}, "bx"), ({"BOXQ_HOST": None}, "ws")):
        fx.client("submit", *JOB_ARGS, "--", "true", **env)
        check(open(fx.ssh_log).read().splitlines()[-1].startswith(f"{host} curl"), f"the host defaults to {host}")
    posts = len(fake.requests)
    for bad, why in ((JOB_ARGS, "no -- command"), ((*JOB_ARGS, "--"), "an empty command"),
                     ((*JOB_ARGS, "--", " "), "a blank command"),
                     (("--card", "moon", *JOB_ARGS[2:], "--", "true"), "an unknown card"),
                     ((*JOB_ARGS[:2], "--class", "media", *JOB_ARGS[4:], "--", "true"), "a class the routes own"),
                     ((*JOB_ARGS[:4], "--expected-s", "0", *JOB_ARGS[6:], "--", "true"), "expected-s 0"),
                     ((*JOB_ARGS[:6], "--name", "a b", "--", "true"), "a name with a space"),
                     ((*JOB_ARGS, "--env", "NOEQ", "--", "true"), "an env without =")):
        rc, out, err = fx.client("submit", *bad)
        check(rc == USAGE and out == b"", f"{why} must be usage (64): {rc} {err!r}")
    check(len(fake.requests) == posts, "a refused command line reached the daemon")


def case_run_done(fx):
    out, err, late = b"o1\nbin\xff\x00end\r\n", b"e1\nwarn\n", (b"late-out\n", b"late-err\n")
    for status, rc in (("done", 0), ("failed", 3)):
        fx.files(out, err)
        fx.serve([{"status": "running"}, {"status": status, "rc": rc}], late=late,
                 gate=lambda: len(fx.pids(fx.tail_pids)) >= 2)
        got, so, se = fx.client(*fx.run_args(), BOXQ_POLL_S="0.5")
        check(got == rc, f"the exit code is the job's rc {rc}: {got} {se!r}")
        check(so == out + late[0], f"stdout is the job's out and what trailed its status, byte for byte: {so!r}")
        check(se.count(err + late[1]) == 1 and out not in se, f"the job's err goes to stderr once, its out not: {se!r}")
        check(fx.rc_file() == f"{rc}\n" and os.listdir(fx.rc_dir) == ["j1.rc"],
              f"the rc mirror: {fx.rc_file()!r} {os.listdir(fx.rc_dir)}")
        fx.gone(fx.tail_pids, 2)
        fx.rm_rc()


def case_lost(fx):
    fx.serve([{"status": "running"}, {"status": "running"}, {"status": "lost"}])
    got, so, se = fx.client(*fx.run_args())
    check(got == 81 and fx.rc_file() == "81\n", f"a lost job exits 81 and mirrors it: {got} {fx.rc_file()!r} {se!r}")


def case_health_down(fx):
    fake = fx.serve([], health=(503, {"error": "starting"}))
    for health, why in (((503, {"error": "starting"}), "a 503"), ((200, {"ok": False}), "ok false"),
                        ((200, {}), "no ok")):
        fake.health = health
        rc, out, err = fx.client("submit", *JOB_ARGS, "--", "true")
        check(rc == 84 and b"daemon-down" in err, f"health {why} must be 84: {rc} {err!r}")
        check(not [r for r in fake.requests if r[0] == "POST"], f"a job was posted past a failed health ({why})")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        closed = f"http://127.0.0.1:{sock.getsockname()[1]}"
    rc, out, err = fx.client("submit", *JOB_ARGS, "--", "true", BOXQ_URL=closed)
    check(rc == 84 and b"daemon-down" in err, f"a daemon that is not there must be 84: {rc} {err!r}")
    rc, out, err = fx.client("attach", "j1", "--rc-dir", fx.rc_dir, BOXQ_URL=closed)
    line = b"boxq: j1 left as it is on the box; resume with: tools/boxq.py attach j1\n"
    check(rc == 84 and line in err and b"poll 2/3 failed" in err and fx.rc_file() is None,
          f"an attach that cannot poll gives up after three, mirrors nothing: {rc} {err!r}")


def case_reconnect(fx):
    """A daemon that drops and returns keeps the job's rc and the bytes already passed on; past the budget the
    attach exits 84 with the resume line and no rc file; a signal while it is down is decided as after any poll."""
    prefix, suffix = b"hello", b" world\n\xff"
    err_a, err_b = b"e1\n", b"warn-b\n"
    fx.files(prefix, err_a)
    fake = fx.serve([{"status": "running"}], gate=lambda: len(fx.pids(fx.tail_pids)) >= 2)
    p = fx.start(*fx.run_args(), BOXQ_POLL_S="0.05", BOXQ_RECONNECT_S="3")
    fx.wait_pids(fx.tail_pids, 2)
    pre, end = b"", time.monotonic() + 5
    while prefix not in pre:
        check(time.monotonic() < end, f"the prefix did not arrive before the blip: {pre!r}")
        ready, _, _ = select.select([p.stdout], [], [], 0.05)
        if ready:
            chunk = os.read(p.stdout.fileno(), 65536)
            if not chunk:
                break
            pre += chunk
    fake.down_until = time.monotonic() + 0.8
    for pid in fx.pids(fx.tail_pids):
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGKILL)
    with open(fx.out, "ab") as f:
        f.write(suffix)
    with open(fx.err, "ab") as f:
        f.write(err_b)
    while time.monotonic() < fake.down_until:
        time.sleep(0.02)
    fake.states = [{"status": "failed", "rc": 3}]
    fake.polls = 0
    fake.gate = None
    try:
        out, err = p.communicate(timeout=15)
    except subprocess.TimeoutExpired:
        p.kill()
        p.communicate()
        raise AssertionError("the client did not end in 15 s after the daemon returned")
    so = pre + out
    check(p.returncode == 3 and fx.rc_file() == "3\n",
          f"a blip ends with the job's rc: {p.returncode} {fx.rc_file()!r} {err!r}")
    check(so == prefix + suffix, f"stdout is the file's bytes, with no gap and no duplicate: {so!r}")
    check(err.count(err_a) == 1 and err.count(err_b) == 1, f"stderr has each err chunk once: {err!r}")
    fx.gone(fx.tail_pids)
    fx.rm_rc()

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        closed = f"http://127.0.0.1:{sock.getsockname()[1]}"
    t0 = time.monotonic()
    rc, out, err = fx.client("attach", "j1", "--rc-dir", fx.rc_dir, BOXQ_URL=closed, BOXQ_POLL_S="0.05",
                             BOXQ_RECONNECT_S="0.35")
    line = b"boxq: j1 left as it is on the box; resume with: tools/boxq.py attach j1\n"
    check(rc == 84 and line in err and fx.rc_file() is None and time.monotonic() - t0 >= 0.35,
          f"past the budget: 84, the resume line, no rc file: {rc} {err!r} {fx.rc_file()!r}")

    # a signal while the daemon is down: a job last seen running is left running (no cancel to post)
    fx.files(b"o\n", b"e\n")
    fake = fx.serve([{"status": "running"}])
    p = fx.start(*fx.run_args(), BOXQ_POLL_S="0.05", BOXQ_RECONNECT_S="5")
    fx.wait_pids(fx.tail_pids, 2)
    fake.down_until = time.monotonic() + 5
    time.sleep(0.45)
    p.send_signal(signal.SIGINT)
    got, so, se = fx.finish(p, 10)
    check(got == 130 and b"left running" in se and fake.cancels() == [] and fx.rc_file() is None,
          f"INT while the daemon is down on a running job: left running, no cancel, no mirror: "
          f"{got} {se!r} {fake.cancels()} {fx.rc_file()!r}")
    fx.gone(fx.tail_pids)

    # a job last seen queued is cancelled, the post waiting for the daemon to return
    fake = fx.serve([{"status": "queued", "wait_reason": "card-busy"}])
    p = fx.start(*fx.run_args(), BOXQ_POLL_S="0.05", BOXQ_RECONNECT_S="5", BOXQ_CANCEL_WAIT_S="10")
    fx.wait_polls(fake, 2)
    fake.down_until = time.monotonic() + 5
    time.sleep(0.45)
    p.send_signal(signal.SIGINT)
    time.sleep(0.3)
    check(p.poll() is None and fake.cancels() == [], "the cancel waits while the daemon is down")
    fake.down_until = 0
    got, so, se = fx.finish(p, 10)
    check(got == 130 and b"elsewhere" not in se and len(fake.cancels()) == 1 and fx.rc_file() == "130\n",
          f"INT on a queued job while the daemon is down: one cancel once it returns: "
          f"{got} {se!r} {fake.cancels()} {fx.rc_file()!r}")
    fx.rm_rc()


def case_refusal(fx):
    fake = fx.serve([], post=(409, {"error": "a sitting is draining the box", "rc": 75}))
    rc, out, err = fx.client("submit", *JOB_ARGS, "--", "true")
    check(rc == 75 and out == b"" and b"a sitting is draining the box" in err,
          f"a 4xx exits with its rc and prints its text: {rc} {err!r}")
    check(b"predicted 90 s" in err, f"the predicted wall is printed before the post: {err!r}")
    for post, why in (((400, {"error": "bad", "rc": 0}), "a refusal that names rc 0"),
                      ((500, {"error": "boom", "rc": 70}), "a 5xx"),
                      ((201, {"status": "queued"}), "a created job with no id"),
                      ((200, {"id": "j1"}), "a 200 where the wire says 201")):
        fake.post = post
        rc, out, err = fx.client("submit", *JOB_ARGS, "--", "true")
        check(rc == 84 and out == b"" and b"bad-reply" in err,
              f"{why} is the daemon off the wire, 84: {rc} {out!r} {err!r}")
    rc, out, err = fx.client("attach", "j9", "--rc-dir", fx.rc_dir)
    check(rc == 64 and b"no-such-job" in err, f"a 404 is 64 no such job: {rc} {err!r}")
    for state, why in (({"status": "claimed"}, "claimed"),
                       ({"status": "done", "rc": None}, "a done job without an rc")):
        fake.states = [state]
        rc, out, err = fx.client("attach", "j1", "--rc-dir", fx.rc_dir)
        check(rc == 84 and b"bad-reply" in err and fx.rc_file() is None,
              f"{why} is named, 84, mirrors nothing: {rc} {err!r}")


def case_passthrough(fx):
    for rc in (80, 83, 85):
        fx.serve([{"status": "running"}, {"status": "failed", "rc": rc}])
        got, so, se = fx.client(*fx.run_args())
        check(got == rc and fx.rc_file() == f"{rc}\n",
              f"the job's rc {rc} passes through unchanged: {got} {fx.rc_file()!r} {se!r}")


def case_queued_line(fx):
    queued = {"status": "queued", "wait_reason": "card-busy"}
    states = [queued, queued, {**queued, "eta_s": 12}, {"status": "running"}, {"status": "running"},
              {"status": "done", "rc": 0}]
    fx.serve(states)
    rc, so, se = fx.client(*fx.run_args())
    lines = se.decode().splitlines()
    waits = [l for l in lines if l.startswith("boxq: j1 waiting")]
    check(rc == 0 and waits == ["boxq: j1 waiting: card-busy, eta unknown"],
          f"one wait line a minute, eta unknown for null: {waits} {rc}")
    moved = [l for l in lines if re.fullmatch(r"boxq: j1 queued \d+ s \(wait_reason card-busy\)", l)]
    check(len(moved) == 1, f"one line when the job leaves the queue: {lines}")
    fx.serve(states)
    rc, so, se = fx.client(*fx.run_args(), BOXQ_NOTE_S="0")
    waits = [l for l in se.decode().splitlines() if l.startswith("boxq: j1 waiting")]
    check(waits == ["boxq: j1 waiting: card-busy, eta unknown"] * 2 + ["boxq: j1 waiting: card-busy, eta 12 s"],
          f"a line every poll at 0 s: {waits}")


def case_int_cancel(fx):
    for sig, want in ((signal.SIGINT, 130), (signal.SIGTERM, 143)):
        fx.files(b"o\n", b"e\n")
        fake = fx.serve([{"status": "running"}])
        p = fx.start(*fx.run_args())
        fx.wait_pids(fx.tail_pids, 2)
        p.send_signal(sig)
        got, so, se = fx.finish(p)
        check(got == want and fake.cancels() == [] and b"left running" in se and b"boxq.py attach j1" in se,
              f"{sig.name} on a running job exits {want}, posts no cancel and names the attach: {got} {se!r} {fake.cancels()}")
        check(fx.rc_file() is None, f"a job left running gets no rc mirror: {fx.rc_file()!r}")
        fx.gone(fx.tail_pids, 2)
    for sig, want in ((signal.SIGINT, 130), (signal.SIGTERM, 143)):
        fake = fx.serve([{"status": "queued", "wait_reason": "card-busy"}])
        p = fx.start(*fx.run_args())
        fx.wait_polls(fake, 2)
        p.send_signal(sig)
        got, so, se = fx.finish(p)
        cancels = fake.cancels()
        check(got == want and b"elsewhere" not in se and fx.pids(fx.follow_pids) == [],
              f"{sig.name} on a queued job exits {want}, the cancel is ours, no tail to stop: {got} {se!r}")
        check(len(cancels) == 1 and cancels[0][0] == "POST" and cancels[0][2] == b"{}",
              f"one POST cancel with {{}}: {cancels}")
        check(fx.rc_file() == f"{want}\n", f"the mirror holds {want}: {fx.rc_file()!r}")
        fx.rm_rc()
    fake = fx.serve([{"status": "running"}], slow=1.0)
    p = fx.start(*fx.run_args())
    fx.wait_requests(fake, 1)
    p.send_signal(signal.SIGINT)
    got, so, se = fx.finish(p)
    check(got == 130 and not [r for r in fake.requests if r[:2] == ("POST", "/jobs")] and b"interrupted" in se,
          f"a signal before the post posts nothing: {got} {se!r} {fake.requests}")
    fx.rm_rc()
    fake = fx.serve([{"status": "queued", "wait_reason": "card-busy"}], cancel_to=None)
    p = fx.start(*fx.run_args(), BOXQ_CANCEL_WAIT_S="1")
    fx.wait_polls(fake, 2)
    t0 = time.monotonic()
    p.send_signal(signal.SIGINT)
    got, so, se = fx.finish(p)
    check(got == 81 and b"j1" in se and time.monotonic() - t0 >= 1.0,
          f"a job still queued after the wait: 81 naming it: {got} {se!r}")
    check(fx.rc_file() is None, "a job that is not terminal gets no rc mirror")
    fake = fx.serve([{"status": "queued", "wait_reason": "card-busy"}], cancel_to=None)
    p = fx.start(*fx.run_args(), BOXQ_CANCEL_WAIT_S="100")
    fx.wait_polls(fake, 2)
    p.send_signal(signal.SIGINT)
    fx.wait_requests(fake, 1, fake.cancels)
    p.send_signal(signal.SIGINT)
    got, so, se = fx.finish(p, 10)
    check(got == 81, f"a second signal ends the wait: {got} {se!r}")


class RecTrap:
    """A Trap that never sleeps and keeps the waits it was asked for."""

    def __init__(self):
        self.sigs, self.waits = [], []

    def wait(self, secs):
        self.waits.append(secs)


def case_flush_cap(fx):
    fx.serve([{"status": "running"}, {"status": "done", "rc": 0}])
    for poll, flush in (("5", 2.0), ("1", 1.0)):
        trap = RecTrap()
        rc = attach_job(Queue(fx.env(BOXQ_POLL_S=poll)), "j1", fx.rc_dir, trap)
        # after the flush only dump's own short waits for its tail -c reads follow
        check(rc == 0 and trap.waits[:2] == [float(poll), flush] and all(w <= 0.1 for w in trap.waits[2:]),
              f"the flush after a terminal status is min(poll, 2 s): poll {poll} s waited {trap.waits}")
        fx.gone(fx.tail_pids)  # the waits are instant, so a tail may be stopped before it recorded its pid
        fx.fake.polls = 0


class SigTrap(RecTrap):
    """A RecTrap that delivers one signal during its first wait, as a signal during the sleep between two polls."""

    def __init__(self, sig):
        super().__init__()
        self.sig = sig

    def wait(self, secs):
        super().wait(secs)
        if len(self.waits) == 1:
            self.sigs.append(self.sig)

    rc = Trap.rc


def case_signal_then_done(fx):
    for before in ({"status": "running"}, {"status": "queued", "wait_reason": "card-busy"}):
        for sig, want in ((signal.SIGINT, 130), (signal.SIGTERM, 143)):
            fake = fx.serve([before, {"status": "done", "rc": 0}])
            trap = SigTrap(sig)
            rc = attach_job(Queue(fx.env()), "j1", fx.rc_dir, trap)
            check(rc == want and fake.cancels() == [] and fx.rc_file() is None,
                  f"{sig.name} after a {before['status']} poll wins over a done reply: {want}, no cancel, no mirror: "
                  f"{rc} {fake.cancels()} {fx.rc_file()!r}")
            fx.gone(fx.tail_pids)
            fx.fake.polls = 0


def case_drain_closed_reader(fx):
    src_r, src_w = os.pipe()
    dst_r, dst_w = os.pipe()
    os.set_blocking(src_r, False)
    os.close(dst_r)  # the reader of our stdout went away (`| head`, `| grep -q`)
    os.write(src_w, b"abc")
    os.close(src_w)
    t = Tail(None, src_r, dst_w)
    try:
        drain([t])
        ok, why = True, ""
    except BrokenPipeError as e:
        ok, why = False, repr(e)
    finally:
        os.close(dst_w)
    check(ok and t.n == 3 and t.src is None,
          f"a closed reader drops the tail's bytes, still counts them and raises nothing: {why} n={t.n}")


class Clock:
    """A fake monotonic clock: only a stub's request or a trap's wait moves it."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class ClockTrap:
    def __init__(self, clock, sigs=()):
        self.clock, self.sigs = clock, list(sigs)

    def wait(self, secs):
        self.clock.t += secs

    rc = Trap.rc


class StubQueue:
    """The attach's Queue with no wire: `poll_s`, the budgets, and a job() and cancel() the case scripts."""

    def __init__(self, poll_s, reconnect_s=0.0, cancel_wait_s=0.0):
        self.poll_s, self.reconnect_s, self.cancel_wait_s = poll_s, reconnect_s, cancel_wait_s
        self.note_s = 60.0


def with_clock(clock, fn):
    real = time.monotonic
    time.monotonic = clock
    try:
        return fn()
    finally:
        time.monotonic = real


def case_reconnect_wall(fx):
    """The reconnect budget is wall time: a poll that hangs its whole timeout counts, not only the sleeps between."""
    clock = Clock()
    q = StubQueue(5.0, reconnect_s=30.0)
    asked = []

    def job(jid, timeout=30.0):
        asked.append(timeout)
        clock.t += timeout  # the daemon never answers
        raise Down("daemon-down", "GET /jobs/j1: no reply")

    q.job = job
    raised = _cancel_run(fx, q, clock, sigs=())
    check(isinstance(raised, Down) and raised.name == "daemon-down" and raised.line == RESUME_LINE.format(jid="j1"),
          f"a budget past ends in 84 with the resume line: {raised!r}")
    # two misses and their sleeps, the third miss at 100 s, then the budget of 30 s: 130 s, not 150 or more
    check(clock.t <= 131.0, f"the budget counts the requests' time: the attach ended at {clock.t:g} s, 131 s at most")
    check(min(asked) < 30.0, f"a poll inside the budget is asked for no longer than what is left: {asked}")


def case_reconnect_step(fx):
    """A long budget backs off in pauses of at most RECONNECT_STEP_S; an ssh that cannot start is not waited out."""
    clock, said = Clock(), []
    q = StubQueue(5.0, reconnect_s=RECONNECT_S)

    def job(jid, timeout=30.0):
        raise Down("daemon-down", "GET /jobs/j1: no reply")

    q.job = job
    real_say = globals()["say"]
    globals()["say"] = said.append
    try:
        raised = _cancel_run(fx, q, clock, sigs=(), quiet=False)
        steps = [float(m.rsplit(" in ", 1)[1].split()[0]) for m in said if "reconnecting, next poll in" in m]
        check(isinstance(raised, Down) and steps and max(steps) == RECONNECT_STEP_S and clock.t >= RECONNECT_S,
              f"pauses stop growing at {RECONNECT_STEP_S:g} s and the budget is spent: max {max(steps or [0]):g}, "
              f"ended at {clock.t:g} s {raised!r}")
        clock = Clock()

        def no_ssh(jid, timeout=30.0):
            raise Down("no-ssh", "/typo: No such file or directory")

        q.job = no_ssh
        raised = _cancel_run(fx, q, clock, sigs=(), quiet=False)
        check(isinstance(raised, Down) and raised.name == "no-ssh" and clock.t < 60,
              f"an ssh that cannot start ends 84 after the polls, not the backoff: {clock.t:g} s {raised!r}")
    finally:
        globals()["say"] = real_say


def _cancel_run(fx, q, clock, sigs=(signal.SIGINT,), quiet=True):
    real_say = globals()["say"]
    globals()["say"] = (lambda msg: None) if quiet else real_say
    try:
        try:
            with_clock(clock, lambda: attach_job(q, "j1", fx.rc_dir, ClockTrap(clock, sigs)))
            return None
        except Refusal as e:
            return e
    finally:
        globals()["say"] = real_say


def case_cancel_deadline(fx):
    """The cancel wait counts from the first signal. A stalled post, a stalled get and a long poll sleep each stay
    inside it: the fixture's stubs spend the timeout they are given."""
    clock = Clock()
    q = StubQueue(1.0, cancel_wait_s=4.0)
    posts = []

    def cancel(jid, timeout=30.0):
        posts.append((clock.t, timeout))
        if len(posts) <= 3:
            raise Down("daemon-down", "POST cancel: no reply")
        return {}

    q.job = lambda jid, timeout=30.0: {"id": jid, "status": "queued", "wait_reason": None}
    q.cancel = cancel
    raised = _cancel_run(fx, q, clock)
    check(raised is not None and raised.name == "cancel-pending" and len(posts) == 4,
          f"three failed posts and an accepted one, the job still queued: cancel-pending: {raised!r} {posts}")
    check(clock.t <= 4.5, f"the wait of 4 s runs from the first signal: the attach ended at {clock.t:g} s")

    clock = Clock()
    q = StubQueue(1.0, cancel_wait_s=4.0)

    def stall_post(jid, timeout=30.0):
        clock.t += timeout
        raise Down("daemon-down", "POST cancel: no reply")

    q.job = lambda jid, timeout=30.0: {"id": jid, "status": "queued", "wait_reason": None}
    q.cancel = stall_post
    raised = _cancel_run(fx, q, clock)
    check(raised is not None and raised.name == "cancel-pending" and clock.t <= 4.5,
          f"a stalled cancel post stays inside 4 s and ends cancel-pending: {clock.t:g} s {raised!r}")

    clock = Clock()
    q = StubQueue(1.0, cancel_wait_s=4.0)
    gets = {"n": 0}

    def stall_get(jid, timeout=30.0):
        gets["n"] += 1
        if gets["n"] > 1:
            clock.t += timeout
        return {"id": jid, "status": "queued", "wait_reason": None}

    q.job = stall_get
    q.cancel = lambda jid, timeout=30.0: {}
    raised = _cancel_run(fx, q, clock)
    check(raised is not None and raised.name == "cancel-pending" and gets["n"] >= 2 and clock.t <= 4.5,
          f"a stalled status get stays inside 4 s and ends cancel-pending: {clock.t:g} s gets={gets['n']} {raised!r}")

    clock = Clock()
    q = StubQueue(10.0, cancel_wait_s=4.0)
    q.job = lambda jid, timeout=30.0: {"id": jid, "status": "queued", "wait_reason": None}
    q.cancel = lambda jid, timeout=30.0: {}
    raised = _cancel_run(fx, q, clock)
    check(raised is not None and raised.name == "cancel-pending" and clock.t <= 4.5,
          f"a poll sleep longer than the budget is cut to it: ended at {clock.t:g} s {raised!r}")


def case_drain_full_pipe(fx):
    """A reader that never reads cannot hold the client: the write does not block, the unwritten bytes are not counted,
    and a SIGTERM during the stall exits 143. Where the platform can shrink a pipe (Linux), each is 4 KiB so the stall
    comes in a few writes; each fill writes at most one pipe buffer, so drain runs before that write can block."""
    src_r, src_w = os.pipe()
    dst_r, dst_w = os.pipe()
    os.set_blocking(src_r, False)
    for fd in (src_r, src_w, dst_r, dst_w) if hasattr(fcntl, "F_SETPIPE_SZ") else ():
        fcntl.fcntl(fd, fcntl.F_SETPIPE_SZ, 4096)
    t = Tail(None, src_r, dst_w)

    def fill():
        while True:
            os.write(src_w, b"x" * min(4096, os.fpathconf(src_w, "PC_PIPE_BUF")))
            drain([t])
            if t.pending:
                return

    th = threading.Thread(target=fill, daemon=True)
    th.start()
    th.join(3)
    stalled = th.is_alive()
    if stalled:
        os.close(dst_r)  # frees the blocked write; the case is red below
    else:
        n_before, held = t.n, len(t.pending)
        os.set_blocking(dst_r, False)
        got = 0
        while True:
            try:
                chunk = os.read(dst_r, 1 << 16)
            except BlockingIOError:
                break
            got += len(chunk)
        drain([t])
        for fd in (dst_r, dst_w):
            os.close(fd)
    os.close(src_w)
    check(not stalled, "a drain into a full pipe blocked for 3 s")
    check(held > 0 and n_before == got,
          f"a full pipe leaves the bytes pending, counted only once written: held={held} n={n_before} read={got}")
    fx.files(b"o" * 3000000, b"e\n")
    fake = fx.serve([{"status": "running"}])
    p = fx.start(*fx.run_args())  # its stdout pipe is never read
    fx.wait_pids(fx.tail_pids, 2)
    time.sleep(1.0)
    p.send_signal(signal.SIGTERM)
    try:
        rc = p.wait(timeout=10)
    except subprocess.TimeoutExpired:
        p.kill()
        p.wait()
        raise AssertionError("a SIGTERM did not end a client whose stdout pipe is full within 10 s")
    check(rc == 143 and fake.cancels() == [], f"SIGTERM with a full stdout pipe exits 143, no cancel: {rc}")
    fx.gone(fx.tail_pids, 2)


def case_pending_replay(fx):
    """Bytes a full stdout pipe left pending are written from the job's file before the rc is mirrored."""
    body = b"P" * (1 << 20) + b"0123456789A"
    fx.files(body, b"")
    fx.serve([{"status": "running"}, {"status": "done", "rc": 0}],
             gate=lambda: len(fx.pids(fx.tail_pids)) >= 2)
    p = fx.start(*fx.run_args(), BOXQ_POLL_S="0.05")
    fx.wait_pids(fx.tail_pids, 2)
    end = time.monotonic() + 3
    while time.monotonic() < end and p.poll() is None and fx.rc_file() is None:
        time.sleep(0.05)
    try:
        out, err = p.communicate(timeout=20)
    except subprocess.TimeoutExpired:
        p.kill()
        p.communicate()
        raise AssertionError("the client did not finish replaying a full stdout pipe within 20 s")
    check(p.returncode == 0 and out == body and fx.rc_file() == "0\n",
          f"pending bytes are replayed before the mirror: rc={p.returncode} got={len(out)} "
          f"want={len(body)} mirror={fx.rc_file()!r} err={err!r}")
    fx.gone(fx.tail_pids)


def case_stalled_tail(fx):
    """A tail that stalls before the file's end, with nothing pending here: the rest is read from the file before the
    rc is mirrored."""
    body = b"hello world\n"
    fx.files(body, b"")
    open(fx.tail_stall, "w").close()
    fx.serve([{"status": "running"}, {"status": "done", "rc": 0}], gate=lambda: len(fx.pids(fx.tail_pids)) >= 2)
    got, out, err = fx.client(*fx.run_args(), BOXQ_POLL_S="0.05")
    check(got == 0 and out == body and fx.rc_file() == "0\n",
          f"a stalled tail's unread rest is read before the mirror: rc={got} out={out!r} mirror={fx.rc_file()!r} err={err!r}")
    fx.gone(fx.tail_pids)


def case_replay_ssh(fx):
    """An ssh that exits 255 while unread bytes are replayed is exit 84, and no rc file is written."""
    body = b"P" * (1 << 20) + b"0123456789A"
    fx.files(body, b"")
    open(fx.replay_fail, "w").close()
    fx.serve([{"status": "running"}, {"status": "done", "rc": 0}],
             gate=lambda: len(fx.pids(fx.tail_pids)) >= 2)
    p = fx.start(*fx.run_args(), BOXQ_POLL_S="0.05")
    fx.wait_pids(fx.tail_pids, 2)
    # Leave stdout unread so the pipe fills and the replay ssh is the one that would finish the file.
    end = time.monotonic() + 3
    while time.monotonic() < end and p.poll() is None and fx.rc_file() is None:
        time.sleep(0.05)
    try:
        out, err = p.communicate(timeout=20)
    except subprocess.TimeoutExpired:
        p.kill()
        p.communicate()
        raise AssertionError("a replay whose ssh fails did not end the client within 20 s")
    check(p.returncode == 84 and fx.rc_file() is None and out != body
          and b"left as it is on the box" in err and b"attach j1" in err,
          f"a replay ssh exiting 255 exits 84 and mirrors nothing: rc={p.returncode} "
          f"got={len(out)} want={len(body)} mirror={fx.rc_file()!r} err={err!r}")
    fx.gone(fx.tail_pids)


def case_replay_signal(fx):
    """A signal while unread bytes are replayed is the signal's exit, and no rc file is written."""
    body = b"P" * (1 << 20) + b"0123456789A"
    fx.files(body, b"")
    fx.serve([{"status": "running"}, {"status": "done", "rc": 0}],
             gate=lambda: len(fx.pids(fx.tail_pids)) >= 2)
    p = fx.start(*fx.run_args(), BOXQ_POLL_S="0.05")
    fx.wait_pids(fx.tail_pids, 3)  # the third pid is the replay tail: the job is already terminal
    p.send_signal(signal.SIGTERM)
    # Do not read stdout yet: a full pipe is what holds the replay, so the signal is what ends it.
    try:
        rc = p.wait(timeout=20)
    except subprocess.TimeoutExpired:
        p.kill()
        p.wait()
        raise AssertionError("a signal during replay did not end the client within 20 s")
    out, err = p.communicate()
    check(rc == 143 and fx.rc_file() is None,
          f"a signal during replay exits 143 and mirrors nothing: rc={rc} mirror={fx.rc_file()!r} "
          f"got={len(out)} err={err!r}")
    fx.gone(fx.tail_pids)


def case_queued_age(fx):
    now, sec = datetime.datetime.now(), datetime.timedelta(seconds=1)
    kst = datetime.timezone(datetime.timedelta(hours=9))
    for raw, lo, hi in (((now - 100 * sec).isoformat(timespec="seconds"), 100, 104),
                        ((now - 50 * sec).astimezone(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), 50, 54),
                        ((datetime.datetime.now(kst) - 30 * sec).isoformat(), 30, 34)):
        age = submitted_age({"submitted": raw})
        check(age is not None and lo <= age <= hi, f"the age of {raw} is about {lo} s: {age}")
    for raw in (None, 7, "not a time", (now + 60 * sec).isoformat()):
        check(submitted_age({"submitted": raw}) is None, f"{raw!r} is no usable submitted time")
    check(submitted_age({}) is None, "a job without submitted has no age")
    submitted = (datetime.datetime.now() - 100 * sec).isoformat(timespec="seconds")
    fx.serve([{"status": "queued", "wait_reason": "card-busy", "submitted": submitted},
              {"status": "running", "submitted": submitted}, {"status": "running", "submitted": submitted},
              {"status": "done", "rc": 0, "submitted": submitted}])
    rc, so, se = fx.client(*fx.run_args())
    m = re.search(rb"queued (\d+) s \(wait_reason", se)
    check(rc == 0 and m and 100 <= int(m.group(1)) <= 120, f"the queued seconds come from `submitted`: {rc} {se!r}")


def case_load_ready(fx):
    """Pids that appear after the idle 10 s, inside the load cap, still satisfy wait_pids.

    Load 80 on 10 cpus makes the cap 80 s. The pids land at 15 s of a fake clock, so a constant
    cap of 10 s fails this case and the scaled cap returns as soon as they appear."""
    path = os.path.join(fx.tmp, "late.pids")
    open(path, "w").close()
    os.environ["BLOOMERY_SELFTEST_LOAD"] = "80"
    os.environ["BLOOMERY_SELFTEST_NCPU"] = "10"
    now = [0.0]
    real_mono, real_sleep = time.monotonic, time.sleep

    def mono():
        return now[0]

    def sleep(seconds):
        now[0] += seconds
        if now[0] >= 15 and os.path.getsize(path) == 0:
            with open(path, "w") as f:
                f.write("11 22\n")

    time.monotonic, time.sleep = mono, sleep
    try:
        fx.wait_pids(path, 2, secs=10)
    finally:
        time.monotonic, time.sleep = real_mono, real_sleep
        os.environ.pop("BLOOMERY_SELFTEST_LOAD", None)
        os.environ.pop("BLOOMERY_SELFTEST_NCPU", None)
    check(fx.pids(path) == [11, 22], f"the late pids were recorded: {fx.pids(path)}")
    check(now[0] >= 15, f"the wait did not see the late pids: t={now[0]}")
    check(now[0] < 20, f"the wait sat out the scaled cap: t={now[0]}")


def case_cancelled_elsewhere(fx):
    for state, want in (({"status": "cancelled", "rc": 137}, 137), ({"status": "cancelled", "rc": None}, 81)):
        fx.serve([{"status": "running"}, {"status": "running"}, state], gate=lambda: len(fx.pids(fx.tail_pids)) >= 2)
        got, so, se = fx.client(*fx.run_args())
        check(got == want and b"boxq: j1 cancelled elsewhere\n" in se and fx.rc_file() == f"{want}\n",
              f"a job cancelled by someone else exits {want} and says so: {got} {se!r} {fx.rc_file()!r}")
        fx.gone(fx.tail_pids, 2)
        fx.rm_rc()


def case_attach_env(fx):
    fx.files(b"out\n", b"")
    fx.serve([{"status": "done", "rc": 0}])
    home = os.path.join(fx.tmp, "home")
    for env, where in (({"BOXQ_RC_DIR": os.path.join(fx.tmp, "envrc")}, "envrc"),
                       ({"BOXQ_RC_DIR": None, "HOME": home}, "home/.cache/bloomery/boxq")):
        got, so, se = fx.client("attach", "j1", **env)
        mirrored = os.path.join(fx.tmp, where, "j1.rc")
        check(got == 0 and so == b"out\n" and os.path.exists(mirrored) and open(mirrored).read() == "0\n",
              f"attach to a finished job prints its out and mirrors into {where}: {got} {so!r} {se!r}")
    check(fx.pids(fx.follow_pids) == [], "a job that finished before the attach is read whole, with no tail -F")
    for path in (fx.out, fx.err):
        os.unlink(path)
    fx.serve([{"status": "cancelled"}])
    got, so, se = fx.client("attach", "j1", BOXQ_RC_DIR=os.path.join(fx.tmp, "envrc"))
    check(got == 81 and so == b"" and se == b"boxq: j1 cancelled elsewhere\n" and fx.pids(fx.follow_pids) == [],
          f"a cancelled job that never wrote a file: 81 and one line: {got} {so!r} {se!r}")


def case_mirror(fx):
    path = os.path.join(fx.rc_dir, "j1.rc")
    os.makedirs(fx.rc_dir, exist_ok=True)
    with open(path, "w") as f:
        f.write("9\n")
    seen, real = [], os.replace

    def spy(src, dst):
        with open(src) as f:
            seen.append((f.read(), open(dst).read()))
        real(src, dst)

    os.replace = spy
    try:
        mirror(fx.rc_dir, "j1", 5)
    finally:
        os.replace = real
    check(seen == [("5\n", "9\n")], f"the new rc is whole in a temp file while the old stands, then renamed: {seen}")
    check(open(path).read() == "5\n" and os.listdir(fx.rc_dir) == ["j1.rc"],
          f"the mirror after: {os.listdir(fx.rc_dir)}")
    blocked = os.path.join(path, "x")
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        mirror(blocked, "j2", 7)
    check("was not written" in err.getvalue(), f"a mirror that cannot be written is a stderr line: {err.getvalue()!r}")


def case_config(fx):
    q = Queue({})
    got = (q.host, q.ssh, q.url, q.poll_s, q.note_s, q.cancel_wait_s, q.reconnect_s)
    check(got == ("ws", ["ssh"], "http://127.0.0.1:8030", 5.0, 60.0, 150.0, 900.0), f"the defaults: {got}")
    q = Queue({"BLOOMERY_BOX": "bx", "BOXQ_SSH": "ssh -o BatchMode=yes", "BOXQ_URL": "http://h:1/", "BOXQ_POLL_S": "2"})
    check((q.host, q.ssh, q.url, q.poll_s) == ("bx", ["ssh", "-o", "BatchMode=yes"], "http://h:1", 2.0), "the env's values")
    check(Queue({"BOXQ_HOST": "h", "BLOOMERY_BOX": "bx"}).host == "h", "BOXQ_HOST beats BLOOMERY_BOX")
    for name, bad in (("BOXQ_POLL_S", "0"), ("BOXQ_POLL_S", "x"), ("BOXQ_POLL_S", "nan"),
                      ("BOXQ_CANCEL_WAIT_S", "-1"), ("BOXQ_NOTE_S", "-1"), ("BOXQ_RECONNECT_S", "-1")):
        try:
            Queue({name: bad})
        except Refusal as r:
            check(r.code == USAGE and name in r.msg, f"{name}={bad} is a usage error naming it: {r.code} {r.msg}")
        else:
            raise AssertionError(f"{name}={bad} was taken")


def case_cancel_cmd(fx):
    fake = fx.serve([{"status": "running"}])
    rc, out, err = fx.client("cancel", "j1")
    check(rc == 0 and out == b"cancelled\n" and fake.requests[-1] == ("POST", "/jobs/j1/cancel", b"{}", "application/json"),
          f"cancel posts {{}} and prints the status: {rc} {out!r} {err!r} {fake.requests}")
    rc, out, err = fx.client("cancel", "j9")
    check(rc == 64, f"cancel of an unknown job is 64: {rc} {err!r}")


def case_tails_stop(fx):
    fx.files(b"x\n", b"y\n")
    q, job, null = Queue(fx.env()), {"id": "j1", "out": fx.out, "err": fx.err}, os.open(os.devnull, os.O_WRONLY)
    kids = follow(q, job, (null, null))
    fx.wait_pids(fx.tail_pids, 2)
    unfollow(kids)
    check(not kids, "unfollow empties the list")
    fx.gone(fx.tail_pids, 2)
    fx.script(os.path.join(fx.tmp, "hung"), f"echo $$ >> {shlex.quote(fx.hung_pids)}\nexec sleep 60\n")
    q.ssh = [os.path.join(fx.tmp, "hung")]
    kids = follow(q, job, (null, null))
    fx.wait_pids(fx.hung_pids, 2)
    unfollow(kids, grace=0.3)
    fx.gone(fx.hung_pids, 2)
    os.close(null)


def case_list(fx):
    argv = "flock q bash -c '[ x = 0123 ] || exit 1\nexport M=1 && bash tools/gpu-gate.sh g'"
    fake = fx.serve([{"status": "running", "argv": argv}])
    rc, out, err = fx.client("list", "--match", "export M=1 && ", "--match", "= 0123 ]")
    check(rc == 0 and out == b"j1\n", f"a running job whose command holds every text is listed: {rc} {out!r} {err!r}")
    rc, out, err = fx.client("list", "--match", "export M=2 && ", "--match", "= 0123 ]")
    check(rc == 0 and out == b"", f"one text it does not hold: not listed: {rc} {out!r} {err!r}")
    n = len(fake.requests)
    rc, out, err = fx.client("list")
    check(rc == 0 and out == b"j1\n" and [r[1] for r in fake.requests[n:]] == ["/jobs"],
          f"with no --match every live job is listed and no command is read: {rc} {out!r} {fake.requests[n:]}")
    fx.serve([{"status": "running", "argv": argv}], listing=[{"id": "j0", "status": "done"}, {"id": "j1", "status": "failed"},
                                                          {"id": "../x", "status": "running"}])
    rc, out, err = fx.client("list", "--match", "export M=1 && ")
    check(rc == 0 and out == b"" and b"a row off the wire is skipped" in err and b"../x" in err,
          f"finished jobs are not listed, and an id off the wire is named and skipped: {rc} {out!r} {err!r}")
    fx.serve([{"status": "running", "argv": argv}], listing=[{"id": "j1\n", "status": "running"}, {"id": "j9", "status": "running"}])
    rc, out, err = fx.client("list", "--match", "export M=1 && ")
    check(rc == 0 and out == b"" and b"skipped" in err,
          f"an id with a trailing newline is off the wire, and a job the daemon no longer knows is skipped: {rc} {out!r} {err!r}")
    fx.serve([{"status": "done", "rc": 0, "argv": argv}], listing=[{"id": "j1", "status": "running"}])
    rc, out, err = fx.client("list", "--match", "export M=1 && ")
    check(rc == 0 and out == b"", f"a job that ended between the list and its read is not listed: {rc} {out!r}")
    fx.serve([{"status": "running", "argv": argv}], listing=[{"id": "j1", "status": "parked"}])
    rc, out, err = fx.client("list", "--match", "export M=1 && ")
    check(rc == 0 and out == b"" and b"a row off the wire is skipped" in err and b"parked" in err,
          f"a status the daemon does not have is named and skipped: {rc} {out!r} {err!r}")
    fx.serve([{"status": "running", "argv": None}])
    rc, out, err = fx.client("list", "--match", "export M=1 && ")
    check(rc == 84 and out == b"" and b"argv is not a string" in err,
          f"a live job whose command is not a string ends the list 84, never an empty list: {rc} {out!r} {err!r}")
    fx.serve([{"status": "running", "argv": argv}], job_reply=(403, {"error": "not yours", "rc": 65}))
    rc, out, err = fx.client("list", "--match", "export M=1 && ")
    check(rc == 65 and out == b"" and b"not yours" in err,
          f"a refusal other than 404 ends the list with its code, never an empty list: {rc} {out!r} {err!r}")
    rc, out, err = fx.client("list", BOXQ_URL="http://127.0.0.1:9")
    check(rc == 84 and b"GET /jobs" in err, f"a daemon that does not answer: 84 naming the list: {rc} {err!r}")


def case_attach_leave(fx):
    fx.files(b"o\n", b"e\n")
    for sig, want in ((signal.SIGINT, 130), (signal.SIGTERM, 143)):
        fake = fx.serve([{"status": "running"}])
        p = fx.start("attach", "j1", "--leave", "--rc-dir", fx.rc_dir)
        fx.wait_pids(fx.tail_pids, 2)
        p.send_signal(sig)
        got, so, se = fx.finish(p)
        check(got == want and fake.cancels() == [] and b"leaving the job as it is" in se,
              f"{sig.name} with --leave exits {want} and posts no cancel: {got} {se!r} {fake.cancels()}")
        check(fx.rc_file() is None, "a job left running gets no rc mirror")
        fx.gone(fx.tail_pids, 2)
    for end, want in (({"status": "failed", "rc": 3}, 3), ({"status": "done", "rc": 0}, 0),
                      ({"status": "cancelled", "rc": 143}, 81), ({"status": "lost", "rc": None}, 81)):
        # the end is answered only once both tails recorded their pids, so a slow host cannot stop one before it did
        fx.serve([{"status": "running"}, end], gate=lambda: len(fx.pids(fx.tail_pids)) >= 2)
        got, so, se = fx.client("attach", "j1", "--leave", "--rc-dir", fx.rc_dir)
        check(got == want and fx.rc_file() is None,
              f"--leave on a job that ends {end}: {want}, no mirror: {got} {se!r} {fx.rc_file()!r}")
        fx.gone(fx.tail_pids, 2)
    fx.serve([{"status": "running"}])
    got, so, se = fx.client("attach", "j9", "--leave", "--rc-dir", fx.rc_dir)
    check(got == 84 and fx.rc_file("j9") is None and se.count(b"poll") >= 2,
          f"--leave on a job the daemon no longer knows: three missed polls, 84, no mirror: {got} {se!r}")
    got, so, se = fx.client("attach", "j9", "--rc-dir", fx.rc_dir)
    check(got == 64, f"without --leave an unknown job stays a usage error: {got} {se!r}")


CASES = [case_list, case_attach_leave, case_submit_body, case_submit_no_prediction, case_run_done, case_lost, case_health_down, case_reconnect,
         case_refusal,
         case_passthrough, case_queued_line, case_int_cancel, case_attach_env, case_mirror, case_config, case_cancel_cmd,
         case_tails_stop, case_flush_cap, case_signal_then_done, case_drain_closed_reader, case_drain_full_pipe,
         case_pending_replay, case_stalled_tail, case_replay_ssh, case_replay_signal, case_reconnect_wall, case_reconnect_step,
         case_cancel_deadline, case_queued_age, case_cancelled_elsewhere, case_load_ready]


def run_cases(label, cases, make_fix, names):
    """Run each case, or only the ones `names` lists, each on its own fixture (make_fix of a fresh temp dir, cleaned up
    after the case); the exit code. tools/boxq-check.py runs its cases through this."""
    known = {c.__name__[5:].replace("_", "-"): c for c in cases}
    unknown = [n for n in names if n not in known]
    if unknown:
        raise Refusal(USAGE, "usage", f"no such case: {', '.join(unknown)}")
    failed = []
    for name, fn in known.items():
        if names and name not in names:
            continue
        with tempfile.TemporaryDirectory(prefix=f"{label}-selftest-") as tmp:
            fx = make_fix(tmp)
            try:
                fn(fx)
                print(f"ok   {name}")
            except Exception as e:  # a case's red line is its message; a traceback only for a non-assertion
                failed.append(name)
                print(f"FAIL {name}: {e if isinstance(e, AssertionError) else repr(e)}")
            finally:
                fx.cleanup()
    if failed:
        print(f"{label} self-test: {len(failed)} red: {', '.join(failed)}")
        return 1
    print(f"{label} self-test: {len(names) or len(known)} cases ok")
    return 0


def self_test(names):
    return run_cases("boxq", CASES, Fix, names)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
