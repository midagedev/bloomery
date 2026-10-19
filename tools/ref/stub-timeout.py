#!/usr/bin/env python3
"""A stand-in timeout for the self-tests of tools/gpu-gate.sh, tools/ref/stack-watch.sh, tools/ref/load-groups.sh,
tools/ref/lease.sh and tools/ref/card-tests/run.sh where the host has none (the Mac), copied in as `timeout`. It takes
coreutils' `timeout [--kill-after=K] S COMMAND...` and keeps its timing and exit codes: S 0 is no bound; otherwise at
S seconds COMMAND's process group gets a TERM, as it does when a TERM is sent here. K seconds after the first of the
two the group gets a KILL if COMMAND is still there (K 0 or none: no KILL). The exit is 124 when S ran out, 137 when
KILL ends the group, else COMMAND's status (128 + the signal). An ALRM sent here is S running out, as coreutils' timeout
takes it. The monitor leads the group and ignores its own TERM while forwarding it. COMMAND inherits the monitor's open descriptors, including a lease held on descriptor 9."""
import os
import signal
import subprocess
import sys
import time

kill_after = 0.0
args = []
for a in sys.argv[1:]:
    if a.startswith('--kill-after='):
        kill_after = float(a.split('=', 1)[1])
    else:
        args.append(a)
deadline = time.monotonic() + float(args[0]) if float(args[0]) > 0 else float('inf')
term_signal = signal.SIGTERM
timed_out = False
proc = None
pending = False


def cleanup(sig):
    global deadline, kill_after, term_signal
    if kill_after:
        term_signal, deadline, kill_after = signal.SIGKILL, time.monotonic() + kill_after, 0
    handler = signal.signal(signal.SIGTERM, signal.SIG_IGN)
    try:
        os.killpg(os.getpid(), sig)
    finally:
        signal.signal(signal.SIGTERM, handler)


def on_term(*_):
    global pending
    if proc is None:
        pending = True
    else:
        cleanup(signal.SIGTERM)


# Set before COMMAND starts, so a TERM that comes as it starts reaches its group.
os.setpgid(0, 0)
signal.signal(signal.SIGTERM, on_term)


def on_alrm(*_):
    global deadline
    deadline = 0.0


signal.signal(signal.SIGALRM, on_alrm)
proc = subprocess.Popen(args[1:], close_fds=False)
if pending:
    cleanup(signal.SIGTERM)
# Short waits, so a deadline a TERM moves takes effect.
while True:
    try:
        rc = proc.wait(timeout=0.1)
        break
    except subprocess.TimeoutExpired:
        if time.monotonic() >= deadline:
            timed_out, deadline = True, float('inf')
            cleanup(term_signal)
if timed_out and rc == -signal.SIGKILL:
    sys.exit(137)
sys.exit(124 if timed_out else 128 - rc if rc < 0 else rc)
