#!/usr/bin/env python3
"""A stand-in timeout for the self-tests of tools/gpu-gate.sh, tools/ref/stack-watch.sh, tools/ref/load-groups.sh and
tools/ref/lease.sh where the host has none (the Mac), copied in as `timeout`. It takes coreutils'
`timeout [--kill-after=K] S COMMAND...` and keeps its timing and exit codes: at S seconds COMMAND gets a TERM, and so does
it when a TERM is sent here; K seconds after the first of the two it gets a KILL if it is still there (K 0 or none: no
KILL). The exit is 124 when S ran out, 137 when the KILL ended COMMAND, else COMMAND's status (128 + the signal). Unlike
coreutils it makes no process group: only COMMAND is signalled."""
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
deadline = time.monotonic() + float(args[0])
term_signal = signal.SIGTERM
timed_out = False
proc = None
pending = False


def cleanup(sig):
    global deadline, kill_after, term_signal
    if kill_after:
        term_signal, deadline, kill_after = signal.SIGKILL, time.monotonic() + kill_after, 0
    proc.send_signal(sig)


def on_term(*_):
    global pending
    if proc is None:
        pending = True
    else:
        cleanup(signal.SIGTERM)


# Set before COMMAND starts, so a TERM that comes as it starts reaches it.
signal.signal(signal.SIGTERM, on_term)
proc = subprocess.Popen(args[1:])
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
