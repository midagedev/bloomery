#!/usr/bin/env bash
# Fixtures shared by the cancellation self-tests. Sourced, never executed.

# CANCEL_TIMEOUT_FORWARD=1 forwards TERM once; unset, the monitor exits and leaves its child's group. SIGALRM is the
# bound running out, as coreutils' timeout takes it from its own timer: the child is killed and the exit is 124, so a
# case can end the bound at the moment its unit is ready instead of at a time after the spawn.
cancel_timeout() {
  cat > "$1" << 'PY'
#!/usr/bin/env python3
import os
import signal
import subprocess
import sys
import time

child = None
pending_term = False
alarmed = False


def stop(*_):
    global pending_term
    if os.environ.get("CANCEL_TIMEOUT_FORWARD") == "1":
        if child is None:
            pending_term = True
        else:
            child.send_signal(signal.SIGTERM)
    else:
        sys.exit(143)


def alarm(*_):
    global alarmed
    alarmed = True


os.setpgid(0, 0)
signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGALRM, alarm)
args = [a for a in sys.argv[1:] if not a.startswith('--kill-after=')]
child = subprocess.Popen(args[1:], close_fds=False)
if pending_term:
    child.send_signal(signal.SIGTERM)
deadline = time.monotonic() + float(args[0])
while True:
    try:
        rc = child.wait(timeout=0.05)
        break
    except subprocess.TimeoutExpired:
        if alarmed or time.monotonic() >= deadline:
            child.kill()
            child.wait()
            rc = 124
            break
sys.exit(128 - rc if rc < 0 else rc)
PY
}

# The child's PID file precedes the monitor's return from fork; sleep marks its wait for the child.
cancel_ready() {
  local sp ip state
  [ -s "$1" ] && read -r sp ip < "$1" || return 1
  state=$(ps -o stat= -p "$ip" 2> /dev/null) || return 1
  case $state in *S*) return 0 ;; esac
  return 1
}

# cancel_alive <pid>: running, not a zombie; a pid ps cannot read is gone.
cancel_alive() { local st; st=$(ps -o stat= -p "$1" 2> /dev/null) || return 1; case $st in '' | *Z*) return 1 ;; esac; }
