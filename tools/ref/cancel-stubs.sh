#!/usr/bin/env bash
# Fixtures shared by the lease and stack-watch cancellation self-tests. Sourced, never executed.

# CANCEL_TIMEOUT_FORWARD=1 forwards TERM once; unset, the monitor exits and leaves its child's group.
cancel_timeout() {
  cat > "$1" << 'PY'
#!/usr/bin/env python3
import os
import signal
import subprocess
import sys

child = None
pending_term = False


def stop(*_):
    global pending_term
    if os.environ.get("CANCEL_TIMEOUT_FORWARD") == "1":
        if child is None:
            pending_term = True
        else:
            child.send_signal(signal.SIGTERM)
    else:
        sys.exit(143)


os.setpgid(0, 0)
signal.signal(signal.SIGTERM, stop)
args = [a for a in sys.argv[1:] if not a.startswith('--kill-after=')]
child = subprocess.Popen(args[1:], close_fds=False)
if pending_term:
    child.send_signal(signal.SIGTERM)
try:
    rc = child.wait(timeout=float(args[0]))
except subprocess.TimeoutExpired:
    child.kill()
    child.wait()
    rc = 124
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
