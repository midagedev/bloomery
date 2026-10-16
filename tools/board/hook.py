#!/usr/bin/env python3
"""The board's event hooks: one line appended to ~/.cache/bloomery/board/events.jsonl per Claude Code hook call.

    hook.py msg | agent-start | agent-stop      # the hook JSON on stdin (tools/board/claude-hooks.json wires them)

Claude Code's documented events: PostToolUse (matcher SendMessage: tool_input.to, tool_input.message), SubagentStart
(agent_id, agent_type) and SubagentStop (agent_id, agent_type, last_assistant_message), each with the common session_id.
The docs give no agent name or model for the two subagent events, so a line holds the agent_type only.

A line is {"ts": epoch_ms, "kind": "msg" | "agent", ...}, one write of under 4 KiB with O_APPEND (atomic). A hook never
blocks and never fails its caller: every path ends exit 0 and an error goes to hook-errors.log in the same directory.
The log is rotated by board.py, never here. BOARD_EVENTS_DIR moves the directory; ~/.claude/sessions is the registry
that turns the hook's session_id into the session's name (the id itself is never stored).
"""

import json
import os
import re
import subprocess
import sys
import tempfile
import time

LINE_MAX = 4000
TEXT_MAX = 200
ROLE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,31}")


def board_dir():
    return os.environ.get("BOARD_EVENTS_DIR") or os.path.expanduser("~/.cache/bloomery/board")


def log_error(msg):
    try:
        os.makedirs(board_dir(), exist_ok=True)
        with open(os.path.join(board_dir(), "hook-errors.log"), "a", encoding="utf-8") as fh:
            fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}\n")
    except OSError:
        pass


def append(event):
    """One line, one write. The line is cut by shortening its text, never split."""
    event = {"ts": int(time.time() * 1000), **event}
    line = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
    while len(line.encode()) >= LINE_MAX and event.get("text"):
        event["text"] = event["text"][: len(event["text"]) // 2]
        line = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
    if len(line.encode()) >= LINE_MAX:
        raise ValueError("event line over 4 KiB")
    os.makedirs(board_dir(), exist_ok=True)
    fd = os.open(os.path.join(board_dir(), "events.jsonl"), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(fd, (line + "\n").encode())
    finally:
        os.close(fd)


def session_name(session_id, sessions=None):
    """The registry's name for this session id (a trailing ` (2)` dropped), else for the nearest ancestor pid that has a
    <pid>.json, else the id's first 8 characters (an id is the only thing left to say, and the board never renders it)."""
    sessions = sessions or os.path.expanduser("~/.claude/sessions")
    by_pid = {}
    try:
        for n in os.listdir(sessions):
            if not re.fullmatch(r"\d+\.json", n):
                continue
            try:
                with open(os.path.join(sessions, n), encoding="utf-8") as fh:
                    j = json.load(fh)
            except (OSError, ValueError):
                continue
            name = re.sub(r"\s*\(\d+\)$", "", str(j.get("name", "")))
            if session_id and j.get("sessionId") == session_id and name:
                return name
            by_pid[n[:-5]] = name
    except OSError:
        pass
    pid = os.getppid()
    for _ in range(6):
        if by_pid.get(str(pid)):
            return by_pid[str(pid)]
        try:
            out = subprocess.run(["ps", "-o", "ppid=", "-p", str(pid)], capture_output=True, text=True, timeout=2).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            break
        if not out.isdigit() or int(out) <= 1:
            break
        pid = int(out)
    return "s:" + str(session_id or "?")[:8]


def first_line(s, n=TEXT_MAX):
    for ln in str(s or "").splitlines():
        if ln.strip():
            return " ".join(ln.split())[:n]
    return ""


def role_of(to):
    """The target as a role name only: a socket address, a path or anything else that is no plain name is `other`."""
    to = str(to or "").strip()
    return to if ROLE.fullmatch(to) else "other"


def build(kind, j, sessions=None):
    """The event for one hook input, or None for an input that is no event of this kind."""
    if not isinstance(j, dict):
        raise ValueError("hook input is not a JSON object")
    sender = session_name(j.get("session_id"), sessions)
    if kind == "msg":
        if j.get("tool_name") != "SendMessage":
            return None
        ti = j.get("tool_input") if isinstance(j.get("tool_input"), dict) else {}
        msg = ti.get("message")
        if not isinstance(msg, str):  # a structured message (a protocol response) has no first line of prose
            msg = ti.get("summary", "")
        return {"kind": "msg", "from": sender, "to": role_of(ti.get("to")), "text": first_line(msg)}
    if kind in ("agent-start", "agent-stop"):
        ev = {"kind": "agent", "phase": kind[6:], "id": str(j.get("agent_id") or "")[:16], "type": first_line(j.get("agent_type"), 60),
              "parent": sender}
        for k in ("model", "name", "description"):
            if isinstance(j.get(k), str) and j[k]:
                ev[k] = first_line(j[k], 80)
        return ev
    raise ValueError(f"unknown kind {kind!r}")


def main(argv):
    try:
        if len(argv) != 1 or argv[0] not in ("msg", "agent-start", "agent-stop"):
            raise ValueError("usage: hook.py msg|agent-start|agent-stop")
        ev = build(argv[0], json.loads(sys.stdin.read() or "null"))
        if ev:
            append(ev)
    except Exception as e:  # a hook must never fail its caller
        log_error(f"hook.py {' '.join(argv)}: {type(e).__name__}: {e}")
    return 0


# ----------------------------------------------------------------------------------------------
# self-test
# ----------------------------------------------------------------------------------------------
def self_test():
    n = bad = 0

    def check(name, cond, detail=""):
        nonlocal n, bad
        n += 1
        if not cond:
            bad += 1
        print(("ok " if cond else "FAIL ") + name + ("" if cond else f": {detail}"))

    me = os.path.abspath(__file__)

    def call(env, kind, stdin):
        return subprocess.run([sys.executable, me, kind], input=stdin, capture_output=True, text=True, env=env, timeout=30)

    def lines(d):
        try:
            with open(os.path.join(d, "events.jsonl"), encoding="utf-8") as fh:
                return [json.loads(x) for x in fh if x.strip()]
        except OSError:
            return []

    with tempfile.TemporaryDirectory(prefix="hook-test-") as tmp:
        home, d = os.path.join(tmp, "home"), os.path.join(tmp, "board")
        sess = os.path.join(home, ".claude", "sessions")
        os.makedirs(sess)
        with open(os.path.join(sess, "777.json"), "w") as fh:
            json.dump({"pid": 777, "sessionId": "abcd1234-0000", "name": "quant (2)", "status": "busy"}, fh)
        env = dict(os.environ, HOME=home, BOARD_EVENTS_DIR=d)
        j = {"session_id": "abcd1234-0000", "hook_event_name": "PostToolUse", "tool_name": "SendMessage",
             "tool_input": {"to": "leader", "message": "done: x\nsecond line"}}
        r = call(env, "msg", json.dumps(j))
        ev = lines(d)
        check("msg: a SendMessage becomes one msg line with the registry's name, the role and the first line",
              r.returncode == 0 and len(ev) == 1 and ev[0].get("kind") == "msg" and ev[0].get("from") == "quant" and ev[0].get("to") == "leader"
              and ev[0].get("text") == "done: x" and isinstance(ev[0].get("ts"), int), (r.returncode, ev))
        check("msg: the session id is not stored", "abcd1234" not in open(os.path.join(d, "events.jsonl")).read(), "id stored")
        j["session_id"], j["tool_input"] = "feedface-1", {"to": "uds:/tmp/cc-socks/9.sock", "message": "y" * 500}
        call(env, "msg", json.dumps(j))
        ev = lines(d)[-1]
        check("msg: an unknown session is its id's first 8 characters, a socket target is `other`, the text is cut to 200",
              ev.get("from") == "s:feedface" and ev.get("to") == "other" and len(ev.get("text", "")) == 200, ev)
        j2 = dict(j, tool_name="Bash")
        before = len(lines(d))
        call(env, "msg", json.dumps(j2))
        check("msg: another tool writes nothing", len(lines(d)) == before, "wrote")
        a = {"session_id": "abcd1234-0000", "hook_event_name": "SubagentStart", "agent_id": "agent-1", "agent_type": "Explore"}
        call(env, "agent-start", json.dumps(a))
        call(env, "agent-stop", json.dumps(dict(a, hook_event_name="SubagentStop")))
        ev = lines(d)[-2:]
        check("agent: start and stop are agent lines with the type, the parent's name and one id",
              [e.get("phase") for e in ev] == ["start", "stop"] and all(e.get("kind") == "agent" and e.get("type") == "Explore"
              and e.get("parent") == "quant" and e.get("id") == "agent-1" for e in ev), ev)
        before = len(lines(d))
        for stdin in ("{not json", "", "[1,2]", "null"):
            r = call(env, "msg", stdin)
            check(f"garbage: stdin {stdin!r} exits 0, writes no event", r.returncode == 0 and len(lines(d)) == before, r.returncode)
        err = open(os.path.join(d, "hook-errors.log")).read() if os.path.exists(os.path.join(d, "hook-errors.log")) else ""
        check("garbage: each bad input is a line in hook-errors.log", err.count("\n") >= 3, err)
        r = call(env, "bogus", "{}")
        check("garbage: an unknown kind exits 0", r.returncode == 0, r.returncode)
        r = call(dict(env, BOARD_EVENTS_DIR="/dev/null/x"), "msg", json.dumps(j))
        check("garbage: an unwritable directory exits 0", r.returncode == 0, r.returncode)
        long = {"session_id": "z", "tool_name": "SendMessage", "tool_input": {"to": "leader", "message": "é" * 3000}}
        call(env, "msg", json.dumps(long))
        raw = open(os.path.join(d, "events.jsonl"), "rb").read().split(b"\n")[-2]
        check("size: a line stays under 4 KiB", len(raw) < LINE_MAX, len(raw))
    print(f"self-test: {'FAIL' if bad else 'ok'} ({n} cases, {bad} failed)")
    return 1 if bad else 0


if __name__ == "__main__":
    if "--self-test" in sys.argv[1:]:
        sys.exit(self_test())
    sys.exit(main(sys.argv[1:]))
