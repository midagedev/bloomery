#!/usr/bin/env python3
"""One role of the review bot on the first engine that finishes it (.github/workflows/review.yml, review-learn.yml).

    tools/review/run.py reviewer|validator|learner BASE_TREE PR_TREE CTX OUT TRACE
    tools/review/run.py --self-test

The engines are tools/review/run-claude.sh and run-codex.sh, which take the same arguments. REVIEW_ENGINE (claude,
the default, or codex) names the one tried first; the other is the fallback. An engine is tried only when its login is
there (ANTHROPIC_API_KEY for claude, CODEX_HOME/auth.json for codex, which the workflow writes from a secret) and its
CLI is on PATH.

An attempt fails when the run exits non-zero (run-codex.sh exits 69 when its sandbox check fails), overruns the
role's MINUTES (its process group is stopped), its usage (tools/review/usage.py) is an error, or OUT lacks what this
run of the role must leave: review.md with a verdict line and decision.txt for the reviewer; a new validator line in
review.md for the validator (removed.json, when there, a JSON array); a candidates.json array for the learner. A failed
attempt's OUT is put back as it was, its trace is kept as TRACE/<role>.<engine>.failed.*, the engine is not tried
again in this job (TRACE/engines.json), and the next engine runs. A malformed review that still has its verdict line is
not a failure here: `post.py check` refuses it after the validator.

An engine writes its transcript, usage and output (stdout and stderr) to a private directory outside TRACE. When the
run ends, by any path but SIGKILL, run.py replaces every credential in those files and only then copies them into
TRACE and the output to stderr: the value of ANTHROPIC_API_KEY, every long string of CODEX_HOME/auth.json before and
after the run (Codex rewrites the file when it refreshes its tokens), and token shapes. A run killed outright leaves
nothing in TRACE. OUT is checked after every run: a credential in it, or anything there but regular files and
directories (a symbolic link could pull a file from outside into the next copy of OUT), ends the role with exit 3, OUT
emptied, no fallback, and nothing posted.

Prints one line per attempt and one for the outcome. Exit 0 when an engine finished the role (TRACE/<role>.engine
names it), 3 as above, 75 when no engine could: the workflow then ends without a review, with a warning.
"""

import contextlib
import io
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from post import TOKEN_RE, VALIDATOR_LINE, VERDICT_RE, secret_file_values, secrets_in, unsafe_entries  # noqa: E402

ENGINES = ("claude", "codex")
NO_ENGINE = 75
SANDBOX_CHECK_FAILED = 69
TIMEOUT = 124
# Per engine run; a role that falls back can take twice this. The workflows' timeout-minutes hold both.
MINUTES = {"reviewer": 20, "validator": 10, "learner": 15}
REASONS = (
    ("limit", re.compile(r"usage limit|rate.?limit|quota|\b429\b|credit balance|too many requests|at capacity"
                         r"|overloaded", re.I)),
    ("auth", re.compile(r"\b40[13]\b|unauthori[sz]ed|invalid.{0,20}(key|token)|refresh token|log ?in|expired", re.I)),
)


def order(preferred, available):
    first = preferred if preferred in ENGINES else "claude"
    return [e for e in (first, *[x for x in ENGINES if x != first]) if available.get(e) is None]


def available_engines(env):
    """{engine: None when it can run, else why not}."""
    home = env.get("CODEX_HOME") or os.path.expanduser("~/.codex")
    login = {"claude": bool(env.get("ANTHROPIC_API_KEY")), "codex": os.path.isfile(os.path.join(home, "auth.json"))}
    return {e: None if login[e] and shutil.which(e, path=env.get("PATH")) else "no login" if not login[e]
            else "not installed" for e in ENGINES}


def read(path):
    try:
        return open(path, encoding="utf-8", errors="replace").read()
    except OSError:
        return None


def json_array(path):
    """None when path holds a JSON array, else what is wrong."""
    name, text = os.path.basename(path), read(path)
    if text is None:
        return f"no {name}"
    try:
        return None if isinstance(json.loads(text), list) else f"{name} is not a JSON array"
    except ValueError:
        return f"{name} is not JSON"


def fresh_array(out, before, name):
    """None when this run wrote OUT/name as a JSON array, else what is wrong."""
    path, old = os.path.join(out, name), os.path.join(before, name)
    if (os.path.isfile(path) and os.path.exists(old) and os.stat(old).st_mtime_ns == os.stat(path).st_mtime_ns
            and read(old) == read(path)):
        return f"{name} is not from this run"
    return json_array(path)


def produced(role, out, before):
    """None when OUT holds what this run of the role must leave, else what is missing."""
    if role == "learner":
        return fresh_array(out, before, "candidates.json")
    body = read(os.path.join(out, "review.md"))
    if body is None:
        return "no review.md"
    if not VERDICT_RE.search(body):
        return "review.md has no verdict line"
    if role == "reviewer":
        return None if os.path.isfile(os.path.join(out, "decision.txt")) else "no decision.txt"
    if body.count(VALIDATOR_LINE) <= (read(os.path.join(before, "review.md")) or "").count(VALIDATOR_LINE):
        return "no validator line from this run in review.md"
    removed = os.path.join(out, "removed.json")
    return json_array(removed) if os.path.exists(removed) else None


def failure(engine, role, rc, usage, out, before, log):
    """None for a finished attempt, else a short reason. usage and log are already free of credentials."""
    err = (usage or {}).get("error") or ""
    if rc == 0 and usage and not usage.get("is_error"):
        missing = produced(role, out, before)
        return f"no output: {missing}" if missing else None
    if engine == "codex" and rc == SANDBOX_CHECK_FAILED:
        m = re.search(r"sandbox check failed: (.*)", log)
        return "sandbox check failed" + (f": {m.group(1)[:120]}" if m else "")
    if rc in (TIMEOUT, 137):
        return "timeout"
    for name, pat in REASONS:
        if pat.search(err):
            return f"{name}: {err[:120]}"
    return f"error rc={rc}" + (f": {err[:120]}" if err else "")


def credentials(env):
    vals = [("env:ANTHROPIC_API_KEY", env.get("ANTHROPIC_API_KEY", ""))]
    home = env.get("CODEX_HOME") or os.path.expanduser("~/.codex")
    return vals + secret_file_values(os.path.join(home, "auth.json"))


def texts(d):
    found = {}
    for root, _, files in os.walk(d):
        for f in files:
            p = os.path.join(root, f)
            found[p] = read(p) or ""
    return found


def redact(text, values):
    for _, v in values:
        if v and len(v) >= 8:
            text = text.replace(v, "[credential removed]")
    return TOKEN_RE.sub("[credential removed]", text)


def publish(raw, trace, role, values):
    """Copy the private run files into TRACE with every credential replaced; the run's output also goes to stderr."""
    for name in sorted(os.listdir(raw)):
        text = redact(open(os.path.join(raw, name), encoding="utf-8", errors="replace").read(), values)
        open(os.path.join(trace, name), "w", encoding="utf-8").write(text)
        if name == f"{role}.log" and text:
            sys.stderr.write(text if text.endswith("\n") else text + "\n")


def launch(engine, role, paths, env):
    """Run one engine's script in its own process group, its output in <role>.log of the TRACE it is given; the other
    engine's credential stays out of its environment. The group is stopped when the run overruns MINUTES[role] or
    run.py is stopped."""
    script = os.path.join(paths[0], "tools", "review", f"run-{engine}.sh")
    child = dict(env)
    if engine == "codex":
        child.pop("ANTHROPIC_API_KEY", None)
    with open(os.path.join(paths[4], f"{role}.log"), "wb") as log:
        p = subprocess.Popen(["bash", script, role, *paths], env=child, stdin=subprocess.DEVNULL, stdout=log,
                             stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return p.wait(timeout=MINUTES[role] * 60)
        except subprocess.TimeoutExpired:
            return TIMEOUT
        finally:
            for sig, wait in ((signal.SIGTERM, 30), (signal.SIGKILL, None)):
                try:
                    os.killpg(p.pid, sig)
                except (ProcessLookupError, PermissionError):  # macOS: EPERM when only zombies are left in the group
                    break
                try:
                    p.wait(timeout=wait)
                except subprocess.TimeoutExpired:
                    pass


def empty(out):
    shutil.rmtree(out)
    os.mkdir(out)


def run_role(role, paths, env, run=launch, say=print):
    out, trace = paths[3], paths[4]
    state_path = os.path.join(trace, "engines.json")
    try:
        dead = json.load(open(state_path, encoding="utf-8"))
    except (OSError, ValueError):
        dead = {}
    avail = available_engines(env)
    tried = [e for e in order(env.get("REVIEW_ENGINE", "claude"), avail) if e not in dead]
    for e in ENGINES:
        if avail[e]:
            say(f"{role}: {e} not tried ({avail[e]})")
        elif e in dead:
            say(f"{role}: {e} not tried (failed earlier in this job: {dead[e]})")
    with tempfile.TemporaryDirectory() as snap:
        before = os.path.join(snap, "out")
        for engine in tried:
            bad = unsafe_entries(out)
            if bad:
                say(f"{role}: OUT holds what is not a regular file ({', '.join(bad[:5])}); nothing is posted")
                empty(out)
                return 3
            if not os.path.exists(before):
                shutil.copytree(out, before, symlinks=True)
            raw = os.path.join(snap, engine)
            os.mkdir(raw)
            secrets = credentials(env)
            try:
                rc = run(engine, role, [*paths[:4], raw], env)
            finally:
                publish(raw, trace, role, secrets + credentials(env))
            secrets += credentials(env)
            try:
                usage = json.load(open(os.path.join(trace, f"{role}.usage.json"), encoding="utf-8"))
            except (OSError, ValueError):
                usage = None
            bad = unsafe_entries(out)
            leaked = [] if bad else secrets_in(list(texts(out).values()), secrets)
            if bad or leaked:
                say(f"{role}: {engine} left in OUT " + (f"what is not a regular file ({', '.join(bad[:5])})"
                                                        if bad else f"a credential ({', '.join(leaked)})")
                    + "; nothing is posted")
                empty(out)
                return 3
            why = failure(engine, role, rc, usage, out, before, read(os.path.join(trace, f"{role}.log")) or "")
            if why is None:
                open(os.path.join(trace, f"{role}.engine"), "w", encoding="utf-8").write(engine + "\n")
                say(f"{role}: done by {engine}" + (f" after {', '.join(tried[:tried.index(engine)])} failed"
                                                   if engine != tried[0] else ""))
                return 0
            dead[engine] = why = redact(why, secrets)
            say(f"{role}: {engine} failed ({why})")
            json.dump(dead, open(state_path, "w", encoding="utf-8"))
            for ext in ("ndjson", "usage.json", "log"):
                p = os.path.join(trace, f"{role}.{ext}")
                if os.path.exists(p):
                    os.replace(p, os.path.join(trace, f"{role}.{engine}.failed.{ext}"))
            shutil.rmtree(out)
            shutil.copytree(before, out, symlinks=True)
    say(f"{role}: no engine finished; no review this run")
    return NO_ENGINE


def self_test():
    with tempfile.TemporaryDirectory() as d:
        paths = [os.path.join(d, n) for n in ("base", "pr", "ctx", "out", "trace", "home", "bin")]
        for p in paths:
            os.mkdir(p)
        paths, home, bindir = paths[:5], paths[5], paths[6]
        out, trace = paths[3], paths[4]
        for e in ENGINES:
            open(os.path.join(bindir, e), "w").write("#!/bin/sh\n")
            os.chmod(os.path.join(bindir, e), 0o755)
        auth = os.path.join(home, "auth.json")
        open(auth, "w").write('{"tokens": {"access_token": "fakefakefakefakefake"}}')
        env = {"ANTHROPIC_API_KEY": "sk-test-key-value", "CODEX_HOME": home, "REVIEW_ENGINE": "codex", "PATH": bindir}
        calls = []

        def fake(results):
            """An engine stand-in: results[engine] is (rc, usage, files written to OUT[, log])."""
            def run(engine, role, paths_, env_):
                calls.append((engine, role))
                rc, usage, files, *log = results[engine]
                for name, text in files.items():
                    if callable(text):
                        text(os.path.join(out, name))
                    else:
                        open(os.path.join(out, name), "w").write(text)
                open(os.path.join(paths_[4], f"{role}.ndjson"), "w").write('{"t": "fakefakefakefakefake"}\n')
                open(os.path.join(paths_[4], f"{role}.log"), "w").write("".join(log))
                json.dump(usage, open(os.path.join(paths_[4], f"{role}.usage.json"), "w"))
                return rc
            return run

        def reset():
            for p in os.listdir(trace):
                os.remove(os.path.join(trace, p))
            calls.clear()
            lines.clear()

        ok = {"is_error": False}
        good = {"review.md": "**Verdict: CLEAN**: fine.\n", "decision.txt": "CLEAN\n"}
        ok_c = (0, {"is_error": False, "engine": "claude"}, good)
        limit = (1, {"is_error": True, "error": "You've hit your usage limit."}, {"partial.md": "x"})
        lines = []

        # Codex first; it hits its limit, OUT goes back, Claude reviews; the trace has no credential; the validator
        # skips Codex. A validator fails when it adds no line of its own or leaves a removed.json that is not an
        # array; a new line alone is enough.
        assert run_role("reviewer", paths, env, fake({"codex": limit, "claude": ok_c}), lines.append) == 0
        assert calls == [("codex", "reviewer"), ("claude", "reviewer")], calls
        assert not os.path.exists(os.path.join(out, "partial.md")) and "limit" in lines[0], lines
        assert lines[-1] == "reviewer: done by claude after codex failed", lines
        assert "fakefake" not in open(os.path.join(trace, "reviewer.codex.failed.ndjson")).read()
        assert open(os.path.join(trace, "reviewer.engine")).read() == "claude\n"
        calls.clear()
        val = {"review.md": good["review.md"] + VALIDATOR_LINE + " removed=0 -->\n"}
        assert run_role("validator", paths, env, fake({"claude": (0, ok, {})}), lines.append) == NO_ENGINE
        assert calls == [("claude", "validator")] and "no validator line" in lines[-2], (calls, lines)
        os.remove(os.path.join(trace, "engines.json"))
        bad = dict(val, **{"removed.json": "{}"})
        assert run_role("validator", paths, dict(env, REVIEW_ENGINE="claude"),
                        fake({"claude": (0, ok, bad), "codex": limit}), lines.append) == NO_ENGINE
        assert "claude failed (no output: removed.json is not a JSON array)" in lines[-3], lines
        os.remove(os.path.join(trace, "engines.json"))
        assert run_role("validator", paths, dict(env, REVIEW_ENGINE="claude"),
                        fake({"claude": (0, ok, val)}), lines.append) == 0

        # Both fail (Codex's sandbox check, Claude's login): 75, OUT as it was, the reasons without the token.
        reset()
        os.remove(os.path.join(out, "decision.txt"))
        before = sorted(os.listdir(out))
        unsafe = (69, None, {}, "run-codex.sh: sandbox check failed: the network was open\n")
        auth_err = (1, {"is_error": True, "error": "x" * 95 + " Invalid API key fakefakefakefakefake"}, {})
        with contextlib.redirect_stderr(io.StringIO()):
            assert run_role("reviewer", paths, env, fake({"codex": unsafe, "claude": auth_err}), lines.append) == 75
        assert sorted(os.listdir(out)) == before, os.listdir(out)
        assert "codex failed (sandbox check failed: the network was open)" in lines[0], lines
        assert "claude failed (auth" in lines[1] and "fake" not in "".join(lines), lines

        # No Codex CLI: only Claude runs. A learner that leaves no candidates.json fails, as does one that leaves the
        # previous run's; an empty array is fine.
        reset()
        os.remove(os.path.join(bindir, "codex"))
        assert run_role("learner", paths, env, fake({"claude": (0, ok, {})}), lines.append) == NO_ENGINE
        assert lines[0] == "learner: codex not tried (not installed)" and "no candidates.json" in lines[1], lines
        reset()
        learned = (0, ok, {"candidates.json": "[]"})
        assert run_role("learner", paths, env, fake({"claude": learned}), lines.append) == 0
        reset()
        assert run_role("learner", paths, env, fake({"claude": (0, ok, {})}), lines.append) == NO_ENGINE
        assert "candidates.json is not from this run" in lines[1], lines

        # A link in OUT (here to the login's directory): exit 3, OUT emptied, the login untouched, no fallback.
        reset()
        link = (0, ok, {**good, "cache": lambda p: os.symlink(home, p)})
        assert run_role("reviewer", paths, dict(env, REVIEW_ENGINE="claude"),
                        fake({"claude": link}), lines.append) == 3 and calls == [("claude", "reviewer")]
        assert os.listdir(out) == [] and os.path.exists(auth) and "(cache)" in lines[-1], lines

        # A credential in OUT: exit 3, OUT emptied.
        reset()
        leak = (0, ok, {**good, "review.md": "**Verdict: CLEAN**: fakefakefakefakefake\n"})
        assert run_role("reviewer", paths, env, fake({"claude": leak}), lines.append) == 3
        assert os.listdir(out) == []

        # Stopped mid-run: the trace still gets the redacted copy, never the raw one.
        reset()

        def stopped(engine, role, paths_, env_):
            fake({"claude": ok_c})(engine, role, paths_, env_)
            raise KeyboardInterrupt
        try:
            run_role("reviewer", paths, dict(env, REVIEW_ENGINE="claude"), stopped, lines.append)
            raise AssertionError("the interrupt was swallowed")
        except KeyboardInterrupt:
            assert "fakefake" not in open(os.path.join(trace, "reviewer.ndjson")).read()

        # The time limit stops the whole process group, a grandchild included. The pid file can still be empty
        # right after launch returns, and a killed grandchild can linger as a zombie until it is reaped: wait
        # for a whole pid, then count only a live process as outliving the limit.
        os.makedirs(os.path.join(paths[0], "tools", "review"))
        pid = os.path.join(d, "pid")
        script = os.path.join(paths[0], "tools", "review", "run-claude.sh")
        open(script, "w").write(f"sleep 60 & echo $! > {pid}; wait\n")
        MINUTES["reviewer"] = 0.02
        assert launch("claude", "reviewer", paths, {"PATH": os.environ.get("PATH", "")}) == TIMEOUT
        for _ in range(50):
            text = read(pid)
            if text and text.endswith("\n") and text.strip().isdigit():
                grandchild = int(text)
                break
            time.sleep(0.1)
        else:
            raise AssertionError("the grandchild's pid never appeared")
        for _ in range(50):
            try:
                os.kill(grandchild, 0)
            except ProcessLookupError:
                break
            if subprocess.run(["ps", "-o", "stat=", "-p", str(grandchild)], capture_output=True,
                              text=True).stdout.startswith("Z"):
                break
            time.sleep(0.1)
        else:
            raise AssertionError("the grandchild outlived the time limit")
    assert order("codex", {"claude": None, "codex": None}) == ["codex", "claude"]
    assert order("gpt", {"claude": None, "codex": "no login"}) == ["claude"]
    print("run.py self-test: ok")


def main(argv):
    if argv[1:] == ["--self-test"]:
        self_test()
        return 0
    if len(argv) != 7 or argv[1] not in ("reviewer", "validator", "learner"):
        print(__doc__, file=sys.stderr)
        return 64
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    return run_role(argv[1], argv[2:], dict(os.environ))


if __name__ == "__main__":
    sys.exit(main(sys.argv))
