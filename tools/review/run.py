#!/usr/bin/env python3
"""One role of the review bot on the first engine that finishes it (.github/workflows/review.yml, review-learn.yml).

    tools/review/run.py reviewer|validator|learner BASE_TREE PR_TREE CTX OUT TRACE
    tools/review/run.py --self-test

The engines are glm, claude and codex. glm and claude both run tools/review/run-claude.sh (headless Claude Code), glm on
GLM through z.ai's Anthropic-compatible endpoint; codex runs run-codex.sh; all take the same arguments. REVIEW_ENGINE is
an ordered, comma-separated list (`glm,codex,claude`; one word is a list of one): the engines it names are tried in that
order, then the ones it leaves out in the order glm, claude, codex. A word that is no engine is named and skipped. Unset,
the order is glm, claude, codex. An engine is tried only when its login is there (ZAI_API_KEY for glm, ANTHROPIC_API_KEY
for claude, CODEX_HOME/auth.json for codex, which the workflow writes from a secret) and its CLI is on PATH (claude for
glm and claude). Each engine's child process gets its own login only: glm gets the key as ANTHROPIC_AUTH_TOKEN with
ZAI_BASE_URL (default https://api.z.ai/api/anthropic) as ANTHROPIC_BASE_URL and REVIEW_MODEL_GLM (default glm-5.3) as
its model, claude gets neither a base URL, a token nor a cloud switch (CLAUDE_CODE_USE_*) of a runner's, and codex
neither key. REVIEW_RUN_ENGINE names the
engine to the script, which passes it to usage.py.

An attempt fails when the run exits non-zero (run-codex.sh exits 69 when its sandbox check fails), overruns the
role's MINUTES (its process group is stopped), its usage (tools/review/usage.py) is an error, or OUT lacks what this
run of the role must leave: review.md with a verdict line and decision.txt for the reviewer; a new validator line in
review.md for the validator (removed.json, when there, a JSON array); a candidates.json array for the learner. A failed
attempt's OUT is put back as it was, its trace is kept as TRACE/<role>.<engine>.failed.*, the engine is not tried
again in this job (TRACE/engines.json), and the next engine runs. A malformed review that still has its verdict line is
not a failure here: `post.py check` refuses it after the validator.

An engine writes its transcript, usage and output (stdout and stderr) to a private directory outside TRACE. When the
run ends, by any path but SIGKILL, run.py replaces every credential in those files and only then copies them into
TRACE and the output to stderr: the values of ANTHROPIC_API_KEY and ZAI_API_KEY, every long string of
CODEX_HOME/auth.json before and after the run (Codex rewrites the file when it refreshes its tokens), and token shapes. A run killed outright leaves
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

ENGINES = ("glm", "claude", "codex")
# The script run-<runner>.sh an engine runs, which is also the CLI it needs.
RUNNER = {"glm": "claude", "claude": "claude", "codex": "codex"}
# What can point Claude Code at an endpoint or a cloud other than the one its engine names.
ROUTES = ("ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX",
          "CLAUDE_CODE_USE_FOUNDRY")
GLM_BASE_URL = "https://api.z.ai/api/anthropic"
GLM_MODEL = "glm-5.3"
# Claude Code enforces 200000 for a model it does not know; glm-5.3 takes 1M.
GLM_CONTEXT_TOKENS = "1310720"
NO_ENGINE = 75
SANDBOX_CHECK_FAILED = 69
TIMEOUT = 124
# Per engine run; a role that falls back through every engine can take three times this. The workflows'
# timeout-minutes hold all three.
MINUTES = {"reviewer": 20, "validator": 10, "learner": 15}
REASONS = (
    ("limit", re.compile(r"usage limit|rate.?limit|quota|\b429\b|credit balance|too many requests|at capacity"
                         r"|overloaded", re.I)),
    ("auth", re.compile(r"\b40[13]\b|unauthori[sz]ed|invalid.{0,20}(key|token)|refresh token|log ?in|expired", re.I)),
)


def words(spec):
    """The distinct words of a REVIEW_ENGINE value, in order."""
    return list(dict.fromkeys(w.strip().lower() for w in (spec or "").split(",") if w.strip()))


def order(spec, available):
    """The engines that can run, in REVIEW_ENGINE's order, then the others in ENGINES order."""
    named = [w for w in words(spec) if w in ENGINES]
    return [e for e in (*named, *[x for x in ENGINES if x not in named]) if available.get(e, "unknown") is None]


def available_engines(env):
    """{engine: None when it can run, else why not}."""
    home = env.get("CODEX_HOME") or os.path.expanduser("~/.codex")
    login = {"glm": bool(env.get("ZAI_API_KEY")), "claude": bool(env.get("ANTHROPIC_API_KEY")),
             "codex": os.path.isfile(os.path.join(home, "auth.json"))}
    out = {}
    for e in ENGINES:
        cli = RUNNER[e]
        if not login[e]:
            out[e] = "no login"
        elif not shutil.which(cli, path=env.get("PATH")):
            out[e] = "not installed" if cli == e else f"{cli} CLI not installed"
        else:
            out[e] = None
    return out


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
    vals = [(f"env:{n}", env.get(n, "")) for n in ("ANTHROPIC_API_KEY", "ZAI_API_KEY")]
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


def child_env(engine, env):
    """The environment one engine's script runs under: its own login, no other engine's and no runner's routing. glm
    is the claude script on z.ai's endpoint: the key becomes ANTHROPIC_AUTH_TOKEN, and every model alias Claude Code
    might ask for is pinned to the one model, since z.ai answers an id it is given without saying what it is."""
    child = {k: v for k, v in env.items() if k != "ZAI_API_KEY" and k not in ROUTES}
    child["REVIEW_RUN_ENGINE"] = engine
    if engine == "codex":
        child.pop("ANTHROPIC_API_KEY", None)
    elif engine == "glm":
        model = env.get("REVIEW_MODEL_GLM") or GLM_MODEL
        child.pop("ANTHROPIC_API_KEY", None)
        child.update(ANTHROPIC_BASE_URL=env.get("ZAI_BASE_URL") or GLM_BASE_URL, ANTHROPIC_AUTH_TOKEN=env["ZAI_API_KEY"],
                     REVIEW_MODEL=model, ANTHROPIC_MODEL=model, ANTHROPIC_DEFAULT_OPUS_MODEL=model,
                     ANTHROPIC_DEFAULT_SONNET_MODEL=model, ANTHROPIC_DEFAULT_HAIKU_MODEL=model,
                     ANTHROPIC_SMALL_FAST_MODEL=model, CLAUDE_CODE_SUBAGENT_MODEL=model)
        child.setdefault("CLAUDE_CODE_MAX_CONTEXT_TOKENS", GLM_CONTEXT_TOKENS)
        child.setdefault("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1")
    return child


def launch(engine, role, paths, env):
    """Run one engine's script in its own process group, its output in <role>.log of the TRACE it is given, under
    child_env. The group is stopped when the run overruns MINUTES[role] or run.py is stopped."""
    script = os.path.join(paths[0], "tools", "review", f"run-{RUNNER[engine]}.sh")
    child = child_env(engine, env)
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
    spec = env.get("REVIEW_ENGINE", "")
    tried = [e for e in order(spec, avail) if e not in dead]
    for w in words(spec):
        if w not in ENGINES:
            say(f"{role}: REVIEW_ENGINE names '{w}', which is no engine ({', '.join(ENGINES)}); skipped")
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
        for e in sorted(set(RUNNER.values())):
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
        assert not os.path.exists(os.path.join(out, "partial.md")) and "limit" in lines[1], lines
        assert lines[0] == "reviewer: glm not tried (no login)" and lines[-1] == "reviewer: done by claude after codex failed", lines
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
        assert "codex failed (sandbox check failed: the network was open)" in lines[1], lines
        assert "claude failed (auth" in lines[2] and "fake" not in "".join(lines), lines

        # Three engines, in REVIEW_ENGINE's order, the rest after in glm, claude, codex order. A word that is no
        # engine is named and skipped. Each failed engine's trace keeps the key out; a key in OUT ends the role 3.
        key = "zai-test-key-value"
        env3 = dict(env, ZAI_API_KEY=key, REVIEW_ENGINE="codex,glm")
        limit_z = (1, {"is_error": True, "error": "API Error: Request rejected (429) \u00b7 [1308][Usage limit reached "
                       "for 5 hour. Your limit will reset at 2026-10-06 16:23:45]"}, {})
        ok_g = (0, {"is_error": False, "engine": "glm"}, good, f"the endpoint echoed {key}\n")
        reset()
        with contextlib.redirect_stderr(io.StringIO()):
            assert run_role("reviewer", paths, env3, fake({"codex": limit, "glm": limit_z, "claude": ok_c}),
                            lines.append) == 0
        assert calls == [("codex", "reviewer"), ("glm", "reviewer"), ("claude", "reviewer")], calls
        assert lines[-1] == "reviewer: done by claude after codex, glm failed" and "glm failed (limit:" in lines[-2], lines
        reset()
        with contextlib.redirect_stderr(io.StringIO()):
            assert run_role("reviewer", paths, dict(env3, REVIEW_ENGINE=""), fake({"glm": ok_g}), lines.append) == 0
        assert calls == [("glm", "reviewer")] and lines == ["reviewer: done by glm"], (calls, lines)
        assert open(os.path.join(trace, "reviewer.engine")).read() == "glm\n"
        text = open(os.path.join(trace, "reviewer.log")).read()
        assert key not in text and "the endpoint echoed [credential removed]" in text, text
        reset()
        assert run_role("reviewer", paths, dict(env, REVIEW_ENGINE="gpt, claude,claude"), fake({"claude": ok_c}),
                        lines.append) == 0
        assert lines == ["reviewer: REVIEW_ENGINE names 'gpt', which is no engine (glm, claude, codex); skipped",
                         "reviewer: glm not tried (no login)", "reviewer: done by claude"], lines
        reset()
        leak_z = (0, ok, {**good, "review.md": f"**Verdict: CLEAN**: {key}\n"})
        assert run_role("reviewer", paths, dict(env3, REVIEW_ENGINE="glm"), fake({"glm": leak_z}), lines.append) == 3
        assert os.listdir(out) == [] and calls == [("glm", "reviewer")] and "env:ZAI_API_KEY" in lines[-1], lines
        reset()
        open(os.path.join(out, "review.md"), "w").write(good["review.md"])
        open(os.path.join(out, "decision.txt"), "w").write(good["decision.txt"])
        # The documented z.ai refusals read as a limit, like the others.
        for msg in (limit_z[1]["error"], "API Error: 429 [1302][Rate limit reached for requests]"):
            assert failure("glm", "reviewer", 1, {"is_error": True, "error": msg}, out, out, "").startswith("limit:"), msg

        # A glm login with no claude CLI is "claude CLI not installed"; no key is "no login".
        nobin = os.path.join(d, "nobin")
        os.mkdir(nobin)
        assert available_engines({"ZAI_API_KEY": key, "PATH": nobin})["glm"] == "claude CLI not installed"
        assert available_engines({"ANTHROPIC_API_KEY": key, "PATH": nobin})["claude"] == "not installed"
        assert available_engines({"ANTHROPIC_API_KEY": key, "PATH": bindir})["glm"] == "no login"
        assert available_engines({"ZAI_API_KEY": key, "PATH": bindir, "CODEX_HOME": home})["glm"] is None
        reset()
        assert run_role("learner", paths, {"ZAI_API_KEY": key, "REVIEW_ENGINE": "glm", "PATH": nobin, "CODEX_HOME": home},
                        fake({}), lines.append) == NO_ENGINE
        assert "learner: glm not tried (claude CLI not installed)" in lines and calls == [], (calls, lines)

        # The order rule: REVIEW_ENGINE's engines first, in its order, then the rest as ENGINES lists them.
        every = dict.fromkeys(ENGINES)
        assert words(" Codex , ,glm,codex") == ["codex", "glm"] and words("") == [] and words(None) == []
        assert order("", every) == ["glm", "claude", "codex"]
        assert order("codex", every) == ["codex", "glm", "claude"]
        assert order("claude,glm", every) == ["claude", "glm", "codex"]
        assert order("glm,codex,claude", every) == ["glm", "codex", "claude"]
        assert order("gpt,codex", every) == ["codex", "glm", "claude"]
        assert order("codex,glm", dict(every, glm="no login")) == ["codex", "claude"]

        # What each engine's script is started with, seen by a script that dumps its environment: the key is the
        # token only glm holds, no engine holds another's login, and a runner's leftover endpoint never reaches claude.
        stubs = os.path.join(paths[0], "tools", "review")
        os.makedirs(stubs)
        for runner in ("claude", "codex"):
            open(os.path.join(stubs, f"run-{runner}.sh"), "w").write(f'env > "{d}/env.$REVIEW_RUN_ENGINE"\n')
        parent = dict(env, ZAI_API_KEY=key, ANTHROPIC_BASE_URL="http://leftover.invalid", ANTHROPIC_AUTH_TOKEN="leftover",
                      CLAUDE_CODE_USE_FOUNDRY="1", REVIEW_MODEL="claude-sonnet-5-5", PATH=os.environ["PATH"])

        def child(engine, **extra):
            assert launch(engine, "reviewer", paths, dict(parent, **extra)) == 0
            return dict(line.split("=", 1) for line in read(os.path.join(d, f"env.{engine}")).splitlines() if "=" in line)

        g = child("glm")
        assert "ANTHROPIC_API_KEY" not in g and "ZAI_API_KEY" not in g, sorted(g)
        assert (g["ANTHROPIC_AUTH_TOKEN"], g["ANTHROPIC_BASE_URL"], g["REVIEW_MODEL"]) == (key, GLM_BASE_URL, GLM_MODEL), g
        pinned = {g[n] for n in ("ANTHROPIC_MODEL", "ANTHROPIC_DEFAULT_OPUS_MODEL", "ANTHROPIC_DEFAULT_SONNET_MODEL",
                                 "ANTHROPIC_DEFAULT_HAIKU_MODEL", "ANTHROPIC_SMALL_FAST_MODEL", "CLAUDE_CODE_SUBAGENT_MODEL")}
        assert pinned == {GLM_MODEL} and g["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == GLM_CONTEXT_TOKENS, g
        g = child("glm", ZAI_BASE_URL="http://z.invalid", REVIEW_MODEL_GLM="glm-x", CLAUDE_CODE_MAX_CONTEXT_TOKENS="5")
        assert (g["ANTHROPIC_BASE_URL"], g["REVIEW_MODEL"], g["ANTHROPIC_MODEL"], g["CLAUDE_CODE_MAX_CONTEXT_TOKENS"]) == (
            "http://z.invalid", "glm-x", "glm-x", "5"), g
        c = child("claude")
        assert c["ANTHROPIC_API_KEY"] == "sk-test-key-value" and c["REVIEW_MODEL"] == "claude-sonnet-5-5", c
        assert not {"ZAI_API_KEY", "CLAUDE_CODE_USE_FOUNDRY", "ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN"} & set(c), sorted(c)
        x = child("codex")
        assert not {"ZAI_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN"} & set(x), sorted(x)
        assert "CLAUDE_CODE_USE_FOUNDRY" not in g, sorted(g)
        assert ("env:ZAI_API_KEY", key) in credentials(parent)
        os.remove(os.path.join(stubs, "run-codex.sh"))

        # No Codex CLI: only Claude runs. A learner that leaves no candidates.json fails, as does one that leaves the
        # previous run's; an empty array is fine.
        reset()
        os.remove(os.path.join(bindir, "codex"))
        assert run_role("learner", paths, env, fake({"claude": (0, ok, {})}), lines.append) == NO_ENGINE
        assert lines[1] == "learner: codex not tried (not installed)" and "no candidates.json" in lines[2], lines
        reset()
        learned = (0, ok, {"candidates.json": "[]"})
        assert run_role("learner", paths, env, fake({"claude": learned}), lines.append) == 0
        reset()
        assert run_role("learner", paths, env, fake({"claude": (0, ok, {})}), lines.append) == NO_ENGINE
        assert "candidates.json is not from this run" in lines[2], lines

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
        os.makedirs(os.path.join(paths[0], "tools", "review"), exist_ok=True)
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
