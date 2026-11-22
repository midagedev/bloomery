#!/usr/bin/env bash
# The reference's own rendering of Inkling 2 chat requests, written as the serve gate's fixture
# (crates/serve/tests/fixtures/inkling2-renders.json): tools/ref/inkling2-chat-render.cpp linked against the oracle
# tree's libraries (llama.cpp PR #32695's build, nothing rebuilt), run on each of the two chat templates the released
# files carry (inkling2-mini-chat-template.jinja, inkling2-small-chat-template.jinja) over every case below.
#   INKLING2_ORACLE  the oracle tree (default /home/user/llama.cpp-pr32695-scale)
#   usage: tools/ref/inkling2-chat-render.sh > crates/serve/tests/fixtures/inkling2-renders.json
# A case the template refuses (`raise_exception`) carries its message in `error` in place of `prompt`.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ORACLE=${INKLING2_ORACLE:-/home/user/llama.cpp-pr32695-scale}
LIBDIR=$ORACLE/build/bin
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
g++ -std=c++17 -O1 -o "$WORK/render" "$HERE/tools/ref/inkling2-chat-render.cpp" \
  -I"$ORACLE/common" -I"$ORACLE/include" -I"$ORACLE/ggml/include" -I"$ORACLE/vendor" \
  -L"$LIBDIR" -lllama-common -lllama -lggml -lggml-base -Wl,-rpath,"$LIBDIR" >&2
python3 -I - "$WORK/render" "$HERE/crates/serve/tests/fixtures" "$LIBDIR" <<'PY'
import json, os, subprocess, sys, tempfile
render, fixtures, libdir = sys.argv[1:4]
TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "description": "Get weather", "parameters": {
        "type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["c", "f"]}},
        "required": ["city"]}}},
    {"type": "function", "function": {"name": "get_time", "parameters": {"type": "object", "properties": {}}}},
]
def user(text): return {"role": "user", "content": text}
def call(i, name, args): return {"id": i, "type": "function", "function": {"name": name, "arguments": args}}
CASES = [
    ("a user turn", {"messages": [user("Hello")]}),
    ("system and user, Korean and emoji", {"messages": [{"role": "system", "content": "You are terse."}, user("한국어로 답해줘 😀")]}),
    ("a conversation", {"messages": [{"role": "system", "content": "S"}, user("Q1"), {"role": "assistant", "content": "A1"}, user("Q2")]}),
    ("content parts", {"messages": [{"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}]}),
    ("tool declarations", {"messages": [user("Weather in Seoul?")], "tools": TOOLS}),
    ("two calls and their results", {"messages": [user("Weather in Seoul and the time?"),
        {"role": "assistant", "content": "", "tool_calls": [call("c1", "get_weather", {"unit": "c", "city": "Seoul"}), call("c2", "get_time", {})]},
        {"role": "tool", "tool_call_id": "c1", "content": "{\"temp\":21}"},
        {"role": "tool", "tool_call_id": "c2", "name": "get_time", "content": "15:00"}, user("thanks")], "tools": TOOLS}),
    ("a thinking turn kept in history", {"messages": [user("2+2?"), {"role": "assistant", "reasoning_content": "Add them.", "content": "4"}, user("and 3+3?")]}),
    ("thinking, text and a call with nested arguments", {"messages": [user("Weather?"),
        {"role": "assistant", "reasoning_content": "Need the tool.", "content": "On it.",
         "tool_calls": [call("c1", "get_weather", {"city": "Seoul", "extra": {"b": [1, 2], "a": "é", "n": 1.5}})]}], "tools": TOOLS}),
    ("effort low", {"messages": [user("Hello")], "reasoning_effort": "low"}),
    ("effort none", {"messages": [user("Hello")], "reasoning_effort": "none"}),
    ("effort max", {"messages": [user("Hello")], "reasoning_effort": "max"}),
    # The server refuses an effort word it does not know before the template does; a kwarg reaches the template.
    ("effort the template does not know", {"messages": [user("Hello")], "chat_template_kwargs": {"reasoning_effort": "bogus"}}),
]
def run(template, body):
    with tempfile.TemporaryDirectory() as d:
        req = os.path.join(d, "req.json")
        effort = body.get("reasoning_effort", body.get("chat_template_kwargs", {}).get("reasoning_effort"))
        req_body = {k: v for k, v in body.items() if k not in ("reasoning_effort", "chat_template_kwargs")}
        json.dump(req_body, open(req, "w"))
        args = [render, template, req] + ([json.dumps(effort)] if effort is not None else [])
        p = subprocess.run(args, capture_output=True, text=True, env={**os.environ, "LD_LIBRARY_PATH": libdir})
    if p.returncode == 0:
        return {"prompt": p.stdout.split("\n=== generation prompt\n")[0]}
    err = [l for l in p.stderr.splitlines() if "Jinja Exception:" in l]
    if not err:
        sys.exit("the reference failed without a template error:\n" + p.stderr[-2000:])
    return {"error": err[-1].split("Jinja Exception: ", 1)[1]}
out = {"reference": "llama.cpp PR #32695 oracle tree, common_chat_templates_apply (its jinja engine) through tools/ref/inkling2-chat-render.cpp; "
                    "regenerate with tools/ref/inkling2-chat-render.sh", "templates": {}}
for key, name in (("mini", "inkling2-mini-chat-template.jinja"), ("small", "inkling2-small-chat-template.jinja")):
    path = os.path.join(fixtures, name)
    out["templates"][key] = {"file": name, "cases": [{"name": n, "body": b, **run(path, b)} for n, b in CASES]}
json.dump(out, sys.stdout, indent=1, ensure_ascii=False)
print()
PY
