"""Reference renders of the Hy4-flash chat template for crates/serve/tests/fixtures/hy4-renders.json.

usage (jinja2 3.1.6):
    python3 render_hy4.py hy4-chat-template.jinja > hy4-renders.json

The environment is render_glm5.py's, line for line (ik_llama.cpp tests/test-jinja.cpp's py_script): SandboxedEnvironment with
trim_blocks, lstrip_blocks and the loopcontrols extension; `tojson` is json.dumps with ensure_ascii=False, indent=None,
separators=None, sort_keys=False as defaults; globals strftime_now and raise_exception. The template's own 22
`'<｜hy_start{}｜>'.format(HYTK)` calls run as Python's `str.format`, which is what bloomery's engine is held to.

`no_think`, the template's off word, is not one of OpenAI's `reasoning_effort` words, which bloomery-serve checks on the body, so
the cases that use it send it in `chat_template_kwargs` (merged over the body's own effort, as the server does).

The variables are the ones bloomery-serve's render_chat (crates/serve/src/api.rs) passes for a body: messages (a content array
flattened to its text parts joined by "\n"), add_generation_prompt (the body's boolean, else true), bos_token and eos_token (the
mock engine's texts), tools and reasoning_effort when present and not null, and chat_template_kwargs merged on top.

Every object in a body is written with its keys sorted, at every level, as the other fixtures are.
"""
import json
import sys
from datetime import datetime

import jinja2
import jinja2.ext as jinja2_ext
from jinja2.sandbox import SandboxedEnvironment

BOS = "<｜begin▁of▁sentence｜>"
EOS = "<｜end▁of▁sentence｜>"

env = SandboxedEnvironment(
    trim_blocks=True,
    lstrip_blocks=True,
    extensions=[jinja2_ext.loopcontrols],
)


def raise_exception(message):
    raise jinja2.exceptions.TemplateError(message)


env.filters["tojson"] = lambda x, ensure_ascii=False, indent=None, separators=None, sort_keys=False: json.dumps(
    x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys
)
env.globals["strftime_now"] = lambda format: datetime.now().strftime(format)
env.globals["raise_exception"] = raise_exception

WEATHER = {
    "function": {
        "description": "Current weather for a city.",
        "name": "get_weather",
        "parameters": {
            "properties": {
                "days": {"description": "How many days ahead.", "type": "integer"},
                "location": {"description": "City name, e.g. Seoul.", "type": "string"},
            },
            "required": ["location"],
            "type": "object",
        },
        "strict": True,
    },
    "type": "function",
}
TIME = {
    "function": {
        "description": "The time now.",
        "name": "get_time",
        "parameters": {"properties": {}, "type": "object"},
    },
    "type": "function",
}
NOTE = {
    "function": {
        "description": "Save a note.",
        "name": "save_note",
        "parameters": {
            "properties": {
                "filter": {"type": "object"},
                "text": {"type": "string"},
            },
            "type": "object",
        },
    },
    "type": "function",
}
# A tool the way the template also takes one: no `function` wrapper, a `defer_loading` flag, and one deferred away.
BARE = {"description": "No wrapper.", "name": "bare", "parameters": {"properties": {}, "type": "object"}, "strict": False}
DEFERRED = {"defer_loading": True, "description": "Not listed.", "name": "deferred", "parameters": {"type": "object"}}


def call(id_, name, arguments):
    return {"function": {"arguments": arguments, "name": name}, "id": id_, "type": "function"}


CASES = [
    ("system and user", {"messages": [
        {"content": "You are terse.", "role": "system"},
        {"content": "Hi", "role": "user"},
    ]}),
    ("user only", {"messages": [{"content": "Hi", "role": "user"}]}),
    ("three turns", {"messages": [
        {"content": "What is the capital of France?", "role": "user"},
        {"content": "Paris.", "role": "assistant"},
        {"content": "And of Italy?", "role": "user"},
    ]}),
    ("reasoning_effort no_think", {"chat_template_kwargs": {"reasoning_effort": "no_think"}, "messages": [{"content": "Hi", "role": "user"}]}),
    ("reasoning_effort low and a system turn", {"messages": [
        {"content": "Be brief.", "role": "system"},
        {"content": "Hi", "role": "user"},
    ], "reasoning_effort": "low"}),
    ("reasoning_effort max", {"messages": [{"content": "Hi", "role": "user"}], "reasoning_effort": "max"}),
    ("reasoning_effort outside the list", {"messages": [{"content": "Hi", "role": "user"}], "reasoning_effort": "medium"}),
    ("reasoning_effort high through the kwargs and low on the body: the kwargs win", {"chat_template_kwargs": {"reasoning_effort": "high"}, "reasoning_effort": "low", "messages": [
        {"content": "Hi", "role": "user"},
    ]}),
    ("past reasoning: only the turn after the last user keeps it", {"messages": [
        {"content": "Hi", "role": "user"},
        {"content": "Hello!", "reasoning_content": "The user greets.", "role": "assistant"},
        {"content": "Again?", "role": "user"},
        {"content": "Hello again!", "reasoning_content": "The user repeats.", "role": "assistant"},
    ], "add_generation_prompt": False}),
    ("reasoning takes precedence over reasoning_content", {"messages": [
        {"content": "Hi", "role": "user"},
        {"content": "Hello!", "reasoning": "first", "reasoning_content": "second", "role": "assistant"},
    ], "add_generation_prompt": False}),
    ("no_think drops the reasoning of the last turn", {"chat_template_kwargs": {"reasoning_effort": "no_think"}, "messages": [
        {"content": "Hi", "role": "user"},
        {"content": "Hello!", "reasoning_content": "The user greets.", "role": "assistant"},
    ], "add_generation_prompt": False}),
    ("a generation prompt after an assistant turn adds none", {"messages": [
        {"content": "Hi", "role": "user"},
        {"content": "Hello!", "role": "assistant"},
    ]}),
    ("raw last assistant: an open prefill", {"chat_template_kwargs": {"raw_last_assistant": True}, "messages": [
        {"content": "Hi", "role": "user"},
        {"content": "Hello, ", "reasoning_content": "kept out", "role": "assistant"},
    ], "add_generation_prompt": False}),
    ("fallback strategy reasoning_toolcall_retry", {"chat_template_kwargs": {"fallback_strategy": "reasoning_toolcall_retry", "reasoning_effort": "no_think"}, "messages": [
        {"content": "Hi", "role": "user"},
        {"content": "Hello, ", "role": "assistant"},
    ]}),
    ("a system turn in the middle", {"messages": [
        {"content": "First.", "role": "system"},
        {"content": "Hi", "role": "user"},
        {"content": "Second.", "role": "system"},
        {"content": "Again", "role": "user"},
    ]}),
    ("tools and a user turn", {"messages": [{"content": "Weather in Seoul?", "role": "user"}], "tools": [WEATHER, TIME]}),
    ("tools and a system turn", {"messages": [
        {"content": "You help.", "role": "system"},
        {"content": "Weather in Seoul?", "role": "user"},
    ], "tools": [WEATHER, BARE, DEFERRED], "reasoning_effort": "low"}),
    ("two calls, their results in the other order", {"messages": [
        {"content": "Weather and time in Seoul?", "role": "user"},
        {"content": "Checking both.", "reasoning_content": "Two tools.", "role": "assistant", "tool_calls": [
            call("call_a", "get_weather", {"days": 3, "location": "Seoul"}),
            call("call_b", "get_time", {}),
        ]},
        {"content": "12:00", "role": "tool", "tool_call_id": "call_b"},
        {"content": "{\"temp\": 21}", "role": "tool", "tool_call_id": "call_a"},
    ], "tools": [WEATHER, TIME]}),
    ("a tool result without an id", {"messages": [
        {"content": "Time?", "role": "user"},
        {"content": "", "role": "assistant", "tool_calls": [call("call_c", "get_time", {})]},
        {"content": "12:00", "role": "tool"},
        {"content": "Thanks.", "role": "user"},
    ], "tools": [TIME]}),
    ("tool calls kept with their reasoning, then a new user turn", {"messages": [
        {"content": "Time?", "role": "user"},
        {"content": "", "reasoning_content": "Ask the clock.", "role": "assistant", "tool_calls": [call("call_f", "get_time", {})]},
        {"content": "12:00", "role": "tool", "tool_call_id": "call_f"},
        {"content": "It is noon.", "reasoning_content": "Report it.", "role": "assistant"},
        {"content": "Thanks.", "role": "user"},
    ], "tools": [TIME]}),
    ("preserved_thinking off with tools", {"chat_template_kwargs": {"preserved_thinking": False}, "messages": [
        {"content": "Time?", "role": "user"},
        {"content": "", "reasoning_content": "Ask the clock.", "role": "assistant", "tool_calls": [call("call_f", "get_time", {})]},
        {"content": "12:00", "role": "tool", "tool_call_id": "call_f"},
        {"content": "Thanks.", "role": "user"},
    ], "tools": [TIME]}),
    ("non-ASCII, nested and typed arguments", {"add_generation_prompt": False, "messages": [
        {"content": "Note it.", "role": "user"},
        {"content": "", "role": "assistant", "tool_calls": [
            call("call_d", "save_note", {"filter": {"n": [1, 2], "q": "서울 ☃", "ok": True, "x": None, "f": 1.5}, "text": "a<b & \"c\""}),
        ]},
    ], "tools": [NOTE]}),
    ("string arguments", {"messages": [
        {"content": "Time?", "role": "user"},
        {"content": "", "role": "assistant", "tool_calls": [call("call_e", "get_time", "{}")]},
    ], "tools": [TIME]}),
    ("a content array of text parts", {"messages": [
        {"content": [{"text": "Part one.", "type": "text"}, {"text": "Part two.", "type": "text"}], "role": "user"},
    ]}),
    ("a null content", {"messages": [
        {"content": "Hi", "role": "user"},
        {"content": None, "role": "assistant", "tool_calls": [call("call_g", "get_time", {})]},
    ], "tools": [TIME]}),
]


def variables(body):
    msgs = []
    for m in body["messages"]:
        m = dict(m)
        if isinstance(m.get("content"), list):
            m["content"] = "\n".join(
                p["text"] for p in m["content"] if p.get("type") == "text" and isinstance(p.get("text"), str)
            )
        msgs.append(m)
    agp = body.get("add_generation_prompt")
    v = {
        "messages": msgs,
        "add_generation_prompt": agp if isinstance(agp, bool) else True,
        "bos_token": BOS,
        "eos_token": EOS,
    }
    for k in ("tools", "reasoning_effort"):
        if body.get(k) is not None:
            v[k] = body[k]
    if isinstance(body.get("chat_template_kwargs"), dict):
        v.update(body["chat_template_kwargs"])
    return v


def main():
    source = open(sys.argv[1], encoding="utf-8").read()
    template = env.from_string(source)
    cases = []
    for name, body in CASES:
        # The body as the test sends it: keys sorted at every level.
        body = json.loads(json.dumps(body, sort_keys=True))
        case = {"name": name, "body": body}
        try:
            case["prompt"] = template.render(**variables(body))
        except Exception as e:  # jinja2 raised: the case records its error
            case["error"] = f"{type(e).__name__}: {e}"
        cases.append(case)
    out = {
        "template": "tokenizer.chat_template of Hy4-flash-Q4_K_M-00001-of-00002.gguf (Tencent's official GGUF, the first shard), read from the GGUF header on the box: 17263 bytes, md5 bec49a793e2864b5e6751900152e988c; hy4-chat-template.jinja is those bytes (hy4-chat-template.source)",
        "reference": "jinja2 3.1.6 (Python 3.14.3, on the lead's Mac in a throwaway venv), the HF chat-template environment of ik_llama.cpp tests/test-jinja.cpp (SandboxedEnvironment with trim_blocks, lstrip_blocks and the loopcontrols extension; tojson = json.dumps with ensure_ascii=False, indent=None, separators=None, sort_keys=False as defaults; globals strftime_now and raise_exception); each case rendered with the variables bloomery-serve's /apply-template passes (messages with a content array flattened to its text parts, add_generation_prompt default true, the mock's bos_token and eos_token, tools and reasoning_effort when not null, chat_template_kwargs merged); every body has its keys sorted at every level; a case jinja2 refused carries its error in place of a prompt. Script: crates/serve/tests/fixtures/render_hy4.py, run as `python3 render_hy4.py hy4-chat-template.jinja > hy4-renders.json`",
        "cases": cases,
    }
    sys.stdout.write(json.dumps(out, ensure_ascii=False, indent=2) + "\n")


main()
