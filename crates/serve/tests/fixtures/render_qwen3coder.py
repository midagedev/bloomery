"""Reference renders of the Qwen3-Coder chat template for crates/serve/tests/fixtures/qwen3coder-renders.json.

usage (transformers 5.x and jinja2 3.1.x; the template is the checkpoint's `chat_template.jinja`, byte for byte,
which qwen3coder-chat-template.source records):
    python3 render_qwen3coder.py qwen3coder-chat-template.jinja > qwen3coder-renders.json

The renderer is Hugging Face's own, `transformers.utils.chat_template_utils.render_jinja_template`, the function
`apply_chat_template` calls: an ImmutableSandboxedEnvironment with trim_blocks, lstrip_blocks, the loopcontrols
extension and the `generation` tag, `tojson` as json.dumps with ensure_ascii=False, and the globals raise_exception and
strftime_now. The one change: `strftime_now` is taken out, because bloomery-serve defines none and the template reads
none.

The variables are the ones bloomery-serve's render_chat (crates/serve/src/api.rs) passes for a body: messages (the role
`developer` sent as `system`, a content array flattened to its text parts joined by "\n"), add_generation_prompt (the
body's boolean, else true), bos_token and eos_token (the mock engine's texts), tools when present and not null, and
chat_template_kwargs merged on top. The template reads no thinking switch and no effort knob.

Bodies keep their keys in the order written here: the template walks a schema's properties and a tool call's arguments
with `|items`, so the order is part of the render, and bloomery's template engine keeps a body's insertion order
(serde_json's `preserve_order`), as jinja2 does.
"""
import json
import platform
import sys

import jinja2
import transformers
from transformers.utils.chat_template_utils import _compile_jinja_template, render_jinja_template

BOS = "<｜begin▁of▁sentence｜>"
EOS = "<｜end▁of▁sentence｜>"

WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "  Current weather for a city.\n",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": " City name, e.g. Seoul. "},
                "days": {"type": "integer", "description": "How many days ahead.", "default": 3, "minimum": 1},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
    },
}
SEARCH = {
    "type": "function",
    "function": {
        "name": "search_files",
        "description": "Search the files of a repository.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "A regular expression.", "examples": ["TODO", "fn main"]},
                "filters": {
                    "type": "object",
                    "properties": {"ext": {"type": "string"}, "max-size": {"type": "integer"}},
                    "required": ["ext"],
                },
                "tags": {"type": "array", "items": {"type": "string"}, "uniqueItems": True, "nullable": None},
                "limit": {"type": "integer", "exclusiveMinimum": 0, "x-internal id": "a$b", "$comment": "cap", "ratio": 0.5},
            },
            "required": ["query", "limit"],
        },
        "return": {"type": "array", "items": {"type": "string"}},
    },
}
TIME = {
    "type": "function",
    "function": {
        "name": "get_time",
        "description": "The time now.",
        "parameters": {"type": "object", "properties": {}},
        "return": "The time as HH:MM.",
    },
}
PING = {
    "type": "function",
    "function": {"name": "ping", "parameters": {"type": "object"}},
}
BARE = {
    "function": {
        "name": "echo",
        "description": "Echo the text.",
        "parameters": {"type": "object", "properties": {"text": {"description": "Anything."}}, "required": ["text"]},
    },
}
HI = [{"role": "user", "content": "Hi"}]


def call(id_, name, arguments):
    return {"id": id_, "type": "function", "function": {"name": name, "arguments": arguments}}


CASES = [
    ("system, two tools and a user turn", {"messages": [
        {"role": "system", "content": "You are a coding assistant."},
        {"role": "user", "content": "Weather in Seoul, and find TODOs."},
    ], "tools": [WEATHER, SEARCH]}),
    ("tools with no system message get the default identity", {"messages": HI, "tools": [TIME, PING, BARE]}),
    ("no tools", {"messages": [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "Again?"},
    ]}),
    ("no generation prompt", {"messages": HI, "add_generation_prompt": False}),
    ("a tool call, its result and the generation prompt", {"messages": [
        {"role": "user", "content": "Weather in Seoul?"},
        {"role": "assistant", "content": "  Checking.\n", "tool_calls": [
            call("call_a", "get_weather", {"location": "Seoul", "days": 3}),
        ]},
        {"role": "tool", "tool_call_id": "call_a", "content": "{\"temp\": 21}"},
    ], "tools": [WEATHER, SEARCH]}),
    ("two calls, two results, then a user turn", {"messages": [
        {"role": "user", "content": "Weather and time in Seoul?"},
        {"role": "assistant", "content": "", "tool_calls": [
            call("call_a", "get_weather", {"location": "Seoul"}),
            call("call_b", "get_time", {}),
        ]},
        {"role": "tool", "tool_call_id": "call_a", "content": "21"},
        {"role": "tool", "tool_call_id": "call_b", "content": "12:00"},
        {"role": "user", "content": "Thanks."},
    ], "tools": [WEATHER, TIME]}),
    ("two tool blocks, the last one ends the prompt", {"messages": [
        {"role": "user", "content": "Go."},
        {"role": "assistant", "tool_calls": [call("c1", "get_time", {})]},
        {"role": "tool", "content": "12:00"},
        {"role": "assistant", "content": "ok", "tool_calls": [call("c2", "get_time", {}), call("c3", "get_time", {})]},
        {"role": "tool", "content": "12:30"},
        {"role": "tool", "content": "13:00"},
    ], "tools": [TIME], "add_generation_prompt": False}),
    ("arguments of every kind, in written order", {"messages": [
        {"role": "user", "content": "Search."},
        {"role": "assistant", "content": "", "tool_calls": [
            call("call_d", "search_files", {
                "query": "a<b & \"c\"\n서울 ☃",
                "filters": {"ext": "rs", "max-size": 10, "deep": [1, 2.5, None, True]},
                "limit": 3,
                "ratio": 0.5,
                "flag": False,
                "nothing": None,
                "tags": ["x", {"y": 1}],
            }),
        ]},
    ], "tools": [SEARCH], "add_generation_prompt": False}),
    ("a call with no arguments, null content", {"messages": [
        {"role": "user", "content": "Time?"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "get_time"}}]},
    ], "tools": [TIME], "add_generation_prompt": False}),
    ("a tool result first", {"messages": [
        {"role": "tool", "content": "orphan"},
        {"role": "user", "content": "Thanks."},
    ]}),
    ("content parts flattened", {"messages": [
        {"role": "system", "content": [{"type": "text", "text": "Be brief."}, {"type": "text", "text": "No lists."}]},
        {"role": "user", "content": [{"type": "text", "text": "서울 날씨? ☃ \"quoted\" <b>"}]},
    ]}),
    ("a developer turn is sent as system, a late system turn renders as its own, an unknown role too", {"messages": [
        {"role": "developer", "content": "Answer in Korean."},
        {"role": "user", "content": "Hi"},
        {"role": "system", "content": "Late."},
        {"role": "narrator", "content": "..."},
    ]}),
    ("string arguments", {"messages": [
        {"role": "user", "content": "Time?"},
        {"role": "assistant", "content": "", "tool_calls": [call("call_e", "get_time", "{}")]},
    ], "tools": [TIME]}),
]


def variables(body):
    msgs = []
    for m in body["messages"]:
        m = dict(m)
        if m["role"] == "developer":
            m["role"] = "system"
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
    if body.get("tools") is not None:
        v["tools"] = body["tools"]
    if isinstance(body.get("chat_template_kwargs"), dict):
        v.update(body["chat_template_kwargs"])
    return v


def render(source, v):
    v = dict(v)
    messages = v.pop("messages")
    tools = v.pop("tools", None)
    add_generation_prompt = v.pop("add_generation_prompt")
    # The server defines no strftime_now: the environment here defines none either.
    _compile_jinja_template(source).environment.globals.pop("strftime_now", None)
    rendered, _ = render_jinja_template(
        [messages], tools=tools, chat_template=source, add_generation_prompt=add_generation_prompt, **v
    )
    return rendered[0]


def main():
    source = open(sys.argv[1], encoding="utf-8").read()
    cases = []
    for name, body in CASES:
        case = {"name": name, "body": body}
        try:
            case["prompt"] = render(source, variables(body))
        except Exception as e:  # jinja2 raised: the case records its error
            case["error"] = f"{type(e).__name__}: {e}"
        cases.append(case)
    out = {
        "template": "qwen3coder-chat-template.jinja, the checkpoint's chat_template.jinja, byte for byte",
        "reference": f"transformers {transformers.__version__} render_jinja_template (jinja2 {jinja2.__version__}, "
        f"Python {platform.python_version()}), the Hugging Face chat-template environment less its strftime_now global; "
        "each case rendered with the variables bloomery-serve's /apply-template passes (see render_qwen3coder.py); a "
        "case jinja2 refused carries its error in place of a prompt. Script: "
        "crates/serve/tests/fixtures/render_qwen3coder.py, run as "
        "`python3 render_qwen3coder.py qwen3coder-chat-template.jinja > qwen3coder-renders.json`",
        "cases": cases,
    }
    sys.stdout.write(json.dumps(out, ensure_ascii=False, indent=2) + "\n")


main()
