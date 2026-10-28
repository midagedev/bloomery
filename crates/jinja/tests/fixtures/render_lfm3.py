"""Reference renders of the LFM3-1.7B chat template for lfm3-renders.json.

usage (jinja2 3.1.x; the chat template is the file's own, `tokenizer.chat_template` of the GGUF, byte for byte):
    python3 render_lfm3.py lfm3-chat-template.jinja > lfm3-renders.json

The environment is Hugging Face's `apply_chat_template` one: an immutable sandboxed environment with
trim_blocks, lstrip_blocks, the loopcontrols extension and the `generation` tag (transformers'
AssistantTracker: the block renders its body); `tojson` is json.dumps with ensure_ascii=False, indent=None,
separators=None and sort_keys=False as defaults; the global raise_exception. `bos_token` is the vocabulary's token
124894, which the template opens with.

Each case's `vars` are the variables the template renders with. A case jinja2 refused carries its error in
place of a prompt.
"""
import json
import sys

import jinja2
import jinja2.ext as jinja2_ext
from jinja2 import nodes
from jinja2.sandbox import ImmutableSandboxedEnvironment

BOS = "<|startoftext|>"


class Generation(jinja2_ext.Extension):
    tags = {"generation"}

    def parse(self, parser):
        lineno = next(parser.stream).lineno
        body = parser.parse_statements(["name:endgeneration"], drop_needle=True)
        return nodes.CallBlock(self.call_method("_generation", []), [], [], body).set_lineno(lineno)

    def _generation(self, caller):
        return caller()


env = ImmutableSandboxedEnvironment(
    trim_blocks=True,
    lstrip_blocks=True,
    extensions=[jinja2_ext.loopcontrols, Generation],
)


def raise_exception(message):
    raise jinja2.exceptions.TemplateError(message)


env.filters["tojson"] = lambda x, ensure_ascii=False, indent=None, separators=None, sort_keys=False: json.dumps(
    x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys
)
env.globals["raise_exception"] = raise_exception

WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name, e.g. 서울."},
                "days": {"type": "integer"},
            },
            "required": ["location"],
        },
    },
}
TIME = {
    "type": "function",
    "function": {"name": "get_time", "description": "The time now.", "parameters": {"type": "object", "properties": {}}},
}
HI = [{"role": "user", "content": "Hi"}]


def call(name, arguments):
    return {"id": "c", "type": "function", "function": {"name": name, "arguments": arguments}}


THINKING = [
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hello!", "reasoning_content": "The user greets.\n"},
    {"role": "user", "content": "Again?"},
    {"role": "assistant", "content": "Hello again.", "reasoning_content": "The user repeats."},
    {"role": "user", "content": "One more."},
]

CASES = [
    ("a user turn", {"messages": HI, "add_generation_prompt": True}),
    ("system and user, thinking off", {"messages": [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "Hi"},
    ], "add_generation_prompt": True, "enable_thinking": False}),
    ("no generation prompt", {"messages": HI, "add_generation_prompt": False}),
    ("thinking kept after the last user turn only", {"messages": THINKING[:4], "add_generation_prompt": True}),
    ("thinking kept in every turn", {"messages": THINKING, "add_generation_prompt": True, "preserve_thinking": True}),
    ("thinking inside the content", {"messages": [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "<think>\nGreet back.\n</think>\n\nHello!"},
        {"role": "user", "content": "Bye"},
    ], "add_generation_prompt": True}),
    ("content parts", {"messages": [
        {"role": "system", "content": [{"type": "text", "text": "Be brief."}, {"type": "text", "text": "No lists."}]},
        {"role": "user", "content": [{"type": "text", "text": "서울 날씨? ☃ \"quoted\" <b>"}, {"type": "image", "image": "x"}]},
    ], "add_generation_prompt": True}),
    ("tools and a user turn", {"messages": [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "Weather in Seoul?"},
    ], "tools": [WEATHER, TIME], "add_generation_prompt": True}),
    ("a tool call and its answer", {"messages": [
        {"role": "user", "content": "Weather in Seoul?"},
        {"role": "assistant", "content": "Checking.", "tool_calls": [
            call("get_weather", {"location": "서울", "days": 3, "detail": {"hourly": True, "n": [1, 2.5, None]}}),
        ]},
        {"role": "tool", "content": "{\"temp\": 21}"},
        {"role": "assistant", "content": "21 degrees."},
        {"role": "user", "content": "Thanks."},
    ], "tools": [WEATHER, TIME], "add_generation_prompt": True}),
    ("two calls, two answers", {"messages": [
        {"role": "user", "content": "Weather and time in Seoul?"},
        {"role": "assistant", "content": "", "tool_calls": [
            call("get_weather", {"location": "Seoul"}),
            {"type": "function", "function": {"name": "get_time", "arguments": {}}},
        ]},
        {"role": "tool", "content": "21"},
        {"role": "tool", "content": "12:00"},
    ], "tools": [WEATHER, TIME], "add_generation_prompt": True}),
    ("a string argument value", {"messages": [
        {"role": "user", "content": "Note."},
        {"role": "assistant", "content": "", "tool_calls": [call("save", {"text": "a\nb", "n": 1, "ok": False})]},
    ], "add_generation_prompt": False}),
    ("the final message continues", {"messages": [
        {"role": "user", "content": "Count."},
        {"role": "assistant", "content": "1, 2, CONTINUE_FINAL_MESSAGE_TAG "},
    ], "add_generation_prompt": False}),
    ("JSON-encoded arguments are refused", {"messages": [
        {"role": "user", "content": "Time?"},
        {"role": "assistant", "content": "", "tool_calls": [call("get_time", "{\"tz\": \"UTC\"}")]},
    ], "add_generation_prompt": False}),
]


def main(path):
    template = env.from_string(open(path, encoding="utf-8").read())
    cases = []
    for name, vars_ in CASES:
        vars_ = dict(vars_, bos_token=BOS)
        entry = {"name": name, "vars": vars_}
        try:
            entry["prompt"] = template.render(**vars_)
        except jinja2.exceptions.TemplateError as e:
            entry["error"] = f"TemplateError: {e}"
        cases.append(entry)
    json.dump(
        {
            "template": "lfm3-chat-template.jinja",
            "reference": f"jinja2 {jinja2.__version__} (Python {sys.version.split()[0]}), the Hugging Face chat-template environment "
            "(ImmutableSandboxedEnvironment with trim_blocks, lstrip_blocks, the loopcontrols extension and the generation tag; "
            "tojson = json.dumps with ensure_ascii=False, indent=None, separators=None, sort_keys=False as defaults; the global "
            "raise_exception); a case jinja2 refused carries its error in place of a prompt. Script: "
            "crates/jinja/tests/fixtures/render_lfm3.py, run as `python3 render_lfm3.py lfm3-chat-template.jinja > lfm3-renders.json`",
            "cases": cases,
        },
        sys.stdout,
        ensure_ascii=False,
        indent=2,
    )
    print()


main(sys.argv[1])
