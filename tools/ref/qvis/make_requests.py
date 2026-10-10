#!/usr/bin/env python3
"""Write requests.jsonl beside this file: the chat requests of the Qwen image-input oracle sets (E, and C's prompts).

One JSON line a request, {"id": ..., "messages": [...]}, in the OpenAI chat shape llama-server takes: a message's content
is a string or an array of {"type": "text", "text": ...} and {"type": "image_url", "image_url": {"url": "<image name>"}}
parts, the name one of tools/ref/clefvis/images.tsv's. dump_mtmd chat renders each through the model's own chat template.

  c1  user [image, text]                                the Clef set A's first prompt shape: one 448x448 image
  c2  user [text, image, image]                         two images (640x480 and 1024x768) in one turn
  c3  user [image, text]                                one image and about 2,000 tokens of text
  e1  user [text, image, text]                          text on both sides of the image
  e2  system, user [image]                              a system turn and an image-only user turn
  e3  user [image], assistant, user [image, text]       an image in each of two user turns, an assistant turn between

The prompts of c1..c3 are the cases of tools/ref/qvis/cases.tsv; e1..e3 are set E only. The text of c3 is a pure function of
the item counter, so the file can be rebuilt anywhere; nothing here reads the model: the token count of c3 is what the
chatids set records (`# tokens_count`), and cases.tsv says what it must be.
"""

import json
from pathlib import Path


def text(t):
    return {"type": "text", "text": t}


def image(name):
    return {"type": "image_url", "image_url": {"url": name}}


def long_text(items):
    return " ".join(
        f"Item {i}: the quick brown fox jumps over the lazy dog near river bend {i}, and the clock strikes {i % 12 + 1}."
        for i in range(1, items + 1)
    )


REQUESTS = [
    ("c1", [{"role": "user", "content": [image("free-448x448"), text("Describe the pattern in this image in one sentence.")]}]),
    (
        "c2",
        [{"role": "user", "content": [text("Compare these two test patterns."), image("free-640x480"), image("free-1024x768")]}],
    ),
    ("c3", [{"role": "user", "content": [image("free-448x448"), text(long_text(72) + " Now describe the image.")]}]),
    (
        "e1",
        [{"role": "user", "content": [text("Look at this picture."), image("free-448x448"), text("Which colour dominates it?")]}],
    ),
    (
        "e2",
        [
            {"role": "system", "content": "You are a careful image analyst."},
            {"role": "user", "content": [image("free-640x480")]},
        ],
    ),
    (
        "e3",
        [
            {"role": "user", "content": [image("free-448x448")]},
            {"role": "assistant", "content": "A test pattern of rings and ramps."},
            {"role": "user", "content": [image("free-640x480"), text("And this one?")]},
        ],
    ),
]

if __name__ == "__main__":
    out = Path(__file__).with_name("requests.jsonl")
    lines = [json.dumps({"id": i, "messages": m}, separators=(",", ":"), ensure_ascii=False) for i, m in REQUESTS]
    out.write_text("\n".join(lines) + "\n")
    print(f"{out}: {len(lines)} requests")
