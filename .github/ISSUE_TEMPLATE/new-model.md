---
name: New model support
about: propose or claim a model architecture to add (open this before you start, so two people do not take one model)
title: "model: <name>"
labels: new-model
---

Read [`docs/contrib/new-model.md`](../../docs/contrib/new-model.md) first.

**The model**
- Name, Hugging Face link and the GGUF you would start from (quantization, size):
- `general.architecture` in its GGUF header:
- Does llama.cpp run it today? (the PR or release, if so):

**What is new about it** — the architecture facts that no attached model has (attention kind, recurrent state, router,
quant types, tokenizer). Everything else should reuse a common owner; name the attached model it is closest to:

**Plan**
- Are you taking it on yourself (yes / proposing only):
- The machine you would test on (card, VRAM, host RAM):
