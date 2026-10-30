# DeepSeek Harness (dsh) on DeepSeek-V4.1-Flash

dsh is DeepSeek's own coding-agent harness (npm `@deepseek-ai/dsh`; 0.2.0-rc.2 is the version surveyed): a web UI,
and one task in a terminal with `dsh headless`. On DeepSeek-V4.1-Flash it is the agent to use: its first request is
the second smallest of the surveyed agents' (~6,600 ids [derived]), and the web UI shows the work as it happens
([`research/coding-agents.md`](research/coding-agents.md)). For other agents, read
[`agents.md`](agents.md#which-coding-agent).

```sh
npm install -g @deepseek-ai/dsh
```

## 1. Start the server

V4.1-Flash with its vision encoder, so dsh can read screenshots:

```sh
bloomery-serve -m <V4.1-Flash GGUF first shard> --mmproj <mmproj-DeepSeek-V4.1-Flash-BF16.gguf> --port 18081
```

`-m` names the first shard of the GGUF set; the paths are yours. The `--mmproj` file is the model's vision encoder:
dsh's `read_image` sends screenshots as images, and the server can only decode them with it. Wait for the `ready`
event on stdout, or `GET /health` to answer 200 ([`agents.md`](agents.md)).

## 2. Write the profile

dsh reads its providers from a profile patch. Write the same file to both `$DSH_HOME/profiles/web/cordis.patch.yml`
and `$DSH_HOME/profiles/headless/cordis.patch.yml`:

```yaml
- id: llm-pi-ai
  config:
    providers:
      bloomery:
        apiKeyEnv: BLOOMERY_DSH_KEY
        api: openai-completions
        baseURL: http://127.0.0.1:18081/v1
        reasoning: off
        compat:
          thinkingFormat: chat-template
          chatTemplateKwargs:
            enable_thinking:
              $var: thinking.enabled
            reasoning_effort:
              $var: thinking.effort
              omitWhenOff: true
        models:
          - id: deepseek-v4.1-flash-1017
            contextWindow: 262144
            maxTokens: 65536
            input: [text, image]
            reasoningEfforts:
              off: none
              high: high
              max: max
- id: agent-default-model
  config:
    provider: bloomery
    model: deepseek-v4.1-flash-1017
```

The `llm-pi-ai` id names pi-ai, the provider layer dsh speaks, so the fields (`api`, `baseURL`, `models`) have the
same shape as pi's `models.json` ([`pi.md`](pi.md)). Three choices in it:

- `reasoning: off` — at effort `high` the model answers long: the first two answers were 16,997 and 13,206 ids, and
  the first page came 12 to 13 min after the server was up. The wait is the model's answer length, not the
  harness's prompt ([`research/coding-agents.md`](research/coding-agents.md) §1).
- `input: [text, image]` — turns on dsh's `read_image`, which needs the server's `--mmproj` (§1).
- `contextWindow` — replace `262144` with the `ready` event's `ctx` before running dsh. The server can fit a smaller
  context to available memory. Keep `maxTokens` below that served window too.

The model `id` can be any name the server answers to: bloomery serves its one model whatever name a request carries.
`BLOOMERY_DSH_KEY` can be any value; the server does not check it.

## 3. Run

The web UI (it prints a `http://127.0.0.1:13080/?token=…` URL to open; `--no-open` keeps dsh from opening a browser
itself):

```sh
BLOOMERY_DSH_KEY=local DSH_HOME=<dir> dsh --profile web --no-open --port 13080
```

One task, no UI:

```sh
BLOOMERY_DSH_KEY=local DSH_HOME=<dir> dsh headless "Make index.html: a three.js page with a rotating cube."
```

## What to expect

Measured on 2026-10-30 on the RTX A6000 with bloomery 0.3.11, with the profile above:

- An agent that checks its own screenshots ran to the 27-minute cap of the run.
- Generation ran at 27 ids/s at a 42K context (`depth`).
- A turn's prompt was mostly reused from the cache: 1,469 new ids with 39,373 reused in one logged turn, 6.77 s.

One limit: servers before PR #1006 (0.3.11 and earlier) refuse a request once a resent history holds more decoded
images than 64 MiB (about 23 screenshots at 1280×720 [derived]). Use smaller screenshots there, or a build after
#1006, which serves resent images from a cache.
