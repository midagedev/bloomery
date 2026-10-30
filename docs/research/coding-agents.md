# Coding agents on bloomery-serve: what each one sends first

A survey (2026-10-30, the lead's Mac and the box, no model run). The question: which coding agent fits a local `bloomery-serve`,
judged by the first request each agent sends. A large first request is a long first prompt pass on a slow machine, and
a field the server or the chat template refuses ends the session before it starts.

Every number is measured from a captured request body, or marked `[derived]` with its terms.

## 0. Result

| Agent | Endpoint | System prompt (chars) | Tools | First request (bytes) | Prompt ids `[derived]` | Extra calls on the first prompt |
|---|---|---|---|---|---|---|
| pi 0.86.1 | `/v1/chat/completions` | 2,715 | 4 | 6,181 | ~1,700 | none |
| DeepSeek Harness (dsh) 0.2.0-rc.2 | `/v1/chat/completions` | 2,592 (a `developer` message) | 24 | 22,776 | ~6,600 | one title call (796 B) |
| Codex 0.153.4 | `/v1/responses` | 19,841 (`instructions` 16,979 + a `developer` item 2,862) | 9 | 39,614 | ~10,800 | none |
| Crush 0.89.0 | `/v1/chat/completions` | 21,805 | 26 | 47,966 | ~13,700 | two title calls (~800 B each) |
| OpenCode 1.19.3 | `/v1/responses` | 44,581 | 10 | 65,781 | ~18,600 | one title call (2,648 B), sent first |
| Claude Code 2.1.314 | `/v1/messages` | 5,962 (3 blocks) + 30,674 in the first messages | 12 (21 from the second call) | 68,225 | ~19,300 | none |

Prompt ids `[derived]` = (system + message + tools-JSON characters) ÷ 3.5. The real count depends on the model's
tokenizer.

What it means for a first turn `[derived]`, as ids ÷ the prompt rate:
- On an M1 Pro running Qwen3-Coder-30B-A3B, Claude Code 2.1.312's first request was 17,410 ids in 91.7 s, measured
  (`docs/claude-code.md`, "What to expect"): about 190 ids/s. At that rate pi's first turn is ~9 s.
- On the RTX A6000 + RTX 3090 running Qwen3.8-Flash-Next (1,574 ids/s measured: 12,352 ids in 7.85 s over a Claude
  Code session), Claude Code's first turn is ~12 s.

## 1. Compatibility with bloomery-serve

Read from the code, not yet run against a live server:

- **Claude Code 2.1.314** sends its environment block as a `system` message after the first user message. bloomery
  0.3.11 passes the role through, and templates that take a system message only first (the Qwen3.6 and Qwen3.8
  families) answer 400 "System message must be at the beginning." Claude Code 2.1.312 did not send it. Fixed after
  0.3.11: such a message renders as a `user` message in its place (`crates/serve/src/anthropic.rs`, #995).
- **Codex 0.153.4** speaks only `/v1/responses` (`wire_api = "chat"` is refused by the CLI). Its `developer` item
  reaches the template as `system` (`crates/serve/src/api.rs`). Its tools include one of `type: "namespace"`
  (`multi_agent_v1`, sub-agent functions nested inside it). bloomery 0.3.11 runs function tools only and refuses any
  other type with 400, so Codex fails on its first request. Fixed after 0.3.11: the functions inside a namespace become
  function tools, and a call to one answers with the `namespace` field Codex reads (`crates/serve/src/responses.rs`).
- **OpenCode 1.19.3**, through its `openai` provider, speaks `/v1/responses`: a `system` item first, `input_text`
  parts, `store: false`, `prompt_cache_key`. The server reads or ignores each of these (`crates/serve/src/responses.rs`,
  module doc). Its title call names its own small model (`gpt-5.4-nano`); the server echoes a model name and serves its
  one model.
- **Crush 0.89.0** speaks `/v1/chat/completions`: `system`, then two `user` messages (a to-do reminder is the second).
  Its title calls name `gpt-6-luna` unless a small model is configured; as above, the name is echoed.
- **DeepSeek Harness (dsh) 0.2.0-rc.2** (npm `@deepseek-ai/dsh`, a web UI with `dsh web`, one task with `dsh headless`)
  speaks `/v1/chat/completions` through a `pi-ai` provider: a `developer` message first, then two `user` messages,
  `chat_template_kwargs` (`enable_thinking`, `reasoning_effort` from the profile), `max_completion_tokens`,
  `stream_options`. The `developer` message reaches the template as `system` (`crates/serve/src/api.rs`). It
  served the DeepSeek-V4.1-Flash 1017 clip of 2026-10-19 on this server.
- **pi** speaks plain `/v1/chat/completions`: `system`, `user`, four tools (`read`, `bash`, `edit`, `write`),
  `stream_options`. Nothing in it is new to the server.

Every agent streams. None puts a `system` item after another message in its first request, except Claude Code.

## 2. Which agent for which machine

- **A slow machine (a Mac):** pi. Its first request is about a tenth of the others', and a first turn on a slow
  prompt pass is the wait a user notices.
- **DeepSeek-V4.1-Flash:** dsh. It is DeepSeek's own harness, its first request is the second smallest here, and
  its web UI shows the work as it happens.
- **A fast card set, a demo:** Claude Code (bloomery after 0.3.11) or OpenCode. Both carry many tools and long system
  prompts, which a fast prompt pass absorbs.
- **Codex:** bloomery after 0.3.11 (its `namespace` tool).
- **Pin the agent's version for a demo or a script.** Agents change their request between patch releases. Claude
  Code changed the role of its environment block from 2.1.312 to 2.1.314, and that change alone broke a rehearsed
  session.

## 3. Method

- A stub HTTP server on `127.0.0.1:18099` recorded each POST's path and JSON body and answered 400.
- Each agent ran from an empty directory with a throwaway `HOME`, pointed at the stub with key `local` and model
  `local`. Prompt: `Make index.html: a three.js page with a rotating cube.`
- Configuration, one line each:
  - pi: `~/.pi/agent/models.json` with a provider of `api: "openai-completions"`; `pi --offline --model local/local -p '…'`.
  - Codex: `config.toml` with `model_providers.<name>.base_url`; `codex exec --skip-git-repo-check '…'`.
  - Crush: `~/.config/crush/crush.json` (a legacy `config.json` is ignored without a warning); `crush run '…'`.
  - OpenCode: `~/.config/opencode/opencode.json`, provider `openai` with `options.baseURL`; `opencode run -m openai/local '…'`.
  - dsh: a `headless` profile whose `profiles/headless/cordis.patch.yml` declares a `pi-ai` provider (`api:
    openai-completions`, `baseURL`, the model) and makes it the default model; `DSH_HOME=<dir> dsh headless '…'`.
    Captured on the box (Linux, node 26.8.2); the others on the Mac.
  - Claude Code: `ANTHROPIC_BASE_URL`, `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_MODEL` (`docs/claude-code.md`), a fresh
    `CLAUDE_CONFIG_DIR`; `claude -p '…' --permission-mode default`.

Not measured: a live session on a model (tool-call success, later turns), and the streamed answers (the stub answered
400 to everything).
