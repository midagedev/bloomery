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

Read from the code. Claude Code and dsh were also run against a live server; their entries say so.

- **Claude Code 2.1.314** sends its environment block as a `system` message after the first user message. bloomery
  0.3.11 passes the role through, and templates that take a system message only first (the Qwen3.6 and Qwen3.8
  families) answer 400 "System message must be at the beginning." Claude Code 2.1.312 did not send it. Fixed after
  0.3.11: such a message renders as a `user` message in its place (`crates/serve/src/anthropic.rs`, #995). Run live
  with the fix (Qwen3.8-Flash-Next UD-Q4_K_XL, RTX A6000 + RTX 3090): one task of 13 turns wrote a three.js page in
  122 s.
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
  served the DeepSeek-V4.1-Flash 1017 clip of 2026-10-19 on this server. Run live after 0.3.11 (DeepSeek-V4.1-Flash
  1017 Q3_K_M, RTX A6000, `--place` unset, the profile's reasoning effort `high`), `dsh headless` with a one-file
  three.js task: no error in 10 requests (the last ended `cancel` when the run's 900 s bound stopped dsh). The first full prompt was 5,483 ids in 18.78 s; every later turn reused the
  cache and sent 15 to 2,718 new ids. Generation ran at 40 to 43 ids/s. The first two answers were 16,997 and 13,206
  ids (398 s and 313 s), and the page was written 12 to 13 minutes after the server came up. At effort `high`, the wait is the model's
  answer length, not the harness's prompt.
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

## 4. Live sweep

Measured 2026-10-30/31, the same agents as §0 against a live server: bloomery commit `28c716d797df` (branch `hsweep`),
Qwen3-Coder-30B-A3B-Instruct Q4_K_S (`Qwen3-Coder-30B-A3B-Instruct-Q4_K_S.gguf`), one RTX 3090,
the whole load on the card (24.94 GB of 25.07 GB), ctx 57344 (65536 does not fit: the plan needs 25.75 GB), one slot,
every lever unset, port 18190. Prompt pass 4.4-6.2k ids/s on the big first prompts; decode 125-200 ids/s. Agent
versions as in §0 (dsh 0.2.0-rc.2 on the box, the others on the Mac through an ssh tunnel). Each agent ran each task
one-shot in a fresh project copy, a 15-minute bound.

- T1: "Create fizzbuzz.py that prints FizzBuzz for 1 to 15, run it with python3, and tell me the output." Pass = the
  file exists and its output is right.
- T2: a planted-bug package (`calc/`, two bugs in two files, 2 of 6 `unittest` tests failing): "Run the tests with
  python3 -m unittest, fix the bugs in the code (not the tests), and run them again." Pass = all tests pass and no
  test file changed.

| Agent | T1 | wall | requests | first ids | T2 | wall | requests | first ids |
|---|---|---|---|---|---|---|---|---|
| pi 0.86.1 | pass | 3.0 s | 3 | 1,706 | **fail** | 4.0 s | 7 | 1,068 (636 reused) |
| Claude Code 2.1.314 | pass | 9.1 s | 3 | 16,714 | pass at the bound | 901.1 s | 244 | 16,692 (22 reused) |
| Codex 0.153.4 | pass | 7.1 s | 3 | 8,954 | **fail** | 16.1 s | 14 | 289 (8,665 reused) |
| OpenCode 1.19.3 | pass | 14.1 s | 4 | 557, then 16,118 | pass | 24.2 s | 9 | 554, then 16,284 |
| Crush 0.89.0 | pass | 7.1 s | 3 | 11,691 | pass | 13.1 s | 10 | 7,481 (4,212 reused) |
| dsh 0.2.0-rc.2 | pass | 5 s | 4 | 142, then 6,010 | **fail** | 2 s | 2 | 139, then 6,007 |

"First ids" is the window's first served request; OpenCode and dsh send a title call first (§0), their main prompt is
the second number. Codex's T2 was a rerun whose conversation shape matched an aborted first attempt (§ below), so its
slot cache still held 8,665 ids of it. §0's `[derived]` first-request estimates run from exact (pi) to 17 % above
these measured counts (codex), the ÷3.5 rule slightly over-counting the coder's prose.

The three failures:

- **pi T2, class server.** The model wrote the edit tool's `edits` array as a Python literal
  (`[{'oldText': …}]`). The Qwen tool-call markup parser refuses a value that is not JSON where the schema typed it
  (`crates/serve/src/qwenxml.rs:158`, `NotJson`), the request ends 500 `tool-call markup: …` (`tool_markup_error`,
  `crates/serve/src/api.rs:4791`), and pi prints the error and exits. At temperature 0 the model writes JSON, so the
  failure needs the sampled run's emission to reproduce.
- **Codex T2, class server.** A markup refusal inside a streamed answer ends the SSE without `response.completed`
  (`crates/serve/src/responses.rs:1590-1602`, the same `tool_markup_error`): the server sends a bare
  `data: {"error": …}` frame with no `event:` line. Codex 0.153.4 reads only `event:` frames,
  so it never sees the error text; it reports "stream disconnected before completion", reconnects five times onto
  the same cached prompt, and fails. A replay of the request rebuilt from codex's session rollout reproduces it:
  `response.output_text.delta` frames, then the bare error frame, then the stream ends. Fixed after 0.3.11: such a
  stream ends with `response.failed` carrying the error (`crates/serve/src/responses.rs`). Against a stub that
  sends each ending, Codex 0.153.4 ran 630 s and never showed the cause on the bare frame; on `response.failed` it
  printed the server's message and stopped in 7 s (it still retries five times, its own policy).
- **dsh T2, class model.** The model emitted `<function=bash> … </tool_call>` without the opening `<tool_call>`
  tag. The parser never arms, the markup reaches dsh as plain text, no tool runs, and the headless run ends in 2 s
  with the tests unfixed. The server passed unrecognized text through, which is the
  defined behaviour.

What else the sweep showed:

- Claude Code's T2 fixes were correct and on disk 12 s in (exactly the reference fix), but it then invoked its
  built-in code-review skill, whose subagents ran the rest of the 900 s bound (244 requests). Its subagent
  conversations interleave on the one slot: after a 20,852-id reuse the next turn reprefilled 14,324 ids, and again
  15,678 a few requests later. An agent that interleaves conversations pays the full prompt pass again per switch.
- OpenCode resolves its working directory from `$PWD`, not from its cwd: with a stale `$PWD` it wrote `fizzbuzz.py`
  into an unrelated directory and the task failed on the spot. Point `$PWD` at the project when driving it
  headlessly.
- Claude Code warns "doesn't recognize the model … assumes a context window of 200k tokens" on a custom model id;
  cosmetic here.
- The slot's prompt cache survives across agents: pi's T2 first request reused 636 ids of T1's state.

The failures share one shape: a local model's tool-call markup discipline is imperfect, and the server's named
refusal — correct per the no-silent-failure contract — turns one bad emission into a failed request. An agent that
could read the refusal (pi does) or a stream that carried the refusal in a frame the agent parses (codex does not)
would let the model retry its own call.
