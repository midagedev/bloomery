# Claude Code on a local model

bloomery serves the Anthropic Messages API (`/v1/messages`, with tool use and streaming) beside the OpenAI one. So
Claude Code runs on a model your own machine serves, on an NVIDIA card or on a Mac. This page starts the server, points
Claude Code at it, and says what to expect.

An agent can do all of it for you. Tell Claude Code (still on its usual model) "install bloomery, start a local coding
model on this machine, and write me a script that runs Claude Code on it". It reads this page and
[`agents.md`](agents.md), and writes the two scripts [below](#a-launcher). That is how the Mac rows in
[What to expect](#what-to-expect) were set up.

## 1. Start a server

**NVIDIA (Linux, WSL2).** Pick a model and start it by name, as [`agents.md`](agents.md) describes:

```sh
bloomery-serve --recommend --json            # what this machine serves, the fastest variant of each
bloomery-serve --model NAME --port 8080      # wait for the "ready" event on stdout
```

**Apple Silicon.** The server is `bloomery-serve-metal` ([`mac.md`](mac.md)). For coding on a 32 GB Mac, start
Qwen3-Coder-30B-A3B's MLX folder:

```sh
bloomery-serve-metal --hf mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit --ctx 49152 --port 8080
```

Wait until `GET /health` answers 200. The first start downloads the folder (17.2 GB), so give it minutes. A start after
that opens it from the cache in seconds. `--ctx` unset is 32,768 positions. That holds Claude Code's first request,
about 17,500 ids, but a long session outgrows it; 49,152 opened and served on a 32 GB M1 Pro.

## 2. Point Claude Code at it

```sh
unset ANTHROPIC_API_KEY
ANTHROPIC_BASE_URL=http://127.0.0.1:8080 \
ANTHROPIC_AUTH_TOKEN=local \
ANTHROPIC_MODEL=local \
ANTHROPIC_SMALL_FAST_MODEL=local \
CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 \
claude --permission-mode default
```

- `ANTHROPIC_BASE_URL` sends every request to the local server. Nothing leaves the machine.
- `ANTHROPIC_AUTH_TOKEN` must be set, and the server does not check it. If the server runs with `--api-key KEY`, use
  that key here instead.
- `ANTHROPIC_MODEL` and `ANTHROPIC_SMALL_FAST_MODEL` can be any name: the server serves its one model whatever name a
  request carries. Claude Code prints `unrecognized_model` for a name it does not know; the warning is harmless.
- **Do not use Claude Code's `auto` permission mode.** It asks the model to judge every command before it runs, and a
  local model can fail that check. With LFM3-1.7B, every Bash call came back blocked. Use `default` (Claude Code asks
  you) or `acceptEdits`.
- Work in a directory whose path has no spaces. Otherwise the model writes `cd "<path>" && …`, and Claude Code asks
  you to approve each such command.

## What to expect

Measured on an Apple M1 Pro (32 GB, macOS 27.0) on 2026-10-30. Conditions:
- bloomery 0.3.11 from Homebrew, Claude Code 2.1.312.
- The task, one `claude -p` run: "Create fizzbuzz.py that prints FizzBuzz for 1 to 15, run it with python3, and tell
  me the output."
- A prompt time is the server's own `prompt call` line.

| Model | Server open | First request (Claude Code's system prompt and tools) | Later requests | The task |
|---|---|---|---|---|
| Qwen3-Coder-30B-A3B, MLX 4-bit, `--ctx 49152` | 11.2 s from the cache | 17,410 ids in 91.7 s | 60 to 169 ids, 0.7 to 1.7 s | correct output, 128 s in all |
| LFM3-1.7B, MLX 4-bit | 0.5 s | 17,496 ids in 20.9 s | about 90 ids, 0.27 s | 88 s in all; its script printed `Fizz 3` instead of `Fizz` |

The first request is slow and the rest are fast. The Mac server keeps the prompt it last read, so each later turn reads
only its new messages. A request that does not start with that prompt, such as a second Claude Code session, reads
from position 0 again. A 1.7B model answers fast but gets small tasks wrong. Use a coding model of the size your
memory holds.

## When a turn takes long

A long turn is usually a long generation, not a hang. The server says so itself: it prints one stderr line when a
request ends, and a `generating` line every 30 s while one runs past that — `serve.log`, where the launcher's first
script writes the server's output. Between the lines, `GET /slots` (`n_past`, `n_decoded`) and `GET /metrics`
(`predicted_tokens_seconds`) show the request still running. The lines' fields are in
[`serving.md`](serving.md).

## A launcher

Two short scripts keep the shell's own settings untouched: one starts the server and waits for it, the other runs
Claude Code on it.

```sh
#!/usr/bin/env bash
# local-model.sh: start the server once and wait until it answers.
set -euo pipefail
port=${PORT:-8080}
curl -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1 && exit 0
nohup bloomery-serve-metal --hf mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit \
  --ctx 49152 --port "$port" > serve.log 2>&1 &
pid=$!
echo "$pid" > serve.pid
until curl -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1; do
  kill -0 "$pid" 2>/dev/null || { tail -n 20 serve.log >&2; exit 1; }
  sleep 5
done
```

```sh
#!/usr/bin/env bash
# claude-local.sh: run Claude Code on the local server.
set -euo pipefail
port=${PORT:-8080}
unset ANTHROPIC_API_KEY
export ANTHROPIC_BASE_URL="http://127.0.0.1:$port" ANTHROPIC_AUTH_TOKEN=local
export ANTHROPIC_MODEL=local ANTHROPIC_SMALL_FAST_MODEL=local CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
exec claude --permission-mode default "$@"
```

On an NVIDIA machine, replace the `bloomery-serve-metal …` line with `bloomery-serve --model NAME --port "$port"`.
