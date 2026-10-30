# pi on a local model

pi is a coding agent with four tools (`read`, `bash`, `edit`, `write`) and a short system prompt: its first request
is about a tenth of the other surveyed agents', so it is the one a slow machine starts fastest
([`research/coding-agents.md`](research/coding-agents.md)). This page starts the server, points pi at it, and says
what to expect. For Claude Code instead, read [`claude-code.md`](claude-code.md); for the comparison of both and the
rest, read [`agents.md`](agents.md#which-coding-agent).

Install pi from npm (`@earendil-works/pi-coding-agent`; 0.86.1 is the version surveyed):

```sh
npm install -g @earendil-works/pi-coding-agent
```

## 1. Start a server

The same server as [`claude-code.md`](claude-code.md) §1. **NVIDIA (Linux, WSL2).** Pick a model and start it by
name, as [`agents.md`](agents.md) describes:

```sh
bloomery-serve --recommend --json            # what this machine serves, the fastest variant of each
bloomery-serve --model NAME --port 8080      # wait for the "ready" event on stdout
```

**Apple Silicon.** The server is `bloomery-serve-metal` ([`mac.md`](mac.md)). For coding on a 32 GB Mac, start
Qwen3-Coder-30B-A3B's MLX folder:

```sh
bloomery-serve-metal --hf mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit --ctx 49152 --port 8080
```

Wait until `GET /health` answers 200. The first start downloads the folder (17.2 GB), so give it minutes. `--ctx`
unset is 32,768 positions; that already holds pi's first request (~1,700 ids [derived]) many times over, so it is
enough to start with.

## 2. Point pi at it

pi reads its providers from `~/.pi/agent/models.json`. Add one provider for the server (the shape is pi's own, from
`docs/models.md` of the installed package):

```json
{
  "providers": {
    "local": {
      "baseUrl": "http://127.0.0.1:8080/v1",
      "api": "openai-completions",
      "apiKey": "local",
      "models": [
        { "id": "local", "contextWindow": 49152 }
      ]
    }
  }
}
```

- `apiKey` is a placeholder. bloomery does not check it, and pi lists a provider's models only when a key exists.
- `id` can be any name: the server serves its one model whatever name a request carries. The `id` is the label pi
  shows and the part after `/` in `--model`.
- `contextWindow` should be what the server serves — the `ready` event's `ctx`, or `--ctx` on a Mac — not pi's
  128,000 default, so pi keeps the session inside the real window.
- The file reloads each time you open `/model`; no restart.

Then run pi on it. `--offline` stops pi's startup network operations, so nothing leaves the machine:

```sh
pi --offline --model local/local
```

Without `--model`, pi starts on its default provider (google), not on this one. One task without the TUI:

```sh
pi --offline --model local/local -p 'Make index.html: a three.js page with a rotating cube.'
```

## What to expect

Measured on an Apple M1 Pro (32 GB) on 2026-10-30: bloomery 0.3.11 from Homebrew, pi 0.86.1, Qwen3-Coder-30B-A3B
MLX 4-bit with `--ctx` unset, the task of [`claude-code.md`](claude-code.md#what-to-expect) ("Create fizzbuzz.py that
prints FizzBuzz for 1 to 15, run it with python3, and tell me the output.") in one `pi -p` run. A prompt time is the
server's own `prompt call` line.

| Server open | First request | Second request |
|---|---|---|
| 10.1 s from the cache | 2,261 ids in 5.26 s | 265 new ids in 0.94 s |

Claude Code's first request on the same Mac and model was 17,410 ids in 91.7 s.

The task did not finish on 0.3.11. The model wrote `fizzbuzz.py` with one line under-indented, then called `edit` to
fix it, writing the `edits` array with a raw newline inside a string. 0.3.11 refuses that value as not JSON, answers
500, and pi stops. The release after 0.3.11 parses it (the newline is escaped), so the edit goes through.

pi's request itself holds nothing bloomery refuses: a system prompt of 2,715 characters, one `user` message, four
tools, 6,181 bytes in all, no extra calls ([`research/coding-agents.md`](research/coding-agents.md) §3).
