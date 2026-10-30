# Starting bloomery from an agent

For a program or an agent that starts `bloomery-serve` and then calls its API. stdout is machine-readable: JSON,
one object a line. stderr is text for people; do not parse it. On a Mac the server is `bloomery-serve-metal`, which
has none of this yet: read [On a Mac](#on-a-mac) instead. To run Claude Code itself on the local model, read
[`claude-code.md`](claude-code.md).

## Rules

1. Start with no `--place`, `--ctx-size` or `--parallel` and no `BLOOMERY_*` variable. The server picks the cards,
   the split and the context itself. Set them only when the user asks.
2. A first start downloads the model. `download` (bytes) in the pick says how much; ask the user first when it is
   large. `--dry-run` downloads nothing.
3. A gated repo needs `HF_TOKEN` in the environment.
4. The exit status is the class of refusal, and the `error` event says what to run instead (`next`).

## On a Mac

Apple Silicon runs `bloomery-serve-metal` ([`mac.md`](mac.md)). It has no `--recommend`, no start by name and no
events, so the steps differ:

1. Install with `brew install midagedev/tap/bloomery`, or with the installer line in the README. If Homebrew stops
   with "Your Command Line Tools are too outdated", use the installer: it needs no build tools.
2. Pick a folder from [mac.md's table](mac.md#which-files-run). On a 32 GB Mac, the coding pick is
   `mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit` (17.2 GB to download). Ask the user before a download that
   large.
3. Start it in the background, with stdout and stderr to a file:

       bloomery-serve-metal --hf mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit --port 8080 > serve.log 2>&1 &

4. Poll `GET /health` until it answers 200. Requests sent before that fail. If the process exits first, `serve.log`
   says why. The exit status is 64 for a flag or value it refuses, 70 when the engine ended the run, and 1 for any
   other refusal.
5. `--ctx` unset is the smaller of 32,768 and the model's own context. Raise it only when the user asks, or when the
   client is Claude Code ([`claude-code.md`](claude-code.md)).

A long request is not a stuck one: the server prints one stderr line when each request ends and a `generating` line
while a long one runs, so `serve.log` tells an agent it is still working.

Do not pass `--place`, `--parallel` or `-ngl`: the Mac server refuses them with 64.

## 1. What this machine serves

    bloomery-serve --recommend --json

One JSON object on stdout, exit 0, schema `bloomery.recommend/1`:

    {"schema":"bloomery.recommend/1","machine":{...},"models":[{"name":"...","seat":"...","pick":{...}|null,"rejected":[...]}]}

`machine` names the build, the cards (`ordinal`, `name`, `cc`, `integrated`, `total`, `free`), the host room and the
cache directory with its free bytes. Each model has a `pick`, the variant this machine runs fastest, or `null` when
none fits. A pick is:

| Field | Meaning |
|---|---|
| `variant` | the catalog variant; `--model NAME:VARIANT` starts it |
| `source` | the `--hf REPO:QUANT` the start downloads |
| `download` | bytes still to download; 0 when the file is cached |
| `cached` | a verified copy is in the cache |
| `place` | the placement the plan chose |
| `ctx`, `slots` | context per slot and slot count the start serves |
| `paged` | weight bytes paged from NVMe; 0 when the cards and host memory hold the model |
| `fit` | `planned`: the server's own plan ran on the file's header; `estimate`: decided from file size alone |

`rejected` lists every other variant with one `why` word: `explicit-only` (Q8_0, Q6_K and Q5_K_M start only when
named), `unverified` (no gate runs the file; the row has `"verified":false`), `not-in-repo`, `offline-uncached`,
`disk-short`, `plan:CODE` (the plan refused it), `host` or `paged` (it holds, but a lower-ranked variant holds
faster), `ranked-below` (a better-ranked variant was picked, and this one would not hold faster). A `ranked-below`
variant may not have been planned: the pick does not read the file list or the header of a variant that cannot beat
the one it holds. To see whether it fits, plan it by name: `bloomery-serve --model NAME:VARIANT --dry-run --json`.

The first pick of a model reads each planned file's header from the hub (a few MB a file, fetched in parallel; a
checkpoint folder also reads its tokenizer and index files, up to tens of MB) and keeps it in
`$BLOOMERY_CACHE/headers`, so the same pick again reads none. Give the first `--recommend --json` on a
machine with no cache a timeout of several minutes: on one workstation (an RTX A6000, 2026-10-29) it took 68 s, the
second one 8 s, and `--model NAME --dry-run --json` for one model 8 s and then 2 s.

## 2. Check one start

    bloomery-serve --model NAME[:VARIANT] --dry-run --json

Picks and plans, downloads nothing, exit 0. The same `machine` block, one `models` entry with every candidate, and
`"start":["--model","NAME:VARIANT",...]`, the arguments a real start would run: the same row at the same pinned
revision, so the start replays the pick. `NAME` is a catalog name (`qwen3.8-flash-next`), an alias or a seat word
(`qwen38`), in any case. When no variant holds, `pick` is `null` and there is no `start`; the `error` event follows
the document and the exit is the refusal's code (the one a real start would end with).

## 3. Start

    bloomery-serve --model NAME[:VARIANT] --port 0 > events.ndjson 2> serve.log &

Read stdout line by line. Every line has `"schema":"bloomery.event/1"`; refuse a schema you do not know.

| Event line | When |
|---|---|
| `{"event":"pick",...}` | a start by name chose a file: the pick block above |
| `{"event":"bound","url":"http://H:P","phase":"loading"}` | the port is bound and answers 503 `loading` |
| `{"event":"ready","url":"http://H:P","model":...,"ctx":...,"slots":...,"place":...}` | requests are served: send them now; `model` is the name the server answers to, and `/v1/models` lists it |
| `{"event":"error","exit":N,"code":"...","message":"...","hint":"...","next":[[...]]}` | before every non-zero exit |
| `{"event":"stopped","cause":"...","exit":N}` | the server ended |

**Wait for the `ready` line, or for `GET /health` to answer 200.** Not before. `cause` is `signal` (SIGINT or SIGTERM,
the orderly stop, exit 0), `shutdown` (`POST /shutdown`, exit 0), `engine` (exit 70) or `listener`.

Until `ready`, every route answers 503, and `GET /health` says where the start is:

    {"status":"loading","phase":"fetch"|"plan"|"load","bytes_done":N,"bytes_total":N,
     "error":{"code":503,"message":"Loading model","type":"unavailable_error"}}

`phase` is `fetch` (the download, with `bytes_done` of `bytes_total`), `plan` or `load`. After `ready`, `/health` is
200 `{"status":"ok",...}`.

A start by name serves under the catalog name it was asked by (`nemotron-3-nano-4b`, not the name the file states;
`--alias NAME` replaces it). A start with `-m PATH` or `--hf REPO` serves under the name its file states.

A catalog name is a start by name. A `--model` value that is an existing path is refused with 64 and a `next` that
names `-m`: a file you already have starts with `-m PATH` or `--hf REPO[:QUANT]`. The automatic pick never takes
Q8_0, Q6_K or Q5_K_M or a row with `"verified":false`; name one to start it (`--model NAME:Q8_0`).

Stop the server with SIGTERM to the pid you started, then wait for `stopped` and exit 0.

`bloomery-serve-ds41` and `bloomery-serve-qwen38`, the binaries of one seat each, speak the same events and exit
codes. A start by name is `bloomery-serve`'s alone.

## Exit codes

| Exit | Code | Meaning | Next |
|---|---|---|---|
| 0 | `ok` | Success, an orderly stop, or a request that only prints. | - |
| 1 | `refused` | A refusal no other code names yet. | read the message |
| 64 | `usage` | A flag, a value, a name or a combination the server refuses; a model whose every row is unverified or explicit-only, started by its name alone. | run with --help; the message names the flag or value; for a model with no automatic row, `next` lists `--model NAME:VARIANT` for each row |
| 65 | `unsupported` | A model or file no seat of the server serves, among them an `--embd-model` file of an architecture no side seat serves; a start by name whose every row the seat's plan refuses as unsupported. | run with --recommend --json for what this machine serves |
| 66 | `no-model` | No model was named, or the one named is not there. | name a model with --model, -m or --hf; the message lists what exists |
| 69 | `no-device` | The machine lacks what the server needs: a driver, a card or a CPU feature. | the message names the driver, device or CPU feature that is missing |
| 70 | `engine` | The engine ended the run. | restart the server; the crash block names the failure |
| 74 | `network` | The hub could not be reached and no cached copy stands in. | check the connection, or set HF_HUB_OFFLINE=1 to start from a cached variant |
| 75 | `retry` | A refusal that clears by itself: the hub is busy, or the load was held back. | retry later |
| 77 | `denied` | The hub refused the credentials the request carried. | set HF_TOKEN and accept the repo's licence on its page |
| 78 | `lever` | A `BLOOMERY_*` value the server refuses. | the message names the lever and the kind of value it takes |
| 80 | `no-fit` | The model does not fit this machine as asked. | choose the next smaller variant, a smaller context or fewer slots |
| 81 | `card-busy` | A fit refusal on a card another process holds. | retry when the process holding the card ends |
| 82 | `disk-short` | The cache's filesystem cannot take the download. | free disk space, or set BLOOMERY_CACHE to a larger drive |
| 83 | `port-in-use` | The address the server was asked to listen on is taken. | use --port 0 or another port |

The `error` event's `next` is a list of argument lists, each runnable as it stands after `bloomery-serve`, for example
`["--model","qwen3.8-flash-next:UD-Q3_K_XL"]` for a `no-fit`.

## Use the server

Take `url` from the `ready` event (`--port 0` gives a free port). OpenAI clients use `URL/v1` as the base URL. Claude
Code runs with `ANTHROPIC_BASE_URL=URL` ([`claude-code.md`](claude-code.md) has the full setup). The cache is `~/.cache/bloomery/hf`; `BLOOMERY_CACHE` moves it, and
`HF_HUB_OFFLINE=1` starts from a cached variant. Every flag and the API: [`serving.md`](serving.md). What each model
needs: [`models.md`](models.md).
