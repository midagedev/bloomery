# Building and running bloomery

This page is for building on your own Linux host. The maintainers build on one workstation through `tools/box.sh`, which copies the tree there and runs the command inside the quotes. On your own host, run the quoted command directly from the repository root.

## Toolchain

| Piece | Version | Where it is pinned |
|---|---|---|
| Rust | `nightly-2026-08-28`, with `rust-src`, `rustc-dev`, `llvm-tools`, `clippy`, `rustfmt` | `rust-toolchain.toml` |
| cuda-oxide (`cuda-device`, `cuda-host`) | git rev `b9847e9515ed3a23096f22567d3eaf0a6e3e440c` | `[workspace.dependencies]` in `Cargo.toml` (NVlabs), source from the `[patch]` fork `midagedev/cuda-oxide` branch `bloomery`; `just deny` fails if either floats. `cargo oxide` builds its backend from that checkout |
| `cargo-oxide` | 0.2.1 | install it as cuda-oxide's README says; `cargo oxide doctor` must pass |
| LLVM | 21.1.8 (`llc`), from the LLVM release tarball | on `PATH` |
| Clang | 21, for bindgen | on `PATH` |
| CUDA toolkit | 13.3 | `CUDA_TOOLKIT_PATH` |
| NVIDIA driver | 615.71.09 on the development machine | — |
| GPU | sm_86 (RTX 3090, RTX A6000) | `--arch sm_86` |
| CPU | x86-64 with AVX2 and FMA | `.cargo/config.toml` |

The nightly moves only when the cuda-oxide pin moves. Other CUDA versions, drivers and GPU architectures are not tested.

An environment like this one works on the development machine. Adjust the paths to your install:

```sh
export PATH=$HOME/.cargo/bin:$HOME/opt/LLVM-21.1.8-Linux-X64/bin:/usr/local/cuda-13.3/bin:$PATH
export CUDA_TOOLKIT_PATH=/usr/local/cuda-13.3
cargo oxide doctor
```

## The two builds

**The GPU build** needs the CUDA codegen backend. Never build a crate with device code (`crates/gpu`, `crates/gpu-deepseek41`, `crates/q3k-gemv`, or `bloomery-gpu-gates` with `--features gpu` or `deepseek41`) with plain `cargo build`. The result is not usable. Use `cargo oxide`:

```sh
cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features deepseek41 --release --bin generate_ds41
```

This builds the V4.1 CLI, the GPU kernels and the host expert tier into one binary, `target/release/generate_ds41`.

**The CPU tier** is compiled with `-C target-cpu=znver3`. `.cargo/config.toml` sets it for `x86_64-unknown-linux-gnu`, so every release build on Linux gets AVX2 and FMA. The bit gates all run under this flag. The kernels also carry their own `#[target_feature]` attributes, so they stay correct if you override `RUSTFLAGS`. A host that is not Zen 3 has not been tested.

`just check`, `just lint` and the `just gate-*` recipes are the maintainers' gates. They go through `tools/box.sh`; see `AGENTS.md` for what each one runs.

## The model file

The engine runs the published Q3_K_M upload as it is, one GGUF set in 9 shards:

```
DeepSeek-V4.1-Flash-Q3_K_M/DeepSeek-V4.1-Flash-Q3_K_M-00001-of-00009.gguf
```

Every tensor keeps the upload's type: the attention, shared expert and engram tensors and the token embedding are K-quants like the routed experts (Q3_K, Q4_K, Q5_K; the output head Q6_K), and each kernel is picked by the type the file gives the tensor (`docs/facts.md` has the type table). The maintainers' gates run on this file by default. A second set, the same upload with the attention and shared expert tensors and the engram tables in Q8_0 and the token embedding in BF16, is the file the engine was first developed against; its reference dumps are kept and `BLOOMERY_V41_MODEL` still selects it; the engine does not need it.

Point the engine at shard 1; the loader follows the split count from there:

```sh
export BLOOMERY_REF_MODEL=/path/to/DeepSeek-V4.1-Flash-Q3_K_M-00001-of-00009.gguf
```

`generate_ds41` has no default model path and stops if this variable is unset.

## The data directory

`$BLOOMERY_DATA` holds everything that is not the model and not the source tree. Its default is `/root/bloomery-data`. The parts `generate_ds41` can read:

| Path under `$BLOOMERY_DATA` | What | Made by |
|---|---|---|
| `router/hotlist-384.txt` | per layer, the routed experts ranked by how often the router picks them | `just trace-router <corpus>` for each corpus, then `just hotlist` |
| `greedy-ds41/prompt<P>.tsv` | the token ids of prompt row P | `just ik-greedy-ds41 <P>` (needs ik_llama.cpp) |
| `engram/corpus-*.ids` | token-id corpora for the draft gate and the router trace | not in this repository |
| `ref_deepseek41/` | the oracle tensor dumps the gates read | `tools/ref/dump.sh` (needs ik_llama.cpp) |

None of these is needed for a plain run with `--tokens`.

## Running `generate_ds41`

`generate_ds41` takes token ids and prints token ids, greedy, one token per step; `bloomery-chat` (below) takes text and streams text through the file's own tokenizer and the sampling chain.

```sh
BLOOMERY_REF_MODEL=... target/release/generate_ds41 --place gate --tokens 671,6102,294,8760,344 -n 32
```

| Flag | Meaning |
|---|---|
| `--tokens a,b,c` | the prompt as token ids; the example is "The capital of France is" without a BOS token (or `--prompt-id P` for a row under `$BLOOMERY_DATA/greedy-ds41/`) |
| `-n N` | tokens to generate (default 32) |
| `--place a` | plan (a): all layers and the head on a card named `A6000`, each routed layer's expert prefix the budget allows, the rest on the host |
| `--place gate` | the same shape on a card named `3090` |
| `--ctx C` | context length (default 32,768); a call past position 16,384 is refused by name — from there the V4.1 reference picks each index top-k inside a two-level candidate mask, which bloomery does not build |
| `--depth D` | feed a fixed synthetic id sequence to depth D before generating; used by the timing tables |
| `--time` | per-step timing; a measurement, which the maintainers only run under the machine-wide lease |

The placement plans are built from this machine's two cards and its RAM (`crates/model/src/placement/workstation.rs`). The card is found by its CUDA device name, so `--place gate` needs a device whose name contains `3090` and `--place a` one that contains `A6000`.

The default placement is `a`. The output has a `load` line (thread pinning, the indexer's `top_k`) and a `tokens` line with the generated ids.

`bloomery-chat` takes text instead of ids (build it with `--bin bloomery-chat` in the command above): `BLOOMERY_REF_MODEL=... target/release/bloomery-chat --place gate --prompt "The capital of France is" -n 64` streams the continuation to stdout, sampled at the reference defaults (`--temp`, `--top-k`, `--top-p`, `--min-p`, `--repeat-penalty`, `--seed`) or with `--greedy`, and applies no chat template.

### HTTP server (`bloomery-serve`)

On the V4.1 engine the server is `bloomery-serve-ds41` (build it with `--bin bloomery-serve-ds41` in the `cargo oxide build` command above): `BLOOMERY_REF_MODEL=... target/release/bloomery-serve-ds41 --host 127.0.0.1 --port 8080` loads the model (the `plan`, `load` and `capture` lines go to stderr, then `listening on http://…`) and serves it with the file's own vocabulary and chat template. `--place a|gate` (default `a`, the serving plan on the A6000) and `--ctx C` are `generate_ds41`'s; `--alias NAME` overrides the model name. At `temperature` 0 a request takes the engine's argmax and its ids are the ones `generate_ds41 --tokens <the prompt's ids>` prints (`"return_tokens": true` on `/completion` returns them); above 0 the sampler crate's chain draws, with no repetition penalty. An engine error ends the server: that request gets a 500 with the engine's message, `/health` answers 503 with a `reason` for two seconds, then the process prints the card, the position and the error to stderr and exits with code 70. `bloomery-serve --host 127.0.0.1 --port 8080 [--model first.gguf]` is the same server on a mock engine, for the API's own gate. Both speak llama-server's API: `POST /v1/chat/completions`, `/completion`, `/tokenize`, `/detokenize`, `/apply-template`, and `GET /v1/models`, `/health`, `/props`, `/slots`, `/metrics`, with llama-server's field names and defaults. `/props` also carries an `engine` object in toktape's shape: `name` (`bloomery`), `version` (the serve crate's version and the commit it was built from, `unknown` in a tree without git, with `mock` appended on the mock engine), the process's `args` and `server_pid`, and on V4.1 the model file (`arch`, `quant` for the type that holds most of the bytes a step reads, `bytes`, `files`, `n_layers`, the expert counts, `ctx_train`) and the placement plan's resident bytes per device (`GPU<n>` by its nvidia-smi index, then `CPU`) and tensor class with the cards' `vram_kv_bytes`, from which toktape takes its ENGINE line (`bloomery <version>`), its flags, the MODEL line and the placement view. The chat prompt is rendered from the GGUF's `tokenizer.chat_template` (`--model` reads the header only), whose parser takes Jinja slices with Python's rules (`a[start:stop:step]`, any part omitted, negative bounds counted from the end, as Qwen3's `messages[::-1]`) along with `split`, `strip`/`lstrip`/`rstrip` with a character set and the `true`/`false` tests. There is one slot: a second generation waits until the first finishes; `/tokenize`, `/detokenize`, `/apply-template`, `/health`, `/props`, `/slots` and `/metrics` answer at once. Streams: `/v1/chat/completions` ends with `data: [DONE]`; `/completion` ends with a chunk carrying `"stop": true` and `timings`, and no `[DONE]`, as llama-server does. `cache_prompt` (default true) keeps the longest prefix of the prompt the cache already holds and reports its length in `timings.cache_n`; `cache_prompt: false` evaluates the whole prompt. On V4.1 the kept prefix is cut to an even length unless it ends one short of the cached length (the ratio-2 compressor pools two positions per row), and the window rows below the cut come back from a shadow copy, so a request whose ids repeat the cached ones evaluates only what follows them, plus at most one position. `/v1/chat/completions` splits V4.1's think span into `reasoning_content`; `reasoning_format: "none"` returns the raw text instead, and `chat_template_kwargs: {"enable_thinking": true}` turns thinking on (the default is off). With `tools`, the DSML the model writes after the think span becomes OpenAI `tool_calls` (`arguments` a JSON string, one id per call, `finish_reason: "tool_calls"`); `tool_choice` accepts `auto` and `none`, and `required` or a named function is a 400. A field the server cannot honor is refused with a 400 that names it: `n_probs`, `response_format` other than `{"type":"text"}`, `json_schema`, `grammar`, `logprobs`, `top_logprobs`, `n > 1` and `tool_choice` other than `auto`/`none`. In `/metrics`, `prompt_tokens_seconds` and `predicted_tokens_seconds` are averages since the server started (a scrape does not reset them), and `kv_cache_usage_ratio` is the last request's `n_past` over the context. `/v1/models` reports the process start as `created`. `timings` are the server's wall clock around engine calls, for bench clients; they are not the project's measured numbers, which come only from the lease runners.

### The levers that matter

| Variable | Effect |
|---|---|
| `BLOOMERY_HOT_LIST=$BLOOMERY_DATA/router/hotlist-384.txt` | each routed layer's card keeps the list's first `n_l` experts instead of the id prefix. Same count and bytes. It was 4.5 ms per step faster (A6000, placement (a), depth 6, `n = 96`, instrumentation off; rig-log 2026-09-24) |
| `BLOOMERY_DRAFT=lookup` | the n-gram lookup draft through the skewed two-row pass. The `tokens` line equals the plain run's. Needs `--ctx` to hold one extra position. Prints a `draft summary` line |
| `BLOOMERY_STEP_STATS=1` | a `stat step` line per step. It slows each step by 2.5–3 ms, so compare instrumented runs only with instrumented runs |
| `BLOOMERY_PIN_MAIN=0` | leave the main thread unpinned |

The full list of runtime levers is at the end of `AGENTS.md`.
