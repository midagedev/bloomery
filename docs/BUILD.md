# Building and running bloomery

This page is for building on your own Linux host. The maintainers build on one workstation through `tools/box.sh`, which copies the tree there and runs the command inside the quotes. On your own host, run the quoted command directly from the repository root.

## Toolchain

| Piece | Version | Where it is pinned |
|---|---|---|
| Rust | `nightly-2026-08-28`, with `rust-src`, `rustc-dev`, `llvm-tools`, `clippy`, `rustfmt` | `rust-toolchain.toml` |
| cuda-oxide (`cuda-device`, `cuda-host`) | git rev `b9847e9515ed3a23096f22567d3eaf0a6e3e440c` | `[workspace.dependencies]` in `Cargo.toml`; `just deny` fails if it floats |
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

The engine is developed against one GGUF set in 9 shards:

```
DeepSeek-V4.1-Flash-Q3_K_M-engramQ8-tokembdBF16-attnQ8/DeepSeek-V4.1-Flash-Q3_K_M-00001-of-00009.gguf
```

It is the published Q3_K_M upload with these tensors replaced:

- the engram tables in Q8_0 (209.2 GB, about 195 GiB), from the uploader's Q8_0 build,
- the attention and shared expert tensors in Q8_0, from the same build,
- the token embedding in BF16.

The routed experts stay as in the Q3_K_M upload (Q3_K, Q4_K, Q5_K). The engine's GPU attention path reads Q8_0, so a plain Q3_K_M file, whose attention tensors are Q3_K, is not known to load yet. Whether the mix is worth it against the plain file is an open experiment. The conversion command will be documented in a later round.

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

Today the CLI takes token ids and prints token ids. Decoding is greedy, one token per step. There is no tokenizer in the engine yet.

```sh
BLOOMERY_REF_MODEL=... target/release/generate_ds41 --place gate --tokens 671,6102,294,8760,344 -n 32
```

| Flag | Meaning |
|---|---|
| `--tokens a,b,c` | the prompt as token ids; the example is "The capital of France is" without a BOS token (or `--prompt-id P` for a row under `$BLOOMERY_DATA/greedy-ds41/`) |
| `-n N` | tokens to generate (default 32) |
| `--place a` | plan (a): all layers and the head on a card named `A6000`, each routed layer's expert prefix the budget allows, the rest on the host |
| `--place gate` | the same shape on a card named `3090` |
| `--ctx C` | context length (default 32,768) |
| `--depth D` | feed a fixed synthetic id sequence to depth D before generating; used by the timing tables |
| `--time` | per-step timing; a measurement, which the maintainers only run under the machine-wide lease |

The placement plans are built from this machine's two cards and its RAM (`crates/model/src/placement/workstation.rs`). The card is found by its CUDA device name, so `--place gate` needs a device whose name contains `3090` and `--place a` one that contains `A6000`.

The default placement is `a`. The output has a `load` line (thread pinning, the indexer's `top_k`) and a `tokens` line with the generated ids.

### HTTP server (`bloomery-serve`)

Today the server runs on a mock engine; the binding to the real engine is the next round. `bloomery-serve --host 127.0.0.1 --port 8080 [--model first.gguf]` speaks llama-server's API: `POST /v1/chat/completions`, `/completion`, `/tokenize`, `/detokenize`, `/apply-template`, and `GET /v1/models`, `/health`, `/props`, `/slots`, `/metrics`, with llama-server's field names and defaults. The chat prompt is rendered from the GGUF's `tokenizer.chat_template` (`--model` reads the header only). There is one slot: a second generation, and any tokenize call, waits until the first finishes; `/health`, `/props`, `/slots` and `/metrics` answer at once. Streams: `/v1/chat/completions` ends with `data: [DONE]`; `/completion` ends with a chunk carrying `"stop": true` and `timings`, and no `[DONE]`, as llama-server does. `cache_prompt` is accepted and ignored (every request evaluates its whole prompt). A field the server cannot honor is refused with a 400 that names it: `n_probs`, `response_format` other than `{"type":"text"}`, `json_schema`, `grammar`, `logprobs`, `top_logprobs`, `n > 1`, `tools` and `tool_choice`. In `/metrics`, `prompt_tokens_seconds` and `predicted_tokens_seconds` are averages since the server started (a scrape does not reset them), and `kv_cache_usage_ratio` is the last request's `n_past` over the context. `/v1/models` reports the process start as `created`. `timings` are the server's wall clock around engine calls, for bench clients; they are not the project's measured numbers, which come only from the lease runners.

### The levers that matter

| Variable | Effect |
|---|---|
| `BLOOMERY_HOT_LIST=$BLOOMERY_DATA/router/hotlist-384.txt` | each routed layer's card keeps the list's first `n_l` experts instead of the id prefix. Same count and bytes. It was 4.5 ms per step faster (A6000, placement (a), depth 6, `n = 96`, instrumentation off; rig-log 2026-09-24) |
| `BLOOMERY_DRAFT=lookup` | the n-gram lookup draft through the skewed two-row pass. The `tokens` line equals the plain run's. Needs `--ctx` to hold one extra position. Prints a `draft summary` line |
| `BLOOMERY_STEP_STATS=1` | a `stat step` line per step. It slows each step by 2.5–3 ms, so compare instrumented runs only with instrumented runs |
| `BLOOMERY_PIN_MAIN=0` | leave the main thread unpinned |

The full list of runtime levers is at the end of `AGENTS.md`.
