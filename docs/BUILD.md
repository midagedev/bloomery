# Building and running bloomery

This page is for building and running bloomery on your own Linux host.

The `just` recipes and `tools/box.sh` are the maintainers' tooling for one remote workstation. `tools/box.sh` copies the tree there over ssh and runs the recipe's command inside the quotes. Those commands take lock files under `/root` (a gate lock per card, a machine-wide timing lease), check the cards first and source an environment file of that machine, so they do not run on your host as written. This page gives the direct commands instead: `cargo oxide build …`, then `target/release/<bin>` with the model file named in an environment variable or a flag.

## Toolchain

| Piece | Version | Where it is pinned |
|---|---|---|
| Rust | `nightly-2026-08-28`, with `rust-src`, `rustc-dev`, `llvm-tools`, `clippy`, `rustfmt` | `rust-toolchain.toml` |
| cuda-oxide (`cuda-device`, `cuda-host`) | declared rev `ec4aa4797956534578a1af010f86252a0b6d8626` (upstream: NVlabs/cuda-oxide, now NVIDIA/cuda-rust), source rev `0af1016c72c2224857d02bead7bdb7cbc10e580b` | `[workspace.dependencies]` in `Cargo.toml` (NVlabs), source from the `[patch]` fork `midagedev/cuda-oxide` at that rev (the declared rev plus the patches `THIRD_PARTY_NOTICES.md` lists); `just deny` fails if either floats |
| `cargo-oxide` | 0.2.1, from the same fork and rev | the install command below; `cargo oxide doctor` must pass |
| LLVM | 21 (`llc` with the NVPTX target); the development machine uses 21.1.8 from the LLVM release tarball | on `PATH`, or `CUDA_OXIDE_LLC` |
| Clang | 21 with its resource headers (`clang-21` or `libclang-common-21-dev` on Ubuntu), for bindgen | on `PATH` |
| CUDA toolkit | 13.3, with the cuRAND headers (`libcurand-dev` on Ubuntu) | `CUDA_TOOLKIT_PATH` |
| NVIDIA driver | 615.71.09 on the development machine | — |
| GPU | sm_86: RTX A6000, RTX 3090 | `--arch sm_86` |
| CPU | x86-64; the build targets Zen 3 (see [The CPU flag](#the-cpu-flag)) | `.cargo/config.toml`, `.cargo/cuda-oxide.toml` |
| `just`, `python3` (3.11 or later) | any | only for the Mac checks and the scripts under `tools/`; the commands on this page need neither |
| `hf` (from `huggingface_hub`) | any | for the downloads below; older releases call it `huggingface-cli download` |

The nightly moves only when the cuda-oxide pin moves. Other CUDA versions, drivers and GPU architectures are not tested.

Install `cargo-oxide` from the fork at the pinned rev, not from upstream's main branch (NVIDIA/cuda-rust, formerly NVlabs/cuda-oxide):

```sh
cargo +nightly-2026-08-28 install --git https://github.com/midagedev/cuda-oxide.git \
  --rev 0af1016c72c2224857d02bead7bdb7cbc10e580b cargo-oxide
```

That `cargo-oxide` builds its codegen backend from the checkout cargo resolved for `cuda-device`, which is the `[patch]` fork at the pinned rev (the fork's `crates/cargo-oxide/src/backend_source.rs`). The maintainers' machine instead keeps one prebuilt backend per rev and points `CUDA_OXIDE_BACKEND` at it; you do not need that. The install above has not been run on a clean host yet.

An environment like this one works on the development machine. Adjust the paths to your install:

```sh
export PATH=$HOME/.cargo/bin:$HOME/opt/LLVM-21.1.8-Linux-X64/bin:/usr/local/cuda-13.3/bin:$PATH
export CUDA_TOOLKIT_PATH=/usr/local/cuda-13.3
cargo oxide doctor
```

cuda-oxide's README says the pipeline prefers the `llc` of the Rust toolchain, then `llc-23`, `llc-22` and `llc-21` on `PATH`; `CUDA_OXIDE_LLC=/path/to/llc` pins one. Which `llc` the development machine's builds pick is set in its environment file, which is not in this repository.

## The CPU flag

Both cargo configuration files force `-C target-cpu=znver3`:

- `.cargo/config.toml` sets it for `x86_64-unknown-linux-gnu` in plain `cargo` builds.
- `.cargo/cuda-oxide.toml` sets it for `cargo oxide` builds (`extra-rustflags`). cargo-oxide passes its own rustflags, which hide the first file's setting, so the flag needs this second owner.

Every bit gate runs under this flag. A host that is not Zen 3 has not been tested; on an older AMD or an Intel CPU a binary built for `znver3` may stop with an illegal instruction. To build for your own CPU, change the flag in **both** files, for example to `target-cpu=native`, and keep the two lists equal. `tools/check-rustflags.sh` (`just check-rustflags`) checks that the two lists are equal and that they still name `znver3`, so it fails after this change; that check is the maintainers'. The kernels carry their own `#[target_feature]` attributes and need AVX2 and FMA in any case.

## Building

Never build a crate with device code (`crates/gpu`, `crates/gpu-deepseek41`, `crates/gpu-glm5next`, `crates/gpu-vision`, or `bloomery-gpu-gates` with its `gpu`, `deepseek41`, `glm5next` or `vision` feature) with plain `cargo build`: the result is not usable. Use `cargo oxide`:

```sh
cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features deepseek41 --release --bin generate_ds41
```

This builds the V4.1 CLI, the GPU kernels and the host expert tier into one binary, `target/release/generate_ds41`. Each model section below names its bins and features; they are the feature sets the maintainers' recipes build with. Two binaries a stranger needs have no device code and build with plain `cargo`: `r8conv` (in `bloomery-model`) and `bloomery-tokenize` (in `bloomery-tokenizer`).

| Binary | Build | Models |
|---|---|---|
| `generate_ds41`, `bloomery-chat`, `bloomery-serve-ds41` | `cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features deepseek41 --release --bin <bin>` | DeepSeek-V4.1-Flash |
| `bloomery-serve-qwen38` | the same, `--features deepseek41` (the feature scopes the server code; it runs no V4.1 code) | Qwen3.8-Flash-Next |
| `bloomery-serve` | `… --features glm5next --release --bin bloomery-serve` (`--model ds41\|qwen38\|glm`; the ds41 and qwen38 seats are the two binaries above) | all three |
| `generate_qwen3moe` | `… --features gpu --release --bin generate_qwen3moe` | Qwen3.8-Flash-Next, Qwen3.6-35B-A3B, Qwen3-30B-A3B |
| `generate_glm5next` | `… --features glm5next --release --bin generate_glm5next` | GLM-5.3-Flash |
| `r8conv` | `cargo build --release -p bloomery-model --bin r8conv` | the V4.1 sidecar |
| `bloomery-tokenize` | `cargo build --release -p bloomery-tokenizer --bin bloomery-tokenize` | every model's text to ids and back |

Build artifacts land in the workspace root `target/`.

## Picking the card

The placement plans are built from the development machine's two cards and its RAM (`crates/model/src/placement/workstation.rs`). A binary finds a plan's card by its CUDA device name: `--place gate` needs one visible device whose name contains `3090`, `--place a` one whose name contains `A6000`, and `--place bp` both. Two visible devices that match the same name are an error (`2 visible devices are named like "3090"`), so on a host with two 3090s make one visible with `CUDA_VISIBLE_DEVICES`. Two 3090s together are not supported: nothing places a model or a draft across two cards of one name. A card named like `3090 Ti` also matches `3090`.

| Binary | `--place` takes | Default |
|---|---|---|
| `generate_ds41`, `bloomery-chat`, `bloomery-serve-ds41` | `a`, `gate`, `bp` | `a` |
| `generate_qwen3moe` on a Qwen3.8 file, `bloomery-serve-qwen38` | `a`, `gate`, `bp` | `a` |
| `generate_glm5next` | `a`, `gate`, `bp` | `gate` |
| `bloomery-serve --model glm` | `a`, `gate`, `bp` | `a` |
| `generate_qwen3moe` on a Qwen3.6 or Qwen3-30B file | refused | CUDA device 0 |

<!-- pending: default-place-a — every binary above but generate_glm5next defaults to --place a (a card named A6000), so on a host with only a 3090 pass --place gate; generate_glm5next is the opposite, its default is gate and an A6000-only host passes --place a -->
On a host with only a 3090, pass `--place gate` to every binary that takes `--place`. `generate_glm5next` is the one whose default is `gate`: on a host with only an A6000, pass it `--place a`. A Qwen3.6 or Qwen3-30B file runs on CUDA device 0; pick the card with `CUDA_VISIBLE_DEVICES`.

V4.1: under `--place gate` adaptive residency is off (`BLOOMERY_RESIDENCY` unset follows the placement: on under `a` and `bp`, off under `gate`).

## Levers

Runtime levers are `BLOOMERY_*` environment variables, the rows of `crates/levers/src/registry.rs`. Each binary reads them once at start and refuses by name a lever it does not act on, a value its kind does not take, and a `BLOOMERY_*` name no row names. So set a lever on the one command it is for, not with `export`. `<bin> --levers` prints what that binary parsed and exits; `just gate-levers` prints the whole table.

| Variable | Effect |
|---|---|
| `BLOOMERY_DRAFT=lookup` | V4.1: the n-gram lookup draft through the skewed two-row pass. The `tokens` line equals the plain run's. Needs `--ctx` to hold one extra position. Prints a `draft summary` line |
| `BLOOMERY_STEP_STATS=1` | a `stat step` line per step. It slows each step by 2.5–3 ms, so compare instrumented runs only with instrumented runs |
| `BLOOMERY_PIN_MAIN=0` | leave the main thread unpinned |

## Model files

Put each model's download in a directory of its own. The r8 sidecar of V4.1 is written beside that directory (below), and the loaders take the first shard's path and follow the split count from there. Each file is the public upload as downloaded; no conversion runs on it. Clef-Flash is the exception ([below](#clef-flash)).

| Model | Upload | First shard (under the download directory) | Size [derived: the upload's file sizes as the Hugging Face API lists them] |
|---|---|---|---|
| DeepSeek-V4.1-Flash `Q3_K_M` | `vcruz305/DeepSeek-V4.1-Flash-GGUF` | `DeepSeek-V4.1-Flash-Q3_K_M-00001-of-00009.gguf` | 347.3 GB in 9 shards |
| GLM-5.3-Flash `UD-Q4_K_XL` | `unsloth/GLM-5.3-Flash-GGUF` | `UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf` | 199.7 GB in 6 shards |
| Qwen3.8-Flash-Next `UD-Q4_K_XL` | `unsloth/Qwen3.8-Flash-Next-GGUF` | `UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf` | 111.3 GB in 4 shards, and the MTP draft `MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`, 2.8 GB |
| Qwen3.6-35B-A3B `Q4_K_M` | `lmstudio-community/Qwen3.6-35B-A3B-GGUF` | `Qwen3.6-35B-A3B-Q4_K_M.gguf` | 21.2 GB, one file |

## DeepSeek-V4.1-Flash

The engine runs the published Q3_K_M upload as it is. Every tensor keeps the upload's type: the attention, shared expert and engram tensors and the token embedding are K-quants like the routed experts (Q3_K, Q4_K, Q5_K; the output head Q6_K), and each kernel is picked by the type the file gives the tensor (`docs/facts.md` has the type table). A second set, the same upload with the attention and shared expert tensors and the engram tables in Q8_0 and the token embedding in BF16, is the file the engine was first developed against; its reference dumps are kept and `BLOOMERY_V41_MODEL` still selects it in the maintainers' tools; the engine does not need it.

```sh
hf download vcruz305/DeepSeek-V4.1-Flash-GGUF --include 'DeepSeek-V4.1-Flash-Q3_K_M-*' \
  --local-dir ~/models/DeepSeek-V4.1-Flash-Q3_K_M
M=~/models/DeepSeek-V4.1-Flash-Q3_K_M/DeepSeek-V4.1-Flash-Q3_K_M-00001-of-00009.gguf

cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features deepseek41 --release \
  --bin generate_ds41 --bin bloomery-chat --bin bloomery-serve-ds41

# the r8 sidecar (below): where it goes, then write it and check it
cargo build --release -p bloomery-model --bin r8conv
target/release/r8conv path "$M"
target/release/r8conv convert "$M" && target/release/r8conv verify "$M"

# token ids in, greedy token ids out
BLOOMERY_REF_MODEL="$M" target/release/generate_ds41 --tokens 671,6102,294,8760,344 -n 32
# text in, streamed text out
BLOOMERY_REF_MODEL="$M" target/release/bloomery-chat --prompt "The capital of France is" -n 64
# the HTTP server
BLOOMERY_REF_MODEL="$M" target/release/bloomery-serve-ds41 --host 127.0.0.1 --port 8080
```

<!-- pending: default-place-a -->
The commands above use the default `--place a`, an A6000. On a single RTX 3090, add `--place gate` to each of the three: the same plan shape on a card named `3090`, with fewer routed experts on the card. The binaries read the model file from `BLOOMERY_REF_MODEL`; they have no default path and stop if it is unset.

**The r8 sidecar.** The host expert tier reads its routed gate and up weights from a second file, the r8 sidecar, laid out for the CPU kernels: `<model dir>-r8/<name>-r8.gguf` beside the model directory, 155.7 GB. For the download above that is `~/models/DeepSeek-V4.1-Flash-Q3_K_M-r8/DeepSeek-V4.1-Flash-Q3_K_M-r8.gguf`; `r8conv path` prints it. `r8conv convert` writes it from the model file and `r8conv verify` checks it against the model file. It is on by default (`BLOOMERY_R8`). Without the file the engine reads the model file instead, with the same output bit for bit, and says so on its `load` line: `load host_tier r8=off (no sidecar at <path>: just r8-sidecar)`. A sidecar that does not match the model file is an error, never a fall-back. Budget the disk for both files. `r8conv convert` picks V4.1's routed stacks by itself; the other models' host tiers also read a sidecar when one is there, but this page does not cover writing one for them.

**Engram** needs no setup. V4.1's engram table is inside the GGUF file and is memory-mapped from it; keep the model files (and the sidecar) on NVMe.

### `generate_ds41`

`generate_ds41` takes token ids and prints token ids, greedy, one token per step; `bloomery-chat` takes text and streams text through the file's own tokenizer and the sampling chain.

| Flag | Meaning |
|---|---|
| `--tokens a,b,c` | the prompt as token ids; the example is "The capital of France is" without a BOS token (or `--prompt-id P` for a row under `$BLOOMERY_DATA/greedy-ds41/`) |
| `-n N` | tokens to generate (default 32) |
| `--place a` | plan (a), the default: all layers and the head on a card named `A6000`, each routed layer's expert prefix the budget allows, the rest on the host |
| `--place gate` | the same shape on a card named `3090` |
| `--place bp` | plan (b′): plan (a) on the A6000, and the 3090 as an expert tier under the host tier |
| `--ctx C` | context length (default 32,768); a call past it is refused by name. From position 16,384 on, each index top-k of layers 24, 28, 32 and 36 is taken inside the two-level candidate mask layer 20 ranks, as in the V4.1 reference |
| `--depth D` | feed a fixed synthetic id sequence to depth D before generating; used by the timing tables |
| `--time` | per-step timing. The binary does not require the lease; the maintainers' numbers come only from runs under it |

The output has a `load` line (thread pinning, the indexer's `top_k`) and a `tokens` line with the generated ids.

`bloomery-chat --prompt "…" -n N` streams the continuation to stdout, sampled at the reference defaults (`--temp`, `--top-k`, `--top-p`, `--min-p`, `--repeat-penalty`, `--repeat-last-n`, `--seed`) or with `--greedy`, and applies no chat template. `--prompt-file PATH` and `--prompt-stdin` take the prompt from a file or stdin; `--no-special` reads special-token text in the prompt as plain text. `-n` defaults to 256; `--place` and `--ctx` are `generate_ds41`'s.

**DSpark.** The DSpark draft (`BLOOMERY_DRAFT=dspark`, the file in `BLOOMERY_DSPARK_MODEL`) is not a public upload; the README says how it was converted. This page does not cover making it.

### HTTP server (`bloomery-serve-ds41`)

`BLOOMERY_REF_MODEL=… target/release/bloomery-serve-ds41 --host 127.0.0.1 --port 8080` loads the model (the `plan`, `load` and `capture` lines go to stderr, then `listening on http://…`) and serves it with the file's own vocabulary and chat template. `--place a|gate|bp` (default `a`, the serving plan on the A6000) and `--ctx C` are `generate_ds41`'s; `--alias NAME` overrides the model name; `--cache-ram MIB` bounds the host prompt cache (default 8192). At `temperature` 0 a request takes the engine's argmax and its ids are the ones `generate_ds41 --tokens <the prompt's ids>` prints (`"return_tokens": true` on `/completion` returns them); above 0 the sampler crate's chain draws, with no repetition penalty. An engine error ends the server: that request gets a 500 with the engine's message, `/health` answers 503 with a `reason` for two seconds, then the process prints the card, the position and the error to stderr and exits with code 70. `bloomery-serve --host 127.0.0.1 --port 8080 [--model first.gguf]` is the same server on a mock engine, for the API's own gate.

Both speak llama-server's API: `POST /v1/chat/completions`, `/completion`, `/tokenize`, `/detokenize`, `/apply-template`, and `GET /v1/models`, `/health`, `/props`, `/slots`, `/metrics`, with llama-server's field names and defaults. `/props` also carries an `engine` object in toktape's shape: `name` (`bloomery`), `version` (the serve crate's version and the commit it was built from, `unknown` in a tree without git, with `mock` appended on the mock engine), the process's `args` and `server_pid`, and on V4.1 the model file (`arch`, `quant` for the type that holds most of the bytes a step reads, `bytes`, `files`, `n_layers`, the expert counts, `ctx_train`) and the placement plan's resident bytes per device (`GPU<n>` by its nvidia-smi index, then `CPU`) and tensor class with the cards' `vram_kv_bytes`, from which toktape takes its ENGINE line (`bloomery <version>`), its flags, the MODEL line and the placement view. The chat prompt is rendered from the GGUF's `tokenizer.chat_template` (`--model` reads the header only), whose parser takes Jinja slices with Python's rules (`a[start:stop:step]`, any part omitted, negative bounds counted from the end, as Qwen3's `messages[::-1]`) along with `split`, `strip`/`lstrip`/`rstrip` with a character set and the `true`/`false` tests.

There is one slot: a second generation waits until the first finishes; `/tokenize`, `/detokenize`, `/apply-template`, `/health`, `/props`, `/slots` and `/metrics` answer at once. Streams: `/v1/chat/completions` ends with `data: [DONE]`; `/completion` ends with a chunk carrying `"stop": true` and `timings`, and no `[DONE]`, as llama-server does. `cache_prompt` (default true) keeps the longest prefix of the prompt the cache already holds and reports its length in `timings.cache_n`; `cache_prompt: false` evaluates the whole prompt. On V4.1 the kept prefix is cut to an even length unless it ends one short of the cached length (the ratio-2 compressor pools two positions per row), and the window rows below the cut come back from a shadow copy, so a request whose ids repeat the cached ones evaluates only what follows them, plus at most one position.

`/v1/chat/completions` splits V4.1's think span into `reasoning_content`; `reasoning_format: "none"` returns the raw text instead, and `chat_template_kwargs: {"enable_thinking": true}` turns thinking on (the default is off). With `tools`, the DSML the model writes after the think span becomes OpenAI `tool_calls` (`arguments` a JSON string, one id per call, `finish_reason: "tool_calls"`); `tool_choice` accepts `auto` and `none`, and `required` or a named function is a 400. A field the server cannot honor is refused with a 400 that names it: `n_probs`, `response_format` other than `{"type":"text"}`, `json_schema`, `grammar`, `logprobs`, `top_logprobs`, `n > 1` and `tool_choice` other than `auto`/`none`. In `/metrics`, `prompt_tokens_seconds` and `predicted_tokens_seconds` are averages since the server started (a scrape does not reset them), and `kv_cache_usage_ratio` is the last request's `n_past` over the context. `/v1/models` reports the process start as `created`. `timings` are the server's wall clock around engine calls, for bench clients; they are not the project's measured numbers, which come only from the lease runners.

## GLM-5.3-Flash

```sh
hf download unsloth/GLM-5.3-Flash-GGUF --include 'UD-Q4_K_XL/*' --local-dir ~/models/GLM-5.3-Flash
M=~/models/GLM-5.3-Flash/UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf

cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features glm5next --release --bin generate_glm5next
cargo build --release -p bloomery-tokenizer --bin bloomery-tokenize

# "The capital of France is", greedy, on a single RTX 3090 (the default --place gate)
target/release/generate_glm5next --model "$M" --tokens 785,6722,315,9621,374 -n 32
# on an A6000
target/release/generate_glm5next --model "$M" --place a --tokens 785,6722,315,9621,374 -n 32
```

<!-- pending: glm-model-flag — generate_glm5next reads the model from --model only, not BLOOMERY_REF_MODEL, and without --model opens /models/GLM-5.3-Flash-UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf -->
`generate_glm5next` takes the first shard with `--model`; it does not read `BLOOMERY_REF_MODEL`, and without `--model` it opens `/models/GLM-5.3-Flash-UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf`, the path the reference sets were dumped from.

It takes token ids only (no BOS is added) and prints a `tokens [a, b, …]` line. `bloomery-tokenize` turns text into ids and ids back into text with the file's own vocabulary:

```sh
IDS=$(target/release/bloomery-tokenize -m "$M" -p "The capital of France is" --ids --no-bos | tr -d '[] ')
target/release/generate_glm5next --model "$M" --tokens "$IDS" -n 32 > glm.out
target/release/bloomery-tokenize -m "$M" --decode -p "$(sed -n 's/^tokens //p' glm.out)"
```

Flags: `-n N` (default 16), `--ctx C` (default 2048, at most 16,384), `--place a|gate` (default `gate`), `--prefill batch|steps` (default `batch`), `--mode graph|eager`, `--plan` (print the plan and exit before the load), `--logits`, `--time [--warm W]`.

The server is `bloomery-serve --model glm` (build it with `--features glm5next --bin bloomery-serve`). At `--place a` and `--place bp` (the 3090 as an expert tier) it runs adaptive residency and the MTP draft by default (`BLOOMERY_RESIDENCY=off` and `BLOOMERY_DRAFT=off` turn them off), and `--plan` prints the plan and those choices and exits before any card is opened:

```sh
cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features glm5next --release --bin bloomery-serve
BLOOMERY_REF_MODEL="$M" target/release/bloomery-serve --model glm --place a --ctx 4096 --host 127.0.0.1 --port 8080
```

It serves one request at a time. A request keeps the slot's prefix back to its last checkpoint (every 512 positions); there is no host prompt cache, and slot save and restore answer 501. GLM-5.3's chat template always opens a thinking span; `chat_template_kwargs: {"reasoning_effort": "low"}` (or `"high"`) makes it shorter than the template's default, `max`.

## Qwen3.8-Flash-Next

<!-- pending: default-place-a -->
```sh
hf download unsloth/Qwen3.8-Flash-Next-GGUF --include 'UD-Q4_K_XL/*' --local-dir ~/models/Qwen3.8-Flash-Next
M=~/models/Qwen3.8-Flash-Next/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf

cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin generate_qwen3moe
cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features deepseek41 --release --bin bloomery-serve-qwen38

# greedy, on an A6000 (the default --place a); add --place gate on a single RTX 3090
BLOOMERY_REF_MODEL="$M" target/release/generate_qwen3moe --prompt "The capital of France is" -n 32
# the HTTP server
BLOOMERY_REF_MODEL="$M" target/release/bloomery-serve-qwen38 --host 127.0.0.1 --port 8080
```

`generate_qwen3moe` reads the architecture from the file's header and opens a Qwen3.8, Qwen3.6 or Qwen3-30B file; it takes exactly one of `--prompt TEXT` (tokenized with the file's vocabulary, no BOS, no chat template) and `--tokens a,b,c`, and prints the ids and the generated text. Flags: `-n N` (default 32), `--ctx C` (default 4096), `--place a|gate` (Qwen3.8 only, default `a`), `--prefill auto|pass|gemm|step`, `--mode graph|eager`, `--time [--warm W]`, `--logits`. Each layer's routed expert prefix goes on the card as its budget holds and the rest on the host (`BLOOMERY_QWEN38_EXPERTS=card`, the default; `host` puts every routed expert on the host).

<!-- pending: default-place-a -->
`bloomery-serve-qwen38` speaks the same llama-server API as `bloomery-serve-ds41` (the think-span split and the DSML tool calls above are V4.1's), with these differences: `--place a|gate|bp` (default `a`; `bp` is refused at the first prompt until the prompt's 3090 tier lands), `--ctx-size C` (or `--ctx`) sizes the stores, a prompt that fills them is a 400 and generation stops there, and `--chat-template-file PATH` replaces the file's template. A request keeps the longest prefix it shares with the slot that the recurrent layers can stand at: every held position, or the nearest checkpoint at or below it. The host prompt cache (`--cache-ram MIB`, 0 off) holds a session's state while another session's request takes the slot. It serves one request at a time.

<!-- pending: mtp-draft-path — the MTP draft path is fixed at /models/Qwen3.8-Flash-Next/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf (crates/refset/src/arch/qwen4exp/mtp.rs) with no lever to move it -->
**The MTP draft** (`BLOOMERY_DRAFT=mtp`, both binaries) opens the draft file at one fixed path, `/models/Qwen3.8-Flash-Next/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`; no lever moves it. To use it, download `MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` from the same upload and put the file, or a symbolic link to it, at that path. The greedy ids with the draft are the plain run's; its speed is not measured yet.

## Qwen3.6-35B-A3B and Qwen3-30B-A3B

```sh
hf download lmstudio-community/Qwen3.6-35B-A3B-GGUF --include 'Qwen3.6-35B-A3B-Q4_K_M.gguf' \
  --local-dir ~/models/Qwen3.6-35B-A3B
M=~/models/Qwen3.6-35B-A3B/Qwen3.6-35B-A3B-Q4_K_M.gguf

cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin generate_qwen3moe

# the whole model on CUDA device 0; choose the card with CUDA_VISIBLE_DEVICES
BLOOMERY_REF_MODEL="$M" target/release/generate_qwen3moe --prompt "The capital of France is" -n 32
```

The same binary opens the Qwen3-30B-A3B-Instruct-2507 `Q4_K_M` file (`unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF`). Both run the whole model on one card and refuse `--place`. <!-- pending: no-glm-qwen36-server -->
`bloomery-serve-qwen38` opens only a Qwen3.8 file; there is no HTTP server for Qwen3.6 or Qwen3-30B yet.

## Clef-Flash

[Clef-Flash](https://huggingface.co/Cloudflare/clef-flash) is Cloudflare's decision model: a Qwen3.5 backbone (`qwen35`, dense) and a small joint schema head that scores every allowed option of every question in one prompt pass. bloomery reads the backbone's Q4_K and Q6_K weights only, and the published Clef GGUFs carry Q8_0 sites, so the file is converted from the BF16 release with llama.cpp mainline (text only, with `--no-mtp`, as the gates ran it):

```sh
hf download Cloudflare/clef-flash --local-dir ~/models/clef-flash/hf
python3 <llama.cpp>/convert_hf_to_gguf.py ~/models/clef-flash/hf --no-mtp --outtype bf16 \
  --outfile ~/models/clef-flash/clef-flash-BF16.gguf
<llama.cpp>/build/bin/llama-quantize ~/models/clef-flash/clef-flash-BF16.gguf \
  ~/models/clef-flash/clef-flash-Q4_K_M.gguf Q4_K_M

cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features clef --release --bin bloomery_serve_clef

# the backbone on one card (see Picking the card), the head on the host
target/release/bloomery_serve_clef --model ~/models/clef-flash/clef-flash-Q4_K_M.gguf \
  --head ~/models/clef-flash/hf/joint_head.safetensors --port 8091

curl -s http://127.0.0.1:8091/v1/systemone -d '{"model": "clef-flash",
  "state": "User: what is the weather in Seoul tomorrow? Tools available: web_search, calculator, calendar.",
  "questions": {"tool": {"type": "choice", "instructions": "Which tool should the agent call next?",
    "criteria": {"web_search": "Look something up online", "calculator": "Do arithmetic",
                 "calendar": "Read or write events", "none": "Answer directly"}}}}'
```

The request and the response are the release's SystemOne format (its `README.md`): question types `choice`, `noul` and `score`, and per question the chosen option with its probabilities. Each response adds `timings` (`prompt_n`, `prompt_ms`, `head_ms`, `cache_n`). `GET /props` names the engine, the build, the model file, its quant and the head file. Flags: `--host` (default `127.0.0.1`), `--port` (default 8091), `--ctx` (default 16384, the release's `max_length`), `--head-config` (default: `joint_head_config.json` beside `--head`). It serves one request at a time and takes text states only.

## The data directory

`$BLOOMERY_DATA` holds everything that is not the model and not the source tree. Its default is `/root/bloomery-data`, the maintainers' path; set it to a directory of your own if you use it. The parts `generate_ds41` can read:

| Path under `$BLOOMERY_DATA` | What | Made by |
|---|---|---|
| `greedy-ds41/prompt<P>.tsv` | the token ids of prompt row P | `just ik-greedy-ds41 <P>` (needs ik_llama.cpp) |
| `engram/corpus-*.ids` | token-id corpora for the draft gate and the router trace | not in this repository |
| `ref_deepseek41/` | the oracle tensor dumps the gates read | `tools/ref/dump.sh` (needs ik_llama.cpp) |

None of these is needed for a plain run with `--tokens`. Without `--tokens` or `--depth`, `generate_ds41` reads `greedy-ds41/prompt0.tsv` and stops with an error that names the file when it is missing. The prose prompts some README rows were measured on are these corpora; they are not in this repository.

## Checks and gates

`just check`, `just lint` and the `just gate-*` recipes are the maintainers' gates. They go through `tools/box.sh`; see `AGENTS.md` for what each one runs. The crates with no device code, no x86_64 and no Linux-only code have native tests that run anywhere: `python3 tools/recipes.py pure-crates` lists them, and `cargo test -p <crate>` runs one.

On a Mac, `just mac-check` and `just mac-lint` run `check` and `lint` as an x86_64-linux cross check that links nothing (`cargo --target x86_64-unknown-linux-gnu`); `just mac-check` then also cross-checks every build shape a gate or build recipe compiles (`just mac-combos`: one `cargo check` per package, mode and feature set, the list from `tools/recipes.py combos`). They need the pinned nightly with the x86_64-linux std, the CUDA headers, a Linux sysroot's headers and libclang, set up in one environment file. `just fmt` and `just mac-fmt-check` format and check with the pinned nightly's rustfmt. The header of `tools/mac-check.sh` lists each piece and how it is made.
