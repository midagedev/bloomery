# bloomery on Apple Silicon

bloomery runs on a Mac's GPU through its own Metal backend: `bloomery-serve-metal` serves the llama-server HTTP API,
and `generate_metal` runs one prompt from the command line. The host side is Rust; the kernels are Metal Shading
Language. Two architectures run on a build after 0.3.9: `qwen35moe` (Qwen3.6-35B-A3B and files with its shapes) and
`qwen3moe` (Qwen3-30B-A3B) ([which files](#which-files-run)). The release carries a signed and notarized Apple
Silicon build; building from source works too. The port's map and design are in [`contrib/mac.md`](contrib/mac.md) and issue
[#593](https://github.com/midagedev/bloomery/issues/593).

## What you need

- An Apple Silicon Mac with macOS 26.2 or newer; an older macOS is refused by name at the start
  (`crates/metal/src/device.rs`). Run on an M1 Pro and an M4 Pro 48 GB here, and on an M3 Ultra and an M5 Ultra by
  [@avlp12](https://github.com/avlp12) (#593).
- Memory: after the open, the model holds 20.3 GB (MLX 4-bit) or 21.8 GB (Q4_K_M) of device memory at a 4,096-row
  context, measured on the M4 Pro. A 48 GB Mac is tested. A 32 GB Mac (M1 Pro) opened and decoded two other Q4_K_M
  files of this shape ([below](#which-files-run)). The GPU working set Metal reports was 0.78 × RAM on the three Macs
  measured (24.96 GiB on an M1 Pro 32 GB, 37.44 GiB on an M4 Pro 48 GB); above 48 GB it is unverified.
- To build from source: [rustup](https://rustup.rs), whose `cargo` comes first in `PATH` (Homebrew's `rust` builds
  with its stable compiler and fails on the pinned crates). `rust-toolchain.toml` pins the nightly, and rustup installs
  it on the first build. The Metal backend links no CUDA code, so plain `cargo` builds it. Xcode's Metal toolchain is
  optional: with `xcrun metal` the build writes a metallib, and without it the kernels compile from their embedded
  source when the binary starts (the release's binaries do this). Every number and gate here ran the source path;
  the metallib path builds, and its gates have not run on the maintainers' Macs. If you have the toolchain (`xcodebuild -downloadComponent MetalToolchain`),
  please run `just lab-metal` and report the output (why: [`contrib/mac.md`](contrib/mac.md), section 1.3).

## Install

```sh
brew install midagedev/tap/bloomery
# or the installer: the latest release, sha256-checked, symlinks into ~/.local/bin
curl -fsSL https://raw.githubusercontent.com/midagedev/bloomery/main/tools/release/install.sh | sh
```

Both install `bloomery-serve-metal`, `generate_metal` and the converter as `bloomery-prepare`. The tarball
`bloomery-<version>-macos-arm64-metal.tar.gz` on the [releases page](https://github.com/midagedev/bloomery/releases)
holds the same three binaries under `bin/` (the converter as `prepare`). The commands below spell the source build's
`target/release/` paths; an installed binary runs by its name.

## Build from source

```sh
git clone https://github.com/midagedev/bloomery && cd bloomery
cargo build --release -p bloomery-metal --bin bloomery-serve-metal --bin generate_metal \
  -p bloomery-model-hf --bin prepare
```

## Which files run

| File | State |
|---|---|
| Qwen3.6-35B-A3B, Q4_K_M GGUF or an MLX 4-bit folder | runs; the numbers are below |
| [Ornith-1.5-35B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF), Q4_K_M | runs, text only: it has Qwen3.6's shapes, and its next-token layer is not read |
| [MiniCPM-V 4.7 35B-A3B](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B-gguf), Q4_K_M | runs on a build after 0.3.9, text only: its `mmproj` is not read. Release 0.3.9 refuses it: `qwen35moe arena: a vocabulary of 248144; the head reads 248320` |
| [Sage-A1.5](https://huggingface.co/InternScience/Sage-A1.5-Q4_K_M-GGUF), Q4_K_M | not run on a Mac yet; it has Qwen3.6's shapes |
| [Qwen3-30B-A3B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507), Q4_K_M | runs on a build after 0.3.9, text only. Release 0.3.9 refuses it: architecture `qwen3moe` |
| [Qwen3.6-35B-A3B](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) `UD-Q4_K_XL` | runs on a build after 0.3.9, text only. Release 0.3.9 refuses it: its Q8_0, Q5_K and f32 tensors |

The mat-vec kernels run Q4_K, Q5_K, Q6_K, Q8_0 and MLX 4-bit weights, and f32 at the delta layers' `ssm_alpha` and
`ssm_beta`. A shared expert may have another type than its routed stacks. The router and the shared-expert gate may be
F32 or MLX 8-bit, which the open dequantizes. Any other type is refused when the file opens, and the error names the tensor.
The evidence for each row is in [`contrib/mac.md`](contrib/mac.md), section 1.1.

## Get a model

Either file works for Qwen3.6:

- **A Q4_K_M GGUF opens directly**, for example `Qwen3.6-35B-A3B-Q4_K_M.gguf` from
  [`lmstudio-community/Qwen3.6-35B-A3B-GGUF`](https://huggingface.co/lmstudio-community/Qwen3.6-35B-A3B-GGUF).
- **An MLX 4-bit folder converts once**, for example
  [`mlx-community/Qwen3.6-35B-A3B-4bit`](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-4bit). `prepare` writes
  a GGUF split that keeps MLX's 4-bit and 8-bit weights as they are stored. It skips the vision tower (text only) and
  checks the tokenizer against the folder's. The split is 19.5 GB; the conversion needs that much free disk plus about
  10 %.

  ```sh
  target/release/prepare PATH/Qwen3.6-35B-A3B-4bit --plan            # prints the plan, writes nothing
  target/release/prepare PATH/Qwen3.6-35B-A3B-4bit OUT/qwen36-mlx4    # writes OUT/qwen36-mlx4-00001-of-00001.gguf
  ```

The Qwen3-30B-A3B MLX 4-bit folder,
[`mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit`](https://huggingface.co/mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit),
converts the same way (`prepare` picks the map by the folder's `model_type`, `qwen3_moe`; the weights stay as stored).
The split is 17.2 GB, and on a build after 0.3.9 `generate_metal` opens and runs it.

A quick check from the command line (the rates print on stderr, `decode_tok_s=`):

```sh
BLOOMERY_REF_MODEL=OUT/qwen36-mlx4-00001-of-00001.gguf \
  target/release/generate_metal --prompt "The capital of France is" -n 32
```

To learn whether a file opens before it loads: `generate_metal --plan [--ctx C]` reads only the file's
header and tensor table. It prints every refusal, not only the first, the bytes the open would allocate, the device's
working set and the host's free memory. It exits 0 when the header's checks pass, 3 when it is refused and 64 when the
header cannot be read; exit 0 does not prove the load fits at run time. A 24 MB cut of a Hub file is enough.

## Serve

```sh
target/release/bloomery-serve-metal -m OUT/qwen36-mlx4-00001-of-00001.gguf --port 8080

curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "What is the capital of France?"}], "max_tokens": 256}'
```

The model thinks first: its reasoning comes back in `reasoning_content`, so leave `max_tokens` room for the answer.
`/health`, `/v1/models`, `/v1/chat/completions` and `/completion` answer as llama-server's do.

The flags it takes: `-m`/`--model`, `--host`, `--port`, `--ctx` (also `--ctx-size` and `-c`; unset, the smaller of
32,768 and the file's `context_length`), `--chat-template-file`, `--alias`, `--queue-depth`, `--api-key` and
`--api-key-file`. Any other flag, `--parallel`, `-ngl`, `--mmproj` and `--flash-attn` among them, is refused by name
with exit 64.

## What it does and does not do yet

- One slot: the server runs one sequence. A request keeps the prefix it shares with the slot back to the last
  checkpoint of the model's recurrent state; a checkpoint is taken at each message start, so a follow-up turn reads
  only its new messages. Any other prompt runs from position 0.
- A long prompt runs in batched passes (flash attention over the pass, the experts over its rows); decode runs one id a
  step.
- No saved prompt cache across slots, no slot files, no image input (MiniCPM-V's `mmproj` is not read), no speculative
  decoding.
- On an M5 (Metal 4, GPU family 17 or newer) the device picks the neural accelerators' tensor-op GEMMs for prompts
  (`gemm_path=tensor` in the stats); M1 to M4 run the SIMD-group GEMMs (`gemm_path=simdgroup`). On an M5 Ultra the
  tensor path took a 2,048-row prompt pass's GEMM class from 203.51 to 92.73 ms (@avlp12, #593). Its sums run in another
  order, so a greedy answer can differ from an M1–M4's after many tokens.

## Numbers

Development-loop numbers from one M4 Pro (48 GB, 20-core GPU, macOS 27.0.1), 2026-10-25, Qwen3.6-35B-A3B, three
rounds, tok/s. The Mac has no timing runner (no lease, no witness blocks), so these are not landing evidence.

| File | Prompt, 512 ids | Prompt, 4,096 ids | Decode, depth about 0 |
|---|---|---|---|
| MLX 4-bit, converted by `prepare` | 943.0 | 1,076.5 | 101.2 |
| Q4_K_M GGUF | 905.3 | 1,044.5 | 85.6 |

mlx-lm and llama.cpp ran on the same files in the same sitting; their rows, every command and flag, and the per-round
tables are in [rig-log](https://github.com/midagedev/rig-log/blob/main/log/2026-10-25.md#mac-m4pro-xeng).

One more Mac, a development-loop number: an Apple M1 Pro 32 GB, macOS 27.0.1, `generate_metal --prompt "The capital of
France is" -n 32`, greedy, with a download running on the same Mac. MiniCPM-V 4.7 and Ornith-1.5-35B (Q4_K_M) each
decoded at 43.4 tok/s ([PR #805's comment](https://github.com/midagedev/bloomery/pull/805) has the commands and output). It is a different chip and different files from the table above.

**Contributors' Macs**: measured by [@avlp12](https://github.com/avlp12) on an M3 Ultra and an M5 Ultra, 2026-10-25, at
main bd0c61345 (its `crates/metal` is 0.3.9's), Qwen3.6-35B-A3B `Q4_K_M`, `generate_metal --tokens <seeded random ids>
-n 128 --ctx 8192`, the mean of two runs after one unscored run, tok/s. The M5 Ultra runs the tensor-op GEMMs, the M3
Ultra the SIMD-group ones. Chat: `bloomery-serve-metal -c 8192`, greedy, a 3,982-id first turn, TTFT on the client.
Their conditions, the engines they ran beside it and the per-op pass times are on
[#593](https://github.com/midagedev/bloomery/issues/593#issuecomment-6391771313).

| Mac | Prompt, 512 ids | Prompt, 4,096 ids | Decode after 512 ids | Chat: first turn TTFT, second turn TTFT |
|---|---|---|---|---|
| M3 Ultra | 2,570.4 | 3,300.0 | 141–142 | 1.26 s, 0.12 s |
| M5 Ultra | 4,248.9 | 5,971.8 | 211–212 | 0.70 s, 0.08 s |
