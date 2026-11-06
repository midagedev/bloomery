# bloomery @VERSION@ — prebuilt Apple Silicon release

bloomery's Metal backend: a Rust host and Metal kernels that serve a mixture-of-experts model on a Mac's GPU, with
llama-server's HTTP API. Built from commit @COMMIT@. Source, measurements and the full README:
https://github.com/midagedev/bloomery (the Mac page: `docs/mac.md`).

## What is in this archive

| File | What it does |
|---|---|
| `bin/bloomery-serve-metal` | llama-server's HTTP API (`/v1/chat/completions`, `/completion`, `/v1/models`, `/health`) on the Mac's GPU |
| `bin/generate_metal` | one prompt from the command line; prints the rates on stderr |
| `bin/prepare` | optional: writes an MLX 4-bit checkpoint folder (mlx-community) as one GGUF split |
| `serving.md` | the HTTP API's reference: its routes, the request fields each reads or refuses and the error objects (its flags and seats are the Linux binary's) |

`bloomery-quantize` (IQ1_S and IQ2_XXS quantization on the card) is not in this archive: it links the CUDA backend, and cargo
refuses to build it without it (`required-features = ["gpu"]`), so a Mac has no binary of it, not even for its `--cpu` path. It is in the
Linux archives (`docs/quantize.md` in the repository).

## Requirements

- An Apple Silicon Mac with macOS 26.2 or newer; an older macOS is refused by name at the start.
- Two model architectures run: `qwen35moe` (Qwen3.6-35B-A3B and files with its shapes) and `qwen3moe`
  (Qwen3-30B-A3B), from a Q4_K_M GGUF file or an mlx-community 4-bit folder. A 48 GB Mac is tested; a 32 GB Mac
  opened and decoded Q4_K_M files of Qwen3.6's shape (`docs/mac.md`).
- The kernels compile from their embedded source on the first start of each process; no Xcode is needed.

## Run

```sh
# an mlx-community 4-bit folder opens as downloaded, fetched once by --hf or named by -m
bin/bloomery-serve-metal --hf mlx-community/Qwen3.6-35B-A3B-4bit
bin/bloomery-serve-metal -m PATH/Qwen3.6-35B-A3B-4bit

# a Q4_K_M GGUF opens directly (for example lmstudio-community/Qwen3.6-35B-A3B-GGUF)
bin/bloomery-serve-metal -m PATH/Qwen3.6-35B-A3B-Q4_K_M.gguf

curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "What is the capital of France?"}], "max_tokens": 256}'
```

The model thinks first: its reasoning comes back in `reasoning_content`, so leave `max_tokens` room for the answer.
The server takes `-m` or `--hf`, `--host`, `--port`, `--ctx` (`--ctx-size`, `-c`), `--chat-template-file`, `--alias`,
`--queue-depth`, `--api-key` and `--api-key-file`; any other flag is refused by name with exit 64. It serves one
sequence, text only.

If macOS says the binary "cannot be opened" after a browser download, clear the quarantine flag:
`xattr -d com.apple.quarantine bin/*`.

MIT licensed (`LICENSE`); third-party notices in `THIRD_PARTY_NOTICES.md`.
