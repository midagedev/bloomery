# Metal-only MLX fixture cuts

Rules files for `fixture cut` of a hub checkpoint folder into a Metal fixture: an MLX folder cut in its stored bits, written as `<family>-mlx` under the fixture root, whose `prepare` split is `<family>-mlxsplit`. The Metal gates open both by name (`crates/metal/src/bin/shared/mlx_split.rs`).

The CUDA fixture tier does not read this directory. A cut with a registry row keeps its rules in `tools/fixture-cuts/` (one file per row, held by `a_cut_fixture_is_its_rules_file`) and its directory in `tools/fixture-dirs.tsv`; the files here have neither a row nor a table entry.

    fixture cut <source folder> <out dir> --spec tools/mlx-fixture-cuts/<name>.json

Each file is held to its source's real shard headers and `config.json` by a test in `crates/model-hf/src/cut.rs`.

| file | source | cut |
|---|---|---|
| `qwen38d-mlx.json` | `mlx-community/Qwen3.8-27B-4bit` at `10c35caafbb80f7dc6a7a432cdd11af10a6d4818` | layers 0 to 3 (three delta layers, one full-attention layer), the embedding, the head and the final norm; no vision tower |
| `llama-mlx.json` | `mlx-community/Llama-3.1-8B-Instruct-4bit` at `90215b22ec18e72f623dde2ea7af4097025160e2` | layers 0 to 3 (every layer alike: GQA, SwiGLU), the embedding, the untied head and the final norm |
| `qwen3-mlx.json` | `mlx-community/Qwen3-8B-4bit` at `545dc4251c05440727734bcd94334791f6ab0192` | layers 0 to 3 (every layer alike: GQA with the per-head q/k norms, SwiGLU), the embedding, the untied head and the final norm |
| `qwen3t-mlx.json` | `mlx-community/Qwen3-4B-Instruct-2507-4bit` at `50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b` | layers 0 to 3 (every layer alike: GQA with the per-head q/k norms, SwiGLU), the embedding and the final norm; the head is the embedding (the folder has no `lm_head`, its config `tie_word_embeddings` true) |
| `qwen2-mlx.json` | `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit` at `019cc73c45c770444708a6dd8690c66243cc5c80` | layers 0 to 3 (every layer alike: GQA of 28 heads over 4 key heads of 128 values with a bias on the q, k and v projections and no head norm, SwiGLU), the embedding, the untied head and the final norm |
| `inkling2-mlx.json` | `mlx-community/Inkling-2-Mini-4bit` at `9856dc53edead71405f8348930937c3638ff21a2` | every language layer 0 to 23, the loop whole (stem 0-5, core 6-17 over its three passes, coda 18-23, each six-layer attention-residual block with its global layer, the four message-gate controllers on the core) and both expert formats (NVFP4 with a global scale on layers 0-19, MXFP8 on 20-23), the stacked routed experts cut to their first 24 of 128 with the router's rows and correction bias and `n_routed_experts` moved, `vision_config` nulled, the embedding, the head and the norms |

## The Metal e2e gate's pin of a cut

A row of `crates/metal/src/bin/gate_metal_qwen35moe_e2e` that opens one of these cuts (`Model::Folder`) holds the session's logits to a committed dump of the same cut, `pins/<family>.txt` and `.f32`. The dump is mlx-lm's own modules in f32 over the exact dequant of the cut's 4-bit weights, written on a Mac with mlx-lm (the Metal gates run on the second M1 Pro, `/Users/d1-mac1/venv-mlx/bin/python`):

    tools/mlx-logits.py dump $BLOOMERY_FIXTURE_ROOT/<family>-mlx <family> --ids ID,ID,… --positions 0,1,… --out DIR

The two files go into `pins/`; a hand-written tail of provenance lines (the folder's revision, the command, the mlx-lm and mlx versions) follows the text's `argmax` line, which the gate's reader ignores. The row's band is derived in `row.rs` from the pin's largest logit and held under half the pin's smallest top-2 gap.
