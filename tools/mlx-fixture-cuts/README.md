# Metal-only MLX fixture cuts

Rules files for `fixture cut` of a hub checkpoint folder into a Metal fixture: an MLX folder cut in its stored bits, written as `<family>-mlx` under the fixture root, whose `prepare` split is `<family>-mlxsplit`. The Metal gates open both by name (`crates/metal/src/bin/shared/mlx_split.rs`).

The CUDA fixture tier does not read this directory. A cut with a registry row keeps its rules in `tools/fixture-cuts/` (one file per row, held by `a_cut_fixture_is_its_rules_file`) and its directory in `tools/fixture-dirs.tsv`; the files here have neither a row nor a table entry.

    fixture cut <source folder> <out dir> --spec tools/mlx-fixture-cuts/<name>.json

Each file is held to its source's real shard headers and `config.json` by a test in `crates/model-hf/src/cut.rs`.

| file | source | cut |
|---|---|---|
| `qwen38d-mlx.json` | `mlx-community/Qwen3.8-27B-4bit` at `10c35caafbb80f7dc6a7a432cdd11af10a6d4818` | layers 0 to 3 (three delta layers, one full-attention layer), the embedding, the head and the final norm; no vision tower |
| `llama-mlx.json` | `mlx-community/Llama-3.1-8B-Instruct-4bit` at `90215b22ec18e72f623dde2ea7af4097025160e2` | layers 0 to 3 (every layer alike: GQA, SwiGLU), the embedding, the untied head and the final norm |
