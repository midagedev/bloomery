# gguf crate — stage 1, round 1-1 results

GGUF v3 loader + scalar reference dequantizer for the six ggml types the
model actually carries, gated against ggml itself (`to_float` from the
vendored ik_llama.cpp build). Everything below was produced on the box
through `tools/box.sh`; `BLOOMERY_DATA=/root/bloomery-data-s1-loader`.

## Gate: how to run

`cargo nextest` is **not installed** on the box (checked 2026-09-19:
`cargo nextest --version` → `error: no such command`). The two tests are
therefore `#[ignore]`d and run with plain cargo. From the repo root on the
Mac:

    ./tools/box.sh 'bash tools/ref/build-dequant.sh'
    ./tools/box.sh '"$BLOOMERY_DATA/bin/dequant_ref"'
    ./tools/box.sh 'cargo test -p gguf -- --ignored --nocapture'

The first command builds `tools/ref/dequant_ref.cpp` against
`$IK/ggml` and installs it to `$BLOOMERY_DATA/bin/dequant_ref`; the second
dumps `$BLOOMERY_DATA/ref/{manifest.txt,<type>.raw,<type>.meta}`; the third
runs `hw_coverage` and `hw_dequant_matches_ggml`. If nextest lands on the
box later, the names already carry the `hw_` prefix the `hw` profile
expects (`cargo nextest run -p gguf --profile hw`).

## Type census, re-derived from the file

Loader output (`hw_coverage`, 2026-09-19), next to the spec's table:

| ggml type | ours | spec | tensor sampled by the oracle |
|---|---:|---:|---|
| f32 (0)   | 108 | 108 | `blk.0.attn_norm.weight` |
| q5_0 (6)  | 26  | 26  | `blk.1.ffn_down_exps.weight` |
| q5_1 (7)  | 1   | 1   | `blk.0.ffn_down.weight` |
| q3_K (11) | 188 | 188 | `token_embd.weight` |
| q4_K (12) | 53  | 53  | `blk.0.attn_output.weight` |
| q6_K (14) | 1   | 1   | `output.weight` |
| total     | 377 | 377 | |

No disagreement with the spec's table. The counts are asserted against the
oracle's `manifest.txt` inside `hw_coverage`, so a re-quantized model file
re-runs the whole census automatically. Header facts: GGUF v3,
377 tensors, 45 KV pairs, `general.architecture = deepseek2`,
`deepseek2.block_count = 27`, `.expert_count = 64`, `.expert_used_count = 6`,
`.embedding_length = 2048`, `.attention.head_count = 16`, `data_base =
3996416`, alignment 32 (the file carries no `general.alignment`, so the
default applies).

## Gate: dequant vs ggml `to_float`

`hw_dequant_matches_ggml` re-dequantizes the oracle's rows from our own
mmap slice and compares float-for-float. **Max absolute difference = 0
(bit-exact) for every type**, against a gate of ≤ 1e-6:

| type | tensor | rows × rowlen | max abs diff |
|---|---|---|---|
| q6_K | output.weight            | 4 × 2048  | 0.000e0 |
| q3_K | token_embd.weight        | 4 × 2048  | 0.000e0 |
| f32  | blk.0.attn_norm.weight   | 1 × 2048  | 0.000e0 |
| q5_1 | blk.0.ffn_down.weight    | 4 × 10944 | 0.000e0 |
| q4_K | blk.0.attn_output.weight | 4 × 2048  | 0.000e0 |
| q5_0 | blk.1.ffn_down_exps.weight | 4 × 1408 | 0.000e0 |

The zero is not luck — it required mirroring the *compiled* library, not
just the source. `objdump -d --disassemble=…` on
`$IK/build/ggml/src/libggml.so` shows the vendored build contracts two of
the five decoders:

- `dequantize_row_q5_1` computes `x0*d + m` as one `vfmadd132ps`;
- `dequantize_row_q4_K` computes `d1*q - m1` as one `vfmsub132ps`;
- `dequantize_row_q5_0`, `_q3_K`, `_q6_K` contain no fused ops.

The Rust ports match that shape exactly: `mul_add` in q5_1 and q4_K
(`(x0 as f32).mul_add(d, m)`, `(q as f32).mul_add(d1, -m1)`), plain
separate multiplies elsewhere. With unfused Rust against fused C the two
FMA types would sit ~1 ulp apart (up to a few 1e-6 at weight magnitudes) —
enough to threaten the 1e-6 gate. If libggml.so is ever rebuilt with
different FP-contract flags, re-run the disassembly check above.

`half_to_f32` (f16→f32, integer-only) is copied from
`crates/q3k-cpu/src/main.rs:47`, which verified it against ggml's
`GGML_FP16_TO_FP32` in stage 0; the conversion is exact, and every K-quant
scale field (d, dmin, m) exercises it in this gate. The file contains no
F16 *tensors*, so the F16 row path itself has no direct oracle coverage
here — it is the same routine on contiguous u16s.

## Not covered / later rounds

- `dequant_row` for `q5_K` errors by design (enum entry exists only so the
  size table is complete for the K-quant family); the model carries none.
- Byte-order: all multi-byte reads are `from_le_bytes`; the box is LE, so
  the BE path is untested by construction (GGUF is LE-only anyway).
- No benchmarks here — this is the reference path by design; speed lives in
  `q3k-cpu`/`q3k-gemv` and is measured by `tools/ref/measure*.sh`.

## Lint state

`cargo clippy -p gguf --all-targets`: **0 warnings, 0 errors** (2026-09-19).
`cargo clippy --workspace --all-targets`: **0 errors**; all remaining
warnings are in the pre-existing crates (`q3k-gemv`, `q3k-cpu`) — none in
`gguf`. `cargo fmt --all -- --check` passes.
