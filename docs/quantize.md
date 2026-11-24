# bloomery-quantize

`bloomery-quantize` is the quantizer: it writes the final quantized GGUF set from the source set in one pass, with no
intermediate set. The routed-expert stacks become IQ1_S or IQ2_XXS on the card (or the host's cores), every other
tensor the type `llama-quantize` gives it by the host's K-quant rules (Q3_K, Q4_K, Q5_K, Q6_K, Q8_0, and the F32/F16
copies), and the metadata, the tensor order, the alignment and the shard layout are the ones `llama-quantize` writes.
The output is byte-identical to `llama-quantize`'s for the same input, importance matrix, rules and options.
`llama-quantize` stays as the oracle: the gates compare against it.

It is in the Linux archives only. It links the CUDA backend, and cargo refuses to build it without the `gpu` feature,
so the Mac archive has no binary of it (`--cpu` included).

## Scope

The architectures are `kimi-k3`, `exaone-moe`, `deepseek41` and `ernie5_5`; the file types are Q3_K_M, Q4_K_M, Q5_K_M, Q6_K,
Q8_0, IQ2_XXS and IQ1_S. Any other architecture, type, or a type resolution that lands on a type not ported (Q2_K, ...) is
refused by name before anything is written. Q3_K is also a rule type: a `--tensor-type-file` rule,
`--output-tensor-type` or `--token-embedding-type` may name it (ggml's `quantize_row_q3_K_ref` without an importance
matrix, `_impl` with one). A rule may also name a type the quantizer copies and does not write (F16, BF16, MXFP4,
FP8_B128): a tensor already in it is copied, any other is refused by name (ERNIE 5.5's lossless file, below).

Q3_K_M is `llama_tensor_get_type_impl`'s Q3_K_M branches, and a rules file is not needed to reach them: the first two
`attn_v`-like tensors Q5_K and the rest Q4_K (Q8_0 at eight experts), `attn_output` Q4_K (Q5_K at eight experts),
`ffn_down` Q5_K for the first `n_layer / 16` blocks and Q4_K after, `output` Q6_K, every other quantizable tensor Q3_K.
`gate-quantize-plan` and `gate-quantize-e2e` hold it to llama-quantize's own plan and bytes on the K3 fixture
(`general.file_type` 12 included), which has eight experts and four blocks, so they reach the eight-expert branches; the
other branches are `tests/resolve.rs`'s, written from the same source. On a DeepSeek-V4.1 source the file type alone
types the tensors the way the `Q3_K_M` file has them (the routed and shared `ffn_down` of blocks 0 and 1 Q5_K and the
other blocks' Q4_K; the grouped `attn_output_a` and `attn_output_b` are not ggml's `attn_output` category and stay
Q3_K); a rules file only changes what the file type does not say.

    bloomery-quantize IN-00001-of-N.gguf OUT/prefix Q4_K_M --imatrix m.imatrix.gguf --tensor-type-file rules.txt \
        --output-tensor-type q6_k --token-embedding-type q4_k --keep-split --allow-requantize

## Usage

    bloomery-quantize [options] <in first shard> <out> <ftype> [nthreads]
    bloomery-quantize [options] --hf DIR <out> <ftype> [nthreads]

    --hf DIR                  the input is the Hugging Face checkpoint folder DIR, read as `prepare` reads it
                              (`model_hf::quantize_source`), and quantized without writing that GGUF set first; exactly
                              one of `<in first shard>` and `--hf`. Needs a build with the `hf` feature
    --hf-arch WORD / --hf-drop GLOB / --hf-shard-gib N   with `--hf`: `prepare`'s `--arch`, `--drop` (repeatable) and
                              `--shard-gib` (48), so the folder's tensors, header and shards are those of the split
                              `prepare` would write; `--keep-split` writes one output shard per such shard
    --imatrix FILE            the importance matrix
    --tensor-type-file FILE   llama-quantize's rules file
    --output-tensor-type T / --token-embedding-type T   the flag types, as llama-quantize's
    --allow-requantize        as llama-quantize's
    --keep-split              the output has the input's number of shards (`<out>-0000K-of-0000N.gguf`)
    --resume                  with `--keep-split`: a whole output shard an earlier run of this same plan wrote is kept
                              (not read, not quantized again), a stale `.partial` is removed and written again, and a
                              whole file that is not this plan's shard refuses the run by name
    --plan                    every tensor's source and target type, bytes in and out, the fate, the derived wall and
                              the room; writes nothing
    --llama-plan TSV          llama-quantize's own `--dry-run` plan, cross-checked tensor for tensor
    --check N                 N rows of every IQ tensor through the host rule too; stops by name on the first byte
    --cpu                     the IQ rule on all cores; no card
    --card N                  the card, by CUDA ordinal
    --batch N                 experts per card call

`nthreads` is accepted for llama-quantize's command line and ignored: the host rules run on the resident pool.
`--help` and `--version` print the usage and `bloomery-quantize <version> (commit <hash>)`.

With `--hf` the bytes are the ones the `prepare` split would have given (`just gate-quantize-hf`: the header and the
output's sha256 per shard equal on a toy K3.1 folder and on a 2-layer cut of the real one). A folder with a shard its
index names missing, a `weight_scale` with no `weight`, or an E8M0 scale byte of 255 is refused by name before any
output remains. A folder `fixture cut` wrote carries its `bloomery.fixture.*` metadata into the GGUF's keys, as the
fixture readers take their types (the version a u32, the card budget a u64), so the fixture tier opens a cut's GGUF as
the fixture it is; a folder with no such keys writes none (`just gate-ernie55-quant`).

`--resume` continues an interrupted run of the same plan. The plan is a pure function of the inputs and the command
line, so a shard an earlier run finished is exactly the shard this run would write, and before any tensor is read every
planned output shard is classified: an absent file is written, a stale `<name>.partial` is removed (the log names it)
and written, and a whole file is kept when its length equals the plan's and its header — every metadata pair and every
tensor's name, dims, type and offset, the bytes this plan's writer would write — equals it, a comparison that reads no
tensor data. A whole file that differs in any of that refuses the run naming the path and the first field that differs,
and removes nothing. A kept shard costs no read and no quantization; the shard line says `kept`, the bytes-done,
elapsed and time-left of the shard line run at the mean rate of the shards this run writes, and the summary line says
how many were kept. A failure under `--resume` removes only the `.partial` it was writing: the whole shards this run
finished and the ones it kept stay, so the next `--resume` goes on (the room check counts only what is left to write).
Without `--resume` any existing output, a stale `.partial` included, is refused and a failed run removes what it made.

Exit: 0 done; 1 a refusal, named; 64 a usage error; 66 a missing input file.

## The type rules

`--tensor-type-file` is llama-quantize's file: whitespace-separated `regex=type` tokens. The first rule whose regex
matches the lowercased tensor name wins. The matcher here is a subset of ECMAScript regex (literals, escaped
punctuation, `.`, `[...]`, `\d \w \s`, groups with `|`, `* + ?`, `^ $`); a pattern outside it is refused by name, not
guessed at. Only `*_exps.weight` stacks of three or more dimensions are quantized here. With `--llama-plan` a tensor
the rules and llama-quantize type differently is a refusal in either direction.

The port writes no fallback type. llama-quantize's `convert_incompatible_tensor` swaps a type whose block does not
divide a tensor's columns for another (Q4_K to Q5_0, Q5_K to Q5_1, Q6_K to Q8_0, the IQ and Q2/Q3 kinds to IQ4_NL); here
a tensor whose columns are not a multiple of 256 under a K-quant type needs a rule naming a type the port writes
(Q8_0), or the run is refused by name with the tensor. K3.1's `tight` map names `attn_k_b` and `ssm_f_b` Q8_0 by rule
(`gen_rules` in `tools/ref/k31-quant.sh`), so those two never reach a K-quant type.

## ERNIE 5.5 Flash Preview (`ernie5_5`)

The input is the checkpoint folder (`--hf`), never a GGUF: projections stored FP8 e4m3 with one f32 scale a 128 by 128
square (FP8_B128 in the GGUF: 128 values a block, each row's block taking its square's scale), MXFP4 routed experts,
BF16 embedding, head, routers and compressor matrices, F32 the rest (the norms, the mHC mixers' tables, the VHA head
mix). No rules file is needed: the rows keyed to `ernie5_5` keep as converted what the conversion names
(`tensor_allows_quantization`: the mHC mixers, the VHA head mix, the compressor's table, the indexer) and what the
reader holds in its stored type alone (the embedding, the head and the routers BF16, the routed stacks MXFP4, the
compressor's matrices as the folder load keeps them), and the file type types the FP8 projections. `DIR` is the
profile's `MODEL` (`tools/ref/models/ernie5_5.sh`), the released folder, whose lossless GGUF is the profile's
`GGUF_MODEL`; `just quantize` runs the same command through the card's gate lock.

| | the command (`--cpu`: nothing needs the card) | the FP8 128-block projections | the MXFP4 routed experts | BF16 and F32 |
|---|---|---|---|---|
| the default | `bloomery-quantize --hf DIR OUT Q8_0 --allow-requantize --cpu` | Q8_0: ggml's reference rule over the FP8 values the loader makes, each value within its derived band | MXFP4, byte for byte the blocks the loader makes of the stored planes | as stored |
| the lossless file | `bloomery-quantize --hf DIR OUT Q8_0 --tensor-type-file RULES --cpu`, RULES the one line `\.weight$=fp8_b128` | FP8_B128, the loader's blocks unchanged: every tensor of the GGUF byte for byte the folder loader's made tensor, nothing converted (no `--allow-requantize`: every tensor is a copy) | MXFP4, byte for byte | as stored |

Both files carry the experts as runs of the file where a folder's are made at load, which is what a hosted load opens:
`gate_ernie55_e2e`'s hosted arms (`--hosted-gguf`, the fixture tier's file the profile's `GGUF_MODEL`) open the lossless
one, the folder's own dense bytes. A cut folder's GGUF is a fixture file (above). `just gate-ernie55-quant` holds the
two rows on the fixture cut: the types under the folder's names and dims, the experts the loader's bytes and the OCP
MXFP4 decode of the stored planes, the Q8_0 blocks the reference rule's, the description reading either GGUF as the
folder, and the lossless file's every tensor the loader's. The released folder's set was written once, with
`--allow-requantize --keep-split --resume --cpu` (`--keep-split`: one output shard a shard of the split `--hf-shard-gib`
plans; the box's NVMe): 4 shards, 155.08 GB, 0 tensors converted, the tool's summary `done in 384 s`; a later run of the
same plan keeps every whole shard (`--resume`).

## The importance matrix's slabs

An entry of the importance matrix holds `ne0` values for each slab of the tensor's rows. For a stacked tensor
(`[ne0, rows, experts]`) a slab is an expert, `rows` rows each, as in llama-quantize (`nrows_per_expert`), and an entry
of any length but `ne0 x experts` is refused. A 2-D tensor is one slab: llama-quantize quantizes every row with the
entry's one `ne0` vector.

A block-diagonal tensor is the one case llama-quantize cannot take. DeepSeek-V4.1's `attn_output_a` is `[4096, 8192]`
with `deepseek41.attention.output_group_count` = 8: row group `g` (rows `g x 1024` on) reads input slice `g` of the
attention output, and the engine's collector writes one entry of `8 x 4096` values for it. llama-quantize needs 4096
there and aborts (`imatrix size 32768 is different from tensor size 4096`). Here the tensor's group count is a header
fact of its architecture (`gguf::arch::row_groups_fact`, the stem that names it and the key that counts it), a group is
a slab, and the plan accepts an entry of `ne0 x groups` values (group `g` quantizes with the `g`-th `ne0` vector) or of
one `ne0` vector (every group takes it, which is llama-quantize's 2-D rule); any other length is refused by name, and so
is a header with no group count, rows the groups do not divide equally, and a block-diagonal tensor of more than two
dimensions. `tests/resolve.rs` holds the bytes: the grouped tensor equals its groups quantized apart as separate 2-D
tensors with their slices, and an entry of one vector equals the plain 2-D tensor.

## What it refuses

Nothing is turned into a plausible value.

- A source row with an Inf or NaN value (F32, F16, BF16), or an MXFP4 block whose E8M0 scale byte is 255 (Inf): named
  with tensor, expert, row and value index.
- An imatrix entry that is Inf, NaN or negative, in the sums or in the counts: named with the tensor and entry.
- A source type the oracle does not cover (anything but F32, F16, BF16, MXFP4), a row that is not a multiple of 256, an
  importance entry of a length the tensor's slabs do not take (above).
- Too little room for the output (checked before anything is written; the output directory's free bytes against the
  bytes the plan writes).
- An output file that exists, or a stale `.partial`, without `--resume`. Shards are written as `.partial` and renamed
  when whole; a failed run removes what it made, and under `--resume` only the `.partial` it was writing.

## How its bytes are proved

- `just gate-quantize-oracle`: the host K-quant rules (`gguf::kquant`: Q3_K, Q4_K, Q5_K, Q6_K, Q8_0) against ggml's
  quantizer (`iq_quant_ref`, linked to the library llama-quantize calls), row for row, with and without an importance
  matrix, over seeded row families. The oracle's libggml-base has no FMA and no F16C instruction, so the host rules use
  none.
- `just gate-quantize-e2e`: the K3 fixture through the host writer (`--cpu`) against `llama-quantize`: whole-file
  sha256 equal, split and not, and with the trunk weights at Q3_K by rule.
- Plan equality: the types and bytes of every tensor equal `llama-quantize --dry-run`'s, tensor for tensor
  (`--llama-plan`, and the writer cross-checks it on every run that has one).
- `just gate-gpu-iq-quant`: the card's bytes against the oracle, `--cpu` equal to the card, `--check`, each refusal.
- `just gate-ernie55-quant` (ERNIE 5.5, the fixture cut and a toy folder, `--cpu`): each expert stack byte for byte the
  folder loader's made tensor and decoding to the OCP MXFP4 value of the stored planes; each Q8_0 block ggml's reference
  rule over the loader's FP8 values; the description reading the GGUF as the folder; under the lossless rule every tensor
  the loader's bytes, none Q8_0; a cut folder's `bloomery.fixture.*` keys in the GGUF.

## Room and time: the resource timeline

Per unit of work, what each resource is busy for (K3.1, the A6000, one stack = 9.865e9 weights of MXFP4 in, IQ1_S out;
the drive rates are the plan's constants, 1.6 GB/s read and 1.5 GB/s write):

| unit | drive read | card | host rules | write |
|---|---|---|---|---|
| one expert stack (~1.75 GB in, ~0.66 GB out) | ~1.1 s | ~14 s | none (the search runs on the card) | ~0.44 s |
| one trunk tensor (2e8 weights BF16 in, Q4_K out) | ~0.25 s | none | ~0.4 s on ~62 threads at ~8e6 weights/s/thread | ~0.02 s |
| all 276 stacks (3 per block x 92; 2.72e12 weights, 1.45 TB in, ~548 GB out) | ~900 s | ~3 950 s | none | ~365 s |
| the trunk (0.114 TB in, ~31 GB out, ~5.7e10 weights) | ~70 s | none | ~115 s | ~20 s |

The measured one-thread host rates (with importance) are Q4_K 9.7e6, Q5_K 9.6e6, Q6_K 3.0e7, Q8_0 4.2e8 weights/s, so
8e6 weights/s/thread is the conservative mix for the plan.

The wall is the card's: the stacks are card-bound by about 12 times over the drive's, and the trunk's host time is 3 % of
the card's. The trunk therefore runs under the card's shadow: a prefetch thread converts the host tensors a few tensors
ahead of the writer on the resident pool while the card works on a stack (the host path of the IQ rule does not
overlap: with `--cpu` the same pool does both and the wall is their sum). The writer takes tensors in output order, so
the file is the same whichever order the conversions finish. The predicted wall for K3.1 on the A6000 is the card's
~3 950 s plus the first read and the last write, ~4 400 s (1 h 13 min, the figure `tools/ref/k31-quant.sh --dry-run` prints);
the trunk adds none of it. Run serially after the card it would add ~115 s. No intermediate set is
written.

`--plan` prints the room the output needs and the derived wall; the wall is derived, not measured. A measured rate comes
from `just quant-measure`, a leased timing runner (`tools/ref/quant-measure.sh`, its prediction card
`docs/cards/quant-measure.card`).
