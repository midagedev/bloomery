# bloomery-quantize

`bloomery-quantize` is the quantizer: it writes the final quantized GGUF set from the source set in one pass, with no
intermediate set. The routed-expert stacks become IQ1_S or IQ2_XXS on the card (or the host's cores), every other
tensor the type `llama-quantize` gives it by the host's K-quant rules (Q4_K, Q5_K, Q6_K, Q8_0, and the F32/F16 copies),
and the metadata, the tensor order, the alignment and the shard layout are the ones `llama-quantize` writes. The output
is byte-identical to `llama-quantize`'s for the same input, importance matrix, rules and options. `llama-quantize`
stays as the oracle: the gates compare against it.

It is in the Linux archives only. It links the CUDA backend, and cargo refuses to build it without the `gpu` feature,
so the Mac archive has no binary of it (`--cpu` included).

## Scope

The architecture is `kimi-k3`; the file types are Q4_K_M, Q5_K_M, Q6_K, Q8_0, IQ2_XXS and IQ1_S. Any other
architecture, type, or a type resolution that lands on a type not ported (Q2_K, Q3_K, ...) is refused by name before
anything is written.

    bloomery-quantize IN-00001-of-N.gguf OUT/prefix Q4_K_M --imatrix m.imatrix.gguf --tensor-type-file rules.txt \
        --output-tensor-type q6_k --token-embedding-type q4_k --keep-split --allow-requantize

## Usage

    bloomery-quantize [options] <in first shard> <out> <ftype> [nthreads]

    --imatrix FILE            the importance matrix
    --tensor-type-file FILE   llama-quantize's rules file
    --output-tensor-type T / --token-embedding-type T   the flag types, as llama-quantize's
    --allow-requantize        as llama-quantize's
    --keep-split              the output has the input's number of shards (`<out>-0000K-of-0000N.gguf`)
    --plan                    every tensor's source and target type, bytes in and out, the fate, the derived wall and
                              the room; writes nothing
    --llama-plan TSV          llama-quantize's own `--dry-run` plan, cross-checked tensor for tensor
    --check N                 N rows of every IQ tensor through the host rule too; stops by name on the first byte
    --cpu                     the IQ rule on all cores; no card
    --card N                  the card, by CUDA ordinal
    --batch N                 experts per card call

`nthreads` is accepted for llama-quantize's command line and ignored: the host rules run on the resident pool.
`--help` and `--version` print the usage and `bloomery-quantize <version> (commit <hash>)`.

Exit: 0 done; 1 a refusal, named; 64 a usage error; 66 a missing input file.

## The type rules

`--tensor-type-file` is llama-quantize's file: whitespace-separated `regex=type` tokens. The first rule whose regex
matches the lowercased tensor name wins. The matcher here is a subset of ECMAScript regex (literals, escaped
punctuation, `.`, `[...]`, `\d \w \s`, groups with `|`, `* + ?`, `^ $`); a pattern outside it is refused by name, not
guessed at. Only `*_exps.weight` stacks of three or more dimensions are quantized here. With `--llama-plan` a tensor
the rules and llama-quantize type differently is a refusal in either direction.

## What it refuses

Nothing is turned into a plausible value.

- A source row with an Inf or NaN value (F32, F16, BF16), or an MXFP4 block whose E8M0 scale byte is 255 (Inf): named
  with tensor, expert, row and value index.
- An imatrix entry that is Inf, NaN or negative, in the sums or in the counts: named with the tensor and entry.
- A source type the oracle does not cover (anything but F32, F16, BF16, MXFP4), a row that is not a multiple of 256, an
  importance row of the wrong length.
- Too little room for the output (checked before anything is written; the output directory's free bytes against the
  bytes the plan writes).
- An output file that exists, or a stale `.partial`. Shards are written as `.partial` and renamed when whole; a failed
  run removes what it made.

## How its bytes are proved

- `just gate-quantize-oracle`: the host K-quant rules (`gguf::kquant`) against ggml's quantizer (`iq_quant_ref`, linked
  to the library llama-quantize calls), row for row, with and without an importance matrix, over seeded row families.
  The oracle's libggml-base has no FMA and no F16C instruction, so the host rules use none.
- `just gate-quantize-e2e`: the K3 fixture through the host writer (`--cpu`) against `llama-quantize`: whole-file
  sha256 equal, split and not.
- Plan equality: the types and bytes of every tensor equal `llama-quantize --dry-run`'s, tensor for tensor
  (`--llama-plan`, and the writer cross-checks it on every run that has one).
- `just gate-gpu-iq-quant`: the card's bytes against the oracle, `--cpu` equal to the card, `--check`, each refusal.

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
