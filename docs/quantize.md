# bloomery-quantize

`bloomery-quantize` quantizes the routed-expert stacks of a GGUF set to IQ1_S or IQ2_XXS, on the card or on the host's
cores. Its bytes are the bytes ggml's `llama-quantize` writes for the same input and importance matrix. It exists so
that a very large model's expert stacks, the bulk of the work, do not wait on one CPU thread per tensor.

It is in the Linux archives only. It links the CUDA backend, and cargo refuses to build it without the `gpu` feature,
so the Mac archive has no binary of it (`--cpu` included).

## What it does and does not do

It writes a set that holds the quantized expert stacks and every other tensor copied as it is. It does not quantize the
trunk (attention, norms, embeddings, the output head). `llama-quantize` does that: it copies a tensor whose type
already equals its target, so running it on the set `bloomery-quantize` wrote gives the same file as running it on the
original input alone.

    bloomery-quantize IN-00001-of-N.gguf OUT/prefix --imatrix m.imatrix.gguf --tensor-type-file rules.txt
    llama-quantize --imatrix m.imatrix.gguf --tensor-type-file rules.txt OUT/prefix-00001-of-N.gguf FINAL.gguf <type>

`--plan` prints the exact `llama-quantize` command for the set.

## Usage

    bloomery-quantize <in first shard> <out prefix> --imatrix FILE --tensor-type-file FILE [options]

    --type IQ1_S|IQ2_XXS  only the stacks the rules type this
    --plan                every tensor's source and target type, bytes in and out, quantized-here or copied, the
                          derived wall, the room needed and the llama-quantize command; writes nothing
    --check N             N rows of every quantized tensor through the host rule too; stops by name
                          (tensor, expert, row, byte) on the first byte that differs
    --cpu                 the host rule on all cores; no card
    --card N              the card, by CUDA ordinal
    --batch N             experts per card call
    --llama-plan TSV      llama-quantize's own `--dry-run` plan, cross-checked against the rules

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

`just gate-gpu-iq-quant` holds the clauses: the card's bytes against ggml's own quantizer (`tools/ref/iq_quant_ref.cpp`,
linked to the library llama-quantize calls) for F32, F16, BF16 and MXFP4 sources; `--cpu` equal to the card; `--check`
stopping on a flipped byte; each refusal above; and an end-to-end clause on the K3 fixture, where `bloomery-quantize`
then `llama-quantize` equals `llama-quantize` alone byte for byte.

## Room and time

`--plan` prints the room the output needs and a wall derived from the card rate and the drive rates in the plan's
constants; the wall is derived, not measured. A measured rate comes from `just quant-measure`, a leased timing runner
(`tools/ref/quant-measure.sh`, its prediction card `docs/cards/quant-measure.card`).
