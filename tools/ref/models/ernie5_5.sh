#!/usr/bin/env bash
# shellcheck shell=bash
# The ernie5_5 profile (ERNIE 5.5 Flash Preview, baidu/ERNIE-5.5-Flash-Preview): what the gates need that is a property of
# the model. Sourced by ref-paths.sh when BLOOMERY_MODEL=ernie5_5; never executed, and it exports nothing.
#
# The name is the GGUF general.architecture value the folder map writes (gguf::arch::ernie55::ARCH), so `grep ernie5_5` finds this
# profile, the registry row, the fixture directory and the family (`ernie55`, tools/families.tsv).
#
#   MODEL   the vendor's checkpoint folder (128 safetensors shards: FP8 128-block projections, MXFP4 routed experts, BF16
#           embedding and head, with the DSpark draft `model.mtp.*` and the FlashIndexer file beside the index): the engine and
#           the gates read it in its own formats, with no GGUF. BLOOMERY_REF_MODEL moves it. The fixture tier's model is the
#           folder `fixture cut` writes (tools/fixture-dirs.tsv, tools/fixture-cuts/ernie55.json): layers 0 (HCA, dense), 1 (HCA,
#           routed) and 7 (latent attention with the DSA indexer, routed) in their stored bits, the dense width 1,024 and the
#           expert width 512, resolved by ref-paths.sh. The cut's card budget (8G) is above its tensor data, so the folder plans
#           whole on one card; the first shard records it (bloomery.fixture.card_budget), which the fixture tier reads:
#             cargo build --release -p bloomery-model-hf --bin fixture
#             target/release/fixture cut /models/ERNIE-5.5-Flash-Preview /models/fixtures/ernie55 \
#               --spec tools/fixture-cuts/ernie55.json --card-budget 8G
#             target/release/fixture verify /models/fixtures/ernie55 --source /models/ERNIE-5.5-Flash-Preview
#   GGUF_MODEL
#           the GGUF `bloomery-quantize --hf` writes of MODEL by the lossless rule (docs/quantize.md): every tensor the
#           folder loader makes, the MXFP4 expert stacks as runs of the file. A folder's stacks are made at load, so no load
#           from the folder splits one between the card and the host; a hosted load opens this file instead (gate_ernie55_e2e's
#           hosted arms, its `--hosted-gguf`; the recipe takes the path from ref-paths.sh's gguf_model). BLOOMERY_REF_GGUF_MODEL
#           moves it. The real tier's is shard 1 of the released folder's set (4 shards, `--keep-split`, 155.08 GB):
#             cargo oxide build --arch $BLOOMERY_ARCH -- -p bloomery-gpu-gates --features gpu,hf --release --bin bloomery-quantize
#             target/release/bloomery-quantize --hf /models/ERNIE-5.5-Flash-Preview --tensor-type-file RULES --allow-requantize \
#               --keep-split --resume --cpu /models/ERNIE-5.5-Flash-Preview-GGUF/ERNIE-5.5-Flash-Preview-F8M4.gguf Q8_0
#   GGUF_FIXTURE
#           ernie55-gguf/ernie55-cut.fp8.gguf: the cut's GGUF under the fixture root ($BLOOMERY_FIXTURE_ROOT, default
#           /models/fixtures), in a directory of the family's own beside its fixture directory (ernie55), because
#           refset::fixture refuses a folder and a GGUF in one directory. The fixture tier's GGUF_MODEL, resolved by ref-paths.sh
#           as the fixture MODEL is, a caller's BLOOMERY_REF_GGUF_MODEL over it; a file that is not there is gguf_model's
#           named refusal, never a run of a hosted arm on another file. The cut's `bloomery.fixture.*` keys carry into the
#           GGUF, so the fixture tier opens it as the fixture it is. RULES is the one line `\.weight$=fp8_b128` (every FP8
#           128-block stays as the loader makes it, nothing is quantized, so no --allow-requantize); `just gate-ernie55-quant`
#           holds what the tool writes to the cut's bytes. The cut's file was written from the cut:
#             target/release/bloomery-quantize --hf /models/fixtures/ernie55 /models/fixtures/ernie55-gguf/ernie55-cut.fp8.gguf \
#               Q8_0 --tensor-type-file RULES --cpu
#
# SC2034: every name here is read by the file that sources this one.
# shellcheck disable=SC2034
MODEL_NAME=ernie5_5
MODEL=${BLOOMERY_REF_MODEL:-/models/ERNIE-5.5-Flash-Preview}
GGUF_MODEL=${BLOOMERY_REF_GGUF_MODEL:-/models/ERNIE-5.5-Flash-Preview-GGUF/ERNIE-5.5-Flash-Preview-F8M4-00001-of-00004.gguf}
GGUF_FIXTURE=ernie55-gguf/ernie55-cut.fp8.gguf
