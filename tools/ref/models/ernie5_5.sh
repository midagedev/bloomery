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
#
# SC2034: every name here is read by the file that sources this one.
# shellcheck disable=SC2034
MODEL_NAME=ernie5_5
MODEL=${BLOOMERY_REF_MODEL:-/models/ERNIE-5.5-Flash-Preview}
