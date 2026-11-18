#!/usr/bin/env bash
# shellcheck shell=bash
# The nemotron-light-folder profile: Nemotron 3.5 Lightning's ModelOpt NVFP4 checkpoint folder
# (nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4), the model of gate_nemotron_h_moe_folder. Sourced by ref-paths.sh when
# BLOOMERY_MODEL=nemotron-light-folder; never executed, and it exports nothing. The folder's architecture is nemotron_h_moe (the
# profile's name picks the folder, the architecture picks the engine); every other name is the nemotron_h_moe profile's.
#
#   MODEL     the real tier's model: the checkpoint folder. BLOOMERY_REF_MODEL moves it.
#   The cut   the fixture tier's model: layers 0, 1 and 5 of the folder (the first Mamba-2, MoE and attention layers, which the
#             committed header fixtures of crates/model-hf stand for every layer of their kind) in their stored bits, the
#             embedding, the NVFP4 head and the final norm, the tokenizer and chat files, `hf_quant_config.json`, no `mtp.*`:
#             tools/fixture-cuts/nemlightf.json, in /models/fixtures/nemlightf. The card budget is above the cut's tensor
#             data (1.72 GB), as the family fixture's (nemotron_h_moe/fixture.rs CARD_BUDGET), so the folder plans whole on
#             one card; the first shard records it (bloomery.fixture.card_budget), which the fixture tier's box command reads:
#               cargo build --release -p bloomery-model-hf --bin fixture
#               target/release/fixture cut /models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
#                 /models/fixtures/nemlightf --spec tools/fixture-cuts/nemlightf.json --card-budget 4G
#               target/release/fixture verify /models/fixtures/nemlightf --source /models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4
__lightf_dir=${BASH_SOURCE[0]%/*}
[ "$__lightf_dir" != "${BASH_SOURCE[0]}" ] || __lightf_dir=.
# shellcheck source=tools/ref/models/nemotron_h_moe.sh
source "$__lightf_dir/nemotron_h_moe.sh"
unset __lightf_dir
# SC2034: read by the file that sources this one.
# shellcheck disable=SC2034
MODEL_NAME=nemotron-light-folder
MODEL=${BLOOMERY_REF_MODEL:-/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4}
