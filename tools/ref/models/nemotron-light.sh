#!/usr/bin/env bash
# shellcheck shell=bash
# The nemotron-light profile: Nemotron 3.5 Lightning's shapes (no latent, 128 experts of 1,856 at the top 6, Q5_0, Q5_1 and Q8_0
# stacks), the model of gate_nemotron_h_moe_card32. Sourced by ref-paths.sh when BLOOMERY_MODEL=nemotron-light; never executed, and it
# exports nothing. The file's GGUF general.architecture is nemotron_h_moe (the profile's name picks the file, the architecture picks
# the engine); every other name is the nemotron_h_moe profile's, so the oracle tree and the reference arms are the same.
#
#   MODEL       the real tier's model: the unsloth UD-Q4_K_XL file, which the box does not keep (no gate loads it): BLOOMERY_REF_MODEL
#               names a file of a contributor's.
#   The fixture the fixture tier's model: blocks 0, 1, 3 and 5 of that file (a Mamba-2, two MoE and an attention block; the
#               Bolt fixture has a latent and IQ2_XXS stacks, this one the 32-value experts and the no-latent MoE),
#               crates/model/src/arch/nemotron_h_moe/fixture.rs `light_spec`, made from the hub's header with no real file:
#                 target/release/fixture fetch nemlight /models/fixtures/nemlight
#                 target/release/fixture verify /models/fixtures/nemlight --profile nemotron-light
#               and its variant `nemmx` (tools/fixture-variants.tsv, `just fixture-requant nemotron-light nemmx`).
#
# SC2034: every name here is read by the file that sources this one.
__light_dir=${BASH_SOURCE[0]%/*}
[ "$__light_dir" != "${BASH_SOURCE[0]}" ] || __light_dir=.
# shellcheck source=tools/ref/models/nemotron_h_moe.sh
source "$__light_dir/nemotron_h_moe.sh"
unset __light_dir
# shellcheck disable=SC2034
MODEL_NAME=nemotron-light
MODEL=${BLOOMERY_REF_MODEL:-/models/Nemotron-3.5-Lightning/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-UD-Q4_K_XL.gguf}
