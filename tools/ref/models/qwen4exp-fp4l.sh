#!/usr/bin/env bash
# shellcheck shell=bash
# The qwen4exp-fp4l profile: Qwen3.8-Flash-Next from NVIDIA's FP4-Local checkpoint (nvidia/Qwen3.8-Flash-Next-FP4-Local at
# b9ce18a6), whose routed stacks store their block scales compressed (`weight_scale_encoding` csf) and whose dense projections
# are MXFP8. Sourced by ref-paths.sh when BLOOMERY_MODEL=qwen4exp-fp4l; never executed, and it exports nothing. The file's GGUF
# general.architecture is qwen4exp.
#
#   MODEL       the checkpoint folder, which the load reads as stored. BLOOMERY_REF_MODEL moves it.
#   The cut     the fixture tier's model, which gate_qwen4exp_nvfp4's fixture tier loads beside the qwen4exp-nvfp4 cut: the same
#               layers 0 to 3 in their stored bits, layer 1's PLE site with its NVFP4 table among them, the embedding, the head
#               and the final mixer; no vision tower and no `mtp.*` (tools/fixture-cuts/qwen38fp4l.json), in
#               /models/fixtures/qwen38fp4l. It is planned under the qwen4exp-nvfp4 cut's card budget, which its smaller dense
#               projections fit:
#                 cargo build --release -p bloomery-model-hf --bin fixture
#                 target/release/fixture cut /models/nvidia__Qwen3.8-Flash-Next-FP4-Local/b9ce18a6887e18c089fdea8509a33e63429a3300 \
#                   /models/fixtures/qwen38fp4l --spec tools/fixture-cuts/qwen38fp4l.json --card-budget 16G
#                 target/release/fixture verify /models/fixtures/qwen38fp4l \
#                   --source /models/nvidia__Qwen3.8-Flash-Next-FP4-Local/b9ce18a6887e18c089fdea8509a33e63429a3300
#               No oracle set holds its forward
#   IK          the default the scripts that expand IKBIN beside every profile take; no ik oracle exists for this profile
#
# SC2034: every name here is read by the file that sources this one, which shellcheck does not see from this file alone.
# shellcheck disable=SC2034
MODEL_NAME=qwen4exp-fp4l
: "${IK:=/home/user/ik-idxkey}"
MODEL=${BLOOMERY_REF_MODEL:-/models/nvidia__Qwen3.8-Flash-Next-FP4-Local/b9ce18a6887e18c089fdea8509a33e63429a3300}
