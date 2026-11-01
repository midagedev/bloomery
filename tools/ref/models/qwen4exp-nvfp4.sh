#!/usr/bin/env bash
# shellcheck shell=bash
# The qwen4exp-nvfp4 profile: Qwen3.8-Flash-Next from NVIDIA's NVFP4 checkpoint (nvidia/Qwen3.8-Flash-Next-NVFP4 at fc694b54),
# the model of gate_qwen4exp_nvfp4. Sourced by ref-paths.sh when BLOOMERY_MODEL=qwen4exp-nvfp4; never executed, and it exports
# nothing. The file's GGUF general.architecture is qwen4exp (the profile's name picks the file, the architecture picks the
# engine); the qwen4exp profile keeps unsloth's K-quant split and its GGUF fixture for the other gates.
#
#   MODEL       the real tier's model: the split `prepare` writes from the checkpoint folder, shard 1. BLOOMERY_REF_MODEL moves it,
#               to prepare's stored-form output or to the checkpoint folder itself, which the load reads as stored.
#   The cut     the fixture tier's model: layers 0 to 3 of the checkpoint (three delta-rule layers, then a selecting attention
#               layer: the checkpoint's interval of 4) in their stored bits, every routed stack NVFP4, the dense projections BF16,
#               layer 1's PLE site with its FP8 table (52.8 GB of the cut's 61.6 GB, read by rows where the folder stores them),
#               the embedding, the head and the final mixer; no vision tower and no `mtp.*`: tools/fixture-cuts/qwen38nvfp4.json,
#               in /models/fixtures/qwen38nvfp4. The card budget holds every routed expert of the cut with the cache and scratch
#               of the gate's 3,072 rows:
#                 cargo build --release -p bloomery-model-hf --bin fixture
#                 target/release/fixture cut /models/nvidia__Qwen3.8-Flash-Next-NVFP4/fc694b54fb0174e0913e6adf86691ef85a4ead47 \
#                   /models/fixtures/qwen38nvfp4 --spec tools/fixture-cuts/qwen38nvfp4.json --card-budget 16G
#                 target/release/fixture verify /models/fixtures/qwen38nvfp4 \
#                   --source /models/nvidia__Qwen3.8-Flash-Next-NVFP4/fc694b54fb0174e0913e6adf86691ef85a4ead47
#               Its oracle sets are the NVFP4 oracle's on the cut (`BLOOMERY_TIER=fixture just dump-ref-qwen4exp-nvfp4 IDS`,
#               refset family fx-hf-qwen4exp-nvfp4)
#   IK          the default the scripts that expand IKBIN beside every profile take; no ik oracle exists for this profile
#
# SC2034: every name here is read by the file that sources this one, which shellcheck does not see from this file alone.
# shellcheck disable=SC2034
MODEL_NAME=qwen4exp-nvfp4
: "${IK:=/home/user/ik-idxkey}"
MODEL=${BLOOMERY_REF_MODEL:-/models/Qwen3.8-Flash-Next-NVFP4/Qwen3.8-Flash-Next-NVFP4-00001-of-00004.gguf}
