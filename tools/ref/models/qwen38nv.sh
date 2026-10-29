#!/usr/bin/env bash
# shellcheck shell=bash
# The qwen38nv profile: the GGUF twin of the Qwen3.8-27B NVFP4 fixture cut, the file mainline llama.cpp reads where ours reads the
# cut folder. What the reference engine and the harnesses need that is a property of the *model*, not of the machine. Sourced by
# ref-paths.sh when BLOOMERY_MODEL=qwen38nv; never executed, and it exports nothing. The writer profile of the family
# `logits-qwen38nv` (crates/refset/src/arch/qwen35/dense.rs); the fixture tier's row of the model is the `qwen38d` profile's
# (tools/fixture-dirs.tsv: the cut folder), so this profile has no fixture row of its own and runs in the real tier only.
#
#   The cut     nvidia/Qwen3.8-27B-NVFP4 at 482ca0f3 (ModelOpt MIXED_PRECISION: NVFP4 feed-forward and head, FP8 attention and
#               delta-rule projections), layers 0 to 3 (three delta-rule layers, one attention layer), the embedding, the head
#               and the final norm, no vision tower and no `mtp.*` (tools/fixture-cuts/qwen38nv.json), in /models/fixtures/qwen38nv:
#                 cargo build --release -p bloomery-model-hf --bin fixture
#                 target/release/fixture cut /models/Qwen3.8-27B-NVFP4 /models/fixtures/qwen38nv --spec tools/fixture-cuts/qwen38nv.json
#                 target/release/fixture verify /models/fixtures/qwen38nv --source /models/Qwen3.8-27B-NVFP4
#   MODEL       the twin: mainline's converter (convert_hf_to_gguf.py at the LCPP_PIN commit) over the cut folder, in a directory
#               of its own under the cut's (the fixture's first-shard glob does not see it):
#                 python3 $LCPP/convert_hf_to_gguf.py /models/fixtures/qwen38nv --no-mtp --outtype bf16 --fp8-as-q8 --outfile /models/fixtures/qwen38nv/twin/qwen38nv-twin.gguf
#               NVFP4 blocks go through as they are (36 bytes per 64 values, the per-block UE4M3 scale bits kept, the tensor scale
#               and the input scale as F32 scalars beside the weight); the FP8 projections are dequantized and written Q8_0
#               (--fp8-as-q8); the embedding, the beta/alpha projections stay BF16 as stored (--outtype bf16), the gammas, taps
#               and vectors F32. These are the types the folder open gives the same tensors, so the twin is the cut folder's own
#               bits in GGUF form; the cut's `text_config.mtp_num_hidden_layers` stays 1 and --no-mtp is what keeps the converter from
#               asking for the dropped block. `python3 tools/ref/dense-logits.py identity <MODEL>` measures the size and sha256
#               that tools/ref/dense-logits.py TARGETS holds for it (None until then: a dump refuses by name). BLOOMERY_REF_MODEL
#               moves it
#   LCPP        the mainline llama.cpp tree that converts and is the numeric oracle: the qwen35 profile's, at LCPP_PIN (NVFP4 is
#               GGML_TYPE_NVFP4 there, with CUDA and CPU kernels; ik reads none, so no ik oracle exists for this profile).
#               dense_logits_ref (tools/ref/dense_logits_ref.cpp) is the binary the qwen38d profile builds; the same one writes these
#               sets (`just build-dense-logits-ref`). IK is the default the scripts that expand IKBIN beside every profile take
#   PROSE_IDS   the prose corpus, the qwen35 profile's file: this model's vocabulary is the 27B's byte for byte, so the ids are one
#               file and PROSE_SHA256 is the check
#   LOGITS_TARGET   tools/ref/dense-logits.py TARGETS' name for MODEL
#   LOGITS_SETS     the oracle sets, `<set>:<P>`, as the qwen38d profile's: the first P ids of PROSE_IDS the prompt, the LOGITS_FORCED
#               ids after them forced; each row mainline's logits (tools/ref/dense-logits.sh), `--cpu-twin` giving each set its CPU
#               floor beside it (<set>.cpu) for the gate's band
#
# SC2034: every name here is read by the file that sources this one, which shellcheck does not see from this file alone.
# shellcheck disable=SC2034
MODEL_NAME=qwen38nv
: "${IK:=/home/user/ik-idxkey}"
MODEL=${BLOOMERY_REF_MODEL:-/models/fixtures/qwen38nv/twin/qwen38nv-twin.gguf}
: "${LCPP:=/home/user/llama.cpp-mainline}"
LCPP_PIN=53ed051ce5e8193652e449f43216ca3859454f49
PROSE_IDS=${BLOOMERY_DATA:-/root/bloomery-data}/qwen35/corpus-prose.ids
PROSE_SHA256=4bdf4171c7e1d0fd37d61b4fd13fdfb66c0dea1806d03b0398f851da5a350223
LOGITS_TARGET=qwen38nv-twin
LOGITS_FORCED=32
LOGITS_CTX=4352
LOGITS_UBATCH=512
LOGITS_SETS=(ref_qwen38nv_logits_p64:64 ref_qwen38nv_logits_p4096:4096)
