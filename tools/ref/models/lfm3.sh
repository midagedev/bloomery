#!/usr/bin/env bash
# shellcheck shell=bash
# The lfm3 (LFM3-1.7B, text) profile: what the reference harness needs that is a property of the model. Sourced by
# ref-paths.sh when BLOOMERY_MODEL=lfm3; never executed, and it exports nothing.
#
# The name is the GGUF general.architecture value, so `grep lfm3` finds this profile, the metadata keys (lfm3.*) and the
# logits sets (crates/refset/src/arch/lfm3).
#
#   MODEL           the Q4_K_M file, the headline one. BLOOMERY_REF_MODEL moves it; a caller's own MODEL= is ignored.
#   LCPP            mainline llama.cpp at LCPP_COMMIT, the commit that added the architecture. ik has no lfm3, so this tree
#                   is the oracle. A tree of its own (tools/ref/qvis/build-lcpp-qvis.sh, `just build-lfm3-lcpp`): a rebuild of
#                   /home/user/llama.cpp-mainline would move the md5s the nemotron_h logits family pins.
#   IK              no ik oracle. The default is only so IKBIN can be formed; nothing here runs it.
#   LCPPBIN         mainline llama-bench of that tree, the depth runner's reference arms (lcpp:, lcpppp:).
#   LCPP_GPU_FLAGS  the whole model on the card the run holds, flash attention on.
#   LFM3_REPO, LFM3_REVISION
#                   the Hugging Face repository the three files come from and its revision, which the `REVISION` file
#                   beside them states (tools/ref/fetch-lfm3.sh writes it after every file matches the table below, and
#                   tools/ref/dump-lfm3.sh refuses a directory whose REVISION is another).
#   LOGITS_SETS     `<set>:<file>:<bytes>:<sha256>` of each file of the repository at LFM3_REVISION that has a logits set
#                   (refset families ref-lfm3-logits, ref-lfm3-q8-logits, ref-lfm3-bf16-logits: one family a file): the
#                   Q4_K_M file is the headline, the Q8_0 file serves the Q8_0 sites, the BF16 file is the one with no
#                   quantization noise. Bytes and sha256 are the repository's own (its LFS pointers).
#
# The logits dump takes no lease and no card: the files are 1.1, 1.8 and 3.5 GB and the dump hides CUDA
# (tools/ref/dump-lfm3.sh over tools/ref/dump-nemotron.sh, the one dump script of the last-position logits sets).
#
# SC2034: every name here is read by the file that sources this one.
# shellcheck disable=SC2034
MODEL_NAME=lfm3
: "${IK:=/home/user/ik-idxkey}"
LFM3_REPO=LiquidAI/LFM3-1.7B-GGUF
LFM3_REVISION=b3f2ae5d73cd3c91bf40e8ba752c648aa05dffdf
LCPP_COMMIT=19cce6919992fdffd92e0ac152caaf1e2b8f254e
: "${LCPP:=/home/user/llama.cpp-19cce691}"
MODEL=${BLOOMERY_REF_MODEL:-/models/LFM3-1.7B-GGUF/LFM3-1.7B-Q4_K_M.gguf}
: "${LCPPBIN:=$LCPP/build/bin/llama-bench}"
: "${LCPP_GPU_FLAGS:=-ngl 99 -fa on}"
LOGITS_SETS=(
  "ref-lfm3-logits:/models/LFM3-1.7B-GGUF/LFM3-1.7B-Q4_K_M.gguf:1099627808:3793e1e6e2266919d6e015287221b41100175a98eddaba92e111aed3b05c93c4"
  "ref-lfm3-q8-logits:/models/LFM3-1.7B-GGUF/LFM3-1.7B-Q8_0.gguf:1846238496:8f6b24182258205e9e01e21f1f24a1f208c1e8115578825d202ce39976af20b6"
  "ref-lfm3-bf16-logits:/models/LFM3-1.7B-GGUF/LFM3-1.7B-BF16.gguf:3466626336:39158a79f56edb85e0c4252ce6447b8d7883ef82ffedd02f0cda7c84da5e9f68"
)
