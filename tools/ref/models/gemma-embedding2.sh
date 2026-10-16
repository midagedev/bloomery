#!/usr/bin/env bash
# shellcheck shell=bash
# The gemma-embedding2 (EmbeddingGemma 2) profile: a bidirectional text encoder (24 layers of width 512, symmetric sliding
# window, head 512 on the global layers, per-layer input projection) whose token rows are projected to 768 and mean pooled,
# and its gemma4v image tower. Sourced by ref-paths.sh when BLOOMERY_MODEL=gemma-embedding2; never executed, and it exports
# nothing. The name is the GGUF general.architecture value, so `grep gemma-embedding2` finds this profile, the metadata keys
# and the refset families (crates/refset/src/arch/gemma_embedding2), which pin the same files, digests and commit.
#
#   MODEL          ggml-org/embeddinggemma-2-GGUF's Q8_0 (309,855,456 bytes, revision bfcd2987), the file the side seat runs.
#                  BLOOMERY_REF_MODEL moves it, and the families then refuse the sets as stale
#   MODEL_SHA256   its sha256
#   MMPROJ         the same repository's mmproj-embeddinggemma-2-BF16.gguf (982,074,784 bytes, projector gemma4v)
#   MMPROJ_SHA256  its sha256
#   LCPP           the mainline tree that is the oracle, built at LCPP_PIN by tools/ref/qvis/build-lcpp-qvis.sh
#                  (just build-emb2-lcpp), the dumper linked by tools/ref/clefvis/build-clefvis.sh (just build-emb2-ref)
#   IMAGES         the image cases by name: tools/ref/clefvis/images/<name>.png (make_images.py), small to large, one under
#                  the tower's minimum size and one a single row of pixels
#   IK             no ik oracle exists for this profile; the default tree is set for the scripts that expand IKBIN beside every
#                  profile
#   EMBED_VOCAB    the tokenizer oracle's vocabulary and set suffix
#   EMBED_CASES    the vocabulary's cases file under crates/tokenizer/tests
#   EMBED_CONTEXT  the server context and batch size
#   EMBED_STEPS    the oracle steps, card steps before CPU twins
#
# The text cases and their ids are the tokenizer oracle's EMBED_VOCAB set (crates/tokenizer/tools/oracle.sh).
#
# SC2034: every name here is read by the file that sources this one.
# shellcheck disable=SC2034
MODEL_NAME=gemma-embedding2
: "${IK:=/home/user/ik-idxkey}"
MODEL=${BLOOMERY_REF_MODEL:-/models/embeddinggemma-2/embeddinggemma-2-Q8_0.gguf}
MODEL_SHA256=2188ac1deca4b77dffefd603c2776a9d76d9d74ec01841392982ebb840b09135
MMPROJ=/models/embeddinggemma-2/mmproj-embeddinggemma-2-BF16.gguf
MMPROJ_SHA256=d2033b3cd0223cfb2e7a2b50dc80a664c78c816d375f5b017ec3ef80ee0bc766
: "${LCPP:=/home/user/llama.cpp-6868a52c}"
LCPP_PIN=6868a52cb337450658006ad2f9281c2eb63f1666
IMAGES=(small-200x200 min-96x64 free-448x448 free-640x480 pad-777x513 strip-1000x1 free-1024x768)

EMBED_VOCAB=gemma4
EMBED_CASES=cases-gemma4.txt
EMBED_CONTEXT=8192
EMBED_STEPS=(text tower server text.cpu tower.cpu server.cpu)
