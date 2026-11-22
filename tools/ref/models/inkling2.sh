#!/usr/bin/env bash
# shellcheck shell=bash
# The inkling2 (Inkling 2 Mini, unsloth's UD-Q4_K_XL looped file) profile: what the reference
# harness needs that is a property of the model. Sourced by ref-paths.sh when BLOOMERY_MODEL=inkling2;
# never executed, and it exports nothing.
#
# The name is the GGUF general.architecture value (the Small's loopless files declare it too), so
# `grep inkling2` finds this profile, the registry row and the fixture directory. The oracle tree
# (LCPP, below) runs the family's files: its tokenizer made the prompts' ids and its forward pass
# dumps the node sets; the meta gate reads headers only.
#
#   MODEL   the Mini's UD-Q4_K_XL file (17.6 GB, one shard, the loop of the core [6, 18) three
#           times). BLOOMERY_REF_MODEL moves it; a caller's own MODEL= is ignored. The Small's
#           six-shard UD-Q2_K_XL under /models/Inkling-2-Small-GGUF/ is the loopless case; the
#           meta gate reads it by its own path.
#   LCPP    the family's oracle tree: PR #32695 at its head 6196d9802 ("convert: register the looped
#           Inkling 2 architecture separately", the naming the Mini's file uses) carrying
#           tools/ref/inkling2-scale.patch — upstream master's a1f422260 ("model : load
#           ffn_gate_up_exps scales", #32657) unedited, which the PR branch lacks: without it the
#           loader refuses both the Mini's and the fixture's files at the tensor count (their fused
#           NVFP4 stacks' per-expert `ffn_gate_up_exps.scale` rows are never created). The tree is
#           /home/user/llama.cpp-pr32695-scale, its own directory (the unpatched
#           /home/user/llama.cpp-pr32695, `tools/ref/build-lcpp-pr.sh build 32695 cpu`, stays at the
#           plain head), the CPU build `just build-inkling2-lcpp` writes; the oracle
#           (tools/ref/inkling2-dump.sh) and the prompts' ids (crates/tokenizer/tools/
#           inkling2-oracle.sh) both come from it.
#   The fixture the fixture tier's model: thirteen layers of the Mini — the stem block whole, one
#           whole core block, a coda layer of the Q8_0 experts — at the experts narrowed to 512,
#           crates/model/src/arch/inkling2/fixture.rs `spec`, planted by:
#             target/release/fixture generate /models/Inkling-2-Mini-GGUF/Inkling-2-Mini-UD-Q4_K_XL.gguf /models/fixtures/inkling2
#             target/release/fixture verify /models/fixtures/inkling2/inkling2-fixture-00001-of-00001.gguf
#
# SC2034: every name here is read by the file that sources this one.
# shellcheck disable=SC2034
MODEL_NAME=inkling2
# The default IK is only so IKBIN can be formed: ik has no inkling2.
: "${IK:=/home/user/ik-idxkey}"
: "${LCPP:=/home/user/llama.cpp-pr32695-scale}"
MODEL=${BLOOMERY_REF_MODEL:-/models/Inkling-2-Mini-GGUF/Inkling-2-Mini-UD-Q4_K_XL.gguf}
