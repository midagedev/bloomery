#!/usr/bin/env bash
# shellcheck shell=bash
# The inkling2 (Inkling 2 Mini, unsloth's UD-Q4_K_XL looped file) profile: what the reference
# harness needs that is a property of the model. Sourced by ref-paths.sh when BLOOMERY_MODEL=inkling2;
# never executed, and it exports nothing.
#
# The name is the GGUF general.architecture value (the Small's loopless files declare it too), so
# `grep inkling2` finds this profile, the registry row and the fixture directory. No oracle yet: the
# reference engine's graph is a later round's, and the meta gate reads headers only.
#
#   MODEL   the Mini's UD-Q4_K_XL file (17.6 GB, one shard, the loop of the core [6, 18) three
#           times). BLOOMERY_REF_MODEL moves it; a caller's own MODEL= is ignored. The Small's
#           six-shard UD-Q2_K_XL under /models/Inkling-2-Small-GGUF/ is the loopless case; the
#           meta gate reads it by its own path.
#   The fixture the fixture tier's model: thirteen layers of the Mini — the stem block whole, one
#           whole core block, a coda layer of the Q8_0 experts — at the experts narrowed to 512,
#           crates/model/src/arch/inkling2/fixture.rs `spec`, planted by:
#             target/release/fixture generate /models/Inkling-2-Mini-GGUF/Inkling-2-Mini-UD-Q4_K_XL.gguf /models/fixtures/inkling2
#             target/release/fixture verify /models/fixtures/inkling2/inkling2-fixture-00001-of-00001.gguf
#
# SC2034: every name here is read by the file that sources this one.
# shellcheck disable=SC2034
MODEL_NAME=inkling2
MODEL=${BLOOMERY_REF_MODEL:-/models/Inkling-2-Mini-GGUF/Inkling-2-Mini-UD-Q4_K_XL.gguf}
