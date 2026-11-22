#!/usr/bin/env bash
# Link inkling2_dump (tools/ref/cb_dump.cpp, with tools/ref/ref_io.h) against the Inkling 2 oracle tree: llama.cpp PR #32695
# at 6196d9802 carrying tools/ref/inkling2-scale.patch (upstream master's a1f422260 unedited, #32657, which the PR branch
# lacks; without it the loader refuses the family's files at the tensor count), CPU, /home/user/llama.cpp-pr32695-scale, whose
# tracked changes must be exactly that patch. This does not configure or rebuild that tree: a rebuild moves the libllama md5
# the oracle sets name. Output: $BLOOMERY_DATA/bin/inkling2_dump and inkling2_dump.build beside it (INKLING2_OUT moves
# them), the record tools/ref/inkling2-dump.sh compares: the source sha256 (cpp then headers), the tree's commit, the
# patch's sha256 and the md5 of the libllama the binary loads. The build is tools/ref/cb-dump.sh's; the family's part is
# inkling2-dump.sh.
set -euo pipefail
exec bash "${BASH_SOURCE[0]%/*}/inkling2-dump.sh" --build
