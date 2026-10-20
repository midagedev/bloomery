#!/usr/bin/env bash
# Link minimaxm3_dump (tools/ref/cb_dump.cpp, with tools/ref/ref_io.h) against the existing mainline build of the minimax-m3
# profile ($LCPP/build/bin: llama.cpp cd0a5b6a1, CPU, no patch; its tracked files must be clean). This does not configure or
# rebuild that tree: a rebuild moves the libllama md5 the oracle sets name. Output: $BLOOMERY_DATA/bin/minimaxm3_dump and
# minimaxm3_dump.build beside it (MINIMAXM3_OUT moves them), the record tools/ref/minimaxm3-dump.sh compares: the source sha256
# (cpp then header), the tree's commit and the md5 of the libllama the binary loads. The build is tools/ref/cb-dump.sh's; the
# family's part is minimaxm3-dump.sh.
set -euo pipefail
exec bash "${BASH_SOURCE[0]%/*}/minimaxm3-dump.sh" --build
