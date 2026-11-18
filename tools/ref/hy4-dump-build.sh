#!/usr/bin/env bash
# Link hy4_dump (tools/ref/cb_dump.cpp, with tools/ref/ref_io.h) against the Hy4-flash oracle tree: llama.cpp at the head of PR #32517
# (0a054d0b0, CPU, no patch, /home/user/llama.cpp-hy4, whose tracked files must be clean). This does not configure or rebuild that
# tree: a rebuild moves the libllama md5 the oracle sets name. Output: $BLOOMERY_DATA/bin/hy4_dump and hy4_dump.build beside it (HY4_OUT
# moves them), the record tools/ref/hy4-dump.sh compares: the source sha256 (cpp then headers), the tree's commit and the md5 of the
# libllama the binary loads. The build is tools/ref/cb-dump.sh's; the family's part is hy4-dump.sh.
set -euo pipefail
exec bash "${BASH_SOURCE[0]%/*}/hy4-dump.sh" --build
