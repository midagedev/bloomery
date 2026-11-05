#!/usr/bin/env bash
# Link kimik3_dump (tools/ref/cb_dump.cpp, with tools/ref/ref_io.h) against the Kimi K3 oracle tree: mainline llama.cpp at 78feec062
# (CPU, no patch, /home/user/llama.cpp-kimik3, `just build-k3-lcpp` builds it; its tracked files must be clean). This does not configure
# or rebuild that tree: a rebuild moves the libllama md5 the oracle sets name. Output: $BLOOMERY_DATA/bin/kimik3_dump and
# kimik3_dump.build beside it (KIMIK3_OUT moves them), the record tools/ref/kimi-k3-dump.sh compares: the source sha256 (cpp then headers),
# the tree's commit and the md5 of the libllama the binary loads. The build is tools/ref/cb-dump.sh's; the family's part is kimi-k3-dump.sh.
set -euo pipefail
exec bash "${BASH_SOURCE[0]%/*}/kimi-k3-dump.sh" --build
