#!/usr/bin/env bash
# Link mistrallarge4_dump (tools/ref/cb_dump.cpp, with tools/ref/ref_io.h) against the existing mainline build
# of the mistral-large4 profile ($LCPP/build/bin: llama.cpp 78feec062, PR 31376's merge, CPU, no patch; its
# tracked files must be clean). This does not configure or rebuild that tree (`just build-ml4-lcpp` builds it): a
# rebuild moves the libllama md5 the oracle sets name. Output: $BLOOMERY_DATA/bin/mistrallarge4_dump and
# mistrallarge4_dump.build beside it (MISTRALLARGE4_OUT moves them), the record tools/ref/mistrallarge4-dump.sh
# compares: the source sha256 (cpp then header), the tree's commit and the md5 of the libllama the binary loads — the
# 12 digits the refset family pins. The build is tools/ref/cb-dump.sh's; the family's part is mistrallarge4-dump.sh.
set -euo pipefail
exec bash "${BASH_SOURCE[0]%/*}/mistrallarge4-dump.sh" --build
