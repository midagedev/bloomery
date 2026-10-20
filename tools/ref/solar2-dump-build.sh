#!/usr/bin/env bash
# Link solar2_dump (tools/ref/cb_dump.cpp, with tools/ref/ref_io.h) against the existing PAI build of the solar-open2 profile
# ($PAI/build/bin: llama.cpp 6ea215d17 plus prometheusAIR's solar_open2 patch). This does not configure or rebuild that
# tree: a rebuild moves the libllama md5 the oracle sets name. Output: $BLOOMERY_DATA/bin/solar2_dump and solar2_dump.build
# beside it (SOLAR2_OUT moves them), the record tools/ref/solar2-dump.sh compares: the source sha256 (cpp then header), the tree's
# commit, the patch's sha256 and the md5 of the libllama the binary loads. The build is tools/ref/cb-dump.sh's; the family's
# part is solar2-dump.sh.
set -euo pipefail
exec bash "${BASH_SOURCE[0]%/*}/solar2-dump.sh" --build
