#!/usr/bin/env bash
# shellcheck shell=bash
# The `llama` architecture's profile: the registry row of the `llama` architecture names its fixture directory in
# tools/fixture-dirs.tsv, and the fixture tier resolves a directory through the profile of the row's name, so the one
# model of this architecture (Nanbeige4.3-33B) is also reachable under it. Sourced by ref-paths.sh when
# BLOOMERY_MODEL=llama; never executed. Everything is the nanbeige43 profile's (tools/ref/models/nanbeige43.sh); the gates
# and the oracle sets name that profile.
# shellcheck source=tools/ref/models/nanbeige43.sh
source "${BASH_SOURCE[0]%/*}/nanbeige43.sh"
# The one line restated: tools/recipes.py profile_dirs reads the /models directory a profile defines from a literal in the profile's own file.
# shellcheck disable=SC2034
MODEL=${BLOOMERY_REF_MODEL:-/models/Nanbeige4.3-33B/Nanbeige_Nanbeige4.3-33B-Q4_K_M.gguf}
