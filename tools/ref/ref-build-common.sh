#!/usr/bin/env bash
# Shared setup for the reference-harness build scripts (tools/ref/build*.sh). This file is
# sourced, never executed, so it only defines variables and functions. The caller sets
# `set -euo pipefail` BEFORE sourcing it, so that a missing or broken copy of this file stops
# the build instead of leaving IK and the flag arrays empty:
#
#   set -euo pipefail
#   # shellcheck source=tools/ref/ref-build-common.sh
#   source "$(dirname "${BASH_SOURCE[0]}")/ref-build-common.sh"
#
# What it owns, so that seven scripts cannot link against seven different ggml builds: the ik
# tree (IK), the data root, the repo root, the compiler and its base flags, and the include and
# link flags for libggml and for libllama+libcommon. Each script keeps what is its own: extra
# include roots and ISA flags, the source file, the output name, and its *_OUT override.
#
# Binaries live outside the tree: tools/box.sh rsyncs it with --delete before every command.
#
# SC2034: every variable here is read by the sourcing script, which shellcheck does not see when
# it checks this file alone. SC2054: the commas are inside one linker argument (-Wl,-rpath,DIR).
# shellcheck disable=SC2034,SC2054
: "${IK:=/home/user/ik_llama.cpp}"
BLOOMERY_DATA=${BLOOMERY_DATA:-/root/bloomery-data}
# Default output directory of the installed reference binaries.
REF_BIN=$BLOOMERY_DATA/bin
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

# The libggml directory is named once: -L and -rpath must point at the same build, or a
# harness links one ggml and loads another at run time.
REF_GGML_LIB=$IK/build/ggml/src
# libggml alone.
REF_GGML_INC=(-I"$IK/ggml/include")
REF_GGML_LINK=(-L"$REF_GGML_LIB" -lggml -Wl,-rpath,"$REF_GGML_LIB")
# libllama + libcommon on top of libggml (the harnesses that load a model through llama.h).
# -l order is the link order; keep ggml last.
REF_LLAMA_INC=("${REF_GGML_INC[@]}" -I"$IK/include" -I"$IK/common" -I"$IK/src")
REF_LLAMA_LINK=(-L"$IK/build/common" -L"$IK/build/src" -L"$REF_GGML_LIB"
  -lcommon -lllama -lggml
  -Wl,-rpath,"$IK/build/src" -Wl,-rpath,"$REF_GGML_LIB")

# ref_cxx <flags...>: the one compiler invocation every harness goes through.
ref_cxx() { g++ -std=c++17 -O2 "$@"; }
