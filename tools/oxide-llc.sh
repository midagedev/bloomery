#!/usr/bin/env bash
# The llc whose PTX a device build carries: LLVM 21's (LLC_MAJOR), named once per build. cuda-oxide takes
# CUDA_OXIDE_LLC first, then the Rust toolchain's own llc (the `llvm-tools` component, LLVM 23 on the pinned nightly),
# then llc-23, llc-22 and llc-21 on PATH; LLVM 23's PTX fuses one more multiply-add into the K-quant walks than
# LLVM 21's, and gate_kquant pins the walks to each other bit for bit. tools/box.sh runs this before every command
# that builds device code (`cargo oxide`, `gate.sh --oxide`, `release/build.sh`), on the box and in local mode, and
# exports what it prints as CUDA_OXIDE_LLC, so cuda-oxide's own order is never reached.
#
# The pick: CUDA_OXIDE_LLC when set, else llc-21 on PATH, else llc on PATH. It must report LLVM major LLC_MAJOR and
# carry the NVPTX target, or it is refused by name.
#
#   tools/oxide-llc.sh              the pick's path on stdout, `oxide-llc: <path> (LLVM <version>)` on stderr
#   tools/oxide-llc.sh --self-test  stub llcs in a temp dir: the pick order and each refusal
#
# Exit codes: 0 the pick; 70 no llc, or one of another major or without NVPTX (each named); 1 a self-test case failed;
# 64 an unknown argument.
# Only bash builtins besides the llc itself, so the self-test runs it under a PATH of stubs alone.
set -euo pipefail

LLC_MAJOR=21

refuse() {
  echo "oxide-llc: $1; the build takes LLVM $LLC_MAJOR's llc: set CUDA_OXIDE_LLC to one (docs/BUILD.md, \"Toolchain\")" >&2
  exit 70
}

# pick: the path, its version line, or a refusal.
pick() {
  local llc from out ver
  if [ -n "${CUDA_OXIDE_LLC:-}" ]; then
    llc=$CUDA_OXIDE_LLC from="CUDA_OXIDE_LLC=$CUDA_OXIDE_LLC"
    [ -x "$llc" ] || refuse "$from is not an executable file"
  elif llc=$(command -v llc-21); then
    from="llc-21 on PATH ($llc)"
  elif llc=$(command -v llc); then
    from="llc on PATH ($llc)"
  else
    refuse "CUDA_OXIDE_LLC is unset and PATH has no llc-21 or llc"
  fi
  out=$("$llc" --version 2>&1) || refuse "$from: \`--version\` failed"
  [[ $out =~ LLVM\ version\ ([0-9]+)(\.[^[:space:]]*)? ]] || refuse "$from: \`--version\` names no LLVM version"
  ver=${BASH_REMATCH[1]}${BASH_REMATCH[2]}
  [ "${BASH_REMATCH[1]}" = "$LLC_MAJOR" ] || refuse "$from reports LLVM $ver"
  [[ $out == *nvptx64* ]] || refuse "$from (LLVM $ver) has no NVPTX target"
  echo "oxide-llc: $llc (LLVM $ver)" >&2
  echo "$llc"
}

self_test() {
  local d self fails=0
  d=$(mktemp -d)
  # shellcheck disable=SC2064 # the path is fixed now
  trap "rm -rf '$d'" EXIT
  self=$(cd "$(dirname "$0")" && pwd)/${0##*/}
  stub() { # stub DIR NAME VERSION [TARGETS]
    mkdir -p "$1"
    printf '#!/bin/sh\nprintf "LLVM (http://llvm.org/):\\n  LLVM version %s\\n\\n  Registered Targets:\\n    %s\\n"\n' \
      "$3" "${4-nvptx64 - NVIDIA PTX 64-bit}" > "$1/$2"
    chmod +x "$1/$2"
  }
  stub "$d/v21" llc 21.1.8
  stub "$d/rust" llc 23.1.0-rust-1.93.0-nightly
  stub "$d/v22" llc 22.1.0
  stub "$d/dash21" llc-21 21.1.8
  stub "$d/nonv" llc 21.1.8 "x86-64 - 64-bit X86: EM64T and AMD64"
  mkdir -p "$d/empty"
  # case NAME WANT_RC WANT_TEXT PATH [CUDA_OXIDE_LLC]: the tool's rc and its stdout plus stderr.
  case_() {
    local name=$1 want=$2 text=$3 path=$4 rc out
    if [ $# -ge 5 ]; then
      out=$(env -i PATH="$path" CUDA_OXIDE_LLC="$5" /bin/bash "$self" 2>&1) && rc=0 || rc=$?
    else
      out=$(env -i PATH="$path" /bin/bash "$self" 2>&1) && rc=0 || rc=$?
    fi
    if [ "$rc" = "$want" ] && [[ $out == *"$text"* ]]; then
      return 0
    fi
    echo "oxide-llc self-test FAIL: $name: rc $rc (want $want), want '$text' in: $out" >&2
    fails=$((fails + 1))
  }
  case_ "CUDA_OXIDE_LLC at LLVM 21 is the pick" 0 "oxide-llc: $d/v21/llc (LLVM 21.1.8)" "$d/rust" "$d/v21/llc"
  case_ "CUDA_OXIDE_LLC at the toolchain's LLVM 23 is refused" 70 "reports LLVM 23.1.0-rust-1.93.0-nightly" "$d/v21" "$d/rust/llc"
  case_ "CUDA_OXIDE_LLC at LLVM 22 is refused" 70 "reports LLVM 22.1.0" "$d/v21" "$d/v22/llc"
  case_ "CUDA_OXIDE_LLC naming no file is refused" 70 "is not an executable file" "$d/v21" "$d/none/llc"
  case_ "llc-21 on PATH comes before llc" 0 "oxide-llc: $d/dash21/llc-21 (LLVM 21.1.8)" "$d/rust:$d/dash21"
  case_ "llc on PATH at LLVM 21 is the pick" 0 "oxide-llc: $d/v21/llc (LLVM 21.1.8)" "$d/v21"
  case_ "llc on PATH at LLVM 23 is refused" 70 "llc on PATH ($d/rust/llc) reports LLVM 23" "$d/rust"
  case_ "no llc is refused" 70 "PATH has no llc-21 or llc" "$d/empty"
  case_ "an llc without NVPTX is refused" 70 "has no NVPTX target" "$d/empty" "$d/nonv/llc"
  if [ "$fails" != 0 ]; then
    echo "oxide-llc: self-test $fails failed" >&2
    return 1
  fi
  echo "oxide-llc: self-test ok (9 cases, 0 failed)"
}

case "${1:-}" in
  --self-test) self_test ;;
  "") pick ;;
  *)
    echo "oxide-llc: unknown argument '$1' (none, or --self-test)" >&2
    exit 64
    ;;
esac
