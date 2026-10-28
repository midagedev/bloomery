#!/usr/bin/env bash
# shellcheck shell=bash
# The CUDA build target: BLOOMERY_ARCH, otherwise native FP4 on GB10 and sm_86 elsewhere.
# The justfile exports this choice so builds and the tools that inspect them agree.
#
# The value is `sm_` and the capability's digits, with the `a` or `f` suffix cargo-oxide accepts: sm_86, sm_121,
# sm_121a. Anything else is refused by name, set-and-empty included: a recipe would hand `--arch ` to cargo-oxide.
# Which archs a tool supports is the tool's own refusal, after this one (the shape pins hold sm_86, sm_121 and sm_121a
# tables, the occupancy column sm_86's and sm_120/sm_121's limits; narrow-scan is sm_86's, and the release archive is
# sm_86's on x86-64 (sm_120a's when BLOOMERY_ARCH names it) and sm_121a's on a GB10).
#
#   bloomery_arch <tool>   ARCH and exported BLOOMERY_ARCH, or the refusal on stderr and 64
#   bash tools/arch.sh     print that value for the justfile
#   ARCH_SHAPE             the value's shape, for a tool that reads an arch elsewhere (a ptx-shapes table's `# arch:`)
ARCH_SHAPE='^sm_[1-9][0-9]{1,2}[af]?$'
bloomery_arch() {
  local v=${BLOOMERY_ARCH-sm_86}
  if [ "${BLOOMERY_ARCH+x}" != x ] && command -v nvidia-smi > /dev/null 2>&1; then
    local capability
    if ! capability=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader); then
      echo "$1: cannot read the CUDA compute capability; set BLOOMERY_ARCH explicitly for a cross build" >&2
      return 64
    fi
    v=sm_121a
    while IFS= read -r capability; do
      if [ "$capability" != 12.1 ]; then v=sm_86; fi
    done <<< "$capability"
  fi
  if ! [[ $v =~ $ARCH_SHAPE ]]; then
    echo "$1: BLOOMERY_ARCH is a CUDA arch, sm_ and the capability's digits with an optional a or f (unset: auto), got '$v'" >&2
    return 64
  fi
  # shellcheck disable=SC2034 # the sourcing tool reads it
  ARCH=$v
  export BLOOMERY_ARCH=$v
}
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  bloomery_arch arch.sh || exit $?
  printf '%s\n' "$ARCH"
fi
