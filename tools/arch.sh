#!/usr/bin/env bash
# shellcheck shell=bash
# The CUDA build target: BLOOMERY_ARCH, otherwise the native block-scaled target of the visible cards' compute capability
# when every card has the same one (sm_120a on 12.0, the RTX 50 series; sm_121a on 12.1, GB10), and sm_86 elsewhere.
# It is read on the machine that builds: tools/box.sh runs it there, after the environment file, for a command that names
# $BLOOMERY_ARCH (every recipe's `--arch {{arch}}`) or spells an arch, and exports the value, so the build and the tools that
# inspect it agree. The justfile does not run it: `just` on a machine with no card would answer sm_86 for the box's cards.
#
# The value is `sm_` and the capability's digits, with the `a` or `f` suffix cargo-oxide accepts: sm_86, sm_121,
# sm_121a. Anything else is refused by name, set-and-empty included: a recipe would hand `--arch ` to cargo-oxide.
# Which archs a tool supports is the tool's own refusal, after this one (the shape pins hold sm_86 tables on cc 8.6 and
# 12.1, sm_120a on 12.0 and sm_121 and sm_121a on 12.1, the occupancy column sm_86's and sm_120/sm_121's limits; the
# release archive is sm_86's or sm_120a's on x86-64 and sm_121a's on a GB10).
#
#   bloomery_arch <tool>   ARCH and exported BLOOMERY_ARCH, or the refusal on stderr and 64
#   bash tools/arch.sh     print that value for the justfile
#   ARCH_SHAPE             the value's shape, for a tool that reads an arch elsewhere (a ptx-shapes table's `# arch:`)
ARCH_SHAPE='^sm_[1-9][0-9]{1,2}[af]?$'
bloomery_arch() {
  local v=${BLOOMERY_ARCH-sm_86}
  if [ "${BLOOMERY_ARCH+x}" != x ] && command -v nvidia-smi > /dev/null 2>&1; then
    local capabilities capability
    if ! capabilities=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader); then
      echo "$1: cannot read the CUDA compute capability; set BLOOMERY_ARCH explicitly for a cross build" >&2
      return 64
    fi
    v=
    while IFS= read -r capability; do
      case $capability in
        12.0) capability=sm_120a ;;
        12.1) capability=sm_121a ;;
        *) capability=sm_86 ;;
      esac
      # Cards of different capabilities share sm_86, the target every one of them loads.
      [ "${v:-$capability}" = "$capability" ] || capability=sm_86
      v=$capability
    done <<< "$capabilities"
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
