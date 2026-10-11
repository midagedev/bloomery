#!/usr/bin/env bash
# shellcheck shell=bash
# The CUDA arch the device builds target: BLOOMERY_ARCH, sm_86 when unset. The justfile's `arch` variable reads the same
# name with the same default, so a recipe's `--arch {{arch}}` and a tool that runs on the box agree. Sourced, never
# executed; one parser, so every tool takes and refuses the same values.
#
# The value is `sm_` and the capability's digits, with the `a` or `f` suffix cargo-oxide accepts: sm_86, sm_121,
# sm_121a. Anything else is refused by name, set-and-empty included: a recipe would hand `--arch ` to cargo-oxide.
# Which archs a tool supports is the tool's own refusal, after this one (the shape pins and the occupancy column
# are sm_86's; the release archive is sm_86's).
#
#   bloomery_arch <tool>   ARCH=<the value>, or the refusal on stderr and 64
bloomery_arch() {
  local v=${BLOOMERY_ARCH-sm_86}
  if ! [[ $v =~ ^sm_[1-9][0-9]{1,2}[af]?$ ]]; then
    echo "$1: BLOOMERY_ARCH is a CUDA arch, sm_ and the capability's digits with an optional a or f (unset: sm_86), got '$v'" >&2
    return 64
  fi
  # shellcheck disable=SC2034 # the sourcing tool reads it
  ARCH=$v
}
