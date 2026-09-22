#!/usr/bin/env bash
# shellcheck shell=bash
# shellcheck disable=SC2034  # every name here is read by the file that sources this one
# The two cards' UUIDs, and nothing else. Sourced, never executed.
#
# timing-card.sh owns the policy — which card is timed, CUDA_VISIBLE_DEVICES, the witness
# lines — and sourcing it moves the caller onto the timing card. A runner that is pinned to
# one card (measure.sh, the stage-0 back-to-back) must not take that policy along with the
# name of a card, so the names live here and both files read them.
#
# A UUID changes when a card is replaced or reseated; one place means the replacement is one
# edit, and no runner can end up naming a card the others no longer know.
GPU_3090=GPU-307fa0f6-daae-24e5-6fd3-cd50620de6b1
GPU_A6000=GPU-8c129fa6-7382-35a5-2464-9ff01d99fcd4
