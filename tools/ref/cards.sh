#!/usr/bin/env bash
# shellcheck shell=bash
# shellcheck disable=SC2034  # every name here is read by the file that sources this one
# The two cards' UUIDs, and nothing else. Sourced, never executed.
#
# timing-card.sh owns the policy — which card is timed, CUDA_VISIBLE_DEVICES, the witness
# lines — and sourcing it moves the caller onto the timing card. A runner that is pinned to
# one card must not take that policy along with the name of a card, so the names live here and
# both files read them. The UUIDs are looked up from the names when this file is sourced — the
# repository is public, and a card's UUID is the machine's, not the code's — so a replaced or
# reseated card is one lookup, not an edit, and no runner can name a card the others no longer
# know.
#
# No silent failure: a name that matches no card, or two, leaves its variable empty with the
# reason in CARDS_ERROR, and the sourcing shell goes on — a runner that needs only the other
# card still runs (a card has fallen off the bus before). A consumer that needs a card refuses
# by name when it finds the variable empty.
GPU_3090='' GPU_A6000='' CARDS_ERROR=''
if ! command -v nvidia-smi > /dev/null 2>&1; then
  CARDS_ERROR='nvidia-smi is not on PATH'
else
  __cards_rc=0
  __cards_out=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader 2>&1) || __cards_rc=$?
  if [ "$__cards_rc" != 0 ]; then
    CARDS_ERROR="nvidia-smi --query-gpu=name,uuid failed (rc $__cards_rc): ${__cards_out%%$'\n'*}"
  elif [ -z "$__cards_out" ]; then
    CARDS_ERROR='nvidia-smi --query-gpu=name,uuid printed no card'
  else
    __cards_3090=0 __cards_a6000=0
    while IFS=, read -r __cards_name __cards_uuid; do
      __cards_uuid=${__cards_uuid# }
      case $__cards_name in
        'NVIDIA GeForce RTX 3090') __cards_3090=$((__cards_3090 + 1)); GPU_3090=$__cards_uuid ;;
        'NVIDIA RTX A6000') __cards_a6000=$((__cards_a6000 + 1)); GPU_A6000=$__cards_uuid ;;
      esac
    done <<< "$__cards_out"
    if [ "$__cards_3090" != 1 ] || [ -z "$GPU_3090" ]; then
      GPU_3090=''
      CARDS_ERROR="${CARDS_ERROR:+$CARDS_ERROR; }'NVIDIA GeForce RTX 3090' matched $__cards_3090 cards"
    fi
    if [ "$__cards_a6000" != 1 ] || [ -z "$GPU_A6000" ]; then
      GPU_A6000=''
      CARDS_ERROR="${CARDS_ERROR:+$CARDS_ERROR; }'NVIDIA RTX A6000' matched $__cards_a6000 cards"
    fi
  fi
  unset __cards_rc __cards_out __cards_name __cards_uuid __cards_3090 __cards_a6000
fi
