#!/usr/bin/env bash
# shellcheck shell=bash
# shellcheck disable=SC2034  # every name here is read by the file that sources this one
# The cards a command runs on. Sourced, never executed.
#
# cards_view (below) reads the cards in view on the machine the command runs on, by UUID. It is the one selector outside
# the box: tools/box.sh's local mode, tools/gpu-gate.sh's and tools/ref/timing-card.sh's take the cards in view from it, so
# the cards a command sees, the lock it holds for them and the card it times are one set. CUDA_VISIBLE_DEVICES selects them
# as it does for CUDA. Its query is bounded as the witness's are (timeout, when there is one).
#
# GPU_3090 and GPU_A6000 are the box's two cards, looked up by name when this file is sourced. timing-card.sh owns the
# box's policy (which card is timed, CUDA_VISIBLE_DEVICES, the witness
# lines), and sourcing it moves the caller onto the timing card. A runner that is pinned to
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

# cards_smi <fields>: nvidia-smi's `--query-gpu=<fields>` list (csv, no header) on stdout with its stderr, its status returned, cut at 30 s where
# there is a `timeout` (a Mac has none). The one query of this file, so a card that hangs the driver ends a call here and nowhere else; read it
# in a substitution (`out=$(cards_smi uuid) || rc=$?`), which leaves no child behind that a signal to the caller must reach.
cards_smi() {
  if command -v timeout > /dev/null; then
    timeout --kill-after=5 30 nvidia-smi --query-gpu="$1" --format=csv,noheader 2>&1
  else
    nvidia-smi --query-gpu="$1" --format=csv,noheader 2>&1
  fi
}

# cards_view: the cards in view, by UUID, space-separated in CARDS_VIEW and counted in CARDS_VIEW_N: CUDA_VISIBLE_DEVICES's
# entries in its order, or every card nvidia-smi lists, in its order, when it is unset. An entry is a card's UUID, or a
# prefix of one that names one card (CUDA takes either), or its index in nvidia-smi's list. nvidia-smi lists by PCI bus,
# and CUDA counts an index in that order only under CUDA_DEVICE_ORDER=PCI_BUS_ID (its default is fastest first), so
# without it an index is taken only where the indices alone name every card. 1, with the reason in CARDS_VIEW_WHY and
# CARDS_VIEW empty, when nvidia-smi fails or lists no card, when CUDA_VISIBLE_DEVICES is set and empty or holds a newline,
# when an entry names no card it lists, more than one, or one already in view, and for an index CUDA's order could read
# as another card. A warning nvidia-smi prints beside its list is not a card.
cards_view() {
  local out rc=0 e u hit listed=()
  CARDS_VIEW='' CARDS_VIEW_N=0 CARDS_VIEW_WHY='' CARDS_VIEW_KIND=''
  out=$(cards_smi uuid) || rc=$?
  if [ "$rc" != 0 ]; then
    CARDS_VIEW_WHY="nvidia-smi --query-gpu=uuid failed (rc $rc): ${out%%$'\n'*}" CARDS_VIEW_KIND=smi
    return 1
  fi
  while IFS= read -r e; do
    case $e in GPU-*) listed+=("$e") ;; esac
  done <<< "$out"
  if [ ${#listed[@]} = 0 ]; then
    CARDS_VIEW_WHY='nvidia-smi --query-gpu=uuid lists no card' CARDS_VIEW_KIND=smi
    return 1
  fi
  if [ -z "${CUDA_VISIBLE_DEVICES+x}" ]; then
    CARDS_VIEW=${listed[*]} CARDS_VIEW_N=${#listed[@]}
    return 0
  fi
  local view=() ents=() n index=0 uuid=0
  if [[ $CUDA_VISIBLE_DEVICES == *$'\n'* ]]; then
    CARDS_VIEW_WHY='CUDA_VISIBLE_DEVICES holds a newline: its entries are one line, split by commas' CARDS_VIEW_KIND=other
    return 1
  fi
  IFS=, read -r -a ents <<< "$CUDA_VISIBLE_DEVICES"
  if [ ${#ents[@]} = 0 ]; then
    CARDS_VIEW_WHY='CUDA_VISIBLE_DEVICES is set and empty: no card is in view' CARDS_VIEW_KIND=other
    return 1
  fi
  # read drops the last empty field; keep it so a trailing comma is refused as an empty entry.
  case $CUDA_VISIBLE_DEVICES in *,) ents+=('') ;; esac
  for e in "${ents[@]}"; do
    hit='' n=0
    case $e in
      GPU-?*)
        uuid=1
        for u in "${listed[@]}"; do
          case $u in "$e"*) hit=$u n=$((n + 1)) ;; esac
        done ;;
      [0-9] | [1-9][0-9]) index=1; [ "$e" -ge "${#listed[@]}" ] || hit=${listed[$e]} n=1 ;;
    esac
    if [ "$n" != 1 ]; then
      CARDS_VIEW_WHY="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES: '$e' names $([ "$n" = 0 ] && echo 'none' || echo "$n") of the cards nvidia-smi lists (${listed[*]})"
      CARDS_VIEW_KIND=other
      [ "$n" != 0 ] || CARDS_VIEW_KIND=gone
      return 1
    fi
    if [[ " ${view[*]-} " == *" $hit "* ]]; then
      CARDS_VIEW_WHY="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES: '$e' names $hit, already in view" CARDS_VIEW_KIND=other
      return 1
    fi
    view+=("$hit")
  done
  if [ "$index" = 1 ] && [ "${CUDA_DEVICE_ORDER:-}" != PCI_BUS_ID ] && { [ ${#view[@]} -lt ${#listed[@]} ] || [ "$uuid" = 1 ]; }; then
    CARDS_VIEW_WHY="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES: an index picks among ${#listed[@]} cards in CUDA's order, fastest first unless CUDA_DEVICE_ORDER=PCI_BUS_ID, and nvidia-smi's list is by PCI bus: name the cards by UUID, or set CUDA_DEVICE_ORDER=PCI_BUS_ID"
    CARDS_VIEW_KIND=other
    return 1
  fi
  CARDS_VIEW=${view[*]} CARDS_VIEW_N=${#view[@]}
}

# cards_pin_check <who> <source> [pin [label]]: a card pin that names no card this machine has is refused by name, never an empty
# device list. The pin is CUDA_VISIBLE_DEVICES, or the value given (a runner's own, like BLOOMERY_TIMING_GPU, with its label); an unset or
# empty pin is no pin (an empty one is the deliberate host-only run). 0 when it is none, or when every entry names a listed card, or when
# cards_view refuses it for a reason that is no missing card (an index in CUDA's order, a card named twice: the callers that need the
# order refuse those themselves). Else 64 and one line on stderr naming who refused, the pin, what set it (<source>: the file or the
# variable) and what nvidia-smi lists: the pin names a card that is gone (cards_view's kind `gone`), or nvidia-smi cannot answer (`smi`).
cards_pin_check() {
  local who=$1 src=$2 label=${4:-CUDA_VISIBLE_DEVICES} out rc=0 listed
  local CUDA_VISIBLE_DEVICES=${3-${CUDA_VISIBLE_DEVICES-}}
  [ -n "$CUDA_VISIBLE_DEVICES" ] || return 0
  cards_view && return 0
  case $CARDS_VIEW_KIND in smi | gone) ;; *) return 0 ;; esac
  out=$(cards_smi name,uuid) || rc=$?
  if [ "$rc" != 0 ] || [ -z "$out" ]; then
    listed="nothing (nvidia-smi rc $rc${out:+: ${out%%$'\n'*}})"
  else
    listed=$(printf '%s' "$out" | tr '\n' ';')
  fi
  echo "$who: $label=$CUDA_VISIBLE_DEVICES, set by $src, names no card this machine has ($CARDS_VIEW_WHY); nvidia-smi lists: $listed — fix the pin where it is set: a card swap leaves the old card's UUID behind" >&2
  return 64
}

# cards_by_name <token>: the one card whose nvidia-smi name holds <token> as a case-insensitive substring (`5090` names `NVIDIA GeForce
# RTX 5090`): its UUID in CARD_UUID and its name in CARD_NAME, and every card nvidia-smi lists counted in CARDS_LISTED_N. 1, with the
# reason in CARD_WHY and CARD_UUID and CARD_NAME empty, when the token is empty, nvidia-smi fails or lists no card, or the token holds
# in the name of none of the cards or of more than one (the names nvidia-smi lists are in the reason: a name that fits two cards picks
# neither). It reads every card nvidia-smi lists, not the cards in view, so a runner that times a card by name sees the card it names
# whatever CUDA_VISIBLE_DEVICES says; that the card is in view is the runner's own check (timing-card.sh's local mode). The one place a
# cells file's `cards` column, or a caller that names a card, becomes a UUID: no card's name is in the tool that asks.
cards_by_name() {
  local tok=$1 out rc=0 line name uuid lc hits=0 listed=''
  CARD_UUID='' CARD_NAME='' CARD_WHY='' CARDS_LISTED_N=0
  if [ -z "$tok" ]; then
    CARD_WHY='no card name was given'
    return 1
  fi
  out=$(cards_smi name,uuid) || rc=$?
  if [ "$rc" != 0 ]; then
    CARD_WHY="nvidia-smi --query-gpu=name,uuid failed (rc $rc): ${out%%$'\n'*}"
    return 1
  fi
  lc=$(printf '%s' "$tok" | tr '[:upper:]' '[:lower:]')
  while IFS= read -r line; do
    # a warning nvidia-smi prints beside its list is not a card: a card's line ends with its UUID
    case $line in *', GPU-'?*) ;; *) continue ;; esac
    name=${line%%, GPU-*}
    uuid=GPU-${line#*, GPU-}
    CARDS_LISTED_N=$((CARDS_LISTED_N + 1))
    listed+="${listed:+; }$name, $uuid"
    case $(printf '%s' "$name" | tr '[:upper:]' '[:lower:]') in
      *"$lc"*) hits=$((hits + 1)) CARD_UUID=$uuid CARD_NAME=$name ;;
    esac
  done <<< "$out"
  if [ "$CARDS_LISTED_N" = 0 ]; then
    CARD_WHY='nvidia-smi --query-gpu=name,uuid lists no card'
    return 1
  fi
  if [ "$hits" != 1 ]; then
    CARD_UUID='' CARD_NAME=''
    CARD_WHY="'$tok' is in the name of $([ "$hits" = 0 ] && echo none || echo "$hits") of the cards nvidia-smi lists ($listed)"
    return 1
  fi
  return 0
}

GPU_3090='' GPU_A6000='' CARDS_ERROR=''
[ "${BLOOMERY_BOX_CARD:-}" != local ] || return 0
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
