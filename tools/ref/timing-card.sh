#!/usr/bin/env bash
# Shared timing-card setup for the GPU runners: depth-gpu.sh, ncu-gpu.sh, nsys-gpu.sh and
# time-gate.sh. This file is sourced, never executed, so it only defines variables and
# functions — nothing here may exit or fail at top level (time-gate.sh runs under `set -e`).
#
#   source "${BASH_SOURCE[0]%/*}/timing-card.sh"
#
# What it owns, so that four runners cannot drift apart:
#   which card is the timing card; CUDA_VISIBLE_DEVICES; the timing card's witness lines (the
#   `card` field of the witness block, which lease.sh owns); what to do when the other card is
#   busy; and the refusal that keeps a runner from timing a binary older than its sources.
#
# Timed numbers are taken on the A6000 and this overrides the 3090 pin in the box env file.
# The 3090 is the gate-and-build card. Numbers from the two cards never belong in one table,
# which is why the first witness line names the card and its power limit.
#
# The two-card mode (AGENTS.md, user 2026-09-28): a model that does not fit one card may carry a
# second table, "A6000+3090", its own and never the A6000's. BLOOMERY_TIMING_CARDS=a6000+3090 turns it
# on for a runner that opts in (TIMING_CARDS_RUNNER=1 before it sources this file: depth-ds41.sh,
# depth-qwen3moe.sh and ik-draft.sh); for any other runner, or any other value, TIMING_GPU and CUDA_VISIBLE_DEVICES are
# left empty and the reason printed, so lease_take refuses the run (64) instead of timing the A6000
# alone. In the mode both cards are visible, the A6000 first (device 0, TIMING_GPU, the lease's
# record) and the 3090 second (TIMING_GPU2); there is no other card (OTHER_GPU empty), so a compute
# process on either is a co-tenant on a timed card: guard_other waits it out or ends the run (75).
# Before the lease timing_cards_precheck refuses a card that does not answer or answers under another
# name (69), a 3090 off its 250 W cap (78), a lease.sh that would record one card (64) and a kernel
# journal it cannot read (69). After each arm timing_cards_arm fails the arm's row on an NVRM Xid
# since the last arm, a card that stopped answering or left its cap, or an engine log that shows
# other than the two cards. The witness's `card` field prints both cards and the Xid count since the
# lease was taken.
# shellcheck source=tools/ref/cards.sh
source "${BASH_SOURCE[0]%/*}/cards.sh"
# now(), the lease and the witness block. guard_other below prints a witness block on its abort path.
# shellcheck source=tools/ref/lease.sh
source "${BASH_SOURCE[0]%/*}/lease.sh"
TIMING_GPU=${BLOOMERY_TIMING_GPU:-$GPU_A6000}
if [ "$TIMING_GPU" = "$GPU_3090" ]; then OTHER_GPU=$GPU_A6000; else OTHER_GPU=$GPU_3090; fi
export CUDA_VISIBLE_DEVICES=$TIMING_GPU
# A DSpark run (BLOOMERY_DRAFT=dspark) puts its draft on the other card, found by name
# (Gpu::for_card), so that card must be visible too; the timing card stays device 0. The draft file
# is the profile's DSPARK_MODEL unless the caller names one. dspark_env prints the same two
# assignments for a runner that sets the lever per arm (depth-ds41.sh); in the two-card mode the cards
# are the mode's two (the draft sits on the 3090 beside the expert tier).
dspark_env() {
  echo "CUDA_VISIBLE_DEVICES=$TIMING_GPU${OTHER_GPU:+,$OTHER_GPU}${TIMING_GPU2:+,$TIMING_GPU2}"
  echo "BLOOMERY_DSPARK_MODEL=${BLOOMERY_DSPARK_MODEL:-${DSPARK_MODEL:-}}"
}
if [ "${BLOOMERY_DRAFT:-}" = dspark ]; then
  export CUDA_VISIBLE_DEVICES=$TIMING_GPU,$OTHER_GPU
  export BLOOMERY_DSPARK_MODEL=${BLOOMERY_DSPARK_MODEL:-${DSPARK_MODEL:-}}
fi

# The two-card mode's state, set in both modes so a runner under `set -u` can read every name:
# TIMING_CARDS (a6000+3090, or empty: one card), TIMING_CARDS_NAME (the rows' card field),
# TIMING_CARDS_WHY (why the value was refused), TIMING_GPU2 (the 3090 in the mode), the cards' PCI
# addresses as the kernel's Xid lines print them (XID_BUS_A, XID_BUS_B, set by the precheck), the
# lease's start instant the Xid count runs from (XID_T0), the count the last arm ended at (XID_BASE), and
# the last check's reason and devices (TWOCARD_WHY, TWOCARD_DEVS: timing_cards_arm).
TIMING_CARDS='' TIMING_CARDS_NAME='' TIMING_CARDS_WHY='' TIMING_GPU2=''
XID_BUS_A='' XID_BUS_B='' XID_T0='' XID_BASE=0 TWOCARD_WHY='' TWOCARD_DEVS=''
case ${BLOOMERY_TIMING_CARDS:-} in
  '') ;;
  a6000+3090)
    if [ "${TIMING_CARDS_RUNNER:-}" != 1 ]; then
      TIMING_CARDS_WHY="BLOOMERY_TIMING_CARDS=a6000+3090, and ${0##*/} has no two-card mode (a runner opts in with TIMING_CARDS_RUNNER=1): it would time the A6000 alone"
    elif [ -z "$GPU_A6000" ] || [ -z "$GPU_3090" ]; then
      __cards_missing=''
      [ -n "$GPU_A6000" ] || __cards_missing='the A6000'
      [ -n "$GPU_3090" ] || __cards_missing="${__cards_missing:+$__cards_missing and }the 3090"
      TIMING_CARDS_WHY="BLOOMERY_TIMING_CARDS=a6000+3090 times both cards, and tools/ref/cards.sh resolved no UUID for $__cards_missing (${CARDS_ERROR:-no reason given})"
      unset __cards_missing
    elif [ -n "${BLOOMERY_TIMING_GPU:-}" ] && [ "$BLOOMERY_TIMING_GPU" != "$GPU_A6000" ]; then
      TIMING_CARDS_WHY="BLOOMERY_TIMING_CARDS=a6000+3090 makes the A6000 device 0 and the timing card; BLOOMERY_TIMING_GPU=$BLOOMERY_TIMING_GPU names another"
    else
      TIMING_CARDS=a6000+3090 TIMING_CARDS_NAME=A6000+3090
    fi
    ;;
  *) TIMING_CARDS_WHY="BLOOMERY_TIMING_CARDS is a6000+3090 (the two-card mode) or unset (one card), got '$BLOOMERY_TIMING_CARDS'" ;;
esac
# One card, and a card tools/ref/cards.sh could not name (its variable empty, the reason in
# CARDS_ERROR): the timing card is the A6000 by default, and which of the two it is decides the
# other card, the lease's record and the model profiles' reference sizing — undecidable when the
# timing card is neither card cards.sh named and a UUID is missing. A run on the card that did
# resolve still runs (the other card's witness lines name the miss; a card off the bus must not
# take the healthy one down with it). The refusal rides TIMING_CARDS_WHY, the branch below
# prints it and lease_take ends the run on it (64): this file is sourced under set -e and may
# not exit itself.
if [ -z "$TIMING_CARDS_WHY" ]; then
  if [ -z "$TIMING_GPU" ]; then
    TIMING_CARDS_WHY="the default timing card, the A6000, has no UUID (tools/ref/cards.sh: ${CARDS_ERROR:-no reason given})"
  elif [ "$TIMING_GPU" != "$GPU_A6000" ] && [ "$TIMING_GPU" != "$GPU_3090" ] && { [ -z "$GPU_3090" ] || [ -z "$GPU_A6000" ]; }; then
    TIMING_CARDS_WHY="the timing card ($TIMING_GPU) is neither card tools/ref/cards.sh named, and a card is missing ($CARDS_ERROR): whether it is the 3090 cannot be told"
  fi
fi
if [ -n "$TIMING_CARDS" ]; then
  # shellcheck disable=SC2034 # TIMING_GPU2 is read by lease.sh's lease_take (the lease's record)
  TIMING_GPU=$GPU_A6000 TIMING_GPU2=$GPU_3090 OTHER_GPU=
  export CUDA_VISIBLE_DEVICES=$GPU_A6000,$GPU_3090
elif [ -n "$TIMING_CARDS_WHY" ]; then
  # A runner that opts in says it itself (timing_cards_mode); for any other this line is the reason.
  [ "${TIMING_CARDS_RUNNER:-}" = 1 ] || echo "[timing-cards] refused: $TIMING_CARDS_WHY" >&2
  TIMING_GPU='' OTHER_GPU=''
  export CUDA_VISIBLE_DEVICES=
fi
# timing_cards_mode: 0 in one-card mode and in the two-card mode; 64 and the reason when
# BLOOMERY_TIMING_CARDS was refused above. A runner that opts in calls it right after sourcing this file,
# dry run or not.
timing_cards_mode() {
  [ -n "$TIMING_CARDS_WHY" ] || return 0
  echo "${0##*/}: $TIMING_CARDS_WHY" >&2
  return 64
}

# The timing card's witness lines: the `card` field of lease.sh's witness block. loadavg is not the
# quiet-machine signal (see docs/quiet-machine.md in rig-log): IO pressure and the actual process
# list are.
# The clocks line is read at the block's instant: at `pre` the card is idle and its SM clock is the idle
# clock, not the run's. The counters (clocks_event_reasons_counters.*, cumulative µs per reason) are
# the run's side: post − pre bounds the time whatever ran on the card in the window spent power-capped
# (sw_power_cap) or thermally slowed, with no sampler inside it. When the driver updates them is not
# characterized, so read the difference over a whole arm, not a short one. event_reasons is the active
# reason bitmask at the instant (nvml.h: 0x1 idle, 0x4 sw power cap, 0x20 sw thermal, 0x40 hw thermal).
# cpu-freq is lease.sh's field of that name.
witness_card() {
  if [ -n "$TIMING_CARDS" ]; then
    witness_cards
    return 0
  fi
  echo "    timing-card: $(nvidia-smi --query-gpu=name,power.limit,clocks.max.sm --format=csv,noheader -i "$TIMING_GPU")"
  echo "    timing-card clocks: $(nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,clocks_event_reasons.active,clocks_event_reasons_counters.sw_power_cap,clocks_event_reasons_counters.sw_thermal_slowdown,clocks_event_reasons_counters.hw_thermal_slowdown,temperature.gpu,power.draw --format=csv,noheader,nounits -i "$TIMING_GPU" | awk -F', ' '{ printf "sm=%s max=%s MHz event_reasons=%s capped_us sw_power=%s sw_thermal=%s hw_thermal=%s temp=%s C power=%s W", $1, $2, $3, $4, $5, $6, $7, $8 }')"
  echo "    cpu-freq: $(cpu_freq_summary)"
  [ -z "${BIN_SHA:-}" ] || echo "    binary: ${BIN_PATH:-?} sha256=$BIN_SHA mtime=${BIN_MTIME:-?}"
  if [ -n "$GPU_3090" ]; then
    echo "    3090-apps: [$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i "$GPU_3090" | tr '\n' ';')]"
  else
    echo "    3090-apps: unresolved (no 3090 UUID${CARDS_ERROR:+: $CARDS_ERROR})"
  fi
  if [ -n "$GPU_A6000" ]; then
    echo "    a6000-apps: [$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i "$GPU_A6000" | tr '\n' ';')]"
  else
    echo "    a6000-apps: unresolved (no A6000 UUID${CARDS_ERROR:+: $CARDS_ERROR})"
  fi
  echo "    gpu: $(nvidia-smi --query-gpu=index,utilization.gpu,power.draw,clocks.sm --format=csv,noheader | tr '\n' ';')"
  echo "    load=$(cut -d' ' -f1-3 /proc/loadavg) io=$(grep '^some' /proc/pressure/io | cut -d' ' -f2) llm.service=$(systemctl is-active llm.service || true)"
}

# Both cards are ours. A compute process on the card we are not timing is most likely another
# round's gate or build: it is recorded, not obeyed — whether the timing card's numbers move
# with load on the other card is a question this witness column will answer later, and it has
# never been measured. BLOOMERY_OTHER_STRICT=1 aborts instead.
# The default witness fields. Every runner sets its own WITNESS right after sourcing this file and
# that list wins; this one exists so guard_other below does not depend on the caller having done
# so — an abort path that prints no witness loses the record where it matters most.
WITNESS=(head-open indent card model)

# guard_other's verdict for the caller's row: ' [other-busy]' after a call that found a compute
# process on the other card, empty after one that did not. The depth runners append it to the
# arm's ROW line as they append guard_cpu's CPU_BUSY_TAG, so a closing table can carry it.
# shellcheck disable=SC2034 # read by the runners that source this file
OTHER_BUSY_TAG=
guard_other() {
  local apps
  if [ -n "$TIMING_CARDS" ]; then
    guard_cards
    return 0
  fi
  if [ -z "$OTHER_GPU" ]; then
    # A card cards.sh could not name has no compute process to read; the witness lines name the
    # miss, and the run goes on — the other card must not take the timed one down with it.
    echo "[other-card] $(now) the other card has no UUID (tools/ref/cards.sh${CARDS_ERROR:+: $CARDS_ERROR}): no compute apps read on it" >&2
    return 0
  fi
  apps=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i "$OTHER_GPU")
  # shellcheck disable=SC2034 # read by the runners that source this file
  OTHER_BUSY_TAG=${apps:+ [other-busy]}
  [ -n "$apps" ] || return 0
  echo "[other-busy] $(now) compute apps on the other card ($OTHER_GPU): [$(echo "$apps" | tr '\n' ';')]" >&2
  if [ "${BLOOMERY_OTHER_STRICT:-}" = 1 ]; then
    witness abort-other >&2
    exit 75
  fi
}

# ---- The two-card mode ("A6000+3090"). Nothing below runs in one-card mode.

# Each card's line in the two-card witness and checks: `<name>, <power.limit> W, <enforced.power.limit>
# W, <bus>` from one nvidia-smi query, bounded as lease.sh's witness queries are (__witness_smi: a card
# off the bus can hang it). __card_query <uuid>: into CARD_Q (the csv line, nounits) and CARD_RC.
__card_query() {
  __witness_smi --query-gpu=name,power.limit,enforced.power.limit,pci.bus_id --format=csv,noheader,nounits -i "$1"
  CARD_Q=$__witness_out CARD_RC=$__witness_rc
}
# __xid_bus <nvidia-smi pci.bus_id>: the address as the kernel's Xid line prints it: 00000000:41:00.0 is
# `PCI:0000:41:00` (the domain's low four digits, bus and device, no function), lower case.
__xid_bus() { sed -E 's/^[0-9A-Fa-f]{4}([0-9A-Fa-f]{4}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2})\.[0-9A-Fa-f]$/\1/' <<< "$1" | tr 'A-F' 'a-f'; }
# __card_ok <label> <uuid> <want in name> <cap W or empty>: CARD_Q for the card, and 1 with TWOCARD_WHY
# when it does not answer, answers under another name, or (a cap given) its power.limit or
# enforced.power.limit is not the cap; the verdict line into CARD_LINE. The cap is compared as a number.
__card_ok() {
  local label=$1 uuid=$2 want=$3 cap=$4 name lim enf bus
  __card_query "$uuid"
  if [ "$CARD_RC" != 0 ] || [ -z "$CARD_Q" ]; then
    TWOCARD_WHY="the $label ($uuid) does not answer nvidia-smi (rc $CARD_RC): a two-card run would load on the other card alone"
    CARD_LINE="$label: unavailable (rc $CARD_RC)"
    return 69
  fi
  IFS=, read -r name lim enf bus <<< "$CARD_Q"
  name=${name# } lim=${lim# } enf=${enf# } bus=${bus# }
  CARD_LINE="$label: $name, power.limit $lim W, enforced $enf W, bus $bus"
  case $name in
    *"$want"*) ;;
    *)
      TWOCARD_WHY="the $label's UUID ($uuid, tools/ref/cards.sh) answers as '$name': not the card the two-card table names"
      return 69
      ;;
  esac
  [ -n "$cap" ] || return 0
  if ! awk -v l="$lim" -v e="$enf" -v c="$cap" 'BEGIN { exit !(l + 0 == c && e + 0 == c) }'; then
    TWOCARD_WHY="the $label's power limit reads $lim W (enforced $enf W), not its $cap W cap (gpu-power-limit.service): the two-card rule keeps it capped"
    return 78
  fi
  CARD_LINE="$CARD_LINE (the $cap W cap: ok)"
}

# timing_cards_arms <our binary> <arm> <kind> <engine> ...: in the two-card mode (0 at once without it),
# before a dry run's lines or the lease: 64 and the reason when the profile has no two-card line
# (TWO_CARD_PLACEMENT: the model fits one card, and the A6000+3090 table is for one that does not) or an
# arm has none. Each arm is three words: the arm as given, its kind (ref for a reference engine, srv for a
# llama-server one; ours, corpus or bin run <our binary>) and its engine. Only mainline llama.cpp's arms
# (lcpp..., the fit arms too, and its server arms lcppsrv...) have a two-card line: the profile's -ts split,
# or the fit over both cards. A server arm is timed inside the same checks as a bench arm: its log passes
# timing_cards_arm (lcpp-warm.sh srv_arm) after the server stopped. A runner whose
# binary has a two-card placement names it in TIMING_CARDS_PLACE (depth-ds41.sh: bp, generate_ds41's plan
# (b′)) and the placement its arms load by in TIMING_CARDS_PLACE_RAN: our arms pass when the two agree and
# are refused with a hint naming it when they do not. Without TIMING_CARDS_PLACE our binaries load one
# card (--place a|gate), and plan (b), `--place b`, is their expected two-card interface.
timing_cards_arms() {
  local bin=$1 a kind eng
  [ -n "$TIMING_CARDS" ] || return 0
  shift
  if [ -z "${TWO_CARD_PLACEMENT:-}" ]; then
    echo "${0##*/}: BLOOMERY_TIMING_CARDS=a6000+3090 and the profile ${MODEL_NAME:-?} has no two-card line (TWO_CARD_PLACEMENT): a model that fits one card has no A6000+3090 table (AGENTS.md)" >&2
    return 64
  fi
  while [ $# -ge 3 ]; do
    a=$1 kind=$2 eng=$3
    shift 3
    case $kind:$eng in
      ref:lcpp* | srv:lcpp*) ;;
      ref:* | srv:*)
        echo "${0##*/}: arm '$a': the two-card table's reference is mainline llama.cpp (the profile's two-card line: $TWO_CARD_PLACEMENT); $eng has no two-card arm" >&2
        return 64
        ;;
      *)
        if [ -n "${TIMING_CARDS_PLACE:-}" ]; then
          [ "${TIMING_CARDS_PLACE_RAN:-}" != "$TIMING_CARDS_PLACE" ] || continue
          echo "${0##*/}: arm '$a': ${bin##*/} under --place ${TIMING_CARDS_PLACE_RAN:-?} loads one card; its two-card placement is --place $TIMING_CARDS_PLACE (BLOOMERY_GEN_PLACE=$TIMING_CARDS_PLACE)" >&2
          return 64
        fi
        echo "${0##*/}: arm '$a': ${bin##*/} loads one card (--place a|gate); its two-card placement, plan (b) (workstation::plan_b), is expected as --place b and does not exist yet, so the A6000+3090 table has no row of ours" >&2
        return 64
        ;;
    esac
  done
}

# timing_cards_precheck: before the lease, in the two-card mode (0 at once without it). The static facts
# only, each printed as a `[timing-cards]` line: both cards answer nvidia-smi by UUID under their names
# (69 otherwise), the 3090's power.limit and enforced.power.limit read 250 W (78), lease.sh writes a
# two-card record (64: an unpatched lease.sh would record the A6000 alone and let a forced 3090 gate
# start beside the run), and the kernel journal answers (69: no Xid reader). A compute process on a card
# is not checked here: guard_other waits it out after the lease. Sets XID_BUS_A and XID_BUS_B; the
# refusal's reason is TWOCARD_WHY, and a dry run prints it and goes on. The optional argument prefixes
# every line (a dry run's `[dry] `).
timing_cards_precheck() {
  local rc=0 out pre=${1:-}
  [ -n "$TIMING_CARDS" ] || return 0
  TWOCARD_WHY=''
  __card_ok A6000 "$GPU_A6000" A6000 '' || rc=$?
  echo "${pre}[timing-cards] $CARD_LINE"
  [ "$rc" = 0 ] || return "$rc"
  XID_BUS_A=$(__xid_bus "${CARD_Q##*, }")
  __card_ok 3090 "$GPU_3090" 3090 250 || rc=$?
  echo "${pre}[timing-cards] $CARD_LINE"
  [ "$rc" = 0 ] || return "$rc"
  XID_BUS_B=$(__xid_bus "${CARD_Q##*, }")
  if [ "${LEASE_CARDS_RECORD:-}" != 1 ]; then
    TWOCARD_WHY="tools/ref/lease.sh records one timing card (no LEASE_CARDS_RECORD): tools/gpu-gate.sh would read the lease as the A6000's and start a forced 3090 gate beside this run; the lease.sh two-card record lands first"
    return 64
  fi
  if ! out=$(timeout --kill-after=5 30 journalctl -k -n 1 -o short-unix --no-pager -q 2>&1); then
    TWOCARD_WHY="journalctl -k does not answer (${out##*$'\n'}): no Xid reader for the witness"
    return 69
  fi
  echo "${pre}[timing-cards] Xid reader: journalctl -k, NVRM Xid lines on PCI:$XID_BUS_A (A6000) and PCI:$XID_BUS_B (3090)"
}

# timing_cards_start: right after lease_take, the lease's start instant, from which the witness counts
# the kernel's Xid lines (XID_T0, whole seconds: a line in the second the lease was taken counts).
timing_cards_start() {
  [ -n "$TIMING_CARDS" ] || return 0
  XID_T0=$(date +%s) XID_BASE=0
  echo "[timing-cards] Xid count from @$XID_T0 ($(date -u -d "@$XID_T0" +%Y-%m-%dT%H:%M:%SZ 2> /dev/null || echo ?)), the lease's start"
}

# xid_read: the kernel journal's `NVRM: Xid` lines since XID_T0 into XID_N (all), XID_NA (the A6000's
# bus), XID_NB (the 3090's) and XID_LAST (the last line); 1 with XID_ERR when the journal does not answer
# or the lease has no start instant yet.
xid_read() {
  local out rc=0 lines
  XID_N=0 XID_NA=0 XID_NB=0 XID_LAST='' XID_ERR=''
  if [ -z "$XID_T0" ]; then
    XID_ERR="no lease start instant (timing_cards_start not run)"
    return 1
  fi
  out=$(timeout --kill-after=5 30 journalctl -k --since "@$XID_T0" -o short-unix --no-pager -q 2>&1) || rc=$?
  if [ "$rc" != 0 ]; then
    XID_ERR="journalctl -k exited $rc: ${out##*$'\n'}"
    return 1
  fi
  lines=$(grep -F 'NVRM: Xid' <<< "$out" || true)
  [ -n "$lines" ] || return 0
  XID_N=$(grep -c . <<< "$lines")
  XID_NA=$(grep -cF "(PCI:$XID_BUS_A)" <<< "$lines" || true)
  XID_NB=$(grep -cF "(PCI:$XID_BUS_B)" <<< "$lines" || true)
  XID_LAST=$(tail -n 1 <<< "$lines")
}

# The two-card witness (witness_card's `card` field in the mode): the mode, each card's name, power limit
# and clocks, the 3090's cap, the Xid count since the lease was taken, then the lines one card prints.
witness_cards() {
  local q lab uuid
  echo "    timing-cards: $TIMING_CARDS_NAME, CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (device 0 the A6000, device 1 the 3090)"
  for lab in A6000 3090; do
    if [ "$lab" = A6000 ]; then uuid=$GPU_A6000; else uuid=$GPU_3090; fi
    __witness_smi --query-gpu=name,power.limit,enforced.power.limit,clocks.max.sm --format=csv,noheader -i "$uuid"
    if [ "$__witness_rc" = 0 ]; then echo "    card $lab: $__witness_out"; else echo "    card $lab: unavailable (rc $__witness_rc)"; fi
    __witness_smi --query-gpu=clocks.sm,clocks.max.sm,clocks_event_reasons.active,clocks_event_reasons_counters.sw_power_cap,clocks_event_reasons_counters.sw_thermal_slowdown,clocks_event_reasons_counters.hw_thermal_slowdown,temperature.gpu,power.draw --format=csv,noheader,nounits -i "$uuid"
    if [ "$__witness_rc" = 0 ]; then
      q=$(awk -F', ' '{ printf "sm=%s max=%s MHz event_reasons=%s capped_us sw_power=%s sw_thermal=%s hw_thermal=%s temp=%s C power=%s W", $1, $2, $3, $4, $5, $6, $7, $8 }' <<< "$__witness_out")
      echo "    card $lab clocks: $q"
    else
      echo "    card $lab clocks: unavailable (rc $__witness_rc)"
    fi
  done
  TWOCARD_WHY=''
  if __card_ok 3090 "$GPU_3090" 3090 250; then echo "    3090 cap: ok ($CARD_LINE)"; else echo "    3090 cap: NOT HELD: $TWOCARD_WHY"; fi
  if xid_read; then
    echo "    xid: $XID_N NVRM Xid line(s) since the lease was taken (@$XID_T0): A6000 $XID_NA, 3090 $XID_NB, other $((XID_N - XID_NA - XID_NB)); last: ${XID_LAST:-none}"
  else
    echo "    xid: unavailable ($XID_ERR)"
  fi
  echo "    cpu-freq: $(cpu_freq_summary)"
  [ -z "${BIN_SHA:-}" ] || echo "    binary: ${BIN_PATH:-?} sha256=$BIN_SHA mtime=${BIN_MTIME:-?}"
  if [ -n "$GPU_3090" ]; then
    echo "    3090-apps: [$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i "$GPU_3090" | tr '\n' ';')]"
  else
    echo "    3090-apps: unresolved (no 3090 UUID${CARDS_ERROR:+: $CARDS_ERROR})"
  fi
  if [ -n "$GPU_A6000" ]; then
    echo "    a6000-apps: [$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i "$GPU_A6000" | tr '\n' ';')]"
  else
    echo "    a6000-apps: unresolved (no A6000 UUID${CARDS_ERROR:+: $CARDS_ERROR})"
  fi
  echo "    gpu: $(nvidia-smi --query-gpu=index,utilization.gpu,power.draw,clocks.sm --format=csv,noheader | tr '\n' ';')"
  echo "    load=$(cut -d' ' -f1-3 /proc/loadavg) io=$(grep '^some' /proc/pressure/io | cut -d' ' -f2) llm.service=$(systemctl is-active llm.service || true)"
}

# guard_cards: guard_other in the two-card mode. Both cards are timed, so a compute process on either as
# an arm starts (another round's functional run or an `any` gate finishing: minutes) is waited out,
# polling every 10 s, and after 10 minutes ends the run: a witness block and rc 75, contention, not a
# result. Every two-card arm is a process of its own, so none of the runner's own is on a card here.
# TIMING_CARDS_POLL (seconds, default 10) is the stub tests' (tools/ref/depth-*-stub.sh) short poll.
guard_cards() {
  local apps i uuid poll=${TIMING_CARDS_POLL:-10}
  # shellcheck disable=SC2034 # read by the runners that source this file
  OTHER_BUSY_TAG=
  for ((i = 0; i < 60; i++)); do
    # Bounded (__witness_smi): a card off the bus can hang the query, and this loop is inside the lease.
    # A query that fails counts as busy, named, so the wait ends at 75 instead of timing a lost card.
    apps=''
    for uuid in "$GPU_A6000" "$GPU_3090"; do
      __witness_smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader -i "$uuid"
      if [ "$__witness_rc" != 0 ]; then
        apps+="$uuid: nvidia-smi rc $__witness_rc"$'\n'
      elif [ -n "$__witness_out" ]; then
        apps+="$__witness_out"$'\n'
      fi
    done
    apps=${apps%$'\n'}
    if [ -z "$apps" ]; then
      [ "$i" = 0 ] || echo "[cards-busy] $(now) both cards are free after $((i * poll)) s" >&2
      return 0
    fi
    if [ "$i" = 0 ]; then
      echo "[cards-busy] $(now) compute apps on a timed card: [$(echo "$apps" | tr '\n' ';')]; waiting up to 10 min" >&2
      witness wait-cards >&2
    fi
    sleep "$poll"
  done
  echo "[cards-busy] $(now) still busy after $((60 * poll)) s: [$(echo "$apps" | tr '\n' ';')]" >&2
  witness abort-cards >&2
  exit 75
}

# timing_cards_arm <engine log> [ours]: after an arm's post witness in the two-card mode (0 at once without
# it): 1 with TWOCARD_WHY when a kernel Xid line came since the last arm's check (between the arms, or
# inside this one: the arm is charged either way, and the count moves on), when the Xid reader failed,
# when a card stopped answering or the 3090 left its cap, or when the engine's log does not show exactly
# the two cards as its devices 0 and 1 (ggml_cuda_init's `found N CUDA devices` and `Device i:` lines): a
# llama-bench that sees one card spreads nothing and would be timed as if it were one card. With `ours`
# the log is one of our binaries': its first `load` record's `cards` (read by tools/bloomery/records.py),
# the devices' own names, must be exactly two, the A6000's then the 3090's (the stage card, then the
# tier). TWOCARD_DEVS is `<device 0> + <device 1>` for the row.
timing_cards_arm() {
  local n d0 d1 rec CARDS
  TWOCARD_WHY='' TWOCARD_DEVS=''
  [ -n "$TIMING_CARDS" ] || return 0
  if ! xid_read; then
    TWOCARD_WHY="the Xid reader failed after the arm: $XID_ERR"
    return 1
  fi
  if [ "$XID_N" -gt "$XID_BASE" ]; then
    TWOCARD_WHY="$((XID_N - XID_BASE)) NVRM Xid line(s) since the last arm's check (A6000 $XID_NA, 3090 $XID_NB since the lease was taken); last: $XID_LAST"
    XID_BASE=$XID_N
    return 1
  fi
  __card_ok A6000 "$GPU_A6000" A6000 '' || return 1
  __card_ok 3090 "$GPU_3090" 3090 250 || return 1
  if [ "${2:-}" = ours ]; then
    rec=$(python3 "${BASH_SOURCE[0]%/*}/../bloomery/records.py" sh - CARDS=load.cards <<< "$1") || {
      TWOCARD_WHY="records.py did not read the engine's load record"
      return 1
    }
    eval "$rec"
    # The record's cards are the devices' own names, each space written `_`.
    d0=${CARDS#\[} d0=${d0%\]} d0=${d0//_/ }
    if [[ $d0 =~ ^[^,]*A6000[^,]*,[^,]*3090[^,]*$ ]]; then
      # shellcheck disable=SC2034 # TWOCARD_DEVS is read by the runners that source this file
      TWOCARD_DEVS="${d0//,/ + }"
      return 0
    fi
    TWOCARD_WHY="the engine's load record names cards ${CARDS:-(none)}, not the A6000 and the 3090: a one-card run in the two-card table"
    return 1
  fi
  n=$(sed -nE 's/.*ggml_cuda_init: found ([0-9]+) CUDA devices.*/\1/p' <<< "$1" | head -n 1)
  d0=$(sed -nE 's/^ *Device 0: ([^,]*),.*/\1/p' <<< "$1" | head -n 1)
  d1=$(sed -nE 's/^ *Device 1: ([^,]*),.*/\1/p' <<< "$1" | head -n 1)
  # shellcheck disable=SC2034 # TWOCARD_DEVS is read by the runners that source this file
  case "$n|$d0|$d1" in
    2\|*A6000*\|*3090*) TWOCARD_DEVS="$d0 + $d1" ;;
    *)
      TWOCARD_WHY="the engine saw ${n:-no} CUDA device(s) (device 0 '${d0:-?}', device 1 '${d1:-?}'), not the A6000 and the 3090: a one-card run in the two-card table"
      return 1
      ;;
  esac
}

# Refuse to measure a binary that is older than the sources it was built from, and record what
# was measured. tools/box.sh syncs source and builds nothing (AGENTS "Profile the binary you
# think you are profiling"), so a runner invoked without its build recipe measures whatever the
# last round left in target/ — that is how an ik_ref column printed an older binary's values.
# rc 2 = no binary or not called from the repo root, rc 3 = stale. Call this before taking the
# lease: a run that will be refused must not first wait half an hour for the lock.
#
# Only files cargo reads are compared. A whole-tree -newer sweep over-refuses, because the sync
# gives every file it transfers the box's own "now" and a RESULTS.md is not a build input.
# crates/oxide-ice-unroll is pruned for the same reason: it is excluded from the workspace on
# purpose (a compiler-bug reproducer that must not compile), so nothing in it is an input to
# any binary this function guards.
# The sha256 is taken once here and printed by every later witness block: it is what makes a
# past log answer "which binary was that row?" without rerunning anything.
assert_fresh_binary() {
  # The find below is relative, so the caller's cwd is part of the contract: from anywhere else
  # it walks nothing, finds nothing newer, and the staleness check silently passes.
  [ -f Cargo.toml ] || { echo "assert_fresh_binary: run from the repo root" >&2; return 2; }
  BIN_PATH=$1
  local newer
  if [ ! -x "$BIN_PATH" ]; then
    echo "no binary at $BIN_PATH — run the matching just build/gate recipe first" >&2
    return 2
  fi
  BIN_SHA=$(sha256sum "$BIN_PATH" | cut -c1-12)
  BIN_MTIME=$(date -u -r "$BIN_PATH" +%Y-%m-%dT%H:%M:%SZ)
  # Cargo writes the binary's dep-info next to it (<bin>.d: every source file of every crate it
  # links, device crates included). With it, only those files, the manifests and the lock count: an
  # edit to a crate the binary does not link (a gate bin, for bloomery-tokenize) is not staleness.
  # Without it, every crates/ source counts.
  local dep=$BIN_PATH.d scope
  if [ -f "$dep" ]; then
    scope='dep-info'
    # shellcheck disable=SC2046 # the dep-info list is space-separated paths without spaces
    newer=$(find $(sed -e 's/^[^:]*: *//' -e 's/\\$//' "$dep" | tr ' ' '\n' | grep -v '^$' | sort -u) \
              Cargo.toml Cargo.lock crates/*/Cargo.toml \
              -newer "$BIN_PATH" -print 2>/dev/null | head -n 5 || true)
  else
    scope="every crates/ source"
    newer=$(find crates Cargo.toml Cargo.lock \
              -path 'crates/oxide-ice-unroll' -prune -o \
              \( -name '*.rs' -o -name 'Cargo.toml' -o -name 'Cargo.lock' \) \
              -newer "$BIN_PATH" -print 2>/dev/null | head -n 5 || true)
  fi
  if [ -n "$newer" ]; then
    echo "[stale-binary] $BIN_PATH (sha256 $BIN_SHA, mtime $BIN_MTIME) is older than its sources:" >&2
    # shellcheck disable=SC2086
    printf '    %s\n' $newer >&2
    echo "    rebuild it with the matching just recipe and rerun; measuring this one would be a wrong number, not a missing one." >&2
    return 3
  fi
  echo "[binary] $BIN_PATH sha256=$BIN_SHA mtime=$BIN_MTIME (newer than its sources: $scope)"
  return 0
}
