#!/usr/bin/env bash
# The host-to-card links under the machine-wide lease (lead-only: `just time-gpu-h2d`): how the cards
# are attached, what one pinned copy stream gives on the timing card, and — under BLOOMERY_CARD=both —
# what the two links give together. Runs on the box through tools/box.sh after the recipe has built
# target/release/h2d_probe (its header has the arms and their lines).
#
#   step 1  attach       nvidia-smi topo -m and every card in view: name, PCI address, the sysfs path
#                        (which names the root port), max and idle link
#   step 2  pinned,pageable   1 GiB x 8 on the timing card: the calibration against docs/facts.md's
#                        A6000 row (pinned 26.28, pageable 21.16 GB/s)
#   step 3  sustained    48 back-to-back copies of 1.6 GB on the timing card: per-copy GB/s min,
#                        median, max and the mean, the link sampled while they run
#   step 4  (both only) sustained on the other card alone, then `both`: the two loops released
#                        together, each card's rate and the sum over the window where both ran
#
# BLOOMERY_H2D_ARMS=backleg (through BLOOMERY_BOX_ENV, with BLOOMERY_LEASE_CARD=docs/cards/zprobe.card)
# replaces the steps above with the decode step's back-leg sitting: the four arms backleg-host,
# backleg-dev, backleg-host+chunks, backleg-dev+chunks on the timing card, then the same four
# reversed (the rotation), both passes printed — the arm's `h2d arm=backleg-…` lines carry the
# round-trip, submit and chunk numbers (h2d_probe's header has the protocol).
#
# The cards are named by name and PCI address, never by ordinal: box.sh puts the 3090 first under
# BLOOMERY_CARD=both, this runner puts the timing card first (CUDA_VISIBLE_DEVICES per step). The
# timing card is timing-card.sh's (the A6000; BLOOMERY_TIMING_GPU moves it). Step 4 runs only when
# box.sh put both cards in view (BLOOMERY_BOX_CARD=both), which also means box.sh found the A6000
# without a compute process; the other card with a compute process at step 4 (another round's gate)
# skips it and ends the run with rc 75. The witness blocks list every card and count the kernel's `NVRM: Xid`
# lines before and after (the 3090 has fallen off the bus under load); every step opens with a line per
# card in its view: name, PCI address, power limit (the 3090's 250 W cap) and max SM clock.
#
# Arguments: --parse-only prints the steps and exits 0 before any check, lease or run. Each step is
# bounded by lease_bounded (BLOOMERY_ARM_BOUND, default 900 s). The run's card: BLOOMERY_LEASE_CARD
# through BLOOMERY_BOX_ENV (docs/cards/pcie-a6000.card; the backleg mode's is docs/cards/zprobe.card).
set -euo pipefail
# shellcheck source=tools/ref/timing-card.sh
source "${BASH_SOURCE[0]%/*}/timing-card.sh"
BIN=target/release/h2d_probe
PARSE_ONLY=0
while [ $# -gt 0 ]; do
  case $1 in
    --parse-only) PARSE_ONLY=1; shift ;;
    *) echo "h2d-pcie.sh: unknown argument '$1' (only --parse-only)" >&2; exit 64 ;;
  esac
done
ARMS_MODE="${BLOOMERY_H2D_ARMS:-default}"
BACKLEG_FOUR="backleg-host,backleg-dev,backleg-host+chunks,backleg-dev+chunks"
BACKLEG_FOUR_REVERSED="backleg-dev+chunks,backleg-host+chunks,backleg-dev,backleg-host"
BOTH=0
[ "${BLOOMERY_BOX_CARD:-}" = both ] && BOTH=1
case "$ARMS_MODE" in
  default)
    steps=("attach" "pinned,pageable" "sustained")
    [ "$BOTH" = 1 ] && steps+=("sustained on the other card" "both")
    ;;
  backleg)
    steps=("$BACKLEG_FOUR" "$BACKLEG_FOUR_REVERSED (rotation)")
    ;;
  *)
    echo "h2d-pcie.sh: unknown BLOOMERY_H2D_ARMS='$ARMS_MODE' (default or backleg)" >&2
    exit 64
    ;;
esac
if [ "$PARSE_ONLY" = 1 ]; then
  echo "[parse] both=$BOTH mode=$ARMS_MODE steps: ${steps[*]}"
  exit 0
fi
assert_fresh_binary "$BIN" || exit $?
if [ "$BOTH" = 1 ]; then VIEW=$TIMING_GPU,$OTHER_GPU; else VIEW=$TIMING_GPU; fi

# xid_lines: the kernel log's NVRM Xid lines so far (dmesg needs root, which the box runs as).
xid_lines() {
  local log
  if ! log=$(dmesg 2> /dev/null); then
    echo "unavailable (dmesg failed)"
    return 0
  fi
  grep -c 'NVRM: Xid' <<< "$log" || true
}

WITNESS=(head indent card gpus gpu-apps busiest)
lease_take
witness pre
xid_pre=$(xid_lines)
echo "    xid-lines: $xid_pre"
rc=0
# step <visible cards> <h2d_probe arguments…>
step() {
  local view=$1 r=0 uuid
  shift
  echo "--- h2d $* on CUDA_VISIBLE_DEVICES=$view $(now)"
  for uuid in ${view//,/ }; do
    echo "    card: $(nvidia-smi --query-gpu=name,pci.bus_id,power.limit,clocks.max.sm --format=csv,noheader -i "$uuid" 2>&1 || true)"
  done
  CUDA_VISIBLE_DEVICES=$view lease_bounded "$LEASE_ARM_BOUND" "$BIN" "$@" || r=$?
  echo "--- h2d $* rc=$r"
  [ "$r" = 0 ] || rc=$r
}
if [ "$ARMS_MODE" = backleg ]; then
  step "$TIMING_GPU" --arm "$BACKLEG_FOUR"
  step "$TIMING_GPU" --arm "$BACKLEG_FOUR_REVERSED"
else
  step "$VIEW" --arm attach
  step "$TIMING_GPU" --arm pinned,pageable
  step "$TIMING_GPU" --arm sustained
  if [ "$BOTH" = 1 ]; then
    # Another round's gate may hold the other card (the lease stops timing runs, not gates): the two-card
    # steps then run beside it, so they do not run.
    guard_other
    if [ -n "$OTHER_BUSY_TAG" ]; then
      echo "[other-busy] step 4 not run: the other card has compute processes" >&2
      rc=75
    else
      step "$OTHER_GPU" --arm sustained
      step "$VIEW" --arm both
    fi
  fi
fi
witness post
xid_post=$(xid_lines)
echo "    xid-lines: $xid_post"
if [ "$xid_pre" != "$xid_post" ]; then
  echo "[xid] the kernel logged NVRM Xid lines during the run ($xid_pre -> $xid_post): the run is void" >&2
  dmesg 2> /dev/null | grep 'NVRM: Xid' | tail -n 5 >&2 || true
  rc=1
fi
echo "h2d-pcie rc=$rc"
exit "$rc"
