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
# assignments for a runner that sets the lever per arm (depth-ds41.sh).
dspark_env() {
  echo "CUDA_VISIBLE_DEVICES=$TIMING_GPU,$OTHER_GPU"
  echo "BLOOMERY_DSPARK_MODEL=${BLOOMERY_DSPARK_MODEL:-${DSPARK_MODEL:-}}"
}
if [ "${BLOOMERY_DRAFT:-}" = dspark ]; then
  export CUDA_VISIBLE_DEVICES=$TIMING_GPU,$OTHER_GPU
  export BLOOMERY_DSPARK_MODEL=${BLOOMERY_DSPARK_MODEL:-${DSPARK_MODEL:-}}
fi

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
  echo "    timing-card: $(nvidia-smi --query-gpu=name,power.limit,clocks.max.sm --format=csv,noheader -i "$TIMING_GPU")"
  echo "    timing-card clocks: $(nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,clocks_event_reasons.active,clocks_event_reasons_counters.sw_power_cap,clocks_event_reasons_counters.sw_thermal_slowdown,clocks_event_reasons_counters.hw_thermal_slowdown,temperature.gpu,power.draw --format=csv,noheader,nounits -i "$TIMING_GPU" | awk -F', ' '{ printf "sm=%s max=%s MHz event_reasons=%s capped_us sw_power=%s sw_thermal=%s hw_thermal=%s temp=%s C power=%s W", $1, $2, $3, $4, $5, $6, $7, $8 }')"
  echo "    cpu-freq: $(cpu_freq_summary)"
  [ -z "${BIN_SHA:-}" ] || echo "    binary: ${BIN_PATH:-?} sha256=$BIN_SHA mtime=${BIN_MTIME:-?}"
  echo "    3090-apps: [$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i "$GPU_3090" | tr '\n' ';')]"
  echo "    a6000-apps: [$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i "$GPU_A6000" | tr '\n' ';')]"
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
