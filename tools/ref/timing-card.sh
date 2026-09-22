#!/usr/bin/env bash
# Shared timing-card setup for the GPU runners: depth-gpu.sh, ncu-gpu.sh, nsys-gpu.sh and
# time-gate.sh. This file is sourced, never executed, so it only defines variables and
# functions — nothing here may exit or fail at top level (time-gate.sh runs under `set -e`).
#
#   source "${BASH_SOURCE[0]%/*}/timing-card.sh"
#
# What it owns, so that four runners cannot drift apart:
#   the two card UUIDs and which one is the timing card; CUDA_VISIBLE_DEVICES; the witness
#   lines every runner prints; what to do when the other card is busy; and the refusal that
#   keeps a runner from timing a binary older than the sources it was built from.
#
# Timed numbers are taken on the A6000 and this overrides the 3090 pin in the box env file.
# The 3090 is the gate-and-build card. Numbers from the two cards never belong in one table,
# which is why the first witness line names the card and its power limit.
# shellcheck source=tools/ref/cards.sh
source "${BASH_SOURCE[0]%/*}/cards.sh"
TIMING_GPU=${BLOOMERY_TIMING_GPU:-$GPU_A6000}
if [ "$TIMING_GPU" = "$GPU_3090" ]; then OTHER_GPU=$GPU_A6000; else OTHER_GPU=$GPU_3090; fi
export CUDA_VISIBLE_DEVICES=$TIMING_GPU

now() { date -u +%Y-%m-%dT%H:%M:%SZ; }

# The witness lines shared by every runner. A caller's own witness() prints the header, calls
# this, and appends whatever else it wants. loadavg is not the quiet-machine signal (see
# docs/quiet-machine.md in rig-log): IO pressure and the actual process list are.
witness_card() {
  echo "    timing-card: $(nvidia-smi --query-gpu=name,power.limit,clocks.max.sm --format=csv,noheader -i "$TIMING_GPU")"
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
# The default witness block. Every runner defines its own right after sourcing this file and
# that definition wins; this one exists so guard_other below does not depend on the caller
# having done so — an abort path that calls an undefined function prints nothing where the
# record matters most.
witness() {
  echo "--- witness $1 $(now)"
  witness_card
}

guard_other() {
  local apps
  apps=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i "$OTHER_GPU")
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
  newer=$(find crates Cargo.toml Cargo.lock \
            -path 'crates/oxide-ice-unroll' -prune -o \
            \( -name '*.rs' -o -name 'Cargo.toml' -o -name 'Cargo.lock' \) \
            -newer "$BIN_PATH" -print 2>/dev/null | head -n 5 || true)
  if [ -n "$newer" ]; then
    echo "[stale-binary] $BIN_PATH (sha256 $BIN_SHA, mtime $BIN_MTIME) is older than its sources:" >&2
    # shellcheck disable=SC2086
    printf '    %s\n' $newer >&2
    echo "    rebuild it with the matching just recipe and rerun; measuring this one would be a wrong number, not a missing one." >&2
    return 3
  fi
  echo "[binary] $BIN_PATH sha256=$BIN_SHA mtime=$BIN_MTIME (newer than every crates/ source)"
  return 0
}
