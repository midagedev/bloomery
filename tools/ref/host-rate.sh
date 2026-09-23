#!/usr/bin/env bash
# bloomery — the V4.1 host-expert leg runner (lead-only: `just time-cpu-v41-host`). Runs on the box
# through tools/box.sh, with the deepseek41 profile picked on the Mac side (BLOOMERY_MODEL=deepseek41).
#
# `bench_v41_host --time` once per thread count, all inside the machine-wide CPU lease, with witness
# blocks before and after. BLOOMERY_THREADS is read once per process when the pool is built, so each
# count is its own process. The pool pins its own workers and the bench pins its main thread the way
# bloomery-decode does, so nothing here wraps the bench in taskset; the bench prints every thread's
# cpu, core and L3 group as the kernel reports them.
#
# Arguments: an optional `--threads 16,32` first (the thread counts in order, default 8,16,24,30,32),
# then the bench's own, passed on unchanged (--rounds, --seconds, --warmup, --arms; `bench_v41_host`
# with no arguments prints them). Environment: BLOOMERY_HOST_BOUND, seconds one thread count may take
# (default 600).
set -euo pipefail
# shellcheck source=tools/ref/ref-paths.sh
source "${BASH_SOURCE[0]%/*}/ref-paths.sh"
# The lease and the witness fields.
# shellcheck source=tools/ref/lease.sh
source "${BASH_SOURCE[0]%/*}/lease.sh"
BIN=target/release/bench_v41_host
THREADS="8 16 24 30 32"
if [ "${1:-}" = --threads ]; then
  [ -n "${2:-}" ] || { echo "host-rate.sh: --threads needs a list, e.g. --threads 16,32" >&2; exit 64; }
  THREADS=${2//,/ }
  shift 2
fi
BOUND=${BLOOMERY_HOST_BOUND:-600}
for t in $THREADS $BOUND; do
  case "$t" in ''|*[!0-9]*|0) echo "host-rate.sh: thread counts and the bound are positive integers, got '$t'" >&2; exit 64 ;; esac
done
[ -f Cargo.toml ] || { echo "host-rate.sh: run from the repo root" >&2; exit 2; }
[ -x "$BIN" ] || { echo "no $BIN — run: just time-cpu-v41-host (it builds the bench first)" >&2; exit 2; }

# Refuse a binary older than any file it was built from. The file set is cargo's own dep-info for this
# binary plus the manifests of the crates in it and the two files that set codegen for every crate — not
# every crate under crates/: the bench does not depend on the GPU crates, so a sync that touched only
# those must not refuse it (cargo would rightly not rebuild it). The workspace Cargo.toml and Cargo.lock
# stay out for the same reason. Same refusal and exit code as timing-card.sh's assert_fresh_binary:
# rc 3 stale, and it runs before the lease.
DEPS=$BIN.d
[ -f "$DEPS" ] || { echo "no dep-info at $DEPS — rebuild with the recipe" >&2; exit 2; }
BIN_SHA=$(sha256sum "$BIN" | cut -c1-12)
BIN_MTIME=$(date -u -r "$BIN" +%Y-%m-%dT%H:%M:%SZ)
sources=$(sed -e 's/^[^:]*://' -e 's/\\$//' "$DEPS" | tr ' ' '\n' | grep -v '^$' || true)
manifests=$(printf '%s\n' "$sources" | sed -n 's#^\(.*/crates/[^/]*\)/src/.*#\1/Cargo.toml#p' | sort -u)
# shellcheck disable=SC2086
newer=$(printf '%s\n' $sources $manifests .cargo/config.toml rust-toolchain.toml \
          | while read -r f; do [ -e "$f" ] && [ "$f" -nt "$BIN" ] && echo "$f"; done | head -n 5 || true)
if [ -n "$newer" ]; then
  echo "[stale-binary] $BIN (sha256 $BIN_SHA, mtime $BIN_MTIME) is older than files it was built from:" >&2
  # shellcheck disable=SC2086
  printf '    %s\n' $newer >&2
  echo "    rebuild it with the recipe and rerun; measuring this one would be a wrong number, not a missing one." >&2
  exit 3
fi
echo "[binary] $BIN sha256=$BIN_SHA mtime=$BIN_MTIME (newer than every file in its dep-info)"

WITNESS=(head loadavg pressure-cpu pressure-io gpus gpu-apps lock-holder cpu cpu-mhz-range meminfo binary model)

lease_take
witness pre
start=$(date +%s)
rc=0
for t in $THREADS; do
  echo "--- threads $t $(date -u +%H:%M:%SZ) load=$(cut -d' ' -f1-3 /proc/loadavg) io=$(grep '^some' /proc/pressure/io | cut -d' ' -f2)"
  t0=$(date +%s)
  if ! BLOOMERY_THREADS=$t BLOOMERY_HOST_LEASE=1 timeout --kill-after=10 "$BOUND" "$BIN" --time "$@"; then
    rc=1
    echo "[threads $t] bench_v41_host failed or passed the ${BOUND} s bound — stopping the sweep" >&2
    break
  fi
  echo "--- threads $t done in $(( $(date +%s) - t0 )) s"
done
echo "--- sweep wall $(( $(date +%s) - start )) s"
witness post
exit "$rc"
