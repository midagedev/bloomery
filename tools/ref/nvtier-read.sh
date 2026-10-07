#!/usr/bin/env bash
# bloomery — the NVMe tier's reader under the machine-wide lease (lead-only: `just time-nvread`). Runs on the
# box through tools/box.sh, under the qwen4exp profile the recipe picks (BLOOMERY_REF_MODEL is Qwen3.8's UD-Q4_K_XL
# split set), after the recipe has built target/release/probe_nvread (its header has the batches, the arms and
# the row text). The card is docs/cards/nvtier-read.card, named to the lease by
# BLOOMERY_BOX_ENV='BLOOMERY_LEASE_CARD=docs/cards/nvtier-read.card'.
#
# Per round: one probe_nvread run under --lease, its rows printed whole, and the drive's own count of what it
# read across the run set beside what the rows say the probe asked of it:
#
#   [reconcile] round <r>: drive read <B> B, probe asked <B> B, ratio <x>
#
# The ratio is the card's no-other-reader and no-overread check; a ratio off 1 by more than 5 % ends the round
# line with [drive-mismatch] (the run is not failed by it: the card's condition judges). The witness blocks carry
# the drive's read counters, and the run prints the drive's queue as it was set (read-ahead, largest request,
# scheduler), which fix what one WILLNEED turns into.
#
# The cache arm allocates page-cache pages for what it reads, and the box's page cache is usually full, so the
# kernel's reclaim can run inside the measured window. Each round prints what the kernel's reclaim did across it
#
#   [reclaim] round <r>: pgscan_direct +<n> pgsteal_direct +<n> pgscan_kswapd +<n> pgsteal_kswapd +<n> allocstall +<n>
#
# (machine-wide counters from /proc/vmstat; the lease keeps other sittings off, not every process), tagged
# [reclaim] when a direct scan or an allocation stall happened: the cache arm then paid for reclaim and its rate
# is not the path's alone. MemFree below MEM_FREE_FLOOR_KB at the first witness is tagged [mem-full]. The CPU guard
# (lease.sh guard_cpu: builds and reference engines on the cores) tags the reconcile line [cpu-busy]. All three
# are flags, not failures: the card's condition judges the sitting.
#
# Arguments, each optional, the rest passed to the probe unchanged (its header lists them):
#   --rounds N       the probe's runs, each under its own seed (default 1)
#   --seed S         the first run's seed (default the probe's own); later rounds add their index
#   --range-from ID  the first expert id of the range case (default 256: half a layer's experts, about 0.8 GB, the
#                    prompt's NVMe-tier read of a layer on a 32 GB host [derived, the nvtier design])
#   --parse-only     print the parsed arguments and exit 0 before any check, lease or run
# Environment: BLOOMERY_ARM_BOUND, seconds one probe run may take (default 900, lease.sh's).
set -euo pipefail
# shellcheck source=tools/ref/ref-paths.sh
source "${BASH_SOURCE[0]%/*}/ref-paths.sh"
# The lease and the witness fields.
# shellcheck source=tools/ref/lease.sh
source "${BASH_SOURCE[0]%/*}/lease.sh"
BIN=target/release/probe_nvread
NROUNDS=1
SEED=$((0x6E7672656164))
RANGE_FROM=256
PARSE_ONLY=0
ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --rounds | --seed | --range-from)
      [ -n "${2:-}" ] || { echo "nvtier-read.sh: $1 needs a value" >&2; exit 64; }
      case $1 in
        --rounds) NROUNDS=$2 ;;
        --seed) SEED=$2 ;;
        --range-from) RANGE_FROM=$2 ;;
      esac
      shift 2 ;;
    --parse-only) PARSE_ONLY=1; shift ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
for v in "$NROUNDS" "$SEED" "$RANGE_FROM"; do
  case "$v" in ''|*[!0-9]*) echo "nvtier-read.sh: --rounds, --seed and --range-from are whole numbers, got '$v'" >&2; exit 64 ;; esac
done
[ "$NROUNDS" -ge 1 ] || { echo "nvtier-read.sh: --rounds is at least 1" >&2; exit 64; }
if [ "$PARSE_ONLY" = 1 ]; then
  echo "[parse] rounds=$NROUNDS seed=$SEED range-from=$RANGE_FROM probe-args=(${ARGS[*]-})"
  exit 0
fi
[ -f Cargo.toml ] || { echo "nvtier-read.sh: run from the repo root" >&2; exit 2; }
[ -x "$BIN" ] || { echo "no $BIN — run: just time-nvread (it builds the probe first)" >&2; exit 2; }

# Refuse a binary older than any file it was built from, as host-rate.sh does: cargo's dep-info for this binary,
# the manifests of the crates in it and the two files that set codegen. rc 3 stale, before the lease.
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

# The drive the model file is on is read from the mount, never guessed; a drive whose counters cannot be read
# is no instrument for the reconcile line, so the run ends by name.
[ -r "$MODEL" ] || { echo "nvtier-read.sh: the model file $MODEL is not readable (profile $MODEL_NAME)" >&2; exit 2; }
DIR=$(dirname "$MODEL")
SRC=$(findmnt -n -o SOURCE --target "$DIR") || { echo "nvtier-read.sh: findmnt cannot name the mount of $DIR" >&2; exit 2; }
DEV=$(lsblk -n -o PKNAME "$SRC" 2>/dev/null | head -n1 || true)
[ -n "$DEV" ] || DEV=$(basename "$SRC")
STAT=/sys/block/$DEV/stat
[ -r "$STAT" ] || { echo "nvtier-read.sh: no counters at $STAT (mount source $SRC, device $DEV)" >&2; exit 2; }

WITNESS=(head loadavg pressure-cpu pressure-io table blockstat meminfo pgmajfault cpu cpu-mhz-range lock-holder binary model)

# queue: the drive's request settings, the numbers one WILLNEED's readahead is cut by.
queue() {
  local q=/sys/block/$DEV/queue f out=
  for f in read_ahead_kb max_sectors_kb nr_requests; do
    out+="$f=$(cat "$q/$f" 2>/dev/null || echo '?') "
  done
  echo "${out}scheduler=$(cat "$q/scheduler" 2>/dev/null || echo '?')"
}
# sectors: sectors read from the drive so far (field 3 of its stat, 512 B each).
sectors() { awk '{print $3}' "$STAT"; }
# reclaim: the kernel's reclaim counters so far: direct scan, direct steal, kswapd scan, kswapd steal, allocation stalls.
reclaim() {
  awk '$1 == "pgscan_direct" {a = $2} $1 == "pgsteal_direct" {b = $2} $1 == "pgscan_kswapd" {c = $2}
       $1 == "pgsteal_kswapd" {d = $2} $1 ~ /^allocstall/ {e += $2}
       END {printf "%.0f %.0f %.0f %.0f %.0f\n", a, b, c, d, e}' /proc/vmstat
}
# MemFree below this at the first witness is tagged [mem-full] (a chosen threshold, not a measured one: a few
# range batches' pages).
MEM_FREE_FLOOR_KB=4194304

# mappers: each process, other than this runner, that maps a file of the model's directory. Its mapping keeps
# those pages in the page cache whatever the probe drops, so the probe would read warm bytes; contention, not a
# failed run: wait for the process to end and start again.
mappers() {
  local f p
  for f in $(grep -l -F "$DIR/" /proc/[0-9]*/maps 2> /dev/null || true); do
    p=${f#/proc/}
    p=${p%/maps}
    [ "$p" != "$$" ] || continue
    echo "pid $p ($(cat "/proc/$p/comm" 2> /dev/null || echo '?'), up $(ps -o etimes= -p "$p" 2> /dev/null | tr -d ' ' || echo '?') s)"
  done
}

OUT=$(mktemp /tmp/nvtier-read.XXXXXX)
trap 'rm -f "$OUT"' EXIT
lease_take
holders=$(mappers)
if [ -n "$holders" ]; then
  echo "[model-mapped] a process maps a file of $DIR, so its pages stay in the page cache: the probe would read warm bytes (exit 75):" >&2
  printf '    %s\n' "$holders" >&2
  exit 75
fi
witness pre
echo "    queue($DEV): $(queue)"
memfree=$(awk '$1 == "MemFree:" {print $2}' /proc/meminfo)
if [ "$memfree" -lt "$MEM_FREE_FLOOR_KB" ]; then
  echo "[mem-full] MemFree ${memfree} kB is under ${MEM_FREE_FLOOR_KB} kB: the cache arm's allocations meet reclaim unless the drops free enough (the [reclaim] lines say)"
fi
rc=0
for ((round = 1; round <= NROUNDS; round++)); do
  seed=$((SEED + round - 1))
  echo "--- nvread round $round seed $seed $(now) load=$(cut -d' ' -f1-3 /proc/loadavg) io=$(grep '^some' /proc/pressure/io | cut -d' ' -f2)"
  CPU_BUSY_TAG=
  guard_cpu "pre r$round"
  before=$(sectors)
  rec_before=$(reclaim)
  r=0
  lease_bounded "$LEASE_ARM_BOUND" "$BIN" --lease --range-from "$RANGE_FROM" --seed "$seed" ${ARGS[@]+"${ARGS[@]}"} > "$OUT" || r=$?
  after=$(sectors)
  rec_after=$(reclaim)
  guard_cpu "post r$round"
  cat "$OUT"
  if [ "$r" != 0 ]; then
    echo "[round $round] probe_nvread failed (rc $r)" >&2
    rc=1
    break
  fi
  asked=$(awk '{for (i = 1; i <= NF; i++) if ($i ~ /^read_bytes=/) { sub(/^read_bytes=/, "", $i); s += $i }} END { printf "%.0f", s }' "$OUT")
  awk -v round="$round" -v b="$before" -v a="$after" -v asked="$asked" -v cpu="$CPU_BUSY_TAG" 'BEGIN {
    drive = (a - b) * 512
    ratio = asked > 0 ? drive / asked : 0
    flag = (ratio < 0.95 || ratio > 1.05) ? " [drive-mismatch]" : ""
    printf "[reconcile] round %d: drive read %.0f B, probe asked %.0f B, ratio %.4f%s%s\n", round, drive, asked, ratio, flag, cpu
  }'
  awk -v round="$round" -v b="$rec_before" -v a="$rec_after" 'BEGIN {
    split(b, x, " "); split(a, y, " ")
    tag = (y[1] > x[1] || y[5] > x[5]) ? " [reclaim]" : ""
    printf "[reclaim] round %d: pgscan_direct +%.0f pgsteal_direct +%.0f pgscan_kswapd +%.0f pgsteal_kswapd +%.0f allocstall +%.0f%s\n", round, y[1]-x[1], y[2]-x[2], y[3]-x[3], y[4]-x[4], y[5]-x[5], tag
  }'
done
witness post
echo "nvtier-read rc=$rc"
exit "$rc"
