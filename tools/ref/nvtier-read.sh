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
# line with [drive-mismatch] and the over-read named on it — the difference in bytes and its sign, the device,
# the round's pre/post sector counts — and the run ends rc 1 after the post witness (the rows stay printed: a
# void sitting is still a record). Beside it the reference judge: the O_DIRECT row `arm=drive case=range` must
# sit within REF_TOL of REF_GBS, or the run measured the drive's state, not the path, and the round prints
# [reference-off] and the run ends rc 1 the same way (a round whose probe printed no reference row is the same
# failure, named "no reference row"). The witness blocks carry
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
#   --self-test      the two round judges (the reference band, the reconcile over-read) on fixed text, the
#                    r0 sitting's rows and numbers, before any check, lease or run; `self-test: ok (<n>
#                    checks)`, or each failure named, rc 1. check-recipes does not run it: a script it names
#                    that calls lease_take makes gate-batch.sh refuse the check as a timed recipe.
# Environment: BLOOMERY_ARM_BOUND, seconds one probe run may take (default 900, lease.sh's).
set -euo pipefail
# shellcheck source=tools/ref/ref-paths.sh
source "${BASH_SOURCE[0]%/*}/ref-paths.sh"
# The lease and the witness fields.
# shellcheck source=tools/ref/lease.sh
source "${BASH_SOURCE[0]%/*}/lease.sh"
# The drive the model is on and its read counter.
# shellcheck source=tools/ref/drive-read.sh
source "${BASH_SOURCE[0]%/*}/drive-read.sh"

# The O_DIRECT reference's band: `arm=drive case=range` outside REF_GBS ± REF_TOL read the drive in another
# state than the sitting's bands were derived for, and the sitting is void (docs/cards/nvtier-read.card).
# REF_GBS is the drive's own 4 MiB sequential read rate [measured, rig-log 09-29#odmax].
REF_GBS=6.12
REF_TOL=0.10

# ref_judge <round> <probe-output-file>: the round's reference judge. Reads the `arm=drive case=range` row's
# median from the probe's own rows; inside the band it is silent and green, outside — or with no such row —
# it prints [reference-off] and returns 1: the sitting measured the drive's state, not the path.
ref_judge() {
  local round=$1 out=$2 med
  med=$(awk '$1 == "nvread" && $2 == "arm=drive" && $3 == "case=range" {
    for (i = 4; i <= NF; i++) if ($i ~ /^median=/) { sub(/^median=/, "", $i); print $i; exit }
  }' "$out")
  if [ -z "$med" ]; then
    echo "[reference-off] round $round: no reference row (arm=drive case=range) in the probe's output: the sitting is void (docs/cards/nvtier-read.card)"
    return 1
  fi
  awk -v round="$round" -v med="$med" -v ref="$REF_GBS" -v tol="$REF_TOL" 'BEGIN {
    lo = ref * (1 - tol); hi = ref * (1 + tol)
    eps = 0.0005  # half the last printed digit of the median: a red line never shows a median inside the band
    if (med + 0 >= lo - eps && med + 0 <= hi + eps) exit 0
    printf "[reference-off] round %d: arm=drive case=range median %s GB/s outside %.3f..%.3f (%.2f GB/s ± %.0f %%, rig-log 09-29#odmax): the sitting is void (docs/cards/nvtier-read.card)\n", round, med, lo, hi, ref, tol * 100
    exit 1
  }'
}

# reconcile_round <round> <before-sectors> <after-sectors> <asked-bytes> <device> <cpu-tag>: the round's
# [reconcile] line, the drive's own count (sectors x 512) beside the bytes the probe's rows say it asked. A
# ratio off 1 by more than 5 % keeps the [drive-mismatch] flag, names the over-read — the difference in bytes
# and its sign, the device, the round's pre/post sector counts — and returns 1: another reader or a read-ahead
# moved the drive while the sitting read it.
reconcile_round() {
  local round=$1 before=$2 after=$3 asked=$4 dev=$5 cpu=${6:-}
  awk -v round="$round" -v b="$before" -v a="$after" -v asked="$asked" -v dev="$dev" -v cpu="$cpu" 'BEGIN {
    drive = (a - b) * 512
    ratio = asked > 0 ? drive / asked : 0
    d = drive - asked
    line = sprintf("[reconcile] round %d: drive read %.0f B, probe asked %.0f B, ratio %.4f", round, drive, asked, ratio)
    if (ratio < 0.95 || ratio > 1.05) {
      why = d >= 0 ? sprintf("another reader on %s, or read-ahead past the asked spans", dev) : sprintf("bytes the probe asked served without reading %s (resident pages a drop missed, or a counter wrap)", dev)
      print line " [drive-mismatch]" cpu sprintf(" over-read %s%.2f GB (drive − asked): %s (sectors %s -> %s)", d < 0 ? "-" : "+", (d < 0 ? -d : d) / 1e9, why, b, a)
      exit 1
    }
    print line cpu
  }'
}

# self_test: `nvtier-read.sh --self-test`, the two round judges on fixed text (the r0 sitting's rows and
# numbers: the reference 4.506 GB/s against 6.12 ± 10 %, the ratio 1.3718) beside a green round, before any
# check, lease or run (it runs on the Mac). One line per check, `ok <name>` or `FAIL <name>: …`; the last
# line is the verdict.
self_test() {
  local fails=0 checks=0 f out rc
  # row <name> <want-rc> <want-out>: the last judge call below, its rc and its whole output.
  row() {
    checks=$((checks + 1))
    if [ "$rc" = "$2" ] && [ "$out" = "$3" ]; then
      echo "ok $1"
    else
      echo "FAIL $1: rc $rc want $2; got [$out] want [$3]"
      fails=$((fails + 1))
    fi
  }
  f=$(mktemp "${TMPDIR:-/tmp}/nvtier-selftest.XXXXXX")
  # A green round: the reference in band (the cache and directN rows beside it in band are not the
  # reference), the drive's count equal to the probe's ask.
  cat > "$f" << 'ROWS'
probe_nvread: 48 layers x 512 experts, 8 threads, seed 121455004442980, lease true
nvread arm=cache case=range experts=256 threads=8 batches=16 warmup=2 redraws=0 bytes=801177600 span_bytes=804323328 read_bytes=14448328704 median=5.900 p10=5.758 p90=6.746 ms=128.560 seed=121455004442980 lease=true
nvread arm=directN case=range experts=256 threads=8 batches=16 warmup=2 redraws=0 bytes=801177600 span_bytes=804323328 read_bytes=14448328704 median=6.189 p10=4.304 p90=6.895 ms=127.094 seed=121455004442980 lease=true
nvread arm=drive case=range experts=256 threads=8 batches=16 warmup=2 redraws=0 bytes=801177600 span_bytes=801189888 read_bytes=14391926784 median=6.117 p10=5.758 p90=6.746 ms=174.660 seed=121455004442980 lease=true
ROWS
  rc=0; out=$(ref_judge 1 "$f") || rc=$?
  row reference-in-band 0 ''
  rc=0; out=$(reconcile_round 1 243964771106 244084822226 61466173440 nvme0n1 '') || rc=$?
  row reconcile-ratio-1 0 "[reconcile] round 1: drive read 61466173440 B, probe asked 61466173440 B, ratio 1.0000"
  # The r0 sitting's reference: 4.506 GB/s, under 6.12 - 10 % — the decoy rows in band must not read as the
  # reference.
  sed -e 's/ median=6.117 p10=5.758 p90=6.746 ms=174.660 / median=4.506 p10=2.954 p90=5.217 ms=174.660 /' "$f" > "$f.2" && mv "$f.2" "$f"
  rc=0; out=$(ref_judge 1 "$f") || rc=$?
  row reference-off-4.506 1 "[reference-off] round 1: arm=drive case=range median 4.506 GB/s outside 5.508..6.732 (6.12 GB/s ± 10 %, rig-log 09-29#odmax): the sitting is void (docs/cards/nvtier-read.card)"
  # The band's other side, and its edge: 6.800 is out, a median at the printed hi itself is in.
  sed -e 's/ median=4.506 / median=6.800 /' "$f" > "$f.2" && mv "$f.2" "$f"
  rc=0; out=$(ref_judge 1 "$f") || rc=$?
  row reference-off-6.800 1 "[reference-off] round 1: arm=drive case=range median 6.800 GB/s outside 5.508..6.732 (6.12 GB/s ± 10 %, rig-log 09-29#odmax): the sitting is void (docs/cards/nvtier-read.card)"
  sed -e 's/ median=6.800 / median=6.732 /' "$f" > "$f.2" && mv "$f.2" "$f"
  rc=0; out=$(ref_judge 1 "$f") || rc=$?
  row reference-at-hi-in-band 0 ''
  # No reference row at all: the probe printed no arm=drive row.
  grep -v '^nvread arm=drive ' "$f" > "$f.2" && mv "$f.2" "$f"
  rc=0; out=$(ref_judge 1 "$f") || rc=$?
  row reference-row-missing 1 "[reference-off] round 1: no reference row (arm=drive case=range) in the probe's output: the sitting is void (docs/cards/nvtier-read.card)"
  # The r0 sitting's over-read: drive 84318838784 B against the probe's 61466173440 B asked, ratio 1.3718.
  rc=0; out=$(reconcile_round 1 243964771106 244129456338 61466173440 nvme0n1 '') || rc=$?
  row reconcile-overread-1.3718 1 "[reconcile] round 1: drive read 84318838784 B, probe asked 61466173440 B, ratio 1.3718 [drive-mismatch] over-read +22.85 GB (drive − asked): another reader on nvme0n1, or read-ahead past the asked spans (sectors 243964771106 -> 244129456338)"
  rm -f "$f"
  echo "self-test: $([ "$fails" = 0 ] && echo ok || echo FAIL) ($checks checks, $fails failures)"
  [ "$fails" = 0 ]
}

BIN=target/release/probe_nvread
NROUNDS=1
SEED=$((0x6E7672656164))
RANGE_FROM=256
PARSE_ONLY=0
SELF_TEST=0
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
    --self-test) SELF_TEST=1; shift ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
for v in "$NROUNDS" "$SEED" "$RANGE_FROM"; do
  case "$v" in ''|*[!0-9]*) echo "nvtier-read.sh: --rounds, --seed and --range-from are whole numbers, got '$v'" >&2; exit 64 ;; esac
done
[ "$NROUNDS" -ge 1 ] || { echo "nvtier-read.sh: --rounds is at least 1" >&2; exit 64; }
if [ "$SELF_TEST" = 1 ]; then
  self_test
  exit $?
fi
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
DEV=$(drive_dev "$MODEL") || exit 2
# The witness's table and blockstat fields read the mount source and the device's counters.
SRC=$(findmnt -n -o SOURCE --target "$DIR")
STAT=/sys/block/$DEV/stat

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
sectors() { drive_sectors "$DEV"; }
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
  reconcile_round "$round" "$before" "$after" "$asked" "$DEV" "$CPU_BUSY_TAG" || rc=1
  awk -v round="$round" -v b="$rec_before" -v a="$rec_after" 'BEGIN {
    split(b, x, " "); split(a, y, " ")
    tag = (y[1] > x[1] || y[5] > x[5]) ? " [reclaim]" : ""
    printf "[reclaim] round %d: pgscan_direct +%.0f pgsteal_direct +%.0f pgscan_kswapd +%.0f pgsteal_kswapd +%.0f allocstall +%.0f%s\n", round, y[1]-x[1], y[2]-x[2], y[3]-x[3], y[4]-x[4], y[5]-x[5], tag
  }'
  ref_judge "$round" "$OUT" || rc=1
done
witness post
echo "nvtier-read rc=$rc"
exit "$rc"
