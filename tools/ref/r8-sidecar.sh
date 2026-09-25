#!/usr/bin/env bash
# The r8 sidecar of the V4.1 file (crates/model/src/r8file.rs): `r8conv convert`, then
# `r8conv verify`, on $BLOOMERY_V41_MODEL (tools/box.sh exports it), both under the machine-wide
# CPU lease. The pair reads the file's routed gate and up stacks (155.7 GB), writes as many beside
# the source and reads them back: minutes of NVMe pressure and major faults a timed run must not
# share the machine with. A tool run like build-ref, not a gate and not in `just gate`.
#
#   ./tools/box.sh 'cargo build --release -p bloomery-model --bin r8conv && bash tools/ref/r8-sidecar.sh [verify]'
#
# `just r8-sidecar` is that line. `verify` skips the conversion and checks an existing sidecar
# again. The sidecar goes where `r8conv path` puts it (r8file::sidecar_path, the one owner of the
# name). One bound covers both commands, R8_SIDECAR_BOUND seconds (1800 unless set): verify gets
# what convert left of it. R8_SIDECAR_DRY=1 prints what would run and exits before the lease. Both
# reach the box through BLOOMERY_BOX_ENV (`BLOOMERY_BOX_ENV='R8_SIDECAR_DRY=1' just r8-sidecar`).
#
# The lease is lease.sh's lease_take, the call the timing runners make, so whatever lease.sh asks of
# a run (a card through BLOOMERY_BOX_ENV, where it checks one) this runner asks too; run it directly,
# not under another lease holder. It prints the lease line, a witness block before and after, the
# sidecar filesystem's sectors read and written, and each command's rc and seconds; the exit status
# is the first failing command's.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
# shellcheck source=tools/ref/lease.sh
source "$HERE/tools/ref/lease.sh"
BIN=$HERE/target/release/r8conv
SRC=${BLOOMERY_V41_MODEL:?BLOOMERY_V41_MODEL unset — run through tools/box.sh, which exports it}
STEP=${1:-convert}
case $STEP in
  convert | verify) ;;
  *)
    echo "r8-sidecar: the one argument is 'verify' (check without converting), got '$STEP'" >&2
    exit 64
    ;;
esac
[ "$#" -le 1 ] || { echo "r8-sidecar: one argument at most, got $#" >&2; exit 64; }
BOUND=${R8_SIDECAR_BOUND:-1800}
case $BOUND in '' | *[!0-9]*) echo "r8-sidecar: R8_SIDECAR_BOUND is seconds, got '$BOUND'" >&2; exit 64 ;; esac
[ -x "$BIN" ] || { echo "r8-sidecar: $BIN is missing — build it first (just r8-sidecar does)" >&2; exit 66; }
OUT=$("$BIN" path "$SRC")

# The sidecar's filesystem: its directory, or the nearest one that exists before convert makes it.
DIR=$(dirname "$OUT")
while [ ! -d "$DIR" ]; do DIR=$(dirname "$DIR"); done
DEV=$(df --output=source "$DIR" | tail -n 1)
# Fields 3 and 7 of the block device's stat: sectors read and written (512 B each), machine-wide.
dev_sectors() {
  echo "[r8-sidecar] $1 $DEV read_sectors/write_sectors: $(awk '{print $3, $7}' "/sys/class/block/${DEV#/dev/}/stat" 2> /dev/null || echo '?')"
}

if [ "${R8_SIDECAR_DRY:-}" = 1 ]; then
  echo "[dry] source: $SRC"
  echo "[dry] sidecar: $OUT (exists: $([ -e "$OUT" ] && echo yes || echo no); .part: $([ -e "$OUT.part" ] && echo yes || echo no))"
  echo "[dry] filesystem: $DEV at $DIR, $(df -B1 --output=avail "$DIR" | tail -n 1 | tr -d ' ') bytes free"
  echo "[dry] binary: $BIN sha256=$(sha256sum "$BIN" | cut -c1-12)"
  echo "[dry] would take the lease $LEASE_LOCK, then run, bound ${BOUND}s in all:"
  [ "$STEP" = verify ] || echo "[dry]   timeout --kill-after=10 $BOUND $BIN convert $SRC $OUT"
  echo "[dry]   timeout --kill-after=10 <the bound's rest> $BIN verify $SRC $OUT"
  exit 0
fi

lease_take
# shellcheck disable=SC2034 # the witness fields read them
MODEL=$SRC
# shellcheck disable=SC2034
WITNESS=(head loadavg pressure-io mem pgmajfault read-sectors busiest)
echo "[r8-sidecar] binary $BIN sha256=$(sha256sum "$BIN" | cut -c1-12)"
witness pre
dev_sectors pre
t0=$(date +%s)
rc=0
if [ "$STEP" = convert ]; then
  timeout --kill-after=10 "$BOUND" "$BIN" convert "$SRC" "$OUT" || rc=$?
  echo "[r8-sidecar] convert rc=$rc secs=$(($(date +%s) - t0))"
fi
if [ "$rc" -eq 0 ]; then
  left=$((BOUND - ($(date +%s) - t0)))
  if [ "$left" -le 0 ]; then
    echo "[r8-sidecar] the ${BOUND}s bound is spent; verify did not run" >&2
    rc=124
  else
    t1=$(date +%s)
    timeout --kill-after=10 "$left" "$BIN" verify "$SRC" "$OUT" || rc=$?
    echo "[r8-sidecar] verify rc=$rc secs=$(($(date +%s) - t1))"
  fi
fi
witness post
dev_sectors post
lease_release
if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
  echo "[r8-sidecar] timed out (exit $rc) — the ${BOUND}s bound" >&2
fi
exit "$rc"
