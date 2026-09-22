#!/usr/bin/env bash
# bloomery — engram 토큰당 비용 러너(B3). 박스에서 tools/box.sh 경유로 돈다.
# 인자는 engram-rate에 그대로 넘어간다(--tokens, --rows-per-token, --seed, --arms, --model-dir).
#
# 이 측정이 재는 것은 NVMe다. 그래서 증인이 CPU 러너와 다르다: loadavg가 아니라
# 드라이브 자체의 점유를 적는다. /sys/block/<dev>/stat의 읽기 I/O 수와 섹터 수를
# 측정 앞뒤로 남기면, 우리가 낸 것보다 많이 읽혔는지가 사후에 판정된다 —
# 드라이브는 공유 자원이고 loadavg는 I/O 바운드 이웃을 늦게 보여준다.
#
# 임대는 CPU 러너와 같은 기계 전역 파일이다. 측정 둘이 같은 드라이브를 동시에 때리면
# 둘 다 무효이므로, 큐 깊이를 재는 이 러너가 임대 밖에서 도는 일은 없다.
set -euo pipefail
# shellcheck source=tools/ref/ref-paths.sh
source "${BASH_SOURCE[0]%/*}/ref-paths.sh"
export BLOOMERY_DATA

BIN=${BLOOMERY_ENGRAM_BIN:-target/release/engram-rate}
[ -x "$BIN" ] || { echo "no engram-rate at $BIN — run: just build-engram" >&2; exit 2; }

DIR=${BLOOMERY_V41_DIR:-/models/DeepSeek-V4.1-Flash-Q3_K_M-engramQ8-tokembdBF16-attnQ8}
# 테이블이 어느 블록 디바이스에 있는지는 추측하지 않는다 — 마운트에서 읽는다.
SRC=$(findmnt -n -o SOURCE --target "$DIR")
# `|| true` 안에서 받는다: set -e 아래에서는 대입 안의 파이프라인이 실패하면 그 줄에서 죽어,
# 다음 줄의 폴백이 영영 안 돈다. 발견이 실패해도 러너는 증인만 약해진 채 계속 가야 한다.
DEV=$(lsblk -n -o PKNAME "$SRC" 2>/dev/null | head -n1 || true)
[ -n "$DEV" ] || DEV=$(basename "$SRC")
STAT=/sys/block/$DEV/stat

LOCK=/root/bloomery-cpu.lock
witness() {
  echo "--- witness $1 $(date -u +%Y-%m-%dT%H:%M:%SZ) ---"
  echo "table: $DIR on $SRC (block device $DEV)"
  echo "loadavg: $(cat /proc/loadavg)"
  echo "pressure-io: $(grep '^some' /proc/pressure/io | head -n1)"
  # 필드 1·3 = 완료된 읽기 I/O 수, 읽은 섹터(512 B) 수. 앞뒤 차이가 드라이브가 실제로 한 일이다.
  echo "blockstat($DEV) read_ios/read_sectors: $(awk '{print $1, $3}' "$STAT")"
  echo "meminfo cached/free kB: $(awk '/^Cached:/{c=$2} /^MemFree:/{f=$2} END{print c, f}' /proc/meminfo)"
  echo "lock-holder-pid: $$"
}

exec 9>"$LOCK"
echo "[lease] waiting for $LOCK ..."
flock -w 1800 9 || { echo "[lease] timed out after 30 min"; exit 75; }
echo "[lease] acquired $(date -u +%H:%M:%SZ)"
witness pre
# 임대 안이라고 바이너리에 말해 준다 — 없으면 줄마다 [not under lease]가 찍힌다.
BLOOMERY_ENGRAM_LEASE=1 "$BIN" --model-dir "$DIR" "$@"
witness post
