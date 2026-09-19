#!/usr/bin/env bash
# mulle — CPU 티어 측정 러너. 박스에서 tools/box.sh 경유로 돈다.
# 두 트랙이 같은 CPU를 동시에 재면 서로 오염되므로 기계 전역 임대(flock)로 직렬화한다.
# 임대 안에서: 증인(loadavg·io 압력·GPU 점유) → ggml CPU 참조($MULLE_DATA/bin/q3k_cpu_ref) → Rust 커널.
# 인자: 없음. 환경: MULLE_DATA(러너가 넘김), MULLE_CPU_RUST(기본은 워크스페이스 target).
set -euo pipefail
export MULLE_DATA=${MULLE_DATA:-/root/mulle-data}
LOCK=/root/mulle-cpu.lock
# 워크스페이스로 합친 뒤 바이너리는 레포 루트 target/에 떨어진다(2026-09-19).
# 옛 경로(crates/q3k-cpu/target)에 낡은 바이너리가 남아 있으면 그걸 재게 되므로,
# 경로가 없으면 조용히 넘어가지 않고 죽는다.
RUST=${MULLE_CPU_RUST:-target/release/q3k-cpu}
[ -x "$RUST" ] || { echo "no rust binary at $RUST — run: just build-cpu" >&2; exit 2; }
witness() {
  echo "--- witness $1 $(date -u +%Y-%m-%dT%H:%M:%SZ) ---"
  echo "loadavg: $(cat /proc/loadavg)"
  echo "pressure-cpu: $(grep '^some' /proc/pressure/cpu | head -n1)"
  echo "pressure-io: $(grep '^some' /proc/pressure/io | head -n1)"
  nvidia-smi --query-gpu=index,name,utilization.gpu,power.draw --format=csv,noheader
  echo "lock-holder-pid: $$"
}
exec 9>"$LOCK"
echo "[lease] waiting for $LOCK ..."
flock -w 1800 9 || { echo "[lease] timed out after 30 min"; exit 75; }
echo "[lease] acquired $(date -u +%H:%M:%SZ)"
witness pre-ref
"$MULLE_DATA/bin/q3k_cpu_ref"
witness post-ref
witness pre-rust
"$RUST"
witness post-rust
