#!/usr/bin/env bash
# mulle — CPU 티어 측정 러너. 박스에서 tools/box.sh 경유로 돈다.
# 두 트랙이 같은 CPU를 동시에 재면 서로 오염되므로 기계 전역 임대(flock)로 직렬화한다.
# 임대 안에서: 증인(loadavg·io 압력·GPU 점유) → ggml CPU 참조($MULLE_DATA/bin/q3k_cpu_ref) → Rust 커널.
# 인자: 없음. 환경: MULLE_DATA(러너가 넘김), MULLE_CPU_RUST(기본 crates/q3k-cpu의 release 바이너리).
set -euo pipefail
export MULLE_DATA=${MULLE_DATA:-/root/mulle-data}
LOCK=/root/mulle-cpu.lock
RUST=${MULLE_CPU_RUST:-crates/q3k-cpu/target/release/q3k-cpu}
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
