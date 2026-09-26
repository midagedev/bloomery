#!/usr/bin/env bash
# bloomery — CPU 티어 측정 러너. 박스에서 tools/box.sh 경유로 돈다.
# 두 트랙이 같은 CPU를 동시에 재면 서로 오염되므로 기계 전역 임대(flock)로 직렬화한다.
# 임대 안에서: 증인(loadavg·io 압력·GPU 점유) → ggml CPU 참조($BLOOMERY_DATA/bin/q3k_cpu_ref) → Rust 커널.
# 인자: 없음. 환경: BLOOMERY_DATA(러너가 넘김), BLOOMERY_CPU_RUST(기본은 워크스페이스 target).
set -euo pipefail
# 데이터 디렉터리 기본값(BLOOMERY_DATA 오버라이드는 그대로 받는다)은 빌드 스크립트와 같은 파일이 소유한다.
# shellcheck source=tools/ref/ref-paths.sh
source "${BASH_SOURCE[0]%/*}/ref-paths.sh"
export BLOOMERY_DATA
# The lease and the witness fields.
# shellcheck source=tools/ref/lease.sh
source "${BASH_SOURCE[0]%/*}/lease.sh"
# 워크스페이스로 합친 뒤 바이너리는 레포 루트 target/에 떨어진다(2026-09-19).
# 옛 경로(crates/q3k-cpu/target)에 낡은 바이너리가 남아 있으면 그걸 재게 되므로,
# 경로가 없으면 조용히 넘어가지 않고 죽는다.
RUST=${BLOOMERY_CPU_RUST:-target/release/q3k-cpu}
[ -x "$RUST" ] || { echo "no rust binary at $RUST — run: just build-cpu" >&2; exit 2; }
WITNESS=(head loadavg pressure-cpu pressure-io gpus lock-holder model)
lease_take
witness pre-ref
lease_bounded "$LEASE_ARM_BOUND" "$BLOOMERY_DATA/bin/q3k_cpu_ref"
witness post-ref
witness pre-rust
lease_bounded "$LEASE_ARM_BOUND" "$RUST"
witness post-rust
