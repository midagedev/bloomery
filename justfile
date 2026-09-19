# mulle 작업 목록. 모든 타깃은 맥에서 치고 박스에서 돈다(tools/box.sh가 rsync한다).
# 맥은 arm64라 CPU 커널이 아예 빌드되지 않는다. 로컬 cargo로 게이트를 돌리려 하지 말 것.

default:
    @just --list

# 빠른 루프: 타입 검사만, 커널은 안 만든다.
check:
    ./tools/box.sh 'cd ~/repo/mulle && cargo check --workspace --all-targets'

# lint. 에러 0이 계약이고 경고 수는 RESULTS/AGENTS에 적힌 기준선과 비교한다.
lint:
    ./tools/box.sh 'cd ~/repo/mulle && cargo clippy --workspace --all-targets'

fmt:
    ./tools/box.sh 'cd ~/repo/mulle && cargo fmt --all'

fmt-check:
    ./tools/box.sh 'cd ~/repo/mulle && cargo fmt --all -- --check'

# GPU 커널 빌드. 디바이스 크레이트는 반드시 cargo oxide로, 평범한 cargo build로는 안 된다.
build-gpu:
    ./tools/box.sh 'cd ~/repo/mulle && cargo oxide build --arch sm_86 -- -p q3k-gemv'

build-cpu:
    ./tools/box.sh 'cd ~/repo/mulle/crates/q3k-cpu && RUSTFLAGS="-C target-cpu=znver3" cargo build --release'

# 측정. 러너가 조용한 기계 규약(GPU 유휴 대기 / 기계 전역 flock)과 증인 기록을 소유한다.
# 측정값을 손으로 모으지 말고 이 두 타깃만 쓴다.
measure-gpu:
    ./tools/box.sh 'cd ~/repo/mulle && bash tools/ref/measure.sh'

measure-cpu:
    ./tools/box.sh 'cd ~/repo/mulle && bash tools/ref/cpu-measure.sh'

# 참조 하네스(ggml에 링크하는 C++). 진실값과 기준 속도의 출처다.
build-ref:
    ./tools/box.sh 'cd ~/repo/mulle && bash tools/ref/build.sh && bash tools/ref/build-cpu.sh'

# 의존성 감사. cuda-oxide가 rev로 고정돼 있는지가 핵심이다.
deny:
    ./tools/box.sh 'cd ~/repo/mulle && cargo deny check'

# 커밋 전에 치는 것. 측정은 포함하지 않는다(조용한 기계가 필요하다).
gate: fmt-check lint build-gpu build-cpu
