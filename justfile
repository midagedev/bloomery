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

# fmt는 맥에서 돈다. box.sh의 rsync가 단방향이라 박스에서 포맷하면 결과가 돌아오지
# 않고 다음 명령에 덮여 사라진다(2026-09-19에 그렇게 한 번 날렸다). cargo fmt는 컴파일을
# 하지 않고 파싱만 하므로 arm64 맥에서 정상 동작한다 — AGENTS.md의 "맥에서 게이트 금지"는
# 빌드가 필요한 것에 대한 규칙이고, 판정은 박스의 fmt-check가 한다.
fmt:
    cargo fmt --all

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

# 오라클 계측기. 1-2부터의 게이트가 읽는 참조 텐서를 $MULLE_DATA/ref/에 만든다.
# ik 빌드가 바뀌면 다시 돌린다 — 참조는 그 빌드의 출력이다.
build-ref-dump:
    ./tools/box.sh 'cd ~/repo/mulle && bash tools/ref/build-dump.sh'

dump-ref:
    ./tools/box.sh 'cd ~/repo/mulle && bash tools/ref/dump.sh'

# 1단계 서브블록 게이트. 각 라운드가 자기 것 하나만 소유한다.
gate-ops:
    ./tools/box.sh 'cd ~/repo/mulle && cargo test -p model --test ops -- --ignored --nocapture'

gate-attn:
    ./tools/box.sh 'cd ~/repo/mulle && cargo test -p model --test attn -- --ignored --nocapture'

gate-ffn:
    ./tools/box.sh 'cd ~/repo/mulle && cargo test -p model --test ffn -- --ignored --nocapture'

gate-moe:
    ./tools/box.sh 'cd ~/repo/mulle && cargo test -p model --test moe -- --ignored --nocapture'

gate-head:
    ./tools/box.sh 'cd ~/repo/mulle && cargo test -p model --test head -- --ignored --nocapture'

# 1단계 1-1 게이트: 디퀀트 오라클을 빌드해 ggml의 to_float 덤프를 만들고, gguf 크레이트의
# hw 테스트가 그것과 대조한다. hw_ 접두는 박스를 요구한다는 뜻이고 기본 실행에서 빠져 있다.
gate-1-1:
    ./tools/box.sh 'cd ~/repo/mulle && bash tools/ref/build-dequant.sh && "$MULLE_DATA/bin/dequant_ref" && cargo test -p gguf -- --ignored --nocapture'

# 커밋 전에 치는 것. 측정은 포함하지 않는다(조용한 기계가 필요하다).
gate: fmt-check lint build-gpu build-cpu gate-1-1
