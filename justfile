# bloomery 작업 목록. 모든 타깃은 맥에서 치고 박스에서 돈다(tools/box.sh가 rsync한다).
# 맥은 arm64라 CPU 커널이 아예 빌드되지 않는다. 로컬 cargo로 게이트를 돌리려 하지 말 것.
#
# 레시피는 절대 `cd ~/repo/bloomery`를 쓰지 않는다. box.sh가 이미 $REMOTE로 들어가고 그 값은
# 워크트리 이름에서 유도된다 — 레시피가 다시 cd하면 워크트리 라운드가 메인 트리를 재게 된다
# (2026-09-19 ffn 라운드가 보고: "gate-ffn은 ~/repo/bloomery로 가는데 이 워크트리는
# ~/repo/bloomery-ffn으로 rsync된다. 둘 다 박스에 있어서 엉뚱한 트리를 시험한다").

default:
    @just --list

# 빠른 루프: 타입 검사만, 커널은 안 만든다.
check:
    ./tools/box.sh 'cargo check --workspace --all-targets'

# lint. 에러 0이 계약이고 경고 수는 RESULTS/AGENTS에 적힌 기준선과 비교한다.
lint:
    ./tools/box.sh 'cargo clippy --workspace --all-targets'

# fmt는 맥에서 돈다. box.sh의 rsync가 단방향이라 박스에서 포맷하면 결과가 돌아오지
# 않고 다음 명령에 덮여 사라진다(2026-09-19에 그렇게 한 번 날렸다). cargo fmt는 컴파일을
# 하지 않고 파싱만 하므로 arm64 맥에서 정상 동작한다 — AGENTS.md의 "맥에서 게이트 금지"는
# 빌드가 필요한 것에 대한 규칙이고, 판정은 박스의 fmt-check가 한다.
fmt:
    cargo fmt --all

fmt-check:
    ./tools/box.sh 'cargo fmt --all -- --check'

# GPU 커널 빌드. 디바이스 크레이트는 반드시 cargo oxide로, 평범한 cargo build로는 안 된다.
build-gpu:
    ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-q3k-gemv'

build-cpu:
    ./tools/box.sh 'cd crates/q3k-cpu && RUSTFLAGS="-C target-cpu=znver3" cargo build --release'

# 측정. 러너가 조용한 기계 규약(GPU 유휴 대기 / 기계 전역 flock)과 증인 기록을 소유한다.
# 측정값을 손으로 모으지 말고 이 두 타깃만 쓴다.
measure-gpu:
    ./tools/box.sh 'bash tools/ref/measure.sh'

measure-cpu:
    ./tools/box.sh 'bash tools/ref/cpu-measure.sh'

# 참조 하네스(ggml에 링크하는 C++). 진실값과 기준 속도의 출처다.
build-ref:
    ./tools/box.sh 'bash tools/ref/build.sh && bash tools/ref/build-cpu.sh'

# 의존성 감사. cuda-oxide가 rev로 고정돼 있는지가 핵심이다.
deny:
    ./tools/box.sh 'cargo deny check'

# 오라클 계측기. 1-2부터의 게이트가 읽는 참조 텐서를 $BLOOMERY_DATA/ref/에 만든다.
# ik 빌드가 바뀌면 다시 돌린다 — 참조는 그 빌드의 출력이다.
build-ref-dump:
    ./tools/box.sh 'bash tools/ref/build-dump.sh'

dump-ref:
    ./tools/box.sh 'bash tools/ref/dump.sh'

# 1단계 서브블록 게이트. 각 라운드가 자기 것 하나만 소유한다.
gate-ops:
    ./tools/box.sh 'cargo test -p bloomery-model --test ops -- --ignored --nocapture'

gate-attn:
    ./tools/box.sh 'cargo test -p bloomery-model --test attn -- --ignored --nocapture'

gate-ffn:
    ./tools/box.sh 'cargo test -p bloomery-model --test ffn -- --ignored --nocapture'

gate-moe:
    ./tools/box.sh 'cargo test -p bloomery-model --test moe -- --ignored --nocapture'

gate-head:
    ./tools/box.sh 'cargo test -p bloomery-model --test head -- --ignored --nocapture'

# 1단계 1-1 게이트: 디퀀트 오라클을 빌드해 ggml의 to_float 덤프를 만들고, gguf 크레이트의
# hw 테스트가 그것과 대조한다. hw_ 접두는 박스를 요구한다는 뜻이고 기본 실행에서 빠져 있다.
gate-1-1:
    ./tools/box.sh 'bash tools/ref/build-dequant.sh && "$BLOOMERY_DATA/bin/dequant_ref" && cargo test -p bloomery-gguf -- --ignored --nocapture'

# 커밋 전에 치는 것. 측정은 포함하지 않는다(조용한 기계가 필요하다).
gate: fmt-check lint build-gpu build-cpu gate-1-1
