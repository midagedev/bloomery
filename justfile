# bloomery 작업 목록. 모든 타깃은 맥에서 치고 박스에서 돈다(tools/box.sh가 rsync한다).
# 맥은 arm64라 CPU 커널이 아예 빌드되지 않는다. 로컬 cargo로 게이트를 돌리려 하지 말 것.
#
# 레시피는 절대 `cd ~/repo/bloomery`를 쓰지 않는다. box.sh가 이미 $REMOTE로 들어가고 그 값은
# 워크트리 이름에서 유도된다 — 레시피가 다시 cd하면 워크트리 라운드가 메인 트리를 재게 된다
# (2026-09-19 ffn 라운드가 보고: "gate-ffn은 ~/repo/bloomery로 가는데 이 워크트리는
# ~/repo/bloomery-ffn으로 rsync된다. 둘 다 박스에 있어서 엉뚱한 트리를 시험한다").

default:
    @just --list

# `--features gpu`(R26): 피처 뒤에 본체가 숨은 바이너리는 그 피처를 켜야 검사된다 —
# gpu-gates 바이너리의 GPU 본체가 끄면 통째로 안 보인다.
# 빠른 루프: 타입 검사만, 커널은 안 만든다.
check:
    ./tools/box.sh 'cargo check --workspace --all-targets --features gpu'

# check와 같은 이유로 `--features gpu`. 이 피처를 켠 것이 기준 계기다(R26).
# lint. 에러 0이 계약이고 경고 수는 RESULTS/AGENTS에 적힌 기준선과 비교한다.
lint:
    ./tools/box.sh 'cargo clippy --workspace --all-targets --features gpu'

# fmt는 맥에서 돈다. box.sh의 rsync가 단방향이라 박스에서 포맷하면 결과가 돌아오지
# 않고 다음 명령에 덮여 사라진다(2026-09-19에 그렇게 한 번 날렸다). cargo fmt는 컴파일을
# 하지 않고 파싱만 하므로 arm64 맥에서 정상 동작한다 — AGENTS.md의 "맥에서 게이트 금지"는
# 빌드가 필요한 것에 대한 규칙이고, 판정은 박스의 fmt-check가 한다.
fmt:
    cargo fmt --all

fmt-check:
    ./tools/box.sh 'cargo fmt --all -- --check'

# 레시피 자체의 점검(맥, grep뿐). 게이트 줄의 `||`는 종료 코드를 삼킨다 — tools/check-recipes.sh 머리말.
check-recipes:
    ./tools/check-recipes.sh

# 아키텍처 축 점검(맥, grep뿐) — docs/arch-split.md 「검사」. 넷 다 엄격하다.
check-arch:
    bash tools/check-arch.sh

# 주석 규약(AGENTS.md Conventions): 엔진 크레이트 src/ 주석에 이슈 번호·날짜 금지, 예외는 `PIN(날짜):`.
check-comments:
    ./tools/check-comments.sh

# GPU 커널 빌드. 디바이스 크레이트는 반드시 cargo oxide로, 평범한 cargo build로는 안 된다.
build-gpu:
    ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-q3k-gemv'

# GPU P0 게이트(docs/gpu-design.md 꾸러미 P0): 라이브러리 gemv가 y_ref 1e-2 안이고, 즉시 실행과
# 그래프 재생의 출력 바이트가 같아야 한다. 호스트 제출 비용도 찍으므로 release 호스트 빌드다
# (dev 호스트는 P0가 재려는 그 숫자를 부풀린다). 3090만 쓴다(박스 env가 UUID를 핀).
# GPU 게이트 바이너리는 박스의 게이트 락 하나에 줄을 선다(빌드는 병렬, 실행만 직렬). 게이트 하나가 모델을 12 GB
# 올리므로 둘이 같은 카드에 겹치면 OOM이다 — 2026-09-22, 세 워크트리의 게이트가 3090에 동시에 올라 하나가
# DriverError(2)로 죽었다. 시간 러너의 임대(/root/bloomery-cpu.lock)와는 다른 락이다: 게이트는 3090, 시간은 A6000.
gate-gpu-p0:
    ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-spike --features gpu --release && flock -w 1800 /root/bloomery-gate.lock ./target/release/gpu-spike'

# GPU 커널 게이트 P1–P3: 트랙마다 자기 바이너리 하나(crates/gpu-gates/src/bin/gate_pN.rs).
# 참조는 bloomery_gpu_gates(gguf 디퀀트 + f64 내적), 밴드는 KERNEL_BAND = 1e-2. 정확성 실행이고 측정이 아니다.
gate-gpu-p1:
    ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin gate_p1 && flock -w 1800 /root/bloomery-gate.lock ./target/release/gate_p1'

gate-gpu-p2:
    ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin gate_p2 && flock -w 1800 /root/bloomery-gate.lock ./target/release/gate_p2'

gate-gpu-p3:
    ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin gate_p3 && flock -w 1800 /root/bloomery-gate.lock ./target/release/gate_p3'

gate-gpu-p4:
    ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin gate_p4 && flock -w 1800 /root/bloomery-gate.lock ./target/release/gate_p4'

gate-gpu-p5:
    ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin gate_p5 && flock -w 1800 /root/bloomery-gate.lock ./target/release/gate_p5'

gate-gpu-p6:
    ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin gate_p6 && flock -w 1800 /root/bloomery-gate.lock ./target/release/gate_p6'

gate-gpu-p9:
    ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin gate_p9 && flock -w 1800 /root/bloomery-gate.lock ./target/release/gate_p9'

# 종단 게이트: 27층 + lm_head + argmax 전체 사슬을 ik CUDA greedy(33프롬프트 × 32스텝)와 대조하고,
# 그래프 재생이 즉시 실행과 토큰 단위로 같은지 본다(정확성 실행, 측정 아님). 이름이 pN이 아닌 이유는
# gate_e2e.rs 머리글 — P9·P10은 커널 꾸러미로 이미 쓰이고 있다. 두 번 돈다: 기본 패스(텐서 코어 flash)와
# `BLOOMERY_FLASH_MMA=0`(스칼라 패스). 레버가 프로세스 시작 때 한 번 읽히므로 프로세스를 가른다. 스칼라 패스
# 쪽이 flash 단계 배가 팔(keyaxis)을 돌린다 — 기본 패스에서는 그 팔이 건너뛰어진다. ARGS는 첫 호출에만 간다.
gate-gpu-e2e *ARGS:
    ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin gate_e2e && flock -w 1800 /root/bloomery-gate.lock ./target/release/gate_e2e {{ARGS}} && BLOOMERY_FLASH_MMA=0 flock -w 1800 /root/bloomery-gate.lock ./target/release/gate_e2e'

# 교사 강제 자의 한 위치(프롬프트 ID, 스텝 S)를 두 팔로 연다: 스칼라 팔이 덤프를 쓰고 MMA 팔이 그것과 대조해
# 두 팔의 로짓 top-k·ik 토큰 순위·강제 스텝별 마진·층별 탭 상대 거리·라우팅 차이를 찍는다. 예: `just forced-probe 12 23`.
forced-probe ID STEP *ARGS:
    ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin forced_probe && BLOOMERY_FLASH_MMA=0 flock -w 1800 /root/bloomery-gate.lock ./target/release/forced_probe --prompt-id {{ID}} --step {{STEP}} --dump target/forced-probe/scalar {{ARGS}} && BLOOMERY_FLASH_MMA=1 flock -w 1800 /root/bloomery-gate.lock ./target/release/forced_probe --prompt-id {{ID}} --step {{STEP}} --dump target/forced-probe/mma --against target/forced-probe/scalar {{ARGS}}'

# 양자화 모델의 정확한 수학(가중치는 정확히 역양자화, 활성·어텐션은 f64, q8·f16 반올림 없음)으로 교사 강제 자의
# 한 프롬프트를 푼다: 스텝마다 정확 top1·마진·ik 토큰 격차. 두 엔진(우리, ik)이 갈리는 자리의 심판이다.
# CPU 64스레드를 쓰므로 기계 전역 임대를 잡는다. 예: `just exact-ref 12 --steps 23`, `--kv f16`은 캐시 f16 반올림 팔.
exact-ref ID *ARGS:
    ./tools/box.sh 'cargo build --release -p bloomery-gpu-gates --bin exact_ref && flock -w 3600 /root/bloomery-cpu.lock ./target/release/exact_ref --prompt-id {{ID}} {{ARGS}}'

# gate-gpu-e2e 교사 강제 팔의 참값 파일: prompts.tsv의 프롬프트마다 exact_ref --emit(32스텝 전부)을 돌려
# $BLOOMERY_DATA/exact-forced-32.tsv 하나로 모은다. 머리 주석에 모델 경로, 강제 토큰 파일(greedy-ik-cuda-32.tsv)의
# sha256, exact_ref 빌드 커밋(맥 워크트리의 git describe — 박스 트리에는 .git이 없다)과 실행한 바이너리의 sha256.
# 임대는 프롬프트마다 따로 잡는다(전체 약 50분 — 다른 라운드가 그 사이에 끼어들 수 있게). 실행하는 바이너리는
# 조각 디렉터리로 복사한 사본이라 도중의 다른 빌드가 바꾸지 못한다. 최종 파일은 .tmp.<pid>에 쓰고 rename한다.
# 강제 토큰 파일을 다시 만들었으면(greedy-ref-cuda) 이것도 다시 — 게이트가 sha256으로 낡은 참값을 거부한다.
build-exact-forced:
    #!/usr/bin/env bash
    set -euo pipefail
    commit=$(git describe --always --dirty --abbrev=12)
    script=$(cat <<'EOF'
    set -euo pipefail
    cargo build --release -p bloomery-gpu-gates --bin exact_ref
    out="$BLOOMERY_DATA/exact-forced-32.tsv"
    forced="$BLOOMERY_DATA/greedy-ik-cuda-32.tsv"
    ids=$(grep -v -e '^#' -e '^$' tools/ref/prompts.tsv | cut -f1)
    parts="$out.parts.$$"
    mkdir -p "$parts"
    cp target/release/exact_ref "$parts/exact_ref"
    for id in $ids; do
      flock -w 3600 /root/bloomery-cpu.lock "$parts/exact_ref" --prompt-id "$id" --emit "$parts/p$id.tsv" > "$parts/p$id.log"
      echo "exact-forced p$id rows=$(grep -vc '^#' "$parts/p$id.tsv")"
    done
    heads=$(for id in $ids; do head -n 1 "$parts/p$id.tsv"; done | sort -u)
    [ "$(printf '%s\n' "$heads" | wc -l)" = 1 ] || { echo "build-exact-forced: parts disagree: $heads" >&2; exit 1; }
    tmp="$out.tmp.$$"
    {
      echo "$heads forced=greedy-ik-cuda-32.tsv forced_sha256=$(sha256sum "$forced" | cut -d' ' -f1) exact_ref_commit=$COMMIT exact_ref_sha256=$(sha256sum "$parts/exact_ref" | cut -d' ' -f1)"
      printf '#id\tstep\texact_top1\texact_top2\texact_margin\n'
      for id in $ids; do grep -v '^#' "$parts/p$id.tsv"; done
    } > "$tmp"
    mv "$tmp" "$out"
    rm -rf "$parts"
    echo "build-exact-forced: wrote $out rows=$(grep -vc '^#' "$out")"
    EOF
    )
    ./tools/box.sh "COMMIT=$commit; $script"

# 한 강제 위치에서 우리 GPU 스칼라 팔의 층·탭별 상대 거리를 세 심판 팔에 대해 한 번에 찍는다: f64(정확),
# ours(우리 활성 양자화 규칙만 흉내 낸 f64 사슬 — K-quant 입력 128값 블록), ik(ik 규칙 — 전부 32값 블록).
# exact_ref가 세 팔을 CPU 임대 아래 차례로 덤프하고(프롬프트당 팔 하나 약 1.5분), forced_probe가 그 셋과 대조한다.
# 덤프와 exact_ref 로그는 박스 target/exact-taps/ID-STEP/. 예: `just exact-taps 12 23`.
exact-taps ID STEP:
    ./tools/box.sh 'cargo build --release -p bloomery-gpu-gates --bin exact_ref && cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin forced_probe && D=target/exact-taps/{{ID}}-{{STEP}} && mkdir -p $D && flock -w 3600 /root/bloomery-cpu.lock ./target/release/exact_ref --prompt-id {{ID}} --steps {{STEP}} --act f64 --dump $D/f64 > $D/f64.log && flock -w 3600 /root/bloomery-cpu.lock ./target/release/exact_ref --prompt-id {{ID}} --steps {{STEP}} --act ours --dump $D/ours > $D/ours.log && flock -w 3600 /root/bloomery-cpu.lock ./target/release/exact_ref --prompt-id {{ID}} --steps {{STEP}} --act ik --dump $D/ik > $D/ik.log && grep -H "top5" $D/f64.log $D/ours.log $D/ik.log && BLOOMERY_FLASH_MMA=0 flock -w 1800 /root/bloomery-gate.lock ./target/release/forced_probe --prompt-id {{ID}} --step {{STEP}} --dump $D/scalar --against $D/f64,$D/ours,$D/ik'

# 얇은 끝-끝 디코드 CLI(greedy, 토큰 하나씩, 프리필 커널 없음). `--time` 없이 토큰만 찍는 것은
# 평범한 실행이고, `--time`은 측정이라 임대가 필요하다 — time-gpu-generate가 그쪽이다.
generate *ARGS:
    ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin generate && ./target/release/generate {{ARGS}}'

# generate의 스텝당 ms(리드 전용): 기계 전역 임대 아래, 증인 블록 전후. 기록이 되는 것은 graph 모드의
# 푸터이고 eager 모드는 호스트 제출 경로의 값이다.
time-gpu-generate *ARGS:
    ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin generate && bash tools/ref/time-gate.sh generate {{ARGS}} --time'

# P0b: 블록 0 FFN 융합 스파이크 — 융합 4런치가 op 8런치와 비트 동일한지(정확성 실행). 시간은 리드가 임대 안에서 `--time`으로 잰다.
gate-gpu-p0b *ARGS:
    ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin gate_p0b && flock -w 1800 /root/bloomery-gate.lock ./target/release/gate_p0b {{ARGS}}'

# P0b 시간(리드 전용): 기계 전역 임대 아래 op 8노드 대 융합 4노드 그래프의 재생 µs, 빈 커널 4/8노드 참조 포함, 증인 블록 전후.
time-gpu-p0b:
    ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin gate_p0b && bash tools/ref/time-gate.sh gate_p0b'

# P8a 블록 0 스텝(ctx_max 64에서 21노드 — fuse1 뒤; 한 구간보다 큰 캐시에서는 flash_merge가 붙어 22)의 재생 µs — 임대·증인, 리드 전용.
time-gpu-p8:
    ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin gate_p8 && bash tools/ref/time-gate.sh gate_p8'

# P8 op별 프로파일(리드 전용): 스텝을 op 단위로 동기 분해한 µs 표(ops=그래프 노드 수와 동일) + sync_floor
# 보정 + refresh_params 호스트 시간 + 같은 프로세스의 그래프 재생 µs + 어텐션/FFN 분할. 임대·증인은 time-gate.sh 소유.
prof-gpu-p8:
    ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin gate_p8 && bash tools/ref/time-gate.sh gate_p8 --profile'

# 커널 런치당 실제 비용(리드 전용): op별 표가 답할 수 없는 것 — 한 행의 net_us가 디바이스 시간인지
# 프로파일 자신의 제출 경로인지 — 를 두 갈래로 가른다. eager는 N런치 뒤 동기화 하나(호스트 제출과
# 디바이스 시간 중 큰 쪽), graph는 같은 N런치를 그래프 하나로 캡처해 재생(스텝의 그래프 노드가 실제로
# 무는 값). 빈 커널 touch가 둘의 바닥이고, f32 gemv는 행 수 셋에서 같은 프로세스로 잰다. 임대·증인은 time-gate.sh 소유.
bench-gpu-kernels *ARGS:
    ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin gate_p8 && bash tools/ref/time-gate.sh gate_p8 --bench-kernels {{ARGS}}'

# MoE 융합 op 8노드 대 융합 4노드의 재생 µs — 임대·증인, 리드 전용.
time-gpu-moe:
    ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin gate_moe_fused && bash tools/ref/time-gate.sh gate_moe_fused'

# P8: 조립된 디코드 스텝(블록 0부터)을 덤프와 대조 — 엔진 자신의 탭.
gate-gpu-p8 *ARGS:
    ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin gate_p8 && flock -w 1800 /root/bloomery-gate.lock ./target/release/gate_p8 {{ARGS}}'

# MoE 융합(P8 준비): 전문가 여섯의 gate·up·swiglu 한 런치 + 결합 한 런치가 op 경로와 비트 동일한지(정확성 실행).
gate-gpu-moe:
    ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin gate_moe_fused && flock -w 1800 /root/bloomery-gate.lock ./target/release/gate_moe_fused'

# P7 뒤 절반: 블록·종단 게이트 하네스의 자기 검증(호스트 전용 — 두 오라클 사이의 알려진 거리를 재현해야 한다).
gate-gpu-block:
    ./tools/box.sh 'cargo run --release -p bloomery-gpu-gates --bin gate_block'

# P10: 가중치 상주 — 모델 파일의 모든 텐서를 커널이 먹는 디바이스 형식으로 올린다(정확성 실행, 측정 아님).
gate-gpu-p10:
    ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin gate_p10 && flock -w 1800 /root/bloomery-gate.lock ./target/release/gate_p10'

# 원시-x 측정 프로브(호스트 전용, 게이트 아님): ref_cuda의 실활성화에 대해 q8_1 양자화 바닥을 잰다.
# 정확 참조(ref_gemv)를 재사용하고 판정은 하지 않는다 — 수치로 밴드·생성기를 재판단하는 건 리드다.
rawx-floor:
    ./tools/box.sh 'cargo run --release -p bloomery-gpu-gates --bin rawx_floor'
# 실제 활성값(ref_cuda 덤프)에서 q8_1 커널의 편차를 잰다: 정확 참조 대비, ik CUDA 자신의 출력 대비. 단언 없는 계측기 —
# KERNEL_BAND를 합성 활성값이 아니라 실분포에서 핀하기 위한 표를 낸다. 정확성 실행이고 시간 측정이 아니다.
probe-gpu-real-x:
    ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin real_x && ./target/release/real_x'

build-cpu:
    ./tools/box.sh 'cd crates/q3k-cpu && RUSTFLAGS="-C target-cpu=znver3" cargo build --release'

# 빌드는 레시피 의존성이 한다(measure-decode와 같은 모양) — 러너가 재는 것은 방금 빌드된
# 바이너리여야 하고, 빌드는 임대·유휴 대기 밖에서 끝나야 한다. build-ref-bench는 두 참조 하네스
# ($BLOOMERY_DATA/bin/q3k_ref·q3k_cpu_ref)를 만든다: cpu-measure.sh는 그것을 부르면서
# 빌드는 하지 않았다. 전부(x4 하네스와 덤프까지)가 필요하면 build-ref다 — 측정 앞에 그것을
# 걸면 매 측정이 gate-qdot의 참조 덤프를 다시 쓰게 된다.
# 측정. 러너가 조용한 기계 규약(GPU 유휴 대기 / 기계 전역 flock)과 증인 기록을 소유한다.
# 측정값을 손으로 모으지 말고 이 두 타깃만 쓴다.
measure-gpu: build-ref-bench build-gpu
    ./tools/box.sh 'bash tools/ref/measure.sh'

measure-cpu: build-ref-bench build-cpu
    ./tools/box.sh 'bash tools/ref/cpu-measure.sh'

# 2026-09-20 사고(q_nope2 무한루크가 gate-mt를 매달아 병렬 에이전트 둘을 '무활동'으로 죽임)의
# 보강. 이 트랙 원격 디렉터리 아래 실행 파일을 물고 있는 고아 프로세스를 찾아 죽인다.
# 고르는 축은 /proc/<pid>/exe이지 cmdline이 아니다 — `pgrep -f "<dir>/target"`은 그 패턴을
# 자기 argv에 든 셸(스캔하는 셸 자신, ssh 핸들러)도 같이 고른다. 자신과 조상 pid는 접두
# 비교 전에 제외한다. 죽이는 것은 TERM → 5초 → KILL. 목록만 보려면 `just box-gc --dry-run`.
# 이 트랙 원격 디렉터리 아래 실행 파일을 문 고아 프로세스를 죽인다. 병렬 트랙 시작·끝에 한 번씩.
box-gc *ARGS='--kill':
    ./tools/box.sh 'bash tools/box-gc.sh {{ARGS}}'

# 박스에 남은 트랙 디렉터리를 로컬 워크트리와 대조한다. 인자 없이 목록, `just box-tracks --remove`로 stale 삭제.
box-tracks *ARGS:
    ./tools/box-tracks.sh {{ARGS}}

# 1-4의 첫 tok/s. 같은 임대 안에서 ik를 같은 파일·같은 조건으로 한 번 더 잰다.
build-decode:
    ./tools/box.sh 'cargo build --release -p bloomery-model --bin bloomery-decode'

measure-decode: build-decode
    ./tools/box.sh 'bash tools/ref/decode-measure.sh'

# 스레드 스윕. 같은 임대 안에서 1/8/16/32를 연달아 돌린다 — plan.md의 "스레드 수는
# 재서 정한다"가 이것이고, 답은 이 티어가 대역폭에 닿는 지점이다.
measure-sweep: build-decode
    ./tools/box.sh 'NOCACHE=0 SWEEP="1 8 16 32 64" bash tools/ref/decode-measure.sh'

# 시간 귀속. 러너가 임대와 증인을 소유한다 — 프로파일 표도 측정이고, 옆에서 빌드
# 하나만 돌아도 site 간 비율이 흔들린다. 레벨 1(배분)과 2(단계)를 연달아 찍는다.
# 같은 임대 안 A/B: `just ab-decode bloomery-<track> ...` (각 트리는 미리 build-decode).
# 인자는 박스 `~/repo/` 아래 **디렉터리 이름**이다 — 맥 절대경로를 줘도 basename으로 바꾼다.
# 맥 셸의 BLOOMERY_AB_ROUNDS·BLOOMERY_AB_ENVS·BLOOMERY_AB_IK는 ssh를 그냥 넘지 않으므로 여기서
# 원격 명령줄 앞에 K=V로 실어 보낸다. BLOOMERY_AB_ENVS는 작은따옴표로 싸서 넘기므로 값 안의
# 작은따옴표 하나가 그 인용을 깬다 — `K=V;K2=V2` 형태(이 변수의 문법 전부)는 안전하다.
#   BLOOMERY_AB_ROUNDS=6 BLOOMERY_AB_ENVS="BLOOMERY_SPIN=0" just ab-decode bloomery-foo
ab-decode *DIRS: build-decode
    #!/usr/bin/env bash
    set -euo pipefail
    names=
    for d in {{DIRS}}; do names="$names $(basename "$d")"; done
    ./tools/box.sh "${BLOOMERY_AB_ROUNDS:+BLOOMERY_AB_ROUNDS=$BLOOMERY_AB_ROUNDS} ${BLOOMERY_AB_ENVS:+BLOOMERY_AB_ENVS='$BLOOMERY_AB_ENVS'} ${BLOOMERY_AB_IK:+BLOOMERY_AB_IK=$BLOOMERY_AB_IK} bash tools/ref/ab-decode.sh$names"

measure-profile: build-decode
    ./tools/box.sh 'bash tools/ref/profile-measure.sh'

# 참조 하네스(ggml에 링크하는 C++). 진실값과 기준 속도의 출처다.
# build-qdot-ref.sh는 ik의 커널 테이블까지 링크하는 x4 하네스 여덟을 짓고 참조 하네스 넷을
# **실행까지** 한다 — gate-qdot의 hw 테스트 넷이 읽는 $BLOOMERY_DATA/ref/*-ik-dot.txt가 그
# 산출물이다. 나머지 넷(*_rate)은 빌드만 한다: 실행은 측정이다.
build-ref:
    ./tools/box.sh 'bash tools/ref/build.sh && bash tools/ref/build-cpu.sh && bash tools/ref/build-qdot-ref.sh'

# measure-gpu·measure-cpu가 거는 좁은 쪽: 두 러너가 실제로 부르는 ggml 링크 하네스
# ($BLOOMERY_DATA/bin/q3k_ref·q3k_cpu_ref)만 짓는다. x4 하네스를 짓고 **실행**하는
# build-qdot-ref.sh는 여기 없다 — 그 실행이 gate-qdot의 참조 덤프를 덮어쓰므로, 측정 하나가
# 다른 트랙의 게이트 입력을 갈아치우는 일이 된다.
build-ref-bench:
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

# GPU 엔진의 오라클: 같은 계측기를 -ngl 99로, $BLOOMERY_DATA/ref_cuda/에. CPU 참조는 건드리지 않는다.
dump-ref-cuda:
    ./tools/box.sh 'BLOOMERY_REF_BACKEND=cuda bash tools/ref/dump.sh'

# 1단계 서브블록 게이트. 각 라운드가 자기 것 하나만 소유한다.
gate-ops:
    ./tools/box.sh 'bash tools/gate.sh -p bloomery-model --test ops -- --ignored --nocapture'

gate-attn:
    ./tools/box.sh 'bash tools/gate.sh -p bloomery-model --test attn -- --ignored --nocapture'

gate-ffn:
    ./tools/box.sh 'bash tools/gate.sh -p bloomery-model --test ffn -- --ignored --nocapture'

gate-moe:
    ./tools/box.sh 'bash tools/gate.sh -p bloomery-model --test moe -- --ignored --nocapture'

gate-head:
    ./tools/box.sh 'bash tools/gate.sh -p bloomery-model --test head -- --ignored --nocapture'

# 1-4 조립 게이트. --release로 도는 유일한 게이트다 — 27블록 전체를 디버그 빌드로 돌리면
# 분 단위로 늘어나고, Rust는 f32를 재결합하지 않으므로 수치는 프로파일과 무관하게 같다.
gate-forward:
    ./tools/box.sh 'bash tools/gate.sh --release -p bloomery-model --test forward -- --ignored --nocapture'

# 1-5 KV 캐시 게이트: 캐시가 있는 경로와 없는 경로의 로짓이 비트 동일한가.
gate-kv:
    ./tools/box.sh 'bash tools/gate.sh --release -p bloomery-model --test kv -- --ignored --nocapture'

# 할당 게이트: 정상 상태 디코드 스텝 하나가 할당자를 몇 번 부르는가(전 스레드 합).
# 속도를 지키는 게이트는 없지만 이 원인 하나는 정수로 환원된다 — 메인 스레드가
# 할당·0 채우기를 하는 동안 워커 전부가 논다.
gate-alloc:
    ./tools/box.sh 'bash tools/gate.sh --release -p bloomery-model --test alloc -- --ignored --nocapture'

# Derived 게이트: 토큰과 무관한 wk_b Q8_0 재양자화를 로드시 한 번으로 옮겼다 —
# 사전 계산이 값을 바꾸지 않는다. 블록은 참조 구현(테스트 안의 예전 두 루프)과
# 바이트 동일, step(명시적 Derived)과 forward(래퍼)의 로짓은 비트 동일.
gate-derived:
    ./tools/box.sh 'bash tools/gate.sh --release -p bloomery-model --test derived -- --ignored --nocapture'

# 행 병렬 게이트: 스레드 수(1/3/32)가 로짓을 한 비트도 안 바꾸는가. BLOOMERY_THREADS는
# 프로세스당 한 번 읽히므로 자식 프로세스 재실행으로 덤프를 뽑아 바이트 비교한다
# (tests/mt.rs의 패턴 설명 참조).
gate-mt:
    ./tools/box.sh 'bash tools/gate.sh --release -p bloomery-model --test mt -- --ignored --nocapture'

# 1-5 프로파일러 게이트: 계측이 로짓을 한 비트도 안 바꾸고, 스텝 시간의 80% 이상을 커버하며,
# 세 site가 전부 살아 있는가. --test-threads=1은 게이트 본체의 set_var이 자식 헬퍼 테스트와
# 경쟁하지 않게 하는 장치다(테스트 파일의 SAFETY 주석 참조).
gate-profile:
    ./tools/box.sh 'bash tools/gate.sh --release -p bloomery-model --test profile -- --ignored --nocapture --test-threads=1'

# 1-5 스레드 풀 게이트: 상주 워커 풀의 분할 전수·커버리지·반복 호출·패닉 전파.
# hw_ 토폴로지 테스트는 #[ignore]라 --include-ignored로 같이 돈다.
gate-threads:
    ./tools/box.sh 'bash tools/gate.sh --release -p bloomery-threads --test pool -- --include-ignored --nocapture'

# 2단계 qdot 게이트: Q3_K×Q8_K 융합 커널이 현재 경로(dequant+roundtrip+f32)와
# 1e-5 안팎에서 일치하고, 정확해(f64)에 더 가깝고, 스칼라 폴백과 비트 동일인가.
# 순수 게이트(rejects_unaligned_k)는 #[ignore]가 아니라 --include-ignored로 같이 돈다.
gate-qdot:
    ./tools/box.sh 'bash tools/gate.sh --release -p bloomery-qdot --test qdot -- --include-ignored --nocapture'

# B3 engram 게이트: V4.1의 NVMe 테이블에서 매핑으로 읽은 행이 같은 오프셋의 pread와
# 바이트 동일이고(선행 읽기를 걸어도 같고), 행 간격이 텐서를 정확히 타일링하는가.
# 정확성 실행이라 임대는 필요 없다 — 속도는 engram-rate가 임대 안에서 잰다.
gate-engram:
    ./tools/box.sh 'bash tools/gate.sh --release -p bloomery-engram --test engram -- --ignored --nocapture'

# engram-rate 바이너리. 측정은 tools/ref/engram-rate.sh가 임대 안에서 돌린다 —
# 이 레시피는 빌드만 한다(러너가 낡은 바이너리를 재는 것을 막는 단계).
build-engram:
    ./tools/box.sh 'cargo build --release -p bloomery-engram --bin engram-rate'

# engram 토큰당 비용 표(기본 팔 여덟 — 캐시 팔은 --ids와 함께 --arms로 부를 때만 돈다). 기계 전역 임대를
# 잡으므로 리드가 조용한 시점에 친다.
measure-engram *ARGS: build-engram
    ./tools/box.sh 'bash tools/ref/engram-rate.sh {{ARGS}}'

# engram-reuse가 읽을 토큰 스트림. 참조 엔진의 토크나이저가 vocab만 읽는다 —
# 측정이 아니라서 임대도, 조용한 기계도 필요 없다(초 단위).
engram-corpus *ARGS:
    ./tools/box.sh 'bash tools/ref/engram-corpus.sh {{ARGS}}'

# 1-4 판정 게이트: 프롬프트 32개의 argmax를 ik와 대조한다. just argmax-ref가 먼저다.
gate-prompts:
    ./tools/box.sh 'bash tools/gate.sh --release -p bloomery-model --test prompts -- --ignored --nocapture'

# ik의 답(프롬프트 32개의 greedy 다음 토큰). 오라클과 달리 파일 하나만 쓰고
# $BLOOMERY_DATA/ref는 건드리지 않는다 — argmax.sh가 끝에서 매니페스트 해시로 확인한다.
build-argmax:
    ./tools/box.sh 'bash tools/ref/build-argmax.sh'

argmax-ref:
    ./tools/box.sh 'bash tools/ref/argmax.sh'

# kv-clear / warmup 재현자. ik가 문맥의 n_eval이 0인 동안 들어온 "BOS 한 토큰" 디코드를
# 워밍업 그래프로 지어 전문가를 전부(64개) 돌린다 — 그 그래프가 쓴 KV가 그 시퀀스 전체를
# 바꾼다. 팔마다 그 술어의 접속사 하나씩만 뒤집는다. 산출물은 스크래치로만 간다.
build-kvclear:
    ./tools/box.sh 'bash tools/ref/build-kvclear.sh'

kvclear-probe *ARGS:
    ./tools/box.sh 'source tools/ref/ref-paths.sh && CUDA_VISIBLE_DEVICES= /root/bloomery-scratch/ikclear/bin/kvclear_probe -m "$MODEL" --prompts tools/ref/prompts.tsv -ngl 0 -c 512 -t 32 {{ARGS}}'

# 1단계 1-1 게이트: 디퀀트 오라클을 빌드해 ggml의 to_float 덤프를 만들고, gguf 크레이트의
# hw 테스트가 그것과 대조한다. hw_ 접두는 박스를 요구한다는 뜻이고 기본 실행에서 빠져 있다.
gate-1-1:
    ./tools/box.sh 'bash tools/ref/build-dequant.sh && "$BLOOMERY_DATA/bin/dequant_ref" && bash tools/gate.sh -p bloomery-gguf -- --ignored --nocapture'

# 커밋 전에 치는 것. 측정은 포함하지 않는다(조용한 기계가 필요하다).
gate: check-recipes check-arch check-comments fmt-check lint build-gpu build-cpu gate-1-1

# B0b V4.1 인벤토리: 분할 GGUF의 헤더만 읽어 텐서 표를 뽑는다(임대 불필요, 텐서 바이트 미접촉).
# 표는 박스의 /tmp에 쓰고 scp로 회수한다 — 박스 작업 트리에 쓰면 다음 box.sh의 rsync --delete가 지운다.
inventory-v41:
    ./tools/box.sh 'cargo run --release -p bloomery-gguf --bin gguf-inventory -- --markdown /tmp/v41-inventory.md /models/DeepSeek-V4.1-Flash-Q3_K_M-engramQ8-tokembdBF16-attnQ8/DeepSeek-V4.1-Flash-Q3_K_M-0000*-of-00009.gguf'
    scp "${BLOOMERY_BOX:-ws}:/tmp/v41-inventory.md" docs/v41-inventory.md

# GPU 헤드(P8의 head 조각): result_norm → lm_head(Q6_K) → argmax를 그래프 하나로 잡아
# 덤프의 마지막 토큰과 대조(정확성 실행, 핀된 헤드 밴드 안).
gate-gpu-head:
    ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin gate_head_gpu && flock -w 1800 /root/bloomery-gate.lock ./target/release/gate_head_gpu'

# bloomery-gpu 라이브러리의 단위 시험(호스트 전용 — 카드를 쓰지 않아 게이트 락을 잡지 않는다).
# 디바이스 크레이트라 cargo oxide test로 돈다; tools/gate.sh --oxide가 상한과 종료 코드를 같이 쥔다.
gate-gpu-lib:
    ./tools/box.sh 'bash tools/gate.sh --oxide -p bloomery-gpu --release --lib'

# ik의 CUDA 답(프롬프트 33개의 다음 토큰): GPU 엔진 종단 게이트의 참조. 카드 선택과 오프로드
# 깊이는 dump.sh와 같다(박스 env의 3090 핀, -ngl 99). ik의 CUDA는 ubatch 하나에 9토큰 이상이
# 들어가면 쓰레기를 낸다(upstream의 MMQ 경로, mainline이 양자화한 파일에서만). 그래서 argmax.sh가 --step-prefill(M=1 경로)로 먹인다 —
# 사정과 되돌리는 법(BLOOMERY_REF_BATCH_PREFILL=1)은 argmax.sh 머리글. 8 GB 모델을 1분 안에
# 올렸다 내리는 정확성 실행이라 임대는 필요 없다. 카드에 다른 컴퓨트 프로세스가 있으면
# 끝나기를 기다린다 — 죽이지 않는다.
argmax-ref-cuda:
    ./tools/box.sh 'BLOOMERY_REF_BACKEND=cuda bash tools/ref/argmax.sh'

# 위의 greedy 연속(프롬프트 다음 토큰부터 32스텝, argmax_ref --gen): gen_ids·gen_margins
# 열을 가진 greedy-ik-cuda-32.tsv를 쓴다. 종단 게이트(gpu-gates::prompts)가 읽는 참조다.
greedy-ref-cuda:
    ./tools/box.sh 'BLOOMERY_REF_BACKEND=cuda BLOOMERY_REF_GEN=32 bash tools/ref/argmax.sh'

# P8b: 조립된 MoE 층(층 1) 스텝을 덤프와 대조 — 어텐션 절반 + 라우터·전문가 여섯·공유 전문가·결합.
# 입력은 오라클의 l_out-0, KV는 오라클의 kv_cache-1 앞 다섯 행. 밴드는 핀하지 않고 표만 찍는다.
gate-gpu-p8b *ARGS:
    ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin gate_p8b && flock -w 1800 /root/bloomery-gate.lock ./target/release/gate_p8b {{ARGS}}'

# PTX 스캔(계측기, 게이트 아님 — 항상 exit 0): 게이트 바이너리가 싣고 있는 디바이스 코드의 엔트리별
# 디포·로컬 왕복·블록 폭 표. 디포를 가진 엔트리가 먼저 나온다. 단언은 gate_p5/gate_p4가 한다 — 머리글 참조.
ptx-scan BIN *ARGS:
    ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin {{BIN}} && bash tools/ptx-scan.sh {{BIN}} {{ARGS}}'
