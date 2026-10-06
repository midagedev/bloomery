# bloomery 작업 목록. 모든 타깃은 맥에서 치고 박스에서 돈다(tools/box.sh가 rsync한다).
# 맥은 arm64라 CPU 커널이 아예 빌드되지 않는다. 로컬 cargo로 게이트를 돌리려 하지 말 것.
#
# 레시피는 절대 `cd ~/repo/bloomery`를 쓰지 않는다. box.sh가 이미 $REMOTE로 들어가고 그 값은
# 워크트리 이름에서 유도된다 — 레시피가 다시 cd하면 워크트리 라운드가 메인 트리를 재게 된다
# (2026-09-19 ffn 라운드가 보고: "gate-ffn은 ~/repo/bloomery로 가는데 이 워크트리는
# ~/repo/bloomery-ffn으로 rsync된다. 둘 다 박스에 있어서 엉뚱한 트리를 시험한다").

# The card check a timed recipe runs first in its box command, before any build (tools/ref/card-precheck.sh).
precheck := 'bash tools/ref/card-precheck.sh'

# serve-qwen38's placement (a the A6000, gate the 3090) and the card it puts
# in box.sh's view.
serve_place := env_var_or_default('SERVE_PLACE', 'a')
serve_card := if serve_place == 'gate' { '3090' } else { 'a6000' }

default:
    @just --list

# `--features gpu`(R26): 피처 뒤에 본체가 숨은 바이너리는 그 피처를 켜야 검사된다 —
# gpu-gates 바이너리의 GPU 본체가 끄면 통째로 안 보인다.
# 빠른 루프: 타입 검사만, 커널은 안 만든다. 의존을 고친 뒤 cargo가 박스 쪽 Cargo.lock을 고쳐 쓰면 lock-back.sh가
# 그것을 이 트리로 가져온다 — box.sh는 한 방향으로만 싣는다.
check:
    ./tools/box.sh 'cargo check --workspace --all-targets --features gpu,bloomery-gpu-gates/deepseek41,bloomery-gpu-gates/vision,bloomery-gpu-gates/glm5next,bloomery-gpu-gates/clef'
    ./tools/lock-back.sh

# check와 같은 이유로 `--features gpu`. 이 피처를 켠 것이 기준 계기다(R26). V4.1 op 게이트의 피처
# `deepseek41`도 켠다 — 그 바이너리들은 이 피처가 있어야 컴파일된다. clippy는 디바이스 코드를 만들지 않으므로
# V4.1 커널의 컴파일러 결함은 여기가 아니라 op 게이트 빌드에서 드러난다.
# lint. 에러 0이 계약이고 경고 수는 RESULTS/AGENTS에 적힌 기준선과 비교한다.
lint:
    ./tools/box.sh 'cargo clippy --workspace --all-targets --features gpu,bloomery-gpu-gates/deepseek41,bloomery-gpu-gates/vision,bloomery-gpu-gates/glm5next,bloomery-gpu-gates/clef'

# fmt는 맥에서 돈다. box.sh의 rsync가 단방향이라 박스에서 포맷하면 결과가 돌아오지
# 않고 다음 명령에 덮여 사라진다(2026-09-19에 그렇게 한 번 날렸다). cargo fmt는 컴파일을
# 하지 않고 파싱만 하므로 arm64 맥에서 정상 동작한다 — AGENTS.md의 "맥에서 게이트 금지"는
# 빌드가 필요한 것에 대한 규칙이고, 판정은 박스의 fmt-check가 한다. rustfmt는 rust-toolchain.toml이 고정한
# 나이틀리의 것이다 — PATH의 Homebrew rustfmt는 다른 판이다. 명령은 fmt-check의 것에서 `-- --check`를 뺀 것이고
# 툴체인을 고르는 것은 tools/mac-check.sh다.
fmt:
    ./tools/mac-check.sh fmt

fmt-check:
    ./tools/box.sh 'cargo fmt --all -- --check'

# 정적 계층을 맥에서 돈다: check·lint 레시피의 박스 명령에 `--target x86_64-unknown-linux-gnu`를 붙인 교차
# 검사(링크 없음), fmt-check의 명령은 고정 툴체인의 rustfmt로. 테스트와 게이트는 여기서 돌지 않는다.
# 환경은 레포 밖 ~/opt/bloomery-mac-env.sh 하나다 — 준비물과 만드는 법은 tools/mac-check.sh 머리말.
# check 다음에 mac-combos를 돈다: 박스 창 전에 라운드가 도는 이 한 줄이 게이트 레시피의 빌드 모양까지 본다.
mac-check:
    ./tools/mac-check.sh check
    ./tools/mac-check.sh combos

# check 레시피는 피처 한 벌만 빌드하므로 `--features gpu` 단독이나 피처 없는 lib 테스트 빌드의 컴파일 오류는
# 여기서만 잡힌다. 목록의 주인은 tools/recipes.py combos다.
# 게이트·빌드 레시피가 컴파일하는 (패키지, 모드, 피처) 모양마다 `cargo check`를 같은 교차 검사로 한 번씩 돈다.
mac-combos:
    ./tools/mac-check.sh combos

# lint의 `^warning:` 수를 AGENTS.md의 기준값과 비교한다. 넘으면 빨강.
mac-lint:
    ./tools/mac-check.sh lint

# fmt-check를 맥에서, 고정 나이틀리의 rustfmt로 돈다.
mac-fmt-check:
    ./tools/mac-check.sh fmt-check

# 순수 크레이트(tools/recipes.py pure-crates가 크레이트 그래프와 소스로 고른 것)의 `cargo test -p`를 맥에서 네이티브로 돈다.
# 개발 루프의 증거이고, 착륙의 증거는 박스의 기록이다.
mac-test:
    ./tools/mac-check.sh test

# 레시피 자체의 점검(맥, grep뿐). 게이트 줄의 `||`는 종료 코드를 삼킨다 — tools/check-recipes.sh 머리말.
check-recipes:
    ./tools/check-recipes.sh

# BASE(기본 main)나 A..B 사이에 바뀐 파일을 입력으로 읽는 gate-* 레시피 목록 — 맥에서, 빌드 없이, 아무것도 돌리지 않는다(tools/affected-gates.sh).
# `--narrow --scan BASE_LOG NEW_LOG …` (ptx-scan logs of the base and the change): the gates that run the changed host path.
affected BASE='main' *ARGS:
    ./tools/affected-gates.sh {{BASE}} {{ARGS}}

# 호스트 rustflags의 두 소유자(.cargo/config.toml, .cargo/cuda-oxide.toml)가 같은지 — 맥에서, 빌드 없음.
check-rustflags:
    ./tools/check-rustflags.sh

# 아키텍처 축 점검(맥, grep뿐) — docs/arch-split.md 「검사」. 넷 다 엄격하다.
check-arch:
    bash tools/check-arch.sh

# 주석 규약(AGENTS.md Conventions): 엔진 크레이트 src/ 주석에 이슈 번호·날짜 금지, 예외는 `PIN(날짜):`.
check-comments:
    ./tools/check-comments.sh

# 레버 읽기 점검(맥, 텍스트뿐): crates/*/src에서 레버 레지스트리 밖에서 환경 변수를 읽는 곳은 그 읽기를 옮길
# 라운드와 함께 tools/levers-direct.txt에 한 줄로 올라 있어야 한다 — 규칙은 tools/check-levers.sh 머리말.
check-levers:
    ./tools/check-levers.sh

# The per-crate unsafe ratchet (Mac, grep only, no build): the unsafe lines under crates/<dir>/{src,tests,benches}
# against the pins in tools/unsafe-ratchet.txt; the counting rule and exit codes are tools/check-unsafe.sh's header.
check-unsafe:
    bash tools/check-unsafe.sh

# The one-bundle-load ratchet (Mac, grep only, no build): a `#[cuda_module]` family's raw `load` outside the
# shared_module! slot — the rule and exit codes are tools/check-loads.sh's header.
check-loads:
    ./tools/check-loads.sh

# `[group('v41-load')]`: 바이너리가 V4.1 전체를 적재하는(body::open — 호스트 세트 populate) 레시피. tools/gate-batch.sh가 한 레인(A)에 차례로 세운다.
# GPU 게이트 바이너리는 tools/gpu-gate.sh 아래에서 카드마다 하나인 게이트 락에 줄을 선다(빌드는 병렬, 실행만 직렬).
# 게이트 하나가 모델을 12 GB 올리므로 둘이 같은 카드에 겹치면 OOM이다. 시간 러너의 임대(/root/bloomery-cpu.lock)와는
# 다른 락이다.
#
# GPU 커널 게이트 P1–P3: 트랙마다 자기 바이너리 하나(crates/gpu-gates/src/bin/gate_pN.rs).
# 참조는 bloomery_gpu_gates(gguf 디퀀트 + f64 내적), 밴드는 KERNEL_BAND = 1e-2. 정확성 실행이고 측정이 아니다.
gate-gpu-p1:
    ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin gate_p1 && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_p1'

gate-gpu-p2:
    ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin gate_p2 && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_p2'

gate-gpu-p3:
    ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin gate_p3 && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_p3'

gate-gpu-p4:
    ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin gate_p4 && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_p4'

gate-gpu-p5:
    ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin gate_p5 && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_p5'

gate-gpu-p6:
    ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin gate_p6 && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_p6'

gate-gpu-p9:
    ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin gate_p9 && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_p9'

# B6: q4k_gemv_sel — 슬롯마다 그 expert 하나로 돈 q4k_gemv와 비트 동일, 범위 밖 id는 슬롯을 건드리지 않는다.
gate-gpu-q4k-sel:
    ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin gate_q4k_sel && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_q4k_sel'

# 모델 없는 K-quant expert 계열(bloomery_gpu::kquant): Q5_K `_sel` down과 gate·up(act 런타임 인자)을 합성 스택에서
# f64 참조의 밴드, 슬롯·m열·폴트·act 절로 보고, 같은 워크의 Q4_K 형제가 q4k_gemv_sel과 비트 동일한지 본다. 모델 파일 없음.
gate-gpu-kquant:
    ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin gate_kquant && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_kquant'

# 두 단계 후보 선택(bloomery_gpu::cand, V4.1 후보 마스크의 커널 넷)을 모델 없이 합성 점수로 호스트 규칙과 비트 대조한다: 블록 키,
# 참조 select_candidate_blocks의 선택(핀, 동점은 낮은 블록, ±0 동점), 경계 이하 무작업, 압축·히스토그램·카운트 뷰, remap, 폴트, 캡처 재생.
gate-gpu-cand:
    ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin gate_cand && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_cand'

# Qwen3.8의 gated-residual hyper-connection(bloomery_gpu::hc_gated)을 모델 없이 파일 모양(4×2560, rank 320)의 합성
# 입력으로 본다: f64 규칙의 밴드, 단계마다 카드 규칙과 비트 동일, 8열과 1열, combine·init, 이름 붙은 거부, 폴트, 로컬 depot.
gate-gpu-hc-gated:
    ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin gate_hc_gated && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_hc_gated'

# GLM-5.3-Flash 라우터(288개 중 8개, sigmoid, ×2.5 — `router::glm5next`)를 모델 없이 합성 입력으로 호스트 규칙과 맞춘다.
# 규칙마다 절이 하나다: logits·ids·weights·batch는 비트, scores는 sigmoid 밴드, ties·guard·fault·graph는 명시한 값.
gate-gpu-glm-router:
    ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features deepseek41 --release --bin gate_glm5next_router && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_glm5next_router'

# IQ2_XS·IQ3_XXS·IQ4_XS·Q2_K 행 코어(bloomery_gpu::iq): ref-synth 합성 행(gate-1-1이 덤프)을 q8_1 8열과 곱해 호스트 규칙과
# 비트 동일한지, ggml 디퀀트 × f32 기준 대비 유도 한계와 핀 안인지 본다. K = 4096과 꼬리가 남는 K = 2304 두 형상.
gate-gpu-iq:
    ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin gate_iq && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_iq'

# V4.1 하이퍼커넥션(B4): 사슬 커널(RMS·분할 K hc_fn gemv·HC_PRE), HC_POST와 접기를 우리 규칙과 V4.1 ik 덤프에 서브층마다
# 대조한다. engram 층(1·14)의 분리 짝(HC_POST 단독 → 접기 단독)이 융합 런치와 비트 동일한지도 본다.
gate-gpu-ds41-hc:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features deepseek41 --release --bin gate_deepseek41_hc && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_deepseek41_hc'

# 종단 게이트: 27층 + lm_head + argmax 전체 사슬을 ik CUDA greedy(33프롬프트 × 32스텝)와 대조하고,
# 그래프 재생이 즉시 실행과 토큰 단위로 같은지 본다(정확성 실행, 측정 아님). 이름이 pN이 아닌 이유는
# gate_e2e.rs 머리글 — P9·P10은 커널 꾸러미로 이미 쓰이고 있다.
gate-gpu-e2e *ARGS:
    ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin gate_e2e && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_e2e {{ARGS}}'

# 하이브리드 MoE 경계 게이트: V2-Lite를 n_l 32와 0(n_l 밖 expert는 호스트)으로 올려 전부 카드인 모델과 대조한다.
# 캡처 노드 수, eager = 재생, 층별 카드 슬롯 비트 동일·호스트 합 밴드, argmax 뒤집힘 밴드, 겹침 레버는 순서만 바꾸는지.
gate-gpu-hybrid *ARGS:
    ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin gate_hybrid && bash tools/gpu-gate.sh gate_hybrid {{ARGS}}'

# 적응형 residency 기계 게이트(모델 파일 없음, 합성 스택): 같은 이력 = 같은 값(복사 시점과 무관), 정적 재배치와 비트 동일,
# 늦은 복사는 스트림이 기다림, 호스트·카드 맵이 한 id를 두 번 또는 0번 서비스하지 않음, reset은 시드로, 호스트 비상주
# 희생자는 이름으로 거부, 고정 시드는 움직이지 않음(adaptres §4 c1–c5·c7).
gate-gpu-swap *ARGS:
    ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin gate_swap && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_swap {{ARGS}}'

# 양자화 모델의 정확한 수학(가중치는 정확히 역양자화, 활성·어텐션은 f64, q8·f16 반올림 없음)으로 교사 강제 자의
# 한 프롬프트를 푼다: 스텝마다 정확 top1·마진·ik 토큰 격차. 두 엔진(우리, ik)이 갈리는 자리의 심판이다.
# CPU 64스레드를 쓰므로 기계 전역 임대를 잡는다. 예: `just exact-ref 12 --steps 23`, `--kv f16`은 캐시 f16 반올림 팔.
exact-ref ID *ARGS:
    ./tools/box.sh '{{precheck}} docs/cards/exact-ref.card && cargo build --release -p bloomery-gpu-gates --bin exact_ref && bash tools/ref/lease-hold.sh --card docs/cards/exact-ref.card -- ./target/release/exact_ref --prompt-id {{ID}} {{ARGS}}'

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
    {{precheck}} docs/cards/exact-ref.card
    cargo build --release -p bloomery-gpu-gates --bin exact_ref
    out="$BLOOMERY_DATA/exact-forced-32.tsv"
    forced="$BLOOMERY_DATA/greedy-ik-cuda-32.tsv"
    ids=$(grep -v -e '^#' -e '^$' tools/ref/prompts.tsv | cut -f1)
    parts="$out.parts.$$"
    mkdir -p "$parts"
    cp target/release/exact_ref "$parts/exact_ref"
    for id in $ids; do
      bash tools/ref/lease-hold.sh --card docs/cards/exact-ref.card -- "$parts/exact_ref" --prompt-id "$id" --emit "$parts/p$id.tsv" > "$parts/p$id.log"
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

# 한 강제 위치에서 우리 GPU 기본 패스(텐서 코어 flash)의 층·탭별 상대 거리를 세 심판 팔에 대해 한 번에 찍는다: f64(정확),
# ours(우리 활성 양자화 규칙만 흉내 낸 f64 사슬 — K-quant 입력 128값 블록), ik(ik 규칙 — 전부 32값 블록).
# exact_ref가 세 팔을 CPU 임대 아래 차례로 덤프하고(프롬프트당 팔 하나 약 1.5분), forced_probe가 그 셋과 대조한다.
# 덤프와 exact_ref 로그는 박스 target/exact-taps/ID-STEP/. 예: `just exact-taps 12 23`.
exact-taps ID STEP:
    ./tools/box.sh '{{precheck}} docs/cards/exact-ref.card && cargo build --release -p bloomery-gpu-gates --bin exact_ref && cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin forced_probe && D=target/exact-taps/{{ID}}-{{STEP}} && mkdir -p $D && bash tools/ref/lease-hold.sh --card docs/cards/exact-ref.card -- ./target/release/exact_ref --prompt-id {{ID}} --steps {{STEP}} --act f64 --dump $D/f64 > $D/f64.log && bash tools/ref/lease-hold.sh --card docs/cards/exact-ref.card -- ./target/release/exact_ref --prompt-id {{ID}} --steps {{STEP}} --act ours --dump $D/ours > $D/ours.log && bash tools/ref/lease-hold.sh --card docs/cards/exact-ref.card -- ./target/release/exact_ref --prompt-id {{ID}} --steps {{STEP}} --act ik --dump $D/ik > $D/ik.log && grep -H "top5" $D/f64.log $D/ours.log $D/ik.log && bash tools/gpu-gate.sh forced_probe --prompt-id {{ID}} --step {{STEP}} --dump $D/engine --against $D/f64,$D/ours,$D/ik'

# 얇은 끝-끝 디코드 CLI(greedy, 토큰 하나씩, 프리필 커널 없음). `--time` 없이 토큰만 찍는 것은
# 평범한 실행이고, `--time`은 측정이라 임대가 필요하다 — time-gpu-generate가 그쪽이다.
generate *ARGS:
    ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin generate && ./target/release/generate {{ARGS}}'

# generate의 스텝당 ms(리드 전용): 기계 전역 임대 아래, 증인 블록 전후. 기록이 되는 것은 graph 모드의
# 푸터이고 eager 모드는 호스트 제출 경로의 값이다.
time-gpu-generate *ARGS:
    ./tools/box.sh '{{precheck}} && cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin generate && bash tools/ref/time-gate.sh generate {{ARGS}} --time'

# P0b: 블록 0 FFN 융합 스파이크 — 융합 4런치가 op 8런치와 비트 동일한지(정확성 실행).
gate-gpu-p0b *ARGS:
    ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin gate_p0b && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_p0b {{ARGS}}'

# P8a 블록 0 스텝(ctx_max 64에서 21노드 — fuse1 뒤; 한 구간보다 큰 캐시에서는 flash_merge가 붙어 22)의 재생 µs — 임대·증인, 리드 전용.
time-gpu-p8:
    ./tools/box.sh '{{precheck}} && cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin gate_p8 && bash tools/ref/time-gate.sh gate_p8'

# P8 op별 프로파일(리드 전용): 스텝을 op 단위로 동기 분해한 µs 표(ops=그래프 노드 수와 동일) + sync_floor
# 보정 + refresh_params 호스트 시간 + 같은 프로세스의 그래프 재생 µs + 어텐션/FFN 분할. 임대·증인은 time-gate.sh 소유.
prof-gpu-p8:
    ./tools/box.sh '{{precheck}} && cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin gate_p8 && bash tools/ref/time-gate.sh gate_p8 --profile'

# 커널 런치당 실제 비용(리드 전용): op별 표가 답할 수 없는 것 — 한 행의 net_us가 디바이스 시간인지
# 프로파일 자신의 제출 경로인지 — 를 두 갈래로 가른다. eager는 N런치 뒤 동기화 하나(호스트 제출과
# 디바이스 시간 중 큰 쪽), graph는 같은 N런치를 그래프 하나로 캡처해 재생(스텝의 그래프 노드가 실제로
# 무는 값). 빈 커널 touch가 둘의 바닥이고, f32 gemv는 행 수 셋에서 같은 프로세스로 잰다. 이어서 그룹 int8 GEMM의
# 팔들을 `gate_gemm --bench-kernels`로 잰다(ARGS, 예를 들어 `--bench-arm`은 이쪽으로 간다). 임대·증인은 time-gate.sh 소유.
bench-gpu-kernels *ARGS:
    ./tools/box.sh '{{precheck}} && cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin gate_p8 --bin gate_gemm && bash tools/ref/time-gate.sh gate_p8 --bench-kernels && bash tools/ref/time-gate.sh gate_gemm --bench-kernels {{ARGS}}'

# The grouped GEMM's counters (ncu, A6000, under the lease, lead-only): gate_gemm --bench-kernels --bench-arm ARM (default
# gemm_q4k_moe_t4096: 128 experts of 768 x 2048 Q4_K, top-8, T = 4096) runs that one arm, and ncu takes 16 of its eager
# gemm_q4k launches after the 64-launch warm-up burst; the form and its levers are in the header of tools/ref/ncu-gpu.sh.
# Expected on main [derived, docs/research/q3next-design-report.md sections 3 and 6; not numbers of record]: L1TEX
# 55-65 % and the top unit, about 1.29e8 LSU data-pipe wavefronts a launch, issue-active 35-40 %, tensor 22-26 %, DRAM
# 10-15 %. Under BLOOMERY_BOX_ENV=BLOOMERY_DRY=1 nothing is built and the runner prints its command line.
ncu-gpu-gemm ARM='gemm_q4k_moe_t4096':
    ./tools/box.sh '{{precheck}} && if [ -z "${BLOOMERY_DRY:-}" ]; then cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin gate_gemm; fi && BLOOMERY_NCU_FORM=gemm BLOOMERY_NCU_GEMM_ARM={{ARM}} bash tools/ref/ncu-gpu.sh'

# The V4.1 prompt projections' counters (ncu, A6000, under the lease, lead-only): one launch each of the joined qkv, q_b,
# wo_a heads and wo_b at m = 8 in the middle full chunk of a layer >= 2 of a P-token prompt (default 512). The launch skip
# comes from the newest nsys-gpu-ds41-prefill trace of the same P (run that first; it must be of this binary),
# and the profiled launches' names, grids, blocks and column counts are checked before the summary: per launch the
# block-step cycles and each unit's demand in cycles. The header of tools/ref/ncu-gpu.sh has the form. Under
# BLOOMERY_BOX_ENV=BLOOMERY_DRY=1 nothing is built and the runner prints its command line.
ncu-gpu-ds41-pp P='512':
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh '{{precheck}} && if [ -z "${BLOOMERY_DRY:-}" ]; then cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features deepseek41 --release --bin generate_ds41; fi && BLOOMERY_NCU_FORM=ds41pp BLOOMERY_NCU_PROMPT={{P}} bash tools/ref/ncu-gpu.sh'

# The Qwen3 prompt's counters (ncu, A6000, under the lease, lead-only): one KERNEL launch (gqa_prefill_flash, the one the
# form's table holds) in layer LAYER of a P-token prompt, in the process of depth-qwen3moe.sh's `<P>` arm (-n 1), at the
# card's own clock. The launch skip is derived from the code (tools/ref/q3pp.py plan: 24 at P = 4096) and the profiled
# launch is proved before the summary: the run's load and plan lines, the name, grid (2,048 x 128 at P = 4096) and the
# tensor pipe's HMMA count, which depends on the ubatch's rows and position (34,078,720). Then the clock, the step's
# cycles and each unit's demand beside them, and the stall composition. BLOOMERY_NCU_BIN=<absolute path> profiles
# another tree's generate_qwen3moe instead (nothing is built). Expected on main for gqa_prefill_flash at P = 4096 [derived,
# docs/research/q3tail-design-report.md 3.2 and 4; not numbers of record]: SM clock 1.46-1.55 GHz, tensor pipe 56-60 %,
# issue 0.36-0.40 a scheduler, 3,400-3,620 SM cycles a block-step. The header of tools/ref/ncu-gpu.sh has the form.
# Under BLOOMERY_BOX_ENV=BLOOMERY_DRY=1 nothing is built and the runner prints the derivation and its command line.
ncu-gpu-qwen3-pp P='4096' LAYER='24' KERNEL='gqa_prefill_flash':
    BLOOMERY_MODEL=qwen3moe ./tools/box.sh '{{precheck}} && if [ -z "${BLOOMERY_DRY:-}" ] && [ -z "${BLOOMERY_NCU_BIN:-}" ]; then cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin generate_qwen3moe; fi && BLOOMERY_NCU_FORM=q3pp BLOOMERY_NCU_PROMPT={{P}} BLOOMERY_NCU_LAYER={{LAYER}} BLOOMERY_NCU_KERNEL={{KERNEL}} bash tools/ref/ncu-gpu.sh'

# 캡처된 그래프의 노드당 값, 비용 모형의 `c_node`(리드 전용): cnode_probe --time이 빈 `touch` 노드 784개와 504개짜리
# 그래프를 잡아, 그래프마다 노드 수와 재생 한 번의 기록을 먼저 확인한 뒤 `touch_us_min`·`touch_us_mean`·
# `touch_us_per_node`를 찍는다. 정확성만 볼 때는 `bash tools/gpu-gate.sh cnode_probe --check`. 임대·증인·A6000 고정은
# time-gate.sh가 쥔다.
time-gpu-cnode:
    ./tools/box.sh '{{precheck}} && cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin cnode_probe && bash tools/ref/time-gate.sh cnode_probe'

# 호스트 → 카드 링크(리드 전용): `nvidia-smi topo -m`과 카드마다 PCI 주소·sysfs 경로·링크를 찍고, A6000에서 1 GiB pinned와
# pageable(facts.md의 교정값), 1.6 GB pinned 복사 48회 연속(복사마다 GB/s의 퍼짐, 복사 중에 읽은 링크)을 잰다.
# BLOOMERY_CARD=both면 3090 단독과 두 카드 동시 복사(카드별 율과 합)도 잰다. 임대·증인·Xid 수는 tools/ref/h2d-pcie.sh가
# 쥐고, 카드는 docs/cards/pcie-a6000.card다. 예: `BLOOMERY_CARD=both BLOOMERY_BOX_ENV='BLOOMERY_LEASE_CARD=docs/cards/pcie-a6000.card' just time-gpu-h2d`.
time-gpu-h2d:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh '{{precheck}} && cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin h2d_probe && bash tools/ref/h2d-pcie.sh'

# 호스트 → 카드 복사가 호스트 티어의 DRAM 읽기에서 뺏는 몫(리드 전용): bench_v41_host --time을 단독으로, h2d_probe의
# pageable-loop 옆에서, staged-loop(스트리밍 설계의 pinned 링) 옆에서 라운드마다 번갈아 돌린다. 복사는 벤치가 비워 둔 코어에
# 고정하고, 라운드마다 두 율과 k = (U_alone − U_with) / C_with를 찍는다. ARGS는 tools/ref/dma-dram-share.sh로 간다(--arms는
# 벤치의 팔을 그대로 넘기고 --rounds, --seconds, --chunk, --fill-threads …). 카드는 docs/cards/dma-dram-share.card다.
# 예: `BLOOMERY_BOX_ENV='BLOOMERY_LEASE_CARD=docs/cards/dma-dram-share.card' just time-dma-dram --arms engine:6`.
time-dma-dram *ARGS:
    BLOOMERY_MODEL=${BLOOMERY_MODEL:-deepseek41} ./tools/box.sh '{{precheck}} && cargo build --release -p bloomery-model --bin bench_v41_host && cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin h2d_probe && bash tools/ref/dma-dram-share.sh {{ARGS}}'

# 유휴 상태 A/B(리드 전용): 가장 깊은 cpuidle 상태(이 박스의 C2, 깨는 데 18 µs)를 모든 CPU에서 켠 팔과 끈 팔을
# 한 임대 안에서 번갈아 잰다. 명령은 A6000에 고정돼 라운드마다 팔 순서를 바꿔 돌고, 원래 값은 모든 종료 경로에서
# 되돌린다. 예: `just cstate-ab 6 env BLOOMERY_HYBRID_NL=32 target/release/generate -n 64 --time`.
cstate-ab ROUNDS *CMD:
    ./tools/box.sh '{{precheck}} && cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin generate && bash tools/ref/cstate-ab.sh {{ROUNDS}} {{CMD}}'

# P8: 조립된 디코드 스텝(블록 0부터)을 덤프와 대조 — 엔진 자신의 탭.
gate-gpu-p8 *ARGS:
    ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin gate_p8 && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_p8 {{ARGS}}'

# MoE 융합(P8 준비): 전문가 여섯의 gate·up·swiglu 한 런치 + 결합 한 런치가 op 경로와 비트 동일한지(정확성 실행).
gate-gpu-moe:
    ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin gate_moe_fused && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_moe_fused'

# V4.1 라우터와 전문가(B4 J·K): 라우터, 라우팅·공유 전문가의 클램프 SwiGLU, combine을 우리 규칙과 V4.1 ik 덤프
# (5토큰 세트와 디코드 스텝 세트)에 층마다 대조한다. 클램프는 합성 입력으로 한계 너머까지 따로 본다.
gate-gpu-ds41-moe:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features deepseek41 --release --bin gate_deepseek41_moe && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_deepseek41_moe'

# V4.1 MoE 조각(B5 1단계, chain G1): step4·d1n 세트의 40층 전부에서 라우터·카드 expert·공유 expert·호스트 티어 합류·combine·
# HC_POST(+접기)를 한 프로세스 안에서 돈다 — op 경로와 비트 동일, 라우터 id는 근접 동률 빼고 정확, combine·스트림·접기는 op 밴드에서
# 옮겨 온 반올림 한계 안의 예측 차이로 덤프와 대조, 재생 = eager(층마다 호스트가 서비스), 종류별 노드 수.
gate-gpu-ds41-chain-ffn:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features deepseek41 --release --bin gate_deepseek41_chain_ffn && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_deepseek41_chain_ffn'

# P7 뒤 절반: 블록·종단 게이트 하네스의 자기 검증(호스트 전용 — 두 오라클 사이의 알려진 거리를 재현해야 한다).
gate-gpu-block:
    ./tools/box.sh 'cargo build --release -p bloomery-gpu-gates --bin gate_block && bash tools/host-gate.sh gate_block'

# P10: 가중치 상주 — 모델 파일의 모든 텐서를 커널이 먹는 디바이스 형식으로 올린다(정확성 실행, 측정 아님).
gate-gpu-p10:
    ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin gate_p10 && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_p10'

# V4.1 적재 게이트 ②: 배치 계획이 카드마다 두는 세그먼트를, 이름으로 찾은 카드에 올리고 계획과 바이트 단위로 대조한다.
# 할당기 반올림도 계획의 항과 바이트까지 같아야 한다. 정확성 실행이지 측정이 아니다. 계획 a는 --plan a.
# 착륙 묶음이 고르지 않는 opt-in이다(gate-* 레시피가 아니라 `just affected`에 잡히지 않는다). 호스트 세트의 populate·lock
# 경로나 두 카드 staging이 바뀌면 이름으로 돌린다 — 계획 (b)가 트리에서 유일한 두 카드 적재다. `--plan bp`는 계획 (b′)
# (A6000은 계획 (a), 3090은 전문가 층)이고, 드래프트 예약을 위해 프로필의 DSPARK_MODEL을 `--draft`로 넘긴다.
[group('solo')]
[group('v41-load')]
stage-gpu-load-v41 *ARGS='--plan b':
    BLOOMERY_MODEL=deepseek41 BLOOMERY_CARD=both ./tools/box.sh 'export BLOOMERY_GATE_V41_LOAD=1 && cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin gate_load_v41 && __d=$(. tools/ref/ref-paths.sh && printf %s "$DSPARK_MODEL") && case " {{ARGS}} " in *" --plan bp "*) set -- --draft "$__d" ;; *) set -- ;; esac && bash tools/gpu-gate.sh gate_load_v41 {{ARGS}} "$@"'

# 같은 게이트 ②의 호스트 절반: 계획 (b)의 호스트 세그먼트 전부(유도 약 196 GB)를 샤드 매핑째 잠근다. 리드 전용이고,
# RAM을 크게 쓰는 트랙이 없을 때만 돈다.
# stage-gpu-load-v41처럼 opt-in이고, 같은 변경에서 이름으로 돌린다.
[group('solo')]
[group('v41-load')]
stage-gpu-load-v41-lock:
    BLOOMERY_MODEL=deepseek41 BLOOMERY_CARD=both ./tools/box.sh 'export BLOOMERY_GATE_V41_LOAD=1 && cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin gate_load_v41 && BLOOMERY_HOST_LOCK=1 bash tools/gpu-gate.sh gate_load_v41 --plan b --lock'

# qdot 커널률(리드 전용): ik 자신의 x4 커널과 Rust qdot-rate를 같은 CPU 임대 안에서 한 코어에 고정해 번갈아 돈다.
# ik 하네스는 build-ref가 짓는다. 인자는 라운드 수(기본 3).
measure-qdot-rate *ARGS:
    ./tools/box.sh '{{precheck}} && cargo build --release -p bloomery-qdot --bin qdot-rate && bash tools/ref/qdot-rate.sh {{ARGS}}'

# V4.1 호스트 expert 다리의 벤치(bench_v41_host) — 정확성 실행이다. 실제 V4.1 파일을 mmap해 층마다 엔진의 디스패치
# (gate+up 묶음 하나, swiglu를 품은 down 하나)를 돌리고, 층 여덟의 expert마다 gate·up·swiglu·down 여섯 행을 같은 바이트의
# f64 참조와 대조한다(밴드 1e-5). 층마다 샤드 지도와 작업 집합의 상주율도 찍는다. 모델은 Mac 쪽 BLOOMERY_MODEL이고
# 없으면 deepseek41이다. BLOOMERY_MODEL=qwen4exp면 Qwen3.8의 routed expert를 같은 벤치로 연다.
bench-cpu-v41-host-check:
    BLOOMERY_MODEL=${BLOOMERY_MODEL:-deepseek41} ./tools/box.sh 'cargo build --release -p bloomery-model --bin bench_v41_host && bash tools/host-gate.sh bench_v41_host --check'

# 같은 벤치의 시간(리드 전용): 스레드 8·16·24·30·32마다 팔 넷(엔진 모양 n_host 6·5·3, 행렬별 디스패치 6)의 토큰당 ms·GB/s,
# 디스패치 수, 상주율. CPU 임대·증인·낡은 바이너리 거부는 host-rate.sh가 쥔다. 조용한 틈에 돈다 — 빌드가 도는 동안 잰
# 수는 증인이 받지 않는다. 모델은 위 레시피와 같이 BLOOMERY_MODEL이고 없으면 deepseek41이다.
time-cpu-v41-host *ARGS:
    BLOOMERY_MODEL=${BLOOMERY_MODEL:-deepseek41} ./tools/box.sh '{{precheck}} && cargo build --release -p bloomery-model --bin bench_v41_host && bash tools/ref/host-rate.sh {{ARGS}}'

# 2026-09-20 사고(q_nope2 무한루크가 gate-mt를 매달아 병렬 에이전트 둘을 '무활동'으로 죽임)의
# 보강. 이 트랙 원격 디렉터리 아래 실행 파일을 물고 있는 고아 프로세스를 찾아 죽인다.
# 고르는 축은 /proc/<pid>/exe이지 cmdline이 아니다 — `pgrep -f "<dir>/target"`은 그 패턴을
# 자기 argv에 든 셸(스캔하는 셸 자신, ssh 핸들러)도 같이 고른다. 자신과 조상 pid는 접두
# 비교 전에 제외한다. 죽이는 것은 TERM → 5초 → KILL. 목록만 보려면 `just box-gc --dry-run`.
# 이 트랙 원격 디렉터리 아래 실행 파일을 문 고아 프로세스를 죽인다. 병렬 트랙 시작·끝에 한 번씩.
# 시팅의 가드를 지나간다(BLOOMERY_BOX_READONLY=1): 아무것도 빌드하지 않고, 찾는 고아가 임대를 쥔 그 프로세스일 수 있다.
# 읽기는 아무것도 동기화하지 않으므로 스크립트를 stdin으로 보낸다: 이 트리의 box-gc.sh가 원격 디렉터리에서 있는 그대로 돈다.
# 한 번도 동기화하지 않은 트랙에는 원격 디렉터리가 없다(box.sh의 66). 거기서 돈 것이 없으니 거둘 것도 없다.
box-gc *ARGS='--kill':
    #!/usr/bin/env bash
    set -uo pipefail
    BLOOMERY_BOX_READONLY=1 ./tools/box.sh 'bash -s -- {{ARGS}}' < tools/box-gc.sh
    rc=$?
    if [ "$rc" = 66 ]; then echo "box-gc: this track has no remote directory on the box (box.sh rc 66): nothing ran there, nothing to collect"; rc=0; fi
    exit "$rc"

# 박스에 남은 트랙 디렉터리를 로컬 워크트리와 대조한다. 인자 없이 목록, `just box-tracks --remove NAME…`로 준 이름만 삭제(그때도 stale일 때).
box-tracks *ARGS:
    ./tools/box-tracks.sh {{ARGS}}

# 이 Mac의 워크트리마다 크기와 착륙 상태를 보인다. 인자 없이 목록, `just mac-disk clean|retire NAME…`로 준 이름만 정리(retire는 LANDED·PRUNABLE일 때만).
mac-disk *ARGS:
    python3 tools/mac-disk.py {{ARGS}}

# 1-4의 첫 tok/s. 같은 임대 안에서 ik를 같은 파일·같은 조건으로 한 번 더 잰다.
# The prebuilt Linux release (tools/release/build.sh's header: the checks it runs on the binaries before it packs):
# built on the box from a committed tree, with this commit as the binaries' --version. The tarball lands in the box
# remote dir's target/release-dist/.
release-build VERSION:
    test -z "$(git status --porcelain --untracked-files=no)" || { echo "release-build: the tree has uncommitted changes" >&2; exit 1; }
    ./tools/box.sh "bash tools/release/build.sh {{VERSION}} $(git rev-parse --short=12 HEAD)"

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
# BLOOMERY_AB_DEPTH=d 는 깊이 d 의 프롬프트를 먼저 프리필한다(러너 머리말) — 어텐션 레버는 깊은 행에서만 보인다.
ab-decode *DIRS: build-decode
    #!/usr/bin/env bash
    set -euo pipefail
    names=
    for d in {{DIRS}}; do names="$names $(basename "$d")"; done
    ./tools/box.sh "${BLOOMERY_AB_ROUNDS:+BLOOMERY_AB_ROUNDS=$BLOOMERY_AB_ROUNDS} ${BLOOMERY_AB_DEPTH:+BLOOMERY_AB_DEPTH=$BLOOMERY_AB_DEPTH} ${BLOOMERY_AB_ENVS:+BLOOMERY_AB_ENVS='$BLOOMERY_AB_ENVS'} ${BLOOMERY_AB_IK:+BLOOMERY_AB_IK=$BLOOMERY_AB_IK} bash tools/ref/ab-decode.sh$names"

# The Mac shell's BLOOMERY_PROFILE_DEPTH and BLOOMERY_PROFILE_LEVELS ride in front of the remote command, as in
# ab-decode; lever env goes through BLOOMERY_BOX_ENV:
#   BLOOMERY_BOX_ENV="BLOOMERY_FLASH_SEGMENTS=8" BLOOMERY_PROFILE_DEPTH=1024 just measure-profile
measure-profile: build-decode
    ./tools/box.sh "${BLOOMERY_PROFILE_DEPTH:+BLOOMERY_PROFILE_DEPTH=$BLOOMERY_PROFILE_DEPTH} ${BLOOMERY_PROFILE_LEVELS:+BLOOMERY_PROFILE_LEVELS='$BLOOMERY_PROFILE_LEVELS'} bash tools/ref/profile-measure.sh"

# 디코드 스텝의 스레드별 perf 표(리드 전용): 임대·증인 아래, 프리필이 끝난 뒤 pid에 붙어 cpu-clock으로 뜬다.
# `perf report --tid`는 메인 스레드의 dot 커널을 통째로 빼고 스핀만 보여 줬으므로(perf 6.8) `perf script`로 집계한다.
# PERF_SECS(기본 2.0)·PERF_TOP·PERF_ANNOTATE·PERF_CALLERS·PERF_KEEP.
perf-decode *ARGS: build-decode
    ./tools/box.sh 'bash tools/ref/perf-decode.sh {{ARGS}}'

# 참조 하네스(ggml에 링크하는 C++). 진실값과 기준 속도의 출처다.
# build-qdot-ref.sh는 ik의 커널 테이블까지 링크하는 x4 하네스 열을 짓고 참조 하네스 다섯을
# **실행까지** 한다 — gate-qdot의 hw 테스트 일곱이 읽는 $BLOOMERY_DATA/ref/의 덤프
# (*-ik-dot.txt 다섯과 q5_K의 to_float 행)가 그 산출물이다. q5_K는 V2-Lite에 없어서 V4.1 첫
# 샤드를 읽는다. 나머지 다섯(*_rate)은 빌드만 한다: 실행은 측정이다.
build-ref:
    ./tools/box.sh 'bash tools/ref/build-qdot-ref.sh'

# The V4.1 file's r8 sidecar (model::r8file): builds r8conv, then tools/ref/r8-sidecar.sh converts and
# verifies it under the CPU lease — 155.7 GB written beside the source on /models. A tool run, not a
# gate; the user decides when. ARGS: `verify` checks an existing sidecar again without converting.
r8-sidecar *ARGS:
    ./tools/box.sh '{{precheck}} && cargo build --release -p bloomery-model --bin r8conv && bash tools/ref/r8-sidecar.sh {{ARGS}}'

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

# V4.1 오라클(B0c): 같은 계측기를 deepseek41 프로필로 돌려 $BLOOMERY_DATA/ref_deepseek41/에 쓴다. CPU 백엔드만
# 쓴다. 프로필의 --defer-experts가 파일 세트 전체의 populate를 건너뛰어 그래프가 닿는 것만 읽는다(page cache가
# 찬 상태에서 1.27 GB; 콜드 읽기량은 재지 않았다). VARIANT(step4, d1, d2와 접미사 -unfused, -every-node)는
# 조용한 프리필 뒤 디코드 한 스텝을 자기 세트에 덤프한다 — models/deepseek41.sh.
dump-ref-v41 *VARIANT:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh 'bash tools/ref/dump.sh {{VARIANT}}'

# qwen3moe(Qwen3-30B-A3B-2507) 오라클: 같은 계측기를 qwen3moe 프로필로 돌려 $BLOOMERY_DATA/ref_qwen3moe/에 쓴다(CPU,
# 5토큰). VARIANT(step4, d1k, d4k와 접미사 -every-node)는 조용한 프리필 뒤 디코드 한 스텝을 자기 세트에 덤프한다 —
# models/qwen3moe.sh. d1k·d4k는 $BLOOMERY_DATA/qwen3moe/corpus-prose.ids의 앞 1,025·4,097개 id를 읽는다(sha256 고정).
dump-ref-qwen3moe *VARIANT:
    BLOOMERY_MODEL=qwen3moe ./tools/box.sh 'bash tools/ref/dump.sh {{VARIANT}}'

# 같은 5토큰의 ik CUDA 덤프(-ngl 99, box env가 핀한 3090)를 $BLOOMERY_DATA/ref_cuda_qwen3moe/에. 가중치 17.5 GiB가 카드에
# 올라가므로 3090에 다른 프로세스가 없을 때 친다.
dump-ref-qwen3moe-cuda:
    BLOOMERY_MODEL=qwen3moe ./tools/box.sh 'BLOOMERY_REF_BACKEND=cuda bash tools/ref/dump.sh'

# Qwen3.6-35B-A3B(qwen35moe) 오라클: ik가 lmstudio Q4_K_M 파일을 CPU로 돌린 노드 덤프(5토큰 배치 세트).
# VARIANT를 주면 조용한 프리필 뒤 디코드 한 스텝을 제 세트로 뜬다(step4, d1k, u1s4, u1s5와 -every-node —
# tools/ref/models/qwen35moe.sh). CPU 임대 아래서 돌므로 카드(docs/cards/q35oracle-dump.card)가 있어야 한다.
dump-ref-qwen35moe *VARIANT:
    BLOOMERY_MODEL=qwen35moe ./tools/box.sh 'bash tools/ref/dump.sh {{VARIANT}}'

# The Qwen3.5 dense (qwen35, Clef's backbone) hidden-state oracle: hidden_ref linked against llama.cpp mainline's build
# (tools/ref/build-hidden.sh), then llama.cpp's result_norm of every position of the first 64, 600 and 4,096 prose ids
# into $BLOOMERY_DATA/ref_qwen35_hidden_p{64,600,4096}/ (the 27B Q4_K_M file; refset family hidden-qwen35) and
# $BLOOMERY_DATA/ref_qwen35_flashq8_hidden_p{64,600,4096}/ (Clef-Flash at bartowski's Q8_0; hidden-qwen35-flash-q8),
# tools/ref/hidden.sh. Each dump puts its whole file on the card box.sh puts in view; no lease (a functional oracle). `--cpu-twin` writes each
# set's CPU twin into <set>.cpu/ instead (no card, 16 threads): the oracle's own floor the gate's bands come from.
build-ref-hidden:
    BLOOMERY_MODEL=qwen35 ./tools/box.sh 'bash tools/ref/build-hidden.sh'

dump-hidden-qwen35 *SETS:
    BLOOMERY_MODEL=qwen35 ./tools/box.sh 'bash tools/ref/hidden.sh {{SETS}}'

# GLM-5.3-Flash(glm5next) 오라클: 같은 덤프 도구를 glm5next 프로필로, ik를 CPU로 돌려 5토큰 배치 세트를
# $BLOOMERY_DATA/ref_glm5next/에 뜬다. VARIANT(step4, d1k, d3kdsa, d16kdsa와 -every-node 접미사)를 주면 조용한 프리필 뒤 디코드 한 스텝을
# 제 세트로 뜬다 — models/glm5next.sh. d1k는 $BLOOMERY_DATA/glm5next/corpus-prose.ids의 첫 1,025개 id를 읽는다(sha256 핀).
# 덤프마다 199.7 GB 샤드 집합 전체를 CPU 임대 아래서 올리므로 카드가 있어야 한다
# (BLOOMERY_BOX_ENV='BLOOMERY_LEASE_CARD=docs/cards/glmref-dump.card').
dump-ref-glm5next *VARIANT:
    BLOOMERY_MODEL=glm5next ./tools/box.sh 'bash tools/ref/dump.sh {{VARIANT}}'

# 같은 5개 id를 ik의 CUDA 백엔드로 $BLOOMERY_DATA/ref_cuda_glm5next/에, TF32는 끈다(NVIDIA_TF32_OVERRIDE=0: 없으면 이 모델에서
# ik CUDA의 top-1 일치가 0.896이었다, rig-log 2026-09-15). 이대로는 돌 수 없다: dump.sh의 -ngl 99가 199.7 GB 파일을 카드 한 장에
# 올리므로 CUDA 세트에는 배치 계획이 먼저 필요하다. GPU 소비자는 아직 없다.
dump-ref-glm5next-cuda:
    BLOOMERY_MODEL=glm5next ./tools/box.sh 'NVIDIA_TF32_OVERRIDE=0 BLOOMERY_REF_BACKEND=cuda bash tools/ref/dump.sh'

# The MTP draft oracles' dumper, GLM-5.3-Flash's and Qwen3.8-Flash-Next's: dump_mtp linked against the ik tree that
# carries the glm5next MTP graph on upstream's qwen4exp one (the glm5next profile's GLM_MTP_IK at GLM_MTP_SHA, checked
# clean); `cmake --build` of its libllama and libcommon first (nothing to do when current). Under the CPU lease:
# BLOOMERY_BOX_ENV='BLOOMERY_LEASE_CARD=docs/cards/glmmtpref-dump.card'.
build-ref-dump-mtp:
    BLOOMERY_MODEL=glm5next ./tools/box.sh 'bash tools/ref/build-dump-mtp.sh'

# Every node ik's MTP context computes while the target decodes 64 positions after the first 64 prose ids, one draft
# token a round, into $BLOOMERY_DATA/ref-mtp/prose64_n64_k1/ (refset family mtp-glm5next). ik on the CPU, CUDA hidden,
# under the CPU lease and the same card; the set is installed from staging only with its `# complete` trailer.
dump-ref-mtp-glm5next:
    BLOOMERY_MODEL=glm5next ./tools/box.sh 'bash tools/ref/dump-mtp.sh'

# Qwen3.8-Flash-Next MTP draft oracle: the same dump_mtp (built by `just build-ref-dump-mtp`) under the qwen4exp profile,
# the shared draft file beside the target (-md): every node ik's MTP context computes while the target decodes 64
# positions after the first 64 prose ids, one draft token a round, into $BLOOMERY_DATA/ref-mtp/qwen4exp_prose64_n64_k1/
# (refset family mtp-qwen4exp). ik on the CPU, CUDA hidden, under the CPU lease and its card
# (BLOOMERY_BOX_ENV='BLOOMERY_LEASE_CARD=docs/cards/mtp-qwen4exp-dump.card'); installed from staging only with its
# `# complete` trailer.
dump-ref-mtp-qwen4exp:
    BLOOMERY_MODEL=qwen4exp ./tools/box.sh 'bash tools/ref/dump-mtp.sh'

# Qwen3.8-Flash-Next(qwen4exp) 오라클: 같은 덤프 도구를 qwen4exp 프로필로, ik를 CPU로 돌려 5토큰 배치 세트를
# $BLOOMERY_DATA/ref_qwen4exp/에 뜬다. VARIANT(step4, d1k, d3k와 -every-node 접미사)를 주면 조용한 프리필 뒤 디코드 한 스텝을
# 제 세트로 뜬다 — models/qwen4exp.sh. d1k와 d3k는 $BLOOMERY_DATA/qwen4exp/corpus-prose.ids의 첫 1,025개와 3,001개 id를
# 읽는다(sha256 핀). d3k의 스텝은 위치 3,000이라 QSA 층이 인덱서가 고른 셀만 본다.
# 덤프마다 111.3 GB 샤드 집합 전체를 CPU 임대 아래서 올리므로 카드가 있어야 한다
# (BLOOMERY_BOX_ENV='BLOOMERY_LEASE_CARD=docs/cards/q38ref-dump.card').
dump-ref-qwen4exp *VARIANT:
    BLOOMERY_MODEL=qwen4exp ./tools/box.sh 'bash tools/ref/dump.sh {{VARIANT}}'

# DSpark draft 오라클: ik `db517b69`에 `ik-dsv41-draft.py`만 얹은 트리(`/home/user/ik-dspark-draft`)를 짓고
# dump_draft를 그 트리의 libllama·libggml에 링크한다. ik 빌드는 CPU 임대 아래서 돈다.
build-ref-dump-draft:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh 'bash tools/ref/build-dump-draft.sh'

# The V4.1 candidate-mask oracle: ik ed27bf7e (its separate V4.1 graph, V41_SEPARATE) with #2507's fix cherry-picked, at
# /home/user/ik-cand, and a dump_ref linked against it in $BLOOMERY_DATA/bin-cand; the ik build runs under the CPU lease
# (BLOOMERY_BOX_ENV='BLOOMERY_LEASE_CARD=docs/cards/candref-build.card'). `just dump-ref-v41 d1c-unfused-every-node` and the
# `-sep` variants dump from it (tools/ref/models/deepseek41.sh).
build-ref-ik-cand:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh 'bash tools/ref/build-ik-cand.sh'

# draft를 켠 greedy 디코드에서 draft 컨텍스트의 노드 전부를 $BLOOMERY_DATA/ref-draft/<세트>/에 쓴다(코드 코퍼스 64 + 32 위치,
# 블록 폭 3). 3090 게이트 락 + CPU 임대; 세트는 staging 뒤 `# complete` 트레일러가 있을 때만 설치된다.
dump-ref-draft:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh 'bash tools/ref/dump-draft.sh'

# 오라클 세트의 정수 사본 검사(B0c): 정수 텐서마다 무손실 사본이 있고, f32 파일이 그 값의 RNE인가.
# 인자는 세트 디렉터리(기본 $BLOOMERY_DATA/ref). V2-Lite의 ref는 v1이라 사본이 없어 빨강이다 — just gate에 넣지 않는다.
check-int-twins *ARGS:
    ./tools/box.sh 'python3 tools/ref/check-int-twins.py {{ARGS}}'

# V4.1 라우터 추적(B10): 토큰 스트림을 프리필로 흘려, 층마다 토큰별로 고른 expert 여섯을 $BLOOMERY_DATA/router/<이름>/에
# 쓴다. CPU 임대 안에서 돌고 30분에서 끊는다. CORPUS가 `oracle`이면 오라클의 다섯 토큰으로 돌려 덤프의 id와 정수로
# 맞춰 본다. 인자는 러너 머리글 참조(--name, --chunk, --max-tokens, --top-k-only).
trace-router CORPUS *ARGS:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh 'bash tools/ref/router-trace.sh {{CORPUS}} {{ARGS}}'

# GLM-5.3-Flash의 라우터 추적: trace-router와 같고, 코퍼스는 $BLOOMERY_DATA/glm5next/corpus-<이름>.ids(GLM 토크나이저),
# 세트 이름은 glm5next-<이름>이다(V4.1 세트와 한 디렉터리를 쓴다).
trace-router-glm5next CORPUS *ARGS:
    BLOOMERY_MODEL=glm5next ./tools/box.sh 'bash tools/ref/router-trace.sh {{CORPUS}} {{ARGS}}'

# Qwen3.8-Flash-Next의 라우터 추적: trace-router와 같고, 코퍼스는 $BLOOMERY_DATA/qwen4exp/corpus-<이름>.ids(Qwen3.8 토크나이저),
# 세트 이름은 qwen4exp-<이름>이다.
trace-router-qwen4exp CORPUS *ARGS:
    BLOOMERY_MODEL=qwen4exp ./tools/box.sh 'bash tools/ref/router-trace.sh {{CORPUS}} {{ARGS}}'

# ik 트리 하나의 wikitext-2 퍼플렉서티(c2048, 4청크, CPU만)를 서빙하는 V4.1 파일로 CPU 임대 아래서 잰다.
# 두 트리를 연달아 돌리면 포트의 A/B다. ARGS: --chunks N.
# 그 밖의 ARGS: --ctx N, --batch N, --ubatch N, --kld-base(KLD 기준 파일을 쓴다), --kld <기준 태그>(그 파일에 대한 KLD) — 러너 머리글.
ik-ppl TREE TAG *ARGS:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh 'bash tools/ref/ik-ppl.sh {{TREE}} {{TAG}} {{ARGS}}'

# 1단계 서브블록 게이트. 각 라운드가 자기 것 하나만 소유한다. gate-ops는 bloomery-model 라이브러리의 단위 시험도
# 같이 돈다(--lib, --include-ignored) — 그것을 도는 레시피가 따로 없다.
gate-ops:
    ./tools/box.sh 'bash tools/gate.sh -p bloomery-model --lib --test ops -- --include-ignored --nocapture'

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

# B1a 배치 게이트: V4.1 샤드 아홉의 헤더만으로 텐서마다 역할·장치·형식·상주 바이트를 정하고, 설계 §5의 두 안
# ((a) A6000 + DDR4, (b) A6000 0–19층 + 3090 20–39층)이 설계의 핀과 같은지 본다. 텐서 바이트·GPU·임대 없이 초 단위다.
# BLOOMERY_PLACEMENT_TABLE=1이면 텐서별 표도 찍는다. (b′) 시험은 DSpark 드래프트 헤더에서 3090 예약을 읽으므로
# 프로필의 DSPARK_MODEL을 내보낸다.
gate-placement:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh '__s=$(. tools/ref/ref-paths.sh && printf %s "$DSPARK_MODEL") && export BLOOMERY_DSPARK_MODEL="$__s" && bash tools/gate.sh --release -p bloomery-model --test placement -- --include-ignored --nocapture && bash tools/gate.sh --release -p bloomery-placement --lib -- --include-ignored --nocapture'

# B5 메타 게이트. 먼저 층 표 단위 시험을 돌린다: 일부러 깨뜨린 층 표 일곱 개를 ik 적재기의 검사 넷과 우리 거부 둘이
# 각각 깨진 그 층에서 거부해야 한다. 이어서 V4.1의 하이퍼파라미터·층 종류·텐서 이름(arch/deepseek41/{hparams,names}.rs)을
# ik가 같은 파일에서 읽은 값(적재 출력과 헤더), ik가 지은 그래프(오라클 매니페스트 둘의 노드), 파일의 텐서 목록과
# 대조한다. 헤더와 매니페스트만 읽으므로 몇 초면 끝난다.
gate-ds41-meta:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh 'bash tools/gate.sh --release -p bloomery-model --lib -- arch::deepseek41 arch::tests --nocapture && __s=$(. tools/ref/ref-paths.sh && printf %s "$DSPARK_MODEL") && export BLOOMERY_DSPARK_MODEL="$__s" && bash tools/gate.sh --release -p bloomery-model --test ds41_meta -- --ignored --nocapture'

# qwen3moe 메타 게이트: 하이퍼파라미터 거부 단위 시험(합성 헤더) 뒤, arch/qwen3moe가 파일에서 읽은 값을 ik 그래프
# (ref_qwen3moe 매니페스트의 노드 형상·op)와 헤더에 대조하고, 텐서 전부가 역할과 허용 타입을 갖는지 본다. 헤더만, 초 단위.
gate-qwen3moe-meta:
    BLOOMERY_MODEL=qwen3moe ./tools/box.sh 'bash tools/gate.sh --release -p bloomery-model --lib -- arch::qwen3moe --nocapture && bash tools/gate.sh --release -p bloomery-model --test qwen3moe_meta -- --ignored --nocapture'

# qwen35moe(Qwen3.6-35B-A3B) 헤더 게이트: 리더가 읽은 파일의 서술과 커버리지 검사의 목록(트리가 아직 못 돌리는 인스턴스, 층별).
gate-qwen35moe-meta:
    ./tools/box.sh 'bash tools/gate.sh --release -p bloomery-model --test qwen35moe_meta -- --ignored --nocapture'

# qwen3moe 커널 게이트(ik CPU 덤프 네 세트: 5토큰 프리필, 깊이 4·1,024·4,096의 디코드 스텝). `rope`는 헤드별 QK RMS
# 노름과 NEOX 로프·K/V 캐시 쓰기(한 런치)를 두 절로, `router`는 소프트맥스 라우터 128/8과 재정규화, 그리고 ubatch 라우터를,
# `down`은 다운 `_sel`(Q6_K, Q4_K)을, `head`는 argmax를 접은 Q6_K 헤드를, `flash`는 GQA 플래시 디코드 두 패스(스칼라·텐서
# 코어)를 잰다. 레시피마다 바이너리 하나.
gate-gpu-qwen3moe-rope:
    BLOOMERY_MODEL=qwen3moe ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin gate_qwen3moe_rope && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_qwen3moe_rope'

gate-gpu-qwen3moe-router:
    BLOOMERY_MODEL=qwen3moe ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin gate_qwen3moe_router && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_qwen3moe_router'

# Qwen3.6-35B-A3B MoE 블록: 257행 게이트 라우터(전문가 256 + 공유 전문가 게이트 행), 조인된 257-전문가 스택, K=512 down `_sel`,
# 아홉 슬롯 FFN과 ubatch 경로를 호스트 규칙과 ik 밴드에 맞춘다. ARGS는 `--case <이름>[,…]`.
gate-gpu-qwen35moe-moe *ARGS:
    BLOOMERY_MODEL=qwen35moe ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin gate_qwen35moe_moe && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_qwen35moe_moe {{ARGS}}'

# Qwen3.6-35B-A3B 전 체인 게이트(한 장): 노드 수 핀(디코드 520, 패스 537 + 4m), 디코드 다섯 스텝 = 다섯 행 패스 = eager
# = 리셋 뒤 반복(logits·저장소 비트 동일), 층별 teacher-forced 탭(GDN·어텐션·MoE)과 자유 주행 l_out을 ik 배치 세트에,
# step4·d1k 세트는 ik의 프리필 상태(cache_s·cache_k·cache_v)를 실어 한 스텝을 댄다.
gate-gpu-qwen35moe-e2e:
    BLOOMERY_MODEL=qwen35moe ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin gate_qwen35moe_e2e && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_qwen35moe_e2e'

# The Clef backbone (Qwen3.5 dense, qwen35) gate: the tensor-core decode flash refused at group 6 before any upload,
# then the prompt call's final-norm hidden states (Tail::Hidden) of the first 64, 600 and 4,096 prose ids, each from a
# reset as one GEMM ubatch (the 64 also as eight passes: the gemv arm), against llama.cpp mainline's result_norm on the
# same file (`just dump-hidden-qwen35`), every position within the derived band: the 27B Q4_K_M file (16.5 GB), then
# Clef-Flash at bartowski's Q8_0 (9.5 GB: every site through the launch its type picks). Fits either card.
gate-gpu-clef-hidden:
    BLOOMERY_MODEL=qwen35 ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin gate_clef_hidden && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_clef_hidden'

# The decide seat of bloomery-serve (gate_clef_serve's doc): Clef-Flash at bartowski's Q3_K_S (4.3 GB) with the
# release's head; the six refusals (a qwen35 file with no head, --model decide with no head, --model qwen3 with
# --head, a head config no row knows, a head narrower than the backbone, a GGUF of llama.cpp's Clef layout) exit
# before any backbone load, then one
# server answers the suite's first request on its row's route only, speaks llama.cpp's /v1/systemone wire (the
# answer's model is the server's, /v1/models, the engine object, an image a 501), the same body again. Either card.
gate-gpu-clef-serve:
    BLOOMERY_MODEL=qwen35 ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features glm5next,clef --release --bin bloomery-serve --bin gate_clef_serve && D=target/clef-serve-gate && rm -rf $D && mkdir -p $D && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_clef_serve --model /models/clef-flash/Cloudflare_clef-flash-Q3_K_S.gguf --head /models/clef-flash/hf/joint_head.safetensors --dir $D'

# clef_hidden on the box (ARGS as its doc: --model, --ids, --out, ...): a qwen35 file's prompt-only pass, every
# position's final-norm hidden state to a file and a `clef hidden` record with the call's functional wall.
clef-hidden *ARGS:
    BLOOMERY_MODEL=qwen35 ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin clef_hidden && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh clef_hidden {{ARGS}}'

# Qwen3.8-Flash-Next (qwen4exp) whole-program gate on the 3090 (its plan's card): the step's node count, graph = eager =
# one pass (logits and every store bit for bit), reset clears, every layer's streams on the batch set within the borrowed
# band with the router's flips named, the step after each set's prompt (4, 1,024 and 3,000 positions), D3K's prompt by
# passes = by steps, and the refusals. Loads the whole host set: alone in a batch, under the big-load lock.
[group('solo')]
[group('v41-load')]
gate-gpu-qwen4exp-e2e:
    BLOOMERY_MODEL=qwen4exp ./tools/box.sh 'export BLOOMERY_GATE_V41_LOAD=1 && cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin gate_qwen4exp_e2e && bash tools/gpu-gate.sh gate_qwen4exp_e2e'

# Qwen3.8's MTP draft program on the 3090 beside the target: first, before any walk, the draft's load — its weights,
# store and row map hold its plan's bytes, its token_embd and output are the target's own buffers at their addresses,
# and Mtp38::open's refusals — then every graph of ik's MTP draft set (mtp-qwen4exp) replayed teacher-forced — eh_proj,
# l_out and the head's logits within their derived bands, the router's flips named, the argmax ik's where its margin
# clears the band — the row-list head against the full head, the target's streams paired with ik's hidden rows, the
# captured walks = the eager ones, and the refusals and the NaN fault. Loads the whole host set: alone in a batch, under
# the big-load lock.
[group('solo')]
[group('v41-load')]
gate-gpu-qwen4exp-mtp:
    BLOOMERY_MODEL=qwen4exp ./tools/box.sh 'export BLOOMERY_GATE_V41_LOAD=1 && cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin gate_qwen4exp_mtp && bash tools/gpu-gate.sh gate_qwen4exp_mtp'

gate-gpu-qwen3moe-down:
    BLOOMERY_MODEL=qwen3moe ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin gate_qwen3moe_down && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_qwen3moe_down'

# qwen3moe 헤드: argmax를 접은 Q6_K 투영을 `q6k_gemv` + `argmax_fault`에 대고 — logits 비트 동일, 토큰과 폴트 워드 동일,
# 동점, NaN 행, NaN 활성, 그래프.
gate-gpu-qwen3moe-head:
    BLOOMERY_MODEL=qwen3moe ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin gate_qwen3moe_head && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_qwen3moe_head'

# 그룹 int8 텐서코어 GEMM(bloomery_gpu::gemm): 실파일 Qwen3 gate(Q4_K)·down(Q4_K, Q6_K) 스택과 V4.1 routed gate(Q3_K)
# 하나, 같은 형상의 합성 스택(+ Q5_K), T ∈ {1,15,16,17,64,511,512,4096} × 라우팅 넷을 f64 참조의 유도 밴드에 대고 잰다.
# fault·거부·그래프 재생 포함. ARGS는 `--case <부분문자열>` 필터. `--bench-kernels`는 판정 대신 런치 값을 잰다
# (리드 전용, `bench-gpu-kernels`).
gate-gpu-gemm *ARGS:
    BLOOMERY_MODEL=qwen3moe ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin gate_gemm && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_gemm {{ARGS}}'

# 선형 어텐션(Gated DeltaNet) 카드 커널 셋 — conv+준비, delta 스텝, 게이트 노름 — 을 호스트 규칙과 비트로 맞춘다.
# 모델 없이 Qwen3.6-35B-A3B 형상의 합성 입력으로 돌고, fault 사이트, 한 토큰의 그래프, 깊이 4,096의 f64 drift 보고를 함께 찍는다.
# ARGS `--case kda_ik`는 그 대신 KDA 층마다 conv·준비·delta·게이트 노름을 ik GLM 배치 세트(ref_glm5next)의 입력에 대 본다.
gate-gpu-linear *ARGS:
    ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin gate_linear && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_linear {{ARGS}}'

# Qwen3.6-35B-A3B full-attention 카드 커널 — rope-256(q/k 노름 + 앞 64차원 NEOX), 디코드 flash 256(두 패스),
# 프리필 flash 256, 게이트 접은 o-proj 양자화기 — 을 모델 없이 16/2×256, n_rot 64, θ 1e7 합성 입력으로 호스트 규칙에 맞추고,
# ik 배치 세트의 층 3 탭 넷과 비교한다.
gate-gpu-qwen35moe-attn:
    ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin gate_qwen35moe_attn && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_qwen35moe_attn'

# GLM-5.3-Flash MLA 층의 카드 부분 — 잠재 RMS + f16 추가, 인덱서 키 LayerNorm + [키; 게이트] 추가(`latent`),
# 창 0·싱크 −∞·스케일 1/16으로 다시 쓰는 V4.1 어텐션 — 을 모델 없이 합성 입력으로 호스트 규칙에 맞춘다.
# 추가는 캐시 비트, 어텐션은 오차 모형에서 끌어낸 밴드, 싱크 −∞는 분할 부분합의 무싱크 접기와 비트로 본다.
gate-gpu-glm-mla:
    ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features deepseek41 --release --bin gate_glm_mla && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_glm_mla'

# GLM-5.3-Flash's k-pool selector on the card (`latent::index_pool`, `kpool::kpool_score`, `qsa::qsa_topk_high`, the
# V4.1 attention over the list): synthetic clauses against the host rules and the exact rule's bands, then every
# latent layer teacher-forced on ik's --dsa step sets (refset `ik-glm5next-dsa`, `just dump-ref-glm5next d3kdsa`
# and `d16kdsa`). Reads no weight but the file's `indexer_compressor_ape`.
gate-gpu-glm-sel:
    BLOOMERY_MODEL=glm5next ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features deepseek41 --release --bin gate_glm_sel && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_glm_sel'

# GLM-5.3-Flash's prompt projections on the tensor-core GEMM (`gpu-glm5next/src/gemm.rs`, no engine caller yet): a KDA
# layer's seven and a latent layer's stack over 512 fixed-seed columns, on the file's own weights (only those tensors
# resident, about 0.3 GB), every output bit for bit the host transcription of `gemm_q8_0p`'s contract
# (`gpu-gates/src/gemm32.rs`); one quantize per input; the refusals by name.
gate-gpu-glm5next-gemm:
    BLOOMERY_MODEL=glm5next ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features glm5next --release --bin gate_glm5next_gemm && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_glm5next_gemm'

gate-gpu-qwen3moe-flash:
    BLOOMERY_MODEL=qwen3moe ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin gate_qwen3moe_flash && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_qwen3moe_flash'

# qwen3moe 체인 커널 셋(3090): Q4_K 임베딩 행(dequant_row·ik inp_embd와 비트 동일), expert gate·up·SwiGLU `_sel`
# (ffn_moe_gate_par 대조), combine(routed_out 대조).
gate-gpu-qwen3moe-experts:
    BLOOMERY_MODEL=qwen3moe ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin gate_qwen3moe_experts && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_qwen3moe_experts'

# qwen3moe 전 체인 게이트(3090 한 장): 노드 수 핀, 층별 teacher-forced·자유 주행 l_out 대조, greedy(ik-greedy-qwen3moe의
# 파일), graph = eager. 텐서 코어 플래시(엔진 경로)로 한 번 돈다. 스칼라 패스는 (u)의 자로 eager 안에서만 돈다.
gate-gpu-qwen3moe-e2e:
    BLOOMERY_MODEL=qwen3moe ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin gate_qwen3moe_e2e && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_qwen3moe_e2e'

# ik CPU의 greedy 연속(프롬프트 0–7, 32토큰)을 $BLOOMERY_DATA/qwen3moe/greedy/에. 프롬프트마다 CPU 임대를 잡는다.
ik-greedy-qwen3moe:
    BLOOMERY_MODEL=qwen3moe ./tools/box.sh 'for p in 0 1 2 3 4 5 6 7; do GEN=32 PROMPT=$p bash tools/ref/ik-greedy.sh || exit $?; done'

# ik KLD 기준 파일 TAG에 대한 qwen3moe 체인의 NLL·KLD·top-1(출력만, 판정 없음).
run-qwen3moe-ppl TAG:
    BLOOMERY_MODEL=qwen3moe ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin gate_qwen3moe_e2e && BLOOMERY_GATE_BOUND=1680 bash tools/gpu-gate.sh gate_qwen3moe_e2e --ppl {{TAG}}'

# ik-ppl을 qwen3moe 파일에(MODEL을 프로필 값으로 넘긴다). ARGS는 ik-ppl.sh 머리글.
ik-ppl-qwen3moe TREE TAG *ARGS:
    BLOOMERY_MODEL=qwen3moe ./tools/box.sh 'MODEL=$BLOOMERY_REF_MODEL bash tools/ref/ik-ppl.sh {{TREE}} {{TAG}} {{ARGS}}'

# qwen3moe 생성 CLI(3090): --prompt 텍스트 또는 --tokens id 목록, -n, --ctx, --mode, --time.
gen-qwen3moe *ARGS:
    BLOOMERY_MODEL=qwen3moe ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin generate_qwen3moe && bash tools/gpu-gate.sh generate_qwen3moe {{ARGS}}'

# V4.1 호스트 expert 티어 게이트(B5). moe::Meta가 파일의 expert 384개를 Hparams와 같게 읽는지 확인한 뒤, 5토큰 세트의
# 모든 층에서 ik의 라우팅을 주입해 호스트 티어가 낸 routed 부분합을 ik의 ffn_moe_out과 도출한 밴드 안에서 대조한다
# (플립은 따로 세고, 층마다 세트가 닿은 클램프 히트 수를 찍는다 — 어느 층이 닿는지는 파일의 것이다). 클램프 자체는
# 층마다 합성 입력으로 한계 너머까지 따로 본다: 토큰 0의 라우팅을 그대로 두고 입력 행을 2의 거듭제곱으로 키워
# silu(g) > L, u > L, u < −L을 모두 넘긴 뒤 combine이 qdot::swiglu_clamp와 비트 동일하고 f64 서술 안인지 확인한다.
# 세트가 라우팅한 expert만 읽는다. 끝으로 크레이트 doctest —
# ops::ShardTensor의 compile_fail(다른 샤드를 가리키는 핸들은 만들 수 없다).
gate-ds41-host:
    ./tools/box.sh 'bash tools/gate.sh --release -p bloomery-model --test ds41_host -- --ignored --nocapture && bash tools/gate.sh --release -p bloomery-model --doc'

# 호스트 합집합 호출 게이트: moe::HostLayer::experts_union_into가 토큰 열 k = 1·2·3·6에서 열마다
# experts_into(그 열, 그 열의 목록)와 비트까지 같은지를 V4.1 라우팅 층 40개 전부(ref_deepseek41의 라우팅)에서
# 지연 팔 둘로 재고, 파일 하나를 받는 진입은 V2-Lite 블록 1에서, 이름 붙은 거부 여섯은 각각 확인한다.
# r8 배치: gate·up을 r8 사이드카에서 행-레인 타일로 읽는 층이 파일 행을 읽는 층과 열마다 비트까지 같다
# (합성 층 하나, 모델 파일 없이 박스 CPU의 AVX2에서).
# --test-threads=1: 두 테스트가 프로세스 전역 지연 레버(ops::set_defer_quant)를 뒤집는다.
gate-union:
    ./tools/box.sh 'bash tools/gate.sh --release -p bloomery-model --test union -- --ignored --nocapture --test-threads=1'

# r8 사이드카(model::r8file) 게이트: gguf::write로 쓴 합성 두 샤드 원본에서 변환, Sidecar::open, verify가
# 통과하고, 원본과의 식별 차이와 변환 거부는 모두 이름으로 거부된다. r8conv 바이너리는 변환부터 검증까지
# 끝까지 돈다. 호스트 티어 적재는 원본이 달라진 사이드카를, 호스트 층은 down까지 담은 사이드카를 이름으로
# 거부한다. 모델 파일은 읽지 않는다.
gate-r8:
    ./tools/box.sh 'bash tools/gate.sh --release -p bloomery-model --test r8file -- --include-ignored --nocapture'

# The V4.1 gate fixture's generator (model::arch::deepseek41::fixture, bin v41fixture) against the real file's header and
# the real DSpark draft's: the plan (the map's nine layers, the metadata overrides, the engine reading the
# header-only files as the source's layer kinds), determinism, every sampled block's scales and RMS, and
# small subset files written, opened and verified end to end (under 1 GB each, removed). Reads headers
# only; writes only under this tree's target/tmp.
gate-fixture:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh '__s=$(. tools/ref/ref-paths.sh && printf %s "$DSPARK_MODEL") && export BLOOMERY_DSPARK_MODEL="$__s" && bash tools/gate.sh --release -p bloomery-model --test fixture -- --include-ignored --nocapture --test-threads=1'

# 1-5 스레드 풀 게이트: 상주 워커 풀의 분할 전수·커버리지·반복 호출·패닉 전파.
# hw_ 토폴로지 테스트는 #[ignore]라 --include-ignored로 같이 돈다.
gate-threads:
    ./tools/box.sh 'bash tools/gate.sh --release -p bloomery-threads --test pool --test helper -- --include-ignored --nocapture'

# 레버 레지스트리 게이트(crates/levers, 호스트만): 모든 행이 형식에 맞는가, 종류가 받지 않는 값과 은퇴한 이름을
# 이름을 대고 거부하는가, 제자리 행과 tools/levers-direct.txt가 서로 맞는가. 레지스트리를 마크다운 표로 찍는다.
gate-levers:
    ./tools/box.sh 'bash tools/gate.sh -p bloomery-levers --lib -- --nocapture'

# 샘플러 게이트: greedy·top-k=1이 엔진 argmax와 같고, top-p·min-p·반복 벌점이 참조의 정의대로 자르고,
# 추첨 빈도가 소프트맥스를 따르며, 같은 시드는 같은 토큰을 내고, 첫 호출 뒤 할당이 0인가. 박스 자원 불필요.
gate-sampler:
    ./tools/box.sh 'bash tools/gate.sh --release -p bloomery-sampler --test sampler -- --nocapture'

# The generation loop (crates/runtime, host only): the stop rule, the lookup draft, and the drafts'
# pick/commit arithmetic against a mock target — plain = draft tokens, a rejected row taken back,
# only kept tokens in the lookup's context, a failed call not run again.
gate-runtime:
    ./tools/box.sh 'bash tools/gate.sh -p bloomery-runtime --lib -- --nocapture'

# The model description crate (crates/models, host only): the shape selector's rows — every model
# the tree runs selects its router and flash instance, a shape between or outside the rows is
# refused by name, no two rows overlap.
gate-models:
    ./tools/box.sh 'bash tools/gate.sh -p bloomery-models --lib -- --nocapture'

# The session crate's host tests (crates/app, a device-linked crate, so cargo oxide test): a fault
# read at a step, a group end or a call end is one SessionError::Fault. They open no card (host group).
[group('host')]
gate-app:
    ./tools/box.sh 'bash tools/gate.sh --oxide -p bloomery-app --release --lib'

# 2단계 qdot 게이트: Q3_K×Q8_K 융합 커널이 현재 경로(dequant+roundtrip+f32)와
# 1e-5 안팎에서 일치하고, 정확해(f64)에 더 가깝고, 스칼라 폴백과 비트 동일인가.
# 순수 게이트(rejects_unaligned_k)는 #[ignore]가 아니라 --include-ignored로 같이 돈다.
gate-qdot:
    ./tools/box.sh 'bash tools/gate.sh --release -p bloomery-qdot --test qdot -- --include-ignored --nocapture'

# B3 engram 게이트: V4.1의 NVMe 테이블에서 매핑으로 읽은 행이 같은 오프셋의 pread와
# 바이트 동일이고(선행 읽기를 걸어도 같고), 행 간격이 텐서를 정확히 타일링하는가.
# 정확성 실행이라 임대는 필요 없다 — 속도는 engram-rate가 임대 안에서 잰다. --lib은 prefetch의 helper 배치
# 단위 테스트(한 cpu 마스크를 물려받은 float helper가 마스크를 넓히는가)를 같은 게이트에 넣는다.
gate-engram:
    ./tools/box.sh 'bash tools/gate.sh --release -p bloomery-engram --lib --test engram -- --include-ignored --nocapture'

# 토크나이저 게이트: 우리 id가 engram 코퍼스 텍스트 전부와 케이스 파일에서 두 parse 모드 모두
# llama-tokenize와 같고, 참조 id가 원문으로 되돌아오는가 — V4.1 어휘(deepseek-v3), qwen3moe 어휘(qwen2), GLM-5.3-Flash 어휘(glm4) 셋 다.
# oracle.sh가 어휘마다 오라클 파일을 먼저 다시 쓴다(어휘만 적재, 임대 없음, 1분 안).
# qwen3moe와 GLM 어휘의 참조는 /home/user/ik-tokref(ik-idxkey와 같은 커밋 + ik#2520 tolower 수정)의 llama-tokenize로 뜬다 —
# ik의 unicode_tolower는 정렬되지 않은 표에 lower_bound를 써서 (?i:'re) 같은 축약이 어긋나고, 그 경로는 qwen2 정규식과
# llama3 정규식(glm4가 쓰는 분할기)이 탄다.
# #2520이 ik에 들어가면 기본 트리로 되돌린다. V4.1 어휘의 참조도 곁가지 트리다: oracle.sh의 기본 TOKENIZE는 /home/user/ik-tilde
# (ik main + `~`를 S에 넣는 한 줄, ik#2528)이고, 참조의 `~/`가 [71520]이 아니면 거부한다 — #2528이 들어가면 되돌린다.
gate-tokenizer:
    ./tools/box.sh 'timeout --kill-after=10 300 bash crates/tokenizer/tools/oracle.sh && TOKENIZE=/home/user/ik-tokref/build/bin/llama-tokenize TOKENIZER_VOCAB=/models/Qwen3-30B-A3B/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf TOKENIZER_SET=tokenizer-qwen3moe timeout --kill-after=10 300 bash crates/tokenizer/tools/oracle.sh && TOKENIZE=/home/user/ik-tokref/build/bin/llama-tokenize TOKENIZER_VOCAB=/models/GLM-5.3-Flash-UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf TOKENIZER_SET=tokenizer-glm5next timeout --kill-after=10 300 bash crates/tokenizer/tools/oracle.sh && bash tools/gate.sh --release -p bloomery-tokenizer --lib --test tokenizer -- --include-ignored --nocapture'
# Clef's request encoder, joint schema head and SystemOne answer (crates/decision, host only) against the
# release's own Python (tools/ref/clef_ref.py's files under $BLOOMERY_DATA/clef/flash/ref): ids and spans
# exact, f32 logits inside the derived band of the f64 referee, the body's shape and top options.
gate-decision-clef:
    ./tools/box.sh 'bash tools/gate.sh --release -p bloomery-decision --lib --test clef -- --include-ignored --nocapture'
# HTTP 서버 게이트(모의 엔진): llama-server JSON 형태, SSE 프레이밍, 정지 규칙(정지 id 목록 전부), V4.1·GLM 채팅 템플릿 렌더링,
# GLM 도구 호출 파서(glmxml), Qwen3의 헤르메스 호출·모델이 여는 생각 범위(hermes), 연결 상한·유휴 연결 종료·컨텍스트 끝의 정지(limits). 박스 자원 불필요.
# Anthropic's Messages API on the chat path (anthropic). Qwen3.6's and Qwen3.8's XML tool calls, and every seat
# template's tool-call markup mapped to its parser (qwenxml). The repetition, frequency and presence penalties as
# llama-server takes them (penalty). A sampled request's drafted passes against its plain steps (sampdraft).
gate-serve:
    ./tools/box.sh 'bash tools/gate.sh -p bloomery-serve --lib --test serve --test dsml --test glmxml --test hermes --test qwenxml --test limits --test anthropic --test penalty --test sampdraft -- --include-ignored --nocapture'

# engram IO 실험실의 시험(crates/engram-lab, 엔진 사용처 없음): 컨텍스트 창의 슬롯 순서, 행 캐시의 LRU를 스택
# 거리 모의와 대조, 캐시가 내주는 바이트. 이름이 gate-가 아니라 lab-이라 `just affected`가 엔진 착륙에서 고르지
# 않는다. V4.1 분할 파일과 engram-corpus 스트림이 있어야 돈다.
lab-engram:
    ./tools/box.sh 'bash tools/gate.sh --release -p bloomery-engram-lab --test lab -- --include-ignored --nocapture'

# engram-rate 바이너리. 측정은 tools/ref/engram-rate.sh가 임대 안에서 돌린다 —
# 이 레시피는 빌드만 한다(러너가 낡은 바이너리를 재는 것을 막는 단계).
build-engram:
    ./tools/box.sh 'cargo build --release -p bloomery-engram-lab --bin engram-rate'

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
# 덤프는 둘이다: V2-Lite의 여섯 타입은 $BLOOMERY_DATA/ref에, V4.1 첫 샤드는 파일의 타입들(공개 Q3_K_M 파일이면
# f32·q3_K·q4_K·q5_K·q6_K의 다섯)을 $BLOOMERY_DATA/ref-v41<세트 접미>에 — 공개 파일은 ref-v41_plain. --include-ignored라 split 리더와 인벤토리의 평범한 테스트도 같이 돈다.
# 박스의 모델 파일에 없는 q2_K·iq2_xs·iq3_xxs·iq4_xs는 --synthetic이 ggml로 양자화한 행과 무작위 코드 행을
# $BLOOMERY_DATA/ref-synth에 덤프하고, i-quant 코드북(iq_tables.rs)은 gen-iq-tables.py --check가 ik 헤더에서
# 다시 뽑아 커밋된 파일과 바이트로 대조한다.
gate-1-1:
    ./tools/box.sh 'source tools/ref/ref-paths.sh && python3 tools/ref/gen-iq-tables.py --check --ik "$IK" && bash tools/ref/build-dequant.sh && "$BLOOMERY_DATA/bin/dequant_ref" && S=$(. tools/ref/models/deepseek41.sh && printf %s "$V41_SET_SUFFIX") && "$BLOOMERY_DATA/bin/dequant_ref" "$BLOOMERY_V41_MODEL" "$BLOOMERY_DATA/ref-v41$S" && "$BLOOMERY_DATA/bin/dequant_ref" --synthetic && bash tools/gate.sh -p bloomery-gguf -- --include-ignored --nocapture'

# 커밋 전에 치는 것. 측정은 포함하지 않는다(조용한 기계가 필요하다).
gate: check-recipes check-rustflags check-arch check-comments check-levers check-unsafe check-loads fmt-check lint gate-1-1 gate-vision gate-gpu-gates-lib gate-gpu-lib gate-sampler gate-levers

# The smoke tier (docs/gates-plan.md 3.1): the static checks, the V4.1 decode step, the V4.1 prompt batch at
# P = 512 alone, V2-Lite end to end — a subset run of unchanged gates, never the landing batch. Two lanes
# through tools/gate-batch.sh (its header: the list, lanes, logs, DONE line); ARGS go to it (--dry-run, --out DIR,
# --lanes 1). It refuses to start while the timing lease is held.
smoke *ARGS:
    ./tools/gate-batch.sh --smoke {{ARGS}}

# The weekly tier: every `weekly-*` recipe (a gate taken out of the landing batch; `just affected` selects only
# `gate-*`, and names a weekly recipe when a file one of its trigger rows in tools/gate-paths.tsv matches changes) in
# one sitting through tools/gate-batch.sh, on the lead's ledger. ARGS go to it (--dry-run, --out DIR, --rerun).
weekly *ARGS:
    ./tools/gate-batch.sh --ledger --weekly {{ARGS}}

# B0b V4.1 인벤토리: 분할 GGUF의 헤더만 읽어 텐서 표를 뽑는다(임대 불필요, 텐서 바이트 미접촉).
# 표는 박스의 /tmp에 쓰고 scp로 회수한다 — 박스 작업 트리에 쓰면 다음 box.sh의 rsync --delete가 지운다.
inventory-v41:
    ./tools/box.sh 'cargo run --release -p bloomery-gguf --bin gguf-inventory -- --markdown /tmp/v41-inventory.md "$BLOOMERY_V41_DIR"/*-0000*-of-00009.gguf'
    scp "${BLOOMERY_BOX:-ws}:/tmp/v41-inventory.md" docs/v41-inventory.md

# GPU 헤드(P8의 head 조각): result_norm → lm_head(Q6_K) → argmax를 그래프 하나로 잡아
# 덤프의 마지막 토큰과 대조(정확성 실행, 핀된 헤드 밴드 안).
gate-gpu-head:
    ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin gate_head_gpu && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_head_gpu'

# m열 커널(q8_0·q8_0 heads·K-quant q3_K·q4_K·q5_K·q6_K·블록 대각 heads·공유 expert gate·up·헤드 argmax): m열 런치의 열 c가
# 그 열 하나를 m = 1로 쏜 결과와 비트 동일한지를 파일의 형식으로 고른 V4.1 실제 행과 합성 K(≤ 8192)에서 m = 1..8 전부
# 본다 — k토큰 스텝이 k개 순차 스텝과 비트 동일하려면 이것이 서야 한다.
gate-gpu-mcol:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features deepseek41 --release --bin gate_mcol && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_mcol'

# bloomery-gpu's lib tests through tools/gate.sh --oxide (its bound, cargo's exit code), with no gate lock, so the
# batch keeps them on the 3090 (lane A). The hw_ tests (graph.rs' capture, replay, host flags and node kinds;
# checkpoint.rs; host/step.rs' boundary) run small kernels on device 0. just gate runs it.
gate-gpu-lib:
    ./tools/box.sh 'bash tools/gate.sh --oxide -p bloomery-gpu --release --lib -- --include-ignored'

# bloomery-gpu-gates 라이브러리의 단위 시험(호스트 전용, 카드·게이트 락 없음). ptx.rs의 컨테이너 판독 계약 —
# 8의 배수 길이 페이로드의 마지막 본문, 번들 경계, 파서가 거부한 섹션은 빈 표가 아니라 오류 — 이 여기서 돈다.
# gpu 피처 없이 빌드하므로 디바이스 크레이트를 컴파일하지 않고, 그래서 평범한 cargo다. just gate가 부른다.
gate-gpu-gates-lib:
    ./tools/box.sh 'bash tools/gate.sh --release -p bloomery-gpu-gates --lib'

# V4.1 오라클 세트 ref_deepseek41의 모든 행과 디코드 스텝 세트의 헤더를 하니스가 읽는지 본다. 호스트 전용이라 카드도
# 게이트 락도 쓰지 않는다. 시험 둘의 출력이 섞이지 않게 한 스레드로 돈다.
gate-ds41-oracle:
    ./tools/box.sh 'bash tools/gate.sh --release -p bloomery-gpu-gates --lib -- --ignored hw_ds41_oracle --nocapture --test-threads=1'

# The --hf resolver (crates/hf, host only): the repo string, set selection from captured API listings (a split-file
# repo, mmproj and imatrix left out, zero and two matches refused), cache paths, the model named once, and the fetch
# through curl on file:// URLs (a resume from a truncated file, a refetch-free second start, size and digest refusals),
# and the offline start (a refused connection lets the one verified cached set stand in; zero or two, or an HTTP error
# from a local port, are refused). No network, no card.
gate-hf:
    ./tools/box.sh 'bash tools/gate.sh -p bloomery-hf --lib -- --nocapture'

# Reference sets (crates/refset, host only): the readers' unit tests, then every family's sets in place
# against the family table — each complete, dumped from the file the tree runs, of the family's ik build.
# The draft set's family needs the profile's DSPARK_MODEL, exported as the dspark recipes do.
gate-refset:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh '__s=$(. tools/ref/ref-paths.sh && printf %s "$DSPARK_MODEL") && export BLOOMERY_DSPARK_MODEL="$__s" && bash tools/gate.sh --release -p bloomery-refset --lib -- --include-ignored --nocapture'

# Which reference sets are stale, one line each: with no arguments every family's sets in place, else
# `<family> <path>...` reads the sets at the paths as sets of that family (host only, reads only).
refset-check *ARGS:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh '__s=$(. tools/ref/ref-paths.sh && printf %s "$DSPARK_MODEL") && export BLOOMERY_DSPARK_MODEL="$__s" && cargo build --release -p bloomery-refset --bin refset-check && bash tools/host-gate.sh refset-check {{ARGS}}'

# V4.1 호스트 스텝 계획: 디코드 걷기 단위 시험(위치 2,101개)을 돈 뒤, V4.1 오라클 세트(배치 세트와 디코드 스텝 세트)의
# 그래프 입력 전부를 계획과 비트 단위로 대조한다. 어떤 검사도 맡지 않은 입력은 빨강이다. 호스트 전용(카드·게이트 락 없음).
gate-ds41-plan:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh 'bash tools/gate.sh --release -p bloomery-model --lib -- arch::deepseek41::plan --nocapture && cargo build --release -p bloomery-gpu-gates --bin gate_deepseek41_plan && bash tools/host-gate.sh gate_deepseek41_plan'

# V4.1 rope 계열(머리 꼬리 rope, ROPE_BACK, 잠재 K/V의 norm·rope·f16 링 기록)을 5토큰 세트와 디코드 스텝 세트 전부에 대조한다.
# rope 사이트는 ik와 비트 동일, K/V 행은 유도한 밴드 안이어야 한다. 게이트가 V4.1 파일의 메타데이터를 읽으므로 모델을 고정한다.
gate-gpu-ds41-rope:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features deepseek41 --release --bin gate_deepseek41_rope && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_deepseek41_rope'

# V4.1 압축기와 인덱스 키(상태 gemv, DS4_COMP 풀링, norm, 꼬리 rope, f16 캐시 행, 링 persist, 인덱스 키의
# norm·rope·Hadamard)를 5토큰 세트와 디코드 스텝 세트의 모든 소스 층에서 대조한다.
gate-gpu-ds41-comp:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features deepseek41 --release --bin gate_deepseek41_comp && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_deepseek41_comp'

# V4.1 인덱서(op F: 쿼리·가중치·점수·top-k 목록)를 d1·d2 세트와 그 unfused 쌍의 내는 층 전부에서 ik 덤프와 대조한다.
# d1n의 항등 목록과 그 위의 attention을 접두 경로와 비트 대조하고, 16,384·32,768행의 합성 깊이 케이스(심어 둔 동점 포함),
# 재실행 비트 동일과 캡처된 재생까지 본다. 목록은 행 오름차순, 동점은 낮은 행.
gate-gpu-ds41-index:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features deepseek41 --release --bin gate_deepseek41_index && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_deepseek41_index'

# V4.1 디바이스 크레이트의 호스트 단위 시험(카드·게이트 락 없음): 오라클 세트가 닿지 않는 큰 위치에서도 rope 표가 ggml 레시피와 같다.
# 락이 없어도 되는 까닭: 이 크레이트의 시험 모듈은 카드를 열지 않는다(Gpu·스트림·디바이스 버퍼를 쓰는 시험이 없다). cargo oxide는 빌드 때문이다.
[group('host')]
gate-gpu-ds41-lib:
    ./tools/box.sh 'bash tools/gate.sh --oxide -p bloomery-gpu-deepseek41 --release --lib'

# V4.1 서버 엔진 결합(gpu-gates의 bind, deepseek41 기능 뒤에 있어 gate-gpu-gates-lib가 빌드하지 않는다)의 호스트 단위 시험:
# 샘플링 스텝이 엔진의 로짓 버퍼 하나를 빌려 쓰고, 길이가 다른 호출자 버퍼는 이름 붙은 오류다. 카드·게이트 락 없음.
[group('host')]
gate-ds41-bind:
    ./tools/box.sh 'bash tools/gate.sh --oxide -p bloomery-gpu-gates --release --features deepseek41 --lib -- bind::'

# The V4.1 binaries' `--place` word (`generate::Place`, behind the deepseek41 feature like bind): each alias and its card
# list (`a6000+3090` is `bp`) planning the machine its plan function makes, a list word planning its own cards, and every
# refusal by name before any plan. Host only, no card or gate lock.
[group('host')]
gate-ds41-place:
    ./tools/box.sh 'bash tools/gate.sh --oxide -p bloomery-gpu-gates --release --features deepseek41 --lib -- generate::'

# ik의 KLD 기준 파일(ik-ppl --kld-base)이 그것을 쓴 실행과 맞는지 본다: 헤더의 ctx·청크 수와 파일 크기가 실행의 결과
# 줄과 같고, 기록마다 ik 양자화기가 쓸 수 있는 모양이며, 파일에서 다시 잰 PPL이 실행이 찍은 PPL과 양자화 밴드 안에서
# 같다. 기본은 kldbase-c2048x4와 kldbase-c512x16이다. 어휘 수를 모델 파일에서 읽으므로 V4.1 프로필로 돈다. 호스트
# 전용이라 카드도 게이트 락도 쓰지 않는다. FAIL-first는 박스 명령 안에서 BLOOMERY_KLD_FILE=<사본 경로>를 준다 — 그
# 파일 하나만 읽고, 파일 이름의 stem이 가리키는 실행의 로그와 맞춘다.
gate-ds41-kld:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh 'bash tools/gate.sh --release -p bloomery-gpu-gates --lib -- --ignored hw_ds41_kld --nocapture'

# V4.1 engram 게이트: 키 norm, 그리고 쿼리 norm·f64 점곱·부호 붙은 제곱근 시그모이드·스트림 갱신을 5토큰 세트와 디코드
# 스텝 세트의 engram 층(1·14)마다 우리 규칙과 ik 규칙 시뮬, 덤프에 대조한다. 토큰마다 gemv → 키 norm → 게이트 사슬도 본다.
# 규칙은 파일의 형식이 고른다: engram_wkv가 Q8_0이면 q8_0_gemv와 ik의 q8_2 규칙, Q3_K(공개 파일)이면 q8_1 → q3k_gemv와
# ik의 q8_K × Q3_K 점곱(engram_kv와 비트 동일), 행 테이블과 gain 둘도 Q8_0/bf16 또는 Q3_K. 그 밖의 형식은 텐서 이름을 대며 거부한다.
gate-gpu-ds41-engram:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features deepseek41 --release --bin gate_deepseek41_engram && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_deepseek41_engram'

# A PLE site's two kernels (Qwen3.8's layer 1, bloomery_gpu::ple) on synthetic inputs, no model: the gate bit for bit
# against its host rule and within a derived per-value bound of the exact f64 values; the dilated conv bit for bit
# against its host rule; then the m-column, ring-wrap, rollback, reset, fault and refusal clauses.
gate-gpu-ple:
    ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin gate_ple && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_ple'

# The V4.1 glue piece (chain G1) on the step4 and d1n sets: the host engram row ids against the dump's int rows,
# the embedding broadcast bit for bit, each engram site's step within the band b4engram's pins propagate, and
# the head end (hc_out bit for bit, the logits against their predicted gap, the argmax); one captured graph
# replayed per set against the eager run, with its node count. The weights load through the engine's role
# formats, so a Q3_K engram_wkv (the public file) runs q8_1 + q3k_gemv, pinned within KERNEL_BAND of the exact
# dot and against ik's Q3_K dot bit for bit; the node count follows the format.
gate-gpu-ds41-chain-glue:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features deepseek41 --release --bin gate_deepseek41_chain_glue && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_deepseek41_chain_glue'

# V4.1 attn_output_a(블록 대각 여덟 그룹)를 q8_0_gemv_heads 한 번의 발사로, attn_output_b는 q8_0_gemv로 돌려 5토큰 세트와
# 디코드 스텝 세트의 층마다 우리 규칙(비트 동일), ik 규칙 시뮬(덤프와 비트 동일), 덤프(값마다 유도한 밴드)에 대조한다.
gate-gpu-ds41-woa:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features deepseek41 --release --bin gate_deepseek41_woa && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_deepseek41_woa'

# V4.1 어텐션(B4): 세트 일곱(5토큰 배치와 디코드 스텝 여섯)의 모든 층을 접두 키와 선택 행 두 경로로 본다. ik 규칙
# 시뮬 대 덤프, 커널 대 우리 f32 규칙과 덤프, 그래프 재생, 깊이 케이스. sink를 읽으려고 V4.1 모델 파일을 연다.
gate-gpu-ds41-attn:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features deepseek41 --release --bin gate_deepseek41_attn && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_deepseek41_attn'

# V4.1 attention 조각(B5 1단계, chain G1): step4·d1n 세트의 40층 전부에 덤프의 입력을 주입하고, 스트림·접기·링·압축기
# 캐시를 유도한 편차(σ 전파, z ≤ 9) 안에서 ik와 대조한다. 스텝이 안 쓴 슬롯은 비트 동일, 재생 = eager, 층 종류별 노드 수.
gate-gpu-ds41-chain-attn:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features deepseek41 --release --bin gate_deepseek41_chain_attn && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_deepseek41_chain_attn'

# 조립된 V4.1 스텝(B5 1단계, pos < 512·선택 없음)을 게이트 배치로 엔진 입구를 거쳐 돌린다. --structure: 캡처된 스텝의
# 커널·메모리 연산 배치 수가 조각들이 예측한 수와 같고, step4 세트의 상태를 주입한 재생이 eager 실행과 비트까지 같다.
# --sets: step4·d1n 세트에서 층마다 덤프의 상태를 주입해 eager 스텝 하나 — engram 행 id와 라우터 id는 정확히(근접 동률은
# 면제), 서브층마다 스트림은 전파 밴드 안, result_output의 argmax. 3090, 게이트 락.
# --select(2단계): d1(top_k 64)·d2 세트, 인덱서 층마다 선택이 실제로 일어나는 자리 — 리스트가 그 층 점수의 정확한 top-k인지,
# 인덱서의 쿼리·가중치가 포락선 안인지, ik 리스트와의 차이가 동률 밴드 안뿐인지; 리스트를 바꿔 끼웠을 때 어텐션이 받는
# 영향은 포락선에 합쳐 잰다.
# --skew-structure, --skew-sets, --skew-api (C4 ktok-skew, shared/ds41_skew.rs), on the same load after those, from a
# reset: tokens t and t+1 as two rows one layer apart on one stream against two one-token steps, bit for bit.
# --skew-structure: the captured nodes exactly twice the step's by kind and kernel name, the batch order, each row's
# shadow table. --skew-sets: after step4 and d1 are injected, both rows' logits, streams, folds and lists and every
# cache, compressor state and history equal the steps in turn, eager and replayed, and after a rollback (t+1 refused)
# one step of another token equals the steps. --skew-api: the engine's entry (step_pair, rollback), graph and eager,
# then the deep cuts. One load.
[group('v41-load')]
gate-gpu-ds41-step *ARGS='--structure --sets --select --skew-structure --skew-sets --skew-api':
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh 'export BLOOMERY_GATE_V41_LOAD=1 && cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features deepseek41 --release --bin gate_deepseek41_step && bash tools/gpu-gate.sh gate_deepseek41_step {{ARGS}} && bash tools/gpu-gate.sh gate_deepseek41_step --slots'

# V4.1 전문가 층(host/tier.rs, twoeng R3)의 루프백: 스테이지와 층을 한 카드(3090, 게이트 배치)의 Gpu 둘에 올린다.
# The gate plan puts each layer's id prefix [0, n_l) on the card; the tier plan moves its last 23 ids (n_l−23 … n_l) to
# device 1 (the tier), the (b′) shape. 기준은 게이트 계획 자체(합집합이 한 카드에)다. 두 적재는
# 차례로, 기준 먼저. --union(T1): 산문 프롬프트 16위치를 한 스텝씩, 탐욕 32스텝, 쌍 4번(그래프),
# 프롬프트와 4스텝(eager)이 기준과 argmax·logits 비트까지 같고 스테이지 그래프 노드 수가 같다; 전제: 층이 가진 층마다
# 라우팅된 슬롯이 한 번 이상 갔다. --fault(T2): 층 카드 폴트가 스텝의 오류·독, 리셋 뒤 기준과 같다. --lost(T3): 층
# 스트림을 호스트 플래그 뒤에 묶으면 데드라인 안에 카드 상실 이름, CardLost 독, 다음 스텝·리셋 거절. --two(T7): 두 장치에
# 같은 전문가가 있는 계획은 업로드 전에 이름으로 거절. 프롬프트 배치(tierbatch, twoeng R4): --batch(B7) P = 512 배치 하나가
# 같은 적재에서 스텝으로 먹인 것과 비트까지 같다; --batch2(B7 그룹) P = 1024 배치 둘을 한 그룹으로 돌린 것이 기준 적재의 배치
# 호출과 비트까지 같고 그룹이 실제로 둘이었다; --bfault(B2) 층 카드 폴트를 올린 뒤의 프롬프트 호출이 같은 폴트로 끝나고 독,
# 리셋 뒤 스텝과 같다; --blost(B3) 층 스트림을 묶은 채 프롬프트 호출이 데드라인 안에 카드 상실 이름과 CardLost 독으로 끝난다.
# --bfeat(B8) 따로 한 번 더 적재해, 적재가 배치 버퍼를 만든 뒤 드래프트 특징 탭을 붙이고 prepare_prefill을 다시 부른다(--place bp가
# 드래프트를 여는 순서). P = 1024를 스텝으로 먹여 위치마다 읽은 특징과, 같은 id의 프롬프트 호출이 넘긴 특징(드래프트 창, 전 위치)이
# 위치·값 비트까지 같다; 전제: 배치 둘이 한 그룹, 넓은 호출은 두 배치 모두에서 행을 넘겼다. 드래프트 헤더는 프로필의 DSPARK_MODEL.
# The lost-card clauses --lost (T3), --blost (B3) and --bfirst (B3f) are not in the landing run: weekly-gpu-ds41-lost
# runs them. 3090, 게이트 락.
[group('v41-load')]
gate-gpu-ds41-tier *ARGS='--union --fault --two --batch --batch2 --bfault --bfeat':
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh 'export BLOOMERY_GATE_V41_LOAD=1 && cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features deepseek41 --release --bin gate_deepseek41_tier && __s=$(. tools/ref/ref-paths.sh && printf %s "$DSPARK_MODEL") && export BLOOMERY_DSPARK_MODEL="$__s" && bash tools/gpu-gate.sh gate_deepseek41_tier {{ARGS}}'

# V4.1 plan (b′) on both cards (--place bp): the stage on the A6000, the expert tier and the DSpark draft on the 3090.
# The reference is the same plan with every tier set joined to the stage card's (one card, no tier code); --loopback, by
# name only, adds the plan with the tier on the A6000 beside the stage (two Gpus on one card). The draft is on the 3090
# in every load; the plans are made under a card budget so the A6000 holds both sets; the stage keeps each layer's id
# prefix and the tier the next ids. --union: the prose prompt (32 ids) step by step, 48 greedy steps, then the prompt
# through the draft and 16 DSpark passes, every token, kept count and logits row bit for bit the reference's, the stage
# graph's node count too; precondition: every tier layer sent a routed slot, the passes kept and rejected a proposal.
# --lost: the tier's stream held behind a host flag, the step named as a lost card within the deadline, no token,
# CardLost, the next step refused; not in the landing run: weekly-gpu-ds41-lost runs it. Two loads, one after the
# other. Both cards (BLOOMERY_CARD=both: both gate locks), alone in a batch.
[group('solo')]
[group('v41-load')]
gate-gpu-ds41-twocard *ARGS='--union':
    BLOOMERY_MODEL=deepseek41 BLOOMERY_CARD=both ./tools/box.sh 'export BLOOMERY_GATE_V41_LOAD=1 && cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features deepseek41 --release --bin gate_deepseek41_twocard && __s=$(. tools/ref/ref-paths.sh && printf %s "$DSPARK_MODEL") && export BLOOMERY_DSPARK_MODEL="$__s" && bash tools/gpu-gate.sh gate_deepseek41_twocard {{ARGS}}'

# The lost-card clauses: gate_deepseek41_tier --bfirst --blost --lost (B3f, B3 and T3: the loopback's tier stream held
# behind a host flag before a fresh load's first prompt call, before a prompt call, before a step, on the 3090, each on
# a load of its own) and gate_deepseek41_twocard --lost (before a step on plan (b′), both cards, one load): the call or
# the step fails within the go deadline and its grace naming the lost card, the host tier is poisoned as a lost card
# (CardLost), the next call or step is refused by that poison and a reset by name. Weekly: `just weekly` runs it, and
# `just affected` names it when a file its triggers in tools/gate-paths.tsv match changes. Alone in a batch.
[group('solo')]
[group('v41-load')]
weekly-gpu-ds41-lost:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh 'export BLOOMERY_GATE_V41_LOAD=1 && cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features deepseek41 --release --bin gate_deepseek41_tier && bash tools/gpu-gate.sh gate_deepseek41_tier --bfirst --blost --lost'
    BLOOMERY_MODEL=deepseek41 BLOOMERY_CARD=both ./tools/box.sh 'export BLOOMERY_GATE_V41_LOAD=1 && cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features deepseek41 --release --bin gate_deepseek41_twocard && __s=$(. tools/ref/ref-paths.sh && printf %s "$DSPARK_MODEL") && export BLOOMERY_DSPARK_MODEL="$__s" && bash tools/gpu-gate.sh gate_deepseek41_twocard --lost'

# 상주 교체의 r8 → Q3_K 풀기(`ds41_r8_q3k_groups`, 그룹 하나를 블록 하나가 제자리에서): 무작위 r8 워드를 V4.1 파트
# 모양(2304행 × 20 슈퍼블록)과 홀수 폭 작은 모양으로 풀어 참조 커널 `ds41_r8_q3k`와 `qdot::unpack_q3k_r8`의 바이트와
# 같고 파트 밖 보호 워드를 건드리지 않는지, 잘못된 파트(16바이트 경계 밖, 21 슈퍼블록, 반 그룹)는 이름 붙은 오류인지 본다.
# 모델 파일 없음.
gate-gpu-ds41-unpack:
    ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features deepseek41 --release --bin gate_deepseek41_unpack && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_deepseek41_unpack'

# V4.1 host streaming in a prompt call (BLOOMERY_HOSTSTREAM, set on in the gate) over the residency at mid-p40-s1, plan
# (b′), groups of 2, the host set locked, both cards, one load. First the residency's own clauses (shared/ds41_residency.rs,
# streaming off): the host-set refusal at load (a host one byte short of the churn pool, named, before the load), then on
# the load c6 (a DSpark pair pass keeping one row folds what one step folds), c1 (the prose history twice, same tokens and
# logits, flips landed), the admitted experts' slots byte for byte a static load's (the r8 unpack), the host set's
# residency answers, the tier entries unmoved, the passes' kinds and kept rows (the prompt call one pass, 0 kept), c3
# (the copy stream held 1 s, same history), the serve seat's reset leaving the residency, c7 (a reset back to the seed,
# no host byte released). Then, streaming on: s2 (a call that streams, then the same call on the placement it left admits
# nothing and gives the same argmax and logits), s1 (a 1,536-id call and 8 steps free and with the copy stream held 1 s:
# the same picks per group and layer, tokens and logits), s3 (4 prose and 4 code windows of 512 ids, streaming off and
# on: where they first differ, on's top-1 margin below 1.5). Both cards, alone in a batch.
[group('solo')]
[group('v41-load')]
gate-gpu-ds41-callstream *ARGS:
    BLOOMERY_MODEL=deepseek41 BLOOMERY_CARD=both ./tools/box.sh 'export BLOOMERY_GATE_V41_LOAD=1 && cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features deepseek41 --release --bin gate_ds41_callstream && bash tools/gpu-gate.sh gate_ds41_callstream {{ARGS}}'

# The same residency clauses on plan (a), the serving plan: every layer and the head on the A6000, no tier card
# (gate_ds41_callstream --place a --only residency; the tier clause asks that the host map holds no tier entry), and
# the static clause, whose load of the dumped card table follows the residency load's teardown. The A6000 alone, the
# host set locked, one load at a time, alone in a batch.
[group('solo')]
[group('v41-load')]
gate-gpu-ds41-residency-a:
    BLOOMERY_MODEL=deepseek41 BLOOMERY_CARD=a6000 ./tools/box.sh 'export BLOOMERY_GATE_V41_LOAD=1 && cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features deepseek41 --release --bin gate_ds41_callstream && bash tools/gpu-gate.sh gate_ds41_callstream --place a --only residency'

# V4.1 long runs on the gate placement, three arms on one load, the host set populated and locked
# (BLOOMERY_HOST_LOCK=1 unless the environment says otherwise: populated pages alone are reclaimed under
# another process's reads). --faults first, the process's first steps after the load: 32 greedy graph
# steps after a reset, each step's faults outside the engram helper read around it; from step 2 on no
# major fault and at most the pinned minor ones. Then, every position through the finite probe before the
# engine steps it: --free, prompt row 7 then up to 330 greedy tokens; --trigger, prompt row 0 plus the 311
# fed ids whose last position selects six layer-34 experts that all score 0, then 16 greedy tokens. Red on
# a fault over the pin, a non-finite stream at any seam, a run of 8 equal generated tokens, an eager argmax
# that is not the engine's token, or a first difference with ik's greedy ids where our margin is not below
# 1.5. Solo: its fault pin is a host-memory count another lane's model load moves. ARGS go after the three
# arms (`-n N`). FAIL-first for the faults arm: BLOOMERY_BOX_ENV="BLOOMERY_HOST_POPULATE=0 BLOOMERY_HOST_LOCK=0".
# 3090, gate lock.
[group('solo')]
[group('v41-load')]
gate-gpu-ds41-long *ARGS:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh 'export BLOOMERY_GATE_V41_LOAD=1 && cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features deepseek41 --release --bin gate_deepseek41_long && BLOOMERY_HOST_LOCK=${BLOOMERY_HOST_LOCK:-1} bash tools/gpu-gate.sh gate_deepseek41_long --faults --free --trigger {{ARGS}}'

# The candidate mask past the 16,384 positions where it first selects, on the live model at the serving context (gate
# placement, 3090), two loads one after the other. G2, gate_deepseek41_long --candidates: 16,448 prose ids through the
# prompt batch, then 32 eager steps with the attention piece's candidate tap armed; at every step layer 20's kept blocks
# are the reference's selection over its own scores and each consumer's compaction and list hold to them, and some list
# the mask changed. G2b, gate_deepseek41_prefill --cand: positions 16,300-16,500 as one batch call and as 200 steps
# from the same keep point, every ring, compressor state, compressed row, index key and the last logits bit for bit.
# The landing batch runs no model-load prefill past 16,384: the wiring at depth lands on gate-gpu-ds41-chain-attn's
# synthetic G2s, and below the bound every candidate launch is a no-op, so the batch-set indexing past it is proved
# here alone. Weekly: `just weekly` runs it, and `just affected` names it when a file its triggers in
# tools/gate-paths.tsv match changes. Alone in a batch.
[group('solo')]
[group('v41-load')]
weekly-gpu-ds41-cand:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh 'export BLOOMERY_GATE_V41_LOAD=1 && cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features deepseek41 --release --bin gate_deepseek41_long --bin gate_deepseek41_prefill && BLOOMERY_HOST_LOCK=${BLOOMERY_HOST_LOCK:-1} bash tools/gpu-gate.sh gate_deepseek41_long --candidates && __s=$(. tools/ref/ref-paths.sh && printf %s "$DSPARK_MODEL") && export BLOOMERY_DSPARK_MODEL="$__s" && bash tools/gpu-gate.sh gate_deepseek41_prefill --cand'

# ik가 V4.1 파일로 prompts.tsv의 행 PROMPT를 greedy로 잇는다(CPU, 디코드마다 토큰 하나, CPU 임대 안) — long 게이트
# --free의 참조, $BLOOMERY_DATA/greedy-ds41/. 행 0은 세 토큰 만에 EOS라 긴 비교는 행 7. 러너 머리말 참조.
ik-greedy-ds41 PROMPT='0':
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh 'PROMPT={{PROMPT}} bash tools/ref/ik-greedy.sh'

# G4: ik의 KLD 기준 파일 TAG($BLOOMERY_DATA/ikppl/TAG.kld)에 대한 우리 NLL·KLD·top-1 — 청크마다 리셋하고 id마다 엔진
# 스텝 하나. 512 × 16청크는 약 8,200스텝이라 한도를 28분으로 올린다. 빨강은 Δ_PPL > +1.5 %.
[group('v41-load')]
run-ds41-ppl TAG:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh 'export BLOOMERY_GATE_V41_LOAD=1 && cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features deepseek41 --release --bin gate_deepseek41_step && BLOOMERY_GATE_BOUND=1680 bash tools/gpu-gate.sh gate_deepseek41_step --ppl {{TAG}}'

# V4.1 decode CLI, functional run (no timing): generate_ds41 feeds the prompt one real step per id, then greedy
# -n tokens, and prints them. The recipe loads the step gate's placement on the 3090 (--place gate), so its tokens
# are comparable with gate-gpu-ds41-long's free arm; later arguments override it (a flag given twice takes its last value) —
# `BLOOMERY_CARD=a6000 just gen-ds41 --place a …` loads the serving placement (a) on the A6000. 3090, gate lock.
[group('v41-load')]
gen-ds41 *ARGS:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh 'export BLOOMERY_GATE_V41_LOAD=1 && cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features deepseek41 --release --bin generate_ds41 && bash tools/gpu-gate.sh generate_ds41 --place gate {{ARGS}}'

# The V4.1 binaries' record schemas (crates/gpu-gates/src/record.rs) into tools/bloomery/schema/, which record.rs's
# checked_in_schemas_are_current holds to the binaries', and the engine's plans the flow model reads
# (generate_ds41 --plan, placement (a), P 128/256/384/512/1536/4096/16384, CED on and off) into tools/flow/plans/:
# P 1536 is weekly-gpu-ds41-flowcounts' three-batch arm, P 16384 the prefill headline's prompt. Loads nothing onto a card;
# runs with the A6000 in view, since --place a plans the stage on the largest visible card.
records-refresh:
    BLOOMERY_MODEL=deepseek41 BLOOMERY_CARD=a6000 ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features deepseek41 --release --bin generate_ds41 --bin bloomery-chat --bin bloomery-serve-ds41 --bin bloomery-serve-qwen38 --bin gate_deepseek41_prefill >&2 && cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features glm5next --release --bin generate_glm5next >&2 && cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin generate_qwen3moe --bin clef_hidden >&2 && for b in generate_ds41 bloomery-chat bloomery-serve-ds41 bloomery-serve-qwen38 gate_deepseek41_prefill generate_glm5next generate_qwen3moe clef_hidden; do target/release/$b --records-schema; done && for P in 128 256 384 512 1536 4096 16384; do for c in on off; do echo "#> tools/flow/plans/ds41-p$P-ced-$c.rec generate_ds41 --plan --depth $P --place a under BLOOMERY_CED=$c" && BLOOMERY_CED=$c target/release/generate_ds41 --plan --depth $P --place a; done; done' | python3 tools/bloomery/records.py refresh

# The flow model's queue entries held to the engine's (3090, placement gate): generate_ds41 -n 2 under
# BLOOMERY_STEP_STATS=1 prints its counter (`stat prefill front`, `stat prefill lb`), at --depth 512 once at the default
# group and once under BLOOMERY_PREFILL_GROUP=1, and at --depth 1536 at the default group: three batches, which the
# group lever of 2 runs as one group (a lone last batch joins the group before it), so the model's group wrap and its
# fold are held too. Each is a tools/gpu-gate.sh run whose log goes to a file; then tools/flow/ds41_prefill.py --counts on
# each log, which compares it layer-batch by layer-batch with the model's config for the log's `call plan` (PG2 or
# PG1) over the engine's plan of that P (tools/flow/plans/, `just records-refresh`). Red unless all three print
# `counts: equal`. Three loads; logs and --counts output in target/flowcounts-gate/. Weekly: `just weekly` runs it,
# and `just affected` names it when a file its triggers in tools/gate-paths.tsv match changes.
[group('v41-load')]
weekly-gpu-ds41-flowcounts:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh 'export BLOOMERY_GATE_V41_LOAD=1 && cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features deepseek41 --release --bin generate_ds41 && D=target/flowcounts-gate && rm -rf $D && mkdir -p $D && BLOOMERY_STEP_STATS=1 bash tools/gpu-gate.sh generate_ds41 --place gate --depth 512 -n 2 > $D/g-default.log && BLOOMERY_STEP_STATS=1 BLOOMERY_PREFILL_GROUP=1 bash tools/gpu-gate.sh generate_ds41 --place gate --depth 512 -n 2 > $D/g1.log && BLOOMERY_STEP_STATS=1 bash tools/gpu-gate.sh generate_ds41 --place gate --depth 1536 -n 2 > $D/g-1536.log && { python3 tools/flow/ds41_prefill.py --counts $D/g-default.log > $D/g-default.counts; rd=$?; python3 tools/flow/ds41_prefill.py --counts $D/g1.log > $D/g1.counts; r1=$?; python3 tools/flow/ds41_prefill.py --counts $D/g-1536.log > $D/g-1536.counts; r3=$?; cat $D/g-default.counts $D/g1.counts $D/g-1536.counts; echo "--counts rc: default group $rd, BLOOMERY_PREFILL_GROUP=1 $r1, P 1536 default group $r3"; [ "$rd" = 0 ] && [ "$r1" = 0 ] && [ "$r3" = 0 ] && grep -qx "counts: equal" $D/g-default.counts && grep -qx "counts: equal" $D/g1.counts && grep -qx "counts: equal" $D/g-1536.counts && echo "weekly-gpu-ds41-flowcounts: PASS"; }'

# The served draft, bit for bit (3090, placement gate): generate_ds41 under BLOOMERY_DRAFT=lookup must emit the plain
# run's tokens. One prompt, both arms, -n 64: the first 128 ids of the code corpus. Red unless the two `tokens` lines are
# identical and the draft arm both kept and rejected a proposal (0 < accepts < proposals in its draft summary).
# Two loads, each its own tools/gpu-gate.sh run: the gate compares the CLI's two arms as shipped; logs in target/draft-gate/.
[group('v41-load')]
gate-gpu-ds41-draft:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh 'export BLOOMERY_GATE_V41_LOAD=1 && cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features deepseek41 --release --bin generate_ds41 && D=target/draft-gate && rm -rf $D && mkdir -p $D && C=$(head -n 128 "$BLOOMERY_DATA/engram/corpus-code.ids" | paste -sd, -) && bash tools/gpu-gate.sh generate_ds41 --place gate --tokens "$C" -n 64 > $D/code-plain.log && BLOOMERY_DRAFT=lookup bash tools/gpu-gate.sh generate_ds41 --place gate --tokens "$C" -n 64 > $D/code-draft.log && grep -h "^draft summary " $D/code-draft.log && grep "^tokens " $D/code-plain.log > $D/code-plain.tok && grep "^tokens " $D/code-draft.log > $D/code-draft.tok && echo "code plain $(cat $D/code-plain.tok)" && echo "code draft $(cat $D/code-draft.tok)" && cmp $D/code-plain.tok $D/code-draft.tok && echo "code: tokens identical" && S=$(grep "^draft summary " $D/code-draft.log) && P=$(echo "$S" | sed -n "s/.*proposals=\([0-9]*\).*/\1/p") && A=$(echo "$S" | sed -n "s/.* accepts=\([0-9]*\).*/\1/p") && [ "$A" -gt 0 ] && [ "$A" -lt "$P" ] && echo "code: accepts $A of $P proposals: both branches ran" && echo "gate-gpu-ds41-draft: PASS"'

# The DSpark loop (3090, placement gate). gate_deepseek41_dsloop: the target's feature tap of the draft's target_layers
# (header only) — node counts pinned, each mean right after the MoE join before its tapped layer, and its features
# equal, bit for bit, the host mean of the eagerly read streams for the eager step, the graph step and both rows of
# a pair pass. Then generate_ds41 prompt row 0, -n 64, plain and BLOOMERY_DRAFT=dspark, both under the same card
# budget so the 8.5 GB draft fits beside the target on the 3090: red unless the tokens lines are identical and the
# draft summary's accepts are neither 0 nor every proposal. Logs in target/dsloop-gate/.
[group('v41-load')]
gate-gpu-ds41-dspark-loop:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh 'export BLOOMERY_GATE_V41_LOAD=1 && cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features deepseek41 --release --bin generate_ds41 --bin gate_deepseek41_dsloop && __s=$(. tools/ref/ref-paths.sh && printf %s "$DSPARK_MODEL") && export BLOOMERY_DSPARK_MODEL="$__s" && bash tools/gpu-gate.sh gate_deepseek41_dsloop --structure --tap && D=target/dsloop-gate && rm -rf $D && mkdir -p $D && export BLOOMERY_CARD_BUDGET=13G && bash tools/gpu-gate.sh generate_ds41 --place gate --prompt-id 0 --ctx 4096 -n 64 > $D/plain.log && BLOOMERY_DRAFT=dspark bash tools/gpu-gate.sh generate_ds41 --place gate --prompt-id 0 --ctx 4096 -n 64 > $D/dspark.log && grep -h "^load draft=\|^draft summary " $D/dspark.log && grep "^tokens " $D/plain.log > $D/plain.tok && grep "^tokens " $D/dspark.log > $D/dspark.tok && echo "plain $(cat $D/plain.tok)" && echo "dspark $(cat $D/dspark.tok)" && cmp $D/plain.tok $D/dspark.tok && echo "tokens identical" && S=$(grep "^draft summary " $D/dspark.log) && P=$(echo "$S" | sed -n "s/.*proposals=\([0-9]*\).*/\1/p") && A=$(echo "$S" | sed -n "s/.* accepts=\([0-9]*\).*/\1/p") && [ "$A" -gt 0 ] && [ "$A" -lt "$P" ] && echo "accepts $A of $P proposals: both branches ran" && echo "gate-gpu-ds41-dspark-loop: PASS"'

# V4.1 배치 프리필(`body::prefill`)을 게이트 배치에서: corpus-prose.ids 앞부분으로 4096스텝 디코드 오라클을 한 번 뜨고(DSpark
# 드래프트의 특징 탭 포함), P ∈ {1, 127, 128, 129, 511, 512, 513, 2300, 4096}과 분할 700+400·1800+1000·300+2700마다 모든 층의 창 링·
# 섀도 행·압축 행·인덱스 키·압축기 상태, 탭 행, logits가 P번 스텝과 비트 동일해야 한다. 3090, 게이트 락. `--cases a,b`·
# `--no-split`은 FAIL-first용 부분 실행, `--seams P`는 판정이 아니라 첫 어긋난 이음매를 찾는 로케이터.
[group('v41-load')]
gate-gpu-ds41-prefill *ARGS='':
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh 'export BLOOMERY_GATE_V41_LOAD=1 && cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features deepseek41 --release --bin gate_deepseek41_prefill && __s=$(. tools/ref/ref-paths.sh && printf %s "$DSPARK_MODEL") && export BLOOMERY_DSPARK_MODEL="$__s" && bash tools/gpu-gate.sh gate_deepseek41_prefill {{ARGS}}'

# V4.1 텍스트 쪽 이미지 주입(`body::prefill_media`, `Session::prompt_media`)을 게이트 배치에서: 공식 비전 오라클 세트
# (deepseek41v, `grad-448`)의 aligner 행과 mmproj 구분자 행으로 스팬을 만들어 세 배치(프롬프트 한가운데, T_MAX 경계를
# 걸침, 프롬프트를 닫음)에 먹이고 조항 (i)–(vi)을 본다: 미디어 자리 임베딩=행 넓힌 값·텍스트 자리=스팬 없는 호출(비트),
# 라우터 id=카드 자체 점수의 호스트 select_n(bias_vl/bias, 커버리지 단언 포함), engram 층의 스트림 항등(−0 예외),
# 분할 불변(한 호출=스팬 앞 텍스트에서 끊은 두 호출, 상태 해시+24 디코드), 이미지로 끝나는 프롬프트 뒤 창의 dead와
# 스냅숏/바이트 재개, 세 거부의 이름. 3090, 게이트 락.
[group('v41-load')]
gate-gpu-ds41-media *ARGS='':
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh 'export BLOOMERY_GATE_V41_LOAD=1 && cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features deepseek41 --release --bin gate_ds41_media && bash tools/gpu-gate.sh gate_ds41_media {{ARGS}}'

# Text in, text out (3090, placement gate). Prompt rows 0 and 7: bloomery-chat --greedy on the row's text must
# tokenize to the row's ids (llama-tokenize's); row 0 must stop at the end-of-generation id (stop=eog); on row 7 its ids
# must be the start of generate_ds41 --tokens <those ids> -n 16, and it must run to stop=length, so that leg compares
# every id; the streamed text must equal the vocabulary's one-shot decode (text_consistent=true); a sampled run at
# --seed 7 on row 0, twice, must print the same ids. Five loads, each its own tools/gpu-gate.sh run; logs in
# target/chat-gate/.
[group('v41-load')]
gate-gpu-ds41-chat:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh 'export BLOOMERY_GATE_V41_LOAD=1 && cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features deepseek41 --release --bin generate_ds41 --bin bloomery-chat && D=target/chat-gate && rm -rf $D && mkdir -p $D && R=$BLOOMERY_DATA/greedy-ds41/prompt0.tsv && T=$(grep -v "^#" $R | head -n 1 | cut -f2) && I=$(grep -v "^#" $R | head -n 1 | cut -f3) && bash tools/gpu-gate.sh bloomery-chat --greedy --place gate --prompt "$T" -n 16 > $D/p0-chat.out 2> $D/p0-chat.err && echo "[$I]" | sed "s/,/, /g" > $D/p0-row.ids && sed -n "s/^prompt_ids //p" $D/p0-chat.err > $D/p0-chat.ids && sed -n "s/^ids //p" $D/p0-chat.err > $D/p0-chat.tok && echo "p0 row ids     $(cat $D/p0-row.ids)" && echo "p0 chat ids    $(cat $D/p0-chat.ids)" && echo "p0 chat greedy $(cat $D/p0-chat.tok)" && grep -h "^chat: \|^text_consistent=" $D/p0-chat.err && echo "p0 text: $(cat $D/p0-chat.out)" && R=$BLOOMERY_DATA/greedy-ds41/prompt7.tsv && T=$(grep -v "^#" $R | head -n 1 | cut -f2) && I=$(grep -v "^#" $R | head -n 1 | cut -f3) && bash tools/gpu-gate.sh bloomery-chat --greedy --place gate --prompt "$T" -n 16 > $D/p7-chat.out 2> $D/p7-chat.err && bash tools/gpu-gate.sh generate_ds41 --place gate --tokens "$I" -n 16 > $D/p7-gen.log && echo "[$I]" | sed "s/,/, /g" > $D/p7-row.ids && sed -n "s/^prompt_ids //p" $D/p7-chat.err > $D/p7-chat.ids && sed -n "s/^ids //p" $D/p7-chat.err > $D/p7-chat.tok && sed -n "s/^tokens //p" $D/p7-gen.log > $D/p7-gen.tok && echo "p7 row ids     $(cat $D/p7-row.ids)" && echo "p7 chat ids    $(cat $D/p7-chat.ids)" && echo "p7 chat greedy $(cat $D/p7-chat.tok)" && echo "p7 generate    $(cat $D/p7-gen.tok)" && grep -h "^chat: \|^text_consistent=" $D/p7-chat.err && echo "p7 text: $(cat $D/p7-chat.out)" && T=$(grep -v "^#" $BLOOMERY_DATA/greedy-ds41/prompt0.tsv | head -n 1 | cut -f2) && bash tools/gpu-gate.sh bloomery-chat --place gate --seed 7 --prompt "$T" -n 16 > $D/seed7a.out 2> $D/seed7a.err && bash tools/gpu-gate.sh bloomery-chat --place gate --seed 7 --prompt "$T" -n 16 > $D/seed7b.out 2> $D/seed7b.err && sed -n "s/^ids //p" $D/seed7a.err > $D/seed7a.tok && sed -n "s/^ids //p" $D/seed7b.err > $D/seed7b.tok && echo "seed 7 a $(cat $D/seed7a.tok)" && echo "seed 7 b $(cat $D/seed7b.tok)" && echo "seed 7 text: $(cat $D/seed7a.out)" && cmp $D/p0-row.ids $D/p0-chat.ids && echo "p0 prompt ids: identical to the row" && test -s $D/p0-chat.tok && grep -q "^chat: .* stop=eog$" $D/p0-chat.err && echo "p0: stopped at the end-of-generation id" && grep -qx "text_consistent=true" $D/p0-chat.err && echo "p0 text: consistent with the decode" && cmp $D/p7-row.ids $D/p7-chat.ids && echo "p7 prompt ids: identical to the row" && test -s $D/p7-chat.tok && case "$(tr -d "[] " < $D/p7-gen.tok)," in "$(tr -d "[] " < $D/p7-chat.tok),"*) echo "p7 greedy ids: a prefix of generate_ds41 (the whole of it unless stop=eog)" ;; *) false ;; esac && grep -qx "text_consistent=true" $D/p7-chat.err && echo "p7 text: consistent with the decode" && grep -q "^chat: .* stop=length$" $D/p7-chat.err && echo "p7: ran to -n, every id compared" && grep -qx "text_consistent=true" $D/seed7a.err && test -s $D/seed7a.tok && cmp $D/seed7a.tok $D/seed7b.tok && echo "seed 7: ids identical" && echo "gate-gpu-ds41-chat: PASS"'

# The HTTP server on the V4.1 engine (3090, placement gate). Prompt row 0: generate_ds41 --tokens <the row's ids> -n 16
# under its own gate-lock hold, then gate_ds41_serve under another: it starts bloomery-serve-ds41 --port 0 --place gate,
# and checks /completion's ids at temperature 0 against generate_ds41's (all 16, or a prefix ending in the
# end-of-generation id), a chat turn streamed and not streamed (same content, [DONE] last), /tokenize of the row's text
# against the row's ids, the same /completion again after those (reset leaves nothing behind), and /props' engine
# object (name and version, the file's header facts, each device's bytes = the plan's, the KV bytes); then kills the
# server it spawned and waits for it. Then the same server under BLOOMERY_DRAFT=dspark with the draft on the A6000
# (BLOOMERY_DSPARK_CARD=A6000, the profile's DSPARK_MODEL): gate_ds41_serve --plain holds its greedy ids to the plain
# server's and generate_ds41's, and checks /props' draft object and row, the draft counts in timings and /metrics, the
# sampled and ignore_eos requests served by plain steps with no draft counted, and prefix reuse under the draft
# (a continuation, a cut, a resume) against
# cache_prompt: false. Then plan (b′) with the draft (--place bp: the A6000 plan (a), the 3090 the expert tier and the
# draft): generate_ds41 --place bp under the draft, then gate_ds41_serve --place bp holds the server's greedy ids to it
# (both under BLOOMERY_RESIDENCY=off: the fixed placement, whose card and host rounding do not move with a flip's timing) and
# checks /props (the A6000, the tier card with the plan's tier bytes and the draft's class, the host) and the draft counts
# in timings. Both cards in view (BLOOMERY_CARD=both: gpu-gate.sh takes both cards' gate locks, and the batch runs it
# alone). Five loads; logs and the raw stream in target/serve-gate/, the draft run's in target/serve-gate/draft/, the
# two-card run's in target/serve-gate/bp/. Weekly: `just weekly` runs it, and `just affected` names it when a file its
# triggers in tools/gate-paths.tsv match changes.
[group('solo')]
[group('v41-load')]
weekly-gpu-ds41-serve:
    BLOOMERY_MODEL=deepseek41 BLOOMERY_CARD=both ./tools/box.sh 'export BLOOMERY_GATE_V41_LOAD=1 && cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features deepseek41 --release --bin generate_ds41 --bin bloomery-serve-ds41 --bin gate_ds41_serve && D=target/serve-gate && rm -rf $D && mkdir -p $D && R=$BLOOMERY_DATA/greedy-ds41/prompt0.tsv && T=$(grep -v "^#" $R | head -n 1 | cut -f2) && I=$(grep -v "^#" $R | head -n 1 | cut -f3) && bash tools/gpu-gate.sh generate_ds41 --place gate --tokens "$I" -n 16 > $D/gen.log && bash tools/gpu-gate.sh gate_ds41_serve --gen $D/gen.log --prompt "$T" --ids "$I" --dir $D && __s=$(. tools/ref/ref-paths.sh && printf %s "$DSPARK_MODEL") && export BLOOMERY_DSPARK_MODEL="$__s" && BLOOMERY_DRAFT=dspark BLOOMERY_DSPARK_CARD=A6000 bash tools/gpu-gate.sh gate_ds41_serve --gen $D/gen.log --prompt "$T" --ids "$I" --dir $D/draft --plain $D && mkdir -p $D/bp && BLOOMERY_RESIDENCY=off BLOOMERY_DRAFT=dspark bash tools/gpu-gate.sh generate_ds41 --place bp --tokens "$I" -n 16 > $D/bp/gen.log && BLOOMERY_RESIDENCY=off BLOOMERY_DRAFT=dspark bash tools/gpu-gate.sh gate_ds41_serve --place bp --gen $D/bp/gen.log --prompt "$T" --ids "$I" --dir $D/bp'

# Qwen3.8's adaptive expert residency on one card (BLOOMERY_RESIDENCY set in the gate at mid-p<P>-s1, P from the plan's
# card experts): the churn pool's host refusal at load, a verify keeping its counted rows only, the same history twice
# with flips landed, every admitted slot byte for byte a static load's, the passes' kinds and kept rows (the prompt one
# pass keeping 0, each step 1, a verify its accepted rows), and a reset back to the seed. Plan (a) on the A6000 (the
# gate plans with the A6000 machine). Loads the whole host set: alone in a batch, under the big-load lock.
[group('solo')]
[group('v41-load')]
gate-gpu-qwen38-residency:
    BLOOMERY_MODEL=qwen4exp BLOOMERY_CARD=a6000 ./tools/box.sh 'export BLOOMERY_GATE_V41_LOAD=1 && cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin gate_qwen38_residency && bash tools/gpu-gate.sh gate_qwen38_residency'

# Qwen3.8 plan (b′) on both cards (--place bp): the stage on the A6000, the expert tier on the 3090, its decode walks (a
# step and the 2-, 3- and 4-row verifies) and its ubatch prompt walk serving the tier. The reference is the same plan
# with every tier set joined to the stage card's (one card, no tier code), the plans made under a card budget so the
# A6000 holds both sets. --union: 32 prose ids step by step, greedy steps, six verifies, greedy steps, then a 600-id
# prompt call and 8 greedy steps, every token and logits row bit for bit the reference's; the stage graphs' nodes the
# reference's less one a tier layer, the tier's graphs eight a tier layer and the fault copy; preconditions: every tier
# layer sent a routed slot, the prompt's tier services one a tier layer. --residency: the tier load under mid-p<P>-s1,
# the history twice the same with flips landed, its ids the residency-off run's or parted at a near tie, the streamed
# 600-id prompt call by the same rule, the tier's set the live map's tier rows, no tier expert's bytes in the host set.
# --prompt4k: plan (b′) and its union at 4,352 positions, a 4,096-id prompt call (the full ubatch window) and 8 greedy
# steps bit for bit. --lost: the tier's stream held behind a host flag, the step named as a lost card, CardLost, the
# next step refused. Up to five loads, one after the other. Both cards (BLOOMERY_CARD=both: both gate locks), alone in a
# batch.
[group('solo')]
[group('v41-load')]
gate-gpu-qwen38-twocard *ARGS='--union --residency --prompt4k':
    BLOOMERY_MODEL=qwen4exp BLOOMERY_CARD=both ./tools/box.sh 'export BLOOMERY_GATE_V41_LOAD=1 && cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin gate_qwen38_twocard && bash tools/gpu-gate.sh gate_qwen38_twocard {{ARGS}}'

# The HTTP server on the Qwen3.8 engine (3090, placement gate). The prompt is the profile's five-id text
# ("The capital of France is") and its ids the profile's REF_TOKENS — ik's llama-tokenize on this model's
# vocabulary, never our own tokenizer (the gate's /tokenize clause would be circular; the derivation command is
# the profile comment's) — and five is at most the eight the gate needs: below Prompt38::GEMM_FROM both engines
# feed the prompt by bit-for-bit passes, so the server's greedy ids are generate_qwen3moe's; at nine or more each
# runs the prompt's last position through a different arm of the ubatch walk. generate_qwen3moe --tokens <those
# ids> -n 16 under its own gate-lock hold, then gate_qwen38_serve under another: it starts bloomery-serve-qwen38
# --port 0 --place gate, and checks /props' engine object against its own plan of the file (architecture
# qwen4exp, each device's bytes = the plan's, the KV bytes), /completion's ids at temperature 0 against
# generate_qwen3moe's (all 16, or a prefix ending in the end-of-generation id), the same /completion again and
# once more after the other requests (a request keeps no prefix: every request prefills from a reset), a chat
# turn streamed and not streamed (same content, [DONE] last), /tokenize of the prompt against REF_TOKENS, and a
# prompt of the served context a 400 the server survives; then one server alone under BLOOMERY_RESIDENCY=mid-p0-s1:
# its residency records and passes, a reset of diff 0, the same request after it with the first one's ids. Three
# loads; logs and the raw stream in
# target/q38-serve-gate/. The build takes the deepseek41 feature, not the qwen family's plain gpu:
# the server surface it links (gpu-gates' bind, serve_client; the serve and sampler crates) sits behind
# that feature today — `just affected` and the recipes.py pins scope it so.
[group('solo')]
[group('v41-load')]
gate-gpu-qwen38-serve:
    BLOOMERY_MODEL=qwen4exp ./tools/box.sh 'export BLOOMERY_GATE_V41_LOAD=1 && . tools/ref/ref-paths.sh && cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features deepseek41 --release --bin generate_qwen3moe --bin bloomery-serve-qwen38 --bin gate_qwen38_serve && D=target/q38-serve-gate && rm -rf $D && mkdir -p $D && T="The capital of France is" && I=$REF_TOKENS && echo "prompt: $T" && echo "ids: $I" && bash tools/gpu-gate.sh generate_qwen3moe --place gate --tokens "$I" -n 16 > $D/gen.log && bash tools/gpu-gate.sh gate_qwen38_serve --gen $D/gen.log --prompt "$T" --ids "$I" --dir $D'

# The qwen3 seat of bloomery-serve (--model qwen3 -m <file>) against generate_qwen3moe, on the qwen3moe profile's file
# (Qwen3-30B-A3B, the MODEL box.sh exports under BLOOMERY_MODEL=qwen3moe) and the qwen35moe profile's (Qwen3.6-35B-A3B,
# that profile's default MODEL, read from tools/ref/models/qwen35moe.sh): per file, one server, then one CLI, one process
# at a time under one gate-lock hold. The server's /completion of a chat turn rendered by its own template (past the
# eight ids a pass takes) gives generate_qwen3moe --tokens <those ids> -n 64 --last-step's ids, and its
# /v1/chat/completions of the same turn is those ids (gate_qwen3_serve's header has the clauses). Logs in
# target/qwen3-serve-gate/<n>/. The build takes glm5next: bloomery-serve links every seat's device bundle.
gate-gpu-qwen3-serve:
    BLOOMERY_MODEL=qwen3moe ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features glm5next,clef --release --bin bloomery-serve --bin generate_qwen3moe --bin gate_qwen3_serve && D=target/qwen3-serve-gate && rm -rf $D && mkdir -p $D && Q36=$(sed -n "s/^MODEL=\${BLOOMERY_REF_MODEL:-\(.*\)}$/\1/p" tools/ref/models/qwen35moe.sh) && test -n "$Q36" && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_qwen3_serve --model "$BLOOMERY_REF_MODEL" --model "$Q36" --dir $D'

# bloomery-serve-qwen38 on the box, for a person to attach a client to (toktape records from it): the A6000 by
# default (SERVE_PLACE=gate for the 3090), the port 8080 unless SERVE_PORT, a four-hour gate bound unless
# BLOOMERY_GATE_BOUND, further ARGS pass through (--alias, --ctx-size, --chat-template-file, and the Qwen3.8 levers
# through the environment). gpu-gate.sh takes the card's gate lock, so no other gate lands on the card it serves
# from, and bounds the run; the address is on the server's stderr.
serve-qwen38 *ARGS:
    BLOOMERY_MODEL=qwen4exp BLOOMERY_CARD={{serve_card}} ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features deepseek41 --release --bin bloomery-serve-qwen38 && BLOOMERY_GATE_BOUND=${BLOOMERY_GATE_BOUND:-14400} bash tools/gpu-gate.sh bloomery-serve-qwen38 --port ${SERVE_PORT:-8080} --place {{serve_place}} {{ARGS}}'

# The same CLI's per-step ms (lead-only): placement (a) on the A6000 under the machine-wide lease, witness blocks
# around it (tools/ref/time-gate.sh). Example: `just time-gpu-ds41 --depth 6 -n 96`.
time-gpu-ds41 *ARGS:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh '{{precheck}} && cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features deepseek41 --release --bin generate_ds41 && bash tools/ref/time-gate.sh generate_ds41 {{ARGS}} --time'

# ik's V4.1 decode on the first 512 ids of corpus-<CORPUS>.ids, plain or with the DSpark draft, under the lease on
# the A6000 (lead-only): the ik twin of `just time-gpu-ds41 --tokens <those ids> -n N`. tools/ref/ik-draft.sh's header
# has the prompt round-trip checks, ik's command line and the summary line. Example: `just time-ik-draft prose 96 dspark`.
time-ik-draft CORPUS N ARM="dspark":
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh '{{precheck}} && cargo build --release -p bloomery-tokenizer --bin bloomery-tokenize && bash tools/ref/ik-draft.sh {{CORPUS}} {{N}} {{ARM}}'

# mainline llama.cpp(V4.1 포트, 프로필의 LCPP)로 같은 corpus-<CORPUS>.ids 첫 512 id 프롬프트를 A6000에서 임대 안에 디코드한다(리드 전용):
# `just time-gpu-ds41 --tokens <그 id> -n N`의 mainline 쌍둥이, llama-completion에 LCPP_CLI_FLAGS. 프롬프트 왕복 검사 둘과 요약 줄은
# tools/ref/ik-draft.sh의 lcpp 팔 그대로다. 예: `just time-lcpp-prompt prose 96`.
time-lcpp-prompt CORPUS N:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh '{{precheck}} && cargo build --release -p bloomery-tokenizer --bin bloomery-tokenize && bash tools/ref/ik-draft.sh {{CORPUS}} {{N}} lcpp'

# V4.1 decode by depth, three engines in one lease on the A6000 (lead-only): our arms `<D>` (generate_ds41 --depth D,
# placement (a)), ik's `ik:<D>` (llama-bench -gp D,96 at the profile's IK_GPU_FLAGS) and mainline's `lcpp:<D>`
# (llama-bench -d D at LCPP_GPU_FLAGS; `lcpp<K>:<D>` sweeps --n-cpu-moe K), alternated, rounds rotated, with per-depth
# ours/reference ratios. tools/ref/depth-ds41.sh's header has the arms, the placement difference and the environment levers.
# BLOOMERY_BOX_ENV='BLOOMERY_AB_ORDER=blocks' runs the arms in engine blocks instead (ours, each corpus, each bin:,
# lcpp, ik), each block's rounds together with its arms rotated inside it, and one discarded DISCARD r0 process first
# (a reference block's arm with the most token draws at its largest --n-cpu-moe, so its engram rows are warm; ours'
# longest prompt); the preheat is then off unless BLOOMERY_PREHEAT=1. The recipe passes BLOOMERY_AB_ROUNDS alone from
# the Mac's environment: every other runner variable goes through BLOOMERY_BOX_ENV.
# Prefill: every `<D>` row also carries pp_tok/s from generate_ds41's `time prompt` row (the D fed steps), and
# `ikpp:<P>`/`lcpppp:<P>` run llama-bench -p P -n 0 at the same flags (`ikpp<U>`/`lcpppp<U>`: -ub U), with a per-P table.
# `prose:<P>[@NAME=VALUE,...]` is ours on the first P ids of corpus-prose.ids (--tokens), in prose-only tables;
# `code:<P>` the same on corpus-code.ids. BLOOMERY_GEN_PLACE=a|gate picks our arms' --place (gate: the 3090 as
# BLOOMERY_TIMING_GPU); under the default order each reference arm's host set is preheated first (BLOOMERY_PREHEAT=0:
# off), rows carry majflt with the measured window's count (lcpp arms through llama-bench --progress) and [cold], and a
# failed arm is a FAIL row the runner goes past (rc 1 at the end). An ours arm under --place a or bp with no
# BLOOMERY_RESIDENCY runs the residency and prompt streaming (the lever's placement default; `@BLOOMERY_RESIDENCY=off`
# is the fixed placement's arm), and the runner holds each arm to what its binary's `residency host` record says.
# With no ours arm generate_ds41 is not built, nor under BLOOMERY_BOX_ENV=BLOOMERY_DRY=1 (the command lines, no lease);
# no arms means the runner's default (6 ik:6), which builds.
depth-gpu-ds41 *ARMS:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh "${BLOOMERY_AB_ROUNDS:+export BLOOMERY_AB_ROUNDS=$BLOOMERY_AB_ROUNDS && }"'{{precheck}} && { ours=; [ -n "{{ARMS}}" ] || ours=1; for a in {{ARMS}}; do case $a in prose:* | code:*) ours=1 ;; *:*) ;; *) ours=1 ;; esac; done; if [ -n "$ours" ] && [ -z "${BLOOMERY_DRY:-}" ]; then cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features deepseek41 --release --bin generate_ds41; fi; } && bash tools/ref/depth-ds41.sh {{ARMS}}'

# Qwen3-30B-A3B 전 카드 디코드를 깊이별로(A6000, 한 임대, 리드 전용): 우리 `<D>`(프롬프트를 실제 D스텝으로 먹임), ik `ik:<D>`(-gp D,96, 프로필
# 플래그)·`ikdef:<D>`(llama-bench 기본값), mainline `lcpp:<D>`(-d D)를 바퀴마다 순서를 돌려 번갈아 재고, 깊이마다 ours/각 참조 비율을 찍는다.
# 팔·플래그 근거·뺀 플래그는 tools/ref/depth-qwen3moe.sh 머리에. 우리 팔이 없으면 generate_qwen3moe를 빌드하지 않는다; 인자 없으면 6 ik:6 lcpp:6.
# mistral.rs arms: `mrs:<D>` (mistralrs bench --depth D, PagedAttention pool at the cache height) and `mrspa0:<D>` (--paged-attn off).
# Prefill: every `<D>` row also carries pp_tok/s from generate_qwen3moe's `time prompt` row, and `ikpp:<P>`/`lcpppp:<P>`
# (llama-bench -p P -n 0; `ikpp<U>`/`lcpppp<U>`: -ub U) and `mrspp:<P>` (mistralrs bench --prompt-len P) with a per-P table.
# `prose:<P>[@NAME=VALUE,...]`는 우리 팔이다. 이 프로필의 $BLOOMERY_DATA/qwen3moe/corpus-prose.ids에서 앞 P개 id를
# --tokens로 넣는다(행 라벨 ours@prose). prose 팔은 같은 P의 prose 팔하고만, 자기 표에서 비교한다.
# Under BLOOMERY_BOX_ENV=BLOOMERY_DRY=1 the runner prints the command lines and exits before the lease, and nothing is built.
depth-gpu-qwen3moe *ARMS:
    BLOOMERY_MODEL=qwen3moe ./tools/box.sh "${BLOOMERY_AB_ROUNDS:+export BLOOMERY_AB_ROUNDS=$BLOOMERY_AB_ROUNDS && }"'{{precheck}} && { ours=; [ -n "{{ARMS}}" ] || ours=1; for a in {{ARMS}}; do case $a in prose:*) ours=1 ;; *:*) ;; *) ours=1 ;; esac; done; if [ -n "$ours" ] && [ -z "${BLOOMERY_DRY:-}" ]; then cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin generate_qwen3moe; fi; } && bash tools/ref/depth-qwen3moe.sh {{ARMS}}'

# Qwen3.6-35B-A3B(qwen35moe) 같은 표: depth-gpu-qwen3moe의 러너·팔·표를 qwen35moe 프로필로 돈다. 우리 팔은 같은
# generate_qwen3moe가 파일 헤더에서 아키텍처를 읽어 Body35로 연다.
# Its prompt runs as its `--prefill` default (auto: ubatches for nine ids or more, `kind=` in the row names the path).
# `prose:<P>`는 depth-gpu-qwen3moe의 prose 팔과 같다. 이 프로필의
# $BLOOMERY_DATA/qwen35moe/corpus-prose.ids 앞 P개 id를 --tokens로 넣고(행 라벨 ours@prose), prose 표에서만 비교한다.
# 참조 트리와 플래그는 tools/ref/models/qwen35moe.sh. BLOOMERY_DRY=1이면 명령줄만, 빌드 없음.
depth-gpu-qwen35moe *ARMS:
    BLOOMERY_MODEL=qwen35moe ./tools/box.sh "${BLOOMERY_AB_ROUNDS:+export BLOOMERY_AB_ROUNDS=$BLOOMERY_AB_ROUNDS && }"'{{precheck}} && { ours=; [ -n "{{ARMS}}" ] || ours=1; for a in {{ARMS}}; do case $a in prose:*) ours=1 ;; *:*) ;; *) ours=1 ;; esac; done; if [ -n "$ours" ] && [ -z "${BLOOMERY_DRY:-}" ]; then cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin generate_qwen3moe; fi; } && bash tools/ref/depth-qwen3moe.sh {{ARMS}}'

# Qwen3.8-Flash-Next (qwen4exp), the same table: depth-gpu-qwen3moe's runner, arms and tables under the qwen4exp
# profile. Our arm is generate_qwen3moe opening Body38 on the A6000, each layer's routed expert prefix on the card as
# its budget holds (BLOOMERY_QWEN38_EXPERTS unset is `card`; `@BLOOMERY_QWEN38_EXPERTS=host` puts every routed expert
# on the host tier), its prompt as its `--prefill` default runs it (`kind=` in the row); the reference is mainline
# llama.cpp, hand-set -ncmoe and fit (models/qwen4exp.sh).
# `prose:<P>[@NAME=VALUE,...]` is ours on the first P ids of $BLOOMERY_DATA/qwen4exp/corpus-prose.ids (--tokens), row
# label ours@prose, compared only with prose arms of the same P in prose's own tables — the lcg prompt's pseudo-random
# ids route heavily skewed, so a lever tuned on it can mean nothing on real text. Each arm pages the ~111 GB host set
# in when it is cold. BLOOMERY_DRY=1 prints the command lines only, nothing built.
depth-gpu-qwen4exp *ARMS:
    BLOOMERY_MODEL=qwen4exp ./tools/box.sh "${BLOOMERY_AB_ROUNDS:+export BLOOMERY_AB_ROUNDS=$BLOOMERY_AB_ROUNDS && }"'{{precheck}} && { ours=; [ -n "{{ARMS}}" ] || ours=1; for a in {{ARMS}}; do case $a in prose:*) ours=1 ;; *:*) ;; *) ours=1 ;; esac; done; if [ -n "$ours" ] && [ -z "${BLOOMERY_DRY:-}" ]; then cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin generate_qwen3moe; fi; } && bash tools/ref/depth-qwen3moe.sh {{ARMS}}'

# GLM-5.3-Flash 디코드를 깊이별로, 프리필을 길이별로(A6000, 한 임대, 리드 전용):
# ours `<D>` (the first D prose ids, run as the binary's `--prefill` default, batches; `kind=` in the row names the feed),
# llama.cpp PR 두 가지 `lcpp27752:<D>`·`lcpp27754:<D>`(-d D)와 `…pp[<U>]:<P>`,
# llama-server 한 요청으로 재는 `…srv:<D>`·`…mtp:<D>`(MTP 초안, 서버가 찍는 draft acceptance 줄을 행에 그대로 싣는다),
# exllamav3 `exl3:<D>`·`exl3pp:<P>`(perf.py, EXL3 4.05 bpw — 다른 양자화라 비율 표에 넣지 않는다). GGUF 팔은 바퀴마다 순서를
# 돌리고 exllamav3 팔은 그 뒤 한 덩어리로 돈다. 팔·플래그 근거는 tools/ref/depth-glm5next.sh와 models/glm5next.sh 머리에.
# 우리 팔이 없거나 BLOOMERY_BOX_ENV=BLOOMERY_DRY=1이면 generate_glm5next를 빌드하지 않는다; 인자 없으면 512 lcpp27754:512.
depth-gpu-glm5next *ARMS:
    BLOOMERY_MODEL=glm5next ./tools/box.sh "${BLOOMERY_AB_ROUNDS:+export BLOOMERY_AB_ROUNDS=$BLOOMERY_AB_ROUNDS && }"'{{precheck}} && { ours=; [ -n "{{ARMS}}" ] || ours=1; for a in {{ARMS}}; do case $a in *:*) ;; *) ours=1 ;; esac; done; if [ -n "$ours" ] && [ -z "${BLOOMERY_DRY:-}" ]; then cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features glm5next --release --bin generate_glm5next; fi; } && bash tools/ref/depth-glm5next.sh {{ARMS}}'

# Qwen3-30B-A3B decode-step kernel timeline (nsys, A6000, under the lease, lead-only): generate_qwen3moe's seed form at each
# depth (default 6 4096), one prefill pass + BLOOMERY_NSYS_N - 1 replays, the cache height depth-qwen3moe.sh uses. The
# header of tools/ref/nsys-gpu.sh has the boundary and the windows. Under BLOOMERY_BOX_ENV=BLOOMERY_DRY=1 nothing is built.
nsys-gpu-qwen3moe *DEPTHS:
    BLOOMERY_MODEL=qwen3moe ./tools/box.sh '{{precheck}} && if [ -z "${BLOOMERY_DRY:-}" ]; then cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin generate_qwen3moe; fi && BLOOMERY_NSYS_DEPTHS="{{DEPTHS}}" bash tools/ref/nsys-gpu.sh'

# Qwen3-30B-A3B prefill kernel table (nsys, A6000, under the lease, lead-only): generate_qwen3moe's timed prompt (the
# depth runner's `<P>` arm) at each length P (default 4096, one ubatch at the default ubatch size), then
# BLOOMERY_NSYS_N - 1 replays; the table is the prompt's window, its units and the head, per kernel, with the grouped
# GEMM summed. The header of tools/ref/nsys-gpu.sh has the boundary. Expected at P = 4096 on main [derived,
# docs/research/q3next-design-report.md; not numbers of record]: gemm_q4k + gemm_q6k about 62 % of the prompt,
# qwen3moe_router_logits 0.36-0.54 ms a launch. Under BLOOMERY_BOX_ENV=BLOOMERY_DRY=1 nothing is built.
nsys-gpu-qwen3moe-prefill *PROMPTS:
    BLOOMERY_MODEL=qwen3moe ./tools/box.sh '{{precheck}} && if [ -z "${BLOOMERY_DRY:-}" ]; then cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin generate_qwen3moe; fi && BLOOMERY_NSYS_FORM=prefill BLOOMERY_NSYS_DEPTHS="{{PROMPTS}}" bash tools/ref/nsys-gpu.sh'

# V4.1 디코드 스텝의 커널 타임라인(nsys, A6000, 임대 안, 리드 전용): 깊이마다 커널 합 대 호스트 합류 빈틈. 인자는 깊이 목록
# 또는 말뭉치 팔(prose:<P>, code:<P> — depth-ds41.sh의 말뭉치 팔이 먹이는 같은 id). BLOOMERY_GEN_PLACE(a|gate|bp)와
# BLOOMERY_TIMING_CARDS=a6000+3090(bp의 두 카드 모드), BLOOMERY_RESIDENCY 등 BLOOMERY_BOX_ENV로 실은 레버가
# 프로파일 대상에 그대로 전해지고 [levers] 줄로 찍힌다; 레지던시가 켜진 실행은 엔진 스트림 옆 복사 스트림 표
# (tools/ref/ds41copy.py)를 더한다. BLOOMERY_BOX_ENV=BLOOMERY_DRY=1이면 빌드도 임대도 없이 명령줄만 찍는다.
nsys-gpu-ds41 *DEPTHS:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh '{{precheck}} && if [ -z "${BLOOMERY_DRY:-}" ]; then cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features deepseek41 --release --bin generate_ds41; fi && bash tools/ref/nsys-ds41.sh {{DEPTHS}}'

# V4.1 prompt batch timeline (nsys, A6000, under the lease, lead-only): generate_ds41 --depth P -n 2 --mode graph --time at
# each prompt length P (default 512), the window from the prompt's first kernel to the first replay, cut into layer-batches
# at ds41_ffn_places and the joins against the run's stat lines; per layer-batch the route window, the union gap and the
# post, the kernel terms of one layer-batch (BLOOMERY_NSYS_LAYER, default 2) and of layers 2-39, the card's idle time and the
# launch queue from the CUDA API trace. The
# header of tools/ref/nsys-ds41.sh and tools/ref/ds41pp.py have the cut. Under BLOOMERY_BOX_ENV=BLOOMERY_DRY=1 nothing is built.
nsys-gpu-ds41-prefill *PROMPTS:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh '{{precheck}} && if [ -z "${BLOOMERY_DRY:-}" ]; then cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features deepseek41 --release --bin generate_ds41; fi && BLOOMERY_NSYS_FORM=prefill bash tools/ref/nsys-ds41.sh {{PROMPTS}}'

# ik KLD 기준 파일 둘을 위치마다 비교한다(P = A, Q = B, 태그는 $BLOOMERY_DATA/ikppl 아래, 호스트만).
# ARGS: --ubatch N(여러 번 줄 수 있다), --ik <ik-ppl --kld 태그>(ik가 찍은 요약과 밴드 안에서 맞는지).
kld-diff A B *ARGS:
    ./tools/box.sh 'cargo build --release -p bloomery-gpu-gates --bin kld_diff && bash tools/host-gate.sh kld_diff "$BLOOMERY_DATA/ikppl/{{A}}.kld" "$BLOOMERY_DATA/ikppl/{{B}}.kld" {{ARGS}}'

# V4.1 본체를 게이트 배치(모든 층과 헤드, 예산 안에서 가장 큰 expert 접두)대로 엔진 입구를 거쳐 3090에 두 번 올린다.
# 세그먼트는 계획의 바이트대로, 슬롯 맵은 그 접두대로 올라가야 하고(카드 사본과 호스트 티어 사본이 같아야 한다), 층마다
# 상태는 KvLayout의 바이트와 같아야 한다. 위치 4·301·1025의 스텝 이미지를 되읽어 계획의 정수·RopeTable의 표와 맞춘다.
# 체인은 캡처돼야 하고(노드 수는 step 게이트의 몫) 합성 깊이는 거부돼야 한다. 두 번째 적재는 첫 번째와 같은 양을 가져가고,
# 드롭은 컨텍스트의 첫 캡처가 쥔 몫만 빼고 전부 돌려줘야 한다. 측정이 아니라 정확성 실행이다(파일 중 카드 몫을 두 번 읽는다).
[group('v41-load')]
gate-ds41-load:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh 'export BLOOMERY_GATE_V41_LOAD=1 && cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features deepseek41 --release --bin gate_deepseek41_load && bash tools/gpu-gate.sh gate_deepseek41_load'

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
    ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin gate_p8b && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_p8b {{ARGS}}'

# PTX 스캔(계측기, 게이트 아님): 게이트 바이너리가 싣고 있는 디바이스 코드의 엔트리별
# 디포·로컬 왕복·블록 폭 표. 디포를 가진 엔트리가 먼저 나온다. 단언은 gate_p5/gate_p4가 한다 — 머리글 참조.
# 바이너리의 cargo 피처는 `--features`로 준다(기본 gpu). V4.1 게이트 바이너리는 `--features deepseek41`.
# 끝의 두 열(jit_regs·jit_local)은 드라이버 JIT가 실제로 잡은 값이다. oxart_jit가 모듈을 카드에 올려 읽으므로
# 게이트 락(tools/gpu-gate.sh) 아래서 돈다.
# BIN 뒤의 말은 ptx-scan.sh의 엔트리 부분 문자열 하나뿐이다. 자리로 준 피처(`gpu,deepseek41`, 피처 이름)나 플래그는
# 빌드 전에 거절한다(tools/scan-args.sh, 세 스캔 레시피 공용) — ARGS로 흘러 기본 피처로 빌드되면 트랙의 바이너리를
# 호스트 전용으로 덮어쓴다.
[arg("FEATURES", long="features")]
ptx-scan BIN FEATURES='gpu' *ARGS:
    @bash tools/scan-args.sh ptx {{BIN}} {{ARGS}}
    ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features {{FEATURES}} --release --bin {{BIN}} --bin oxart_jit && cargo build --release -p bloomery-gpu-gates --bin oxart_ptx && bash tools/ptx-scan.sh {{BIN}} {{ARGS}}'

# PTX 스필 래칫 — 빌드 시점 게이트. generate_ds41(deepseek41)과 gate_e2e(V2-Lite)의 ptx-scan 표에서 엔트리마다
# spill(ptxas 스필 저장 바이트)·jit_local(드라이버 JIT의 스레드당 로컬 바이트)을 tools/ref/ptx-shapes.tsv의 핀과 대조한다.
# 핀 위든 아래든 다른 값, 핀 없는 새 엔트리, 스캔에서 사라진 핀 행이 전부 빨강이다 — 스필은 비트를 안 바꾸고 속도만
# 바꿔 비트 게이트가 전부 지나가므로(ds41_attn_seg_sel의 8 B 스필을 눈으로 찾았다), 0이 아닌 핀은 PIN 줄로 사유를 단다.
gate-ptx-spill:
    ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu,deepseek41 --release --bin generate_ds41 --bin oxart_jit && cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin gate_e2e --bin oxart_jit && cargo build --release -p bloomery-gpu-gates --bin oxart_ptx && bash tools/ptx-spill-check.sh tools/ref/ptx-shapes.tsv generate_ds41 gate_e2e'

# 인자: `<엔트리> n,t,…`(분기 결정 한 줄을 따라간 경로), `<엔트리> list`(목록).
# SASS 스캔(ptx-scan의 짝, 계측기): 첫 대기 전에 발행된 전역 로드 수를 루프마다, 그리고 한 경로를 따라 센다.
# 자리로 준 피처와 sass-scan.sh가 받지 않는 플래그는 ptx-scan처럼 빌드 전에 거절한다(tools/scan-args.sh).
[arg("FEATURES", long="features")]
sass-scan BIN FEATURES='gpu' *ARGS:
    @bash tools/scan-args.sh sass {{BIN}} {{ARGS}}
    ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features {{FEATURES}} --release --bin {{BIN}} && cargo build --release -p bloomery-gpu-gates --bin oxart_ptx && bash tools/sass-scan.sh {{BIN}} {{ARGS}}'

# 공유 로드·스필 스캔(계측기, 게이트 아님): 엔트리마다 PTX 본문의 ld.shared(폭별)·ld.global·st.shared·shfl·fma·
# cvt.f16 수와, ptxas가 만든 SASS의 STL·LDL·LDS(폭별) 수를 한 줄에 센다. ptx-scan의 spill 열(저장+적재 바이트)이
# 어느 쪽에서 왔는지, 공유 로드가 벡터로 합쳐졌는지를 명령 한 줄로 가른다. 카드가 필요 없다(ptxas·cuobjdump만).
# 자리로 준 피처는 ptx-scan처럼 빌드 전에 거절한다(tools/scan-args.sh). 예: `just lds-scan gate_hc_gated hc_gated_up_mix`.
[arg("FEATURES", long="features")]
lds-scan BIN FEATURES='gpu' *ARGS:
    @bash tools/scan-args.sh lds {{BIN}} {{ARGS}}
    ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features {{FEATURES}} --release --bin {{BIN}} && cargo build --release -p bloomery-gpu-gates --bin oxart_ptx && bash tools/lds-scan.sh {{BIN}} {{ARGS}}'

# 판정 비교(맥에서 돈다): 두 트리에서 같은 레시피를 돌리고, 빌드 줄·시간·pid·스레드 id를 가린 뒤 출력을 비교한다.
verdict-diff *ARGS:
    python3 tools/verdict-diff.py {{ARGS}}

# 바퀴마다 팔 순서를 돌리고, 임대는 기존 러너가 잡는다. `run --dry-run`은 계획만 찍는다.
# GPU A/B(리드 전용): (트리, env) 팔을 시간 레시피로 번갈아 돌리고 팔별 평균·SD·첫 팔 대비 차와 그 구간을 낸다.
gpu-ab *ARGS:
    python3 tools/gpu-ab.py {{ARGS}}

# DSpark 드래프트 파일 읽기: MXFP4 디코드 단위 시험(스케일 바이트 256개 × 코드 16개 전부)을 돈 뒤, 엄격한 리더로
# 드래프트를 열어 hparams·텐서 표·바이트 타일링을 보고, mxfp4_ref 덤프(just build-ref)와 비트 대조하고, 전문가 스케일
# 바이트를 전부 훑는다. 바이트 표를 찍는다. 대상 쪽 텐서(token_embd·output)를 읽으려고 V4.1 프로필로 돈다. 호스트 전용.
gate-dspark-read:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh 'bash tools/gate.sh --release -p bloomery-gguf --lib -- quant --nocapture && cargo build --release -p bloomery-gpu-gates --bin gate_dspark_read && __s=$(. tools/ref/ref-paths.sh && printf %s "$DSPARK_MODEL") && export BLOOMERY_DSPARK_MODEL="$__s" && bash tools/host-gate.sh gate_dspark_read'

# DSpark 드래프트의 두 커널(f32 가중치 HC_PRE, bf16 Markov 헤드 + argmax)을 카드에서 호스트 규칙과 비트 단위로 대조한다.
# dsref 세트(code64_n32_w3) 블록 0에 대한 ik와의 거리는 찍기만 한다(진단, 핀 아님).
gate-gpu-dspark-hc:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features deepseek41 --release --bin gate_dspark_hc && __s=$(. tools/ref/ref-paths.sh && printf %s "$DSPARK_MODEL") && export BLOOMERY_DSPARK_MODEL="$__s" && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_dspark_hc'

# DSpark 드래프트의 routed MoE(MXFP4 expert gate·up·SwiGLU와 down+결합, 128개 중 3개 라우터)를 카드에서 호스트 규칙과
# 비트 단위로, f32 참조와는 q8_1 한계와 핀으로 대조한다. dsref 세트 블록 0 층 0에 대한 ik와의 거리는 찍기만 한다.
gate-gpu-dspark-experts:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features deepseek41 --release --bin gate_dspark_experts && __s=$(. tools/ref/ref-paths.sh && printf %s "$DSPARK_MODEL") && export BLOOMERY_DSPARK_MODEL="$__s" && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_dspark_experts'

# DSpark 드래프트를 3090에 올리고(텐서별 카드 형식, 그룹별 바이트 = 인벤토리, cuMemGetInfo 증감), 특징→KV 그래프를
# dsref 세트 블록 0–3에 대조한다: main_x와 층마다 링 행이 ik의 q8_1 활성값 규칙이 허용하는 유도 한계 안.
gate-gpu-dspark-kv:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features deepseek41 --release --bin gate_dspark_kv && __s=$(. tools/ref/ref-paths.sh && printf %s "$DSPARK_MODEL") && export BLOOMERY_DSPARK_MODEL="$__s" && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_dspark_kv'

# DSpark 드래프트의 블록 패스와 헤드를 ik가 떨군 dsref 세트와 대조한다. ik 규칙 arm(블록 인과 마스크, SwiGLU clamp 없음 —
# ik 드래프트가 도는 방식)은 블록 0–2의 모든 탭을 유도 밴드로 판정하고, 엔진이 도는 기준 규칙(model.py) arm은 두 규칙이
# 갈리지 않는 행만 판정한다. 블록마다 draft_argmax, w=5 비트 동일, 캡처 노드 수 = 커널 목록(패스 73 + 8w, 링 append 8). DraftBody가
# 로드 때 잡은 그래프는 블록 0–2·폭 1–5에서 eager와 비트 동일, 세트 전체에서 id와 링이 eager 쌍둥이와 같다. 수락 수는 찍기만 한다.
gate-gpu-dspark-graph:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features deepseek41 --release --bin gate_dspark_graph && __s=$(. tools/ref/ref-paths.sh && printf %s "$DSPARK_MODEL") && export BLOOMERY_DSPARK_MODEL="$__s" && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_dspark_graph'

# DSpark Markov 헤드 단독 초안(E6)의 오프라인 수락률 — 시간을 재지 않는 CPU 실행이고 임대를 잡지 않는다. 먼저 self-test,
# 다음 코퍼스 스트림마다 앞 50k토큰의 acc/positions·top-2·top-4와 E5 markov1 재계산, 이전 토큰별 argmax 표의 해시와
# 위치별 경로와의 불일치 수(0이어야 한다)를 찍는다. 목표는 코퍼스 텍스트 자체다. 인자는 bin에 그대로 간다(--tokens N).
markov-accept *ARGS:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh 'cargo build --release -p bloomery-model --bin markov-accept && bash tools/host-gate.sh markov-accept --self-test && S=$(. tools/ref/ref-paths.sh && printf %s "$DSPARK_MODEL") && D=$BLOOMERY_DATA/engram && bash tools/host-gate.sh markov-accept "$S" $D/corpus-code.ids $D/corpus-prose.ids $D/corpus-prose-all.ids $D/corpus-korean.ids $D/corpus-threads.ids {{ARGS}}'

# V4.1 비전 오라클(V0): 공식 체크포인트의 image_processor.py·vision.py를 박스 torch로 3090에서(게이트 락 아래)
# tools/ref/vision/images/에 돌려 $BLOOMERY_DATA/ref-vision/deepseek41v/에 쓴다. 두 번 돌려 바이트 동일할 때만 설치하고,
# MANIFEST에 리비전·샤드 sha256·torch·Pillow·카드·mmproj를 적는다.
dump-ref-vision:
    ./tools/box.sh 'bash tools/ref/vision/dump-vision.sh'

# The V4.1 vision fork comparison (tools/ref/vision/): smalinin's llama.cpp fork at its pin, built under
# ~/repo/bloomery-visref-fork on the box with its row-injection harness visref_fork (build-fork.sh, CPU only, ~4 min).
build-visref-fork:
    ./tools/box.sh 'bash tools/ref/vision/build-fork.sh'

# The fork comparison's set: the official encoder's rows of tools/ref/vision/scenes/ (dump-vision.sh, set
# deepseek41v-scenes), then the fork's greedy answers and logits on each scene and two text controls per scene
# (visref.sh) into $BLOOMERY_DATA/ref-visref/deepseek41-scenes, family visref-deepseek41. Not timed, no lease; the
# fork runs under the card's gate lock and the V4.1 load lock.
dump-ref-visref:
    ./tools/box.sh 'BLOOMERY_VISION_SET=deepseek41v-scenes VISION_IMAGES=tools/ref/vision/scenes bash tools/ref/vision/dump-vision.sh && bash tools/ref/vision/visref.sh'

# JPEG decoder fixtures: dump-jpeg.py on the box encodes crops of tools/ref/vision/images/ with the system Pillow and
# decodes each file the way the reference does (its header lists the files); the set comes back whole into
# crates/vision/tests/fixtures/jpeg/, which the image tests read. Another libjpeg writes other bytes: commit the set whole.
dump-ref-jpeg:
    #!/usr/bin/env bash
    set -euo pipefail
    out=/tmp/$(basename "$PWD")-dump-jpeg
    ./tools/box.sh "rm -rf $out && /home/user/ft/bin/python3 tools/ref/vision/dump-jpeg.py --out $out"
    got=$(mktemp -d) && chmod 755 "$got"
    BLOOMERY_BOX_READONLY=1 ./tools/box.sh "tar -C $out -cf - ." | tar -C "$got" -xf -
    grep -q '^# complete' "$got/MANIFEST.tsv"
    rm -rf crates/vision/tests/fixtures/jpeg
    mkdir -p crates/vision/tests/fixtures
    mv "$got" crates/vision/tests/fixtures/jpeg
    ls -l crates/vision/tests/fixtures/jpeg

# V4.1 비전 호스트 게이트(V1): 리사이즈 플랜과 패드 기하, 패드된 u8 이미지와 bf16 패치(리샘플 유무 모두 비트 동일),
# 스팬 id·타입을 그 세트와 대조하고, mmproj 헤더의 hparams와 모든 텐서의 이름을 검사한다.
# It also runs tests/jpeg.rs: the JPEG decoder against Pillow on the fixtures of dump-ref-jpeg, within the pinned frontier.
gate-vision:
    ./tools/box.sh 'bash tools/gate.sh --release -p bloomery-vision --lib --test vision --test jpeg -- --include-ignored --nocapture'

# V4.1 이미지 인코더 카드 게이트(V2): 전체 사슬(ViT 32블록 + aligner, bf16 텐서코어 GEMM, 비인과 어텐션, 2D RoPE)을
# 비전 오라클 세트의 모든 이미지에 돌려 공식 vision.py와 탭별로 대조한다 — embed·blk0·forced 탭은 절대 핀, 자유 실행 탭은
# 참조의 자기 민감도 행(MANIFEST `# sensitivity`)의 K배 안; 블록 0은 crates/gpu-vision의 호스트 규칙과 연산별로 대조. 3090, 게이트 락.
gate-gpu-vision:
    ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features vision --release --bin gate_vision_encoder && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_vision_encoder'

# V4-Flash 인벤토리 게이트: deepseek41 모듈이 V4-Flash 파일(deepseek4)에서 읽은 모델 값·층 종류(비율 4는 자기 인덱스 키로
# top-k, 비율 128은 전 행, 0–2층 해시 라우팅), 텐서 수·역할별·타입별 바이트, 설계 §5 (a)·(b)·게이트 배치의 카드·호스트 줄과
# 카드 예산이 담을 expert 수[유도], 엔진이 거절하는 기능 목록을 tests/common/deepseek4_pins.rs에 대조한다. 헤더만, 초 단위.
gate-deepseek4-meta:
    BLOOMERY_MODEL=deepseek4 ./tools/box.sh 'bash tools/gate.sh --release -p bloomery-model --lib -- arch::deepseek41 --nocapture && bash tools/gate.sh --release -p bloomery-model --test deepseek4_meta -- --ignored --nocapture'

# glm5next (GLM-5.3-Flash) header gate: the reader's refusals on synthetic headers, then its description of the
# file and the coverage check's list. Headers only, seconds; needs all six shards.
gate-glm5next-meta:
    ./tools/box.sh 'bash tools/gate.sh --release -p bloomery-model --lib -- arch::glm5next --nocapture && bash tools/gate.sh --release -p bloomery-model --test glm5next_meta -- --ignored --nocapture'

# glm5next (GLM-5.3-Flash) end-to-end: the program loaded once by the gate placement on the 3090 (every routed expert
# on the host tier, the host set of about 185 GB populated), against ik's sets (refset `ik-glm5next`): the step's
# node count, graph = eager bit for bit, every layer's streams on the batch set within the derived band, and the argmax
# after each step set's prompt — here the two 4-token step sets (`--step-sets short`); the 1,024- and 3,070-position
# sets are weekly-gpu-glm5next-e2e-long's. Right after the load, before those, the card experts (shared/glm5next_card.rs):
# the slot map holds each layer's id prefix [0, n_l) in ascending order, only on layers whose stacks the card reads; at
# slots 0, n/2 and n-1 each stack's `_sel` and the gate·up are that expert's own upload from the file, bit for bit; the
# card copy of the map is the host map at its row offsets. Loads the whole model: alone in a batch, and under the
# big-load lock the V4.1 loads take.
[group('solo')]
[group('v41-load')]
gate-gpu-glm5next-e2e:
    BLOOMERY_MODEL=glm5next ./tools/box.sh 'export BLOOMERY_GATE_V41_LOAD=1 && cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features glm5next --release --bin gate_glm5next_e2e && bash tools/gpu-gate.sh gate_glm5next_e2e --step-sets short'

# The same gate's drafted slots pass (`--only stagger-draft`, (h1)-(h4)) on a NextN load of its own at SLOT_CTX, its plan
# counting two slots: each slot's verify rows, two a slot, as one pass of four (`GpuModel::verify_slots` and
# `commit_slots`) against each slot's drafted run alone, in graph and eager mode; forced kept rows; the refusals while a
# pass waits for its commit and of the passes a NextN load does not run, a pass that keeps every row
# (`GpuModel::step_slots`) among them; the captured four-row pass against the verify pair's. The gate's run with no
# `--only` leaves these clauses out. Loads the whole model once: alone in a batch, and under the big-load lock the V4.1
# loads take.
[group('solo')]
[group('v41-load')]
gate-gpu-glm5next-stagger:
    BLOOMERY_MODEL=glm5next ./tools/box.sh 'export BLOOMERY_GATE_V41_LOAD=1 && cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features glm5next --release --bin gate_glm5next_e2e && bash tools/gpu-gate.sh gate_glm5next_e2e --only stagger-draft'

# The same gate's long step sets (`--only main --step-sets long`): the load at CTX and its clauses, with the 1,024-position
# set and the 3,070-position `--dsa` set, the only comparisons with ik past 4 positions and the only end-to-end run past
# the dense limit. Weekly: `just weekly` runs it, and `just affected` names it when a file its triggers in
# tools/gate-paths.tsv match changes.
[group('solo')]
[group('v41-load')]
weekly-gpu-glm5next-e2e-long:
    BLOOMERY_MODEL=glm5next ./tools/box.sh 'export BLOOMERY_GATE_V41_LOAD=1 && cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features glm5next --release --bin gate_glm5next_e2e && bash tools/gpu-gate.sh gate_glm5next_e2e --only main --step-sets long'

# glm5next (GLM-5.3-Flash) NextN draft: the target loaded twice by its NextN plan on the 3090, without the next-token
# layer and with it (`Body::open_placed_nextn`), against ik's MTP draft set (refset `mtp-glm5next`): the set's MTP input
# (no position mask), the walk's refusals, every graph of the set replayed from ik's inputs with each block's proposal the set's
# draft, the two loads' plain runs bit for bit, and the drafted windows (`MtpDraft<Body>`) the plain run's ids with a
# rejected and an accepted row, the prompt in batches and by steps. Loads the whole model twice: alone in a batch, and
# under the big-load lock the V4.1 loads take.
[group('solo')]
[group('v41-load')]
gate-gpu-glm5next-mtp:
    BLOOMERY_MODEL=glm5next ./tools/box.sh 'export BLOOMERY_GATE_V41_LOAD=1 && cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features glm5next --release --bin gate_glm5next_mtp && bash tools/gpu-gate.sh gate_glm5next_mtp'

# glm5next (GLM-5.3-Flash) on plan (b′): the stage on the A6000, the expert tier on the 3090 (`app::arch::glm5next::open_pair`
# at --place bp, two KDA lanes, under a 16 GiB card budget on both cards), against a reference load of the same plan whose
# A6000 holds the stage's and the tier's experts with no tier: the step leg (32 prompt steps, 48 greedy), the pair leg (16
# verifies of two rows, a third rejected) and the call leg (one 512-position prompt batch, 4 steps after) bit for bit the
# reference's, tokens, kept counts and logits rows; the stage's step graph of the reference's node count; every tier layer
# sent a routed slot by each leg, the call's batch served by the tier. Two loads, one after the other. Then two more
# runs, each under its own bound: --residency (the plain and the NextN load under mid-p0-s1 beside the tier: the same
# history twice with flips landed, and no tier expert moved — the host map's tier entries, the tier's set and the stage
# card's copy of the map), and --nextn (a NextN plan with a next-token expert on the tier refused by name before any
# upload, then the drafted ids and windows on the two cards against the one-card NextN load of the union, the walk on
# the stage card). Two loads each. Then --records: generate_glm5next's load record names the A6000 alone under --place a
# and the A6000 then the 3090 (with the tier's experts) under bp, one child process a placement. Both cards
# (BLOOMERY_CARD=both: both gate locks), alone in a batch.
[group('solo')]
[group('v41-load')]
gate-gpu-glm5next-twocard:
    BLOOMERY_MODEL=glm5next BLOOMERY_CARD=both ./tools/box.sh 'export BLOOMERY_GATE_V41_LOAD=1 && cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features glm5next --release --bin gate_glm5next_twocard --bin generate_glm5next && bash tools/gpu-gate.sh gate_glm5next_twocard && bash tools/gpu-gate.sh gate_glm5next_twocard --residency && bash tools/gpu-gate.sh gate_glm5next_twocard --nextn && bash tools/gpu-gate.sh gate_glm5next_twocard --records'

# GLM's adaptive expert residency on the gate placement (BLOOMERY_RESIDENCY set in the gate at mid-p<P>-s1, P half the
# plan's least card slots a layer): the churn pool's host refusal at load, the batch prompt one pass keeping 0 and each
# step 1, every admitted slot byte for byte the file's on a Q4_K layer and on the Q5_K gate/up layer (per-layer parts),
# the steps prompt refused by name, the same history twice with flips landed, and a reset back to the seed. Loads the
# whole host set: the e2e gates' batching rule.
[group('solo')]
[group('v41-load')]
gate-gpu-glm5next-residency:
    BLOOMERY_MODEL=glm5next ./tools/box.sh 'export BLOOMERY_GATE_V41_LOAD=1 && cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features glm5next --release --bin gate_glm5next_residency && bash tools/gpu-gate.sh gate_glm5next_residency'

# The GLM seat of bloomery-serve (--model glm) on the 3090, placement gate, against generate_glm5next on the same card
# (gate_glm5next_serve's header has the clauses). Two processes, each under its own gate-lock hold and bound, the loads
# one after the other: --arm plain first runs the seat with both levers unset and --plan at --place a and gate (the
# unset rule's records, no load), then holds the served greedy ids of one chat turn's prompt to the plain CLI's under
# BLOOMERY_DRAFT=off BLOOMERY_RESIDENCY=off and under BLOOMERY_DRAFT=mtp BLOOMERY_RESIDENCY=off (three loads); --arm
# drafted (BLOOMERY_DRAFT=mtp BLOOMERY_RESIDENCY=mid-p0-s1, the clip's levers) holds the draft's records and counts, the
# residency's records, its reset and the same ids after it, and the ids against the CLI under the same levers through
# the first landed flip (two loads). Logs in target/glm-serve-gate/{plain,plain/mtp,drafted}/.
# Loads the whole host set: alone in a batch, under the big-load lock. Both cards in view: the --plan arms' a and bp
# plan on the largest card and the next-largest; the loads stay on the 3090 (--place gate).
[group('solo')]
[group('v41-load')]
gate-gpu-glm5next-serve:
    BLOOMERY_MODEL=glm5next BLOOMERY_CARD=both ./tools/box.sh 'export BLOOMERY_GATE_V41_LOAD=1 && cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features glm5next,clef --release --bin generate_glm5next --bin bloomery-serve --bin gate_glm5next_serve && D=target/glm-serve-gate && rm -rf $D && mkdir -p $D && bash tools/gpu-gate.sh gate_glm5next_serve --arm plain --dir $D/plain && bash tools/gpu-gate.sh gate_glm5next_serve --arm drafted --dir $D/drafted'

# glm5next decode CLI, functional run (no timing): generate_glm5next feeds --tokens one step per id, then greedy -n
# tokens. The gate placement on the 3090 unless --place a (and BLOOMERY_CARD=a6000). 3090, gate lock.
[group('v41-load')]
gen-glm5next *ARGS:
    BLOOMERY_MODEL=glm5next ./tools/box.sh 'export BLOOMERY_GATE_V41_LOAD=1 && cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features glm5next --release --bin generate_glm5next && bash tools/gpu-gate.sh generate_glm5next --place gate {{ARGS}}'

# qwen4exp(Qwen3.8-Flash-Next) 헤더 게이트: qwen35moe 리더의 변형 행이 합성 헤더에서 이름으로 거절하는 것들, 그다음 파일의
# 서술·역할별 텐서·커버리지 검사의 목록과 헤더에서 세운 계획(카드·호스트·캐시 바이트). 헤더만, 초 단위. 마지막 호출은
# 호스트 티어가 48층을 모두 서빙하는지와 네 층의 단계별 밴드다(네 층에서 expert 30개의 행을 읽는다, 초 단위). 샤드 넷이 다 있어야 한다.
# 그 사이에 MTP 초안 파일 둘(shared와 아닌 것)을 타깃 헤더에 대어 읽은 서술을 고정한다. 앞 호출이 빨개도 뒤 호출은 모두 돌고,
# 종료 코드는 넷 중 하나라도 0이 아니면 0이 아니다.
gate-qwen4exp-meta:
    ./tools/box.sh 'bash tools/gate.sh --release -p bloomery-model --lib -- arch::qwen35moe arch::tests::a_qwen4exp --nocapture; lib=$?; bash tools/gate.sh --release -p bloomery-model --test qwen4exp_meta -- --ignored --nocapture; meta=$?; bash tools/gate.sh --release -p bloomery-model --test qwen4exp_mtp_meta -- --ignored --nocapture; mtp=$?; bash tools/gate.sh --release -p bloomery-model --test qwen4exp_host -- --ignored --nocapture; host=$?; echo "gate-qwen4exp-meta rc: lib $lib meta $meta mtp $mtp host $host"; [ "$lib" = 0 ] && [ "$meta" = 0 ] && [ "$mtp" = 0 ] && [ "$host" = 0 ]'

# The chat route trace (D2's input, not a gate, lead-only, in the lead's A6000 window): one bloomery-serve-ds41
# process (one load, plan (a) on the A6000) with BLOOMERY_ROUTE_TRACE=OUT and the step feed runs every prompt row of
# PROMPTS in file order, greedy (/apply-template, /tokenize, /completion at temperature 0, n_predict 256,
# cache_prompt false), and the engine writes every position's routed ids per layer into OUT, a new directory, as a
# router set with the slot kinds and a call row per prompt call; the driver seals it once the server has exited
# (route_trace.py seal) and joins contexts.tsv (one row per request). OUT is an absolute path or target/…: a path
# elsewhere in the remote tree is refused by name, since the next sync (rsync --delete) removes it. tools/ref/route-trace-chat.py's header has the contract. Predicted wall about 13-15 min [derived]. It
# takes neither the A6000 gate lock nor the V4.1 load lock (the driver is not a target/release binary
# tools/gpu-gate.sh can run): run it inside a hold. The server runs under the driver's own bound (--bound, 1800 s),
# and every request under an HTTP timeout, so the driver ends when the server does.
route-trace-chat OUT PROMPTS='tools/ref/data/d2-prompts-ko.tsv,tools/ref/data/d2-prompts-en-code.tsv':
    BLOOMERY_MODEL=deepseek41 BLOOMERY_CARD=a6000 ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features deepseek41 --release --bin bloomery-serve-ds41 && python3 tools/ref/route-trace-chat.py run --server target/release/bloomery-serve-ds41 --out {{OUT}} --prompts {{PROMPTS}}'

# 서버 soak(M2, 리드 전용, 게이트 아님): bloomery-serve-ds41을 A6000에 배치 (a)로 띄우고, 시드를 고정한 요청 묶음을
# MINUTES분 보낸다. 30초마다 표본을 떠서 메모리 누수를 판정한다. 상한은 MINUTES분에 900초를 더한 값이다.
[group('solo')]
[group('v41-load')]
soak-ds41 MINUTES='30':
    BLOOMERY_MODEL=deepseek41 BLOOMERY_CARD=a6000 ./tools/box.sh 'export BLOOMERY_GATE_V41_LOAD=1 && cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features deepseek41 --release --bin bloomery-serve-ds41 --bin soak_ds41_serve && D=target/soak && rm -rf $D && mkdir -p $D && BLOOMERY_GATE_BOUND=$(( {{MINUTES}} * 60 + 900 )) bash tools/gpu-gate.sh soak_ds41_serve --minutes {{MINUTES}} --dir $D'
