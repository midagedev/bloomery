# qwen3 seat 컨텍스트 크기: 현황과 할 일

2026-10-04 기준. ctxauto(`7fab4e21`)와 memguard(`85106d5a`) 착륙 뒤의 모습이다. 수치는 `[유도]`가 아니면 코드 사실이거나 실측 인용이다. 이 문서는 조사와 제안이고, 채택된 항목은 `plan-triage.md` 카드로 옮겨간다(여기가 두 번째 트래커가 되지 않게).

## 지금 상태

- **상한 = 파일의 학습 컨텍스트**(`<arch>.context_length`; Qwen3-30B-A3B-2507 파일은 262,144, `model/src/arch/qwen3moe/hparams.rs`의 헤더 값). 그 너머 확장 경로는 없다 — rope 테이블이 plain window 스펙(`RopeSpec::window`, `gpu/src/arch/qwen3moe/body.rs:583`, `body35.rs:1105`)이고 YaRN/스케일링 항을 읽는 곳이 없다(qwen3 계열에서; V4.1 `gpu-deepseek41`은 `rope.scaling.*`을 읽는다).
- **기본값**(`--ctx`/`--ctx-size` 없을 때, `serve_seats/qwen3.rs` `default_ctx`): whole-card 로드는 학습값을 카드 빈 바이트에 맞는 최대치로 캡한다 — 바닥 4096, 1024 배수(`CTX_GRAN`) 단위 이진탐색(`qwen3moe_place.rs` `whole_ctx_qwen3`). 캡이 바인딩되면 stderr에 한 줄.
- **memguard 뒤의 갈림길**: 파일 전체가 카드 빈 바이트에 안 들어가면 `--place` 없는 실행도 placed 플랜으로 떨어지고(`why=whole_does_not_fit`), placed 로드는 **항상 4096**(solver가 컨텍스트와 카드 expert를 트레이드하는 영역이라 플래그 소유라는 주석, `qwen3moe_place.rs:260-261`). 같은 머신에서 카드 상황에 따라 기본 ctx가 수만 ↔ 4096으로 튄다 — 이 문서의 할 일 2번이 닫을 갭.
- **명시적 `--ctx C`**는 그대로 이긴다(llama-server 표기 `--ctx-size`도 받음). `--ctx 0`은 거부.
- **런타임**: 위치가 ctx를 넘으면 HTTP 400 `exceed_context_size_error`(`serve/src/api.rs:1490`). `/props`는 `n_ctx`와 `engine.model.ctx_train` 둘 다 보고한다.
- **`--parallel N`**: N슬롯이 ctx짜리 캐시 하나를 턴제로 공유한다(선점 시 re-prefill). llama-server는 슬롯마다 full n_ctx를 따로 줘서 메모리 N배 — 자원의 의미가 다른 호환 surface다(메모리 절약 설계, 결함 아님).
- **KV는 f16 고정**, 양자화 옵션 없다.

### 산수 [유도]

K+V = 2(K,V) × 4 KV헤드 × 128 dim × 2 B(f16) = 2,048 B/포지션·층. 48층(공개 스펙)이면 **96 KiB/포지션**, 학습값 262,144 포지션에 **≈ 24 GiB**. Q4_K_M weights ≈ 18.6 GB와 함께:

- 24 GB 3090: 불가 — 기본 탐색이 수만 행(빈 바이트 ÷ 96 KiB)에서 캡. gate_qwen3_serve 주석의 "tens of thousands of rows"와 일치.
- 48 GB A6000: 여유 ≈ 29 GB > 24 GiB라 **학습값 전체가 도달 가능**[유도] — "VRAM이 먼저 바인딩"은 3090 이야기다.
- llama.cpp `--cache-type-k q8_0`면 KV가 ≈ ½라 같은 카드에서 도달 ≈ 2배.

## 경쟁 엔진 비교

| 엔진 | 기본 ctx | 메모리에 안 들어갈 때 | 학습 ctx 너머 | KV 양자화 |
|---|---|---|---|---|
| llama-server (llama.cpp) | 4096 (`--ctx-size 0`이어야 학습값) | 할당 실패로 **시동 안 됨** | YaRN 등 `--rope-scaling` | `--cache-type q8_0` |
| vLLM | 모델 학습값 | **에러로 거부**(`--max-model-len` 줄이라고 함) | config의 rope_scaling 따름 | fp8 KV |
| bloomery qwen3 seat | 학습값을 VRAM에 맞게 **자동 탐색** (≥4096, 1024 배수) | 알아서 캡 + stderr 한 줄 | 없음 (학습값이 하한) | 없음 (f16 고정) |

판정: **자동 캡 기본값은 우위**(llama-server 기본 4096은 실사용자에게도 "너무 작다"로 악명이고 `-c 0` + VRAM 부족이면 시동 실패; vLLM은 거부). 갭은 셋 — placed 4096 고정(결함), KV 양자화 부재(실질 손해, 도달 ≈ ½), YaRN 부재(실수요 낮음).

출처: [llama.cpp 기본 4096과 `--ctx-size 0`](https://blog.vorona.ca), [vLLM KV 캐시 vs max_model_len 거부](https://www.answeroverflow.com) — 2026-10-04 검색.

## 할 일 (우선순위 순)

### 1. `kvq8` — KV 캐시 q8_0 (M–L)

같은 카드에서 도달 컨텍스트 ≈ 2배[유도]. 경계: `gpu/src/arch/qwen3moe/{body,body35}.rs`의 KV 면(f16 plane → q8_0 블록)과 flash 읽기 경로 dequant, `kv_bytes`(`qwen3moe_place.rs:166`)와 `whole_ctx_*` 탐색의 예산 항. keep-prefix/cut은 바이트 경로라 불변. 게이트: e2e 밴드 + forced_exact 개수 핀(정확도는 진단 — 「Performance first, accuracy opt-in」), `gate-ptx-spill` 새 엔트리 핀. 우선 **opt-in 레버**(`registry.rs` 행 + `--cache-type-k` 철자)로 내고, 기본 전환은 별도 결정. q8_0 gemv 형식은 weights쪽에 이미 있다.

### 2. `placectx` — placed/memguard 폴백 로드의 자동 ctx (S–M)

`default_ctx`가 whole 로드에만 붙어 있다. 폴백 placed 플랜은 4096 고정이라 카드 상황에 따라 기본이 튄다(위). 플래그 없을 때만 탐색하되 solver의 ctx↔expert 트레이드를 존중하는 규칙이 필요하다(예: whole과 같은 floor에서 출발, solver가 그 안의 최적 배분을 고르거나, 최소한 `why=whole_does_not_fit` 줄에 `ctx=4096 (pass --ctx)` 한 줄). 0.2.1 후보 — 채택되면 이 문서 대신 triage 카드가 소유.

### 3. `yarn` — 학습값 너머 rope 확장 (S–M, 최하)

GGUF `rope.scaling.*`을 읽는 YaRN 경로. llama.cpp 패리티지만 품질을 깎는 모드라 correctness-vs-ggml 정책상 명시적 opt-in이어야 하고, 이 파일이 256K를 들고 있는 이상 실수요가 낮다. 요구가 생기면(V4.1처럼 학습값이 짧은 파일) 그때.

### 4. `slotsem` — 슬롯 ctx 의미 문서화 (XS)

llama-server는 슬롯당 n_ctx, 우리는 공유 캐시 턴제. module doc에는 있지만 사용자 문서(README 서빙 절) 검토 한 줄.

## 사용자 결정 대기

- `kvq8`: opt-in 레버로 시작(권고) — 기본 전환 여부와 시기.
- `placectx` 스코프: 0.2.1(채널 부채) vs 0.3.0.
