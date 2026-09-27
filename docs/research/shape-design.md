# 모델 모양은 스펙에서 온다 — 상수 설계 (03, 2026-09-27)

사용자 요청(2026-09-27): "전문가 수 같은 상수를 고정하지 않고 다이내믹하게 쓸 수 있게 설계해 줘", 이어서
"트레이드오프 잘 봐서 억지로 하진 말고 더 나은 설계를 지향해 줘". 이 문서의 결론은 **모두 런타임으로**가 아니다.
모델마다 코드에 박힌 값을 없애는 것이 목표다. 그래서 값마다 "이 값이 무엇을 정하는가"를 묻고, 그 답에 따라 네 부류로
나눈다. 비용은 모두 [유도]이고, 박스 확인은 각 라운드의 게이트가 한다.

## 1. 지금 무엇이 박혀 있나

같은 사실이 crate마다 따로 적혀 있다. 모델 하나를 더할 때마다 이 사본이 하나씩 늘었다.

| 자리 | 값 | 무엇의 모양 |
|---|---|---|
| `crates/gpu/src/router.rs:38,41` | `N_EXPERT` 64, `N_USED` 6 | V2-Lite 라우터 |
| `crates/gpu-deepseek41/src/router.rs:60,63` | 384, 6 | V4.1 라우터 |
| `crates/gpu-deepseek41/src/experts_mxfp4.rs:63,66` | 128, 3 | DSpark 초안 전문가 |
| `crates/gpu/src/arch/qwen3moe/router.rs:72,75` / `:2215,2217` | 128/8, 256/8 | Qwen3, Qwen3.6 라우터 |
| `crates/gpu/src/arch/qwen3moe/plan.rs:147` | `EXPERTS` 256, `TOP_K` 8 | Qwen3.6 계획 |
| `crates/gpu-deepseek41/src/chain/ffn.rs:118` | `assert!(N_USED == 6)` | handoff 커널과 combine |
| `crates/model/src/moe.rs:753` | `EXPERTS_INTO_MAX` 8 | 호스트 전문가 목록 배열 |
| `crates/gpu/src/host/step.rs:485,706,787` | `[_; EXPERTS_INTO_MAX]` | 호스트 티어 페이지 목록 |
| `gpu-gates` 여러 곳 (`block.rs:49`, `gate_qwen35moe_e2e.rs:124`) | 6, 256/8 | 게이트가 가정한 모양 |

반면 모델 파일은 이 값을 이미 한 곳에 담고 있다. `crates/models`의 `ModelSpec`이 층마다
`Moe { experts, top_k, … }`를 헤더(`expert_count`, `expert_used_count`)에서 읽는다. 그리고 `bloomery-models`는 Mac에서
네이티브로 빌드되고 테스트된다(`cargo test -p bloomery-models --lib` rc 0, 2026-09-27). **단일 소유자는 이미 있고, 사본만
정리하면 된다.**

## 2. 네 부류와 결정

### (a) 호스트 쪽 크기 — 로드 때 정한다

`EXPERTS_INTO_MAX` 배열(`moe.rs`, `host/step.rs`, `host/batch.rs`, `bench_v41_host.rs`), `PageLayout`의 `n_used`.

- **결정**: 로드 때 스펙에서 크기를 정해 할당한다(`Derived`나 로드 시 scratch). 상수 상한도 없앤다.
- **비용 0 [유도]**: "스텝은 로드 때 할 일을 하지 않는다"는 규칙 덕분이다. 크기가 로드 때 정해지면 스텝에는 할당이 없다.
  `just gate-alloc`의 스텝당 할당 수 래칫이 그대로 지킨다.
- **값**: Qwen3.8(10)이 이 부류에서 막혀 있다. `PageLayout`이 `n_used > EXPERTS_INTO_MAX`를 이름 붙여 거부한다(hostab
  발견). 첫 라운드가 이것을 푼다.
- **억지로 하지 않는 것**: `DEFER_MAX_COLS`(8)와 `Q8Blocks32`의 m ≤ 8은 모델 모양이 아니라 엔진이 고른 묶음 기하다.
  모델이 바뀌어도 이 값을 바꿀 이유가 없으므로 상수로 둔다. 이름을 `…_CAP`처럼 "용량"으로 읽히게 하는 것만 검토한다.

### (b)와 (c) 기기 커널의 값 — 첫걸음은 언제나 인스턴스 행이다

이 부류에서는 **순서**가 설계의 절반이다(aa 검토, 2026-09-27). 같은 날 K 라운드 둘(glmkda, q38gqa)이 새 모델을 받으려고
기존 몸체를 제네릭으로 바꿨다. 그 결과 기존 Qwen3.6 엔트리 다섯의 md5와 명령 수가 움직였고, "추가" 부류였어야 할 커밋이
착륙 묶음에서 멈췄다. 되돌린 방법(kfix)이 이 부류의 규칙이 되었다.

1. **새 모양의 첫걸음은 인스턴스 행이다(add 부류).** 새 모양은 기존 엔트리 옆에 새 이름의 엔트리로 들어간다. 기존 엔트리가
   닿는 소스는 글자 그대로 둔다. 새 엔트리의 몸체가 제네릭이어야 하면 새 이름의 사본을 만든다. launcher 하나가
   스펙 값으로 인스턴스를 고르고, 없는 값은 이름을 대고 거부한다. 부르는 쪽은 인스턴스 이름을 모른다.
   - 증명: ptx-scan이 base에 새 엔트리만 더한 것과 같다. 기존 엔트리가 닿는 항목의 텍스트가 base와 같다는 호출 그래프
     검사(`callgraph.py`)가 Mac 쪽 증명이다.
   - 예: slot8의 `ds41_ffn_handoff_8`, `ds41_ffn_post_8`, `_streams_8`, `ds41_ffn_card_acc_8`, `ds41_card_gather_8`.
     launcher는 `PageLayout.n_used`로 6과 8 중에서 고른다. q38gqa의 `_p4`, q38router의 `_512`도 같은 모양이다.
2. **여러 행을 한 엔트리로 합치는 일은 나중의 별도 변경이다.** 런타임 `n_used`로 바꾸거나 배열을 없애는 것이 그런 합치기다.
   기존 엔트리의 명령이 바뀌므로 (b) 증명이 필요하다: 소유 게이트 비트 동일, 자원 열 보고, 움직이면 같은 lease의 A/B.
   예: 사본이 된 GQA·GDN 몸체의 합치기는 train 5 항목이다.
   이미 이렇게 한 곳이 하나 있다. Qwen 라우터의 픽 수 `used`는 shapes 1단계에서 런치 인자가 되었다(선택을 lane이 하나씩
   들어 배열이 없어졌고, 비트 동일로 예측). 이것도 이 부류의 합치기라, 소유 게이트 비트 동일로 증명한다.

같은 값이라도 무엇을 정하느냐에 따라 합칠 수 있는 형태가 갈린다.

| 값이 정하는 것 | 예 | 합칠 때의 형태 |
|---|---|---|
| 루프 경계나 오프셋만 | handoff의 `if d < N_USED` | 런치 인자 |
| 순서만 보관하는 배열 | V4.1 combine의 `[f32; N_USED]` 셋 | 같은 순서로 흘려 계산해 배열 제거(fma를 j 오름차순). slot8의 `_8` 몸체가 이미 이 형태다 |
| lane이 든 레지스터 배열 | 라우터 `PER_LANE`(logit), GQA `HEAD` | 인스턴스로 남는다. 런타임 크기의 배열은 로컬 메모리로 가고 `no_local_depot`이 금한다 |
| 선택 횟수 | 라우터 `USED` | 런타임(lane이 픽을 든다) |

**cuda-oxide 제약(nvlabs-ledger §12, kfixgqa가 찾음).** 인스턴스 몸체 안에서 unroll할 루프의 상한은 상수 하나나 const
파라미터 하나여야 하고, 그 위의 산술(`CONST / PARAM`)이면 안 된다. 제네릭 몸체의 MIR에서 그 산술은 상수로 접히지 않고,
unroll 패스는 상수 op에서만 trip count를 읽는다. 그러면 루프가 펼쳐지지 않고, 루프 카운터로 인덱싱한 배열이 로컬 depot으로
간다. `gqa_prefill_flash_256_p4`가 jit_local 24로 이 경우를 밟았다. 필요한 몫은 엔트리 쪽에서 const 파라미터로 계산해
넘긴다(`{ MMA_ROWS / PACK_4 }`). `callgraph-bounds.py --unroll`이 Mac에서 이 경우를 잡는다. 포크에서 이 결함을 고치면
이 제약은 풀린다.

### (d) 게이트 파일의 모양 — 무엇을 재는지에 따라

- **모델 파일을 여는 게이트**(`gate_qwen35moe_e2e`의 256/8 같은 것)는 자기가 연 파일의 스펙에서 모양을 읽는다. 엔진과
  같은 출처다. 그러면 게이트와 엔진이 다른 값을 가정해 조용히 맞는 일이 없다.
- **모델 없는 합성 게이트**(`gate_linear`, `gate_kquant`, `block.rs`)의 모양은 시험 입력으로 고른 점이라 상수로 둔다. 억지로
  바꾸지 않는다. 다만 "어느 모델의 모양"이라고 주장하는 상수는 선택기의 표를 읽게 한다.

## 3. 선택기 — 한 곳에서 고르고, 없으면 이름 붙여 거부

선택기는 `crates/models/src/shape.rs`다(shapes 1단계, 커밋 d85a14f). `ModelSpec`이 `models`에 있고, runtime은 이미 models에
의존하므로 이 자리가 맞다.

- **라우터의 키는 모양만이 아니라 규칙까지다.** `MoeShape { experts, top_k, rule }`이고, `rule`은
  `RouterRule { score, bias, norm, gated }`로 층의 `Moe`에서 읽는다. 폭만으로는 몸체를 고를 수 없다. 전문가 128개는 Qwen3의
  softmax 재정규화 몸체이기도 하고 DSpark의 편향 √softplus 몸체이기도 하다. 표의 행은 규칙 하나만 받는다.
- **표**: `ROUTERS`(행 6개: V2-Lite, V4.1, DSpark, Qwen3, gated 256·512)와 `GQA`(행 3개). 각 행은 몸체, `per_lane`,
  top-k 범위, 코드 위치를 가진다. `select_router`와 `select_gqa`는 행을 돌려주거나 `ShapeRefused { shape, why }`로 거부한다.
  가장 가까운 행으로 떨어지는 폴백은 없다(조용한 실패 금지).
- **GLM-5.3-Flash**(sigmoid + bias, 288/8): 지금은 `NoRouterRule`로 이름 붙여 거부한다. aa의 glmprog가 부르는
  `gpu-deepseek41`의 GLM 라우터 엔트리(`glm5next_router`, `_scores`, `_pick`)에 호출자가 생기면 그 엔트리를 가리키는 행이
  된다.
- 기기 쪽 상수는 const fn(`router_row`, `router_row_is`, `gqa_row`)으로 표의 행에 묶인다. 모델 coverage 표(`coverage.rs`)도
  같은 표를 읽는다(커밋 a38b710). 사본이 하나 줄었다.
- **Mac에서 증명한다**: `cargo test -p bloomery-models --lib` 10개(트리가 돌리는 모델마다 행이 있다, 행 사이의 모양은 거부된다,
  GLM은 이름으로 거부된다), 뮤턴트 7개가 빨강이다. 박스의 `*-meta` 게이트는 실제 헤더로 같은 선택을 한 번 더 확인한다.

## 4. 라운드 순서

| 순서 | 라운드 | 부류 | 증명 | 상태 |
|---|---|---|---|---|
| 1 | 호스트 목록을 로드 때 크기로(`EXPERTS_INTO_MAX` 배열, `PageLayout`의 상한) | (a) | 호스트 게이트, `gate-alloc` 래칫, V4.1 비트 동일, ptx-scan 불변 | **먼저, 곧.** Qwen3.8(`n_used` 10)을 막는 유일한 항목이고 q38prog의 첫 부팅 전에 필요하다 |
| 2 | 선택기와 네이티브 테스트, crate 상수 사본을 표에 묶기, coverage | 소유자 | Mac 네이티브 테스트, ptx-scan 불변 | shapes 1단계, 브랜치 `shapesc`(train 5) |
| 3 | 새 모양의 인스턴스 행(`_8`, `_p4`, `_512`) | (b)/(c)의 첫걸음, add | ptx-scan = base + 새 엔트리, 호출 그래프 검사 | slot8, q38gqa, q38router |
| 4 | Qwen 라우터 `used` 런타임화 | 합치기 | 소유 게이트 비트 동일, 자원 열 | shapes 1단계(9a9ac3e) |
| 5 | 모델을 여는 게이트가 스펙을 읽게 | (d) | 해당 게이트 녹색 | shapes 1단계 |
| 6 | 사본이 된 몸체를 합치기(GQA 256/p4, GDN/KDA, handoff/combine 6·8) | 합치기 | 소유 게이트 비트 동일, 자원 열, 움직이면 A/B | train 5 이후 |
| 7 | V4.1 라우터를 선택기에 연결 | 소유자 | ptx-scan 불변 | shapes 2단계 |

## 5. AGENTS.md에 들어갈 문장 (aa 수락, train 5에 문서와 함께 착륙)

> **A model's shape is a value from its spec, never a crate constant.** Expert count, top-k, heads and head width come
> from `ModelSpec`. A compile-time bound is either a row of an instance table the spec selects at load, or a capacity
> the engine chose for itself; a shape no row serves is refused by name, never mapped to the nearest one. A new shape is
> a new instance row whose existing entries keep their bytes; merging rows is a separate change proven bit-identical.
