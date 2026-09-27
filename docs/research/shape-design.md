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

### (b) 기기 커널에서 루프 경계나 오프셋만 정하는 값 — 런처 인자로

handoff 커널의 `if d < N_USED`(`chain/ffn.rs:221`). 이 값은 레지스터 배열 크기를 정하지 않는다.

- **결정**: `n_used`를 런치 인자로 넘긴다. 런처는 페이지(`PageLayout`)가 담을 수 있는지 확인하고, 넘으면 이름 붙여 거부한다.
- **비용 [유도]**: 스레드당 비교 하나가 상수에서 레지스터로 바뀐다. 커널은 hidden(4,096) 스레드로 한 번 돌고, 어차피
  그 비교를 한다. 명령 수 변화는 0~1이고 시간 효과는 잣대 안이다.
- **증명 부류**: 기존 엔트리의 PTX가 바뀌므로 이동이 아니다. 소유 게이트(ds41-step, -prefill, -chain-ffn, e2e) 비트
  동일, 자원 열 보고, 타이밍 A/B 없음.

### (c) 레지스터 배열 크기·unroll·lane 배치를 정하는 값 — 커널마다 따져서

여기가 트레이드오프의 중심이다. 선택지는 셋이다.

1. **상수 인스턴스 표**: 몸체 하나를 const generic으로 두고, 지원하는 모양마다 엔트리를 하나씩 둔다. 로드 때 스펙이
   인스턴스를 고른다. 비용은 0이다. 모양이 새로 오면 표에 한 줄을 더한다.
2. **컴파일 상한 + 런타임 n**: 배열은 MAX 크기로 잡고 루프를 `i < n`으로 가린다. 엔트리 하나가 MAX 이하의 모든 모델을
   받는다. 대가로 레지스터를 MAX 기준으로 쓰므로 점유율이 떨어질 수 있다.
3. **배열을 없애는 재구성**: 배열이 순서를 보관하기 위해서만 있으면, 같은 순서로 흘려 계산해 배열을 없앤다. 이것이
   가능하면 가장 낫다. 상수도 인스턴스도 필요 없다.

커널별 판단:

| 커널 | 배열이 하는 일 | 결정 | 근거 |
|---|---|---|---|
| V4.1 combine (`card_sum_elem`, `combine_post_at`의 `[f32; N_USED]` 셋) | 슬롯 순서대로 fma하려고 모아 둔다 | **3. 배열 제거** | `acc = fma(down_j, w_j, acc)`를 j 오름차순으로 바로 흘리면 순서가 같다. 결과가 비트 동일하고 배열도 없다. `n_used`는 런치 인자가 된다 |
| Qwen 게이트 라우터 `route_warp_gated<PER_LANE, USED>` | `PER_LANE`: lane이 들고 있는 logit 배열. `USED`: top-k 선택 횟수 | **`PER_LANE`은 1(인스턴스), `USED`는 런타임** | logit 배열을 런타임 크기로 두면 로컬 메모리로 가고, 트리의 `no_local_depot` 계약이 이를 금한다. 선택 횟수는 배열을 키우지 않는 루프 횟수라 런타임으로 둬도 된다. 선택 결과를 lane j가 j번째 픽으로 들고 있으면 배열이 없다(라운드가 코드로 확인) |
| V4.1 라우터(384/6), V2-Lite(64/6), DSpark 초안(128/3) | 위와 같은 구조 | 같은 규칙 | `PER_LANE` = n_expert/32로 인스턴스 {2, 4, 8, 12, 16}이면 알려진 모델이 다 들어간다 |
| GQA flash `HEAD` | smem 타일과 레지스터 벡터의 폭 | **1. 인스턴스(128, 256)** | 헤드 폭은 타일 기하 자체다. q38gqa가 이미 이 모양이다 |
| GQA group | 블록당 쿼리 헤드 수 | **이미 동적** | q38gqa에서 `PACK`은 상수, 키 헤드당 `packs`는 런타임이 됐다. group이 PACK의 배수이기만 하면 모두 받는다 |
| 인덱서 `PER_LANE` (V4.1) | lane당 값 | 그대로 | V4.1 한 모델의 기하다. 다른 모델이 쓰지 않는 동안에는 바꿀 이유가 없다 |

상한+런타임(2)을 기본으로 삼지 않는 이유가 있다. 라우터·어텐션처럼 레지스터가 빡빡한 커널은 MAX 기준 레지스터가
점유율을 깎는다. 이런 비용은 ptxas를 돌려 봐야 안다. 그래서 2는 "ptxas가 MAX에서도 점유율이 같다고 말할 때만" 쓴다.

### (d) 게이트 파일의 모양 — 무엇을 재는지에 따라

- **모델 파일을 여는 게이트**(`gate_qwen35moe_e2e`의 256/8 같은 것)는 자기가 연 파일의 스펙에서 모양을 읽는다. 엔진과
  같은 출처다. 그러면 게이트와 엔진이 다른 값을 가정해 조용히 맞는 일이 없다.
- **모델 없는 합성 게이트**(`gate_linear`, `gate_kquant`, `block.rs`)의 모양은 시험 입력으로 고른 점이라 상수로 둔다. 억지로
  바꾸지 않는다. 다만 "어느 모델의 모양"이라고 주장하는 상수는 1절의 인스턴스 표를 읽게 한다.

## 3. 선택기 — 한 곳에서 고르고, 없으면 이름 붙여 거부

- `crates/models`에 `shape.rs`를 둔다. 스펙에서 뽑은 모양 두 개와 선택 함수 하나다.
  - 모양: `MoeShape { experts, top_k }`, `AttnShape { n_head, n_kv, head }`.
  - 선택: `MoeShape → RouterInst`(`PER_LANE`), `AttnShape → GqaInst`(`HEAD`, `PACK`).
- 표에 없는 모양은 `ShapeRefused { what, shape }`로 이름 붙여 거부한다. 가장 가까운 인스턴스로 떨어지는 폴백은 없다(조용한
  실패 금지).
- 기기 crate는 인스턴스 id를 받아 엔트리를 고른다. crate마다 있던 `N_EXPERT`/`N_USED` 상수는 없어진다. 커널 상수는 인스턴스
  표의 한 줄이 되고, 그 줄은 자기가 받는 모양을 선언한다. 로드 때 스펙 → 인스턴스를 확인한다.
- **Mac에서 증명한다**: 네이티브 테스트 하나가 "트리가 돌리는 모델마다 인스턴스가 있다"를 확인한다. 알려진 모양은
  V2-Lite 64/6, V4.1 384/6, Qwen3 128/8, Qwen3.6 256/8, Qwen3.8 512/10, GLM-5.3-Flash는 헤더 값이다. 박스의 `*-meta`
  게이트는 실제 헤더로 같은 선택을 한 번 더 확인한다.

## 4. 라운드 순서

| 순서 | 라운드 | 부류 | 증명 | 크기 |
|---|---|---|---|---|
| 1 | 호스트 목록을 로드 때 크기로 | (a) | 호스트 게이트, `gate-alloc` 래칫, V4.1 비트 동일(`ptx-scan` 불변) | M |
| 2 | `shape.rs` + 네이티브 커버리지 테스트, crate 상수 사본 제거 | 소유자 | Mac 네이티브 테스트, `ptx-scan` 불변(값이 같으므로) | S–M |
| 3 | handoff `n_used` 인자, combine 배열 제거 | (b), (c)-3 | 소유 게이트 비트 동일, 자원 열 | S |
| 4 | 라우터 `USED` 런타임화(`PER_LANE` 인스턴스 유지) | (c) | 라우터 게이트 비트 동일, ptxas 레지스터·점유율 | M |
| 5 | 모델을 여는 게이트가 스펙을 읽게 | (d) | 해당 게이트 녹색 | S |

1과 2는 V4.1의 기기 코드를 바꾸지 않는다. 3과 4는 기존 엔트리의 명령을 바꾸므로 소유 게이트 비트 동일이 증명이다.
커널 시간은 유도상 잣대 안이라 타이밍 A/B는 열지 않는다. 다만 예측이 빗나가면(자원 열이 움직이면) 그때 연다.

hostab(지금 진행 중인 호스트 티어 이동)에는 넣지 않는다. hostab은 이동 부류라 V4.1 명령이 그대로여야 한다. 이 설계의
라운드들은 hostab이 착륙한 뒤에 그 위에서 연다.

## 5. AGENTS.md에 들어갈 한 줄 (제안, aa 착륙)

> **A model's shape is a value from its spec, never a crate constant.** Expert count, top-k, heads and head width come
> from `ModelSpec`. A compile-time bound is either a row of an instance table the spec selects at load, or a capacity
> the engine chose for itself; a shape no row serves is refused by name, never mapped to the nearest one.
