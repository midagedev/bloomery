# Rust 코드 품질 전략 — 리뷰와 리팩토링의 기준

이 문서는 리뷰어와 리팩토링 라운드가 인용하는 규칙표다. 규칙마다 번호(`R1`…)가 있고, 리뷰 보고는 `경로:줄 — R번호 — 한 줄 — 크기`로 적는다. 근거는 Rust 공식 권고(API Guidelines, Rustonomicon, Unsafe Code Guidelines, clippy 린트 등급)와 이 레포에서 실제로 난 사고다. `AGENTS.md` Conventions가 도메인 계약(정확성·측정·주석)을 담고, 이 문서는 **코드의 모양**을 담는다. 둘이 충돌하면 AGENTS.md가 이긴다.

## 0. 기준선

계기: `just lint`(`--features gpu`) 출력의 `grep -c '^warning:'` — 타깃마다 한 번씩 세므로 에이전트의 유니크 계수보다 높다. **같은 계기로 전후를 잰다.** 래칫 다운만.

| 계기 | 2026-09-22 밤 (리뷰 전, `685462d`) | 2026-09-22 새벽 (리뷰 1회차 후, `0ff785e`) |
|---|---|---|
| `just lint` 경고 | **308** (gpu 피처 없이 260; 09-21 아침 144) | **221** |
| `undocumented_unsafe_blocks` | 123 | **63** — q3k-gemv 49, q3k-cpu 9(스테이지 0 스파이크, MUL-10/11 몫), model 4, gpu 1(`flash.rs`) |
| `too_many_arguments` 경고(allow 없이) | 12 | 17 (8/7 ×12, 9/7 ×2, 11/7 ×3) — R8 라운드 대상; allow에는 전부 `reason` |
| `chunks_exact` 상수 → `as_chunks` | 17 | 23 (R17 2차 — 핫 패스는 A/B 동반) |
| `crates/gpu` `as` 캐스트 | 637 | 미재측(R5 라운드 전) |
| 가장 큰 파일 | `model.rs` 3901, `flash.rs` 2628 | `flash.rs` 3241(a4d·a4e 커널), `model.rs` 3963 |

리뷰 1회차(agy 3 라운드)와 리팩토링 4 라운드(opus: `qtools`·`gatesdedup`·`cpumech`·`gpusafety`, 각 한 축)가 하루 만에 늘어난 116을 되돌리고 그 아래로 87을 더 내렸다. 리뷰 미완: `flash.rs`·`model.rs`·`gate_p5`·`gate_e2e`(a4e 뒤로 미룬 4파일), `GpuError` 열거형(R9), `*Args` 구조체(R8).

## 1. unsafe — 범위는 최소, 불변식은 타입에

- **R1** `unsafe` 블록 하나에 연산 하나, 바로 위에 `// SAFETY:` 한두 줄로 **그 연산의** 불변식(어느 인덱스가 왜 범위 안인가). 함수 전체를 감싸는 `unsafe {}`는 금지. (`undocumented_unsafe_blocks` 123건이 이 위반.)
- **R2** 같은 불변식이 세 곳 이상에서 반복되면 그것을 **타입**으로 만든다 — 경계가 증명된 뷰(cuda-device `view.rs`의 `StaticTileMut32`·`InBounds32` 같은 모양), 또는 안전한 래퍼 함수 하나. 불변식을 주석으로 열 번 쓰는 것은 열 번 틀릴 기회다.
- **R3** `unsafe fn`은 doc 주석 `# Safety` 절이 있어야 한다(`missing_safety_doc = deny`, 이미 계약). 호출자가 지켜야 할 것만 쓴다.
- **R4** 원시 포인터 산술은 커널 안에만. 호스트 오케스트레이션(`model.rs`·`lib.rs`)에서 `*const`/`*mut`를 만지면 그 자리가 잘못된 층이다 — 버퍼 타입의 메서드로 내린다.

## 2. 타입이 계약을 든다

- **R5** **단위가 다른 수는 타입이 다르다.** 원소 수와 바이트 수, 행과 키와 차원, 워드와 f16 쌍 — `usize` 하나로 다 부르면 8배 예약(ik #2501: `alloc`이 원소 수를 받는데 바이트 수를 넘김)이 컴파일된다. 호스트 코드에서 이런 값이 함수 경계를 넘을 때는 newtype(`Elems(usize)`, `Bytes(usize)`, `Rows`, `Keys`) 또는 최소한 이름에 단위(`n_elems`, `bytes`). 커널 본체 안의 `as` 캐스트는 허용하되(레지스터 산술), 호스트 코드의 좁히는 `as`(`u64 as u32`, `usize as i32`)는 `try_from`으로 — 실패가 불가능하면 그 이유를 `expect("...")`에 쓴다.
- **R6** 형상 상수는 `const`와 const generic으로, 관계는 `const { assert!(...) }`로 컴파일 타임에 박는다(`KEY_TILE == 32`, smem 스트라이드 `≡ 16 (mod 128)`, `MMA_KEYS == MMA_WARPS * MMA_NTILE`). 런타임 `assert!`는 런타임에만 알 수 있는 값(캐시 길이)에만.
- **R7** 결과를 버리면 안 되는 함수는 `#[must_use]`(런치 빌더, 밴드 계산, `segments_for`). 상태를 바꾸는 메서드는 `&mut self`로 — 내부 가변성(`Cell`/`RefCell`/`OnceLock`)은 레버(env 한 번 읽기)와 캐시에만 쓰고 그 사유를 한 줄 적는다.
- **R8** 인자가 8개를 넘는 함수는 파라미터 구조체로(`too_many_arguments` 12건). 커널 런처는 `LaunchArgs`류 구조체 하나 + 스트림. 필드 이름이 단위를 든다(R5).

## 3. 오류 — 라이브러리는 열거형, 바이너리는 아무거나

- **R9** 라이브러리 크레이트(`gguf`·`model`·`gpu`·`qdot`·`threads`)의 공개 함수는 그 크레이트의 오류 열거형(`LoadError`·`ModelError`·`GpuError`…)을 돌려준다. `Box<dyn Error>`·`String` 오류는 바이너리(`gpu-gates/src/bin/*`, `q3k-*`)에서만.
- **R10** `unwrap()`·`expect()`는 게이트·테스트·`main`에서만, 라이브러리 크레이트에서는 0. 예외는 "실패가 불가능함을 타입이 증명하지 못하는" 자리 하나로, `expect("why this cannot fail")` 형태여야 한다.
- **R11** 오류 변형은 **호출자가 다르게 처리할 만큼만** 나눈다. 변형마다 사람이 읽을 컨텍스트(어느 텐서, 어느 커널)를 든다. 드라이버 오류 코드를 그대로 올리는 변형에는 우리 쪽 문맥을 덧붙인다.

## 4. 함수·모듈·중복

- **R12** 오케스트레이션 함수는 두 화면(AGENTS). 커널 본체는 쪼개지 않는다(`#[target_feature]`·`#[kernel]` — 측정된 10–13 % 손실). 커널의 **상수·인덱스 산술**은 `const fn` 헬퍼로 빼도 된다(인라인되고 SASS가 같음 — 뺐으면 `ptx-scan`으로 같음을 보인다).
- **R13** 가시성 기본은 `pub(crate)`. `pub`은 다른 크레이트가 실제로 부르는 것만. 공개 항목에는 doc 주석 한 문장(무엇을 받고 무엇을 돌려주나).
- **R14** **두 번째 사본이 생기는 순간 공통 코어를 뽑는다**(const generic 또는 `#[inline(always)]` 코어 + 얇은 엔트리 — kprobe의 `seg_pass::<TWICE>`가 그 모양). 세 번째 사본은 금지. 지금의 `flash_latent`/`_q8`/`_seg` 셋과 a4c의 `head_groups`/a4d의 `mma_groups`가 이 위반이다.
- **R15** 한 파일이 한 주제. `model.rs` 3901줄은 스텝 오케스트레이션·스크래치 소유·프로브·씨앗 캐시가 섞여 있다 — 분리 축은 "누가 소유하나"(버퍼 소유 = `residency`, 런치 순서 = `step`, 계측 = `probe`).

## 5. clippy — 등급과 래칫

- **R16** 린트 등급 셋: `deny`(적중 0인 것만 — `unsafe_op_in_unsafe_fn`, `missing_safety_doc`), `warn`(0을 목표로 내려가는 것), 그리고 **사유가 있는 `allow`** — 반드시 `#[allow(clippy::x, reason = "…")]`로 항목마다, 크레이트 전역 allow는 금지. 경고 총수는 `AGENTS.md` Known state에 적힌 기준선과 비교해 **오르면 그 라운드가 빨강**이다.
- **R17** 기계적으로 닫히는 것은 한 라운드에 몰아 닫는다(`chunks_exact` 상수 → `as_chunks`, `is_multiple_of`, `len() < 1`, 빈 줄, doc 들여쓰기 — 50건). 그 커밋은 **비트 동일 게이트 + 같은 임대 A/B**를 붙인다(R20).
- **R18** `cargo fmt`는 맥에서, 커밋 전 항상. `just deny`(의존성 핀)는 머지 게이트.

## 6. 디바이스 코드 (cuda-oxide 커널)

- **R19** 커널은 `#![no_std]`·할당 없음·패닉 없음. 인덱스 검사는 `debug_assert!` 대신 **경계가 증명된 접근**(R2) 또는 `SAFETY`가 붙은 `get_unchecked`. 공유메모리 레이아웃(스트라이드·패딩·정렬)은 `const` + `const { assert! }`(R6)로, 뱅크 충돌은 cuda-device `swizzle::conflict_degree`로 컴파일 타임에 센다. 블록 형상은 `#[launch_contract]`에 선언하고(동적 smem 포함 — 48 KB 초과는 `dynamic_shared`로 선언하면 런치가 알아서 켠다), 런처의 런타임 검사와 이중으로 두지 않는다.
- 툴체인이 못 하는 것(`#[unroll]` 실패, 인트린식 부재)은 우회 전에 `docs/upstream/nvlabs-ledger.md`에 한 줄(AGENTS). "없다"는 cuda-oxide 트리와 호스트 크레이트(`cuda-core`·`cuda-bindings`) 둘 다 본 뒤에만.

## 7. 리팩토링의 조건

- **R20** 리팩토링 커밋은 **동작 불변**을 게이트로 증명한다: 토큰 동일(`gate-gpu-e2e` 648노드, `gate-gpu-p8`), 밴드 불변(`gate-gpu-p5`), PTX 모양 불변(`ptx-scan`: 디포 0·레지스터·smem 같음). 그리고 **같은 임대 A/B**로 속도 불변(±1 % 자 — 비트 동일이 속도 동일은 아니다, AGENTS). 둘 중 하나라도 없는 리팩토링 커밋은 머지하지 않는다.
- **R21** 한 라운드 = 한 축. unsafe 주석 라운드, 파라미터 구조체 라운드, 중복 제거 라운드를 섞지 않는다 — diff가 읽히고, 회귀가 나면 축 하나로 귀속된다.
- **R22** 리팩토링은 **동작 변경과 같은 커밋에 두지 않는다**(룩 코어를 UI 커밋에 섞은 사고의 코드판). 커널 성능 라운드가 지나가며 이름을 고치고 싶으면 별도 커밋.

## 7′. 리뷰 1회차(2026-09-22, agy 세 라운드)에서 확정한 예외와 추가 규칙

리뷰어가 제안한 것 중 리드가 원본에서 확인해 받아들인 것만. 기각한 것은 사유와 함께 끝에.

- **R1 보강** — 커널의 연속 벡터 로드(4·8워드 `get_unchecked` 묶음)는 **묶음 하나에 SAFETY 하나**로 충분하다: 청크 경계(`base + 8 <= len`)를 증명하는 한 줄. 공유메모리 배열(`SharedArray`)의 집합적 접근은 표준 문구를 쓴다 — `// SAFETY: block-shared, N == blockDim.x, written before the barrier that publishes it`. "as above"·"as in X"·"cuda-oxide shared array access"는 SAFETY가 아니다(12곳).
- **R8 보강** — `#[kernel]` 엔트리는 인자 수 제한에서 **면제**(디바이스 ABI가 인자를 그대로 받는다). 호스트 `enqueue_*` 런처는 면제가 아니다 — `*Args` 구조체 하나를 받아 커널 런치에 펼친다.
- **R12 보강** — `#[kernel]` 본체는 `#[target_feature]` 본체와 같은 면제(AGENTS의 측정 근거). 게이트 바이너리의 `main`은 선형 파이프라인이라 120줄이 아니라 **논리 단계(로드 / eager / graph / 타이밍) 단위 분리**까지만 요구한다.
- **R9 예외** — `gpu-gates`의 `lib.rs`(게이트 보조 라이브러리)는 `Box<dyn Error>`를 유지한다: 소비자가 전부 게이트 바이너리고 실패는 곧 종료다. `crates/gpu`의 `GpuError = Box<dyn Error>`는 예외가 **아니다** — R9 위반이고 라운드 대상.
- **R10 예외** — 포이즌 락 회수(`lock().unwrap_or_else(|e| e.into_inner())`)는 `unwrap`이 아니다. 벤치 루프 안 `launch().unwrap()`(gate_moe_fused/p0b `us_per_replay`)은 바이너리라 R10 대상 아님.
- **R23 (신설)** — **부동소수 비교·정렬은 `total_cmp`**. `partial_cmp().unwrap()`은 NaN에서 패닉이고 NaN은 게이트가 잡아야 할 값이지 패닉 사유가 아니다.
- **R24 (신설)** — **`let x = e.unwrap() else { … }`는 죽은 코드다**(unwrap이 먼저 패닉). `ok_or_else(..)?` 또는 `match`. 검증 헬퍼의 `ok: &mut bool` 누적 인자도 같은 부류 — `bool`/`Result`를 돌려준다.
- **R25 (신설)** — **그래프 핸들은 그것이 주소를 잡은 버퍼보다 먼저 선언한다**(필드는 선언 순서로 드롭). `Head`·`GpuModel`·`Stage`가 이미 그 순서이고, 새 구조체도 같은 순서 + 한 줄 주석.
- **R26 (신설)** — **피처 뒤에 본체가 숨은 바이너리는 그 피처를 켜고 lint한다.** `just lint`·`just check`는 `--features gpu`. 러너(`depth-gpu.sh`·`ncu-gpu.sh`·`nsys-gpu.sh`)는 바이너리를 **빌드하거나 빌드 해시를 증인에 찍는다** — 빌드하지 않은 바이너리를 재는 것은 낡은 바이너리 사고의 형태다(2026-09-22 밤 `ik_ref` 열이 그렇게 옛 값을 찍었다).
- **R27 (신설)** — 통합 테스트가 내부 함수를 부르기 위한 `#[doc(hidden)] pub`은 금지. 인라인 유닛 테스트 또는 게이트가 관찰하는 출력으로.

**기각**: ① `flash_row_scalar`/`flash_row_avx2` 트윈의 공통 코어 추출(cpu 리뷰 1순위) — `#[target_feature]` 본체 분리 금지(AGENTS, 측정 10–13 %)와 정면 충돌. 트윈은 의도된 형태고 `TWIN` 주석이 계약이다. 남는 것은 트윈 **밖**의 소프트맥스 스캔 산술을 `#[inline(always)]` 코어로 빼되 SASS·A/B 동일을 보이는 것 — 별도 라운드, 지금 아님. ② `head.rs` 드롭 순서 결함 — 이미 그래프가 먼저다(오탐; 규칙 R25로만 받음). ③ "cuda-oxide가 구조체 인자를 못 받는다"는 리뷰어 주장은 미확인이라 R8 보강의 근거로 쓰지 않았다(ABI 이유로 충분).

## 8. 리뷰 절차

1. 리뷰어(조사 라운드)는 크레이트 하나씩 읽고 위반을 `경로:줄 — R번호 — 한 줄 — 크기(XS/S/M/L)`로 낸다. 심각도는 리뷰어가 매기지 않는다 — 규칙 번호가 곧 분류다. 리뷰어는 코드를 고치지 않는다.
2. 리드가 항목마다 셋 중 하나로 처분한다: 지금 라운드 / 트래커 / 근거를 들어 기각(그 근거는 이 문서의 규칙 예외로 올라오거나 `allow(reason)`이 된다).
3. 리팩토링 라운드(구현)는 축 하나(R21), 파일 화이트리스트, R20의 게이트·A/B를 완료 조건으로 받는다.
4. 라운드가 끝나면 §0 표를 다시 재고 이 문서를 갱신한다. 표가 오르는 방향으로 움직이면 그 라운드가 원인이다.
