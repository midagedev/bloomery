# 게이트 선택의 커널 폐포 증명 설계 — 원문

2026-09-26. opus 서브에이전트 설계 라운드 `gatesel`의 보고 원문이다(스펙 `specs/wave-m6/spec-gatesel.md`, 박스 없음, git과 착륙 로그만 읽음). [유도] 수치는 라운드의 추론이다. 리드가 원본에서 확인한 것: `Gpu::with_device`가 같은 번들을 아홉 번 적재한다(`crates/gpu/src/lib.rs:1877-1886` — `kernels::load`와 `XKernels::load` 여덟), `ptx::normalize`가 번호 붙은 생성 심볼을 줄기로 지워 엔트리 digest가 전역 초기값·공유 메모리 크기를 보지 않는다(`crates/gpu-gates/src/ptx.rs:418-442`). 처분은 `docs/plan-triage.md`의 gatesel 항목에 있다.

## 보고 (`gatesel`, 워크트리 `/Users/hckim/repo/bloomery-gatesel`, base `af929ae`)

**gatesel 설계 보고.** 박스는 쓰지 않았고 저장소에는 아무것도 쓰지 않았습니다. git은 읽기만 했고, 사용자가 제 세션에 직접 입력한 것은 없습니다. 스크래치 파일은 `/private/tmp/claude-501/-Users-hckim-repo-bloomery/02d044bc-8d2a-4055-91fa-4ae541a4916f/scratchpad/gatesel/`에 있습니다: `rustitems.py`, `backtest.py`, `wall.py`, `bt-all2.tsv`, `aff-*.txt`, `dry-all.txt`.

**결론:** 지금까지의 착륙 이력에서는 증명으로 건너뛸 수 있는 항목이 아주 적습니다.
- 소스 수준 방안(c)은 백테스트에서 사실상 0이라 만들지 않는 것이 맞습니다.
- 커널 본문만 바뀐 착륙은 116개 중 **0개**였습니다. 커널 변경에는 항상 호스트 쪽 변경이 같이 들어왔습니다.
- 큰 이득이 나는 경우는 q3router 같은 "아키텍처 국소" 착륙 하나입니다(3,143 → 394 s [유도]). 이것은 **함수 단위 바이너리 동일성(b)이 성립할 때만** 증명됩니다.
- 그래서 **7번 측정을 먼저** 하고, 결과에 따라 만들지 말지 정하자는 것이 권고입니다.

## 1. 방안 표

| 방안 | 무엇이 건너뛴 항목을 빨갛게 만들 수 있나 | 크기 | 착륙당 박스 비용 | 건너뛰는 비율 [유도] |
|---|---|---|---|---|
| (a) 바이너리 전체 동일성 | 동일성 판정 자체는 건전합니다. 다만 거의 성립하지 않습니다. ① `.oxart`가 `SHF_ALLOC`입니다(oxide-artifacts `lib.rs:793`@b9847e9). 번들 크기가 바뀌면 그 뒤의 배치가 모두 밀립니다. ② 트랙 경로가 바이너리에 박힙니다(`env!("CARGO_MANIFEST_DIR")` 6곳, 예: `gate_e2e.rs:667`). ③ CGU 재분할 | S. 섹션 고정 링크 플래그, 리드 고정 원격 디렉터리, PT_LOAD 해시 | 선택된 bin 선빌드(대부분 실행 항목과 공유), 해시 수 초 | 측정 필요. 커널이 바뀐 착륙에서는 ① 때문에 0 |
| (b) 함수 단위, 재배치 기호화 | 링크된 바이너리에 남은 **모든** 함수·데이터 객체를 비교하므로 도달성 논증이 필요 없습니다(`--gc-sections`가 이미 미참조 함수를 뺐습니다). 남는 위험은 CGU 재분할로 무관한 함수의 코드가 바뀌는 것뿐이고, 그 경우 보수적으로 실행합니다 | L. `-Wl,--emit-relocs`, 익명 데이터 경계, 경로 문자열 정규화 | 선빌드 + 지문. 50 MB bin 40개 objdump 병렬 ≈ 수십 초 [미측정] | 측정 필요. q3router에서 48/59 후보 |
| (c) 소스 hunk + 커버리지 | const fn·중첩 `impl`·`impl Trait` 반환·매크로·줄 번호 상수. 항목 수준 hunk는 전부 크레이트 닫힘으로 폴백 | L. 정확한 Rust 파서 + 커버리지 빌드 + 항목별 커버리지 실행 | 녹색 기록마다 계측 빌드와 재실행(≈ 착륙 묶음 1배) | 항목 ≤ 276/5,988 (4.6 %, 상한), 창 안 벽시계 0 |
| (d) (b) + 번들 동일성, 이후 엔트리 단위 | (b)와 같고, 디바이스는 아래 2절의 닫힘 | (b) + S(번들 해시) + M(발사 기록) | 위와 같음 | 1단 ≈ N/X 항목 2.8 %, 2단 q3router류 |

## 2. 디바이스 쪽과 발사 집합 (질문 1–2)

**PTX가 바이너리에 들어가는 경로.**
- 디바이스 크레이트마다 번들 하나, 그 안에 PTX 페이로드 하나입니다(bloomery-gpu, bloomery-gpu-deepseek41, gpu-vision).
- 이 중 ptx-scan이 실제로 읽는 것은 둘입니다. `generate_ds41`는 modules=2, `gate_e2e`는 modules=1이고, 둘 다 `calls=`가 없습니다(fixup6 `gb1/g-gate-ptx-spill.log:40-43`).
- 파서 경로: `ptx.rs:107-113`, `:154-166`. Cubin이 PTX를 가리면 거부합니다(`:131-149`).
- 제네릭 커널은 0개입니다. 그래서 같은 빌드 안의 모든 bin이 같은 rlib 멤버, 즉 같은 번들을 싣습니다.

**모듈 적재가 번들 전체에 걸립니다.**
- `#[cuda_module]`의 `load`는 `load_embedded_module(ctx, CARGO_PKG_NAME)`을 부르고(cuda-macros `mod.rs:180,278`), 이것은 **번들 PTX 전체**에 대해 `cuModuleLoadData`를 합니다(cuda-host `embedded.rs:56,170`).
- 이어서 `from_module`이 그 모듈의 **모든** 커널을 이름으로 찾습니다(`mod.rs:124`).
- `Gpu::with_device`는 이것을 bloomery-gpu에 대해 9번 합니다(`crates/gpu/src/lib.rs:1877-1886`).
- 결과: 한 게이트의 디바이스 닫힘은 "발사한 엔트리" + "적재하는 번들 전체가 JIT된다"입니다. 발사하지 않는 엔트리가 JIT에 실패해도 그 번들을 적재하는 모든 게이트가 빨갛게 됩니다.

**정규화 본문 + 피호출 `.func`만으로는 닫힘이 아닙니다.** `normalize`에서 빠지는 것은 넷입니다(`ptx.rs:418-442`, `:55-59`).
- `__device_global_N` 참조의 번호를 지웁니다. 그래서 전역 **초기값**이 안 보입니다. 예: IQ 코드북은 호스트 크레이트 상수 `gguf::iq_tables`에서 오는데, 이것이 디바이스 전역 하나가 됩니다(`iq.rs:47-53,100-102`). 정의된 전역은 external linkage를 유지합니다(llvm-export `function.rs:189`).
- `__shared_mem_N` 선언의 크기가 안 보입니다.
- 본문은 `.func`에서 끝나므로 피호출 함수가 안 보입니다(`:203-212`).
- 모듈 헤더(`.version/.target`)가 안 보입니다.
- 나머지 하나, 모듈 footprint(코드 + 전역이 차지하는 카드 메모리)는 텍스트가 아니라 적재 결과라서 digest 밖입니다. 쓰이는 곳: `gate_load_v41.rs:380-387`이 "적재 뒤 free ≥ 바닥"을 단언합니다. 배치 자체는 상수로 계산하므로(`workstation.rs` CONTEXT/MARGIN) 단조 논증이 성립합니다.
- 그러므로 지금 관행("기존 PTX 엔트리 digest 동일"로 30개를 뺀 것, `docs/gates-plan.md:70`)은 **충분한 증명이 아닙니다.**

**있는 도구로 엔트리별 비교가 되는가.** 본문은 됩니다: `oxart_ptx --norm`(`oxart_ptx.rs:54-62`)과 md5 블록(`ptx-scan.sh:300-310`). 다만 digest 블록이 ptxas 열과 게이트 락 아래 JIT 열까지 요구하고(`ptx-scan.sh:150-196`), 위의 모듈 범위 선언을 못 봅니다.

**발사 집합: 동적 기록을 택합니다.**
- 정적 방식은 성립하지 않습니다.
  - 발사 메서드는 커널마다 생성되고, 적재 때 찾아 둔 `CudaFunction`으로 `cuda_core::launch_kernel*_on_stream`을 부릅니다(`launchers.rs:1108-1155`). 엔트리 이름은 적재 시점(`from_module`)에만 나오고 발사 지점에는 없습니다.
  - `GpuModel<Body>`나 serve의 `Engine` trait 너머로 Rust 호출 그래프를 풀어야 하는데, 링크된 바이너리에서는 발사 메서드가 인라인됩니다.
  - 그래서 정적 방식은 건전하지 않거나, 결국 "모든 커널"이 됩니다.
- 이름으로 함수를 찾는 곳은 워크스페이스에서 `oxart_jit.rs:69` 하나뿐입니다(grep).
- 동적 기록 방법은 둘입니다.
  - ① CUPTI injection(`CUDA_INJECTION64_PATH`): 드라이버의 모든 발사 CBID와 그래프 노드 API를 잡습니다. 모르는 kernel-node CBID가 오면 기록을 incomplete로 표시합니다. 엔진 코드를 바꾸지 않습니다. 크기 M, 발사당 비용은 미측정입니다.
  - ② cuda-macros 훅: 발사당 relaxed atomic 하나입니다. 대신 포크 패치라서 pin 이동이 필요합니다.
  - 권고는 ①입니다.
- 그래프 캡처 중 발사는 같은 경로를 지나므로 노드마다 한 번씩 기록됩니다. 리플레이는 기록할 필요가 없습니다. 기록 내용은 args·env·card·데이터에 따라 달라지고, 이것들은 이미 원장 키에 있습니다.

## 3. 백테스트 (`affected A..B --no-box`, 오늘의 도구)

**조건.** 창(`1de374c` 이후) 코드·도구 착륙 13개. 확장 표본은 09-24 이후 `crates/`를 건드린 first-parent 116개(선택 ≥ 1인 착륙 109개, 항목 5,988개)입니다.

**분류기와 그 한계.** 정규식 렉서입니다. 중첩 `impl`·`no_mangle`·`extern` 토큰이 있으면 I, 파싱 실패도 I, `#[cfg(test)]` mod와 `#[test]` 안이면 X로 분류합니다. 분류 결과: I 4,750 / 스크립트·레시피·Cargo 같은 비 Rust 입력 960 / B 111 / X 115 / N 52 / **K·M만 있는 것 0**.

**벽시계 모델.** `max(A,B)+X`이고, 차선은 러너 dry-run에서 가져왔습니다(A 41 / B 48 / X 3). 항목 시간 출처: fixup6 로그 43개, `gate-times.tsv` 23개, gates-plan 1개, 나머지 32개는 45 s 기본값. ptx-spill은 조용할 때 값 59 s를 썼습니다.

| 착륙 | 파일 | 선택 | (c) 건너뜀(상한) | (a) | (b)+번들 | 오늘 벽시계 → 방안별 |
|---|---|---|---|---|---|---|
| 823aac8 q3fix2 | 21 | 90 | 24 (CPU, B차선) | 0 [유도: `fault.rs` I hunk가 모든 GPU 경로에] | 바이너리 필요, ≈0 예측 | 3,143 → 3,143 |
| e0fea62 qdot | 2 | 85 | 0 | 바이너리 필요 | 바이너리 필요, ≈0 예측 (`model.rs:35` 경유 hybrid→qdot) | 3,143 → 3,143 |
| c58cb37 hostserve | 9 | 84 | 0 | 바이너리 필요 | ≈0 예측 | 3,143 → 3,143 |
| ef00f78 q3router | 5 | 59 | 0 | 측정 필요 (CGU) | 1단 0(bloomery-gpu 번들 바뀜), **2단 후보 48** | 3,143 → 2단 394 |
| 5089bba fixup6 | 14 | 65 | 0 | 0 | 0 (아래 설명) | 3,143 → 3,143 |
| e8605c7 | 5 | 2 | 1 | 자기 bin 변경이라 실행 | 실행 | 182 |
| e4ae993 | 2 | 3 | 0 | 레시피 텍스트라 실행 | 실행 | ≈393 |
| 1de374c · 5b793b7 · 1633232 · afe86d5 · 05ec804 · 085b1c3 (도구만) | — | 0 | — | — | — | 검사 7개 ≈14 |

- **fixup6이 0인 이유.** q4k_sel에 커널 `q8_1_quantize_sel`이 새로 들어왔습니다(ptx-shapes +2). 그러면 생성되는 `LoadedModule`과 `from_module`이 바뀝니다. 이 모듈은 `Gpu::with_device`(`lib.rs:1880`)에서 적재되므로 **모든 GPU 바이너리의 호스트 코드가 바뀐 것**입니다. 리드가 fixup6에서 뺀 29개는 어느 방안으로도 증명되지 않습니다.
- **확장 표본 합계.** 벽시계 합 239,552 s → (c) 상한 230,092 s (−3.9 %) → N/X만 건너뛸 때 232,517 s (−2.9 %) [유도].
- **많은 게이트가 발사하지 않는 커널만 바꾼 착륙.** ef00f78(qwen3moe 라우터 전용 커널)이 해당합니다. 다만 같은 착륙에 `q8f32.rs` 새 헬퍼(I)가 있어서 파일 수준 판정으로는 "공유 호스트 변경"이 되고, (b)여야만 잡힙니다. 파일 수준에서 V4.1 전용으로 분류된 것은 6개(ed3d6f3, 8dd878f, 675e09f, 6ec8baa, f56e770, 4f230f5)이고, qwen3moe 전용은 0개입니다.

## 4. 권고 · 증명 줄 · 녹색 기록

**권고.** (c)는 만들지 않습니다(백테스트 결과). (a)는 채택하지 않습니다. 커널이 바뀐 착륙에서는 `.oxart`가 배치를 밀어 동일성이 깨지고, 핀·경로 고정 두 가지를 해도 CGU에 취약합니다. **7번 측정 결과를 먼저 보고 (d)로 갑니다.**
- 1단: (b) 호스트 함수 표 + 번들 원시 페이로드 sha256.
- 2단: 번들이 바뀌었어도 다음이 모두 성립하면 건너뜁니다.
  - 발사한 엔트리의 닫힘 digest가 같다. 닫힘 = 본문 + 이름으로 해석한 모듈 범위 선언과 초기값 + `.func` + 헤더.
  - 같은 묶음에서 `gate-ptx-spill`이 녹색이다. 이것이 적재 시 JIT 증명입니다(`ptx-scan.sh:193`, 모든 모듈을 적재). `affected`가 `crates/gpu`가 바뀔 때마다 선택하므로 따로 탐침을 만들 필요가 없습니다.
  - 번들 footprint가 녹색 때보다 크지 않다(`oxart_jit`에 `cuMemGetInfo` 차이 출력 추가).

**건너뛰어도 그대로 두는 예외.**
- `gate-batch.sh:114-133`의 한계 목록 전부를 상속합니다.
- 추가로 적을 것:
  - `GO_DEADLINE` 10 s(`hybrid.rs:457`)와 `ANSWER_DEADLINE` 10 s(`launcher.rs:96`): 박스가 멈추면 입력이 같아도 빨개집니다.
  - solo 게이트의 page-cache·mincore 핀, Xid 79.
- `recipes.py`의 `never` 조건(참조 트리, smoke, 절대경로 ARGS, `--lanes 1`의 `any`)도 상속합니다.

**증명 줄 형식.**
```
gate-gpu-ds41-hc rc=skip proof=host:9f3c…@gate_deepseek41_hc bundles=bloomery-gpu:1a2b…(entries 31/31 same, jit=gate-ptx-spill@batch, fp 4MiB<=4MiB),bloomery-gpu-deepseek41:=  data=manifest:ab12… env=BLOOMERY_GATE_CARD=3090 green-at=<commit> <date> lane=A
```
- 증명이 불완전하면 `rc=run proof-incomplete=<이유>`를 찍고 실행합니다.
- 강제 실행은 기존 `--rerun`으로 합니다.

**녹색 기록.** 원장 파일은 그대로 두고, 키는 `bloomery-gate-key 2`로 올립니다(한 번 전량 재실행). `<ledger>.parts/<key>.parts`에 줄을 더합니다: `host <bin> <sha>`, `bundle <crate> <sha>`, `entry <crate> <name> <md5>`(발사 기록분), `footprint <crate> <card> <bytes>`. 소스 `.rs` file 부분은 host/bundle 부분으로 대체하고, 스크립트·데이터·레시피 부분은 그대로 둡니다.

## 5. 구현 라운드

| 순서 | 라운드 | 파일 | 크기 | 충돌 |
|---|---|---|---|---|
| 0 | 리드: 6절 측정 | — | 박스 ≈10분 | 시팅 밖, 임대 불필요 |
| 1 | `gatesel-dev` | `crates/gpu-gates/src/ptx.rs`(closure + 테스트, 기존 normalize로 FAIL-first), `oxart_ptx.rs --closure`, `ptx-scan.sh` digest 블록 | S | 없음 |
| 2 | `gatesel-host` | 새 `tools/host-fingerprint.py`(박스), `.cargo/config.toml`·`cuda-oxide.toml`에 `-Wl,--emit-relocs`, check-rustflags | L((a)이면 M) | cargo 전역이라 전 키 이동. 파동 경계에 착륙, bloomery-ee 라운드 전량 재빌드 |
| 3 | `gatesel-ledger` | `tools/recipes.py prove`, `tools/gate-batch.sh`(ledger_plan 앞 prepare 빌드+지문 한 번, skip 줄) | M | **toolsfix 뒤**에 착륙. 그 전이적 스크립트 워커를 재사용하고 `exec_closure` 중복을 제거 |
| 4 | `gatesel-launch` | `tools/ref/launch-record.c` + 빌드 스크립트, `oxart_jit.rs` footprint, prove 2단 규칙 | M | 3 뒤 |

어느 단계도 박스 30분 규칙에 가깝지 않습니다.

**fixture와 합친 예측** [유도].
- fixture 티어는 A를 646–783 s로 줄입니다(fixture 보고 §6). 그러면 무거운 착륙의 속박 차선이 B가 됩니다.
- q3router류: fixture만 쓰면 ≈986 s(B), 2단까지 쓰면 394 s.
- fixup6류: gatesel 0이라 fixture의 674–783 s 그대로입니다.

## 6. 나중 슬롯의 측정 하나 (돌리지 않음)

- **절차.** base `af929ae`에서 `generate_ds41`와 `gate_e2e`를 빌드합니다. 이어 탐침 셋을 따로 적용합니다.
  - P1: `crates/gpu`에 호출되지 않는 `pub fn` 하나 추가.
  - P2: qwen3moe 라우터 커널 본문만 수정.
  - P3: `arch/qwen3moe/router.rs`와 `q8f32.rs`에 함수 추가(q3router 모양).
  - 각각 base와 (a)(PT_LOAD − `.oxart`) 및 (b)(`objdump -dr` 함수 표)로 비교합니다.
- **기대값.**
  - (a): P2는 다름(`.oxart` 이동). P1·P3는 CGU 병합이 그대로면 같음(불확실). 참고로 스펙의 "(a)는 no"라는 전제는 고칠 필요가 있습니다. 호출되지 않는 함수는 gc-sections가 뺍니다.
  - (b): P1·P2·P3 모두 V4.1 bin 함수 표가 같음. 단 CGU 탓에 bloomery-gpu 함수 일부가 달라질 수 있음(0 이상).
- **비용.** 빌드 4회 × ≈75 s(fixup6 30.1 + 41.1 + 3.8 s) + 비교 → ≈10 박스분 [유도].
- **판정.** (b)에서 무관한 함수가 크게 흔들리면 이 설계를 접고 fixture만 갑니다.

## 7. 범위 밖 개선 후보 (보고만)

- `crates/gpu/src/lib.rs:1877-1886`: 같은 번들을 9번 적재합니다. 매번 실행 파일 읽기 + `cuModuleLoadData`입니다(`embedded.rs:56-67`). 한 번 적재하고 `from_module`로 나누면 됩니다. S, 비용은 미측정.
- `ptx-scan.sh:49-57`: "digest = 같은 코드"라는 주장이 모듈 범위 선언 때문에 거짓입니다. 1번 라운드로 고치거나 문구를 정정합니다. XS.
- `gate_e2e.rs:667` 외 5곳: `env!("CARGO_MANIFEST_DIR")`가 트랙 경로를 바이너리에 박습니다. XS.
- `tools/recipes.py` `cmd_box_manifest`: libc·libstdc++ 버전이 매니페스트에 없어서 glibc가 올라가도 키가 안 움직입니다. XS.
- `gate-batch.sh:114-133`: 10 s 데드라인 둘(`hybrid.rs:457`, `launcher.rs:96`)이 한계 목록에 빠져 있습니다. XS.
- `tools/recipes.py` `select`: `#[cfg(test)]` 전용 hunk가 모든 bin을 선택합니다. 백테스트에서 115건이었습니다. `--lib` 항목만 남기면 건전하지만 렉서가 정확해야 합니다. S.

## 8. 모델

opus(Opus 5.5)로 돌았습니다.
