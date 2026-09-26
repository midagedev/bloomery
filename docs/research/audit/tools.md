# audit-tools (TL) 보고

경로는 모두 `/Users/hckim/repo/bloomery` 기준 상대 경로입니다. rig-log는 `/Users/hckim/repo/rig-log`(`9e223e4`)입니다.

## 0. 읽은 범위

- **기준 커밋.** 시작할 때 `4786c1e`였고, 읽는 동안 main이 `fa61362`로 6커밋 움직였습니다(`b00af8f` q3fa, `031b842` q3swz, `9903025` q3pp.py, `e2b2b68` srcpage.py 신설, 문서 둘). 인용 줄 번호는 `4786c1e`의 것입니다. 새 커밋이 건드린 `q3pp.py`와 `plan-triage.md`는 `git show 4786c1e:`로 다시 재서 같은 값을 확인했습니다. `NCU_*` 개수만 작업 트리 기준이라 srcpage.py가 섞였습니다.
- **참조 트리.** ik_llama.cpp `41b17995`, mistral.rs `b0f26d5cd`를 봤습니다. exllamav3(박스)는 열지 않았습니다.
- **전부 읽음.**
  - 도구: `tools/box.sh`(116), `tools/ref/depth-ds41.sh`(651), `tools/ref/measure.sh`(52), `tools/gate.sh`, `tools/gpu-gate.sh`, `tools/check-recipes.sh`(126), `tools/check-rustflags.sh`, `tools/ref/ref-paths.sh`, `tools/ref/models/deepseek4.sh`
  - 설정: `Cargo.toml`, `.cargo/config.toml`, `.cargo/cuda-oxide.toml`, `deny.toml`, `rust-toolchain.toml`
  - 문서: `AGENTS.md`(669), `CLAUDE.md`(38), `docs/gates-plan.md`(136), `docs/cards/README.md`, `docs/cards/prefillgroup-pp.card`
- **머리말·구조·표본만 읽음.**
  - `tools/recipes.py`(2,470: 1–80, 154–410, 850–895, 함수 목록)
  - `tools/gate-batch.sh`(1,060: 1–120, 255–335, 함수 목록)
  - `tools/ref/lease.sh`(484: 1–81, 함수 목록)
  - `tools/flow/ds41_prefill.py`(3,156: 1–90, 179–202, 함수 목록, `ROWS` 1775–1792)와 `constants.tsv`(kind 열 집계)
  - `tools/ref/ds41pp.py`(1–60, 127–163), `tools/ref/q3pp.py`(92–145), `tools/ref/card.py`(228–245), `tools/gpu-ab.py`·`tools/verdict-diff.py`(머리말)
  - `tools/ref/depth-qwen3moe.sh`(함수 목록 + 공유 함수 diff), justfile(1–60, 115–126, 928–945, grep)
  - `docs/plan-triage.md`(1–32행 + 바이트·취소선 집계), `docs/plan.md`·`docs/plan-ledger.md`(절 머리), docs 15개의 머리
  - `docs/upstream/nvlabs-ledger.md`와 `docs/research/rustmodern-report.md`(grep)
- **grep으로 전수 조사.** crates·tools·justfile의 `BLOOMERY_*` 전부를 뽑았고, 엔진 레버 파서는 전부 그 자리에서 읽었습니다. rig-log의 레버 판정 줄, `tools/` 커밋 215개도 훑었습니다.
- **안 읽음.** `ncu-gpu.sh`(노브·폼 수만 셈), `nsys-*.sh`, `timing-card.sh`, `lease-hold.sh`, `ik-*.sh`, `dump*.sh`, `router*`, `ptx-scan.sh`, `sass*`, `box-gc.sh`, `box-tracks.sh`, `check-arch.sh`, `check-comments.sh`, `card.py` 본문, `card-tests/`, C++ 하네스, `docs/research/*` 본문 81개, triage 33행 이후, ledger·`v41-inventory`·`v41-placement`·`arch-split`·`roofline`·`oracle`·`facts` 본문.
- **실행한 것.** 맥에서 `tools/gate-batch.sh --classes`를 한 번 돌렸습니다(분류 인쇄만). 뒤에 `git status --porcelain`이 비어 있고 `target/gate-batch/`에 새 파일이 없음을 확인했습니다. 박스 명령, 잠금, 게이트는 쓰지 않았습니다.

## 1. 지금 아는 것 (시드 밖)

1. **레버에 공용 파서도 목록도 없습니다.** 엔진 레버는 쓰는 자리에서 각자 환경을 읽습니다. 파서 관용구가 약 11가지, 오류 경로가 넷(panic, `GpuError`, `PlacementError`, `GateError` 문자열)입니다. 뜻 모를 값을 조용히 기본값으로 받는 레버가 13개입니다. 대부분 "No silent failure"(AGENTS.md:308, 09-24)보다 먼저 들어왔습니다(`e218fb8` threads 09-19, `a7e7c17` 09-20).
2. **레버가 프로세스 전역 `OnceLock`이라 시험이 두 팔을 한 프로세스에서 못 돕니다.** 자기 바이너리를 다시 띄우는 자식 프로토콜이 7개(`current_exe()`: tests/attn.rs:992·1067·1359·1543·1692, mt.rs:65, profile.rs:65)이고, 덮어쓰기 훅이 둘입니다(`ops::set_defer_quant` ops.rs:2289, attn.rs prefetch override). 테스트 본문이 직접 이렇게 적습니다: "the only way a process-wide flag can be: a re-exec'd child"(tests/attn.rs:968).
3. **게이트 묶음의 벽시계는 모델 적재가 지배합니다.** fixup5 묶음은 레시피 48개에 2,167 s였고, 30 s 미만 32개(합 1,154 s)가 "거의 전부 적재 비용"입니다. V4.1 게이트 25개 × 적재 30 s ≈ 12.5분입니다[유도, gates-plan §1]. prefill 게이트는 한 팔 386 s, 두 팔 745 s입니다[실측, 같은 곳].
4. **측정 프로토콜 개정 한 번이 도구 파일 약 30개를 고칩니다.** `085b1c3` predcard 31, `66c121a` 33, `9ed73ff` 29개입니다[실측, `git show --stat`].
5. **justfile 구조를 세 프로그램이 따로 읽고, 스크립트 워커 둘의 규칙이 다릅니다.**
   - `just --dump` 소비자: recipes.py:113·1721, gate-batch.sh:224(내장 파이썬), check-recipes.sh:58(내장 파이썬).
   - 워커: gate-batch.sh:259·304는 `tools/…\.sh`와 `.sh/.bash`만 따라갑니다. recipes.py:327·856–893은 `tools|crates/` 경로 전부와 확장자 여덟을 따라갑니다.
6. **레시피 분류와 기본 프로필.** `--classes` 결과는 고정(3090) 53, 균형 89, 타이밍 50, solo 3입니다. 레시피 줄의 `BLOOMERY_MODEL=`은 deepseek41 52, qwen3moe 21, deepseek4 1이고, 나머지는 기본 프로필 `deepseek2`(V2-Lite)로 돕니다. 이 기본값은 ref-paths.sh:35와 recipes.py:79 두 곳에 있습니다.
7. **바이너리 출력 줄의 스키마 주인이 여럿입니다.**
   - `SMOKE`: 쓰는 곳 4(generate.rs:404, generate_ds41.rs:970·1230, generate_qwen3moe.rs:230), 파싱하는 도구 8
   - `time prompt`: 쓰는 곳 2(generate_ds41.rs:837, generate_qwen3moe.rs:204), 읽는 곳 6
   - `stat prefill split`: 읽는 곳 3
8. **흐름 모델은 값을 손으로 옮겨 적었고, 그 인용이 이미 어긋났습니다.** 상수 148개(실측 86, 유도 57, 가정 5)와 백테스트 행 약 104개가 파이썬 튜플(`ROWS`, ds41_prefill.py:1751~)입니다. Rust 코드 인용은 `path:line` 54곳입니다. `4786c1e`에서 틀린 것:
   - `body/prefill.rs:69`(T_MAX)와 `:72`(CHUNK) 인용 → 지금은 모듈 주석이고, 상수는 :92·:95에 있습니다.
   - `:877-882` "an eager stream" 인용 → 지금은 `StepImage` 생성입니다.
   - `batches` 137-146 인용 → 실제 :162
   - `chunks` 466-476 인용 → 실제 :841
9. **배치·청크 분할 규칙이 세 벌입니다.** Rust prefill.rs:162·841, ds41pp.py:127·138, ds41_prefill.py:179·190입니다. `T_MAX=512`와 `CHUNK=8`도 두 파이썬 파일에 손으로 적혀 있습니다(ds41pp.py:50-51, ds41_prefill.py:152-153). q3pp.py:92-145는 반대로 Rust 소스를 정규식으로 읽고 바뀌면 이름을 대고 거부합니다. 같은 문제를 세 전략이 푸는 셈입니다.
10. **열린 일의 주인이 둘입니다.** CLAUDE.md와 AGENTS.md 마지막 관례는 트래커 MUL을 지정합니다. 그런데 커밋 메시지의 `MUL-` 참조는 09-20의 62회 뒤로 하루 0–1회입니다[실측, 커밋 본문 grep — 트래커 자체 상태는 조회하지 않았습니다]. 대신 `docs/plan-triage.md`가 09-24에 23번, 09-25에 147커밋 중 89번, 09-26에 92커밋 중 52번 고쳐졌습니다.
11. **매 라운드가 싣는 문서 크기.** AGENTS.md는 53,502바이트이고 모든 세션·라운드에 실립니다(`@AGENTS.md`). triage는 255,118바이트이고 Read 도구 기준 약 138k 토큰입니다.
12. **쌍둥이 러너가 이미 갈라졌습니다.** `depth-ds41.sh`와 `depth-qwen3moe.sh`는 651행 중 378행이 다릅니다(공통 약 273행). 차이는 이렇습니다.
    - `guard_cpu`(CPU 경합 가드): ds41 쪽에서만 5번 부르고 qwen3moe 쪽은 0번입니다.
    - `ratio_table`: ds41 쪽만 `base` 인자와 `[cpu-busy]` 집계가 있습니다(507–537 대 529–560).
    - `tree_line`: ds41 쪽만 box.sh의 `BLOOMERY_GIT_COMMIT`를 읽습니다(263–275 대 276–282).

## 2. 구조 발견 (보상/비용 순)

### TL-1 레버는 쓰는 자리마다 환경을 직접 읽는 프로세스 전역 싱글턴이다

**어디·지금 모양.** 엔진 레버 약 40개가 각자 `std::env::var`를 부릅니다(표). 공용 헬퍼는 파일 안에만 있습니다(`lever_var` ops.rs:2718, `flag` hybrid.rs:208).
- 한 레버를 여러 곳이 읽습니다. `STEP_STATS`는 6곳(glue.rs:1105, gate_deepseek41_prefill.rs:332, generate_ds41.rs:548·608·916·1042), `PIN_MAIN`은 바이너리 5곳, `STEP_PAIR`는 3곳입니다.
- 두 바이너리는 엔진이 고른 값이 아니라 환경을 다시 읽어 "무엇이 돌았나"를 찍습니다(`CardExperts::from_env` generate_ds41.rs:492, gate_deepseek41_prefill.rs:365).
- 셸도 같은 레버를 해석합니다(`BLOOMERY_DRAFT=dspark`를 timing-card.sh:32, depth-ds41.sh:437에서 문자열로 비교).
- 문서는 AGENTS.md:538–669(132행) 산문 하나이고, 엔진 레버 다섯이 빠졌습니다: `DEFER_QUANT`, `PREFILL`, `POPULATE`, `WEIGHTS`, `EXPERT_LOG`.
- 끝 문장 "Each is read once, at first use"는 `RowsLevers::from_env`(glue.rs:1102, `OnceLock` 아님)에서 거짓입니다.

**어떻게 쌓였나.** 라운드마다 같은 바이너리 팔을 하나씩 더했습니다: `e218fb8`(09-19), `a7e7c17`, `61c752b`, `bdf7914`, `4e4e643`, `382578a`, `0c330bb`, `43944e4`, `9c8ebf0`, 그리고 CED·`9d61a13`→`237321a`·`eb3e08a`. 파서 모양은 들어온 날의 규칙을 따랐습니다. 09-24 이전 것은 조용한 기본값, 이후 것은 이름 붙은 오류입니다.

**오늘 만든다면.** 환경 변수는 운반 수단으로 남깁니다. `BLOOMERY_BOX_ENV`가 레시피를 고치지 않고 박스까지 레버를 싣는 길이기 때문입니다(box.sh:75–90). 바뀌는 것은 해석하는 자리와 값의 모양입니다.
- **등록부 하나.** `LeverSpec { name, class, default, parse, doc }`를 작은 크레이트 하나에 둡니다. `--levers`가 표를 인쇄하고, 잘못된 값은 한 오류 경로로 보냅니다.
- **바이너리 가장자리에서 한 번 해석.** 결과는 타입 값(`OpenCfg { ced, prefill, group, card_experts, launch_thread … }`, `HostCfg { populate, lock, card_dontneed }`)이고, 그 값을 쓰는 생성자가 받습니다. 로드 줄은 엔진이 쓴 값을 찍습니다.
- **트리 안 선례.** `attn_heads_split`는 `halves`를 인자로 받고, 시험은 두 팔을 한 프로세스에서 돕니다(tests/attn.rs:556). 환경은 맨 위 디스패치(attn.rs:2481)에서만 읽습니다.
- **참조 엔진.**
  - ik는 `gpt_params` → `llama_model_params`/`llama_context_params` 구조체로 넘기고, 환경은 인자와 한 표로 짝짓습니다(common/common.cpp:791–800 `LLAMA_ARG_*`).
  - mistral.rs는 타입 설정을 생성자에 넘깁니다(`NormalSpecificConfig` pipeline/normal.rs:397, `PagedAttentionConfig` paged_attention/mod.rs:197). mistralrs-core 전체의 `std::env::var("…")`는 14곳입니다[실측].
- **전역으로 남길 것.** 진짜 프로세스 전역인 `THREADS`/`SPIN`/`POISON`/`PROFILE`만 전역에 둡니다.

**보상.**
- 조용한 수용 13개가 닫힙니다. 예: `STEP_STATS=true`면 계측이 조용히 꺼지고 러너는 `stat` 줄 없이 끝납니다.
- 재실행 자식 7개와 덮어쓰기 훅 2개가 사라집니다.
- **게이트 흐름(자원 시간선).** V4.1 게이트 레시피 하나의 벽시계는 다음 구간의 **합**이고, 차선은 레시피를 직렬로 돕니다.
  1. rsync(box.sh:20, `-c`로 트리 전체; 미측정)
  2. 빌드 확인(박스 CPU; 미측정)
  3. 락 대기(차선 안 0, 최대 1,800 s 뒤 rc 75, gpu-gate.sh:44–52)
  4. 적재 30.0 s(NVMe·페이지 캐시 → DRAM, PCIe H2D; SM은 논다)
  5. 케이스(SM + 호스트 티어)
- **환경 팔 하나가 곧 프로세스 하나, 적재 하나입니다.**
  - prefill 게이트는 386 s → 두 팔이면 745 s입니다.
  - `gate-gpu-e2e`는 바이너리를 두 번 띄웁니다(justfile:115–116).
  - `-q3ksplit`는 별도 레시피입니다(justfile:936).
- **적재 공유의 선행 조건.** gates-plan §3.5의 적재 공유(−12분 이상[유도, 거기])는 팔이 환경인 한 팔을 흡수하지 못합니다. `CARD_EXPERTS=expert/slot`, `PREFILL_GROUP=1`, `Q3K_SPLIT=2`, `FLASH_MMA=0`이 생성자 인자가 되어야 한 적재 안의 케이스가 됩니다. 즉 이 발견이 벌크 흐름(모델·배치당 적재 1회)의 선행 조건입니다.
- AGENTS의 132행 산문이 생성물로 바뀝니다.

**비용·증명.** 기본값을 바꾸지 않고 읽는 시점도 open에 둡니다("step does no load-time work"). 따라서 **move 등급**입니다.
- `just ptx-scan` 동일(레버는 호스트 코드), 구조 줄 동일(노드 수, eager = replay, e2e 집합), 소유 게이트 출력은 `tools/verdict-diff.py`로 동일.
- 엄격해지는 것은 잘못된 값뿐이므로 레버마다 FAIL-first 거부 시험을 붙입니다.
- 시간 A/B는 필요 없습니다. 예외는 CPU 디스패치(ops.rs, deepseek2/attn.rs)입니다. AGENTS:356–365가 같은 임대 A/B를 요구하므로, 접근자 본문만 바꾸거나 뒤로 미룹니다.

**위험·주인·순서.**
- aa: gpu-deepseek41, gpu-gates 바이너리, `gpu/src/{flash,lib,model/launcher}.rs`, placement, deepseek2/attn.rs, threads.
- `[03]`: hybrid.rs(HYBRID_*·HOST_*·CARD_DONTNEED), qwen3 ubatch, flash_gqa, ops.rs(POISON·DEFER_QUANT·STEAL*), moe.rs(EXPERT_LOG). r8host 착륙 뒤에 합니다.
- 순서: 등록부와 open 레버 → 게이트 안 팔(gates-plan §3.5와 한 라운드) → `[03]` 조각 → AGENTS 표 교체.

#### TL-1 표 — 엔진·바이너리가 읽는 `BLOOMERY_*`

분류: C 운영 설정, A 같은 바이너리 A/B 팔, T 게이트가 대조하는 쌍둥이(없애면 오라클이 사라짐), D 디버그·계측, M 모드. "조용"은 뜻 모를 값이 오류 없이 기본값이 된다는 뜻입니다. "문서"는 AGENTS 레버 산문에 있는지입니다.

| 레버 | 읽는 곳 | 파서 → 기본 | 조용 | 문서 | 분류 | 판정·출처 | 오늘 |
|---|---|---|---|---|---|---|---|
| THREADS | threads/src/lib.rs:334 | 파싱 실패·0 → 물리 코어 | 예 | 예 | C | — | 전역, 오류 |
| SPIN | threads/src/lib.rs:339 | 실패 → DEFAULT_SPIN | 예 | 예 | C | — | 전역, 오류 |
| POISON | model/src/ops.rs:150 | `!= "0"` → 켬 | 예 | 예(443) | D | — | 전역 0/1 |
| DEFER_QUANT | ops.rs:2284 (+override :2289) | `!= "0"` → 켬 | 예 | 아니오 | T | `f3270b9` +1.4 % 4/4; tests/union.rs:419·679·718이 양 팔 대조 | [03] 결정 |
| STEAL | ops.rs:2731 | 0/1, panic | 아니오 | 예 | A | rig-log 09-21:14(게이트 실행, 판정 아님) | 인자 [03] |
| STEAL_BLOCKS | ops.rs:2751 | 양수, 4, panic | 아니오 | 예 | A 정착 | AGENTS:541 "2/8/16 no better"(`bdf7914`) | 상수로 [03] |
| EXPERT_LOG | model/src/moe.rs:590 | 경로 | — | 아니오 | D | expert-union-dump.sh:18 | 등록부 D [03] |
| PROFILE | model/src/profile.rs:84, bloomery-decode.rs:56 `set_var` | 실패 → 0, >2 → 2 | 예 | 예 | D | — | 0/1/2, 오류 |
| FLASH_SIMD | model/src/arch/deepseek2/attn.rs:1088 | `== "0"` → 스칼라 | 예 | 예 | T | rig-log 09-20:59(15×) | 인자 |
| KV_PREFETCH | attn.rs:1108 | `== "0"` → 끔 | 예 | 예 | A | `6391233` | 인자 |
| KV_PREFETCH_ROWS | attn.rs:1111 | 16, panic | 아니오 | 예 | A 정착 | 09-23#cpu-kv-prefetch-distance | 상수 + 인자 |
| FLASH_SEGMENTS | attn.rs:1200 | 1..32, panic | 아니오 | 예 | T(1 = 밴드 기준) | 09-23#cpu-splitk-attention | 인자 |
| ATTN_BUNDLE | attn.rs:1224 | 1/8, panic | 아니오 | 예 | T | rig-log 09-24:8 | 인자 |
| ATTN_HALVES | attn.rs:2483 | 1/2, panic | 아니오 | 예 | A 정착(느림) | `fa8f54b` | **삭제** |
| FLASH_MMA | gpu/src/flash.rs:255 | 0/1, panic | 아니오 | 예 | T | rig-log 09-22-n; justfile `=0` 3줄 | 인자 |
| FLASH_SEG | flash.rs:279 | KEY_TILE 배수, panic | 아니오 | 예 | A(스윕) | — | 인자 |
| GQA_MMA | gpu/src/flash_gqa.rs:112 | 0/1, panic | 아니오 | 예 | T | `43944e4` | 인자 [03] |
| Q3K_SPLIT(_ITERS) | gpu/src/lib.rs:247·254 | 1/2/4/8 → 1; ≥1 → 16 | 아니오 | 예 | A **미결** | `3bf2e2e`, 예측 −60…−150 µs 미측정 | 인자 |
| HYBRID_NL | gpu/src/hybrid.rs:118 | usize, 오류 | 아니오 | 예 | C(V2-Lite) | rig-log 09-23:341–355 | 설정 [03] |
| HYBRID_OVERLAP | hybrid.rs:127 | 0/1, 오류 | 아니오 | 예 | T(gate-gpu-hybrid) | 09-23:353–354 켬 6.921 대 끔 8.288 ms | 인자 [03] |
| HOST_POPULATE/LOCK, CARD_DONTNEED | hybrid.rs:198–200 | 0/1, 오류 | 아니오 | 예 | C | 09-25:123 | `HostCfg` [03] |
| LAUNCH_THREAD | gpu/src/model/launcher.rs:44 | 0/1, 오류 | 아니오 | 예 | A **미결** | 09-25:12 +0.96 % ± 2.1 % (n=3) | 인자 |
| QWEN3_UBATCH | gpu/src/arch/qwen3moe/ubatch.rs:64 | 1..4096, 오류 | 아니오 | 예 | C + T | `ded6c51` | 설정 [03] |
| ENGRAM_HELPER | gpu-deepseek41/src/chain/glue.rs:1104 | `== "0"` → 직접 | 예 | 예 | A | `382578a`; 09-25:82 | 인자 |
| STEP_STATS | glue.rs:1105 외 5곳 | `== "1"` | 예 | 예 | D | — | 한 번 해석 |
| CED | gpu-deepseek41/src/body/ced.rs:212 | on/off, 오류 | 아니오 | 예 | A 정착 + 보정 팔 | 09-25#v41-prefill-resit +5.0 %/+49 % | 설정 |
| PREFILL | body/prefill.rs:117 | batch/steps, 오류 | 아니오 | 아니오 | M | `43cd107` 2.97× | 모드 |
| PREFILL_GROUP | body/prefill.rs:346 | 1..8 → 2, 오류 | 아니오 | 예 | A 정착 + 조율 축 | 09-26#prefillgroup-ab | 설정 |
| CARD_EXPERTS | chain/ffn/batch.rs:1438 (+재독 2곳) | tile/expert/slot, 오류 | 아니오 | 예 | A 정착 | 09-26#cardtile-ab | 인자; `expert` 존폐 |
| HOT_LIST | model/src/placement/hot_list.rs:18 | 경로, 오류 | 아니오 | 예 | C | `e32664a` | 설정 |
| CARD_BUDGET | placement/card_budget.rs:15 | 바이트, 오류 | 아니오 | 예 | C | — | 설정 |
| PIN_MAIN | generate_ds41.rs:410, bloomery_serve_ds41.rs:132, bloomery_chat.rs:226, bloomery-decode.rs:122, bench_v41_host.rs:3415 | "0"만 → 떠 있음 | 예 | 예 | A | 09-25:12 −1.2 % ± 3.4 % | 설정 |
| DRAFT | generate_ds41.rs:741 (+셸 2) | off/lookup/dspark, 오류 | 아니오 | 예 | M | `c00274c` | CLI 플래그 |
| STEP_PAIR | generate_ds41.rs:604·751·928 | `== "1"` | 예 | 예 | A 답함(E12) | 09-24:99, 09-25:183 | **삭제** |
| CHECK_FINITE | generate_ds41.rs:585 | 0/1, 오류 | 아니오 | 예 | D | — | CLI 플래그 |
| POPULATE | model/src/bin/bloomery-decode.rs:129 | `!= "0"` | 예 | 아니오 | A 정착 | `a7e7c17` +3.5 %/+18 % | **삭제** |
| WEIGHTS | bloomery-decode.rs:132 | anon/huge, 그 밖 → 기본 | 예 | 아니오 | A 정착(무효) | 09-21:46 | **삭제** |

경로·시험·러너 쪽은 가족 단위로 적습니다.

| 가족 | 수 | 소유 | 비고 |
|---|---:|---|---|
| 경로(`MODEL`, `REF_MODEL`, `REF_MODEL_PROFILE`, `V41_MODEL`, `V41_DIR`, `V4_MODEL`, `Q5K_MODEL`, `DSPARK_MODEL`, `DATA` …) | 약 15 | box.sh:41–74, ref-paths.sh, models/*.sh, Rust 시험 | `BLOOMERY_MODEL`이 도구에서는 프로필 이름(ref-paths.sh:35), 시험에서는 파일 경로(model/tests/common/model_path.rs:6, prompts.rs:21, union.rs:1011, qdot.rs:20)입니다. box.sh:46–47이 그래서 내보내지 않습니다. → TL-6 |
| `*_CHILD_DUMP`·`*_BAND_DUMP` | 7 | model/tests | TL-1의 결과물 |
| `CHAIN_ATTN_LAYERS/MUTATE` | 2 | gate_deepseek41_chain_attn.rs:165·181 | FAIL-first 도구, 남김 |
| 핸드셰이크 `HOST_LEASE`, `ENGRAM_LEASE`, `LEASE_HELD` | 3 | bench_v41_host.rs:3499, engram-rate.rs:760, lease.sh | |
| `NCU_*` | 22 | ncu-gpu.sh(677행, 폼 넷: generate·ds41pp·gemm·q3pp) | 스위스 군용 도구 |
| `NSYS_*` / `REF_*` / `AB_*` / `LEASE_*` / `GATE_*` | 10 / 14 / 6 / 4 / 4 | nsys·dump·depth·lease·gate 러너 | |
| `GEN_*`, `DECODE_*`, `ARM_BOUND`, `DRY`, `OTHER_STRICT`, `CPU_BUSY_*` | 약 12 | 러너마다 골라 붙임 | TL-7 |

고유 이름은 전체 167개입니다[실측, `grep -rhoE 'BLOOMERY_[A-Z0-9_]+' crates tools justfile | sort -u`].

### TL-2 AGENTS.md와 CLAUDE.md가 규칙이 아니라 상태와 역사를 싣는다

**어디·지금 모양.** AGENTS.md 669행 중 약 232행이 상태·역사·참조입니다[유도, 행 범위].
- 프리필 착륙 연대기: 376–426(51행)
- 「Known state, 2026-09-20」: 489–537(49행)
- 레버 산문: 538–669(132행)

틀린 사실도 싣습니다.
- 「Layout」(278–305)에 `gpu`, `gpu-deepseek41`, `gpu-gates`, `gpu-spike`, `engram`, `vision`, `gpu-vision`이 없습니다. `.rs` 156,971행[실측], 즉 제품 엔진 전부입니다.
- 「Never」 셋째 규칙(16–22)은 측정이 `measure.sh`·`cpu-measure.sh`에 속한다고 합니다. 그런데 measure.sh는 stage-0 러너입니다. 임대 없이 증인 필드만 쓰고(15–17), 바운드가 없고, 증인 사이에서 `cargo oxide run`을 돕니다(50–51). rig-log 전체에 두 이름이 없습니다[grep 빈 출력].
- "All 13 subsystem gates"라고 하지만 게이트 레시피는 95개입니다.
- MUL-10·MUL-11 참조(AGENTS:519·539, Cargo.toml 린트 주석)가 있습니다.
- CLAUDE.md에 취소선이 남아 있습니다(`~~구현 라운드는 outsource…~~`).

**어떻게 쌓였나.** 착륙마다 헤드라인 문장이 붙었습니다(`dfa8b5d` … `eb3e08a`). 틀린 것은 취소선으로 정정해 10쌍이 됐습니다. 레버는 들어올 때마다 산문을 한 덩이씩 받았습니다.

**오늘 만든다면.**
- AGENTS.md에는 규칙만 둡니다. 사례는 rig-log 앵커 링크 하나로 대신합니다.
- 헤드라인 숫자는 README 표 하나(행마다 rig-log 앵커)에 둡니다.
- 린트 기준선은 `docs/rust-quality.md` §0 하나에 둡니다(AGENTS:510이 이미 그렇게 말합니다).
- 레버 표는 TL-1이 생성합니다.
- 「Never hand-run」은 지금 러너의 이름을 댑니다.
- 근거: 사용자 규칙 "CLAUDE.md에는 히스토리를 쌓지 않는다 — 현재 유효한 규칙만"(~/.claude/CLAUDE.md §10). 참조로 mistral.rs AGENTS.md는 164행, 취소선 0개, 첫 절이 크레이트 목록입니다[실측].

**보상.** 모든 라운드가 싣는 53.5 KB의 약 3분의 1이 빠집니다[유도]. 라운드를 오도하는 셋(빠진 크레이트, stage-0 러너 규칙, "13 gates")이 사라집니다.

**비용·증명.** 문서만이라 게이트가 필요 없습니다. 레버 절만 TL-1 뒤에 합니다.

**주인.** aa이고 언제든 할 수 있습니다.

### TL-3 출력 줄 스키마와 엔진의 구조 사실에 도구 쪽 주인이 따로 있다

**어디·지금 모양.** 1절의 7–9항 그대로입니다.
- 러너는 `sed`/`awk`로 필드를 뽑습니다(depth-ds41.sh:406–410, 475–493).
- ds41pp.py:52–60은 커널 이름 집합(`PROLOGUE`, `ENGRAM_ROWS`, `JOIN`, `PLACES`)으로 트레이스를 자릅니다.
- 흐름 모델에는 분할·계획의 사본이 있습니다(ds41_prefill.py:179–366, `batches` … `batch_acts`, 약 188행).

**어떻게 쌓였나.** 필요한 필드가 생기면 바이너리에 `key=value`를 붙이고(`e4c37cf`, `afe86d5`), 파이썬 쪽은 규칙을 옮겨 적었습니다. 흐름 모델은 시팅마다 손으로 보정했습니다(`be9a4c3`, `3693dd9`).

**오늘 만든다면.**
- **기록 모듈 하나.** Rust 한 모듈(`gpu-gates/src/record.rs`)이 모든 기록 줄의 주인이 됩니다. 한 형식(`rec=… v=… key=value` 또는 JSON 한 줄)에 `--records-schema`를 둡니다.
- **`plan` 기록.** 엔진이 자기 계획을 내보냅니다: 배치, 청크, CED 층별 시작, 층-배치별 런치 목록. 이미 순수 함수입니다(prefill.rs:162, :841, ced.rs:328). 흐름 모델과 ds41pp는 이 기록을 읽습니다.
- **NVTX 범위.** 트레이스는 NVTX 범위로 자릅니다. 장부가 이미 이 방향을 적었습니다(plan-ledger.md:1012).
- **파서·인용.** 파이썬 파서는 모듈 하나로 모읍니다. 코드 인용은 줄 번호가 아니라 심볼로 하고, 백테스트는 보관된 기록 파일을 읽습니다.
- 참조: llama-bench의 printer 추상(CSV/JSON/MD/SQL, examples/llama-bench/llama-bench.cpp:186–208, 2140)과 `scripts/compare-llama-bench.py`.

**보상.**
- 쓰는 곳 7과 읽는 곳 17이 모듈 둘로 모입니다.
- 흐름 모델이 코드 이동에 조용히 썩는 경로가 닫힙니다(오늘 넷이 틀림).
- 약 188행과 백테스트 전사 약 104행이 기록 읽기로 바뀝니다.

**비용·증명.** 기록 줄은 **추가**이고 호스트 코드뿐입니다.
- ptx-scan 동일, 게이트 출력 동일.
- 파서는 하나씩 옮기고, 보관된 옛 로그가 같은 표를 내는지 확인합니다(`ds41pp tables`, flow `--backtest`·`--self-test` 동일).
- 박스 시간은 스모크 하나뿐입니다.

**주인.** aa이고, generate_qwen3moe·depth-qwen3moe·q3pp는 `[03]`입니다. Rust 쪽은 지금 할 수 있고, 러너 파서 교체는 boxlease 뒤입니다. TL-7보다 앞서야 합니다.

### TL-4 게이트가 셸 문자열로 선언되고, 세 프로그램이 거기서 구조를 캐낸다

**어디·지금 모양.** 게이트는 justfile의 한 줄 셸 문자열입니다(예: 928–929). 이것을 되읽는 코드가 셋입니다.
- recipes.py: 셸 렉서, 래퍼 제거, cargo CLI 파서(154–410, 257행).
- gate-batch.sh: 파이썬 셋을 heredoc으로 품습니다(`PYPLAN` 192–640 사이 약 445행, `PYBAL`, `PYAPPEND`). 그 안의 두 번째 워커가 `gpu-gate.sh` 호출 형태와 임대 취득을 정규식으로 찾아 차선을 정합니다(255–335).
- check-recipes.sh: 셸 텍스트 린트(8–37)와 `just --dump`를 다시 읽는 내장 파이썬(53–116).

팔 커버리지는 산문에만 있습니다. AGENTS는 prefill 게이트를 기본·`expert`·`slot`으로 돈다고 하지만 레시피는 기본 하나입니다. 배치 러너 기록(`gate-times.tsv` 684행, `gate-ledger.tsv` 327행)에서 `CARD_EXPERTS=slot`은 3회, `PREFILL_GROUP=1`은 2회, `CARD_EXPERTS=expert`는 **0회**입니다. 라운드가 레시피를 직접 돌리면 이 기록에 남지 않으니 "한 번도 안 돌았다"는 뜻은 아닙니다.

**어떻게 쌓였나.** 사고가 날 때마다 "텍스트를 읽어 판단"하는 층이 하나씩 붙었습니다: `|| echo`(`ef9e579`) → 린트, GPU 락(`9a339d3`) → 락 린트, 카드별 락(`2fb3893`) → `any` 형태, 두 차선(`e4ae993`) → 텍스트로 차선 읽기, 원장 → recipes.py `key`.

**오늘 만든다면.**
- **`gates.toml` 매니페스트가 주인입니다.** 필드는 name, package, target, features, runner, card(3090/any/both), solo, profile, args, arms입니다.
- justfile에는 `gate NAME [ARM]` 하나와 정적 검사만 남깁니다.
- recipes.py는 매니페스트를 읽고, 모듈 트리 워크와 원장 키는 그대로 둡니다. gate-batch는 차선·카드를 매니페스트에서 읽습니다.
- 러너가 늘 gate.sh나 gpu-gate.sh이므로 셸 린트가 막던 부류는 구성상 사라집니다.
- 타이밍 레시피 50개는 매니페스트 밖(TL-7)에 둡니다.
- 참조: ik는 `llama_test`/`llama_target_and_test` + LABEL로 선언하고 CTest가 소유합니다(tests/CMakeLists.txt:17–83).

**보상.**
- 셸 해석 코드가 최대 약 800행 줄어듭니다[유도, 상한: 257 + 약 445 + 약 94 — PYPLAN 중 매니페스트 뒤에도 남는 목록 검증·균형 부분을 빼지 않은 값].
- 두 워커의 규칙 차이가 닫힙니다.
- 팔 커버리지가 데이터가 됩니다.

**비용·증명.** 도구만 바뀝니다.
- 같은 `--list`에서 `gate-batch --dry-run` 계획이 전후 동일해야 합니다.
- 고정 구간(최근 착륙 20개)의 `recipes.py affected A..B` 출력이 동일해야 합니다.
- 박스 시간은 스모크 하나뿐입니다.

**주인·순서.** aa입니다. boxlease(gate-batch.sh, gpu-gate.sh) **착륙 뒤**에 합니다. 이행기에는 기존 레시피에서 매니페스트를 한 번 생성합니다.

### TL-5 열린 일과 역사의 부기가 주인 둘, 사본 여럿으로 퍼졌다

**어디·지금 모양.**
- **triage.** triage:3은 "아직 할 일만 … 착륙한 줄은 지운다"라고 선언합니다. 실제로는 취소선 141쌍, "착륙" 154회, 255,118바이트이고, 09-25 커밋 147개 중 89개가 이 파일을 고쳤습니다. 병렬 트랙의 충돌 지점입니다.
- **MUL.** 1절 10항 그대로입니다.
- **plan.md.** plan.md:5는 여전히 「지금 (2026-09-25 새벽)」입니다.
- **ledger.** ledger는 654,074바이트이고, triage:3이 근거가 필요하면 라운드를 거기로 보냅니다.
- **docs/research.** 81문서, 3,004 KB입니다. `errsrc/`에는 원시 로그 528 KB와 md5까지 같은 스크립트 두 벌이 있습니다.
- **숫자 사본.** 헤드라인 숫자가 다섯 곳에 있습니다(예: `5,304`가 AGENTS·triage·plan·rig-log에).
- **코드 주석.** 코드 32파일이 `gpu-design.md work package …`, `decision N`을 인용합니다.

**어떻게 쌓였나.** `5c66b38` plan 재편, 09-25 triage 재작성 뒤로 착륙 줄을 지우지 않고 취소선과 요약을 쌓았습니다.

**오늘 만든다면.**
- 열린 일은 트래커 행(MUL) 또는 한 항목 한 행의 표 파일에 두고, 착륙 커밋이 그 행을 닫습니다.
- plan.md는 현재 상태만 담고 교체합니다(덧붙이지 않음).
- 측정은 rig-log, 이력은 커밋 메시지에 둡니다.
- 보고서와 원시 데이터는 `docs/` 밖에 보관합니다. 설계 문서 색인은 하나만 둡니다.
- 참조 엔진의 레포 docs에는 사용자 문서만 있습니다(ik `docs/` 11개, mistral.rs는 사이트)[ls].

**보상.** 충돌 지점 하나가 사라지고, 라운드가 138k 토큰짜리 파일을 읽지 않고, 숫자 사본이 하나로 줍니다.

**비용.** 문서·트래커만입니다. **사용자 결정이 필요합니다.** CLAUDE.md가 MUL을 지정하므로, 선언된 주인(MUL)과 실제 주인(triage 파일) 중 어느 쪽으로 합칠지 사용자에게 물어야 합니다.

**주인.** aa이고, 한국어 산문이라 리드가 직접 씁니다.

### TL-6 경로·프로필이 셸 프로필, box.sh 문자열, Rust 기본값에 나뉘었다

**어디·지금 모양.**
- `BLOOMERY_MODEL`이 두 뜻입니다.
- `/root/bloomery-data` 기본값이 주석 아닌 줄 30곳에 있습니다. qdot/tests/qdot.rs 한 파일에만 9곳입니다. gpu-gates에는 이미 한 소유자 `data_dir`(lib.rs:106)가 있습니다.
- V4.1 기본 파일의 주인이 둘입니다(gguf/src/v41.rs:31, `models/deepseek41.sh`). `d771085`가 스스로 "the two owners"라고 적었습니다.
- 기본 프로필이 두 곳에 있습니다.
- box.sh:57–74는 프로필을 원격 셸 안에서 세 번 source하는 문자열을 만듭니다.

**어떻게 쌓였나.** V2-Lite 하나에서 시작해 V4.1(`78fbe2a`), Qwen3(`1fb022a`), V4(09-25)가 차례로 붙었습니다. 모델마다 셸 프로필 하나, 경로마다 환경 변수 하나가 늘었습니다.

**오늘 만든다면.**
- `profiles.toml` 하나를 둡니다. Rust는 경로 모듈 하나로 읽되 기본값 없이, 없으면 이름 붙은 오류를 냅니다(box.sh가 늘 내보내므로 기본값은 가림막일 뿐입니다). 파이썬은 `tomllib`, 셸은 한 줄 getter로 읽습니다.
- `BLOOMERY_MODEL`은 프로필 이름 한 뜻으로 쓰고, 시험의 파일 경로에는 다른 이름을 줍니다.
- 기본 프로필은 제품 모델로 하거나 기본 없이 늘 명시하게 합니다.
- 참조: mistral.rs `toml-selectors/`(plain, gguf, speculative-gguf …)와 `from-config`.

**보상.** 기본값 사본 30개와 이중 주인이 하나로 모이고, box.sh의 인용 문자열이 줄어듭니다.

**비용·증명.** move 등급입니다(게이트 출력·ptx-scan 동일). box.sh와 ref-paths.sh를 건드리므로 **boxlease 뒤**에 합니다. TL-4의 `profile` 필드와 같은 라운드가 자연스럽습니다. Qwen3 프로필은 `[03]`이 확인합니다.

### TL-7 측정 러너 23개가 프로토콜을 손으로 조립한다

**어디·지금 모양.** 러너 23개가 프로토콜 조각을 골라 붙입니다[grep].
- `guard_cpu` 1개(depth-ds41만), `guard_other` 8개, `assert_fresh_binary` 10개, `DRY` 6개, `lease_bounded` 14개. 나머지 러너는 직접 `timeout`을 겁니다(depth-ds41.sh:350·352·462·464).
- `measure.sh`는 임대가 없습니다.
- 팔 문법이 다섯 가지입니다(4절).
- 비율 통계는 awk 두 벌(이미 다름)과 파이썬 둘로 구현됐습니다. t 표는 `9ed73ff`에서 tdist.py 하나로 모였습니다.
- depth-ds41.sh:92–98의 "V4.1's ours is one decode step per token today"는 `43cd107`(09-25) 뒤로 거짓입니다.

**셸이 치른 값.**

| 사고 | 출처 | 부류 |
|---|---|---|
| `\|\| echo`가 종료 코드를 삼킴 | `ef9e579`, gate.sh:8–13 | 언어·구조(매니페스트·타입 러너가 없앰) |
| 레시피가 락을 직접 잡음 | `9a339d3`, check-recipes.sh:19–26 | 구조 |
| 맥 bash 3.2에서 `set -u` | `c0ae6b8` | 언어 |
| `declare -A` 팔 붕괴, `\| tail` rc | 스코프 파일이 이름을 댐. 커밋 로그에서는 못 찾았고 메모리 기록(09-23, 09-25)에만 있음 | 언어 |
| mtime 전달 → 낡은 바이너리 | box.sh:13–15 | 환경(시계 차이) |
| rsync `--delete`가 `*.ptx`·`*.ll`을 지우는 경합 | `ea5365e`, `c2c5ab8`, ledger #18 | 환경·업스트림 |
| 한 방향 동기화로 Cargo.lock 유실 | `737eeba` | 환경 |
| 임대 대기자가 익명 | `66c121a` | 프로토콜 균일성 |

**어떻게 쌓였나.** 옆 러너를 복사해 새 러너를 만들었습니다(depth-qwen3moe.sh는 `1785648`). 프로토콜 개정은 모든 러너에 소급해야 했습니다(1절 4항).

**오늘 만든다면.**
- 러너는 파이썬 프로그램 하나로 둡니다. 임대, 카드, 증인, 가드, 바운드, 신선도, dry-run, 회전, 통계를 **필수 단계**로 갖습니다.
- 엔진마다 어댑터를 둡니다(ours-ds41, ours-q3, ik, lcpp, mrs, bin:). 어댑터는 명령 생성과 TL-3 기록 파서로 이뤄집니다.
- 팔은 데이터로, 문법은 하나로 둡니다.
- 셸은 box.sh와 락·바운드 래퍼만 남깁니다.

**보상.**
- 개정 한 번이 한 파일로 끝납니다.
- 가드 누락 부류가 닫힙니다. 리드 빌드가 B1 시팅에 `[cpu-busy]`를 남긴 부류를 잡은 가드가 지금 Qwen3 러너에는 없습니다.
- 시팅 벽시계는 줄지 않습니다. 이 발견은 속도가 아니라 무효 행 부류를 닫는 것입니다.

**비용·증명.** 도구 L입니다. 러너가 시간 조건을 바꿀 수 있으므로 **박스 시간이 듭니다.**
- 팔마다 dry-run 명령줄이 전후 동일해야 합니다.
- 보정 카드(kind: calibration)로 기록된 행 하나를 그 구간 안에서 재현하는 시팅 한 번이 필요합니다(30분 이하).
- 그래서 지금은 보상/비용이 가장 낮습니다.

**주인·순서.** aa이고, Qwen3 어댑터는 `[03]`입니다. boxlease와 TL-3 뒤에 합니다.

## 3. 삭제

| 대상 | 근거 | 커버리지 변화 |
|---|---|---|
| `ATTN_HALVES=2` 경로(attn.rs:2468–2495 + 분할 본문) | `fa8f54b` "bit-identical, gated, and slower"; AGENTS:229–246 한 경로 원칙 | tests/attn.rs:556의 halves 2 반복이 빠짐(날짜·사유 필요). V2-Lite 존폐 판정에 종속(audit-cpu) |
| `WEIGHTS=anon/huge`(bloomery-decode.rs:130–136) | rig-log 09-21:46: 큰 페이지 무효, anon은 잡음 속으로(4/6), "레버만 남겼다" | 없음. `gguf::Weights::Resident`는 유지(oracle.rs:313, r8file.rs:258). V2-Lite 종속 |
| `POPULATE=0`(bloomery-decode.rs:128–129) | `a7e7c17` +3.5 %/+18 % | 없음. V2-Lite 종속 |
| `STEP_PAIR=1` 팔(generate_ds41.rs:604–605, 751–756, 928과 그 가지) | rig-log 09-24:99, 09-25:183. `grep -rn STEP_PAIR tools justfile docs/cards docs/plan.md docs/gates-plan.md` 빈 출력 | 없음. `step_pair` 자체는 DRAFT가 씀 |
| `STEAL_BLOCKS` 레버 본문(ops.rs:2741–2765, panic 시험 4390–4400). 상수 4는 유지 | AGENTS:541 / `bdf7914` | panic 시험 둘. `[03]`, r8host 뒤. V4.1 union도 이 경로를 씀 |
| `docs/research/errsrc/{errsrc-judge.py,errsrc-sim-all.sh}` 루트 사본 | `tools/` 사본과 md5 동일; gate_e2e.rs:398은 `tools/` 쪽을 인용 | 없음 |
| AGENTS 「Layout」·「Known state」 → 재작성·이동. depth-ds41.sh:92–98 → 정정 | TL-2, TL-7 | 없음 |
| stage-0 러너(`measure.sh`, `cpu-measure.sh`, `build.sh`, `build-cpu.sh`, `q3k_ref.cpp`, `q3k_cpu_ref.cpp`, `measure-gpu/-cpu`, `decode-measure.sh` 계열) | 조건부. stage-0·CPU 엔진 존폐는 audit-gpucore/audit-cpu 몫 | 크레이트와 함께 |
| `tools/ref/window-union.py`(340행, `051d51e` 하나) | justfile·tools·crates·docs·rig-log 어디에도 이름이 없음 — 확신 낮음, 리드 확인 필요 | 없음 |

삭제로 분류하지 않은 것:
- `DEFER_QUANT=0`: 오라클 쌍둥이라 `[03]`이 결정합니다.
- `CARD_EXPERTS=expert`: 배치 기록 0회입니다. 매니페스트에 올리든 지우든 결정해야 합니다.
- `LAUNCH_THREAD`: 미결입니다. 다음 시팅에서도 잣대 안에 있으면 삭제 후보입니다.

## 4. 교차 패턴

1. **줄 번호 인용은 하루 만에 썩습니다.** 흐름 모델 54곳(넷은 이미 틀림), triage·plan, 코드 주석의 문서 절 인용이 그렇습니다. check-comments.sh는 이슈 번호와 날짜만 잡습니다.
2. **환경 변수가 만능 운반 수단입니다.** 설정, 팔, 경로, 노브, 핸드셰이크 167개가 이름 공간 하나를 씁니다. 뜻 충돌과 조용한 수용이 여기서 나옵니다.
3. **선언 대신 텍스트에서 구조를 캐냅니다.** 셸 렉서, 워커 둘, `src_const`, 커널 이름 집합, `sed` 필드 추출이 그렇습니다.
4. **프로토콜을 파일마다 옵트인으로 조립합니다.** 러너 23개가 그렇고, 종료 코드(75·124·137·64·70·3·2)도 스크립트마다 다시 적습니다(gate.sh:25, gpu-gate.sh:11–12, card.py:80, box.sh:33–39).
5. **팔 문법이 다섯 가지입니다.**
   - ab-decode `BLOOMERY_AB_ENVS="K=V;K=V"`
   - depth-gpu `BLOOMERY_GPU_ARMS="D:C …"`
   - depth-ds41/q3 `<D>@K=V,…`
   - gpu-ab `--arm NAME=TREE[:K=V,…]`
   - gate-batch `NAME[@K=V,…][:ARGS]`
6. **스위스 군용 도구가 있고, 한 도구가 두 언어로 쓰였습니다.** ncu-gpu.sh(677행, 노브 22, 폼 4), gate-batch.sh(bash에 파이썬 셋), recipes.py(하위 명령 여섯 + 박스 매니페스트 + 해시 캐시), check-recipes/check-rustflags의 heredoc 파이썬이 그렇습니다.
7. **계약과 주석이 역사를 싣습니다.** 취소선 쌍이 AGENTS 10, plan 42, triage 141, ledger 333개이고, justfile 주석에 날짜 박힌 사고담이 있습니다(2–7행 등).

## 5. 역사처럼 보이지만 남는 것

- `.cargo/config.toml` + `cuda-oxide.toml` + `check-rustflags.sh`: cargo-oxide의 `CARGO_ENCODED_RUSTFLAGS`가 주인 둘을 강제합니다(nvlabs-ledger §6). 업스트림 또는 포크 패치가 합칠 때까지 유지합니다.
- box.sh의 `/*.ptx`·`/*.ll`·`/.oxide-artifacts/` 제외(ledger #18), mtime을 싣지 않는 `-c` 동기화, `lock-back.sh`: 실제 방어입니다.
- 예측 카드(`docs/cards/`, `card.py`, `lease_take`): 살아 있는 derive-first 강제 장치입니다.
- 쌍둥이 팔(`FLASH_SIMD=0`, `ATTN_BUNDLE=1`, `FLASH_SEGMENTS=1`, `FLASH_MMA=0`, `GQA_MMA=0`, `HYBRID_OVERLAP=0`, `DEFER_QUANT=0`): 게이트의 오라클입니다. 모양만 인자로 바꿉니다.
- `CED=off`, `PREFILL_GROUP=1`: 흐름 모델 보정 팔이고, G는 조율 축입니다(G 8 예측 +38.2 %, `3693dd9`).
- `PREFILL=steps`: `CHECK_FINITE`가 강제하는 모드이고(generate_ds41.rs:541), nsys 디코드 폼도 씁니다.
- `Q3K_SPLIT`과 `gate-gpu-ds41-prefill-q3ksplit`: 미결이고, 분할 커널을 도는 유일한 게이트입니다.
- `models/deepseek4.sh`와 `gate-deepseek4-meta`: 계획된 다음 대상입니다.
- `deny.toml`: rev 고정 장치입니다.
- plan-ledger.md: 보관소로는 맞습니다. 라운드가 읽으러 가지 않게만 하면 됩니다.

## 6. 오늘 다시 짓는다면

**배치.**
- `crates/levers`: 등록부, 타입 설정(`OpenCfg`, `HostCfg`, `PrefillCfg`, `AttnCfg`), 해석기 하나, `--levers`
- `gpu-gates/src/record.rs`: 기록 구조체 전부, 순수 함수가 내는 `plan` 기록, NVTX
- `gates.toml`, `profiles.toml`
- `tools/bloomery/`(파이썬): `records`, `manifest`, `affected`(recipes.py의 살아남는 절반), `batch`, `sit`(러너), `flow`
- 셸: `box.sh`와 `gate/gpu-gate/host-gate.sh`만
- 문서: AGENTS(규칙), README(헤드라인 표), `docs/`(facts, BUILD, HARDWARE, rust-quality, 살아 있는 설계), 트래커, rig-log, 보관소

**흐름.** 바이너리 가장자리에서 한 번 해석 → 설정 값으로 적재 1회 → 케이스·팔 여럿 → 기록 줄 → 도구는 기록만 읽음.

**이행 (각 단계가 게이트를 녹색으로 유지).**
1. AGENTS·CLAUDE를 정리합니다. 레버 절은 빼고 문서만 고칩니다.
2. `record.rs`와 `plan`을 추가합니다. 호스트만 바뀌므로 ptx-scan 동일, 게이트 출력 동일입니다. 파서를 하나씩 옮기며 보관 로그로 같은 표를 확인하고, flow는 `--backtest` 동일로 확인합니다.
3. `crates/levers`와 aa 쪽 open 레버를 옮깁니다. move 등급(ptx-scan 동일, verdict-diff 동일, 레버별 FAIL-first)입니다. 게이트는 프로세스 안 팔로 바꾸고 재실행 자식을 삭제합니다(gates-plan §3.5와 한 라운드).
4. r8host 뒤에 `[03]` 조각을 옮깁니다.
5. boxlease 뒤에 매니페스트 생성기 → 일반 레시피 → recipes.py·gate-batch가 매니페스트 읽기(dry-run·affected 동일)로 가고, 셸 해석을 삭제합니다. 같은 라운드에 `profiles.toml`을 넣습니다(move).
6. AGENTS 레버 표를 생성물로 바꿉니다.
7. `sit.py`로 depth 쌍둥이를 먼저 옮기고 보정 시팅을 한 번 합니다. 이어 nsys/ncu, 나머지 순서로 옮기며 bash 러너를 지웁니다.
8. 사용자 결정 뒤 열린 일을 트래커(또는 행 파일)로 옮기고, 보고서·데이터를 보관합니다.

## 7. 범위 밖 개선 지점 (보고만)

- `crates/gpu-deepseek41/src/chain/glue.rs:1102`: `RowsLevers::from_env`가 `OnceLock`이 아니어서 AGENTS:669와 어긋납니다. XS.
- `crates/gpu-gates/src/bin/generate_ds41.rs:492`, `gate_deepseek41_prefill.rs:365`: 엔진이 쓴 값 대신 환경을 다시 읽어 인쇄합니다. XS.
- Rust 시험의 `/root/bloomery-data` 기본값 약 22곳: 공용 헬퍼 하나로 모을 수 있습니다(`data_dir` 선례). S.
- 코드 주석 32파일의 `gpu-design.md work package …`: check-comments.sh가 못 잡는 역사 부류입니다. S.
- `tools/gate-batch.sh:259·304`: 워커가 `.sh/.bash`만 따라가 `python3 tools/…` 안의 호출을 못 봅니다. XS이고, TL-4가 되면 사라집니다.
- `crates/threads/src/lib.rs:330–341`, `crates/model/src/profile.rs:79–86`: 조용한 기본값·clamp가 주석에 설계로 적혀 있습니다("No silent failure" 이전 결정). XS.
- `Cargo.toml` 린트 주석 `적중 54 — … MUL-10`: AGENTS의 58과 수가 다르고, 트래커를 참조합니다. XS.
