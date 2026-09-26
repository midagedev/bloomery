# audit-cpu 보고 (ID 접두 CPU)

## 0. 읽은 범위

**기준 커밋과 실행한 것**
- 읽은 커밋은 `4786c1e`(main, 시작 시점)이고, 그 뒤에 착륙한 것은 보지 않았다.
- 박스 명령은 없다. 레포 파일은 하나도 쓰지 않았다.
- 도구 실행은 하나다. 맥에서 `python3 tools/recipes.py affected 9c8ebf0^..9c8ebf0 --no-box`를 돌렸다. 이 도구는 `git archive`를 임시 디렉터리에 풀고 `cargo metadata --offline --locked --no-deps`만 부르며, 끝나면 그 디렉터리를 지운다. 결과는 §1 F2에 있다.
- 세션 스크래치패드에 게이트 목록 파일 하나를 썼다(레포 밖).

**직접 읽은 것**
- 전부 읽음: `model/src/lib.rs`, `arch/qwen3moe/{mod,host}.rs`, `arch/deepseek2/mod.rs`, `model/tests/alloc.rs`.
- 구간만 읽음:
  - `ops.rs` [03]: 1–60, 896–1110, 1160–1202, 1300–1335, 2276–2315, 3140–3260, 그리고 전체 항목 목록.
  - `moe.rs` [03]: 600–960, 1531–1600, 그리고 항목 목록.
  - `arch/mod.rs` 1–120, `deepseek2/derived.rs` 1–80, `hparams.rs` 1–40과 grep, `attn.rs`의 `MlaParams`(351–420), `profile.rs` 1–80, `kv.rs` 1–60, `placement.rs` 1–80·733–775, `deepseek41/hparams.rs` 항목 목록, `fixture.rs` 1–60.
- 머리말만 읽음: `placement/{host_lock,workstation}`, `deepseek41/{host,place}`, `dspark`, `fileio`, bin `bloomery-decode`·`r8conv`·`v41fixture`·`markov-accept`(1–40)·`bench_v41_host`(1–140과 grep).
- 그 밖에:
  - `gguf/src/lib.rs` 1–31·170–300·항목 목록, `v41.rs` 머리.
  - `qdot/src/lib.rs`의 절 경계, `col_bytes`, `swiglu`/`swiglu_clamp`(4205–4256).
  - serve: `bloomery-serve.rs` 머리, `api.rs` 470–510, `sampling.rs` 머리, `genloop.rs` 325–390.
  - 대조용으로 범위 밖 gpu-gates의 `bind.rs` 150–175, `bloomery_chat.rs` 280–332, `gate_p1.rs` 머리, `gpu/src/{lib.rs,model.rs}` 머리, `hybrid.rs` 1100–1146·1990–2145.
- 문서와 도구:
  - 전부: `docs/plan.md`, `docs/facts.md`.
  - 부분: `docs/arch-split.md` 1–150, `docs/rust-quality.md`의 표와 규칙.
  - grep으로: `plan-triage.md`, `plan-ledger.md`.
  - 도구: `tools/recipes.py`의 affected·key 부분, `tools/ref/card.py` 머리말.
  - rig-log: 09-24 앞부분, 09-26 #uniondispatch-ab, 09-23 grep.

**하위 조사 셋에서 받은 것**(Explore, opus, 읽기 전용): ① qdot 전체 ② serve·tokenizer·sampler·threads ③ 스테이지 0 크레이트와 CPU 러너.

원본에서 다시 확인한 주장:
- `bloomery-serve`가 모의 엔진 전용이라는 것.
- `api.rs`의 거부 목록에 penalty가 없다는 것.
- `sampling.rs` 머리말, `bind.rs:152-174` 매핑, chat과 genloop이 두 루프라는 것.
- `BLOOMERY_PIN_MAIN` 파싱 다섯 곳.
- `engram/src/prefetch.rs:543-553`과 `gpu/src/model/launcher.rs:312-322`이 diff로 동일하다는 것.
- qdot 절 경계 다섯, `col_bytes`의 `_ =>` 팔, `gguf::activation_format`.
- gpu-spike만 부르는 엔진 API 셋(grep), `gate_p1`의 스테이지 0 FNV 핀, `rust-quality.md`의 58 분해.
- `qdot-rate-mt`·`pool-rate`·`governor-ab.sh`·`depth-decode.sh`에 러너가 없다는 것, `hunyuan-dense`, `.params()`, `oxide-ice-unroll`에 `rev`가 없다는 것, `DraftProps`가 생성되지 않는다는 것.

다시 확인하지 않은 주장은 본문에 "(하위 조사)"로 표시했다.

**읽지 않은 것**
- model: `placement.rs` 본문(80–730, 775–1724), `placement/{hot_list,card_budget,host_lock}` 본문, `deepseek41/{hparams 본문, plan.rs, kv, roles, names, place 본문}`, `dspark.rs` 본문, `r8file.rs` [03].
- CPU 엔진: `deepseek2/attn.rs` 본문(`MlaParams` 밖), `forward.rs`·`ffn.rs`·`head.rs` 본문.
- `ops.rs` 60–896·1335–2276·2315–3140·3260–4579, `moe.rs` 1–600·960–1530, `model/tests/*` 본문 대부분.
- gguf의 `quant.rs`·`write.rs`·`iq_tables.rs`·`gguf-inventory`.
- tokenizer·threads·sampler 본문, serve `template.rs`·`api.rs` 대부분.
- q3k-gemv·q3k-cpu 본문, oxide-ice-unroll의 variants.

## 1. 지금 아는 것

### 종이 위 분해

질문("지금 안다면 이렇게 짓지 않았을 곳")을 세 항으로 나눴다.
- ① 누가 무엇을 쓰나: 크레이트 밖 import, 호출자 grep.
- ② 안 쓰는 경로가 무엇을 끌고 다니나: 줄 수, `just affected`가 고르는 게이트 수, 레버, lint.
- ③ 재구성마다 증명 클래스: AGENTS의 변경 클래스 표.

세 항 모두 코드, grep, rig-log, 게이트 시간 파일로 닫혔다. 박스 실행이 필요한 항은 없었다.

유도로 못 닫는 항은 하나다. CPU6(디코드 호스트 호출을 UnionCall 한 열로 옮기기)의 속도인데, 오늘의 카드 종류로는 잴 수 없다(F6, §4-5).

### 자원 시간선 (제품 경로)

| 단위 | 호스트 CPU | 카드 | 벽시계 | 근거 |
|---|---|---|---|---|
| V4.1 디코드 스텝 (A6000, plan (a), 깊이 6) | 호스트 다리 28.49 ms (40층, 층당 평균 0.69) | 커널 합 17.36 ms | 41.73 ms, 그중 노출된 합류 빈틈 23.91 | rig-log 09-24#v41-step-nsys |
| 호스트 다리의 대역 | 132–136 GB/s (순수 읽기 140–145, STREAM 147.7) | — | 호스트 DRAM 바운드 | 09-24 E4, `plan-ledger.md:295` |
| V4.1 프리필 층-배치 (G 착륙 뒤) | lcg: union 68.6 / 73.8 ms로 호스트 바운드 | prose: 카드 바운드(모형 49.5 ms, duty 0.87) | max | `plan.md` 「흐름 모델」 09-26 저녁 |

V2-Lite CPU 전용 엔진(`bloomery-decode`)은 이 표의 어느 자원에도 올라 있지 않다. 그런데 AGENTS가 디스패치 경로 라운드에 요구하는 A/B(`ab-decode`, `AGENTS.md:356`)와 `measure-profile`은 바로 그 엔진을 잰다.

그 체제는 제품 경로와 모양이 다르다.
- routed expert 하나가 4,460,544 B다[유도: gate·up Q3_K 1408×2048이 각 1,239,040 B, down Q5_0 2048×1408이 1,982,464 B].
- V4.1 슬롯은 16,773,120 B(`plan.md` 타겟 프로필)라서, V2-Lite expert는 그 1/3.76이다[유도].
- 어텐션·dense·헤드까지 CPU가 한다.

그래서 디스패치당 바이트 법칙(`facts.md:108`)에 따라, 이 계기는 디스패치 고정비를 V4.1 호스트 다리보다 크게 보고, 호스트 expert 변경의 효과는 묽게 본다.

### 사실 (시드 밖)

**F1. `crates/model`을 크레이트 밖에서 쓰는 것은 세 부류뿐이다.**
- arch 메타: deepseek41·qwen3moe·dspark 전부, deepseek2는 `Hparams`·`names`·`Derived`·`attn::MlaParams`.
- `placement`.
- 호스트 티어: `moe::{HostLayer, HostScratch, UnionScratch, experts_into, EXPERTS_INTO_MAX, UNION_MAX_COLS, Meta}`, `ops::{Tensor2, Tensor2View, ShardTensor, first_non_finite_col, matmul_q}`.

`ffn`·`head`·`kv`·`profile`·`deepseek2::forward`·`attn`(`MlaParams` 제외)을 크레이트 밖에서 쓰는 코드는 없다.
- 명령: `grep -rnE 'model::(ffn|head|kv|profile)\b|deepseek2::forward|deepseek2::attn::[a-zA-Z_]+' crates --include='*.rs' | grep -v '^crates/model/' | grep -v 'attn::MlaParams'`
- 결과: doc 주석 두 줄(`gpu/src/arch/deepseek2/dispatch.rs:95`, `qdot/src/lib.rs:3955`)뿐이다.

**F2. CPU 전용 편집 하나가 게이트를 거의 다 고른다 [실측].**
- 대상: `9c8ebf0`(09-24 CPU 어텐션 커밋). 바뀐 파일은 `arch/deepseek2/attn.rs`와 `tests/attn.rs` 둘이다.
- 결과: 게이트 레시피 **58개 중 54개**가 선택됐다. `gate-gpu-e2e`, `gate-gpu-ds41-step`, `gate-gpu-load-v41` 등이 들어 있다.
- 58은 그 커밋의 justfile 기준이다. 오늘은 95개이고 그중 `gate-gpu-*`가 60이다. 모든 GPU bin이 `bloomery-gpu`를 거쳐 `bloomery-model`에 의존하므로, 오늘 같은 편집이면 60개 전부가 들어간다[독해].
- 기제: `lib_files()`가 의존 크레이트의 lib 트리 전부를 넣는다(`tools/recipes.py:643-660, 674-733`). `--ledger` 키도 같은 폐포를 쓰므로(`recipes.py:1299`) 원장의 초록도 전부 무효가 된다.
- 게이트 시간[유도: `~/.cache/bloomery/gate-times.tsv`, 09-26 21:09]: 오늘 `gate-gpu-*` 60개의 중앙값 합은 3,398 s, CPU 엔진 전용 게이트 10개는 31.5 s다.

**F3. CPU 엔진은 09-24 이후 rig-log에 나오지 않는다.**
- 마지막 행은 09-24 #cpu-head-bundle-tile(V2-Lite CPU, 깊이 4096에서 69.38 대 ik 66.92)이다.
- 09-25와 09-26 로그에는 `bloomery-decode`를 쓴 행이 없다(grep).
- `plan.md`의 M1–M5에 CPU 전용 디코드 축은 없다.
- README:22는 V2-Lite를 "CPU (`bloomery-decode`) and GPU — the engine's first model"로, README:151은 "that path still runs and is gated"로 적는다.

**F4. 단언하는 할당 래칫은 하나뿐이고, 제품 경로 위에 있지 않다.**
- 유일한 단언은 V2-Lite CPU 스텝의 `LIMIT = 660`(`model/tests/alloc.rs:94`)이다.
- V4.1 스텝 게이트와 하이브리드 게이트는 `allocs_per_step`를 찍기만 한다(`gate_deepseek41_step.rs:2642-2648`, `gate_hybrid.rs:732-733`).

**F5. 호스트 호출이 두 벌이고, 둘이 비트 동일하다는 것이 계약이다.**
- 디코드: `HostLayer::experts_into`(`moe.rs:1536`) → `serve`(`moe.rs:910`) → `matmul_q_group_into` + `matmul_q_group_swiglu_into`. 풀 디스패치 둘에 호출자 합이다.
- 프리필: `experts_union_into`(`moe.rs:1577`) → `UnionCall`(`ops.rs:3712`). 8열 이하의 좁은 호출은 "the decode and verify shapes"로, claim 디스패치 둘에 호출자 합이다(`ops.rs:3152-3160, 3810`; rig-log 09-26#uniondispatch-ab).
- 계약은 `gpu/src/hybrid.rs:1127-1132`에 있고, 게이트는 `model/tests/union.rs:486`이다.

**F6. 증명 체계에 틈이 있다.**
- AGENTS는 디스패치 경로를 건드린 라운드에 같은 임대 A/B를 요구한다(`AGENTS.md:356`).
- `card.py`는 대역이 0을 품는 ab 카드를 바퀴 수와 무관하게 거부한다(`tools/ref/card.py:66-69`).
- A/B 없이 되는 클래스는 구조 동일(`AGENTS.md:182`)뿐이다.

**F7. 스테이지 0 빌드 둘이 `just gate`에 있는데, 엔진은 그 크레이트들을 쓰지 않는다.**
- `build-gpu`는 `q3k-gemv`만 짓는다(justfile:61-62). `build-cpu`는 `q3k-cpu`만 짓는다(justfile:331-332). 둘 다 `just gate`(justfile:744)에 있다.
- 두 크레이트를 import하는 코드는 없다(하위 조사 grep).
- 같은 목록의 `gate-gpu-lib`(justfile:773)이 엔진 디바이스 크레이트를 이미 짓고 시험한다.
- lint 175 시점의 `undocumented_unsafe_blocks` 58은 q3k-gemv 49와 q3k-cpu 9였다(`docs/rust-quality.md:11`). 두 크레이트는 `b4436d6`(09-19) 뒤로 바뀌지 않았으므로, 오늘 144개 중 58이 그대로 이 둘에 있다[유도].

**F8. V2-Lite만 옛 API 모양에 남아 있다.**
- 메타를 두 곳이 읽는다. 둘 다 `attention.head_count`·`key_length`·`rope.dimension_count`를 읽는다.
  - `deepseek2/hparams.rs`: `Hparams::read`, 오류는 `PlacementError::Metadata`.
  - `deepseek2/attn.rs:369-420`: `MlaParams::read`, 오류는 `ModelError::MissingTensor("metadata …")`.
- model에서 `&Gguf`를 받는 함수 29개는 거의 전부 V2-Lite 경로(deepseek2, head, moe, ops)에 있다. 그 뒤에 들어온 코드는 전부 `&Split`/`ShardTensor`를 쓴다(grep 집계).
- `derived.rs:32`는 `attn.rs`에서 `MlaParams`·`Q8Block`·`quantize_q8_0`을, `:36`은 CPU `head::HeadPlan`을 가져온다. 그래서 GPU V2-Lite가 적재 때 `attn.rs`의 코드를 실행한다.

**F9. serve가 요청 필드를 조용히 무시한다.**
- `bloomery-serve`는 모의 엔진만 서빙한다(`bloomery-serve.rs:1-9`).
- 실서버는 `repeat_penalty: 1.0, repeat_last_n: 0`을 박아 넣는다(`gpu-gates/src/bind.rs:152-163`).
- 요청의 `repeat_penalty`, presence, frequency 필드는 읽지도 거절하지도 않는다. `grep -n 'repeat\|penalty' crates/serve/src/*.rs`의 결과는 `/props` 보고(`api.rs:560-565`)뿐이다.
- 샘플러가 거절한 파라미터는 조용히 argmax로 떨어진다(`bind.rs:167-174`).
- 이것은 `api.rs:477`의 "A field this server cannot honor is a 400 naming it, never a 200 that ignores it"와, AGENTS의 "No silent failure"에 어긋난다.

## 2. 구조적 발견 (payoff/cost 순)

### CPU1 — "CPU 엔진" 크레이트가 GPU 엔진의 메타·배치·호스트 티어 크레이트가 됐는데, CPU 엔진이 아직 그 안에 있다

**어디**
- `model/src/lib.rs:1-21`: 크레이트 문서 "The CPU engine"과 결정 셋, 그리고 `Slot`.
- CPU 엔진 모듈: `arch/deepseek2/{attn,forward}.rs`, `ffn.rs`, `head.rs`, `kv.rs`, `profile.rs`, `bin/bloomery-decode.rs`.
- `moe.rs:43-326·446-750` [03]: CPU MoE(`Buckets`·`MoETrace`·`route_inner`·gather/scatter·`moe_ffn_with`·`log_experts`).
- `ops.rs`의 CPU 진입점 [03]: `matmul_q`, `matmul_q_batch`, `matmul_q_group`, `rms_norm`, `f32_tensor`, `4056-4276`의 `matmul_q_one`/`matvec_q_local`/`matmul_q_multi`, `26-160`의 `Tensor2` 자유 목록.

**규모**
- src는 5,679줄이다[실측: deepseek2 6파일 + ffn·head·kv·profile + bloomery-decode]. 여기서 GPU가 쓰는 hparams·names·derived·plan·mod·`MlaParams`를 빼면 약 4,600줄[유도].
- 테스트 10파일 3,539줄[실측], 러너 5개 385줄[실측].

**쌓인 경위**
- 1단계(09-19~21)가 CPU 엔진으로 시작했다: `fcd59e8` KV, `9b6ad4a` 계측기, `4d1b00e` Derived, `b37e8ef` 그룹 디스패치.
- GPU 단계는 이 크레이트를 V2-Lite 메타와 호스트 티어의 공급자로 import했다.
- `9c43ea6`(09-23, arch-split M1)이 모델을 아는 코드를 `arch/deepseek2`로 옮겼지만 크레이트는 나누지 않았다. `arch-split.md`의 전제는 "호스트 코드의 경계는 모듈 디렉터리와 기계 검사로 충분하다"였다.
- 게이트 선택과 원장(`just affected`, `--ledger`)은 그 뒤에 생겼고, 크레이트 그래프를 단위로 삼는다.
- CPU 어텐션 라운드는 09-24까지 이어졌다(`0f01679`, `9c8ebf0`).

**오늘 짓는다면 (두 단계)**
- (b) V2-Lite CPU 엔진 전부를, 그 게이트·러너·레버와 함께 잎 크레이트 `bloomery-cpu`로 옮긴다. GPU 크레이트는 여기에 의존하지 않는다. `bloomery-model`에는 arch 메타, placement, 호스트 티어만 남는다.
- (a) 그다음 `bloomery-cpu`를 은퇴시킬지는 사용자가 정한다(README가 지원 경로로 적는다).
- 참조 엔진도 같은 모양이다.
  - ik와 llama.cpp는 모델 메타(`src/llama-arch.cpp`, `llama-hparams.cpp`, `llama-model-loader.cpp`)와 계산(`ggml/src/iqk`, `ggml-cuda`)을 가른다.
  - mistral.rs는 모델을 모르는 `mistralrs-quant`(`mistralrs-quant/src/moe/mod.rs:1`, "MoE support that is independent of any specific quant scheme")와 `mistralrs-core`(모델, `device_map`)를 가른다.
  - 어느 쪽도 디딤돌 모델의 두 번째 엔진을 메타 크레이트 안에 두지 않는다.

**얻는 것**
- (b)의 이득:
  - CPU 전용 편집이 CPU 게이트 약 10개만 고른다. 오늘은 54/58 이상이다(F2).
  - GPU 원장을 무효로 만들지 않는다.
  - 지금은 리드가 ptx-scan 규칙으로 과선택을 손으로 좁힌다(`AGENTS.md:91` 부근). 그 판단 없이도 그래프의 답이 맞아진다.
- (a)까지 가면 추가로 빠지는 것:
  - 게이트 레시피 10개(attn, ffn, moe, head, forward, kv, alloc, mt, profile, prompts)와 `gate-ops`·`gate-derived`의 일부.
  - 러너 5개와 레시피 6개(`build-decode`, `measure-decode`, `measure-sweep`, `ab-decode`, `measure-profile`, `perf-decode`).
  - 레버 약 11개: `FLASH_SIMD`, `KV_PREFETCH`, `KV_PREFETCH_ROWS`, `FLASH_SEGMENTS`, `ATTN_BUNDLE`, `ATTN_HALVES`, `WEIGHTS`, `POPULATE`, `PROFILE`, `EXPERT_LOG`, 그리고 러너의 `PROFILE_DEPTH`.
  - qdot의 V2-Lite 전용 가족 735줄[유도: 절 표시 사이]과 그 테스트: Q6_K `2318-2587`, Q5_1 `3306-3513`, cells `3948-4106`, `dot_f32` `4107-4158`, `sum_sq_f64` `4159-4204`.

**비용과 증명**
- (b)는 이동 클래스다.
  - 디바이스 코드가 움직이지 않으므로 모든 bin의 `just ptx-scan`이 같다.
  - 구조 줄(그래프 노드 수, eager = 재생, e2e 집합)이 같다.
  - CPU 게이트는 레시피만 `-p bloomery-cpu`로 바뀌고 출력 비트는 같다. A/B는 없다.
  - 선행 조건은 CPU4다. `attn.rs`에 GPU V2-Lite가 적재 때 부르는 코드(`MlaParams::read`, `quantize_q8_0`)가 있으므로(F8), 그것이 먼저 빠져야 경계가 깨끗하다.
- (a)는 게이트 제거이므로 커버리지 변경이고, 날짜 붙은 이유가 필요하다.
  - `gate-prompts`(CPU argmax 대 ik)는 V2-Lite GPU의 `gate-gpu-e2e`(ik CUDA 33프롬프트 × 32스텝, justfile:110)가 이미 대조한다.
  - `gate-alloc`은 제품 쪽에 짝이 없으므로 CPU3이 먼저다.

**위험, 주인, 순서**
- 주인은 aa다. `moe.rs`·`ops.rs` 몫은 [03]이다.
- r8host(`moe.rs`·`ops.rs`·`r8file.rs` 편집 중)와 boxlease(`tools/ref/*.sh`) 뒤에 한다. CPU4 뒤, (a)는 CPU3과 사용자 결정 뒤다.
- 위험은 낮다(이동). 남는 판단은 `ops.rs`의 공유 디스패치를 어느 크레이트에 둘지인데, CPU6이 끝나면 답이 단순해진다.

### CPU2 — 스테이지 0 크레이트는 박물관인데 `just gate`가 그걸 짓는다

**어디**
- 크레이트: `crates/q3k-gemv`(1,911), `crates/q3k-cpu`(743), `crates/oxide-ice-unroll`(1,677과 REPRO 문서), `crates/gpu-spike`(345).
- justfile: `build-gpu` 61, `build-cpu` 331, `measure-gpu`/`measure-cpu` 341–345, `build-ref-bench` 429, `gate` 744.
- 러너: `tools/ref/measure.sh`(52), `cpu-measure.sh`(26).

**쌓인 경위**
- 스테이지 0 라운드는 09-19에 있었다(RESULTS 파일이 전부 09-19, 하위 조사).
- `crates/gpu`가 "the stage-0 CUDA kernels packaged as a library"(`gpu/src/lib.rs:1`)로 그 커널들을 가져갔다.
- `gate_p1`이 K=2048 출력을 스테이지 0 포트의 FNV 해시에 이미 핀했다(`gate_p1.rs:9-13`).
- qdot이 q3k-cpu의 커널을 대체했다. 크레이트들은 `just gate`에 남았다.

**오늘 짓는다면**
- 없앤다: q3k-gemv, q3k-cpu, 그 레시피와 러너, `build-ref-bench`, `q3k_cpu_ref`.
  - 하위 조사에 따르면 `measure.sh`는 임대 밖에서 3090을 쓰고 측정 흐름 안에서 빌드한다. `docs/gpu-design.md:73`은 `measure-gpu`가 이미 깨졌을 수 있다고 적는다.
- gpu-spike의 고유 검사 둘은 `crates/gpu`의 hw 시험으로 옮긴다: 2노드 그래프의 eager = 재생, 그리고 실패하거나 패닉하는 캡처 몸체가 스트림을 묶지 않는다는 것(`gpu-spike/src/main.rs:79-178`, 하위 조사).
- 그러면 gpu-spike와, 그것만 부르는 엔진 API 셋이 함께 빠진다(grep 확인): `Gpu::gemv_q4k`(`gpu/src/lib.rs:3153`), `probe_q4k_launch_us`(`:3176`), `Probe::enqueue_two_phase`(`probe.rs:190`).
- 그 뒤에는 `q3k_ref` 데이터(`attn.q4k`, `x_m*.f32`, `y_ref_*`)를 읽는 곳이 없다.
- `oxide-ice-unroll`은 핀 이동(oxidefork ①) 때 은퇴한다(`docs/upstream/nvlabs-ledger.md:41`). 그 전에도 의존에 `rev`가 없어서(`crates/oxide-ice-unroll/Cargo.toml:9-10`, 재현기 설명과 모순), box.sh 밖에서 빌드하면 업스트림 HEAD를 푼다.

**얻는 것**
- lint 144 중 58이 빠진다[유도]. 그러면 MUL-10(`deny`)이 닿는 거리에 온다.
- `just gate`에서 엔진에 대해 아무것도 증명하지 않는 빌드 둘이 빠진다.
- Rust 2,654줄[실측]이 빠진다.
- AGENTS의 "Never" 예시와 측정 소유자(`AGENTS.md:9-19`)가 살아 있는 러너를 가리키게 된다.

**비용과 증명**
- 엔진 코드가 바뀌지 않으므로 ptx-scan 비교도 A/B도 필요 없다.
- `just gate`의 구성 변경은 커버리지 변경이므로 날짜 붙은 이유를 단다: "`gate-gpu-lib`(justfile:773)가 고정 툴체인으로 엔진 디바이스 크레이트를 짓고 시험한다. `build-gpu`는 아무도 import하지 않는 q3k-gemv만 지었다."

**위험, 주인, 순서**
- 주인은 aa다.
- boxlease(`tools/ref/*.sh`, `lease.sh`의 stage0 증인 필드) 뒤에 한다.
- gpu-spike 몫은 gpucore가 캡처 실패 검사를 `crates/gpu`로 받은 뒤에 한다(인계 의존).
- ice는 핀 이동과 한 착륙으로 한다.

### CPU3 — 래칫과 계기가 첫 모델의 CPU 경로에 박혀 있다

**어디**
- `model/tests/alloc.rs:68-94`: 유일한 단언 할당 래칫이고, 대상은 V2-Lite CPU 스텝이다.
- 제품 쪽은 찍기만 한다(F4).
- `AGENTS.md:356`: 디스패치 경로 A/B를 `ab-decode`로 한다.
- `AGENTS.md:443`: `gate-alloc`.
- `profile.rs`(442줄)와 ops·moe·ffn·head·attn의 훅.
  - `profile::report`를 부르는 곳은 `bloomery-decode.rs:192,268`뿐이다(grep).
  - 그런데 호스트 티어의 핫 경로가 그 표에 기록한다: `moe.rs:910-958` `serve`의 `host_x_quant`·`host_h_quant`·`host_sum`.

**쌓인 경위**
- `gate-alloc`의 PIN 줄은 09-20~21의 CPU 디코드 라운드들에서 왔다.
- b5host는 V4.1 호스트 변경을 V2-Lite `ab-decode`로 지켰고, "빌드 간 1 % 아래라 판정하지 않았다"(`plan-ledger.md:988`).

**오늘 짓는다면**
- 할당 핀은 제품 스텝에 둔다. `gate-gpu-ds41-step`이 이미 찍는 `allocs_per_step`를 단언으로 바꾼다.
  - 한도는 착륙 때 그 게이트가 스스로 찍은 값으로 하고, 숫자는 그 라운드가 정한다.
  - FAIL-first는 스크래치 재사용 하나를 일부러 풀어서 빨강을 확인한다.
- 디스패치 경로 계기는 제품의 다리로 한다: `time-cpu-v41-host`의 엔진 팔과 `depth-gpu-ds41`.
- 호스트 티어 계측은 제품이 찍는 stat 줄 하나로 한다(`HybridStats`, `stat prefill split`). `profile` 훅은 CPU 엔진 쪽에만 둔다.

**얻는 것**
- AGENTS가 인용하는 유일한 정수 속도 가드가 출하 경로를 지킨다.
- expert가 3.76배 작은 프록시 체제[유도]가 호스트 티어 라운드를 판정하지 않게 된다.
- 호스트 티어가 `profile::level()` 분기와 `record_time` 호출을 벗는다.

**비용과 증명**
- 단언 추가는 새 핀이므로 FAIL-first가 필요하다.
- 훅 제거는 타이머만 걷는 호스트 전용 변경이다.
  - 비트 동일 게이트(`gate-union`, `gate-ds41-host`, `gate-gpu-hybrid`)와 ptx-scan 동일로 증명한다.
  - 호출당 원자 load 하나를 층당 0.69 ms와 견주면 비용 모형이 0을 주므로 A/B는 없다.

**위험, 주인, 순서**
- 주인은 aa이고, 스텝 게이트 파일은 gates-ds41 몫이라 이 규칙을 넘긴다.
- CPU1(a)보다 먼저 한다.

### CPU4 — V2-Lite만 옛 모양(`&Gguf` + `TensorInfo`, 메타 리더 둘)으로 호스트 티어를 탄다

**어디**
- 메타 리더 둘(F8).
- `ops.rs`: `Weight::in_file`(1067–1080), `ExpertStack::in_file`(1042–1045).
- `moe.rs`: `expert_view`(327), `MoeBlockPlan`(355–445), 자유 함수 `experts_into`(767–805), 자유 함수 `experts_union_into`(1406, 호출자는 `tests/union.rs:1079` 하나).
- `gpu/src/arch/deepseek2/mod.rs:87-131, 424`: `PlanHost`가 `Derived::new(gguf)`의 CPU 스텝 계획으로 호스트 티어를 만든다.

**쌓인 경위**
- 1단계 API는 `Gguf` 위에 지어졌다(09-19/20).
- b5host(`292a484`)가 V4.1에 `ShardTensor`와 `HostLayer`를 들였고, qwen3moe·deepseek41은 `host.rs`에서 `HostLayer`를 짓는다. V2-Lite만 한 파일 경로에 남았다.
- `arch-split.md`의 b5plan 정정(09-23)은 이미 "V2-Lite는 1샤드 Split으로 연다"고 적었다.

**오늘 짓는다면**
- `arch::deepseek2::host::layer(split, hp, l) -> HostLayer`를 qwen3moe·deepseek41과 같은 모양으로 둔다.
- `MlaParams`는 `Hparams`에서 유도한다. 리더 하나, 오류 타입 하나(`Metadata { key, detail }`)가 된다.
- wk_b 재양자화는 CPU 스텝 계획 없이 `arch::deepseek2`의 적재 함수로 한다.
- `Gguf`는 `Split` 아래의 샤드 리더가 된다.

**얻는 것**
- ops와 moe의 텐서 손잡이와 호스트 API가 하나가 된다. `Weight::in_file`, `ExpertStack::in_file`, 두 자유 함수, GPU용 `MoeBlockPlan`이 사라진다.
- GPU V2-Lite가 CPU 엔진에서 떨어진다. CPU1(b)의 선행 조건이다.

**비용과 증명**
- 비트 동일 클래스다. 호출은 같은 `serve()`와 같은 그룹 디스패치를 타고, 바이트 슬라이스의 출처만 바뀐다(같은 바이트).
- 등식 하나를 확인해야 하고, 확인했다.
  - `HostLayer`는 `serve(.., Some(limit))`를 부른다. V2-Lite의 한도 0.0은 `qdot::swiglu_clamp`가 `limit <= 1e-6`일 때 `swiglu` 자신을 부르는 경로로 간다(`qdot/src/lib.rs:4231-4250`).
  - 같은 함수이므로 같은 비트다. 부동소수 클래스가 아니다.
- 게이트는 `gate-gpu-hybrid`, `gate-derived`, `gate-gpu-e2e`다. `tests/union.rs:1079`의 v2lite 케이스는 `HostLayer` 형으로 옮긴다.
- ptx-scan이 같고 디스패치 구조가 그대로이므로 A/B는 없다.

**위험, 주인, 순서**
- `arch/deepseek2`는 aa다. `gpu/src/arch/deepseek2/mod.rs`는 gpucore 파일이라 인계한다. `moe.rs`·`ops.rs`의 삭제 몫은 [03]이다.
- r8host 뒤, CPU1(b) 앞에 한다.

### CPU5 — serve: 서버 이야기 둘, 샘플러 둘, 생성 루프 둘, 디코더 셋, 조용한 무시

**어디**
- 모의 전용 bin: `bloomery-serve.rs:1-9, 128-131`.
- 샘플러 둘:
  - `serve/src/sampling.rs`(146줄). 머리말 "used until the sampler crate is plugged in"은 `4afdd65` 뒤로 거짓이다.
  - `sampler` 크레이트.
  - RNG(SplitMix64 대 xoshiro256**), NaN 규칙, penalty 유무가 서로 다르다(하위 조사).
- 파라미터 구조체 둘: `serve::SamplingParams`(`engine.rs:218`)와 `sampler::SamplerParams`. 그리고 F9의 조용한 무시.
- 생성 루프 둘: `serve/src/genloop.rs:331-436`과 `gpu-gates/src/bin/bloomery_chat.rs:285-355`.
  - 멈춤: 앞은 `vocab.eos()` 하나, 뒤는 `tok.eog()` 집합이다.
  - 샘플러 이력: 앞은 생성분만, 뒤는 프롬프트와 생성분이다.
  - 문맥 한계 판정도 다르다.
- UTF-8 디코더 셋: `tokenizer/src/decode.rs:76-134`, 그 사본 `bind.rs:105-146`, 그리고 `mock.rs:358-389`.
- 슬롯 저장·복원은 모의에서만 닿는다(`bloomery_serve_ds41.rs:188` `slot_save_path: None`).

**쌓인 경위**
- serve(`ae9abba`, 09-24)는 엔진 결속 전에 모의 엔진 위에서 태어났다.
- sampler(`560c31a`)는 나중에 팩토리로 꽂혔다(`4afdd65`).
- chat bin은 따로 자랐다.

**오늘 짓는다면**
- serve가 `sampler`에 의존하고 `SamplerParams`를 그대로 받는다. 파라미터 구조체가 하나가 되고, penalty에 닿을 수 있고, 못 받는 값은 이름 붙은 400이 된다.
- 생성 루프는 serve의 것 하나를 chat도 몬다.
- 디코더는 tokenizer의 것을 쓰고, 어휘를 소유하는 형을 하나 더한다.
- 모의 엔진은 게이트 픽스처라는 것을 이름으로 밝힌다(예: `bloomery-serve-mock`).
- 참조: mistral.rs는 샘플러 모듈 하나(`mistralrs-core/src/sampler.rs`)를 쓴다.

**얻는 것**
- 조용한 실패 부류 하나를 닫는다: 요청 필드 무시, 거절된 파라미터의 argmax 폴백.
- 같은 시드가 chat과 server에서 같은 토큰을 낸다.
- `sampling.rs` 146줄과 루프·디코더 사본이 빠진다.

**비용과 증명**
- 호스트 전용이고 A/B는 없다.
- 게이트는 `gate-serve`, `gate-gpu-ds41-serve`, `gate-gpu-ds41-chat`, `gate-sampler`다. 모의의 시드 기대값이 RNG를 따라 바뀌면 날짜 붙은 재핀이 필요하다.

**위험, 주인, 순서**
- 주인은 aa(serve)이고, `bind.rs`·`bloomery_chat.rs`는 gpu-gates bin 주인 몫이다.
- 다른 발견과 독립이다.

### CPU6 [03] — 디코드는 그룹 엔진, 프리필은 UnionCall: 같은 수학을 두 엔진이 한다

**어디**
- 디코드 쪽:
  - `moe.rs:752-958`: `EXPERTS_INTO_MAX`, 자유 함수 `experts_into`, `HostScratch`, `serve`. 그리고 `moe.rs:1536-1570`.
  - `ops.rs:1128-1197`: 진입점 넷.
  - `ops.rs:1339-2760`: `group_core`, `run_group`, `DeferredSlots` 등. 약 1,420줄이고[유도: 줄 번호], 레인·비용 코드 일부는 union과 공유한다.
  - `ops.rs:2282-2313`: 레버 `BLOOMERY_DEFER_QUANT`와 `#[doc(hidden)] pub fn set_defer_quant`. R27(`docs/rust-quality.md:77`) 위반이다.
- 프리필 쪽: `UnionCall`, `ops.rs:3140-4056`.

**쌓인 경위**
- 그룹 엔진은 CPU 디코드에서 났다(`b37e8ef`, 09-21). `DeferredSlots`의 +1.4 %는 CPU 디코드 스코어보드의 값이다(`plan-ledger.md:834`).
- b5host가 이것을 V4.1 디코드에 재사용했다.
- uniondispatch(`9626c7f`)가 union을 다섯 디스패치로 다시 쓰면서 좁은 claim 형태를 "the decode and verify shapes"(`ops.rs:3152`)용으로 만들었다. 디코드는 그리로 옮기지 않았다.

**오늘 짓는다면**
- 호출을 하나로 한다: `HostLayer::experts(x: Tensor2View, lists, out, scratch)` = `UnionCall`. 디코드는 그 한 열 경우다.
- 참조: ik의 CPU MoE도 토큰 수와 무관하게 op 하나다(`ggml/src/ggml.c:18389` `ggml_compute_forward_mul_mat_id`, `:18708` `…_up_gate`).

**얻는 것**
- 빠지는 것: `serve`, `HostScratch`, 자유 함수 `experts_into`, `Entry` 행렬, `DeferredSlots`, `BLOOMERY_DEFER_QUANT`(`grep -c BLOOMERY_DEFER_QUANT AGENTS.md` → 0), `set_defer_quant`.
- CPU1(a)까지 가면 `group_core`, `run_group`, `matmul_q_group*` 전부가 빠진다.
- `HostExperts` 트레이트(`hybrid.rs:1117-1146`, gpucore 파일)의 메서드도 하나가 된다.

**비용과 증명**
- 비트는 이미 게이트가 쥐고 있다(열마다 비트 동일, `union.rs:486`).
- 속도가 막힌 곳이다.
  - 예측 Δ ≈ 0이다[유도: 다리가 132–136 GB/s에서 DRAM 바운드이고, 두 형태 모두 호출당 풀 디스패치 둘].
  - 0을 품은 ab 카드는 거부된다(F6).
  - 09-24의 `engine:5` 24.89 ms 같은 다른 임대의 값과 비교하는 것은 AGENTS가 금한다.
- 그래서 길은 둘이다.
  - (i) 한 열 UnionCall이 `serve`와 같은 풀 디스패치 수와 같은 레인 절단을 낸다는 것을, 셀 수 있는 구조 주장(디스패치 수와 레인 경계의 단위 시험)으로 만들고 이동 클래스로 착륙한다.
  - (ii) 리드가 동등성(한쪽 비열등) 카드 종류를 먼저 들인다(§4-5).

**위험, 주인, 순서**
- [03]이고, r8host 뒤에 한다.
- 디코드는 헤드라인이다. 레인 절단이 달라지면 자 밖으로 움직일 수 있으므로 (i)이 안전하다.

### CPU7 — "타입 → 활성값 형식" 규칙의 주인이 열둘이고, 기본 팔이 Q3_K다

**어디**
- 타입 있는 주인: `gguf/src/quant.rs:1120-1128` `activation_format`.
- qdot의 match 표 11개(하위 조사: `qdot/src/lib.rs:136, 173, 193, 221, 242, 278, 323, 370, 390, 442, 765`).
  - 그중 `col_bytes`(`:221-227`)와 `quantize_col`(`:256` 부근)은 `_ =>` 팔이 q8_K 경로다. 나열되지 않은 타입은 조용히 Q3_K 규칙을 탄다.
- `ops.rs:2668-2694`의 `tile_units`는 qdot 타일의 명령 수를 손으로 옮겨 적었다.
- `ops.rs:3166-3178`은 Q3_K와 Q5_0을 형식의 대리로 쓴다.

**쌓인 경위**
- 가족이 들어올 때마다 팔을 더했다: Q5_K `c9cdf7e`, IQ3_XXS·MXFP4 `55b5249`.

**오늘 짓는다면**
- 타입 기술자 하나를 둔다. gguf에는 블록 기하·활성값 형식·k 단위를, qdot에는 타일 종류와 비용 단위를 둔다.
- 모든 디스패치와 크기 계산이 그 기술자를 읽는다.
- match는 망라형으로 해서, 새 타입이 빠지면 컴파일 오류가 되게 한다.

**얻는 것**
- 다음 타입(Qwen 최신 파일의 형식)이 표 하나만 만진다.
- 잠복한 조용한 오경로 부류를 닫는다.

**비용과 증명**
- 호스트 상수 정리이므로 비트 동일(`gate-qdot`, `gate-union`, `gate-ds41-host`)이다. ptx와 무관하고 A/B는 없다.

**위험, 주인, 순서**
- 주인은 aa다.
- qdot을 만지는 03의 h3tile-b·r8 라운드 뒤에 한다.

### CPU8 [03] — `bench_v41_host`(3,529줄)가 판정 끝난 팔과 엔진이 떠난 흐름을 들고 있고, 그 때문에 `ops.rs`에 진입점 하나가 산다

**어디**
- `bench_v41_host.rs`: `:1-140`(모양 설명), `:179` `UNION_CHUNK`, `:936-1030` `union_chunks_call`, `:1204-1460`(청크 계획 위의 r8·q 모양), `:1770` `ThpCopy`, `:2752-2802` `Shape`.
- `ops.rs:1165-1197` `matmul_q_group_cols_into`와 `Entry::Cols`. 주석이 직접 말한다: "The engine's host union does not call it … the host bench's per-chunk union arms do".

**판정 끝난 팔**
- `read-mmap`/`read-thp`: E4 "매핑은 원인이 아니다"(`plan-ledger.md:295`).
- `union`(청크 흐름): 엔진이 떠난 흐름이다(`9626c7f`; rig-log 09-26#uniondispatch-ab, 디스패치 161 → 5).
- `engine-sep`: 디스패치당 바이트 법칙(MUL-35, `facts.md:108`).
- 판정이 남은 것: `unionr8`/`unionq`는 r8host가 정한다.

**오늘 짓는다면**
- 벤치는 엔진의 호출만 몬다.
- 커널 변형도 그 호출을 통해서만 잰다. r8host가 r8을 `UnionCall`에 넣으면 그 모양이 된다.

**얻는 것**
- `matmul_q_group_cols_into`, `Entry::Cols`, `UNION_CHUNK` 기계, `ThpCopy`, 읽기 모양이 빠진다. 줄 수는 세지 않았다.

**비용과 증명**
- 벤치 전용이다. `--check` 정확성 팔은 남는다. ptx와 게이트는 바뀌지 않는다.

**위험, 주인, 순서**
- [03]이고, r8host 착륙 뒤에 한다.

## 3. 삭제

| 대상 | 증거 | 커버리지 변경 |
|---|---|---|
| `model/src/bin/markov-accept.rs`(714)와 레시피 `markov-accept`(justfile:1138) | E6 "답함 — 닫힘"(`plan-ledger.md:296`, rig-log 09-24#markov-head-alone-offline) | 없음(게이트 아님) |
| `arch/qwen3moe/host.rs`(49)와 `pub mod host` | `grep -rn 'qwen3moe::host\|qwen3moe::{.*host' crates --include='*.rs'` → 빈 결과 | 없음 |
| 자유 함수 `moe::experts_union_into`(`moe.rs:1406`) | 호출자는 `tests/union.rs:1079` 하나 | 그 케이스를 `HostLayer` 형으로 옮긴다(CPU4) |
| `qdot/src/bin/qdot-rate-mt.rs`(261), `threads/src/bin/pool-rate.rs`(100) | justfile·tools·AGENTS grep → `facts.md:108` 인용과 `plan-ledger.md:1605` 수동 명령뿐. `AGENTS.md:16` "Never hand-run a benchmark". 판정은 09-20 MUL-35 | 없음 |
| `tools/ref/governor-ab.sh`(25) | 레시피 없음(`plan-triage.md:234`의 언급뿐) | 없음 |
| `tools/ref/depth-decode.sh`(71) | `grep -n depth-decode justfile` → rc 1. AGENTS는 인용만 한다 | CPU1(a)와 함께 지운다. 엔진이 남으면 레시피를 준다 |
| `BLOOMERY_ATTN_HALVES` 팔 | AGENTS 레버 절 자체가 "bit-identical, gated, and slower … so the default is 1" | `tests/attn.rs:556-583`의 halves 2 케이스, 날짜 붙은 이유 |
| `BLOOMERY_WEIGHTS=anon\|huge`(`bloomery-decode.rs:130-137`) | `cpu-dispatch-plan.md:63` "큰 페이지 무차이", `plan-ledger.md:834` "+1 %였다가 잡음으로", E4. `gguf::Weights::Resident`는 남긴다(§5) | 없음 |
| bench 팔 `read-mmap`/`read-thp`/`ThpCopy`/`union`(청크)/`engine-sep`, `ops::matmul_q_group_cols_into`·`Entry::Cols` [03] | CPU8 | 없음(벤치). r8host 뒤 |
| `serve/src/sampling.rs`의 참조 샘플러 | CPU5 | `gate-serve` 시드 기대값 재핀 가능 |
| tokenizer `hunyuan-dense` 별칭(`pretok.rs:213`) | `grep -rn hunyuan crates tools docs justfile README.md AGENTS.md` → tokenizer 소스 세 곳뿐, 게이트 없음 | 없음(시험 없는 지원 주장 제거) |
| `sampler::Sampler::params()` | `grep -rn '\.params()' crates --include='*.rs'` → 다른 타입의 `body.params()` 하나뿐 | 없음 |
| `gguf::v41::MIXED`(`v41.rs:24`) | `plan.md:9`: 혼합 파일은 시팅 10 뒤 지운다. 그때 함께 | 없음 |
| 스테이지 0: q3k-gemv, q3k-cpu, `build-gpu`, `build-cpu`, `measure-gpu`, `measure-cpu`, `build-ref-bench`, `measure.sh`, `cpu-measure.sh`, `q3k_cpu_ref`. 이후 gpu-spike(엔진 API 셋과 `q3k_ref` 데이터 포함). 핀 이동 때 `oxide-ice-unroll` | CPU2 | `just gate` 두 항목(이유는 CPU2) |
| `#[doc(hidden)] pub fn set_defer_quant`와 `BLOOMERY_DEFER_QUANT` [03] | R27, CPU6 | union·ops 테스트의 두 팔 |
| CPU1(a) 시: CPU 엔진 레버 약 11개, 게이트 10개(+일부 2), 러너 5개, qdot 가족 5개 | CPU1 | 날짜 붙은 이유. CPU3 선행 |

## 4. 가로지르는 패턴

1. **크레이트와 모듈 머리말이 현재가 아니라 출생을 적는다.**
   - `model/src/lib.rs:1` "The CPU engine".
   - `ops.rs:1-4` "fast path in `crates/q3k-cpu`".
   - `bloomery-decode.rs:3-9` "matmul_q still materializes f32 … q3k-cpu holds it; stage 1 does not call it". qdot 융합 경로가 이미 붙어 있으므로 거짓이다.
   - `serve/src/sampling.rs:1-3`, `threads/src/lib.rs:10-14`(하위 조사).
   - 범위 밖의 `gpu/src/lib.rs:1`과 `gpu/src/model.rs:1` "an architecture's CPU `forward::step` is its reference". 어느 GPU 코드도 그것을 부르지 않는다.
   - `check-comments.sh`는 날짜와 이슈 번호만 잡아서 이 부류를 못 본다. 착륙 검수 항목이 하나 필요하다.
2. **래칫, 의무 A/B, 프로파일이 첫 모델의 경로에 있고, 제품 경로는 숫자를 찍기만 한다**(CPU3). 게이트 쪽 감사 범위에도 같은 모양이 있을 가능성이 크다.
3. **레버 목록이 등록부가 아니다.**
   - `BLOOMERY_DEFER_QUANT`, `BLOOMERY_WEIGHTS`, `BLOOMERY_POPULATE`, `BLOOMERY_EXPERT_LOG`, `BLOOMERY_HOST_LEASE`는 AGENTS에 한 번도 나오지 않는다(grep).
   - 코드에서 `"BLOOMERY_[A-Z_]+"` 읽기를 모아 (이름, 크레이트, 기본값, 판정 앵커) 표를 만들고 check가 AGENTS와 대조하면, 판정 끝난 팔이 저절로 드러난다.
4. **러너 없는 계측 bin과 스크립트가 "Never hand-run a benchmark"와 공존한다**(`qdot-rate-mt`, `pool-rate`, `governor-ab.sh`, `depth-decode.sh`). 러너를 주거나 지우거나 둘 중 하나다.
5. **증명 체계에 틈이 있다(리드가 재구성 라운드를 계획하기 전에 닫아야 한다).**
   - AGENTS는 디스패치 경로 변경에 같은 임대 A/B를 요구하고, `card.py`는 0을 품은 ab 대역을 거부한다.
   - 그래서 디스패치 구조를 바꾸면서 Δ ≈ 0을 예측하는 재구성(CPU6과, 앞으로의 호스트 경로 정리)은 입증할 길이 없다.
   - 재구성 라운드는 셀 수 있는 구조 동일(디스패치 수, 레인 경계)로 설계하거나, 리드가 동등성(한쪽 비열등) 카드 종류를 먼저 들여야 한다.
6. **경계를 넘어 주인이 둘 이상인 것들.**
   - 타입 → 형식 규칙: qdot 11곳과 gguf.
   - 친화성 코드 넷: threads(`lib.rs:596`), engram(`prefetch.rs:474-567`, 하위 조사), gpu(`launcher.rs:255-322`), q3k-cpu. engram과 launcher의 `cpu_list`는 diff로 동일하다.
   - `BLOOMERY_PIN_MAIN` 파싱 다섯 곳(두 가지 형태).
   - 샘플러 둘, 생성 루프 둘, 디코더 셋, V2-Lite 메타 리더 둘.
7. **첫 파일의 모양(`&Gguf`)이 공유 코드(ops, moe)의 API에 남아 있다**(CPU4).

## 5. 역사처럼 보이지만 남는 것

- `gguf::Weights::Resident`: `BLOOMERY_WEIGHTS` 팔은 판정이 끝났지만, `r8file::Sidecar::open`이 이 인자를 받고(`tests/r8file.rs:258`) gguf 오라클 테스트가 쓴다.
- gpu-spike: `gate-gpu-p0`는 살아 있다(원장상 09-26에 초록 3회, 하위 조사). 캡처 실패 검사는 저장소에서 여기뿐이다. `crates/gpu`가 받을 때까지 남긴다.
- V4.1 fixture(`arch/deepseek41/fixture.rs`, `v41fixture`): 사용자 09-26 "실 모델 대신 훨씬 작은 픽스처" 방향이다. 소비 게이트가 아직 안 붙었을 뿐이다.
- qdot IQ3_XXS·MXFP4: V4-Flash가 다음 대상이다(사용자 결정 09-25). 러너가 아직 없을 뿐이고, IQ3_XXS 속도 측정은 `plan-triage.md:77`에서 대기 중이다.
- qdot Q5_0: GPU V2-Lite 하이브리드 호스트 티어가 쓴다(`gpu/src/arch/deepseek2/mod.rs:107-131` → `experts_into`).
- `lib.rs`의 결정 셋과 `Slot`: CPU 엔진과 함께만 움직인다. 따로 지우지 않는다.
- qdot의 스칼라/AVX2 TWIN 쌍: `#[target_feature]` 본체를 쪼개지 말라는 규칙(AGENTS, rust-quality 기각 ①)이 만든 의도된 형태다.
- `BLOOMERY_STEAL`·`BLOOMERY_STEAL_BLOCKS`: 판정(09-23)은 V2-Lite 프록시에서만 났다. union의 144행 훔치기 블록이 읽으므로, 호스트 티어에서 재기 전까지 남긴다.
- `deepseek41/hparams.rs`(1,511줄)의 모델 둘(`Model::{Deepseek41, Deepseek4}`): `arch-split.md` 원칙 3의 예외로 정한 형태다.
- `placement`의 역할 기반·아키텍처 무지 설계: mistral.rs의 `DeviceMapper`가 expert 단위 배치를 못 하는 것을 보고 정한 것이다(`facts.md`).
- `r8file`/`r8conv`와 `placement/host_lock.rs`: 앞의 둘은 비행 중(r8host)이고, `host_lock.rs`는 살아 있는 레버(`BLOOMERY_HOST_*`)다.
- serve의 `DraftProps`(`engine.rs:211`, 생성되는 곳 0): DSpark 서빙이 계획에 있으므로 남긴다. 다만 시험 없는 표면이다.
- tokenizer의 `collapse_table.rs`(생성물): 남기되, `gen-iq-tables.py --check` 같은 재생성 검사를 붙인다(하위 조사: 지금은 검사가 없다).

## 6. 오늘 다시 짓는다면

```
crates/gguf      Split(샤드 1..n; Gguf는 샤드 리더), Value, TensorInfo,
                 quant(디퀀트 포트 + 타입 기술자 하나), write
                 — 모델 경로(v41.rs)는 도구 프로필·게이트 도우미로
crates/qdot      (타입, 형식)별 AVX2 커널; 디스패치는 기술자를 읽는다(망라형 match);
                 행·열 타일·행-레인 타일; 인코더; swiglu(_clamp)
crates/threads   풀 + 친화성 전부(topology, pin_to, sibling_of, cpu_list) + PIN_MAIN 파싱 하나
crates/model     GPU 엔진이 싣고 서빙하는 것:
  arch/{mod, deepseek2/{hparams, names, host, wk_b}, deepseek41/*, qwen3moe/*, dspark}
  placement/*    역할 → 장치, 계획, hot list, 예산, host set populate/lock
  host/          Tensor2·Weight·ShardTensor; HostLayer(Split + spec);
                 호출 하나 experts(layer, x: Tensor2View, lists, out, scratch) = UnionCall
                 (≤8열: 풀 디스패치 2 + 호출자 합, 넓으면 5); UnionScratch; r8 리더(03)
  bins           v41fixture, r8conv, bench_v41_host(엔진 호출만)
crates/cpu       (잎; 은퇴는 사용자 결정) V2-Lite CPU 엔진 + bloomery-decode + 그 게이트·러너
crates/serve     HTTP + 생성 루프 하나 + 슬롯; sampler·tokenizer 디코더에 의존; 모의는 게이트 픽스처
crates/sampler, tokenizer   그대로(쓰지 않는 별칭·메서드만 뺀다)
스테이지 0       없음(ice는 핀 이동 때); gpu-spike의 검사 둘은 crates/gpu hw 시험
```

**흐름**
- 디코드: GPU 스텝 → `Hybrid` → `HostExperts::experts`(한 열) → `UnionCall`의 좁은 형태.
- 프리필: 같은 호출의 넓은 형태.
- 계기: 할당 핀은 `gate-gpu-ds41-step`, 디스패치 경로 계기는 `bench_v41_host` 엔진 팔과 `depth-gpu-ds41`, 계측은 stat 줄 하나.

**이행** (각 단계가 게이트를 초록으로 유지한다)
1. §3의 판정 끝난 삭제 묶음. 정적 검사와 영향 게이트로 확인하고, 커버리지 변경에는 날짜 붙은 이유를 단다. boxlease 뒤.
2. 스테이지 0 제거(CPU2). `just gate` 구성 변경에 이유를 달고 lint를 다시 잰다. gpu-spike는 gpucore 인계 뒤, ice는 핀 이동과 함께.
3. 제품 스텝 할당 핀(CPU3). 새 단언과 FAIL-first. gates-ds41에 규칙을 넘긴다.
4. V2-Lite를 Split + HostLayer + 메타 리더 하나로(CPU4). 비트 동일 클래스, ptx-scan 동일, A/B 없음. r8host 뒤.
5. CPU 엔진을 잎 크레이트로(CPU1 b). 이동 클래스다. 착륙 뒤 CPU 전용 변경에 `just affected`를 한 번 돌려, 선택이 CPU 게이트뿐인지 확인한다(구조 줄).
6. [03] 디코드를 UnionCall 한 열로(CPU6). 구조 동일 단위 시험을 쓰거나, 동등성 카드를 들인 뒤에 한다.
7. 서로 독립인 셋: serve 통합(CPU5), 타입 기술자(CPU7), 벤치 정리(CPU8 [03], r8host 뒤). 모두 비트 동일이고 A/B가 없다.
8. (사용자 결정) `crates/cpu` 은퇴. 게이트 10개 제거에 날짜 붙은 이유를 달고, 러너 5개·레버 약 11개·qdot 가족 5개를 함께 뺀다.

## 7. 범위 밖에서 본 개선 자리

- **gpu-gates 생성 쪽 사본들(S)**: `bloomery_chat.rs:285-355`의 생성 루프 사본(CPU5), `print_plan` 세 벌(`generate_ds41.rs:764`, `bloomery_serve_ds41.rs:203`, `bloomery_chat.rs:258` — 하위 조사가 둘은 공백만 다르다고 diff로 확인), `Place` 두 벌(`generate_ds41.rs:206`, `generate.rs:27`).
- **찍기만 하는 할당 수(S)**: `gate_deepseek41_step.rs:2642-2648`과 `gate_hybrid.rs:732`의 `allocs_per_step`(CPU3).
- **출생을 적는 머리말(XS)**: `gpu/src/lib.rs:1`, `gpu/src/model.rs:1`(§4-1).
- **친화성 사본(M, 트리아지에 있음)**: `gpu/src/model/launcher.rs:312-322`와 `engram/src/prefetch.rs:543-553`이 diff로 동일하다. `threads`로 모은다.
- **낡은 문서 셋(XS)**
  - `tools/box.sh:7`: 낡은 사용 예 `cargo oxide run q3k_gemv`(하위 조사).
  - `docs/upstream/nvlabs-ledger.md:9,41`: #1314를 열림으로 적는다. `839ce81`과 README:161은 머지로 적는다(하위 조사).
  - `AGENTS.md:9-19`: 측정 소유자로 `measure.sh`·`cpu-measure.sh`를 적는다. `:502-517`의 lint 분해는 175 시점 값이다.
- **CPU 엔진에 기대는 GPU 게이트 둘**
  - `gate_p8b.rs:278`: V2-Lite 스케일을 CPU MoE의 `Meta::read`로 읽는다. `deepseek2::Hparams`로 바꾼다(XS).
  - `gate_hybrid.rs:329-330`: 호스트 기준으로 CPU 엔진 진입점 `model::ops::matmul_q`를 쓴다. CPU1(a) 때 `HostLayer` 호출이나 qdot 직접 호출로 바꾼다(S).
- **tokenizer pre 이름(XS)**: V4.1의 pre 이름을 테스트와 justfile은 `deepseek-v3`로, 문서는 `joyai-llm`으로 적는다(하위 조사). 파일 헤더에서 읽어 단언한다.
- **serve 조용한 폴백(XS, CPU5에 포함)**: `bind.rs:167-174`에서 거절된 샘플러 파라미터가 argmax로 떨어진다. 이름 붙은 400으로 바꾼다.
- **cuda-oxide 원장(XS)**: `oxide-ice-unroll`의 의존에 `rev`가 없다. 핀 이동 전까지 재현기가 핀과 다른 소스를 풀 수 있다.
