## audit-gates-ds41 보고 (ID 접두 GD) — 읽은 커밋 `4786c1e`

### 0. 읽은 범위

- 범위 안 파일은 34개, 38,654줄이다(`wc -l` 실측). 묶음별로 보면 op 게이트 8개 12,428줄, chain 게이트 3개 5,546줄, 본체 게이트 7개(step·skew·long·prefill·dsloop·load·plan) 8,737줄, DSpark 게이트 5개 5,297줄, 제품·도구 bin 8개 5,947줄, `shared/` 3개 699줄이다.
- **전부 읽은 것**: 스펙 두 파일, `oracle/{mod,deepseek41}.rs`(419줄), `gguf/src/v41.rs`(86줄), lib `generate.rs`(207줄), bin 31개와 `shared/` 3개의 모듈 doc 전체, justfile의 V4.1·DSpark 레시피 전부(`:100-125`, `:225-330`, `:440-505`, `:750-1162`).
- **구조까지 읽은 것**: 다음의 `main`/`run`/인자 해석이다. `gate_deepseek41_step.rs:225-300`, `gate_deepseek41_prefill.rs:137-220`, `gate_deepseek41_long.rs:241-330`, `generate_ds41.rs`(항목 개요, `:200-248`, `:520-560`, `:850-905`), `bloomery_chat.rs:258-332`, `bind.rs` 머리, `engine.rs:1-113`, `gate_load_v41.rs`·`gate_deepseek41_load.rs`의 doc. lib.rs는 항목 목록과 `:34-163`, `:530-690`, `:1100-1160`만 읽었다.
- **grep·difflib로 표집한 것**: 중복 함수와 상수 전부(아래 수치).
- **보지 않은 것**: op 게이트 8개의 검사 본문(doc과 중복 헬퍼만 봤다), `gate_deepseek41_chain_{attn,glue}.rs` 본문, `gate_dspark_graph.rs` 본문, `gate_ds41_serve.rs` 본문, `exact_ref`·`forced_probe`·`kld_diff` 본문, 박스의 `$BLOOMERY_DATA` 목록. 마지막 것은 감사 규칙상 참조 트리가 아니어서 읽을 수 없었다.

### 0′. 종이 위 분해와 자원 시간선

박스 명령은 한 번도 실행하지 않았다(ref 트리 읽기도 하지 않았다). 이 보고의 수치는 전부 읽은 값이거나 [유도]다.

**분해.** V4.1 착륙 묶음의 비용은 Σ(적재 + 오라클 + 케이스 + rsync·빌드 확인)으로 나뉜다.

- **적재**: 한 번에 30.0–41.9 s다. `target/gate-batch/lead-smoke-1/g-gate-gpu-ds41-prefill.log`에 "loaded in 30.0 s", `g-gate-gpu-ds41-step.log:19`에 "resident in 41.9 s"로 찍힌 런타임 값이다.
- **prefill 게이트의 디코드 오라클**: 4,097스텝이다(`gate_deepseek41_prefill.rs:199`). 스모크 로그의 "513 steps in 20.4 s"는 스텝당 39.8 ms이므로, 4,097스텝이면 160–190 s다[유도; 깊은 스텝은 더 느리다].
- **항목 시간**: `~/.cache/bloomery/gate-times.tsv`(677행)에서 항목마다 마지막 5행의 중앙값을 냈다. 3090 차선의 V4.1 항목 합은 1,925 s다. 그중 본체 게이트 6개가 1,037 s, 제품 레시피 4개(chat·draft·dspark-loop·serve)가 615 s, op 게이트 8개가 157 s, chain 게이트가 34 s다. solo 차선은 389 s(faults 61 + load-v41 163 + lock 166), A6000 차선에 가는 V4.1 항목은 51.5 s뿐이다[유도].

**자원 시간선(게이트 프로세스 하나 기준).**

- 적재 30–42 s 동안은 호스트 CPU·DRAM(페이지 캐시와 populate)과 PCIe가 바쁘고, 카드 SM은 논다.
- 그 뒤 케이스 구간에서는 카드와 호스트 티어가 번갈아 바쁘다.
- 묶음 벽시계는 max(A, B) + X다. V4.1 게이트 배치는 3090을 이름으로 부르므로 A 차선이 임계 경로다.
- 적재는 다른 자원의 그늘에 들어가지 않는 직렬 항이다. 그래서 첫 레버는 적재를 빠르게 하는 것이 아니라 **벌크**다. 적재 한 번에 케이스 여럿을 돌린다.

**유도로 닫히지 않는 항 둘**(박스 실행을 한다면 이것만이다).

- (a) V4.1 프로세스 둘을 두 카드에서 동시에 돌릴 때의 호스트 DRAM. 같은 파일을 공유 매핑하면 들어간다고 예상한다. 다만 AGENTS에 호스트 집합 214 GB와 카드 48 GB가 264 GB를 넘어 populate 상주 검사가 846페이지 차이로 실패한 실측이 있다.
- (b) 게이트 배치를 A6000에서 돌렸을 때의 비트 동일성. 인덱서 그리드가 SM 수에 맞춰진다(`crates/gpu-deepseek41/src/indexer.rs:1024`). top-k가 분할과 무관하다면 같은 비트가 나온다는 것은 가설이다.

### 1. 지금 아는 것 (시드 목록 밖)

1. **전체 모델 적재는 9개 bin에만 있다.** `body::open`을 부르는 것은 step·skew·long·prefill·dsloop·load·generate_ds41·chat·serve이고, op·chain 게이트는 `Split`을 열어 텐서 일부만 읽는다.
   - 전체 V4.1 목록 한 번이면 적재가 **25회** 일어난다. 3090 차선이 22회(step 1, skew 1, long 1, prefill 1, q3ksplit 1, ds41-load 2, dspark-loop 3, draft 4, chat 6, serve 2), solo가 3회다. justfile `:858-958`, `:1054`, `:313-321`과 각 bin의 open 위치를 세었다.
   - `docs/gates-plan.md` §3.5의 "25개 게이트 바이너리가 각자 `body::open`을 부른다"는 16개 bin에 대해 틀렸다. 적재 횟수 25는 맞지만, 대부분은 제품 레시피가 `generate_ds41`을 되풀이해 띄워서 생긴다.
2. **게이트 라이브러리는 이미 V4.1 디바이스 크레이트를 이름으로 부른다**(`engine.rs:29,47`, `715118b` 이래). 그런데 `generate.rs:6-13`, `shared/ds41_dspark.rs:27-28`, `shared/ds41_finite.rs:16-19`는 "라이브러리는 부르지 않는다"를 이유로 V4.1 하니스를 lib 밖에 두었다.
   - `AnyEngine::Deepseek41` 팔은 두 호출자 모두 거부한다(`generate.rs:296-303`, `gate_e2e.rs:750-752`). ledger `:952`에도 "컴파일만"으로 적혀 있다.
3. **`bloomery-chat`은 프롬프트를 토큰마다 디코드 스텝 하나로 먹인다.** 경로는 `bloomery_chat.rs:292`의 `g.prefill` → `generate.rs:141` → `GpuModel::step`(`crates/gpu/src/model.rs:812-853`, 토큰마다 한 body)이다.
   - chat은 `0399efc`(09-24)에 들어왔고, 배치 프리필 `43cd107`(09-25, 스텝 피드의 2.97배, rig-log 09-25#ds41batch-pp)은 `generate_ds41`과 serve에만 연결됐다.
   - chat·serve·`bind.rs`에는 드래프트가 없다(grep `draft|dspark|lookup` 결과 없음). DSpark 루프는 게이트 크레이트의 bin 모듈(`shared/ds41_dspark.rs`, `generate_ds41.rs:1012-1177`)에 있다. 그래서 README의 51.3 tok/s는 id CLI로만 나온다.
4. **엔진은 여러 카드에 걸친 배치를 거부한다**(`crates/gpu-deepseek41/src/body.rs:127-135`). 반면 `gate-gpu-load-v41`의 기본값은 `--plan b`(두 카드)다(`justfile:313`).
5. **참조 세트 다섯 계열 중 파일을 확인하는 것은 ik 노드 덤프 하나뿐이다.** `_plain` 접미사(`gguf/src/v41.rs:33-71`)와 `# model` 줄(`oracle/mod.rs:86-104`)로 확인한다.
   - dsref(DSpark)는 `# build`만 본다. 읽기 함수가 4벌이다(`gate_dspark_experts.rs:823`, `_kv.rs:99`, `_hc.rs:498`, `_graph.rs:215`). 필드 해석의 `unwrap_or(0)`이 10곳이다.
   - greedy tsv의 헤더는 `basename "$MODEL"`이다(`tools/ref/ik-greedy.sh:104`). 두 V4.1 파일은 샤드 이름이 같아(`oracle/mod.rs:87-90`) 헤더로 구분되지 않고, 리더는 `#` 줄을 건너뛴다.
   - KLD base는 n_vocab만 본다(`kld.rs:132`).
   - README:48과 triage `:83`에 따르면 greedy·KLD·dsref 셋이 혼합 파일로 떠 있다. 그래서 `gate-gpu-dspark-graph`는 `f1d1168`(09-25 04:52) 이래 모든 착륙 묶음에서 FAIL 61–63줄의 **표준 빨강**이다(triage `:39`, `:230`, `:235`, `:252`, `:254`, `:260`). 리드는 FAIL 줄의 md5를 손으로 비교하고 있다.
   - long `--free`와 step `--ppl`은 다른 파일의 참조와 비교하면서 그 사실을 말하지 않는다. 재생성 시팅은 09-25에 승인됐고(triage `:102`) 아직 열리지 않았다.
6. **"all three must pass"(`gate_deepseek41_prefill.rs:108`)와 "1 and 2 must pass"(`:111`)라고 적힌 팔을 도는 레시피가 없다.** justfile을 grep하면 `BLOOMERY_DRAFT`와 `BLOOMERY_Q3K_SPLIT`만 나오고, `CARD_EXPERTS`·`PREFILL_GROUP`·`CED`는 0건이다. gate-times에는 손으로 넣은 `@ENV` 행이 G=1 한 행, slot 두 행뿐이다.
7. **레시피는 195개**다(`just --dump`). 그중 `gate-*`가 95개이고, 셸 논리 토큰(sed·grep·cmp·case·for·if·`$(`·test)을 가진 것이 56개다. DSpark 경로 export는 10벌이다. `box.sh`는 REF·V41 모델 경로만 한 번 export한다(`box.sh:57,66`).
8. **`MANIFEST.tsv`를 읽는 자리가 세 크레이트에 13곳**이다: gpu-gates lib, dsref 4, vision 1, `gate_deepseek41_load.rs:730`, `model/tests` 5, `vision/tests` 1.
9. **ik 빌드 `db517b69`가 bin에 5벌 박혀 있다**(`gate_deepseek41_skew.rs:99`, dspark 4). 오라클은 09-23에 이 트리로 다시 떴다(`527d84c`).

### 2. 구조적 발견 (이득/비용 순)

**GD1 — 참조 세트의 출처에 주인이 없다: 표준 빨강, 그리고 다른 파일 참조와의 조용한 비교**

- **위치**: 위 1-5의 다섯 계열 리더. 계열마다 출처 검사가 다르다.
  - ik 덤프는 파일·아키텍처를 본다.
  - dsref는 빌드만 본다. `IK_BUILD`·`DSREF_SET`가 4벌이다(`gate_dspark_*.rs:56-152`).
  - greedy와 KLD는 파일을 보지 않는다.
- **쌓인 경위**: 파일 전환(`f1d1168`)에 접미사 규칙이 따라간 것은 ik 덤프뿐이었다(`gguf/src/v41.rs`). 나머지 계열은 각자의 bin이 따로 자라났다.
- **오늘 짓는다면**: 호스트 전용 크레이트 `refset` 하나를 둔다.
  - 계열 표: 이름 → 생성 레시피, 모델 파일 신원(1번 샤드의 전체 경로 + 크기, 또는 헤더 digest), ik 빌드, 소비자.
  - 매니페스트 리더 셋: ik v1/v2, dsref, vision.
  - `RefError::{Missing, Stale{dumped_from, runs}, Unfinished}`: 다른 파일에서 뜬 세트는 비교하기 전에 이름으로 거부한다.
  - 생성기(`ik-greedy.sh`, `ik-ppl.sh`, `dump-draft.sh`)는 `dump.sh`처럼 `# model <전체 경로>`를 쓴다.
  - llama.cpp·mistral.rs에 해당하는 것은 없다. 두 엔진은 덤프 오라클 체계를 두지 않는다.
- **이득**:
  - 매 착륙 묶음의 FAIL 61–63줄이 이름 붙은 한 줄("stale reference … dumped from MIXED, the tree runs PUBLIC")이 된다. 손으로 하는 md5 비교가 사라진다.
  - long `--free`와 `--ppl`의 녹색이 doc이 말하는 내용을 다시 보증한다.
  - 13곳 중 gpu-gates 쪽 8곳이 리더 하나로 합쳐진다.
  - 승인된 시팅 10 뒤 혼합 파일을 지우는 작업(triage `:102`)이 기계 검사로 막히지 않고 닫힌다.
- **비용·증명**: S–M, 게이트 동작 변경이다.
  - FAIL-first: 지금 트리에서 혼합 파일 dsref와 greedy가 이름으로 거부되는지 본다.
  - 재생성 뒤 dspark-graph, long, `--ppl`이 녹색이어야 한다. 엔진 코드는 건드리지 않고, 타이밍 A/B는 필요 없다.
- **위험·순서**: 시팅 10(CPU 잡 셋, 몇 시간, 이미 승인)을 **먼저** 돌린다. 그 전에 검사를 착륙시키면 long `--free`가 녹색에서 이름 붙은 빨강으로 바뀐다(정직하지만 커버리지 공백). 주인은 aa다.

**GD2 — V4.1 하니스가 둘 곳이 없어 복사와 `#[path]`로 산다**

- **위치**: 실측한 사본은 다음과 같다.
  - 세트 주입: `gate_deepseek41_skew.rs:197-469`의 273줄 중 250줄이 `gate_deepseek41_step.rs:535-811`과 같다(difflib).
  - 라우터 동률 수학: `ulp32`·`softplus64`·`softplus_err`가 3벌이다(`gate_deepseek41_moe.rs:577-612`, `_chain_ffn.rs:632-663`, `_step.rs:2274-2305`). `tie_margin`은 2벌(`_chain_ffn.rs:669-693`, `_step.rs:2310-2343`), ULP 상수 넷은 3벌이다.
  - 어텐션 밴드 상수: `_chain_attn.rs`와 `_step.rs:225-236`에 있다.
  - dsref 리더 4벌, `top_k_override` 3벌(14줄 동일: `_index.rs:152`, `_skew.rs:197`, `_step.rs:535`), `print_plan` 4벌, `Place` 2벌(`generate.rs:27`, `generate_ds41.rs:206`), `q8_block`·`post_elem`·`node_to_hc` 각 2벌이다.
  - 모델 여는 서두(`Split::open` 두 번 + `Hparams::read` + `body::open(plan_gate, CTX_MAX)`)가 본체 게이트 6벌이다.
  - 사본의 출처를 가리키는 주석("the step gate's `set_state`", "the MoE chain gate's `tie_margin`")이 12개다.
  - 합치면 약 570줄[유도, 실측 사본 크기의 합]이다. 여기에 `#[path]` 포함 10곳, `#[allow(dead_code, reason=…)]` 4곳이 붙는다.
- **쌓인 경위**: `499bced`(09-23 하니스 라운드)는 op 게이트 쪽 사본을 lib로 옮겼다(−318줄). 이어 chain·step·skew·prefill 게이트는 V4.1 타입(`Deepseek41Model`, `Body`)이 필요했는데 lib가 그 크레이트를 부를 수 없다고 적혀 있어서, bin 사이 복사로 자랐다.
  - ledger `:952`는 "다음 하니스 라운드(M)"로 미뤘다. 그 사이 규칙 자체는 `engine.rs:29`가 이미 깨고 있었다.
- **오늘 짓는다면**: `crates/gates-ds41` 크레이트(lib + bin)를 만들고 lib가 `gpu-deepseek41`을 부른다.
  - lib에 둘 것: `SetState`(read·inject: step·skew·chain 공용), `Envelope`(라우터 동률, 어텐션 밴드, q8 규칙 항), `FiniteProbe`, shadow 표, 폴트 심기, `open_gate_placement()`, `print_plan`.
  - gpu-gates lib에는 모델과 무관한 조각만 남긴다: `exit_with`·`verdict`·`KERNEL_BAND`·`ref_gemv`·ptx·nodes·`ik_q8_2`·`act_rule`.
  - 이렇게 하면 `#[path]`, `deepseek41` 피처의 가짜 main, "라이브러리는 부르지 않는다" 주석 셋이 함께 사라진다.
  - 참조: mistral.rs는 크레이트별 `tests/`를 둔다(`mistralrs-quant/tests`, `mistralrs-paged-attn/tests`). 하니스와 대상 크레이트를 한 의존 방향에 둔다.
- **이득**: 약 570줄 중복이 사라지고 `#[path]` 10곳이 없어진다. GD3의 전제 조건이다(공유 프로세스는 공유 하니스를 필요로 한다).
- **비용·증명**: M, 이동 클래스다. 이동한 bin마다 `just ptx-scan` 표가 같아야 한다(디바이스 크레이트는 건드리지 않는다). 게이트마다 `just verdict-diff` 출력이 빌드 줄만 빼고 같아야 한다. `check-recipes`가 녹색이어야 한다. 타이밍 A/B는 필요 없다.
- **위험·순서**: 낮다. 레시피의 `-p`와 `--bin` 경로가 모두 바뀐다. 주인은 aa다. GD3보다 먼저 한다.

**GD3 — 게이트 하나 = 프로세스 하나 = 적재 한 번, 레버가 프로세스 env라 팔마다 적재와 오라클을 다시 한다**

- **위치**: 적재 25회는 위 1-1에 있다. 레버 읽기 위치는 다음과 같다.
  - `crates/gpu-deepseek41/src/body/prefill.rs:342-359`(`BLOOMERY_PREFILL_GROUP`)
  - `chain/ffn/batch.rs:1434-1449`(`CardExperts::from_env`)
  - `body/ced.rs:210-220`
  - `crates/gpu/src/lib.rs:247`(`BLOOMERY_Q3K_SPLIT`)
  - `generate_ds41.rs:741`(`BLOOMERY_DRAFT`), `STEP_PAIR`, `CHECK_FINITE`
  - 그래서 prefill 팔(`@BLOOMERY_CARD_EXPERTS=…`, `@BLOOMERY_PREFILL_GROUP=1`)과 q3ksplit이 각각 적재 30–42 s와 오라클 160–190 s를 되풀이한다.
  - chat·draft·dspark-loop·serve 레시피는 셸 프로그램이다(3,498·1,224·1,248·564자). `tokens`·`draft summary` 줄을 sed·grep·cmp로 뜯고, 적재를 15회 한다.
- **쌓인 경위**: 라운드마다 자기 bin과 자기 레시피를 더했다. 팔은 "한 번 읽는 env" 규약(AGENTS 레버 목록)으로 남았다.
  - `docs/gates-plan.md` §3.5가 적재 공유를 M으로 계획했지만, env 레버가 있는 한 팔끼리는 적재를 공유할 수 없다.
- **오늘 짓는다면**:
  - 레버는 값이다: `PrefillConfig{card_experts, group, ced}`, `Draft{Off, Lookup, Dspark}`. 로드 때 넣거나 호출에 싣는다(이미 `set_indexer_top_k`, `set_mode`가 그 모양이다). env 해석은 제품 CLI 경계에서 한 번만 한다.
  - 게이트는 배치(placement)마다 프로세스 하나로 돈다.
    - gate 배치 1회: step·skew·long(free·trigger)·dsloop·prefill과 팔 전부, chat·draft·serve의 토큰 동일성을 in-process 검사로.
    - DSpark 예산 배치 1회(13G).
    - ds41-load 2회(재적재 회계가 계약이다).
    - faults는 solo로 따로 둔다.
    - 케이스 사이에는 `reset`과 `take_fault`를 한다. prefill 게이트가 이미 한 프로세스 안에서 fault → reset → clean을 도는 선례다. 순서를 바꿔 한 번 돌려 케이스 독립을 증명한다.
  - 참조: mistral.rs는 레버를 타입 있는 `RuntimeOptions`로 둔다(`mistralrs-cli/src/args/mod.rs:582`).
- **이득**[유도]:
  - 3090 차선 적재가 22회에서 4–6회가 된다. 16–18회 × 30–42 s = **480–756 s/묶음**이다.
  - prefill 팔 하나마다 적재 + 오라클 = 190–232 s가 준다.
  - 셸 레시피 넷이 타입 있는 검사가 된다.
  - V4.1 3090 차선 1,925 s는 q3ksplit 삭제(GD6)까지 더하면 약 820–1,100 s가 된다.
  - 두 번째 레버(시간선): gate 배치는 3090을 이름으로 부른다(`workstation.rs:124-129`). `BLOOMERY_CARD_BUDGET`은 A6000에서 3090 예산을 슬롯 단위로 똑같이 흉내 낸다(AGENTS). 배치를 카드와 무관하게 바꾸면 DSpark 프로세스를 B 차선에 둘 수 있다. 단 0′의 (a)·(b)가 열린 항이다.
- **비용·증명**: M–L이다.
  - 레버를 값으로 바꾸는 것은 로드 시점 호스트 코드라서 이동 클래스다: ptx-scan 동일, 노드 수, eager = replay. 한 프로세스 안의 팔 결과가 프로세스별 결과와 같다는 것은 verdict-diff로 본다.
  - 케이스 통합은 verdict 줄 동일로 증명하고, 시간은 gate-times가 잰다. 타이밍 A/B는 필요 없다.
- **위험·순서**: 긴 수명의 프로세스가 누수를 드러낼 수 있다(triage `:222`의 `staging_view` 같은 것). 드러나는 것은 이득이다.
  - 레버를 값으로 바꾸는 쪽은 ds41 감사 범위(`gpu-deepseek41`)라 조율이 필요하다.
  - 순서는 GD2 → 레버 값화 → GD3이다. 주인은 aa다. qwen3 게이트에도 같은 모양이 있을 것이다 [03].

**GD4 — 제품 표면이 게이트 크레이트에 있고 드라이버가 셋이다: chat은 배치 프리필이 없고, chat·serve에는 드래프트가 없다**

- **위치**: 드라이버 셋이 따로 있다.
  - `generate_ds41.rs`(1,429줄)의 `drive`: 자기 `Place`, 프롬프트 리더, 피드 방식 넷(batch·steps·checked·dspark) × 디코드 방식 넷(plain·pair·lookup·dspark). env 레버 5개와 서로 거부하는 조합이 있다.
  - lib `Generator`(`generate.rs`, "one real step per id" — 배치 전의 믿음을 담은 추상): chat이 쓴다.
  - `bind.rs`(599줄) `Ds41Engine`: serve가 쓰며 프리필 클로저를 넘긴다.
  - 출력 줄은 규약 없이 파싱된다: `SMOKE`를 tools 9개 파일이, `time step`을 5개, `time prompt`를 6개, `stat prefill`을 4개가 정규식으로 읽는다.
- **쌓인 경위**: `0399efc`(chat, 09-24) → `43cd107`(배치 프리필, serve와 generate_ds41에만 연결) → `1dcaef6`(DSpark 루프, `shared/`에 들어감).
- **오늘 짓는다면**:
  - `Session`(V4.1 엔진 쪽): 배치 배치로 열기, capture, `prompt(ids)`(배치), step, pair, rollback, 선택적 `Draft`, `stats()`를 한 주인이 갖는다.
  - `crates/cli`의 `bloomery run|chat|serve|bench`: serve 크레이트의 `Engine`을 `Session` 위에 얹는다. bench는 버전이 붙은 레코드를 내고, 러너는 파서 모듈 하나로 읽는다.
  - 참조: mistral.rs `mistralrs-cli/src/args/mod.rs:43`(Serve), `:65`(Run), `:188`(Bench), `commands/bench.rs`. 한 바이너리에 라이브러리 하나다.
- **이득**:
  - chat이 배치 프리필을 얻는다. `43cd107` 때 스텝 피드의 2.97배였고, 그 뒤 배치 쪽만 91.2에서 144.6(lcg pp512)으로 올랐다. 스텝 피드는 다시 재지 않았다.
  - chat·serve가 DSpark를 얻는다(1.146–1.156배, rig-log 09-25#dspark-loop-tps의 generate_ds41 수치).
  - 드라이버 셋이 하나로, `Place` 둘과 `print_plan` 넷이 하나로 준다. 제품 코드가 게이트 lib에서 빠진다.
- **비용·증명**: M이다.
  - `Session` 추출은 이동 클래스다: run·chat·serve의 토큰이 오늘과 같아야 한다.
  - chat의 피드 전환은 prefill 게이트 계약(비트 동일 상태)이 증명한다.
  - bench 레코드는 바이트 호환으로 옮기고, 러너 dry-run 파싱으로 증명한다. 공개 수치를 내려면 시간 행 하나를 선택적으로 잰다.
- **위험·순서**: `generate_ds41`의 줄 형식에 러너 다섯 이상(`depth-ds41.sh`, `nsys-ds41.sh`, `ds41pp.py`, `gpu-ab.py`, `cstate-ab.sh`)과 `tools/flow/ds41_prefill.py`가 기대 있다. GD3 뒤에 한다. 주인은 aa이고, 엔진 쪽은 ds41 감사와 조율한다.

**GD5 — 적재 게이트가 둘이고, 하나는 엔진이 돌 수 없는 배치 (b)를 기본으로 잰다**

- **위치**: `gate_load_v41.rs`(823줄, `Weights::load_placed`로 두 카드, solo 163 s + lock 166 s)와 `gate_deepseek41_load.rs`(1,035줄, 엔진 입구, 74 s)다.
  - 겹침: 후자의 (i)은 전자의 check 1이라고 스스로 적는다. Hparams → plan 파이프라인은 3벌이다(ledger `:952`: `body.rs:68-88`, `gate_load_v41.rs:145-150`, `gate_deepseek41_load.rs:87-89`).
- **쌓인 경위**: `b7b9e49`·`e886a4b`(B1b, 09-23 오전, 엔진 입구가 생기기 전) → `715118b`(`body::open`과 ds41-load, 같은 날).
- **오늘 짓는다면**: 엔진 입구 적재 게이트 하나만 둔다.
  - 담을 것: ds41-load의 (i)–(vi), load-v41의 메모리 잔차(2)와 독립 패킹 되읽기(3), 페이지 캐시를 비운 뒤의 호스트 상주·잠금(4, `--lock`). 첫 적재 전에 샤드를 fadvise로 비우면 (4)와 (v) 재적재를 한 프로세스에서 할 수 있다.
  - 배치 (b)의 산술은 이미 `model/tests/placement.rs:659`와 `deepseek4_meta.rs:472`가 평가하므로 그쪽에만 남긴다.
- **이득**: bin 하나(823줄)가 빠지고, solo 레시피 2개(329 s) 대부분이 ds41-load 프로세스로 들어간다. 파이프라인 사본 하나가 준다.
- **비용·증명**: S–M, 커버리지 변경이다. 날짜 붙은 사유는 "엔진이 두 카드 배치를 거부한다(`body.rs:127-135`)"이다. 옮긴 검사마다 FAIL-first를 한다(load-v41의 기존 변이 그대로).
- **위험**: 두 카드 업로드 경로를 도는 유일한 시험이 사라진다. 다만 그 경로를 쓰는 엔진 길이 없다. 두 카드 엔진이 착륙하면 그 게이트와 함께 돌아온다. 주인은 aa다.

**GD6 — 엔진이 남긴 팔의 게이트 행렬은 손으로 적는 `@ENV` 항목뿐이다: 정착한 팔은 지우고, 남길 팔은 in-process 케이스로**

- **위치**: `gate_deepseek41_prefill.rs:107-111`(팔 계약), justfile(팔 0건), `gate-gpu-ds41-prefill-q3ksplit`(`justfile:936`, 347 s). AGENTS 스스로 "두 번째 경로는 제 게이트가 없으면 썩는다"고 적고 있다.
- **판정 상태**:
  - `slot`: "기본과 slot은 갈리지 않는다"(rig-log 09-25#v41-prefill-s14). 그 뒤 tile이 기본이 됐다(`237321a`).
  - `expert`: tile이 1.245/1.167로 이겼다(09-26#cardtile-ab).
  - G(`BLOOMERY_PREFILL_GROUP`)는 1–8의 매개변수이고 흐름 모형이 G 8을 예측한다(`3693dd9`).
  - `CED=off`는 흐름 모형 보정 팔이다(`4786c1e` 메시지).
  - q3ksplit은 예측 −60…−150 µs/스텝, 0.26–0.64 %(`3bf2e2e`)다. 13바퀴의 ±0.5 % 자에 걸친다[유도, AGENTS 자]. 시팅은 09-25부터 대기 중이고(triage `:122`), 접기 순서가 long free 궤적과 d1 동률을 옮겼다.
- **오늘 짓는다면**:
  - 정착한 팔(`slot`, 곧 `expert`)은 엔진과 함께 지운다.
  - 매개변수(G)와 계측기 팔(CED off)은 GD3 이후 같은 프로세스 안의 스모크 부분집합 케이스로 매 묶음 돈다. 팔 하나당 `--cases 512 --no-split` 비용만 든다.
  - q3ksplit은 다음 시팅 창에서 판정하거나 지운다.
- **이득**: 착륙마다 347 s(q3ksplit), 커널 둘, ptx-shapes 핀 둘이 준다. "must pass"라는 문장이 실제 커버리지가 된다.
- **비용·증명**: S(GD3 뒤)다. 삭제는 이동 클래스다(남는 팔의 비트는 그대로, prefill 게이트 녹색).
- **위험·주인**: 엔진 레버 삭제는 ds41·gpucore 감사 범위다. 게이트 쪽 항목은 aa다.

**GD7 — B4 op 게이트 8개(12,428줄)가 각자 하니스를 갖는다: test-backend-ops식 표 하나로**

- **위치**: `gate_deepseek41_{attn,comp,engram,hc,index,moe,rope,woa}.rs`. 셋 다 "B4 gate form"(커널 대 우리 규칙의 전사, ik 규칙 시뮬 대 덤프, 커널 대 덤프)이고, 합쳐서 157 s다.
- **쌓인 경위**: B4 파동에서 op마다 한 라운드였다. `499bced`가 세트·매니페스트 리더를 합쳤지만 사이트 순회·재실행 비트 검사·센티널 캐시·인자 해석은 bin마다 남았다.
- **오늘 짓는다면**: ik_llama.cpp `tests/test-backend-ops.cpp:300`의 `struct test_case`와 `build_graph`(`:311`) 모양을 따른다. 그 파일은 2,626줄 한 bin에 case 34개다.
  - `OpCase{ sets, load, run, ours, ik, bands }` 표와 러너 하나를 둔다.
  - 2층("ik 규칙 시뮬 대 덤프")은 덤프와 호스트 시뮬에만 의존하므로 디바이스 크레이트를 링크하지 않는 호스트 시험으로 뗀다. 그러면 원장이 커널 변경 때 건너뛸 수 있다.
- **이득**: 코드가 준다. 줄 수 감소폭은 가설이고 재지 않았다. 시간 이득은 작다(157 s 중 일부).
- **비용·증명**: L이다. 판정 줄 동일(verdict-diff)로 증명하고, 밴드 상수는 그대로 옮긴다.
- **위험·순서**: 맨 마지막(GD2 뒤)이다. 주인은 aa다.

### 3. 삭제

1. **`gate_deepseek41_attn.rs:168` `T1_SINK_DEFECT_BUILDS`와 결함 경로**(`:187-242` 필드 doc, `:494-498`, `:556-563`, `:1066-1089` `ik_index_sinks`, 약 50줄).
   - 근거: `527d84c`(09-23)가 오라클을 `db517b69`로 다시 떴다. skew는 다른 빌드를 거부하고(`skew.rs:99`, `:213-224`), 표의 시험은 스텝 세트가 배치 세트의 빌드를 공유할 것을 요구한다(`oracle/deepseek41.rs:177-219`).
   - 가설: 박스 세트 목록은 읽지 못했다. 커버리지 변화는 없다(살아 있는 데이터를 읽지 않는다).
2. **`engine.rs`의 `AnyEngine::Deepseek41` 팔**: 두 호출자가 모두 거부한다(위 1-2). ledger `:952`에 "컴파일만"으로 적혀 있다. 지우면 lib가 디바이스 크레이트를 부르지 않는다는 문장이 참이 된다. 커버리지 변화는 없다.
3. **`bench_v41.rs`(1,746줄), `bench-gpu-v41-check`, `time-gpu-v41`, `tools/gpu-ab.py:49`의 INSTRUMENTS 항목.**
   - 근거: rig-log의 마지막 언급은 2026-09-23이고 gate-times 행이 없다. 배치 상수가 옛 믿음이다(`:84-91`의 `N_SLOTS` 6, `EXPERT_LO` 2). 실제 스텝의 커널 표는 `nsys-gpu-ds41`이 낸다.
   - 단 `c_node`(plan.md:70, `time-gpu-v41`의 빈 `touch` 그래프)는 작은 프로브로 남겨야 한다.
4. **step `--greedy`**(`gate_deepseek41_step.rs:2563-2654`, 약 90줄)와 **`run-ds41-greedy`**.
   - 근거: long `--free`가 같은 `GREEDY_MARGIN` 규칙을 더 긴 실행에서 판정한다(`long.rs:515-530`). `--greedy`만의 출력(allocator·fault)은 출력만 할 뿐이고(AGENTS의 "no print-only tests"), fault는 `--faults`가 핀한다.
5. **`gate-gpu-ds41-prefill-q3ksplit`과 레버·커널**: GD6의 조건을 따른다(판정 또는 삭제).
6. **`BLOOMERY_CARD_EXPERTS=slot`과 prefill doc `:107-108`의 slot 언급**: GD6(엔진 쪽은 ds41 감사).
7. **`gate-gpu-load-v41`의 배치 (b)**: GD5, 날짜 붙은 사유와 함께.
8. **확신이 낮은 후보**:
   - `kld_diff.rs`(462줄) + `kld-diff`: ubsplit 질문용으로 만들었다(`5467188`, ledger `:429`에서 09-23 15:10 닫힘). 러너가 없고 rig-log 언급도 없다. KLD base를 다시 뜰 때 쓸 수도 있으니 리드가 판단한다.
   - `forced_probe.rs`(334줄) + `forced-probe`: V2-Lite MMA 진단이고 마지막 rig-log는 09-22다. V2-Lite를 어떻게 할지에 따른다.

### 4. 다른 곳에도 있을 패턴

- **"라이브러리는 디바이스 크레이트를 부르지 않는다" → `#[path]`와 복사.** model/tests(`#[path]` 대상 분할)와 qwen3 게이트(`src/qwen3moe.rs` 381줄이 lib에 있다)에도 같을 가능성이 크다 [03].
- **프로세스 env 레버 → 팔마다 프로세스.** 그 결과 serve는 요청마다 프리필·드래프트를 고를 수 없다.
- **텍스트 규약.** 러너가 정규식으로 줄을 읽고, 표준 빨강을 FAIL 줄 md5로 손 비교한다. 판정을 `Check{id, value, bound, pass}` 레코드로 내면 묶음 러너가 "알려진 빨강"을 기계로 대조할 수 있다.
- **파일 정체성.** 두 V4.1 파일의 샤드 이름이 같아서 basename을 쓰는 헤더는 전부 모호하다(`ik-greedy.sh:104`).
- **레시피 안의 셸 프로그램**: 56/195개. DSpark export는 10벌이다.
- **출처를 가리키는 주석은 R14 위반의 자백이다**: "the step gate's …"류 12개. grep으로 찾을 수 있다.
- **ik 빌드 id가 bin에 5벌** 있다. 계열 표 한 곳에 있어야 한다.

### 5. 역사처럼 보이지만 남는 것

- **`BLOOMERY_PREFILL=steps`**: `nsys-ds41.sh:30`의 디코드 형태가 쓴다.
- **`exact_ref`**: `gate_e2e` forced_exact 핀의 참값 파일을 짓는 V2-Lite f64 심판이다(AGENTS가 이름을 댄 정확도 핀). V2-Lite가 남는 동안 남는다. V4.1에는 f64 심판이 없고, 그 몫은 op 게이트의 규칙 전사가 나눠 진다.
- **`gate_deepseek41_plan`(2 s)**: 호스트 계획 정수 = ik 그래프 입력을 증명하는 유일한 검사다. load의 (iii)과 계약이 다르다.
- **`gate-gpu-ds41-faults`의 단독 프로세스와 solo 등급**: 호스트 메모리 핀이라 공유 프로세스에 넣으면 안 된다(gates-plan 3.3).
- **ds41-load의 두 번 적재**: 재적재 회계가 계약이다.
- **op 게이트의 3층 형식**: 회귀를 op 하나로 좁힌다(GD7은 모양만 바꾼다).
- **`BLOOMERY_STEP_PAIR=1`**: 09-25 패스 가격 측정에 썼다(rig-log 09-25 `:183`). bench 팔로 옮긴다.
- **G 매개변수와 `CED=off`**: 흐름 모형이 쓴다.
- **`_plain` 접미사 규칙**: 일반화할 대상이지 지울 대상이 아니다.
- **`kld.rs`**: step `--ppl`(09-24에 사용)과 `gate-ds41-kld`가 읽는다.
- **`gate_dspark_read`(11 s)**: MXFP4 디코드 핀이다.

### 6. 오늘 이 범위를 다시 짓는다면

```
crates/refset/      호스트 전용: 매니페스트 리더 셋(ik v1/v2, dsref, vision), 계열 표(생성 레시피·파일 신원·ik 빌드·소비자), RefError
crates/gates-core/  (지금 lib에서 V4.1·제품을 뺀 것) Check 레코드와 판정 줄, exit_with, KERNEL_BAND, ref_gemv, ptx, nodes, ik_q8_2, act_rule
crates/gates-ds41/  lib → gpu-deepseek41: Harness(Session 하나, 세트들), SetState, Envelope, FiniteProbe, shadow 표, 폴트 심기
                    bin gates-ds41 {ops|chain|body|load|dspark}, gates-ds41-faults(solo)
gpu-deepseek41      Session{open(placement, ctx, Config), prompt, step, pair, rollback, draft, stats}; Config는 값(env 없음)
crates/cli/         bin bloomery {run|chat|serve|bench}; env·플래그 → Config는 여기서 한 번; bench는 버전 붙은 레코드
```

**착륙 묶음의 흐름**: A 차선은 `body`(gate 배치, 적재 1회: 팔과 제품 토큰 검사 포함), `dspark`(예산 배치 1회), `load`(2회), `ops`·`chain`(부분 읽기). X 차선은 faults다.

**이전 순서.** 각 단계에서 게이트는 녹색을 유지한다.

1. 시팅 10(승인됨) → `refset`과 출처 거부. 게이트 동작 변경이며 FAIL-first로 증명한다.
2. 박물관 삭제: 결함 빌드 경로, `AnyEngine` V4.1 팔, step `--greedy`, `bench_v41`(`c_node` 제외). verdict-diff와 ptx-scan 동일로 증명한다.
3. `gates-ds41` 크레이트로 이동하고 사본을 lib로 올린다. 이동 클래스다.
4. 레버를 값으로 바꾼다(ds41 감사와 조율). 이동 클래스이며 노드 수와 eager = replay로 증명한다.
5. 배치마다 프로세스 하나(gates-plan §3.5를 이 모양으로). verdict 줄 동일과 순서를 뒤집은 1회 실행으로 증명한다.
6. `Session`과 `bloomery` CLI. 토큰 동일, prefill 계약, 러너 dry-run으로 증명한다.
7. `slot`·q3ksplit 처분.
8. op 표. 판정 줄 동일로 증명한다.

### 7. 범위 밖에서 본 개선 자리

- **`crates/gpu-gates/src/bin/generate.rs:287-303`, `gate_e2e.rs:744-755`**: `AnyEngine::open`이 모델을 적재한 **뒤에** 다른 아키텍처를 거부한다. 적재 전에 `expect_arch`를 하고 Deepseek2만 열면 된다(S).
- **env 레버가 디바이스 크레이트 안에서 읽힌다**: `crates/gpu-deepseek41/src/body/prefill.rs:113-125,342-359`, `chain/ffn/batch.rs:1434-1449`, `body/ced.rs:210-220`, `crates/gpu/src/lib.rs:247`. 해석은 CLI 경계로 올려야 한다(M, ds41·gpucore 감사).
- **`crates/model/tests/{common/manifest.rs, common/v41set.rs, ds41_host.rs:117, ds41_meta.rs:587, qwen3moe_meta.rs:59}`**: 매니페스트 리더 5벌을 `refset`으로(M, ledger `:896`).
- **`crates/serve/src/bin/bloomery-serve.rs`**: mock 전용 서버 bin이다. `bloomery serve`의 엔진 선택 하나로 합칠 수 있다(S–M).
- **`tools/ref/ik-greedy.sh:104`**: `basename`을 전체 경로로(XS).
- **justfile의 DSPARK export 10벌**: `box.sh:58-66`처럼 한 번 export하면 된다(XS).
- **`gate_deepseek41_load.rs:730`**: `step_of`가 MANIFEST를 다시 해석한다. `RefHeader`로 바꾼다(XS, ledger `:899`).
