# 다시 짓는다면 — 사후 설계 감사와 재구성 계획

2026-09-26. 같은 날 사용자 지시 넷에서 나온 문서다.
- 감사: "처음부터 지금 아는 것을 알았더라면 이런 구조로 만들지 않았을 부분 … 이 관점에서 코드 감사"
- 이행: "조사한 것을 토대로 처음부터 지금 스펙을 목표로 설계를 잡았다면이라는 가정을 잡고 모두 리팩토링"
- 확장: "애당초 설계할때 qwen뿐 아니라 glm 5.3 flash나 기타 최신 모델이 들어올 설계를 미리 잡아야해"
- 변형: "qwen도 여러 바리에이션 지원할 수 있어야 하고"

읽기 전용 감사 라운드 일곱이 `4786c1e`를 읽었다(일부는 `fa61362`까지). 범위는 이렇게 나눴다.

| 범위 | ID | 보고서 |
|---|---|---|
| gpu 코어 | GC | `docs/research/audit/gpucore.md` |
| V4.1 엔진 | DS | `docs/research/audit/ds41.md` |
| V4.1 게이트 | GD | `docs/research/audit/gates-ds41.md` |
| 공통 게이트 | GG | `docs/research/audit/gates-core.md` |
| CPU 크레이트 | CPU | `docs/research/audit/cpu.md` |
| 도구·문서 | TL | `docs/research/audit/tools.md` |
| Qwen3 [03] | Q3 | `docs/research/audit/qwen3.md` |

이 문서는 설계 문서다. 항목마다 근거 줄과 수치는 보고서에 있고, 여기에는 판단과 순서만 적는다. 숫자는 보고서가 읽은 값이고, 계산한 값에는 [유도]를 붙였다. 리드는 핵심 주장 일곱을 소스와 rig-log에서 직접 다시 확인했다(§8).

## 1. 지금 스펙 — 무엇을 위해 짓는가

- **지금 도는 모델 둘, 그리고 다음 모델들.** 다음 모델(GLM-5.3 Flash와 최신 공개 모델, Qwen 계열의 변형들)이 들어올 자리가 목표 모양의 첫 축이다(§2-1).
  - **DeepSeek V4.1 Flash**(공개 GGUF): 카드에는 dense 그래뉼과 층별 expert 접두(또는 hot list)를 두고, 호스트 RAM 264 GB에 expert 약 214 GB를 두고 CPU(AVX2 `qdot`, 고정 풀)가 계산한다. 초안(DSpark)은 둘째 카드에서 돈다.
  - **Qwen3 MoE**: 카드에 통째로 올라가고, 프리필은 GEMM 우배치(최대 4096)다.
- **두 헤드라인.** 디코드 tok/s(깊이를 적는다)와 프리필 pp512·pp4096, 둘 다 A6000에서 잰다. 공개 비교는 llama.cpp와 mistral.rs다.
- **계약.**
  - 성능이 먼저이고, 엔진은 선택마다 경로 하나만 싣는다. 더 정확한 변형은 검증 쪽에만 둔다.
  - 조용한 실패는 없다. 정의되지 않은 입력은 이름 붙은 오류가 되고, 커널은 fault word를 올린다.
- **자원 시간선(지금).**
  - 디코드: 산문, 깊이 512, A6000에서 토큰당 22.3 ms[유도, 44.8 tok/s]. 층마다 카드 → 호스트 서비스 → 카드 합류가 이어지고, 벽은 호스트 다리가 정한다.
  - 프롬프트(G 뒤, 층-배치당): lcg는 호스트 union 69 / 74 ms로 호스트 바운드, 산문(hot list 384)은 카드 바운드.

**이 스펙에 없는 것**: V2-Lite(디딤돌 모델), CPU 전용 디코드 엔진(`bloomery-decode`), stage 0(q3k-gemv·q3k-cpu·gpu-spike). 셋 다 게이트·레버·러너·계약을 싣고 있다(§3의 A).

## 2. 오늘 짓는다면 — 목표 모양

```
crates/gguf        Split(샤드 1..n) · 타입 기술자 하나(블록 기하, 활성값 형식, k 단위)
crates/qdot        (타입, 형식)별 AVX2 커널 — 디스패치는 기술자를 읽는 망라형 match
crates/threads     풀 + 친화성 전부 + PIN_MAIN 파싱 하나
crates/levers      레버 등록부(이름·종류·기본값·파서·문서), 타입 설정(OpenCfg·HostCfg·PrefillCfg), --levers
crates/model       GPU 엔진이 싣는 것만: arch 메타, placement(본체가 신고한 카드 항까지 포함),
                   host/(HostLayer, 호출 하나 = UnionCall — 디코드는 한 열)
crates/gpu         ctx(카드 하나) · error(거부/오염/드라이버/OS/복구) · fault(판독 주인 하나)
                   · graph(Graphs<K> 캐시, HostFlags 할당기 하나) · weights(KqRows<F>, 입구 하나)
                   · launch(가족별 모양 선택기) · kernels · host(HostTier + StepPort + BatchPort)
                   · model(GpuModel: 카드 하나·상주·Option<HostTier>, 작은 ChainBody + 능력 트레이트)
crates/ops         모델을 모르는 연산 라이브러리: 어텐션 계열(MLA·GQA·슬라이딩·선형/GDN·희소 인덱서),
                   norm·rope 변형, MoE 라우터·결합, 양자화 형식별 gemv/GEMM — 상수는 인자, 타일 크기는 단형 표
crates/runtime     seq/(SeqState: 위치·롤백 주인) · sched/(decode m=1 캡처, verify m≤8 캡처,
                   prompt m≤512 eager + CED + G — 겹침 부품 하나) · exchange/ · 추측 디코딩 원천(MTP·초안·n-gram)
crates/models      ModelSpec(층마다 층 종류) + 계열마다 짧은 층 프로그램: deepseek(V2 … V4.1 변형),
                   qwen(dense·MoE·Next·3.5/3.6 변형), glm, …; 역할 이름표; 적재 때 커버리지 검사
crates/app         Session(open·prompt·step·rows·rollback·draft·stats) + bin bloomery {run|chat|serve|bench}
crates/serve       HTTP + 생성 루프 하나 + sampler 하나; 모의 엔진은 게이트 픽스처
crates/refset      참조 세트 판독기 하나, 계열 표(생성 레시피·파일 신원·ik 빌드), RefError::Stale
crates/gates       연산 계열 게이트(모델 파일 없이, 각 모델의 차원 집합으로) + 모델마다 e2e 하나
                   (배치(placement)마다 프로세스 하나, 팔은 프로세스 안 케이스)
gates.toml         게이트 등록부(한 행 = 게이트 하나; card·solo·profile·arms·tier)
profiles.toml      모델 경로·프로필의 주인 하나
tools/bloomery/    records · manifest · affected · batch · sit(측정 러너 하나) · flow
```

**흐름.**
- 레버와 경로는 바이너리 가장자리에서 한 번 해석하고, 설정 값으로 한 번 적재한다. 그 뒤 케이스·팔을 여럿 돌리고, 기록 줄을 낸다. 도구는 기록만 읽는다.
- 디코드는 지금과 같다: 캡처된 스텝 하나와 `StepPort`. 프롬프트는 같은 층 프로그램을 배치 폭으로 돌리고, 교환은 `BatchPort`가 맡는다.
- 참조 엔진과 같은 자리에서 가른다: 8열 이하 gemv / 그 위 GEMM(ik `mmvq.cuh:10`), 층 몸체는 하나(llama.cpp `deepseek41.cpp`, mistral.rs `deepseek2.rs:951`).

**V2-Lite의 자리.** 전용 코드를 `arch/deepseek2`(GPU)와 잎 크레이트 `crates/cpu`(CPU 엔진)에 가둔다. 은퇴는 대체 게이트가 생긴 뒤 디렉터리 하나의 결정이다(§7 결정 1).

`crates/gpu`와 `ops`·`runtime`·`models`의 경계, 크레이트 이름은 modelvocab 표를 받은 뒤 확정한다. 위 목록은 주인과 흐름을 정하는 초안이다.

### 2-1. 다음 모델이 들어올 자리 (사용자, 09-26)

사용자 요구 두 줄이 목표 모양의 첫 축이다.
- "애당초 설계할때 qwen뿐 아니라 glm 5.3 flash나 기타 최신 모델이 들어올 설계를 미리 잡아야해"
- "qwen도 여러 바리에이션 지원할 수 있어야 하고"

지금 트리는 모델 둘에 맞춰 자랐다.
- Qwen3 형상 커널이 `qwen3moe_*` 이름으로 `HEAD 128`·`GROUP 8`·`N_EXPERT 128`·`NORM_K 2048`을 박고 있다. 다음 Qwen(qwen35moe: GQA 16/2 × 256, GDN 30층)과 이미 맞지 않는다(Q3 §1-11).
- `dflash_router`는 모델 상수가 컴파일 시간 크기라서 생긴 `ds41_router`의 본문 사본이다(DS §4).
- 모델마다 층 오케스트레이션을 처음부터 짓는다. V4.1만 1.5만 줄이다(DS6).
- `ChainBody`는 모델마다 거부 기본값을 더했다(GC2).

오늘이라면 다섯 원칙으로 짓는다. 연산 어휘와 대상 목록은 조사 라운드 `modelvocab`의 표로 확정한다.

1. **연산 라이브러리는 모델을 모른다.** 커널은 연산과 모양 계열로 부른다. 모델 상수(헤드 차원, GQA 그룹, expert 수, top-k, hidden)는 인자로 넘기고, 레지스터 타일처럼 컴파일 시간 크기가 필요한 것만 단형 변형 표에서 적재 때 고른다. 어느 커널이 어느 쪽인지는 03의 `kernelshape`가 가린다.
2. **모델은 서술이다.** 타입 있는 `ModelSpec`이 층마다 층 종류(full/sliding/linear 어텐션, MLA·GQA, dense/MoE, MTP)와 변형 매개변수를 갖고, 역할 이름표가 GGUF 텐서를 역할에 잇는다. 한 계열의 변형(Qwen의 dense·MoE 크기·Next/GDN·3.5/3.6)은 모델 타입 하나의 매개변수와 층 종류로 나타낸다. 변형마다 크레이트를 두지 않는다.
3. **층 프로그램은 짧고 하나다.** 새 모델이 쓰는 코드는 연산을 조합한 층 프로그램 하나다(llama.cpp `src/models/<arch>.cpp`의 자리). 디코드·검증·프롬프트 스케줄, 호스트 티어 교환, 배치(placement), 그래프 캡처, 추측 디코딩(MTP 층, 별도 초안, n-gram)은 런타임이 모든 모델에 준다.
4. **적재 때 커버리지를 검사한다.** 파일이 요구하는 연산·양자화 타입·층 종류 가운데 모르는 것이 있으면 이름 붙은 거부로 끝낸다(조용한 실패 없음).
5. **게이트는 연산 단위다.** 연산 계열 게이트는 모델 파일 없이 여러 모양(각 모델의 차원 집합)을 돈다. 모델마다 필요한 것은 e2e와 참조 세트(`refset`) 하나다.

## 3. 지금 모양이 오늘 설계와 갈라진 곳 — 부류별

일곱 보고서의 발견 51개가 아래 열 부류로 모인다. 부류 하나가 여러 범위에서 되풀이된다는 것 자체가 발견이다. 라운드마다 레버·팔·특례·게이트·러너 옵션을 하나씩 더하는 방식이 이 모양을 만들었다.

### A. 디딤돌이 아직 기본 자리에 있다

| 발견 | 지금 | 오늘 |
|---|---|---|
| GG1 | 커널 게이트가 V2-Lite를 띄울 때의 꾸러미 번호(P0b–P10)로 짜여 있다. 공유 커널의 새 계약(q8_1 거부 `6123133`, f16 전수 `031b842`, fault 사이트마스크 `7928a0f`)이 V2-Lite 게이트에 절로 얹힌다 | 모델 파일 없는 커널 계열 게이트(`kquant`, `elem`) — 선례 `gate_iq`, mistral.rs `grouped_mmq_packed_cuda_tests.rs` |
| GC7 | V2-Lite 전용 코드 6,577줄[유도]과 계기(`StepProbe`, taps, `probe_cfg` 분기 8, `tick` 43)가 공용 모듈(`flash.rs` 등)에 섞여 있다 | V2-Lite 전용은 `arch/deepseek2` 아래로. 은퇴는 디렉터리 하나의 결정 |
| CPU1 | `crates/model`은 GPU 엔진의 메타·배치·호스트 티어 크레이트가 됐는데 CPU 엔진(약 4,600줄[유도])이 아직 그 안에 있다. CPU 전용 편집 하나가 게이트 58개 중 54개를 고른다(`9c8ebf0`으로 실측) | CPU 엔진은 잎 크레이트 `crates/cpu`. GPU 크레이트는 거기에 의존하지 않는다 |
| CPU2 | stage 0(q3k-gemv, q3k-cpu, gpu-spike; 2,654줄)을 `just gate`가 짓는다. 엔진은 import하지 않는다. lint 144 중 58이 여기 있다[유도] | 없앤다. gpu-spike의 고유 검사 둘은 `crates/gpu` hw 시험으로 |
| CPU3·GG6 | 단언하는 할당 래칫(`LIMIT = 660`), 스레드 불변, 프로파일러, 디스패치 A/B(`ab-decode`)가 모두 V2-Lite CPU 경로에 있다. 제품 스텝은 할당 수를 찍기만 한다 | 래칫과 계기를 제품 스텝과 호스트 티어에 |
| CPU4 | V2-Lite만 옛 API(`&Gguf` + `TensorInfo`, 메타 리더 둘)로 호스트 티어를 탄다 | `Split` + `HostLayer` + 메타 리더 하나 |
| TL-6 | 기본 프로필이 `deepseek2`(두 곳)이고, `BLOOMERY_MODEL`이 도구에선 프로필 이름, 시험에선 파일 경로다. 기본 경로 사본이 30줄 | `profiles.toml` 하나, 기본값 없음 |

### B. 한 층을 두 번 짓고, 커널의 8이 API로 샜다

| 발견 | 지금 | 오늘 |
|---|---|---|
| DS6 | V4.1 디코드 체인(8,274줄)과 프롬프트 체인(6,738줄)이 같은 층을 따로 짓는다. 이미 갈라진 곳: joined Q3_K만 받는 배치, 교환 둘, observer 둘 | 층 프로그램 하나를 m토큰 블록 위에서. 스케줄 셋: decode(m=1 캡처), verify(m≤8 캡처), prompt(m≤512 eager + CED + G). pair와 G는 같은 겹침 부품의 두 매개변수 |
| DS2 | `HC_MAX_TOKENS = 8`, `Q8Act::with_k` 1..=8, dense m 1..=8, 8토큰 이미지 거부 — 넷이 API 상한이 되어 프롬프트가 "청크마다 도는 디코드 스텝"으로 남았다. 청크별 카드 작업은 층-배치당 ≤ 6.9 ms[유도, 상한]이고 산문에서는 임계 경로에 있다 | 단위는 배치 T, 8은 커널 안의 타일 폭. 토큰별 연산은 T 위 launch 하나 |
| Q3-3 | m행 디코드 패스의 주인이 둘이다: Qwen3 `Prefill` 패스와 골격의 `step_pair`(m = 2 고정). `Head::with_m`은 셋째 조각이다 | 골격의 `step_rows(m)`, m마다 그래프 하나. V4.1 pair = m 2 |
| Q3-5·GC6 | `Q8Act`가 m ≤ 8을 할당에 굳히고 세 순열을 다 쓴다(P 4096 Qwen3에서 읽지 않는 쓰기 약 5.6 GB[유도]). 가중치 형식을 적재 때 타입으로 굳히지 않아 행 바이트 식이 64곳이다 | 활성값 `Act{planes, cap, k}`, 가중치 `KqRows<F>`, 가족별 모양 선택기(exllamav3 `select_gemm_shape`) |
| GC8 | V4.1 프롬프트는 "스텝과 비트 동일" 계약이라 m ≤ 8 코어 위에 섰고, Qwen3는 GEMM + 밴드다. 모델마다 프리필 계약이 다르다 | 공통 계약 하나(결정 5) |

### C. 타입이 옛 믿음을 담고 있다

| 발견 | 지금 | 오늘 |
|---|---|---|
| GC1 | `GpuModel.stages: Vec<Stage>`. 여러 스테이지는 한 번도 두 카드가 아니었다(`load_staged` 호출자 0). 거부 줄 26개, `Option<Residency>`, 그래프 주인 다섯 | `GpuModel { gpu, weights, body, head, host: Option<HostTier>, graphs: Graphs<Chain>, … }`. 상주하지 않은 모델은 표현할 수 없게 |
| GC2 | `ChainBody`가 11메서드 설계에서 20메서드로 컸고, 7개는 "한 본체만 구현, 나머지 거부" 기본값이다. 엔진 표면이 셋이다(`gpu::Engine`, `serve::Engine`, `AnyEngine`) | 작은 핵심 트레이트 + 능력 트레이트(`HostServed`, `Rows`, `Rollback`, `Instrumented`). 런타임 거부가 컴파일 오류가 된다 |
| DS3 | V4.1 `Body` 한 타입에 필드 약 30, `pub fn` 48(게이트 전용 약 20). restore 루프 둘, 위치 주인 둘, `top_k` 둘 | `SeqState`(위치·롤백 주인, 카드 없이 시험) + `KvCache` + `Inspect`(feature) |
| GC4 | `GpuError::Shape`가 만능(생성 587곳). 호출자가 가르는 것은 `Fault`·`Poisoned`뿐이고, 게이트 7곳이 문구를 대조한다 | 가르는 축 셋: 호출 거부 / 모델 오염 / 카드·컨텍스트 소실. 셋째만 서버 재시작 |

### D. 한 사실에 주인이 여럿이다

| 사실 | 주인들 | 발견 |
|---|---|---|
| 카드↔호스트 교환 | 스텝용 플래그·memop(hybrid.rs [03]) / 배치용 세트·이벤트·상태 기계(`ffn/batch.rs`) | GC3 → `HostTier` 아래 `StepPort`·`BatchPort` |
| fault 판독, 매핑 메모리, 캡처 중 여부 | 각각 둘 | GC3 |
| 카드 바이트 | 계획 산술(`SCRATCH 64 MiB [assumed]`) / 실제 할당(첫 프롬프트의 배치 버퍼 238 MB, fault word …). 적재 게이트 잔차로만 대조 | GC5 → 본체가 자기 카드 항을 계획에 신고(mistral.rs `DeviceMappedModelLoader`) |
| launch 구조 | attention(enqueue에서 셈) / `queue` 식 / Python 흐름 모형·ds41pp | DS5 → 세는 스트림 래퍼 + `--plan` 덤프 |
| 출력 줄 스키마 | `SMOKE` 쓰는 곳 4·읽는 곳 8, `time prompt` 2·6 | TL-3 → `record.rs` 하나 |
| 컴파일 모양 | gate_p4·p5·p6의 런타임 단언 / `ptx-shapes.tsv` | GG2 → 래칫 표 하나 |
| 참조 매니페스트 | 판독기 13곳(세 크레이트) + Python 4 | GD1·GG5 → `crates/refset` |
| 타입 → 활성값 형식 규칙 | gguf 1 + qdot match 11(`_ =>` 팔이 조용히 Q3_K 규칙) | CPU7 → 기술자 하나, 망라형 match |
| 샘플러·생성 루프·UTF-8 디코더 | 2 · 2 · 3 | CPU5 |
| 친화성 코드, `PIN_MAIN` 파싱 | 4 · 5 | CPU §4 |
| 배치·청크 분할 규칙 | Rust 1 + Python 2 | TL §1 |

### E. 레버가 프로세스 env라서, 팔 하나가 곧 프로세스 하나다

- **TL-1**: 엔진 레버 약 40개가 쓰는 자리마다 `std::env::var`를 부른다. 파서 관용구가 11가지이고, 13개는 뜻 모를 값을 조용히 기본값으로 받는다. 레버가 프로세스 전역이라 시험이 두 팔을 한 프로세스에서 못 돈다(재실행 자식 7개). 레버를 **바이너리 가장자리에서 한 번 해석해 타입 값으로 생성자에 넘긴다**(ik `llama_context_params`, mistral.rs `NormalSpecificConfig`). env는 운반 수단으로 남는다.
- **GD3**: 게이트 하나 = 프로세스 하나 = V4.1 적재 한 번이다. V4.1 전체 목록 한 번이 적재 25회(한 번에 30–42 s)다. 팔이 env인 한 적재를 나눌 수 없다. TL-1 뒤에는 **배치(placement)마다 프로세스 하나**가 되고, 3090 레인 적재는 22회에서 4–6회로 준다(묶음당 480–756 s[유도]).
- **Q3-1**: 우산 레시피(`gate-gpu-qwen3moe-kernels`)가 개별 여섯을 다시 돌려 묶음마다 88–131 s[실측]를 쓴다. e2e는 스칼라 팔로 통째로 한 번 더 돈다.

### F. 제품이 게이트 크레이트에 산다

- **GD4·GG4·GC2**: `generate_ds41`(1,429줄, 피드 넷 × 디코드 넷), lib `Generator`("한 id에 스텝 하나" — 배치 이전의 믿음), 서버 바인딩 `bind.rs`가 드라이버 셋이다.
  - chat은 프롬프트를 토큰마다 디코드 스텝으로 먹인다(리드 확인: `bloomery_chat.rs:292` → `generate.rs:141` → `GpuModel::step`). 배치 프리필 `43cd107`은 스텝 피드의 2.97배였고, 그 뒤 빨라진 것은 배치 쪽뿐이다.
  - chat과 serve에는 드래프트가 없다. README의 DSpark 51.3 tok/s는 `generate_ds41`로만 나온다.
  - 오늘이라면: 엔진 쪽 `Session` 하나와 `bloomery {run|chat|serve|bench}`(mistral.rs `mistralrs-cli`).
- **CPU5**: serve가 요청의 `repeat_penalty` 등을 읽지도 거절하지도 않고, 거절된 샘플러 파라미터는 조용히 argmax로 떨어진다(`bind.rs:167-174`). `api.rs:477` 자신의 규칙("못 지키는 필드는 400")과 AGENTS "No silent failure"에 어긋난다.

### G. 도구가 엔진의 사실을 소스에서 다시 캐낸다

- **TL-4·GG7**: 게이트가 justfile의 셸 문자열이고, 세 프로그램(recipes.py 2,470줄, gate-batch.sh의 내장 파이썬, check-recipes.sh)이 거기서 구조를 캐낸다. 오분류 사례가 있다(ptx-spill을 "디바이스 코드 없음"으로). → `gates.toml` 등록부.
  - 오늘 faultstep 배치에서 원장 키가 레시피 매개변수로 받은 cargo 피처(`{{FEATURES}}`)를 버려, 코드가 바뀐 트리의 ptx-scan을 초록으로 건너뛰었다. 이 부류의 실측 사례다.
- **Q3-8·DS5·TL-3**: `q3pp.py`는 Rust 소스를 정규식으로 읽어 launch 순서를 다시 유도하고, 흐름 모형은 분할 규칙을 옮겨 적었다. 흐름 모형의 Rust 줄 인용 54곳 중 넷이 이미 틀렸다. → 엔진이 자기 계획을 기록으로 낸다.
- **TL-7·Q3-6**: 측정 러너 23개가 프로토콜 조각(가드, 신선도, dry-run, 바운드)을 골라 붙인다. `guard_cpu`는 depth-ds41에만 있다. depth 러너 쌍둥이는 172줄이 글자 그대로 같다. 팔 문법이 다섯 가지다. → 러너 하나 + 엔진 어댑터.

### H. 판정이 끝난 팔과 박물관 코드가 계속 값을 치른다

§4의 삭제 목록. 대표적인 것만 적는다.
- `CARD_EXPERTS=expert|slot`(tile이 1.245 / 1.167로 이김, #cardtile-ab)
- 스칼라 flash 세그먼트 패스와 탐침 8개(MMA가 기본이 된 09-22에 "두 번째 경로"로 남음)
- `StepProbe`의 split 팔 넷(09-22-b에서 판정)
- `STEP_PAIR=1`, `ATTN_HALVES=2`, `WEIGHTS`, `POPULATE`
- 박물관 bin: bench_join, h2d_probe, real_x, rawx_floor, markov-accept, qdot-rate-mt, pool-rate
- engram 실험실 1,877줄(DS4)

레버에 "판정 → 삭제 날짜" 칸이 없어서, 판정 뒤에도 "롤백용"으로 남았다(GC §4).

### I. 문서가 규칙이 아니라 상태와 역사를 싣는다

- **TL-2**: AGENTS.md 669줄 중 약 232줄이 상태·역사다. 모든 라운드가 53.5 KB를 싣는다.
  - 틀린 사실도 있다. 「Layout」에 gpu 크레이트 일곱(157k줄)이 없고, 「Never hand-run」은 stage-0 러너를 측정 주인으로 적고, "13 subsystem gates"(실제 95개)라고 한다.
- **TL-5**: 열린 일의 선언된 주인은 MUL인데 실제 주인은 `plan-triage.md`(255 KB, 취소선 141쌍, 09-25 커밋 147개 중 89개가 이 파일을 고침)다.
- 크레이트·모듈 머리말이 현재가 아니라 출생을 적는다: "The CPU engine", "stage-0", "P5", "(package P10)" 등. `check-comments.sh`는 날짜와 이슈 번호만 잡는다.

### J. 참조 데이터의 출처에 주인이 없다

- **GD1**: 참조 세트 다섯 계열 중 파일 신원을 확인하는 것은 ik 노드 덤프 하나뿐이다. greedy·KLD·dsref 셋은 혼합 파일에서 떴다.
- 그래서 `gate-gpu-dspark-graph`는 `f1d1168`(09-25) 뒤로 모든 착륙 묶음에서 FAIL 61–63줄의 **표준 빨강**이고, 리드가 md5를 손으로 비교한다. long `--free`와 step `--ppl`은 다른 파일의 참조와 조용히 비교한다.
- 재생성 시팅(시팅 10)은 승인됐지만 열리지 않았다. → `crates/refset`과 `RefError::Stale`, 그리고 시팅 10.

## 4. 삭제 목록

"아무도 안 부른다"는 보고서가 돌린 grep이 증거다. 게이트를 지우거나 팔을 빼는 것은 커버리지 변경이라 착륙 커밋에 날짜 붙은 이유를 단다. 커널을 지우면 ptx-scan 행이 빠지므로 증명 규칙을 먼저 넓힌다(§6-0 ①).

| 대상 | 근거 | 커버리지 변화 | 주인 |
|---|---|---|---|
| `CARD_EXPERTS=expert\|slot`, `Phase`, `CardExperts`, `act_h`, 1024 탈출구, shadow 행렬, `ds41_expert_gate_up_tok` | #cardtile-ab(tile/expert 1.245 ± 0.060 / 1.167 ± 0.011), #v41-prefill-s14(slot 갈리지 않음) | prefill 게이트의 두 팔 실행. grouped 커널은 게이트 기준이면 게이트 쪽으로 | aa |
| `Body::step_plan`, `Body::engram_layers` | grep: 정의뿐 | 없음 | aa |
| `STEP_PAIR=1` 팔(`generate_ds41`) | E12 답함(rig-log 09-24, 09-25) | 없음(`step_pair`는 DRAFT가 씀) | aa |
| `GpuModel::load`·`load_staged` | 호출자 0(grep rc 1) | 없음 | aa |
| `argmax_rows`, 비폴트 `argmax` | 엔진 호출 0 | gate_p4 절을 `_fault` 판으로 | aa |
| 스칼라 flash 세그먼트 패스·탐침 8·`TWICE`·`flash_merge2_q8`·`BLOOMERY_FLASH_MMA=0` | 09-22-n "두 번째 경로, 지울지는 다음 결정" | gate-gpu-e2e의 둘째 프로세스, keyaxis 팔 | aa |
| `StepProbe` split 팔 넷 | 09-22-b 판정(−31.9, −35.9 µs) | gate_e2e 중첩 팔 검사 | aa |
| `Head::graph`·`capture`·`launch` | 게이트만 씀 | 헤드 단독 eager=replay(스텝 그래프가 지킴) | aa |
| `join_probe.rs`, probe.rs `gap_*`·`two_phase`, bench_join(+cstate-ab 빌드), h2d_probe, real_x, rawx_floor | 결과는 rig-log에 있음 | 없음(게이트 아님) | aa |
| `bench_v41`(1,746줄) — `c_node` 프로브만 작게 남김 | 마지막 사용 09-23, 옛 배치 상수 | 없음 | aa |
| step `--greedy`, `run-ds41-greedy` | long `--free`가 같은 규칙을 더 길게 판정 | 없음 | aa |
| `AnyEngine::Deepseek41` 팔 | 두 호출자가 적재 뒤 거부 | 없음 | aa |
| `T1_SINK_DEFECT_BUILDS` 경로(약 50줄) | 오라클이 `db517b69`로 다시 떴다(`527d84c`) — 박스 세트 목록 확인 필요 | 없음 | aa |
| lib.rs 죽은 pub 함수 셋, `ds41_host.rs`의 `Set` 사본 | 외부 호출 0 / `v41set.rs`와 같음 | 없음 | aa / [03] 인접 |
| p0b·moe_fused의 `--time` 팔과 레시피 둘 | 판정이 `gpu-design.md:79, :94`에 | 없음 | aa |
| `gate_load_v41`의 plan (b) 기본값 | 엔진이 두 카드 배치를 거부(`body.rs:127-135`, 리드 확인) | solo 적재 두 번(약 329 s[유도])을 opt-in으로 | aa |
| engram 실험실(cache.rs, reuse.rs, `SeededRows`, `Context`, bin 둘; 1,877줄) | 엔진 import 0 | 실험실 시험과 `measure-engram`이 bench로 | aa |
| `markov-accept`, `qdot-rate-mt`, `pool-rate`, `governor-ab.sh`, `window-union.py` | 답함 / 러너 없음 / 이름 없음 | 없음 | aa |
| `ATTN_HALVES=2`, `WEIGHTS=anon\|huge`, `POPULATE=0` | 판정 끝(느림 / 무차이) | halves 2 시험 케이스 | aa(CPU 엔진) |
| tokenizer `hunyuan-dense` 별칭, `Sampler::params()` | 시험 없는 지원 주장 / 호출 0 | 없음 | aa |
| stage 0 전부(+ `just gate`의 `build-gpu`·`build-cpu`, `measure-gpu/-cpu`, `build-ref-bench`) | CPU2 | `just gate` 두 항목 | aa |
| `docs/research/errsrc/` 스크립트 사본 | `tools/` 사본과 md5 같음 | 없음 | aa |
| **[03]** 우산 레시피, `GQA_MMA` env(선택자는 API 하나로), `qwen3moe/host.rs`, `qwen3moe_router` 엔트리, `enqueue_combine` 래퍼, `RouterOut::new`, `set_defer_quant`, `STEAL_BLOCKS` 본문, `HYBRID_OVERLAP=0`, `bench_v41_host`의 판정 끝난 팔 | Q3·CPU·DS 보고 | 03이 판단 | [03] |

**결정 뒤에 지우는 것**
- `Q3K_SPLIT` 가족: 판정하든지 지우든지. 게이트가 3090 레인에서 묶음마다 337–357 s를 쓴다.
- `LAUNCH_THREAD`: 판정하든지 지우든지. 판정하려면 13바퀴 시팅이 필요하다.
- `gemm_q5k`: IMMA 결정 뒤.
- `kld_diff`, `forced_probe`: 확신이 낮다.
- V2-Lite 전부(결정 1).

## 5. 남기는 것 — 역사처럼 보이지만 살아 있다

- **쌍둥이 팔.** 게이트의 오라클이라 남긴다: `FLASH_SIMD=0`, `ATTN_BUNDLE=1`, `FLASH_SEGMENTS=1`, `DEFER_QUANT=0` [03], `GQA_MMA=0`(자로서; env 선택자만 뺀다). `CED=off`와 `PREFILL_GROUP=1`은 흐름 모형 보정 팔이다. `PrefillMode::Steps`는 prefill 게이트의 비트 기준이다.
- **살아 있는 방어.** box.sh의 동기화 제외·`-c`·`lock-back.sh`, 예측 카드(`card.py`, `lease_take`), `deny.toml`, `.cargo/config.toml` + `cuda-oxide.toml` + `check-rustflags`(cargo-oxide가 주인 둘을 강제한다; nvlabs-ledger §6).
- **계획된 대상.** `iq.rs`와 `gate_iq`(IQ 모델), `models/deepseek4.sh`와 V4 메타(V4-Flash가 다음), V4.1 fixture(작은 게이트용 방향), qdot IQ3_XXS·MXFP4.
- **판정이 아니라 설계.**
  - 디코드 flash(split-K)와 프리필 flash 둘: llama.cpp의 vec/mma 분업과 같은 이유다.
  - 8이라는 컷 자체(참조 셋 모두 8), `gemm.rs` 매크로 뼈대.
  - 단형 `GpuModel<B>`(토큰 경로에 `dyn` 없음), `Seam`(llama.cpp `cb_eval` 모양), `take_host_refusal`.
  - `ENGRAM_HELPER`, `RowsArrival`(구조 정정으로 남기기로 함), 호스트 rope 표(ik와 비트 정확).
- **오늘은 V2-Lite만 줄 수 있는 커버리지.** `gate-gpu-hybrid`(호스트 티어 = 전부 카드 비트 동일)와 `gate-gpu-e2e`(ik CUDA 토큰 대조). 대체가 생길 때까지 남는다.

## 6. 이행 계획

### 6-0. 재구성 전에 닫을 증명 규칙의 틈 셋

1. **삭제 클래스.** 좁힌 착륙 규칙은 "ptx-scan 표가 base와 같다"를 요구한다. 그런데 커널을 지우면 행이 빠진다. AGENTS 변경 클래스 표에 한 줄을 날짜와 함께 더한다: "지운 엔트리를 뺀 나머지 행과 digest가 base와 같다 + 지운 엔트리마다 엔진 호출 0의 grep". 첫 적용은 1파동이다.
2. **Δ ≈ 0인 구조 변경.** AGENTS는 디스패치 경로를 건드리면 같은 임대 A/B를 요구하고, `card.py`는 0을 품은 ab 대역을 거부한다. 그래서 "빨라지지 않고 같은 일을 다른 모양으로" 하는 재구성(CPU6, DS6의 디코드 쪽)은 지금 입증할 길이 없다(CPU §4-5). 두 길이 있다.
   - 셀 수 있는 구조 동일로 증명한다: 디스패치 수, 레인 경계, 캡처 노드 목록, ptx-scan 동일.
   - 동등성(한쪽 비열등) 카드 종류를 `card.py`에 들인다.
   - 1순위는 앞의 것이고, 뒤의 것은 필요해질 때 연다.
3. **원장 키.** 레시피 매개변수로 받는 cargo 대상·피처는 키 폐포에 없다. 그런 항목은 skip하지 않게(`never`) 하고, 자가 시험에 넣는다. boxlease 직후 리드 픽스업이다.

### 6-1. 파동

모델 확장성(§2-1, modelvocab 결과로 확정)을 기준으로 순서를 잡는다. 한 파동은 aa 라운드 넷 이하, 파일 경계는 겹치지 않게 하고, 착륙은 03과 직렬로 한다.

| 파동 | aa 라운드 | 증명 | 선행 |
|---|---|---|---|
| **1 삭제** | `ds41del`(DS1, 죽은 `Body` 메서드, `STEP_PAIR=1`) · `gatesdel`(박물관 bin, `AnyEngine` V4.1 팔, step `--greedy`, `T1_SINK`, dead lib fn, `--time` 팔, load-v41 plan b opt-in) · `engramlab`(DS4 + `map_token` 이름 붙은 오류 + 적재 때 `token_map` 검사) · `shapefacts`(GG2: 컴파일 모양 단언 → ptx-shapes 래칫 열) | 삭제 클래스(6-0 ①), 커버리지 변경마다 날짜 사유 | boxlease, 원장 키 픽스업 |
| **2 한 주인** | `levers`(TL-1, aa 레버) · `records`(TL-3·DS5: 기록 모듈, 엔진의 `--plan`, 세는 스트림 래퍼) · `gpumodel`(GC1 + GC2: 카드 하나 `GpuModel`, 작은 `ChainBody` + 능력 트레이트) · `refset`(GD1·GG5, 시팅 10과 짝) | move: ptx-scan 동일 + 구조 줄 / 호스트 전용 | 1파동 |
| **3 모델 서술과 연산 라이브러리** | `modelspec`(`ModelSpec`, 층마다 층 종류, 적재 때 커버리지 검사) · `opslib`(모델 이름 없는 커널 계열, 상수는 const 표 — 03 `kernelshape`와 짝) · `session`(GD4·GC2: `Session` + `bloomery` CLI, chat 배치 프리필·드래프트) · `v2fence`(GC7 2단계, CPU1 b: V2-Lite를 `arch/deepseek2`와 `crates/cpu`로) | move, 커버리지 검사는 FAIL-first | 2파동, modelvocab |
| **4 층 프로그램 하나** | `layerprog`(DS6: 층 프로그램 + decode/verify/prompt 스케줄, Q3-3 `step_rows`) · `batchwide`(DS2: 토큰별 연산을 배치 폭으로) · `gatesproc`(GD3: 배치마다 프로세스 하나) · `gatestoml`(TL-4·TL-6) | 디코드 move(노드 목록 동일), 프롬프트 launch 목록 동일은 `--plan` 덤프로, DS2는 산문 A/B 1회 | 3파동, r8host |
| **5 새 모델** | modelvocab 추천 대상 첫 모델을 층 프로그램 하나로 올린다 — 재구성이 맞았는지의 시험대 | 새 모델 e2e + 참조 세트 | 4파동 |

03 파동은 03의 계획을 따른다: `q3prune` → `q3input`·`q3gates` → `gemmsplit`·`q3act` → Q3-3 → 호스트 티어(`hostcfg` → `benchprune` → `hostone` → `hybridgate`) → `q3plan`, 그리고 `kernelshape` 설계 라운드. 공유 타입은 설계자와 서명자를 나눈다: `Act`(03 설계, aa 서명), `step_rows`(aa 설계, 03 서명), `HostTier`(03 설계, aa 서명).

리드 몫(라운드 밖): 원장 키 픽스업, AGENTS·CLAUDE를 규칙만으로(TL-2; 레버 절은 `levers` 뒤), 열린 일의 주인 정리(결정 2 뒤).

## 7. 사용자가 정할 것

1. **V2-Lite(와 stage 0, CPU 엔진)의 은퇴.** 추천은 격리를 먼저 하고, 은퇴는 대체 게이트가 생긴 뒤에 하는 것이다.
   - 오늘 V2-Lite만 주는 커버리지가 둘이다. `gate-gpu-hybrid`는 호스트 티어가 전부 카드에서 돌린 결과와 비트 동일함을 본다. `gate-gpu-e2e`는 ik CUDA와 33프롬프트 × 32스텝 토큰을 대조한다.
   - 대체 후보: V4.1 픽스처 부분 집합(606 MB, 전부 카드 가능) 위의 hybrid 계약, 그리고 새 대상 모델의 작은 변형으로 하는 참조 토큰 대조.
   - AGENTS "Performance first"가 f64 심판으로 부르는 `exact_ref`와 forced_exact 핀도 V2-Lite 전용이다. 은퇴 때 정확도 자를 ik 덤프 밴드 + KLD로 옮길지, 서빙 모델용 심판을 새로 지을지를 같이 정한다.
   - stage 0은 대체가 필요 없다. `gate-gpu-lib`가 이미 엔진 디바이스 크레이트를 짓고 시험한다. 1파동 뒤 바로 지울 수 있다(`oxide-ice-unroll`은 핀 이동 때).
2. **열린 일의 주인.** CLAUDE.md는 MUL을 지정하지만 실제 주인은 `plan-triage.md`다. 한 항목 한 행 표 파일로 바꿔 주인으로 삼을지, MUL로 옮길지 정한다.
3. **30분 넘는 착륙 배치.** 재구성이 `crates/gpu`를 건드리면 GPU 게이트가 전부 돈다(30–90분). 이번 재구성 동안 일괄 승인할지 정한다. 타이밍 시팅은 따로 묻는다.
4. **판정하든지 지우든지.**
   - `Q3K_SPLIT`: 예측이 자 아래라 13바퀴 시팅이 필요하다. 게이트가 착륙마다 5.6–6분을 쓴다.
   - `LAUNCH_THREAD`: 판정하려면 13바퀴가 필요하다.
   - 추천은 둘 다 지우고 아이디어는 트리아지에 남기는 것이다.
5. **프리필 계약(GC8).** 모델 공통 계약을 "프롬프트는 스텝과 밴드, 비트는 배치 크기와 무관"으로 둘지 정한다. 지금 V4.1은 비트 동일, Qwen3는 밴드다. 층 프로그램 하나(DS6)와 연산 라이브러리의 GEMM 분기가 이 결정에 닿는다. 값은 호스트 항이 줄어든 뒤에 생긴다.

## 8. 리드가 원본에서 다시 확인한 것

- chat이 프롬프트를 토큰마다 스텝으로 먹인다: `bloomery_chat.rs:292` → `generate.rs:138-144`(`self.model.step(ids)`)
- `AnyEngine`의 V4.1 팔은 두 호출자가 모두 거부한다(`gate_e2e.rs:744-752`, `bin/generate.rs:287-303`).
- `gate_load_v41`의 기본값은 `PlanId::B`("the serving target")인데, 엔진은 두 카드 배치를 거부한다(`body.rs:127-135`).
- Qwen3 우산 레시피는 개별 여섯을 다시 짓고 돈다(`justfile` `gate-gpu-qwen3moe-kernels`).
- `qwen3moe::host`를 부르는 곳은 없다(grep rc 1).
- `gate_p8`이 Qwen3 GEMM 벤치(`gemm_arms`)를 품고 있고, `ncu-gpu.sh`의 `GEMM_BIN` 기본값이 `gate_p8`이다.
- 할당 래칫은 V2-Lite CPU forward에만 걸려 있다(`tests/alloc.rs:13-14`, `LIMIT = 660`, PIN 다섯).
- **엇갈린 주장 하나를 가렸다.** `HYBRID_OVERLAP=0`을 ds41·gates-core 감사는 "판정 기록 없음"으로 적었다. 실제로는 rig-log 2026-09-23.md:353-354에 판정이 있다(n = 32, 켬 6.921 대 끔 8.288 ms). 두 감사의 grep이 한국어 "겹침"을 못 잡았다. 끔 팔은 `gate_hybrid.rs:1561`이 설정(`overlap: false`)으로 돈다. 지우면 커버리지 변경이고, 03이 판단한다.
- **감사 중에 새로 난 결함 하나.** faultstep 착륙 배치에서, 원장 키가 레시피 매개변수로 받은 cargo 피처를 버렸다(`recipes.py:748`). 그래서 코드가 바뀐 트리의 ptx-scan이 초록으로 skip됐다. G 부류의 실측 사례다(§6-0 ③).

## 9. 03에게 넘긴 것 [03]

Q3-1 … Q3-8 전부, 그리고 다음을 넘겼다(09-26 메시지).
- hybrid.rs의 교환 프로토콜 소유(GC3)
- `REPLAY` 인자화, `enum Chain`을 model.rs로
- `DEFER_QUANT`는 시험 전용 오라클로, `STEAL_BLOCKS`는 상수로, `HYBRID_OVERLAP=0`은 판정 있음(§8)
- CPU6(디코드 호스트 호출을 `UnionCall` 한 열로), CPU8(`bench_v41_host`의 판정 끝난 팔)
- `fault.rs` `Q5Quant`(V2-Lite 전용)

형상 커널을 크레이트 루트로 옮기고 상수를 인자·단형 표로 바꾸는 일은 사용자 요구(§2-1)에 맞춰 미루지 않기로 했다. 03의 `kernelshape` 설계 라운드가 커널마다 컴파일 시간 크기가 필요한지를 가린다.
