# 모델 축 — 아키텍처별 구현을 어디서 가르는가

2026-09-22 밤. 지금 트리는 모델이 하나(DeepSeek-V2-Lite, GGUF `general.architecture` = `deepseek2`)라
모델을 아는 코드와 모르는 코드가 한 파일에 섞여 있어도 아무것도 아프지 않았다. V4.1-Flash(`deepseek41`)가
들어오면 그 섞임이 값을 받는다 — B1·B2·B4·B5가 전부 `gpu/model.rs`·`gpu/model/dispatch.rs`·`gpu/weights.rs`·
`model/derived.rs`를 지나고, 그 파일들은 V2-Lite의 게이트 15개가 증명하는 파일이다. 나누지 않으면 V4.1 라운드마다
증명이 "V2-Lite 게이트 전부 초록"이 되고, 두 모델의 불변식이 한 함수 안에서 `if`로 갈린다.

이 문서는 **무엇을 어디에 두는가**를 정하고, 그 이동을 라운드로 자른다. V4.1 코드는 여기서 한 줄도 쓰지 않는다.
op 목록은 [`research/v41-ops.md`](research/v41-ops.md), 파일 내용은 [`v41-inventory.md`](v41-inventory.md),
GPU 경로의 결정 1~7은 [`gpu-design.md`](gpu-design.md)다. 이 문서는 그 결정들 위의 여덟째다.

## 원칙 셋

1. **타입이 다른 것은 아키텍처 타입으로, 값만 다른 것은 표로.** 두 모델 사이에 *타입*이 다른 것은 넷이다 —
   KV 토폴로지, 스텝 입력, 로드 시 계획, 층 사슬. 텐서 이름·메타데이터 키·전문가 수·양자화 타입은 *값*만 다르다.
   값은 표 한 장으로 끝내고(`plan.md` 규칙: "이름 표는 표로, arch마다 여섯 파일에 흩뿌리지 않는다"), 타입은
   아키텍처 모듈이 소유한다. 토큰 경로에 `dyn`은 두지 않는다(같은 규칙: "1단계 디스패치는 monomorphic").
2. **커널은 모델을 모른다.** 결정 6 그대로 — 형상(K, 행 수, 행 바이트)은 런치 인자이고 커널 파일은 형상의 이름을
   가진다. 모델을 아는 호스트 코드는 세 층뿐이다: **계획**(로드 시 — 어떤 텐서·어떤 키·어떤 파생 가중치),
   **사슬**(캡처 시 — 어떤 런치를 어떤 순서로), **탭**(게이트 — 오라클의 어떤 이름과 대조). 셋이 `arch/` 아래로 간다.
3. **이름은 파일이 말하는 문자열이다.** 모듈·크레이트·도구 프로필·오라클 디렉터리 전부 `general.architecture`의
   값을 그대로 쓴다: `deepseek2`, `deepseek41`. 줄임말(`ds2`)은 쓰지 않는다 — `grep deepseek41`이 모듈, 메타데이터
   키, 도구 프로필, 게이트를 한 번에 찾는 것이 이 규칙의 값이다.

## 무엇이 어디로

| 항목 | deepseek2(지금) | deepseek41(`v41-ops.md`·인벤토리) | 갈래 |
|---|---|---|---|
| 메타데이터 키 | `deepseek2.*` — u64는 `arch_get_u64`가 접두를 파일에서 읽고, **f32 넷은 `attn.rs:380-396`이 `deepseek2.` 리터럴**(`gate_head_gpu.rs:99`·`gate_p5.rs:1201`도) | `deepseek41.*` + `deepseek41.engram.{multipliers,primes,offsets,token_map,pad_id,…}` | **표** — 접두는 파일에서, `gguf`에 `arch_get_f32`를 더해 리터럴을 없앤다 |
| 텐서 이름·역할 | `attn_q`·`attn_kv_a_mqa`·`attn_kv_b`·`attn_output`, `ffn_{gate,up,down}{,_exps,_shexp}`, `ffn_gate_inp` — `gpu/model/scratch.rs`의 `LayerNames` 17개 + 파생 1 | `attn_q_a`·`attn_q_a_norm`·`attn_q_b`·`attn_kv`·`attn_sinks`·`attn_output_a`·`attn_output_b`, `hc_{attn,ffn}_{fn,base,scale}`, `exp_probs_b{,_vl}`, `engram_{embd,k,q,wkv}`, `attn_compressor_{kv,norm,gate}`, `indexer.{attn_q_b,attn_k,k_norm,proj}` | **표** — 아키텍처마다 이름 표 하나(`arch/<name>/names.rs`) |
| 층 종류 | dense 1층(0) + MoE 26층, 판정은 `ffn_gate_inp` **존재**로(층 번호 아님) | 40층 전부 MoE·hc. 그 위에 층별 플래그: engram **1·14**, 압축기 소유 **2·8·14·20**(게이트는 2·8·14 — 비율 2; 20은 비율 1), 인덱서 **2·8·14·20·24·28·32·36**, 윈도우 전용 rope 10k **0·1**(인벤토리 텐서 존재로 읽은 것, 0·1은 op 문서) | **아키텍처 타입** — `LayerKind`는 층 번호가 아니라 **텐서 존재**로 로드 시 결정(`routed`와 같은 규칙) |
| 블록 골격 | pre-norm 잔차: `attn_norm → attn → add → ffn_norm → ffn → add` (`forward::block_common`) | 하이퍼커넥션: 스트림 4벌, `pre`로 접어 들어가고 `post`·Sinkhorn `comb`으로 섞으며 믹스는 한 서브층 늦게 소비 | **아키텍처 사슬** |
| 어텐션 | MLA — 잠재 K, rope 꼬리 64, `wk_b` 흡수, 헤드별 `wv_b`; flash는 `[k_rope ; kv_compressed]` 576행 | MLA 아님 — 512차원 잠재 하나가 K=V, 64헤드 직접 내적, 헤드별 sink, 출력 꼬리 역-rope, 블록 대각 `wo_a`→`wo_b`; 윈도우 128행 + 고른 512행 = 최대 640행 | **아키텍처 사슬** + deepseek41 전용 커널(sink 어텐션·역-rope·그룹 출력 투영·압축기·인덱서·hc·engram 게이트) |
| KV 토폴로지 | 층마다 `[ctx_max, 576]` f16 평면 하나(`Vec<DeviceTensor<u16>>`), 슬롯 표는 하나 | 층마다 128행 링(윈도우) + **비율 그룹 넷이 공유**하는 압축 행(2–7·8–13·14–19·20–39) + 인덱서 키 | **아키텍처 타입** — 결정 2의 `(seq, pos)` 슬롯 표는 공유, **행 저장소**만 아키텍처 것. 그룹 공유 캐시는 "층 슬롯당 버퍼 하나"로는 표현이 안 된다 — 이것이 타입이 갈리는 첫 자리다 |
| 스텝 입력 | 토큰 id, 위치(`step_params` 1버퍼) | 토큰 id, 위치, **engram 48행(13,056 B, 한 스텝 앞서 발행)**, 압축 층의 **호스트 색인 계획**(`v41-ports.md`: 비율 기계는 그래프 분기가 아니라 스텝마다 호스트가 세우는 계획) | **아키텍처 타입** — `ChainBody::Input` |
| MoE | 라우터 softmax → top-6 → 정규화·스케일, **64 → 6 핀**(`MoeDims::read`가 전문가 수·슬롯 수·텐서별 양자화 타입을 핀) | `√softplus` + 선택 편향 → top-6 → 재정규화 ×1.5, 384 → 6, SwiGLU ±10, 합 `Σ+1e-20` | 라우터·점수는 **아키텍처 커널**; 전문가 `_sel` 본체는 형상이 같으니 **공유**되되 타입이 갈린다 — gate/up은 q3_K(있음), down은 **q5_K(커널도 디퀀트도 없다**, `quant.rs:198`·`weights.rs:253`), shared 전문가는 q8_0(우리 융합 gate/up은 q3_K 전용); 전문가 **배치와 호스트 티어는 공유 서비스**(아래 「직교」) |
| 헤드 | `output_norm` + `output`, eps는 `mla.eps` | 같음 + 마지막 FFN의 `pre`로 스트림을 접는 마지막 접기 | **거의 공유** — 접기는 사슬 끝에서, `Head`는 eps를 계획에서 받는다 |
| 파생 가중치 | `wk_b`의 Q8_0 재양자화(`Derived`, 로드 시 CPU) | 알려진 것 없음(hc `comb`·층 종류별 rope 표는 후보 — B4가 정한다) | **아키텍처 계획** |
| 가중치 형식 | `DevWeight`: K-quant(Q3/Q4/Q6)·Q5_0·Q5_1·F32 + 파생 Q8_0. **파일 텐서로서의 Q8_0 팔이 없고, BF16이 없고, Q5_K는 `weights.rs:253/356`이 거부** | q8_0 **332개(216 GB)**, bf16 45개, q5_K 2개(인벤토리 `## types`) | **공유 크레이트의 일**(B1의 이웃) — 아키텍처가 아니다 |
| 오라클·탭 | `$BLOOMERY_DATA/ref_cuda`(ik main CUDA 덤프), 탭 `l_out-N`·MLA 중간, `gpu-gates/lib.rs`의 `DEFAULT_MODEL` 상수 | 오라클 v3(B0c, V4.1 포트에서), 정수 일치 탭 셋(engram 행 id·인덱서 top-k·라우터 top-6) + `l_out` | **표** — `gpu-gates/src/oracle/<name>.rs`(디렉터리·매니페스트·탭 이름), 하네스는 공유 |
| 참조 엔진·도구 | ik `main`, `-mla 3 -fa 1 -fmoe 1`, `prompts.tsv`, `ref-paths.sh`의 `MODEL` 기본값 | **다른 트리** — 우리 V4.1 포트(#2455, `~/repo/upstream/v41-ports/mine`), 다른 플래그, 다른 프롬프트·PPL 셋 | **표** — `tools/ref/models/<name>.sh`, `BLOOMERY_MODEL`로 고른다. 증인 블록이 카드처럼 **모델 이름**을 찍고, 두 모델의 숫자는 한 표에 놓지 않는다 |
| 바이너리 | `generate`·`gate_e2e`·`bloomery-decode`가 `GpuModel`/`forward::step`을 직접 부른다 | 같은 바이너리 | 열 때 `general.architecture`를 **한 곳**에서 읽어 `AnyEngine` 팔을 고른다. 모르는 값은 그 문자열을 든 오류 한 줄(exit 1) |

## 코드 형태

### 크레이트와 디렉터리

    crates/gguf                 공유. GgmlType에 q8_0(가중치)·bf16 팔, arch_get_f32
    crates/qdot, threads        공유
    crates/engram               공유 서비스(부르는 것은 deepseek41만) — B3, 비행 중
    crates/model                호스트 쪽 모델 코드
      src/{ops,profile,head,ffn,moe,kv}.rs    공유 — kv.rs는 SlotTable·KvRows(행 폭은 계획이 준다)
      src/arch/mod.rs                         `Arch` 열거형 + `Arch::detect(&Gguf)` — general.architecture를 읽는 유일한 자리
      src/arch/deepseek2/{attn,derived,forward,names}.rs   지금의 파일이 그대로 이사한 것
      src/arch/deepseek41/{hparams,names,plan}.rs          B4가 처음 만든다 — 순전파는 없다(아래)
    crates/gpu                  런타임(Gpu·Graph·Weights·DeviceTensor·Q8Act) + 커널 전부(지금 있는 파일 그대로)
      src/model.rs                            `GpuModel<B: ChainBody>` — 스테이지·모드·캡처/재생·pos·헤드. 공유 골격
      src/arch/deepseek2/{dispatch,scratch,names,seed,lookup,kernels,probe}.rs → `impl ChainBody`
    crates/gpu-deepseek41       새 크레이트(S0 스파이크 결과에 따라): V4.1 전용 #[cuda_module]들 + arch/deepseek41 사슬 → `impl ChainBody`
    crates/gpu-gates            lib.rs 하네스 공유; src/oracle/{deepseek2,deepseek41}.rs; bin은 gate_p*(deepseek2)와 gate_deepseek41_<op>
    tools/ref/models/{deepseek2,deepseek41}.sh   ref-paths.sh가 BLOOMERY_MODEL로 소스

`crates/gpu`에 남는 커널 가운데 지금 deepseek2만 쓰는 것이 있다(`flash.rs`의 `[rope ; latent]` 행 어텐션,
`fused.rs`의 `kv_norm_rope_append`, `model/kernels.rs`의 헤드별 래퍼 — 독해다, 재지 않았다). **옮기지 않는다.**
커널은 형상의 이름을 가지고 형상은 인자로 받으니 라이브러리에 있어도 거짓이 아니고, 옮겨서 얻는 성질이 없다 —
deepseek2 커널의 컴파일러 결함은 어차피 deepseek2 게이트를 붉게 한다. 반대로 **deepseek41 전용 디바이스 코드는
자기 크레이트**를 가진다: cuda-oxide의 ICE(`crates/oxide-ice-unroll`이 그 재현기다)가 V4.1 커널 하나에서 나면
같은 크레이트의 V2-Lite 게이트 전부가 함께 붉어지는데, 크레이트가 다르면 V4.1 빌드만 깨진다. 이것이 이 설계에서
크레이트 경계를 세우는 **유일한** 이유이고, 그래서 그 경계는 디바이스 코드에만 있다. 호스트 코드의 경계는 모듈
디렉터리와 기계 검사(아래)로 충분하다.

두 크레이트의 디바이스 코드가 한 바이너리에 링크되고 한쪽의 `#[cuda_module]`이 다른 크레이트의 `cores::` 본체를
부를 수 있는지는 **이 트리에 증거가 없다** — `gpu-spike`·`gpu-gates`는 자기 `#[cuda_module]`이 없다. S0가 그것을
잰다. 안 되면 대안은 `crates/gpu/src/arch/deepseek41/` 아래 두고 cargo 피처 `deepseek41`로 V2-Lite 게이트 빌드에서
빼는 것이다(격리 성질은 같고, `--features gpu`처럼 기준 계기에 피처 하나가 더 붙는 값을 치른다). 어느 쪽이든
`gpu-spike`·`gpu-gates`가 쓰는 "선택적 의존 + `gpu` 피처" 형태를 새 크레이트도 갖는다 — 평범한 `cargo check
--workspace`가 임베드 번들 앵커 심볼을 못 풀어 죽지 않게.

### 트레이트 둘, 둘 다 작다

```rust
/// What the shared skeleton drives at capture time. Monomorphic: `GpuModel<B>`.
pub trait ChainBody: Sized {
    /// Per-replay host values: deepseek2 = token/pos/n_keys/cs; deepseek41 adds the
    /// engram rows and the compressed-index plan.
    type Input;
    fn load(gpu: &Gpu, gguf: &Gguf, w: &Weights, layers: Range<usize>, ctx_max: usize) -> Result<Self, GpuError>;
    fn refresh(&mut self, stream: &CudaStream, input: &Self::Input) -> Result<(), GpuError>;
    /// One layer's capturable body: no allocation, no synchronization.
    fn enqueue_layer(&mut self, gpu: &Gpu, w: &Weights, slot: usize, embed: bool, obs: &mut Observer<'_>) -> Result<(), GpuError>;
    fn reset(&mut self, gpu: &Gpu) -> Result<(), GpuError>;
    fn head_eps(&self) -> f32;
    fn resident_bytes(&self) -> usize;
}

/// What a binary or a gate drives, once per token. Static dispatch only.
pub trait Engine {
    fn step(&mut self, tokens: &[u32]) -> Result<u32, GpuError>;
    fn reset(&mut self) -> Result<(), GpuError>;
    fn seed_depth(&mut self, rows: usize) -> Result<(), GpuError>;
    fn pos(&self) -> u32;
    fn resident_bytes(&self) -> usize;
    fn arch(&self) -> Arch;
}
impl<B: ChainBody> Engine for GpuModel<B> { /* the skeleton, once */ }

pub enum AnyEngine {
    Deepseek2(GpuModel<deepseek2::Body>),
    Deepseek41(GpuModel<deepseek41::Body>),
}
```

`GpuModel`의 골격 — 그래프 드롭 순서, 캡처 정체성 `(layer, embed)`, 모드 전환이 캡처를 버리는 규칙, `pos` 전진,
argmax 회수 한 번 — 은 미묘한 코드 400줄이고 두 벌을 두면 AGENTS가 금하는 이중 장부다. 그래서 골격은 `B`에
대해 제네릭 한 벌이고, `Stage`는 `Residency<B> { weights: Weights, body: B }`를 든다. 지금 `GpuModel`의 `mla`와
`moe` 필드는 `B` 안으로 들어간다(둘 다 deepseek2의 형상이다).

`AnyEngine`은 열거형이고 `Box<dyn Engine>`이 아니다. 세 번째 아키텍처를 더하면 팔 하나가 늘고 컴파일러가 빠진
`match`를 전부 지목한다 — 이 레포가 상수 쌍에 `const _: () = assert!`를 두는 것과 같은 종류의 래칫이다. `Engine`
트레이트는 게이트 하네스와 `generate`가 `fn drive<E: Engine>`로 두 팔을 한 코드로 몰기 위한 것이지 가상 호출을
위한 것이 아니다.

여덟 개짜리 연관 타입을 가진 `Arch` 트레이트는 만들지 않는다. 아키텍처 둘로는 무엇이 정말 공통인지 모르고,
지어낸 추상은 셋째 모델이 올 때 틀린 자리에 있다. `ChainBody`와 `Engine`, 그리고 `Arch::detect` 하나면
지금 필요한 다형성은 전부 덮인다.

### 스텝 입력이 타입인 이유 — engram

deepseek41의 `Input`은 토큰과 위치만이 아니다. 다음 토큰의 engram 48행 id는 지금 토큰이 정해지는 순간 계산된다
(앞 3토큰만 필요), 그래서 `step`의 끝(argmax 직후)에서 다음 스텝의 행을 **발행**하고 다음 `step`의 처음에서
**기다려** 13 KiB를 디바이스 버퍼로 올린 뒤 재생한다. 호출자가 argmax를 그대로 되먹이지 않으면(프롬프트 강제,
호스트 샘플러) 발행한 id와 실제 id가 어긋난다 — 그때는 그 자리에서 동기로 다시 발행하고 **miss 계수기**를 올린다.
이 전부가 `GpuModel<deepseek41::Body>::step` 안이고 `Engine` 표면은 모른다. 서버(C1)가 호스트 샘플러를 가지면
`hint_next(token)` 하나를 더한다 — 지금은 아니다.

### KV — 결정 2는 살고 저장소만 갈린다

`(seq, pos)` 두 차원 키의 슬롯 표는 공유 타입이다. 갈리는 것은 그 슬롯이 가리키는 **행 저장소**: deepseek2는
층 슬롯마다 평면 하나, deepseek41은 층마다 링 128행 + 그룹 넷이 공유하는 압축 행 + 인덱서 층의 키. 그룹 공유는
"층 l의 캐시"가 아니라 "그룹 g의 캐시"이므로 `kv[slot]`으로는 표현이 안 되고, 그래서 `ChainBody`가 자기 저장소를
소유한다. 두 모델이 같은 슬롯 표를 쓰면 추측 디코딩의 되감기(C4)는 한 곳에서 한 번만 짓는다 — `v41-ports.md`가
셋 포트에서 본 "되감기 소유자가 둘이라 죽은" 사례가 이 결정의 근거다.

## 직교인 것 — 아키텍처 축이 아니다

이것들을 `arch/` 아래에 두면 B2의 V2-Lite 연습이 deepseek2 코드에 갇힌다. 공유 서비스로 두고 아키텍처 사슬은
**호출만** 한다.

- **전문가 배치(B1)**: 텐서마다 (디바이스, dtype) 표. 표는 아키텍처 계획이 텐서의 *역할*을 알고 만들고, 공유
  `Weights` 로더가 소비한다.
- **호스트 티어(B2)**: `crates/model`의 전문가 gemv(풀 위) + 고정(pinned) 스테이징 + 합류 수단. 라우터는 아키텍처
  것이지만 그 뒤 "선택된 전문가를 gate/up/down `_sel`로 돌려 합친다"는 두 모델이 같은 모양이다 → 공유
  `moe::enqueue_experts(placement, …)` 하나를 두 사슬이 부르고, **B2의 경계는 그 함수 안 한 곳**이다.
- **engram(B3)**: 공유 크레이트, 부르는 것은 deepseek41의 `Body`뿐.
- **DSpark(C4)**: 같은 상주 가중치 위의 **두 번째 사슬**(드래프트 층 + 본 경로의 층 입력 평균). `GpuModel`이 캡처
  그래프를 `Chain` 종류별로 들 수 있게 필드를 스칼라가 아니라 맵으로 둔다 — 문만 열어 두고 짓지 않는다. deepseek2는
  `Chain::Decode` 하나다. **우리 V4.1 파일에는 DSpark 텐서가 없다**(`v41-op-map` 발견: `mtp_nextn` 티어 0개, 포트의
  그래프도 MTP 꺼짐을 단언) — C4는 그 텐서를 실은 다른 양자화 파일이 먼저다.
- **프리필 m>1(A5)**: `Input`이 m을 든다. 두 사슬 다.

## 검사 — 기계가 잡는 것

- `tools/check-arch.sh`(맥, grep뿐 — `check-recipes.sh`와 같은 부류, `just check-arch`):
  ① `arch/deepseek41/` 아래에서 `deepseek2`를 `use`하는 줄, 그 반대 방향도 없다
  ② `blk.N.<name>` 문자열 리터럴과 `deepseek2.`·`deepseek41.` 접두 키는 `arch/`와 `tools/ref/models/` 밖에 없다.
     맨 접두 `"blk."` 하나는 이름이 아니라 GGUF 블록 규약이라 걸리지 않는다(2026-09-23 — 공유 로더의 층 번호 파싱)
  ③ `general.architecture`를 읽는 자리는 `arch/mod.rs` 하나다
- 커널 파일은 모델 이름을 모른다(결정 6) — ②가 `crates/gpu/src/*.rs`에도 걸린다.
- 이관 라운드의 증명은 전부 「Derive first」의 **이동 클래스**다: `just ptx-scan` 표 동일(`gate_p5`·`gate_p8` 54행),
  그래프 노드 수 불변(지금 648), eager = replay, e2e 집합 동일(두 패스), CPU 게이트 레시피 전부·GPU 게이트 15개
  초록, lint 수는 오르지 않는다. 시간은 재지 않는다 — 스텝의 호스트 코드가 그래프 모드에서 스텝당 0회 도는 것은
  gpucast에서 이미 세운 논거다.

## 라운드

크기는 추정(S 반나절 이하 / M 하루)이고 끝나면 실제 소요를 옆에 적는다. 라운드마다 축 하나(R21).

| id | 라운드 | 파일 경계 | 증명 | 앞 | 크기 |
|---|---|---|---|---|---|
| S0 | **스파이크 — 크레이트 밖 디바이스 코드**: 새 크레이트에 `#[cuda_module]` 하나, `bloomery_gpu::cores::q3k_row_dot`(`pub`으로)을 부르는 래퍼 하나. `cargo oxide`가 빌드하는가, PTX가 크레이트 안 래퍼와 동일한가(`ptx-scan`), `.oxart` 멤버 둘이 한 게이트 바이너리에 링크되는가, `cargo oxide build` 시간의 차(전·후 각 3회) | 새 `crates/gpu-xcrate-spike`(머지하지 않음), `cores.rs` 가시성 한 줄 | 셋에 대한 답 + 시간 표. 실패면 `docs/upstream/nvlabs-ledger.md`에 한 줄 먼저 | — | S |
| M1 | **`crates/model`의 arch 이관**: `attn.rs`·`derived.rs`·`forward.rs`가 `arch/deepseek2/`로, `kv.rs`는 슬롯 표·`KvRows`만 남기고 폭은 계획에서, `arch/mod.rs`의 `Arch::detect`, `MlaParams::read`의 f32 리터럴 넷 → `arch_get_f32`(`gguf`) | `crates/model/src/**`, `crates/gguf/src/lib.rs`(getter 한 개), `crates/gpu`·`gpu-gates`는 **`use` 줄만** | CPU 게이트 레시피 전부 동일, `gate-derived` 바이트 동일, `gate-alloc` 수 동일, lint 불상승, `check-arch` 초록 | — | S~M |
| M2 | **`crates/gpu`의 arch 이관**: `GpuModel<B: ChainBody>`, `Residency<B>`, `dispatch`·`scratch`·`names`·`seed`·`lookup`·`kernels`·`model/probe`가 `arch/deepseek2/`로, `mla`·`moe` 필드가 `Body` 안으로, `Head`의 eps는 `head_eps()`에서 | `crates/gpu/src/model.rs`, `crates/gpu/src/model/**` → `arch/deepseek2/**`, `lib.rs`의 `mod`·`pub use` 줄 | `ptx-scan` 54행 동일, 648노드, eager = replay, e2e 집합 동일(두 패스), GPU 게이트 15개, lint 불상승 | M1 | M — **`model.rs`를 만지는 라운드라 한 시점에 하나** |
| M3 | **게이트·바이너리**: `gpu-gates/src/oracle/deepseek2.rs`(디렉터리·매니페스트·탭 이름 표), `DEFAULT_MODEL` 상수 → 도구 프로필이 주는 경로, `generate`·`gate_e2e`·`bloomery-decode`가 `AnyEngine` 위에서, 모르는 아키텍처는 오류 한 줄 exit 1 | `crates/gpu-gates/src/**`, `crates/model/src/bin/bloomery-decode.rs` | 게이트 15개 출력 줄 동일; **FAIL-first**: V4.1 파일을 `generate`에 주면 지금은 어디서 어떻게 죽는지 기록 → 뒤에는 `unsupported architecture "deepseek41"` 한 줄 | M2 | S |
| M4 | **도구 프로필**: `tools/ref/models/deepseek2.sh`(MODEL·IK 트리·플래그·프롬프트·PPL 셋·오라클 디렉터리), `ref-paths.sh`가 `BLOOMERY_MODEL`(기본 deepseek2)로 소스, 증인 블록이 모델 이름을 찍음, `check-arch.sh` 신설 | `tools/ref/**`, `tools/check-arch.sh`, `justfile` 레시피 한 줄 | 빈 환경에서 해석값 바이트 동일(tools6의 방법), `build-ref` md5 동일, shellcheck 불상승 | — | S |

**파동**: {S0 ‖ M1 ‖ M4} → {M2} → {M3}. 첫 파동 셋은 파일이 겹치지 않는다(새 크레이트 / `crates/model` /
`tools`). 리드 직렬 지점은 M2와 M3의 머지 둘이다. **B1·B2·B4·B5는 M2 뒤에 연다** — 같은 파일을 이관이 먼저
지나야 하기 때문이고, 비행 중인 `engram`(새 크레이트)과 `v41ops`(문서)는 이 파일들을 안 만지니 지금이 창이다.
머지 순서는 규율대로 비트 불변 라운드부터(전부 그렇다).

## 하지 않는 것

- 이 라운드들에서 V4.1 코드를 쓰지 않는다. `arch/deepseek41/`은 B4의 첫 op가 만든다 — 빈 디렉터리도 미리 두지 않는다.
- deepseek2만 쓰는 커널을 `crates/gpu`에서 꺼내지 않는다(위 — 얻는 성질이 없다).
- deepseek41의 CPU 순전파는 없다. `crates/model`이 deepseek41에 주는 것은 계획(hparams·이름·배치 표)과 호스트
  티어의 op다. deepseek2의 순전파는 1단계의 산출물이자 C1 서버의 엔진으로 그대로 산다.
- 트레이트 객체·층 단위 `dyn`·디스패치 안의 `if arch`는 셋 다 쓰지 않는다.

## 열린 것 — 재서 닫는다

- ~~크레이트 밖 디바이스 코드(S0). 이 답이 `crates/gpu-deepseek41`인지 피처인지를 정한다.~~ **닫힘(2026-09-23, 브랜치 `s0`
  스파이크)**: `crates/gpu-deepseek41`이다. 다른 크레이트의 `#[cuda_module]`이 `bloomery_gpu::cores::q3k_row_dot`을 인라인해
  PTX가 명령 단위로 같고(정규화 md5 동일, 라벨에 `bloomery_gpu__cores__q3k_row_dot_1col`이 남는다), 64행 비트 동일, `.oxart`
  둘이 한 프로세스에 뜬다(앵커는 크레이트·버전으로 구분된다). 조건 둘: ① 공유 본체는 `pub` + `#[inline(always)]`(코어 14/14
  이미 충족, `cores`·`q3k_row_dot`·`q8_1_quant_block`을 `pub`으로) ② **커널 엔트리 이름은 바이너리 전역 유일** — cuda-oxide의
  호스트 커널 심벌 `cuda_oxide_kernel_246e25db_<entry>`가 크레이트·모듈로 네임스페이스되지 않아 같은 이름은 링크 실패
  (`nvlabs-ledger` #12). 그래서 `crates/gpu-deepseek41`의 엔트리는 접두를 단다(`ds41_` — B4의 첫 op가 정한다). 피처 폴백은 필요 없다.
- ~~V4.1 커널 모듈 하나가 `cargo oxide build`에 더하는 시간~~ S0가 잰 것은 가시성 확대의 값뿐이다(`gate_p8b` 증분 재빌드 중앙값
  10.31 → 9.99 s, 3표본 — 차이가 분해되지 않는다). 새 디바이스 크레이트 하나의 값은 그 크레이트에 의존하는 바이너리가
  생길 때(B4) 같은 방법으로 잰다.
- V4.1 파일의 메타데이터 키 전수 — 인벤토리는 텐서만 적었다(샤드 1에 kv 68개). 포트의 로더가 **요구하는** 키는
  `v41-op-map`이 원본에서 확인했다(`deepseek41.engram.{layer_ids, head_count, key_length, max_ngram_size, pad_id,
  multipliers, primes, offsets, token_map}` 아홉 개, 하나라도 없으면 로드 실패) — 우리 파일이 그 값을 무엇으로 갖는지는
  `gguf-inventory`에 키 표를 더해야 읽힌다(헤더만, 임대 없음).
- **계획은 기본값을 만들지 않는다.** 포트는 `hyper_connection.sinkhorn_iterations`가 없으면 조용히 **3**(참조 20),
  `hyper_connection.epsilon`이 없으면 rms eps로 떨어진다(`v41-op-map` 발견). 우리 `arch/deepseek41/hparams.rs`는
  `rms_eps`처럼 없는 키를 오류로 낸다 — 파일이 값을 안 가지면 그 사실이 게이트 줄이지, 리터럴이 아니다.
