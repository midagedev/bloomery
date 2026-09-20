# Intel CPU 기반 LLM 구동 — 최근 자료 조사 (2025~2026 중심)

작성 2026-09-20. 조사 범위: 웹 공개 자료만(WebSearch/WebFetch). 대상: Intel의 CPU LLM 추론 기술 중 **알고리즘/스케줄링/메모리 계층 자산**. 산출 기준 원장: `/Users/hckim/repo/bloomery/docs/RESULTS-mul35-saturation.md`(2026-09-20).

**우리 박스 제약(이식성 판정 기준)**: AMD 5975WX Zen 3, 32코어 단일 소켓·단일 NUMA, AVX2+FMA+F16C+BMI2+VAES+VPCLMULQDQ까지(AVX-512·VNNI·AMX·GFNI 없음), STREAM 147.7 GB/s.
**원장 병목 코드**: (A) flash_attn_latent 스칼라 벽 #1 사이트 18.8% — (B) 오케스트레이션 잔여 ~10.2 ms/step(예측 변수 = 디스패치당 바이트) — (C) 직렬 activation quant 4.03 ms — (D) swiglu 직렬 1.37 ms — (E) lm_head Q6_K 유일 대역폭 벽(81% STREAM) — (F) q_nope2 Q8_0 스칼라. **진행 중(본 조사에서 중복 취급 금지)**: flash SIMD화, 직렬 quant 풀 이양, Q8_0 내적 maddubs화.

---

## 0. 한눈에: 우리 박스 기대가치 Top 5 (상세 근거는 §3)

1. **gate·up 결합 GEMM + SiLU·mul 에필로그 융합** (§2.1-1) — D+C+B 동시 공격, Intel이 DeepSeek MoE에서 실사용.
2. **동적 activation quant를 소비 GEMM의 입력 인출에 융합** (§2.1-2) — C의 상위 해법(진행 중 '풀 이양'과 상호보완).
3. **MoE 라우팅: topk_ids 정렬로 전문자별 블록 배치 GEMM + 게더/스캐터 커널화** (§2.1-3) — B(디스패치 굵기) 직격.
4. **KV 쓰기·패킹을 디코드/어텐션 커널에 융합("Load Once Pack Twice")** (§2.1-4, §2.1-6) — A의 데이터 경로 + 직렬 kvr/latent 접착부.
5. **디스패처 전용 코어 분리·예약 + 스텝 정적 캡처** (§2.3-1, §2.1-5) — B의 저비용 정돈(Intel/vLLM 양쪽 BKM).

---

## 1. 지형 2025-2026: Intel CPU LLM 스택의 무게중심 이동

- Intel의 CPU LLM 자산이 정리·통합 국면: **intel/ipex-llm 저장소 2026-01-28 아카이브**(읽기 전용, 이후 릴리스 노트는 상세 없음 — https://github.com/intel/ipex-llm/releases, 열람 2026-09-20), 문서는 intel-analytics/ipex-llm로 리다이렉트(https://github.com/intel-analytics/ipex-llm). 구형 순수-C++ 런타임 **neural-speed는 2024-08-30 아카이브**(https://github.com/intel/neural-speed). 살아있는 최전선은 **SGLang/vLLM + AMX** 조합(§2.1, §2.3).
- Intel 스스로 명시: Xeon 6 DeepSeek 백엔드는 "only supports CPUs with Intel AMX support"(AMX 없으면 느림) — https://www.lmsys.org/blog/2025-07-14-intel-xeon-optimization (2025-07). **즉 2025-2026 Intel 자료의 성능 숫자는 거의 전부 AMX 전제이며, AVX2에 이식 가능한 것은 알고리즘/융합/스케줄링 층이다.**
- 소프트웨어만의 이득 사례(참고 상한): MLPerf Inference v6.1(2026-09)에서 Xeon 6980P Llama-3.1-8B 서버 처리량이 동일 실리콘에서 소프트웨어 최적화만으로 v6.0 대비 2.4x — https://www.storagereview.com/news/mlperf-inference-v6-1-5-7x-per-accelerator-gains-a-512-gpu-run-and-vera-rubins-first-peer-reviewed-numbers (2026-09).
- 2026-02 배포 가이드 "A Practical Guide to CPU-Optimized LLM Deployment on Intel Xeon 6"(NUMA-aware 자동 병렬화, BF16, 청크 프리필) 존재 확인(검색 발췌, 직접 URL 미확보 — community.intel.com / swcatalog.intel.com, 2026-02-16~17).

---

## 2. 자료별 아이디어 (형식: 요지 / 출처 / 병목 매핑 / 이식성 / 기대 기제)

### 2.1 LMSYS×Intel×SGLang — "Cost Effective Deployment of DeepSeek R1 with Xeon 6 CPU" (핵심 자료)

출처: https://www.lmsys.org/blog/2025-07-14-intel-xeon-optimization (2025-07-14). Intel 커뮤니티 재게시 https://community.intel.com/t5/Blogs/Tech-Innovation/Artificial-Intelligence-AI/Cost-Effective-Deployment-of-DeepSeek-R1-with-Intel-Xeon-6-CPU/post/1704597 (2025-07-21). 플랫폼: 듀얼소켓 Xeon 6980P(128C/소켓), MRDIMM 8800. DeepSeek-R1-671B INT8 기준 llama.cpp 대비 TTFT 6~14x, **TPOT 2~4x**(TPOT 이득이 작은 이유를 "디코드가 메모리 대역폭 바운드" 때문이라 명시 — 원장 진단과 동일 구조). MoE INT8으로 **메모리 대역폭 효율 85%(실효 1.45 TB/s)** 달성.

**1) gate·up 결합 GEMM + SiLU·mul 에필로그 융합**
- 요지: FFN의 gate·up 두 GEMM을 `A×[B1,B2]=[C1,C2]` 단일 GEMM으로 합치고 SiLU(C1)*C2를 GEMM 에필로그에 융합해 "추가 load/store 제거".
- 병목: **D(swiglu 직렬 1.37ms) + C(moe g/u에서 같은 xb를 두 번 양자화하는 중복, 원장 §7-2) + B(gate/up 디스패치 52→26으로 감소)**.
- 이식성: **AVX2 가능** (융합은 순수 알고리즘; 에필로그는 FMA).
- 기대 기제: activation 중간 버퍼의 메모리 왕복이 사라지고 디스패치 수가 절반으로. 원장 기준 유도: D 소멸(1.37) + 중복 quant 제거(batch Q3_K quant 절반) + 디스패치 감소분. Intel은 동일 계열 융합(KV 버퍼)이 +12% TPOT임을 실측 — 여러 융합의 합성 효과 추정 근거.

**2) 동적 activation quant의 GEMM 입력 인출 융합**
- 요지: BF16→UINT8 동적 양자화를 activation fetch(가중치 스트리밍 시작 시점)에 융합; 입력 크기에 따라 AVX512 vs AMX 커널 선택.
- 병목: **C(직렬 quant 4.03 ms)**.
- 이식성: **AVX2 가능**(융합 구조 자체는 ISA 무관; 개별 커널만 ISA별).
- 기대 기제: 진행 중인 '풀 이양'은 패스를 병렬화하는 것이고, 이쪽은 **패스 자체를 소비자 커널 안으로 흡수**하는 더 강한 형태. 두 방향은 병행 가능(융합 남는 사이트만 풀 이양). 원장 §3의 세금(Instant 쌍당 35ns 추정)도 함께 사라짐.

**3) MoE 라우팅: topk_ids argsort → 전문자별 블록 배치 GEMM**
- 요지: SGLang GPU 방식을 CPU로 이식 — "run argsort on topk_ids"로 토큰을 전문자별로 정렬·블록화해 순차적 전문자 루프와 잦은 토켄 게더를 없앰.
- 병목: **B(디스패치 굵기 — 원장 H1: 달성 GB/s는 MB/디스패치의 포화 함수) + 직렬 moe_expert_io 0.303ms + moe_trace 0.320ms**.
- 이식성: **AVX2 가능** (정렬/블록 조직은 ISA 무관).
- 기대 기제: 6전문가×26층의 얇은 디스패치를 전문자 블록 단위로 굵히면 원장 잔여 10.2ms 중 MoE 관련분 축소. Q6_K(172MB/디스패치=81% STREAM)·Q5_1(16.8MB=75GB/s)이 보여주는 '굵은 디스패치 포화' 재현이 목표.

**4) KV 버퍼 세팅을 디코드 커널에 융합**
- 요지: KV 캐시 갱신(버퍼 세팅)을 디코드 커널에 융합, torch의 암묵 dtype 변환·TensorImpl 생성·TensorIterator 복사 제거 — **+12% TPOT 실측**.
- 병목: **직렬 kvr·latent 접착부 + A(flash) 에필로그**.
- 이식성: **AVX2 가능**.
- 기대 기제: 스텝당 직렬 접착부 소계(~0.8ms f32_tensor·rms_norm·kvr 등) 중 KV 기록분을 워커 내로. Intel 숫자(+12%)가 이 계열 융합의 실효 상한 참고치.

**5) 그래프 모드로 호스트 오버헤드 제거**
- 요지: torch.compile 그래프 모드로 파이썬/런타임 오버헤드 제거 실험 — "예비 결과로 TPOT 추가 10% 개선".
- 병목: **B(오케스트레이션 잔여 10.2 ms)**.
- 이식성: **AVX2 가능(ISA 무관)** — 우리는 Rust라 파이썬 오버헤드는 없지만, "스텝 전체를 정적으로 캡처해 단계 경계의 동적 판단을 없앤다"는 개념은 유효(원장 §5 파킹/깨움 캐스케이드 관리와 연결).
- 기대 기제: 디스패치 경로의 조건분기·파킹/깨움 빈도 축소. Intel 수치는 10%(예비)로 보수적.

**6) MLA "Load Once Pack Twice" — KV 1회 로드로 K/V 이중 포맷 동시 패킹**
- 요지: KV를 2개 LUT+프리페치로 1회만 메모리에서 읽어, K 버퍼[NT용]와 V 버퍼[NN용]을 동시에 패킹(각 32레인). MLA 최적화 전체로 "바닐라 대비 약 1.9x". head folding으로 head 차원을 GEMM으로 접고 배치에 따라 블록 크기 6→22 가변.
- 병목: **A(flash_attn_latent — KV f16, 평균 키 54.5, 1152B/키)**.
- 이식성: **AVX2 가능**(데이터 배치·프리페치·포맷 전처리는 ISA 무관; 포맷 자체는 소비 커널에 맞추면 됨).
- 기대 기제: 진행 중 flash SIMD화와 독립적으로 **데이터 경로**(KV 로드 횟수·패킹 레이아웃)를 개선 — SIMD화와 곱으로 합성. 원장 H2: flash는 대역폭 0.2%라 이득은 지연 감소 쪽.

**7) Flash Decoding — KV 청크 분할 병렬화 + 리듀스**
- 요지: 디코드 어텐션에서 KV를 청크로 갈라 병렬도를 확보한 뒤 부분 결과를 리듀스. 병렬 축 [B,H,qBlocks]이 단일 요청에선 [1,H,1]로 붕괴하는 문제의 해법.
- 병목: **A**.
- 이식성: **AVX2 가능**(분할-리듀스는 알고리즘).
- 기대 기제: 우리는 N=96×16head라 병렬도가 이미 넓음 — 이득 제한적(원장 MUL-29 행 분할과 중첩). 낮은 우선순위.

**8) 프리필 flash: prefix(사각)/extend(삼각) 분리 + L1/L2 맞춤 타일링 + 변환 융합**
- 요지: 프리필 어텐션을 prefix(전체 키 조회, 사각형)와 extend(새 토큰, 하삼각)으로 분리해 중복 계산 스킵; 블록 크기를 S_i, m_i, S*가 L1/L2에 들어가게 선택; 온라인 소프트맥스 모멘텀 갱신에 dtype 변환을 융합.
- 병목: **A의 타일 설계 원칙**(우리 프롬프트 6토큰·디코드 위주라 직접 이득은 작음).
- 이식성: **알고리즘 부분은 AVX2 가능**(Intel은 GEMM에 AMX, pointwise에 AVX512 사용 — 이 분배 자체는 AMX 필요).
- 기대 기제: flash SIMD 커널의 타일 파라미터(원장 b=0.1118 ms/키 기준 최적 블록) 선택에 적용.

**9) FP8→BF16 언팩: NaN/denorm 검사 스킵 + 지그재그 L2 블로킹**
- 요지: FP8 가중치를 BF16으로 푸는 비용을 NaN/denorm 검사 생략으로 절반(60-70사이클→반감)하고, 언팩된 BF16 블록을 지그재그 패턴으로 L2에 배치.
- 병목: 일반적 **dequant/중간 버퍼 캐시 관리**(우리의 dequant·quant 접착부에 응용).
- 이식성: **개념은 AVX2 가능**(FP8 자체는 우리와 무관).
- 기대 기제: 중간 activation 버퍼의 L2 재사용률 향상 — 원장 C·D 개선의 보조.

**10) S8S8 에뮬레이션 `(A+128)×B − 128×B`**
- 요지: AVX512-VNNI가 U8S8만 지원하므로 부호있는 활성을 오프셋 트릭으로 에뮬레이션.
- 병목: 해당 없음(참고).
- 이식성: **AVX-512·VNNI 필요 — Zen 3 불가**. VNNI 없는 AVX2에서의 등가 기법이 바로 진행 중인 maddubs(Q8_0) 계열임 — Intel의 트릭이 우리 진행 작업의 필요성을 역으로 뒷받침.

**11) NUMA TP + 공유메모리 올리듀스(통신 3%)**
- 요지: 멀티 GPU TP를 멀티 NUMA에 매핑, torch.distributed 대신 커스텀 공유메모리 all-reduce로 통신 비용 3% 달성. KV 중복 접근은 미해결(DP-MLA는 향후 과제).
- 병목: 없음 — **우리는 단일 NUMA라 그대로는 무관(N/A)**.
- 이식성: N/A(다만 'CCD 4개×L3 32MiB를 캐시 도메인으로 취급한 스케줄링'으로 변주할 여지는 있음 — 추정).

### 2.2 neural-speed / BesTLA — AVX2 폴백 경로의 실체 (아카이브된 자산)

출처: https://github.com/intel/neural-speed (아카이브 2024-08-30), 커널 문서 https://github.com/intel/neural-speed/blob/main/neural_speed/core/README.md (열람 2026-09-20). Intel의 NeurIPS 2023 논문 "Efficient LLM Inference on CPUs"(https://arxiv.org/abs/2311.00502, 2023-11 — ggml 대비 최대 1.6x, KV 사전할당으로 재할당 제거)의 런타임 후신.

**12) BesTLA ISA 디스패치 표와 AVX2 경로의 존재**
- 요지: Intel 순수 C++ 커널 라이브러리가 **AVX2를 기본 경로로 지원**(int8 compute 포함). 단 AVX-VNNI 없는 AVX2에서 int7 비대칭/int8 대칭이 "수치 오버플로 우려"라고 명시.
- 병목: **F(Q8_0 등 int8 계열 커널)**의 참고 구현.
- 이식성: **AVX2 가능(코드 차용 가능 — Apache-2.0)**.
- 기대 기제: 오버플로 회피 레이아웃·그룹 스케일 처리를 벤치마크 대조용으로 직접 열람 가능. 'ik 대비 95-99% 커널률' 검증에 제3의 구현 확보.

**13) QKV/FFN/MHA 융합 목록**
- 요지: fc(gemm+bias)에 QKV 융합(GPT-J·LLaMA), **FFN 융합(LLaMA·ChatGLM 등)**, fused MHA를 라이브러리 차원에서 제공.
- 병목: **D+C+B**(§2.1-1과 같은 방향의 원조 구현).
- 이식성: **AVX2 가능**.
- 기대 기제: 융합 지점·레이아웃의 구체적 설계 표본.

**14) 하이브리드 비트(층별 int4×int2 혼합) + 그룹 크기 가이드**
- 요지: 층마다 bits·알고리즘·group size를 다르게 혼합(int1~int8, fp4, nf4). group=128 int4가 group=32와 정확도가 거의 같으면서 더 빠름. group=-1(채널별)은 재학습 필요.
- 병목: 포맷 설계(전 사이트에 걸친 바이트 예산).
- 이식성: **AVX2 가능**.
- 기대 기제: 우리 Q3_K 디코드 사이트의 '디스패치당 바이트 vs 정확도' 재설계 참고 — 다만 재양자화 비용이 커 우선순위는 낮음(추정).

### 2.3 vLLM CPU 백엔드 (Intel 주도)

출처: 공식 문서 https://docs.vllm.ai/en/latest/getting_started/installation/cpu/index.html (열람 2026-09-20; AVX2 = "Limited features"로 명시, AMD는 Zen4+로 AVX512 권장).

**15) 프레임워크용 코어 예약 + 스레드 바인딩**
- 요지: 서빙 프레임워크에 1-2코어를 예약하고(예: 0-30 바인드, 31 예약) `VLLM_CPU_OMP_THREADS_BIND`로 코어 고정, 하이퍼스레드는 한 물리 세트만 사용, 블록 크기는 32배수 권고.
- 병목: **B(오케스트레이션 — 디스패처가 워커와 코어를 다투는 구조 정리)**.
- 이식성: **AVX2 가능(ISA 무관)**.
- 기대 기제: 원장의 디스패처-워커 관계(풀 32워커 + 디스패처)에서 디스패처 전용 코어 분리는 저비용·저위험 정돈. 효과 수치는 미상(문서는 정성 권고).

**16) "paged attention은 메모리 바운드 — 컴퓨트보다 접근 최적화"**
- 요지: vLLM CPU가 oneDNN 커널을 쓰는 이유를 Intel 개발자가 "paged attention은 극도로 메모리 바운드라 메모리 접근 최적화가 연산보다 중요"로 설명.
- 출처: https://github.com/vllm-project/vllm/issues/10694 (2024-11-26).
- 병목: **원장 H1(달성 GB/s 예측 변수)과 독립적으로 같은 결론** — 우리 진단의 외부 상호 검증.
- 이식성: 무관(진단 근거).

**17) CPU에서의 continuous batching·paged attention·prefix caching 상용화**
- 요지: vLLM CPU 백엔드가 GPU 백엔드와 동일 기능(연속 배칭, PagedAttention, 프리픽스 캐싱, 청크 프리필, TP)을 CPU에서 제공(https://docs.vllm.ai — 상기 문서; Red Hat 검토 https://www.redhat.com/en/blog/rethinking-cpu-gpu-split-llm-inference, 열람 2026-09-20). NUMA 노드를 TP 랭크로 취급.
- 병목: B 참조(스케줄러 설계).
- 이식성: **AVX2 가능**(스케줄러는 ISA 무관; 단 vLLM CPU 자체는 AVX2 기계에서 기능 제한).
- 기대 기제: 배칭 스케줄러 비교 대상 확보.

### 2.4 oneDNN (uxl Foundation, 구 Intel)

출처: v3.13 릴리스 https://github.com/oneapi-src/oneDNN/releases/tag/v3.13 (2026-07-17, 태그 API로 확인).

**18) AVX2에서 GEMV형 matmul 성능 개선**
- 요지: v3.13에서 "bf16, f16, f32 matmul의 단위 M/N/K 차원(GEMV형) 성능 개선"이 **AVX2 명령세트 프로세서 대상**으로 명시됨.
- 병목: **E의 f32/f16 이웃 사이트**(F32 라우터 13.4 GB/s, 활용도 21% — 라우터는 F32 GEMV형). lm_head(Q6_K) 자체는 qdot 영역이라 직접 아님.
- 이식성: **AVX2 가능(라이브러리 차원에서 이미 이식됨)**.
- 기대 기제: GEMV형 특화(단위 차원 분기)가 얇은 디스패치 사이트의 커널률을 올리는 표준 수단임을 최신 릴리스가 확인. 우리 엔진엔 해당 분기 설계 참고 가치.

**19) u4/s4 가중치 + 그룹 스케일 u8/s8 matmul 개선 / f8 SDPA 서브그래프**
- 요지: int4-가중치 + int8-활성 + 그룹 스케일 조합의 matmul 개선, Graph API의 양자화 SDPA(어텐션) 서브그래프 성능 개선.
- 병목: 포맷 방향 참고(B/A).
- 이식성: 개선 커널 자체는 대부분 **AVX-512/AMX 필요**로 보임(릴리스 노트는 ISA 미표기 — 연도·세부 미상), 그래프 융합 아이디어(quant→SDPA를 한 서브그래프로)는 **AVX2 가능(개념)**.
- 기대 기제: '양자화+어텐션을 하나의 융합 그래프로'의 공식 구현 사례.

### 2.5 OpenVINO GenAI

출처: OpenVINO GitHub 릴리스 https://github.com/openvinotoolkit/openvino/releases (열람 2026-09-20), Intel Distribution 릴리스 노트(intel.com, 2025-07-22 항목 — 검색 발췌), 문서 https://docs.openvino.ai (열람 2026-09-20).

**20) CPU에서 continuous batching + Paged Attention 성능 개선의 상용화**
- 요지: 2025년 릴리스(2025.1 계열)에서 "Paged Attention 성능·기능 개선", "GenAI API의 continuous batching 사용 시 CPU LLM 성능 개선", VLM의 continuous batching 기본화가 명시됨.
- 병목: **B(스케줄러)**.
- 이식성: **AVX2 가능 — OpenVINO GenAI는 AVX2 기기에서도 동작하는 Intel 계열 참조 구현**(ISA 요구가 AVX2급).
- 기대 기능: 연속 배칭+페이지드 KV의 CPU 스케줄러를 소스 수준에서 열람 가능한 제3 구현(llama.cpp·ik와 다른 관점).

### 2.6 llama.cpp 계열 Intel 기여

**21) ggml AMX 백엔드의 CPU 백엔드 재통합**
- 요지: Intel AMX 백엔드가 CPU 백엔드로 재통합(이슈 https://github.com/ggml-org/llama.cpp/issues/10359, 2024-11-17; 병합 PR #10570, 2024-12 — 검색 요약 기준). AMX 필요 레이아웃 변환을 버퍼 타입에서 처리.
- 병목: 없음 — Zen 3에는 **AMX 필요라 이식 불가**.
- 의의: ik_llama.cpp와 같은 계보에서 Intel 기여는 AMX 쪽에 집중됨 — AVX2 사이드의 알고리즘 자산은 §2.1·§2.2에 있음.

### 2.7 논문 (Intel CPU 실험/관련)

**22) NoMAD-Attention — 곱셈 없는 어텐션 (NeurIPS 2024)**
- 요지: 어텐션의 MAD 연산을 SIMD 레지스터를 초고속 룩업 테이블로 쓰는 in-register lookup으로 치환(사전학습 모델 그대로, 4비트 LLaMA-7B 16k 문맥에서 최대 2x, 품질 유지 주장).
- 출처: https://arxiv.org/abs/2403.01273 (2024-03-02 게시; NeurIPS 2024 게재로 보임 — 정확 트랙 미상). 실험이 Intel CPU.
- 병목: **A의 알고리즘 대안**(SIMD화 진행 중 작업과 직교 — 연산 종류 자체를 바꿈).
- 이식성: **AVX2 가능(개념)** — 단 256비트 레지스터는 룩업 테이블 용량이 작아 효과 감소 가능(추정).
- 기대 기제: 원장 H2의 스칼라 벽(259ns/행-키 ≈ f32 FMA ~1,100개)을 연산 수 자체로 깨는 경로. 품질 리스크 사전 검증 필요.

**23) QUOKA — 질의 지향 KV 선택 스파스 어텐션**
- 요지: 청크 프리필에서 대표 질의(평균 질의와 코사인 유사도 낮은 것)가 미는 키 집합만 남기는 훈련 불필요 스파스 어텐션 — Intel Xeon에서 어텐션 최대 ~7x, 사용 KV 88% 감소, 정밀도 근사 유지.
- 출처: https://arxiv.org/abs/2602.08722 (2026-02-09). 저자 소속은 페이지상 미표기(Intel 아님 — Qualcomm 계열 저자명).
- 병목: A와 유사 사이트이나 **프리필 전용** — 우리(프롬프트 6토큰, 디코드 T=1)와는 거리.
- 이식성: AVX2 가능(알고리즘). 우선순위 낮음.

**24) FairyFuse — 삼진 가중치 fused 커널**
- 요지: {-1,0,+1} 삼진 가중치로 GEMV의 곱셈 전부를 AVX-512 masked add/sub 루프로 치환("부동소수 곱셈 0개"), 16x 가중치 압축으로 메모리 바운드→컴퓨트 바운드 전환(루프라인 분석). 커널 29.6x, 단일 Xeon 8558P에서 llama.cpp Q4_K_M 대비 e2e 1.24x(32.4 tok/s), 품질 근사 손실 없음.
- 출처: https://arxiv.org/abs/2604.20913 (2026-04-22).
- 병목: **E(대역폭 벽)를 비트 폭으로 무너뜨리는 노선**의 2026년 계량 사례.
- 이식성: 구현은 **AVX-512 필요**로 명시; "압축으로 바운드 전환 + 비트별 마스크 연산" 개념 자체는 AVX2에서도 변용 가능하나 Zen 3에선 이득 축소 추정(곱셈이 이미 FMA 1클럭).
- 기대 기제: 참고 — 우리 Q3_K→초저비트 재양자화는 비용이 크고 lm_head만이 벽이라 적용면 좁음.

**25) IPEX-LLM 셀프 스페큘러티브 디코딩**
- 요지: CPU BF16 추론에서 Self-Speculative Decoding으로 "FP16/BF16 지연 기준 ~30% 개선" 주장.
- 출처: https://github.com/intel-analytics/ipex-llm README(열람 2026-09-20; 도입 연도 미상 — 문서상 명시 없음).
- 병목: 디코드 전체(스텝당 가중치 재방문 절감).
- 이식성: **ISA 무관(알고리즘)**.
- 기대 기제: CPU 디코드는 대역폭 바운드라 검증(배치) 패스의 한계효율이 높음 — 단 T=1 샘플링에서 수용률·드래프트 선택이 관건, DeepSeek-V2-Lite(MLA+MoE)용 셀프 드래프팅 설계 필요. 추정.

**26) Efficient LLM Inference on CPUs (역사 기준점)**
- 요지: Intel ITREX INT4 WOQ + 전용 런타임 — ggml 대비 1.3~1.6x(그룹 128), KV 캐시 재할당을 사전할당으로 제거. ISA 지원 매트릭스(AVX2/AVX512/VNNI/AMX) 존재만 명시, 경로별 알고리즘 설명은 없음.
- 출처: https://arxiv.org/abs/2311.00502 (2023-11-01, v2 2023-12-07).
- 이식성: 무관(배경).

---

## 3. 기대가치 순위 Top 5 (우리 박스 기준 — 원장 §7 후보와의 중복 여부 병기)

| 순위 | 아이디어 | 공격 병목 | 원장 유도 절감(근거) | 비고 |
|---|---|---|---|---|
| 1 | **gate/up 결합 GEMM + SiLU·mul 에필로그**(§2.1-1) | D+C+B | swiglu 1.37ms 소멸 + g/u 이중 quant 제거(원장 §7-2가 지목한 중복 156/312쌍) + 디스패치 52→26 | 원장 §7 후보 4번의 Intel 실증 버전. Intel 실측 근거(동일 계열 융합 +12% TPOT) |
| 2 | **activation quant를 소비 GEMM 인출에 융합**(§2.1-2) | C | 4.03ms 중 융합되는 사이트 전량(상한) | 진행 중 '풀 이양'과 배타가 아님 — 융합 잔여분만 풀 이양하는 2단계 구성 가능 |
| 3 | **MoE topk 정렬 전문자 블록 배치**(§2.1-3) | B | 잔여 10.2ms 중 MoE 게더/스캐터·얇은 디스패치분(수 ms 추정) | 원장 §7 '배치 굵게/패킹'의 구체 설계 표본을 Intel이 제공 |
| 4 | **KV 쓰기·패킹 융합 + Load Once Pack Twice**(§2.1-4,6) | A 데이터 경로 + 직렬 kvr/latent | 직렬 접착부 일부 + flash 지연(간접). Intel MLA 전체 1.9x 참고 | flash SIMD(진행 중)와 곱으로 합성되는 독립 축 |
| 5 | **디스패처 코어 분리·예약 + 스텝 정적 캡처**(§2.3-15, §2.1-5) | B | 원장 §5 파킹/깨움 0.1-0.7ms/step + 잔여 일부(추정) | 저위험 정돈. vLLM BKM + Intel 그래프 모드(+10%) 이중 근거 |

순위 밖 언급: NoMAD-Attention(A의 알고리즘 대안, 품질 리스크 — 실험 가치 있음), oneDNN v3.13 AVX2 GEMV 개선(F32 라우터 참고), OpenVINO GenAI 스케줄러(비교 대상).

---

## 4. 출처 총목록

- LMSYS×Intel DeepSeek R1 Xeon 6: https://www.lmsys.org/blog/2025-07-14-intel-xeon-optimization (2025-07-14); Intel 재게시 https://community.intel.com/t5/Blogs/Tech-Innovation/Artificial-Intelligence-AI/Cost-Effective-Deployment-of-DeepSeek-R1-with-Intel-Xeon-6-CPU/post/1704597 (2025-07-21)
- neural-speed: https://github.com/intel/neural-speed (아카이브 2024-08-30); 커널 문서 https://github.com/intel/neural-speed/blob/main/neural_speed/core/README.md (열람 2026-09-20)
- Efficient LLM Inference on CPUs: https://arxiv.org/abs/2311.00502 (2023-11)
- ipex-llm: https://github.com/intel/ipex-llm/releases (아카이브 2026-01-28; 2.2.0 = 2026-04-07, 2.1.0 = 2025-08-22); README https://github.com/intel-analytics/ipex-llm (열람 2026-09-20)
- vLLM CPU 문서: https://docs.vllm.ai/en/latest/getting_started/installation/cpu/index.html (열람 2026-09-20); oneDNN 커널 논의 https://github.com/vllm-project/vllm/issues/10694 (2024-11-26); CPU-GPU 분할 검토 https://www.redhat.com/en/blog/rethinking-cpu-gpu-split-llm-inference (열람 2026-09-20)
- oneDNN v3.13: https://github.com/oneapi-src/oneDNN/releases/tag/v3.13 (2026-07-17)
- OpenVINO: https://github.com/openvinotoolkit/openvino/releases (열람 2026-09-20); https://docs.openvino.ai (열람 2026-09-20); Intel Distribution 릴리스 노트 2025-07-22 항목(intel.com — 검색 발췌)
- llama.cpp AMX 재통합: https://github.com/ggml-org/llama.cpp/issues/10359 (2024-11-17; 병합 PR #10570, 2024-12 — 검색 요약 기준)
- NoMAD-Attention: https://arxiv.org/abs/2403.01273 (2024-03-02)
- QUOKA: https://arxiv.org/abs/2602.08722 (2026-02-09)
- FairyFuse: https://arxiv.org/abs/2604.20913 (2026-04-22)
- MLPerf Inference v6.1 Xeon 6980P 2.4x(소프트웨어만): https://www.storagereview.com/news/mlperf-inference-v6-1-5-7x-per-accelerator-gains-a-512-gpu-run-and-vera-rubins-first-peer-reviewed-numbers (2026-09)
- "A Practical Guide to CPU-Optimized LLM Deployment on Xeon 6": community.intel.com / swcatalog.intel.com (2026-02-16~17 — 검색 발췌, 직접 URL 미확인)
