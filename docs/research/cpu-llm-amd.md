# AMD 최신(2025–2026) CPU 기반 LLM 구동 자료 조사 — bloomery 이식 판정용

작성 2026-09-20. 웹 전용 조사(WebSearch/WebFetch/웹리더). 박스 미접출, 저장소 미기록.
판정 기준: **ThreadRipper Pro 5975WX = Zen 3 (Castle Peak), 32코어, 단일 소켓·단일 NUMA, L3 128MiB(4×32MiB CCD), DDR4 8채널, STREAM 147.7 GB/s. ISA 상한: AVX2+FMA+F16C+BMI2+VAES+VPCLMULQDQ (AVX-512/VNNI/AMX/GFNI 없음).**

원장(RESULTS-mul35-saturation.md) 병목 대응표: ① flash 스칼라 벽(6.01 ms/step, 진행 중: flash SIMD), ② 오케스트레이션(풀 평균 9.1/32 가동, '디스패치당 바이트' 법칙, 잔여 ~10.2 ms), ③ 직렬 quant 4.03 ms(진행 중: 풀 이양), ④ q_nope2 Q8_0 스칼라(진행 중: maddubs), ⑤ lm_head Q6_K = 유일한 대역폭 법(81% STREAM), ⑥ 직렬 접착부(swiglu 1.37 + 게더/스캐터 등 ~3.3 ms).
**진행 중 항목(flash SIMD·quant 풀 이양·Q8_0 maddubs)과 중복되는 제안은 "외부 확인용"으로만 기재.**

---

## 1. AMD PACE — Platform Aware Compute Engine (LLM 전용 CPU 엔진, vLLM 대비 1.60×)

- **요지**: AMD가 5세대 EPYC용으로 만든 LLM 추론 엔진의 최적화 원칙 — ① 커널 블로킹을 L1/L2 크기에 정렬해 matmul이 캐시 상주, ② 가중치 행렬을 **SIMD 타일 차원에 맞는 블록 레이아웃으로 미리 패킹**(reorder_direct류)해 런타임 리오더 제거, ③ **인라인 패킹 + QKV 융합**으로 커널 런치 오버헤드·메모리 트래픽 절감, ④ SDPA 인과 타일 스킵 + 인라인 RoPE 융합, ⑤ GQA를 네이티브 처리(expand-reshape-copy 회피), ⑥ KV 캐시를 **L2 크기에 자동 튜닝된 고정 크기 블록의 SLAB 풀 할당기**로 관리. 전체 geomean vLLM 대비 1.60×, 스펙 조합 시 2.5–3.2×.
- **출처**: https://www.amd.com/en/developer/resources/technical-articles/2026/amd-pace---high-performance-platform-aware-compute-engine.html (2026-04)
- **병목 매핑**: ② 오케스트레이션 — "커널 런치 오버헤드 절감"은 우리의 디스패치당 바이트 법칙과 같은 진단(AMD도 잔여를 '런치 오버헤드'로 부름). 가중치 사전 패킹=디스패치 굵히기의 수단. ⑥ 직렬 접착부(GQA 게더 흡수). ① flash(SLAB attention 구조 참고).
- **이식성**: 블로킹·사전 패킹·타일 스킵·SLAB 풀·GQA 네이티브 = **Zen 3 AVX2 가능**(알고리즘/레이아웃). SLAB attention의 통합 AVX-512 디스패처 자체는 **Zen 4+ 필요** — 우리는 진행 중인 flash SIMD(AVX2)로 자체 구현 중이므로 구조(온라인 softmax·타일 분할)만 참조.
- **기대 기제(추정)**: 패킹·융합으로 사이트 내 잔여(~10.2 ms)의 상당분 축소 — 잔여가 디스패치 굵기에 반비례한다는 우리 대조군(Q6_K 0.29 ms)과 정합. 1.60×는 BF16/VNNI 하드웨어(EPYC 9755) 기준 참고치.

## 2. PARD 병렬 드래프트 스펙 디코딩 — 디코드의 산술 강도를 k배 올리기 (AMD 실측 3–4×)

- **요지**: AMD의 스펙 디코딩 노선. 자회귀 디코드는 메모리 대역폭 바운드라 가중치를 k토큰 검증에 재사용해 산술 강도를 k배 올림. PARD는 드래프트를 시퀀스 축이 아닌 **병렬(배치) 축으로 세워 드래프트의 직렬 지연을 제거**, K=12 스펙 토큰으로 3–4× — Llama-3.1-8B 380 tok/s(듀얼 EPYC 9755). PACE의 멀티토큰 디코드는 PARD용 최적화와 명시 연계.
- **출처**: PARD 소개 https://www.amd.com/en/developer/resources/technical-articles/accelerating-generative-llms-interface-with-parallel-draft-model-pard.html (2025-04); PARD+PACE https://www.amd.com/en/developer/resources/technical-articles/2025/speculative-llm-inference-on-the-5th-gen-amd-epyc-processors-wit.html (2025-07-23); 제품 페이지 https://www.amd.com/en/products/processors/server/epyc/ai/9005-inference.html
- **병목 매핑**: **⑤ lm_head 대역폭 벽을 포함한 스텝 고정비 전체**(직렬 quant 4.03 + 오케스트레이션 잔여 ~10.2 + flash 고정분)를 수용 토큰 k로 나눔. k=4 수용이면 고정비 1/4. 스텝 수 자체가 줄어 파킹·스프레드 문제도 함께 상각.
- **이식성**: **Zen 3 AVX2 가능** — 순수 알고리즘(드래프트 모델·검증 배치). 검증 pass는 N=k 배치가 되어 matmul이 두꺼워져 %STREAM도 개선(디스패치당 바이트 법칙에 순풍).
- **기대 기제(추정)**: DeepSeek-V2-Lite Q3_K T=1에서 자기회귀 스펙(작은 드래프트) 또는 PARD식 병렬 드래프트로 수용률 β·k에 따라 **1.5–2.5×**. AMD 수치(3–4×)는 BF16 하드웨어·대형 모델 기준 상한 참고.

## 3. llama.cpp ZenDNN 백엔드 — Milan(Zen 3) 명시 지원, Q8_0 동적 양자화 경로

- **요지**: AMD가 llama.cpp에 넣은 공식 CPU 백엔드(ggml-zendnn). **MUL_MAT와 MUL_MAT_ID(MoE)만** 가속하고 나머지는 표준 CPU 폴백. 데이터타입 FP32/BF16(Zen 4/5 전용, 구형은 FP32 폴백)/**Q8_0(활성값을 동적으로 Q8_0 양자화 후 정수 matmul)**. 지원 CPU에 **EPYC 7003 = Milan = Zen 3 명시**. 권장: `ZENDNNL_MATMUL_ALGO=1`(Blocked AOCL DLP), NUMA에서 `numactl --cpunodebind=0 --membind=0`. 성능 "1.1–2×"(문서), RFC 벤치에서 F32 Qwen2-7B tg 최대 2.95×(EPYC 9004 96스레드) — 단 **MoE(Mixtral)는 이득 미미**, Q8_0 이득은 주로 프리필.
- **출처**: 문서 https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/ZenDNN.md (열람 2026-09, 문서 등록 2025-12경); RFC 토론 https://github.com/ggml-org/llama.cpp/pull/17684 + PR #17690 "ggml-zendnn: add ZenDNN backend for AMD CPUs" (2025-12-02, z-vishal); RFC 주간 보도 https://buttondown.com/llama.cpp (2025-12-01 주)
- **병목 매핑**: ④ Q8_0 — "활성값을 Q8_0로 동적 양자화해 maddubs류 정수 GEMM"이 AMD의 공식 경로라는 **외부 확인**(우리 진행 중 Q8_0 maddubs와 동일 방향; 신규 작업 아님). MUL_MAT_ID = ⑥ 전문가 게더를 matmul 연산 자체로 흡수(아이디어 4와 연결).
- **이식성**: **Zen 3 AVX2 가능**(Milan 지원이 명시). 단 Q3_K/Q4_K 등 K-quant는 가속 대상 밖이라 우리 양자 체계에 직접 갖다쓰는 이득은 제한적 — **기법 참고용**.
- **기대 기제**: 직접 채택보다 두 가지 교훈 — (i) 동적 Q8_0 경로의 정당성, (ii) MoE에서 별도 게더/스캐터 패스가 왜 느린지 AMD도 MUL_MAT_ID로 회피했다는 사실.

## 4. ZenDNN 6.0 — MoE 그룹 GEMM 원시연산(group_matmul_direct)과 FP16 (2026-07)

- **요지**: ZenDNN 6.0의 축은 ① **MoE 모델 최적화**(LowOHA의 `group_matmul_direct` — 전문가별 그룹 GEMM을 단일 직접 호출로), ② FP16 기능 지원(6세대 EPYC 대상), ③ vLLM 호환 창 확대. 5.2/5.2.1이 vLLM V1 플러그인+양자화 지원으로 "x86 CPU AI 추론 재정의" 노선을 깐 뒤, 6.0이 MoE·저정밀로 확장.
- **출처**: https://www.amd.com/en/developer/resources/technical-articles/2026/zendnn-6-0-fp16-inference-and-moe-acceleration.html (2026-07-10); ZenDNN 5.2 https://www.amd.com/en/developer/resources/technical-articles/2026/zendnn-5-2-accelerating-vllm-inference-on-amd-epyc-cpus.html (2026-03-13); 5.2.1 https://www.amd.com/en/developer/resources/technical-articles/2026/zendnn-5-2-1-on-amd-epyc-cpus.html (2026-04-24); 지원 매트릭스(EPYC 3–5세대 명시) https://www.amd.com/content/dam/amd/en/documents/developer/version-5-2-1-documents/zendnn/zendnn-5.2.1-support-matrix.pdf (2026); 저장소(LowOHA API 목록) https://github.com/amd/ZenDNN
- **병목 매핑**: ②⑥ MoE 배치 사이트 — 전문가 6개 gate/up/down의 게더(`moe_expert_io` 0.303 ms)·얇은 전문가 디스패치를 그룹 GEMM 하나로 흡수하면 디스패치 수 자체가 줄고 배치가 두꺼워짐.
- **이식성**: 라이브러리는 EPYC 중심(3세대까지 지원하나 BF16 등 핵심 경로는 **Zen 4+ 필요**); `group_matmul_direct` **아이디어 자체는 Zen 3 AVX2 가능** — 우리 matmul_q_batch에 전문가 인덱스를 인자로 받는 그룹 변형을 추가하는 형태. FP16 경로는 6세대 EPYC 대상 = **무관**.
- **기대 기제(추정)**: 게더 접착부 ~0.3 ms 직접 절감 + 전문가 디스패치 굵기 증가로 batch Q3_K/Q5_0(활용도 34–35%)의 잔여 축소에 기여 — 합쳐 1 ms/step 남짓(보수 추정).

## 5. ZenDNN 5.2의 배치 정책 — "와이드 패브릭은 감익한다: 128코어 1개 < 64코어 2개"

- **요지**: 단일 vLLM 인스턴스가 128코어 전부를 쓰면 "그토록 넓은 컴퓨트 패브릭 관리 복잡성과 메모리 경합"으로 체감 성능이 감익한다는 AMD 공식 진단. 해법: **코어를 인터리브(짝수 코어)로 묶어 2×64코어 인스턴스로 분할 + membind** → 동기화 오버헤드 감소·DRAM 대역폭 포화. 또한 `TORCHINDUCTOR_FREEZING=1`(파라미터 동결)로 메모리 풋프린트·**L3 캐시 지역성** 개선. vLLM V1 플러그인으로 네이티브 vLLM-CPU 대비 최대 2배+ 향상(모델별 상이, AMD 인용).
- **출처**: https://www.amd.com/en/developer/resources/technical-articles/2026/zendnn-5-2-accelerating-vllm-inference-on-amd-epyc-cpus.html (2026-03-13)
- **병목 매핑**: ② 오케스트레이션 — 풀 9.1/32 가동의 거울상. 우리는 단일 모델이라 다중 인스턴스가 아니라 **(a) 워커 수 스윕(32 미만), (b) CCD별 행 소유권 고정, (c) 청크 할당을 CCD 4개의 L3 정합**으로 번역해야 함. FREEZING의 교훈 = 변하지 않는 텐서를 지역성 좋게 배치(우리 활성값/KV가 작아 이미 유리하나 접착부 텐서 재할당 점검거리).
- **이식성**: **Zen 3 AVX2 가능** — 순수 정책/스케줄링.
- **기대 기제(추정)**: 저비용 실험(스레드 수만 바꿔 측정). 원장 §5 파킹 분석(스핀 20,000회 예산)과 맞물려 파킹 빈도가 줄면 잔여 감소 — 수치 근거는 미확보, 측정 필요.

## 6. llama.cpp NUMA/스레드 현장 지식 — 스레드 수 스윕과 행 소유권(퍼스트터치)

- **요지**: llama.cpp 커뮤니티(2025-03 개시, 2026-02까지 활발)의 합의: 대형 CPU에서 **총 코어수보다 적은 스레드가 최적**인 일이 흔함(32코어 중 -t 20, 192코어 중 -t 48 사례) — "단일 타일 분량의 스레드를 넘으면 감응". 유지자(slaren) 제안: **가중치 행을 NUMA 노드로 분할하고 소유 노드의 스레드만 그 행을 처리**(mmap 퍼스트터치보다 확실). `--numa distribute`는 콜드 페이지캐시에서만 제대로 동작. VTune 관측: 잘못된 노드의 KV 페이지로 10k ctx에서 13→4 tok/s. "EPYC/Xeon은 MoE 전문가/FFN에 필요한 대역폭은 충분하나 어텐션(2차)이 약점" — 하이브리드 필요 주장.
- **출처**: https://github.com/ggml-org/llama.cpp/discussions/12303 (2025-03-10 개시, 2026-02까지 활동)
- **병목 매핑**: ② 오케스트레이션/파킹 — 스레드 수 스윕은 즉시 실험 가능. 행 소유권 고정은 단일 NUMA 박스에서는 효과 제한적이나 **CCD 단위 친화도(L3 32MiB×4 정합)**로 재해석 가능.
- **이식성**: **Zen 3 AVX2 가능**.
- **기대 기제(추정)**: 부정적 결과 가능성을 포함한 측정 과제. Leaseweb EPYC 벤치(듀얼 9334, LLM에 24스레드 사용, CPU 활용 20–30% — https://blog.leaseweb.com/2026/04/05/amd-epyc-llm-inference-benchmark-cpu-vs-gpu/ , 2026-04-05)가 "LLM 디코드에서 낮은 스레드 효율"이 범용 현상임을 뒷받침.

## 7. AOCL 5.1 / aocl-dlp — 저정밀 LPGEMM은 사실상 Zen 4+ (BF16-on-AVX2 표방의 함정)

- **요지**: AOCL 5.1(2025-05)은 Zen 4/5 최적화 커널, 저정밀 배치 GEMM API(LPGEMM), "AVX2에서 BF16 지원"을 내걸었으나, AMD GitHub 이슈(2025-07-02)에서 **BF16 경로는 AVX-512를 먼저 고르고 없으면 다운그레이드 대신 실패/크래시**한다는 보고 — Zen 3는 BF16 네이티브가 아예 없다(공식 답변: "Zen4부터 추가"). BLIS의 BF16 마이크로커널은 JIT 생성되나 AVX512_BF16 하드웨어 게이트.
- **출처**: AOCL 5.1 발표 https://www.amd.com/en/developer/resources/technical-articles/2025/what-s-new-in-aocl-5-1--faster--and-more-scalable-math-libraries.html (2025-05-21); 반박 이슈 https://github.com/amd/aocl-dlp/issues/2 (2025-07-02); 저정밀 원시연산 저장소 https://github.com/amd/aocl-dlp ; BLIS LPGEMM 배경 https://www.cs.utexas.edu/users/flame/ (발행일 미상)
- **병목 매핑**: 직접 공격 사이트 없음 — 채택 불가 판정 자체가 산출물. BLIS의 **GEMV/블로킹·스레딩 정책(친화도)** 문서만 일반 참고 가치.
- **이식성**: **Zen 4+ AVX-512 필요**(BF16·INT8/VNNI 경로 전부). 우리 박스에는 AOCL 저정밀 = **무관**. 교훈: "AVX2에서 돈다"는 표방도 실행 계획(AVX-512 경로 선택)을 열어봐야 확인된다.
- **기대 기제**: 해당 없음(판정 기록).

## 8. EPYC 9005 AI/ML 튜닝 가이드(문서 #58482) — BIOS/OS 노브

- **요지**: AMD 공식 EPYC 9005(Turin) AI/ML 튜닝 가이드: 결정론(Determinism = Power), APBDIS(부스트 비활성화), cTDP, NPS(NUMA-per-socket) 등 BIOS 권장 + 워크로드별 설정. Phoronix가 이 가이드 설정의 AI/ML 성능 영향을 실측 소개(2024-11-29). docs.amd.com 포털은 JS/로그인 벽으로 본문 열람 불가(존재와 문서번호는 확인).
- **출처**: https://docs.amd.com/r/en-US/58482 (개정일 미상, 2024-12 발행 계열); Phoronix "AMD BIOS Tuning Guide Impact For Boosting AI/ML Performance On EPYC 9005 Series" https://www.phoronix.com (2024-11-29; 본문 403으로 상세 미확인); 보조: https://lenovopress.lenovo.com (2026-04경), https://documentation.suse.com (EPYC 9005 튜닝)
- **병목 매핑**: 간접 — 원장 §5의 파킹 가설 (i)(부스트→정상 클럭 전환)과 연결되는 관찰거리(부스트 거동이 표준화되면 스텝 스프레드 35–50%의 일부 설명 가능).
- **이식성**: 기타(환경) — WR 플랫폼에도 유사 노브(Determinism/부스트) 존재하나 박스 미접촉 원칙상 기록만.
- **기대 기제(추정)**: 낮음 — STREAM 147.7 GB/s로 이미 환경은 측정 완료. 스프레드 원인 규명에만 참고.

## 9. 명시적 무관 항목 (조사 중 배제 근거)

- **Ryzen AI 소프트웨어 스택의 NPU 부분**: NPU 가속 = 우리 박스에 NPU 없음 → 무관. CPU 코어 관련 부분은 ZenDNN(zentorch)으로 수렴(https://www.amd.com/en/developer/zendnn.html , 2026 열람).
- **ZenDNN 6.0 FP16**: 6세대 EPYC(Venice) 대상 기능 — Zen 3 무관.
- **APEX(arXiv, 2026-01)**: CPU-GPU 비동기 분리(디코드 어텐션/KV를 CPU로) — GPU 전제라 무관 (https://arxiv.org/abs/2512.16473 인접 문헌; CPU-GPU 혼합 계열).
- **Intel AMX 계열 MoE CPU 서빙**(sglang/lktransformers 등): AMX 필요 — llama.cpp 토론 #12303 내 언급으로만 존재 확인.
- **"Trillion-Parameter LLM on Ryzen AI Max+ Cluster"(2026-02)**: llama.cpp RPC 클러스터링 가이드 — 다중 기계 분산이라 무관.

---

## 우리 박스 기준 기대가치 순위 (top 5)

| 순위 | 아이디어 | 공격 병목 | 이식성 | 기대(근거) |
|---|---|---|---|---|
| 1 | **스펙 디코딩(PARD식 병렬/자기회귀 드래프트 + k토큰 검증 배치)** — §2 | ⑤ lm_head 벽 포함 **스텝 고정비 전체**(quant 4.0 + 잔여 10.2 + flash 고정분)를 k로 상각; 검증 pass는 두꺼운 배치 matmul이 되어 디스패치당 바이트 법칙에 순풍 | Zen 3 AVX2 가능(순수 알고리즘) | AMD 실측 3–4×(BF16 EPYC) / 우리 추정 **1.5–2.5×**(수용률·k 의존, 추정). 원장 §7의 커널·패킹 계열과 직교하는 유일한 대형 레버 |
| 2 | **가중치 사전 패킹(청크 패널 정렬) + QKV/인라인 융합(PACE 방식)** — §1 | ② 오케스트레이션: 디스패치당 바이트 법칙의 잔여(~10.2 ms). AMD도 같은 진단("런치 오버헤드")으로 1.60× | Zen 3 AVX2 가능(레이아웃/스케줄링) | Q6_K·Q5_1 대조군(잔여 0.29/0.02 ms)이 상한 입증 — 얇은 사이트(Q4_K 53회, Q3_K 108회)에 잔여 대부분이 있음 |
| 3 | **MoE 게더/스캐터의 그룹 GEMM 흡수(MUL_MAT_ID·group_matmul_direct·GQA 네이티브)** — §3·§4 | ⑥ 접착부(moe_expert_io 0.303 + wv_b 게더 0.416 ms) + 전문가 디스패치 굵기 | Zen 3 AVX2 가능 | AMD가 MoE를 전용 원시연산으로 뺀 이유와 동일 — 합쳐 ~1 ms/step(보수 추정) |
| 4 | **워커 수 스윕 + CCD(L3) 정합 친화도** — §5·§6 | ② 파킹/감익(풀 9.1/32) | Zen 3 AVX2 가능, **거의 공짜 실험** | AMD(128<2×64)·llama.cpp(-t 20/32)·Leaseweb(활용 20–30%) 3중 제보 — 수치는 미확보, 측정 과제 |
| 5 | **(외부 확인) Q8_0 동적 양자화·maddubs 경로** — §3 | ④ q_nope2 | 진행 중과 동일 작업 | AMD 공식 백엔드가 같은 선택 — 신규 이득 없음, 방향 정당화만 |

**총평(원장과의 정합성)**: AMD 2025–2026 자료 전체가 우리 원장의 진단과 동일한 그림을 그린다 — "커널은 이미 대역폭 근처, 남은 것은 런치 오버헤드(오케스트레이션)와 고정비 상각(스펙)이다". 차이는 AMD가 이를 BF16/AVX-512 하드웨어로 푸는 반면 우리는 AVX2+레이아웃+스케줄링으로 풀어야 한다는 점. 반대로 BF16·VNNI·FP16 경로(AOCL 저정밀, ZenDNN 6.0 FP16)는 전부 Zen 4+ 게이트로 확인되어 우리 박스에서는 배제 확정.

---
*검증 한계 메모: amd.com 본문 일부(PACE·PARD·ZenDNN 5.2)는 웹리더 판독, ZenDNN 6.0 세부는 발표문+검색 스니펫 교차 확인 — 벤치 수치 인용은 각주에 표기한 신뢰도로 판단할 것. docs.amd.com 58482는 포털 벽으로 2차 출처(Phoronix/레노보/SUSE)로만 존재 확인. Phoronix 본문은 403으로 제목·날짜만 확인.*
