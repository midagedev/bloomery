# 광역 생태계 조사 — 2025-2026 CPU 기반 LLM 추론의 새 기법 (bloomery 리서치)

작성 2026-09-20. 범위: Intel/AMD 공식 자료 밖의 생태계 (llama.cpp 본류·ik_llama.cpp 포크·학계/산업 논문). 방법: 웹 전용(WebSearch/WebFetch), 박스 미접촉·bloomery 저장소 미기록 (유일한 예외: 처지 공유용 `docs/RESULTS-mul35-saturation.md` 열람). 이식 필터: Zen 3 / AVX2+FMA+F16C+BMI2+VAES+VPCLMULQDQ, 32코어 단일 NUMA, L3 128MiB, STREAM 147.7 GB/s.

**우리 원장 기준 요약** (RESULTS-mul35-saturation.md): N=96 디코드 31.9 ms/step = 41.5 GB/s (STREAM 28%), 평균 가동 워커 9.1/32. ik는 82.78 tok/s = **12.07 ms/step ≈ 108 GB/s (STREAM ~73%)** — 같은 기계·같은 ISA·(원장 §1) 거의 같은 커널률(Q4_K 82-86%, Q5_0 91%, Q6_K 95%, Q5_1 98-99%)에서. 즉 ik는 같은 1.31 GB/step을 우리의 ~2.6배 속도로 흘려보낸다. 커널이 아니라 **어떻게 풀코어를 물려서 대역폭을 채우는가**의 차이다.

---

## 1. 최우선: ik_llama.cpp의 2.6배는 어디서 나오는가 — 정황 증거

### 1.1 결정적 정황: ik의 PR 벤치마크 기계가 우리와 동일 기종이다
- ik_llama.cpp PR #332 (2025-04-17)의 벤치마크 머신은 명시적으로 **"Ryzen-5975WX CPU (vanilla AVX2)"**, `-t 32 -fa`, Q8_0 KV. 우리와 같은 기계·같은 ISA에서 잰 수치라 그대로 비교 근거로 쓸 수 있다. URL: https://github.com/ikawrakow/ik_llama.cpp/pull/332
- 이 PR에서 LLaMA-3.1-8B TG: ik 16.5 → 11.7 t/s (ctx 0→16k) vs mainline 16.9 → 6.5 ("Mainline TG is 55.5% of ik_llama.cpp at 16k tokens"). ctx가 길어질수록 격차가 벌어진다 — KV 접근 경로(FA)의 우열이 TG 격차의 주원인이라는 우리 원장 H2 판정(§6: flash가 #1 사이트 18.8%, 스프레드 전부)과 동일 구조.

### 1.2 기제 1: CPU FlashAttention을 GEMV가 아니라 GEMM으로, 그리고 스레드 분할 축을 바꿔다
- PR #332: "The gains without FA are fairly minor and come from **a different way of distributing the work between the threads for the K\*Q and V\*softmax(K\*Q) matrix multiplications**" — FA 없이도 K·Q / V·softmax 두 내적의 **스레드 분배 방식**만 바꿔서 이득. FA 있을 때는 "very significant", **TG에서 GEMV 대신 GEMM을 쓰도록 구현을 교체**. URL: https://github.com/ikawrakow/ik_llama.cpp/pull/332
- N=96 배치에서 Q는 96행짜리 행렬 — ik는 이것을 GEMM(96×d·d×keys)으로 처리한다. 우리 flash_attn_latent는 원소별 스칼라 f32 mul_add(0.28 GB/s, 259 ns/행-키)다. ik의 108 GB/s 총성능과 우리의 41.5 GB/s 차이의 최대 단일 항목.
- PR #410 (2025-05-13, "Better CPU FA performance for DeepSeek-Lite"): DS-Lite Q4_0, 7950X, `-mla 3 -fmoe -fa -rtr` — TG가 ctx에 비례해 개선 (8192: +12.9%, 15360: +24.7%, 22.40→27.93 t/s). **DeepSeek-Lite(MoE+MLA)의 latent FA 경로가 바로 우리와 같은 그래프**라는 직접 증거. URL: https://github.com/ikawrakow/ik_llama.cpp/pull/410

### 1.3 기제 2: MLA 그래프를 'TG 최적'으로 유지 — PP 최적 전환은 배치에서 독이다
- PR #282 (2025-03-23): MLA에서 n_tokens > n_head이면 'PP 최적' 그래프(Dk=192/Dv=128, 추가 matmul 2개)로 전환하는데, 배치 TG에서 이것이 "clearly kills performance". ik는 전환 임계를 128 토큰으로 올렸고 "'TG optimized' is better than 'PP optimized' for prompt lengths up to 64 tokens". DeepSeek-R1 tg128: 기본 2.84 t/s ↔ 강제 PP-최적 **0.99** t/s ↔ 강제 TG-최적 2.87. URL: https://github.com/ikawrakow/ik_llama.cpp/pull/282
- 우리 N=96은 128 미만 → ik는 이 구간에서 **latent FA(Dk=576/Dv=512) 그래프**를 그대로 쓴다. 우리 엔진도 latent 경로를 쓰므로 구조는 일치 — 차이는 그 내부가 SIMD GEMM이냐 스칼라냐.

### 1.4 기제 3: MoE 게이트/업 융합 — 활성화 양자화 1회 공유 + 장벽(barrier) 감소
- PR #229 (2025-02-23, `-fmoe`): up·gate·activation을 단일 ggml op로 융합. 핵심 인용: 세 mul_mat_id가 각자 전문가를 탐색하고 복사하던 것을 한 번으로 줄이고, gate와 up이 **"(already quantized) second operand를 공유"** 해 중복 양자화 제거, "There is a barrier after each operation"였던 것의 장벽数 축소. CPU 이득은 TG 1-2% (작지만 방향이 우리와 동일). URL: https://github.com/ikawrakow/ik_llama.cpp/pull/229
- 우리 원장 §3·§7: 게이트/업에서 **같은 xb를 두 번 양자화(312쌍 중 156 중복)**, 직렬 quant 4.03 ms, swiglu 직렬 1.37 ms — ik가 커널 안에서 이미 없앤 바로 그 비용들.

### 1.5 기제 4: ggml 구조 — 텐서당 1노드·스레드당 연속 행 청크·노드별 장벽
- ik를 포함한 ggml 계열은 레이어 텐서 하나 = 그래프 노드 하나: 배치 96이면 M=96 GEMM 한 번이 32스레드에 **행(N) 방향으로 연속 분할**되고(https://github.com/ikawrakow/ik_llama.cpp — iqk 커널 구조, Codeberg 미러: https://codeberg.org/ikawrakow/illama), 노드 사이만 장벽으로 동기화한다. 즉 ffn_down_exps 309 MB 사이트는 스레드당 ~10-12 MB 연속 스트림이 32개 병렬 — 우리 원장의 예측 변수 '디스패치당 바이트'(Q6_K 172 MB→119 GB/s, Q5_1 16.8 MB→75 GB/s, 2-10 MB→47-51 GB/s)의 포화 영역에 ik 전체가 놓여 있다. 96×2048 양자화 활성화(~100-200 KB)는 L1/L2에 상주하므로 스레드는 순수하게 웨이트를 스트리밍한다.
- 우리 엔진은 같은 사이트를 2-10 MB짜리 53-108회 디스패치로 쪼개 풀에 흘리고(스핀 20,000회 후 파킹) — 사이트 내 오케스트레이션 잔여 ~10.2 ms/step의 직접 원인. ik와의 19.8 ms 격차에서 flash SIMD(≈5)를 빼면 나머지 대부분이 이 구조 차이다.
- ik 위키 (2024-07, TG 비교): Q4_1 AVX2 2스레드 5.11→11.45 t/s(2.24배), Q5_1 1.96배, Q8_0 1.57배 — iqk_mul_mat 커널 자체도 낮은 스레드 수에서 2배급. 단 "TG performance being limited by memory bandwidth", 라이젠은 4스레드에서 피크. URL: https://github.com/ikawrakow/ik_llama.cpp/wiki/July-2024:-Token-generation-performance-comparison
- 커뮤니티 측정 (2025-12, 이슈 #1083): "ik_llama.cpp is much, much faster in pp (5x even) and slightly faster in tg (20%)" — 단일 시퀀스 기준; N=96 배치에서는 위 1.2-1.5 기제가 커진다. URL: https://github.com/ikawrakow/ik_llama.cpp/issues/1083

### 1.6 기제 5: 스레드 정책 문서
- ik parameters.md: `-t` "Try to match the number of physical CPU cores. Avoid odd numbers", `-tb` 배치/프롬프트용 별도, `-fa` 기본 on, `-rtr` 런타임 리팩(인터리브 변형이 있으면), `--defer-experts`(전문가 mmap 상주 지연 — 로드 시간), CUDA 한정 `GGML_CUDA_HOST_MALLOC_THP`(THP). URL: https://github.com/ikawrakow/ik_llama.cpp/blob/master/docs/parameters.md
- TG 튜닝 문서: 오버새추레이션 경고, "start with 1 and double". URL: https://github.com/ikawrakow/ik_llama.cpp/blob/master/docs/development/token_generation_performance_tips.md

**2.6배 종합 (정황 증거의 합)**: (i) latent FA의 SIMD·GEMM·스레드-분할 재설계(§1.2-1.3, 우리 flash 6.01 ms + 스프레드 기울기 0.112 ms/키의 상당분), (ii) 텐서당-1노드·연속 행 청크 GEMM 구조로 32코어 전역을 대역폭 포화(§1.5, 잔여 10.2 ms의 상당분), (iii) 커널 내 활성화 양자화·전문가 IO 융합(§1.4, 직렬 4-7 ms 분), (iv) 낮은 스레드 수에서도 1.5-2.2배인 iqk 커널 자체(§1.5 위키). 커널률이 91-99%로 같으므로 (iv)는 작고, (i)+(ii)+(iii)이 주축 — 원장 §7의 예측(flash SIMD + 양자화 풀 이양으로 ~21 ms, 잔여가 오케스트레이션)과 독립적으로 일치한다.

---

## 2. 아이디어 카드

### A. flash를 '키-분할 GEMM + F16C'로 — 진행 중 라운드의 설계 보강자
1. 요지: CPU FA의 TG 병렬화는 K·Q와 V·softmax의 **스레드 분배 축 설계**가 핵심이고, 배치(N=96)에서는 GEMV가 아니라 GEMM 경로를 탄다.
2. 출처: https://github.com/ikawrakow/ik_llama.cpp/pull/332 (2025-04), https://github.com/ikawrakow/ik_llama.cpp/pull/410 (2025-05)
3. 병목: flash_attn_latent (6.01 ms/step, #1 사이트, 0.28 GB/s 스칼라 벽, 스프레드 35-50%의 전부).
4. 이식성: **가능** — ik가 같은 "vanilla AVX2" 5975WX에서 구현·측정. F16C로 half→f32 변환을 벡터화하는 것도 동일 커널족.
5. 기대: 원장 예측 ~5.0 ms 절감(6.01→~1.0). 추가 단서: DS-Lite 개선은 ctx가 길수록 커진다(+24.7% @15k) — 우리 ctx 기울기 0.102 ms/토큰의 대부분이 여기서 온다.

### B. '텐서당 1디스패치, 스레드당 연속 행 청크' 패킹 — 잔여 10.2 ms의 직접 타깃
1. 요지: ggml식으로 같은 k의 사이트를 하나의 노드로 묶어 32스레드에 행 방향 연속 청크(십수 MB)로 나누고 노드 경계에만 동기화하면 '디스패치당 바이트' 포화곡선의 최상단(75-119 GB/s)에 근접한다.
2. 출처: https://github.com/ikawrakow/ik_llama.cpp/pull/229 ("There is a barrier after each operation" — 장벽 구조 언급), https://github.com/ikawrakow/ik_llama.cpp/wiki/To-tile-or-not-to-tile (타일링 전략, 2024-07), https://github.com/ikawrakow/ik_llama.cpp/pull/332 (-t 32 벤치마크)
3. 병목: 사이트 내 오케스트레이션 잔여 ~10.2 ms/step (Q4_K 53호출 3.118 ms, Q3_K singles 108호출 4.582 ms, 활용도 21-35%).
4. 이식성: **가능** — 순수 스케줄링, ISA 무관. 원장이 이미 Q6_K(0.29 ms 잔여)·Q5_1(0.02 ms)로 '굵은 디스패치의 달성 가능성'을 내부 증명.
5. 기대: 상한 ~10 ms (원장 §7 진단 항목). k 혼재(Q4_K 2,048/2,816) 결합 불가 사이트는 세그먼트 통합이라도. ik 전체 108 GB/s=STREAM 73%가 실증 상한선.

### C. 활성화 양자화의 '커널 내 1회화' + gate/up 공유 — 진행 중 라운드의 설계 반영점
1. 요지: gate·up이 같은 입력을 공유하면 양자화된 두 번째 피연산자를 만들어 **한 번만** 만들어 공유하고, 전문가 탐색·행 복사도 세 연산에서 한 번으로 줄인다.
2. 출처: https://github.com/ikawrakow/ik_llama.cpp/pull/229 (2025-02) — "(already quantized) second operand" 공유, CPU TG 1-2%/PP 3-4%
3. 병목: 직렬 activation quant 4.03 ms/step + moe_expert_io 0.303 + moe_trace 0.320; 그중 Q5_0 q8_2_x4 인코더 1.55 ms(값당 ~7-8 ns).
4. 이식성: **가능** — 데이터 흐름 재배치만. 진행 중인 '직렬 quant 풀 이양'에 더해, **게더 후 공유 입력당 1회 인코딩**을 에필로그가 아니라 프로로그에.
5. 기대: 원장 상한 ~3.5 ms(풀 이양) + 중복 156쌍 제거로 batch Q3_K quant 절감. ik 수치(1-2%)는 단일 시퀀스 기준이라 우리(직렬 비중 12.6%)에서는 이득이 더 크게 나올 여지.

### D. KV 캐시 q8_0 양자화 (-ctk/-ctv 상당)
1. 요지: KV를 f16→q8_0로 저장해 flash 읽기 바이트를 절반으로; ik의 DS-Lite CPU FA 벤치마크도 Q8_0 KV 조합으로 잡았다.
2. 출처: https://github.com/ikawrakow/ik_llama.cpp/pull/332 (2025-04, "Q8_0 KV cache" 벤치마크), https://github.com/ikawrakow/ik_llama.cpp/pull/410 ("improves DeepSeek-Lite CPU TG performance with Q8_0 KV cache")
3. 병목: flash (1,152 B/키 → ~576 B/키, ctx 기울기 b=0.1118 ms/키 절감) 및 장기적으로 KV 대역폭.
4. 이식성: **가능** — q8_0 KV dot은 AVX2 maddubs로 처리 가능(VAES 불필요). 단 우리 flash는 현재 대역폭이 아니라 지연 벽(0.28 GB/s)이므로 **A(flash SIMD) 선행 후에야 이득이 드러남**.
5. 기대: flash SIMD화 이후 ctx 기울기 ~절감(94스텝 누적 ~10 ms/런의 절반 이하로). 품질(MLA latent의 q8 근사)은 사전 검증 필요.

### E. Q8_0 내적의 maddubs '부호 트릭' — 진행 중 Q8_0 커널의 레시피
1. 요지: VNNI 없는 AVX2에서 int8 GEMM은 활성화를 부호 없는 바이트(y+128)로 먹이고 bias로 보정, 또는 maddubs 부호 트릭 — upstream llama.cpp가 이미 이 경로를 다듬었다.
2. 출처: https://github.com/ggml-org/llama.cpp/compare/ggml-org:30e1216...ggml-org:2bd5dac.diff — "feed the activations to VNNI as unsigned bytes (y + 128) and correct with bias[]; without VNNI the kernels use the maddubs sign trick"
3. 병목: q_nope2_absorbed Q8_0 (1.31 ms/step, 스칼라 2.7 GB/s/코어 = qdot의 16-25%).
4. 이식성: **가능** — 그 자체가 'VNNI 없는' 경로. Zen 3용.
5. 기대: 원장 순위 3, ~0.85 ms. ik의 AVX-VNNI PR군(README: PR 1446·1455·1467·1474·1482)은 Zen 3에 없어 불가 — 대신 이 트릭이 정답.

### F. 웨이트 리패킹/인터리브 (런타임)
1. 요지: Q4_0류를 AVX2 친화 인터리브 레이아웃으로 로드 시간에 재배치해 GEMV/GEMM 접근 패턴 정렬 — llamafile이 원조, upstream이 '온라인 리팩'으로 도입, ik는 -rtr.
2. 출처: https://github.com/ikawrakow/ik_llama.cpp/blob/master/docs/parameters.md (-rtr "May improve performance on some systems"), https://www.phoronix.com/news/Mozilla-Llamafile-0-8-2 (2024-05, AVX2 프롬프트 처리 1.4-2.3배), upstream 런타임 리팩 실재는 버그 리포트로 확인 (https://github.com/ggml-org/llama.cpp — "Q4_0 with runtime repacking", 2024-12)
3. 병목: Q4_K 53호출(2,048행 k-혼재)·Q5_0/Q5_1 GEMV류의 접근 패턴. 단 우리 최대 사이트들은 이미 68-70 GB/s(대역폭 46-48%)라 리팩보다 패킹(B)이 우선.
4. 이식성: **조건부 가능** — 인터리브 변형은 Q4_0/Q4_K 중심; K-quant(Q3_K·Q5_0·Q6_K)에는 ik에도 변형이 없음(README: k-quants는 인터리브 없음). 로드 시 1회 비용.
5. 기대: 낮음(수 %). B·A 이후 잔여 사이트 대상으로만.

### G. T-MAC / LUT-GEMM — 계산량을 LUT로 치환
1. 요지: 저비트 웨이트×활성화 MAC을 비트 단위 룩업테이블로 치환, 디퀀트 제거 — llama.cpp 대비 최대 4배·에너지 70% 절감 주장.
2. 출처: https://arxiv.org/abs/2407.00088 (T-MAC, EuroSys 2025, 2024-06/2025-03 개정), https://arxiv.org/abs/2206.09557 (LUT-GEMM, 2022)
3. 병목: 이론상 커널-한계 사이트(q_nope2). 단 우리 원장의 결론은 대부분 사이트가 대역폭/오케스트레이션 한계 — LUT는 **바이트는 줄이지 못하고 FLOP만 줄인다**.
4. 이식성: **부분** — pshufb 기반 LUT는 AVX2 가능(GFNI 불요), 단 Q8_0/K-quant가 아니라 2-4비트 전용 설계.
5. 기대: 낮음 — 우리 병목 구도(대역폭 28% 활용)와 정면으로 안 맞음. 기록용.

### H. KTransformers (SOSP'25) — CPU 전문가 GEMM 튜닝의 참고점
1. 요지: MoE 전문가를 CPU DRAM에 두고 AMX/INT8로 밀어낸 극단 사례. 프리필 286.55 t/s (2소켓 Xeon+GPU) vs llama.cpp 10.31 = 27.8배. 핵심은 **전문가 행렬의 NUMA-국소화·코어 수 맞춤 --cpu_infer·핫/콜드 전문가 배치**.
2. 출처: https://github.com/kvcache-ai/ktransformers (SOSP 2025 "Unleashing the Full Potential of CPU/GPU Hybrid Inference for MoE Models"; README: "AMX/AVX512/AVX2 optimized kernels for INT4/INT8", AVX2-only 백엔드 2026-03 추가), https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/DeepseekR1_V3_tutorial.md (numactl 노드 고정, USE_NUMA, --cpu_infer)
3. 병목: 없음 직접 — 단일 NUMA라 NUMA 기법 무효, AMX 불가. 참고점: **전문가 GEMM을 코어 수에 정확히 맞춘 상시 워커로 돌린다**는 운영 원칙만 우리 상주 풀과 공유.
4. 이식성: 대부분 **불가/무관** (AMX·NUMA). AVX2 백엔드 존재만 체크 포인트.
5. 기대: 없음(참고).

### I. 전문가 계층화(프리페치·캐싱·SSD) 계열
1. 요지: MoE 전문가를 CPU↔GPU/SSD 간 스트리밍하는 serving 연구는 활발 (DuoServe-MoE: 2단계 프리페치+캐싱, MoE-Infinity: 활성화 기반 캐싱, SSD 계층 에너지 분석).
2. 출처: https://arxiv.org/abs/2509.07379 (DuoServe-MoE, 2025-09, v2 2026-04), MoE-Infinity (ASPLOS 2024 — DuoServe 관련조작으로 확인), SSD 계층 분석 (arxiv, 2025-08, "SSD Offloading for LLM Mixture-of-Experts Weights")
3. 병목: 무관 — 우리 전문가는 전부 DRAM 상주(ffn_down_exps 309 MB/step가 최대 사이트지만 이는 대역폭·패킹 문제지 계층 문제가 아님).
4. 이식성: **무관**.
5. 기대: 없음. 단 N=96에서 토큰당 top-6 전문가의 **중복 제거·행 재그룹**(같은 전문가끼리 모아 한 GEMM으로)은 이 문헌의 '전문가-배치' 발상의 CPU판 — B와 같은 결론.

### J. THP/2MB 페이지 — 미검증, 저비용 A/B 후보
1. 요지: 대용량 웨이트 스트리밍의 dTLB 미스를 2MB 페이지로 줄이는 시도는 널리 권고되나, CPU-only llama.cpp 계열의 **공개 정량 벤치마크는 확보하지 못했다**. ik에 CPU용 노브는 없고 CUDA 호스트 버퍼의 `GGML_CUDA_HOST_MALLOC_THP` 환경변수만 확인.
2. 출처: https://github.com/ikawrakow/ik_llama.cpp/blob/master/docs/parameters.md (2025), https://docs.kernel.org/mm/transhuge.html (커널 문서 — 일반 근거)
3. 병목: 전 스트리밍 사이트의 실효 대역폭(특히 B 시행 후 측정).
4. 이식성: **가능** — mmap 영역에 madvise(MADV_HUGEPAGE). macOS가 아니라는 전제(리눅스 런타임일 때).
5. 기대: 불확실(수 % 미만 추정). '노벨'로 1회 A/B 측정 권고 — 우리 같은 스트리밍 패턴(1.31 GB/step)이 수혜 후보.

### K. 기타 확인 사항
- **"Pika"라는 CPU 디코딩 시스템 논문은 2025-2026 검색 범위에서 발견되지 않았다** (Pika Labs는 비디오 생성 회사로만 확인). 혹시 다른 철자/저자라면 재조회 필요.
- upstream llama.cpp: FA가 2025-09 기본 on(PR #15434)이지만 **CPU에서는 FA가 무이득·역이득이라 끄라는 컨센서스** (https://github.com/ggml-org/llama.cpp — 토론; Unsloth 모델카드도 "disable flash attention" 권고 확인). 즉 'FA 유무'가 아니라 'FA의 CPU 구현 품질'이 차이 — ik만 CPU FA를 진지하게 튜닝(README: PR 332·410·429 "Option to enable or disable the CPU FA kernels", 2025-05-17).
- upstream의 Q4_K '온라인 리팩'·ARM I8MM(https://developer.arm.com/community/arm-community-blogs/b/ai-blog/posts/optimize-llama-cpp-with-arm-i8mm-instruction, 2025-06)는 타 ISA 참고.
- FastDecode(KV 스트리밍)·InfiniGen류: GPU 중심이라 이 조사에서 제외.

---

## 3. 기대가치 Top 5 (우리 원장 §7과의 중복 제거 기준)

| 순위 | 아이디어 | 근거 사이트·수치 | 기대(유도) |
|---|---|---|---|
| 1 | **B: 텐서당-1노드·스레드 연속 청크 패킹** (§1.5) | 잔여 10.2 ms; ik 전체 108 GB/s(73% STREAM); 내부 대조군 Q6_K/Q5_1 | 상한 ~10 ms/step — 원장 진단 항목과 동일하나 ik의 실현 구조(노드=텐서, 행-분할, 노드별 장벽)가 구체 설계도를 제공 |
| 2 | **A: flash SIMD + GEMM 전환·분할축 보강** (§1.2-1.3) | flash 6.01 ms·0.28 GB/s; ik 5975WX AVX2 실측 | ~5.0 ms(진행 중이므로 증분은 ik PR #332의 '분할 축'·PR #410의 DS-Lite 특화에서) |
| 3 | **C: quant 1회화·gate/up 공유·전문가 IO 통합** (§1.4) | 직렬 quant 4.03 + expert_io 0.30 + trace 0.32; ik PR #229 구조 | ~3-4 ms(진행 중 라운드에 '공유 입력 1회 인코딩'·'전문가 탐색 1회' 추가) |
| 4 | **D: KV q8_0** (A 선행 후) | 1,152 B/키→~576; ik DS-Lite 벤치 조합 | ctx 기울기 절감(N=96 런 누적 수 ms) + KV 상주 L3 여유 |
| 5 | **E: q_nope2 maddubs(진행 중) + J: THP A/B(노벨)** | 스칼라 2.7 GB/s/코어; E는 upstream 레시피와 정합 | ~0.85 ms + (J은 미검증·측정 비용만) |

**전략적 결론**: ik의 2.64배는 '마법 커널'이 아니라 (1) FA의 SIMD·GEMM 재설계, (2) 노드=텐서·연속 청크·최소 장벽의 그래프 스케줄링, (3) 커널 내 융합(양자화·전문가 IO)의 합이다. 우리 진행 중 3건(flash SIMD, quant 풀 이양, Q8_0 maddubs)이 정확히 (1)(3)(보조)에 대응하며, 남은 미개척 영역은 (2) — 원장의 '배치 굵게/패킹' 후보와 동일하고, 이번 조사는 그 상한(73% STREAM)과 실현 형태(ggml 노드 구조)를 외부 증거로 못박았다.

---

## 4. 출처 색인
- ik_llama.cpp README: https://github.com/ikawrakow/ik_llama.cpp (Codeberg 미러: https://codeberg.org/ikawrakow/illama)
- ik PR #332 (2025-04-17, 5975WX AVX2, CPU FA TG GEMM): https://github.com/ikawrakow/ik_llama.cpp/pull/332
- ik PR #410 (2025-05-13, DS-Lite CPU FA): https://github.com/ikawrakow/ik_llama.cpp/pull/410
- ik PR #282 (2025-03-23, MLA TG/PP 그래프 임계): https://github.com/ikawrakow/ik_llama.cpp/pull/282
- ik PR #229 (2025-02-23, -fmoe 융합): https://github.com/ikawrakow/ik_llama.cpp/pull/229
- ik parameters.md / tg tips: https://github.com/ikawrakow/ik_llama.cpp/blob/master/docs/parameters.md · https://github.com/ikawrakow/ik_llama.cpp/blob/master/docs/development/token_generation_performance_tips.md
- ik 위키 TG 비교 (2024-07): https://github.com/ikawrakow/ik_llama.cpp/wiki/July-2024:-Token-generation-performance-comparison
- ik 이슈 #1083 (2025-12): https://github.com/ikawrakow/ik_llama.cpp/issues/1083
- upstream igemm maddubs/VNNI diff: https://github.com/ggml-org/llama.cpp/compare/ggml-org:30e1216...ggml-org:2bd5dac.diff
- llamafile AVX2 (2024-05): https://www.phoronix.com/news/Mozilla-Llamafile-0-8-2
- KTransformers: https://github.com/kvcache-ai/ktransformers · https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/DeepseekR1_V3_tutorial.md (SOSP 2025)
- T-MAC (EuroSys 2025): https://arxiv.org/abs/2407.00088 · LUT-GEMM: https://arxiv.org/abs/2206.09557
- DuoServe-MoE: https://arxiv.org/abs/2509.07379
- THP(커널 문서): https://docs.kernel.org/mm/transhuge.html
- Arm I8MM (2025-06): https://developer.arm.com/community/arm-community-blogs/b/ai-blog/posts/optimize-llama-cpp-with-arm-i8mm-instruction
