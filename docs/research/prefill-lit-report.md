# 프리필 문헌 조사 — 원문

2026-09-25. opus 서브에이전트 라운드 `prefilllit`의 보고 원문이다(스펙 `specs/wave-m5/spec-prefilllit.md`). 종합과 리드 확인 목록은 [`prefill.md`](prefill.md)에 있다. 이 원문의 [I] 수치는 조사자의 추론이고, 리드가 원본에서 확인한 것은 종합 문서가 따로 적는다.

## 1. 요약

- **V4.1의 가장 큰 프리필 레버는 모델 구조 자체에 있다.** 공식 논문(arXiv 2609.19969 §2.2, §3.2.2)에 따르면 V4.1은 Causal Encoder-Decoder(CED) 구조다. 디코더 20층(20–39)의 전역 KV는 인코더 출력 H_20에서 투영한다. 그래서 프리필은 하위 20층만 전 토큰에 돌면 된다. ik main은 40층 전부를 모든 프롬프트 토큰에 돈다(`src/graphs/build_deepseek4.cpp:1561@20f7a72`).
  - **정확한 쪽:** 디코더를 마지막 L/2×n_win = 2560 토큰에만 돌려도 ik 전체 계산과 같다. 층별로 필요한 범위가 줄어드는 삼각형 형태로 세면 절감이 더 크다[I]. P=4096에서 연산 1.54배, P=512에서 1.07배다.
  - **근사 쪽:** DeepSeek이 실제 배포하는 Decoder SWA Bounded Replay는 디코더를 마지막 128 토큰에만 돌린다. **이 모드는 출력을 바꾼다.** 논문이 "not mathematically equivalent"라고 적고 후학습에서 적응시켰다. 따라서 ik 오라클과는 밴드가 아니라 KL로 비교해야 하고, 채택은 사용자 결정이다.
- **호스트 티어는 P=512에서 대역폭과 연산의 경계에 걸리고, 스트리밍 교차점은 약 2300 토큰이다.** 같은 CPU(5975WX)에서 ik의 dense pp512 실측이 약 3.8–4.1 TOPS다. V4.1 호스트 전문가 212.9 GB를 한 번 읽는 데 1.5 s, 계산에 1.6 s가 든다[I]. 스트리밍 교차점은 ik 규칙(64×32 = 2048)과 거의 같다. P=4096에서는 CPU 계산과 전문가 스트리밍을 동시에 돌리고 끝나는 시각을 맞추는 분할이 약 820 tok/s 천장을 준다. CED 정확 판을 더하면 약 990이다[I].
- **같은 급 박스에서 공개된 V4.1 pp 숫자는 없다.** 가장 가까운 기준은 DeepSeek-V3/R1을 CPU 전문가로 돌린 ubergarm(7965WX Zen 4 + A6000, pp512 약 110)과 KTransformers v0.2.1(pp 101–112)이다. 둘 다 CPU 쪽 실효 연산이 약 4.5 TOPS로 우리 CPU급이다. V4.1로 환산한 선은 70–300 tok/s[I]이고, 오늘 43 tok/s의 2–7배다. Qwen3는 llama.cpp 4090 실측 7246에서 환산해 A6000 pp512 약 2850이 선이다[I].

## 2. 질문별 결과

표기 규칙: [M-paper] 논문 실측, [M-blog/docs] 공식 문서·README·메인테이너 PR 실측, [C-code] 코드 독해(path:line@commit), [I] 내 추론. 커뮤니티 글은 "(커뮤니티)"를 붙였다.

박스 상수는 로컬 문서에서 가져왔다. 호스트 읽기는 STREAM 147.7 GB/s, 엔진 132–137 GB/s다. PCIe H2D는 pinned 26.2 GB/s, pageable 14.4 GB/s다(`docs/research/v41-ports.md:38`, 어느 카드로 쟀는지 적혀 있지 않다). 두 카드 사이 P2P는 없다(`docs/gpu-design.md:37`). **`docs/facts.md`에는 PCIe 세대와 레인 수가 없다.** "4.0 ×16"은 `plan-triage.md:32`에만 있다.

카드 피크는 GA102 whitepaper Table 3과 Appendix에서 가져왔다[M-blog/docs].

| 카드 | INT8 텐서 dense | FP16 텐서, FP32 누산 | FP16 텐서, FP16 누산 |
|---|---:|---:|---:|
| A6000 | 309.7 TOPS | 154.8 TFLOPS | 154.8 TFLOPS |
| 3090 | 284 TOPS | 71 TFLOPS | 142 TFLOPS |

V4.1 상수는 다음과 같다. 40층, d 5120, 전문가 384개 중 top-6, 전문가 FFN 2304다(논문 §4.2.1). 전문가 하나는 16.77 MB다. 토큰당 routed 연산은 17.0 GFLOP이고, dense 연산은 약 13.8 GFLOP이다[I, `facts.md` 바이트에서 유도]. plan (a)는 카드에 전문가 2,668개를 둔다. 따라서 호스트 전문가 집합은 12,692 × 16.77 MB = 212.9 GB다[I].

Qwen3-30B-A3B 상수는 다음과 같다. d 2048, 48층, q 헤드 32개, kv 헤드 4개, 전문가 128개 중 top-8, 전문가 FFN 768이다(HF config.json). 프리필 토큰당 약 5.5 GFLOP에 attention이 더해진다[I].

### Q1. 하이브리드 CPU+GPU MoE 프리필

| 시스템 | 출처/연도 | 하드웨어 | 모델 | 프롬프트 | 프리필 결과 | 기제·교차 규칙 | 출처 |
|---|---|---|---|---|---|---|---|
| llama.cpp op offload | 코드 | — | 전부 | ubatch 크기 | — | 배치 토큰 수가 32 이상이면 CPU 가중치를 ubatch마다 GPU로 복사한다. MUL_MAT_ID는 토큰 수(ne[2])로 판정하므로 전문가당 토큰이 아니다. 기본 32, `GGML_OP_OFFLOAD_MIN_BATCH`로 바꾼다. 쓰인 전문가만 연속 구간으로 묶어 복사한다. ids를 동기화한 뒤 같은 스트림에서 복사하므로 겹침이 없다. mmap을 끄면 CPU 가중치가 pinned 호스트 버퍼에 놓인다 [C-code] | `ggml-cuda.cu:5608-5626,5797@84e76d8`; `ggml-backend.cpp:1693-1781@84e76d8`; `llama-model.cpp:1076-1090@84e76d8` |
| llama.cpp #27425 (커뮤니티) | 2026 | RTX 5060 Laptop, Ryzen AI 7 350 | Llama-3.1-8B, 절반 CPU | pp8–32 | 교차점: Q4_K_M 15.5 토큰, Q8_0 28.7 토큰 | 교차점이 가중치 바이트에 비례한다 [M-blog/docs] | issue #27425 |
| ik_llama.cpp | 코드 + PR #520 | 작성자 박스(명시 없음) | DeepSeek-Lite IQ4_KS, 전문가 CPU | pp4096 | ub128: 345 → 723, ub4096: 1878 → 2071 t/s | MoE는 배치×k ≥ 32×E일 때만 offload한다. 즉 전문가당 토큰 32 이상이다. V4.1은 2048, Qwen3-235B는 512 토큰이다 [C-code][M-blog/docs] | `ggml-cuda.cu:5227-5256@20f7a72`; `github-data/pull_requests/520…md` |
| 같은 PR 댓글 (커뮤니티) | 2025-06 | 2×3090 PCIe 3.0 x8, DDR4 2채널 | DS-V3 IQ1_S 130 GiB | pp512 / pp4096 | 스트리밍 시 pp512 27.9, pp4096 166 t/s | 스트리밍 경로는 ubatch당 비용이 일정해서 pp가 P에 비례해 는다 [M-blog/docs] | 같은 파일 |
| KTransformers | SOSP'25 | 2× Xeon 8452Y 36c, 소켓 내 220 GB/s, A100-40 / 4080-16, PCIe 4.0 | DS-3 Int4, DS-2 Int8, QW-2 | 32–8192 | 프리필 4.62–19.74배(대 Fiddler·llama.cpp). Fiddler식 기준선은 70.02 tok/s | 프리필 전문가를 CPU AMX로 계산한다(21.3 TFLOPS). 동적 스케줄링이 프리필 최대 1.83배다. Expert Deferral은 디코드 전용이다 [M-paper] | SOSP25-chen.pdf §1, §3.2, §3.3, §6.2, Fig. 11 |
| KTransformers 문서 | README | 2× Xeon 6454S, DDR5-4800 8ch×2, 4090D | DS-V3 Q4_K_M | 500 / 1K / 2K / 4K / 8K | v0.2 97.32(대 llama.cpp 10.31), v0.2.1 111 / 112.5 / 102 / 101, v0.3 AMX 286.55(2K) | 전문가 CPU, MLA/KV GPU [M-blog/docs] | DeepseekR1_V3_tutorial.md |
| KT AMX 문서 | README | Xeon4 + 4090 | Qwen3 MoE, DS-V3 | — | 프리필 최대 347 tok/s. llamafile 시절 DS-V3는 91 | AMX int8 35 TOPS. 전문가당 토큰 >4일 때 AMX, ≤4일 때 AVX-512 [M-blog/docs] | AMX.md:19,76,117,127 |
| kt-kernel 층별 프리필 | 코드 + 문서 | — | V4-Flash 등 | 임계 이상 | — | 임계 이상이면 층 하나 분량의 GPU 슬롯 2개를 쓰고, 층 N 계산 중에 N+1을 전송한다. 임계 이하면 하이브리드다. 임계는 AMX 튜토리얼 1024–4096, **AVX2 튜토리얼 400**, V4-Flash 도커 기본 2048이다 [C-code][M-blog/docs] | sglang `kt_ep_wrapper.py:2784-2820,5700-5760@541ddc3`; AVX2-Tutorial.md:124; DeepSeek-V4-Flash.md |
| kt Kimi-K2-Thinking | 문서 | 1×4090 48GB, 2× Xeon 8488C, DDR5-4800 | Kimi-K2 native INT4 | 2K / 8K / 30K | 53 / 184 / 290 tok/s | 층별 GPU 프리필 [M-blog/docs] | kt-kernel/Kimi-K2-Thinking-Native.md:141-150 |
| exllamav3 스트리밍 프리필 | 코드 | 보정: gen5 x16 / x8, gen4 x4, CPU 12스레드 | 512-expert 모델 | rows ≥ 32 | — | 전문가마다 count ≥ stream_t면 pinned 링 → VRAM 링으로 DMA한 뒤 GPU에서 dequant+GEMM한다. 꼬리는 CPU가 **동시에** 처리한다. stream_t = max(8, round(8·√(25/bw))). 실험적 pinned arena는 스테이저 memcpy 병목을 없앤다 [C-code] | `moe_cpu_host.py:101-117,1196-1220,1272-1308@6b84a21` |
| Fiddler | ICLR'25 | Quadro RTX 6000 + Xeon 6126 PCIe3; RTX 6000 Ada + Xeon 8480+ PCIe4 | Mixtral-8x7B fp16 | 512–4096 | 긴 프리필 TTFT 1.07배(대 DS-MII), 1.65배(대 Mixtral-Offloading) | 전문가마다 cpu_lat(s) > gpu_lat(s)+trans_lat()이면 GPU. cpu_lat는 s에 선형, gpu는 상수. CPU 커널은 AVX512_BF16 [M-paper] | arXiv 2402.07033v2 §3.3, Alg. 1, Fig. 5 |
| HybriMoE | DAC'25 | Xeon 5220R 10코어 제한, A6000 | Mixtral, DS-V2-Lite, Qwen2-57B, llama.cpp 4비트 | 32–1024 | 프리필 평균 1.33배(대 KTransformers) | CPU·GPU·전송 세 타임라인을 시뮬레이션으로 채운다. GPU는 캐시된 고부하, CPU는 비캐시 저부하, 고부하 비캐시는 전송 우선 [M-paper] | arXiv 2504.05897v1 §IV-B, §VI-B1 |
| MoE-Lightning | ASPLOS'25 | T4/L4, Xeon 24–32c | Mixtral, DBRX | ≤2048 | 프리필 수치 없음 | 프리필은 전부 GPU에서(각주 7). HRM: P = min(P_peak, B·I, B_link·I_link) [M-paper] | arXiv 2411.11217v1 §3.2 Eq.7, §4 fn.7 |
| MoE-Gen | 2025 | A5000 + EPYC 7453 28c, PCIe4 32 GB/s | DS-V2 236B, DS-R1 | 512 | DS-V2 787 tok/s(대 DeepSpeed 109), DS-R1 204 | 전문가는 GPU에서만 돌고 층별로 스트리밍한다. GPU 포화에 전문가당 토큰 2^10, PCIe를 가리려면 2^11 이상이 필요. 대배치 오프라인 [M-paper] | arXiv 2503.09716 Table 7, Fig. 3, §4.1 |
| Klotski | 2025 | 3090 + Xeon 5318Y PCIe4 | Mixtral bf16 | 512 | 종합 처리량만 | GPU 전용. n·(tc_A+tc_G)+tc_hot ≥ t_IO_G+(K+1)t_IO_E [M-paper] | arXiv 2502.06888 §7 |
| CoX-MoE | DAC'26 | Xeon 8452Y 36c(AMX), RTX 6000 Ada 등 | Mixtral, V2-Lite, Qwen3-30B BF16 | 97/800, B=1024 | FlexGen·MoE-Lightning 대비 1.7–2.4배 | AMX 144 TFLOPS 대 AVX-512 18. "AVX is insufficient"라고 명시 [M-paper] | arXiv 2605.17889v1 §1, §2.2, §6.2 |
| ProMoE | 2024 | 엣지 GPU | — | — | 프리필 평균 2.20배(최대 3.21) | 예측 프리페치, GPU 계산 [M-paper, 초록만] | arXiv 2410.22134 초록 |
| Qwen3-235B (커뮤니티) | 2025-06 | 4 GPU + CPU 48스레드(세부 불명) | IQ4_XS | pp4096 / pp1024 | ub4096: 스트리밍 287 대 CPU(rtr) 169. ub1024: CPU 188 대 스트리밍 약 120 | 이 박스의 교차점은 전문가당 64–256 토큰 사이 [M-blog/docs] | ik `github-data/discussions/491…md` |

종합.
- 방식의 선택은 전문가당 토큰 수 하나로 갈린다. 배치 1이나 짧은 프롬프트는 CPU 계산, 대배치 오프라인은 스트리밍이다. llama.cpp는 이를 토큰 수로 잘못 재지만 ik는 전문가당 토큰으로 고쳤다.
- 우리 박스의 교차점[I]을 유도했다. 전문가 하나 스트리밍에 16.77 MB ÷ 26.2 GB/s = 0.64 ms가 든다. CPU에서는 토큰·전문가당 70.8 MFLOP ÷ 4.0 TOPS = 17.7 µs다. 따라서 s* ≈ 36 토큰/전문가이고 B* ≈ 36 × 384/6 ≈ 2300 토큰이다.
- 일반식은 s* = b·R_cpu / (2·β_link)이고 b는 가중치당 바이트다. 전문가 크기와 무관하다. pageable 링크면 B*는 약 4200이다.
- **AVX2로 옮겨 오는 것과 못 오는 것이 갈린다.** KTransformers v0.3·SOSP의 AMX 수치(21.3 TFLOPS, 35 TOPS)와 CoX-MoE는 옮겨 오지 않는다. 우리 int8 실효 약 4 TOPS는 KT AMX int8의 약 11%다. llamafile 시절 KT v0.2.x(DS-V3 97–112)가 우리와 가까운 유사물이다.
- 최선의 설계는 두 극단 중 하나가 아니다. **끝나는 시각을 맞추는 동시 분할**이다. HybriMoE의 시뮬레이션 채움과 exllamav3의 "꼬리는 CPU, 뜨거운 것은 스트림"이 이 형태다. Fiddler식 전문가별 부등식은 CPU와 링크가 병렬로 돈다는 점을 반영하지 않는다[I].
- kt의 AVX2 튜토리얼 임계(400)가 AMX 튜토리얼(1024–4096)보다 낮다. 약한 CPU는 더 일찍 스트리밍한다는 벤더 쪽 신호다. 반면 exllamav3의 stream_t=8은 12스레드 보정값이라 우리 32코어에 그대로 쓰면 안 된다.

### Q2. 층별 파이프라인, 전문가 합집합, 프롬프트 길이 의존

| 항목 | 출처 | 내용 |
|---|---|---|
| 층 N+1 전송과 층 N 계산 겹침 | kt `kt_ep_wrapper.py:2784-2820@541ddc3` [C-code] | 전체 층 슬롯 2개와 transfer·postprocess·control 스트림을 쓴다. "launched after layer N's compute so layer N+1 transport can overlap that compute" |
| 겹침 없음 | llama.cpp `ggml-backend.cpp:1717-1760@84e76d8` [C-code] | ids를 호스트로 읽어 동기화한 뒤, 쓰인 전문가를 split 스트림에서 `tensor_set_async`로 복사한다. 다음 층 선반입은 없다 |
| DAG 임계경로로 모듈 배치 선택 | MoE-Gen §4.3–4.4 [M-paper] | B, b_a, b_e, ω, 버퍼를 DP로 골라 HtoD와 계산을 겹친다 |
| 가중치 페이징 | MoE-Lightning §4.1 [M-paper] | 가중치를 마이크로배치 수만큼 쪼개 전송 4종을 끼워 넣는다 |
| 백그라운드 가중치 스트림 | MoE-Prefill arXiv 2605.02960 [M-paper] | "the long, compute-bound forward passes of large-batch prefill open a per-layer window wide enough to stream expert weights in the background". 포화 임계를 강제해 겹침을 보장한다 |
| 합집합(전문가 가중치 한 번 읽기) | KT SOSP §3.2 [M-paper] | 전문가 가중치를 세로로 나눠 작업화한다. 입력은 L3, 가중치 블록은 L2에 맞춘다. Gate를 전문가에 걸쳐 한 작업으로 합쳐 층당 fused batch 2개로 줄인다 |
| 합집합(카드) | exllamav3 `moe_batch_recon.py:1-47@6b84a21` [C-code]; llama.cpp PR #15525 [M-blog/docs] | 뜨거운 전문가를 row 수로 정렬해 16개씩 묶는다(패딩 ≤1.1배). pointer-table dequant 후 strided-batched fp16 GEMM. llama.cpp는 MoE 헬퍼를 디바이스로 옮겨 동기화를 없앴고, 3090 V2-Lite ub16–64에서 1.4–1.6배다 |
| 실측 편중 | MoE-Prefill Fig. 4 [M-paper] | Qwen3-30B-A3B에서 64K 토큰 기준, 전문가 간 max/min 토큰 수가 48층 합산 16.15배이고 층 내부는 더 치우쳤다(A100, vLLM) |
| 편중과 스케줄링 | KT SOSP §3.2 [M-paper] | 프리필 "uneven expert activations". 동적 작업 도둑질로 최대 1.83배 |

균등 라우팅에서 전문가당 토큰 수는 이항분포로 계산했다[I, 스크래치 `tokstats.py`]. 실제 라우터는 이보다 더 치우친다.

| 모델 | P | 전문가당 평균 | 표준편차 | 활성 전문가(개) | 기대 최댓값 | 1–99분위 |
|---|---:|---:|---:|---:|---:|---|
| V4.1 (384, k=6) | 128 | 2.0 | 1.4 | 333 / 384 | 7.2 | 0–6 |
| V4.1 | 512 | 8.0 | 2.8 | 383.9 / 384 | 17.4 | 2–15 |
| V4.1 | 2048 | 32 | 5.6 | 384 | 49.8 | 20–46 |
| V4.1 | 4096 | 64 | 7.9 | 384 | 88.7 | 46–83 |
| Qwen3 (128, k=8) | 512 | 32 | 5.5 | 128 | 47.0 | 20–45 |
| Qwen3 | 4096 | 256 | 15.5 | 128 | 297 | 221–293 |

종합.
- V4.1은 P=512부터 호스트 전문가가 **전부** 쓰인다. 프리필 한 번은 호스트 집합 212.9 GB를 최소 한 번 읽는 일이다. 캐시와 예측으로 줄일 바이트가 없다.
- 균등 라우팅에서 P=512 최댓값은 평균의 2.2배이고 4096에서는 1.4배다. Qwen3 실측(16.15배 max/min)은 균등보다 훨씬 치우쳤다. 그러니 CPU 쪽 작업 분배는 정적 분할이 아니라 도둑질이어야 한다(KT 1.83배).
- 층별 겹침의 참조 설계는 kt-kernel(슬롯 2개 + 후속 층 전송)과 exllamav3(copy 스트림 + VRAM 링 + CPU 꼬리 동시)다. llama.cpp와 ik의 op offload는 겹치지 않는다.
- 스트리밍 비용은 ubatch마다 일정하다. 그래서 ubatch를 프롬프트 전체(최대 4096)로 잡는 것이 그 자체로 레버다. ik #520 댓글에서 pp가 P에 비례하는 것이 이 때문이다.

### Q3. Ampere(sm_86)의 양자화 프리필 GEMM

| 커널 | 출처 | GPU | 모델·타입 | m | 결과(피크 대비는 [I]) | 역양자화 전략 |
|---|---|---|---|---|---|---|
| llama.cpp MMQ + int8 MMA | PR #7921 [M-blog/docs] | RTX 3090 | Llama-3 8B Q3_K_S pp2048 | ub16 / 64 / 256 / 512 / 1024 | 923 / 1944 / 3305 / 3487 / 3576 t/s. 약 14.5 GFLOP/tok으로 환산하면 int8 284 대비 4.7 / 9.9 / 16.9 / 17.8 / 18.3% | 활성값 q8_1 + IMMA. 작성자: 텐서코어의 차이는 약 10%, 이용률이 낮아서 |
| llama.cpp dense | Discussion #15013 [M-blog/docs] | 3090 / A6000 | Llama-2 7B Q4_0 pp512 | 512 | 3090: 5175, A6000: 4914(FA 켜면 5662) t/s. 피크 대비 약 24% / 21% | MMQ |
| llama.cpp MoE | PR #13199, #15525 [M-blog/docs] | 3090 | DeepSeek-V2-Lite Q4_0 | pp512 ub512 / pp2048 ub2048 | 4848 / 4544 t/s. 활성 약 4.2 GFLOP/tok으로 약 20 TOPS, 피크의 7% | MUL_MAT_ID MMQ. dense보다 약 3.4배 덜 효율 |
| llama.cpp MoE | PR #16991 [M-blog/docs] | 4090 / 5090 | Qwen3-30B-A3B Q4_K_M | pp512 | 7246 / 7362 t/s | MMQ. 5090이 4090과 같다. 연산이 아닌 병목이 있다[I] |
| llama.cpp 선택 규칙 | `mmq.cu:266-~320@84e76d8` [C-code] | Turing 이상 | K-quant 전부 | 전부 | — | int8 MMA가 있으면 **모든 배치에서 MMQ**. cuBLAS dequant는 강제할 때만 |
| ik MoE | `ggml-cuda.cu:3234-3250@20f7a72` [C-code] | — | — | 토큰 ≤ 32·E | — | mmq_id 융합 경로. 초과하면 전문가별 gather + 일반 matmul |
| exllamav3 | `exl3.py:10,137-144@6b84a21` [C-code] | — | EXL3 트렐리스 | rows ≤ 144 / > 144 | — | 144행 이하는 융합 양자 커널, 넘으면 fp16 재구성 + hgemm. 3비트 K-quant는 없다 |
| Marlin | arXiv 2408.11743 Table 2 [M-paper] | 3090 / A6000 | Llama-2 7B / 13B, 4비트 | 1 / 16 / 32 / 64 / 128 | FP16 대비 3090 2.69 / 2.30 / 1.84 / 1.28 / 1.11배, A6000(13B) 3.17 / 2.77 / 2.39 / 2.01 / 1.23배 | W4A16 lop3 레지스터 역양자화 + HMMA. 4/8비트 전용, IMMA 미사용 |
| QServe | arXiv 2405.04532 §3.1 [M-paper] | A100 / L40S | W4A8 | m > 78 | L40S Llama-2-7B 2394 대 TRT W8A8 1271 tok/s | 모든 연산을 IMMA로. W4A16은 m<78에서만 유리 |

종합.
- 3–4비트 K-quant를 Ampere에서 큰 m으로 돌리는 프로덕션 경로는 사실상 llama.cpp·ik의 MMQ 하나다. 활성값을 q8_1로 양자화해 IMMA에 넣는다.
- 역양자화 후 FP16 HMMA 방식은 A6000에서 연산 천장이 절반이다(154.8 대 309.7). 3090에서는 FP32 누산 HMMA가 71로 **4분의 1**이다. Marlin의 이득도 m=128에서 1.1–1.2배로 사라진다.
- 달성률은 dense가 int8 피크의 18–24%, MoE가 약 7%(V2-Lite pp512)다. MoE에서 전문가당 m이 작아 타일이 비고, 전문가 경계에서 패딩과 스케줄 손실이 생긴다.
- A6000 V4.1 카드 몫(dense 13.8 + 카드 전문가 약 4.3 + attention)을 MMQ급 약 50 TOPS로 잡았다[I]. P=4096에서 약 1.5 s로 호스트 항보다 작다. 그러나 겹치지 않으면 무시할 수 없다.

### Q4. MLA + 압축 KV + 희소 인덱서의 프리필

| 항목 | 출처 | 내용 |
|---|---|---|
| CED | V4.1 논문 §2.2, §3.2.2, §4.2.1 [M-paper] | 인코더 20층, 디코더 20층이다. 디코더 전역 KV는 H_20에서 투영한다. "compute only the first half of the layers during the prefill phase". 정확한 디코더 SWA 재구성은 마지막 (L/2)×n_win 토큰이 필요하다. 배포는 Bounded Replay(마지막 n_win=128)이고 "not mathematically equivalent"다. 활성 파라미터는 프리필 8B, 디코드 16B |
| CSA2 | V4.1 논문 §2.3, §4.2.1 [M-paper] | 인코더 m=2, 디코더 m=1이다. 압축은 **겹침 없음**(CSA의 2m 중첩 창과 절대 위치 임베딩 제거). 인덱서 K는 main KV를 투영한다. Full / Reindex / Reuse 모드. 인덱서 32헤드 × 128, top-k 512, n_win 128 |
| 계층 희소 인덱서 | V4.1 논문 §2.3.2 [M-paper] | 디코더 전용이다. Full 층이 후보 풀(블록 2048 × 8 = 16,384 위치)을 만들고 Reindex 층은 풀만 채점한다. 학습 때도 같게 적용된다 |
| DSA 비용 | V3.2 논문 §2.3, Fig. 3 [M-paper] | 핵심 attention O(L²)가 O(Lk)가 된다. 인덱서는 O(L²)지만 "much less computation". H800 비용 곡선. **짧은 프롬프트는 "masked MHA mode to simulate DSA"** |
| ik V4.1 그래프 | `build_deepseek4.cpp:1561,1710-1714@20f7a72` [C-code] | 40층 전부를 n_tokens 전부에 돌리고, 출력 행은 마지막에 고른다. CED 생략이 없다. 열린 PR #2512의 분리 경로 설명에도 인코더 전용 프리필은 없고 "chunked indexer score"만 있다 |
| FlashMLA | README@ba89a34 [M-blog/docs] | sparse prefill은 per-query indices를 받는 `flash_mla_sparse_fwd`로 H800에서 640 TFLOPS다. **SM90/SM100 전용**이다 |
| vLLM DSA | vLLM 블로그 2025-09-29 [M-blog/docs] | DeepGEMM `fp8_mqa_logits`로 q×n 로짓을 만든다. ks/ke로 인과성. 융합 top-k, 별도 인덱서 K 캐시. 수치 없음 |
| sm_86의 V4 | kt `DeepSeek-V4-Flash.md` [M-blog/docs] | 3090에서 NSA sparse MLA는 Triton 폴백, 인덱서는 tilelang 경로다 |
| NSA | arXiv 2502.11089 §3.3, §5 [M-paper] | A100 Triton에서 64K 전진 9.0배(8K 4배). 선택 블록 중요도는 압축 attention 점수를 재사용한다. GQA 그룹이 선택을 공유한다 |
| Chunked prefill | Sarathi-Serve §4.3, Fig. 14 [M-paper] | Yi-34B TP-2에서 chunk 512는 오버헤드 약 25%, 2048은 거의 0. chunk 257은 256보다 32% 느리다(타일 양자화) |

종합.
- **프롬프트 전체에 대해 본질적으로 순차인 부분은 없다.** ratio-2 압축은 겹침 없는 2토큰 묶음이고, ratio-1은 투영이다. engram은 앞 3토큰 id의 해시라 id가 정해지면 병렬이다. 128 링은 인과 마스크다. 인덱서 top-k는 쿼리마다 독립이다. ubatch 경계의 미완 압축 묶음만 상태가 넘어간다[I, 논문 §2.3에서].
- 인덱서 FLOP는 P=4096에서 토큰당 약 0.11 GFLOP로 전체 약 33 GFLOP의 0.3%다[I]. 128K에서는 약 6%다(인코더 Full 3층이 64K 키를 스캔하고, 디코더는 후보 풀 16,384로 묶인다).
- **정확한 짧은 프롬프트 지름길이 있다.** 보이는 압축 항목 수가 top-k(512) 이하이면 선택은 "전부"다. 따라서 m=2 층은 P ≤ 1024, m=1 층은 P ≤ 512에서 인덱서 채점과 gather를 비트 불변으로 생략할 수 있다. V3.2의 "masked MHA mode"가 같은 생각이다.
- CED의 정확한 이득을 층별로 셌다[I]. 디코더 층 39−j는 마지막 1+127j 토큰에서만 계산하면 된다. 창 128이 층마다 뒤로 127씩 번지기 때문이다. 층 20은 2414 토큰이고, H_20은 인코더가 전 토큰에 갖고 있다. 디코더 토큰·층 합은 24,150으로 논문의 사각형 상한 51,200보다 작다.
- 그 결과 토큰·층 비율은 P=512에서 0.937, 4096에서 0.647, 32K에서 0.518이다. DSpark 초안기(창 128)가 읽는 37–39층 입력은 이 범위에 들어간다.

### Q5. AVX2 CPU의 배치 전문가 계산

| 항목 | 출처 | 수치 |
|---|---|---|
| ik pp512, **Ryzen-5975WX(우리와 같은 CPU)**, Llama-3.1-8B | ik wiki Jan-2025 [M-blog/docs] | ik 대 mainline(tinyBLAS 켬): Q3_K_S 267.69 대 118.88, Q4_K_S 291.90 대 148.69, Q6_K 264.83 대 104.15, Q8_0 251.75 대 165.14, F16 152.65 대 91.61 t/s. 약 14.1 GFLOP/tok으로 환산하면 약 3.8–4.1 TOPS, 코어당 약 120–130 GOPS[I] |
| ik CPU 전용 Mixtral-8x22B Q4_K_M pp512 | ik discussion 258, 2025-03-23 [M-blog/docs] | 5975WX에서 61 t/s. 활성 약 77 GFLOP/tok으로 약 4.7 TOPS[I] |
| Zen 3 명령 처리량 | uops.info [M-blog/docs] | VPMADDUBSW ymm: TP 0.5, 지연 3, FP03. VPMADDWD ymm: 같다. 따라서 int8 사슬은 코어·사이클당 32 MAC이고, 3.6 GHz × 32코어에서 약 7.4 TOPS 피크[I]. ik는 그 50–55% |
| 대역폭-연산 경계 | 계산 [I] | 4.0 TOPS ÷ 147.7 GB/s = 27 op/B. Q3_K는 m=1에서 4.65 op/B이므로 경계는 m≈5.8 토큰/전문가. V4.1 P=512(평균 8)가 경계 위다 |
| KT AMX·AVX-512 | SOSP §2.2, §3.2; AMX.md [M-paper][M-blog/docs] | AMX 커널 21.3 TFLOPS BF16(1소켓 36c), int8 35 TOPS. PyTorch AMX 5.4, AVX-512 1.8. 전문가당 ≤4 토큰에서 AVX-512가 이긴다 |
| kt AVX2 백엔드 | kt-kernel README, AVX2-Tutorial [M-blog/docs] | AVX2는 LLAMAFILE 백엔드 또는 BF16/FP8/GPTQ_INT4/RAWINT4. 층별 GPU 프리필 임계 400 |
| exllamav3 CPU 층위 | `moe_mul1.cpp:39-80,196-199@6b84a21` [C-code] | Scalar / Avx2 / Bw / Vnni / Vbmi. AVX2는 byte-sum을 vpmaddubsw로 흉내 낸다 |

종합.
- 같은 실리콘에서 ik의 dense 프리필 실효가 약 4 TOPS다. 이것이 호스트 티어의 연산 쪽 눈금이다.
- 우리 CPU는 KT AMX int8 속도의 약 11%만 갖는다. AMX에 기대는 결과는 전부 옮겨 오지 않는다.
- 다른 곳에서 DeepSeek CPU 전문가 프리필이 보인 실효도 이 눈금과 같은 급이다[I]. KT v0.2.1은 llamafile로 약 4.5 TOPS(111 t/s × 40.9 GFLOP), ubergarm의 7965WX는 약 4.5 TOPS다.
- **V4.1은 P=512에서 전문가당 약 8 토큰으로 경계(m≈6)에 걸린다.** 호스트 212.9 GB를 140 GB/s로 읽는 데 1.52 s, 512 × 12.7 GFLOP ÷ 4 TOPS에 1.63 s가 든다. P≤128은 대역폭 바운드라 배치의 이득이 거의 없다.
- oneDNN과 llamafile 원문 블로그는 확보하지 못했다. 5.2절 실패 목록에 적었다.

### Q6. 같은 급 한 대 박스의 공개 프리필 숫자

| 엔진·설정 | 하드웨어 | 모델 | P | pp tok/s | 비교 가능성 |
|---|---|---|---|---:|---|
| ik, `-ot exps=CPU -rtr` (ubergarm, 커뮤니티) | TR PRO **7965WX 24c Zen 4(AVX-512)**, DDR5 약 225 GB/s, A6000 | DS-R1 Q2_K_R4 238.7 GiB | 512 | 109–112 | 가장 가깝다. 다만 ISA와 대역폭이 우리보다 우위 |
| KTransformers v0.2.1 (llamafile CPU) | 2× Xeon 6454S, 2×8ch DDR5-4800, 4090D | DS-V3 Q4_K_M | 1K / 2K / 4K / 8K | 111 / 112.5 / 102 / 101 | 중간. 코어 64개, AVX-512 |
| KTransformers v0.3 AMX | 같음 | 같음 | 2K | 286.55 | 불가(AMX) |
| ik op-offload 스트리밍 (커뮤니티) | 2×3090 PCIe 3.0 x8 | DS-V3 IQ1_S | 512 / 4096 | 27.9 / 166 | 스트리밍 모양만 참고 |
| kt 층별 프리필 | 1×4090 48 GB + 2× 8488C AMX | Kimi-K2 INT4 | 2K / 8K / 30K | 53 / 184 / 290 | P 의존의 모양만 참고 |
| ik CPU 전용 | **5975WX**(같은 CPU) | Mixtral-8x22B Q4_K_M | 512 | 61 | CPU 눈금으로 비교 가능 |
| MoE-Gen (대배치) | A5000 + EPYC 7453 | DS-V2 / DS-R1 | 512 | 787 / 204 | 불가(오프라인 처리량) |
| llama.cpp 전 카드 | 4090 / 5090 | Qwen3-30B-A3B Q4_K_M | 512 | 7246 / 7362 | Qwen3 선의 출발점 |

- V4.1 선을 두 방식으로 환산했다[I]. 우리 호스트 몫은 토큰당 12.7 GFLOP(routed 17.0의 75%로 가정)다.
  - 코어 수와 ISA 비율로 환산하면 약 70–100 t/s다. KT 111에 코어 0.5배, AVX2 코어당 0.55배, FLOP 비 2.4–3.2배를 곱했다.
  - 같은 실리콘의 실효 4 TOPS로 환산하면 약 300 t/s가 위쪽이다.
  - 선은 70–300 t/s이고, 오늘 43 tok/s의 2–7배다. 이 폭은 박스 실측으로만 좁혀진다.
- Qwen3 A6000 선: 7246(4090) × 0.414(V2-Lite pp512에서 3090/4090 비) × 0.95(dense pp512에서 A6000/3090 비) ≈ 2850 tok/s[I].
- 디코드 전용 숫자는 표에서 뺐다. KT V4-Flash "20+ tok/s", ik #2449의 7–8 tok/s, #2444의 13.6 t/s가 그것이다.

## 3. Q7. 이 박스에 맞춘 아이디어 순위와 하지 말 것

아래 수치는 전부 [I]이고 산수는 표 아래에 적었다. 기준선은 V4.1이 모든 P에서 약 43 tok/s이고, Qwen3는 미측정이다.

| 순위 | 기법 | 출처 | 기제 | 엔진에 필요한 것 | 주된 위험 |
|---|---|---|---|---|---|
| 1 | 호스트 티어 배치 MoE(합집합, 전문가 우선 순회) | KT SOSP §3.2, ik iqk_mul_mat | 전문가 가중치를 ubatch당 한 번 읽고 토큰을 모은다. 활성값 q8 한 번, gate·up 융합, 작업 도둑질 | 전문가별 토큰 목록(CSR), m=2–64 AVX2 GEMM 타일, L2 블로킹, 도둑질 큐 | 작은 m에서 효율이 dense보다 낮다 |
| 2 | 카드 m행 IMMA GEMM + 카드 MUL_MAT_ID | llama.cpp MMQ(#7921, #15525), `mmq.cu:266` | q8_1 활성값, K-quant 블록 스케일을 IMMA로. 디바이스 쪽 전문가 정렬 | A5 커널(m 16–512), 전문가 경계 타일, ubatch 최대 4096 | cuda-oxide 커널 개발량, 레지스터와 공유메모리 |
| 3 | CED 정확 재생(삼각형) | V4.1 논문 §2.2, §3.2.2; ik는 미구현 | 인코더 20층은 전 토큰, 디코더 층 39−j는 마지막 1+127j 토큰 | 층별 행 범위, H_20의 층 20 압축 KV와 인덱서 K, 링 채움 | 창·링 경계 버그. 최종 상태 동등 게이트가 필요 |
| 4 | 동시 분할: 스트림(뜨거운 것) + CPU(꼬리), 끝나는 시각 균형 | HybriMoE Eq.2, exllamav3 `submit_prefill`, Fiddler | 층 l+1의 정적 핫셋(hot list)을 층 l 중에 선반입한다. 라우팅 뒤 나머지는 CPU | pinned 소스(26.2 대 14.4 GB/s), 슬롯 2개, copy 스트림 | A6000에 빈 VRAM이 없고(plan (a)가 약 48.3 GB), DMA가 DRAM을 두고 경합 |
| 5 | 층별 전체 스트리밍(P ≥ 약 2300에서만) | kt layerwise, llama.cpp/ik op offload | 층의 호스트 전문가 전부를 카드로 보내 GPU에서 계산 | 4와 같다. ubatch는 프롬프트 전체 | P가 작으면 진다 |
| 6 | 두 카드 스트리밍(3090이 스트리밍 카드) | [I], 링크별 x16 가정 | 3090에 슬롯을 두고 자기 링크로 받는다. 카드 사이 활성값은 호스트 경유 | 링크 토폴로지 확인, 층마다 교차 복사 약 168 MB | 3090 Xid 79 이력, 토폴로지 미검증 |
| 7 | CED Bounded Replay(근사) | V4.1 논문 §3.2.2(DeepSeek 배포 모드) | 디코더를 마지막 128 토큰에만 | 3의 일반화 | **출력이 바뀐다**. ik 오라클과 KL 비교, 사용자 결정 |
| 8 | 짧은 프롬프트 인덱서 생략(정확) | V3.2 §2.3 "masked MHA mode" | 보이는 압축 항목 ≤512이면 top-k는 전부 | 조건 분기 한 줄과 동등 게이트 | 얻는 몫이 작다 |
| 9 | 긴 프롬프트 chunked 인덱서와 후보 풀 | ik PR #2512, 논문 §2.3.2 | 쿼리를 잘라 [chunk × P/m] 로짓을 제한 | 청크 루프, int8 인덱서 내적 | 16K 이상에서만 의미가 있다 |

| 기법 | V4.1 P=512 | V4.1 P=4096 | Qwen3 P=512 | Qwen3 P=4096 |
|---|---:|---:|---:|---:|
| 1. 호스트 배치 MoE(CPU만) | 314 천장, 현실 약 160 | 314 천장 | — | — |
| 2. 카드 IMMA GEMM | 카드 항 0.19 s(가려져야 함) | 카드 항 1.5 s | 선 약 2850, 천장 약 8800 | 천장 7000–14000 |
| 5. 전체 스트리밍만 | 63(짐) | 504(pageable이면 277) | — | — |
| 4. 1+5 동시 분할 | 377 | 818 | — | — |
| 3. + CED 정확 | 398 | 989 | — | — |
| 6. + 두 카드 스트리밍 | 461 | 1493 | — | — |
| 7. + Bounded Replay(근사) | 약 +7%(바이트 바운드) | 약 1100(4.과 결합) | — | — |
| 8. 인덱서 생략 | ≤2% | 적용 없음(P>1024) | — | — |

산수.
- 상수: H = 212.9 GB, 호스트 FLOP/tok F_h = 0.75 × 17.0 = 12.7 GFLOP. 호스트 활성 몫 0.75는 가정이다(균등이면 0.826). R_cpu = 4.0 TOPS, β_h = 140 GB/s, β_link = 26.2 GB/s. 카드 R = 약 50 TOPS(3090 MMQ의 피크 17.8% × 309.7). 카드 FLOP/tok은 약 20(dense 13.8 + 카드 전문가 4.3 + attention).
- CPU 항: T_h = max(H/β_h, P·F_h/R_cpu). P=512은 max(1.52, 1.63) = 1.63 s이고, P=4096은 13.05 s다.
- 스트림 항: T_s = H/β_link = 8.13 s. 동시 분할은 T = T_h·T_s/(T_h+T_s)다.
- CED 정확: 계산 바운드 항에 f(P) = (20P + Σ_{j=0}^{19} min(P, 1+127j)) / 40P를 곱한다. f(512) = 0.937, f(4096) = 0.647이다. 스트림 바이트는 줄지 않는다.
- Bounded Replay: f′(P) = (20P + 20·min(P,128)) / 40P이고, f′(512) = 0.625, f′(4096) = 0.516이다. P=512에서는 디코더 전문가도 한 번은 읽어야 하므로(128 토큰에서 384개 중 333개 활성) 이득이 작다.
- 두 카드: T_s = 4.07 s.
- Qwen3 천장: P=512은 512 × 5.7 GFLOP ÷ 50 TOPS = 58 ms에서 8800. P=4096은 4096 × 7.1 ÷ 50–100 TOPS.

순서 제안[I]은 2(Qwen3와 공통, 카드 항 제거) → 1 → 3 → 4/5 → 6이다. 7은 사용자 결정이 필요하다. 4와 5의 선결은 VRAM이다. A6000 plan (a)는 2,668 × 16.77 MB + dense 3.6 GB ≈ 48.3 GB로 꽉 찬다. 층당 호스트 몫은 약 5.3 GB(317 전문가)라 슬롯 2개에 10.6 GB가 필요하다. 방법은 둘이다. 프리필 동안 상주 전문가 약 630개를 내렸다가 다시 올리거나(10.6 GB ÷ 26.2 GB/s ≈ 0.4 s, P=4096이면 싸고 P=512이면 비싸다), 3090을 스트리밍 카드로 쓴다.

**하지 말 것.**
- **AMX·AVX-512·VNNI에 기대는 커널과 KT의 AMX 타일링.** 우리에게 없는 ISA다(KT AMX.md). CoX-MoE §1도 "AVX is insufficient"라고 적는다.
- **FlashMLA·DeepGEMM의 sparse prefill과 fp8 인덱서 코드를 가져오는 것.** SM90/SM100 전용이고(FlashMLA README 지원표) sm_86에는 FP8이 없다. 알고리즘만 옮긴다.
- **Marlin/W4A16식 역양자화 후 HMMA를 큰 m에서 쓰는 것.** 4/8비트 전용이고 m=128에서 이득이 1.1–1.2배로 사라진다(Marlin Table 2). 3090의 FP32 누산 HMMA는 71 대 int8 284다(GA102).
- **QServe식 재양자화.** ik 오라클과 결별하고, A100/L40S에서만 쟀다.
- **프리필에 Expert Deferral을 쓰는 것.** KT가 디코드 전용으로 쓴 이유가 "almost complete coverage of experts"다(SOSP §3.3).
- **프리필에 전문가 캐시나 예측 프리페치를 쓰는 것(ProMoE, MoE-Infinity, fMoE류).** P=512에서 384개 전부가 쓰인다(Q2 표).
- **P가 약 2300보다 작을 때 V4.1 전문가를 스트리밍하는 것과 pageable 소스로 스트리밍하는 것.** P=512 스트리밍은 63 tok/s로 CPU의 약 314보다 느리다. pageable은 14.4 GB/s다.
- **대배치 오프라인 설계를 이식하는 것(MoE-Gen, MoE-Lightning, Klotski, CoX-MoE).** 전문가당 2^10–2^11 토큰이 필요하다(MoE-Gen §4.1). 단일 사용자 프롬프트는 이를 못 채운다.
- **exllamav3의 stream_t=8을 그대로 쓰는 것.** 12스레드 보정값이다. 우리 식은 s* ≈ 36이다.

## 4. 문헌이 답하지 못하는 것과 그것을 가를 측정

1. **5975WX에서 m=2/8/32/64 토큰/전문가일 때 Q3_K gate·up과 Q4_K down GEMM 실효.** 합성 전문가 묶음으로 호스트 커널을 마이크로벤치하고 dense 4 TOPS 눈금과 비교한다. 1번 아이디어의 "현실" 계수를 정한다.
2. **V4.1 프리필의 라우팅 편중과 hot list 아래 카드 활성 몫.** 512·4096 프롬프트의 층별 max/mean과 카드 적중 비율을 잰다. 라우터 id 덤프로 답하며 `tools/ref/router-hotlist.py`가 있다. 가정 0.75를 대체한다.
3. **200 GB가 넘는 mmap 호스트 집합에서 pinned DMA가 되는가.** cuMemHostRegister(읽기 전용 플래그) 대 pinned 스테이징 링을 비교한다. 달성 GB/s와, 동시 DMA 아래 CPU GEMM의 감속을 잰다.
4. **카드별 PCIe 세대·폭과 두 카드 동시 H2D 합.** `nvidia-smi -q`의 링크 정보와 두 스트림 복사 시험으로 잰다. 26.2 GB/s를 어느 카드에서 쟀는지도 확인한다.
5. **이 박스에서 ik와 llama.cpp의 pp512·pp4096.** 각자 최적 플래그로 잰다(시팅 P). 특히 ik `-ub 4096`에서 규칙 2048 위아래를 본다.
6. **CED 정확 재생의 동등성.** 최종 SWA 링, 압축 KV, 마지막 logits를 전체 프리필 경로와 비교하는 게이트를 둔다. 근사 모드를 원한다면 프롬프트 세트에서 ik 대비 KL을 잰다.
7. **cuda-oxide IMMA m행 GEMM의 A6000 달성률.** m=16–512를 재서 llama.cpp MMQ 17–24%와 비교한다.
8. **프리필 중 스트리밍 슬롯의 자리.** 상주 전문가를 내렸다 올리는 비용과 3090 경로 비용을 같은 임대에서 잰다.

## 5. 출처

### 5.1 받아서 읽은 것

논문과 공식 문서:
- arXiv 2609.19969 PDF, DeepSeek-V4.1-Flash 논문(pdftotext): https://arxiv.org/pdf/2609.19969
- arXiv 2512.02556 PDF, DeepSeek-V3.2: https://arxiv.org/pdf/2512.02556
- KTransformers SOSP'25 PDF(pdftotext): https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf
- Fiddler: https://arxiv.org/html/2402.07033v2
- MoE-Gen: https://arxiv.org/html/2503.09716
- MoE-Lightning: https://arxiv.org/html/2411.11217v1
- Klotski: https://arxiv.org/html/2502.06888
- CoX-MoE: https://arxiv.org/html/2605.17889v1
- HybriMoE: https://arxiv.org/html/2504.05897v1
- ProMoE 초록: https://arxiv.org/abs/2410.22134
- fMoE 초록: https://arxiv.org/abs/2502.05370
- MoE-Infinity 초록: https://arxiv.org/abs/2401.14361
- MoE-Prefill PDF: https://arxiv.org/pdf/2605.02960
- 활성 전문가 균형(수치 미사용): https://arxiv.org/html/2512.09277
- FreeBalance PDF(수치 미사용): https://arxiv.org/pdf/2608.14205
- Marlin: https://arxiv.org/html/2408.11743
- QServe: https://arxiv.org/html/2405.04532
- Sarathi-Serve: https://arxiv.org/html/2403.02310
- NSA: https://arxiv.org/html/2502.11089
- MegaBlocks 초록(수치 없음): https://arxiv.org/abs/2211.15841
- GA102 whitepaper v2: https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf
- vLLM DSA 블로그: https://vllm.ai/blog/2025-09-29-deepseek-v3-2
- uops.info VPMADDUBSW: https://uops.info/html-instr/VPMADDUBSW_YMM_YMM_YMM.html
- uops.info VPMADDWD: https://uops.info/html-instr/VPMADDWD_YMM_YMM_YMM.html
- Qwen3 config: https://huggingface.co/Qwen/Qwen3-30B-A3B/raw/main/config.json

KTransformers와 kt-kernel:
- https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/DeepseekR1_V3_tutorial.md
- https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/AMX.md
- https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/README.md
- raw@c40722b: `doc/en/AMX.md`, `doc/en/DeepSeek-V4-Flash.md`, `doc/en/kt-kernel/AVX2-Tutorial.md`, `doc/en/kt-kernel/Kimi-K2-Thinking-Native.md`, `doc/en/Kimi-K2-Thinking.md`, `doc/en/kt-kernel/experts-sched-Tutorial.md`, `doc/en/benchmark.md`(https://raw.githubusercontent.com/kvcache-ai/ktransformers/c40722bf04c494f2492b7eb9e86ef01a4ede45b3/doc/en/…)
- kvcache-ai/sglang `python/sglang/srt/layers/moe/kt_ep_wrapper.py@541ddc3`: https://raw.githubusercontent.com/kvcache-ai/sglang/541ddc37cbc92c60dc748db5ff1a2aad0b069a80/python/sglang/srt/layers/moe/kt_ep_wrapper.py

llama.cpp:
- raw@84e76d8: `ggml/src/ggml-cuda/ggml-cuda.cu`, `ggml/src/ggml-backend.cpp`, `src/llama-model.cpp`, `ggml/src/ggml-cuda/mmq.cu`(https://raw.githubusercontent.com/ggml-org/llama.cpp/84e76d8a23162eca70490da131945ebec1f09bf4/…)
- https://github.com/ggml-org/llama.cpp/issues/18530
- https://github.com/ggml-org/llama.cpp/issues/27425
- https://github.com/ggml-org/llama.cpp/discussions/15013
- https://github.com/ggml-org/llama.cpp/pull/7921 와 본문 API
- PR 본문 API: https://api.github.com/repos/ggml-org/llama.cpp/pulls/13199, …/15525, …/16991

ik_llama.cpp:
- https://github.com/ikawrakow/ik_llama.cpp/wiki/Jan-2025:-prompt-processing-performance-comparison/20f0a968188aee4e27e1fc09bc2cd30e9f845abb
- 위키 raw: https://raw.githubusercontent.com/wiki/ikawrakow/ik_llama.cpp/Jan-2025:-prompt-processing-performance-comparison.md
- raw@20f7a72: `ggml/src/ggml-cuda.cu`, `ggml/CMakeLists.txt`, `ggml/src/CMakeLists.txt`, `src/graphs/build_deepseek4.cpp`, `src/llama-dsv4.cpp`
- raw@20f7a72, `github-data/`: `pull_requests/520 - Better strategy for GPU offload.md`, `issues/255 …`, `discussions/258 …`, `discussions/491 …`. 내려받았으나 인용하지 않은 것: `discussions/357`, `223`, `164`, `25`
- 이슈 API: https://api.github.com/repos/ikawrakow/ik_llama.cpp/issues/2449, …/2455, …/2512, …/2523, …/2444

exllamav3와 FlashMLA:
- exllamav3 raw@6b84a21: `exllamav3/modules/quant/exl3.py`, `modules/block_sparse_mlp_cpu.py`, `modules/moe_batch_recon.py`, `model/moe_cpu_host.py`, `exllamav3_ext/cpu/moe_mul1.cpp`(https://raw.githubusercontent.com/turboderp-org/exllamav3/6b84a21b6f1e5da3f291b9e1019061f0de788279/…)
- FlashMLA raw@ba89a34: https://raw.githubusercontent.com/deepseek-ai/FlashMLA/ba89a3466e9470ad08ab39738d4e7bb66989e1e7/README.md

커뮤니티:
- https://huggingface.co/blog/Doctor-Shotgun/llamacpp-moe-offload-guide (수치 없음)

### 5.2 실패하거나 확보하지 못한 것

- https://justine.lol/matmul/ — 인증서 만료로 실패했다.
- https://web.archive.org/web/2024/https://justine.lol/matmul/ — 도구가 거부했다.
- ktransformers `doc/en/kt-kernel/kt-kernel_intro.md` raw — 0바이트로, 경로가 틀렸다.
- 받지 않은 것: oneDNN AVX2 int8 GEMM 수치, Machete, SGLang·vLLM fused_moe의 A6000 수치, MegaBlocks 본문, DeepSeek-V4 기술보고서(arXiv 2606.19348). 이들로는 어떤 수치도 주장하지 않았다.

### 5.3 범위 밖에서 본 개선 여지(보고만, 손대지 않음)

- **`docs/facts.md`에 PCIe 세대·폭과 카드별 링크 사실이 없다.** 스펙은 거기서 읽으라고 했다. 한 줄로 크기 XS다.
- **`docs/research/v41-ports.md:38`의 26.2 GB/s 측정이 어느 카드인지 적혀 있지 않다.** XS다.
- **`AGENTS.md`의 BLOOMERY_CARD_DONTNEED 설명은 "196 GB host set"이다.** plan (a)의 2,668 슬롯에서 유도하면 212.9 GB다. 배치 기준을 명시해 맞출 필요가 있다. XS다.
