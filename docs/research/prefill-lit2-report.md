# 프리필 레버 문헌 조사 2 — 원문

2026-09-26. opus 서브에이전트 조사 라운드 `prefilllit2`의 보고 원문이다(스펙 `specs/wave-m6/spec-prefilllit2.md`, 박스 없음). 인용은 라운드가 이번에 받은 원문의 한 줄이다. 리드가 표본 둘을 대조했다: llama.cpp `4e74811` `ggml/src/ggml-cuda/mmvq.cuh:3`(GitHub raw를 직접 받아 같은 줄 확인)과 `mmvq.cu:185`. [유도] 수치는 라운드의 추론이다. 처분은 `docs/plan-triage.md`에 있다.

박스, 저장소, git은 건드리지 않았고 산출물은 모두 스크래치에만 두었습니다. 인용한 줄은 전부 이번 라운드에 받은 원문에 있습니다. 코드는 SHA를 고정한 raw 파일이고, 논문·글은 `fetch.py`로 받은 텍스트에서 grep으로 확인했습니다. 원문은 `/private/tmp/claude-501/-Users-hckim-repo-bloomery/02d044bc-8d2a-4055-91fa-4ae541a4916f/scratchpad/prefilllit2/`의 `lcpp/ vllm/ exl3/ mrs/ ik/ papers/ gh/` 아래에 있고, 조사한 커밋 SHA는 `heads.txt`에 적었습니다.

## 요약

- **가장 큰 레버는 문헌보다 우리 저장소 안에 있습니다.** `crates/gpu/src/gemm.rs`에는 이미 grouped int8 텐서코어 GEMM(`gemm_q3k`, `gemm_q4k`)이 있습니다. `gate_gemm`은 V4.1 실제 gate 스택(Q3_K, 5120 → 2304, 384 experts)으로 이 커널을 검증합니다.
  - V4.1 grouped shadow는 지금 워프 하나가 행 하나를 맡아 슬롯을 순서대로 도는 gemv이고, 실효는 약 3.0 TOPS입니다[derived].
  - llama.cpp MMQ MUL_MAT_ID, exllamav3 `moe_coop`, Marlin MoE는 모두 "한 전문가의 슬롯을 MMA의 한 차원에 올린다"는 같은 형태입니다.
  - 그 경로로 바꾸면 prose shadow는 53 → 3.6–7.3 ms/층-배치입니다. prose pp512는 151 → 약 192(+27 %)[derived], lcg는 오늘 0입니다(shadow가 union 그늘 아래).
  - 단, gemv의 워프 트리와 합 순서가 달라 **프리필 = 스텝 비트 불변식을 옮깁니다**(float sum order 클래스, 사용자 결정).
- **B4(프로젝션 넷을 IMMA로)도 같은 커널의 "전문가 하나짜리 스택"입니다**(gemm.rs 헤더). B1 뒤 route에서 −8.3…−11.9 ms/층-배치[derived]이고, 같은 비트 문제로 역시 사용자 결정입니다.
- **메가커널, 프로젝션 그리드의 stream-K, PDL은 모두 0에 가깝거나 쓸 수 없습니다.** B1 뒤 런치 간격은 0.35 ms/층-배치이고, PDL은 cc 9.0 전용입니다.
- **스트리밍은 새 공개 수치 둘이 우리 유도와 맞습니다.** 한 대 박스의 V4 계열 pp 실측(vLLM A100)과, 전체 층 이중 버퍼를 끄면 19–26 %를 잃는다는 FreeToken의 수치입니다.

---

## 1. 질문별 결과

### Q1. Ampere에서 작은 m 양자화 GEMM/GEMV의 그리드 형상

**llama.cpp** @ `4e7481175cbd4759df8bee2f1c1a0073effbebd7`

- **m ≤ 8은 MMVQ다.** `ggml/src/ggml-cuda/mmvq.cuh:3` — "`#define MMVQ_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVQ kernels.`"
  - sm_86(Ampere)은 아키텍처별 분기에 걸리지 않고 `mmvq.cu:427` "`return ne11 <= MMVQ_MAX_BATCH_SIZE;`"로 빠진다.
  - `mmvq.cu:322` — "`// k-quants cost more to decode and mvq redoes that per column, so MMQ wins sooner.`"
- **MUL_MAT_ID의 Q3_K는 Ampere에서 5열까지만 MMVQ다.** `mmvq.cu:291-292`에서 Turing 이상 Ada 미만은 `get_mmvq_mmid_max_batch_turing_plus`로 가고, `mmvq.cu:185` — "`case GGML_TYPE_Q3_K:    return 5;`"
- **MMVQ 형상은 5–8열에서 블록당 워프 2개 × 2행이다.** K는 워프끼리 나눈다.
  - `mmvq.cu:464` "`return 2;`"(calc_nwarps, 5–8열)
  - `mmvq.cu:591` "`return 2;`"(calc_rows_per_block)
  - `mmvq.cu:626` — "`constexpr int blocks_per_iter = vdr * nwarps*warp_size / qi;`"
- **8열을 넘으면 MMQ(IMMA)이고, Ampere 설정은 256스레드, launch-bounds 최소 1블록, I = 128 × J 타일, stream-K다.**
  - `mmq-config-ampere.cuh:148` — "`CASE(GGML_TYPE_Q3_K, 256, 1, 128,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K, MMQ_ITER_K, true, false);`"
  - `mmq.cuh:959` — "`// The mul_mat_q kernel implements "stream-k" work partitioning as described in https://arxiv.org/abs/2301.03598`"
- **stream-K 그리드는 SM 수이고, 타일 효율이 90 % 이상일 때만 타일링으로 돌아간다.** `mmq.cuh:1452` — "`const dim3 block_nums_stream_k(GGML_CUDA_CC_IS_NVIDIA(cc) && tiles_efficiency_percent >= 90 ? ntiles_dst : nsm, 1, 1);`"
- **J는 열 타일 수가 가장 적은 값으로 고른다.** `mmq.cuh:1506` "`if (ntiles_x < ntiles_J_best) {`"
  - MoE에서 NVIDIA는 전문가당 평균이 아니라 토큰 수로 J를 잡는다. `mmq.cu:300` — "`// On RDNA3 and RDNA4 it is faster to pick the tile size against this value instead of ne12.`"
  - 빈 열 타일은 `mmq.cuh:1120` "`if (jt*J >= col_diff) {`"에서 건너뛴다.

**vLLM Marlin / MoE-Marlin** @ `ddd6fbca148a`

- **영구형 그리드다.**
  - `csrc/libtorch_stable/quantization/marlin/marlin.cu:362` — "`int blocks = sms * exec_cfg.blocks_per_sm;`"(MoE판은 `moe/marlin_moe_wna16/ops.cu:341`)
  - `marlin_template.h:242` — "`// Each threadblock processes one "stripe" of the B matrix with (roughly) the`"
- **MoE의 m 타일은 전문가당 평균 토큰으로 8/16/32/48/64 중에서 고른다.** `vllm/model_executor/layers/fused_moe/experts/marlin_moe.py:343` — "`if M * topk / E / block_size_m < 0.9:`"
- 두 커널 모두 cp.async 다단 파이프라인이다(`marlin_template.h:52` "`const int stages,  // number of stages for the async global->shared`").

**Machete는 Hopper 전용이다.** `csrc/libtorch_stable/quantization/machete/machete_mainloop.cuh:3` — "`//   cutlass/gemm/collective/sm90_mma_tma_gmma_rs_warpspecialized_mixed_input.hpp`", `:9` "…`to more closely match the shape of the wgmma instructions`". sm_86에는 wgmma가 없다.

**exllamav3** @ `6b84a21b6f1e`(이전 조사와 같은 커밋이지만 아래 `moe_coop`은 그때 다루지 않았다)

- **GEMM은 stream-K형 슬라이스다.**
  - `exllamav3_ext/quant/exl3_gemm_inner.cuh:99` "`int num_slices = gridDim.x;`"
  - 그리드는 SM 수로 자른다(`exl3_kernel_map.cu:173` "`int max_slices = size_k / tilesize_k * size_n / tilesize_n;`")
- **작은 배치 MoE는 한 전문가의 슬롯 묶음을 MMA 행으로 올리고, 블록 안 16워프가 k를 나눈다.**
  - `exl3_moe_coop_kernel.cuh:11` — "`//                 tile with those slots as the rows of one m16 MMA, so an expert's weights are read`"
  - `:40` "`constexpr int ROWS = 8;`"
  - `:336` "`// block splitting k across its 16 warps.`"
- **타일 폭은 런타임 분기가 아니라 따로 인스턴스화한다.** `exl3_moe_kernel.cuh:21` — "`// 128-register budget and the 16-row path pays for tiles it never runs (measured +60-70% on`"

**mistral.rs** @ `2370966bb91e`: 프리필 grouped MoE는 dp4a 타일이다(IMMA 아님).
- `mistralrs-quant/kernels/moe_grouped/moe_grouped.cu:712` — "`// Grid: (ceil(N/MMQ_Y), num_experts), Block: (32, MMQ_NWARPS)`"
- `:715` "`#define MMQ_X 64`"

**MonoMoE** (arXiv 2609.04244, H200, 디코드 전용): 토큰 타일을 MMA의 N 차원에 두고 CTA를 전문가 가중치 타일로 나누는 weight-major 영구 커널이다.
- https://arxiv.org/html/2609.04244 — "MonoMoE places the complete decode-step token tile on the fine-grained tensor-core N dimension and partitions CTAs over expert-weight tiles"
- 토큰이 늘면 이득이 준다: "At B = 32 only 8% remains—the baseline GEMMs already sustain \sim 67% of peak."

**Stream-K** (https://arxiv.org/abs/2301.03598): "our Stream-K parallelization of GEMM produces a peak speedup of up to 14$\times$ and 6.7$\times$"

SGLang과 kt-kernel의 GPU 쪽 작은 m 커널은 받지 않았다. 그래서 거기서 나온 주장은 없다.

### Q2. 하이브리드 CPU+GPU MoE 프리필

- **vLLM PR #56118 / #56120** (2026-09, 박스 2× A100-PCIE-40GB + EPYC 9654, **DeepSeek-V4-Flash**)
  - https://api.github.com/repos/vllm-project/vllm/issues/56120 — "| 4 137 | 439.4 s | **5.8 s** | 76× |"(CPU 프리필 대 층별 GPU 가중치 스트리밍)
  - 같은 표에서 2,091토큰은 5.6 s다. 그래서 373 / 713 tok/s @ 2,091 / 4,137, 단일 A100이다[derived].
  - "**Kernel-time breakdown of a 16 K prefill** … MoE kernels 31 %, elementwise kernels 29 %, H2D 19 %, dense GEMM 18 %, **attention + indexer 2.9 %**."
  - #56118: "the GPU path (stream one layer's weights H2D, compute, free) is a separate, out-of-tree mechanism."
- **FreeToken** (arXiv 2608.16157)
  - https://arxiv.org/html/2608.16157v1 — "Taking an FP4 deployment of DeepSeek-V4-Flash as an example, it requires transferring roughly 140 GB of expert weights and adds … five seconds on RTX 4090- and 3090-class desktops (PCIe 4.0 x16, \sim 25 GB/s)"
  - 140 GB ÷ 5.6 s ≈ 25 GB/s[derived]라서 위 vLLM의 2k–4k 평평한 벽시계가 PCIe 바닥이라는 것과 맞는다.
  - 설계: "It allocates two full-layer buffers … a dedicated transfer stream simultaneously loads the complete expert set of layer l+1 … the transfer can start before that layer's routing is even known."
  - 수치: "Disabling the second buffer serializes transfer against computation and costs 19% of throughput at 4k tokens, 25% at 8k, and 26% at 16k"(PCIe 5.0 ×16, 52.7 GB/s)
- **OSDI'26 로컬 MoE** (arXiv 2606.10493, 2× EPYC 9355 24ch DDR5 + 2× RTX 5090)
  - https://arxiv.org/html/2606.10493v1 — "we use the AVX FP8 kernel for requests up to 4K tokens and switch to SLP/DSLP for longer contexts."
  - 링 길이: "For short prompts (transfer-dominated), we set the length to the number of experts of a layer … we shrink the length to 2 (ping–pong)"
- **ATSInfer** (arXiv 2607.10183): https://arxiv.org/html/2607.10183v1 — "KTransformers delivers low throughput on consumer platforms because its optimized AMX-based CPU kernels are unavailable on consumer CPUs"
- **ik_llama.cpp** @ `1aaf7105be6e`, `ggml/src/ggml-cuda.cu:5245`: 오프로드 임계를 가중치당 바이트에 비례시킨다. "`min_batch_size = int(1.*ctx->offload_batch_size_per_byte*row_size/src0->ne[0]);`"
- KTransformers HEAD `c40722b`는 이전 조사가 읽은 커밋과 같다. 새 자료는 없다.

### Q3. 런치 오버헤드와 스케줄링

- **런치 큐 깊이.** https://forums.developer.nvidia.com/t/kernel-launches-blocking-when-1024-kernels-in-a-queue/33866 — "Now we found the actual number of kernels in a queue when the others become blocking, is 1024." 2014년 사용자 경험값이다. 우리 실측 Q 1,068 activity와 같은 급이다.
- **런치 비용.**
  - Hazy(H100): https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles — "running on a CUDA stream incurs a launch cost of about 2.1 microseconds, and with CUDA graphs the launch cost only decreases to around 1.3 microseconds"
  - MPK(B200): https://arxiv.org/html/2512.22219 — "a kernel-per-operator execution of Qwen3-8B issues 293 kernel launches per token. On B200, each launch costs 3.8 \mu s in eager execution … and 0.8 \mu s with CUDA Graphs … its in-kernel scheduler accounts for only 0.28% of total runtime"
- **MPK는 A100에서 돈다.**
  - "In all experiments, MPK reserves four SMs for schedulers … The remaining SMs are used as workers."(A100은 108 SM 중 104가 워커)
  - 가변 배치: "MPK generates multiple t Graphs specialized for representative batch sizes, using powers of two up to the maximum batch size."
- **MonoMoE의 해석**: "Under CUDA graphs, these gaps are small relative to total kernel runtime. The dominant opportunity lies within the kernels themselves"
- **CUDA 그래프 기능과 아키텍처 요건.**
  - PDL은 https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html — "Available starting with devices of compute capability 9.0"
  - 조건 노드와 디바이스 그래프 런치는 https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html 에 있고, 그 페이지에는 compute capability 요건이 적혀 있지 않다. sm_86에서 도는지는 확인하지 않았다.
  - 인용: "device graph structure is fixed at time of instantiation and cannot be updated without re-instantiation"

### Q4. 희소·압축 어텐션 프리필(pre-Hopper)

- **vLLM #57144** (2026-09-16, A800 sm_80, **V4.1-Flash**): https://api.github.com/repos/vllm-project/vllm/issues/57144
  - DeepSeek 레퍼런스 TileLang 커널: "| `sparse_attn` (BF16 gather + online softmax + attn sink) | **works** | compiles and matches a torch reference, max rel err 2.9e-3 (bf16-level) at h=64, d=512, topk=512 |"
  - 막히는 것은 fp8/fp4 MMA와 DeepGEMM 인덱서다: "The indexer requires DeepGEMM, which has no sm_80 support."
- **vLLM #56120**: "**Indexer**: the scoring pass (`_indexed_d512_split_score/value`) is kept as-is; only the fp8 encode/decode changes to uint8 on Ampere."
- **LiteTopK / LiteDSA** (vLLM #48726, SM100): https://api.github.com/repos/vllm-project/vllm/issues/48726
  - 이웃 쿼리를 묶어 KV 합집합을 한 번만 읽는다. V4 C128A에서: "The measured union size is only 1.002× that of a single token."
  - LiteTopK는 근사다: "| 262,144 | 13.0469 | 10.6117 | 1.2266× | 99.9977% |"(recall)
- **DeepSelect** (vLLM #56464): "Blackwell-only for now: DeepSelect upstream compiles sm_100a/sm_103a only."
- **FSA** (arXiv 2508.18224, H20/H200): https://arxiv.org/html/2508.18224v2 — "This strategy reaches kernel efficiency only when each Grouped Query Attenti[on group has a large number of query heads]". FSA는 루프 순서를 뒤집어 이 문제를 푼다.
  - V4.1 MLA는 64헤드가 KV 선택을 공유하므로 기존 루프 순서로 이미 타일이 찬다[I].

---

## 2. 우리 레버에 대응

기준값은 flowcal §2 / §5(main `9626c7f`, A6000, lcg / prose-in)이다. 층-배치당 벽시계를 **route + max(union, shadow)**로 두면 두 P = 512 관측을 재현한다[derived].
- prose: 29.8 + max(34.9, 53.0) = 82.8(관측 84.75)
- lcg: 29.8 + 62.6 = 92.4(관측 94.0)

### grouped shadow 형상 (llama.cpp MMQ-ID, exllamav3 `moe_coop`, Marlin MoE, MonoMoE) — 가장 큰 항

**오늘의 실효**
- 슬롯 수는 22.1 ms ÷ 23.51 µs = 940(lcg), 53.0 ÷ 23.51 = 2,254(prose)다.
- 카드 전문가가 층당 약 70개이므로 전문가당 13.4 / 32.1열이다.
- 슬롯당 35.39 M MAC이면 33.3 / 79.8 GMAC이고, 실효는 약 3.0 TOPS다. A6000 int8 피크 309.7의 1 %다[derived].

**재료는 저장소에 있다**
- `crates/gpu/src/gemm.rs`의 `gemm_q3k` / `gemm_q4k`: BM 128 × BN 64, IMMA, 카드 위 route table, 블록은 (타일, 128행 slab)마다 하나다.
- `crates/gpu-gates/src/bin/gate_gemm.rs:5-10`이 V4.1 gate 스택을 검증한다.

**예측[derived]**
- 속도 R은 우리 Qwen3 전 경로에서 약 44 TOPS 상당이다(pp4096 8,024 × 약 5.5 GFLOP/tok, 5.5는 이전 조사의 [I]). 여기에 BN 64 열 채움(21 % / 50 %)을 곱하면 9–22 TOPS다. 그러면 shadow는 lcg 7.2 ms, prose 7.3 ms다.
- BN 16 / 32 변형이면 lcg 1.8 ms(DRAM 바닥 1.17 GB ÷ 700 GB/s = 1.7), prose 3.6 ms다.
- 벽시계: prose P 512은 84.75 − (53.0 − 34.9) = 66.65 ms → **192 tok/s(+27 %)**, prose P 4096은 86.3 − 16.4 = 69.9 → **266(+23 %)**이다. lcg는 shadow가 이미 union 그늘 아래라 0이다(h3tile-b 뒤 union 46.4에서도 0).

**cuda-oxide 표현**: 가능하다(이미 있음). 두 가지를 더해야 한다.
- `gemm_swiglu_quant`에 `swiglu_clamp`의 limit이 없다(gemm.rs에 limit/clamp가 없음).
- 활성값은 `GemmAct`(같은 `q3k_quantize_q8_1`)로 받아야 한다.

### 프로젝션 그리드

- **B1 앞의 청크 루프(m = 8).** llama.cpp라면 MMVQ(2워프 × 2행/블록, K 분할)를 쓴다. qkv가 1,792 ÷ 2 = 896블록이 되어, 오늘의 224블록(0.44웨이브)보다 웨이브가 채워진다[derived]. 다만 B1이 청크 루프를 없애므로 **B1 뒤에는 0**이다.
- **B1의 m-col 커널에 stream-K.** 웨이브가 28 / 520 / 260 / 81이라 꼬리는 qkv에서 1/28 = 3.5 % 이하다. 약 0.03 ms/층-배치로 **약 0**이다[derived].
- **B4(IMMA).** 프로젝션 넷은 126.6 M MAC/토큰 × 512 = 64.8 GMAC/층-배치다. 20–44 TOPS에서 2.9–6.5 ms이고, B1의 14.8 대비 **−8.3…−11.9 ms**다. lcg pp512는 B1의 148.1 → **164–172(+11…+16 %)**[derived]이다.
  - gemm.rs 헤더: "with a one-expert stack the dense projection". 새 커널이 아니라 배선이다.
  - B4 GEMM의 qkv는 14 slab × 8타일 = 112블록 / 168 자리(0.67웨이브)다. stream-K를 붙여도 약 0.1 ms 이하다[derived].

### G 스케줄러와 런치 큐

- 큐가 1,024 근처에서 막힌다는 것은 우리 Q = 1,068과 맞는다. B1 뒤 924 < Q이므로 호스트는 막히지 않는다(flowcal).
- **디바이스 간격.** 우리 0.898 µs/activity는 이미 그래프 수준(0.8–1.3 µs)이다. 그래프로 바꿔도 디바이스 간격은 거의 그대로다.
- **route 그래프 캡처.** B1 뒤 호스트 enqueue 4.5 ms/층-배치가 절감의 상한이다(lcg는 serial이라 86.4 → 81.9, +5.5 % 이하)[derived]. 가변 T는 MPK처럼 2의 거듭제곱 버킷과 노드 파라미터 갱신으로 받는다.
- **메가커널.** B1 뒤 간격은 0.35 ms/층-배치로 벽시계의 0.4 %다[derived]. MonoMoE도 큰 m에서는 8 %뿐이다. **라운드를 열지 않는다.** PDL은 cc 9.0이라 쓸 수 없다.

### 호스트 union

새 AVX2 자료는 없다. ATSInfer도 AMX 부재가 소비자 CPU의 약점이라고 적어 "AMX 이식 금지"를 뒷받침한다. h3tile·cpugemm-lit 결론은 바뀌지 않는다.

### 스트리밍 교차점

- **바닥.** 214 GB ÷ 26.2 GB/s = 8.2 s/프롬프트이므로 전부 스트리밍할 때 pp4096 ≤ 500이다[derived]. FreeToken(140 GB, 약 25 GB/s에서 5 s)과 vLLM A100 V4-Flash(4,137토큰에 5.8 s)가 같은 법칙을 실측으로 보인다.
- **설계.** 라우팅 전에 다음 층 전체를 받고 버퍼 둘을 쓴다. 둘째 버퍼를 끄면 −19…−26 %다. P ≥ 512에서 384 전문가가 모두 쓰이므로 라우팅과 무관한 선반입이 정확하다.
- **VRAM.** 층당 호스트 몫 5.3 GB × 2 = 10.6 GB다. A6000은 가득 차 있으므로 3090을 스트리밍 카드로 쓰거나 상주 전문가를 내린다(이전 조사와 같음).
- **ik의 바이트 비례 임계.** 이전 조사의 s* = b·R_cpu / (2 β_link)와 같은 형태다. 레버가 아니라 교차 확인이다.

### 희소 어텐션 프리필

- P ≤ 1024(m = 2 인코더)와 P ≤ 512(m = 1)에서는 top-512가 전부라 dense causal이다(이전 조사).
- **B3(배치 전체 query).** LiteDSA의 "이웃 쿼리의 합집합 ≈ 1.002×"가 P = 4096 인코더 층에서 배치 query 한 번에 KV를 한 번 읽는 근거다. 항은 attention 4.10 ms/층-배치 중 −1.3…−1.9 ms다[derived, cardroute B1+B3 − B1].
- DeepSeek TileLang `sparse_attn`(h 64, d 512, topk 512)이 sm_80에서 맞는다는 것이 레퍼런스 형상이다. cuda-oxide로 쓸 수 있다(BF16/f32 gather + online softmax, fp8 MMA 없음).
- vLLM A100 16K에서 attention + indexer가 2.9 %이므로 P ≤ 4096에서 이 항은 작다.

## 3. `prefill-lit-report.md`를 뒤집거나 고치는 곳

1. **「같은 급 박스에서 공개된 V4.1 pp 숫자는 없다」(§1, §6)를 일부 고친다.**
   - V4 계열 한 대 박스 실측이 생겼다. V4-Flash, A100-PCIE-40GB, 층별 스트리밍, 2,091 / 4,137 / 16,039토큰에서 5.6 / 5.8 / 17.7 s다.
   - 같은 박스의 CPU 하이브리드는 112.9 / 439.4 s였다.
   - V4.1 자체의 수치는 여전히 없다.
2. **§Q3 「int8 MMA가 있으면 모든 배치에서 MMQ」를 보탠다.** HEAD `4e74811`에서 sm_86은 dense ≤ 8열이 MMVQ이고, MUL_MAT_ID Q3_K는 ≤ 5열이 MMVQ다. MMQ는 그 위이고, 그리드는 stream-K(블록 수 = nsm, 효율 90 % 이상이면 타일링)다.
3. **§Q1 ik 규칙 「배치×k ≥ 32×E」를 보탠다.** HEAD `1aaf710`에는 `offload_batch_size_per_byte`가 있어 임계가 가중치당 바이트에 비례한다. 이전 조사가 유도한 법칙과 같은 형태다.
4. **「FlashMLA sparse는 SM90/SM100 전용이라 알고리즘만 옮긴다」를 보탠다.** DeepSeek의 레퍼런스 TileLang `sparse_attn`이 sm_80에서 컴파일되고 맞는다(#57144 실측). vLLM에는 V4용 A100 Triton sparse-MLA 프리필이 있다(#56120).
5. **§Q3 종합 「3–4비트 K-quant 큰 m 경로는 사실상 MMQ 하나」를 보탠다.** 작은 m MoE에는 exllamav3 `moe_coop`(전문가 슬롯 묶음을 MMA 행으로, 16워프가 k 분할)가 둘째 설계다. Marlin MoE의 block_size_m 규칙(전문가당 평균 토큰으로 8–64)이 타일 폭 선택의 참조다.
6. **카드 R 약 50 TOPS 가정**은 우리 Qwen3 실측에서 유도한 약 44 TOPS 상당[derived]과 같은 급이다. 뒤집히지 않는다.

## 4. 순위 목록 (상위 5)

| 순위 | 할 것 | 예측 항 [derived] | 필요한 박스 측정 하나(기대값) | 비고 |
|---|---|---|---|---|
| 1 | **grouped shadow를 `gemm_q3k` / `gemm_q4k` 경로로**: `swiglu_clamp` limit 변형, `GemmAct`, 필요하면 BN 16/32 인스턴스(exllamav3처럼 따로) | prose shadow 53 → 3.6–7.3 ms/층-배치, pp512 prose 151 → 약 192, pp4096 prose 216 → 약 266. lcg는 0 | **prose shadow는 lcg의 슬롯당 비용을 옮긴 값이고 prose에서 잰 적이 없다.** 그래서 flowcal에 이미 있는 5분 실행이 먼저다: prose 프롬프트 팔 + `BLOOMERY_STEP_STATS=1` → `card_in`. 기대 45–53 ms. 35 ms 이하면 prose에서도 shadow가 union 그늘이라 이 레버는 0이다 | float sum order 클래스(σ 대 `exact-forced-32.tsv`, `gate-gpu-e2e` 개수 핀, 확인 1회). **프리필 = 스텝 비트가 깨지므로 사용자 결정** |
| 2 | **B4**: 프로젝션 넷을 전문가 하나짜리 스택의 `gemm_q3k`로 | route −8.3…−11.9 ms, lcg pp512 148 → 164–172(B1 위) | B1 착륙 시팅의 `card_proj`(0분, 기대 14.8 [9.2–16.6]). B4가 이기는 폭은 이 값 − 2.9…6.5다 | 1과 같은 비트 문제로 **사용자 결정**. 1과 한 번에 묶으면 재핀도 한 번이다 |
| 3 | **스트리밍 이중 버퍼**: 라우팅 전 다음 층 전체 선반입, 링 = 층 하나 × 2, 3090을 스트리밍 카드로 | FreeToken 기준 이중 버퍼가 없으면 −19…−26 %. 우리 pp4096 all-stream 천장은 약 500 | flowcal의 DRAM 프로브(`bench_v41_host union5:` + 채움 스레드, GPU 불필요, 5분): 채움 ≥ 26.3 GB/s에서 union −4…−10 % | hoststream-design과 같은 방향이고, 문헌이 순서를 확인한다 |
| 4 | **B1 뒤 route의 그래프 캡처**(T 버킷 + 노드 파라미터 갱신) | 호스트 enqueue −4.5 ms/층-배치 이하, lcg +5.5 % 이하 | B1 착륙 시팅의 `enqueue`(0분, 기대 4.5). 1 ms 이하면 열지 않는다 | 디바이스 간격은 이미 0.9 µs라 줄 것이 없다 |
| 5 | **B3**: 배치 전체 query의 causal MLA flash(P ≤ 1024는 dense, 4096 인코더 층은 합집합 한 번 읽기) | −1.3…−1.9 ms/층-배치 | B1 뒤 nsys 프리필 폼의 attention 항(기대 4.1 ms/층-배치 유지) | 비트 불변 설계가 가능하다(q3pflash의 V4.1판) |

하지 말 것:
- 메가커널, 영구형 route(간격 0.35 ms)
- B1 그리드에 stream-K(약 0)
- PDL(cc 9.0)
- Machete(wgmma)
- LiteTopK(recall < 100 %라 ik 오라클과 결별)
- DeepSelect(sm_100a)
- FSA 루프 뒤집기(64헤드 MLA에는 필요 없음)

## 5. 범위 밖에서 본 개선 여지(보고만, 손대지 않음)

- `crates/gpu/src/gemm.rs`의 `gemm_swiglu_quant`에 clamp limit이 없다. V4.1 shadow에 쓰려면 한 인자가 필요하다. 크기 S.
- `gate_gemm`은 V4.1의 **gate** 스택만 실제 파일로 검증한다. V4.1 down(Q4_K, 2304 → 5120) 모양은 Qwen3 모양의 합성 스택뿐이다. 크기 XS–S.
- `AGENTS.md`의 런치 큐 설명에 "Q = 1,068"과 공개 경험값 1,024의 대응이 없다. 크기 XS.

## 6. 실행 모델

opus(Opus 5.5)로 돌았습니다.
