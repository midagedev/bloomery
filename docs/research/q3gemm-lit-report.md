# q3gemm-lit — exllamav3·mistral.rs(Marlin, CUTLASS grouped, MMQ 포트)가 가중치를 텐서코어에 먹이는 방식과 우리 GEMM에 옮길 것 (라운드 보고, 2026-09-25)

리드 메모: 파동 25 대기 창의 박스 없는 조사 라운드(opus, 스펙 `specs/wave-m6/spec-q3gemm-lit.md`)다. 사용자가 "exllama나
mistral.rs 쪽 리서치"를 물어 연 라운드이고, `q3next-design`의 레버 A·B′ 모양을 정한다. 리드가 원본과 대조한 인용 넷:
`mistralrs-quant/src/gguf/packed_affine.rs:95`(rank 2만), `:297`("Off by default"), exllamav3 `modules/block_sparse_mlp.py:43`
(`FUSED_DET` 기본 "1"), `mistralrs-core/src/moe/experts/backends.rs:1506`(`grouped_moe_mmq_pair_packed`) — 넷 다 일치.
결론: A는 2단·2 블록/SM·20워드 피치 그대로(3–4단·swizzle·`ldmatrix`는 우리 커널에서 0[유도]), B′는 A 뒤 go(512스레드,
`q8_1_quant_vals` 재사용으로 비트 동일 구성). 공개 비교에 닿는 사실: mistral.rs의 GGUF MoE 프리필은 llama.cpp MMQ 포트
(int8 텐서코어)이고, 어떤 플래그로도 GGUF 전문가를 Marlin·CUTLASS로 보낼 수 없다 — ~~우리 표의 mrspp 행은 그 엔진의 최선 경로다.~~ 정정(리드, 09-26): GEMM 경로에만 맞다 — 우리 mistral.rs 바이너리는 flash-attn 없이 빌드돼 프롬프트 attention이 eager 경로였다(`mrs-pp4096-report.md`, rig-log 09-26#mrs-noflash).
아래는 라운드의 보고 원문이다.

**q3gemm-lit 보고** (리드 `bloomery-ee`에게)

결론부터 적습니다.

1. **A는 참조 셋이 모두 쓰는 방식입니다.** Marlin, exllamav3, CUTLASS는 모두 가중치를 `cp.async`로 공유 메모리에 올립니다. 우리 A도 같은 방식이고, 모양은 "2단, 2 blocks/SM, 20워드 피치 그대로"가 맞습니다.
   - 3–4단, swizzle, `ldmatrix`는 우리 커널에서 얻는 것이 0입니다 [유도].
   - 참조 두 엔진은 적재 때 가중치를 재배열합니다. 이것만 우리에게 남는 추가 레버이고, 가중치 쪽 전역 파면을 약 −120 줄입니다 [유도]. 다만 디코드 커널이 같은 버퍼를 GGUF 순서로 읽으므로 사본이나 공용 레이아웃이 필요합니다.
2. **B′는 go입니다.** 네 엔진 어디에도 프리필 GEMM 에필로그에서 activation과 requant를 같이 하는 곳은 없습니다. 그래도 우리 쪽은 구성상 비트 동일로 만들 수 있습니다.
   - 모양: 512스레드 한 블록에서 워프 0–7이 gate, 8–15가 up을 맡습니다. 에필로그는 기존 `q8_1_quant_vals`를 그대로 부릅니다.
   - 순서는 A 다음입니다.
3. **설계 라운드 기록에서 정정할 것이 넷입니다.**
   - mistral.rs의 프리필 MoE는 `indexed_moe` dp4a가 아닙니다. llama.cpp MMQ 포트(int8 텐서코어)입니다.
   - 부분 타일은 11 %가 아니라 타일의 21.7 %입니다. 비용은 약 3 %가 아니라 오늘 5.8 %, A 뒤 3.7 %입니다 [유도].
   - exl3 coop은 bsz ≤ 32 한 커널이 아니라 bsz ≤ 8, 런치 둘입니다.
   - exl3 결정적 합산은 이 트리에서 기본값입니다.

경로 약어는 다음과 같습니다. 인용 라인은 모두 원문입니다.
- `M=/home/user/mistral.rs/mistralrs-quant/`
- `MC=/home/user/mistral.rs/mistralrs-core/src/moe/experts/`
- `X=/home/user/exllamav3-src/exllamav3/exllamav3_ext/quant/`
- `XM=/home/user/exllamav3-src/exllamav3/modules/`
- `C=/home/user/.cudaforge/git/checkouts/cutlass-7127592069c2fe01/include/cutlass/`
- `L=/home/user/llama.cpp-mainline/ggml/src/ggml-cuda/`
- `G=/Users/hckim/repo/bloomery/crates/gpu/src/gemm.rs`

## 1. 바뀐 파일
없습니다. 연구 라운드라 편집, 빌드, 게이트, 시간 재는 실행이 없습니다. 다른 에이전트에게 메시지도 보내지 않았습니다.

## 2. 실행한 것 (읽기만)

박스 읽기는 약 15회이고, 모두 `/Users/hckim/repo/bloomery`에서 `./tools/box.sh '<read>'`로 했습니다. 참조 소스는 `tar | base64`로 스크래치패드에 복사해 읽었고, 박스에는 아무것도 쓰지 않았습니다. 트리 커밋은 다음과 같습니다(git log 대신 `.git/HEAD`와 ref 파일로 확인, git 설정은 건드리지 않음).

| 트리 | 커밋 |
|---|---|
| exllamav3-src | `0740edc2da56` (detached) |
| mistral.rs | `d5ae0f18f217` (master) |
| llama.cpp-mainline | `53ed051ce519` (m-latest) |
| ik-idxkey | `db517b690a8c` (v41/idxkey-fix, 오라클 빌드와 일치) |
| CUTLASS | `7127592069c2` (mistral.rs 빌드가 고정: `M/build.rs:2` `const CUTLASS_COMMIT: &str = "7127592069c2fe01b041e174ba4345ef9b279671";`) |

mistral.rs 바이너리는 이 트리에서 빌드된 것입니다. `.git/logs/HEAD`의 clone 타임스탬프가 `1789643997`(2026-09-17 20:19 KST)이고, `target/release/mistralrs`의 mtime이 같은 날 21:05 KST입니다. rig-log 행도 `d5ae0f18f`로 적혀 있습니다.

웹 출처는 세 개이고 모두 직접 가져왔습니다.
- GA102 백서 v2 PDF: https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf (pdftotext로 추출)
  - Table 3 RTX A6000: `Peak FP16 Tensor TFLOPS … with FP32 Accumulate 154.8/309.6`, `Peak INT8 Tensor TOPS 309.7/619.4`
  - Appendix A RTX 3090: `Peak FP16 Tensor TFLOPS with FP32 … 71/142`, `Peak INT8 Tensor TOPS … 284/568`
  - `28 KB L1 + 100 KB Shared Memory`
- CUDA Ampere 튜닝 가이드: https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html
  - "For GPUs with compute capability 8.6, shared memory capacity per SM is 100 KB."
  - "CUDA reserves 1 KB of shared memory per thread block."
- Marlin 논문: https://arxiv.org/abs/2408.11743
  - "batchsizes up to 16-32 can be supported with close to maximum (4×) quantization speedup, and larger batchsizes up to 64-128 with gradually decreasing…"
  - Marlin은 디코드와 작은 배치가 설계 영역입니다.

### 종이 위 분해와 자원 시간선 (round 규칙)
- **단위:** T=4096의 꽉 찬 Q4_K 블록-스텝 하나(128×64×128)입니다.
- **항:** 설계 라운드의 L1TEX 파면 예산(`/Users/hckim/repo/bloomery/docs/research/q3next-design-report.md` §3)을 그대로 씁니다. 2,496 = 가중치 1,024 + B 프래그먼트 LDS 512 + s8 LDS 512 + d8 128 + 활성값 cp.async 약 320. A 뒤에는 1,880입니다.
- **시간선(GEMM 런치 하나):**
  - 호스트 CPU: enqueue 몇 µs뿐이고 카드와 겹칩니다.
  - PCIe, 호스트 DRAM, NVMe: 0.
  - 카드: L1TEX 62–66 % > issue 38 % > tensor 26 % > L2→SM 약 34 % > DRAM 12 %.
  - 벽시계는 합이 아니라 L1TEX 하나의 max입니다(모형 기준이고, ncu 대조는 아직입니다).
- **흐름 레버:**
  - 비동기 = A: SM 안에서 가중치 로드를 연산 아래로 옮깁니다.
  - 벌크 = B′: gate·up·SwiGLU·양자화를 한 런치로 묶고 f32 왕복 403 MB/층을 없앱니다.
  - SIMT: 이미 ubatch 전체에 걸쳐 있습니다.
- **그늘 아래라 가치 0인 항:** issue를 줄이는 `ldmatrix`와 `lop3`(issue는 바인딩이 아님), 3번째 단(지연은 이미 가려짐), swizzle(뱅크 충돌이 이미 0).
- 모든 질문이 코드와 산술로 닫혀서 새 박스 실행은 없습니다.

## Q1. 가중치 스테이징 표

regs는 모두 미측정입니다(빌드와 `ptxas -v`를 돌리지 않았습니다).

| 엔진 | 단 수 | cp.async가 싣는 것 | smem 배치 / 충돌 회피 | 가중치 프래그먼트 읽기 | 디퀀트 위치·형식 | 스케일·영점 | MMA | 스레드 / smem / 블록/SM (sm_86) |
|---|---|---|---|---|---|---|---|---|
| **Marlin** (mistral.rs 사본) | **4**: `M/kernels/marlin/marlin/marlin.cuh:32` `static constexpr int pipe_stages = 4;`, `marlin_kernel.cuh:769` `cp_async_wait<stages - 2>();` | A(f16) `:675`, **B(가중치)** `:685` `cp_async4(&sh_b_stage[b_sh_wr_delta * i + b_sh_wr + j], B_ptr[i] + j);`, 스케일 `:712` | A: XOR `:578` `(i % a_gl_rd_delta_o) ^ row` → `ldsm4` `:779`. B: **적재 때 재배열**해서 스레드 선형(`:494` `b_sh_rd = threadIdx.x * b_thread_vecs`), swizzle 없음 | LDS.128 `:785` `frag_b_quant[k % 2][i] = *reinterpret_cast<I4 *>(` | 레지스터. lop3 매직으로 f16 `:147` `int lo = lop3<(0xf0 & 0xcc) \| 0xaa>(q, LO, EX);` | 그룹 16/32의 f16. 자기 smem 단을 쓰고 새 그룹일 때만 fetch. 스케일은 **MMA 전에 f16으로 곱함** `:243` `frag_b[0] = __hmul2(frag_b[0], s);`, 영점 `:1041` | m16n8k16 f16→f32 `:36` | 256(`marlin.cuh:22`), 동적 96 KB(`:1417`) → 1블록/SM, 격자 = SM 수(`:1696` `int blocks = sms;`), 타일 k64×n256(`:1465`) |
| **exllamav3** GEMM | 4(shape 2, `X/exl3_kernel_map.cuh:57` `16, 32, 128, 4, 3`), MoE는 **3**(`X/exl3_moe_common.cuh:14`). 프래그먼트 링 3단 | A(Hadamard 회전한 f16) `X/exl3_gemm_inner.cuh:258`, **B(trellis)** `:270` `if (pred_b_gl[i]) cp_async(sh + EXL3_GEMM_BASE_THREADS * i + t, gl + load_b_gl[i]);` | A: XOR `:133` `k ^ ((m >> A_SWIZZLE_SHIFT) & A_SWIZZLE_MASK)` → `ldsm4` `:296`. B: 16×16 블록으로 미리 타일링, 선형 | 레인이 자기 조각을 디코드 `:307` `dq_dispatch<bits, cb>(shb, lane_id << 3, …)` | 레지스터. 비트스트림 + 코드북으로 f16 `X/exl3_dq.cuh:30` `return decode_3inst<cb>(w0);` | 그룹 스케일 없음(입출력의 suh/svh + Hadamard) | m16n8k16 f16, sm_86에서는 **fp16 누산** `:16-17` `#define EXL3_GEMM_H_ACC 1` | 512(K32, `exl3_kernel_map.cuh:63`), `SMEM_MAX (90 * 1024)` `:7` → 1블록/SM, 협력 슬라이스 `:99` |
| **CUTLASS grouped** (mistral.rs) | **4**: `M/kernels/cutlass_moe/grouped_mm_2x.cu:30-31` `…IdentityThreadblockSwizzle, 4, … kDeviceOnly>`. `C/gemm/threadblock/mma_multistage.h:489` `cp_async_wait<Base::kStages - 2>();` | A와 B 둘 다 bf16: `:311` `cp_async_zfill<kSrcBytes, kCacheOpA>(`, `:346` `…kCacheOpB>(` | Crosswise XOR `C/layout/tensor_op_multiplicand_sm75.h:185` `partition_contiguous_residual ^ (partition_strided_residual % 4);` (배치는 `default_mma_core_sm80.h:1446-1451`) → ldmatrix | ldmatrix | 없음. 비양자화 bf16만: `M/src/moe/cutlass.rs:130` `"cutlass moe path is bf16-only"` | — | m16n8k16 bf16→f32 | 128×128×32, 워프 64×64 → **4워프**. smem 4 × 16 KB = 64 KB [유도] → 1블록/SM. 격자 `grouped_mm_2x.cu:52` `cached = (max_active > 0 ? max_active : 1) * sm_count;` |
| llama.cpp MMQ (대조) | **없음.** `L/mmq.cuh`와 `mmq-load-tiles.cuh`에 `cp_async` 0건 | — | 가중치를 동기로 읽고 int8로 풀어 smem에 둠(설계 라운드 `mmq-load-tiles.cuh:711-786`) | LDS | smem에 풀기 전 | half2 `d·sc`/`dmin·m` | int8 mma | 256, 점유율 1, I128 × J≤128(`L/mmq-config-ampere.cuh:168` `CASE(GGML_TYPE_Q4_K, 256, 1, 128,  64, …, true, false);`), stream-k |
| **우리** (`24bc2b8`) | **2**, 활성값만(`G:711` `cp_async_wait_group(1);`) | q6 코드 `G:583`, s8 `G:596`, d8 `G:600`. **가중치는 싣지 않음**: 스텝 머리에서 동기 로드(`G:696` `$dec!(@load $w, base0 + sb * $sb_units, hh, t),`) | 활성값은 `G:99` `const B_COL_W: usize = 36;`(4워드 패딩). 가중치는 smem 없음 | 레인당 LDG.32 16개 | 레지스터. AND/시프트 `G:205` `let even = [$r0[i0] & m, …` | 헤더 12 B의 6비트 정수 스케일을 **i32 결과에 곱함**(`G:225-226` `mma_m16n8k32_s32_s8` → `isum += sc*d`), 128값마다 f32 접기(`G:366`) | m16n8k32 s8→i32 | 256, 정적 21,504 B, `G:1138` `#[launch_bounds(256, 2)]` → 2블록/SM, 128 regs |

공통점: 참조 셋은 모두 **1블록/SM + 깊은 파이프라인**이고, 우리는 **2블록/SM + 얕은 파이프라인**입니다.

## Q2. mistral.rs의 GGUF → Marlin, 그리고 mrspp 행을 만든 커널

**받는 타입.** Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K입니다(`M/src/gguf/packed_affine.rs:45-53`, `M/kernels/gguf_affine_packed/marlin_gguf_affine_repack.cu:116-127`).

**Q4_K의 매핑.** `:124` `MRS_FORMAT_TRAITS(MRS_GGUF_AFFINE_Q4_K, BlockQ4K, QK_K, 4, 32);`
- 슈퍼블록을 32값 Marlin 그룹 여덟 개로 풉니다.
- 그룹마다 스케일과 오프셋을 f32로 곱합니다: `:265` `scale = d * static_cast<float>(quant_scale);`, `:266` `offset = dmin * static_cast<float>(quant_min);`.
- 그 곱을 f16으로 반올림합니다: `:411` `scales[output_index] = Scalar<T>::from_float(scale);`.
- 결과: 6비트 sc/m은 d와의 f16 곱으로만 남습니다. ggml에는 없는 반올림이 하나 더 생깁니다.
- 니블은 그대로 Marlin 프래그먼트 순서로 재배열됩니다(`:330` `pack_order`, `repack_payload_u4_kernel` `:292`).
- 다른 K-quant: Q5_K는 8비트 페이로드와 그룹 32(`:125`), Q6_K는 8비트와 그룹 16(`:126`), Q3_K는 4비트와 그룹 16입니다.

**MMA.** `M/kernels/marlin/marlin_affine_f16.cu:17` `marlin_matmul<half, weight_type, true, bits, true>(` → `is_zp_float`입니다.
- f16 활성값 × f16 디퀀트 가중치(w = q·s − z를 f16으로 계산) → f32 누산입니다.
- 우리의 정확한 int8×int8→i32와는 수치 계약이 다릅니다.

**언제 쓰이는가.** 기본은 꺼져 있습니다.
- `packed_affine.rs:297` `// Off by default: the repacked copy is a second full set of weights`, `:304` `_ => Backend::Off,`. `MISTRALRS_GGUF_AFFINE_BACKEND=on|auto`일 때만 켜집니다.
- rank 2 가중치만 받습니다: `:95` `… || weight.shape().rank() != 2 {`. 즉 dense 투영만이고 **전문가(rank 3)는 절대 가지 않습니다.**
- 최소 배치는 Q4_K 8(`M/src/gguf/mod.rs:36`), Q6_K 128(`packed_affine.rs:31`)입니다.
- 켜져 있으면 `forward_raw`가 MMQ보다 먼저 시도합니다(`mod.rs:444`).

**GGUF MoE의 기본 경로 (mrspp 행 P=512와 4096).**
1. 전문가가 prequantized로 판정됩니다: `MC/backends.rs:250` `Some(_) => true`.
2. Fast 백엔드로 갑니다: `MC/mod.rs:166-167` `MoEExpertsBackendImpl::Fast(FastExpertsWeights::load_prequantized(`. 양자화된 경우는 `MC/config.rs:262` `Ok(Self::Fast)`입니다.
3. prefill이고 토큰이 32개 이상이면 grouped 경로입니다(`MC/backends.rs:1127` `|| forward.shape.num_tokens < GROUPED_PREFILL_MIN_TOKENS`의 반대편, `:23`이 32).
4. gate·up: `:1506` `GroupedGateUp::Packed(mistralrs_quant::grouped_moe_mmq_pair_packed(`. 이것은 llama.cpp MMQ 포트입니다.
   - int8 텐서코어: `M/kernels/mmq_gguf/mmq_mma.cuh:879` `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32`
   - sm_86 설정: `mmq_gguf.cuh:3574-3575` `__launch_bounds__(…, 1)`
   - stream-k 켜짐: `mmq_instance_q4_k.cu:225-227`
   - 전문가 스택은 3-D QTensor입니다(`M/src/gguf/fast_mmq.rs:1061` `weight.shape().dims3()`).
5. down: `:1650` `grouped_moe_mmq_from_glu_packed(`. 별도 `quantize_mmq_q8_1_glu`(`mmq_quantize.cu:238`) 뒤 MMQ입니다(`fast_mmq.rs:1114-1115`).
6. `indexed_moe` dp4a는 디코드(토큰 32개 미만) 경로뿐입니다(`backends.rs:1434` `indexed_moe_fused_decode(`).

**플래그로 경로를 바꿀 수 있는가.** 없습니다.
- `MISTRALRS_MOE_BACKEND=cutlass`는 양자화 전문가를 이름 붙여 거부합니다: `config.rs:151` `"MISTRALRS_MOE_BACKEND={backend} requires raw, unquantized expert weights"`.
- affine 플래그는 rank 3에 닿지 않습니다.
- 따라서 **어떤 플래그도 GGUF 전문가를 Marlin이나 CUTLASS로 보내지 못합니다.**

**pp4096 하락.** mistral.rs의 pp4096 1,321은 pp512 3,471보다 토큰당 2.6배 느립니다. 두 P 모두 같은 MMQ 경로라 GEMM으로는 설명되지 않습니다. P² 항(attention prefill)이 후보이고, 이 라운드 범위 밖입니다.

## Q3. 에필로그 퓨전 (B′)

**exl3 coop.** 헤더 주석은 `X/exl3_moe_coop_kernel.cuh:3` `// Fused decode-shaped MoE kernels (bsz 1..MAX_BSZN), two ordinary launches per layer:`입니다.
- 커널 A 블록 하나가 하는 일: (전문가 run ≤ 8슬롯, 열 그룹, 투영, split-k) GEMV 타일을 계산하고, 부분합을 전역 `gu_g`/`gu_u`에 씁니다.
- 동기화 대상: (슬롯, 128열 청크)마다 완료 카운터입니다. `:548-557` `arrive_last`, `:553` `*sh_flag = (atomicAdd(counter, 1) == expected - 1);`.
  - 앞에 `__threadfence(); __syncthreads();`가 있습니다.
  - 기대 도착 수는 `GPC * nproj * p.ksplit_a`입니다(`:758`).
- 마지막 도착 블록이 하는 일:
  - 부분합을 L2에서 다시 읽습니다(`:773` `load_f4_cg`).
  - split-k를 고정 순서로 더합니다.
  - Hadamard와 svh, bias를 적용합니다.
  - activation을 계산합니다(`:793` `float a0 = act_gate(p.act, p.gated, g0, u0, p.act_limit);`).
  - down의 suh를 곱하고 Hadamard를 한 뒤 **fp16으로 저장합니다**(`:800` `store_h4(p.act_out + off, …)`).
- requant는 없습니다. exl3 GEMM이 fp16 활성값을 받기 때문입니다.
- down은 별도 런치 B이고, 거기서도 last-arrival로 슬롯을 고정 순서로 더합니다.
- 카운터 리셋은 형제 커널이 합니다(`:695`, `:832`).
- 배치 상한: `XM/block_sparse_mlp.py:44` `MAX_BSZN = 8`, `:40` `ROWS = 8`.

**exl3 프리필 MoE 커널.** activation은 GEMM 에필로그가 아닙니다. 그룹 배리어 사이에 전역 fp16 임시 버퍼를 한 번 더 도는 패스입니다(`X/exl3_moe_kernel.cuh:207` `had_hf_r_128_guad_inner`).

**나머지 엔진.**
- llama.cpp: GLU를 MMVQ 에필로그에만 넣습니다(`L/mmvq.cu:872` `// fuse gate, bias, scales, and glu_op into the up projection`). 조건은 `L/ggml-cuda.cu:1812-1813` `//we only support fusion for ncols_dst = 1`이라 디코드 전용이고 requant는 없습니다.
- ik: `/home/user/ik-idxkey/ggml/src/ggml-cuda.cu:3073` `ggml_cuda_moe_up_gate_unary` 뒤 별도 `ggml_fused_mul_unary`(`:3302`, `:3332`)입니다.
- mistral.rs: 별도 `quantize_mmq_q8_1_glu`(`mmq_quantize.cu:238`)로, 우리 `gemm_swiglu_quant`와 같은 자리입니다.
- CUTLASS: 에필로그는 `LinearCombination`(`grouped_mm_2x.cu:28`)뿐이고, 그 뒤 `M/src/moe/cutlass.rs:294` `let act = super::cuda::act_and_mul(…)`, 이어서 2차 grouped GEMM입니다.

**판정.** 넷 중 어느 엔진도 프리필 GEMM 에필로그에서 activation과 requant를 같이 하지 않습니다. B′에는 선례가 없고, exl3 coop은 디코드, fp16, requant 없음입니다.
- 우리 쪽 선례: last-arrival 티켓 방식은 디코드 커널에서 +133 µs로 측정됐습니다(rig-log `log/2026-09-25.md#qwen3fuse-regression-nsys`의 (c)).
- **B′가 비트 동일인 이유(구성상):**
  - `GEMM_BM` = 128이 down 입력의 q8_1 128값 블록과 정확히 겹치므로, 한 블록이 열의 128값을 모두 가집니다.
  - `q8_1_quant_vals`(`/Users/hckim/repo/bloomery/crates/gpu/src/lib.rs:420`)는 레지스터의 값을 받는 코어입니다. `gemm_swiglu_quant`도 이 함수를 부릅니다(`G:1487`).
  - amax는 max라 순서에 무관하고, s8은 정수 합이며, 코드는 값마다 독립입니다.

## Q4. 전문가 불균형과 타일 스케줄

**유도** (균등 라우팅, c ~ Binomial(8T, 1/128), 블록 비용 F + V·nt_live, F는 가중치 파면 1,024(오늘)/408(A 뒤), V = 184 = 64+64+16+40):

| T | c | 타일/전문가 | 부분 타일 | 완전 포장 대비 초과 | 게이트 격자 wave tail 상한 |
|---|---|---|---|---|---|
| 4096 | 256 ± 15.9 | 4.483 | 0.975개 = **타일의 21.7 %** | **5.8 %** (오늘: 1,024×0.483 + 184×0.44 = 576 / 9,984), **3.7 %** (A 뒤) | 3,443블록 / 168 = 20.50 wave → ≤2.5 % |
| 512 | 32 ± 5.6 | 1.000 (전부 부분, nt_live 4.44) | 100 % | 47 % / 30 % | 768 / 168 = 4.57 wave → ≤9.4 % |

- T=512의 47 %는 스케줄링 손실이 아닙니다. 전문가마다 가중치 슬랩을 한 번은 읽어야 하므로, F를 32열에 나누는 것 자체가 구조 항입니다. 이것이 46.6 → 70.5 TOPS의 T 추세이고, 레버는 F를 낮추는 A뿐입니다.
- wave tail 두 값은 상한입니다. 부분 블록이 먼저 끝나고, 혼자 남은 블록은 파이프를 독점하므로 실제는 그보다 작습니다.

**참조 스킴과 비교:**
- **exl3 M_TILE 64/32/16** (`X/exl3_moe_kernel.cuh:171`): 토큰 축(패딩 행)을 줄이는 장치이고, 타일마다 B를 다시 디퀀트합니다. 우리는 n-타일 8열 단위로 이미 건너뛰고, 남는 항은 F 재읽기라 **0**입니다.
  - 전문가 할당은 동적 티켓(`:283` `sched[2 + group_idx] = num_groups + atomicAdd(&sched[0], 1);`)입니다.
  - 전문가 안에서는 SM 그룹(`exl3_moe_common.cuh:10` 기본 8)이 k 슬라이스를 락으로 합칩니다(`X/exl3_gemm_inner.cuh:838`).
  - 행 256을 넘는 전문가는 fp16으로 디퀀트한 뒤 cuBLAS로 갑니다(`XM/moe_batch_recon.py:5`, 상한 `block_sparse_mlp.py:40` `FUSED_ROWS_WIDE … 256`). T=4096의 우리 분포(256±16)라면 절반쯤이 그쪽입니다.
- **CUTLASS problem visitor**: 정적 라운드로빈(`C/gemm/kernel/grouped_problem_visitor.h:151` `tile_idx += grid_size;`, `gemm_grouped.h:319`)이고 문제마다 ceil 그대로라 **0**입니다.
- **llama.cpp stream-k** (`L/mmq.cuh:1067` `int kbc = …/ gridDim.x;`, MoE 빈 타일 `:1112` `if (jt*J >= col_diff) {`):
  - tail만 회수합니다(≤2.5 % @4096).
  - 하지만 k를 나눠 fixup에서 더하므로 f32 k순 접기 계약이 깨집니다. 비트 동일이 아니라 오차 모형 클래스입니다.
  - MoE에서는 kbc 공간이 전체 토큰 수(ncols_max)로 패딩됩니다.
- **Marlin stripe** (`marlin_kernel.cuh:343`): 같은 k 분할 클래스이고, 전역 합산은 `:1137` `// results in FP16 (but still reduce with FP32 compute).`입니다.
- **비트 동일한 대안:** route 표에서 부분 타일을 끝에 두는 LPT 순서입니다. 이득은 tail 상한 이하(@4096)이고 T=512에서는 0입니다.

**결론.** 정확 계약 아래에서 눈금을 넘는 스킴은 없습니다. BN 128(A2)은 스케줄이 아니라 F를 나누는 레버입니다. 64열 단위당 오늘 2,640 → 2,128, A 뒤 1,949 → 1,745 [유도]입니다.

## Q5. `gemm_block!`으로 옮길 수 있는 것 [유도]

기준은 설계 모형의 블록-스텝당 파면입니다(오늘 2,496, A 뒤 1,880).

| 요소 (출처) | 계약 | L1TEX Δ / A 파라미터 | 판정 |
|---|---|---|---|
| 가중치 cp.async → smem, 활성값과 같은 commit 그룹 (셋 모두) | 바이트 동일 | 가중치 1,024 → 약 408 (설계) | **A 그 자체, go** |
| 단 수 4 (Marlin, CUTLASS) / 3 (exl3 MoE) | — | 3단 = 3 × (10,496 + 10,240) + 512 = 62,720 B → 2 × 63,744 > 102,400이라 **1블록/SM**. 우리 스텝 벽시계는 2 × 3,780–4,000 ≈ 7,600–8,000 cyc(A 뒤 ≈5,700–6,000)로 전역 지연(수백–1,000 cyc [가정])의 5배 이상. CUTLASS 스텝은 k=32 = 128×128×32/512 MAC/clk ≈ 1,024 cyc라 4단이 필요. 동적 smem은 레포에 선례 있음(`/Users/hckim/repo/bloomery/crates/gpu/src/flash.rs:1247` `dynamic_shared = 56064,`), 막는 것은 점유율뿐 | **2단 유지** |
| `ldmatrix` (참조는 f16 A에만 씀. 양자화 B는 Marlin/exl3 모두 재배열 + LDS) | 동일 바이트 | qs에 `ldmatrix.x4` 두 번(행 g, g+8 × 청크 4개. 행 주소 16 B 단위 5g+c mod 8이 모두 다름) = 8파면 = LDS.32 8개와 **동일**. issue만 워프-스텝당 −6(752의 0.8 %) | 0 |
| swizzle 대신 20워드 피치 (CUTLASS `:185`, exl3 `:133`, Marlin `:578`은 모두 2의 거듭제곱 f16 행) | — | WT 피치 20워드는 헤더 16 + qs 64 B로 **패딩이 0**. 뱅크 {0,20,8,28,16,4,24,12}+t가 LDS.32에서 모두 다르고, LDS.128은 5g mod 8로 모두 다름 | 0 (피치 유지) |
| 활성값 패딩 제거 (`G:99` 36 → 32워드 + XOR) | — | 0파면, 단당 −1,024 B. s8을 i16으로 바꾸면(A3) 활성값 2단 + 가중치 3단 = 17,920 + 30,720 + 512 = 49,152 ≤ 50,176이 되지만, 3단 자체가 불필요 | 보류 |
| 헤더를 슈퍼블록당 한 번 (A0, smem 형태) | 동일 | 헤더 LDS.128 워프-스텝당 8 → 4, 블록 −32 | go (A 안) |
| 적재 때 슬랩 연속 재배열 (Marlin repack, `repack_payload_u4_kernel`, exl3 사전 타일링) | 니블·isum 불변, 위치만 이동 | A의 전역 측 약 200 → 약 80 (cp.async 20개 × 4줄), **−120 (−6.4 %)**, L2 부분 섹터 과다 fetch 제거. 대가: Q4_K를 읽는 디코드 커널도 같은 버퍼를 씀 → 사본(전문가 Q4_K 약 13.6 GB = 226.5 MB × 48 + 113 MB × 24 [유도]; mistral.rs도 같은 이유로 기본 off) 또는 디코드 재배열 | 제안만 (A 측정 뒤) |
| 1블록/SM + BN 128, ≤255 regs (Marlin 1블록, CUTLASS 128², llama.cpp J≤128, 점유율 1) | 동일 | 64열 단위당 1,880 → 1,676 (−11 %). 타일 초과 3.7 → 4.1 %. A 없이는 워프로 가리던 LDG 지연이 노출(`G:77` 주석의 전제) | A2, A 뒤 |
| B′ 투영별 워프 분할 | 비트 동일 (위 Q3) | 활성값 스테이지를 공유해 cp.async 320 → 단위당 160. 에필로그 약 1,100 / 32 단위-스텝 ≈ +34 → 단위당 약 1,754 (−7 %). 층당 −403 MB DRAM, 런치 −1 | go, A 뒤 |
| f16 MMA, 디퀀트 후 f16, MMA 전 스케일, fp16 누산, 디퀀트 + cuBLAS | **계약 변경** | 전이 불가. 게다가 A6000 f16 천장은 int8의 절반(154.8 vs 309.7 = **2.0×**, 3090은 71 vs 284 = **4.0×**). exl3 `EXL3_GEMM_H_ACC`는 GeForce의 fp32 누산 반속을 우회하는 장치 | 불가 |
| `lop3`/`prmt` 매직 (int → f16) | — | 우리는 `& 0x0f0f0f0f`와 `>> 4`(`G:203-211`) 뒤 s8 MMA라 이미 최소 형태. issue는 그늘 | 0 |
| stream-k / stripe (k 분할) | **f32 순서 변경** | tail ≤2.5 % | 불가 (정확 계약) |

참조 엔진의 속도는 메모리 쪽 설계(파이프라인, 재배열)에서 옵니다. 연산 천장은 우리가 2배 높습니다.

## 권고

**A의 모양:** 설계 그대로 가되 세 가지를 확정합니다.
- 256스레드, 2블록/SM, **2단**, 정적 smem 41,984 B(< 48 KB라 동적 smem 불필요).
- WT 피치 20워드 유지(swizzle 없음), qs는 LDS.32 넷, 헤더 LDS.128은 슈퍼블록당 한 번(A0 흡수).
- `ldmatrix`와 3–4단은 넣지 않습니다(둘 다 가치 0 [유도]).
- 파면 모형으로 A 1,880 → A0 포함 1,848입니다.
- 다음 후보 순서: 적재 때 재배열(−120, 디코드 소비자 해법이 전제) → A2(BN 128, 1블록/SM, −11 %).

**B′: go, A 다음.**
- 모양:
  - 512스레드 한 블록, 워프 0–7 gate, 8–15 up, 같은 (타일, 슬랩).
  - `launch_bounds(512, 1)` = 128 regs로 오늘과 같은 상한이고, SM당 16워프도 같습니다.
  - smem은 2 × (9,216 + 1,024 + 256 + 2 × 10,240) + 512 = **62,464 B**로, flash 선례처럼 `DynamicSharedArray`를 씁니다.
- 에필로그:
  - gate 워프가 f32 타일을 smem에 씁니다. 열 피치 132워드로 뱅크 충돌 없음 [유도], 33,792 B.
  - 배리어 뒤 up 워프가 자리에서 `silu_mul`을 계산합니다.
  - 배리어 뒤 16워프가 (열, 블록) 64개를 4라운드로 나눠 **`q8_1_quant_vals`를 그대로 호출**합니다. `gemm_swiglu_quant`와 파티션이 같으므로 비트 동일이 구성상 보장됩니다.
- 버리는 모양 둘:
  - (a) 스레드 하나가 두 투영을 모두 계산: acc+isum 128 regs → 1블록 8워프로 떨어집니다.
  - (b) exl3 last-arrival: f32 면 하나가 남고 fence와 atomic이 붙습니다. qwen3fuse (c)에서 +133 µs로 측정된 모양입니다.
- 이 모양도 `q8_1_quant_vals`가 q3/q4 면을 계속 쓰는 것은 그대로입니다(C(i)가 다룰 몫).

## 3. 예측 대 결과, 정정

스펙이 예측을 요구하지 않았고 박스 실행도 없습니다. 설계 라운드 기록에 대한 정정은 이렇습니다.

| 항목 | 설계 라운드 | 이 라운드 |
|---|---|---|
| (1) mistral.rs 프리필 MoE | `indexed_moe` dp4a | llama.cpp MMQ 포트, int8 m16n8k32 (위 Q2 사슬) |
| (2) 부분 타일 | 11 %, 약 3 % | 21.7 %, 5.8 %(오늘) / 3.7 %(A 뒤) |
| (3) exl3 coop | bsz ≤ 32, 한 커널 | bsz 1..8, 런치 둘(+rot) |
| (4) exl3 결정적 합산 | 기본은 atomic | 이 트리에서 결정적이 기본: `XM/block_sparse_mlp.py:43` `FUSED_DET = os.environ.get("EXL3_MOE_FUSED_DET", "1") != "0"` |

## 4. 못 한 것
- 참조 커널들의 regs와 우리 `gemm_q4k` regs(빌드와 ptxas 금지).
- L1TEX 바인딩 가정의 대조. 설계 라운드가 제안한 ncu `gemm_q4k` T=4096 실행이 판정하고, 표의 Δ는 모두 그 모형 안의 값입니다.
- 전역 지연 사이클은 측정하지 않은 가정입니다.
- mistral.rs pp4096 하락의 원인.
- exl3 `moe_batch_recon.py`의 세부와 kernel map의 shape 선택 로직은 깊게 읽지 않았습니다.
- 참고: `box.sh`를 main 트리에서 돌렸으므로 매 호출이 기본 원격 `~/repo/bloomery`로 rsync했습니다. 지시대로이고, 트리는 `24bc2b8`에서 clean이었습니다.

## 5. 스펙 밖 개선 지점 (보고만)
- `G:76-78`: "sixteen warps … hides a step's weight loads"는 A 뒤에 사실이 아니게 됩니다. A와 함께 고칠 주석입니다. XS.
- `G:99`: `B_COL_W` 4워드 패딩은 단당 1,024 B입니다. smem이 묶일 때(3단, B′+BN128) XOR swizzle로 회수할 수 있습니다. S.
- `/Users/hckim/repo/bloomery/docs/research/q3next-design-report.md:89`, `:206`, `:207`, `:208`: 위 정정 (2), (4), (3), (1)입니다. 리드 문서라 손대지 않았습니다. XS.
- 업스트림 후보(mistral.rs): pp4096이 pp512보다 토큰당 2.6배 느린데 MoE GEMM 경로는 같습니다. nsys 한 번으로 P² 항인지 확인하면 이슈가 될 수 있습니다. S(조사).
- 업스트림 참고(mistral.rs packed affine): Q4_K의 `d·sc`를 그룹마다 f16으로 반올림합니다(`repack.cu:265`, `:411`). 설계상 선택이라 결함은 아니지만, 정밀도 비교 때 알아둘 반올림입니다. XS.

## 6. 모델
opus(Opus 5.5)로 스폰되어 실행했습니다.
