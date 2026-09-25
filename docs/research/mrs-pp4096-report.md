# mrs-pp4096 조사 보고 (mistral.rs Qwen3-30B-A3B GGUF 프리필이 P = 4096에서 토큰당 2.6배 느린 이유, 2026-09-26 새벽)

> 리드 메모(2026-09-26). 아래는 `mrs-pp4096` 라운드(opus, 읽기 전용, 박스 실행·git 변경 없음)의 보고 원문이다. 결론은
> 우리 빌드 조건이다: 박스의 mistral.rs 바이너리는 `--features cuda`로만 빌드돼(리드가 fingerprint를 직접 다시 읽음 —
> `["code-execution", "cuda", "default"]`, flash 산출물 0개) 프롬프트 attention이 eager 경로(`naive_sdpa`, P × P 점수 전체)로
> 간다. 격차가 전부 그 항이라는 것은 유도이고 nsys 확인(§5)은 아직 안 했다. 처분:
> - 공개 표(README, rig-log 09-25의 mistral.rs 프리필 행 다섯, 09-26 q3gemma-ab)에 빌드 조건을 달았다(rig-log 09-26#mrs-noflash).
> - flash-attn 빌드와 같은 임대 재측정은 빌드가 30분을 넘을 수 있어 사용자 승인 뒤에 한다.
> - 업스트림 이슈는 내지 않는다(권장 빌드가 이미 피한다). eager 경로 개선(§8의 7)·candle softmax(§8의 8)는 트리아지로.
> - 러너 증인 보강(§8의 2·4: 빌드 피처를 증인에, flash 빌드 두 번째 경로)은 도구 항목으로.

---

# mrs-pp4096 보고 (리드 `bloomery-ee`에게)

## 결론

- **공개 표의 mistral.rs pp 행은 `flash-attn` 피처 없이 빌드된 바이너리를 잰 것입니다.** fingerprint에 `features= ["code-execution", "cuda", "default"]`만 있습니다. 그래서 프롬프트 attention이 eager 폴백(`naive_sdpa`)으로 갑니다. 이 경로는 P×P 점수를 전부 만들고, 층마다 32×P×P 원소 위로 일곱 번 지나갑니다. P = 4096에서 층당 24.7 GB, forward당 1.19 TB입니다 [derived].
- **이 항 하나가 토큰당 0.288 → 0.757 ms(격차 0.469 ms)의 사실상 전부입니다** [derived]. 나머지 P² 항은 호스트에서 만드는 P×P 마스크 하나이고, 격차의 1–2 %입니다.
- **mistral.rs의 결함이 아니라 우리 빌드 선택입니다.** 업스트림의 sm_86 릴리스 바이너리, install.sh, 문서 셋 다 Ampere에서 `flash-attn`을 켭니다. master `2370966bb`의 해당 파일 13개는 우리 트리와 바이트까지 같습니다. 그러니 고칠 곳은 우리 쪽입니다: 표에 각주를 달고, flash-attn 빌드로 다시 잽니다.
- **flash-attn 빌드를 예측하면 pp512 약 3,860–4,470, pp4096 약 3,630–4,190입니다** [derived]. 비선형 항 L(4096)이 llama.cpp처럼 줄면 pp4096은 약 6,300까지 갑니다. 어느 쪽이든 llama.cpp 기본 ub 행(4,318 / 4,205)과 같은 급이고, 우리(6,402 / 7,490)보다는 아래입니다.
- **업스트림 이슈는 올리지 않기를 권합니다.** 초안은 §7에 두었습니다.
- 디코드 행(`mrs:<D>`)은 영향이 없습니다(§2 끝).

## 0. 종이 위 분해와 자원 시간선 (박스 실행 0회)

**분해에 쓴 항.** 코드에서 가져온 것:
- 한 forward의 query 청크 수 ⌈P/1024⌉
- 청크마다 P² 패스 7개와 원소당 바이트
- 호스트 마스크 P²

이미 잰 숫자: TTFT 두 개(rig-log 09-25#q3ubatch-ab)와 whitepaper 천장값.

유도로 닫히지 않는 항은 하나입니다. **P = 4096에서 비어텐션 커널 합 L(4096)**, 즉 eager 어텐션 패스들의 실효 대역폭입니다. 기대값은 §5에 적었습니다.

**자원 시간선, P = 4096, 층당** [derived]:

| 자원 | 층당 바쁜 시간 | 근거 |
|---|---|---|
| 카드 SM | ≈ 63.8 ms (3,060 ms / 48) = 어텐션 42–44 + 선형 19–22 (L이 줄어드는 경우 ≈ 51 + 12) | 스트림 하나라 커널이 직렬로 돈다 |
| 호스트 CPU | 런치 약 70개 × 약 5 µs [가정] ≈ 0.3–0.5 ms, 카드 그늘 아래 / forward당 한 번, layer 0 전에 마스크 21–38 ms (이동안 카드는 논다) | `layers_masker.rs:82-84` |
| PCIe | 층당 0 / u8 마스크 16.8 MB 한 번(약 1.2 ms) | pageable 14.4 GB/s (`docs/facts.md:66`, 카드 미상) |
| 호스트 DRAM · NVMe | 0 (가중치는 카드에 상주) | — |
| 벽시계 | **카드 커널의 합 + 호스트 프롤로그(여기만 합)** | — |

레버는 항을 줄이는 것이 아니라 흐름을 바꾸는 것입니다: SIMT 융합(flash)이 P² 패스 일곱 개를 없앱니다. 그것이 곧 `flash-attn` 빌드 스위치입니다. eager 경로 안에서 항을 줄이는 방법은 §8의 7번에 있고, 폭은 −9 %, −37.5 %로 작습니다.

## 1. P = 4096 요청의 경로 (Q1)

**벤치 클록.** 요청 전송 직전부터 첫 스트림 토큰까지입니다.
- 시작: `bench.rs:431` `let request_start = Instant::now();` → `:432` `sender.send(req).await?;`
- 끝: `:479` `time_to_first_token: first_token.duration_since(request_start),`
- 처리율: `:280` `*prompt_len as f32 / ttft_seconds`
- 클록 안에 드는 것: 스케줄러 픽업, 입력 준비, 호스트 마스크, forward 한 번, 샘플 한 번, 채널.
- 워밍업: 같은 길이의 요청 한 개 전체(`:201-216`).
- 프로파일러 범위(`:249-253` 시작, `:318-320` 종료)는 측정 반복만 덮습니다.

**스케줄러 설정.** 벤치는 `.with_max_seqs(1)`(`:140`)과 `.with_prefix_cache_n(0)`(`:144`)만 주고 토큰 예산은 주지 않습니다. 그래서 기본값을 씁니다: `scheduler/mod.rs:22` `pub const DEFAULT_MAX_NUM_BATCHED_TOKENS: usize = 4096;`, `:23` `…PREFILL_CHUNK_TOKENS: usize = 512;`

**청크 결정.**
- CUDA에서는 스케줄러가 청크를 직접 봅니다: `engine/mod.rs:605` `pipeline.device().is_cuda() && !pipeline_metadata.is_xlora,`
- 단독 요청이라 `latency_bounded`는 false입니다: `paged_attention/scheduler.rs:347-350` `let latency_bounded = self … .any(|seq| get_mut_arcmutex!(seq).is_completion());`
- 그래서 청크 크기는 4096입니다: `:178` `let chunk_size = (self.prefill_token_budget(latency_bounded) / batch_size).max(1);`
- 512 상한은 디코드가 상주할 때만 겁니다: `:179` `if self.scheduler_visible_prompt_chunks && latency_bounded {`
- 청크 계획: `prompt_chunks.rs:268` `let mut end = (pos + chunk_size).min(next_feature_start).min(total_len);` → [0, P) 하나. Qwen3-MoE는 hybrid가 아니라 블록 정렬이 없습니다.
- 첫 청크 표시: `inputs_processor.rs:1418` `is_first_prompt_chunk: chunk_offset_toks == 0 && !has_any_cache_hit,`

→ **P = 512와 P = 4096은 각각 P토큰짜리 forward 한 번입니다. 512 청크 8개가 아닙니다.** 어텐션 안의 `ATTENTION_CHUNK_SIZE` 1024는 별개의 청킹입니다. query 축을 1024행씩 자르고, 각 조각은 키 P개를 전부 봅니다.

## 2. 프리필 attention (Q2)

**빌드 증거.**
- mistralrs-core fingerprint: `["code-execution", "cuda", "utoipa"]`
- `target/release/{deps,.fingerprint}`에 flash 산출물이 0개입니다.
- fingerprint JSON의 mtime `2026-09-17 21:05:30.91`이 바이너리 mtime `21:05:30.73`과 같습니다.
- 트리 HEAD는 `d5ae0f18f`이고 dirty 파일 하나(`mistralrs-quant/src/gguf/cpu.rs`)는 #2430 CPU 패치라 CUDA 경로와 무관합니다.
- flash 사용 여부는 컴파일 타임에 정해집니다: `utils/mod.rs:308-310` `#[cfg(not(any(feature = "flash-attn", feature = "flash-attn-v3")))]` / `pub const fn using_flash_attn() -> bool {` / `false`

**경로.**
1. `qwen3_moe.rs:727` `let attention_mask = CausalMasker.make_causal_mask(`
   → `layers_masker.rs:207` `if !cfg.force_custom && crate::using_flash_attn() && input_ids.device().is_cuda() {`가 false
   → 호스트에서 `flat_map` P×P 마스크(`:82-84`)
   → `:227` `Ok(crate::attention::AttentionMask::Custom(mask))` (rank 2, bf16).
2. `qwen3_moe.rs:234` `paged_attn.forward(`
   → 첫 청크에서는 gather 경로가 꺼집니다(`paged_attention.rs:778` `let mask_is_prefill = …`가 false, 캐시된 접두사 없음)
   → `try_regular_prompt`(`:1584`) → `:1620` `Sdpa.run_attention(`.
3. `attention/mod.rs:284` `if let AttentionMask::Custom(mask_tensor) = mask {` → `:292` `return self.run_attention_noflash(q, k, v, Some(mask_tensor), sdpa_params, do_causal);`
4. `:483` `let k = repeat_kv(k.clone(), sdpa_params.n_kv_groups)?;`(헤드 4 → 32)
   → `:486` `if mask.is_some_and(|x| x.rank() == 2) || mistralrs_quant::distributed::use_nccl() {` → `:487` `return naive_sdpa(`.
5. `naive.rs:38` `chunked_attention(…)`, 청크 크기는 `attention/mod.rs:62` `pub(crate) const ATTENTION_CHUNK_SIZE: usize = 1024;`. 청크마다 사각형 전체를 계산하고, 가려질 상삼각까지 계산합니다:
   - `:40` `MatMul.matmul_affine_mul(q_chunk, &k.t()?, …)`
     = `mistralrs-quant/src/lib.rs:893` `self.matmul(a, b)? * scale` (cuBLAS bf16 GEMM 뒤 `affine_bf16` 패스가 따로 붙음. `:892`에 `// TODO(EricLBuehler): Optimize this by using the gemm parameter?`. candle `tensor.rs:3069` `bin_trait!(Mul, mul, |v| v, |_| 0.);`)
   - `:49` `att = att.broadcast_add(mask)?;`
   - `:55` `…to_dtype(candle_core::DType::F32)?`
   - `:57` `candle_nn::ops::softmax_last_dim(&att)?`
   - `:59` bf16로 되돌림
   - `:61` `MatMul.matmul(&att, v)`

**dtype.** 활성값과 점수는 bf16(러너 설명 `depth-qwen3moe.sh:60`), softmax는 f32, 마스크는 P×P bf16입니다.

**점수 원소당 바이트** [derived]:

| 패스 | P = 512 | P = 4096 |
|---|---:|---:|
| QKᵀ 쓰기 | 2 | 2 |
| `* scale` 읽기 + 쓰기 | 4 | 4 |
| mask add 읽기 + 쓰기 + 마스크 | 4.06 (0.5 MiB 청크가 L2에 남음) | 6 (8 MiB 청크 > L2 6 MiB) |
| bf16 → f32 | 6 | 6 |
| softmax (max 읽기 / exp 읽기+쓰기 / 정규화 읽기+쓰기) | ≈ 8 (상주 행 1,344 × 2 KiB = 2.6 MiB) | 20 (1,344 × 16 KiB = 21 MiB) |
| f32 → bf16 | 6 | 6 |
| PV 읽기 | 2 | 2 |
| **합** | **32.1** | **46** |

- 상주 행 1,344 = SM 84개 × 블록 16개(블록당 워프 1개).
- 층당 원소 수 32P²: P = 512에서 8,388,608, P = 4096에서 536,870,912.
- forward당 바이트: P = 512에서 12.9 GB, **P = 4096에서 1,185 GB** (536,870,912 × 46 × 48).
- GEMM FLOPs 16,384·P²/층: P = 4096에서 13.2 TFLOP이고, 154.8 TFLOPS에서 바닥이 85 ms라 바이트 항 아래에 숨습니다.

**이 경로를 바꾸는 스위치.**
- 빌드 피처 `mistralrs-cli/Cargo.toml:65` `flash-attn = ["cuda", "mistralrs-core/flash-attn", "mistralrs-server-core/flash-attn"]`
  → `using_flash_attn()`이 true(`utils/mod.rs:314-316`) → 마스크가 `CausalFlash` → FA2 varlen causal(`backends/flash.rs:118-119` `let causal = flash_params.map_or(default_causal, |p| p.causal);`).
- 이때 K/V 복제도 없습니다: GQA 8이 `attention/mod.rs:63` `FLASH_ATTN_NATIVE_MAX_GQA_GROUP: usize = 8;`와 같고, 복제 조건은 `:303` `&& sdpa_params.n_kv_groups > FLASH_ATTN_NATIVE_MAX_GQA_GROUP`입니다.
- `flash-attn-v3`은 Hopper 전용입니다. `cudnn`은 `candle-core/cudnn`만 켭니다(`mistralrs-core/Cargo.toml:131`).
- **런타임 플래그로는 바꿀 수 없습니다.** `using_flash_attn`이 `const fn`이고, `--paged-attn off`(mrspa0 팔)도 같은 마스크를 들고 같은 `Sdpa.run_attention`으로 갑니다(`qwen3_moe.rs:266-274`).
- 우리 바이너리는 이 피처 없이 빌드됐습니다(위 fingerprint).

**디코드.** 디코드 클록은 첫 스트림 토큰부터라 프롬프트가 안 들어갑니다. 디코드 계획도 이 모델에서는 flash 피처와 무관합니다: `paged_attention/plan.rs:697` `AttentionBackendKind::Standard => Ok(Self::PagedAttention),`. FlashInfer 분기는 `cuda`와 `unix` 조건만 봅니다(`:685`).

**세 엔진의 프리필 attention 비교.** 우리는 q3pflash(텐서코어 flash), llama.cpp는 `-fa on` FA, mistral.rs는 이 빌드에서 eager이고 피처를 켜면 FA2(Dao) varlen causal입니다.

## 3. 그 밖의 P² 항, P·(증가) 항 (Q3)

- **호스트 마스크**(`layers_masker.rs:80-86`): P² u8을 `flat_map`으로 만들고, forward당 한 번이며, 그동안 카드가 놉니다.
  - P = 4096에서 1,680만 원소 × 1–2 ns[가정] + 재할당 복사 + H2D 16.8 MB ≈ **21–38 ms** [derived]
  - P = 512에서 ≈ 0.3–0.5 ms
  - 기기별 사본은 기기당 한 번뿐입니다(`device_map/mask.rs:32-36`).
- **KV 쓰기, `repeat_kv`**(`layers_utils.rs:8` `Tensor::cat(&vec![&x; n_rep], 2)?…`, 층당 64 MiB), **청크 cat**: 전부 O(P)입니다. `repeat_kv` 합이 약 3 GB, 약 4 ms입니다 [derived].
- **청크마다 접두사를 다시 읽는 일은 없습니다.** 청크가 하나이고 gather 경로가 꺼져 있습니다.
- **RoPE**는 O(P)이고 표는 적재 때 만듭니다.
- **MoE**는 GPU dispatch를 쓰고 동기화가 없습니다(`moe/experts/backends.rs:1473` "Build dispatch tables on GPU (no CPU-GPU sync)"; `moe_grouped.cu:1105-1132`는 count, 1-스레드 prefix, scatter). O(P·k)이고, MMQ의 `ncols_max`는 P입니다. 같은 구조로 llama.cpp가 ub 4096에서 6,904를 냅니다.
- **`maybe_synchronize`**(`naive.rs:11-25`): 여유 메모리가 4 GiB 미만일 때만 동기화합니다. 이 구성에서는 해당 없음 [derived].
  - 업스트림 #2322(핫패스의 sysinfo)는 통합 GPU만의 문제입니다: `memory_usage.rs:92` `if super::normal::is_integrated_gpu(device) {`
- **할당**: cudarc `cuMemAllocAsync` 풀(`cudarc-0.19.9 core.rs:1539-1540`)이라 forward 안에서 재사용됩니다. 작은 항입니다.
- **logits**는 마지막 위치 하나만 계산합니다(`pipeline/mod.rs:1017-1035`).

## 4. 토큰당 분해 (Q4)

**천장값 출처.**
- GA102 whitepaper v2 Table 3, RTX A6000 (내가 직접 받음, `nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf`): `Peak BF16 Tensor TFLOPS 154.8/309.6` (FP32 누산), `Peak INT8 Tensor TOPS 309.7/619.4`, `Memory Bandwidth 768 GB/sec`, `L2 Cache Size 6144 KB`, `SMs 84`.
- Ampere 튜닝 가이드(직접 받음): "The maximum number of thread blocks per SM is … 16 for GPUs with compute capability 8.6", "…for compute capability 8.6 it is 48" (워프).

**측정값.** TTFT(512) = 512 / 3,470.9 = **147.5 ms**, TTFT(4096) = 4096 / 1,321.1 = **3,100.5 ms**.

**분해** [derived; 어텐션 대역폭 560–768 GB/s, 호스트 1–2 ns/원소 가정]:

| 항 | P = 512 (ms / ms·tok⁻¹) | P = 4096 (ms / ms·tok⁻¹) |
|---|---|---|
| eager 어텐션 (12.9 GB / 1,185 GB) | 16.8–23.1 / 0.033–0.045 | 1,543–2,117 / 0.377–0.517 |
| 호스트 마스크 | 0.3–0.5 / 0.001 | 21–38 / 0.005–0.009 |
| 선형 L (MMQ dense·routed, 글루, 요청·샘플) | 123.9–130.4 / 0.242–0.255 (나머지) | 575–1,044 / 0.14–0.255 |
| 합 / 측정 | = 147.5 | 2,139–3,199 / **3,100.5** |

- L(512)의 교차 확인: llama.cpp pp512의 벽시계 전체가 118.6 ms입니다. mistral.rs의 선형 항은 그 1.04–1.10배입니다.
- L(4096)의 아래쪽 끝(575)은 llama.cpp의 ub 512 → ub 4096 토큰당 비율 0.625를 적용한 값입니다.

**격차 0.469 ms/토큰의 귀속.**
- 어텐션 +0.332…0.484, 마스크 +0.005…0.008, 선형 −0.115…0 (토큰당으로는 줄어들기만 한다).
- 선형 항이 토큰당 그대로라면 P = 4096 어텐션은 2,019–2,089 ms여야 합니다. 이는 1,185 GB를 **567–587 GB/s**(768의 74–76 %)로 읽는 속도이고, **격차 전체가 어텐션 몫입니다.**
- 768보다 낮은 이유는 PTX에 근거가 있습니다(`target/release/build/candle-kernels-e16b7607269c3d15/out/reduce.ptx`, `.target sm_86`, `softmax_f32`):
  - max 루프는 독립 `ld.global.f32`가 4개입니다.
  - exp 루프와 정규화 루프는 `ld`와 `st`가 번갈아 놓입니다(`x`와 `dst`에 `__restrict__`가 없음). 그래서 워프당 로드가 하나씩만 떠 있습니다.
  - 블록당 워프 1개, SM당 블록 16개(`candle-nn/src/ops.rs:362-363` `grid_dim: (n_rows as u32, 1, 1),` `block_dim: (1, 32, 1),`)이라 48 워프 중 16 워프만 씁니다.

**두 점 맞춤으로 일관성 확인** (상수항 c = 0 가정):
- b = (3,100.5 − 8 × 147.5) / (4096² − 8 × 512²) = 1.308e-4 ms/tok², a = 0.2211 ms/tok.
- P² 항은 4096에서 2,195 ms(TTFT의 71 %), 512에서 34 ms입니다.
- 4096 값은 어텐션 + 마스크 유도값과 맞습니다. 512 값이 유도(17–23 ms)보다 큰 것은 원소당 바이트가 46 → 32로 떨어지는데 맞춤은 P² 계수 하나로 강제하기 때문입니다. 그래서 이 맞춤은 일관성 확인으로만 씁니다.

**flash-attn 빌드 예측** [derived]:
- 마스크가 사라지고, FA2 causal 8,192·P²/층 = 6.6 TFLOP을 77–93 TFLOPS[가정: 피크의 50–60 %]로 돌려 71–86 ms입니다. P = 512에서는 1.3–2.2 ms입니다.
- pp512 = 512 / (L + 1.3…2.2) = **3,860–4,470**
- pp4096 = 4096 / (L(4096) + 71…86) = **3,630–4,190** (선형 항이 토큰당 그대로이거나 맞춤의 a일 때). L(4096)이 llama.cpp처럼 줄면 최대 약 6,300.
- 판정을 가르는 것은 §5의 L(4096)입니다.

## 5. 확인 실행 하나 (리드가 실행)

```
MISTRALRS_BENCH_CUDA_PROFILER_RANGE=1 nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop -t cuda -o mrs-pp4096 \
  /home/user/mistral.rs/target/release/mistralrs bench --format gguf -f <MODEL> --prompt-len 4096 --gen-len 1 \
  --iterations 1 --warmup 1 --max-seq-len 4352 --pa-context-len 4384
nsys stats --report cuda_gpu_kern_sum mrs-pp4096.nsys-rep
```

A6000에서, 임대 안에서 돌립니다. 벽시계는 적재 + 요청 두 번(약 6 s) + stats로, 수 분입니다 [추정]. 프로파일러 범위가 워밍업을 빼 줍니다.

**기대 커널 표** (측정 forward 한 번, [derived]):

| 커널 | 런치 | 바이트 | 기대 ms |
|---|---:|---:|---:|
| `softmax_f32` | 192 | 515 GB | 700–1,100 |
| `cast_bf16_f32` + `cast_f32_bf16` | 각 192 (+ 작은 것) | 각 155 GB | 합 420–580 |
| `badd_bf16` (어텐션 add, 잔차 add는 각 약 50 MB) | 192 + 잔차 | 103–155 GB | 150–260 |
| `affine_bf16` | 192 | 103 GB | 140–190 |
| cuBLAS bf16 텐서코어 GEMM (QKᵀ, PV) | 384 | 119 GB, 13.2 TFLOP | 130–250 |
| copy2d/ucopy (`repeat_kv`, 청크 cat) | 약 150 | 약 4.6 GB | 5–10 |
| **어텐션 소계** | | ≈ 1.15–1.20 TB | **1,550–2,380 (측정 총합으로 보면 ≈ 2,000–2,100)** |
| `mul_mat_q`, `mul_mat_q_stream_k_fixup`, `quantize_mmq_q8_1(_glu)`, `moe_dispatch_*`, `moe_weighted_reduce_flat`, norm/rope/topk | 층당 약 20–30 | | **575–1,044 (≈ 950–1,050)** |
| layer 0 전 카드 유휴 (호스트 마스크) | | | 21–38 |

런치 192 = query 청크 4 × 48층입니다.

**반증 조건.** 어텐션 소계가 1.4 s 미만이거나, 선형 커널 합이 1.3 s를 넘으면(토큰당 512의 1.25배 초과) 선형 경로에 두 번째 항이 있다는 뜻입니다.

**진짜 고침(b).** `--features "cuda flash-attn"`으로 다시 빌드합니다.
- `mistralrs-flash-attn/build.rs:15` `KERNEL_FILES: [&str; 53]`에 CUTLASS 고정 커밋(`:9`)까지 받아야 해서 빌드 시간을 모르고, **30분을 넘길 수 있습니다. 사용자 승인 항목입니다.**
- 트리 밖 `CARGO_TARGET_DIR`에 빌드해 `d5ae0f18f` 바이너리를 남겨 두면, 같은 임대에서 A/B를 할 수 있습니다.

## 6. 업스트림 상태 (Q5)

**검색 쿼리.** 모두 `gh`로 읽기만 했고, 결과는 스크래치의 `gh-issues.txt`, `gh-prs.txt`에 있습니다.
- `gh search issues --repo EricLBuehler/mistral.rs "<q>"`: "prefill slow", "slow prefill", "long prompt", "prompt processing", "TTFT", "flash attention", "flash-attn", "naive_sdpa", "softmax", "qwen3 moe gguf", "qwen3moe prefill", "attention chunk", "prefill performance", "prompt throughput"
- `--include-prs`로 추가: "quadratic", "O(n^2)"(검색 구문이 거부함), "without flash", "flash-attn feature", "attention slow", "prefill tok/s", "pp4096", "mistralrs bench", "bench prompt-len", "qwen3 moe", "Qwen3-30B-A3B"
- `gh search prs`: "prefill", "flash attention", "flash-attn", "naive_sdpa", "sdpa", "softmax", "attention mask", "causal mask", "matmul_affine_mul", "long context", "prompt chunk", "chunked prefill", "cublaslt attention", "eager attention"
- GraphQL discussions: "prefill", "flash attention", "long prompt", "prompt processing", "TTFT", "slow"

**같은 증상은 없습니다.** 가장 가까운 것들과, 왜 이 증상이 아닌지:
- #1624(병합 안 됨) → candle-vllm #255/#256: 토큰 65,535개를 넘으면 `quantize_q8_1`의 gridDim.y 때문에 죽는 문제.
- #1591: 청킹 재작업.
- #2322 / #2325 / #2346: 통합 GPU의 sysinfo 문제.
- #2024: 여러 시퀀스의 bucket/preempt 문제.
- #153: 2024년 코드.
- #1310: 메모리 문제.

**master 대조.**
- master `2370966bb`(2026-09-25)는 우리 트리보다 `ahead_by=12 behind_by=0 files=46`입니다.
- 인용한 13개 파일(attention/mod.rs, naive.rs, layers_masker.rs, utils/mod.rs, quant lib.rs, qwen3_moe.rs, paged_attention.rs, 두 scheduler, bench.rs, cli/core Cargo.toml, install.sh) 모두 `SAME`입니다. 12개 커밋 중 어텐션 파일을 건드린 것은 없습니다. `paged_attention/plan.rs`(접두사·디코드 계획)와 candle 핀 이동(#2450)만 있습니다.
- **원인은 master에도 그대로입니다.** 다만 권장 빌드가 이미 이를 피합니다:
  - `release.yml:315` "flash-attn covers the attention path.", `:321` sm 86 `features: "cuda flash-attn nccl"`
  - `install.sh:367-369` cc ≥ 80이면 `flash-attn`
  - docs `cargo-features.md:25` "NVIDIA Ampere or Ada: `cuda flash-attn cudnn`"
- flash 경로 자체는 우리 트리보다 먼저 들어왔습니다(#1319, 2025-05-07 병합). 즉 우리 트리가 고침보다 오래된 것이 아니라, 우리가 빌드에서 그 경로를 빼먹은 것입니다.

## 7. 이슈 초안 — 올리지 않기를 권함

올리면 "flash-attn으로 빌드하라"는 답으로 닫힐 가능성이 큽니다. 초안:

> **Eager CUDA attention (no flash-attn) makes long-prompt prefill 2.6x slower per token**
>
> Measured: Qwen3-30B-A3B Q4_K_M GGUF, RTX A6000, v0.9.3 (d5ae0f18f), built with `--features cuda` only. `mistralrs bench --prompt-len P --gen-len 1`: 3,471 tok/s at P=512, 1,321 tok/s at P=4096.
>
> Cause (from code): without flash-attn, `make_causal_mask` returns a P×P mask (`layers_masker.rs:227`), so `run_attention_noflash` calls `naive_sdpa` (`attention/mod.rs:486`). It keeps full 32×P×P scores and makes seven passes over them. The scale is its own pass (`mistralrs-quant/src/lib.rs:893`). Derived: about 25 GB memory traffic per layer at P=4096.
>
> Suggestion: fold the scale into the GEMM and skip keys after each chunk's last query, or warn when CUDA runs without flash-attn.
>
> I used an AI assistant to read the code; the numbers are from our runs.

## 8. common.md 절

1. **변경 파일: 없음.** 스크래치에만 사본을 두었습니다: `/private/tmp/claude-501/-Users-hckim-repo-bloomery/b4e4a3ef-e524-4e47-91af-3afadc891c1e/scratchpad/{gh-issues.txt,gh-prs.txt,master/,tree/,ga102.txt,ampere-tuning.html}`

2. **판정을 싣는 읽기 전용 명령의 출력.** 게이트는 없는 조사 라운드입니다.
   - `mistralrs-cli-89b1636c39f8875b/bin-mistralrs.json features= ["code-execution", "cuda", "default"]`
   - `ls deps | grep -i flash` → 빈 출력, `---no flash artifacts above means none`
   - `2026-09-17 21:05:30.729265176 +0900 mistralrs`
   - master 대조에서 `SAME` 13줄, `ahead_by=12 behind_by=0 files=46`
   - softmax_f32 PTX에 `ld.global` 15개(위의 루프 패턴).

3. **예측 대 결과.** §4, §5의 예측만 있습니다. 확인 실행은 리드의 몫이라 결과는 없습니다.

4. **하지 못한 것.**
   - nsys, 시간 측정, 재빌드: 스펙이 금지했고, 재빌드는 승인이 필요합니다.
   - flash 빌드에서 프리필이 어디로 들어가는지(packed varlen인지 `try_regular_prompt`인지)는 끝까지 추적하지 않았습니다. 둘 다 FA2 varlen causal입니다.
   - master의 candle 핀 이동 전후로 softmax 커널을 비교하지 않았습니다.
   - 호스트 마스크의 원소당 1–2 ns는 가정입니다.
   - 실행 로그에 `FlashAttention is enabled.`(`normal.rs:969`)가 있었는지는 확인하지 못했습니다. 러너가 그 줄을 에코하지 않습니다.

5. **스펙 밖에서 보인 개선 지점** (보고만, 손대지 않음):
   1. `tools/ref/depth-qwen3moe.sh:370` REF_BATCH 문구는 512 청킹처럼 읽힙니다. 단독 요청은 P ≤ 4096이면 forward 한 번입니다. XS.
   2. `depth-qwen3moe.sh:123`의 "does not print its cargo features"는 사실이 아닙니다. fingerprint JSON이나 `mistralrs doctor`의 `Build features:`(`cargo-features.md:70`)를 증인에 넣고, `:421` grep에 `FlashAttention is enabled`를 더하면 이번 일을 잡았을 것입니다. XS.
   3. `depth-qwen3moe.sh:145-149`는 기본값의 출처를 `mistralrs-cli src/args/mod.rs`라고 적습니다. 벤치는 `mistralrs_for_server_builder.rs:320-321`을 거쳐 `scheduler/mod.rs:22-23`의 값을 씁니다. XS.
   4. `tools/ref/models/qwen3moe.sh:64-66`은 MRSBIN이 하나뿐입니다. flash 빌드용 두 번째 경로가 있으면 같은 임대 A/B 팔을 둘 수 있습니다. S.
   5. `docs/research/q3gemm-lit-report.md:9`의 "mrspp 행은 그 엔진의 최선 경로다"는 GEMM에만 맞고 어텐션에는 틀립니다. 리드 문서.
   6. `README.md:69-74`와 rig-log 09-25의 mistral.rs 행(`:140`, `:329`, `:439`, `:478`)에 빌드 조건(flash-attn 없음) 각주가 필요하거나, 다시 재야 합니다. 리드.
   7. 업스트림 eager 경로. 성능 항목이라 트리아지로 보내고 바로 PR하지 않습니다.
      - (a) `mistralrs-quant/src/lib.rs:891-894` scale 패스 제거: 46 B 중 4 B, −9 %.
      - (b) `attention/mod.rs:121-173`에서 causal일 때 청크마다 키를 자르기: ×0.625, −37.5 %.
      - (c) flash 없이 CUDA로 돌 때 경고 한 줄. 각 XS–S.
   8. candle `softmax`(`candle-kernels/src/reduce.cu:233`, `candle-nn/src/ops.rs:362-363`): 블록당 워프 1개(16/48 워프)이고 `__restrict__`가 없습니다. huggingface/candle 후보입니다. S.

6. **모델.** opus(Opus 5.5)로 스폰돼 그대로 돌았습니다.
