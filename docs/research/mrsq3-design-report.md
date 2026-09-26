# mrsq3 보고 (mistral.rs Qwen3-MoE 프롬프트 경로의 항별 분해, 설계, 2026-09-26)

> 리드 메모: 공개 표의 Qwen3 pp4096 칸이 mistral.rs flash-attn 빌드와 1.015 ± 0.009로 비긴 뒤(rig-log 09-26#mrs-flash), 그 원인을 코드로 분해한 조사 라운드다. 박스 실행은 없다. 이 보고의 수치는 우리 창(nsys `ef00f78`)을 빼면 모두 유도이고, 코드로 닫지 못한 항(P 4096의 MMQ 합, 300–370 ms)은 §3.7의 nsys 한 번이 가른다. 결론과 다음 할 일은 `docs/plan-triage.md`의 mistral.rs 줄에 옮겼다. 아래는 라운드 보고 원문이다.


# mrsq3 보고 — mistral.rs Qwen3-MoE 프롬프트 경로의 항별 분해 (박스 실행 0회)

## 요약

- **청크.** P = 4096은 forward 한 번입니다. 512 청크 8개가 아닙니다.
  - 벤치 요청은 단독이라 `latency_bounded`가 false입니다(`paged_attention/scheduler.rs:347-350`). 그래서 청크 크기가 `max_num_batched_tokens` 4096입니다(`:176-198`, `scheduler/mod.rs:22-23`).
  - attention은 이번 청크의 q/k/v 위에서 도는 FA2 causal이고, 자라는 캐시를 다시 읽지 않습니다.
- **P = 4096에서 비기는 구조**:
  - mistral.rs가 짧은 항: attention(FA2) −4…−24 ms, q/k 노름·rope·KV 쓰기 −12…−17 ms, 라우터 logits −10.5…−12 ms.
  - 우리가 짧은 항: int8 GEMM +10…+61 ms, 활성값 양자화 +4…+12 ms.
  - 합이 +8 ms(519 대 511)입니다[유도].
- **P = 512의 격차 30.8 ms는 거의 전부 MMQ 몫입니다**(+29…+38 ms)[유도].
  - MMQ 타일이 128열이라, 토큰이 32 ± 5.6개인 전문가마다 곱셈을 4배로 합니다. 우리는 64열 타일이라 2배입니다.
  - MMQ는 SM당 1블록이고 단일 버퍼입니다.
- **유도가 닫지 못한 항은 하나, P = 4096의 MMQ 합(`mul_mat_q*` + fixup)입니다.**
  - 구조 모형으로는 260–363 ms입니다.
  - 벽시계에서 나머지 항(글루 + FA2 + 호스트)을 빼면 312–363 ms입니다.
  - 카드 밴드는 **300–370 ms**로 잡았습니다. 이것이 nsys로 잴 항입니다.
- **1.10× 도달 여부.**
  - 1.10×가 되려면 창이 471.9 ms 이하, 즉 **−39.5 ms**여야 합니다.
  - q3gemmb가 예측값(−58.9)대로 나오면 혼자서 1.147×입니다.
  - 같은 커널에서 레버 A가 0이었으니 q3gemmb가 0이라고 두면, FA + H1 + Q만으로는 1.066–1.091×로 모자랍니다. 꼬리 사다리(F2–F4, route)까지 끝내야 1.116–1.150×입니다(라운드 여섯 안팎).
  - 반대쪽 위험: mistral.rs에는 고칠 수 있는 비효율 두 개(빈 블록, 8배 gather-quantize)가 있고, 고치면 그쪽 pp4096이 약 8,310–8,830으로 오릅니다[유도].

## 1. 변경 파일

없습니다. 레포·박스 어디에도 쓰지 않았습니다. 빌드·게이트·타이밍·프로파일 실행과 git 상태 변경도 없습니다. 박스 접근은 `sed`/`grep`/`ls`/`stat`/`git log`뿐입니다.

## 2. 판정을 싣는 읽기 전용 출력

```
d5ae0f18f2170f10d30880cb7d21fb0880410e7b / " M mistralrs-quant/src/gguf/cpu.rs"
-rwxrwxr-x 2 user user 541793192 Sep 26 07:46 /home/user/mistral.rs/target/release/mistralrs
mistralrs-cli-984fb0872fa721d5/bin-mistralrs.json 2026-09-26 07:46:40.767727322 +0900   (flash-attn 포함)
mistralrs-quant/src/gguf/fast_mmvq.rs:53:pub const MMVQ_MAX_BATCH: usize = 8;
mistralrs-quant/src/gguf/mod.rs:36:pub(crate) const GGUF_AFFINE_MIN_BATCH: usize = 8;
packed_affine.rs:300-305  Ok("on")|Ok("auto") => Auto ... _ => Backend::Off     (Marlin 재포장은 기본 꺼짐)
mmq_gguf.cuh:3575  __launch_bounds__(ggml_cuda_get_physical_warp_size()*mmq_get_nwarps_device(), 1)
mmq_gguf.cuh:215   #define MMQ_TILE_NE_K 32     → smem 512 + 128·76·4 + 128·144 = 57,856 B > 51,200 → SM당 1블록
mmq_gguf.cuh:3519  float sum[mmq_x*mmq_y / (nwarps*warp_size)]   (타일 전폭 계산)
mmq_gguf.cuh:3717  if (jt*mmq_x >= col_diff) {   (빈 타일은 블록 하나를 받아서 건너뜀)
fast_mmq.rs:1478   num_tokens as i64,   (MoE MMQ의 ncols_max = 토큰 수)
flash_fwd_launch_template.h:274  Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>, Is_dropout, Is_causal
flash_fwd_kernel.h:1125  const int m_block = blockIdx.x;   grid(num_m_block, b, h) (:62)
backends.rs:23  const GROUPED_PREFILL_MIN_TOKENS: usize = 32;
scheduler/mod.rs:22-23  DEFAULT_MAX_NUM_BATCHED_TOKENS 4096 / DEFAULT_MAX_PREFILL_CHUNK_TOKENS 512
```

## 3. 분해와 예측

### 3.0 자원 시간선 (층당, 유도)

| 자원 | P = 4096 | P = 512 |
|---|---|---|
| 카드 SM (스트림 하나, 직렬) | ≈ 10.7 ms | ≈ 2.1 ms |
| 호스트 CPU (eager, 층당 런치 ≈ 33 × 5–15 µs [가정]) | 0.17–0.5 ms, 카드 그늘 아래. forward 안에 동기화 없음(디스패치 표를 GPU에서 만듦 `backends.rs:1474-1483`, 할당은 스트림 순서) | 같음, 여전히 그늘 |
| PCIe | 층당 0. forward당 H2D 메타 < 1 MB, D2H logits 0.6 MB | 같음 |
| 호스트 DRAM·NVMe | 0 (가중치 카드 상주) | 0 |
| 벽시계 | 카드 커널 합 + 호스트 앞뒤 2–6 ms [가정] | 같음 |

우리도 같은 모양입니다(q3tail: 호스트가 5.4 ms에 다 넣고 카드가 유일한 바쁜 자원). 흐름 수준(비동기·벌크)의 차이는 없고, 차이는 전부 커널 안에 있습니다. 단 하나의 흐름 차이는 MoE 타일 목록입니다. 우리 `gemm_route`는 살아 있는 타일만 발사하고, mistral.rs는 전문가 × ⌈P/128⌉ 격자를 전부 띄운 뒤 빈 블록을 버립니다.

### 3.1 커널 순서 (한 층, P 토큰, bf16 활성값)

`qwen3_moe.rs:514-533` 순서입니다.

1. **입력 RmsNorm.** `layers.rs:404-414` → candle `rms_norm`. 행당 1024스레드 블록 하나입니다(`candle-nn/src/ops.rs:580-583`, candle `35d7ae7`).
2. **q/k/v 투영.** `qwen3_moe.rs:200-201` → `ops.rs:6739` `try_fused_quantized_qkv`.
   - 공유 양자화 활성값 경로는 GGUF에서 None입니다(`lib.rs:2385`).
   - Marlin 재포장은 기본 꺼짐입니다(`packed_affine.rs:300-310`, `lib.rs:2416`).
   - 셋의 dtype이 같으면 `fast_mmq::fused_qkv`로 갑니다(`lib.rs:2445-2448`): `quantize_mmq_q8_1` 한 번에 MMQ 셋(`fast_mmq.rs:387-447`).
   - v가 Q6_K인 24층에서는 `lib.rs:2436-2437`에서 None이 되어 `ops.rs:6758-6762`의 세 `forward`로 갑니다: quantize와 MMQ가 각각 한 번씩(`gguf/mod.rs:299-320`).
   - 24/24 분할은 우리 창의 런치 수(`gemm_q4k` 288 = 층당 6, `gemm_q6k` 48)로 확인한 유도입니다. v와 down이 같은 24층에서 Q6_K입니다.
3. **q/k RMS 노름 + rope.** 커널 하나(`layers.rs:3076-3098` → `:2858` → `ops.rs:5587`). 출력은 토큰 우선 배치(`:5736`)라 뒤따르는 transpose가 복사를 만들지 않습니다.
4. **prompt attention.**
   - `qwen3_moe.rs:234` → `paged_attention.rs:1584-1628` `try_regular_prompt` → `attention/mod.rs:296-297` flash 가능.
   - GQA 8 ≤ 8이라 K/V 복제가 없습니다(`:300-318`).
   - `:359-367`에서 transpose 후 `flash_attn`, 곧 `flash_attn_v2`(`backends/flash.rs:59-118`)입니다.
   - 커널은 FA2 `flash_fwd_kernel`이고, sm86 causal hdim128은 **64×64 타일, 4워프**입니다(`flash_fwd_launch_template.h:268-274`).
   - m-블록은 오름차순(`flash_fwd_kernel.h:1125`)이라 LPT가 아닙니다.
5. **KV 쓰기.** `write_kv_cache`(`paged_attention.rs:1630-1640`), paged입니다. 이 청크의 attention은 캐시를 읽지 않습니다.
6. **o 투영.** transpose + reshape는 복사 없음(`qwen3_moe.rs:278-283`). `plain`, 즉 quantize + MMQ입니다(`fast_mmq.rs:762-765`).
7. **잔차 add** (candle bf16, `qwen3_moe.rs:527`).
8. **post-attn RmsNorm.**
9. **라우터 logits.** `qwen3_moe.rs:403` candle `Linear`, bf16 cuBLAS GEMM(T × 2048 × 128).
10. **top-k.** `ops.rs:274` → `sort.cu:1345-1366`: softmax, top-8, renorm을 워프당 한 행으로 합니다.
11. **디스패치.** `moe_grouped.cu:1105-1132`: memset → count(atomic) → prefix(**스레드 1개 순차**, `:645-657`) → memcpy D2D → scatter(atomic).
12. **gate·up.**
    - prefill ≥ 32 토큰이라 `forward_grouped`로 갑니다(`backends.rs:1121-1133`).
    - `grouped_pair_packed`(`backends.rs:1505-1517`, `fast_mmq.rs:1290-1499`)는 이렇게 돕니다:
      - `quantize_mmq_q8_1`이 **정렬된 32,768행을 gather**해서 양자화합니다(`:1404-1418`). x를 8번 읽습니다.
      - MoE MMQ 두 번이 f32 packed 출력 하나에 씁니다(`:1382`, `:1461-1485`). `ncols_max` = 토큰 수(`:1478`)입니다.
13. **down.**
    - `grouped_from_glu_packed`(`backends.rs:1648-1661`, `fast_mmq.rs:1251-1284`)입니다.
    - `quantize_mmq_q8_1_glu<float>`는 f32 gate/up을 읽어 SiLU·곱·양자화를 한 번에 합니다. k = 768이 `k_padded` 1024로 패딩됩니다(`:1092`).
    - 이어서 MoE MMQ가 f32 [T·8, 2048]을 씁니다(`:1133-1165`).
14. **가중합.** `moe_weighted_reduce_flat_bf16`, 한 패스로 bf16을 씁니다(`backends.rs:1678-1680`, `moe_grouped.cu:679-703`, grid (T, 8) × 256).
15. **잔차 add.**

MMQ 공통(`mmq_instance_q4_k.cu`):
- mmq_y 128(`:10-13`). mmq_x는 `ncols_max`로 고르니 P ≥ 128이면 128입니다(`:118-145`).
- stream-k는 타일 효율 ≥ 90 %이면 타일당 블록 하나이고, 아니면 84블록 + fixup입니다(`:63-72`, `:225-227`).
- `__launch_bounds__(256, 1)`에 smem 57,856 B라 SM당 1블록(8워프)입니다.
- 루프는 load → sync → mma → sync의 단일 버퍼입니다(`mmq_gguf.cuh:3523-3555`).
- MoE 출력은 f32입니다(`:3988`).

### 3.2 작업량 (코드의 모양, 유도)

- **층당 토큰당 MAC**: q 8.389M + k 1.049M + v 1.049M + o 8.389M = 18.874M(dense), MoE 8 × 3 × 768 × 2048 = 37.75M.
- **유효 int8 연산**: P = 4096에서 22.26 TOP, P = 512에서 2.78 TOP.
- **MMQ가 실제로 도는 패딩 포함 연산.** 전문가 열 c ~ Bin(8P, 1/128)입니다.
  - P = 4096: c = 256 ± 15.9라 E⌈c/128⌉ = 2.488, 318.5열에 해당합니다. MoE 192.4 + dense 77.3 G MAC/층 → **25.9 TOP**(패딩 24 %). 우리 64열 타일은 4.488 타일, 12 %입니다.
  - P = 512: 전문가 전부 128열 한 타일 → **8.35 TOP**(MoE 4배, 우리는 2배).
- **빈 타일 블록(SM당, forward 합)**:
  - P = 4096: gate 22,672 + up 22,672 + down 60,458 = 층당 105,802 → **SM당 60,458**.
  - P = 512: 층당 10,752 → **SM당 6,144**.
  - 블록당 0.3–0.8 µs[가정]로 두면 18–48 ms / 1.8–4.9 ms입니다.
- **MoE MMQ 출력 쓰기(f32)**: P = 4096에서 층당 469 MB로 22.5 GB이고 690 GB/s로 33 ms입니다. SM당 1블록이라 에필로그가 가려지지 않아, 드러나는 몫을 0–33 ms로 둡니다.

### 3.3 mistral.rs 커널별 종이 표 (한 forward, 유도)

| 커널 | 런치 P=4096 | 바이트/일 | ms @4096 | 런치 P=512 | ms @512 |
|---|---:|---|---:|---:|---:|
| `mul_mat_q<Q4_K,128,false>` + `<Q6_K,…>` | 288 + 48 | 패딩 25.9 TOP | **300–370 (구조 260–363, 벽 합 312–363)** | 288 + 48 | 82–91 |
| `mul_mat_q_stream_k_fixup` (k, v, o / 512에서는 q 포함) | 144 | 작음 | 위에 포함 | 192 | 위에 포함 |
| `flash_fwd_kernel<…128,64,64,4…, causal>` | 48 | 블록-타일 66,560(우리와 같음), 6.7 TFLOP 발행 | 65–85 (1.5 GHz 피크의 50–65 % [가정]) | 48 | 1.5–3 |
| `quantize_mmq_q8_1<bf16>` (qkv 또는 q·k·v 따로, o, MoE gather) | 192 | 층당 평균 314.8 MB | 22–30 | 192 | 2.7–4 |
| `quantize_mmq_q8_1_glu<float>` | 48 | 239 MB | 16–22 | 48 | 2.1–3 |
| `moe_weighted_reduce_flat_kernel` | 48 | 285 MB | 20–25 | 48 | 2.5–3.5 |
| candle `rmsnorm` | 97 | 33.6 MB | 5–12 | 97 | 0.6–1.5 |
| `qk_rms_norm_rope` | 48 | 75.5 MB | 5–8 | 48 | 0.7–1 |
| candle `badd` bf16 | 96 | 50.3 MB | 7–9 | 96 | 0.9–1.3 |
| `reshape_and_cache` | 48 | 16.8 MB | 1–2 | 48 | 0.15–0.3 |
| cuBLAS bf16 gemm (라우터) | 48 | 2.15 GFLOP | 1.5–3 | 48 | 0.4–0.8 |
| `moe_router_topk_kernel` | 48 | 1 MB | 0.5–1 | 48 | 0.2–0.4 |
| dispatch 3종 + memset + memcpy | 240 | 작음 | 1.2–2.5 | 240 | 0.8–1.5 |
| 임베딩, 최종 norm, 헤드(마지막 위치) | 수 개 | — | 0.5–1.5 | — | 0.3–0.8 |
| 호스트 앞뒤 (창 밖) | — | — | 2–6 | — | 2–4 |
| **합 / 측정 벽시계** | | | **= 519** | | **= 105.9** |

- P = 512의 MMQ 82–91 ms는 벽시계에서 나머지를 뺀 값입니다. 패딩 연산 기준으로 92–107 TOPS, 유효 연산 기준으로 31–33 TOPS입니다.
- P = 4096에 같은 패딩 처리율을 대면 242–282 ms이고, 빈 블록 18–48과 쓰기 노출 0–33을 더해 260–363 ms입니다.
- 벽시계는 312–363 ms를 요구합니다. 두 대역이 겹치는 곳이 MMQ의 유효 처리율 61–71 TOPS이고, 우리 GEMM은 73.7입니다.

### 3.4 우리와 항별 비교, P = 4096

우리 값은 실측(nsys `ef00f78` 창 511.1 ms), mistral.rs 값은 유도입니다.

| 항 | 우리 | mistral.rs | Δ(그쪽 − 우리) | 기제 | 우리 레버 |
|---|---:|---:|---:|---|---|
| int8 GEMM / MMQ | 302.1 | 312–363 | **+10…+61** | MMQ 128열 타일 패딩 24 %(우리 12 %). 빈 타일 블록 SM당 1,260/층. SM당 1블록·단일 버퍼 | q3gemmb(격차를 더 벌림) |
| attention 본체 | 89.0 | 65–85 | **−4…−24** | 블록-타일 수(66,560)와 SM당 8워프는 같음. FA2의 타일당 비-HMMA 명령이 적을 것으로 봄(SASS 미확인). 마스크는 대각 타일에만 | FA −9…−17, FS/FB, 천장 −35…−38 |
| q/k 노름 + rope + KV 쓰기 | 21.9 | 5–10 | **−12…−17** | 우리는 f64 합(FP64 바운드), f32 입출력 169.9 MB/층. 그쪽은 f32 합, bf16 92 MB | H1 −7.4…−10.0, H2 −9…−12. 바이트 차는 레버 없음 |
| 라우터 logits | 13.5 | 1.5–3 | **−10.5…−12** | 그쪽은 bf16 텐서코어(cuBLAS). 우리 계약(ggml f32 라우팅 비트)이 배제 | **레버 없음**(§3.6) |
| 라우팅(top-k, route/디스패치) | ≈ 3.9 | 1.7–3.5 | −0.4…−2 | 우리 `gemm_route`는 블록 1개 | 다중 블록 route ≤ −2 |
| 활성값 양자화 | 18.0 | 22–30 | **+4…+12** | 그쪽 gather가 x를 8번 읽음(134 MB/층) | Q −8.2…−8.5, F2 −7.7…−9.6 |
| swiglu + quant | 19.6 | 16–22 | −3.6…+2.4 | 우리가 아무도 안 읽는 q3/q4 면을 씀(50.3 MB/층) | Q |
| MoE 합산 + 잔차 add 둘 | 30.3 (combine 23.2 + add 7.1) | 27–34 | ≈ 0 | 그쪽은 bf16 잔차 | F4 −6.6…−7.0, F5′ |
| rms_norm | 11.4 | 5–12 | −6…+0.6 | f32 대 bf16. candle은 행당 블록 하나 | F2 |
| 임베딩·헤드, 호스트 | ≈ 1.5 / ≈ 0 | 0.5–1.5 / 2–6 | +1…+6 | | — |
| **합** | **511.2** | **519** | **+8** | | |

- **레버가 없는 차이 둘.**
  - bf16 잔차 흐름: 노름·add·qk-노름 바이트가 절반. ggml f32 계약과 e2e 핀이 배제합니다.
  - bf16 라우터.
- **P = 512 (우리 값은 유도: 벤치 TOPS와 창을 P에 맞춰 줄인 값)**:
  - 우리 GEMM ≈ 53.5 (MoE 1.86 TOP @ 47 TOPS + dense 0.93 @ 66), flash ≈ 2, 라우터 ≈ 1.9, head_norm ≈ 2.7, 꼬리 ≈ 13–16, 합 ≈ 73–77 (측정 75.1).
  - 그쪽 MMQ 82–91 대 우리 53.5로 +29…+38입니다. 이것이 격차 30.8 ms의 전부입니다.

### 3.5 pp4096 사다리 (7,890 대비, 유도, 카드 합 = 벽시계)

| 지점 | 창 ms | pp4096 | 대 7,890 |
|---|---:|---:|---:|
| 지금 (main) | 511.4 | 8,009 | 1.015 |
| **1.10× 선** | **471.9** | 8,679 | 1.100 |
| + q3gemmb (예측 −58.9) | 452.5 | 9,052 | 1.147 |
| + FA (−9…−17) | 435.5–443.5 | 9,236–9,405 | 1.171–1.192 |
| + H1, Q (−15.6…−18.5) | 417.0–427.9 | 9,572–9,823 | 1.213–1.245 |
| + F2, F3, F4, route (−21.5…−24.3) | 392.7–406.4 | 10,079–10,430 | 1.277–1.322 |
| q3gemmb = 0이고 FA + H1 + Q만 | 475.9–486.8 | 8,414–8,607 | 1.066–1.091 |
| q3gemmb = 0이고 F2–F4, route까지 | 451.6–465.3 | 8,803–9,070 | 1.116–1.150 |

- **넘는 조건.** q3gemmb가 혼자 −39.5 ms 이상(예측의 67 %), 또는 FA + H1 + Q와 함께 −4.0…−14.9 ms 이상이면 됩니다. 레버 A는 같은 커널에서 예측과 달리 0이었습니다.
- **여유 위험.** mistral.rs가 빈 블록(18–48 ms)과 gather의 8배 읽기(llama.cpp의 `dedup_bcast` 식이면 ≈ 8 ms)를 고치면 그쪽이 4096 / (519 − 26…56) = **8,310–8,830**이 됩니다. 그 위의 1.10×는 우리 창이 422–448 ms 이하여야 하므로, q3gemmb와 꼬리 사다리 대부분이 필요합니다.

### 3.6 새 레버 제안: 없음 (라우터는 기제만)

라우터를 읽었습니다(`crates/gpu/src/arch/qwen3moe/router.rs:883-979`): 32토큰 × 32행 블록, 512스레드, 워프당 8×8 f32 타일, 3단 cp.async, 64값마다 배리어.

한 런치 = 426K 사이클 @1.515 GHz에 대한 유닛별 수요 [유도]:

| 유닛 | 수요 (사이클) | 비율 |
|---|---:|---:|
| FMA | 99.9K | 23 % |
| 발행 | 132.6K | 31 % |
| LDS 파면 | 99.9K | 23 % |
| L2 → SM 262 MB | ≈ 2 TB/s[가정]로 ≈ 198K | ≈ 47 % |
| DRAM | ≈ 12 % | |

웨이브 7 / 6.1로 꼬리가 13 %입니다. 최대 수요가 스텝의 ≈ 47 %이므로 AGENTS 09-26 규칙상 지연 바운드이고, 유닛 하나를 줄이는 레버는 0으로 예측됩니다. 점유율(SM당 1블록, 16워프)이 바뀌기 전에는 레버가 없습니다. 천장은 FMA 바닥 66 µs, 즉 −10.3 ms입니다.

### 3.7 확인 실행 (리드가 한 임대에서; 나는 돌리지 않음)

```
BLOOMERY_BOX_ENV='BLOOMERY_LEASE_CARD=docs/cards/mrsq3-nsys.card' tools/box.sh '
 tools/ref/lease-hold.sh --card docs/cards/mrsq3-nsys.card -- bash -c "
  export CUDA_VISIBLE_DEVICES=<TIMING_GPU, tools/ref/timing-card.sh가 고르는 A6000>
  for P in 4096 512; do
   MISTRALRS_BENCH_CUDA_PROFILER_RANGE=1 nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop \
     -t cuda -o /root/nsys-mrs-flash-p\$P --export=sqlite \
     /home/user/mistral.rs/target/release/mistralrs bench --format gguf -f \$MODEL \
     --prompt-len \$P --gen-len 1 --iterations 1 --warmup 1 --max-seq-len \$((P+1)) --pa-context-len \$((P+33))
   nsys stats --report cuda_gpu_kern_sum,cuda_gpu_trace --format csv -o /root/nsys-mrs-flash-p\$P /root/nsys-mrs-flash-p\$P.nsys-rep
  done"'
```

- 벽시계는 적재 두 번과 요청 네 번으로 수 분입니다[추정]. `bench.rs:249-253`의 프로파일러 범위가 워밍업을 뺍니다.
- **P = 512도 같이 돌 가치가 있습니다.** MMQ 기제 셋이 P에 따라 다르게 자라기 때문입니다: 패딩 연산 ×3.1, 빈 블록 ×9.8, 출력 쓰기 ×8.
  - `cuda_gpu_trace`의 grid로 dense MMQ(q 1024블록, k/v/o 84블록)와 MoE MMQ(24,576 / 65,536블록)를 가릅니다. dense는 빈 블록도 패딩도 없어 순수 처리율을 줍니다.
  - dense 비(4096/512)는 ≈ 7–8, MoE 비는 3.4–4.4로 예측합니다. MoE 비가 3.1을 넘는 몫이 빈 블록 + 쓰기 몫입니다.
- **기대 표.** §3.3과 같습니다. 무효 조건: `flash_fwd_kernel` 48런치가 아니거나, P×P 크기의 softmax/cast 커널이 보이면 flash 경로가 아니므로 그 행을 버립니다.

카드 초안 (`docs/cards/mrsq3-nsys.card`, 리드가 씀):

```
kind: profile
question: which terms of mistral.rs's Qwen3-30B-A3B Q4_K_M prompt forward are shorter than
  ours at P = 4096, and is the MMQ sum the term the paper decomposition leaves open?
term: sum of mul_mat_q* and mul_mat_q_stream_k_fixup* kernel time (nsys cuda_gpu_kern_sum) in one
  P = 4096 prompt forward of mistral.rs d5ae0f18f built with cuda flash-attn, A6000 300 W
unit: ms
predict: 300..370
condition: one lease; A6000 pinned through TIMING_GPU; the binary target/release/mistralrs of
  2026-09-26 07:46 KST (fingerprint mistralrs-cli-984fb0872fa721d5, rig-log 09-26#mrs-flash sha256
  8da64b84d5e5); MISTRALRS_BENCH_CUDA_PROFILER_RANGE=1 limits capture to the measured iteration;
  the trace has 48 flash_fwd_kernel launches and no P x P softmax, else the run is void; the
  witness shows no other compute on the A6000; P = 512 in the same lease splits the MMQ terms
decide-in: write the per-term table to rig-log; ladder order stays q3gemmb, FA, H1/Q; the margin
  question rests on q3gemmb landing at least 4..15 ms beside FA+H1+Q
decide-below: FA2 or glue carries 40+ ms more than the byte/FLOP model; read those rows, and if
  flash_fwd_kernel > 90 ms drop FA behind the tail levers
decide-above: FA2 + glue are under 145 ms, so FA2 < 65 ms: move FA/FS/FB ahead of the tail
  levers and re-derive the attention row
```

## 4. 못 한 것

- **FA2와 MMQ의 SASS·레지스터.** 박스 명령 제한(sed/grep/ls) 때문에 cuobjdump를 쓰지 않았습니다. FA2 65–85 ms는 대역일 뿐입니다.
- **가정 넷**: 빈 블록 하나의 비용 0.3–0.8 µs, 호스트 앞뒤 2–6 ms, candle rmsnorm과 add의 대역 효율, L2 대역 ≈ 2 TB/s.
- **활성값 dtype bf16**은 러너 설명과 이전 보고에서 가져왔고, 로더의 기본값을 직접 읽지 않았습니다.
- **Q6_K 24/24 분할**은 우리 런치 수에서 유도했고, GGUF 헤더를 읽지 않았습니다.
- **벤치 토큰의 라우팅 분포.** 합성 반복열이라 균등 이항분포보다 치우칠 수 있고, 그러면 패딩이 늘어납니다. 확인하지 않았습니다.

## 5. 스펙 밖 개선 지점 (보고만)

1. `mistralrs-quant/src/gguf/fast_mmq.rs:1478`, `mmq_instance_q4_k.cu:18`: MoE 격자를 `ncols_max` = P로 잡아, P = 4096에서 MMQ 블록의 92 %가 빈 블록입니다(forward당 18–48 ms[유도]). llama.cpp도 NVIDIA에서 같습니다(`/home/user/llama.cpp/ggml/src/ggml-cuda/mmq.cu:249-252`). 우리 `gemm_route` 같은 압축 타일 목록이 해법입니다. 업스트림 성능 후보(두 레포), 트리아지로. M.
2. `fast_mmq.rs:1404-1418`: 정렬 행을 gather-quantize해서 x를 8번 읽습니다(층당 134 MB). llama.cpp는 `dedup_bcast`로 한 번 읽고 흩뿌립니다(`mmq.cu:191-193`, `:232-234`). 업스트림 후보. S.
3. `mistralrs-quant/kernels/moe_grouped/moe_grouped.cu:645-657`: prefix sum을 스레드 하나가 순차로 합니다. 작지만 층마다 지연입니다. XS.
4. `docs/research/q3tail-design-report.md:328`: mistral.rs MoE를 gather 경로(`backends.rs:1181`, 네 패스 combine)로 적었습니다. 프리필 ≥ 32 토큰은 grouped 경로이고 가중합이 한 패스입니다(`backends.rs:1121-1133`, `moe_grouped.cu:679-703`). 리드 문서 정정 한 줄. XS.
5. `packed_affine.rs:300-305`: `MISTRALRS_GGUF_AFFINE_BACKEND=on`이면 dense q/k/v/o가 Marlin w4a16(f16 텐서)으로 갑니다. 전문가는 3-D라 제외됩니다(`:95`). 공개 표는 기본값으로 재는 것이 맞지만, 추가 팔로는 대략 중립으로 예측합니다[유도: dense 3.71 T MAC을 60–90 TFLOPS로 82–124 ms, MMQ의 dense 몫은 89–104 ms]. 정보용. XS.
6. `crates/gpu/src/arch/qwen3moe/router.rs:892-898`: 512블록이 SM 84개에 6.1 웨이브라 꼬리가 13 %입니다. 블록 모양을 바꿀 때 같이 볼 항입니다. XS.

## 6. 모델

opus(Opus 5.5)로 스폰되어 그대로 돌았습니다.
