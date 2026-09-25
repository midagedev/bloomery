# q3router 보고 (Qwen3 ubatch 라우터 logits를 레지스터 타일로, 퓨즈드 라우터와 비트 동일, 2026-09-26 새벽)

> 리드 메모(2026-09-26). 아래는 `q3router` 라운드(opus)의 보고 원문이다. 같은 임대 A/B(rig-log 09-26#q3router-ab, A6000):
> pp4096 7,439 → **8,024**(1.079 ± 0.013), pp512 6,383 → **6,815**(1.068 ± 0.012), 디코드 그대로. nsys에서
> `qwen3moe_router_logits`는 런치당 1.19 → **0.281 ms**(48런치 57.2 → 13.5 ms). 예측(0.12–0.20 ms, pp4096 8,200–8,270)보다
> 느렸다 — 틀린 항은 발행 효율 가정(70–100 %)이다. 60.1 M 워프 명령을 0.281 ms에 내면 스케줄러당 발행이 약 40 %이고, 512블록이
> 84 SM에 1블록/SM(127 regs × 512스레드)이라 7웨이브(이상적 6.1)로 꼬리가 붙는다[유도]. 착륙은 비트 동일·게이트 녹색·+7.9 %로 한다.
> 5절의 개선 지점 중 ④(`ubatch.rs` 주석)와 ⑥(`q8f32.rs` top-k)은 리드가 착륙 커밋에서 고쳤고, ①②③⑤는 트리아지로 갔다.

---

# q3router 보고 (리드 `bloomery-ee`에게)

`qwen3moe_router_logits`를 새로 짰습니다. 512스레드 블록이 토큰 32 × 전문가 행 32를 맡고, 워프마다 8행 × 8토큰 레지스터 타일을 들며, 블록의 행·토큰 값은 `cp.async`로 공유 메모리에 3단(스테이지마다 64값) 올립니다. 모든 logit은 퓨즈드 디코드 라우터와 비트까지 같습니다(12개 T 전부 `ubatch_eq_fused_bits=true`).

- 소유 게이트는 전부 초록이고, FAIL-first 둘은 예측대로 빨강이 났다가 복원됐습니다.
- 예측은 T=4096에서 런치당 **0.12–0.20 ms**(오늘 1.19 ms), pp4096 8,200–8,270, pp512 6,790–7,120입니다.
- 시간을 재는 실행은 하지 않았습니다.
- 세션 리밋으로 한 번 멈췄고, 리드의 재개 메시지 뒤 같은 지점에서 이어 끝냈습니다. 어드바이저는 첫 호출이 시간 초과였고, 둘째 호출의 지적은 반영했습니다(판정은 바뀌지 않음).

## A. Part A — 코드 전에 종이에서 한 분해

### A.1 오늘 커널 (base `5dc685c`)

| 항 | T = 4096 | T = 512 |
|---|---:|---:|
| 격자 | 블록 8,192개 × 256스레드(워프 65,536) | 블록 1,024개 |
| FMA | 1.074 G(워프 FFMA 33.55 M) | 134 M |
| L1TEX 파면(LDG 128 B) | 워프·반복당 9(w 1 + x 8) → 37.75 M | 4.72 M |
| L2→SM | 하한 1.074 GB(블록의 8워프가 L1로 x 공유) … 상한 4.83 GB(공유 없음) | 0.134–0.60 GB |
| 명령 | 워프당 ≈2,570 → 168 M | 21 M |
| DRAM | 36.6 MB | 4.6 MB |

ptx-scan(base) 원문:
```
qwen3moe_router_logits            256     no         0         0     51        0     42      0      0              5       42         0
```
SASS(base, `just sass-scan gate_qwen3moe_e2e -- --exact qwen3moe_router_logits`) 원문:
```
  loop qwen3moe_router_logits 0x23c0-0x2d20 LDG=36 issued_before_first_wait=6 after=30 first_wait=0x24e0
```
- m>1 루프는 4배 전개돼 있고 4반복에 151명령입니다.
- ptxas는 열 가드를 분기가 아니라 술어(`@P1..@P6`)로 바꿨습니다.
- 반복 하나에 LDG 9 + FFMA 8 + 주소 계산 18 + 루프 약 3이 듭니다. 주소가 많은 이유는 열마다 런타임 `k` 오프셋이라 `IADD3`+`IMAD.X` 두 개씩 붙기 때문입니다.

**스캔 전에 적은 예측은 틀렸습니다.** regs 72–96, blk/SM 2–3, 첫 대기 전 로드 1–2개로 예측했는데, 실제는 42 regs, 5블록/SM(40워프), 로드 6개였습니다.

**1.19 ms에서 무엇이 묶는가** (84 SM, 1.6–1.8 GHz):
- SM 안 처리율 유닛은 모두 여유가 큽니다. FFMA 4.7–5.2 %, issue 23–26 %, L1TEX 21–24 %, DRAM 31 GB/s(피크의 4 %)입니다.
- 비행 중 로드는 40워프 × 6–9 = 240–360줄이고 완료율은 0.21–0.24줄/clk/SM입니다. Little 법칙으로 평균 지연은 1,000–1,700사이클입니다.
- 결론은 **지연·MLP 한계**입니다.
  - x(32 MB)는 L2(6 MB)보다 커서 반복마다 새로 만나는 줄이고, 한 청크의 16블록과 8워프가 같은 비행 중 미스를 기다립니다.
  - GPU 전체에서 비행 중인 x는 약 26청크 × 8줄 ≈ 27 KB뿐입니다. 27 KB / 31 GB/s = 0.86 µs이니 반복마다 DRAM 왕복 하나를 치르는 셈입니다.
  - 이 모형으로 19.5웨이브 × 64반복 × 0.93 µs ≈ 1.16 ms[유도]가 나오고, 실측은 1.19 ms입니다.
- 다른 읽기도 가능합니다. L1이 형제 워프의 미스를 합치지 못해 L2(B_L2 ≥ 1.59 TB/s @1.41 GHz)가 포화했다고 보면, x의 L1 적중률이 50 % 이하여야 합니다. 둘을 가르는 것은 ncu인데 이번에는 재지 않았습니다. **어느 읽기든 고르는 모양은 같습니다.**
  - B_L2 하한은 q3gemma ncu의 `L2 Cache Throughput 47.68 %`와 설계 보고서의 GEMM L2 바이트로 역산했습니다[유도].

### A.2 모양 비교와 선택

| | D(설계안, smem 없음) | S8(8워프 smem) | **S(선택)** |
|---|---|---|---|
| 블록 / 워프 타일 | 32행×16토큰 / 4×16 | 32×16 / 8×8 | 512스레드, 32×32 / 8×8 |
| 격자 T=4096 / 512 | 1,024 / 128 | 1,024 / 128 | 512 / 64 |
| 로드/반복·워프 | LDG 20 | LDS 16 | LDS 16 |
| L2→SM | 0.403 GB(L1 공유가 된다는 전제), 안 되면 1.34 | 0.403 | **0.268, 정확히 줄마다 한 번** |
| 명령 | 71.8 M(주소 40/127) | — | 60.1 M(실측 SASS) |
| regs / 상주 | 64+20+기저 40 → 128 초과 위험 | 2블록/SM | **127**, 1블록 = 16워프 |

S를 고른 이유는 넷입니다.
1. SM 밖으로 나가는 요청을 줄이고, 줄마다 한 번만 가져오게 정확히 만듭니다(L1 동작에 기대지 않음).
2. 안쪽 루프에 전역 주소 계산이 없습니다.
3. FMA 64개당 로드가 16개로 D(20개)보다 적습니다.
4. 지연은 cp.async 2단(32 KB) 비행이 덮습니다. GPU 전체에서 x 약 1.3 MB가 비행 중이니 오늘의 약 50배입니다.

한계도 둘 적습니다.
- regs가 127이라 상한 128에 붙어 있습니다. 늘면 gate-ptx-spill이 잡습니다.
- T=512에서는 64블록이라 20 SM이 놉니다. 1블록/SM 상주라 128블록으로 쪼개도 SM당 최대 일은 같고, 얻을 수 있는 상한은 512-프롬프트당 약 0.3 ms[유도]라 열지 않았습니다.

### A.3 예측

스테이지 루프 SASS는 199명령/패스(FFMA 128, LDS 35, LDGSTS 2, BAR 1)이고, 워프당 7,332명령, 합 60.1 M입니다.

- **T=4096: 0.12–0.20 ms/런치**(중심 0.15). issue 항 0.112–0.176 ms(클럭 1.45–1.6 GHz, 발행 효율 70–100 %)와 L2 항 0.077–0.168 ms(1.6–3.5 TB/s) 중 큰 쪽에 꼬리를 더했습니다.
- **T=512: 0.022–0.035 ms.** 오늘 값 0.13–0.19 ms는 P=512 nsys가 없어서 유도한 값입니다.
- **pp4096**: 48층 × Δ = −47.5…−51.4 ms, 546.9 → 495.5–499.4 ms → **8,200–8,270 tok/s**. nsys 창 549.6 ms 기준이면 498.2–502.1 ms입니다.
- **pp512**: Δ −4.6…−8.1 ms, 80.0 → 71.9–75.4 ms → **6,790–7,120 tok/s**.
- 판정 기준: 0.10 ms보다 빠르면 L2가 3 TB/s를 넘고 issue 모형이 과대였다는 뜻입니다. 0.20 ms보다 느리면 ncu로 발행 효율과 대기 원인을 봐야 합니다.

## B. 자원 시간선 (ubatch 한 층, P=4096)

- 호스트 CPU는 런치 약 20개를 3–6 µs씩 비동기로 넣고, 카드 그늘 아래 있습니다.
- PCIe, 호스트 DRAM, NVMe는 층 안에서 쓰지 않습니다.
- 카드 SM이 층당 약 11.4 ms를 직렬로 씁니다. 벽시계는 카드 한 스트림의 합이고, 라우터 logits는 임계 경로 위에 있어 Δ가 그대로 보입니다.
- 이번 레버는 SIMT입니다. 커널 안에서도 복사(cp.async)와 계산(LDS+FFMA)을 3단으로 겹칩니다.
- 비동기 레버는 제안만 합니다(아래 개선 지점 ③).

## C. 비트 논증

각 (행, 토큰) 쌍에서 레인 l의 합은 +0.0에서 시작해 `f32::mul_add(w[row·k+32·it+l], x[t·k+32·it+l], acc)`를 `it` 오름차순으로 더합니다.
- 스테이지 순서, 그리고 청크 j=0 다음 1이 곧 `it` 오름차순입니다.
- cp.async는 바이트를 그대로 복사하므로 값이 같습니다.
- 끝으로 `gemv_lane_sums`(퓨즈드와 같은 xor 버터플라이)를 거칩니다.
- `it`를 워프 사이에 나누지 않고, 리듀스 순서도 바꾸지 않았습니다.
- 범위 밖 토큰은 `n_tok−1`로 스테이징하고 저장하지 않습니다.

## D. 참조 엔진

- **llama.cpp** `53ed051ce`: `src/llama-graph.cpp:2025`, `ggml/src/ggml-cuda/ggml-cuda.cu:1824-1878`
  - F32 게이트는 ne11 ≤ 3이면 mmvf, ≤ 16이면 mmf를 탑니다.
  - 그보다 크면 `cublasSgemm`(`:1546`)으로 가는데, 핸들이 `CUBLAS_TF32_TENSOR_OP_MATH`(`common.cuh:1544`)로 만들어집니다. 대배치 logits가 TF32 텐서코어 곱이라는 뜻이라 정밀도와 합 순서가 모두 다릅니다.
- **exllamav3**: 게이트가 fp16입니다(`block_sparse_mlp.py:176-183`).
  - bsz 1은 f16→f32 fmaf 레인 체인에 셔플 리듀스를 씁니다(`routing.cu:208-239`). 우리 산술과 같은 모양입니다.
  - bsz > 1은 `hgemm`(`:254-266`)입니다.
- **mistral.rs** `d5ae0f1`: `layers::linear_no_bias`(`models/qwen3_moe.rs:365`) → `gate.forward`(`:403`) → cuBLAS GEMM입니다.
- 셋 다 벤더 GEMM에서 속도를 얻고, 셋 다 우리 비트 계약을 깹니다. S는 exl3의 bsz=1 산술을 레지스터 타일로 일반화한 모양입니다.

## 1. 바뀐 파일

- `/Users/hckim/repo/bloomery-q3router/crates/gpu/src/arch/qwen3moe/router.rs`
  - 새 logits 커널과 기하 상수, compile-time assert, `store_rows`를 넣었습니다.
  - `enqueue_ubatch`에 k의 64 배수 조건과 16 B 정렬 조건을 이름 붙은 거부로 넣었습니다. 이 두 거부는 새로 생긴 오류 경로입니다.
  - 격자는 ⌈T/32⌉ × 4입니다. `ROW_GROUPS`는 지웠습니다.
- `/Users/hckim/repo/bloomery-q3router/crates/gpu/src/q8f32.rs`
  - `TILE`, `fma_rows`, `f32_tile_chunk`와 모듈 문서 한 단락을 더했습니다. 기존 몸체는 그대로입니다.
- `/Users/hckim/repo/bloomery-q3router/crates/gpu-gates/src/bin/gate_gemm.rs`
  - `ROUTER_TOKENS`와 그 문서 줄을 고쳤습니다.
  - **모듈 문서 51–58행(라우터 T 목록 단락)도 고쳤습니다.** q3gemma의 GEMM 절과는 겹치지 않지만, diff 적용 순서를 정할 때 참고해 주십시오.
- `tools/ref/ptx-shapes.tsv`는 고치지 않았습니다. 엔트리 이름이 같고 값이 0/0이라 기존 행이 그대로 핀입니다.

## 2. 게이트와 증거 (원문)

`just gate-gpu-gemm --case router` (복원 트리, `Compiling bloomery-gpu` 있음):
```
gemm case=router T=1 ubatch_eq_fused_bits=true logits_in_band=true worst_err_over_band=0.007 PASS
… T=7 0.007 / T=8 0.007 / T=9 0.008 / T=15 0.007 / T=16 0.007 / T=17 0.012 / T=31 0.012 / T=33 0.013 / T=63 0.008 / T=512 0.017 — 모두 ubatch_eq_fused_bits=true logits_in_band=true PASS
gemm case=router T=4096 ubatch_eq_fused_bits=true logits_in_band=true worst_err_over_band=0.024 PASS
gemm case=router refusals tokens_past_buffers=refused no_tokens=refused short_input=refused weight_rows=refused ubatch_over_limit=refused ubatch_empty=refused PASS
```

**FAIL-first (a)**: `store_rows`에서 s0과 s1을 바꿔 저장했습니다. 예측은 12개 T 전부 빨강이었습니다.
```
   Compiling bloomery-gpu v0.1.0 (/root/repo/bloomery-q3router/crates/gpu)
gemm case=router T=1 ubatch_eq_fused_bits=false logits_in_band=false worst_err_over_band=20516.467 FAIL
… T=7,8,9,15,16,17,31,33,63,512 전부 FAIL …
gemm case=router T=4096 ubatch_eq_fused_bits=false logits_in_band=false worst_err_over_band=64815.334 FAIL
```

**FAIL-first (b)**: 부분 타일에서만 `live = n − tw − 1`로 바꿨습니다. 예측은 {1,7,9,15,17,31,33,63} 빨강, {8,16,512,4096} 초록이었고, 정확히 그렇게 나왔습니다.
```
gemm case=router T=1 ubatch_eq_fused_bits=false … FAIL
gemm case=router T=8 ubatch_eq_fused_bits=true logits_in_band=true worst_err_over_band=0.007 PASS
gemm case=router T=15 ubatch_eq_fused_bits=false logits_in_band=false worst_err_over_band=35685.833 FAIL
gemm case=router T=16 ubatch_eq_fused_bits=true … PASS
gemm case=router T=33 ubatch_eq_fused_bits=false logits_in_band=false worst_err_over_band=24434.939 FAIL
gemm case=router T=512 … PASS / T=4096 … PASS
```

복원은 두 번 모두 `restored md5 500bd28efaed48e8fa64fba06575e731 orig 500bd28efaed48e8fa64fba06575e731`로 확인했고, 그 뒤 빌드에 `Compiling bloomery-gpu` 줄이 있었습니다.

`just gate-gpu-qwen3moe-kernels`:
```
gate_qwen3moe_qknorm: 192 (set, layer) sites, 0 failed — PASS
gate_qwen3moe_rope: 192 (set, layer) sites and the graph, 0 failed — PASS
fault op=qwen3moe_router_logits+route n=9 token 8 overflows: … token refused=true other tokens bit-identical=true PASS
gate_qwen3moe_router: PASS / gate_qwen3moe_down: PASS / gate_qwen3moe_flash: PASS / gate_qwen3moe_experts: PASS
```

`just gate-gpu-qwen3moe-e2e` (두 팔, 각각 PASS 63줄, FAIL 0줄):
```
load resident_bytes=19013528264 ctx=1344 layers=48 flash_mma=true ubatch=4096 …
structure graph_nodes=604 (want 604) kernel=604 memcpy=0 (want 0) memset=0 host=0 (want 0) other=0 PASS
gate_qwen3moe_e2e: PASS
load … flash_mma=false … / structure graph_nodes=604 (want 604) … PASS / gate_qwen3moe_e2e: PASS
```

`just gate-gpu-e2e`: `gate_e2e: PASS — the whole chain picks the greedy reference's first token …`가 두 실행 모두에서 나왔습니다.

`just gate-ptx-spill`:
```
ptx-spill bin=generate_ds41 entries=146 pinned=146 nonzero=ds41_attn_seg_stage(spill=8,jit_local=8),flash_latent_seg_v2(spill=36,jit_local=16),flash_merge2_q8(spill=40,jit_local=8) violations=0 PASS
ptx-spill bin=gate_e2e entries=94 pinned=94 nonzero=flash_latent_seg_v2(spill=36,jit_local=16),flash_merge2_q8(spill=40,jit_local=8) violations=0 PASS
```

`just ptx-scan gate_qwen3moe_e2e`를 base와 트리에서 떴습니다. md5 블록 94줄을 diff하니 다른 줄은 `qwen3moe_router_logits` 하나뿐이고, 나머지 93개는 같습니다. 트리 행:
```
qwen3moe_router_logits            512     no         0         0    128        0    127  49152      0              1      127         0
```

정적 검사:
- `just check` rc=0(`Finished`), Cargo.lock 변화 없음.
- `just lint` rc=0, `grep -c '^warning:'` = **144**. 내 파일에서 나온 경고는 0입니다.
- `just fmt-check` rc=0.
- `just check-recipes`: `check-recipes: ok`
- `check-comments: ok`, `check-arch: ok`
- `box-gc`: 처음과 끝 모두 `found 0 process(es)`

## 3. 예측 대 결과

- 비트 동일: 예측대로 12/12.
- FAIL-first 두 패턴: 예측과 정확히 같습니다.
- 새 커널: regs 90–124로 예측했는데 127이 나왔습니다(스필 0). 스테이지당 명령은 약 175로 예측했는데 199가 나와서 issue 항을 다시 잡았습니다.
- 시간 예측(A.3)은 리드의 시팅(`depth-gpu-qwen3moe`, `nsys-gpu-qwen3moe-prefill`)으로 판정합니다.

## 4. 못 한 것

- 시간 측정은 하지 않았습니다(스펙대로).
- ncu가 없어서 A.1의 두 읽기(지연 대 L2 포화)를 가르지 못했습니다.
- 새로 넣은 거부 둘(k의 64 배수 조건, 16 B 정렬 조건)은 gate_gemm 경계 밖이라 거부 조항을 추가하지 못했습니다.

## 5. 스펙 밖 개선 지점 (보고만)

- ① `/Users/hckim/repo/bloomery-q3router/crates/gpu/src/q8f32.rs:74` `f32_lane_partials`의 m>1 몸체: 가드된 열별 FMA와 반복당 주소 18개가 MLP를 6으로 묶습니다. 디코드 퓨즈드 라우터(m 2–8)와 `f32_gemv`가 같은 몸체를 씁니다. `q8_0_lane_partials_mcol` 식의 `const M` 변형이 답입니다. 크기 S.
- ② gate_gemm router 케이스에 두 공백이 있습니다. n_tok 뒤에 센티널이 없어 과다 저장(`c <= live`)이 초록으로 지나가고, 새 거부 둘을 시험하는 조항도 없습니다. 크기 S.
- ③ 비동기 제안: `/Users/hckim/repo/bloomery-q3router/crates/gpu/src/arch/qwen3moe/ubatch.rs:564`의 `enqueue_quantize_gemm`과 라우터 logits는 둘 다 `a.normed`만 읽으니 두 번째 스트림에서 겹칠 수 있습니다. 층당 최대 약 0.1 ms, P=4096 프롬프트당 약 5 ms[유도]입니다. 크기 M.
- ④ `/Users/hckim/repo/bloomery-q3router/crates/gpu/src/arch/qwen3moe/ubatch.rs:37-38`의 "router … groups of eight"는 이제 틀린 문장입니다(32토큰 블록, 클램프 스테이징). 크기 XS.
- ⑤ 새 스테이지 루프 199명령 중 33개가 주소·회전·분기입니다. 시팅에서 issue 한계로 나오면 루프를 3배 전개해 버퍼 번호를 정적으로 만드는 것이 다음 수입니다. regs 여유가 없는 것이 위험 요소입니다. 크기 S.
- ⑥ `/Users/hckim/repo/bloomery-q3router/crates/gpu/src/q8f32.rs:5`의 "top-6"은 V2-Lite 기준 문구이고, Qwen3는 top-8입니다. 크기 XS.

## 6. 모델

opus(Opus 5.5)로 스폰되어 돌았습니다.
