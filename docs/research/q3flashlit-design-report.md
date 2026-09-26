# q3flashlit 보고 (FA2와 우리 GQA 프리필 flash의 SASS 판독, 설계, 2026-09-26)

> 리드 메모(2026-09-26). 아래는 조사 라운드 `q3flashlit`(opus)의 보고 원문이다. 박스 실행은 없다. mistral.rs 바이너리(sha256 `8da64b84d5e5`)의
> `flash_fwd_kernel`과 우리 `generate_qwen3moe`의 PTX(main, q3fa)를 SASS로 풀어 루프 한 바퀴를 셌다. 결론: 두 커널은 하는 일이 같다
> (HMMA 34,078,720/런치, 블록-타일 66,560, 그리드 2,048, L2→SM 바이트, SM당 8워프). 그런데 FA2는 텐서 파이프 86–96 %, 우리는
> 55–64 %이고, 우리 쪽에만 발행 모형이 설명하지 못하는 짝-스텝당 약 2,000–2,300 사이클(창 ≈ 28 ms)이 남는다[유도]. q3fa가 비-HMMA를
> 56 % 깎고도 pp4096이 +1.06 %만 움직인 것은 이 잔차가 명령 수가 아니라는 뜻으로 읽힌다. 가장 뚜렷한 구조 차이는 스텝 모양이다:
> FA2는 QK 뒤 배리어 다음에 softmax를 두어 exp·팩이 PV HMMA 그늘에 들어가고, 우리는 softmax 뒤에 배리어가 있다. 첫 레버 후보는
> 「FA2 스텝 모양」(비트 유지, M, 창 0…−35 ms)이고, 여는 판단은 ncu 시팅 16(main·q3fa·FA2 세 행, 카드 `q3ncu-flash`·`fa2-ncu`)이
> 한다. 어느 읽기에서도 양수인 것은 프롤로그 다이어트(−1.8…−3.9 ms)뿐이다. 이 라운드가 찾은 q3tail 보고의 오류(「FA2가 K/V L2 트래픽
> 8배」)는 그 문서에 줄을 긋고 정정했다. 모형 상수(MUFU·LDSM·BAR 지연 등)는 가정이라고 보고가 스스로 밝히고 있다.


**결론.** FA2와 우리 커널은 하는 일이 같습니다. HMMA 수, 블록-타일 66,560, 블록 길이 분포, L2→SM 바이트, SM당 8워프가 모두 같습니다[유도 + SASS]. 그런데 FA2는 텐서 파이프를 86–96 % 쓰고 우리는 55–64 %입니다.

SASS에서 직접 만든 스케줄러 모형 여러 가지로 확인했습니다. 모든 모형이 FA2는 약 5 % 안으로 맞힙니다. 우리는 짝-스텝당 약 2,000–2,300 사이클이 설명되지 않고 남습니다. 이 잔차는 q3fa가 비-HMMA를 56 % 줄여도 거의 움직이지 않았습니다. 이것이 이번 라운드의 발견입니다.

L2 용량 가설은 LRU 유도로 닫았습니다(≈ 0). 남은 후보는 셋이고, 셋을 가르는 것은 ncu입니다.
- 배리어 뒤에 드러나는 softmax 지연 사슬을 같은 SM의 다른 블록 워프가 가리지 못하는 경우
- 명령어 fetch(L0 i-cache)
- (가능성 낮음) 점유율이나 cp.async 도착 지연

첫 레버는 **FA2 스텝 모양**입니다(비트 유지, M). 상한은 창 −31…−35 ms(P = 4096), 하한은 0입니다. 레버를 열지는 ncu 시팅이 정합니다.

### 1. SASS 표 두 개(와 main)

조건: 워프-타일 하나 = 내부 루프 한 바퀴(마스크 없는 타일)입니다. 도구는 `tools/sass_inflight.py --step`의 사본이고, 분류는 도구의 분류를 따랐습니다.

| | FA2 (mistral.rs) | 우리 main | 우리 q3fa(미커밋) |
|---|---:|---:|---:|
| 스텝 명령 | **494** | 1,265 | **624** (마스크 타일 774) |
| HMMA.16816.F32(.BF16) | 128 | 128 | 128 |
| 비-HMMA | **366** | 1,137 | **496** |
| FP32 (FMUL/FFMA/FADD) | 134 (68/34/32) | 258 (194/0/64) | 194 (130/0/64) |
| 정수·주소 (IMAD, IADD3, LEA, SHF, ISETP, MOV, LOP3) | 35 | 약 370 | 약 50 |
| 부동 비교·선택 (FMNMX/FSETP/FSEL/PLOP3) | 40 | 238 | 42 |
| MUFU | 34 | 32 | 32 |
| LDSM / LDS(@!PT 더미) / LDGSTS / STS | 72 / 6 / 16 / 0 | 64 / 48 / 16 / 15 | 64 / 6 / 16 / 0 |
| F2FP / HADD2 / SHFL | 16 / 0 / 4 | 16 / 32 / 4 | 16 / 32 / 4 |
| BAR / DEPBAR+LDGDEPBAR | 2 / 4 | 2 / 4 | 2 / 4 |
| 제어 (BRA, BSSY/BSYNC, WARPSYNC, CALL, VOTE) | 3 | 19 | 34 |
| 정적 stall 합 | 1,448 | 2,874 | 1,887 |
| 레지스터 / smem / SM당 블록 | 182 / 동적 49,152 / **2** | 255 / 정적 34,816 / 2 | 194 / 34,816 / 2 |
| 루프 코드 크기(실행 경로의 128 B 줄) | **7.9 KB 연속, 점프 2** | 20.1 KB, 점프 3 | 10.6 KB(18 KB 범위에 흩어짐), 점프 5 |

q3fa 보고의 1,137 → 496은 그대로 재현됐습니다. 마스크 타일은 제 분기 선택으로 774가 나왔고 보고의 750과 다릅니다.

**루프 한 바퀴를 HMMA 구간 기준으로 나누면**(`seg.py`) 이렇습니다.

| | QK 앞 | QK 구간 안 | QK→PV 사이에 드러나는 구간 | PV 구간 안(HMMA 그늘) |
|---|---|---|---|---|
| FA2 | 40 | 32 (BAR 1개 포함) | **201, MUFU 13, BAR 없음** | **92, MUFU 21·FADD 26·F2FP 12** |
| q3fa | 54 | 38 | **378, MUFU 32, BAR 1개** | 22 (LDSM만) |
| main | 140 | 132 | 837, MUFU 32, BAR 1개 | 22 |

**점유율.** FA2는 레지스터 184 × 32 = 5,888/워프라 11워프입니다. smem은 (49,152 + 1,024) × 2 = 100,352 ≤ 102,400이라 2블록입니다. 우리도 2블록이라 둘 다 SM당 8워프, 스케줄러당 2워프입니다.

**명령**(박스는 읽기 전용 `ssh ws`, `nice -n 19`)
- FA2
  - `cuobjdump -xelf all mistralrs`
  - `strings | c++filt`로 인스턴스를 찾음: `mistralrs.16.sm_86.cubin`, `_Z16flash_fwd_kernelI23Flash_fwd_kernel_traitsILi128ELi64ELi64ELi4ELb0ELb0EN7cutlass10bfloat16_tE19Flash_kernel_traitsILi128ELi64ELi64ELi4ES2_EELb0ELb1ELb0ELb0ELb1ELb1ELb0ELb0EEv16Flash_fwd_params`
  - `cuobjdump -sass -fun … -res-usage`(REG:182)
  - `sass_inflight.py --step 0x7440 --decisions 0x7cc0=t fa2.sass`
- 우리
  - 대상 트리 둘: `/root/repo/bloomery`(flash 소스 md5 `ba6d1e…` = main)와 `/root/repo/bloomery-q3fa`(`ebb447…` = Mac q3fa 워크트리)
  - 각 트리의 `target/release/generate_qwen3moe`에 `objcopy --only-section=.oxart`
  - 각 트리의 `oxart_ptx`, 이어서 `ptxas -arch=sm_86 -v`(13.0: 255 / 194 regs, spill 0), `cuobjdump -sass`
  - main 스텝: `--step 0xfbc0 --decisions 0x103c0=n,0x11a90=t,0x11c50=t,0x13e50=n,0x144f0=t,0x14600=n,0x14c60=n`
  - q3fa 스텝: `--step 0xf250 --decisions 0xf2c0=n,0xf9a0=n,0x101e0=n,0x10480=t,0x108f0=t,0x12bc0=n,0x13030=n,0x13080=n,0x136d0=n`
- 인스턴스 주의: FA2 인스턴스(even_MN = true)는 nsys 이름에서 읽은 것이 아닙니다.
  - 근거는 둘입니다. mistral.rs가 cu_seqlens를 null로 넘기고(`mistralrs-flash-attn/src/flash.rs:180`), dispatch가 b = 1·seq = k에서 varlen을 쓰지 않습니다(`mistralrs-core/src/attention/backends/flash.rs:69-71`)[유도].
  - 다른 even_MN 인스턴스와는 경계 검사만 다릅니다.

### 2. 구조 차이: 기제, Δ, 모형과 적합

**측정을 짝-스텝으로 바꾼 값.** 짝-스텝 = 스케줄러 하나에서 두 블록 워프가 각자 타일 하나씩 도는 시간이고, 텐서 수요는 4,096 사이클입니다(GA10x 128 FMA/clk/파티션, `ga102.txt:997`, A6000 FP16·BF16 FP32 누산 154.8 TFLOPS).

- 메이크스팬은 이벤트 모형으로 구했습니다(홀로 남은 블록이 더 빠른 것까지 반영). FA2 430–450, 우리 LPT 397–409 타일-스텝입니다.
- 클록 대역은 1.5–1.6 GHz입니다. 중심값은 이미 잰 1,515 MHz 증인입니다.

| | 런치 | 짝-스텝 사이클 | 텐서 비율 | 텐서 밖 초과분 |
|---|---:|---:|---:|---:|
| FA2 | 1.276 ms (측정) | 4,253–4,752 | **86–96 %** | 157–656 |
| main | 1.854 ms (측정) | 6,800–7,472 | 55–60 % | 2,704–3,376 |
| q3fa | ≈ 1.741 ms [pp4096 +1.06 %에서 유도] | 6,385–7,017 | 58–64 % | 2,289–2,921 |

곁들여 나온 유도 하나: FA2가 텐서 한계를 넘을 수는 없으므로 FA2 클록은 1.38–1.44 GHz 이상입니다.

**선형 분해**[유도]. 짝-스텝 = 4,096 + 0.34 × (짝당 비-HMMA 발행) + R로 두면 우리는 R ≈ 1,900–2,600, FA2는 R ≈ −100…400입니다. 계수 0.34는 main→q3fa에서 발행 1,282개를 줄여 약 435 사이클을 번 값입니다. R의 차이 ≈ 2,000–2,300 사이클/짝 = 런치당 0.55–0.63 ms = **창 ≈ 28 ms**이고, 측정된 틈과 같습니다.

**모형 적합.** 모형은 모두 SASS 제어 워드(stall, scoreboard)로 돌렸습니다. 상수는 전부 **가정**입니다: HMMA 파이프 16, MUFU 점유 8·지연 20, LDSM 30, SHFL 30, cp 400(2,000까지 스윕), BAR 20.

| 모형 | FA2 | main | q3fa | 판정 |
|---|---:|---:|---:|---|
| 스케줄러 1개, 워프 2개 겹침, GTO / LRR | 4,379 / 4,081 | 5,580 / 5,897 | 4,352 / 4,674 | FA2 적합, 우리는 −16…−37 % |
| 스케줄러 4개 × 2블록, 실제 BAR, SM 공용 smem 포트, GTO | 4,942 | 6,090 | 5,418 | q3fa −15…−23 % |
| 같은 모형, LRR, 동위상 시작 | 5,626 | 7,915 | 6,517 | FA2가 18 % 넘침 |
| 순수 합(한 워프 스텝 × 2) | 5,060 | 8,274 | 6,262 | q3fa만 적합, FA2와 main은 넘침 |
| **측정** | 4,253–4,752 | 6,800–7,472 | 6,385–7,017 | |

q3fa/FA2 비는 측정이 1.50이고 모형은 모두 ≤ 1.24입니다. **세 점을 함께 맞히는 닫힌 모형은 없습니다.** 우리 쪽 잔차는 SASS 발행 모형 밖에 있습니다.

**D0 같은 일**[유도]
- HMMA는 런치당 34,078,720으로 같습니다.
- 블록-타일은 둘 다 66,560입니다. FA2는 64 m-block × 32 헤드이고 블록 길이가 m + 1입니다. 우리는 qt 512개 × KV 헤드 4개이고 길이가 ⌊qt/8⌋ + 1입니다. 두 경우 모두 **길이 1–64가 각 32개씩**입니다.
- 그리드는 둘 다 2,048입니다(FA2 `dim3 grid(num_m_block, params.b, params.h)`, `flash_fwd_launch_template.h:62`).
- **q3tail 정정**: L2→SM 트래픽은 둘 다 66,560 × 32 KB = 2.18 GB/런치로 같습니다. 「FA2가 K/V L2 트래픽 8배」(`q3tail-design-report.md:326`)는 틀렸습니다. FA2 블록은 위치를 8배 덮는 대신 헤드를 1/8만 덮기 때문입니다.

**D1 행 묶음**(우리 8위치 × 8헤드, FA2 64위치 × 1헤드)
- 타일당 K/V smem은 둘 다 32 KB입니다.
- 마스크 타일은 둘 다 블록당 정확히 하나(2,048)이고 평균 절반이 가려집니다.
  - FA2는 대각 타일을 첫 반복으로 떼어 냅니다(`n_masking_steps`, `flash_fwd_kernel.h:334-338`). 키는 내림차순으로 돕니다(`:300` `n_block = n_block_max - 1`).
  - 우리는 오름차순이고 마스크 경로가 루프 본문 안에 있습니다.
- Δ는 0입니다[유도]. 차이가 나는 곳은 프롤로그(Q가 f32 32 KB/블록, FA2는 bf16 16 KB)뿐이고 D6에서 다룹니다.

**D2 파이프라인과 배리어 위치.** 둘 다 타일당 배리어 2개, `wait_group 0`입니다.
- FA2는 [BAR1] V 복사 → QK → **[BAR2] 다음 K 복사 → softmax → PV** 순입니다(`flash_fwd_kernel.h:417-464`, BAR2는 `:430-431`, K는 `:433`, softmax는 `:443`, PV는 `:464`). 그래서 exp·팩 작업의 일부가 PV HMMA 그늘에 들어가고(위 구간 표), 다음 K는 softmax와 PV 동안 옵니다.
- 우리는 QK → softmax → 재스케일 → **[BAR2]** → 다음 K → PV 순입니다(`flash_gqa_prefill.rs:457-461`). 배리어가 exp와 PV HMMA의 교차 배치를 막습니다.
- Δ
  - 겹침 모형에서는 0…−7 %입니다. 한 워프 스텝 차 약 600 사이클을 다른 워프가 가립니다.
  - 겹침이 없는 읽기(동위상 락스텝)에서는 −2 × (3,131 − 2,530) ≈ 짝당 −1,200, 약 −18 %, 창 ≈ −16 ms입니다.
- q3fa가 평평했던 것은 정성적으로 이렇게 설명됩니다. 드러난 구간이 발행에 묶인 것이 아니라 의존 사슬에 묶여 있으면 명령을 깎아도 그 구간은 줄지 않습니다. 사슬: FMNMX 직렬 16 → SHFL 2 → exp → MUFU 파이프 32 × 8 = 256 → FADD 사슬 16 → BAR → 스테이징 → LDSM → 첫 HMMA.

**D3 softmax**
- FA2
  - `exp2f(tensor * scale - max_scaled)` = FFMA + MUFU.EX2입니다(fast-math, `softmax.h:87`, 빌드 플래그 `build.rs:179 --use_fast_math`).
  - acc_o 재스케일은 매 타일 무조건 합니다(`softmax.h:156-159`, `Check_inf` 가드 `:75`).
  - 행 합은 끝에 한 번 쿼드로 합칩니다(`:171`).
- 우리(q3fa)
  - FMUL scale, FADD, FMUL log2e, MUFU 순입니다.
  - 재스케일은 vote로 건너뜁니다.
  - 합은 f16으로 반올림한 가중치로 합니다(HADD2 32는 계약이라 유지).
- 차이는 워프-타일당 비-HMMA 130개이고, 계수 0.34로 짝당 약 90 사이클, 창 ≈ −1 ms입니다.

**D4 레지스터·점유율.** 같습니다(2블록, 8워프). Δ는 0입니다. 다만 드라이버 carveout 선택은 ncu가 확인할 항목입니다.

**D5 스케줄 순서**
- FA2는 m이 가장 빠르고 헤드가 바깥입니다(`:62`, `flash_fwd_kernel.h:1125` `m_block = blockIdx.x`). 메이크스팬은 평균보다 +8.5…+13.6 %입니다.
- 우리 LPT는 +0.2 %입니다. 따라서 짝-스텝으로 보면 FA2가 **1.50–1.59배** 빠릅니다.
- FA2 순서에서는 같은 SM에 올라간 두 블록의 길이가 달라 위상이 흩어집니다. 우리 LPT는 길이가 같은 블록을 짝지어 동위상 락스텝이 됩니다.
- Δ는 겹침 모형에서 0이고, 락스텝 읽기에서는 잔차 전부까지입니다.

**D6 L2 용량**(자문이 제안해 유도, **닫음**). 6 MB 완전연관 LRU, 16 KB 단위로 계산했습니다(`l2lru.py`).
- DRAM에서 읽는 K/V는 우리 LPT 55.4 MB, 헤드 순차 LPT 17.3, 헤드 짝 30.1, 오름차순 219, FA2 8.4 MB입니다(고유 8.4 MB).
- 우리 DRAM 합은 약 55 + Q 67 + Y 67 ≈ 190 MB로, 약 0.27 ms/런치입니다(1.85 ms 대비). 순서 레버의 가치는 ≤ 0.055 ms/런치이고 그늘 아래라 ≈ 0입니다.
- 층 0 flash가 약 9 % 빠른 것(q3tail)도 클록에 비례하는 SM 안 바운드와 맞습니다.

**D7 루프 코드 크기.** FA2 7.9 KB, q3fa 10.6 KB, main 20.1 KB입니다. GA102 백서는 L0 i-cache가 있다고만 하고 크기는 적지 않습니다(`ga102.txt:378`). 이 항은 **L0가 10.6 KB보다 작을 때에만** q3fa와 FA2를 가릅니다. 모형에 넣지 않았습니다.

**D8 프롤로그**
- SASS 사실: 루프 머리 앞에 워프당 3,877명령, LDG 40개, 첫 대기 전에 떠 있는 로드는 5개뿐입니다. 소프트웨어 `f32_to_f16_bits`(`gguf/src/quant.rs:481`, 분기가 있음)이고, FA2 커널 전체는 2,630명령입니다.
- 비용: 블록 짝 시작마다 약 8–11 K 사이클, 런치의 2–4.4 %입니다. DRAM 왕복 4번은 **가정**입니다.

### 3. llama.cpp 행

| | llama.cpp MMA flash(`/home/user/llama.cpp-mainline`) |
|---|---|
| 묶음 | ncols1 8 × ncols2 8 = 64행으로 **우리와 같습니다**(`fattn.cu:190` `64/ncols2`, gqa 8) |
| 설정 | `(128,128,64, 128thr, occ 2, nbatch_fa 64, K2 64, V2 64, 64, 2단, Q_in_reg)`(`fattn-mma-f16.cuh:62`) |
| 배리어 순서 | softmax가 V 배리어 앞입니다. **우리와 같고 FA2와 다릅니다**(`:642-644`, `:994-1003`) |
| 누산 | VKQ가 half2입니다(`:1090`). 우리는 f32. expf는 `-use_fast_math` 아래입니다(`:803`, `CMakeLists.txt:186`) |
| 블록 순서 | kbc 연속이고 jt가 안쪽, 헤드가 바깥입니다(`:1908-1924`). 오름차순이라 LPT가 아니고 위상이 흩어집니다 |
| 수치 | 엔진 전체 pp4096 @ ub 4096 6,860(09-26 #mrs-flash-qwen3 임대). **flash 커널 시간은 잰 적이 없습니다.** SASS는 뽑지 않았습니다 |

셋이 갈리는 곳
- 배리어 위치: FA2 `flash_fwd_kernel.h:430-443` / 우리 `flash_gqa_prefill.rs:457-461` / llama `:994-1003`
- 순서: FA2 `launch_template.h:62` / 우리 `flash_gqa_prefill.rs:164-165` / llama `:1908-1924`
- 누산: FA2·우리 f32, llama half2

### 4. 레버 순위

창 기준 Δ이고, 기준선은 main pp4096 8,818(창 464.5 ms, flash 89.0 ms)과 pp512 7,706입니다. 우리 P = 512 flash는 잰 적이 없고, q3tail의 ≈ 1.8 ms/창은 유도값입니다.

| # | 레버 | Δ창 P = 4096 | Δ창 P = 512 | 비트 | 레지스터 / smem / 점유 | 크기 | 옳다면 ncu가 보일 것 |
|---|---|---|---|---|---|---|---|
| **1** | **FA2 스텝 모양**: V 배리어를 QK 바로 뒤로, 다음 K를 softmax 앞에, exp·팩을 PV HMMA와 섞음. 마스크 타일은 루프 밖으로 떼어 연속 ≈ 8 KB 루프로. f는 exp(m_old − m_new)로 무조건 계산 | **0…−35**. 상한은 FA2 짝-스텝 × 우리 메이크스팬 = 1.126–1.214 ms/런치, pp4096 → 9,440–9,540[유도] | 0…−0.7 | 유지. 조건 두 가지(아래) | ≈ 180–210 / 34,816 / 2블록 | M | 전: 소프트맥스 뒤 BAR PC(q3fa `0x13020`)와 PV 입구 LDSM에 barrier·short_scoreboard가 몰리거나 no_instruction ≥ 10 %. 후: 텐서 ≥ 80 %, math_pipe_throttle(텐서)가 1위 |
| 2 | FS 스태거(LPT는 유지하고 위상만 흩음) | 락스텝 읽기일 때만 0…−(잔차) | 약 0 | 유지 | +2 / 0 / 같음 | S | 텐서 약 60 %에서 math_pipe_throttle ≥ 25 % |
| 3 | 프롤로그 다이어트: 로드 32개를 한 번에 띄우고, 하드웨어 `cvt.rn.f16x2.f32` + NaN select로 `sign\|0x7e00` 유지 | −1.8…−3.9 | −0.2…−0.35 | 유지(NaN까지 select로) | 0 / 0 / 같음 | S | SourceCounters에서 루프 앞 PC 샘플 3–5 %. **어느 읽기에서도 양수인 유일한 레버** |
| 4 | softmax 더 깎기(scale·log2e FFMA 접기, vote 제거) | 약 −1 | 약 0 | 접기는 이동(밴드 + PIN) | | S | 1번에 흡수 |
| 5 | carveout 속성 | 0 예상 | | 유지 | | XS | `sm__warps_active` ≈ 4/SM일 때만 |
| 6 | 헤드 묶음 순서(L2 용량) | ≈ 0(≤ 2.6, 그늘 아래) | | 유지 | | S | 기각(D6) |
| 7 | 3단 cp.async / 키 타일 128 / f16 VKQ | ✗ | | | 106,496 > 102,400 B / 레지스터 초과 / 비트·정밀도 | | 기각 |

**1번이 비트를 유지하는 조건 둘**
- **`Check_inf`에 해당하는 select(분기 없이)가 필요합니다.** 떼어 낸 마스크 타일에서 한 행 절반이 전부 가려져 m = m′ = −∞이면 −∞ − (−∞) = NaN이 되어 `o`를 오염시킵니다(모듈 문서 `flash_gqa_prefill.rs:43-46`의 경우). 유한한 m에서는 ex2.approx(±0) = 1이고, 첫 타일에서는 −∞ → 0이라 기존 값과 같습니다.
- T가 8의 배수가 아닐 때 마지막 블록은 count 0인 행이 있어(`warp_lo = 0`) 일반 경로를 탑니다.

**처음 열 레버는 1번입니다.** 29–35 ms 틈을 덮는 상한을 가진 레버는 이것뿐이고, 유력한 두 읽기(드러나는 softmax 사슬·락스텝, fetch)를 한 번에 겨냥합니다. 단, 아래 시팅에서 세 가지를 확인한 뒤에 엽니다.
- (a) q3fa를 직접 재서 ≈ 1.74 ms인가
- (b) SM당 8워프인가
- (c) 스톨이 루프 PC에 몰려 있는가(DEPBAR의 long_scoreboard나 프롤로그가 아니라)

시팅 전에 하나를 열어야 한다면 3번뿐입니다.

### 5. ncu 시팅만 가를 수 있는 것(층 24, P = 4096)

1. **q3fa 커널 시간을 직접 잴 것.** 지금 값은 pp에서 유도한 것이고, 잔차 결론이 여기에 기댑니다. 기대값은 main 1.85–1.87 ms, q3fa 1.70–1.78 ms입니다. q3fa가 ≤ 1.60이면 잔차가 줄어듭니다. 드라이버가 PTX를 JIT하므로 CUPTI registersPerThread를 ptxas의 194와 대조합니다.
2. 클록 1.46–1.55 GHz. `sm__warps_active` ≈ 8/SM(16.7 %), 두 빌드 모두.
3. 텐서: main 55–60 %, q3fa 58–64 %.
4. 스톨 구성
   - **제 읽기**(드러나는 사슬 + 락스텝): barrier 15–25 %, short_scoreboard 15–25 %, wait 15–20 %, math_pipe_throttle 10–20 %, no_instruction < 8 %, long_scoreboard < 5 %. math_pipe_throttle은 텐서와 XU(MUFU) 몫을 `pipe_xu`로 나눠 봐야 합니다.
   - **fetch 읽기**: no_instruction ≥ 15 %가 루프 전체 PC에 고르게 퍼짐. L0 < 10.6 KB가 전제입니다.
   - **cp.async 읽기**: DEPBAR PC에서 long_scoreboard ≥ 15 %.
5. **추가 행 권고**
   - 같은 ncu 섹션을 mistral.rs `flash_fwd_kernel`에도. 기대값은 텐서 86–96 %, math_pipe_throttle 1위(≥ 35 %), no_instruction < 3 %, barrier < 10 %, lts 처리율 75–90 %입니다(1.71 TB/s 평균에서 유도). 이 행은 이 타일 모양이 텐서 한계에서 필요로 하는 L2→SM(약 1.85 TB/s)도 재 줍니다. 그래서 현실적인 목표는 순수 텐서 천장(1.05–1.12 ms)이 아니라 FA2와 같은 수준입니다.
   - llama.cpp `-ub 4096` nsys 한 번으로 flash 시간 한 행.

### 6. 스펙 밖 개선 지점(보고만, 손대지 않음)

- `crates/gpu/src/flash_gqa_prefill.rs:217-258` + `crates/gguf/src/quant.rs:481`: Q 스테이징이 소프트웨어 f16 변환이고 로드를 직렬 4묶음으로 합니다. 루프 전 3,877명령입니다. 레버 3이고 S입니다.
- `flash_gqa_prefill.rs:393-394`: `warp::shuffle_xor_f32`가 SHFL마다 `BRA.DIV` + `WARPSYNC`를 냅니다. nvcc의 `__shfl_xor_sync`는 맨 SHFL.BFLY라 FA2 루프에는 없습니다. nvlabs 장부 후보이고 XS입니다.
- `flash_gqa_prefill.rs:306`(`cp_async_cg_16`)는 `LDGSTS.E.BYPASS.128`이고, FA2는 `.LTC128B`입니다. L2 적중이 약 97 %라 지금은 ≈ 0입니다. XS, 기록만.
- `docs/research/q3tail-design-report.md:326`: 「K/V L2 트래픽 8배」는 틀렸습니다(같음). 또 FA2는 키를 내림차순으로 돕니다(`flash_fwd_kernel.h:300`). XS이고 리드가 한국어로 고칠 몫입니다.
- `tools/sass_inflight.py`에 `--help`가 없습니다(`unknown flag --help`). 스펙이 이 플래그를 가리킵니다. XS.
- `docs/cards/q3ncu-flash.card`: 판정 규칙에 점유율 줄, q3fa 커널 시간 확인, FA2 비교 행을 넣을 것. XS.
- 이번에 만든 두-워프·네-스케줄러 모의기(`sim.py`, `sim4.py`)는 ncu로 상수를 보정하면 `tools/`에 올릴 만합니다. S.

### 7. 출처, 편차, 모델

- **출처 줄**
  - mistral.rs: `flash_fwd_launch_template.h:62, :274`, `flash_fwd_kernel.h:300, :334-338, :414-464, :1125`, `softmax.h:75, :87, :156-159, :171`, `kernel_traits.h:109`, `mistralrs-flash-attn/build.rs:179`
  - llama.cpp: `fattn-mma-f16.cuh:62, :642-644, :803, :994-1003, :1090, :1908-1924`, `fattn.cu:190`, `CMakeLists.txt:186`
  - 백서: 로컬 사본 `ga102.txt:355`(L2 6,144 KB), `:378`(L0, 크기 표기 없음), `:997`(텐서코어당 128 FMA), A6000 표 `:559-582`
  - 측정: rig-log `log/2026-09-26.md` #mrs-nsys·#q3fa-ab, `log/2026-09-25.md:565`
- **산출물(스크래치)** `/private/tmp/claude-501/-Users-hckim-repo-bloomery/b4e4a3ef-e524-4e47-91af-3afadc891c1e/scratchpad/`
  - SASS 목록: `fa2.sass`, `main_flash.sass`, `q3fa_flash.sass`
  - 스크립트: `seg.py`, `sim.py`, `sim4.py`, `runsim.py`, `sched.py`, `sched2.py`, `l2lru.py`
- **편차**
  - 바뀐 파일과 git 변경은 없습니다.
  - 박스는 `box.sh`가 아니라 읽기 전용 `ssh ws`로 접근했습니다. box.sh의 rsync가 main 원격 디렉터리를 덮어쓰기 때문입니다. box-gc는 돌리지 않았습니다(제 원격 디렉터리가 없습니다).
  - 기존 PTX에 `objcopy`/`oxart_ptx`/`ptxas`를 돌렸습니다. `tools/sass-scan.sh`와 같은 조립 단계이고 cargo 빌드는 아닙니다.
  - `cuobjdump -xelf all`이 `/tmp/q3flashlit`에 653 MB를 썼고 끝에 지웠습니다.
  - 임대·홀드 확인은 박스 명령마다 먼저 했고 모두 비어 있었습니다(`lease=0 hold=0`, 이후 무출력). 카드, 게이트 락, 남의 프로세스는 건드리지 않았습니다.
  - 자문의 L2 용량 가설은 제 LRU 유도로 기각했습니다. 자문도 두 번째 검토에서 동의했습니다.
- **모델**: opus(Opus 5.5)로 스폰되어 그대로 돌았습니다.
