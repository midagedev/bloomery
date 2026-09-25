# cpugemm-lit — AVX2 CPU MoE GEMM 커널 설계 조사 (라운드 보고, 2026-09-25)

리드 메모: 파동 25 대기 창의 박스 없는 조사 라운드(opus). 스펙 `specs/wave-m6/spec-cpugemm-lit.md`의 여섯 질문에 대한 답이고,
`h3tile` 카드(`docs/plan-triage.md`)와 plan 「흐름 모델」 사다리 4의 정정(DRAM 바닥 c ≤ (W − a)/m̄ = 11.3 µs)의 출처다.
아래는 라운드의 보고 원문이다.

# `cpugemm-lit` 보고: AVX2 CPU MoE GEMM 커널 설계 조사

`c ≤ 16.6 µs`는 **Q3_K gate·up을 8행 행-레인 타일로 바꾸면 닿는다[derived]**. 이 방법은 비트 동일이다. 다만 스펙의 기준 셋 중 두 개가 틀렸다.

- **열당 MAC 수.** 스펙의 38.5 M은 shape에서 나오지 않는다. 3 × 5120 × 2304 = **35.39 M**이다. gate·up(Q3_K)이 23.59 M, down(Q4_K)이 11.80 M이다. 근거는 슬롯 16,773,120 B = 10,137,600 + 6,635,520 B, `docs/research/dsm-report.md:46` "k=5120 rows=2304" [derived].
- **DRAM 바닥.** `c ≤ W/m̄ = 16.6`은 고정항 a = 40.4 µs를 빠뜨렸다(`docs/plan.md:101`). 실제 조건은 a + c·m̄ ≤ W이고, **c ≤ (126.4 − 40.4)/7.6 = 11.3 µs**다. c = 16.6이어도 t(7.6) = 166.6 µs로 W의 1.32배, 아직 계산 바운드다 [derived].
- **"ik·mainline은 행 타일"이라는 전제는 절반만 맞다.** V4.1 P512(m̄ 7.6)에서 ik의 Q3_K는 우리와 같은 열 타일이다. 행 인터리브는 `_R4`/`_R8` 재패킹 타입과 nrc_y ≥ 32일 때만 쓴다. mainline의 MUL_MAT_ID에는 GEMM이 아예 없다(Q5 표).

변경한 파일은 없고 박스 잡도 없다. 참조 트리는 `ssh ws 'sed -n …'`로 읽기만 했다. `tools/box.sh`는 실행 전에 rsync(`--delete`)를 걸어 원격 디렉터리에 쓰므로 쓰지 않았다. 박스 작업이 없어서 `box-gc` 시작·끝 명령도 생략했다. 코드 커밋은 ik `db517b69`, mainline(V4.1 포트) `5210c7c5`, exllamav3 `0740edc2`, mistral.rs `d5ae0f18`(→ candle `35d7ae7`), 우리 `26bcb4a`다. 클록은 `docs/facts.md`에 없어서 3.6 GHz(4.0이면 괄호)로 두었다.

## 0. 척도(모두 [derived])

- **지금 속도.** c = 27.02 µs면 35.39 M/27.02 µs = 1.31 TMAC/s다. 코어당 40.9 GMAC/s이고, 코어·사이클당 11.4 MAC(10.2)이다.
- **Zen 3 발행 단위.** `vpmaddubsw`와 `vpmaddwd` 모두 FP0/3에서 1 uop, TP 0.5다 [M-blog uops.info]. `vpaddd`는 FP0/1/2/3, `vpunpckldq`·`vpshufb`는 FP12, `vpsignb`는 FP01이다. `vpbroadcastd ymm,m32`는 1 uop, TP 0.5인데 **Zen 3 포트 줄이 없다**(Zen 4는 FP1/2).
- **정수 사슬 상한.** maddubs 하나에 madd 하나가 붙는 꼴은 코어·사이클당 32 MAC이다. 32코어·3.6 GHz면 3.69 TMAC/s다.
- **FP03 슬롯이 판별 축이다.** 우리 h1fold Q3_K 타일은 열·슈퍼블록마다 maddubs 8 + madd 8 + bsum madd 1 = 17 FP03 / 256 MAC다(30 MAC/cycle). 열 하나당 FP03 바닥은 gate·up 92,160 × 8.5 + down 46,080 × 약 7(FP12 바운드)이다. 합이 약 1.1 M 코어-사이클이고, 32코어로 나누면 **약 9.6 µs**다.
- **실측 27.0은 FP03 바닥의 2.8배다.** 명령 수로 맞춰 보았다. hosttile 시점 명령 수는 Q3_K 65 + 71/8, Q4_K 48.5 + 약 5다. 열·전문가당 9.28 M 명령이고, 27.02 µs에 맞추면 **IPC ≈ 2.98**이다.
- **h1fold 뒤 예측.** 같은 IPC로 Q3_K 30.6 + 99/8을 넣으면 6.43 M 명령, **c ≈ 18.7 µs**다. 리드 예측 f 0.45 → 19.1과 맞는다.
- **남는 병목은 명령 발행이고, 원인은 아직 모른다.** 후보는 C = 8 스필 15/15, 활성값 L2 스트리밍, 발행 폭이다. 이것이 유도로 못 닫는 한 항이다. 박스 측정 하나로 가릴 수 있다: union 벤치를 `perf stat -e cycles,instructions`와 Zen 3 dispatch-stall 이벤트로 돌린다. 예측값은 IPC 3.0 ± 0.3이다.

## 1. 질문별 표

### Q1. 레지스터 블로킹과 산술 강도

| 설계 | 타일(행 × 열) | 누산기 | 가중치 unpack | 활성 L1 바이트/MAC | FP03 / 256 MAC(열당) | 출처 |
|---|---|---|---|---|---|---|
| 우리 h1fold | 1 × C ≤ 8 | sumi C개 + f32 1개 | 필드마다 u = q3l\|h≪2를 한 번(열 공유) | 1.14 | 17 | `crates/qdot/src/lib.rs:1015-1043,1063-1180@26bcb4a` [C-code] |
| ik Q3_K(AVX2 기본) | 1 × nrc_y ≤ 8 | sumi·accd 각 nrc_y개 | HighBit3 `mh = 0x04` → 0..7, 슈퍼블록 128값마다 | 약 1.1 | 17(−4는 float fma) | `iqk_gemm_kquants.cpp:485-498,532-551,677-720,1825-1828@db517b69`; `iqk_common.h:140,400-406` [C-code] |
| ik Q3_K_R4(재패킹) | 4행을 레인에 × nrc_y | acc nrc_y개(레인 = 4행 × 서브블록 2) | ib(4행 × 32값)마다 약 20 uop 공유 | 0.25(32 B 적재 + pshufd 4) | 10 | `iqk_gemm_kquants.cpp:1354-1487`, 1478 "Quants are in 0...8, so we can add add up all of them as int16_t without overflowing" [C-code] |
| ik Q8_K_R8(nrc_y ≥ 32 즉석 변환) | 8행을 레인에 × nrc_y | acc nrc_y개 | 변환을 한 번 하고 전 열에 재사용 | 약 0.1 | 16(sign_epi8 둘 추가) | `iqk_gemm_kquants.cpp:1836-1880`; `iqk_mul_mat.cpp:256` [C-code] |
| mainline gemm_q4_K_8x8(dense 전용) | 8 × 16(q8_Kx4 네 개) | f32 32개(acc_rows 16 + acc_min_rows 16) → 스필 | 8행 인터리브를 적재 때, 서브블록마다 16 활성 행이 공유 | 약 0.3(+ lhs 셔플 48 / 2048 MAC) | 9.5(i16 선합산 8개) | `ggml-cpu/arch/x86/repack.cpp:2818-2841,3074-3112@5210c7c` [C-code] |

DRAM 강도는 **어느 설계든 2.11·m MAC/바이트**다(16.77 MB당 35.39 M·m). 슬라이스가 캐시에 있으면 m개 열이 DRAM을 한 번만 읽기 때문이다. 133 GB/s면 DRAM 천장은 281·m GMAC/s다.

| m | DRAM 천장 | 오늘(1.31 T) | h1fold(1.89 T) | 행-8 Q3(2.23 T) | 행-8 Q3 + Q4 q8_K(2.88 T) |
|---:|---:|---|---|---|---|
| 1 | 0.28 T | DRAM | DRAM | DRAM | DRAM |
| 4 | 1.12 T | DRAM(m* 4.7) | DRAM | DRAM | DRAM |
| 8 | 2.25 T | 계산 1.7배 | 계산 1.2배 | 균형 | DRAM |
| 16 | 4.50 T | 계산 | 계산 | 계산 | 계산 |

한계율만 본 값이고 a는 뺐다 [derived]. 같은 CPU에서 ik가 낸 최고 AVX2 dense 속도와 견주면 이렇다(Llama-3.1-8B 6.98 GMAC/토큰[derived], 5975WX 32T pp512).

- Q3_K 199.22 t/s ≈ 43.5 GMAC/s/코어.
- Q3_K_R4 262.34 ≈ **57.2**.
- Q8_K_R8 293.72 ≈ 64.

c = 16.6은 66.6 GMAC/s/코어가 필요하다. **ik의 최고 dense 속도보다 위다.** c = 11.3이면 98이 필요하다.

### Q2. 디퀀트 재사용(m > 8 포함)

| 설계 | m ≤ 8 | m > 8 | 출처 |
|---|---|---|---|
| 우리 | 행 바깥·런 안쪽. ⌈m/8⌉ 런마다 L1의 행을 다시 unpack한다(h1fold: 슈퍼블록마다 99 고정 + 30.6·C) | 같음, 런 추가 | `crates/model/src/ops.rs:1507-1535`; triage h1fold 카드 [C-code] |
| ik | `mul_mat_NxM` k_x_step 64행 블록 바깥, 열 스텝(≤ 8) 안쪽. m > 8은 고르게 나눈다(12 → 6 + 6) | **m ≥ 32부터** Q3_K → Q8_K_R8, Q4_K → Q8_1을 32–64행씩 thread-local에 변환해 전 열에 재사용한다. up_gate는 64행 | `iqk_mul_mat.cpp:61-123,256-257,740-766,807-839` [C-code]. PR #531: "u-batch sizes of 1024 tokens or more"가 되어야 DeepSeek급 MoE에서 효과가 난다 [M-blog] |
| mainline | MoE는 재사용 없음: Q3_K는 (행, 열)마다 `vec_dot`. 재패킹된 Q4_K도 활성 행마다 gemv 한 번 | 같음 | `ggml-cpu.c:1494-1528`(1525 `vec_dot(…, 1)`); `repack.cpp:4497-4513` [C-code] |
| exllamav3 | k-major 밴드, 디코드를 토큰 ≤ 4(MAX_M)가 공유 | 청크 추가 | `moe_mul1.cpp:52-55,69@0740edc` [C-code] |

### Q3. 행 타일 대 열 타일(union 모양)

| 방향 | 활성 L1 바이트/MAC | 가중치 L1 바이트/MAC(m = 7.6 / 1) | 레지스터 16개 안에서 |
|---|---:|---|---|
| 열(1 × 8, 지금) | 1.14(8열 × 5.9 KB = 47 KB, 32 KB L1D 초과 → 행마다 L2) | 0.057 / 0.43 | sumi 8 + u + sc + 마스크·셔플 → 15/15 스필 |
| **행-레인(8행 × C열)** | **0.14**(4 B 브로드캐스트 / 32 MAC) | 0.057 / 0.43 | acc C(ymm 하나 = 8행 i32) + 가중치 4 + i16 부분합 1 + 스케일 1 + bcast 1 → C = 6이면 여유, C = 8이면 빠듯 |
| 4행 × 8열을 지금 열 레이아웃으로 | 0.29 | 같음 | 누산기 32 + u 4 + sc 4 + q8 1 = 41 ymm → 불가 |

비트 동일 구성은 이렇다. 레인 = (행, 16값 서브블록)으로 맞추고, maddubs 4개를 i16으로 먼저 더한다(한도 8·7·127 = 7,112). 그 다음 **정수** 스케일 벡터 [s_rj, s_rj]로 madd 한 번을 한다. ik R4의 `madd(ones)` + float fma 자리를 이 정수 madd가 대신하고, uop 수는 같다. 슈퍼블록마다 정수 sumi가 정확하고, float 단계 acc + (dcol·d_r)·sumi는 행 레인에서 같은 곱·같은 순서다. 그래서 `dot_row`와 bit for bit로 유지된다. 이것이 AGENTS.md의 "integer-path reorder" 클래스다. (advisor는 선합산이 비트를 깬다고 보았다. 스케일을 float로 옮길 때만 그렇다.)

- 수: 256 MAC당 열마다 약 28.6 명령(FP03 10–11), 공유 unpack 약 31(스케일을 i8로 미리 펼치면)이다. C = 8에서 32.5 대 43.0, **−24 %**다. Q3_K 명령은 3.96 M → 3.0 M, **c ≈ 15.9 µs**다.
- 교차 확인: 같은 CPU에서 ik PR #134가 Q3_K → Q3_K_R4로 **1.317배**를 냈다. 3.96/1.317 = 3.0 M으로 같은 값이 나온다.
- a도 내려간다. Q3_K 고정분 99 → 31/슈퍼블록-행이라 약 −18 µs, **a ≈ 30**이다(h1fold가 71 → 99로 올린 7 µs 포함).
- down(Q4_K × q8_2_x4)은 **행-레인으로 비트 동일하게 옮길 수 없다.** ik의 8-레인 float 누산과 `hsum_float_8` 순서가 한 열 비트에 묶여 있다(`lib.rs:2225-2380`). 남는 길은 down 활성값을 q8_K로 바꾸는 것이다. 이는 mainline의 Q4_K vec_dot_type이고(`repack.cpp:4534-4535`), 약 27명령/256으로 **c ≈ 12.3 µs**가 된다. 그러면 a 30 + 12.3 × 7.6 = 123.5 < W로 **DRAM 바운드**다. 대가는 디코드를 포함한 Q4_K/Q5_K 경로의 반올림 변경과 재핀이다.

### Q4. VNNI 없는 int8 포화 회피

| 설계 | 방법 | MAC당 비용 | 출처 |
|---|---|---|---|
| 우리 Q3_K | u 0..7(부호 없음) × q8 ±127, 쌍 ≤ 1778. −4는 `madd(−4·s, bsum)` 정수로 | 슈퍼블록·열마다 madd 1 + 적재 1(약 0.8 %) | `lib.rs:1044-1054,1120-1127` [C-code] |
| ik Q3_K | 같은 u(`mh = _mm256_set1_epi8(0x04)`), −4는 float `process_mins_and_scales_16(…, -4.f*d, …)` | 열마다 madd + cvt + fma | `iqk_gemm_kquants.cpp:496,539` [C-code] |
| mainline Q3_K | maddubs 둘(q3l, q3h) + sub(우리 h1fold 전 모양) | 32 MAC당 maddubs 2 | `arch/x86/quants.c:1848-1870` [C-code] |
| 부호 접기(ik Q6_K·Q8_K_R8, 우리 Q6_K) | `sign_epi8(x, x)`로 부호 없는 쪽, 부호는 활성값으로 | maddubs마다 psignb 2(FP01). PR #141: 넘치는 판 354 → 올바른 판 293 t/s(**−17 %**) | `iqk_gemm_kquants.cpp:877-882,1864-1872`; `lib.rs:1737-1741` [C-code][M-blog] |
| 활성값 영역 | q8_K 코드는 −127..127(음수 쪽은 구성상 −127, +127에서 한쪽 클램프)이라 −128 없음 | — | `lib.rs:597` [C-code] |
| exllamav3 | bytesum-first: 곱 바이트를 `maddubs(·, 0x01)`로 먼저 더하고 x는 쌍 밖에서 madd | 코드북 전용 | `moe_mul1.cpp:44-48,1228-1240` [C-code] |

i16 선합산 한도: Q3 u 0..7은 16값 14,224. Q4_K 0..15는 16곱 30,480 < 32,767(mainline이 이 가장자리를 쓴다, `repack.cpp:3074-3093`). Q5_K 0..31은 8곱까지(31,496).

### Q5. 스레드 분해

| 엔진 | 균형 단위 | 층당 join | 출처 |
|---|---|---|---|
| 우리 | 8 expert 청크 × {gate·up, down} 디스패치. 레인은 `row_cost` = 바이트 × 열(선형)로 자르고, 레인당 4블록을 훔친다 | **80**(317 expert → 40청크 × 2). 꼬리 약 15 µs/join이면 약 1.2 ms/층[derived] | `moe.rs:959,1195-1250`; `ops.rs:2093,2238-2270` [C-code] |
| ik(`GGML_EXPERT_CHUNKING` 기본 ON) | (expert, 행 조각) 단위. `chunks_per_expert = min(nth, rows/32)` → up_gate 32 × 72행, down 32 × 160행. 원자 카운터 하나, expert-major 순서, 활성값은 op당 한 번 양자화 | op당 배리어 1(양자화 뒤) + op 끝 → 층당 약 4 | `ggml.c:18261-18333,18633-18680`; `ggml/CMakeLists.txt:98` [C-code] |
| mainline | expert 순차 루프, expert마다 미리 리셋한 카운터(expert 사이 배리어 없음). 16 × 16 청크 수가 4·nth 미만이면 nth로 행 분할 | op당 1 | `ggml-cpu.c:1657-1735` [C-code] |
| exllamav3 | 단계별(gate·up → down). 프리필이면 GEMV 전체를 워커에 스트라이드(같은 expert를 동시에 읽어 L3 공유), 적으면 8타일 그룹 평탄 분할 | 단계당 1 | `moe_mul1.cpp:1942-1964` [C-code] |
| mistral.rs(candle) | 활성 행마다 가중치 행 quad 풀 디스패치(가중치를 활성 행마다 다시 읽는다). AVX2 Q3_K에는 `vec_dot_2/4` 오버라이드가 없고, x86 재패킹은 AVX-512 VNNI 전용(Q4K/Q6K) | 활성 행마다 | candle `k_quants.rs:69-80,2440-2460`; `repack_x86.rs:1-5,43-52@35d7ae7` [C-code] |

### Q6. 비교 가능한 CPU의 공개 수치(조건이 적힌 것만)

| 출처 | CPU·스레드 | 모델·타입 | ubatch | 값 | 비고 |
|---|---|---|---|---|---|
| ik PR #134 | 5975WX, 32T | Llama-3.1-8B, Q3_K 대 Q3_K_R4 | pp512 | 199.22 → 262.34 t/s(1.317×) | 같은 CPU·같은 타입의 열 타일 대 행 타일 [M-blog] |
| ik PR #141 | 5975WX | Llama-3.1-8B, Q8_0_R4 대 Q8_K_R8 | pp512 | 234.40 → 293.72(넘치는 판 354) | 부호 접기 비용의 실측 [M-blog] |
| ik PR #531 | 7950X(Zen 4, AVX-512) | Llama-8B Q3_K → Q8_K_R8 | pp512 | 178.56 → 344.45 | ISA가 달라 모양만 [M-blog] |
| ik 디스커션 #223 | Epyc 7773X Zen 3 64c, CPU 전용, -t 63 | DS-R1 Q4_K_M, rtr | **ubatch 미기재**(프롬프트 약 500) | 58.58 t/s | 조건 일부만 [M-blog, 커뮤니티] |

MoE CPU 전용이면서 조건이 다 적힌 Zen 3 수치는 이것 말고 찾지 못했다. 나머지(5975WX Mixtral 61, ubergarm, KT)는 `docs/research/prefill-lit-report.md` Q5·Q6 행 그대로다.

### 결론 한 문단

- **타일:** Q3_K gate·up을 8행 행-레인 × C(6–8)열 타일로 바꾼다. i16 선합산 + 정수 스케일 madd, 적재 때 8행 인터리브 재패킹, 스케일은 i8로 미리 펼친다(W +3.6 %).
- **스레드:** 층마다 gate·up 한 op, down 한 op를 (expert, 행 조각) 원자 큐로 돌린다(ik 모양). x는 호출당 한 번 양자화한다.
- **효과(유도):** c 18.7 → **약 15.9 µs**, a 48 → 약 30 µs, 비트 동일. t(7.6) ≈ 151 µs = 1.19 W로, 아직 DRAM 바닥은 아니다.
- **코드 비용:** 새 재패킹 타입과 그 로더. 호스트 집합 214 GB를 적재 때 익명 메모리로 재패킹하면 264 GB 램의 페이지 캐시·populate 흐름과 충돌한다. 그래서 **오프라인으로 재패킹한 파일**(ik `-rtr`의 오프라인판)이 맞다. 여기에 타일 커널, 레인 비용 모델, 큐 디스패치가 붙는다.
- **열린 것:**
  1. **리드·사용자 결정:** DRAM 바닥(c ≤ 11.3)까지 가려면 down Q4_K/Q5_K 활성값을 q8_2_x4에서 q8_K로 바꿔야 한다(c ≈ 12.3). 반올림 변경과 재핀이 따르고 ik 패리티가 깨진다.
  2. IPC ≈ 3의 원인. perf stat 한 번으로 가린다.
  3. `vpbroadcastd`의 Zen 3 포트(FP12라면 셔플과 겹친다).
  4. join 꼬리의 실제 크기.

## 2. 권고가 따른 설계와 엔진 사이의 차이

- **행-레인 + i16 선합산:** ik `_R4`의 레인 배치(`kquants.cpp:1354-1487`)와 mainline 8x8의 적재 때 인터리브·i16 깊이(`repack.cpp:3074-3112`)를 따랐다.
- **ik와 다른 점:** 스케일을 float fma가 아니라 정수 madd로 건다. 이것이 비트 동일성의 조건이다.
- **mainline과 다른 점:** 활성 행 인터리브(q8_Kx4)와 누산기 32개 스필을 버리고, 4 B 브로드캐스트와 누산기 C개로 간다.
- **exllamav3:** 같은 2D 발상(k-major 밴드, 토큰 ≤ 4 공유)이지만 코드북 전용이라 설계 참고만 했다.
- **mistral.rs:** AVX2 K-quant GEMM이 없다.
- **스레드:** ik EXPERT_CHUNKING을 따랐다. mainline(expert별 카운터)은 가깝고, exllamav3는 정적 스트라이드다.

## 3. 출처(이번 라운드에 받은 것)

- https://uops.info/html-instr/VPMADDUBSW_YMM_YMM_YMM.html — "Latency: 3 / Throughput: 0.50 / Port usage: FP0/3"(Zen 3)
- https://uops.info/html-instr/VPMADDWD_YMM_YMM_YMM.html — "Measured (loop): 0.50", 1×FP03
- https://uops.info/html-instr/VPBROADCASTD_YMM_M32.html — Zen 3 "Measured (loop): 0.50", uop 1, 포트 줄 없음(Zen 4 "FP1/2")
- https://uops.info/html-instr/VPADDD_YMM_YMM_YMM.html — 0.25, FP0/1/2/3
- https://uops.info/html-instr/VPUNPCKLDQ_YMM_YMM_YMM.html — "1*FP12"
- https://uops.info/html-instr/VPSHUFB_YMM_YMM_YMM.html — 1×FP12
- https://uops.info/html-instr/VPSIGNB_YMM_YMM_YMM.html — "1*FP01"
- https://github.com/ikawrakow/ik_llama.cpp/pull/134 — "Here is `PP-512` for LLaMA-3.1-8B on … `AVX2` (Ryzen-5975WX)", 32T 199.22 / 262.34
- https://github.com/ikawrakow/ik_llama.cpp/pull/141 — 354 t/s 넘치는 판, `_mm256_sign_epi8` 교정 뒤 293
- https://github.com/ikawrakow/ik_llama.cpp/pull/531 — Ryzen-7950X Q3_K 178.56 → 344.45, "u-batch sizes of 1024 tokens"
- https://github.com/ikawrakow/ik_llama.cpp/discussions/223 — "Epyc 7773X (64 cores, 128 threads), one socket, 8x128GB RAM", 58.58
- https://github.com/ggml-org/llama.cpp/discussions/11765 — CPU 전용 pp 수치 없음(TG만), 인용할 값 없음
- 코드는 위 표의 `path:line@commit`이다. ik `github-data/`의 PR 사본은 URL 페치로 대조했다.

## 4. 범위 밖 개선 여지(보고만, 손대지 않음)

- **`docs/plan.md:101`** DRAM 바닥 "c가 W/m̄ = 16.6 아래"는 a를 빠뜨렸다. c ≤ (W − a)/m̄ = 11.3이 맞다. 스펙의 "38.5 M MAC"도 35.39 M이다. XS.
- **`crates/model/src/moe.rs:1218-1219`** gate와 up이 같은 `cols`를 별도 슬롯으로 받아 x 열을 두 번 양자화한다. 청크별 재양자화 위에 2배가 더 붙는다(unionreal이 받을 몫). S.
- **`crates/qdot/src/lib.rs:459-461`** `dot_row_cols`가 행·런마다 `check_row`를 C번 부른다. 슬롯마다 한 번 검증으로 올리면 c의 약 1–2 %[derived]다. S.
- **`crates/model/src/ops.rs:1522-1525`** 런마다 슬라이스 8개를 경계검사하며 새로 만든다. 1 % 미만. XS.
- **공개 비교용 사실:** mainline MUL_MAT_ID CPU 경로는 GEMM이 없다. Q3_K는 (행, 열)마다 vec_dot, 재패킹 Q4_K는 활성 행마다 gemv다(`ggml-cpu.c:1525`, `repack.cpp:4497-4513`, 3D 적격 조건은 `repack.cpp:4796-4799`). rig-log의 ik 165.8 대 llama.cpp 104.6이 여기서 온다. 문서 한 줄, XS.
- `crates/qdot` C = 8 Q3_K 타일 스필 15/15는 이미 triage에 있다. 행-레인 설계로 가면 없어진다.

opus로 스폰됐다(Opus 5.5).
