# cardroute-design — V4.1 프리필 카드 route의 항별 분해와 레버 (2026-09-26 새벽, 설계 라운드)

리드 메모. 시팅 14가 드러낸 카드 route 30.35 ms/층-배치(A6000, plan (a), lcg, P 512)를 종이 위에서 나눈 opus 설계 라운드의
보고를 원문 그대로 싣는다(박스는 좁은 읽기만, 빌드·게이트·타이밍 없음). 리드 확인: 포럼 인용 셋은 라운드가 받아 둔 원문에
있고(`forum-246254`·`-218750`·`-33866`), 코드 주장 넷(`body/prefill.rs:675-680` 캡처 거부, `gpu/src/cores.rs:1118-1121` m열
비트 계약, `gpu-deepseek41/src/hc.rs:81` `HC_MAX_TOKENS = 8`, cuda-core 0.3.1 `src/simt/launch.rs:899-911` 런치마다 속성 질의 —
보고의 `launch.rs` 경로는 `simt/` 아래다)은 원본과 맞다. 처분: 첫 카드 라운드는 **B1**(`docs/plan-triage.md`의 `ds41proj`),
그 전에 박스 실행 둘(P 384 큐 판정, nsys 잔여 판정); A·C는 B1에 지배된다; B4는 프리필 = 스텝 비트 불변식을 옮기므로 사용자
결정. 이 보고로 정정한 것: ds41bulk 커밋과 리드의 시팅 14 기록이 쓴 "청크별 런치 바닥"은 틀렸다 — 의존 런치 간격은 층·배치당
1.4–3.6 ms이고 나머지는 청크마다 다시 도는 커널 실행이다; 리드가 쓴 "파트 C로 route를 반으로 → 615 / 798"은 근거 없는 반감
가정이었다.

---

Q의 적합 대역 1,080–1,140으로 다시 계산한 예측과 B1의 큐 여유 대역이 아래 절에 들어갔다. 모든 절을 이 값에 맞췄다. 보고는 아래와 같다.

---

**결론.** card route는 커널 실행에 묶여 있다. 호스트 발행률과 런치 간격으로는 실측의 작은 몫만 설명된다. 호스트 enqueue 시간의 대부분은 드라이버 런치 큐가 가득 차서 막힌 시간이다. 첫 레버는 B1이다. B1은 프로젝션을 청크 루프 밖으로 빼서 배치 전체 폭으로 한 번씩 돈다. B1의 큰 값은 route 단축보다 호스트 해방이다. 큐에 한꺼번에 쌓이는 항목이 큐 깊이 아래로 내려간다. 그러면 한 스레드로도 G 스케줄러가 다음 route를 union 뒤에 숨긴다. route를 더 줄이는 B2, B3, B4는 hoststream 스트리밍이 호스트를 줄인 뒤에야 크게 값을 한다.

A6000, plan (a), lcg, P 512, layer-batch당:

| 항목 | 값 |
|---|---|
| 실측 route, sitting 14의 wait + enqueue | 30.35 ms |
| 호스트 발행 상한 [derived] | 5.2 ms 이하 |
| 런치 간격 [derived] | 1.4–3.6 ms |
| 종이 위 커널 합 [derived] | 19.5 ms, 대역 16.3–23.4 |
| 설명 안 된 잔여 [derived] | 10.9 ms, 실측의 36 % |
| 큐 깊이 적합값 Q [derived] | 1,080–1,170 항목 |
| route 항목, 오늘과 B1 뒤 [derived] | ≈1,633, ≈463 |

pp4096 tok/s [derived], A6000, plan (a):

| 형태 | lcg | prose-in |
|---|---:|---:|
| G1 오늘, 모델 | 174.0 | 245.8 |
| G8, 오늘 한 스레드 | 190.2 | 277.3 |
| G8과 B1 | 236.3–237.2 | 388.6–391.5 |
| G8과 B1, 스트리밍 ring 128 | 483.3–502.3 | 543.5–573.2 |

코드 인용은 모두 main `a27e542` 기준이다. 작업 트리는 깨끗했다. 저장소 파일, 빌드, 게이트, 타이밍 실행, git 상태는 건드리지 않았다. 박스는 두 `flock -n` 검사가 통과한 뒤 이름 붙인 파일만 `sed -n`과 `grep -n`으로 좁게 읽었다.

리드에게 박스 실행 두 개를 제안한다. 둘 다 30분 안에 끝난다.
1. **큐 깊이 판정**: A6000 plan (a), lcg, P 384, `BLOOMERY_STEP_STATS=1`, CARD_EXPERTS 두 arm. 세 가설의 예측은 4절에 있다.
2. **잔여 판정**: A6000에서 P 512 layer-batch 하나를 nsys로 추적한다. 잔여 10.9 ms를 커널별로 가른다. B2를 열지 말지와 B4의 크기가 여기서 정해진다.

모델 행은 다시 돌릴 수 있다. 스크립트 셋이 `recal-tables.py`를 통해 recal 모델을 읽는다. 경로는 끝의 산출물 목록에 있다. 쓰는 용어는 다음과 같다.
- **dm**: 레버가 512 토큰 layer-batch 하나에서 줄이는 route ms다. 음수가 빠른 쪽이다. 모델은 dm/512를 토큰당 route 비용에 더한다.
- **토큰당 route 비용**: P 512의 30.35 ms를 CED 가중 토큰 483.9로 나눈 값이다. s14-recheck는 512로 나눠서 5.8 % 낮았다. P 4096 값은 같은 sitting의 32.9 ms를 486.5로 나눈 값이다.
- **G**: 한 층을 같이 지나는 배치 수다. G1이 오늘 순서다.
- **호스트 자유**: 다음 route를 발행하는 동안 호스트가 막히지 않는다고 둔 행이다.
- **한 스레드**: 배치마다 큐 막힘 18.7 ms를 벽시계에 더한 행이다.
- **ring**: hoststream 스트리밍의 링 슬롯 수다.
- **N_r, N_s**: route와 shadow가 layer-batch 하나에서 큐에 넣는 항목 수다. 커널, 복사, event 기록, 스트림 대기를 각각 하나로 센다.
- **t_c**: 카드가 route 항목 하나를 소비하는 평균 시간이다.

lcg pp512와 pp4096 보정값은 모델 123.0과 174.0, 실측 121.4와 169.4다.

**1. 분해와 판정**

항목 수. P 512, CED 켬, 40층 평균, [derived]:

| 항목 | 청크당 | layer-batch당 | 근거 |
|---|---:|---:|---|
| attention 청크 커널 | 21.2 | | `chain/attn.rs:1430` |
| fork와 join 스트림 연산 | 4 | | `gpu/src/graph.rs:251`, `:283` |
| route의 청크별 norm_quant | 1 | | `chain/ffn/batch.rs:1207` |
| full 청크 60.1개 소계 | 26.2 | 1,575 | `body/prefill.rs:968` |
| latent 전용 청크 1.6개 | 4 | 6.4 | `body/prefill.rs:947` |
| engram, 층 1과 14에서만 | 14 | 44.8 | `chain/glue/batch.rs:215` |
| route 꼬리: scores, pick, places, D2H 셋, event | | 7 | `chain/ffn/batch.rs:1181`, `:961` |
| **route 합 N_r** | | **≈1,633** | ds41bulk 계수 ≈1,670 |
| shadow N_s, expert arm | 8 | 486 | `chain/ffn/batch.rs:1454` |
| shadow N_s, slot arm | 10 | 601 | 같은 함수 |

청크 커널 21.2는 층 종류별 커널 수의 평균이다. 기본 층이 20개, indexer 층이 22개, compressor source 층이 29–30개다. CED를 끄면 N_r은 ≈1,730이고 N_s는 517과 640이다. 합 1,633은 ds41bulk 계수와 2 % 안에서 맞는다.

커널별 하한. 공개 파일 기준이고 attention 가중치는 모두 Q3_K, 256 가중치당 110 B다. decode 실측은 A6000, m = 1 값이며 `docs/plan-ledger.md:1159-1185`에 있다.

| 커널 | grid × block | 청크당 읽기 | 700 GB/s 하한 µs [derived] | decode m = 1 실측 µs | 근거 |
|---|---|---:|---:|---:|---|
| joint qkv gemv | 행/8 × 256 | 3.94 MB | 5.6 | ≈7.5 [derived] | `gpu/src/lib.rs:2380` |
| q_b gemv | 4,096 × 256 | 18.0 MB, indexer 층 20.25 | 25.7–28.9 | 31.06 | 같은 곳 |
| wo_a heads m-col | 1,024 × 256 | 14.4 MB | 20.6 | 26.07 | `gpu-deepseek41/src/dense.rs:900` |
| wo_b gemv | 640 × 256 | 18.0 MB | 25.7 | 31.74 | `gpu/src/lib.rs:2380` |
| attention seg | 320 × 512 | 키 행 | 계산 바운드 | depth 6에서 10.9, ≈1024에서 16.2 | `gpu-deepseek41/src/attn.rs:1089` |
| attention merge | 512 블록 | 부분합 | 작은 커널 | | `gpu-deepseek41/src/attn.rs:1090` |
| norm_quant, K 5120과 1280 | m × 1024 | | 작은 커널 | ≈6, 3.71 | `gpu/src/fused.rs:634` |
| hc_pre | 40 × 256 | | | 10.97–11.04 | `gpu-deepseek41/src/hc.rs:879` |
| quantize, heads와 wo_b 입력 | m·n_sb·2 × 32 | | 작은 커널 | 2.15, 2.24 | `gpu/src/lib.rs:2182` |

청크당 가중치 읽기는 54–65 MB이고, 700 GB/s에서 하한은 약 0.09 ms다 [derived]. ds41bulk 커밋도 같은 값을 적었다. 이는 실측 청크당 0.45 ms의 5분의 1이다.

분해. A6000, plan (a), lcg, P 512:

| 항 | 청크당 µs [derived] | layer-batch당 ms [derived] | 잡은 방법 |
|---|---:|---:|---|
| 프로젝션 넷, m = 8 | ≈160, 140–185 | 9.6, 8.4–11.1 | 정수 경로 발행률 |
| attention: seg, merge, commit, append | ≈49, 38–65 | 2.9, 2.3–3.9 | decode 실측에 m = 8 행을 곱함 |
| 작은 커널: norm_quant, quantize, 전치, rope, hc | ≈36.5 | 2.2 | decode 실측, 또는 최소 2–4 µs |
| source와 indexer, 40층 평균 | 3.3 | 0.2 | decode 실측 |
| route norm_quant | 6 | 0.36 | decode 실측 |
| 런치 간격 | 37, 21–54 | 2.2, 1.3–3.2 | 26.2 × 0.85–2.23 µs |
| route 꼬리와 engram | | 1.9 | |
| **합** | ≈292, 238–358 | **19.5, 16.3–23.4** | |
| 실측 | | A6000 30.35, 3090 28.9–29.3 | sitting 14, ds41bulk |
| **잔여** | | **10.9, 7–14** | 실측의 36 % |

프로젝션 항은 산술로 잡았다. m-column 코어의 열 1..7은 가드된 블록이다. 블록마다 u64 로드 둘, d8 로드 하나, dp4a 넷, 정수 곱셈-누산 사슬, f32 누산 하나를 한다. 근거는 `gpu/src/cores.rs:1069-1300`이다.

CUDA 가이드 11.1.1의 Table 3에서 sm_86 값은 다음과 같다. 32비트 정수 덧셈, 곱셈, 시프트, 비교, 비트 연산이 SM당 클럭당 64개다. "All other type conversions"는 16개다. dp4a 행은 표에 없다.

이 값으로 워프 반복 하나가 m = 8에서 약 84.5 SM 사이클, m = 1에서 약 40 사이클이 된다 [derived]. 1.8 GHz에서 프로젝션의 유효 대역 상한은 약 394 GB/s다 [derived].

| 가설 | 예측, layer-batch당 ms [derived] | 판정 |
|---|---:|---|
| 발행률 바운드 | 1,633 × 3.2 µs = 5.2 이하 | 기각 |
| 간격 바운드 | 1,633 × 0.85–2.23 µs = 1.4–3.6 | 기각 |
| 실행 바운드 | 16.3–23.4에 잔여 | 채택, 소거법 |

호스트는 항목당 약 3.2 µs를 쓴다 [derived]. 3090 P 513의 실측 enqueue 3.5 ms를 막히지 않은 항목 약 1,106개로 나눈 값이다. 카드는 항목당 18.7 µs를 쓴다 [derived]. 호스트가 5.8배 빠르다. 호스트의 런치 하나는 이런 경로를 거친다.
- 런처가 매 호출 `prepare_*`를 부른다. 근거는 `gpu/src/lib.rs:2208-2210`이다.
- `__prepare`가 장치 속성 8개와 함수 속성 3개를 묻는다. 근거는 cuda-core 0.3.1 `launch.rs:899-908`이다.
- 질의마다 컨텍스트를 스레드에 묶는다. 원자 교환 하나와 `cuCtxGetCurrent` 하나다. 근거는 cuda-core `context.rs:382`이다.
- 발행할 때 인자 Vec를 할당하고 `cuLaunchKernel`을 부른다. 근거는 cuda-macros `launchers.rs:269`이다.

Rust 쪽과 `cuLaunchKernel`의 몫은 따로 재지 않았다. 그래도 판정은 달라지지 않는다. 3.2 µs를 다 합쳐도 카드보다 빠르다.

graph 노드 간격은 A6000 실측이다. 784 노드에서 0.852 µs, 504 노드에서 0.871 µs다. eager 빈 커널은 3090에서 런치당 2.233 µs로 쟀고, rig-log 2026-09-21에 기록되어 있다.

**ds41bulk 해석의 정정.** ds41bulk 커밋은 card_out을 청크당 런치 하한으로 읽었다. 의존 런치 약 26개가 각각 약 14 µs를 쓴다는 해석이다. 그런데 의존 런치 간격은 0.85–2.23 µs다. 나머지 약 12 µs는 커널 실행이다. 그래서 런치 수만 줄이는 레버는 카드에서 layer-batch당 약 2.3 ms 넘게 벌지 못한다 [derived].

**잔여.** 잔여 10.9 ms는 청크별 커널의 m = 8 팽창이라고 부른다. 모형은 m = 1 대비 약 1.85배 팽창을 준다. 실측을 맞추려면 2.6–2.85배가 필요하다. 제안한 nsys 추적이 이 차이를 커널별로 가른다.

**DRAM 교란.** 두 카드의 route는 4 % 차이인데 DRAM 대역은 22 % 차이다. DRAM 바운드가 아니라는 힌트다. 다만 3090 값은 250 W 캡과 gate placement에서, A6000 값은 plan (a)에서 나왔다. 그래서 증명으로 쓰지 않는다.

**2. 자원 타임라인**

오늘. A6000, plan (a), lcg, P 512, layer-batch당 ms, sitting 14:

| 자원 | 바쁜 시간 | 비고 |
|---|---:|---|
| 호스트 enqueue | 18.8 | 큐 막힘 ≈12.0, 자체 발행 6.8 [derived] |
| 호스트 wait | 11.7 | route D2H 대기 |
| 호스트 union | 72.3 | |
| 호스트 copy, upload, join | ≈1.8 [derived] | non-union에서 wait와 enqueue를 뺀 값 |
| 카드 route | 30.4 | 호스트의 enqueue와 wait 동안 돈다 |
| 카드 shadow | 12.6 | 3090 gate placement, ds41bulk 뒤 실측. union 아래 숨는다 |
| PCIe D2H와 H2D | ≈0.8 [derived] | 21 MB |
| **벽시계** | **104.6** | 호스트 한 스레드의 합 |

오늘 벽시계는 합이다. route가 union 앞에 그대로 선다.

G 스케줄러 모양. P 4096, CED로 비지 않은 layer-batch 225개의 평균, ms, [derived]:

| 형태 | 호스트 끝, prose-in / lcg | route | 벽시계, prose-in / lcg | 묶는 자원, 40층 중 | pp4096, prose-in / lcg |
|---|---|---:|---|---|---|
| G1 오늘 | 73.8 / 104.3 | 32.2 | 74.1 / 104.6 | 합 | 245.8 / 174.0 |
| G8 한 스레드 | 47.1 / 77.1 | 32.2 | 65.6 / 95.7 | 호스트와 큐 막힘 | 277.3 / 190.2 |
| G8 호스트 자유 | 47.1 / 77.1 | 32.2 | 47.4 / 77.4 | 호스트 40, 호스트 40 | 384.4 / 235.1 |
| G8 스트리밍 ring 8 | 38.8 / 46.9 | 32.2 | 39.1 / 47.5 | 호스트 40, 호스트 16과 stream 24 | 465.1 / 383.1 |
| G8 스트리밍 ring 128 | 35.6 / 39.5 | 32.2 | 35.9 / 39.7 | 호스트 32와 stream 8, 호스트 40 | 507.4 / 458.0 |

표의 "stream"은 카드의 스트리밍 경로다. 모형에서 이 경로는 G개의 route가 끝난 뒤에 시작한다.

합이 max로 바뀌는 조건:
- G 스케줄러는 배치 g의 union 동안 카드에 배치 g+1의 route를 돌린다. 그동안 호스트가 막히지 않아야 합이 max가 된다.
- 한 스레드라면 다음 route를 발행하다가 큐에서 막힌다. 배치마다 18.3–19.4 ms다 [derived]. 이는 앞 배치의 shadow가 큐에 남아 있다고 둔 값이다. 발행 순서에 따라 9–24 ms로 움직인다 [derived].
- 호스트를 푸는 길은 셋이다. C는 막힘을 다른 스레드로 옮긴다. A는 route를 graph 발행으로 바꾼다. B1은 한꺼번에 쌓이는 항목을 큐 깊이 아래로 내린다.

**핵심 관찰.** 호스트가 풀린 G8에서는 모든 행에서 route만으로 묶이는 층이 40층 중 하나도 없다. 그래서 스트리밍 전에는 route 레버가 G8에서 0–7 %만 낸다. 스트리밍이 호스트를 줄이면 stream 쪽이 묶기 시작한다. route가 stream의 앞부분이므로, 그때부터 route 레버가 벽시계에 거의 그대로 반영된다.

**3. 레버**

레버 정의:
- **A**: 층마다 route와 shadow를 CUDA graph로 캡처해 재생한다.
- **B1**: 입력 프로젝션과 출력 프로젝션을 청크 루프 밖으로 빼서 배치 전체 폭으로 한 번씩 돈다. m-column 코어는 그대로 쓴다.
- **B2**: m-column 코어를 블록당 8열에서 16–32열로 넓힌다.
- **B3**: attention을 배치 전체의 query로 한 번에 돈다.
- **B4**: 프로젝션 넷을 IMMA GEMM으로 돈다.
- **C**: prefill 발행을 launcher thread로 옮긴다.

예측은 pp tok/s [derived], A6000, plan (a) 기준이다. G1 칸은 pp512 / pp4096, G8 칸은 pp4096이다.

| 레버 | dm, ms [derived] | G1 lcg | G1 prose-in | G8 호스트 자유, lcg / prose-in | G8 스트리밍 ring 128, lcg / prose-in |
|---|---|---|---|---|---|
| 오늘 | 0 | 123.0 / 174.0 | 177.9 / 245.8 | 235.1 / 384.4 | 458.0 / 507.4 |
| A | −2.3…0 | 123.0–125.6 / 174.0–177.7 | 177.9–183.5 / 245.8–253.1 | 235.1–236.2 / 384.4–388.2 | 458.0–479.9 / 507.4–539.0 |
| B1 | −4.5…−2.6 | 125.9–128.2 / 178.1–181.3 | 184.2–189.1 / 254.1–260.5 | 236.3–237.2 / 388.6–391.5 | 483.3–502.3 / 543.5–573.2 |
| B1+B2 | −8.3…−5.0 | 128.8–133.0 / 182.1–187.9 | 190.4–199.7 / 262.3–274.4 | 237.4–238.9 / 392.3–397.1 | 507.4–541.8 / 581.5–643.3 |
| B1+B3 | −6.4…−3.9 | 127.5–130.5 / 180.3–184.5 | 187.5–194.3 / 258.5–267.3 | 236.9–238.0 / 390.6–394.3 | 496.4–521.5 / 563.4–606.3 |
| B1+B3+B4 | −17.1…−10.8 | 136.3–145.6 / 192.5–205.2 | 207.4–229.5 / 284.3–312.9 | 240.0–243.0 / 400.8–409.8 | 571.1–648.5 / 698.6–877.0 |
| C | 0 | 오늘과 같다 | 오늘과 같다 | 235.1 / 384.4 | 458.0 / 507.4 |

G8 칸의 오늘 행은 호스트가 풀렸다고 둔 값이다. 실제로 그 값에 닿으려면 C, A, B1 중 하나가 있어야 한다. 한 스레드 G8의 기준값은 lcg 190.2, prose-in 277.3이다. ring 8 행과 prose-x 행은 `cardroute-levers.out`에 있다.

AGENTS 「Derive first」 표에서 두 줄을 그대로 옮긴다.
- "launch count only | Δt = ΔN × c_node, predicted | the owning gate | once, only if occupancy moves too"
- "fold a launch's work into a neighbour kernel | Δt = −ΔN × c_node − the removed kernel's time + the work every block of the host grid now repeats or waits on × its blocks; a grid already near ~700 GB/s pays that last term in full | the owning gate | once (qwen3fuse set that term to 0: two folds predicted faster, each measured +0.13 ms)"

A는 첫 줄에 해당한다. B1과 B3는 move/split 줄에 첫 줄이 겹친다. 둘째 줄의 마지막 항은 B1에서 0이다. B1은 일을 이웃 커널에 접지 않고 같은 일을 더 넓은 grid로 옮기기 때문이다.

**A. route graph**
- 기제: 오늘 prefill은 캡처 중이면 오류를 낸다. 배치를 발행하는 도중에 호스트가 serve를 돌기 때문이다. 근거는 `body/prefill.rs:675-680`이다. 그래서 층마다 graph가 둘 필요하다. 하나는 route에서 D2H와 event까지다. 다른 하나는 shadow, upload, join이다.
- 달라지는 값: position, ring slot, compressor와 indexer 상태, fault word다. 배치 파라미터는 이미 배치마다 H2D로 올라간다. q3graph처럼 고정 주소 슬롯에 복사한 뒤 재생하면 된다. 근거는 `arch/qwen3moe/prefill.rs:24-37`이다. 대안은 exllamav3처럼 재생마다 포인터를 고치는 방식이다.
- 모양 수 [derived]: 정렬된 all-full 층만 캡처한다. 40층에 graph 80–120개, 노드 약 84,000개다. CED 부분층과 512의 배수가 아닌 꼬리 배치는 eager로 돈다. eager로 돈 층 수는 load 줄과 stat 줄에 찍는다.
- 로드 비용 [derived]: 캡처는 0.26 s 이상이고, 메모리는 25–85 MB다.
- Δ [derived]: 카드에서 −2.3…0 ms다. 큐가 차 있으면 eager 간격이 이미 최소일 수 있어서 아래 끝이 0이다. 호스트 발행은 0.9–1.3 ms로 준다. 이 값은 decode graph의 노드당 호스트 비용 0.42–0.64 µs에서 잡았다.
- 게이트: 재생과 eager의 비트 동일성, 그리고 `gate-gpu-ds41-prefill`.
- 크기 L. 파일: body/prefill.rs, gpu/src/graph.rs, chain/attn.rs, chain/ffn/batch.rs.
- 위험: 큰 graph 발행 하나가 큐에서 몇 항목을 차지하는지 공개되어 있지 않다. 큐가 바이트 단위라는 추정도 있다. A가 호스트를 푼다는 것은 아직 가설이다.

**B1. 프로젝션 배치 폭**
- 기제: 입력 프로젝션은 층 입력만 읽는다. ds41bulk 보고가 이 점을 적었다. 그래서 hc_pre, norm_quant, joint qkv, q_b, rope_tail은 청크 루프 앞에서 배치 전체로 돈다. 출력 쪽 rope back, quantize, wo_a, wo_b, hc_post는 루프 뒤에서 돈다. 루프 안에는 ring 순서에 묶인 kv append, indexer, source, staged attention, commit만 남는다.
- 바꿀 커널: hc의 토큰 상한이 8이다. 근거는 `gpu-deepseek41/src/hc.rs:81`이다. 이 상한을 올리거나 런처가 8개씩 돈다. gemv와 heads m-col 런처는 열 묶음을 grid로 편다.
- 비트: 비트 동일이다. 근거는 `gpu/src/cores.rs:1118-1121`의 계약 "column c of an m-column span is the single-column span on that column bit for bit"이다. 게이트는 `gate-gpu-ds41-prefill`의 비트 동일성과 `gate-gpu-e2e`다.
- 항목 [derived]: route는 약 463이 된다. shadow까지 더하면 한꺼번에 큐에 있는 항목이 expert arm 949, slot arm 1,064다.
- 확장 [derived]: shadow의 청크별 커널도 같은 방식으로 배치 폭으로 옮길 수 있다. 그러면 한꺼번에 쌓이는 항목이 약 480으로 준다. 큐 깊이가 650이어도 호스트가 풀린다. Q3_K 밖 타입의 m-column 계약은 이번에 읽지 않았으므로 shadow 커널의 비트 계약은 따로 확인해야 한다.
- Δ [derived]: 카드에서 −4.5…−2.6 ms다. 런치 약 840개의 간격과 작은 grid의 wave 꼬리가 사라진다. 예를 들어 wo_b의 640 블록은 SM당 6블록, 84 SM 기준으로 1.27 wave다.
- 스크래치 [derived]: T 512에서 약 256 MB, 128 토큰 부분 블록으로 나누면 약 64 MB다.
- 크기 M. 파일: chain/attn.rs, body/prefill.rs, gpu-deepseek41/src/hc.rs, gpu/src/lib.rs, gpu-deepseek41/src/dense.rs.
- 위험: slot arm의 큐 여유가 1.6–6.5 %뿐이다 [derived]. 확장을 같이 넣거나 P 384 실행으로 Q를 확정해야 한다. 스크래치가 plan (a)의 카드 예산을 먹는다.

**B2. 넓은 열**
- 기제: m-column 코어는 Q3_K 가중치를 한 번 풀어서 블록의 모든 열에 쓴다. 열을 16–32개로 늘리면 풀기 비용이 더 많은 열에 나뉜다.
- 여는 조건: nsys가 프로젝션이 정수 경로에 묶여 있음을 보일 때만 연다.
- 비트: 비트 동일이다. B1과 같은 계약에 기댄다.
- Δ [derived]: B1 위에 −3.8…−2.4 ms다.
- 게이트: occupancy/geometry 부류다. ptxas `-v`의 레지스터 수로 occupancy를 스펙에 적는다. `gate-ptx-spill` 핀을 새로 잡는다.
- 크기 M. 파일: gpu/src/cores.rs, gpu/src/lib.rs.
- 위험: 누산기 레지스터가 늘어 SM당 블록이 6개보다 줄 수 있다. 오늘 q3k gemv는 레지스터 40개를 쓴다.

**B3. attention 배치 폭**
- 기제: attention 커널은 이미 T 토큰 런치를 1 토큰 런치 T개의 블록으로 한 grid에 담는다. 근거는 `gpu-deepseek41/src/attn.rs:47`이다. 배치 전체로 돌리려면 창의 키를 배치 폭 버퍼와 배치 전 ring에서 읽어야 한다. ring은 창이 128이라 뒤 청크가 앞 키를 덮기 때문이다. exllamav3가 바로 이 모양이다. 청크의 kv 행과 갱신 전 ring을 같이 읽고, ring은 attention이 끝난 뒤에 갱신한다.
- 비트: 토큰마다 같은 키를 같은 순서로 읽으면 비트 동일이다. 게이트는 `gate-gpu-ds41-prefill`이다.
- 메모리 [derived]: seg 부분합이 T 512, segment 10에서 약 671 MB다. 64–128 토큰 슬랩으로 나누거나 seg와 merge를 합쳐야 한다.
- Δ [derived]: −1.9…−1.3 ms다.
- 크기 L. 파일: gpu-deepseek41/src/attn.rs, chain/attn.rs, body/prefill.rs.
- 위험: indexer의 query별 top-k와 compressor 목록도 배치 폭에 맞춰야 한다.

**B4. IMMA GEMM**
- 기제: IMMA GEMM 경로를 프로젝션에 쓴다. a5gemm은 A6000, T 512에서 dense 66.4 TOPS와 MoE 47.3 TOPS를 쟀다. 프로젝션 넷은 layer-batch당 64.8 GMAC이고 1.95–2.75 ms에 끝난다 [derived].
- Δ [derived]: B1 위에 −10.6…−6.9 ms다. 잔여를 프로젝션 몫으로 얹은 시간에서 GEMM 시간을 뺀 값이다.
- 비트: 비트 동일이 아니다. GEMM은 128값 블록마다 `acc = fma(d8, d·f32(isum), acc)`를 k가 커지는 순서로 접는다. 근거는 `gpu/src/gemm.rs:40-43`이다. gemv의 합 순서와 다르다. prefill = step bits 불변식이 깨지므로 band와 PIN이 필요하다. hoststream on arm과 같은 부류이고 사용자가 결정할 사항이다.
- 크기 M. 파일: gpu/src/gemm.rs의 런처, chain/attn.rs.
- 위험: decode는 gemv를 쓰므로 prefill 뒤 decode 궤적이 갈린다.

**C. launcher thread**
- 기제: decode용 launcher thread는 이미 있다. 근거는 `gpu/src/model.rs:363`과 `:627`이다. prefill의 route와 shadow 발행을 그 스레드로 옮기면 큐 막힘도 그 스레드로 간다. 2014년 포럼 답글이 같은 모양을 권했다. "a ring buffer that is serviced by a separate blocking thread".
- Δ: G1에서는 0이다. 메인 스레드가 어차피 route를 기다리기 때문이다. G8에서는 pp4096이 lcg 190.2에서 235.1로, prose-in 277.3에서 384.4로 오른다 [derived].
- 크기 M–L. Body의 발행 경로를 다른 스레드로 넘길 수 있게 나눠야 한다.
- 필요 조건: B1이 큐 아래로 내려가면 C는 필요 없다. A가 호스트를 풀어도 마찬가지다. P 384가 Q를 B1 항목 수보다 낮게 보이고 B1 확장도 불가능할 때 C가 대비책이 된다.

추천 순서:
1. 박스 실행 두 개를 먼저 한다. P 384로 큐를 판정하고 nsys로 잔여를 판정한다.
2. B1과 그 shadow 확장. 호스트를 푸는 레버이고 비트 동일이다.
3. G 스케줄러, 즉 hoststream 1단계. 예측 표의 G8 호스트 자유 칸에 해당한다.
4. hoststream 스트리밍. 예측 표의 ring 128 칸에 해당한다.
5. B3, 그리고 nsys가 정수 경로 바운드를 보이면 B2.
6. B4는 사용자의 비트 결정 뒤에 한다.

A와 C는 B1에 지배된다. hoststream이 밀리면 G1 칸이 기준이 되고, 그 순서에서는 B1+B3+B4가 유일한 큰 prefill 이득이다. pp4096 400을 넘는 경로 [derived]:
- prose-in은 G8 호스트 자유에 B1+B3+B4를 더하거나, 스트리밍 ring 8만으로 넘는다.
- lcg는 스트리밍과 B1이 함께 있어야 넘는다.

**4. 큐 판정**

NVIDIA 문서에는 큐 깊이가 없다. Robert_Crovella가 2023-03-15 NVIDIA 포럼에 이렇게 썼다.
> "It's not adjustable, and there are no specifics published about it."

같은 답글의 앞부분이 큐의 범위를 정한다.
> "There is a launch queue for kernels (really, for all asynchronous work issued to the GPU from host code)"

기제는 같은 사람이 2022-06-25에 적었다.
> "The queues provided to support asynchronous work issuance are not of infinite depth. When the queue becomes full, a kernel launch changes from an asynchronous , non-blocking call, to a synchronous blocking call, waiting for a queue slot to open up, before it can put the new kernel launch in the queue and return control to the host thread."

깊이에 대한 관찰은 다음과 같다.
- njuffa, 2023-03-16: "it was determined to be on the order of 1K launches. However, I suspect the depth of the queue is specified in bytes"
- Robert_Crovella, 2023-03-16: "the queue capacity can vary greatly (in terms o number of “items”) depending on the mix of “items” that you put into it, demonstrating a number as low as ~100."
- jan_brothanek, 2014-07-03: "the actual number of kernels in a queue when the others become blocking, is 1024." 같은 스레드에 인용된 그의 다른 문장은 "We tried to split the kernels to streams but the behavior is still the same."이다.

판정: 기제는 확인된다. 깊이는 공개되지 않았고 항목 구성에 따라 변한다. 그래서 깊이는 우리 데이터로 맞춘다.

호스트는 route, D2H, event, shadow를 차례로 발행한 뒤 event를 기다린다. 근거는 `body/prefill.rs:1018-1034`이다. 모형은 다음과 같다.

```
enqueue = (N_r + N_s − Q) × t_c
wait    = (Q − N_s) × t_c
```

두 식을 더하면 route 시간 N_r × t_c가 된다. 그래서 한 arm 안에서는 두 식이 서로 독립이 아니다. 독립 검정은 arm 사이의 차이다. 두 arm은 route가 같고 shadow 항목만 다르며, slot arm이 115개 더 많다 [derived].

| slot − expert, layer-batch당 ms | Δenqueue | Δwait |
|---|---:|---:|
| 실측, A6000 plan (a), sitting 14 | +2.4 | −2.5 |
| 실측, 3090 gate placement, ds41bulk | +2.1 | −2.5 |
| Q 모형 [derived] | +2.1 | −2.1 |
| 고정 꼬리 ≈650, shadow를 세지 않음 [derived] | 0 | 0 |
| 큐 없음, 호스트 비례 [derived] | +0.4 | −0.4 |

arm 차이를 맞추는 것은 Q 모형뿐이다. arm이 바뀌어도 적합한 Q는 거의 그대로다. N_s가 115 바뀌는 동안 A6000의 Q는 17만큼 움직인다.

| 카드, arm | enqueue | wait | Q [derived] |
|---|---:|---:|---:|
| A6000, expert | 18.8 | 11.7 | 1,112 |
| A6000, slot | 21.2 | 9.2 | 1,095 |
| 3090, expert | 18.3 | 11.7 | 1,099–1,138 |
| 3090, slot | 20.4 | 9.2 | 1,081–1,121 |

3090 행의 폭은 t_c를 두 가지로 잡은 차이다. 하나는 wait + enqueue에서, 다른 하나는 card_out에서 잡았다. shadow를 64 청크 전부로 세면 적합값이 약 30 올라간다. 그래서 Q는 1,080–1,170으로 적는다.

P 513은 배치당 약 1,106 항목이다. 발행하는 동안 카드가 약 190개를 소비하므로 큐 점유는 약 916이 되어 Q 아래다 [derived]. 이는 실측 enqueue 3.5 ms가 순수 발행 시간이라는 것과 맞는다.

B1 뒤에 한꺼번에 쌓이는 항목은 949–1,064다. Q 대역보다 아래이고 650보다는 한참 위다. 그래서 Q의 값이 중요하다. Q 모형이 맞으면 B1만으로 호스트가 풀린다. 650이 맞으면 B1의 shadow 확장이나 C가 더 필요하다.

판별 실행은 A6000 plan (a), lcg, P 384, `BLOOMERY_STEP_STATS=1`, CARD_EXPERTS 두 arm이다. 입력값은 N_r ≈1,243, N_s 371과 457, card_out 22.4–23.0 ms다 [derived]. 예측은 layer-batch당 ms [derived]다.

| 가설 | expert enqueue / wait | slot enqueue / wait | slot − expert |
|---|---|---|---|
| Q 모형, 1,080–1,140 | 8.5–9.9 / 12.8–14.2 | 10.1–11.5 / 11.2–12.6 | +1.6 / −1.6 |
| 고정 꼬리 ≈650 | 10.7–11.0 / 11.7–12.0 | 10.7–11.0 / 11.7–12.0 | 0 / 0 |
| 큐 없음 | 5.2 / 17.2–17.8 | 5.4 / 17.0–17.6 | +0.3 / −0.3 |

expert arm의 wait 하나만으로도 세 가설의 예측이 겹치지 않는다. arm 차이가 두 번째 판별자다. sitting 14의 arm 차이는 이미 Q 모형 쪽을 가리킨다.

**5. 참조 설계**

박스 경로의 뿌리는 ik가 `/home/user/ik-idxkey`, llama.cpp가 `/home/user/llama.cpp-v41`, exllamav3가 `/home/user/exllamav3-src`, mistral.rs가 `/home/user/mistral.rs`다.

| 엔진 | 프롬프트 배치의 attention | 프로젝션 | graph | 근거 |
|---|---|---|---|---|
| ik_llama.cpp, 수치 기준 | ubatch 전체에 `ggml_flash_attn_ext`, sink, F32 정밀도를 쓴다. 원본 주석은 "DSV4 uses the generic CPU FA path here for numerical correctness"다. 토큰이 하나면 get_rows, 아니면 top-k 마스크다 | ubatch 전체 matmul이고, wo_a는 3차원 matmul이다 | MUL_MAT_ID의 ne[2]가 1이 아니면 graph를 끈다. batch > 1 규칙은 `if (false && …)`로 꺼져 있다 | `src/graphs/build_deepseek4.cpp:581-591`, `:1351-1360`, `:1425-1436`, `ggml/src/ggml-cuda.cu:4493-4494`, `:4518-4532` |
| llama.cpp V4.1 브랜치 | ubatch 전체 `build_attn_mha`에 top-k 마스크 | ubatch 전체 `build_lora_mm` | 읽은 검사에서는 MUL_MAT_ID sync fallback일 때만 끈다. 실제 prompt 배치에서 graph를 쓰는지는 재지 않았다 | `src/models/deepseek41.cpp:617-629`, `:785-796`, `ggml/src/ggml-cuda/ggml-cuda.cu:1872-1898`, `:2549-2575` |
| exllamav3 | 청크 단위 `dsa_attn`이다. 청크의 kv 행과 갱신 전 ring을 읽고, ring은 attention 뒤에 갱신한다 | 청크 전체에 `_project_qkv`와 `exl3_mgemm` | 배치 graph는 B ≤ 8, S ≤ 16, R ≤ 32에서만 쓴다. 전체 스텝 graph는 seq ≤ 16에서만 쓴다. GraphedParams가 재생마다 포인터를 고친다 | `exllamav3/modules/dsv4.py:1307`, `:1557-1558`, `:1604`, `:1648-1652`, `:1735-1738`, `:1748-1756`, `exllamav3/exllamav3_ext/graph.cuh` |
| mistral.rs | V2/V3 MLA만 있고 CSA와 indexer가 없다. seq 전체 q에 paged, flash, sdpa를 쓴다 | seq 전체 | decode 배치 버킷에만 graph를 쓴다 | `mistralrs-core/src/models/deepseek2.rs:295-300`, `:409-464`, `mistralrs-core/src/pipeline/cuda_graph.rs:402-418` |
| bloomery 오늘 | 8 토큰 청크마다 seg와 merge | 8 토큰 청크마다 m = 8 gemv | prefill은 캡처를 거부한다 | `body/prefill.rs:675-680` |

넷이 갈리는 곳:
- **프로젝션**: 청크마다 프로젝션을 도는 엔진은 우리뿐이다. 나머지 셋은 ubatch, seq, 또는 청크 전체로 돈다. B1이 이 차이를 닫는다.
- **attention**: exllamav3가 우리와 가장 가깝다. 청크 단위, ring, attention 뒤 ring 갱신이 같고 청크 폭만 다르다. B3의 설계 참조로 쓴다.
- **ik의 attention**: ik는 DSV4 attention에 일반 CPU FA 경로를 쓴다고 주석에 적었다. 그래서 GPU attention 설계 참조가 되지 못하고 수치 기준으로만 남는다.
- **mistral.rs**: V4의 CSA와 indexer가 없다. MLA의 seq 전체 prefill 참조일 뿐이다.
- **graph**: 큰 prompt 배치를 graph로 돈다고 확인된 엔진은 없다. exllamav3만 작은 모양을 포인터 패치로 graph에 넣는다. 이것이 A의 대안 설계다.

**6. 범위 밖 개선 후보**

보고만 하고 손대지 않았다. 원장과 트리아지에서 중복을 찾았지만 없었다.

| 위치 | 한 줄 | 크기 |
|---|---|---|
| `crates/gpu/src/lib.rs:2208-2210` | 런처가 매 호출 `prepare_*`를 부른다. 주석은 "Caching per shape is P8's business"라며 미뤄 두었다. 모양별 캐시 후보다 | S |
| `crates/gpu-deepseek41/src/chain/attn.rs:2727-2744` | `q3_k()`와 `vector()`가 런치마다 이름으로 가중치를 찾는다. "The step does no load-time work" 규칙에 걸린다. `Derived`로 한 번 풀어 둘 후보다 | S |
| cuda-core 0.3.1 `src/launch.rs:899-908`, `src/context.rs:555` | `__prepare`가 런치마다 장치 속성 8개와 함수 속성 3개를 묻는다. 장치마다 한 번이면 된다. nvlabs-ledger에 한 줄 올릴 후보다 | 원장 XS, 업스트림 S |
| cuda-macros `src/cuda_module/launchers.rs:269`, `:334`, `:355` | 런치마다 인자 Vec를 할당한다. 고정 배열로 바꿀 업스트림 후보다 | XS |
| `crates/gpu/src/graph.rs:243-244` | fork와 join 이벤트를 기본 플래그로 만들지만 타이밍이 필요 없는 이벤트다. 타이밍을 끈 이벤트로 바꿀 후보다. 효과는 재지 않았다 | XS |
| `crates/gpu-deepseek41/src/chain/glue/batch.rs:215` | engram_rows를 토큰마다 한 번씩 런치한다. 층 1과 14에서 청크당 한 번으로 묶을 후보다 | S |

**7. 실행 모델**

이 라운드는 opus, 즉 Opus 5.5로 돌았다.

산출물:
- `/private/tmp/claude-501/-Users-hckim-repo-bloomery/02d044bc-8d2a-4055-91fa-4ae541a4916f/scratchpad/hs/cardroute-levers.py`와 `cardroute-levers.out`: 레버 행
- `/private/tmp/claude-501/-Users-hckim-repo-bloomery/02d044bc-8d2a-4055-91fa-4ae541a4916f/scratchpad/hs/cardroute-timeline.py`와 `cardroute-timeline.out`: 자원별 합과 묶는 자원
- `/private/tmp/claude-501/-Users-hckim-repo-bloomery/02d044bc-8d2a-4055-91fa-4ae541a4916f/scratchpad/hs/cardroute-queue.py`와 `cardroute-queue.out`: 큐 적합과 P 384 예측
- `/private/tmp/claude-501/-Users-hckim-repo-bloomery/02d044bc-8d2a-4055-91fa-4ae541a4916f/scratchpad/cpg-11.1.1.html`: 가이드 표 원본
- `/private/tmp/claude-501/-Users-hckim-repo-bloomery/02d044bc-8d2a-4055-91fa-4ae541a4916f/scratchpad/forum-33866.json`, `forum-218750.json`, `forum-246254.json`: 포럼 원문

Sources:
- [NVIDIA forum: Kernel launches blocking when 1024 kernels in a queue](https://forums.developer.nvidia.com/t/kernel-launches-blocking-when-1024-kernels-in-a-queue/33866)
- [NVIDIA forum: cudaLaunchKernel Hangs When Interleaving Multiple Empty Kernels and cudaLaunchHostFunc](https://forums.developer.nvidia.com/t/cudalaunchkernel-hangs-when-interleaving-multiple-empty-kernels-and-cudalaunchhostfunc/218750/2)
- [NVIDIA forum: Cuda kernel blocking launch](https://forums.developer.nvidia.com/t/cuda-kernel-blocking-launch/246254)
- [CUDA C Programming Guide 11.1.1, Arithmetic Instructions](https://docs.nvidia.com/cuda/archive/11.1.1/cuda-c-programming-guide/index.html#arithmetic-instructions)
