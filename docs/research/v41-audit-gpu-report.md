조언대로 확인을 끝냈고 더 읽지 않고 보고서를 씁니다.

## B13 감사 ① — V4.1 GPU 스텝 보고서

**요약.** 가장 큰 발견은 코드가 계획과 다르다는 것입니다. engram의 토큰 전용 작업 6커널이 0층 그늘이 아니라 임계 경로에서 돕니다(`body.rs:480`). 그래서 코드 기준 GPU 임계는 12.95 ms가 아니라 약 13.46 ms이고, 이것을 그늘로 옮기면 −0.51 ms/토큰입니다. 나머지 성능 행은 각각 0.3 ms 이하입니다. 품질·게이트 행 중에서는 G2가 노드 수만 세고 어느 커널이 그늘에 있는지는 핀하지 않는 것이 가장 중요합니다. 위 불일치가 게이트에 걸리지 않은 이유가 이것입니다.

### 1. 발견

**질문 1 — 별칭 층 하나의 런치 순서.** 격자는 코드에서 읽었습니다. µs는 b5plan §1.1 값이고 B11b 시점 상수입니다.

| # | 커널 | 격자(코드) | 읽는 바이트(형상) | 묶는 항 | µs |
|---|---|---|---|---|---|
| 1 | `ds41_hc_pre` | 20480/512 = 40 | 211,200 + 81,920 | t_floor, Sinkhorn 꼬리 | 4.1 [가정] |
| 2 | `rms_norm` (별칭 층) | 1 블록 | 40,960 | 1블록 지연 | ≤7.4 [유도] |
| 3 | q_a q8_0 5120→1280 | 160 | 6,963,200 | t_floor(14.0) > B/BW(10.7) | 14.0 |
| 4 | q_a_norm | 1 | 10,240 | 1블록 지연 | 3.3 |
| 5 | q_b q8_0 1280→32768 | 4096 | 44,564,480 | BW | 69.8 |
| 6 | `rope_tail` q | 64·32/256 = 8 | ~16 K | c_node | 1.7 |
| 7 | kv q8_0 5120→512 | 64 | 2,785,280 | t_floor(8.0) > B/BW(4.3) | 7.9 |
| 8 | `kv_norm_rope_append` | 1 | ~4 K | 1블록 지연 | 3.2 |
| 9 | `attn_seg_sel` | 4 × (⌈128/64⌉ + ⌈top_k/64⌉) = 40 | ≤655,360 | 연산·지연 | 10.1 |
| 10·11 | merge, `rope_tail` back | 64, 8 | 1,310,720 | 지연 | 2.7 |
| 12 | wo_a heads 8×(4096→1024) | 1024 | 35,651,584 | BW | 60.8 |
| 13 | wo_b q8_0 8192→5120 | 640 | 44,564,480 | BW | 72.0 |
| 14 | `ds41_hc_post` + 접기 | 20 | ~100 K | 지연 | 2.1 |
| 15 | `norm_quant` | 1 | 20,480 | 1블록 지연 | 7.4 [유도] |
| 16 | `ds41_router` f32 | 48 | 7,864,320 | t_floor f32(13.5, 격자 48) | 16.3 |
| 17 | `ds41_ffn_handoff` | 20 | PCIe 쓰기 20 KB | 지연 | ~1.5 [가정] |
| 18 | go (memop 배치, op 5개) | – | – | c_node + sys barrier | |
| – | 그늘 | hc_pre 40, gate_up 1728, q8_1, q4k_sel, shexp gate_up 288, shexp down 640 | ~50 MB | 호스트 다리 안 | ~100 [b5plan] |
| 19 | wait (memop 배치, op 2개) | – | – | RT | 1.5 [가정] |
| 20 | `ds41_ffn_post` | 20 | hsum 20 KB PCIe + res 80 KB | 지연 | 3.8 |

- **층 합.** 임계 경로는 289.6 µs/층입니다. b5plan은 291.7이고, 차이 −2.1은 combine과 HC_POST가 이미 한 커널로 합쳐진 몫입니다.
  - **BW에 묶인 것:** q_b, wo_a, wo_b 셋이 202.6 µs(70 %)이고 합칠 수 없습니다.
  - **t_floor에 묶인 gemv:** q_a와 kv 둘이 21.9 µs입니다.
  - **나머지 지연·c_node 노드:** 12개와 memop 2개가 65 µs(22 %)입니다.
- **스텝 합.** 40층에 소스·인덱서·engram 층 추가분과 헤드 0.84를 더하면 설계 기준 12.95 ms입니다. 코드는 engram의 6커널을 임계 경로에 두므로 약 13.46 ms입니다(P1).
- **합칠 후보.** 트리아지 ⑤에 이미 있는 것만 남습니다: HC_PRE 갈래, rope 융합, q_a+kv, norm 접기, 여러 블록 argmax.
- **n = 1에서 겹치지 않는 부분.** 층 l의 wait부터 층 l+1의 go까지 전부 호스트 다리의 그늘 밖입니다. 다음 층 입력이 combine 출력에 걸려 있기 때문입니다. 겹치려면 배치가 필요합니다(C3, ktok).

**질문 2 — 합류 핸드셰이크.**
- 층당 memop 배치 둘이 전부입니다. 스텝당 80개이고 이벤트와 호스트 노드는 0입니다.
  - go 배치는 op 5개입니다: sys barrier, Lyr 쓰기, sys barrier, Gen +1, region seq +1.
  - wait 배치는 op 2개입니다: Cnt ≥ 1을 기다린 뒤 −1.
- 호스트는 층마다 풀 작업 하나(`wait_go`, 32스레드가 Gen에서 스핀)를 돌고, 이어서 `experts_into`의 디스패치 둘을 돌고, 끝에 Cnt에 `fetch_add`를 합니다.
- **overlap을 켠 경우** wait는 shexp down 뒤입니다(`ffn.rs:1020`). 끈 경우는 go 바로 뒤입니다(`ffn.rs:954`).
- **wait를 더 늦출 수는 없습니다.** `ffn_post`가 hsum을 읽고, 그 뒤 모든 것이 post 출력을 읽습니다.
- **go를 더 앞당길 수는 없습니다.** router가 먼저 끝나야 합니다. 다만 router와 go 사이의 handoff 런치는 없앨 수 있습니다(P4).
- 그늘의 GPU 일은 약 100 µs이고 호스트 다리는 약 660 µs/층입니다. 그래서 층마다 약 0.56 ms의 GPU 유휴가 있고, P1과 P2가 그 자리를 씁니다.

**성능 행** (Δ 순, 조건은 전부 n=1, A6000 + 호스트 (a), R1)

| path:line | 시선 | 종류 | 기제 | Δ [유도] | 클래스·증명 | 크기 | 간섭 |
|---|---|---|---|---|---|---|---|
| P1 `crates/gpu-deepseek41/src/body.rs:480` | perf | async | 층 루프 앞 단일 스트림에서 `enqueue_engram_kv` 6커널(`engram_wkv` q8_0 6144→25600, 167,116,800 B × 2사이트)이 0층 attention보다 먼저 돈다. 이 작업은 토큰 id에만 의존하는데 0층 호스트 다리(0.85 ms, GPU 몫 67 µs)의 그늘에 넣을 훅이 `ffn.rs`의 `enqueue_shadow`에 없다. 트리아지의 "hook 없음(노드 1개, S)"에는 값이 없고, 12.95는 이것이 그늘에 있다고 가정한 수다 | −0.51 ms/토큰, 깊이 무관 (b5plan §1.2 254.2 µs × 2; B11c 뒤라면 −0.48) | 호스트 디스패치 순서 이동, 비트 동일. G2 노드·종류 수 같음, replay = eager, `--sets` 판정 줄 같음. 시간 A/B 1회 | S | 아래 셋 모두 `body.rs`·`ffn.rs`·`glue.rs`를 건드린다. b5prof nsys가 이 6커널의 임계 위치를 바로 확인할 수 있다. ktok이 같은 파일을 읽는 중 |
| P2 `crates/gpu-deepseek41/src/chain/ffn.rs:1020` (wait 직전) | perf | async·캐시 | 층마다 약 0.56 ms의 GPU 유휴에서 다음 층의 t_floor·지연 사이트 가중치(attn_kv 2.79 MB, hc_fn 0.21 MB, q_a 앞쪽 ~1 MB)를 L2(6 MB)에 미리 읽는다. t_floor는 행을 걷는 적재 사슬의 지연이라 L2 히트면 줄어든다. 사이에 끼는 것은 `ffn_post`(~100 KB)뿐이다 | −0.15…−0.3 ms (kv 8.0→~4, q_a 14→~11 µs/층; 가정: L2 지연 ≈ DRAM 왕복 417–599 ns의 절반, 이 카드에서 재지 않음) | 캐시 효과라 측정이 먼저다. 구성상 비트 동일, 그늘 노드 +40. A/B 1회. cuda-oxide에 `prefetch.global.L2`가 없으면 원장에 한 줄 | S | 없음 |
| P3 `crates/gpu/src/hybrid.rs:1094` `wait_go` | perf | batching(호스트) | go를 기다리는 풀 작업이 끝날 때 조인이 한 번 있고, 이어 gate/up 디스패치가 새로 시작된다. 워커가 Gen을 본 뒤 ids를 직접 읽고 결정적 플랜의 자기 청크로 바로 들어가면 층당 조인 하나와 디스패치 하나가 빠진다. 트리아지의 gate/up→down 조인과는 다른 조인이다 | ≤ −0.28 ms (상한: B2h 디스패치 고정비 6.9 µs × 40) | 호스트 디스패치 경로, 비트 동일. 두 배 탐침으로 값을 먼저 잰다(`HybridStats.straggle_ns`가 일부를 이미 준다). 같은 임대 A/B | M | 감사자 ②와 B2의 호스트 티어 |
| P4 `crates/gpu-deepseek41/src/chain/ffn.rs:930` | perf | fusion | 트리아지 ⑥의 D2H 제거는 절반만 닫혔다. memcpy가 handoff 커널이 됐을 뿐 런치 하나가 임계에 남아 있다. norm이 매핑 페이지에도 x를 쓰고 router 마지막 블록이 ids·weights·seq·sel을 쓰면 런치가 없어진다 | −0.09…−0.15 ms ((c_node 0.85 + 20블록 커널 1.5–3 µs) × 40) | 런치 수만 바뀜, 비트 동일. chain_ffn과 G2(커널 −40) | S–M | 없음 |
| P5 `crates/gpu-deepseek41/src/chain/attn.rs:1263`, `:1269` | perf | fusion | 인덱서 8층에서 attn q_b와 인덱서 q_b(5,570,560 B)가 같은 `q_a_normed`를 두 런치로 읽는다. 소스가 아닌 인덱서 4층은 `rms_norm` 뒤에 따로 q8_1 양자화를 돈다. `norm_quant`가 이미 있다 | −0.02…−0.04 ms (8 × ~2.9 + 4 × 2.95 µs) | 행 연결은 비트 동일. norm_quant 대체는 `fused.rs:125`와 `elem.rs:285`가 같은 규칙일 때만 비트 동일(트리아지 ⑧은 둘을 사본이라 적었다) | XS–S | 없음 |

**품질·게이트 행**

| path:line | 시선 | 종류 | 기제 | Δ | 클래스·증명 | 크기 | 간섭 |
|---|---|---|---|---|---|---|---|
| Q1 `crates/gpu-gates/src/bin/gate_deepseek41_step.rs:757` | quality | gate | G2는 커널과 memop 개수만 센다. 층마다 go와 wait 사이에 어떤 커널이 있는지(그늘 집합)는 핀하지 않는다. 그래서 P1의 배치가 설계에서 벗어난 것을 아무 게이트도 보지 못했다 | – | 게이트 추가: 캡처 순서나 `cuGraphGetEdges`로 층별 그늘 집합을 예측 표와 대조. FAIL-first는 지금 코드(engram 6커널이 그늘 밖) | S | 없음 |
| Q2 `crates/gpu-gates/src/bin/gate_deepseek41_step.rs:774` | quality | gate | replay = eager는 STEP4와 D1에서만 핀된다. D1은 top_k 64라 격자가 segs 3이고, 서빙은 파일 top_k라 segs 10이다. STEP4는 파일 top_k지만 선택이 없어 목록이 항등이다. 선택이 살아 있고 파일 top_k인 상태(D2)의 replay 동일성은 핀이 없다 | – | D2를 structure 루프에 추가 | XS–S | 없음 |
| Q3 justfile, gates 전체 | quality | gate | `BLOOMERY_HYBRID_OVERLAP=0` 모드를 도는 레시피나 게이트가 없다(grep 0건) | – | replay = eager를 overlap 0에서 한 번 | XS–S | 없음 |
| Q4 `crates/gpu-deepseek41/src/chain/glue.rs:13-15` | quality | error(문서) | 모듈 문서는 engram 작업이 "layer 0's host-leg shadow"에 있다고 쓰지만 코드는 그렇지 않다. AGENTS의 "주석은 지금 참인 것" 위반 | – | P1과 함께 고친다 | XS | P1 |
| Q5 `crates/gpu-deepseek41/src/hc.rs:551,565`, `compress.rs:504,521,524,532`, `indexer.rs:462,715,725,735,803`, `engram_gate.rs:371`, `chain/ffn.rs:306,308` | quality | unsafe | R1 보강 위반: "as above", "as in …" 형태의 SAFETY 14곳. `hybrid.rs:642,644,651`은 경계를 적었으므로 제외 | – | ptx-scan 표 동일 | XS × 14 | 없음 |
| Q6 `body.rs:283,298`, `chain/attn.rs:953`, `chain/ffn.rs:769,812,1172`, `hybrid.rs:245,254` | quality | duplication | R2: 층 번호를 조각 인덱스로 바꾸는 `checked_sub(start)…get` 불변식이 7곳 이상에 반복된다. newtype이나 헬퍼 하나로 | – | 이동, 게이트 초록 | S | 없음 |
| Q7 `crates/gpu-deepseek41/src/hc.rs:951`, `hc.rs:1018`, `experts.rs:494` | quality | 죽은 경로 | `ds41_hc_pre_mix`, `ds41_hc_post_streams`, `ds41_moe_combine`를 부르는 곳은 `gate_deepseek41_hc`와 `gate_deepseek41_moe`뿐이다. 체인은 `ds41_ffn_post`와 `hc_post`를 쓴다. 엔진이 돌지 않는 경로를 게이트가 핀한다 | – | 처분은 리드(참조 팔로 둘지 지울지) | S | 없음 |
| Q8 `crates/gpu/src/head.rs:141`, `chain/attn.rs:417`, `chain/attn.rs:646` | quality | error | R10: enqueue 경로의 런타임 `assert_eq!(m, 1)`은 `new`가 이미 m=1로 만든다. `expect`는 `Table::ALL`의 위치라 const로 박을 수 있다. R5: `ctx_max() as usize`인데 `body.rs:804`는 같은 값을 `try_from`으로 받는다 | – | ptx-scan 불변 | XS | 없음 |

### 2. 모델이 기각한 것

- **0·1층 routed expert를 카드에 올리기.** Q5_K는 카드 포맷이 없어 `placement.rs:187`이 두 층을 전부 호스트로 보냅니다. 그러나 (a)는 VRAM에 묶여 있고 라우팅이 균등하면 VRAM 바이트당 줄어드는 호스트 바이트는 어느 층이나 같습니다. 슬롯을 옮겨도 합은 약 0입니다. B12의 뜨거운 집합이 들어오면 다시 봅니다.
- **csa 소스의 인덱스 키 경로를 건너뛰기.** ratio 2라 두 스텝에 한 번은 쓰지 않는 키를 계산합니다. 3층 × 6.7 µs × ½ ≈ 10 µs/토큰이라 조건 노드를 도입할 값이 없습니다.
- **wait 앞에서 post의 comb·r 항을 미리 계산.** 층당 ≤1 µs × 40 = 0.04 ms인데 합 순서가 바뀌어 e2e 재핀이 필요합니다.
- **go를 둘로 나누기(router가 도는 동안 호스트가 x를 양자화).** 절약은 호스트 1–3 µs이고 임계 경로에 memop이 하나 늘어(≥0.85 µs + barrier) 합은 약 0입니다. 호스트가 router를 직접 도는 것은 7.86 MB ÷ 128 GB/s = 61 µs로 GPU의 16.3 µs보다 느립니다.
- **handoff의 20 KB 이중 쓰기를 없애고 router가 페이지를 직접 읽기.** router의 384행 워프가 PCIe로 20 KB씩 읽게 되므로 지금 설계가 맞습니다.
- **q_b, wo_a, wo_b의 명령 수 레버.** 615–680 GB/s 근처라 「모델」 표대로 0입니다.
- **q_b, wo_a, wo_b를 더 적은 비트로 재양자화.** 층당 124.8 MB, 토큰당 4.99 GB, 650 GB/s에서 7.7 ms로 GPU의 가장 큰 항입니다. 그러나 같은 파일로 ik와 KLD 패리티를 맞추는 계약을 깨므로 코드 레버가 아닙니다.
- **pageable 메모리에서 이미지 H2D(24.2 KB/스텝).** ≤10 µs/토큰입니다.
- **n = 1에서 층 사이 겹침.** 다음 층 입력이 combine 출력에 걸려 있습니다. HC_PRE의 mix는 y에 대해 아핀이지만 q8_1 양자화는 선형이 아닙니다. 겹치려면 배치가 필요합니다(C3, ktok).

### 3. 원인 축소 항목

1. **모델이 실측보다 0.8 ms 느리다.** 유도는 B11c 뒤 R1 39.7에 P1의 0.51을 더한 40.2 ms이고, 실측은 39.4 ms @ n=1, 깊이 4096, A6000입니다. 가를 측정은 b5prof의 nsys입니다. 층별 wait 노드 길이와 `HybridStats.leg_ns/served`로 호스트 다리의 실효 GB/s(3.418 GB ÷ Σleg)를 구합니다. 가정은 128.8이고, 137이면 −1.6 ms로 차이가 전부 설명됩니다. 나머지는 JOIN 0.8 가정입니다.
2. **합류 왕복의 분해.** 지금은 V2-Lite 실측 18–22 µs/층만 있습니다. nsys에서 go·wait 노드와 `ffn_post` 길이를 호스트 스탬프와 맞춰 봅니다. 크게 나오면 레버가 셋입니다.
   - 층 번호를 handoff 이미지에 넣으면 sys barrier가 하나로 줍니다.
   - memop wait 대신 `ffn_post`가 Cnt에서 직접 스핀하면 노드가 하나 줄고 폴링 지연이 바뀝니다.
   - 호스트가 hsum과 카운터를 BAR1(A6000 64 GiB, DMA-BUF 속성 152 = 1 측정)으로 VRAM에 쓰면 폴링과 읽기가 로컬이 됩니다.
3. **깊이 기울기가 거꾸로다.** 깊이 6은 40.32 ms, 4096은 39.37 ms로 얕은 쪽이 0.95 ms 느립니다[tok/s에서 유도]. 팔마다 `HybridStats.host_slots/served`(라우팅이 정하는 호스트 바이트)와 스텝 스레드의 majflt·minflt(프로세스 시작 직후의 콜드 engram·embd 행)로 가릅니다. 두 번째 텔레메트리는 트리아지 b5gen ⑪에 따르면 아직 없습니다.
4. **1블록 커널.** 층마다 임계 경로에 5–6개(norm·norm_quant·q_a_norm·kv_append)이고 헤드에도 있습니다. 시간은 b5plan의 [가정/유도]뿐입니다(norm+Q8 7.4는 외삽). b5prof의 커널별 표에서 실제로 7 µs 근처로 나오면, 블록마다 20 KB 벡터의 Σx²를 다시 계산하는 여러 블록 형태가 국소 레버가 됩니다. 이것은 norm 접기 카드(−0.4…−0.6)와 같은 항이므로 둘 중 하나만 고릅니다.
5. **스텝 사이 GPU 유휴.** argmax 끝부터 다음 embed 시작까지의 간격을 nsys에서 읽고 engram 조회 0.31 ms를 뺍니다. 남는 것(`plan_into`, `image.build`, H2D, 런치)이 대상입니다. 헤드가 도는 약 0.85 ms 동안 호스트가 놀고 있으므로 위치에만 의존하는 일을 그 창으로 옮길 수 있는지 봅니다.

### 4. 읽지 못한 것

- **커널 본문.** `attn.rs`의 세그먼트 매크로 본문, `compress.rs`, `index_key.rs`, `indexer.rs` 점수·top-k, `engram_gate.rs`, `experts.rs`, `q8f32.rs`, `fused.rs`, `elem.rs`, `model/kernels.rs`는 런치 사이트와 grep만 봤습니다. `params.rs`도 grep만 봤습니다.
- **게이트.** chain_attn, chain_ffn, chain_glue, op 게이트 본문은 grep만 했습니다. step 게이트는 헤더와 structure 부분만 읽었습니다. `ptx.rs`는 함수 목록만 봤습니다.
- **컴파일 산출물.** ptx-scan 표와 SASS는 받지 않았으므로 레지스터와 점유율은 트리아지 값을 썼습니다.

### 5. 모델

opus로 스폰됐고 Opus 5.5(claude-opus-5-5)로 돌았습니다.
