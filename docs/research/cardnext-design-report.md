# 카드 쪽 다음 레버 설계(G 스케줄러, grouped shadow 타일) — 원문

2026-09-26. opus 서브에이전트 설계 라운드 `cardnext`의 보고 원문이다(스펙 `specs/wave-m6/spec-cardnext.md`, 박스 없음; 흐름 모형 `tools/flow/ds41_prefill.py`를 Mac에서 돌리고 모형에 없는 스케줄·커널 모양은 스크래치 확장으로 더함 — 확장은 G = 1에서 원 모형과 소수점까지 같음). [유도] 수치는 라운드의 추론이다. 요지: grouped shadow는 잔류 부족이 아니다 — gate·up은 슬롯마다 가중치를 DRAM에서 다시 읽고(R), down은 캐시에 상주한 채 지연에 묶였다. 권고 순서는 T(전문가 × 행 타일 × 8열 타일, 비트 그대로, prose pp512 +31 %) 먼저, 그다음 G(층을 넘는 발행, lcg pp4096 +26 %). 처분은 `docs/plan-triage.md`에 있다.

## 보고 (`cardnext`, 워크트리 `/Users/hckim/repo/bloomery-cardnext`, base `74fe84c`)

## cardnext 보고 — G 스케줄러와 grouped shadow 설계

박스, 저장소 파일, git 상태는 건드리지 않았습니다. 수치는 Mac에서 흐름 모형(`tools/flow/ds41_prefill.py`, 74fe84c, 수정 없음)을 돌려 얻었고, 모형에 없는 스케줄과 커널 모양은 스크래치 확장으로 더했습니다. 파일은 `/private/tmp/claude-501/-Users-hckim-repo-bloomery/02d044bc-8d2a-4055-91fa-4ae541a4916f/scratchpad/cardnext/`에 있습니다.
- `flowx.py`: 확장 코드
- `table2.txt`: pp 표
- `timeline.txt`: 층-배치당 자원 시간선
- `launcher.txt`: launcher 스레드 상한

확장은 두 가지로 검증했습니다.
- 확장 경로를 G = 1로 돌리면 스톡 모형과 소수점까지 같습니다(148.15 / 166.2).
- 스톡 B1, B1+G 행은 `--predict` 출력(148.1 / 208.6 / 166.2 / 237.0, 252.4 / 235.7)을 그대로 재현합니다.

**먼저 알릴 것: 스펙 전제 하나가 코드·트레이스와 맞지 않습니다.** 스펙은 grouped shadow가 "프로젝션처럼 잔류가 모자라 지연에 묶였다"고 봤습니다. 실제 그리드는 GU 40웨이브, down 107웨이브로 카드를 꽉 채웁니다. GU는 슬롯마다 가중치를 DRAM에서 다시 읽는 모양과 맞고(R), down은 캐시에 상주한 채 지연에 묶여 있습니다(2절).

---

### 1. G 스케줄러

**1.1 종이 분해의 결론: 이득은 G의 폭이 아니라 "층을 넘는 발행(wrap)"에서 나옵니다.**
- 오늘 G = 1은 route(i) → shadow(i) → wait → union(i) → post(i)의 합입니다.
- 제안은 그룹의 (층 l, 배치 j) 항목을 층 우선 순서로 한 줄로 세우고, 항목 i마다 이 순서로 돌리는 것입니다:
  - [S(i), R(i+1)] 발행
  - wait routed[s(i)]
  - union(i)
  - [U(i) upload, J(i) join, tap] 발행
- wrap은 배치 g−1, 층 l에서 배치 0, 층 l+1로 넘어가는 경계에서도 R을 미리 발행한다는 뜻입니다.
- 이 순서가 맞는 조건은 R(l+1, j)가 J(l, j) 뒤에 발행되는 것입니다. J(l, j)는 g항목 앞의 끝에서 발행되므로 **g ≥ 2**면 성립합니다.
- g = 1이면 오늘 순서로 되돌아갑니다(R(i+1)을 post(i) 뒤에 발행).
- 스톡 모형의 G는 층마다 route(0, l)을 노출합니다(wait 1.8 ms/lb). wrap은 이것을 그룹의 층 0에서 한 번으로 줄입니다.

**1.2 버퍼는 네 부류입니다.** hoststream 스케치는 `cap > UNION_MAX_COLS`를 풀어 G×512 용량을 잡자고 했습니다. 엔진 스트림이 하나라 FIFO가 대부분을 지켜 주므로 그렇게 할 필요가 없습니다.

| 부류 | 버퍼 | 처리 | 바이트/여분 배치 [derived] |
|---|---|---|---|
| (i) 층의 위치 간 상태 | ring, compressor 풀, index key, 가시 개수 | 그대로 둡니다. 층 안의 배치 순서가 보존되므로 (l, j)가 (l, j+1)보다 FIFO에서 앞섭니다 | 0 |
| (ii) 배치 잔차 | `hc` 쌍 `prefill.rs:332`(2×512×4×5120×4 B), `folds` 쌍 `:333` | 배치마다 둡니다(G=2는 쌍 그대로, G≥4면 G+1 풀) | 83.9 + 21.0 MB |
| | `lists` `:335`(65 청크 × 쓰는 층 8 × 8 × top_k 512 × 4 B). 쓰는 층이 쓰고 뒤층이 읽습니다(`body.rs:1618`) | 배치마다 | 8.5 MB |
| | FfnBatch `hc` 혼합(512×24×4 B). engram 층이 읽고(`prefill.rs:1133`) head도 읽습니다(`:1305`) | 배치마다. 공유하면 14층에서 비트가 어긋납니다 | 0.05 MB |
| | `params` `:323`, AttnChain 청크 word 행(`with_rows(CHUNKS_MAX)`), AttnBatch rope `tables`, `taps` | 배치마다 | ≤ 12.6 + ≤ 12.6 + ~0.4 MB |
| (iii) 항목의 임시값 | FfnBatch `x, ids, weights, sel, probs, acc, shexp, hsum, h_all, act_h_all, down_all, order, start`, AttnBatch 서브블록 스크래치, `staging`, glue | 한 벌로 둡니다(근거는 아래) | 0 |
| (iv) 호스트 교환 | pinned `host_x/ids/w/sum`(`batch.rs:836-839`), `routed`(`:840`) | served 항목 순번의 홀짝으로 두 벌 | pinned +21.0 MB, event +1 |

(iii)을 한 벌로 둘 수 있는 이유는 R(i+1)이 쓰는 값(x, probs, ids, weights, sel)을 읽는 쪽이 모두 FIFO에서 먼저 돌기 때문입니다.
- S(i)는 같은 발행 묶음 안에서 R(i+1)보다 앞섭니다.
- J(i)는 acc, hsum, shexp, hc 혼합을 읽는데, R(i+1)은 이것들을 쓰지 않습니다. 이들을 쓰는 S(i+1)과 U(i+1)은 J(i) 뒤입니다.

(iv)를 두 벌로 둬야 하는 이유는 호스트가 FIFO 밖에 있기 때문입니다.
- R(i+1)의 D2H가 union(i)가 읽는 중인 `host_x`를 덮습니다.
- union(i+1)이 아직 U(i)가 올리지 않은 `host_sum`을 덮습니다. ev(i+1)은 FIFO에서 U(i)보다 앞입니다.
- `routed`가 하나면 serve(i)가 R(i+1)의 기록까지 기다립니다. 비트는 같고 겹침만 사라지는 **조용한 성능 실패**라서, 교환 세트가 자기 event를 타입으로 소유하게 합니다.

**1.3 메모리.** G=2는 잔차 한 벌을 더 씁니다. 114–139 MB [derived]로, B1 뒤 남은 966,655,756 B 안에 듭니다. 한 벌에 들어가는 것:
- 고정분 113.4 MB: hc·folds 쌍 104.9, lists 8.5, 혼합 0.05
- 상한분 ≤ 25.6 MB: params, word 행, tables

G=8(hoststream 때)은 쌍 형식으로 0.79–0.97 GB라 한계에 닿습니다. 풀 형식은 0.43–0.61 GB이고, 배치 계획의 한 항으로 올려야 합니다. 어느 경우든 할당이 모자라면 `AttnChain::batch`처럼 이름 붙은 거부를 냅니다.

**1.4 겹침을 막는 것(모두 이름 있는 규칙)**
1. g = 1인 그룹은 오늘 순서로 돕니다. P ≤ 512는 한 배치라 G 이득이 0입니다. 반쪽 분할은 ds41overlap이 +19 ms/층으로 기각했습니다.
   - 규칙: 홀수 개 배치의 꼬리 한 개는 앞 쌍에 붙여 g=3으로 만듭니다. 잔차가 한 벌 더 들고, 풀 형식이면 +61–87 MB입니다.
2. CED로 T = 0인 항목은 숨을 union이 없습니다. 그 항목의 카드 일(잠재 청크)만 FIFO에 흐릅니다.
3. 그룹 경계에서는 `plan_batch`의 동기 H2D(`prefill.rs:1041-1043`)가 스트림을 비웁니다. 그룹마다 route 한 번과 prologue가 노출되고, P4096 G2에서 약 0.6 %입니다 [derived].
4. fault 읽기(`:949`), `batch_features` D2H, head는 그룹 끝으로 옮깁니다.
   - **의미 변화 하나:** 여러 배치의 fault가 오늘은 첫 배치 뒤에 멈춰 그 배치의 (층, site)를 냅니다. G에서는 그룹 전체의 atomic-min을 냅니다. 상태 비트는 그대로이고 오류 내용만 달라질 수 있습니다.
   - FAIL-first: 2배치 중 배치 1에 fault를 심어, 이름 붙은 오류와 poison이 나오는지 확인합니다.
5. 그룹 도중 오류(union의 호스트 거부 등): 스트림을 synchronize한 뒤(진행 중인 DMA가 없게) 오늘의 `take_back`으로 갑니다. `holds.wrote`는 그룹의 모든 위치를 그룹 첫 런치 전에 표시합니다. 표시만 하고 쓰지 않은 행을 복원해도 같은 값이라 안전합니다. `take_host_refusal`은 디코드 스텝 경로이고, 배치 경로는 `serve`가 오류를 직접 돌려줍니다(`prefill.rs:1249`).
6. 캡처는 계속 거부합니다(`:877`, eager 그대로).

**1.5 비트 논증(prompt = decode 스텝, 비트 그대로)**
- 발행 순서를 바꿔도 각 런치가 읽는 값은 같습니다.
  - (i)은 층 l의 attention만 위치 순서대로 쓰고, 층 안의 배치 순서가 보존됩니다.
  - (ii)는 배치 전용입니다.
  - (iii)은 위 FIFO 논증으로 안전합니다.
  - union은 열마다 독립이고 상태가 없습니다(gate-union이 고정합니다).
  - fork 스트림은 attention 호출 안에서 합류합니다.
- **비트가 움직일 수 있는 곳**은 (ii)나 (iv)를 한 벌로 잘못 분류한 경우뿐입니다. FAIL-first 넷이 이것을 게이트에 걸리게 합니다.
  - F1 혼합 공유 → 14층 빨강
  - F2 `host_x` 한 벌 → 빨강
  - F3 g=1에서 wrap → 빨강
  - F4 `routed` 공유 → 비트는 녹색, `wait_lb` ≈ route로 성능 실패(타입으로 표현할 수 없게 만듭니다)
- `gate_deepseek41_prefill`의 observer는 seam을 층 우선 순서로 받게 됩니다. 배치 우선 순서를 가정하는지 확인하고, G=1과 G=2 두 팔을 둡니다.

**1.6 예측** pp tok/s [derived]. A6000, plan (a), 뜨거운 목록 384, CED on, B1 위.

| | lcg 512 | lcg 4096 | prose-in 512 | prose-in 4096 |
|---|---:|---:|---:|---:|
| B1 (G1) | 148.1 | 208.6 | 166.2 | 237.0 |
| 스톡 G (층 우선, G≤8, wrap 없음) | 148.1 | 252.4 | 166.2 | 235.7 |
| **G wrap, G=2** | 148.1 | **258.8** [253.8–261.1] | 166.2 | 234.1 [229.7–248.9] |
| G wrap, G≤8 | 148.1 | 257.6 | 166.2 | 235.1 |

- G=2와 G≤8의 차이는 0.5 % 안입니다. 모형이 흔들리는 폭이지 기제가 아닙니다.
- lcg P4096에서 G 뒤 벽시계는 71.9 ms/lb이고 가장 바쁜 자원(host 66.4)의 1.08배입니다. union 합이 벽시계의 88 %입니다.
- prose는 카드가 묶어서 G가 0입니다(card 76.9 ≥ 벽시계 78.6).
- launcher 스레드의 상한은 t_issue를 0으로 둔 경우로 +2.9 %라 열지 않습니다.
- 트레이스의 층별 shadow 편차(10–43 ms)를 넣으면, lcg G 뒤 34층 중 6층(12, 14, 16, 17, 18, 21)이 카드 병목입니다. 균등 모형보다 −1.9 %입니다 [derived, `an-pp512-box.txt` 층별 열].
- **밴드 주의:** 제 밴드는 full_res_lat, t_issue, gap_act, union 앵커 셋, 타일 대안만 흔들었습니다. 스톡 `band_of`처럼 모든 상수를 훑은 자가 아닙니다.

---

### 2. grouped shadow

**2.1 오늘의 기하** (ptxas는 ds41proj `final-ds41-full.txt`, 트레이스는 nsyspp sqlite L2–39 평균)

| 커널 | 그리드 × 블록 | regs, blk/SM | 웨이브 | 워프의 일 |
|---|---|---|---|---|
| `ds41_expert_gate_up_grouped` (`batch.rs:329`) | n_card × ⌈2304/8⌉ = 20,160–20,448 × 256 | 40, 6 | 40.0 | 한 행(gate+up 4,400 B)이 전문가의 슬롯을 **한 개씩 차례로** 걷습니다. 슬롯마다 1열 반복 2×10회(SASS 1열 루프 104명령, `q3k_gemv` 0x2d0–0x940) |
| `q4k_gemv_grouped` (`q4k_sel.rs:205`) | n_card × 640 = 44,800–45,440 × 256 | 48, 5 | 106.7 | 한 행(1,296 B), 슬롯마다 3반복 |

- 트레이스(A6000, afe86d5, lcg P512)는 층별 GU 1.74–23.5 ms, down 0.92–11.9 ms를 보입니다. n_card는 70–71로 고정인데 16배가 벌어집니다.
- GU:down 비는 층마다 2.0 ± 0.1이고, 평균은 GU 10.22, down 5.01 ms/lb입니다.
- 웨이브는 슬롯이 아니라 n_card로 정해지므로 lcg와 prose가 같습니다. 슬롯은 각 워프의 걷기를 늘립니다. prose의 가장 뜨거운 전문가는 λ가 87–237이라 워프 하나가 슬롯 237개를 차례로 걷습니다.

**2.2 유닛별 수요 대 스텝** (슬롯당, lcg 평균, S = 694 [lcg 모형])

| | 스텝 | DRAM | issue | 판독 |
|---|---:|---:|---:|---|
| GU | 14.7 µs | 재독이면 10.14 MB / 674 GB/s = 15.0 µs (≈ 100 %) | 2304 × 2,150 명령 / (336 × 2.03 GHz) = 7.3 µs (49 %) | **R-정합: DRAM 묶임** |
| down | 7.2 µs | 전문가당 1회 ≈ 1.0 µs (14 %) | I4(1) = 120 가정 시 2.9 µs (41 %) | 지연에 묶임(스텝에 닿은 유닛 없음) |

R을 가리키는 근거는 비율 검정입니다. 절대값이 아닌 이유는 트레이스의 층별 편차가 균등 S를 부정하기 때문입니다.
- (a) GU:down = 2.0이 층마다 일정하므로 둘 다 같은 층별 개수(S_l)에 비례합니다.
- (b) down이 슬롯마다 재독한다면 860–920 GB/s가 필요한데, 이는 피크 768보다 큽니다. 그래서 down은 캐시에 상주합니다. SM당 40워프 × 1,296 B = 52 KB로 L1에 들어가고, 84 SM 합 4.35 MB는 L2 6 MB 안입니다.
- (c) GU는 SM당 48 × 4,400 B = 211 KB로 L1 128 KB보다 크고, 합 17.7 MB는 L2 6 MB보다 큽니다. 순환 접근이라 재독이 DRAM으로 갑니다.
- (d) 그러면 GU의 슬롯당 시간은 바이트 / BW입니다.
- 정직하게 적으면, "둘 다 issue 묶임"(issue 수요 비 2.09)도 (a)와는 맞습니다. 판정은 T 착륙 런의 `dram__bytes_read`가 합니다. 예측은 R이면 GU ≈ S_l × 10.14 MB, 캐시 가설이면 ≈ n_card × 10.14 MB입니다.
- 스펙의 "잔류 부족, 지연 묶임"과 flowcal의 "DRAM 바닥의 8배"(바닥을 전문가당 1회로 셈)는 GU에 대해 정정이 필요합니다.

**2.3 제안: T = (전문가, 행 타일, 8열 타일) 작업 항목** (B1 ColGroups의 전문가판)
- **GT (gate·up):**
  - 블록 = (e, 행 타일 ρ, 열 타일 t). 워프 = 한 행이고, 그 전문가 run의 ≤ 8슬롯을 m-column 코어 `q3k_row_dot`(`cores.rs:1074`)로 한 번에 걷습니다.
  - gate와 up을 한 커널에서 돕니다. 모양은 `ds41_shexp_gate_up_q3k_mcol`(`dense.rs:673`, 47 regs, 5/SM)입니다.
  - 입력은 새 gather가 슬롯 순서로 복사한 q8_1 열입니다. 열당 5,280 B의 바이트 복사로, 슬롯 3,072개면 16.2 MB, 0.01–0.03 ms입니다.
  - 출력은 order 위치 j에 씁니다.
- **DT (down):** `q4k_gemv_mcol`의 m-column 코어(48 regs, 5/SM)로 run의 연속 열(start[e]+8t…)을 걷습니다. 입력의 q8_1은 order 위치 열로 양자화합니다(`q8_1_quant_block`은 위치와 무관합니다). 결과는 `order[j]`로 `down_all[slot]`에 흩뿌려서 `ds41_ffn_card_acc`를 바꾸지 않습니다.
- **그리드:** 상주형(84 × 5 × k 블록)으로 버킷 커널이 낸 타일 표를 stride로 돕니다. 타일 수는 기기의 값입니다.
  - 항목 순서를 (e, ρ, t)로 두면 동시에 도는 블록이 같은 가중치 행을 L2에서 나눠 씁니다.
  - 항목 수: GT는 lcg 33,955(81웨이브), prose 83,779(200웨이브). DT는 75,456 / 186,176입니다.
- **레지스터 제약: ≤ 48 regs(5 blk/SM).** 49–56이면 4/SM으로 떨어지고 e의 아래끝이 낙관이 됩니다. ptxas `-v`와 새 `ptx-shapes.tsv` 행으로 확인합니다. 넘으면 gate와 up을 두 런치로 나눕니다(각 39 regs급, 6/SM). swiglu는 같은 f32 입력이라 비트가 같습니다.

**2.4 예측** T 512, L2–39 평균, ms/lb [derived]. e는 새 커널의 issue 효율입니다. 0.45는 오늘 down, 0.70은 q_b의 issue-active 68.8 %에서 잡았습니다.

| | 오늘(슬롯당 23.51 µs) | 새 합계 | GT = max(issue@100%/e, DRAM) | DT |
|---|---:|---:|---|---|
| lcg | 16.3 (트레이스 15.2) | **5.3** [4.1–6.4] | 1.92/e, DRAM 1.06–1.77 → 2.7–4.3 | 0.78–1.17/e → 1.4–2.2 |
| prose-in | 48.9 (타일당 해석 39.7) | **14.6** [11.5–17.8] | 5.20/e, DRAM 1.05–4.38 → 7.4–11.5 | 2.2–3.4/e → 4.0–6.2 |

- 층-배치당 shadow는 lcg 22.1 → 12.3, prose 53.0 → 21.1입니다.
- **비트 계약:**
  - (슬롯, 행) 값마다 m-column 코어의 열 c가 1열 호출과 비트가 같습니다. Q3_K는 `cores.rs:1062`, `:1118-1121`, Q4_K는 `cores.rs:640-645`(q4k_acc의 split 반올림)입니다. 오늘 grouped 커널이 그 1열 호출이고, 그것이 슬롯별 스텝 커널과 비트가 같습니다.
  - 워프 트리 축약은 열마다 따로 돌고, `swiglu_clamp`는 같은 함수입니다.
  - card_acc는 토큰의 여섯 슬롯을 j 순서로 합하는데, 이 순서가 그대로입니다. 한 카드 전문가의 슬롯도 order 안에서 id 순입니다.
  - K = 2304(Q4_K)와 K = 5120(Q3_K)의 mcol은 이미 공유 전문가 prefill 경로에서 prompt = steps 게이트로 고정돼 있습니다.
- 대안은 둘이지만 지금은 권하지 않습니다.
  - IMMA GEMM(a5gemm MoE 47.3 TOPS)이면 prose가 ~3 ms입니다. 그러나 비트가 바뀌어 B4처럼 사용자 결정 사항입니다.
  - 16열 타일은 prose GT를 −24 % 줄이지만 레지스터가 늘어 B2 성격입니다.

---

### 3. 순서, 결합 예측, 결정

| | lcg 512 | lcg 4096 | prose 512 | prose 4096 |
|---|---:|---:|---:|---:|
| B1 | 148.1 | 208.6 | 166.2 | 237.0 |
| B1+T | 148.1 (+0) | 208.6 (+0) | **218.0** [212.0–239.4] (+31 %) | **300.5** [292.6–328.6] (+27 %) |
| B1+T+G2 wrap | 148.1 | **262.7** [257.6–265.0] | 218.0 | **376.9** [357.7–403.8] |

**G의 이득은 T에 따라 달라집니다.**
- lcg: T 전 +24.1 %, T 뒤 +25.9 %로 거의 같습니다. 카드는 G 뒤 busy 36.8 ms/lb로 host 66.4의 그늘 아래라, T는 트레이스 편차 몫인 +1.9 %만 줍니다.
- prose: T 전 −1.2 %(카드 76.9가 묶음), T 뒤 **+25.4 %**입니다(card 45.5 대 host 39.1, 벽시계 49.4 = 1.09× card).
- T+G 뒤 prose P4096도 여전히 카드가 묶습니다. 다음 항은 route 23.8과 shadow의 청크별 부분 7.3 ms/lb입니다.

**권고 순서: T 먼저, 그다음 G.**
- T는 크기 M이고 비트가 같으며 메모리가 들지 않습니다. 유일한 P512 레버(prose +31 %)이고 G의 prose 몫을 엽니다.
- G는 크기 L입니다.
- 둘 다 `chain/ffn/batch.rs`를 만지므로 직렬로 둡니다.
- P512 lcg는 둘 중 어느 것도 움직이지 못합니다(route + union 직렬). 그 칸은 h3tile-b의 몫입니다.

**결정 (각 착륙 A/B, 효과가 자 밖이라 2바퀴)**

T의 A/B(`BLOOMERY_CARD_EXPERTS=tile` 대 `expert`)는 prose 512 +28…+44 %, 4096 +23…+39 %를 예측합니다.
- 대역 안: G 라운드로 갑니다.
- 아래: `card_in`과 ncu의 GT issue-active를 읽습니다. e < 0.45면 레지스터·점유가 원인이므로 gate와 up을 나눕니다.
- 위: prose의 오늘 값에 꼬리가 있었다는 뜻입니다. G의 prose 몫을 다시 유도합니다.

G의 A/B(`BLOOMERY_PREFILL_GROUP=1` 대 `2`)는 lcg 4096 +23…+27 %를 예측합니다.
- 대역 안: lcg는 host union이 88 %이므로 hoststream이나 h3tile-b로 갑니다. prose는 route(B3/B4)와 청크별 shadow로 갑니다.
- 아래: `wait_lb`가 route 수준이면 event 공유(F4)입니다. `enqueue_lb` > 8이면 큐 막힘이나 t_issue 문제입니다.
- 위: 겹침 아래서 union이 더 빠른 것입니다. T sweep 뒤 k(T)를 다시 잡습니다.

---

### 4. 구현 라운드

**① `cardtile` (M, 먼저)**
- 파일:
  - `crates/gpu-deepseek41/src/chain/ffn/batch.rs`: 커널 모듈(버킷이 타일 표를 냄, gather, `ds41_expert_gate_up_tiles`), `enqueue_grouped_experts` `:1495`, FfnBatch의 order 위치 버퍼 16.2 MB
  - `crates/gpu/src/q4k_sel.rs`: `q4k_gemv_tiles`와 order 위치 양자화기 변형(`fused.rs`, `quant.rs`는 ee 소유라 건드리지 않습니다)
  - `tools/ref/ptx-shapes.tsv`
  - `gate_q4k_sel` 및 grouped 단위 케이스: run 0, 1, 7, 8, 9, 16, 17, 237
  - `gate_deepseek41_prefill`: tile, expert, slot 세 팔
- 레버: `BLOOMERY_CARD_EXPERTS=tile|expert|slot`, 기본 tile. expert는 한 파동 동안 A/B 팔로 두고 지웁니다(두 번째 경로는 썩습니다).
- FAIL-first:
  - 타일 경계 off-by-one → 빨강
  - scatter를 잘못된 슬롯에 씀 → 빨강
  - run이 n_slots를 넘음 → 이름 붙은 ExpertId fault

**② `prefillgroup` (L, T 뒤)**
- 파일:
  - `body/prefill.rs`: feed 그룹화, prologue와 그룹 체인 분리, 슬롯별 잔차, 통계 `group=`, 층 우선 seam
  - `chain/ffn/batch.rs`: `Exchange` 두 벌과 소유 event, 혼합 슬롯별
  - `chain/attn/batch.rs`: tables 슬롯별
  - `chain/attn.rs`: `with_rows(CHUNKS_MAX × G)`
  - `generate_ds41.rs`: `load group= group_bytes=`
  - 게이트: G1/G2 팔, fault 케이스
- 레버: `BLOOMERY_PREFILL_GROUP`, 적재 때 한 번 읽음. 기본 2이고 1이 오늘 순서입니다. 잘못된 값이나 예산 초과는 이름으로 거부합니다.
- **`body.rs`는 건드리지 않습니다**(hostreset이 편집 중). `Batch` 타입 이름을 유지하면 필요 없습니다.
- ee 파일(`hybrid.rs`, `fault.rs`, `moe.rs`)도 필요 없습니다. `serve_batch`의 시그니처는 그대로입니다.

---

### 5. 박스 측정 (한 묶음, 합 ~5분 + 착륙 런)
1. **G의 항 t_issue(A6000):** 추가 시간 0. B1 착륙 시팅의 stat 줄(`enqueue_lb`, `entries_*`, `card_proj_lb`)을 읽습니다. 예측은 enqueue 4.5 ms/lb(P512 lcg), `card_proj` 14.8 [9.2–16.6]입니다. 결정은 enqueue ≤ 5.5면 설계대로, > 8이면 G 라운드에 launcher를 넣는 것입니다.
2. **T의 항, prose의 오늘 shadow:** ~5분. 카드는 아래와 같고, 러너는 prose 프롬프트 팔에 `BLOOMERY_STEP_STATS=1`(B1 바이너리)입니다.
   ```
   kind: calibration
   question: is the prose prompt's shadow the per-slot reading (R) that sizes the tile lever
   term: card_in_lb ms @ prose prompt P 512, B1 binary, A6000 plan (a), hot list 384, CED on, CARD_EXPERTS expert
   unit: ms
   predict: 45..53
   condition: one prompt under BLOOMERY_STEP_STATS=1 (card event pairs), lease held, no other card work
   decide-in: build cardtile as designed (prose +27..+31 % at G1)
   decide-below: below the union 34.9 the shadow already hides at G1: G first, then cardtile
   decide-above: the hot experts' serial walks add a tail the per-slot model misses: cardtile first, re-derive its gain upward
   reason: the flow model's prose shadow is lcg's per-slot cost transferred, never traced
   ```
3. 오늘 커널의 ncu 판별은 **착륙 검증으로 미룹니다.** 사전 카드로 쓰면 결과가 R이든 아니든 행동이 "T를 짓는다" 하나라, card.py 규칙상 결정할 수 없는 카드(UNDECIDABLE)가 됩니다. T 착륙 때 일반 ncu 폼으로 돕니다(`BLOOMERY_NCU_KERNELS`, `DEPTHS=512`, `SKIP=2`, `COUNT=2`/4, 약 10분, `dram__bytes_read`와 hit rate는 기본 수집). 이때 새 커널의 e도 같이 교정합니다.

---

### 6. 범위 밖 개선 후보 (보고만)
- `crates/gpu-deepseek41/src/chain/ffn/batch.rs:221` `ds41_card_buckets`: 1,024스레드 블록 하나가 전문가마다 n_slots를 두 번 직렬로 훑어 129 µs/lb(실측)입니다. 병렬 히스토그램으로 바꾸면 −0.12 ms/lb입니다. XS–S.
- `batch.rs:1593` 청크별 shadow: shexp mcol 42.5 µs(288블록, 0.69웨이브), q4k_mcol 31.5 µs(1.5웨이브), hc_pre 25.4 µs로 합 7.3 ms/lb입니다. T+G 뒤 prose의 다음 카드 항이고, B1처럼 ColGroups로 배치 폭 처리가 가능합니다. M.
- `body/prefill.rs:1041-1043` pageable 동기 H2D가 그룹마다 스트림을 비웁니다. pinned 스테이징과 비동기 복사로 바꾸면 G2 P4096 기준 ~0.6 %입니다 [derived]. S.
- `tools/flow/ds41_prefill.py:742` `shadow_lb`는 층별 균등 lcg를 가정하는데, 트레이스는 n_card 고정에서 16배 편차를 보입니다. 층별 카드 슬롯 수를 stat 줄에서 받아 backtest하면 됩니다. S.
- `crates/gpu-gates/src/bin/generate_ds41.rs`: `stat prefill`에 층별 card 슬롯(= 6T − host 슬롯)을 한 줄 추가하면 커널의 슬롯당 비용을 층마다 바로 읽을 수 있습니다. XS.
- launcher 스레드(C)는 G 뒤 +2.9 % 상한이므로 열지 않습니다.

### 7. 실행 모델
opus(Opus 5.5)로 실행했습니다. 박스, 빌드, 게이트, git은 쓰지 않았습니다.
