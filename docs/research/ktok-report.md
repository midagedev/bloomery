ktok 라운드를 마쳤습니다. 지도, 설계 셋, 측정 하나가 모두 나왔습니다. 측정 판정은 **선형**입니다. 호스트 expert 티어는 k행을 주면 k배 가까이 걸리고, 단축은 가장 커도 행당 −1.35 %여서 판정을 뒤집을 크기가 아닙니다. 설계에서 가장 큰 발견은 (a′) 층 어긋남 파이프라인입니다. 이 설계는 한 토큰 사슬 둘을 층 단위로 엇갈려 놓은 것이어서 k-token 커널이 필요 없고, 순차 스텝과 비트 동일하며, k=1 롤백에 복원할 상태가 없습니다.

## 1. 바꾼 파일

- `/Users/hckim/repo/bloomery-ktok/crates/model/src/bin/bench_v41_host.rs` 한 파일만 바꿨고, 탐침 코드는 전부 여기에 있습니다.
  - 팔 문법에 `<shape>:<n_host>x<rows>`를 더했고, 새 모양 `engine-sep`(행마다 따로 호출)와 `engine_rows_layer`(모든 행을 그룹 하나로)를 넣었습니다.
  - 기본 팔과 `--check`는 그대로입니다. 행 1의 추첨 `draw(n_host·1)`은 이전 추첨과 같고, `check()`는 `engine_layer`를 직접 부릅니다.
  - 행들은 작업 집합 24슬롯에서 한 번에 뽑은 `n_host·rows`개의 서로 다른 expert를 나눠 가져서 겹치지 않습니다. 행마다 활성 열도 다릅니다. 스펙이 제안한 `+i·64 mod 384`와 같은 효과입니다. 브랜치를 버릴지 레버로 남길지는 리드가 정합니다.
- 리드 전용 레시피 `time-cpu-v41-host`는 스펙 §2가 이번 임대 작업 하나를 명시적으로 승인해서 돌렸습니다.

## 2. 실행한 명령과 판정 줄

**측정**(임대 하나, 32스레드 = 박스 물리 코어 수, 서빙 기본값):
```
BLOOMERY_REMOTE='~/repo/bloomery-ktok' just time-cpu-v41-host --threads 32 --rounds 3 --seconds 24 --warmup 3 --arms engine:6,engine:5,engine:5x2,engine:5x3,engine:5x4,engine-sep:5x4
[binary] target/release/bench_v41_host sha256=2c57ee69eae1 mtime=2026-09-23T18:02:33Z (newer than every file in its dep-info)
[lease] held by pid 537959 at 2026-09-23T18:02:57Z
[lease] netdata: frozen
check layers=[0, 1, 2, 7, 14, 27, 34, 39] experts_per_layer=6 sites=200 failed=0 worst_rel_err=8.689e-7 band=1e-5
v41host working_set experts_per_layer=24 bytes=16172974080 pages=3951384 resident_at_open=3388899 populate_s=2.19 resident_populated=3951384 (100.000 %)
summary threads=32 arm=engine:6 rounds=3 tokens=816 ms_min=29.364 ms_mean=29.661 round_means=[29.655,29.724,29.604] gbps_mean=136.32 gbps_best=137.69 admissible=yes
summary threads=32 arm=engine:5 rounds=3 tokens=972 ms_min=24.619 ms_mean=24.892 round_means=[24.921,24.848,24.907] gbps_mean=135.36 gbps_best=136.86 admissible=yes
summary threads=32 arm=engine:5x2 rounds=3 tokens=492 ms_min=48.588 ms_mean=49.109 round_means=[49.091,49.145,49.093] gbps_mean=137.22 gbps_best=138.69 admissible=yes
summary threads=32 arm=engine:5x3 rounds=3 tokens=327 ms_min=73.217 ms_mean=73.960 round_means=[73.972,73.774,74.134] gbps_mean=136.67 gbps_best=138.06 admissible=yes
summary threads=32 arm=engine:5x4 rounds=3 tokens=243 ms_min=98.527 ms_mean=99.357 round_means=[99.223,99.560,99.289] gbps_mean=135.65 gbps_best=136.79 admissible=yes
summary threads=32 arm=engine-sep:5x4 rounds=3 tokens=243 ms_min=98.530 ms_mean=99.552 round_means=[99.781,99.342,99.532] gbps_mean=135.38 gbps_best=136.79 admissible=yes
--- sweep wall 151 s
rc=0
```

**증인 블록**:
```
--- witness pre 2026-09-23T18:02:57Z ---
loadavg: 0.73 2.39 3.80 1/1245 538039
pressure-cpu: some avg10=0.00 avg60=0.00 avg300=0.00 total=18448707
pressure-io: some avg10=0.00 avg60=0.05 avg300=0.10 total=57901621
0, NVIDIA GeForce RTX 3090, 0 %, 29.23 W
1, NVIDIA RTX A6000, 0 %, 18.86 W
gpu-apps: []
cpu-mhz min/max: 400.000 4304.996
--- witness post 2026-09-23T18:05:28Z ---
loadavg: 27.39 13.74 7.91 1/1227 539596
pressure-cpu: some avg10=0.00 avg60=0.00 avg300=0.00 total=18475612
pressure-io: some avg10=0.00 avg60=0.16 avg300=0.28 total=59095794
gpu-apps: []
cpu-mhz min/max: 400.000 4524.279
```
끝난 뒤 loadavg 27은 벤치 자신의 32스레드입니다.

**팔마다 토큰당 호스트 바이트**(벤치 출력): engine:5 = 3,369,369,600, engine:5x2 = 6,738,739,200, engine:5x4와 sep:5x4 = 13,477,478,400, engine:6 = 4,043,243,520.

**행당 ms를 행 1과 비교한 값.** 구간은 세 바퀴 평균으로 낸 95 %이고 t₂ = 4.303입니다. 바퀴 평균의 SD는 0.06–0.24 %입니다.

| 팔 | 행당 ms | 행 1 대비 |
|---|---|---|
| 5x2 묶음 | 24.555 | −1.35 % ± 0.42 |
| 5x3 묶음 | 24.653 | −0.96 % ± 0.71 |
| 5x4 묶음 | 24.839 | −0.21 % ± 0.59 |
| 5x4 따로 | 24.888 | −0.02 % ± 0.67 |

5x4 묶음은 추세에서 뺍니다. down 그룹이 20쌍이라 `MAX_DEFER_SLOTS` 16을 넘고, 그러면 호출자 쪽 결합으로 떨어집니다. k=3 대비 행당 +0.19 ms가 그 비용입니다[유도].

**lint**(`just lint`, 전체 출력을 파일로 받아 `grep -c '^warning:'`):
- 탐침 첫 판은 173이었습니다. `chunks_exact_to_as_chunks` 경고 2개가 대상 2개에서 세어졌습니다.
- `as_chunks`로 고친 뒤 **169, rc=0**으로 기준과 같습니다. 이 수정은 반복 순서를 바꾸지 않는 lint 전용 재작성이고, 측정은 수정 전 바이너리로 했습니다.

**box-gc**: 처음과 끝 모두 `found 0 process(es)`, `gc-done`.

## 3. 예측과 결과

예측은 임대 전에 `…/scratchpad/ktok/prediction.txt`에 적었습니다(18:02:49Z, 증인 18:02:57Z보다 앞).

| 팔 | 예측 (128–130 GB/s) | 측정 |
|---|---|---|
| engine:6 (B2h 교정 짝) | 31.1–31.5 ms | 29.66 ms (−5 %) |
| engine:5 행 1 | 26.2–26.6 | 24.89 (−5 %) |
| 5x2 | 52.4–53.3 | 49.11 |
| 5x3 | 78.7–79.9 | 73.96 |
| 5x4 묶음 | 104.9–106.5 + 교란 | 99.36 |
| sep 5x4 | 4 × 행 1 | 99.55 (4 × 24.89 = 99.57) |

- **빗나간 항은 `BW_host`입니다.** 오늘은 135–137 GB/s로, 128–130이 아닙니다. 교정 짝도 같은 −5 %로 빗나갔습니다.
  - 원인 후보[유도, 측정 아님]는 조용한 기계입니다. 증인이 loadavg 0.73, pressure 0, GPU 앱 없음이었고, plan.md의 B2b 메모에도 병렬 빌드 중 호스트 다리가 흔들렸다고 적혀 있습니다.
  - B5 분할의 호스트 26.54 ms(3.409 GB)에 해당하는 오늘 값은 24.89 ms(3.369 GB)입니다. `BW_host` 재교정은 리드 판단입니다.
- **k 비율 판정은 이 빗나감과 무관합니다.** 같은 임대에서 교차해 잰 비율이기 때문입니다.
- **판정은 선형입니다.** 제가 이름 붙인 준선형 신호는 k=2 −1.35 %로 나타났지만 작습니다.
  - 기제는 gate+up 디스패치의 대역 향상(137.1 → 138.8 GB/s)과 디스패치 고정비 공유입니다.
  - 뒤집기 문턱은 행당 ~0.6배였는데 측정은 0.986배 이상입니다.
- **설계 귀결**: 티어는 호출당 한 토큰 그대로 두고, 층마다 k번 부르면 됩니다. 행마다 따로 호출해도 k배와 같으므로(−0.02 %) `experts_into`를 다시 쓸 이유가 없습니다.
- **탐침이 값을 매기지 못한 것**: 행마다 서비스를 따로 열면 서비스마다 go/wait 왕복이 붙습니다. B2b의 왕복 18–22 µs로 k=2면 패스당 ~0.8 ms입니다[유도]. 층마다 go 하나에 k행을 실으면 이 비용이 없습니다.

### 지도: 스텝이 한 토큰을 가정하는 자리

변경 클래스는 AGENTS.md 표 기준입니다. 캡처 열의 "k별 캡처"는 k ∈ {1..K_MAX}마다 그래프를 하나씩 캡처한다는 뜻이고, 버퍼는 공유합니다. 대부분의 커널이 토큰 수를 런치 인자와 격자로 받아 캡처 때 굳히기 때문입니다.

| 자리 | 가정 | k-token에 필요한 것 | 클래스 | 크기 | 캡처 |
|---|---|---|---|---|---|
| `body.rs:68`, `:825`, `:832`, `:853` | `STEP_TOKENS=1`로 StepRows·이미지·목록을 할당 | K_MAX로 할당 | 런치 모양 | XS | k별 |
| `body.rs:691`, `:696` | `plan_into(&[token])`, history에 하나만 push | 토큰 슬라이스로 계획, `truncate(a+1)` API | 호스트 | XS | — |
| `body.rs:1058` 링 `min(ctx_max, W)`행 + `rope.rs:432` slot = pos % 행수 | W행 링 | 아래 attention 행 참고 | — | — | — |
| `attn.rs:19–23`(op) | 토큰 t가 슬롯 0..vis[2t]를 순서대로 읽음. 링을 감는 배치는 계약 밖 | 링이 차면 k행 append가 토큰 0의 창을 지우고 미래 행을 보이게 함. 선택 (i) 토큰마다 append+score를 교차(층당 attn k번), 선택 (ii) W+K_MAX−1 링과 셀 범위 가림 | (i) 런치 수, (ii) 합산 순서 | (i) S, (ii) M | (i) k별 |
| `chain/attn.rs:324–346` `plan_words` | token(0)의 pos, len, 표 넷, n_visible[0]만 gather | 토큰마다 필드 | 런치 모양 | S | k별 |
| `chain/attn.rs:612`, `:686` | tokens ≠ 1 거부, scratch m=1 | K_MAX로 할당 | 런치 모양 | XS | k별 |
| **`chain/attn.rs:701`** | `gm != 1` 거부. 인덱스 키는 투영을 토큰 우선(`k[g·WIDTH]`)으로 읽고 gemv는 `y[r·m+c]`로 씀 | **k=2면 비율 1 스트림(20층) max_groups=2라 오늘 거부됨.** strided 읽기나 전치 출력 | 런치 모양 | S | k별 |
| `chain/attn.rs:1284` 대 `indexer.rs:256` | 인덱서는 `ints[n_vis_at+t]`를 연속으로 읽고 워드는 [len, n_vis] 쌍 | 연속 n_vis 필드를 따로 둠(m=1에서만 우연히 맞음) | 런치 모양 | XS | 파라미터 워드 가능 |
| 압축 행 가시성 | 토큰별 접두 개수 `vis[2t+1]`, 같은 층의 압축기가 먼저 씀 | 없음. 이미 블록 안 인과 | — | 0 | 파라미터 워드 |
| `compress.rs:94` CompGeom, `plan.rs` StreamStep | 이미 k 토큰 (ik ubatch 전사) | `:701`만 | — | 0 | — |
| 인덱서 op | 행별 n_vis·top-k·rope, 행별 커널 | 위 `:1284`만 | — | XS | — |
| `hc.rs:83` `HC_MAX_TOKENS=8` | 런치당 최대 8토큰 | K_MAX ≤ 8 | — | 0 | k별 |
| `q8f32.rs:23–27`, `lib.rs:2039` | q8_0 gemv는 m=1에서 워드 단위 레인, m>1에서 32값 청크 레인. m ≤ 8 | **열로 묶으면 합산 순서가 바뀜.** 사이트: q_a, q_b, kv, wo_b, 인덱서 q_b, shared down, engram_wkv. m=1 순서의 m>1 변형 커널, 또는 엔벨로프 게이트 | 합산 순서 | S | — |
| `cores.rs:817` q3_K | `q3k_row_dot_1col`은 "같은 적재, 같은 누적 순서"라고 문서화 | 이를 고정하는 핀을 grep으로 못 찾음 | — | XS 게이트 | — |
| `enqueue_output` `Q8_0GemvHeadsArgs` | m 필드가 없는 한 토큰 커널 | m 인자, 또는 k번 런치 | 런치 모양 | S | k별 |
| rope, norm, kv append, tail rope | 이미 행별 m | 1을 k로 | 런치 모양 | XS | k별 |
| `chain/ffn.rs:693`, `:695`, `:697`, `:977`, `:1019` | act_x 1열, sel·h·down N_USED, HC_PRE tokens:1, shexp m=1 | k·N_USED 슬롯과 슬롯에서 토큰 열로 가는 색인 | 런치 모양 | M | k별 |
| `router.rs:5` | "토큰당 런치 하나", 티켓으로 마지막 블록이 선택 | k번 런치, 또는 격자 y = 토큰에 토큰별 티켓. 비트 동일은 `f32_lane_partial_1col` 재사용 조건 | 런치 수 | S | k별 |
| `experts.rs:4` | 런치당 한 토큰(활성 1열) | 슬롯마다 토큰 열 | 런치 모양 | S | k별 |
| `chain/ffn.rs:134` handoff, `:215` post | 한 토큰의 ids·weights·x, 결합 한 행 | k행 | 런치 모양 | S | k별 |
| `hybrid.rs:886`, `:1042`, BoundaryShape | handoff 영역 하나, hsum 하나, x 한 열 | 영역 k개(또는 k행 영역)와 hsum k개. `captured`를 (층, 행)으로 | 프로토콜 | S–M | — |
| `moe.rs:745`, `:811` | `EXPERTS_INTO_MAX` 8, `x.ne1 != 1` 거부 | **그대로 둠**(k번 호출이 k배와 같다고 측정됨) | — | 0 | — |
| `glue.rs:253`, `:417`, `:425`, `:464`, `:470`, `:496`와 embed 커널 | 한 토큰 | k행 | 런치 모양 | S | k별 |
| `head.rs:180`, `:212`, `plan.rs:213` | 입력 한 행, argmax 하나, out_ids는 마지막만 | 검증 패스는 k+1행 전부, 프리필은 마지막만 | 런치 모양 | S | k별 |
| `model.rs:671` | `step`이 토큰마다 한 몸통 | `step_block(tokens)`와 `rollback(pos)` | 호스트 | S | — |
| `params.rs:203` | m ≤ W(128) | K_MAX의 다음 벽 | — | — | — |

### (a) k-token 포워드

- **층 순서는 오늘과 같고 m=k입니다.** HC_PRE(k) → norm(k) → 압축기(⌈k/r⌉+1 그룹, `:701` 수정이 전제) → query(k) → kv와 append → 인덱서(행별 n_vis, n_vis ≤ top_k이면 항등 목록. 행별 커널은 이미 있음) → attention → wo_a·wo_b(k) → HC_POST(k) → ffn: norm(k), router(k), handoff(k행). 이어서 층마다 go 하나에 k행을 싣고 호스트는 한 서비스 안에서 `experts_into`를 k번 부릅니다. 카드 expert k·6슬롯, shared m=k, 결합 k행, head k행 logits와 argmax가 뒤따릅니다.
- **attention과 링.** 스펙의 "링이 점수 전에 k개 새 행을 이미 들고 있다"는 W행 링에서는 틀립니다. 선택지는 둘입니다.
  - (i) 층 안에서 토큰마다 append 다음 score를 교차합니다. W링과 오늘의 슬롯 공간 세그먼트 절단이 그대로이고, k번 순차 스텝과 비트 동일합니다. 노드는 +2·40·(k−1)이고 c_node로 0.07–0.14 ms입니다[유도]. **1라운드에 권합니다.**
  - (ii) k행 런치 하나에 W+K_MAX−1 링, 셀 범위 가림으로 커널을 다시 씁니다. 롤백이 공짜지만 깊이 ≥ W에서 오늘의 한 토큰 비트가 움직입니다. ik 대비 σ 밴드 안이지만 오늘 핀이 다시 박힙니다. 프리필용입니다.
- **engram**: 드래프트 위치의 n-gram은 드래프트 토큰을 포함합니다. 그래도 괜찮습니다. 그 위치의 logits는 앞 토큰이 전부 수락될 때만 쓰이고, 그때 n-gram은 실제와 같습니다. 수락된 접두의 행은 뒤만 돌아보므로 거절 토큰에 의존하지 않습니다.
- **V2-Lite 기준**(`deepseek2/attn.rs:1054` `visible_end`): 키는 pos ≤ q.pos일 때 보이고, 절단은 보이는 개수 위에서 합니다. 그래서 프리필 행과 같은 위치의 디코드 스텝이 같은 세그먼트를 자릅니다. (i)은 이 원칙을 지킵니다.

### (a′) 층 어긋남 파이프라인

행 A는 수락된 토큰 t, 행 B는 드래프트 t+1입니다. 카드 스트림 하나, 층 l의 순서는 이렇습니다.

```
… go_A(l) ; [A 그늘: HC_PRE, 카드 expert, shared] ;
wait_B(l−1) ; combine_B(l−1)+HC_POST ; attn_B(l) ; router_B(l) ; handoff_B(l) ; go_B(l) ; [B 그늘] ;
wait_A(l) ; combine_A(l)+HC_POST ; attn_A(l+1) ; router_A(l+1) ; go_A(l+1) ; …
호스트(서비스 순서 = go 순서): … H[B,l−1], H[A,l], H[B,l], H[A,l+1] …
```

- **의존성은 모두 채워집니다.** attn_B(l)은 A의 층 l KV(attn_A(l)이 이미 씀)와 B 자신의 l−1 출력만 읽습니다. 압축기 상태도 순차와 같은 순서로 A 다음 B입니다.
- **호스트가 쉬지 않습니다.** H[A,l] 동안 카드가 할 일은 combine_B부터 go_B(l)까지입니다. 이는 카드의 행·층 몫 ≤ 0.49 ms[유도: all-card 토큰 그래프 19.64 ms / 40]이고, 호스트 행·층 0.62 ms[측정 24.89/40]보다 짧습니다.
- **패스 시간은 ≈ 2H + D입니다.** 분할이 H = 26.54, G = 12.95일 때 53.08 + D이고, 오늘의 H = 24.89로는 49.78 + D입니다. 같은 분할의 2행 평범 배치는 66.03 + D입니다. 이 배치 값은 2행 카드 임계가 1행과 같다고 가정한 것입니다(밀집 바이트를 한 번만 읽음). D는 첫 층의 go 전 몫과 마지막 결합, 2행 head로 ~0.3–1 ms이며 유도값입니다.
- **투기 디코딩 손익분기**(패스당 1+α 토큰, 기준 H+G): 어긋남은 α > 0.34, 평범 배치는 α > 0.67입니다[유도, D = 0].
- **필요한 것**: 행마다 HC streams와 fold 핑퐁, ffn scratch(rout, sel, h, down, sh_y, y, hc_out, mixes), 인덱서 목록, handoff 영역·hsum·x, 이미지 워드가 따로 있어야 합니다. 인덱서 목록은 A의 l_i+1층이 읽기 전에 B가 덮어쓰므로 행마다 필수입니다. attention scratch는 공유해도 됩니다.
  - `Hybrid.captured`는 (층, 행)이 됩니다. 카운터 하나로 됩니다. wait 순서가 서비스 순서와 같아 FIFO로 동작하고, seq 검사도 go 순서를 따릅니다.
  - **`experts_into`는 재진입이 필요 없습니다.** 서비스는 결정 스레드에서 직렬이고, `Ds41Host`는 `&mut self`에 scratch 하나입니다.
- **k=2 드래프트(세 행, 두 층 어긋남)로 일반화됩니다.** ≈ 3H + D이고 조건은 같습니다.
- **비트 동일**: 각 행이 한 토큰 커널을 같은 상대 순서로 돌리므로 순차 스텝과 정확히 같습니다.

### (b) 롤백

| 상태 | 어디 | 되감기 규칙 | 패스당 비용 [유도] |
|---|---|---|---|
| 창 링 | `LayerKv.ring`, W행, slot p%W | 거절 위치 p′가 셀 p′−W를 지웁니다. 다음 스텝 a+1은 ≥ a+2−W가 필요하므로 복원할 행은 max(0, k−r−1)입니다(k는 마지막 드래프트 색인). 드래프트 1개면 0, 2개면 ≤ 1행/층이고 패스 전에 스냅샷합니다. (ii) 링이면 공짜 | k=1은 0, k=2는 40 KB D2D |
| 압축 행 | 2·8·14층(비율 2), 20층(비율 1), 행 p/r | 공짜입니다. n_vis 밖이라 안 읽히고 다시 디코드할 때 덮어씁니다 | 0 |
| 인덱스 키 | 그룹당 한 행 | 공짜, 같은 이유 | 0 |
| 압축기 상태 | 2·8·14층, values+scores 2×512 f32 = 층당 8 KB | 거절 위치 ≥ a+2가 있을 때(드래프트 ≥ 2)만 필요한 슬롯이 파괴됩니다. 드래프트 1개면 공짜입니다. 링에서 다시 계산할 수 없으므로(다른 투영) 패스 전 24 KB 스냅샷, 거절 때만 eager 복원 | D2D 노드 ~3개, ~2.6 µs |
| streams, fold, 목록, scratch | 스텝마다 | 없음 | 0 |
| 이미지, pos | 호스트 | a+1에서 다시 빌드 | 0 |
| history(engram, planner) | `Body.history` | a+1로 truncate | 0 |
| hybrid served/seq | `Hybrid.served` | 이어서 증가(위치가 아님) | 0 |

**게이트는 "k 스텝 후 r로 롤백 = r+1개 순차 스텝, 비트 단위"입니다.**
- (a′)와 (a)(i)에서는 달성 가능합니다. 단, 열로 묶는 런치가 m=1 열 순서를 지켜야 합니다.
- q8_0 m>1 사이트에서 깨집니다. m=1 순서의 변형 커널을 만들든지, 그 사이트만 엔벨로프 게이트로 가야 합니다.
- router와 experts의 새 k행 커널은 1열 코어를 재사용할 때만 같습니다.
- (a)(ii)는 오늘의 비트를 옮긴 뒤에야 양쪽이 같은 커널이 되어 정확해집니다.

### (c) 프리필과 K_MAX

- **오늘 K_MAX = 8입니다.** 벽이 셋 있습니다.
  - `HC_MAX_TOKENS` 8 (`hc.rs:83`)
  - q8_0·q3_K·q4_K·f32 gemv의 m ≤ 8 (`lib.rs:2039`, `q8f32.rs:769`)
  - `Q8Act` m 1..=8
- 스펙이 든 `tensor.rs:106`은 K축의 `Q8ACT_MAX_K`이지 토큰 수 제한이 아닙니다.
- **다음 벽**: 이미지 m ≤ W=128 (`params.rs:203`), part_v scratch가 토큰당 ~1.3 MB[유도], 호스트의 `MAX_DEFER_SLOTS` 16과 `EXPERTS_INTO_MAX` 8.
- **k=8 청크 프리필은 토큰당 ~25 ms로, 오늘 39.4 ms입니다.** 4096토큰이 ~100 s로 오늘 160 s입니다[유도: 호스트 선형 측정 × 균등 라우팅에서 서로 다른 expert 45.4/48, 카드는 그늘 안이라고 가정].
- **k ≥ 64에서만 큰 몫이 열립니다.** 서로 다른 expert 비율이 k=16에서 −11 %, k=64에서 −37 %입니다[유도, 균등 라우팅]. 이를 쓰려면 호스트가 expert별로 합쳐 다열 입력을 받고, 카드는 텐서코어 GEMM을 써야 합니다. K_MAX 8 너머의 별도 라운드입니다.

### 구현 라운드 제안 (넷, 각 ≤ M)

1. **`ktok-skew`** (M): (a′) 드래프트 1개. 파일은 `body.rs`, `chain/ffn.rs`(행별 scratch), `hybrid.rs`, `model.rs`(verify와 rollback API), 새 게이트 바이너리입니다. 게이트는 2행 패스와 r∈{0,1} 롤백이 순차 스텝과 비트 동일한지이고(logits, 링, 행, 키, 상태), G2 노드 ≈ 2×1,068[유도]입니다. 시간 A/B는 리드가 합니다. 가장 작고 정확하며 2H+D를 직접 확인합니다.
2. **`ktok-kernels`** (M): 1 다음 순서입니다(`chain/ffn.rs` 공유). router 2-D 격자, experts 슬롯-토큰, handoff·post k행, embed·engram·head k행, 인덱스 키 gm>1(`:701`), q8_0 m>1을 m=1 레인 순서로, `plan_words` 토큰별 필드와 연속 n_vis. 게이트는 op마다 m=k의 열 c가 m=1과 비트 동일한지입니다.
3. **`ktok-batch`** (M): (a)(i) 토큰별 attention 교차, 창과 압축기 스냅샷·복원, k별 캡처, k 스텝+롤백 = 순차 게이트.
4. **`prefill8`** (M): K=8 청크 프리필과 측정. K > 8(GEMM, 호스트 합집합)은 그 뒤 따로입니다.

## 4. 못 한 것

- 없습니다.
- 평범 배치의 "2행 카드 임계 ≈ 1행"은 측정하지 않은 가정입니다.
- q3_K m=1 = m>1 열 0은 문서로만 확인했고 핀은 찾지 못했습니다.

## 5. 스펙 밖 개선 지점 (보고만, 손대지 않음)

- `crates/model/src/bin/bench_v41_host.rs:425`: `blocks()`가 시간 재는 토큰 안에서 디스패치마다 출력 블록을 할당해 할당기가 시간에 섞입니다. XS.
- `crates/model/src/ops.rs:1599`: `MAX_DEFER_SLOTS` 16을 넘는 호스트 그룹은 호출자 쪽 결합으로 떨어집니다. 20쌍에서 행당 ~0.19 ms입니다[유도]. XS–S.
- `crates/gpu/src/cores.rs:817`: `q3k_row_dot_1col` = 다열 열 0의 비트 핀이 없습니다. XS 게이트.
- `docs/plan.md`의 「모델」 `BW_host` 행(128–130)이 오늘 135–137 GB/s입니다. B5 분할의 호스트 26.54 ms도 같은 방향입니다. 재교정은 리드 판단입니다.

## 6. 모델

opus로 스폰되어 Claude Opus 5.5(`claude-opus-5-5[1m]`)로 돌았습니다.
