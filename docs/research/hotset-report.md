hotset 라운드가 끝났습니다. 레버를 켜지 않으면 기존 게이트의 판정 줄이 모두 기준과 같습니다(Proof A). 목록을 켜면 카드가 목록의 id를 정확히 들고, 슬롯마다 `_sel`이 그 expert만 올린 gemv와 비트 동일합니다(Proof B). 확인이 필요한 발견이 둘 있습니다.
- **64개 목록은 plan (b)를 거부합니다.** plan (b)의 A6000 층은 expert 149개를 두는데 목록에는 64개뿐이라, 로더가 설계대로 거부합니다. 전체 순위 목록(384개)은 plan (b)도 통과합니다.
- **목록을 켜면 단계 게이트의 d1n head 줄이 "diagnostic"으로 바뀝니다.** 게이트 전체는 PASS이고 FAIL은 0입니다. 다만 그 세트에서 스텝이 ik의 경로를 벗어나 envelope 판정이 적용되지 않았습니다.

## 1. 변경 파일
- `crates/model/src/placement.rs`: `ExpertList` 타입을 새로 만들었습니다. 정렬되고 중복이 없으며 목록 순서가 곧 슬롯 순서입니다. 아울러:
  - `Segment.experts`는 목록 타입이 됐고, `span`은 연속 구간 여러 개를 돌려주는 `spans`로 바뀌었습니다.
  - 접두 전제를 깔던 `place_routed`를 공개 함수 `routed_row`(카드 목록과 호스트 여집합)와 `whole_on_card`로 바꿨습니다.
  - `plan()`은 레버를 읽어 `plan_with(…, hot)`에 넘깁니다.
  - 겹침 검사 `segment_shape`는 "각 expert가 정확히 한 번"으로 바뀌었습니다.
- `crates/model/src/placement/hot_list.rs`(신규): 목록 파일 파서, `BLOOMERY_HOT_LIST`를 한 번만 읽는 `from_env`, `card_list`(층의 앞 n_l개), 유닛 테스트 3개.
- `crates/model/src/placement/host_lock.rs`: 호스트 여집합을 구간마다 잠급니다.
- `crates/gpu/src/weights.rs`: `load_rows(model, &[Row])`를 새로 두고 `load_placed`는 여기에 위임합니다. 목록 expert는 스택당 버퍼 하나로 모아 올립니다. 접두는 지금처럼 파일 바이트를 빌려 쓰므로 복사가 없습니다. "rows do not lead" 거부는 지웠습니다.
- `crates/gpu/src/arch/deepseek2/mod.rs`: 가짜 `Plan`/`Machine`/`CardTotals`를 지웠습니다. `hybrid_row`는 `routed_row(ExpertList::prefix(n_l))`를 쓰고, 적재는 `load_rows`로 합니다.
- `crates/gpu-gates/src/bin/gate_deepseek41_load.rs`: 검사 셋을 더했습니다.
  - 기대 슬롯 맵을 plan의 세그먼트가 아니라 목록 파일(없으면 접두)에서 직접 만듭니다.
  - expert 바이트 검사 줄: Σ n_l × expert 하나의 카드 바이트.
  - check vi: 층마다 슬롯 0, n/2, n−1에서 `_sel`을 그 expert만 올린 gemv와 비트로 비교합니다.
- `crates/gpu-gates/src/bin/gate_load_v41.rs`: read-back 표본과 page union을 목록의 id에서 계산합니다.
- `crates/model/tests/placement.rs`: 표 출력을 새 타입에 맞췄습니다. hw 테스트 `hw_placement_hot_list_keeps_the_counts`를 더했습니다(흩어진 합성 목록에서 n_l과 모든 총량이 접두 plan과 같음).
- `tools/ref/router-hotlist.py`(신규): `--self-test` 포함.
- `justfile`: `hotlist` 레시피를 trace-router 뒤에 넣었습니다. 레시피 설명은 한국어 두 줄이라, 리드가 다시 쓸 수 있습니다.
- `AGENTS.md`: 레버 목록 끝에 한 줄을 넣었습니다.
- `Cargo.lock`은 바뀌지 않았습니다.

**"plan이 개수, 파일이 id"의 해결.** 스펙은 파일 줄을 "sorted ids"라고 했지만, 정렬된 목록을 잘라 쓰면 덜 뜨거운 id가 아니라 큰 id를 버리게 됩니다. 그래서 파일은 열도 순(가장 뜨거운 것 먼저, `# order rank`)으로 쓰고, 로더가 층마다 앞 n_l개를 가져가 정렬합니다. 층의 id가 n_l보다 적으면 거부합니다. 그러므로 `--n`은 어떤 plan의 n_l보다 커야 하고, 레시피 기본값은 384입니다.

## 2. 증명

**Proof A (레버 없음).** 첫 편집 전에 스냅샷 트리(원격 `~/repo/bloomery-hotset-base`)에서 기준을 떴습니다. 판정 줄(PASS/FAIL/test)을 비교했습니다.

| 게이트 | 기준 대비 |
|---|---|
| gate-placement | 동일, 새 테스트 한 줄 추가 |
| gate-gpu-hybrid | 동일 |
| gate-gpu-e2e | 동일 |
| gate-gpu-ds41-step --sets | 동일(116줄) |
| gate-gpu-load-v41 | 동일(17줄) |
| gate-gpu-load-v41-lock | 동일(19줄) |
| gate-ds41-load | 새 검사 줄 둘 추가, 마지막 요약 줄 문구만 바뀜 |

gate-ds41-load의 새 줄:
```
check i 3090 expert bytes: Σ n_l × {16773120} B per expert = 14894530560 B, the plan's expert bytes 14894530560 B, resident routed stacks 14894530560 B: PASS
check vi 3090: 342 (stack, slot) pairs on 38 layers, slots 0, n/2 and n-1: the `_sel` gemv at each slot is its listed expert's own gemv bit for bit: PASS
```

**Proof B (`BLOOMERY_HOT_LIST=/root/bloomery-data/router/hotlist-64.txt just gate-ds41-load`, rc 0).**
```
check i 3090: 997 segments, 23043910080 B resident, plan dense + experts 23043910080 B: PASS
check i 3090 expert bytes: Σ n_l × {16773120} B per expert = 14894530560 B, the plan's expert bytes 14894530560 B, resident routed stacks 14894530560 B: PASS
check i 3090 slot map: 40 layers x 384 experts, 888 on the card, the plan's hot lists hold 888, host copy equal: PASS
check vi 3090: 342 (stack, slot) pairs on 38 layers, slots 0, n/2 and n-1: the `_sel` gemv at each slot is its listed expert's own gemv bit for bit: PASS
```

FAIL-first: 로더의 gather를 구간 역순으로 바꿔 한 번 돌렸습니다(rc 1). 되돌린 파일은 `cmp`로 원본과 같음을 확인했습니다. 슬롯 맵과 바이트 검사는 통과했고 check vi만 잡았습니다.
```
check vi 3090: 342 (stack, slot) pairs on 38 layers, slots 0, n/2 and n-1: the `_sel` gemv at each slot is its listed expert's own gemv bit for bit: FAIL
FAIL: check vi 3090: layer 2 blk.2.ffn_down_exps.weight slot 0: `_sel` differs from expert 14's own gemv first at row Some(0)
FAIL: check vi 3090: … and 264 more
```

전체 순위 목록(`hotlist-384.txt`)에서는 gate-placement, gate-gpu-load-v41, gate-gpu-load-v41-lock이 모두 rc 0입니다. `hotlist-64.txt`에서는 plan (b)가 이렇게 거부됩니다.
```
hot list /root/bloomery-data/router/hotlist-64.txt: layer 2 lists 64 experts, the plan keeps 149 on its card
```

**Proof C (`just gen-ds41 --depth 6 -n 16`, 3090 gate 배치, 접두 대 목록).** 같은 입력 id로 첫 생성 토큰부터 갈립니다.
```
prefix tokens [16, 223, 11668, 4063, 305, 9793, 8037, 16, 223, 53606, 271, 5, 223, 4377, 7831, 947]
list   tokens [271, 5, 53606, 271, 372, 1999, 344, 436, 1240, 48037, 427, 344, 260, 2395, 11202, 396]
```
원인을 가르려고 목록을 켠 채 `gate-gpu-ds41-step --sets`를 돌렸습니다. PASSED, PASS 57줄, FAIL 0줄입니다. 값이 움직였으니 목록이 실제로 쓰였습니다. 두 head 줄 모두 top1은 ik와 같습니다.
```
head set=ref_deepseek41_step4_every_node: top1 11111 (margin 2.0293 over 260) ik top1 11111 (margin 3.8761 over 223) … diagnostic
head set=ref_deepseek41_d1n_every_node: top1 5873 (margin 0.3339 over 892) ik top1 5873 (margin 0.2516 over 892), logits_rel=1.753e-2 carried=NaN … diagnostic (the step left ik's path)
```
해석: 목록은 어떤 expert를 카드 경로(q8_1)로 돌리고 어떤 것을 호스트 경로(q8_K)로 돌릴지를 바꿉니다. 그 반올림 차이가 여기서 선택 하나를 뒤집은 것으로 보입니다. 입력은 depth 6의 lcg 프롬프트라 여백이 작습니다. 비트 동일은 확인하지 않았습니다. 기준에서 PASS였던 d1n head 줄이 목록에서는 diagnostic이 된 점은 리드가 판단할 항목입니다.

**C′ (카드 적중 개수).** telem(`c1e733e`)이 이 브랜치의 base에 없어서, 스펙의 대안 경로로 쟀습니다. 각 라우터 세트의 첫 16토큰 선택 3,840개를 두 맵에 대조했습니다. 목록이 이 토큰들로 학습됐으므로 in-sample 값입니다.

| 세트 | plan (a) 접두 | plan (a) 목록 | gate plan 접두 | gate plan 목록 |
|---|---|---|---|---|
| code | 651 (17.0 %) | 2025 (52.7 %) | 244 (6.4 %) | 1393 (36.3 %) |
| prose | 613 (16.0 %) | 1552 (40.4 %) | 186 (4.8 %) | 1023 (26.6 %) |
| korean | 625 (16.3 %) | 2117 (55.1 %) | 300 (7.8 %) | 1632 (42.5 %) |
| threads | 596 (15.5 %) | 1531 (39.9 %) | 213 (5.5 %) | 912 (23.8 %) |

카드가 있는 38층만 세면 plan (a)에서 접두 16.3–17.8 %, 목록 42.0–58.0 %입니다.

**나머지 점검.**
- `just check` rc 0이고, body.rs(`gpu-deepseek41`)는 수정 없이 컴파일됐습니다.
- `just lint`는 경고 169개로 base 스냅샷의 169와 같습니다. 처음에는 175였는데, 내 경고 둘(`single_range_in_vec_init`, `map_entry`)을 고친 뒤 영향받은 두 적재 게이트를 다시 돌려 둘 다 PASS입니다.
- `gate-ops` rc 0이고 hot_list 유닛 테스트 3개가 통과합니다.
- check-recipes, check-arch, check-comments, fmt-check 모두 ok입니다. 박스의 self-test도 ok입니다.

## 3. 예측 대 결과
- 기대 C′는 접두 약 16.7 %, 목록 40–74 %였습니다 [유도]. plan (a) 측정값은 접두 15.5–17.0 %, 목록 39.9–55.1 %로 대역 안입니다.
- 카드 바이트는 Σ n_l × 16,773,120 B로 예측했고, gate plan에서 888 × 16,773,120 = 14,894,530,560 B가 plan과 상주 스택에서 모두 일치했습니다.
- 시간 측정은 하지 않았습니다. `depth-ds41.sh`는 `$BIN`을 환경 그대로 실행하므로 스크립트 변경 없이 레버가 전달됩니다. 명령은 `BLOOMERY_BOX_ENV="BLOOMERY_HOT_LIST=/root/bloomery-data/router/hotlist-384.txt" just depth-gpu-ds41 …`입니다.

## 4. 하지 못한 것
- **목록 파일은 레포에 없습니다.** 박스 `/root/bloomery-data/router/`에 `hotlist-64.txt`와 `hotlist-384.txt`가 있고, 두 파일의 층별 앞 64개는 같습니다.
- **원격 aux 디렉터리가 남아 있습니다.** `~/repo/bloomery-hotset-base`(기준 스냅샷 빌드)는 리드가 `just box-tracks --remove`로 지워야 합니다. 마지막 box-gc에서 고아 프로세스는 0개였습니다.
- **body.rs 수정은 필요 없었습니다.**

목록 파일의 첫 세 층입니다. 0·1층은 plan에서 n_l이 0이라 쓰이지 않습니다.
```
0	56,2,292,323,116,242,21,39,237,183,69,35,40,182,164,229,317,30,321,68,…
1	160,85,361,213,318,217,5,67,193,180,280,108,340,12,259,57,274,133,383,377,…
2	187,120,105,323,132,166,352,35,27,380,161,216,241,254,213,14,268,151,311,58,…
```

접두 가정 표와 처분입니다.

| 자리 | 처분 |
|---|---|
| `placement.rs:780-808` `place_routed` | `routed_row(list)`와 호스트 여집합으로 교체 |
| `placement.rs:339-362` `Segment::span` | 구간 여러 개를 돌려주는 `spans`로 교체 |
| `weights.rs:196-201` 카드당 세그먼트 둘 거부 | 유지. 스택당 버퍼 하나 설계에서 여전히 참 |
| `weights.rs:320-324` "rows do not lead" | 삭제. 대신 gather |
| `weights.rs:457-459` `bytes.get(..n)` | 모은 바이트나 빌린 바이트를 받도록 바꿈 |
| `gate_deepseek41_load.rs:315-316` | 기대 맵을 목록 파일이나 접두에서 만듦 |
| `gate_load_v41.rs:27,403` 및 page union | id별 구간으로 바꿈 |
| `host_lock.rs` span | 구간마다 잠금 |
| `deepseek2/mod.rs:566-612` | 가짜 Plan 삭제, `routed_row` + `load_rows` |

AGENTS.md에 넣은 줄:
```
`BLOOMERY_HOT_LIST=<path>` (placement: a hot list file from `tools/ref/router-hotlist.py`;
each routed layer's card keeps the file's first `n_l` ranked ids instead of the id prefix `[0, n_l)`,
same counts and bytes; unset is the prefix; a layer listing fewer than the plan's `n_l` is refused).
```

## 5. 스펙 밖 개선 지점 (보고만)
- `crates/gpu-deepseek41/src/experts.rs:298-301`: `ExpertGateUp.wg` 문서가 "experts in id order from 0"이라고 합니다. 정확히는 "the slot order of the card's list"입니다. 한 줄이고 engshadow 파일입니다.
- `crates/gpu/src/weights.rs`: 목록 gather는 바이트 사본 뒤에 `words_of` 사본을 한 번 더 만들어, 적재 중 스택 크기의 2배를 잡습니다. KQuant에서 워드로 바로 모으면 하나로 줄어듭니다(S).
- `crates/gpu-gates/src/bin/gate_load_v41.rs` check 3: 목록일 때 experts 구간 목록이 한 줄로 수백 자가 됩니다. 구간 수만 찍는 편이 읽기 쉽습니다(XS).
- `crates/gpu/src/hybrid.rs:3,98,156`과 `crates/gpu/src/model.rs:127,144,420,455`: "[0, n_l)" 문서는 V2-Lite 경로에서는 여전히 참이지만, V4.1 경로와 헷갈릴 수 있습니다(XS).
- `crates/gpu-gates/src/bin/generate_ds41.rs:334`: plan 줄이 목록 사용 여부를 찍지 않아, 로그만으로는 접두 실행과 목록 실행을 구별할 수 없습니다(XS, telem 파일).
- 목록 실행에서 gen-ds41가 첫 토큰부터 갈린 것은 lcg 프롬프트의 작은 여백 탓일 수 있습니다. 실제 텍스트 프롬프트로 접두 대 목록을 교차 확인하면 반올림인지 확정됩니다(S, 박스 2분).

## 6. 모델
opus(Opus 5.5)로 실행했습니다.
