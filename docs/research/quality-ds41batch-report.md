== /Users/hckim/.claude/projects/-Users-hckim-repo-bloomery/02d044bc-8d2a-4055-91fa-4ae541a4916f/subagents/agent-aquality-ds41batch-1ec5d5525ed10934.jsonl ==
## quality-ds41batch 리뷰 보고 (opus, 파일 변경 없음, 박스 작업 없음)

대상은 main `26bcb4a`이고, `43cd107` 이후 이 코드를 바꾼 커밋은 없습니다. 제 발견이 ds41ced 트리(`/Users/hckim/repo/bloomery-ds41ced`, 읽기만 했음)에도 그대로 남아 있는지는 표시한 항목만 확인했습니다. lint는 돌리지 않았습니다(리뷰 라운드이고 여기는 Mac입니다).

가장 중요한 항목은 둘입니다.
- `staging_view`의 수동 반납이 오류 경로에서 빠집니다. ds41ced에도 그대로 있습니다.
- `draft/stage.rs`의 `View`에는 `Drop`이 없습니다. 창을 하나 만들 때마다 `Arc<CudaContext>` 참조가 하나씩 샙니다.

둘 다 "window 해제 사본이 7개" 부류에서 나옵니다. 그래서 첫 번째로 권하는 라운드는 `Window` 타입 하나를 만드는 R14 라운드입니다.

### 발견 (규칙별, 심한 것부터)

**R28/R14/R2 — 창(window) 해제**
1. `/Users/hckim/repo/bloomery/crates/gpu-deepseek41/src/chain/attn.rs:1617-1637` — R14/R28 — `staging_view`는 `ManuallyDrop` 창을 돌려주고, 호출자가 1637에서 `DeviceTensor::release`로 반납해야 합니다. 그런데 1618-1625의 `enqueue_staged(..)?`가 실패하면 반납하지 않고 빠져나가 `Arc<CudaContext>` 참조가 샙니다. 메모리 해제가 아니라 UB는 아닙니다. ds41ced `attn.rs:1646-1666`에도 같은 모양이 남아 있습니다. 올바른 형태는 Drop 가드입니다 — S
2. `/Users/hckim/repo/bloomery/crates/gpu-deepseek41/src/span.rs:52-57` — R28 — `release`는 "drop에서 한 번만 호출"이라는 호출자 계약에 기대는 안전한 fn입니다. 제대로 하려면 "한 번만 take"를 `Drop` 안에 구조로 넣어야 합니다. 즉 `bloomery_gpu::window()`가 `Drop`을 가진 `Window<'a,T>`/`WindowMut`를 돌려주면 `release`라는 fn이 아예 필요 없어집니다. 차선은 `unsafe fn release`에 `/// # Safety`를 다는 것입니다 — S
3. 같은 해제가 7곳에 있습니다: `span.rs:59-69`(Drop 둘), `chain/attn.rs:263-284`(`View`), `draft/stage.rs:17-76`(`View`, Drop 없음 — 목록 밖), `/Users/hckim/repo/bloomery/crates/gpu/src/tensor.rs:94-103`(`PartedBuffer`), `tensor.rs:148-167`(`DeviceTensor::window`/`release`), `body.rs:369`, `/Users/hckim/repo/bloomery/crates/gpu/src/weights.rs:344-345` — R14 — `tensor.rs`의 `window()` 옆에 공통 코어 하나를 둡니다. 호스트 코드만 바뀌므로 비트 중립입니다 — M
4. `/Users/hckim/repo/bloomery/crates/gpu-deepseek41/src/chain/attn.rs:1719-1736` — R14 — `rows_of`는 `View`(attn)를 돌려주고 `rows_of_mut`는 `span::SpanMut`를 돌려줍니다. 한 쌍이 두 창 타입을 씁니다 — XS
5. `/Users/hckim/repo/bloomery/crates/gpu-deepseek41/src/span.rs:111` — R1 — `// SAFETY: as in span`은 R1 보강에 따르면 SAFETY가 아닙니다 — XS

**R1/R3/R4 — unsafe**
6. `/Users/hckim/repo/bloomery/crates/gpu-deepseek41/src/chain/ffn/batch.rs:563-567` — R1 — `unsafe {}` 블록 하나에 FFI 복사(`dtoh`) 셋, SAFETY는 하나뿐입니다 — XS
7. `chain/ffn/batch.rs:349-368`, `:267-270` — R1 — 349는 버퍼 다섯(`hc`·`acc`·`hsum`·`shexp`·`res`)의 서로 다른 경계를 한 블록에 담았습니다. R1 보강이 허용하는 연속 로드 묶음이 아닙니다. 267은 `store4`와 `fold` 쓰기, 두 연산입니다 — XS
8. `chain/ffn/batch.rs:340-344`, `:626-629`, `:657-660` — R3 — `unsafe fn`(`join_post_at`, `dtoh`, `htod`)의 계약이 `/// # Safety` 절이 아니라 `/// SAFETY:` 문단에 있습니다. private이라 lint가 잡지 않습니다 — XS
9. `chain/ffn/batch.rs:626-686` — R4/R14 — 체인 크레이트 안에 원시 `sys::cuMemcpy{DtoH,HtoD}Async_v2`와 `as_mut_ptr().cast()`가 있습니다. 같은 FFI가 `gpu/src/hybrid.rs:972`, `gpu/src/graph.rs:518`, `gpu/src/lib.rs:1875`, `chain/glue.rs:552`에도 있습니다. `bloomery_gpu`에 pinned↔device 비동기 복사용 unsafe fn 한 쌍을 두면 됩니다 — S

**R6/R5 — 형상과 단위**
10. `/Users/hckim/repo/bloomery/crates/gpu-deepseek41/src/body/prefill.rs:55-61,270,476,483` — R6 — `CHUNKS_MAX = T_MAX/CHUNK + 1`은 `T_MAX % CHUNK == 0`일 때만 맞는데 `const { assert! }`가 없습니다. 이 전제가 깨지면 476의 `plans.iter_mut().zip(cuts)`가 조용히 잘리고, 483의 `plans[..n]`이 이름 없이 패닉합니다. `chunks()`는 `u/CHUNK + 2`를 예약해 CHUNKS_MAX와 경계가 다릅니다. 둘 중 하나로 통일하고, `cuts.len() > CHUNKS_MAX`는 이름 있는 거부로 바꿉니다 — XS
11. `chain/ffn/batch.rs:476`, `chain/attn.rs:964` — R6/R14 — `if slots <= 8 {with_k} else {with_slots}`가 두 벌 있습니다. 8은 `gpu/src/tensor.rs:235`의 이름 없는 리터럴입니다. `CHUNK <= 8`, `CHUNK·N_USED <= Q8ACT_MAX_SLOTS(64)`도 load 때 런타임 오류로만 드러납니다. `Q8ACT_MAX_COLS`를 공개하고, `Q8Act::for_cols`와 const assert를 둡니다 — XS
12. `chain/ffn/batch.rs:823,126-127,172-174,232,288` — R6 — 그리드의 `div_ceil(8)`과 커널의 `%256`·`/256*8`·`t/32`가 `THREADS/32`를 이름 없이 씁니다. 런치 계약의 `6`·`24`·`4`는 `N_USED`·`HC_MIX`·`HC_STREAMS`입니다(매크로가 상수 경로를 받는다면 바꿉니다) — XS
13. `chain/attn.rs:786-792` + `body/prefill.rs:311,536,606` — R5 — `with_rows`의 `rows`/`row`는 디코드에서는 pair pass의 행이고, 배치에서는 청크 번호 k입니다(`with_rows(.., CHUNKS_MAX)`, `enqueue_layer_staged(.., l, k, m, ..)`). 같은 파일의 `StageIo.rows`는 스테이징의 행입니다. `usize` 하나가 세 가지 뜻을 가집니다 — XS
14. `body/prefill.rs:128,263,479,580,631,690` + `chain/attn.rs:172` — R5 — 위치가 `BatchSeam.first`·`BatchRun.pos`에서는 u32, `StageIo.first`에서는 usize이고, 그 사이를 `as u32`로 왕복합니다. `chain/glue/batch.rs:203`은 u32끼리 곱해서 `as usize`로 바꾸는데, 63행은 usize로 곱합니다 — XS
15. `/Users/hckim/repo/bloomery/crates/gpu-deepseek41/src/transpose.rs:78` — R5 — `rows * m`을 검사 없이 곱합니다. wrap되면 길이 검사를 통과합니다. `checked_mul`로 바꿉니다 — XS

**R14 — 사본**
16. `chain/glue/batch.rs:264-269`, `chain/ffn/batch.rs:926-934` — R14 — "m==1이면 바로 쓰고, 아니면 raw에 쓴 뒤 transpose"는 `chain/attn.rs:1768-1785` `project()`의 세 번째·네 번째 사본입니다. 같은 모듈(`TransposeKernels`)을 `attn.rs:864`, `ffn.rs:806`, `glue/batch.rs:97`에서 세 번 load합니다. 한 곳이 소유하고 빌려주게 합니다. 런치 순서가 같으므로 비트 중립입니다 — S
17. `chain/attn.rs:1483,2015-2016,2149` — R14/R5 — row-major join의 부분 오프셋(`r0=q_lora_rank`, `w_at=q_lora_rank+head_dim`, `(COMP_SCORE,width)`)을 손으로 더합니다. 부분 길이는 945-953의 `qkv_parts`/`q_parts`가 이미 알고 있습니다. `RowMajor<N>{rows}`의 `part(i,m)` 한 곳으로 모읍니다 — S
18. `body/prefill.rs:535-707` — R14 — 토큰-major 창 `span(WHAT, buf, at*s4, m*s4)`가 15번, `for (k,r) in cuts…{(at,m)=(r.start-b,r.len())}`가 6번 나옵니다. `chain/ffn/batch.rs`에도 `at*N_USED`·`at*n`이 있습니다. `Chunk{k,at,m,first}` 이터레이터와 `tokens(buf, per, at, m)`로 줄입니다 — S
19. `chain/ffn/batch.rs:1061` = `chain/attn.rs:1706`(`count_of`, `glue/batch.rs:255`와 `attn.rs:1746`에도 인라인 사본), 레이어 인덱스 해석 `ffn/batch.rs:953`·`:1026`·`chain/ffn.rs:1009` — R14 — XS
20. `chain/attn.rs:752`, `chain/glue.rs:356`, `chain/ffn.rs:1393` — R14 — `q8act_bytes` 세 벌이 각자 `Q8Act::alloc`의 공식을 다시 유도합니다. `Q8Act::device_bytes()` 하나로 둡니다 — S
21. `chain/ffn/batch.rs:263,318` vs `:346` — R14 — `b = 4*(i/n)*n + i%n`이 세 번 나옵니다. `const fn`으로 빼고 ptx-scan이 같음을 보입니다. `_post_batch`/`_streams` 트윈 자체는 이미 받아들인 R14 모양입니다(공유 `join_post_at` + 얇은 진입점, 커널 ABI 차이). 라운드를 쓸 가치가 없습니다 — XS
22. `/Users/hckim/repo/bloomery/crates/gpu/src/hybrid.rs:1380-1395` vs `:1494-1509`, `:1406-1454` vs `:1603-1629` — R14 — catch_unwind→poison 래퍼, 호스트 리스트 규칙(`is_none_or(slot==HOST)`), finite fold가 `serve_one`의 사본입니다. `fn host_list(..) -> n`으로 모읍니다. fixup4가 이 파일을 고쳤으니 fixup4와 대조해야 합니다 — S
23. `chain/ffn.rs:1469-1472` vs `:1507-1510` — R14 — `UnionScratch::new`의 lazy insert가 두 벌입니다 — XS
24. `/Users/hckim/repo/bloomery/crates/gpu-gates/src/bind.rs:480-486` — R14 — `Generator::prefill`(`generate.rs:141`)의 "빈 입력·ctx 초과" 검사를 복제하고 그 함수를 우회합니다. `/Users/hckim/repo/bloomery/crates/gpu-gates/src/bin/generate_ds41.rs`의 `passes: depth.div_ceil(body::T_MAX)` 두 곳은 `prefill.rs:180`의 절단 규칙을 바이너리가 다시 유도하는 것입니다. prefill이 pass 수를 돌려주게 합니다 — XS

(스펙이 말한 "`chunks()` vs 게이트 쪽" 트윈은 main 게이트에 없습니다. 게이트에는 청크를 도는 코드가 없고, 대응하는 사본은 24의 배치 수 재유도입니다.)

**R12/R8/R13**
25. `chain/attn.rs:1320-1652` — R12 — `enqueue_layer_at`가 333줄입니다(배치 분기 전에는 약 265줄). 7″④의 R12 라운드에 합칩니다 — M
26. `body/prefill.rs:501-715` — R12 — `enqueue_batch_chain`이 215줄입니다. ds41ced에서는 661-917로 256줄이 되어 더 커졌습니다. `enqueue_batch_shadow`(`ffn/batch.rs:781-936`, 156줄)와 게이트의 `seams`(240-366)도 같은 부류입니다 — S
27. `/Users/hckim/repo/bloomery/crates/gpu-deepseek41/src/router.rs`의 `enqueue_router_into`·`launch`(인자 9개), `chain/attn.rs:1297-1328`, `chain/glue/batch.rs:186-199` — R8 — 호스트 쪽에 새 `allow(too_many_arguments)`가 다섯 개 생겼습니다(7′: 호스트 런처는 면제 아님). `RouterArgs{w,x,bias,scale}`로 묶고, `StageIo`는 `AttnIo`의 `Option`으로, `(params,m)`은 `EngramStep`으로 넣습니다 — S
28. `FfnBatch`·`ChunkIo`·`JoinIo`·`GlueBatch`·`PromptRows`·`StageIo`·`TransposeKernels`(+ `lib.rs:38`의 `pub mod transpose`)·`enqueue_layer_staged`·`enqueue_router_into`·`card_sum_elem`·`join_elem`·`span_mut`·`prepare_union`·`pub use CHUNK` — R13 — 크레이트 밖 호출이 없습니다(grep). `FfnBatch::cap`(`ffn/batch.rs:509`)은 호출이 아예 없습니다 — S

**조용한 실패 금지**
29. `hybrid.rs:1429-1433` — 조용한 실패(판정 요청) — `out.fill(NAN); Ok(())`는 스펙 5번이 말한 바로 그 부류입니다. 설계로 문서화돼 있고(카드 norm의 fault 워드가 이름 있는 오류가 된다), 배치에서는 `prefill.rs:447`의 fault 읽기와 마지막 배치 head readback이 그걸 보증합니다. 다만 이 경로는 `batch_*` 통계를 세지 않습니다 — 결정 사항
30. `body/prefill.rs:474,531` — 청크 목록이 비면 `cuts.first().map_or(0, …)`가 위치 0을 돌려줍니다. ds41ced `:633/:695`에도 남아 있습니다 — XS
31. `chain/attn.rs:1662-1663` — `rows == 0` 거부보다 먼저 `st.first % rows.max(1)`로 clamp합니다. 순서만 바꾸면 됩니다. ds41ced `:1788`에도 남아 있습니다 — XS
32. `body/prefill.rs:358-371` — 문서는 "배치가 돈 것보다 많은 행이면 거부"라고 하지만 코드는 `u > T_MAX`만 봅니다. 이전 배치의 행이 조용히 반환될 수 있습니다 — XS
33. `body/prefill.rs:481` + `:443` — `history`를 먼저 늘린 뒤 fault가 아닌 오류로 빠지면 history가 pos보다 앞서 있게 됩니다. 413의 이름 있는 거부가 막지만, reset 전까지 모델이 멈춰 있습니다 — 결정 사항
34. `/Users/hckim/repo/bloomery/crates/gpu-gates/src/bin/gate_deepseek41_prefill.rs:507,511` — 표에 없는 레이어가 `map_or(0)`와 `unwrap_or(0)`를 거쳐 "행 0개"가 되고, 양쪽 빈 md5가 같아 통과합니다. `:308-312,331-335`의 `f32::max` fold는 NaN을 버립니다(R23). `:285`의 `n4 = streams.len()/T_MAX`는 할당 크기에서 토큰 폭을 역산합니다. `BatchSeam`이 폭을 들고 있어야 합니다 — XS

**주석**
35. `body/prefill.rs:38-41` — "a later round can cut … today every layer runs"는 계획·이력 문장입니다. 날짜·숫자 스캔은 새 파일 전부 깨끗합니다. ds41ced(CED 구현)와 대조가 필요합니다 — XS

**스텝이 load 시점 작업을 하지 않는다**
36. `chain/ffn/batch.rs:727,794` — `LayerWeights::resolve`(이름 해시 조회 약 9개)가 청크마다 두 번 돕니다(배치당 64×43×2회). `chain/attn.rs:1369-1392,1520,1605`, `chain/glue/batch.rs:205-213,253,276,290`, `prefill.rs:651`(`CardStacks::of`)도 같은 부류입니다. 흐름 모델로 보면 union 그늘(94 ms/layer) 아래라 지금 값은 약 0입니다 — 트래커에만 올리고 라운드는 열지 않습니다 — S
37. `hybrid.rs:1455-1459`(레이어마다 `lists` Vec, 이미 fixup 목록), `body/prefill.rs:269`(배치마다 `chunks()` Vec) — XS

### 라운드 절단 제안
- **A. R14 `Window` 타입 라운드**(비트 중립, 한 축): 1-4와 `draft/stage.rs`를 묶고, `DeviceTensor` 창에는 형상을 가진 가드를 붙입니다. 증명은 ptx-scan 표 동일, ds41-step 1169 / skew 2338 / prefill PASSED / e2e, 그리고 디스패치 경로에서 창을 만드는 곳이 바뀌므로 같은 임대 A/B입니다. ds41ced가 착륙한 뒤에 합니다(세 파일을 다시 썼기 때문).
- **B. R14 레이아웃·스크래치 헬퍼 라운드**(비트 중립): 16-21, 11. 증명은 ptx-scan 동일, prefill 비트 동일, 배치 경로를 건드리므로 pp512 A/B입니다.
- **C. R13 기계 라운드**(7″⑦ 방식, 컴파일러가 판정): 28. 여기서 드러나는 dead_code(`FfnBatch::cap` 등)는 같은 라운드에서 지웁니다.
- **D. R8 `*Args` 라운드**: 27. **R12 분할**(25-26)은 B 뒤에 합니다. B가 두 함수를 줄여 놓기 때문입니다.
- **XS fixup5 묶음**: 5-8, 10, 12-15, 23-24, 30-32, 34-35, 37, 22의 NaN 경로 통계.
- **R4 비동기 복사 헬퍼**(9)는 `crates/gpu/src/{graph,lib}.rs`에 닿습니다. 두 리드가 공유하는 파일인지 소유를 먼저 정해야 합니다.
- **필요한 결정**:
  - (a) 29: 배치 경로에는 서비스 카운터가 없으므로 이름 있는 오류를 돌려주는 비용이 사실상 0입니다. 그래도 스텝 경로와 같은 NaN 전달 계약을 유지할지, 오류로 바꿀지 정해야 합니다.
  - (b) 33: 배치 중간 오류에서 poison할지, history를 잘라 되돌릴지.
  - (c) 36: 가중치를 레이어당 한 번 호이스트할지(S), `Derived`로 load 때 해석할지(M).
  - (d) `FfnBatch::serve:584-586`의 8 MB 복사: 근본 원인은 `Tensor2`가 `Vec`을 소유하는 것이고 `model/moe.rs`, 즉 bloomery-ee 트랙입니다. 리드 경계를 넘는 결정입니다.

### lint 예상
기준선 167에서 움직이지 않을 것으로 봅니다(유도한 값이고 측정하지 않았습니다). 새 unsafe 블록은 전부 SAFETY가 있고, R8 자리는 전부 `allow(reason)`이며, R3 대상은 private이라 세지 않습니다. C 라운드가 dead_code를 잠깐 드러낼 수 있지만 같은 라운드에서 지우면 순증은 0입니다.

### 목록 밖에서 본 개선 자리
- `/Users/hckim/repo/bloomery/crates/gpu-deepseek41/src/draft/stage.rs:17-33,125-131` — `View`에 `Drop`이 없어, `f32s`/`u32s`로 창을 만들 때마다(`draft/block.rs:919-972`, 한 번 enqueue에 다섯 개) `Arc` 참조가 하나씩 샙니다. 또 `&self`로 만든 창에 `DerefMut`이 있어 공유 빌림을 거쳐 쓰기가 됩니다 — S. A 라운드에서 닫힙니다.
- `chain/ffn/batch.rs:400-401,730,808` — route 루프가 모든 청크를 끝낸 뒤에 shadow 루프가 돌고, `act_x`가 청크별이 아니라 토큰 수별이라서 `enqueue_norm_quant`가 청크마다 두 번 돕니다. 청크마다 반복되는 작업이라 먼저 값을 유도해 봐야 합니다 — S
- `/Users/hckim/repo/bloomery/crates/gpu/src/hybrid.rs:1273` — `host_mut`가 `&mut H`를 통째로 엽니다. `HostExperts::prepare_union`을 기본 no-op로 두고 `Hybrid::prepare_batch()`로 좁히는 편이 낫습니다 — XS
- `chain/attn.rs:1409-1456` — norm 분기 넷 중 두 쌍의 본문이 같습니다. 배치 이전부터 있던 R14입니다 — XS
- `gate_deepseek41_prefill.rs:168-169`(Split을 두 번 엶), `:395`(`T_MAX_TOKENS` 별칭), `:186`(P=4096의 다음 스텝 비교가 n/a로 빠짐, ORACLE 4097은 이미 fixup 목록); `generate_ds41.rs`의 `feed_mode`를 두 번 계산함(load 줄과 512-516) — XS

opus로 스폰되어 돌았습니다.
