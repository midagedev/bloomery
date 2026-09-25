# quality-ds41bulk — 착륙 `9d61a13`의 R-규칙 리뷰 (2026-09-25 밤, 대기 창 라운드)

리드 메모. 시팅 14가 박스를 쥔 동안 돌린 읽기 전용 라운드(opus)의 보고를 원문 그대로 싣는다. 처분은
`docs/plan-triage.md`의 「ds41bulk 남긴 것」에 있다. 리드 판정 셋: MED 1(리셋 뒤 거짓 `QuantColumn` 폴트 —
`h_all` 전 열 양자화)은 실재하고 후보 1과 같은 수정으로 닫는다(fixup6). MED 2(그룹 커널의 `min` clamp 둘, 지금
호출자로는 도달 불가)는 `ExpertId`로 올린다 — 비트 불변, PTX md5 이동. MED 4(디코드 `ds41_router` md5 이동에
같은 임대 디코드 A/B가 없음)는 리드가 수용한다: 유한 투표는 런치 바닥 아래 명령 몇 개[유도 ≈ 0]이고, 다음
디코드 시팅의 기준선이 그것을 잰다. MED 5(폴트 뒤에도 `select`가 그럴듯한 상위 6 id를 쓴다)는 설계대로다 —
메모리 접근이 정의되고 그 스텝은 `Poisoned`로 버려지며 리드백이 이름 붙은 오류를 낸다; "plausible value"
조항은 호스트로 나가는 값에 관한 것이다. 후보 8(`resolve`가 route와 shadow에서 층당 두 번)은 커밋 메시지의
"층당 한 번"을 정정한다.

---

**quality-ds41bulk 리뷰 보고: 9d61a13, 읽기 전용**

HIGH는 없습니다. 비트와 경계가 걸린 기제를 다섯 가지 확인했고 모두 성립합니다. MED 가운데 리드가 먼저 볼 것은 세 가지입니다. 리셋 뒤 거짓 fault, 두 grouped 커널의 조용한 clamp, 호스트 런처의 R8 위반입니다. 박스 명령, 파일 수정, git 상태 변경은 하지 않았습니다. 판정은 전부 원본 코드를 읽고 내렸습니다.

## 1. HIGH: 없음

아래 다섯 기제를 확인했고 모두 성립합니다.

- **q8_1 평면의 열 레이아웃.** `Q8Act`는 열마다 연속으로 잡힌다(/Users/hckim/repo/bloomery/crates/gpu/src/tensor.rs:277). 그래서 청크별 `dtod`가 `at·q`에 복사하는 방식이 `q3k_row_dot`이 읽는 열 `col0 + slot/6`을 그대로 보존한다.
- **블록 상대 슬롯과 절대 열.** `order`의 슬롯 번호는 블록 기준이다. 반면 `h`와 `down` 스팬은 `at·6`에서 시작한다. 전체 `h_all` 양자화와 down의 `col0 = at·6`이 같은 절대 열을 가리키므로 어긋남이 없다.
- **`start` 표의 낡은 항목.** 버킷의 전문가 수는 `gate.rows/ff`에서, down의 전문가 수는 `down.rows/n`에서 온다. `check_card`가 둘을 같게 강제하므로 낡은 `start` 항목은 읽히지 않는다(/Users/hckim/repo/bloomery/crates/gpu-deepseek41/src/chain/ffn.rs:1081).
- **스트림 순서.** 모든 작업이 엔진 스트림 하나에서 돈다.
  - `routed` 이벤트와 pinned 버퍼는 `serve`의 동기 대기가 끝난 뒤에 다음 기록이 들어간다.
  - 레이어 L의 upload가 레이어 L+1의 download보다 먼저 들어가므로 `host_sum`에 경합이 없다.
- **카드 타이밍 이벤트.** `elapsed_ms`는 두 이벤트를 먼저 동기화한다(cuda-core 0.3.1 `event.rs:134`). 모든 레이어가 매 배치 자기 마크를 기록하므로 낡은 이벤트 쌍을 읽지 않는다.

## 2. MED

1. **리셋 뒤 거짓 `QuantColumn` fault.** /Users/hckim/repo/bloomery/crates/gpu-deepseek41/src/chain/ffn/batch.rs:1426
   - 기제: `h_all`의 `cap·6`열을 전부 양자화한다. 이번 호출이 쓰지 않은 열도 포함된다(HOST 슬롯, 블록 밖 슬롯). 그런데 `reset`은 `FfnBatch`를 지우지 않는다(/Users/hckim/repo/bloomery/crates/gpu-deepseek41/src/body.rs:1958).
   - 발생 순서: 한 호출이 카드 슬롯에 비유한 SwiGLU 값을 쓴다. 이 호출은 이미 fault다. `reset`을 한다. 다음 호출에서 그 열이 HOST이거나 블록 밖이면, 정상 입력인데 fault가 오른다.
   - 영향: 요란한 실패이고, 다른 열의 비트는 바뀌지 않는다. 열마다 따로 양자화하기 때문이다.
   - slot 팔도 청크 안 HOST 슬롯에서 같은 모양을 이미 갖고 있다.
   - 수정: 블록 구간 `at·6..u·6`만 양자화한다. 이것이 개선 후보 1과 같은 작업이다.
2. **조용한 clamp와 무동작.** batch.rs:343과 /Users/hckim/repo/bloomery/crates/gpu/src/q4k_sel.rs:183
   - `j1.min(n_slots)`가 범위를 넘는 run 끝을 조용히 자른다. `j0 > j1`이면 아무 일도 하지 않고 넘어간다.
   - 지금 호출자로는 도달할 수 없다.
   - 수정: 둘 다 `ExpertId`를 올린다. 비트는 그대로이고 PTX md5만 바뀌므로 `gate-ptx-spill` 재핀과 `gate-gpu-ds41-prefill` 1회가 필요하다.
3. **R8, 호스트 런처가 allow로 가려짐.** 7′ 보강에 따르면 호스트 `enqueue_*`는 면제가 아니다. 대상 다섯 곳:
   - /Users/hckim/repo/bloomery/crates/gpu-deepseek41/src/router.rs:701 `enqueue_router_rows`, 인자 11개
   - q4k_sel.rs:369 `enqueue_gemv_q4k_grouped`, 인자 11개
   - batch.rs:1311 `enqueue_grouped_block`
   - batch.rs:1356 `enqueue_grouped_experts`
   - batch.rs:1454 `enqueue_batch_shadow_chunk`
4. **R20, R21, R22: 디코드 경로 변경이 프리필 라운드에 섞임.** router.rs:137
   - 디코드 `ds41_router`의 md5가 바뀌었고 fault 범위가 넓어졌다. 이 변경이 프리필 라운드에 같이 들어왔다.
   - 같은 임대 디코드 A/B가 없다. 예측 비용은 약 0이다 [derived]. 리드가 명시적으로 수용할 항목이다.
5. **-inf bias 의미와 fault 뒤 출력.** router.rs:462, router.rs:37-43
   - 투표가 -inf bias를 거부한다. ik는 이 값을 꼴찌로 정렬한다. -inf로 전문가를 가린 파일은 이름 있는 오류로 거부된다. 지금 파일은 finite라 영향이 없다.
   - fault 뒤에도 `select`는 finite 값 가운데 top-6이라는 그럴듯한 id를 쓴다. places, 버킷, 호스트 union이 readback 전에 그 id를 소비한다.
   - 메모리 접근은 정의돼 있다. 다만 AGENTS의 "never writes a plausible value"와 문구가 충돌한다. 설계 판단은 리드 몫이다.
6. **R6, const assert 부재.**
   - `BUCKET_THREADS`, `BUCKET_EXPERTS`, 리터럴 1024 사이의 관계를 고정하는 assert가 없다(batch.rs:60-61, 188-197, 222). count 패스는 스레드 수가 전문가 수 이상이어야 맞다.
   - `8 = THREADS/32` 매직 넘버도 같다: batch.rs:326, batch.rs:1400, q4k_sel.rs:166, q4k_sel.rs:411.
7. **R14, 사본.**
   - 레이어 인덱스 관용구가 batch.rs에 셋이다(1636, 1705, 1741). 이번 커밋의 `batch_block`이 세 번째를 만들었다. ffn.rs:1015에도 있고, 옆 ffn.rs:897에 `cfg_of`가 이미 있다.
   - `nanos` 사본 둘: batch.rs:1072, /Users/hckim/repo/bloomery/crates/gpu-deepseek41/src/body/prefill.rs:427
   - grouped run 순회 골격이 크레이트 두 곳에 있다: batch.rs:324-365, q4k_sel.rs:163-183
8. **R12, 두 화면 초과 함수.**
   - `enqueue_batch_shadow_chunk` 164줄(batch.rs:1454-1617). 이전 `enqueue_batch_shadow`는 156줄이었다.
   - `enqueue_batch_chain` 264줄(prefill.rs:827-1090). 257줄에서 늘었다.
   - `FfnBatch::new` 92줄(batch.rs:756)은 필드 목록이라 경계선이다.
   - 게이트 헬퍼 `router_cases` 111줄(/Users/hckim/repo/bloomery/crates/gpu-gates/src/bin/gate_deepseek41_prefill.rs:411)은 main이 아니므로 7′ 면제 밖이다.
   - 커널 엔트리 `ds41_router` 102줄은 면제다.

## 3. LOW

**R1: 연산 여럿을 담은 unsafe 블록**
- batch.rs:244-247
- batch.rs:1498-1501: `dtod` 두 번과 그 안의 `?`
- router.rs:467-470
- router.rs:540-543

**R3: `# Safety` 절 대신 doc 안에 SAFETY 문단을 쓴 비공개 unsafe fn**
- router.rs:443
- router.rs:483
- batch.rs:1039

**R13: 가시성과 doc**
- `ChunkIo`가 crate 안에서만 쓰이는데 pub으로 재수출된다(ffn.rs:89).
- `BlockIo`(batch.rs:1153)와 `ServeTimes`(batch.rs:745)도 pub이다.
- doc 두 개가 `experts` 필드에 붙었고 `host_x`에는 doc이 없다(batch.rs:669-677).
- `PrefillStats`의 pub 필드들이 doc 하나를 공유한다(prefill.rs:352-369).

**R7, R13: 레버 이중 읽기**
- 두 바이너리가 load 줄을 찍으려고 `BLOOMERY_CARD_EXPERTS`를 다시 읽는다(gate_deepseek41_prefill.rs:274, /Users/hckim/repo/bloomery/crates/gpu-gates/src/bin/generate_ds41.rs:481).
- 그러는 사이 `FfnBatch::experts()`는 호출자가 0이다(batch.rs:862).
- AGENTS.md:558은 버퍼를 만들 때 한 번 읽는다고 적고 있다.

**R11**
- 잘못된 값의 오류가 그 값을 보여 주지 않는다(batch.rs:716-719).

**R28**
- 안전한 `pub fn q3k_row_dot`을 다른 크레이트에서 두 번 새로 부른다(batch.rs:352-353). 기존 부채다.

**R14, 스타일**
- `ChunkIo`를 만드는 코드가 두 번 나온다: batch.rs:1195-1205와 batch.rs:1281-1291
- 똑같은 `BlockIo` 리터럴이 두 번 나온다: prefill.rs:1012-1017과 prefill.rs:1024-1029
- `impl PrefillStats` 블록이 둘이다: prefill.rs:372, prefill.rs:382
- 불필요한 블록 중괄호가 있다: batch.rs:1280-1294

**조용한 실패 규칙의 문면 위반. 전부 도달 불가다.**
- `n_expert != N_EXPERT`이면 return한다: router.rs:280, router.rs:389
- `col >= cols`이면 return한다: batch.rs:164
- `enqueue_ns`의 `saturating_sub`가 회계 역전을 가린다: prefill.rs:375

**결합**
- 범용 크레이트가 V4.1 상수 `UNION_MAX_COLS`에 assert로 묶였다(tensor.rs:201).

**주석 규칙은 깨끗하다.**
- 추가 주석에 날짜, 이슈 번호, 측정 ms가 없다. grep으로 확인했다.
- 추가된 unsafe 블록마다 SAFETY가 붙어 있다. 스캔으로 확인했다.

## 4. 커널 쪽 점검

| 커널 | 스레드→데이터 매핑 | 쓰기 고유성 | 배리어 | smem / 블록 |
|---|---|---|---|---|
| `ds41_router_scores` router.rs:266 | 그리드 48·⌈t/8⌉. 블록의 타일 `b%48`이 행 8개, 워프 w가 행 `8·tile+2w`와 그다음 행, 그룹 `b/48`이 토큰 8개. 레인 k = lane+32i, 누산기 16개 | lane 0이 `probs[(t0+c)·384+r]`에 쓴다. 고유하다 | 없음. return은 블록 균일이다. 누산기 16개 전부 `lane==0 && c<cols` 가드 전에 reduce하므로 잘린 마지막 그룹도 안전하다 | 0 / 128 |
| `ds41_router_pick` router.rs:372 | 블록 하나가 토큰 하나. 스레드가 전문가 `tid+128i` 3개를 싣는다. 워프 0이 선택 | `ids/weights[6·token+s]` 고유 | 414의 배리어에 전원 도달한다. 389의 return은 배리어 앞이고 블록 균일이다 | 3,096 B / 128 |
| `ds41_router`(디코드, 수정) router.rs:137 | 기존과 같다. `publish`와 `select`만 공유로 바뀌었다 | 기존과 같다 | 177, 192, 214. 196의 return은 공유 `LAST`에 대해 균일이고 배리어 뒤다 | 3,100 B / 128 |
| `ds41_card_buckets` batch.rs:199 | 블록 1개에 1024스레드. 스레드 tid < n_experts가 슬롯 전체를 세고, 스레드 0이 직렬 prefix, 이어서 order 패스 | count는 tid, start는 스레드 0, order는 run이 서로 겹치지 않아 고유 | 234, 254. 조기 종료 없음 | 4,096 B / 1024 |
| `ds41_expert_gate_up_grouped` batch.rs:306 | 블록이 전문가 `e = b/⌈ff/8⌉`, 워프가 행 r. run의 각 슬롯에 대해 `q3k_row_dot` 2회와 reduce 2회 | `h[slot·rpe+r]`. `order`가 카드 슬롯의 순열이면 고유하다. 버킷이 그걸 보장한다 | 없음. return과 슬롯 분기는 워프 균일이다 | 0 / 256 |
| `q4k_gemv_grouped` q4k_sel.rs:147 | 위와 같은 매핑, rpe = n_embd | `y[slot·rpe+r]` 고유 | 없음. 워프 균일 | 0 / 256 |
| `ds41_expert_gate_up_tok`(수정) batch.rs:130 | 기존과 같고 fault 인자만 늘었다 | 기존과 같다 | 없음. return은 워프 균일 | 0 / 256 |

## 5. 개선 후보. 보고만 하고 손대지 않았다

1. **XS~S** | batch.rs:1426 | 블록 구간만 양자화하도록 열 수를 덮어쓰는 인자를 둔다. MED 1도 함께 닫힌다.
2. **M** | batch.rs:1207-1215, 1481-1501 | route가 계산한 q8 평면을 버리고 shadow가 청크마다 다시 계산한다. route에서 블록 평면을 한 번에 쓰면 P = 512에서 레이어·배치당 약 64 런치가 빠진다 [derived]. shared expert가 `q3_all` 위의 `Q8Act` 뷰를 읽어야 한다.
3. **S** | batch.rs:1400 | 그리드가 n_experts × ⌈ff/8⌉ 블록인데 대부분 `j0 == j1`로 바로 끝난다. 버킷이 비지 않은 전문가 목록을 내도록 한다.
4. **M** | batch.rs:342-365 | 워프 하나가 run을 직렬로 돈다. 뜨거운 전문가의 run 길이가 꼬리를 정한다. 가중치 디코드가 슬롯마다 반복되므로 gather 열 방식의 다열 코어를 쓸 수 있다 [derived, 측정 없음].
5. **XS~S** | batch.rs:235-253 | 버킷의 스레드 0 직렬 prefix를 블록 scan으로 바꾼다. union 그늘 아래라 지금 가치는 약 0이다 [derived].
6. **XS** | batch.rs:218 | 레인 가드 없이 모든 스레드가 fault를 올린다.
7. **XS** | batch.rs:822-828 | slot 팔에서도 블록 버퍼를 할당한다. 커밋 자체의 증가분이 약 89 MB다.
8. **XS** | batch.rs:1193, 1268 | `LayerWeights::resolve`가 route와 shadow에서 한 번씩, 레이어당 두 번 돈다. 커밋 메시지는 한 번이라고 적었다.
9. **XS** | batch.rs:91-97 | places가 `id ≥ n_expert`를 조용히 HOST로 보낸다. 기존 코드다. 규칙대로면 `ExpertId`를 올려야 한다.

## 6. 모델

opus로 돌았다(Opus 5.5).
