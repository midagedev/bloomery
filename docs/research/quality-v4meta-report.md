# quality-v4meta 리뷰 보고 (착륙한 `8bfa287`의 R-규칙 검토, 2026-09-25 밤)

> 리드 메모(2026-09-25). 아래는 `quality-v4meta` 라운드(opus, 정적 독해만, 박스 없음)의 보고 원문이다. 처분: HIGH 없음.
> MED 넷(V4 파일의 V4.1 전용 텐서가 이름으로 거절되지 않고 `Unread`/engram 역할로 조용히 실림 — `PlanInputs::read`의
> 전체 거절이 걷히는 순간 조용한 수용; `LayerKind`의 불가능 조합(R5) → enum 셋; `row`·`fail_if_bad` 셋째 사본(R14);
> 거절 판정 `CardFormat::of` 대 배치 `of_routed`)은 v4 사슬의 다음 고리 앞에 픽스업 카드 `v4quality`(S)로 묶는다 —
> 트리아지 참조. XS 항목(문구·죽은 이름·문서)은 그 카드에 같이 싣는다. "거절 9묶음 194건"은 10묶음으로 정정했다.

---

== /Users/hckim/.claude/projects/-Users-hckim-repo-bloomery/02d044bc-8d2a-4055-91fa-4ae541a4916f/subagents/agent-aquality-v4meta-7b4407a5efc822f6.jsonl ==
# quality-v4meta 리뷰 보고 (착륙한 `8bfa287`, 정적 독해만)

**결론.** 스펙 정의대로의 HIGH는 없습니다. 조용히 기본값으로 바뀌는 누락 키는 없고, 체인·디스패치 코드에서 `Model`로 가지를 치는 곳도 없습니다. 가장 무거운 발견은 커밋 메시지와 코드가 어긋나는 한 곳입니다. 메시지는 V4 파일의 V4.1 전용 텐서를 이름으로 거절한다고 적었지만, 코드는 `indexer.attn_k`와 `engram_embd`만 거절합니다. 지금은 `PlanInputs::read`가 파일 전체를 거절하므로 닿지 않습니다. 거절 목록이 줄어드는 순간 조용한 수용이 됩니다.

## 발견 표

| 규칙 | 위치 | 내용 | 심각도 | 크기 |
|---|---|---|---|---|
| 조용한 실패 | `hparams.rs:765-808`, `roles.rs:44-50` | V4 파일의 `exp_probs_b_vl`은 `Role::Unread`로 무시된다. `engram_wkv`·`engram_k`·`engram_q`는 `EngramDense`/`EngramGain`으로 카드에 올라가고 읽히지 않는다. 커밋 메시지의 "refused by name"이 구현돼 있지 않다 | MED (v4body가 "a model without engram sites"를 걷는 순간 HIGH) | S |
| 조용한 실패 | `hparams.rs:863-870`, `:976-983` | 짝 없는 텐서를 보지 않는다. ratio 0 층의 `attn_compressor_gate`·`_ape`, V4.1의 `indexer_compressor_gate`/`_ape`/`_norm`처럼 `_kv` 없이 온 텐서가 역할 접두사로 카드에 실린다 | LOW (기형 파일만 해당) | S |
| R5 | `hparams.rs:346-378` | `LayerKind`의 `stream: Option<Stream>`와 `dense: Option<DenseStream>`가 불가능한 `(Some, Some)`을 허용한다. `ratio()`는 그때 stream을 조용히 고른다. 불가능 조합이 둘 더 있다: `hash_routed && !routed`, `index_keys && index_compressor.is_some()` | MED (다음 층 종류와 함께 퍼진다) | S |
| R14 | `deepseek4_meta.rs:52,61` / `ds41_meta.rs:71,131` / `qwen3moe_meta.rs:99,114` | `row`·`fail_if_bad` 사본이 셋째가 됐다. `tests/common`이 이미 있다 | MED | S |
| 일관성 | `place.rs:110` vs `placement.rs:366` | 거절 판정은 `CardFormat::of`를 쓰고 배치는 `of_routed`를 쓴다. IQ3_XXS/MXFP4는 배치상 host에 남고 qdot가 둘 다 돌리므로, 이 항목은 정확성 결함이 아니라 성능 한계의 거절이다 | MED | XS |
| R9/R11 | `hparams.rs:133,488`, `plan.rs:92`, `place.rs:120` | "engram 없음" 거절 주인이 셋이다. 키 두 개, 오류 열거형 두 개(`ModelError`/`PlacementError`)다. `RowTypes::engram()`은 모델 필드가 없어 `Model::Deepseek4`를 박아 둔다 | LOW | XS |
| 테스트 계약 | `deepseek4_meta.rs:385` | `capacity()`가 플래너의 바닥 산술을 테스트 안에서 다시 유도한다. 플래너 식이 바뀌어도 이 값은 안 움직인다 | LOW–MED | S |
| 테스트 계약 | `deepseek4_meta.rs:460-509` | `hw_deepseek4_plans` 하나가 두 계약(배치 산술, 거절 목록)을 진다 | LOW | XS |
| R13 | `names.rs:213` | `indexer_compressor_norm`의 호출자가 0이다 | LOW | XS |
| R13 | `arch/mod.rs:28,32` | `DEEPSEEK4`·`deepseek41_model`이 `pub`인데 크레이트 안에서만 불린다 | LOW | XS |
| 문서 | `placement.rs:148-149` | `gathered_rows` 문서가 TokenEmbedding·EngramTable만 적는다. `HashTable`도 이제 `Some(1)`이다 | LOW | XS |
| 문구 | `placement.rs:769` Display | "194 feature(s)"가 실은 (층, 기능) 쌍 수다. 목록은 10묶음이다 | LOW | XS |
| 문구 | `hparams.rs:699` | 1차원 압축기 텐서가 `unwrap_or(0)`로 폭 0이 된다. 거절은 되지만 사유가 "projects to 0 values"로 틀린다 | LOW | XS |
| 모양 | `body.rs:2479,2485,2493` | `hp.engram()?`를 세 번 부르고 둘은 층 루프 안이다. 적재 시점이라 비용은 아니다 | LOW | XS |
| 모양 | `params.rs:125-141` | `ImageDims::of`가 engram 없는 모델에 `map_or(0)`을 쓴다. 뜻은 맞고 아래층도 0을 처리한다. 다만 의미 없는 `engram_row_bytes` 인자를 받는다 | LOW | XS |
| check-arch 머리말 | `check-arch.sh:36` | 머리말은 "목록을 손으로 적지 않는다"인데 `deepseek4`가 손으로 적혀 있다 | LOW | XS |

## 1. 거절 함수와 도달 시점

**`Hparams::read` 안, 적재 시점:**
- `arch/mod.rs:32` `deepseek41_model`은 모르는 아키텍처 문자열을 거절한다.
- `hparams.rs:765` `no_foreign_tensors`는 V4.1에서 `attn_compressor_ape`·`indexer_compressor_kv`·`ffn_gate_tid2eid`를, V4에서 `indexer.attn_k`만 거절한다.
- `:792` `no_engram`은 V4에서 `engram_embd`와 `engram.layer_ids` 키를 거절한다.
- `:847` `own_streams`는 V4 층 표를 본다. ratio 0 층의 압축기·인덱서, 빠진 kv·gate, 인덱서 없는 index 압축기, 스트림 종류당 비율 하나, 셋째 비율을 거절한다.
- `:814` `compressor_of`는 한 행도 두 행도 아닌 투영 폭을 거절한다.
- `:1103` `hash_site`는 해시 표와 `hash_layer_count`를 양방향으로 대조하고, 라우터 없는 층의 표를 거절한다.
- `:1136` `collapse`는 `output_hc_*` 유무를 모델과 대조한다.
- `:738` `segments`와 `:136` `RowTypes::read`.

**`PlanInputs::read` 안, 적재 시점:** `place.rs:77` `unimplemented()`가 10개 기능 문자열로 194건을 낸다. 해시 3, ape 41, overlap 21, 인덱서 자기 압축기 21, dense 20, iq3_xxs 42, mxfp4 43, 모델 전역 3이다.

**배포 진입점은 전부 `PlanInputs::read`를 거친다.** `body::open`(`body.rs:122`), serve, chat, `generate_ds41`, 적재 게이트가 그렇다. `Hparams::read`를 직접 부르는 V4.1 게이트 바이너리는 그보다 먼저 V4 파일을 거절한다. `expect_arch`가 `"deepseek41"`과 문자열을 비교하거나(`lib.rs:127`), `split.architecture()`를 직접 검사하기 때문이다.

**뒤에 있는 방어선도 전부 적재 시점이다.** `Body::derive`(`body.rs:1881`)는 `open` 뒤에서만 닿는다. `Planner::from_file`(`plan.rs:92`)과 `engram()` 접근자 20곳은 Body 조립 때 불린다. 한 곳은 첫 prefill의 `make_batch`에서 불리지만, 같은 오류가 `body.rs:2100`에서 먼저 난다. **스텝 때에만 터지는 거절은 없다.**

**문구 하나.** "hyper-connection head" 항목은 머리만 말한다. `Collapse::Head`는 모든 서브층이 자기 mix로 접는다는 뜻(ik `dsv4_hc_lag` false)도 담는다. v4hc가 머리만 구현하고 거절을 걷을 위험이 있다.

## 2. `engram()` 24곳, 모양별

| 모양 | 곳 | None일 때 |
|---|---|---|
| `?` in `Result` | lib 7곳(`body.rs:2479/2485/2493`, `glue.rs:624`×2, `glue.rs:1213`×2), 게이트 바이너리 13곳(engram 1, skew 1, chain_glue 4, chain_attn 1, step 6) | 이름 붙은 오류. 오류는 `GpuError::Model`로 가며 어느 자리가 물었는지는 잃는다(R11) |
| `map_or(0, …)` | `params.rs:139` | 정의된 기본값 0. 뜻은 맞고 `ImageLayout`과 `rows()`(`params.rs:735`)가 0을 처리한다. 지금 호출자는 모두 그 전에 `?`로 거절돼서 사실상 도달 불가다. 그 사실을 타입이 들지는 않는다 |
| `ok_or_else` | `plan.rs:92` | 이름 붙은 오류지만 주인이 다르다(키 `engram.max_ngram_size`) |
| `is_none()` | `place.rs:120` | 의도한 거절 항목 |
| `expect` | `ds41_meta.rs:391` "a V4.1 file has engram sites" | 불변식 문구가 맞다 |

**스텝마다 도는 조회는 없다.** 모든 lib 자리가 생성자 안에 있다. `Glue::with_rows`, `StepRows::open`, `engram_row_bytes`(open 시점, 그리고 메모이즈되는 `make_batch`)다.

## 3. 타입 모양 (R5, R8)

`LayerKind`는 불 묶음과 Option 쌍의 혼합이라 R5의 사례입니다.

- **권장 모양은 세 가지 enum이다.** `enum Attends { Window, Selected(Stream), Whole(DenseStream) }`, `enum Ffn { Dense, Routed, HashRouted }`, `enum IndexKeys { None, Projected, Compressed(Compressor) }`. 이러면 불가능한 조합과 `ratio()`/`compressed()`의 우선순위 가지가 사라진다.
- **`Compressor { gated, ape, overlap }`는 그대로 둔다.** 세 독립 플래그이고 조합이 실제 파일에 있다. `ape && !gated`를 거절할지는 v4comp 몫이다.
- **usize 단위 인자 수.** `rows`·`cols`·`bytes`·`pos` 이름의 usize 인자는 새로 0개다. 대신 u64 `row` 인자 둘이 "행당 값 수"를 뜻해 단위가 이름과 어긋난다(`compressor_of`, `kv.rs` `state_bytes`). `hash_site`는 bool 위치 인자가 둘이다.
- **R8.** 7개를 넘는 인자는 없다. 최대는 `layer_kinds`의 6개다.

## 4. 죽은 경로

- **`unimplemented()`**는 `PlanInputs::read`와 `deepseek4_meta.rs:482`가 부른다. 출력은 `PlacementError::Unimplemented`의 Display(`placement.rs:769`, `unimplemented_list`)다.
- **`describe()`**의 호출자는 `deepseek4_meta.rs:46`뿐이다. 통합 테스트용 `pub`이지만 `doc(hidden)`이 아니라 R27 위반은 아니다. R13 관점의 기록만 남긴다.
- **죽은 것은 `names::indexer_compressor_norm` 하나다.**
- `Model::name`·`DEEPSEEK4`는 크레이트 안에서만 불린다. 커밋의 "Left" 항목(`bloomery-decode.rs:109`, `engine.rs:53`)이 예정된 호출자라 죽은 것이 아니라 대기 중이다.

## 5. 중복

- **플랜 핀.** V4.1 플랜 핀은 `tests/common`이 아니라 `tests/placement.rs:37-235`에 인라인 상수로 있다. 파일별 `FilePins`(MIXED/PUBLIC)라는 표 모양은 이미 거기 있다.
- **구조체 겹침.** V4 `HostPin`은 V4.1 것과 필드 네 개가 같다. V4 `CardPin`은 V4.1 것의 부분집합에 `capacity`가 붙은 모양이다. `plan_rows`(`deepseek4_meta.rs:395-458`)는 `placement.rs`의 `pins()`를 얇게 다시 쓴 사본이다.
- **줄 수는 약속하지 않는다.** `Model` 키 표로 합치려면 V4.1 핀이 먼저 공통 파일로 옮겨가야 한다. V4에는 카드 headroom·eligible·n_l 핀이 없어 박스 1회로 새로 재야 한다. 그래서 절감 줄 수를 근거 있게 적을 수 없다.
- **hparams 블록은 복제가 아니다.** ds41_meta와 필드 목록은 같다. 그러나 ds41_meta는 ik 값과 출처 열에 대조하고, deepseek4_meta는 헤더 덤프에 대조하며 오라클이 없다. 같은 모양 다른 계약이다. 표 기반으로 합치려면 모델별 출처 열이 필요하다.
- **V4 전용 계약.** 층 종류, 자기 스트림 소유, 해시 역할, family·type별 바이트, expert 바이트, 거절 목록, capacity다.
- **`deepseek4_pins.rs:169-180`**의 `IQ3_XXS_LAYERS`·`ALL_LAYERS`·`COMPRESSED_LAYERS`는 손으로 적은 배열이다. `SELECTED ∪ DENSE` 등에서 유도할 수 있다.
- **셸 프로필 차이.** 경로 말고도, V4 프로필은 V41_*, DSPARK_MODEL, IK_GPU_ENV, NCMOE/플래그 넷, REF_TOKENS, REF_DUMP_ARGS, step 변형을 뺐다. 같은 값은 IK, LCPP, LCPPBIN, REF_CTX, REF_DUMP_LEASE다. IK·LCPP 기본값은 `deepseek41.sh:148,158`과 같은 리터럴을 따로 적은 것이라 드리프트 여지가 있다. `dump.sh:66-72`는 REF_TOKENS 부재와 `ref_step_variant` 부재를 배열 확장(205행)보다 먼저 이름으로 거절하므로, 이 프로필로 잘못 돌려도 의도한 메시지에서 멈춘다.

## 6. check-arch 규칙 ②

- **다른 규칙은 접기가 필요 없다.** ①과 ④는 디렉터리·모듈 경로에 걸리는데 deepseek4에는 그런 것이 없다. ③은 `general.architecture` 문자열만 본다.
- **접기가 가리는 것도 없다.** `"deepseek4.` 리터럴은 arch 밖에 없다. 유일한 적중은 `hparams.rs:1357`의 테스트 상수다.
- **남는 위험은 런타임 접두사다.** 게이트 바이너리 세 곳이 `Arch::Deepseek41.name()`으로 ik override 키를 만든다(`gate_deepseek41_skew.rs:199`, `step.rs:418`, `index.rs:155`). V4 파일이었다면 틀린 키가 나온다. 지금은 V4.1 전용 게이트라 닿지 않는다.
- **제안.** 접힌 이름을 `tools/ref/models/*.sh` 파일명에서 유도하면 머리말 계약이 다시 참이 된다(XS).

## 7. lint 패턴

추가된 줄 기준 실측입니다.

| 항목 | 개수 |
|---|---|
| 새 `unsafe` | 0 |
| 새 `#[allow]` | 0 |
| 인자 7개를 넘는 함수 | 0 |
| 좁히는 `as` 캐스트 | 0 (넓히는 `usize as u64` 4곳만) |

## 공통 보고 절

1. **변경 파일은 없다.** 조사 라운드라 파일을 고치지 않았다.
2. **게이트는 돌리지 않았다.** 박스·빌드 없이 정적 독해만 했다. 쓴 명령은 `git show 8bfa287`, `grep -rn --include='*.rs' …`, `sed -n`이다.
3. **예측은 해당이 없다.**
4. **확인 못 한 것은 둘이다.** `just lint` 경고 수와 `just check-arch` 종료 코드는 박스 전용이라 재지 않았다. 커밋 메시지는 lint 167과 check-arch 녹색을 적고 있다.
5. **질문 밖의 개선 여지:**
   - `docs/plan-triage.md:54`의 "거절 9묶음 194건"은 핀상 10묶음이다(iq3_xxs와 mxfp4가 따로다). XS.
   - `body.rs:1881` `Body::derive`가 적재 중 `Hparams::read`를 두 번째로 돈다. 이미 있던 것이다. XS.
   - `host.rs:24`는 `kind.routed`만 보고 `hash_routed`를 보지 않는다. v4body가 해시 라우팅을 걷을 때 host 계층이 점수 라우팅으로 조용히 돈다. 스펙에 적어 둘 것. XS.
   - `gpu/src/lib.rs:332`의 `From<ModelError>`는 호출 자리 문맥을 붙이지 않는다(R11). XS–S.
   - `hparams.rs:778` `(c.index_keys).then(|| (true, …))`의 상수 `true` 튜플은 모양이 어색하다. XS.
   - `bind.rs:210` `gathered_or_unread`의 HashTable 누락은 커밋 "Left"에 이미 있다.
6. **모델은 opus(Opus 5.5)다.** 지정 스폰 그대로이고, 판정 전에 어드바이저를 한 번 불렀다.
