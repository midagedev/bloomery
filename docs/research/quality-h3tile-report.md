# quality-h3tile 리뷰 보고 (h3tile-a `ffb6b41`의 R-규칙 리뷰, 2026-09-26 새벽)

> 리드 메모(2026-09-26). 아래는 `quality-h3tile` 라운드(opus, 읽기 전용, 박스·git 변경 없음)의 보고 원문이다. 대상은
> h3tile-a 착륙 `ffb6b41`의 세 파일이다. HIGH는 0건, MED는 7건이고 전부 비트를 바꾸지 않는다. 처분은 이렇다.
> - MED 1·2·3·5·6·7과 LOW(`narrow`가 `BLOOMERY_POISON` NaN 채움을 뺀 사본, `Cells`의 `unsafe impl`에 `T: Send` 경계
>   없음, reason 없는 allow 등)는 트리아지 「fixup (ee) — quality-h3tile」 목록으로 간다.
> - MED 4(`Cells` 불변식 여섯 사본)와 범위 밖 10(공유 셀 타입 하나로 `Cells`·`SharedOut`·`SharedQuantCols` 통합)은
>   h3tile-b가 큐를 엔진으로 옮길 때 한다.
> - 범위 밖 7(`ops.rs:2124-2138`: 좁은 디스패치에서 x 슬롯을 참가자 한 명이 직렬로 양자화하고 31명이 스핀)은 union
>   성능 항이라 트리아지에 따로 둔다.
> - `r8_gate_up` 관찰 (a)–(e)는 h3tile-b-design 보고와 대조할 입력이다. 특히 (c) unit당 17.6 KB 조각 흐름의 램프가
>   a로 간다는 가설(δ ≈ 2.4 µs/unit[유도]), 그리고 (e) unionr8은 익명 메모리 사본을, unionq는 파일 매핑을 읽는다는
>   차이다. (e) 때문에 rig-log 09-25#h3tile-ab의 "같은 큐 안에서 커널만 바꾼 비"에 단서를 덧붙였다.

---

**quality-h3tile 보고**

`ffb6b41`의 세 파일을 읽기 전용으로 검토했습니다. 빌드·게이트·박스 명령과 git 상태 변경은 한 번도 하지 않았습니다. 로컬에서 python으로 산술 확인을 한 번 했습니다(레포·박스와 무관한 wrap 탐색). 검토 중 main이 `823aac8`(q3fix2)로 움직였는데, 검토한 세 파일과 `ops.rs`는 `ffb6b41` 이후 그대로입니다. `moe.rs`만 +4/−3 바뀌었고, 인용한 moe 줄 번호는 현재 트리 기준입니다.

결론부터: HIGH는 없습니다. MED 7건은 모두 비트를 바꾸지 않아 재핀할 게이트가 없습니다.

### 0. 종이 위 분해와 자원 시간선

**산술로 닫은 항**
- **i16 합:** `t`의 i16 레인 하나는 maddubs 4개 × 곱 2개로 곱 8개(u ≤ 7, |q| ≤ 128)입니다. 그래서 |t| ≤ 7,168이고 + 쪽은 7,112입니다.
  - 폴드는 `madd`로 바로 i32에 더하고, `v4 = 4s`는 절댓값이 128 이하입니다. 곱을 8개보다 많이 더하는 i16 경로는 없습니다.
  - 포화까지 여유는 4.5배입니다.
- **i32 부분합:** 가장 큰 경우가 ≈ 14·2^18 + 2^19 + 458,752 ≈ 4.65M으로 2^23보다 작습니다. 주석의 "±2^23"은 맞습니다.
- **`check_r8`의 `nb * 880`·`nb * 296` wrap:** 워크스페이스 프로필에 `overflow-checks`가 없어서 release에서는 wrap합니다.
  - 그래도 두 값을 동시에 할당 가능한 크기로 만드는 k는 없습니다. 격자로 확인하면 짧은 해는 (37, 55), 즉 nb = 1의 배수뿐이고, 296·nb가 작게 wrap하는 창에서는 880·nb가 약 0.97·2^64에 놓입니다.
  - 따라서 wrap을 통한 범위 밖 읽기는 없습니다.
- **두 점 적합의 항 귀속:** t = 160·(a + c·m)을 m = 8, 16에서 풀면 run 수가 m/8에 정확히 비례합니다.
  - 그래서 run마다 드는 항(check_r8, 커널의 super-block당 unpack, 열마다 도는 `check_row`)은 모두 c로 갑니다. a에는 unit당·호출당 항만 남습니다.
  - V4.1은 ff = 2,304, embd = 5,120입니다[유도: facts.md의 shexp down 6,635,520 B, `token_embd` 284,416,000 B, vocab 129,280]. expert당 unit은 288개, 호출당(4 expert) 1,152개입니다.
  - unit당 오버헤드 δ는 expert당 a를 9δ µs 올립니다. unionq의 Δa 21.4 µs는 δ ≈ 2.4 µs/unit에 해당합니다.

**자원 시간선** (팔 `4x8u0.125`, 층 호출 하나 = expert 4개, m = 8, rig-log 적합 기준)
- **호스트 CPU(32 스레드):** union 831 µs, unionq 967 µs, unionr8 706 µs.
- **DRAM:** expert당 16.77 MB[유도] × 4를 W 바닥으로 읽으면 506 µs입니다.
- **PCIe·NVMe·카드:** 0입니다.
- **벽시계:** max(CPU, DRAM)이고, 세 팔 모두 CPU가 결정합니다.
  - unionr8은 expert당 DRAM 바닥보다 약 50 µs 위에 있고, 그 사이를 막는 항이 a = 59.4입니다. 큐의 고정비는 다른 자원 그늘에 가려지지 않은 벽 항입니다.
  - expert당 126.4 µs 아래로 내려가면 m = 8에서 CPU를 더 줄여도 이득은 0입니다.

### 1. HIGH — 없음

확인한 것:
- **읽기 범위:** 커널의 모든 읽기가 그룹 super-block [0, 880)과 열 블록 [0, 296) 안에 있습니다(§4). nb개 분량은 `check_r8`이 보장합니다.
- **`copy_nonoverlapping`(`bench_v41_host.rs:1197`):** at + 8 ≤ m·ff = `data.len()`입니다. row-lane 모드에서는 ff % 8 ≠ 0을 시작 시점 repack이 `RowGroup`으로 거부합니다.
- **데이터 경쟁 없음:**
  - unit (i, g)는 `fetch_add`로 한 번만 잡히고, 블록 2i+m의 (열, 행 8g..8g+8) 셀을 혼자 씁니다.
  - xq는 열마다 claimer 하나가 씁니다. 전체 슬라이스는 `done.fetch_add(Release)` 후 `load(Acquire) == k`를 본 뒤에만 만들어집니다. RMW 사슬이라 release sequence가 k개 쓰기를 모두 잇습니다.
  - 넓은 호출에서 xq는 채움 디스패치의 join 뒤에 읽힙니다. gate_up·downs·out도 모두 join 뒤에 읽힙니다.
- **`bench:1177` `let c = run[t.min(run.len() - 1)];`:** 숨은 클램프가 아니라 정의된 패딩입니다. `:1180`에서 `&acols[..run.len()]`로 잘려 커널에도 쓰기에도 가지 않습니다.
- **헬퍼 규칙:** 커널이 부르는 헬퍼는 `r8_sub_block` 하나이고 `#[inline(always)]`(`lib.rs:1543`)입니다.
- **시간 구간 할당:** `union_r8_call`은 할당하지 않습니다. plan·narrow는 용량 안에서 움직이고, 배열은 스택에 있습니다.

### 2. MED

1. **k = 0에서 이름 없는 panic**
   - 위치: `crates/qdot/src/lib.rs:569` `.chunks_exact(group_bytes)`
   - 기제: k = 0은 `UnalignedK`를 통과하고 need = 0이라 빈 버퍼가 길이 검사도 통과합니다. 그다음 `group_bytes = 0`에서 std의 "chunk size must be non-zero" panic이 납니다.
   - 입력: `repack_q3k_r8(&[], 8, 0, &mut [])`
   - 같은 k = 0을 `dot_q3k_r8_cols`는 `Ok`(값 0)로, `dot_row`는 `Ok(0.0)`로 받습니다.
   - 수정: `if row_bytes == 0 { return Ok(()); }`. 크기 XS.
2. **Column 모드(unionq)의 조용한 잘림과 무검사 타입**
   - 위치: `bench_v41_host.rs:1130` `let groups = ff / R8;`, `:1201` `let rb = w.len() / ff;`, `:1206` `GgmlType::Q3_K,`
   - 기제: ff % 8 ≠ 0이면 끝의 행들을 계산하지 않고, 이전 토큰의 셀이 그대로 SwiGLU로 갑니다. unionq에서는 타입과 k = embd를 아무도 확인하지 않습니다.
   - 벤치 전체로는 `check_r8_arms`가 union과 비교해 잡지만, 함수 자체는 조용합니다. V4.1 파일로는 촉발되지 않습니다.
   - 수정: 진입에서 이름 붙은 거부. 크기 XS.
3. **R28 — 단언 없는 원시 쓰기**
   - 위치: `bench:1147` `.map_or(std::ptr::null_mut(), |t| t.data.as_mut_ptr()),`
   - 기제: `:1197`, `:1217`의 원시 쓰기가 두 조건에 기댑니다. `outs.len() == 2 * mats.len()`이 아니면 `dst[2i+m]`이 null 패딩입니다. 각 `outs[p]`가 `{ff, plan.cols(a+p/2).len()}`로 narrow되지 않았으면 `data.len()`을 넘어 씁니다.
   - 둘 다 단언이 없습니다. 유일한 호출자는 지킵니다.
   - 수정: 진입에 `assert_eq!` 두 줄. 크기 XS.
4. **R2·R4 — 같은 불변식을 여섯 곳에서 반복**
   - 위치: `bench:695` `struct Cells<T>(*mut T);`
   - 기제: 호스트 원시 포인터 여섯 곳(`:892`, `:1000`, `:1094`, `:1162`, `:1197`, `:1217`)이 "열 c는 이 참가자 것"을 매번 다시 적습니다. 세 곳이 넘으면 타입으로 올려야 합니다.
   - `ops::SharedOut`(`ops.rs:377`)·`SharedQuantCols`에 이은 세 번째 사본입니다.
   - 크기 S. h3tile-b가 엔진으로 옮길 때 하나로 합치면 됩니다.
5. **R8·R16 — reason 없는 allow**
   - 위치: `bench:921`, `:1113` `#[allow(clippy::too_many_arguments)]`
   - 인자가 10개·9개로 R8 한도 8을 넘는데, allow에 `reason`이 없습니다. 크기 XS.
6. **R14 — 스칼라 unpack의 두 번째 사본**
   - 위치: `lib.rs:1427` `fn q3k_scales6(blk: &[u8]) -> [u8; 16] {`, `:1443` `fn q3k_codes`
   - `dot_q3k_q8k_scalar`의 `:1139` `let word = |i: usize| match i {` / `:1145`, 그리고 `:1162`의 사본입니다.
   - 정수 연산이라 합쳐도 비트가 같고, 기존 스칼라 게이트가 다시 증명합니다. 크기 S.
7. **R6 — 레이아웃 불변식에 const assert 없음**
   - 위치: `lib.rs:1412` `const R8_CODES: usize = R8_SCALES + 96;`
   - 커널의 경계 증명(마지막 바이트 879)이 `R8_CODES + 8 * R8_PAIR == Q3K_R8_BLOCK`에 기대는데 `const { assert! }`가 없습니다. 크기 XS.

### 3. LOW (규칙별)

- **R1**
  - `lib.rs:1601` `unsafe {`: 본체 전체를 감싸는 형태지만 크레이트 기존 형태와 같습니다.
  - `:1614` `let base = group.as_ptr().add(Q3K_R8_BLOCK * sb);`, `:1660`에는 자기 SAFETY 줄이 없습니다.
  - `bench:706`/`:708` `unsafe impl<T> Send/Sync for Cells<T> {}`: T 경계가 없습니다(`T: Send`가 맞습니다).
- **R2/R6:** `lib.rs:1454` `fn r8_pack_block(rows: &[&[u8]; Q3K_R8_ROWS], dst: &mut [u8])`는 호출자가 이미 `&mut [u8; 880]`을 들고 있습니다.
- **R5:** `bench:937` `*entry = (l.experts[s].id as u32, w);`는 호스트에서 좁히는 `as`입니다.
- **R11:** `lib.rs:68` `RowGroupBytes { have: usize, need: usize },`는 세 버퍼 중 어느 것인지 말하지 않습니다.
- **R12:** `bench:958-1100` `union_r8_call`이 142줄로 두 화면을 넘습니다.
- **R14**
  - `bench:712` `fn narrow`는 `set_cols`(`ops.rs:220`)의 사본인데 `BLOOMERY_POISON`(`:228`) NaN 채움을 뺐습니다. poison 모드에서 놓친 셀이 NaN 대신 0.0이 됩니다.
  - `bench:164` `const UNION_INLINE_COLS: usize = 8;`는 `moe.rs:966`의 사본입니다.
  - 테스트: `tests/qdot.rs:3139-3147`·`:3167-3172`가 `:2847-2855`·`:2862-2867`을 복사했습니다.
- **조용한 실패(잠재)**
  - `bench:2674` `_ => Ok(GateUpTile::Column),`
  - `bench:1127-1128`의 이른 `return Ok(())`: 지금은 도달하지 않지만, 도달하면 claims를 건너뜁니다.
- **스텝 안 로드타임 작업:** `bench:771` `.position(|x| x.id == e as usize)` (무시할 만한 크기).
- **문서:** `bench:110-111` "`unionq` against `union` the dispatch"라고 하지만 양자화 배치도 다릅니다(`ops.rs:2124-2128`).
- **테스트:** `tests/qdot.rs:3224`은 "before any kernel runs"라고 하지만, 거부 뒤 `out`이 그대로인지는 보지 않습니다.

### 4. 커널 점검 `dot_q3k_r8_tile_avx2::<C>`

**레인 사상**
- **코드 벡터:** W_f(f ∈ 0..8)의 바이트 b는 행 b/4, sub-block 2p + f/4, 값 4(f%4) + b%4입니다.
  - j = 2p는 `[a&7, a>>3&7, (a>>6&3)|(c>>4&4), b&7]`, j = 2p+1은 `[b>>3&7, (b>>6&3)|(c>>5&4), c&7, c>>3&7]`입니다.
  - `srli/slli_epi16` 12곳의 이웃 바이트 번짐은 모두 마스크 밖으로 떨어집니다.
- **maddubs:** i16 레인 k는 행 k/2입니다. 네 개를 합치면 레인 k는 그 행의 값 8개입니다.
- **scale:** `v`의 레인 k는 s(행 k/2, 2p + k%2)입니다. `dup_lo`/`dup_hi` 뒤 dword r은 [s_{r,j}]×2이고, `madd(s,t)`의 dword r은 s_{r,j}·Σ16 u·q입니다.
- **폴드:** dword r은 4(s_{r,2p}·bsum_{2p} + s_{r,2p+1}·bsum_{2p+1})입니다.
- **나머지:** `d`의 레인 r은 행 r이고, `acc[c]`의 레인 r은 `out[c][r]`입니다.
- **부동 단계:** 한 열 커널 `:1120` `acc += dcol * d * hsum_i32(sumi) as f32`와 연산·순서가 같습니다.

**바이트 대조** (그룹 super-block 하나)
- 커널 읽기: [0,16) d, [16,112) l0·l1·hb, p마다 [112+96p, +96) a·b·c. 마지막 바이트는 879입니다.
- `r8_pack_block`은 880바이트를 정확히 한 번씩 씁니다(16 + 96 + 768). 비트 필드까지 일치하고, `r8_scale`/`r8_code`도 같은 역사상입니다.
- 열 블록에서는 [0,4), [8,264), [264,296)만 읽습니다.

**정렬 fault:** 가능한 명령이 없습니다. 모두 `loadu`, `read_unaligned`, `storeu`이고, VEX 메모리 피연산자는 정렬을 요구하지 않습니다.

**C = 8 레지스터**
- 열 루프를 가로질러 살아 있는 값: `sumi[8]` + `w[4]` + `s` + `v`(`:1688`까지) = 14개, 여기에 임시 2–3개로 16–17 ymm입니다.
- `acc[8]`까지 더하면 22개 + 임시입니다. super-block당 한 번만 만지는 `acc`가 스필 대상입니다.
- 라운드가 보고한 "17 > 16"과 맞습니다. `:1677-1678`의 b·c 재로드가 두 개를 더 살려 두지 않게 하는 장치입니다.

### 5. 게이트 강도

`hw_q3k_r8_tile_matches_dot_row`가 덮는 것:
- C 전부: c = 1..=8, 시작 슬롯 전부, canary 열로 C 너머 쓰기 검출.
- i16 경계 그룹: u = 7 × 전부 −128 열이 −7,168, 전부 +127 열이 +7,112.
- scale −32·31.
- 전부 ±127 열(−128도 포함).
- 실제 V4.1 행 32그룹(nb 20), V2-Lite(nb 8), 합성(nb 2).

**거부 경로:** `check_r8`의 여섯 거부와 repack의 네 거부 모두 테스트가 닿습니다.

**닿지 않는 것**
- repack의 k = 0 panic(MED 1).
- `lib.rs:604` `if !has_features(GgmlType::Q3_K) {` 분기(박스에서는 도달 불가).
- `RowGroupBytes`의 Display.

**약점**
- 합성 그룹은 행마다 d가 같아서(`:3158` `0x3C00`, `:3159` `0x3800`) d 레인 치환은 V4.1 행만 잡습니다.
- 그룹 시작이 늘 16바이트 정렬이라, 정렬 로드로 바꾸는 편집을 결정적으로 잡지 못합니다.
- 스칼라 미러는 V4.1 32그룹 중 2그룹에서만 돕니다.

### 6. 범위 밖 개선 후보 (보고만, 손대지 않음)

**랜딩 라운드가 올린 여섯 개**
1. `bench:626` `let lists: Vec<&[(u32, f32)]> = ...` — **확인.** union 팔의 시간 구간 안에서 층마다 할당합니다. 효과는 약 0.01 %입니다. XS.
2. `bench:2565` — **rows > 8에서만 확인.** x 양자화, down pre-pass(`ops.rs:2195`), 열 합, stash 디스패치를 빠뜨립니다. m = 8에서는 정확합니다. XS.
3. `moe.rs:1233-1244` — **부분 확인.** 한 청크 안에서는 슬롯을 공유합니다(`ops.rs:1345-1352`). 다만 좁은 호출은 청크마다 x를 다시 양자화합니다. 벤치 팔에서는 0입니다. S.
4. `lib.rs:481` `nb = check_row(w, wrow.len(), a.len(), k)?;` — **확인.** `compute_rows`(`ops.rs:1714`)가 행·run마다 부르므로 rows × cols번 검사합니다. S.
5. `lib.rs:2726` — **확인.** 살아 있는 값이 2C + 5개라 C = 8에서 21개 + 임시입니다. M.
6. `ops.rs:2348` — **Q4_K/Q5_K는 확인.** Q3_K는 해당 없습니다(C + 6 ≈ 14로 레지스터에 들어갑니다). S.

**내가 찾은 것**
7. `ops.rs:2124-2128`·`:2138` `wait_ready`: 좁은 디스패치의 x 슬롯 하나(≤ 8열)를 참가자 한 명이 직렬로 양자화하는 동안 31명이 스핀합니다. 열 단위 claim으로 바꾸면 정지가 k·t_q에서 약 t_q로 줄어듭니다[유도]. S.
8. `set_cols`를 `pub`으로 열어 `narrow` 사본을 없앱니다(poison 동작도 복원). XS.
9. moe의 `UNION_INLINE_COLS`를 공개합니다. XS.
10. `ops`에 제네릭 공유 셀 타입 하나를 두고 `Cells`·`SharedOut`·`SharedQuantCols`를 통합합니다. S.
11. `lib.rs:1127`이 `q3k_scales6`/`q3k_codes`를 쓰게 합니다(MED 6의 처분). S.
12. `bench:523`의 `Tally::rows`가 첫 시간 토큰 안에서 자랍니다(기존, 모든 팔 공통). XS.

**`r8_gate_up`에서 본 것** (h3tile-b 입력용 관찰입니다. 원인 판정이 아니고 측정하지 않았습니다.)
- (a) **전역 카운터:** `bench:1132` `let next = AtomicUsize::new(0);` 하나를 모든 참가자가 unit마다 잡습니다(호출당 1,152회). 잡기 전에는 매번 `stop`(`:1133`, `:1163`)을 읽는데, 둘을 가르는 정렬이 없습니다. 엔진 쪽은 `ops.rs:2373` `#[repr(align(64))]` `struct Lane`로 레인별 카운터를 둡니다.
- (b) **출력 false sharing:** unit 출력은 열마다 32 B라, 이웃 unit(연속 claim이니 다른 스레드)이 64 B 줄 하나의 두 절반을 씁니다. 비용은 소스만으로는 유도할 수 없습니다. 이 항은 열 수에 비례하므로 적합에서 c 쪽으로 갑니다. 16행 unit이면 구조적으로 사라집니다.
- (c) **가중치 흐름 조각화:** 스레드의 흐름이 17.6 KB 조각(8행 × 2,200 B)으로 gate·up 영역을 오갑니다. union은 스레드당 약 1.27 MB 연속 레인입니다. unit당 램프는 a로 갑니다. δ ≈ 2.4 µs 중 claim RMW(수백 ns급)가 설명하는 몫은 작습니다.
- (d) **`QuantClaims` 장벽**(`:896`): k ≤ 8에서 인코드 한 번 분량이라 작습니다.
- (e) **unionr8의 a가 unionq보다 7.8 µs 큰 이유는 소스에 없습니다.**
  - unit당 코드는 unionr8 쪽이 더 가볍습니다.
  - 두 팔은 읽는 메모리부터 다릅니다. unionr8은 익명 메모리 사본(`bench:657` `vec![0u8; ...]`, 고정되지 않은 scoped 스레드가 first-touch)을, unionq는 파일 매핑을 읽습니다.
  - 그래서 페이지 크기(THP)가 다르고, "unionr8 대 unionq = 커널만"(`:110`)은 엄밀하게는 참이 아닙니다.

### 7. 모델

opus(Opus 5.5)로 스폰되어 돌았습니다. 어드바이저 호출은 시간 초과로 응답이 없었습니다.
