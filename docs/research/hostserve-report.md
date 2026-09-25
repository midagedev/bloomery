# hostserve 보고 (V4.1 호스트 union의 배치 이음매 — 빌린 x, 호출별 제외 집합, 그룹 꼬리, 2026-09-26 새벽)

> 리드 메모(2026-09-26). 아래는 `hostserve` 라운드(opus)의 보고 원문이다. 세 가지가 한 커밋으로 착륙했다.
> - xview: union이 `Tensor2View`(빌린 열 창)를 받아 `FfnBatch::serve`의 층·배치당 x 복사 10.5 MB와 `x_host` 힙이 사라졌다.
>   비트는 직전 리드 배치 로그의 prefill 게이트 md5 줄·hybrid `n_l=` 줄과 같다(절대 기준 대조). 예측 Δ는 층·배치당
>   −0.45…−0.83 ms, P = 512 프롬프트의 0.39–0.71 %라 자보다 작아 시간 A/B는 없다[유도].
> - 제외 집합: `Hybrid::serve_batch(.., exclude, ..)`, 거부 넷(스펙의 셋 + 카드 id — 리드가 받음, 조용한 누락을 막는다).
> - 그룹 꼬리: `UnionScratch::new_tail(embd, ff, groups, slots)`, 예산을 넘는 계획은 512열 호출로 잘라 돌고 `cut_calls`에 센다.
>   카드 쪽 단계 ②(hoststream, 69)는 사용자 결정 ⓐ대로 꼬리를 on 팔(밴드)에서만 쓴다.
> - 5절의 개선 지점: `hybrid.rs:1586` 직렬 유한성 검사와 `serve_batch_one`의 호출당 `Vec`은 R1 uniondispatch로,
>   `copy_ns`·`stat prefill split` 항목·카드 cap 상수는 69 쪽 파일이라 69에게 넘겼다.

---

hostserve 라운드 보고입니다(보고 대상 `bloomery-ee`). 세 가지를 모두 만들었습니다. 내 소유 게이트는 전부 초록이고, FAIL-first 세 건은 각각 빨강을 확인한 뒤 복원했으며 복원은 md5로 확인했습니다. git 상태는 건드리지 않았고, 사용자의 직접 입력은 없었습니다.

## 1. 변경 파일

모두 `/Users/hckim/repo/bloomery-hostserve` 아래이고, `git diff --stat` 기준 8개 파일, +892 / −159입니다.

- `crates/model/src/ops.rs`
  - `Tensor2View<'a>`를 새로 만들었습니다. `Tensor2::view()`와 `From<&Tensor2>`로 한 블록을 빌려줍니다.
  - `GroupInput::{Cols, Quantized}`, `QuantizedCols::fill/is_of`가 view를 받습니다.
  - `SlotBlock`을 넣었습니다. 입력은 view로, par 블록은 헤더만 참조합니다.
  - 양자화 공유 판정을 `same_block`(데이터 주소와 모양)으로 바꿨습니다.
- `crates/model/src/lib.rs`: `Tensor2View`를 re-export합니다.
- `crates/model/src/moe.rs`
  - union 입구들(`experts_union_into`, `HostLayer::experts_union_into`, `check_union_call`, `serve_union`)이 view를 받습니다. `impl Into<Tensor2View>` 형태라 `bench_v41_host`처럼 `&Tensor2`로 부르던 호출자는 그대로 컴파일됩니다.
  - 꼬리용 상수 셋, `UnionScratch::new_tail`, `max_slots()`, `cut_calls()`, `UnionPlan::fits`를 추가했습니다.
  - 예산을 넘는 계획은 512열 호출로 잘라 돌리고, 계획이 선 뒤의 본체는 `serve_planned`로 분리했습니다.
- `crates/model/tests/union.rs`
  - 꼬리 scratch 거부 4건을 추가했습니다.
  - `hw_union_tail_matches_batches_v41`를 추가했습니다. 두 층 × 두 deferral 팔에서, 꼬리 케이스 두 개는 한 번에(whole) 돌고, store를 넘는 케이스와 block을 넘는 케이스는 잘려서(cut) 돕니다.
- `crates/gpu/src/hybrid.rs`
  - `HostExperts::experts_union_into`가 `x: Tensor2View`를 받습니다.
  - `Hybrid::serve_batch`에 `exclude: &[u32]`를 추가하고, HOST 결정에서 제외합니다.
  - `HybridStats::batch_excluded_slots`, 이름 붙은 거부 넷을 내는 `check_exclude`를 추가했고, 거부 경로를 view에 맞췄습니다.
- `crates/gpu-gates/src/bin/gate_hybrid.rs`: Stub을 view에 맞췄고, `exclusion_arm`을 추가했습니다(집합 넷 × 40열, 700열 한 건, 거부 넷).
- `crates/gpu-deepseek41/src/chain/ffn/batch.rs`
  - 호출부만 바꿨습니다. `x_host` 복사 대신 `Tensor2View::new(&self.host_x[at*n..u*n], n, u - at)?`를 넘기고, `serve_batch(.., &[], ..)`로 부릅니다.
  - 읽는 곳이 없어진 `x_host` 필드(cap × n_embd f32, 10.5 MB 힙)와 그 초기화를 지웠고, 구조체 문서 한 줄을 고쳤습니다.
- `crates/gpu-deepseek41/src/chain/ffn.rs`: `Ds41Host::experts_union_into` 시그니처와 import만 바꿨습니다.

`HostExperts`의 나머지 구현체 `PlanHost`(`crates/gpu/src/arch/deepseek2/mod.rs:106`)와 `Zero`(`crates/gpu/src/model/launcher.rs:456`)는 `experts_union_into`를 재정의하지 않고 트레이트 기본값(이름 붙은 거부)을 씁니다. 그래서 고칠 것이 없었습니다.

## 2. 게이트·증명: 명령과 실제 출력

박스 명령은 전부 `BLOOMERY_REMOTE='~/repo/bloomery-hostserve' ./tools/box.sh "if flock -n /root/bloomery-cpu.lock true && test ! -e /root/bloomery-ee-hold; then <cmd>; else exit 199; fi"`로 감쌌고, 막히면 60초 간격으로 최대 30회 재시도했습니다. 처음에 홀드 파일 때문에 6회 대기했습니다.

**정적 검사**

- box-gc: 처음과 끝에 `bash tools/box-gc.sh --kill`을 돌렸고, 둘 다 `box-gc: found 0 process(es) under /root/repo/bloomery-hostserve/target`.
- check: `cargo check --workspace --all-targets --features gpu,bloomery-gpu-gates/deepseek41,bloomery-gpu-gates/vision` → `Finished \`dev\` profile … in 17.51s`, rc=0.
  - lock-back은 같은 임대 조건으로 박스의 `~/repo/bloomery-hostserve/Cargo.lock`을 읽어 비교했습니다. `box lock == worktree lock`이고 `git diff --stat Cargo.lock`는 비어 있습니다.
- lint(같은 feature로 clippy, 출력 전체를 파일로 받음): `grep -c '^warning:'` = **144**, rc=0. 마지막 테스트 편집 뒤 다시 돌려도 144였습니다.
  - 내 파일에 걸린 경고는 `crates/model/src/ops.rs:2977` `collapsible_if` 하나인데, `HEAD:2848`에 있던 같은 코드가 내 추가분만큼(+129줄) 밀린 것입니다. 새로 생긴 경고는 0개입니다.
- fmt-check: `cargo fmt --all -- --check` rc=0 (박스에서 두 번).
- Mac에서: `check-comments: ok`, `check-arch: ok`, `self-test: ok (92 gate recipes, 0 failures)`, `check-recipes: ok`.

**gate-union**: `bash tools/gate.sh --release -p bloomery-model --test union -- --ignored --nocapture --test-threads=1`

최종 트리 출력:
```
layer=0 down=Q5_K k=4096 slots=3042 union=192 max_cols_per_expert=35 experts_across_batches=192 defer: whole diff_cells=0 prepass: whole diff_cells=0 PASS
layer=0 down=Q5_K k=1236 slots=888 union=189 max_cols_per_expert=13 experts_across_batches=157 defer: whole diff_cells=0 prepass: whole diff_cells=0 PASS
layer=0 down=Q5_K k=4096 slots=24576 union=384 max_cols_per_expert=90 experts_across_batches=384 defer: cut diff_cells=0 prepass: cut diff_cells=0 PASS
layer=0 down=Q5_K k=4096 slots=4096 union=4 max_cols_per_expert=1044 experts_across_batches=4 defer: cut diff_cells=0 prepass: cut diff_cells=0 PASS
(layer=2 down=Q4_K 네 줄, 모두 같은 값으로 PASS)
PASSED: union tail — one call over 4096 and 1236 columns through a tail scratch of 8192 slots equals the calls one batch at a time, … a routing past the budget, or with an expert past a batch's columns, runs cut into batch calls, counted
test result: ok. 4 passed; 0 failed; …   gate-union rc=0
```
꼬리 scratch 거부 네 줄은 모두 이름이 붙어 있습니다:
- `… expected [5120, 8], got [5120, 0]`, `… got [5120, 9]` (묶음 수)
- `… UNION_BATCH_SLOTS..=EXPERTS_INTO_MAX · columns slots: expected [4096, 8192], got [4095, 2]`, `… got [8193, 2]` (슬롯 예산)

**나머지 게이트**

- gate-ops(`--lib --test ops`): `test result: ok. 33 passed`, `8 passed`, rc=0.
- gate-alloc: `step 1: 603 allocator calls` … `step 7: 603`로 PIN 603과 같고, rc=0.
- gate-gpu-hybrid: `exclusion verdict PASS`, 마지막 줄 `gate_hybrid: PASS — … a batch service leaves exactly its exclusion set's slots off the host, over one batch or more, and refuses a malformed set by name.`, rc=0. 제외 줄의 예:
  - `set=[9, 12]: … excluded 35 (want 35) host 79 (want 79) stub experts 79 PASS`
  - `cols=700 set=[9, 12]: … excluded 527 (want 527) host 1569 (want 1569) … PASS`
- gate-gpu-ds41-prefill(기본 팔, `BLOOMERY_MODEL=deepseek41`): `PASSED: gate_deepseek41_prefill`, rc=0.

**비트 동일성을 절대 기준으로 확인한 것 (xview의 증명)**

gate-union은 상대 비교라서 이것만으로는 비트가 고정되지 않습니다. `experts_union_into`와 `experts_into`(`matmul_q_one` → `SlotBlock::Input(x.view())`, ops.rs:2853) 양쪽이 모두 바뀐 코드를 거치기 때문입니다. 그래서 기준 로그 두 개와 직접 비교했습니다.
- prefill 게이트의 16개 케이스 md5 줄(ring, state, shadow, rows, keys; P=1…4096, split, rollback 포함)이 직전 착륙 배치 로그 `…/scratchpad/wave-m6/lead-q3fix2-gb/g-gate-gpu-ds41-prefill.log`과 diff가 비었습니다(`md5 lines identical`). 배치 경로와 한 열 경로를 둘 다 덮습니다.
- gate_hybrid의 결정적 `n_l=` 줄(structure, eager_vs_replay, per_layer band, e2e disagree/sigma)이 같은 배치의 `g-gate-gpu-hybrid.log`와 같습니다. 다른 것은 `host_tier … (3090 gate load, not a timing)` 두 줄뿐입니다.

**FAIL-first 세 건** (모두 빨강 → 복원 → md5 일치 → 초록)

| 건 | 변이 | 빨강 줄 | md5 (원본 → 변이 → 복원) |
|---|---|---|---|
| (a) 제외 | 호출마다 host 슬롯을 딱 하나 더 제외로 셈: `let mut extra = !exclude.is_empty();` + guard `\|\| std::mem::take(&mut extra)` | `set=[9, 12]: … excluded 36 (want 35) host 78 (want 79) stub experts 78 FAIL`, `[8,10,11,13,14,15] … excluded 80 (want 79) FAIL`, `cols=700 … excluded 528 (want 527) FAIL`, `gate_hybrid: FAILED: one or more checks above did not pass`, rc=1, `Compiling bloomery-gpu` | 8fb848ae… → aad660f5… → 8fb848ae… |
| (b) 폭에 따른 합 순서 | `serve_planned`의 pool 가지에서 `k > UNION_MAX_COLS`일 때 열의 리스트를 뒤집어 합산 | whole 케이스 `diff_cells=211779 / 69087 / 213645 / 69437 FAIL`, `8 (layer, k, arm) tail calls differ …`, `test result: FAILED. 3 passed; 1 failed`, rc=101, `Compiling bloomery-model` | 2bb6811e… → c45209e4… → 2bb6811e… |
| (c) block 한계 | `fits`에서 전문가당 block 조항을 뺌 | `panicked at crates/model/src/ops.rs:222:9: Tensor2::set_cols: 995 columns of 2304 past the block's capacity 1179648`, rc=101, `Compiling bloomery-model` | 2bb6811e… → 7d5ba183… → 2bb6811e… |

- (a)의 예측이 맞았습니다. 빈 집합과 host 전부를 제외한 집합은 PASS를 유지했습니다(더 뺄 슬롯이 없음). 복원 후 `Compiling bloomery-gpu`, `exclusion verdict PASS`, rc=0.
- (b)에서 cut 케이스는 예측대로 `diff_cells=0 PASS`였습니다. 잘린 조각은 512열 이하라 뒤집히지 않기 때문입니다. 복원 후 4 passed, rc=0.
- (c)는 원래 cut 케이스가 store 한계(24,576 > 8,192)만 건드리고 block 한계는 건드리지 않아서 뒤늦게 추가했습니다. block 한계를 넘는 케이스(열마다 전문가 1개, 네 전문가 중 하나)를 게이트에 넣고 이 FAIL-first로 확인했습니다. 복원 후 최종 실행은 위 gate-union 출력입니다.

## 3. 예측과 결과 (Part A)

**자원 시간선.** 단위는 P=512의 layer-batch 하나이고, S14에서 잰 값입니다.
- 호스트 호출 스레드는 enqueue 18.8 + wait 11.7 + copy 0.77 + union 72.3 ms를 직렬로 씁니다. 카드 작업(union 전 route 30.4 ms, shadow)은 union 그늘에 들어갑니다.
- 그래서 벽시계는 호출 스레드 시간의 합입니다. copy는 route 대기와 union 사이에 끼어 있고 어떤 그늘에도 들어가지 않습니다.

**A1 xview**
- 바이트: layer-batch마다 512 × 5120 × 4 B = **10.49 MB**를 읽고 같은 양을 씁니다. 따로 `x_host` 힙 10.49 MB도 없어집니다.
- 시간: S14에서 잰 `copy_lb`가 0.75–0.83 ms입니다. 단일 스레드 memcpy로 약 13 GB/s입니다 [실측에서 유도].
- Δ: **−0.45…−0.83 ms/layer-batch** [derived].
  - 상한은 copy 전체입니다.
  - 하한은 x를 처음 읽는 곳이 바뀐 몫(최대 약 0.3 ms)을 뺀 값입니다. 첫 읽기는 호출 스레드의 유한성 검사 `non_finite(x.col(j))`(hybrid.rs:1586)입니다. 예전에는 방금 L3에 쓴 x_host를 읽었고, 이제는 DMA가 쓴 host_x를 DRAM에서 읽습니다.
- 프롬프트 기준: P=512, 40층이면 18–33 ms로, 4,659 ms(109.9 tok/s)의 **0.39–0.71 %** [derived]입니다. 자(4회 ±1 %)보다 작으므로 시간 A/B는 할 가치가 없습니다.
  - 확인 방법은 다음 시팅의 `stat prefill split`에서 `copy` 칸이 약 0으로 떨어지는지 보는 것입니다. 지금 `copy_ns`는 view 생성 시간만 잽니다.
- union 안에서 x를 읽는 곳과, 모두 같은 값을 같은 순서로 읽는 근거:
  - `moe.rs:1228` `check_union_call`: 모양만 봅니다.
  - `moe.rs:1323` cut 경로: 같은 메모리의 부분 slice입니다.
  - `moe.rs:1372` `xq.fill` → `ops.rs:692/717`: 열마다 `x.col(t)`를 양자화합니다. `Tensor2::col`(ops.rs:206)과 같은 slice 식입니다.
  - `moe.rs:1402–1403` `Quantized/Cols` → group_core
    - 모양 검사(ops.rs:1263–1279)
    - `SlotBlock::Input`(ops.rs:1303) → claim된 양자화가 `s.x.col(t)`(ops.rs:2068)와 `q.x.col(t)`(ops.rs:2257)를 읽습니다.
    - `xq.is_of`(ops.rs:1394)와 공유 판정 `is_input`(ops.rs:1439)은 `same_block`(ops.rs:305)으로 판정합니다. 한 블록의 view는 주소와 모양이 같으므로 예전의 `&Tensor2` 주소 비교와 같은 쌍이 공유합니다.
  - `moe.rs:1393/1421`의 채움값은 `[..2n]`, `[..n]`으로 잘려 나가 읽히지 않습니다.
  - `hybrid.rs:1586/1609/1611`(거부 경로)도 같은 값을 읽습니다.
  - par 블록은 claimant가 raw 포인터로 쓰기 때문에 `&[f32]`로 만들지 않고 `SlotBlock::Par(&Tensor2)`로 헤더만 참조합니다(쓰기 중인 메모리에 공유 참조가 겹치는 UB를 피하려는 것).
  - view가 덮는 범위 `&host_x[at·n..u·n]`, ne1 = u − at는 예전 `extend_from_slice`가 복사하던 범위와 정확히 같습니다.

**A2 제외 집합**
- 층의 host 시간 = Σ_{e∈H∖S} t(m_e), t(m) = max(W, a + c·m)입니다. 집합 S에 든 항만 정확히 빠집니다 [derived].
- 통계 `batch_excluded_slots` = Σ_{e∈S} m_e입니다. 스트리밍되는 전문가는 뜨거운 쪽(m ≥ 4.28)이므로 Δunion ≈ a·|S| + c·excluded_slots로 대조할 수 있습니다.
- 예: 16개 × 30열이면 0.45 + 11.03 ≈ 11.5 ms/layer-batch [derived, 예시].
- 게이트에서 개수는 정확했습니다(35/79, 79/35, 527/1569).

**A3 꼬리 호출**
- recal 곡선(꼬리 전문가 114개, host rank 200–313, 512열당 λ 1.4 → 0, 포아송)으로 계산했습니다 [derived]. 층당, 한 그룹 기준:

| G | 512열 호출 G번 | 한 번에 | 절감 | layer-batch당 |
|---|---|---|---|---|
| 4 | 26.61 ms | 12.98 ms | 13.63 ms | 3.41 ms |
| 8 | 53.21 ms | 19.22 ms | 33.99 ms | 4.25 ms |

- union 72.3–76.1 ms 대비 4.5–5.9 %입니다.
- 상한 (G−1)·W는 전문가당 379.2 / 884.8 µs이고, 층당 그룹 합으로 43.2 / 100.9 ms입니다.
- 폭 독립성의 근거: 청킹(`UNION_CHUNK`)과 타일 묶음은 누가 한 dispatch를 같이 쓰는지만 바꿉니다. 각 (행, 열)의 dot, 열 양자화, 리스트 순서대로 0부터 하는 열 합은 그 열과 그 열의 리스트만의 함수입니다. FAIL-first (b)가 게이트가 이 조건을 보고 있다는 증거입니다.

**tail scratch — 무엇을 골랐고 몇 바이트인가 (리드 요청)**

`UnionScratch::new_tail(embd, ff, groups, slots)`로 정했습니다.
- block은 한 batch 폭(512열)으로 두고, store는 `slots`개로 예산을 잡습니다.
- 예산을 넘는 계획이나 512열을 넘는 전문가가 있으면 512열 호출로 잘라 돌립니다. 비트는 같고 `cut_calls()`에 셉니다.
- 권장 예산은 `S = UNION_BATCH_SLOTS = 4096`입니다.
  - 꼬리 전문가는 정의상 바닥 아래(512열당 m < 4.28)이므로 batch당 114 × 4.28 ≈ 488 슬롯 이하, G=8이면 3,904 < 4,096입니다.
  - 기대값은 batch당 Σλ ≈ 80, G=8에서 약 640 슬롯이라 6배 여유가 있습니다.

| scratch | block | store | xq | plan | 합계 |
|---|---|---|---|---|---|
| 오늘의 batch `new(512)` | 197.13 | 83.89 | 3.03 | 0.15 | **284.20 MB** |
| `new_tail(G=4, 4096)` | 197.13 | 83.89 | 12.12 | 0.59 | 293.73 MB |
| **`new_tail(G=8, 4096)` (권장)** | 197.13 | 83.89 | 24.25 | 1.18 | **306.45 MB** |
| `new_tail(G=8, 8192)` (게이트) | 197.13 | 167.77 | 24.25 | 1.18 | 390.33 MB |
| 전문가당 최악 배치로 4096열 | 1577.06 | 671.09 | 24.25 | 1.18 | 2,273.6 MB |

- 마지막 줄은 h3tile-b가 말한 ≈2.25 GB와 일치합니다.
- `new_tail` scratch 하나가 batch 호출도 처리합니다. 512열 계획은 구성상 항상 맞습니다. 그래서 카드 쪽은 batch scratch를 교체하면 되고, 늘어나는 것은 **+22.25 MB**(xq와 plan)뿐입니다.
- h3tile-b의 174–229 MB는 block을 꼬리 폭에 맞춘 셈입니다. 그렇게 하면 batch용 scratch(284 MB)가 따로 필요해 합계가 더 커집니다.

**API 시그니처**
```rust
// ops.rs
#[derive(Clone, Copy, Debug)] pub struct Tensor2View<'a> { ne0: usize, ne1: usize, data: &'a [f32] }
pub fn new(data: &'a [f32], ne0: usize, ne1: usize) -> Result<Tensor2View<'a>, ModelError>; // ne0·ne1 ≠ len → 이름 붙은 Shape 오류
pub fn ne0(self) / ne1(self) / data(self) -> &'a [f32] / col(self, i) -> &'a [f32];
impl Tensor2 { pub fn view(&self) -> Tensor2View<'_> }   // len ≠ ne0·ne1이면 이름 붙은 panic
// hybrid.rs
pub fn serve_batch(&mut self, layer: usize, x: Tensor2View<'_>, ids: &[u32], weights: &[f32],
                   exclude: &[u32], out: &mut [f32]) -> Result<(), GpuError>;
pub batch_excluded_slots: u64; // HybridStats
// moe.rs
pub const UNION_TAIL_MAX_GROUPS: usize = 8;
pub const UNION_TAIL_MAX_COLS: usize = 4096;
pub const UNION_BATCH_SLOTS: usize = 4096;
pub fn new_tail(embd: usize, ff: usize, groups: usize, slots: usize) -> Result<UnionScratch, ModelError>;
pub fn max_slots(&self) -> usize;
pub fn cut_calls(&self) -> u64;
pub fn experts_union_into<'x>(…, x: impl Into<Tensor2View<'x>>, …)  // HostLayer도 같은 형태
```
- `exclude`는 순증가해야 하고 전부 host 전문가여야 합니다. 거부는 넷입니다: `expert 9 after 12: the set must ascend` / `expert 9 twice` / `expert 16 past the layer's 16 experts` / `expert 3 is on the card`.
- 넷째(카드 id) 거부는 스펙의 세 가지 밖입니다. 조용히 무시하면 집합 항목이 소리 없이 사라지므로 넣었습니다. 받을지는 리드가 판단해 주십시오.
- UNION_MAX_COLS에 대한 예외는 전역 상한을 올리지 않고, scratch 타입(`new_tail`)이 열 상한 G×512를 가지는 방식입니다.

## 4. 하지 않은 것, 그리고 step ②가 필요로 하는 것

**하지 않은 것**
- prefill 루프 배선(카드 쪽 step ②)은 스펙 밖이라 하지 않았습니다.
- 제외 통계는 `HybridStats`에만 있습니다. `stat prefill split` 줄은 HybridStats를 싣지 않고, 그 파일(`prefill.rs`, `generate_ds41`)은 내 경계 밖이라 추가하지 않았습니다.
- xview의 시간 A/B는 자보다 작아서 하지 않았습니다.

**step ②에 필요한 호출**
1. scratch: `Ds41Host`의 `UnionScratch::new(embd, ff, UNION_MAX_COLS)`(ffn.rs:1478, :1516)를 `UnionScratch::new_tail(embd, ff, G, UNION_BATCH_SLOTS)`로 바꿔야 합니다. 바꾸기 전에는 G×512열 호출이 `check_union_call`에서 "host union: at most the columns the scratch was made for"로 거부됩니다.
2. batch g마다: `hybrid.serve_batch(layer, Tensor2View::new(&group_x[g·512·n..(g+1)·512·n], n, 512)?, ids_g, w_g, &sorted(S ∪ T), out_g)`
3. 그룹의 마지막 route 뒤: `hybrid.serve_batch(layer, Tensor2View::new(&group_x[..G·512·n], n, G·512)?, ids_all, w_all, &sorted(S ∪ W), tail_out)`. 그다음 열마다 out_g + tail_out을 더합니다. 두 집합 모두 순증가로 병합해야 합니다.
4. 버퍼: `FfnBatch::host_x`는 한 batch만 담습니다(cap ≤ UNION_MAX_COLS, batch.rs:762). 꼬리의 view는 G×512열이 연속으로 놓인 pinned 그룹 버퍼에서 나와야 합니다. batch마다 따로 두면 모으는 복사(10.5 MB × G)가 필요해 xview로 없앤 비용이 돌아옵니다.
5. 비트: warm_j + tail_j는 한 호출의 리스트 순서 합과 부동소수 순서가 다릅니다. 꼬리를 쓰는 곳에서는 "bits = steps"가 깨집니다.
   - 결정 ⓐ에 따르면 `BLOOMERY_HOSTSTREAM=1`(band와 count pin)에서만 허용됩니다.
   - step ①(`BLOOMERY_PREFILL_GROUP`, bits = steps)에서 꼬리를 쓰려면 두 호출의 down을 보관해 두었다가 슬롯 순서로 한 번에 더해야 합니다. 보관량은 batch당 512 × 6 × 20,480 B = 62.9 MB, G=8이면 503 MB [derived]입니다. 이번 라운드에서는 만들지 않았습니다.
6. `tensor.rs:201`: 꼬리에는 필요한 것이 없습니다. host만 쓰는 호출이고 UNION_MAX_COLS는 512 그대로이므로 assert가 유지됩니다.
   - step ①이 카드 버퍼를 G×512로 키울 때는 `Q8ACT_MAX_SLOTS`, batch.rs:762의 cap, prefill.rs:68의 `T_MAX`를 host 상수 대신 카드 쪽 상수와 n_used 기준으로 다시 써야 합니다. 지금은 host의 `UNION_MAX_COLS`와 리터럴 6에 묶여 있습니다.

**엔진 비교 — 따른 설계와 갈리는 곳**
- llama.cpp `ggml_compute_forward_mul_mat_id`(ggml/src/ggml-cpu/ggml-cpu.c:1550): src1을 제자리에서 wdata로 한 번 변환하고(`from_float(… src1->data …)`, :1626/:1639) 전문가별 행 매핑을 만듭니다(:1659). 활성의 float 복사는 없습니다.
- ik(ggml/src/ggml.c:18452, `iqk_mul_mat_moe` :18594): 구조는 같습니다. SER이 `i02 < 0 || i02 >= n_as`를 건너뛰고(:18550) 그 행을 0으로 채웁니다(:18532–18535).
- exllamav3: pinned 플래그를 쓰는 worker handoff(exllamav3_ext/cpu/moe_handoff.h:5–6, 가중치 스테이징 :49–50)가 있고, 전문가 범위 샤드를 커널 안에서 마스킹해 범위 밖 선택은 정확히 0이 됩니다(modules/block_sparse_mlp.py:939–940).
- mistral.rs: GGUF CPU MoE는 전문가 가중치를 f32로 역양자화한 뒤 `UnquantLinear::gather_forward`로 처리합니다(mistralrs-quant/src/gguf/cpu.rs:3, :74–82). 제외 기능은 없습니다.
- 따른 것:
  - xview는 ggml과 ik의 방식입니다(제자리 읽기 + 호출당 한 번 변환). 우리의 `xq.fill`이 그들의 wdata `from_float`, 우리의 열 리스트가 그들의 행 매핑에 해당합니다.
  - 제외는 두 방식 모두 따르지 않았습니다. 그들은 건너뛰기를 라우팅 데이터 안에(−1이나 범위로) 넣고 0 행이라는 정의된 출력을 냅니다. 우리는 라우팅을 건드리지 않고 검증된 집합을 옆으로 넘기며, 해당 슬롯은 아예 계산하지 않고 셉니다. "조용한 실패 금지" 규칙 때문입니다.

## 5. 스펙 밖에서 본 개선 여지 (보고만, 손대지 않음)

- `crates/gpu/src/hybrid.rs:1586`: `non_finite`가 512 × 5120 값을 호출 스레드에서 직렬로 훑습니다. xview 뒤에는 이것이 DRAM의 host_x를 처음 읽는 곳이라 약 0.45–0.55 ms/layer-batch가 듭니다 [derived]. `xq.fill`이 이미 pool에서 모든 값을 읽으니, 열별 유한 플래그를 돌려받게 하면 약 −0.4 ms/layer-batch입니다. 크기 S.
- `crates/gpu-deepseek41/src/chain/ffn/batch.rs`의 `copy_ns` 문서와 `take_serve_times`, 그리고 prefill split의 `copy` 칸(prefill.rs describe, generate_ds41): 이제 view 생성만 잽니다. 이름을 바꾸거나 없애야 합니다. XS, fixup6 쪽 파일입니다.
- `stat prefill split`에 `batch_excluded_slots` 델타와 scratch의 `cut_calls`를 찍으면, 시팅마다 Σt(m)을 union ms와 바로 대조할 수 있습니다. XS.
- `crates/gpu/src/hybrid.rs` `serve_batch_one`의 `let lists: Vec<&[(u32, f32)]> = …collect()`: 호출마다 512항목 Vec을 할당합니다(decode step 밖이라 gate-alloc 대상은 아님). 재사용 버퍼로 바꿀 수 있습니다. XS.
- batch.rs:762, prefill.rs:68, tensor.rs:201: 카드 cap이 host 상수와 리터럴 6에 묶여 있습니다(4절 6번). S.
- ik ggml.c:18550: n_as 이상인 id를 SER 센티널처럼 조용히 0 행으로 처리합니다(assert가 :18551에서 주석 처리됨). 업스트림 견고성 후보지만 우선순위는 낮습니다. XS.

## 6. 모델

opus(Opus 5.5)로 스폰되어 실행했습니다.
