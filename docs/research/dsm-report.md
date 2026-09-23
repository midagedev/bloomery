# dsm 라운드 보고: m ≤ 8 열의 커널이 m = 1 합산 순서를 유지함

Q8_0 gemv, Q8_0 heads, Q3_K, Q6_K 모두 m열 런치의 열 c가 그 열을 m = 1로 런치한 결과와 비트 동일합니다. 출력 토큰도 같습니다. V4.1 실제 행에서 m = 1..8 전부 확인했습니다. 기존 ptx-scan 행 중 바뀐 것은 없고, 새 행 3개만 추가됐습니다. 경계 밖 파일은 두 개를 건드렸습니다. `elem.rs`는 4절 첫 항목에, `GemvOut::TokenMajor`는 그 둘째 항목에 적었습니다. 커밋은 하지 않았고 git 상태도 바꾸지 않았습니다.

## 1. 변경 파일

- `crates/gpu/src/q8f32.rs`
  - `q8_0_lane_partials_mcol::<M>` (535행): m=1과 같은 레인 순서를 쓰는 M열 몸통입니다.
  - `store_sums` (790행)를 추가했습니다.
  - 새 엔트리는 `q8_0_gemv_mcol` (993행)과 `q8_0_gemv_heads_mcol` (1060행)입니다.
  - `enqueue_q8_0_gemv`는 m > 1을 새 엔트리로 보냅니다. `enqueue_q8_0_gemv_mcol`과 `enqueue_q8_0_gemv_heads_mcol` 런처도 추가했습니다.
  - `GemvOut`, `Q8_0GemvMcolArgs`, `Q8_0GemvHeadsMcolArgs`를 추가했습니다.
  - 호스트 유닛 테스트 `q8_0_mcol_column_is_1col` (1529행)을 추가하고 모듈 문서의 계약 문단을 고쳤습니다.
- `crates/gpu/src/elem.rs` (경계 밖)
  - 새 엔트리 `argmax_rows`와 런처 `enqueue_argmax_rows`를 추가했습니다.
  - 기존 `argmax` 본체는 공통 코어 `argmax_block`으로 뽑았습니다.
- `crates/gpu/src/head.rs`
  - `Head::with_m(m)`와 `tokens()`, `m()`을 추가했습니다. `new`는 `with_m(1)`이고, 호출자는 고칠 필요가 없습니다.
  - m = 1이면 기존 `enqueue_argmax`, m > 1이면 `argmax_rows`를 부릅니다. m = 1 그래프는 그대로입니다.
- `crates/gpu-gates/src/bin/gate_mcol.rs`: 새 게이트입니다. `--features gpu`로 빌드하고 V4.1 파일에서 돕니다.
- `justfile`: `gate-gpu-mcol` 레시피를 추가했습니다. 두 줄짜리 한국어 주석은 제가 썼으니 리드가 다듬어 주세요.
- `lib.rs`와 `cores.rs`는 최종 트리에서 변경이 없습니다. FAIL-first 돌연변이는 백업에서 복원했습니다.

## 2. 증명과 게이트 (실제 출력)

**레인 규칙.**
- Q8_0: 기존 m > 1은 레인이 32값 청크(`32·it + L`)를 가졌습니다. 새 규칙은 m = 1과 같이 레인 L이 워드 `L, L+32, …`를 갖고, 열마다 합을 하나씩 둡니다.
- Q8_0 heads: 기존은 m = 1뿐이었습니다. 새 엔트리는 헤드별 창에 같은 몸통을 씁니다.
- Q6_K: 원래 열 c가 `f_c += a·(e_c·drow·sc)`로 m = 1과 같은 식이라, 스펙이 허용한 대로 고치지 않고 핀만 걸었습니다.
- Q3_K: 핀 가능한 진입점이 따로 필요 없었습니다. `enqueue_gemv_q3k`의 m열과 m = 1을 비교하는 것으로 충분했습니다.

**FAIL-first (`just gate-gpu-mcol`).** 성격이 다른 두 종류입니다.
- q8_0 빨강은 **실제 결함**입니다. 라우팅이 없는 기존 청크 커널에서 나왔습니다.
  ```
  q8_0 site=blk.0.attn_q_a.weight k=5120 rows=1280 col_vs_m1_misses[m=1..8]=[0, 2283, 3429, 4569, 5733, 6891, 8061, 9213] mcol_m1_eq_decode=true FAIL
  q8_0 site=blk.0.attn_output_b.weight k=8192 rows=4099 col_vs_m1_misses[m=1..8]=[0, 7516, 11312, 15105, 18879, 22638, 26396, 30139] mcol_m1_eq_decode=true FAIL
  ```
  (7개 사이트와 합성 k=128/480/992/1024 모두 같은 모양입니다.)
- 나머지 넷은 **의도적 돌연변이**입니다. 출고 커널은 처음부터 일치했습니다.
  - heads: 버터플라이 순서를 뒤집었습니다.
  - q6k: 열 1..7을 `e·(drow·sc)`로 재결합했습니다.
  - q3k: 다열 경로 `q3k_row_dot`의 **열 0 항만** 재결합했습니다. `q3k_row_dot_1col`은 건드리지 않았습니다. 그래서 m=2..8에서 불일치가 1557로 일정합니다.
  - argmax_rows: 항상 열 0을 읽게 했습니다.
  ```
  q8_0_heads site=blk.0.attn_output_a.weight groups=8 rank=1024 group_k=4096 col_vs_m1_misses[m=1..8]=[5343, 10668, 15980, 21287, 26723, 32131, 37480, 42827] FAIL
  kquant site=blk.0.ffn_gate_exps.weight k=5120 rows=2304 col_vs_m1_misses[m=1..8]=[0, 1557, 1557, 1557, 1557, 1557, 1557, 1557] FAIL
  kquant site=output.weight k=5120 rows=129280 col_vs_m1_misses[m=1..8]=[0, 84928, 169572, 253810, 339007, 423732, 508804, 594477] FAIL
  head m=2 normed_rows_eq_m1=true logit_misses=85672 tokens=[94658, 94658] tokens_eq_m1=false tokens_eq_host_argmax=false graph_eq_eager=true FAIL
  ```

**초록 (최종 커널).**
```
q8_0 site=blk.0.attn_q_a.weight k=5120 rows=1280 col_vs_m1_misses[m=1..8]=[0, 0, 0, 0, 0, 0, 0, 0] token_major_misses[m=1..8]=[0, 0, 0, 0, 0, 0, 0, 0] mcol_m1_eq_decode=true PASS
  (q_b, kv, output_b, indexer.attn_q_b, ffn_down_shexp, engram_wkv, synthetic k=128/480/992/1024 모두 같은 줄 PASS)
q8_0_heads site=blk.0.attn_output_a.weight groups=8 rank=1024 group_k=4096 col_vs_m1_misses[m=1..8]=[0, 0, 0, 0, 0, 0, 0, 0] PASS
kquant site=blk.0.ffn_gate_exps.weight k=5120 rows=2304 col_vs_m1_misses[m=1..8]=[0, 0, 0, 0, 0, 0, 0, 0] PASS
kquant site=output.weight k=5120 rows=129280 col_vs_m1_misses[m=1..8]=[0, 0, 0, 0, 0, 0, 0, 0] PASS
head m=8 normed_rows_eq_m1=true logit_misses=0 tokens=[94658, 117657, 97021, 91719, 52695, 69443, 13532, 51770] tokens_eq_m1=true tokens_eq_host_argmax=true graph_eq_eager=true PASS
PASSED: column c of every m-column launch (m = 1..8) is the m = 1 launch of column c, bit for bit — q8_0, q8_0 heads, q3_K, q6_K and the head's tokens
```
`mcol_m1_eq_decode=true`는 새 엔트리를 m = 1로 쏜 결과가 기존 디코드 커널과 같다는 뜻입니다.

**`just ptx-scan gate_p3`, base 대 최종.** `diff`는 추가 3행, 변경 0행입니다. `q8_0_gemv`, `q8_0_gemv_heads`, `argmax`, `q6k_gemv`, `q3k_gemv` 행은 동일합니다.
```
> argmax_rows                       256     no         0         0      0        0     22     64      0              6       22         0
> q8_0_gemv_heads_mcol              256     no         0         0    864       48     78      0      0              3       78         0
> q8_0_gemv_mcol                    256     no         0         0    864       48     72      0      0              3       72         0
```
첫 빌드에서는 두 mcol 엔트리 모두 depot YES(ld.local 5, st.local 8, jit_local 32)였습니다. 원인은 `store_sums`가 `sums[c]`를 런타임 인덱스로 읽은 것이었습니다. 이름 붙인 스칼라로 바꿔 depot no가 됐습니다.

**나머지 게이트** (모두 최종 커널):

| 게이트 | 결과 |
|---|---|
| `gate-gpu-p3` | `PASSED: f32 and q8_0 gemv within 1e-5 of the f64 reference; eager == graph replay` (m=8은 이제 mcol, 예 `K=2048 rows=64 m=8 max_rel_err=2.624e-7`) |
| `gate-gpu-p6` | `shape op=q8_0_gemv fma=48 fma_floor=48 … PASS`, `shape op=q8_0_gemv_heads fma=40 fma_floor=40 … PASS`, `PASSED: router ids exact …` |
| `gate-gpu-p4` | `shape op=argmax geometry block_reqntid=256 ARGMAX_THREADS=256 … PASS`, `PASSED: elem kernels …` |
| `gate-gpu-head` | `argmax device=8913 host=8913 oracle=8913 ik_pin=8913 PASS`, `graph graph_nodes=4 eager_vs_replay_bit_identical=true PASS` |
| `gate-gpu-lib` | `test q8f32::tests::q8_0_mcol_column_is_1col ... ok`, `test result: ok. 6 passed; 0 failed` |
| `gate-gpu-e2e` | `graph graph_nodes=648 want=648 eager_vs_graph_identical=true ok`, `gate_e2e: PASS — …` |
| `gate-gpu-ds41-chain-glue` | `PASSED: gate_deepseek41_chain_glue — …` (Head 변경 확인용) |
| `just check` | rc=0 |
| `just lint` | `grep -c '^warning:'` = **169**, 기준 169. 문서 수정 뒤 재실행도 169 |
| Mac 검사 | `check-recipes: ok`, `check-comments: ok`, `check-arch: ok`, `cargo fmt --all -- --check` rc 0 |

`gate-gpu-mcol`의 마지막 초록 실행 뒤에는 **주석만** 바뀌었습니다. `store_sums`의 SAFETY 문구(lint 170에서 169로 복구)와 `q8f32.rs` 모듈 문서, `check_gemv_geometry` 문서입니다. 그 뒤로 lint, fmt, check-comments는 다시 돌렸습니다.

## 3. 예측과 결과

- **비트 동일:** 구조상 동일하다고 예측했고, 게이트가 m = 1..8에서 0 불일치를 보였습니다.
- **Q6_K:** 고치지 않아도 이미 열별로 재현된다고 예측했고, 결과는 0 불일치였습니다.

**레지스터와 점유율 (ptxas).** 임시 프로브 엔트리를 빌드해 M별로 쟀고, 프로브는 삭제했습니다.

| M | 1 | 2 | 4 | 6 | 8 |
|---|---|---|---|---|---|
| regs | 40 | 40 | 54 | 60 | 76 |

- 합친 엔트리는 모든 M의 최댓값이 걸립니다. `q8_0_gemv_mcol`은 72 regs로 블록당 18,432 레지스터, SM당 3블록, 48워프 중 24워프(50 %)입니다. heads는 78 regs로 역시 3블록입니다.
- 비교로 m = 1 디코드 커널 `q8_0_gemv`는 57 regs, SM당 4블록(67 %)입니다. 새 엔트리를 따로 둔 이유가 이 m = 1 레지스터를 지키기 위해서입니다.
- spill은 0이라 k = 8 상한은 필요 없습니다.
- 대역폭 측면 [유도]: 24워프 × 32레인 × 호이스트한 워드 4개 × 4 B = SM당 약 12 KB가 비행 중입니다. 필요량은 936 GB/s × 약 600 ns / 82 SM ≈ 6.8 KB이므로 50 % 점유로도 충분하다고 봅니다. 타이밍은 리드 몫입니다.

**가중치 1회 읽기.** 워프가 스텝마다 코드 워드 32개(연속 128 B)와 스케일 4개를 한 번 읽고 M열 전부에 dot합니다. 가중치 바이트는 m과 무관하고, 런치 수는 k에서 1로 줄어듭니다. 활성값 M·k·4 B(≤ 256 KB)는 L1/L2에서 읽습니다. m = 1 바이너리는 바뀌지 않았으므로(ptx-scan 행 동일) 디코드 A/B는 필요 없습니다.

## 4. 못 한 것과 경계 확장

1. **`crates/gpu/src/elem.rs`는 스펙 경계 밖입니다.** `argmax_rows`를 추가했고, 두 번째 사본을 만들지 않으려고(R14) 기존 `argmax` 본체를 `argmax_block`으로 뽑았습니다.
   - 이동 클래스이고, 증명은 ptx-scan `argmax` 행 동일(regs 22, smem 64), p4 `argmax_geometry` PASS, `gate-gpu-head` graph_nodes=4 PASS입니다.
   - 되돌리려면 `argmax_rows`만 남기고 원래 본체를 복원하면 됩니다.
2. **`GemvOut::TokenMajor`는 스펙에 없던 추가입니다.** 스텝의 요소 연산(rms_norm, rope, kv append)은 토큰 우선으로 읽는데, gemv 계약은 `y[r·m+c]`입니다(ktok 맵 `attn.rs:701`). 그래서 m > 1 체인 사이트가 바로 쓸 수 있게 토큰 우선 출력을 열었고, 게이트에도 `token_major_misses`로 넣었습니다. heads mcol 출력은 원래부터 토큰 우선입니다.
3. **청크 경로 호출자.**
   - 청크 **순서**에 핀으로 기대는 호출자는 없습니다. gate_p3 m=8은 1e-5 밴드이고, 이제 mcol을 타고 통과합니다. ds41 체인은 `STEP_TOKENS=1`이라 전부 m = 1입니다.
   - 그래서 런처에서는 청크 경로를 대체했습니다. 하지만 청크 몸통은 `q8_0_gemv` 엔트리 안에 남겼습니다. 이 엔트리의 m = 1 ptx-scan 행이 바뀌면 안 되고, gate_p6 `router_shape`가 그 엔트리의 fma floor 48(청크 열 8 + 40)을 핀하고 있기 때문입니다.
   - 후속(이동 클래스): 청크 몸통 삭제, p6 floor를 48에서 40으로 재핀, 같은 임대 A/B.
4. **체인 사이트 헌크 (보고만, skew 소유).**
   - m > 1의 `enqueue_q8_0_gemv`는 `y[r·m+c]`를 냅니다. 그래서 체인 사이트는 이 라우팅을 그대로 타지 말고 `enqueue_q8_0_gemv_mcol{out: GemvOut::TokenMajor}`를 써야 합니다. 대상은 `chain/attn.rs:1053`(kv), `:1174`(q_a), `:1185`(q_b), `:1224`(wo_b, x는 wo_a 출력), `:1265`(indexer q_b), `chain/glue.rs:464`(engram_wkv, 리터럴 1을 m으로), `chain/ffn.rs:1062`(shared down, 리터럴 1을 m으로)입니다.
   - `:1208`의 wo_a는 `enqueue_q8_0_gemv_heads_mcol{m, x_col_stride: n_head*head_dim, y_col_stride: o_groups*o_lora_rank}`로 바꿉니다.
   - m = 1에서는 기존 런치를 유지하는 것을 권합니다. 그래야 디코드 바이너리가 바뀌지 않습니다.
   - 헤드는 `Head::with_m(k)`와 `tokens()`를 씁니다. 로짓은 `[v*m + c]`이고, 입력은 행 우선 `m*hidden`입니다.
5. 타이밍은 하지 않았습니다(리드 몫). 박스 작업은 전부 30분 안에 끝났고, CPU 임대는 매번 비어 있었습니다.

## 5. 스펙 밖 개선 지점 (보고만)

- `crates/gpu/src/q8f32.rs`의 `q8_0_gemv`: m > 1 청크 몸통이 이제 런처에서 닿지 않는 죽은 경로입니다. 삭제하면서 p6 floor를 재핀해야 합니다. 크기 S.
- `crates/gpu/src/q8f32.rs`의 `f32_gemv`와 `q8_0_gemv` 저장부: 8갈래 가드 저장이 두 벌 있고, `store_sums`로 대체할 수 있습니다. 이동 클래스이고 ptx-scan 동일로 증명합니다. 크기 XS.
- `crates/gpu/src/q8f32.rs`의 `f32_gemv` m > 1: 여전히 청크 순서입니다. 라우터는 토큰별 1col이라 ktok 사이트는 아니지만, m열 f32 gemv를 쓰는 날이 오면 같은 문제가 생깁니다. 크기 S.
- `crates/gpu/src/model/kernels.rs:81`의 `q8_0_gemv_heads`: `#[allow(clippy::too_many_arguments)]`에 `reason`이 없어 R16 위반입니다. 크기 XS.
- `crates/gpu/src/model/kernels.rs`의 `enqueue_q8_0_gemv_heads`: 이 함수와 새 heads mcol 런처가 검사 여섯 개를 중복합니다. 공통 검사 함수로 합칠 수 있습니다. 크기 S.
- K-quant gemv(q3k, q4k, q6k)도 m > 1에서 `y[r·m+c]`만 냅니다. k토큰 체인의 K-quant 사이트에서는 같은 전치 문제가 생깁니다. 크기 M.
- `mcol` 엔트리 하나가 모든 M에 M=8 레지스터(72)를 씁니다. m=2..4가 흔하다면 M ≤ 4용 엔트리(≤ 54 regs, 4블록)와 M > 4용으로 나누는 게 이득일 수 있습니다. 타이밍을 보고 판단할 일입니다. 크기 S.

## 6. 모델

opus로 스폰되었고, Opus 5.5로 실행했습니다.
