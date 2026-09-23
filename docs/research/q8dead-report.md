# q8dead 라운드 보고

`q8_0_gemv`의 죽은 청크 몸통을 지웠습니다. 새 커널은 예측대로 40 레지스터와 SM당 6블록으로 컴파일됩니다. p6 floor는 48에서 40으로 재핀했고 FAIL-first까지 확인했습니다. 지정한 GPU 게이트 다섯과 정적 검사가 모두 녹색이고 lint는 168입니다. `q8_0_gemv` 행이 바뀌었으므로 리드의 같은 임대 A/B가 필요합니다.

## 1. 변경 파일

- **`/Users/hckim/repo/bloomery-q8dead/crates/gpu/src/q8f32.rs`**
  - `q8_0_lane_partials`를 통째로 지웠습니다. 호출자가 `q8_0_gemv` 하나뿐이었습니다. 그 함수의 가중치 디코드 설명은 `q8_0_lane_partial_1col`의 문서와 Safety 절로 옮겼습니다.
  - `q8_0_gemv`는 이제 단일 열 커널입니다. `m_cols` 인자를 ABI에서 뺐고, contract는 `x.len() >= k`와 `y.len() >= n_rows`입니다. 몸통은 1col 호출, `reduce_sum_f32` 한 번, 저장 한 번입니다. 호스트 쪽 `enqueue_q8_0_gemv`는 `m`을 넘기지 않고, 모듈 문서도 이에 맞춰 고쳤습니다.
  - `f32_gemv`의 8갈래 가드 저장을 `store_sums`로 바꿨습니다.
- **`/Users/hckim/repo/bloomery-q8dead/crates/gpu/src/model/kernels.rs`**: 81행 `#[allow(clippy::too_many_arguments)]`에 `reason`을 달았습니다. q8f32.rs에 있는 문자열을 그대로 썼습니다(R16).
- **`/Users/hckim/repo/bloomery-q8dead/crates/gpu-gates/src/bin/gate_p6.rs`**: `router_shape`의 q8 floor를 `GEMV_COLS + 4·(…)`에서 `4·(Q8_STEP_UNROLL + Q8_OTHER_WORDS)`로 바꿨습니다. 새 PIN 줄을 달고 derivation 문자열도 고쳤습니다. 이제 틀린 문장이 된 "`q8_0_gemv` has the same general body" 한 문장도 고쳤습니다. 경계는 "floor pin only"였지만, 이 문장은 바꾸는 핀의 근거 설명이라 함께 고쳤습니다.

## 2. 증명과 게이트 (실제 출력)

**ptx-scan.** gate_p3와 gate_p6는 같은 번들을 싣습니다(`bytes=1105656 modules=1`). 그래서 이후 스캔은 gate_p3 하나로 했습니다.

`just ptx-scan gate_p3`, base 대 1단계 후 diff:
```
< q8_0_gemv  256  no  0  0  48  11  57  0  0  4  57  0
---
> q8_0_gemv  256  no  0  0  40  10  40  0  0  6  40  0
```
다른 행은 전부 같습니다. `f32_gemv`, `q8_0_gemv_heads`, `q8_0_gemv_mcol`, `q8_0_gemv_heads_mcol` 모두 그대로입니다.

3·4단계 후 1단계 후 표와 diff: `TABLE-IDENTICAL`

**엔트리별 명령어 수.** 박스에서 ptxas와 `cuobjdump -sass`로 셌습니다.

| 엔트리 | PTX 명령어 base | PTX 명령어 1단계 후 | SASS 명령어 base | SASS 명령어 1단계 후 |
|---|---|---|---|---|
| q8_0_gemv | 697 | 415 | 1216 | 568 |
| f32_gemv | 376 | 376 | 816 | 816 |

3단계 후에도 네 엔트리의 수가 모두 1단계 후와 같습니다.

**p6 FAIL-first.** 새 커널에 옛 floor 48, `just gate-gpu-p6` rc=1:
```
shape op=q8_0_gemv fma=40 fma_floor=48 (cols=8 + 4*(Q8_STEP_UNROLL=4 + 6)) local_depot=false FAIL
GPU GATE RED: gate_p6 (exit 1)
```

**p6 녹색.** floor 40, rc=0:
```
shape op=f32_gemv fma=19 fma_floor=16 (cols=8 + LANE_UNROLL=8) local_depot=false PASS
shape op=q8_0_gemv fma=40 fma_floor=40 (4*(Q8_STEP_UNROLL=4 + 6)) local_depot=false PASS
shape op=q8_0_gemv_heads fma=40 fma_floor=40 (4*(Q8_STEP_UNROLL=4 + 6)) local_depot=false PASS
PASSED: router ids exact / probs+weights within 1e-6 of the route_inner reference; ...
```

**나머지 게이트.** 모두 rc=0입니다.
- `gate-gpu-p3`: `PASSED: f32 and q8_0 gemv within 1e-5 of the f64 reference; eager == graph replay`
- `gate-gpu-mcol`: `PASSED: column c of every m-column launch (m = 1..8) is the m = 1 launch of column c, bit for bit — q8_0, q8_0 heads, q3_K, q6_K and the head's tokens`
- `gate-gpu-e2e`: `gate_e2e: PASS` 2회로, MMA 팔과 `BLOOMERY_FLASH_MMA=0` 팔입니다. forced_exact 줄은 `forced_exact positions=1056 disagree=43 buckets=[18 21 3 1] ge_floor=4 pin<=6 max_margin=1.182 ok`입니다.
- `gate-gpu-ds41-chain-ffn`: `gate_deepseek41_chain_ffn: 121 checks across 2 sets and 40 layers, 0 failed, 0 waived (near tie) — host served 160 layers, 490 slots — PASS`

**정적 검사.** `just fmt-check`, `check-recipes`, `check-arch`, `check-comments`, `check`, `lint` 모두 rc=0입니다.
- 각 출력은 `check-recipes: ok`, `check-arch: ok`, `check-comments: ok`입니다.
- lint는 `grep -c '^warning:'`로 **168**이고, 기준 169 이하입니다.
- `git status`에는 수정한 파일 셋만 있습니다. Cargo.lock은 변하지 않았습니다.

`just box-gc`는 처음과 끝에 돌렸고, 둘 다 `found 0 process(es)`였습니다.

## 3. 예측과 결과

1단계 빌드 전에 예측을 적어 두었습니다. 새 몸통은 `q8_0_gemv_heads`가 도는 1col 몸통과 같으므로 그 행을 따른다고 봤습니다.

| 항목 | 예측 | 실측 |
|---|---|---|
| fma | 48 → 40 | 40 |
| cvt.f16 | 11 → 10 | 10 |
| regs | 57 → 약 40, 48 이하 구간 | 40 |
| blk/SM | 4 → 6 | 6 |
| 다른 엔트리 행 | 불변 | 불변 |

블록 수의 유도는 이렇습니다. 레지스터 한도는 65536 / (40 × 256) = 6.4이고, 워프 한도는 48 / 8 = 6입니다. 둘 중 작은 쪽이 6입니다.

- **A/B 예측.** 디코드 경로의 시간 변화는 0 이하로 봅니다. 로드와 합산 순서가 같아 비트가 같고, occupancy가 4블록에서 6블록으로 오르며, 명령어가 절반 가까이로 줄었습니다.
- **A/B가 필요한 이유.** `q8_0_gemv` 행이 바뀌었으므로 리드의 같은 임대 A/B가 필요합니다.
- **3단계.** 이동 클래스이고 표가 같으므로 추가 A/B는 필요 없습니다.

## 4. 못 한 것과 스펙에서 벗어난 판단

- **`m_cols`를 커널 ABI에서 뺐습니다.** 남기면 쓰지 않는 인자가 되고, m > 1 호출이 조용히 틀린 값을 냅니다. 모듈이 비공개라 이 엔트리를 부르는 곳은 `enqueue_q8_0_gemv` 하나이고, 그 공개 시그니처는 그대로입니다.
- **3단계는 `f32_gemv`에만 적용했습니다.** 1단계 뒤 `q8_0_gemv`의 저장은 한 칸짜리라 가드 저장 자체가 사라졌습니다. 거기에 `store_sums`를 쓰면 `(row, 1, 1, [s,0,…])` 모양의 무의미한 호출이 됩니다.
- **Mac에 `timeout`이 없습니다.** 그래서 박스 작업은 레시피를 직접 불렀고, 박스 쪽 상한은 `gate.sh`와 `gpu-gate.sh`가 걸었습니다. 모든 작업이 30분 안에 끝났습니다.
- **타이밍은 돌리지 않았습니다.** 리드 몫입니다.

## 5. 스펙 밖 개선 지점 (보고만, 손대지 않음)

- **`crates/gpu/src/q8f32.rs`의 `q8_0_launch_dims`**: 셋째 반환값 `m`을 `q8_0_gemv` 경로가 버립니다(`_`). m = 1 경로 전용 dims를 두거나 반환형을 나누면 더 정직해집니다. 크기 XS.
- **`crates/gpu/src/q8f32.rs`의 `f32_lane_partials`와 `f32_gemv`**: m > 1은 여전히 청크 순서의 굶는 모양입니다. dsm도 지적한 항목이고, 엔진 호출처는 모두 m = 1입니다. 크기 S.
- **`crates/gpu-gates/src/bin/gate_p6.rs`의 `GEMV_COLS` 문서**: 이제 `f32_gemv` 하나에만 해당합니다. 이름이나 문서를 f32 전용으로 좁힐 수 있습니다. 크기 XS.
- **`crates/gpu-gates/src/bin/gate_p6.rs`의 두 floor**: 이제 `q8_fma_floor`와 `heads_fma_floor`가 같은 식입니다. 상수 하나로 합치고 PIN 줄도 하나로 줄일 수 있습니다. 크기 XS.
- **`crates/gpu/src/model/kernels.rs:81`의 allow 위치**: `#[kernel]` 앞에 있어서, `#[launch_contract]` 뒤에 두는 q8f32.rs 관례와 다릅니다. 순서만 통일하면 되는 문제입니다. 크기 XS.
- **`crates/gpu/src/model/kernels.rs`의 `enqueue_q8_0_gemv_heads`**: heads mcol 런처와 검사 여섯 개가 중복됩니다. dsm 보고와 같은 항목이고, 공통 검사 함수로 합칠 수 있습니다. 크기 S.

## 6. 모델

opus로 스폰되어 claude-opus-5-5로 실행했습니다.
