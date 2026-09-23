engshadow 라운드를 끝냈습니다. engram의 토큰 전용 작업 6커널이 이제 0층과 13층의 호스트 다리 그늘 안에서 돌고, 이를 핀하는 그늘 집합 게이트는 수정 전 코드에서 FAIL, 수정 후 PASS입니다. 엔진이 돌리지 않는 커널 셋은 지웠습니다. 요청하신 모든 게이트는 초록입니다. 시간 A/B는 돌리지 않았습니다(리드 몫).

## 1. 변경 파일

A+B(게이트 추가, 그늘 이동)와 C(죽은 커널 삭제)는 서로 다른 파일이라 따로 커밋할 수 있습니다. 예외는 C의 네 파일인데, 그 안의 hunk는 전부 C입니다.

**A+B**
- `/Users/hckim/repo/bloomery-engshadow/crates/gpu-gates/src/bin/gate_deepseek41_step.rs`: `--structure`에 Q1 그늘 집합 검사(`shadow_sets`)를 넣었고 헤더 문서도 고쳤습니다.
- `/Users/hckim/repo/bloomery-engshadow/crates/gpu-deepseek41/src/chain/ffn.rs`: 타입이 있는 훅 `ShadowWork` 트레이트와 `FfnPiece::enqueue_shadowed(…, extra)`를 넣었습니다.
  - 기존 `enqueue`는 빈 목록으로 위임합니다. 그래서 경계 밖의 chain_ffn 게이트는 고치지 않아도 됩니다.
  - wait 발행을 `enqueue_shadow`에서 `enqueue_shadowed`로 올렸습니다. 발행 순서는 같습니다.
  - 모듈 문서의 순서도와 런치 수 설명을 고쳤습니다.
- `/Users/hckim/repo/bloomery-engshadow/crates/gpu-deepseek41/src/chain/glue.rs`: 사이트 하나 단위의 `enqueue_engram_kv_at`와 `EngramKv`(ShadowWork)를 넣었습니다.
  - 기존 `enqueue_engram_kv`는 사이트마다 `enqueue_engram_kv_at`을 부르는 루프가 됐습니다. chain_glue 게이트가 이것을 씁니다.
  - 모듈 문서 11–17행(Q4)을 지금 참인 내용으로 고쳤습니다.
- `/Users/hckim/repo/bloomery-engshadow/crates/gpu-deepseek41/src/body.rs`: 층 루프 앞의 `enqueue_engram_kv` 호출을 지웠습니다.
  - `LayerStep.shadow_site`를 새로 뒀습니다. 다음 층이 사이트면 그 층 번호입니다.
  - MoE 서브층이 `enqueue_shadowed`로 그 사이트의 작업을 받습니다.
  - 모듈 문서의 스텝 순서와 `layer_steps` 문서를 고쳤습니다.

**C**
- `/Users/hckim/repo/bloomery-engshadow/crates/gpu-deepseek41/src/hc.rs`: `ds41_hc_pre_mix`와 `ds41_hc_post_streams` 커널, 두 런처, 모듈 문서의 두 항목을 지웠습니다. `post_launch`의 `Option` 인자는 호출자가 하나뿐이라 `usize`로 줄였습니다.
- `/Users/hckim/repo/bloomery-engshadow/crates/gpu-deepseek41/src/experts.rs`: `ds41_moe_combine` 커널과 `enqueue_moe_combine`, 결합 계약 문서, 쓰이지 않게 된 `N_USED` import와 단언을 지웠습니다.
- `/Users/hckim/repo/bloomery-engshadow/crates/gpu-gates/src/bin/gate_deepseek41_hc.rs`: pre_mix 검사(`op_bit`·`op_vs_dump_*`)와 split 형태 검사(`split_eq_fused_bit`)를 지웠습니다.
  - 헤더에 `PIN(2026-09-24): removed — …` 줄을 넣었습니다.
  - 출력 전용 `nodes` 줄과 PASSED 문구를 지금 있는 커널에 맞게 고쳤습니다.
- `/Users/hckim/repo/bloomery-engshadow/crates/gpu-gates/src/bin/gate_deepseek41_moe.rs`: `combine_site`, `combine_host`, 결합용 Dev 버퍼 여섯 개를 지웠습니다.
  - 층 그래프에서 combine 런치를 뺐습니다. `NODES`는 8에서 7이 되고 `PIN(2026-09-24): 8 → 7 …` 줄을 달았습니다.
  - 헤더에 `PIN(2026-09-24): removed — …` 줄을 넣었습니다.
  - 엔진 결합의 커버리지는 줄지 않습니다. chain_ffn 게이트가 `ds41_ffn_post`를 `ffn_out-L`에 대해 이미 핀합니다.

## 2. 증명과 게이트 — 실행 명령과 실제 출력

**Q1 FAIL-first.** 게이트만 바꾸고 엔진은 base 그대로 둔 트리입니다. `BLOOMERY_REMOTE='~/repo/bloomery-engshadow' just gate-gpu-ds41-step --structure`, rc=1.
```
shadow: walked 1110 nodes from the root in one chain, 80 memop batches (ops: go {5}, wait {2}), 0 other: PASS
shadow layer=0 got=[ds41_hc_pre,ds41_shexp_gate_up,q8_0_gemv] want=[ds41_hc_pre,ds41_shexp_gate_up,q8_0_gemv,ds41_glue_engram_rows,q8_0_gemv,ds41_engram_key_norm]: FAIL
shadow layer=13 got=[ds41_hc_pre,ds41_expert_gate_up,q3k_quantize_q8_1,q4k_gemv_sel,ds41_shexp_gate_up,q8_0_gemv] want=[…,ds41_glue_engram_rows,q8_0_gemv,ds41_engram_key_norm]: FAIL
shadow engram token-only work: 2 of 2 sites found, at [before the first go (critical path); before the first go (critical path)]; kernels outside every shadow: 6: FAIL
GPU GATE RED: gate_deepseek41_step (exit 1)
```

**방식.** 게이트 바이너리 안에서 엔진의 `enqueue_chain`을 `cuStreamBeginCapture_v2`/`cuStreamEndCapture`로 직접 캡처했습니다. `Graph`의 핸들은 경계 밖 `crates/gpu`에서 private입니다.
- 순서는 `cuGraphGetRootNodes`가 정확히 1개인지 확인한 뒤, `cuGraphNodeGetDependentNodes_v2`의 후속 노드를 하나씩 따라가서 얻습니다. 한 스트림 캡처라 사슬이고, 전체 노드 수와 같은지도 대조합니다. 결정적이고 비용이 작습니다.
- 커널 이름은 `cuGraphKernelNodeGetParams_v2`와 `cuFuncGetName`으로 읽습니다.
- go와 wait는 `cuGraphBatchMemOpNodeGetParams`의 op 수로 구분됩니다(go 5, wait 2).
- 템플릿은 인스턴스화하지 않고 파괴합니다.
- overlap 레버가 0이면 명확한 오류로 거부합니다.

**A+B, `--structure --sets --select`.** rc=0.
```
graph: nodes=1110 kernels=1030 memops=80 other=0 predicted kernels=1030 memops=80: PASS
graph set=ref_deepseek41_step4_every_node replay_bit_identical_to_eager=true (logits, 48 state buffers, 6 compressor buffers): PASS
graph set=ref_deepseek41_d1_every_node replay_bit_identical_to_eager=true (logits, 48 state buffers, 6 compressor buffers): PASS
shadow set=[ds41_hc_pre,ds41_shexp_gate_up,q8_0_gemv,ds41_glue_engram_rows,q8_0_gemv,ds41_engram_key_norm] layers=0: PASS
shadow set=[ds41_hc_pre,ds41_shexp_gate_up,q8_0_gemv] layers=1: PASS
shadow set=[ds41_hc_pre,ds41_expert_gate_up,q3k_quantize_q8_1,q4k_gemv_sel,ds41_shexp_gate_up,q8_0_gemv] layers=2,…,12,14,…,39: PASS
shadow set=[…,ds41_shexp_gate_up,q8_0_gemv,ds41_glue_engram_rows,q8_0_gemv,ds41_engram_key_norm] layers=13: PASS
shadow engram token-only work: 2 of 2 sites found, at [layer 0's shadow; layer 13's shadow]; kernels outside every shadow: 0: PASS
PASSED: gate_deepseek41_step
```

**`--sets`/`--select`를 base와 대조했습니다.** base 트리는 `git archive ba7b949`로 풀어 aux 원격 디렉터리 `~/repo/bloomery-engshadow-base`에서 같은 레시피로 돌렸습니다. 판정 줄만이 아니라 모든 seam 수치를 포함한 438줄 전체를 비교했습니다.
```
diff sets-base.txt sets-new.txt → SETS+SELECT OUTPUT IDENTICAL
md5 f141b5187509a8684d1f4ea2c0697142 (both)
```

**ptx-scan, 이동 클래스 증명.** base는 첫 편집 전에 떴고, 다른 하나는 A+B 상태입니다. 명령은 `just ptx-scan gate_deepseek41_step --features deepseek41`입니다.
```
83 lines each; diff → IDENTICAL; md5 74fae2e659bcdea1eeceba608e6977ed (both)
```

**ptx-scan, C 뒤.** A+B 표와의 diff입니다. 정확히 세 행이 빠졌고 나머지 80행은 같습니다.
```
< ds41_hc_post_streams  256 no 0 0 16 0 38 0 0 6 38 0
< ds41_hc_pre_mix       256 no 0 0 15 0 30 0 0 6 30 0
< ds41_moe_combine      256 no 0 0  5 0 25 0 0 6 25 0
```

**Q7 죽은 코드 확인.** 크레이트 전체에서 심볼 단위로 찾았습니다.
- `enqueue_pre_mix`는 `gate_deepseek41_hc.rs:967`에서만 불렸습니다.
- `enqueue_post_streams`는 `gate_deepseek41_hc.rs:1066`에서만 불렸습니다.
- ds41 쪽 `enqueue_moe_combine`은 `gate_deepseek41_moe.rs:1267,1269,1272,1910`에서만 불렸습니다. `crates/gpu/src/moe_fused.rs`의 같은 이름 함수는 V2-Lite용 별개 함수라 그대로 뒀습니다.
- 커널 이름 세 개는 hc.rs와 experts.rs 밖에서 호출되지 않았습니다.

**최종 트리 배치.** 편집 없이 순차로 돌렸고, 단계마다 임대가 비어 있는지 먼저 확인했습니다. 전부 rc=0입니다.
```
fmtcheck rc=0   check rc=0   lint rc=0   ptx rc=0   hc rc=0   moe rc=0   glue rc=0   ffn rc=0   structure rc=0
```
- lint는 `grep -c '^warning:'` 169로 기준선 169와 같습니다. 바뀐 파일에 걸린 경고는 0이고, 로그에 `Checking bloomery-gpu-deepseek41`·`Checking bloomery-gpu-gates`가 있어 다시 검사된 것을 확인했습니다.
- moe 게이트:
  ```
  graph layer=2 nodes=7 (want 7) replays_eq_eager=true tickets_zero=true PASS
  gate_deepseek41_moe: 477 sites across 4 sets and 40 layers, 0 failed — PASS
  ```
- chain-glue 게이트:
  ```
  PASSED: gate_deepseek41_chain_glue — host row ids exact, …, the captured step replaying bit-identically at its node count
  ```
- chain-ffn 게이트:
  ```
  gate_deepseek41_chain_ffn: 121 checks across 2 sets and 40 layers, 0 failed, 0 waived (near tie) — host served 160 layers, 490 slots — PASS
  ```
- 최종 트리의 structure는 위 A+B와 같은 줄로 전부 PASS입니다.
- hc 게이트는 PASSED 문구만 고친 뒤 fmt-check와 함께 다시 돌렸습니다. 둘 다 rc=0입니다.
  ```
  PASSED: gate_deepseek41_hc — chain, HC_PRE, HC_POST and folds bit-identical to our rule; HC_POST and folds bit-identical to ik's dump; the chain inside its predicted gap
  ```
- `just check-recipes`는 `check-recipes: ok`, `tools/check-comments.sh`는 `check-comments: ok`입니다.
- `gate-gpu-e2e`는 V2-Lite(`--features gpu`) 게이트이고 ds41용은 없어서 돌리지 않았습니다.

## 3. 예측과 결과

**사이트별 첫 소비 층.** 층 l의 engram 단계는 그 층 attention 앞에서 돕니다(`body.rs:481-490`). 그래서 첫 소비자는 사이트 층 자신의 `glue.enqueue_engram`입니다.

| 사이트 | 층 (`hp.engram.layer_ids`) | 첫 소비자 | 합법인 가장 늦은 그늘 | 고른 곳 |
|---|---|---|---|---|
| 0 | 1 | 1층 `enqueue_engram` (`glue.rs` `enqueue_engram`, `body.rs` 루프의 `step.engram`) | 0층 | 0층 |
| 1 | 14 | 14층 `enqueue_engram` | 13층 | 13층 |

첫 층에 사이트를 두는 것은 `layer_steps`가 이미 거부합니다. 따라서 모든 사이트는 L−1 그늘을 가집니다. 예측을 반으로 줄일 경우는 없습니다.

**그늘 안 순서.** 조각 자신의 일(HC_PRE, 카드 expert, shared expert)을 먼저 두고, 그 뒤 engram 3커널, 그 뒤 wait입니다. join(`ds41_ffn_post`)의 입력이 engram 바이트 뒤에 줄 서지 않게 하려는 것입니다. engram 출력은 다음 층 또는 13층 뒤에야 필요합니다.

**수용량 [유도].** 사이트 하나가 167,116,800 B(254.2 µs, B11c 런치 측정은 237.99 µs)입니다. 두 사이트를 서로 다른 층에 나눠서 층마다 한 사이트만 들어갑니다.
- 0층: 호스트 다리 약 0.85 ms에 자기 일 약 67 µs + 약 250 µs가 들어갑니다.
- 13층: 약 0.66 ms에 약 100 µs + 약 250 µs가 들어갑니다.

**예측과 결과 대조.**

| 항목 | 예측 | 결과 |
|---|---|---|
| G2 노드 | 1110 | 1110 |
| 커널 | 1030 | 1030 |
| memop | 80 | 80 |
| replay = eager | 두 세트 모두 | 두 세트 모두 PASS |
| `--sets`/`--select` 출력 | base와 같음 | 438줄 바이트 동일 |
| 0층·13층 그늘 | 3커널씩 추가 | 그대로 나옴 |
| ptx-scan | A+B는 동일, C는 세 행 감소 | 그대로 나옴 |

**nsys 배치.** 명령은 `BLOOMERY_REMOTE='~/repo/bloomery-engshadow' just nsys-gpu-ds41 6 1024`입니다. 임대 아래 A6000 300 W에서 돌았고, 두 카드 모두 앱이 비어 있었고 loadavg는 1.15였습니다. 0층과 13층 합류 바로 앞 커널이 `ds41_engram_key_norm`입니다. 즉 6커널이 go와 wait 사이에 있습니다.

| 깊이 | 층 | bridge (µs) | overlap (µs) | exposed (µs) | 합류 앞 커널 |
|---|---|---|---|---|---|
| 6 | 0 | 2672.8 | 313.3 | 2353.1 | `ds41_engram_key_norm` |
| 6 | 13 | 717.7 | 349.0 | 362.5 | `ds41_engram_key_norm` |
| 1024 | 0 | 2133.0 | 311.9 | 1814.8 | `ds41_engram_key_norm` |
| 1024 | 13 | 678.2 | 347.0 | 325.3 | `ds41_engram_key_norm` |

- 그늘 안 카드 일(overlap)은 두 층 모두 호스트 다리 안에 들어갑니다.
- 이 실행의 스텝 시간(p50 45.6/45.3 ms, `warm=0`, 프로파일러 아래)은 판정용 수치가 아닙니다. base nsys와도 비교하지 않았습니다.

**시간 예측은 검증하지 않았습니다.** 예측은 −0.48…−0.51 ms/토큰(약 −1.2 %) @ n=1, 모든 깊이, A6000 + 호스트입니다. 리드가 돌릴 A/B는 base `ba7b949`의 `generate_ds41`과 이 트리의 것을 한 임대 안에서 번갈아 4–6바퀴 도는 것입니다. 깊이 6과 1024 두 행을 보고 구간으로 판정합니다. 자는 4바퀴에서 ±1.0 %입니다. `ab-decode`는 V2-Lite CPU용이라 ds41에는 `depth-gpu-ds41` 계열을 두 바이너리에 쓰는 방식이 필요합니다.

## 4. 하지 못한 것

- **시간 A/B.** 스펙대로 리드 몫으로 남겼습니다.
- **aux 디렉터리 덮어쓰기.** 마지막 box-gc를 `BLOOMERY_REMOTE='~/repo/bloomery-engshadow-base'`로 돌렸는데, box.sh가 먼저 rsync를 해서 그 디렉터리를 이 워크트리 사본으로 덮었습니다. base 판정 줄은 이미 회수한 뒤였고, 그 디렉터리는 더 쓰지 않습니다. `just box-tracks --remove`로 지우시면 됩니다. 박스 트리의 `target/`도 수백 MB 남아 있습니다.

## 5. 스펙 밖 개선 지점 (보고만)

- **`gate_deepseek41_step.rs` `SHADOW_CARD`/`SHADOW_HOST` 상수:** 기대 표가 커널 엔트리 이름에 묶여 있습니다. 커널 이름을 바꾸면 이 게이트가 빨개지는데, 의도한 동작입니다. 조각이 자기 그늘 커널 목록을 내보내면 한 곳이 소유하게 됩니다(S).
- **nsys 0층 bridge 2.1–2.7 ms:** 0층은 전부 호스트 expert(Q5_K)인 층입니다. 계획이 적은 0.85 ms의 2.5–3배이고, 스텝에서 가장 큰 exposed입니다(프로파일러 아래, `warm=0`, 판정 아님). 원인 축소 항목으로 올릴 만합니다(측정이 먼저).
- **`glue.rs` `Glue::enqueue_engram_kv`:** 이제 부르는 곳이 `gate_deepseek41_chain_glue.rs:335` 하나뿐입니다. 그 게이트가 사이트 단위 `enqueue_engram_kv_at`으로 옮기면 지울 수 있습니다(XS).
- **`gate_deepseek41_hc.rs` 사이트의 `engram_next`:** 이제 `form=` 라벨과 `nodes` 출력 줄에만 쓰입니다(XS).
- **`gate_deepseek41_step.rs` `shadow_sets`:** 스펙의 Q3(overlap=0 모드를 도는 레시피 없음)는 여전히 열려 있습니다. 이 검사는 그 모드를 거부만 합니다(XS–S).
- **`chain/ffn.rs` `enqueue_shadowed`:** 인자가 self 포함 8개라 `too_many_arguments` allow를 달았습니다. 조각들이 공유 버퍼를 평평하게 받는 기존 패턴과 같습니다. 나중에 `FfnIo`에 넣을 수도 있습니다(S).

## 6. 모델

opus로 스폰됐고 Opus 5.5(claude-opus-5-5)로 돌았습니다.
