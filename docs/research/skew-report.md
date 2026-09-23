# ktok-skew 라운드 보고

2행 어긋난 패스, `rollback`, 새 게이트를 구현했습니다. 두 행의 결과는 순차 두 스텝과 비트까지 같습니다. 한 행 공유 결함을 심은 FAIL-first는 빨강, 복원본은 녹색입니다. 노드 수는 예측과 정확히 같은 2220 = 2060 + 160입니다. 기존 게이트, ptx-scan 커널 표, check, lint(169)도 모두 기준 그대로입니다.

## 1. 바꾼 파일
- `/Users/hckim/repo/bloomery-skew/crates/gpu/src/hybrid.rs`: 페이지를 행 두 개로 나눴습니다.
  - 행마다 따로 두는 것은 layer 워드, Cnt 워드, 핸드오프 이미지, hsum입니다.
  - `Chain::{Step, Pair}`를 새로 두었고, 캡처 목록은 체인마다 따로 둡니다. 두 그래프가 공존합니다.
  - generation 검사는 한 번에 떠 있는 행 수만큼 앞선 것까지 허용합니다.
  - 기존 API는 모두 행 0 래퍼로 남겼습니다. `PAYLOAD_OFF`는 256에서 512가 됐습니다.
- `crates/gpu-deepseek41/src/chain/ffn.rs`: 행별 `FfnRow`를 두었고, 층 하나를 두 절반(`enqueue_go_half`, `enqueue_join_half`)으로 나눴습니다. `enqueue_shadowed`는 두 절반을 이은 것이라 노드 순서가 같습니다.
- `chain/attn.rs`: step words 사본을 행마다 둡니다(`enqueue_step_of`, `enqueue_layer_of`). scratch는 공유합니다.
- `chain/glue.rs`: engram 사이트 scratch를 행마다 둡니다. `EngramKv`에 `row` 필드를 더했습니다.
- `body.rs`: 행마다 streams, folds, 이미지 사본, 목록을 둡니다.
  - 한 행의 발사는 `Parts` 도우미 하나가 맡고, 단일 스텝(행 0)과 pair 패스가 그것을 같이 씁니다.
  - 새 API는 `enqueue_pair`, `decode_pair`, `rollback`, `row_buffers`, `history`, `row_bytes`입니다.
- `crates/gpu/src/model.rs`: `ChainBody`에 pair 관련 메서드 넷을 더했습니다. 기본 구현은 거절이라 V2-Lite 쪽은 건드리지 않았습니다.
  - `GpuModel`에 `step_pair`, `capture_pair`, `pair_logits`, `rollback`, `pair_graph_nodes`를 더했습니다.
  - 두 번째 head는 첫 `step_pair` 때 만들고, `resident_bytes`에 포함됩니다.
- 새 파일 `crates/gpu-gates/src/bin/gate_deepseek41_skew.rs`, 그리고 `justfile`에 레시피 `gate-gpu-ds41-skew` 하나(재독 뒤 마지막에 추가)를 더했습니다.

**행을 하나 더 두는 비용**은 디바이스 964,300 B이고 게이트가 찍은 값입니다. 페이지는 행마다 33,024 B의 pinned 호스트 메모리입니다.
```
load: 40 layers; a row adds: hyper-connection streams 163840 B, folded inputs 40960 B, step image 24688 B, selection lists 16384 B, attention step words 1440 B, ffn row 299164 B, engram site rows 417824 B
```
두 번째 행은 적재 때 늘 할당합니다. 캡처는 주소를 고정하기 때문입니다. 그래서 `gate_deepseek41_load`의 "besides the state" 출력 줄이 움직입니다. 이 줄은 핀이 아니라 출력만 합니다.

## 2. 명령과 판정 줄
**`BLOOMERY_REMOTE='~/repo/bloomery-skew' just gate-gpu-ds41-skew`** (rc=0, 복원 후 `Compiling bloomery-gpu-deepseek41` 확인)
```
graph pair: nodes=2220 kernels=2060 memops=160 other=0 []; 33 kernel names, each twice the step's: true: PASS
graph pair batches: 160 in the pass's order (go 5 ops, wait 2): PASS
graph pair prologue: [gather_pairs,ds41_glue_embed] per row, then row 0's layer 0 up to its go (17 kernels): PASS
shadow pair: 80 goes, row 0 40 and row 1 40 of 40 layers match the step's shadow table; engram token-only work 4 runs (want 4 = 2 rows x 2 sites), 4 inside a shadow: PASS
set ref_deepseek41_step4_every_node pair eager == two steps (both rows' logits, streams, folds, lists; 48 cache and 6 compressor buffers; history): PASS
set ref_deepseek41_step4_every_node pair replay == pair eager: PASS
set ref_deepseek41_step4_every_node pair + rollback(5) + step(t2) == step(t) + step(t2): PASS
set ref_deepseek41_d1_every_node pair eager == two steps (...): PASS
set ref_deepseek41_d1_every_node pair replay == pair eager: PASS
set ref_deepseek41_d1_every_node pair + rollback(302) + step(t2) == step(t) + step(t2): PASS
api Graph: step_pair(344, 11111) at 4 == step, step (tokens [11111, 16] vs [11111, 16], both logits bit for bit, pos 6): PASS; step_pair + rollback(5) + step(1613) == step, step (token 1 vs 1, logits): PASS
api Eager: ... PASS; ... PASS
PASSED: gate_deepseek41_skew
```
- 두 세트 모두 oracle build `db517b69`을 확인했습니다.
- 두 위치를 한꺼번에 되감는 `rollback`은 두 세트 모두에서 거절됐습니다(PASS).

**FAIL-first**는 `body.rs` 845행을 `self.lists.get_mut(row)`에서 `get_mut(0)`으로 바꿔 돌렸습니다. 행 1이 행 0의 목록을 씁니다. 파일은 백업 사본으로 되돌렸습니다.
```
set ref_deepseek41_step4_every_node pair eager == two steps (...): FAIL — row 0 lists 8 words, row 1 lists 31 words
set ref_deepseek41_d1_every_node pair eager == two steps (...): FAIL — row 0 logits 129280 words, row 0 streams 40960 words, row 0 folds 10240 words, row 0 lists 388 words, row 1 logits 129280 words, row 1 streams 40960 words, row 1 folds 10240 words, row 1 lists 506 words, caches 38402 words, compressor state 4096 words
set ref_deepseek41_d1_every_node pair + rollback(302) + step(t2) == step(t) + step(t2): FAIL — row 0 logits 129280 words, ...
GPU GATE RED: gate_deepseek41_skew (exit 1)
```

**첫 실행의 그늘 FAIL은 게이트의 기대값 오류였고, 완화한 것이 아닙니다.** 행 0의 0층 go 뒤에는 wait가 없어서 행 1의 0층 go 앞 커널 17개가 곧바로 붙습니다. 이것이 (a′) 스케줄의 프롤로그이고, 호스트 다리가 그 17개도 가립니다. 코드는 그대로 두고, 게이트에 프롤로그 기대값과 그 검사를 더했습니다.

**다른 게이트** (모두 rc=0, 전부 `BLOOMERY_REMOTE='~/repo/bloomery-skew'`)
- `just gate-gpu-ds41-step --structure --sets`: `graph: nodes=1110 kernels=1030 memops=80 other=0 predicted kernels=1030 memops=80: PASS`
  - shadow 줄이 전부 PASS입니다. `replay_bit_identical_to_eager=true`는 step4와 d1 둘 다입니다.
  - 마지막 줄은 `PASSED: gate_deepseek41_step`입니다.
- `just gate-gpu-e2e`: `gate_e2e: PASS — ...`
- `just gate-gpu-hybrid`: `gate_hybrid: PASS — ...`. 페이지 배치와 Cnt를 바꿨기 때문에 V2-Lite 경로도 돌렸습니다.
- `just gate-gpu-ds41-chain-ffn`: `121 checks ... 0 failed ... PASS`
- `just ptx-scan --features deepseek41 gate_deepseek41_step`과 `--features gpu gate_e2e`를 base와 새 트리에서 떴습니다.
  - 커널 행 표는 81행과 58행 모두 같습니다. gate_e2e는 표 md5가 `1989a031…`로 같습니다.
  - deepseek41 바이너리는 배너의 섹션 바이트만 1436080에서 1435976으로 다릅니다.
  - PTX를 뽑아 비교하면 차이는 이름 없는 공유 배열(`__shared_mem_N`)의 번호와 선언 순서뿐입니다. 번호를 지우고 정렬한 diff는 0줄입니다(`mod1 sorted_after_rename=0`, `mod2 sorted_after_rename=0`). 커널을 건드리지 않은 bloomery-gpu 번들에도 같은 번호 재부여가 있습니다.
- `just check` rc=0. `Cargo.lock`은 바뀌지 않았습니다.
- `just lint` rc=0이고 `grep -c '^warning:'`은 **169**, error는 0입니다. 마지막 편집(`resident_bytes`) 뒤에 다시 잰 값입니다.
- `check-recipes: ok`, `check-comments: ok`, `check-arch: ok`, `cargo fmt --check` rc=0입니다.
- `box-gc`는 처음과 끝 모두 `found 0 process(es)`, `gc-done`입니다.

## 3. 예측과 결과
예측은 첫 코드 편집 전에 적었습니다. 파일은 `…/scratchpad/skew/prediction.txt`이고 시각은 19:05:55Z입니다.

| 항목 | 예측 [유도] | 측정 |
|---|---|---|
| pair 노드 | 2220 = 2060 + 160, other 0 | 2220 = 2060 + 160, other 0 |
| 커널 이름별 개수 | 단일 스텝의 정확히 2배 | 33개 이름 모두 2배 |
| memop 순서 | go A0·B0, 층마다 wait·go 넷, 끝 wait 둘 | 일치 |
| FAIL-first | step4는 목록만, d1은 logits·스트림·캐시까지 | 일치 |
| ptx-scan | 커널 행 표 동일 | 일치 |

**스펙 §2의 "압축기 상태는 건드리지 않음"을 정정합니다.**
- B의 스텝(p+1)은 persist로 잔여 슬롯 `(p+1)%r`을 덮어씁니다.
- 대체 스텝(p+1)은 어떤 그룹도 읽기 전에 같은 슬롯을 다시 persist합니다. 대체 스텝이 읽는 링 슬롯은 q < p+1 위치의 잔여뿐이고, `(p+1)%r`은 그중에 없습니다(`plan.rs:401-433`).
- 그래서 한 위치 되감기는 복원할 것이 없습니다. d1의 (iv)가 압축기 상태 4096워드를 포함해 비트 동일로 이를 확인했습니다.
- 두 위치 되감기는 창 링과 압축기 슬롯 모두 복원이 필요합니다. 그래서 `rollback`은 마지막 한 위치만 받습니다.

**E12 시간 측정(리드 몫).** 예측은 패스당 ≈ 2H + D = 49.8 + D ms [유도, ktok]입니다. 아래 hunk를 적용한 트리에서 같은 임대 안에 레버 팔과 레버 없는 팔을 교대로 돌리면 됩니다.
```
BLOOMERY_BOX_ENV="BLOOMERY_STEP_PAIR=1" just depth-gpu-ds41 6 1024
just depth-gpu-ds41 6 1024
```
pair 팔의 시간 행은 패스 하나, 즉 두 위치의 ms입니다. 따라서 SMOKE의 `tok/s(p50)`는 패스/s이고 **×2를 해야 tok/s**입니다.

**`generate_ds41` hunk** (probes 라운드의 파일이라 적용하지 않았습니다, 약 10줄)
```
+        let pair = std::env::var("BLOOMERY_STEP_PAIR").is_ok_and(|v| v == "1");
+        let mut draft = next;
         for _ in 1..a.n_gen {
             let t0 = Instant::now();
-            next = m.step(&[next])?;
+            if pair {
+                [draft, next] = m.step_pair(next, draft)?;
+            } else {
+                next = m.step(&[next])?;
+            }
```
여기서 드래프트는 직전 패스의 행 A argmax입니다. 롤백 없이 매 패스를 수락으로 치므로(수락률 1.0) 순수한 타이밍 팔입니다. B의 토큰은 패스를 시작할 때 필요해서, 같은 패스의 A argmax는 쓸 수 없습니다.

## 4. 못 한 것과 주의
- 시간은 재지 않았습니다(리드 몫).
- 박스에 보조 디렉터리 `~/repo/bloomery-skew-base`가 남았습니다. ptx-scan base용이고, 그 안의 target이 수백 MB입니다.
  - 마지막에 box.sh 호출을 잘못 한 번 해서 그 디렉터리를 **새 트리로 덮어** rsync했습니다. 그래서 더는 base가 아닙니다.
  - `box-tracks --remove`가 aux로 잡으니 워크트리를 치울 때 함께 지우면 됩니다.
- 두 번째 ptx 실행은 zsh가 `set -- $pair`에서 단어를 나누지 않아 인자 오류로 실패했습니다. bash로 다시 돌린 결과가 위의 값입니다.

## 5. 스펙 밖 개선 지점 (보고만, 손대지 않았습니다)
- `gate_deepseek41_skew.rs`: step 게이트의 `open_set`, `set_state`, `comp_ring`, `inject`, `walk`, `describe` 약 350줄을 복제했습니다. lib는 ds41 크레이트를 이름할 수 없으므로 `src/bin/common/`을 `#[path]` 모듈로 두어 한 주인으로 모으는 것이 후보입니다. 크기 S.
- `SHADOW_CARD`, `SHADOW_HOST`, `ENGRAM_KV` 표가 이제 bin 셋에 있습니다. ffn 조각이 이 표를 내보내면 주인이 하나가 됩니다. 크기 S.
- `hybrid.rs`: `Boundary.hsum` 필드는 `pages[0].hsum`과 같은 메모리를 가리키는 두 번째 창입니다. `deepseek2/dispatch.rs:999`가 필드를 직접 읽어서 남겼습니다. 크기 XS.
- `HybridStats.go_early`는 pair 모드에서 "다음 행의 go가 이미 도착"한 경우까지 셉니다. pair의 통계로 쓰려면 의미를 다시 정해야 합니다. 크기 XS.
- `GpuModel::logits()`는 `step_pair` 뒤에 행 A의 logits를 돌려줍니다. 문서에 한 줄이 필요합니다. 크기 XS.

## 6. 모델
opus로 스폰되어 Claude Opus 5.5(`claude-opus-5-5[1m]`)로 돌았습니다.
