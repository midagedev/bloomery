# V2-Lite·CPU 엔진 격리: 디렉터리 셋, 검사되는 목록 하나, 라운드 넷 (라운드 `v2fencedesign`)

**상태.** 3파동 `v2fence`의 설계 메모다(2026-09-27, 박스 없음, 트리 `cdd90f3`). 사용자 결정 1(09-27)은 V2-Lite와 CPU 엔진을 먼저 격리하고, 대체 게이트가 선 뒤에 은퇴시킨다는 것이다. 이 메모는 그 격리를 종이 위에서 설계했다. 아래 본문은 라운드의 보고를 그대로 옮긴 것이고(§12 모델 줄만 뺐다), 리드의 처분은 이 머리에 적는다. 줄 번호는 `cdd90f3` 기준이다. del2(stage 0 삭제)가 그 뒤에 착륙하므로, 라운드 스펙을 쓸 때 다시 짚는다.

**리드의 처분(09-27).**
- **§10 리드 질문 1 — K2로 간다.** 커널은 옮기지 않는다. V2-Lite만 부르는 커널은 `tools/v2fence.txt` 목록에 올리고 은퇴 때 delete 급으로 지운다. 3파동 `opslib`가 커널을 모델 이름 없는 계열로 다시 짜므로, 지금 `arch/` 아래로 옮기면 한 번 더 옮기게 된다. §11의 불일치(결정 8의 문서는 "커널은 루트"라고 하는데 Qwen3 커널이 `arch/qwen3moe`에 있음)는 opslib가 정리한다.
- **질문 2 — opslib의 계열 인스턴스.** V2-Lite의 Softmax(64, 6) 라우터와 latent 플래시는 opslib 계열의 인스턴스로 남긴다. 라우터 (E, K) 코어와 latent LATENT × ROPE 계열에 V2-Lite의 모양이 하나 더 붙는 것이라 비용이 거의 없고, 커버리지가 남는다. Q5_0/Q5_1은 modelspec의 커버리지 검사가 대상 파일(Qwen3.6, GLM-5.3-Flash)에서 찾지 못하면 V2-Lite와 함께 은퇴한다.
- **질문 3·4 — 엔진 표면.** `gpu::Engine`은 3파동 `session`이 지운다(제네릭 사용자는 `gate_e2e.rs:250` 하나이고, 그 게이트는 V2-Lite 모델을 직접 쓰게 한다). `v2fence-gates`는 그 뒤, modelspec step 4 전에 간다. modelspec은 deepseek2 리더를 새로 만들지 않는다. V2-Lite는 격리 안에서 자기 `Hparams`를 읽는다.
- **질문 5 — 등록부의 경로 리터럴 규칙은 다음 도구 라운드가 가진다.** `registry.rs`의 `.rs` 경로 리터럴이 그 파일을 레버 크레이트를 링크하는 모든 레시피의 입력으로 만든다. `recipes.py`가 lib 입력에서 그런 리터럴을 빼야 CPU 격리의 효과(87 → 11)가 선다. levers2의 L9(`BLOOMERY_GATE_BOUND` 행의 `tools/gate.sh` 리터럴, `why` 95개)와 같은 부류이므로 한 규칙으로 닫는다. v2fence-cpu보다 먼저 착륙해야 한다.
- **질문 6 — 예.** `ATTN_HALVES`·`POPULATE`·`WEIGHTS`의 삭제는 v2fence-cpu의 첫 커밋(delete 급, 케이스마다 날짜 사유)으로 한다. `FLASH_SEG`의 은퇴는 v2fence-gpu가 한다(`flash.rs`).
- **질문 7 — 예.** v2host와 v2fence-cpu는 v2host 앞의 base 하나에 대해 한 착륙 묶음으로 검증한다. 둘 다 판정 줄이 base와 같다고 예측하기 때문이다. 빨강이 나오면 v2host의 트리에서 따로 돌려 가른다.
- **질문 8 — KLD 빨강 줄을 exact_ref 자리의 후보로 받는다.** 만드는 것은 은퇴 준비 라운드(3파동 뒤)다. AGENTS "Performance first"의 원칙대로 이것은 버그 잡이이지 정밀도 순위가 아니다.
- **hosttier 질문 열하나 — 03의 답(09-27).** 줄 번호는 `31fe26e` 기준이고, r8land가 `hybrid.rs`·`union.rs` 줄을 17~35줄 민다. `REPLAY`와 재생 감시는 그 전에 답이 왔다. 지우기로 했고, gpumodel이 이름 붙은 hunk로 가져간다.
  1. `hybridgate`는 gate_hybrid의 스텁 절반만 잇는다. 하나는 거부 팔이다(`Stub` 타입, `refusal_tier`·`serve_step`·`plant`·`answer_ok`·`show`·`recorded_once`·`reset_cases`·`refusal_arm`, card_sel 절, :751–1368). 다른 하나는 hostone D가 지울 때까지의 배제 팔이다(:1369–1478). V2-Lite 팔(구조, eager = replay, 층별 밴드, e2e argmax, 엔진 NaN 절)은 V2-Lite 커버리지라 `v2fence-gates`가 옮긴다. off 팔은 hostcfg가 `HYBRID_OVERLAP` 은퇴와 함께 날짜 사유로 지운다. 순서는 03의 제안대로다. hybridgate(S)가 hostcfg 바로 뒤, v2fence 전에 가서 스텁 절반을 모델 없는 게이트로 먼저 뺀다. v2fence가 먼저 준비되면 03에게 알린다. 그때는 v2fence가 통째로 옮기고 03이 거기서 hunk로 뺀다.
  2. `HYBRID_OVERLAP`은 hostcfg가 은퇴시킨다(등록부 Retired, `=0` 가지 삭제). `hybrid.rs`와 aa의 `chain/ffn.rs` 가지는 03이 hunk 목록을 준다.
  3. `HybridConfig`와 NL 읽기는 V2-Lite 전용이라 격리가 옮긴다. V4.1은 plan에서 카드 집합을 받고, HostCfg는 NL을 흡수하지 않는다. hostcfg가 먼저 착륙하면 `Levers`에는 `n_l`만 남는다.
  4. v2host 뒤 죽는 자유 함수 둘(`moe.rs`)과 `Weight::in_file`·`ExpertStack::in_file`(`ops.rs`)은 v2host가 지운다. 마지막 호출자를 없애는 변경이 피호출자도 지워야 lint가 오르지 않는다. 03 파일이므로 이름 붙은 hunk와 착륙 전 목록을 보낸다. 03은 hosttier §2.6의 그 행들을 v2host가 닫는 것으로 적는다.
  5. 03이 `moe.rs`·`ops.rs`를 편집하는 순서는 이렇다. r8land(착륙 중), hostcfg(levers2 뒤; `ops.rs`의 POISON·STEAL·STEAL_BLOCKS·DEFER_QUANT, `moe.rs`의 EXPERT_LOG), 빈 창, hostone C·D다. C·D는 B 뒤, 곧 gpumodel·q3act B·hostone A 뒤에 간다. benchprune과 hostone A·B는 두 파일을 만지지 않는다. v2host와 v2fence-cpu는 hostcfg 착륙부터 hostone C 발사 전까지의 창에 간다. 03이 hostcfg 착륙 때 한 줄 알린다. 그 창에 aa의 라운드가 날고 있으면 C가 기다린다.
  6. 둘 다 아니다. C는 V4.1 디코드 경로(`generate_ds41`)를 바꾸므로 `ab-decode`(V2-Lite CPU 엔진)는 해당하지 않는다. D의 `group_core` 편집은 도달 불가 가지의 삭제라 비트와 grep으로 증명한다. `bench_v41_host`는 `--check` 판정 동일로만 들어간다. 사용자가 비열등 팔을 원하면 V4.1 시팅의 디코드 행에서 두 바이너리를 돌리고, 여유는 4라운드 잣대다. C 발사 전에 사용자에게 묻는 질문이 이것이다.
  7. `profile.rs` 훅은 둔다. 호스트 티어의 청크 스큐 계기는 둘이다. 하나는 union 경로다(`ops.rs`의 `run_row_pool`·`claim`·`UnionRows`·`record_union_rows`). 다른 하나는 `moe.rs`의 serve 훅이고, C가 serve를 지울 때까지 남는다. CPU 엔진 자체의 훅은 CPU 엔진과 함께 나간다(deepseek2 attn·forward·plan, `ffn.rs` swiglu, `moe.rs` route·setup·io·trace, `head.rs`, 그룹 엔진 쪽). `profile.rs`는 `crates/model`에 남는다(`crates/cpu`가 model에 의존한다). 레벨은 v2fence가 `levers-direct.txt` 목록대로 바꾼다. 제자리에서 읽던 것을 main에서 파싱해 한 번 설정하게 하고, 두 엔진을 도는 빈 모두에 적용한다.
  8. `tests/ds41_host.rs`의 `Meta` 단언은 지운다. 날짜 사유: `moe::Meta`가 CPU 엔진과 함께 나가고 V4.1 호스트 티어는 `Hparams`만 읽는다. 한 크레이트의 두 리더를 맞대던 절이 할 일을 잃는다.
  9. 받는다(03 시험 파일, 이름 붙은 hunk와 목록). `union.rs`의 V2-Lite 케이스는 줄 번호가 아니라 `hw_union_file_matches_per_column_v2lite`라는 이름으로 찾는다.
  10. 03이 `q3gates`에서 옮긴다. gate_p8의 GEMM 벤치 팔(`gemm_arms`, `--bench-arm`)은 GEMM 계열의 게이트 빈(`gate_gemm`)으로 간다. `bench-gpu-kernels`의 GEMM 몫, `ncu-gpu-gemm`, `tools/ref/ncu-gpu.sh`의 `GEMM_BIN`은 새 자리를 가리키게 하고, bench 줄은 글자 그대로 둔다. q3gates는 q3input 착륙 직후라 v2fence보다 먼저다.
  11. q3act B에는 이음이 없다(타입만 바뀐다). C는 기록기 본체에 평면별 행 수를 붙인다. V2-Lite 기록기 셋(`q5.rs:429` 짝의 b팔, `flash.rs:941`·`:1294`, §3의 6·7·9번)은 진입 안에서 모든 평면 행 수를 `m_cols`로 넘긴다. 새 커널 매개변수도 런처 변경도 없으니 v2fence가 만날 새 이음이 없다. 세 엔트리의 PTX 행은 C의 ptx-scan diff에 나온다. 03이 이 제약을 q3act 계획에 적는다.
  - **§6 표가 달라지는 곳.**
    - 순서는 hostcfg → hybridgate(03) → v2host·v2fence-cpu(한 묶음, hostone C 전 창) → v2fence-gpu ∥ v2fence-gates다.
    - `v2fence-gates`는 gate_hybrid의 V2-Lite 팔만 옮긴다.
    - v2fence-cpu의 범위에 `profile.rs` 레벨을 main에서 파싱하도록 바꾸는 일이 더해진다.
    - recipes.py의 경로 리터럴 규칙(질문 5의 처분)은 여전히 v2fence-cpu보다 먼저 착륙해야 한다.
- **§11**의 `Gpu::with_device`가 크레이트 번들을 아홉 번 적재한다는 지적(M)은 트리아지에 올린다. 먼저 `cuMemGetInfo` 전후로 잰다. cuda-oxide의 `#[cuda_module]` 적재가 부모 적재를 나누지 못하는 것이라면 nvlabs 원장에 한 줄 적고 포크에서 고친다(사용자 규칙 09-27).

## §0 권고

1. 격리는 디렉터리 셋과 검사되는 목록 하나로 한다.
   - `crates/cpu/`: CPU 엔진.
   - `crates/gpu/src/arch/deepseek2/`: V2-Lite GPU 호스트.
   - `crates/model/src/arch/deepseek2/`: CPU와 GPU가 같이 읽는 서술(`Hparams`, `names`, 새 `mla.rs`·`wkb.rs`·`host.rs`).
   - `tools/v2fence.txt`: 은퇴 체크리스트.
   - 셋째 디렉터리는 피할 수 없다. CPU 엔진과 GPU가 서로에 의존할 수 없기 때문이다.
2. 커널은 옮기지 않는다(K2, 권고). V2-Lite만 부르는 커널은 목록에 올리고, 은퇴 때 delete 급으로 지운다.
   - 근거: 결정 8(`docs/gpu-design.md:39`, `crates/gpu/src/arch/mod.rs:5-8`)이 커널은 루트에 두고 호스트 세 층만 `arch/` 아래에 둔다고 정했다. 3파동 opslib도 라우터 (E, K)와 latent LATENT × ROPE를 계열로 만든다.
   - GC7 2단계 문안(커널도 `arch/` 아래로, K1)과 다르다. K1 계획도 §2에 적었다. 아래 절은 모두 K2 기준이다.
3. CPU4는 옛 API를 격리 안에 남기지 않고 옮긴다.
   - V2-Lite 호스트 티어를 `Split` + `HostLayer`로 옮긴다.
   - GPU 적재는 CPU의 `Derived` 대신 `MlaParams::read(gguf, l)`과 `wkb` 함수를 부른다.
   - 비트 동일은 구성으로 선다: `swiglu_clamp(limit 0)` = `swiglu`, 같은 바이트, 같은 `serve`.
4. 라운드는 넷이다: `v2host`(S–M, API 변경) → `v2fence-cpu`(L, delete 한 커밋 + move) → `v2fence-gpu`(S, move) ∥ `v2fence-gates`(M). 뒤 둘은 선행 조건이 같이 풀리면 하나로 합친다.
5. 효과에는 조건이 있다. 등록부의 경로 리터럴(`registry.rs:32,34`)이 `crates/cpu` 파일을 레버 크레이트를 링크하는 모든 레시피의 입력으로 만든다. `recipes.py`가 `.rs` 리터럴을 lib 입력에서 빼야 87 → 11이 된다.
6. 침식 방지 규칙은 셋이고, 각각 FAIL-first를 붙인다.
   - ⑤ `bloomery-cpu`를 의존하는 크레이트가 0이다.
   - ①·④의 OWN에 `crates/cpu/src/`를 넣는다.
   - ⑥ V2-Lite 이음과 V2-Lite 전용 커널을 닫힌 목록으로 둔다.
7. 대체 게이트:
   - gate_hybrid의 스텁 팔(`gate_hybrid.rs:751-1478`, 모델 파일 없음)은 3파동에 떼어 살린다(S).
   - 구조, eager = replay, 거부와 reset은 V4.1 step이 이미 덮는다.
   - ik 토큰 대조는 Qwen3 e2e (g)가 이미 한다.
   - exact_ref의 자리에는 KLD 빨강 줄이 가장 싸다(S). 채택 여부는 리드가 정한다.
8. 레버:
   - `PIN_MAIN`·`PROFILE`은 `bloomery-decode`의 `at_main`으로 옮긴다.
   - CPU 쌍둥이 다섯은 격리 안에서 제자리에 두고 경로만 바꾼다.
   - `ATTN_HALVES`·`POPULATE`·`WEIGHTS`·`FLASH_SEG`는 은퇴(Retired)시킨다.
   - `FLASH_MMA`는 `left: []`로 만든다.
9. 03 서명이 필요한 곳:
   - `moe.rs`: CPU MoE 반출, 자유 함수 둘 삭제.
   - `ops.rs`: `pub` 셋, `in_file` 둘 삭제.
   - `hybrid.rs`: `HybridConfig` 이동(선택).
   - 호스트 티어 시험 넷.
   - `fused.rs`·`flash_gqa*`·`gemm*`·`qdot`·Qwen3 디렉터리·`r8file.rs`·`deepseek41/host.rs`는 0이다.

**종이 위 분해.**
- 옮길 줄 [유도, 범위는 §1]:
  - CPU 엔진 ≈5,025줄 + 시험 ≈3,658줄.
  - model에 남는 V2-Lite 서술 ≈868줄.
  - GPU 호스트 쪽 이동 ≈330줄(`model/probe.rs` 272 + `HybridConfig` ≈40 + `q8_derived` ≈12). K1이면 커널 쪽 ≈5,267줄이 더해진다.
- 제품 스텝 Δt:
  - V4.1·Qwen3의 디코드와 프리필은 0이다 [유도]. 커널, 런치, 호스트 디스패치, 배치가 하나도 바뀌지 않는다.
  - V2-Lite 하이브리드 스텝은 같은 `serve`를 부른다. 달라지는 것은 전문가마다 뷰 셋 대신 슬라이스 셋을 자르는 것뿐이라 ns 단위다. 이 경로는 제품 경로가 아니다.
  - 그래서 어느 격리 라운드도 같은 임대 A/B를 빚지지 않는다. 임대를 잡는 실행이 없으니 카드도 없다.
- 박스 비용은 착륙 묶음뿐이다. 값은 따뜻한 빌드 중앙값이다: 레시피마다 `~/.cache/bloomery/gate-times.tsv`의 마지막 5행 중앙값, 2026-09-27 07:32에 읽음.
  - 차가운 빌드는 더 길다. 예를 들어 `gate-gpu-e2e`의 옛 행은 142–216 s이고 최근은 37–49 s다. `gate-ds41-load`는 행이 없다.
  - gate-gpu-* 65개 합은 ≈4,412 s다. 그중 V2-Lite GPU 게이트 17개가 ≈177 s, CPU 게이트 11개가 ≈35 s다.
  - V2-Lite는 공유 편집 묶음의 ≈4 %다. 따라서 격리는 공유 편집의 묶음 벽시계를 거의 줄이지 않는다.
  - 이득은 구조에 있다: 은퇴가 목록 하나로 끝나고, GPU → CPU `Derived` 결합이 없어지고, 옛 호스트 API가 지워진다.
  - 시간 이득은 두 곳이다. CPU 전용 편집은 전체 목록 대신 11개(≈35 s)만 돈다. `v2fence-cpu`의 호스트 전용 착륙은 ≈575 s다 [유도: CPU 11개 35 + gate-ops 10 + gate-union 17 + gate-ds41-host 5 + gate-gpu-hybrid 57 + ds41-step 42 + ds41-prefill 409].

**자원 시간선(단위 = 라운드 하나의 착륙).**
- 맥: 정적 검사뿐이다.
- 박스 호스트: 빌드. cargo 락이 두 레인의 빌드를 직렬화한다(`tools/gate-batch.sh:47-48`).
- 3090 레인: V4.1 `--place gate` 게이트는 3090에서만 돈다(AGENTS). 최장 항목은 ds41-prefill 409 s다.
- A6000 레인: `any` 게이트가 돈다.
- 벽시계는 긴 레인 쪽 시간 + 빌드다. 전체 목록이면 30–90분이다(rebuild §7-3).

**흐름 레버.**
- 벌크: `v2host`와 `v2fence-cpu`를 v2host 전 base 하나에 대해 한 묶음으로 검증하면 전체 묶음이 하나 빠진다. 둘 다 verdict가 base와 같다고 예측하기 때문이다.
- 비동기: `v2fence-cpu` ∥ `v2fence-gpu`를 워크트리 둘로 돌린다. 등록 파일의 주인은 한 라운드로 정한다.
- 제품 스텝의 항은 격리로 움직이지 않으므로 줄일 항이 없다.

## §1 Q1 인벤토리

**정정(측정·유도).**
- **GC7 "6,577줄"은 오늘 6,279줄이다.**
  - `wc -l crates/gpu/src/arch/deepseek2/*.rs` = 3,659, `wc -l crates/gpu/src/{router,q5,moe_fused}.rs crates/gpu/src/model/probe.rs` = 2,620.
  - 줄어든 곳은 `arch/deepseek2` 3,815 → 3,659, `model/probe.rs` 414 → 272다.
  - GC7 합에는 공유 파일 안의 V2-Lite 전용 부분 ≈2,647줄이 빠져 있다 [유도]: `flash.rs:740-2526` 1,787, `elem.rs` 커널 넷 ≈344, `fused.rs` 둘 ≈270, `model/kernels.rs` heads_pair ≈246. 합은 ≈8,926줄이다.
- **GC7의 "계기가 공용 모듈에 섞여 있다(분기 8, `tick` 43)"는 낡은 전제다.**
  - probe 설정 분기점은 5곳이고 전부 `arch/deepseek2/dispatch.rs`에 있다: `:173 :608 :641 :882 :1170`. `grep -n 'skip_quant\|pad_per_layer'`로 확인했다.
  - `tick(` 호출은 38곳이다(dispatch 37 + taps 1, `grep -c 'tick('`).
  - 격리 밖에 있는 것은 정의 파일 `model/probe.rs`(:203, :248)와 `GpuModel`의 `set_probe`·`seed_depth`·층 캡처뿐이다. 뒤의 셋은 gpumodel이 옮긴다.
- **CPU1 "≈4,600줄"은 ≈5,025줄이다 [유도, 줄 범위].**

  | 부분 | 계산 | 줄 |
  |---|---|---:|
  | `attn.rs` | 3,240 − 공유 259(:246-331, :353-515, :2364-2373) | 2,981 |
  | `derived.rs` | 358 − `build_block` 38(:321-358) | 319 |
  | `forward.rs` | | 331 |
  | `plan.rs` | | 131 |
  | `head.rs` | | 90 |
  | `kv.rs` | | 191 |
  | `ffn.rs:65-93` | | 29 |
  | `Slot`(`lib.rs:36-42`) | | 7 |
  | `moe.rs:24-746` | 723 − `expert_view` 28 − `expect_stack` 27 | 668 |
  | `bloomery-decode.rs` | | 278 |

  - 시험은 ≈3,658줄이다(`wc -l crates/model/tests/*.rs`; `moe.rs`는 CPU 케이스 :1-224만 셈).
  - 차이의 대부분은 CPU1이 세지 않은 CPU MoE다.
- **CPU1 "58개 중 54개"는 오늘 94개 중 87개다.**
  - `python3 tools/recipes.py why crates/model/src/arch/deepseek2/attn.rs` = 87, `head.rs` = 84.
  - 게이트 수는 `just --summary | tr ' ' '\n' | grep -c '^gate-'` = 94다.
  - 87과 84의 차이 셋(`gate-levers`·`gate-qdot`·`gate-threads`)은 등록부의 경로 리터럴이 들여온다.
  - del2가 `gate-gpu-p0`(del2.md:59)와 `gate-gpu-ds41-prefill-q3ksplit`(:66)을 지우면 85/92가 된다 [유도].

**분류(K2).**

| 파일·항목 | 분류 |
|---|---|
| `model/arch/deepseek2/attn.rs` | CPU 2,981 + 서술 259(`MlaParams`, `RopeParams`, `rope_yarn_ramp`, `find`, `Q8Block`/`quantize_q8_0`, `wkb_row_bytes`). 서술을 쓰는 곳: GPU `mod.rs:30`, `dispatch.rs:22`, `scratch.rs:14`, `taps.rs:173`, `exact_ref.rs:44` |
| `derived.rs` | CPU `Derived` + 서술 `build_block`(wk_b requant). GPU는 `derive_blocks`(mod.rs:458-493)와 `gate_p10.rs:70`에서 `Derived`를 거쳐 쓴다 |
| `forward.rs`, `plan.rs`, `model/{head,kv}.rs`, `bin/bloomery-decode.rs`, `Slot` | CPU |
| `hparams.rs` 417, `names.rs` 93, `mod.rs` 15 | V2-Lite 서술(GPU `pins.rs:10`, `scratch.rs:15`와 공유) |
| `ffn.rs` | `swiglu`·`swiglu_timed`는 호스트 티어가 쓴다(`ops.rs:6`). dense 부분 :65-93은 CPU |
| `profile.rs` 442 | 공유. 훅은 `ops.rs`에 47줄, `moe.rs` 호스트 부분에 4줄 있다(`grep -c 'profile::'`, `awk 'NR>=747'`). 레벨을 세우는 것은 CPU 엔진뿐이다 → §10 |
| `moe.rs` | CPU :24-746 / 호스트 티어 :747-(03) / 공유 `expert_view` :327, `expect_stack` :376 / V2-Lite 전용 옛 API: `experts_into` :767-805, `expert_of` :807-814, `experts_union_into` :1406-1424 |
| `ops.rs` | 호스트 티어(03). CPU가 쓰는 `pub(crate)` 셋: `SharedOut` :461, `ErrGate` :2875, `matvec_q_local` :4148. V2-Lite 전용 `ExpertStack::in_file` :1040, `Weight::in_file` :1067 |
| `placement.rs` | 공유. `CardFormat::Q5_0/Q5_1`(:242-246, :276-277, :328-331, :452-453)은 이음이다 |
| GPU `arch/deepseek2/*` 3,659 | V2-Lite GPU 호스트 |
| `gpu/src/model/probe.rs` 272 | V2-Lite 계기. 머리말 :1-3의 "shared by every architecture"는 거짓이다 |
| `gpu/src/model.rs` | 공유 골격. V2-Lite 전용 항목: `Engine` impl :1354-1378, `load_full`의 `HybridConfig::from_levers` :434-437, `set_probe` :621, `seed_depth` :660, 층 캡처 :1171-1264(gpumodel이 옮긴다) |
| `model/lookup.rs` | `q8_derived` :18은 V2-Lite 전용 |
| `model/kernels.rs` | `q3k_gemv_heads_pair`(:178-281, :458-599)는 V2-Lite 전용 커널. `gather_pairs`·`q8_0_gemv_heads`는 V4.1 공유 |
| `flash.rs` | 공유 :1-736(V4.1 `attn.rs:56`과 `flash_gqa.rs:54`가 가져가는 상수·도우미·`mma_segment_walk!`). V2-Lite 전용 커널 :740-2526, 상수·`seg_keys` 등 :90-107·:143-187·:241-278 |
| `router.rs` 528, `q5.rs` 1,426, `moe_fused.rs` 394 | V2-Lite 전용 커널(호출자 기준). `q5.rs`의 `pack_q5_0/1`(:926-937)은 `weights.rs:13`이 쓴다 |
| `elem.rs` | V2-Lite 전용: `embed_rows` :357, `rope` :513, `card_sel` :557, `weighted_sum` :639(호스트 :945, :1078, :1135, :1232). 게이트 전용 `half_decode`·`half_encode_check`는 공유 f16 도우미의 계약이다(gate_p4) |
| `fused.rs`(03) | V2-Lite 전용 `gate_up_swiglu_q3k` :428, `down_add_q5_1` :483(호스트 :741, :826). `norm_quant`·`kv_norm_rope_append`는 공유 |
| `hybrid.rs`(03) | 공유. `HybridConfig`·`from_levers`·NL 읽기(:95-171)는 V2-Lite 전용이다(문서 :136-139). `overlap`은 V4.1도 읽는다(`gpu-deepseek41/src/body.rs:2253`) |
| `fault.rs`(03) | `Q5Quant = 3`(:71)은 `q5.rs:331`만 올린다. `step_order::DEEPSEEK2`는 :161-173 |
| `weights.rs` | `DevWeight::Q5_0/Q5_1/Q8_0Derived`(:46-51, :67, :626-637)는 이음이다 |
| `gpu/src/lib.rs` | `Deepseek2Model` :66. V2-Lite 전용 셋 넷을 적재한다(:2095-2101, :2185-2193) |
| `gpu-deepseek41/src/draft/load.rs:255-256` | V4.1 크레이트 안의 V2-Lite 이음(`CardFormat` 망라 match) |
| gpu-gates lib | V2-Lite 전용: `block.rs` 788, `oracle/deepseek2.rs` 26, `ref_dir` 기본값 :341-353, `V2_LITE_ROUTER`/`route_ref` :614-623, `engine.rs`의 Deepseek2 팔, `prompts.rs`의 `compare_forced_exact`/`read_exact`. `compare_greedy`는 Qwen3와 공유 |
| V2-Lite 빈 20개, 14,482줄(`wc -l`) | gate_hybrid 1,688, p10 1,220, p8 1,156, p5 1,301, p4 1,044, p6 954, q4k_sel 950, e2e 903, p0b 776, moe_fused 703, exact_ref 678, generate 492, p8b 445, p9 394, p1 366, p3 348, forced_probe 332, p2 272, head_gpu 234, block 226 |
| 데이터·도구 이음 | gate-1-1의 V2-Lite 여섯 타입 덤프(Q5_0/Q5_1은 여기뿐, justfile:709-715), qdot 덤프 넷(`build-qdot-ref.sh:14`), `tests/ops.rs`의 stage-1 오라클, 기본 프로필(`ref-paths.sh:35`, `recipes.py:83`), `lease.sh:358` DECODE_BIN, `ncu-gpu.sh:230` |

한 가지 주의할 점이 있다. `gate_p8.rs`에는 Qwen3 GEMM 벤치가 같이 산다(`gemm_arms`, :763-778, :916). 은퇴 전에 빼야 한다.

## §2 Q2 절단

**목표 트리(K2).**
- `crates/cpu/`(패키지 `bloomery-cpu`, lib `cpu`)
  - 옮겨 오는 것: `attn.rs`, `derived.rs`(`build_block` 대신 `model::arch::deepseek2::wkb` 호출), `forward.rs`, `plan.rs`, `head.rs`, `kv.rs`(+`Slot`), `ffn.rs`(dense), `moe.rs`(CPU MoE 668줄), `bin/bloomery-decode.rs`.
  - 시험 11개는 `#[path = "../../model/tests/common/…"]`로 공통 하네스를 읽는다. 복사하지 않는다.
  - 의존: model, gguf, qdot, threads, levers. dev 의존: refset.
- `crates/model/src/arch/deepseek2/`: `mod`, `hparams`, `names`, 그리고 새 파일 셋.
  - `mla.rs`: `attn.rs`의 세 범위를 그대로 옮긴다.
  - `wkb.rs`: `derived.rs:321-358`을 그대로 옮긴다("op order is the contract").
  - `host.rs`: `layer(split, hp, embd, l) -> Result<Option<HostLayer>>`, `swiglu_limit: 0.0`. `deepseek41/host.rs:15-44`와 같은 모양이다. `embd`는 호출자의 `hidden`에서 받는다. V2-Lite `Hparams`에는 `n_embd`가 없다.
- `crates/model`의 나머지(`ops`, 호스트 티어 `moe`, `ffn`의 swiglu 둘, `profile`, `placement*`, `r8file`, `fileio`, `arch/{deepseek41,qwen3moe,dspark}`): 호스트 티어는 03이, 나머지는 aa가 맡는다.
- `crates/gpu/src/arch/deepseek2/`로 오는 것:
  - `probe.rs`(← `model/probe.rs`)
  - `HybridConfig`/NL 읽기(← `hybrid.rs:95-171`, 03 서명)
  - `q8_derived`(← `model/lookup.rs:18-29`)
  - gpumodel 뒤의 `Instrumented`. inherent `impl GpuModel<Body>`로 바꾼다(`mod.rs:593-681`의 `HybridTaps`가 선례다).
- 커널과 그 호스트 래퍼는 루트에 남는다. 목록은 `tools/v2fence.txt`의 kernel 절에 둔다.
- `crates/gpu-gates`:
  - V2-Lite 빈은 `src/bin/deepseek2/`로 옮긴다. `[[bin]] path`만 바뀌고 이름과 레시피는 그대로다.
  - 스텁 팔은 모델 없는 새 빈으로 뗀다.
  - `engine.rs`의 Deepseek2 팔을 지운다(§6).

**K1(GC7 문안대로 커널도 옮기는 경우).**
- 추가로 옮기는 것:
  - 통파일: `router.rs`, `q5.rs`, `moe_fused.rs`. `q5.rs`의 pack 둘은 `weights.rs` 옆 공유 자리에 남긴다. 옮기면 `weights.rs:13`이 arch 경로를 적게 되어 ④ 위반이다.
  - 부분: `flash.rs:740-2526`과 V2-Lite 상수, elem 넷, fused 둘(03), kernels heads_pair.
  - `Gpu`의 셋 넷(lib.rs:2095-2101, :2185-2193, :2382-2415)은 V2-Lite 몸체가 적재한다.
- 크기: ≈5,267줄 [유도], M–L.
- 증명:
  - rows+md5가 동일하다(gemmsplit `760825c`가 선례다). `bytes=`는 움직일 수 있다(ledger #22).
  - V4.1·Qwen3 프로세스의 번들 적재가 9 → 5가 된다 [유도]. 적재 경로가 바뀌므로 주인 게이트는 `gate-ds41-load`(justfile:1029, 적재·드롭 바이트를 본다), `stage-gpu-load-v41`(:284), `gate-gpu-qwen3moe-e2e`(:574)다.

**절단 뒤 cargo 그래프(del2 뒤; 오늘 값은 `cargo metadata --format-version 1 --no-deps`로 확인).**
- `cpu` → model, gguf, qdot, threads, levers (dev: refset)
- `model` → gguf, levers, qdot, threads (dev: refset)
- `gpu` → gguf, model, threads
- `gpu-deepseek41` → engram, gguf, gpu, levers, model
- `gpu-gates` → engram, gguf, levers, model, refset, 그리고 opt로 gpu, gpu-deepseek41, gpu-vision, sampler, serve, threads, tokenizer, vision
- `gpu-vision` → gguf, gpu, vision
- `qdot` → gguf, threads
- `threads` → levers
- `engram`, `refset`, `serve`, `tokenizer`, `vision` → gguf
- 불변식: 어떤 패키지도 `bloomery-cpu`를 의존하지 않는다(모든 종류 포함).

**CPU4 권고: 옮긴다.**
- 옛 API를 격리 안에 두는 안은 성립하지 않는다.
  - `PlanHost`가 `Derived`를 품는 한(`mod.rs:85-131`), GPU가 `crates/cpu`에 의존하거나 `Derived`·`MoeBlockPlan`·`Meta`·자유 함수 둘·`in_file` 둘이 공유 model에 남는다.
  - 증명은 move로 싸지만 격리 목표를 못 맞춘다.
- 이주하는 안은 비트 동일이 구성으로 선다.
  - `qdot::swiglu_clamp`는 `limit > 1e-6`일 때만 자르고, 그 밖에는 `swiglu`를 부른다(`crates/qdot/src/lib.rs:4234-4250`). `ops.rs:1304-1309`가 `None`과 `Some(0.0)`을 이 둘로 보낸다.
  - 바이트가 같다. `facts.md:26-28`에 따르면 routed gate/up은 전부 Q3_K@2048, down은 Q5_0@1408이다. `UnionStack::of_call`이 이 모양을 받는다는 것은 v2lite union 케이스(`union.rs:1026-1098`)가 이미 보였다. `HostLayer::build`도 같은 검사를 쓴다(moe.rs:1490).
  - `MlaParams::read(gguf, l)`은 `Derived::new`가 부르는 것과 같은 호출이다(`derived.rs:197`; `assemble`은 블록 0만 읽는다, `mod.rs:184`).
  - `HostLayer`의 정상 경로는 할당하지 않는다. `ShardTensor::file/expert`(`ops.rs:931-975`)는 오류 클로저에서만 할당한다.
- 증명 비용(API 변경): 주인 게이트가 비트 동일해야 한다. 먼저 볼 줄은 verdict-diff가 비교하는 `gate_hybrid.rs:733` `steady allocs_per_step`이다. §6에 전부 적었다.

**세 층.**
- (1) 구조: 위의 절단.
- (2) 침식 방지(규칙마다 스크래치 사본에서 FAIL-first):
  - ⑤ `crates/*/Cargo.toml`과 루트 `[workspace.dependencies]` 가운데 `bloomery-cpu`를 이름으로 부르는 것은 `crates/cpu/Cargo.toml`뿐이어야 한다. FAIL-first: `crates/gpu-gates/Cargo.toml`에 한 줄을 넣으면 빨강이 된다.
  - ①·④: OWN을 `^crates/([^/]+/src/arch/|gpu-($ALT)/src/|cpu/src/)`로 바꾸고(`check-arch.sh:36`), ①은 `crates/cpu/src`를 deepseek2 디렉터리로 센다. FAIL-first: 변경 전에는 `crates/cpu/src/attn.rs`의 `use model::arch::deepseek2::mla::…`가 ④에서 빨강이다. 변경 뒤에는 `use model::arch::qwen3moe::…`를 심으면 ①에서 빨강이다. 허용 목록 줄 `crates/model/src/bin/bloomery-decode.rs use`는 없어진다.
  - ②는 그대로 둔다. CPU 파일에 `"blk.`·`"deepseek2.` 리터럴이 0개이기 때문이다(`grep -n '"blk\.\|"deepseek2\.'`). 앞으로 `crates/cpu`에 이름 문자열이 생기면 빨강이 되는 것이 원하는 동작이다.
  - ⑥ `tools/v2fence.txt`는 `kind<TAB>path<TAB>pattern<TAB>why` 형식이고, 다음을 검사한다.
    - (a) 격리 경로 밖의 V2-Lite 표지 적중(`deepseek2|Deepseek2|DEEPSEEK2|V2_LITE|v2lite|V2-Lite|Q5Quant|Q8_0Derived|CardFormat::Q5_|DevWeight::Q5_`)은 모두 seam 줄과 맞아야 한다.
    - (b) 적중이 없는 줄은 낡은 줄이라 빨강이다. 목록은 줄어들기만 한다.
    - (c) kernel 줄마다 `enqueue_*` 호출자는 V2-Lite 빈을 빼면 전부 `arch/deepseek2/` 아래에 있어야 한다.
    - (d) recipe 줄은 레시피가 있고 `BLOOMERY_MODEL=deepseek2`를 적어야 한다.
    - FAIL-first: `arch/qwen3moe/dispatch.rs`에 `enqueue_router_topk`를 심으면 빨강이다. seam 적중 한 줄을 지우면 stale로 빨강이다.
    - 오늘 적중은 ≈40파일이다(격리 밖 grep). CPU 시험이 옮겨 가면 ≈25파일로 준다 [유도].
  - `recipes.py`: `.rs`를 가리키는 문자열 리터럴은 lib 입력이 아니다(`:553` `_LITERAL`, `:716`). 이유는 Rust 원천이 모듈 트리나 `include!`로만 바이너리에 닿기 때문이다. FAIL-first: `why crates/cpu/src/attn.rs`가 87이면 빨강이다. recipes.py의 자기 시험에 한 케이스를 더한다.
- (3) 디버깅과 삭제:
  - 은퇴는 세 디렉터리와 `src/bin/deepseek2/`를 지우고 목록을 비우는 일이다. 빈 목록으로 ⑥이 초록이면 남은 것이 없다는 증명이 된다.
  - `crates/cpu/src/lib.rs` 머리말에 "아무도 의존하지 않는다"를 적는다.
  - `bloomery-decode`의 출력 경로와 러너는 그대로다.
  - V2-Lite 레시피는 프로필을 명시한다. 그래야 TL-6의 기본값 삭제가 이 목록 하나로 끝난다.

## §3 Q3 공유 모듈 안의 계기

| 항목 | 자리 | 결정 |
|---|---|---|
| `StepProbe`·`Observer`·`tick`·`OpTime`·`profile_reps` | `gpu/src/model/probe.rs` | 옮긴다 → `arch/deepseek2/probe.rs`(triage :265의 이름 바꾸기 포함). 다른 사용자는 gpumodel이 지우는 거부 impl 둘뿐이다 |
| probe 설정 분기 5, `tick` 38 | 이미 `arch/deepseek2/dispatch.rs`·`taps.rs` | 남긴다(격리 안, gate_p8과 `generate`가 쓴다) |
| `set_probe`·`seed_depth`·층 캡처 | `model.rs:621, :660, :1171-1264` | gpumodel이 `Instrumented`·taps.rs로 옮긴다. 격리 라운드는 트레이트를 inherent impl로 바꾸고 트레이트를 지운다(구현자가 하나뿐) |
| `Engine` impl + AnyEngine의 V2-Lite 팔 | `model.rs:1354-1378`, `engine.rs:25-47` | 지운다. 순서는 §6 |
| `HybridConfig` + `BLOOMERY_HYBRID_NL` | `hybrid.rs:95-171`, `model.rs:434-437` | 옮긴다 → `arch/deepseek2`(03 서명). `overlap`은 남긴다 |
| `q8_derived` | `model/lookup.rs:18` | 옮긴다 |
| `FaultSite::Q5Quant`, `step_order::DEEPSEEK2` | `fault.rs:71, :161-173` | 남긴다. 코드 3은 고정값이고 게이트가 핀한다(`gate_p1.rs:334`). 목록에 올리고, 은퇴 때 코드 3은 예약만 하고 재사용하지 않는다 |
| `DevWeight::Q5_*`/`Q8_0Derived`, `CardFormat::Q5_*` | `weights.rs`, `placement.rs`, V4.1 `draft/load.rs:255-256` | 남긴다(목록) |
| `seg_keys` + FLASH_SEG/FLASH_MMA 읽기 | `flash.rs:241-260` | FLASH_SEG는 은퇴시키고 → `MMA_SEG_KEYS` 상수. FLASH_MMA 거절은 `at_main`으로 옮긴다(§7) |
| `profile.rs` 훅 51줄 | `ops.rs` 47, `moe.rs` 호스트 4 | 03에 묻는다(§10). 격리는 손대지 않는다 |
| `gap_*` 커널 | `probe.rs`(gate_p8 전용) | 남긴다(목록 kernel 절) |

## §4 Q4 게이트

| 레시피(justfile 줄) | 격리 뒤 |
|---|---|
| gate-attn :480, ffn :483, moe :486(CPU 케이스), head :489, forward :494, kv :498, alloc :504, derived :510, mt :516, profile :522, prompts :687 | `-p bloomery-cpu`로 바꾼다. attn은 `--lib`을 더한다(attn.rs 단위 시험 둘이 옮겨 온다, :3132) |
| gate-ops :477 | model에 남는다. lib 단위 시험이 셋 줄어든다(attn 둘, moe `touched_mask…`). `tests/ops.rs`는 V2-Lite 오라클 이음이다 |
| gate-union :607 | 남는다. v2lite 케이스는 `HostLayer`로 다시 가리킨다(Q5_0 union 커버리지). moe 호스트 케이스(`tests/moe.rs:225-282`)도 여기로 온다 |
| gate-1-1 :714, gate-qdot :642 | 그대로 둔다(데이터 이음, 목록) |
| gate-gpu-p0b :189, p1 :80, p2 :83, p3 :86, p4 :89, p5 :92, p6 :95, p8 :253, p8b :1048, p9 :98, p10 :276, e2e :118, hybrid :123, moe :257, head :735, q4k-sel :102, block :272 | 레시피는 그대로다. `BLOOMERY_MODEL=deepseek2`를 명시하고, 빈 import만 바뀐다. 목록에 올린다 |
| gate-ptx-spill :1064 | 그대로(gate_e2e를 스캔한다. 은퇴 때 Qwen3 빈으로 바꾼다) |
| exact-ref :129, build-exact-forced :138, exact-taps :175 | 그대로(forced_exact 핀은 `gate_e2e.rs:159` `FORCED_PIN = 6`) |
| build-decode :349 | `-p bloomery-cpu`로 바꾼다. measure-decode :352, measure-sweep :357, ab-decode :369, measure-profile :379, perf-decode :385는 불변이다(`lease.sh:358` 경로가 같다) |
| generate :180, time-gpu-generate :185, cstate-ab :249, gpu-ab :1079, time-gpu-p8 :193, prof-gpu-p8 :198, bench-gpu-kernels :205, ncu-gpu-gemm :214, argmax-ref :695, argmax-ref-cuda :1038, greedy-ref-cuda :1043, dump-ref :418, dump-ref-cuda :422, build-argmax :692, build-kvclear :701, kvclear-probe :704 | 그대로(목록). :205와 :214는 gate_p8 안의 Qwen3 GEMM 벤치다 |

**효과(방법 포함).**
- CPU 파일(`attn.rs`, 나중에 `crates/cpu/src/attn.rs`): 오늘 87 [측정] → 11 [유도].
  - 방법: ⑤가 역의존을 0으로 만든다. 그러면 이 파일을 컴파일하는 타깃은 `bloomery-cpu` 자신의 lib·시험·bin뿐이다. 그것을 짓는 게이트는 위 표의 11개다.
  - 조건: recipes.py의 리터럴 규칙이 있어야 한다. 없으면 등록부 `ATTN` 리터럴 때문에 87이 그대로다.
  - 착륙 라운드가 새 트리에서 `why`로 확인한다.
- V2-Lite GPU 파일(`arch/deepseek2/dispatch.rs`): 58 [측정] → 58 [유도, 같은 lib 트리]. del2 뒤에는 56이다.
  - B안으로 V2-Lite 전용 디바이스 크레이트를 두면 ≈12다 [유도]. V2-Lite GPU 항목을 부르는 빈은 p0b, p1, p2, p5, p6, p8, p8b, p9, p10, e2e, hybrid, moe_fused다(grep).
  - B안은 권하지 않는다. 결정 8은 크레이트 경계를 V4.1 디바이스 코드에만 세운다. 비용은 `tensor`·`model::{lookup,kernels}`·`launch_u32`·`one_shard`를 pub으로 여는 것, generate_ds41 표가 delete 급이 되는 것, gate_e2e가 modules=2가 되는 것이다.

## §5 Q5 V2-Lite만 주는 커버리지

**정정 둘.**
- `gate-gpu-hybrid`는 "호스트 티어 = 전부 카드 비트 동일"이 아니다. 실제 계약은 다음과 같다(`gate_hybrid.rs:16-27, :83-97`):
  - 카드 슬롯은 비트 동일, 호스트 슬롯은 0이다.
  - 호스트 합은 `BAND_FACTOR` 3.0 × 오차 모델 안에 있다(:183).
  - argmax는 뒤집힘 밴드(`TAIL_FACTOR` 9.0, `SIGMA_OURS` 0.367, :186-189) 밖에서 같다.
  - 비트 단위인 것은 eager = replay(:13-15)와 겹침 off = on(:30-31)뿐이다.
- `gate-gpu-e2e`가 ik 토큰으로 판정하는 것은 인덱스 0뿐이다. 그 뒤의 발산은 출력만 한다(`gate_e2e.rs:25-30`). 핀은 exact 진실 대비 forced_exact다.

**1. gate-gpu-hybrid**
- 구조와 eager = replay: V4.1 step `--structure`(`gate_deepseek41_step.rs:8-22`)가 덮는다.
- 거부와 reset: 같은 게이트 :23-47이 덮는다.
- 층 합성: `gate_deepseek41_chain_ffn.rs:12-50`이 덮는다. 조각 = op 경로 비트 동일(호스트 몫 포함), 그리고 ik 덤프 대비 유도 밴드로 본다. 같은 모델의 전부-카드보다 강한 오라클이다.
- 스텁 팔(:751-1478, ≈728줄): 모델 없는 빈으로 뗀다. move 급, S, 3파동에 된다.
- 겹침 off 팔(:1554-1583): V4.1이 이 레버를 읽고(`body.rs:2253`) 켜짐을 요구한다(`gate_deepseek41_skew.rs:792`). V4.1 step에 off 팔을 더하거나(S) 레버를 은퇴시킨다(03).
- 대체가 없는 것: "실모델 hybrid argmax = 전부 카드". 606 MB 부분 집합은 `bloomery.fixture.subset`를 달고 있고, `0b95904` 메시지는 엔진이 그것을 거부해야 한다고 적는다. 그런데 그 키는 fixture verify만 읽는다(`fixture.rs:2138, :2231`). 엔진 경로를 먼저 결정해야 하고, M–L이라 3파동은 아니다.

**2. gate-gpu-e2e**
- 첫 토큰 대조: Qwen3 e2e (g)가 같은 `MARGIN_FLOOR` 기준으로 본다(`gate_qwen3moe_e2e.rs:32-37`). 프롬프트 0–7(justfile:578-580)이다. 33개로 넓히는 것은 선택이고 S–M이다.
- forced_exact: 아래 3번.
- seed_depth(4): V2-Lite `generate`의 계기라 은퇴와 함께 없어진다. 대체는 필요 없다.
- replay와 결정론: Qwen3 (r)와 V4.1 step이 덮는다.

**3. exact_ref / forced_exact**
- exact_ref(678줄)는 MLA 전용 f64 심판이다. 대체 후보는 셋이다.
  - (i) V4.1 `--ppl TAG`와 `PPL_RED`(:80-86): 코드는 있으나 레시피 기본 인자(justfile:843)에 들어 있지 않다.
  - (ii) Qwen3 `--ppl`: 출력만 한다(justfile:582-584). 빨강 줄을 더하면 S다.
  - (iii) Qwen3용 exact 심판: L. V2-Lite 크기에서도 빌드 한 번이 ≈50분이다(justfile:134 주석).
- AGENTS의 "정확도 핀 = 버그 잡이" 원칙에서는 (ii)가 가장 싸다. 3파동에 할 수 있다. 채택 여부와 박스 시간은 리드가 정한다. 박스 시간은 유도하지 않았다.

**그 밖에 은퇴를 막는 것.**
- 할당 래칫(`tests/alloc.rs`, LIMIT 660): V4.1 스텝 쪽 핀으로 옮긴다. S.
- 스레드 불변(`tests/mt.rs`): 호스트 티어에 둔다. S.
- 프로파일러 게이트: 03에 달렸다.
- GG1 공유 커널 계약(q8_1 거부, f16 전수, 사이트마스크, 컴파일 모양): opslib의 모델 없는 계열 게이트로 옮긴다. M.
- `tests/ops.rs`·`gate_head_gpu`·`gate_q4k_sel`의 V2-Lite 데이터: 각각 S.
- gate-1-1·gate-qdot의 Q5_0/Q5_1: `--synthetic`로 옮긴다. S.

## §6 Q6 라운드

| 라운드 | 파일 경계 | 크기 | 증명·예측 | 순서 |
|---|---|---|---|---|
| `v2host`(CPU4) | 새 `model/arch/deepseek2/{mla,wkb,host}.rs`; GPU `mod.rs`(PlanHost :85-131, derive :273-282, load_hybrid :414-439, derive_blocks :458-493); `moe.rs`의 삭제 :767-814·:1406-1424, `ops.rs`의 삭제 :1040·:1067(03); 빈 `gate_hybrid`(:130, :320-339, :1499-1503), `gate_p10`(:70, :132-137, :241, :279-294, :324-326), `gate_p8b` :275(`hp.experts.scale`, 같은 키), `exact_ref` :44-45; 시험 `moe.rs:225-282`, `union.rs:1026-1098` | S–M | API 변경. 예측은 다음 세 가지다. ① `generate_ds41`·`gate_e2e`·Qwen3 빈의 ptx-scan rows+md5가 동일하고 `bytes=`는 움직일 수 있다(ledger #22). ② verdict-diff가 동일하다: hybrid(:733 allocs 줄 포함), e2e, p8, p8b, p10, derived. ③ 다시 가리킨 두 시험은 출력 문구가 바뀐다. 이것은 이름 붙은 예측 차이다 | gpumodel 뒤. q3act-1이 `mod.rs`를 건드리면 직렬로 간다. hostone 전 |
| `v2fence-cpu`(CPU1 b) | 새 `crates/cpu` 전체. model에서 빠지는 것: CPU 파일, `moe.rs` CPU 부분(03), `ops.rs`의 pub 셋(03), `ffn.rs:65-93`, `lib.rs` `Slot`. justfile 12곳, `registry.rs` `ATTN`/`DECODE`, `levers-direct.txt` :14·:16·:24, check-arch ⑤·①·④, recipes.py. 첫 커밋은 `ATTN_HALVES`·`POPULATE`·`WEIGHTS` 삭제 | L | 첫 커밋은 delete(halves-2 케이스 `tests/attn.rs:556` 제거에 날짜 사유), 나머지는 move다. ptx-scan은 위와 같다. 착륙은 호스트 전용 규칙을 따른다: CPU 11개 + ops + union + ds41-host + hybrid + ds41-step·-prefill + 정적 검사(≈575 s). lint ≤ 133. `why crates/cpu/src/attn.rs` = 11 | v2host 뒤. 03의 `moe.rs`·`ops.rs` 창이 비어 있을 때 |
| `v2fence-gpu` | `model/probe.rs` → arch, `model.rs:31`, `hybrid.rs` HybridConfig(03), `lookup.rs` `q8_derived`, `Instrumented` → inherent impl, FLASH_SEG 은퇴, `levers-direct.txt:17` | S | move. ptx-scan rows+md5가 동일하다. V2-Lite GPU 게이트의 verdict가 동일하다. 노드 핀 줄이 불변이다(p8 `NODES_BLOCK0` 등, hybrid 구조, V4.1 step, Qwen3 `NODES_CHAIN`) | gpumodel, q3act-1(act-planes :421-425), levers2(`at_main(acts_on)`) 뒤 |
| `v2fence-gates` | `engine.rs` 팔 삭제(gate_e2e·generate는 `Deepseek2Model`을 직접 쓴다), `gpu::Engine`, 스텁 빈 분리, `src/bin/deepseek2/`, justfile 프로필 명시, `tools/v2fence.txt`와 ⑥ | M | move와 분할. gate_e2e의 ptx-scan이 동일하다. 새 두 게이트의 verdict 줄 합집합이 옛 gate_hybrid의 줄과 같다 | session의 `gpu::Engine` 결정 뒤, modelspec step 4 전 |

- gpumodel이 V2-Lite 경로에서 건드리는 것:
  - `open_hybrid`와 plain open 생성자.
  - `Instrumented`.
  - 층 캡처를 taps.rs로 옮기는 것.
  - "레이어 범위 없음". 이것은 `gate_p8b`의 `load_blocks(LAYER..LAYER+1)`과 `gate_p10`의 staging에 닿는다. 격리의 base는 gpumodel의 V2-Lite 생성자를 이어받는다.
- opslib가 V2-Lite 커널을 계열 인스턴스로 삼는지는 K2의 kernel 목록만 바꾼다. 라운드 경계는 바꾸지 않는다.

## §7 Q7 레버

| 레버(`registry.rs`) | 오늘 | 격리 뒤 |
|---|---|---|
| `PIN_MAIN` :145-164 | `bloomery-decode.rs:122`, `!= "0"` | converted. `at_main`의 Flag를 쓴다(triage :148의 조용한 읽기가 닫힌다). bench_v41_host의 `left`(R03)는 그대로 |
| `PROFILE` :261-284 | `profile.rs:83-89`(조용한 파싱, triage :139) | converted. `bloomery-decode`의 main에서 Count 0–2로 읽어 `profile::set_level`로 넘긴다. 자식 재실행 시험(`tests/profile.rs:78-84`)도 같은 파싱을 쓴다. 03의 훅 답(§10)에 달렸다 |
| `FLASH_SIMD` :285, `KV_PREFETCH` :299, `KV_PREFETCH_ROWS` :312, `FLASH_SEGMENTS` :329, `ATTN_BUNDLE` :348 | `attn.rs:1086-1217`, 제자리 | kept. 격리 안의 쌍둥이 팔이다(rebuild §5). 행의 파일만 `crates/cpu/src/attn.rs`로 바꾼다. `== "0"` 읽기(:1088, :1108)는 행의 Kind로 죈다(XS) |
| `ATTN_HALVES` :362 | `attn.rs:2481` | Retired(rebuild §4). halves=2 팔과 케이스를 지운다 |
| `FLASH_SEG` :376 | `flash.rs:250` | Retired. 트리에서 이 레버를 세우는 레시피·러너가 0이다(`grep -rnw BLOOMERY_FLASH_SEG`). A6b에서 값이 정해졌다(`plan-ledger.md:709`) |
| `POPULATE` :544, `WEIGHTS` :558 | `bloomery-decode.rs:129-136` | Retired(§4). `gguf::Weights::Resident`는 r8file이 쓰므로 남는다 |
| `FLASH_MMA` :595(Retired, `left` flash.rs) | `seg_keys`에서 panic | `left: []`. V2-Lite 빈이 `at_main`을 부르면 은퇴한 이름을 거절한다 |
| 인접 행: `EXPERT_LOG` :247 | [03], CPU MoE | 파일이 `crates/cpu/src/moe.rs`로 바뀐다. 라벨은 03에 묻는다 |
| 인접 행: `HYBRID_NL` :436, `OVERLAP` :454 | [03] | §10 |
| 인접 행: `REF_MODEL` | `levers-direct.txt:24` | 빈과 함께 경로만 옮긴다 |

## §8 Q8 03 파일 hunk (K2)

- `moe.rs`
  - :24-746을 `crates/cpu`로 옮긴다. `expert_view`와 `expect_stack`은 남긴다. 머리말 :1-22는 호스트 티어 설명으로 새로 쓴다.
  - 단위 시험 :1609-1636을 같이 옮긴다.
  - v2host 뒤 호출자가 0인 :767-814와 :1406-1424를 지운다.
  - `Meta`가 나가면 `tests/ds41_host.rs:328-336`의 V4.1 `Meta` 단언도 없어진다. 날짜 사유를 붙인다.
- `ops.rs`
  - `SharedOut` :461, `ErrGate` :2875, `matvec_q_local` :4148을 `pub`으로 연다. 경계에 SAFETY 계약을 적는다.
  - `in_file` :1040, :1067을 지운다.
- `hybrid.rs`: :95-171의 `HybridConfig`·`from_levers`·NL 읽기를 옮긴다. 선택 사항이다.
- 시험: `tests/moe.rs:225-282`, `tests/union.rs:1026-1098`, `tests/ds41_host.rs:328-336`. `tests/ops.rs:28-37`은 목록에만 올린다.
- 없음: `fault.rs`(목록만), `fused.rs`(K2), `quant.rs`(`cdd90f3`에 없다), `qdot`(doc 두 줄 :16, :3955은 선택), Qwen3 디렉터리, `flash_gqa*`(가져가는 것이 전부 공유로 남는다, :54-57), `gemm*`(:1118 `half_bits_to_f32`는 남는다), `r8file.rs`, `deepseek41/host.rs`(host.rs의 틀일 뿐).
- K1이면 `fused.rs:428-535`과 `:741-902`를 옮기는 hunk가 더해진다.

## §9 위험과 먼저 볼 것

- 리터럴 규칙이 없으면 CPU 격리의 효과는 87 그대로다. 처음 볼 것은 `why crates/cpu/src/attn.rs`다.
- AGENTS의 "ptx-scan 동일"은 rows+md5로 읽는다. 헤더의 `bytes=`는 호스트 전용 변경에도 움직인다(ledger #22, `nvlabs-ledger.md:30`).
- v2host의 비트 동일은 세 전제에 선다: limit 0 등식, 같은 바이트, 할당 0. 하나라도 틀리면 `gate_hybrid.rs:733` 줄이나 층별 줄이 DIFFERS로 이름을 댄다. hybrid를 가장 먼저 base 대비로 돌린다.
- 다시 가리킨 시험, 옮긴 단위 시험, gate-ops·gate-attn의 통과 개수 줄은 예측된 차이다. 스펙에 줄 단위로 적는다.
- 03과의 겹침:
  - `moe.rs`·`ops.rs`를 편집하는 hostcfg와 hostone의 창.
  - q3act-1의 `arch/deepseek2/*`·`q5.rs`·`flash.rs`. 격리와 동시에 돌리지 않는다.
- 선행 조건:
  - gpumodel의 레이어 범위 생성자.
  - levers2의 `at_main(acts_on)` 서명.
  - modelspec step 4의 AnyEngine.
- K1을 고르면 `Gpu::new`의 모듈 적재 수가 바뀐다. 그러면 `gate-ds41-load`의 적재·드롭 바이트를 먼저 본다.
- `bench-gpu-kernels`와 `ncu-gpu-gemm`이 gate_p8에 기대고 있다. 은퇴 전에 03이 옮겨야 한다.
- 규칙 위반 한 건: `/tmp/x_attn`에 정렬 목록을 한 번 썼고 바로 지웠다(스크래치 경로 밖이었다).
- 이 메모는 박스 명령을 하나도 돌리지 않았고, 카드도 필요하지 않았다.

## §10 열린 질문

**리드에게.**
1. K2(결정 8)로 갈지, K1(GC7 문안)로 갈지.
2. opslib가 Softmax(64, 6) 라우터(`modelspec-design.md:672`), latent 플래시, Q5_0/Q5_1을 계열 인스턴스로 보는지, 아니면 V2-Lite와 함께 은퇴시키는지.
3. AnyEngine의 Deepseek2 팔(`engine.rs:37-47`)과 modelspec step 4 가운데 누가 먼저인지. modelspec §6에는 deepseek2 리더가 없다.
4. `gpu::Engine`을 session 뒤에 지울지. 제네릭 사용자는 `gate_e2e.rs:250` 하나다.
5. recipes.py 규칙을 어느 라운드가 가질지.
6. FLASH_SEG 은퇴와 §4 셋의 삭제를 v2fence-cpu의 첫 커밋으로 해도 되는지.
7. v2host와 v2fence-cpu를 한 묶음으로 검증할지.
8. exact_ref의 자리를 KLD 빨강 줄로 받을지.

**hosttier 메모에게(답을 추측하지 않는다).**
1. `hybridgate`는 gate_hybrid의 후계인가? 그렇다면 스텁 팔(:751-1478), off 팔, 층별 밴드 가운데 무엇을 가져가는가?
2. `HYBRID_OVERLAP`을 은퇴시키는가?
3. `HybridConfig`와 NL을 격리가 옮겨도 되는가, 아니면 hostcfg의 `HostLevers`가 흡수하는가?
4. v2host 뒤 죽는 자유 함수 둘과 `in_file` 둘은 v2host에서 지우는가, hostone에서 지우는가?
5. hostcfg, benchprune, hostone이 `moe.rs`와 `ops.rs`를 편집하는 창은 언제인가?
6. hostone의 같은 임대 A/B는 `ab-decode`(V2-Lite CPU 엔진)로 하는가, `bench_v41_host`로 하는가?
7. `bloomery-decode`가 격리된 뒤 `profile.rs` 훅 51줄을 호스트 티어의 계기로 둘 것인가? 둔다면 레벨은 누가 세우는가?
8. `tests/ds41_host.rs`의 `Meta` 단언을 지워도 되는가?
9. v2lite union 케이스와 moe 호스트 케이스를 `HostLayer`로 다시 가리키는 것을 받는가?
10. gate_p8의 GEMM 벤치는 언제, 누가 옮기는가?
11. q3act의 `quant.rs`가 V2-Lite 쓰기 자리(`q5.rs:429`, `flash.rs:941`, `:1294`)에 새 이음을 만드는가?

## §11 스펙 밖 개선 지점(보고만)

- `gpu/src/lib.rs:2176-2196`: `Gpu::with_device`가 크레이트 번들 전체를 9번 적재한다. `#[cuda_module]`의 `load`는 `load_named(ctx, env!("CARGO_PKG_NAME"))` → `load_embedded_module`이다(cuda-oxide 체크아웃 `cuda-macros/src/cuda_module/mod.rs:175-180, :266-281`). `from_parent` 뷰로 한 번만 적재할 수 있다. 메모리는 재지 않았으니 `Gnew` 전후 `cuMemGetInfo`로 확인할 것. M.
- `gpu/src/arch/mod.rs:5-8`의 결정 8과 Qwen3 커널의 실제 자리(`arch/qwen3moe/{experts:111, proj:28, router:480, head_argmax:63}`)가 갈라져 있다. 문서를 고치거나 결정을 다시 해야 한다. XS.
- `model/probe.rs:1-3`의 머리말이 거짓이다. XS.
- `model/src/lib.rs:1-21`과 `ops.rs:1-4`의 머리말이 낡았다. XS.
- `moe.rs:156-160`과 `derived.rs:174-176`의 주석 "ModelError has no metadata variant"는 틀렸다(`lib.rs:78-79`). `MlaParams::read`의 metadata 오류도 `MissingTensor`로 나간다. S.
- `gpu-gates/Cargo.toml:30`의 gpu-spike 주석은 del2 뒤에 낡는다. XS.
- AnyEngine이 Qwen3를 적재한 뒤에 거절한다(`engine.rs:37-47`). XS.
- V2-Lite 세트에는 refset family가 없다(`crates/refset/src/arch`에는 deepseek41, deepseek41v만 있다). M.
- 기본 경로가 두 곳에 복사돼 있다(`tests/common/model_path.rs:5-8`, `union.rs:1008-1011`). XS.
- `fault.rs:161-213`의 모델별 스텝 순서 표가 공유 파일에 있다. 각 몸체의 const로 옮길 수 있다. S.

