== /Users/hckim/.claude/projects/-Users-hckim-repo-bloomery/e5714b41-3850-42be-a722-79e1f225c3b5/subagents/agent-aspots-2239ec560b5d0d21.jsonl ==
스펙 밖 개선 항목 177개를 모두 main 트리와 대조했습니다. 트리아지 ①–⑫(`plan.md:467–587`)는 리드가 이미 구조화한 트래커라 다시 세지 않았습니다. 필요하면 두 번째 패스로 하겠습니다.

**결과 요약**

| 권고 | 개수 |
|---|---|
| DO-NOW 지금 병렬 가능 (A 4, B 6, C 4, D 3, E 4) | 21 |
| DO-NOW 리드 한국어 문서 (L) | 10 |
| DO-NOW 보류 (H-load 3, H-engpre 3, H-ngdraft 4, H-just 4) | 14 |
| TRACKER | 73 |
| DROP | 59 |
| 합계 | 177 |

**가장 쓸모 있는 발견 넷**
- **audit-gpu Q5의 SAFETY "as above" 14곳은 하나도 고쳐지지 않았습니다.** grep 패턴을 넓히니 `indexer.rs` 5, `compress.rs` 4, `hc.rs` 2, `chain/ffn.rs` 2, `engram_gate.rs` 1이 전부 남아 있습니다.
- **`kernels.rs:81`의 allow reason은 고쳐졌지만 형제 둘은 그대로입니다.** q3_K heads 커널의 `:153`과 `:226`에 reason이 없습니다.
- **`moe.rs:285`의 unwrap은 DO-NOW에서 뺐습니다.** 스텝마다 도는 `route_inner` 안이라, cpu1의 할당 항목과 한 라운드로 묶고 같은 임대 A/B를 해야 합니다.
- **많은 보고 항목이 이미 닫혀 있습니다.** 닫은 커밋은 `4f230f5`, `f56e770`, `c1e733e`, `e0919a3`, `76c9ad8`, `ef2f43f`, `e32664a`이고, roofline.md 정정분도 포함됩니다.

**묶음과 소유 파일**

두 묶음에 같은 파일이 들어간 경우는 없습니다.
- **A**: `tools/ref/{dequant_ref.cpp,host-rate.sh,engram-corpus.sh,nsys-gpu.sh}`
- **B**: `crates/gpu/src/{model/kernels.rs,q8f32.rs,hybrid.rs,lib.rs,arch/deepseek2/dispatch.rs}`
- **C**: `crates/gpu-deepseek41/src/{experts.rs,indexer.rs,compress.rs,hc.rs,engram_gate.rs,chain/ffn.rs,chain/attn.rs}`
- **D**: `crates/gpu-gates/src/bin/{gate_p6.rs,gate_deepseek41_hc.rs,gate_deepseek41_step.rs}`
- **E**: `crates/model/src/arch/{dspark.rs,mod.rs,deepseek41/hparams.rs,deepseek2/attn.rs}`, `crates/gguf/src/quant.rs`, `crates/qdot/src/lib.rs`
- **L**: `plan.md`, `roofline.md`, `research/{spec-decoding-report,hybrid,host-surplus}.md`, `ncu-gpu.sh`의 한국어 주석
- **H**: wave 15가 쥔 파일이라 그 라운드가 착륙한 뒤에 합니다.
  - H-load: `body.rs`, `gate_load_v41.rs`, `bench_v41_host.rs`
  - H-engpre: `glue.rs`, `crates/engram`
  - H-ngdraft: `generate_ds41.rs`, `gpu/src/model.rs`, `depth-ds41.sh`
  - H-just: `justfile`, `models/deepseek41.sh`

**리드가 미리 알아야 할 것**
- 행 33과 166은 게이트가 찍는 판정 줄을 바꿉니다. 그 diff는 회귀가 아닙니다.
- 행 22와 167은 `#[kernel]` 속성이나 `#[cuda_module]` 안의 주석을 고칩니다. 증명은 `cargo oxide build`와 ptx-scan 표 동일입니다.
- C와 D는 비행 중인 engpre, ngdraft와 같은 크레이트를 씁니다. 파일은 겹치지 않지만 병합 순서는 확인해 주세요.

**분류 기준**
- 분류는 correctness / lint-quality / perf / docs / measurement-tooling 다섯 가지만 썼습니다.
- 업스트림 항목은 correctness, 게이트 항목은 measurement-tooling로 넣고 내용 칸에 표시했습니다.
- "미확인"으로 남은 것은 행 53, 54, 132, 135, 137, 138, 141, 158입니다.
- 행 173의 리터럴 6은 main에서 두 번 grep해도 찾지 못해 TRACKER에 두었습니다.

| # | 출처 | 보고 위치 | main에서 유효? | 내용 | 크기 | 분류 | 권고 |
|---|---|---|---|---|---|---|---|
| 1 | cuda-13 §8 | `tools/ref/ncu-gpu.sh:25` | 없음 — `:30`에서 2025.3.1로 이름 고정 | ncu 래퍼가 최신 판을 고름 | XS | measurement-tooling | DROP 이미 고정 |
| 2 | cuda-13 §8 | `ncu-gpu.sh:16-18` | 예 `:16` | 주석이 node 출력을 "2025.3 기본값" 덕으로 적음. 러너가 `--graph-profiling node`를 직접 넘긴다(`:115`) | XS | docs | DO-NOW(L, 한국어 주석) |
| 3 | cuda-13 §8 | `ncu-gpu.sh` KEEP | 해당 없음 | KEEP에 "Shared Memory Spilling Requests" 추가 | XS | measurement-tooling | DROP 2025.3.1 고정으로 무의미 |
| 4 | cuda-13 §8 | `tools/ref/nsys-gpu.sh:28` | 예 `:30` `/usr/local/cuda/bin/nsys` | nsys도 래퍼 경로. 판 고정 또는 판 검사 | XS | measurement-tooling | TRACKER 어느 판에 둘지 결정(㊱은 호스트 타임라인이 필요) |
| 5 | cuda-13 §8 | `nsys-gpu.sh:83` | 예 `:74` `ORDER BY start` | 행을 `graphNodeId`로 노드에 직접 묶기 | S | measurement-tooling | TRACKER |
| 6 | cuda-13 §8 | `tools/ptx-scan.sh:42` | 없음 — `:130-131`이 `ptxas-version=`을 배너에 찍음 | ptxas 판을 표 머리에 | XS | measurement-tooling | DROP 이미 됨 |
| 7 | cuda-13 §8 | 박스 `bloomery-env.sh` | 맥에서 확인 불가 | `CUDA_OXIDE_LIBDEVICE` 명시 | XS | measurement-tooling | TRACKER 기계 설정(리드) |
| 8 | cuda-13 §8 | `docs/RESULTS-mul30-gpu-scout.md:15,:73` | `:15`는 정정됨, `:73`은 업스트림 README 인용 | 13.2 대 13.3 요건 | XS | docs | DROP |
| 9 | draft-accept-e5 §5, markov-e6 §5 | `spec-decoding-report.md:14,56` 대 `dspark-cost.md:65` | 예 (0.36 셋, 0.34 하나) | 손익분기 0.34/0.36 불일치 | XS | docs | DO-NOW(L) |
| 10 | draft-accept-e5 §5 | `plan.md:622` E5 행 | 없음 — 「네 코퍼스」는 취소선 안(`:623`) | 코퍼스 수 | XS | docs | DROP |
| 11 | draft-accept-e5 §5 | `tools/ref/engram-corpus.sh:64` | 예 — prose-all이 로캘 `sort`(`:24-25`가 의도라 적음) | 앞머리만 읽는 분석이 prose를 다시 잰다. 머리 주석 한 줄 | XS | docs | DO-NOW(A) 주석만. 정렬 변경은 .ids 재생성·E5/E6 재핀이라 TRACKER |
| 12 | dsm §5, q8dead §5, plan `:21` | `q8f32.rs` `q8_0_gemv` m>1 청크 몸통 | 없음 — `76c9ad8`(q8dead) | 죽은 m>1 몸통 삭제 + p6 재핀 | S | perf | DROP 됨 |
| 13 | dsm §5 | `q8f32.rs` 8갈래 저장 두 벌 | 없음 — `f32_gemv`가 `store_sums`(`:784`), `76c9ad8` | store_sums 통합 | XS | lint-quality | DROP 됨 |
| 14 | dsm §5, q8dead §5 (중복) | `q8f32.rs:65` `f32_lane_partials`/`f32_gemv` m>1 | 예 `:103-128` | m>1이 여전히 청크 순서(엔진 호출처는 m=1) | S | correctness | TRACKER `#[cuda_module]` 안, ptx-scan·A/B |
| 15 | dsm §5, q8dead §5 (중복) | `model/kernels.rs:81` allow reason | `:81` 고침(`76c9ad8`). 형제 `:153`·`:226`은 reason 없음 | R16 allow reason | XS | lint-quality | DO-NOW(B) `:153`·`:226` |
| 16 | dsm §5, q8dead §5 (중복) | `model/kernels.rs:404` `enqueue_q8_0_gemv_heads` | 예 (검사 여섯이 heads_mcol 런처와 중복) | 공통 검사 함수 | S | lint-quality | TRACKER 호스트 런처 R14 묶음 |
| 17 | dsm §5 | K-quant gemv m>1 `y[r·m+c]` | 예 (q3k·q4k·q6k) | k토큰 체인의 전치 문제 | M | correctness | TRACKER C3/C4·ktok |
| 18 | dsm §5 | `mcol` 엔트리 M=8 레지스터 | 예 | M≤4용 엔트리 분리 | S | perf | TRACKER 타이밍 먼저 |
| 19 | q8dead §5 | `q8f32.rs:1055` `q8_0_launch_dims` 셋째 반환 `_` | 예 `:1055`·`:1263` | m=1 경로 전용 dims 또는 반환형 분리 | XS | lint-quality | DO-NOW(B) 호스트 코드 |
| 20 | q8dead §5 | `gate_p6.rs:627` `GEMV_COLS` 문서 | 예 | f32_gemv 전용으로 문서·이름 좁히기 | XS | docs | DO-NOW(D) 문서만 |
| 21 | q8dead §5 | `gate_p6.rs:643,648` 두 floor | 예 (같은 식, PIN 두 줄) | 상수 하나로 | XS | lint-quality | TRACKER PIN 줄 재저작 |
| 22 | q8dead §5 | `model/kernels.rs:81` allow 위치 | 예 (`#[kernel]` 앞, q8f32.rs는 `#[launch_contract]` 뒤 `:852`) | 속성 순서 통일 | XS | lint-quality | DO-NOW(B) 15와 같이. 증명은 `cargo oxide build` + ptx-scan 동일 |
| 23 | dsread §5, plan `:21` | `placement.rs:180` `CardFormat::of` MXFP4 | 예 `:192` → None | MXFP4 카드 네이티브 형식 | S | perf | TRACKER dsmx 몫 |
| 24 | dsread §5 | `placement.rs:180` BF16 → f32 넓힘 | 예 `:191` `Bf16AsF32` | Markov 132→265 MB | S | perf | TRACKER dsgraph 결정 |
| 25 | dsread §5 | `arch/dspark.rs` 끝 helper | 예 `:683-707` 대 `deepseek41/hparams.rs:8-38` | 메타 helper 사본 → `arch/mod.rs` 공용 | S | lint-quality | DO-NOW(E) 이동 클래스, ~40줄 |
| 26 | dsread §5 | `gguf/src/lib.rs:698` fallback 표 | 부분 — `GgmlType` 이름을 먼저 보므로 승격 id는 이긴다 | 손 표가 id를 가림 | XS | lint-quality | DROP 구조가 이미 막음 |
| 27 | dsread §5 | `tools/ref/dequant_ref.cpp:40` `write_file` | 예 `:38` fopen/fwrite | 비원자적 쓰기, 형제는 `write_atomic` | XS | measurement-tooling | DO-NOW(A) |
| 28 | dsread §5 | `gguf-inventory.rs:382,588` MXFP4 보고 | 예 (시험 없음, `gguf/tests`에 MXFP4 0건) | 경로 핀 시험 | XS | measurement-tooling | TRACKER 새 게이트 판단 |
| 29 | dsread §5 | `gguf/src/quant.rs:671,726` chunks_exact | 예 | lint 넷 → as_chunks | XS | lint-quality | TRACKER 핫 패스라 mechlint가 남긴 줄(⑧) — A/B 동반 |
| 30 | engshadow §5, skew §5 (중복) | `SHADOW_CARD`/`SHADOW_HOST`/`ENGRAM_KV` | 옮김: bin 둘(`gate_deepseek41_step.rs:822-834`, `_skew.rs:86-95`) | 조각이 그늘 표를 내보내 주인 하나로 | S | lint-quality | TRACKER |
| 31 | engshadow §5, plan `:21` | nsys 0층 bridge 2.1–2.7 ms | 측정 항목. E13/E15가 engram 0.48 항은 노출된 적 없다고 답함 | 원인 축소 | S | measurement-tooling | TRACKER ㊱과 합침 |
| 32 | engshadow §5, plan `:21` | `glue.rs:463` `enqueue_engram_kv` | 예 (호출처 `gate_deepseek41_chain_glue.rs:335` 하나) | 게이트를 `_at`으로 옮기고 삭제 | XS | lint-quality | DO-NOW(H-engpre) |
| 33 | engshadow §5 | `gate_deepseek41_hc.rs` `engram_next` | 예 `:765,835,864,1045,1173` | 라벨·출력 줄에만 쓰임 | XS | lint-quality | DO-NOW(D) hc 게이트의 `form=` 라벨·`nodes` 줄이 바뀐다 |
| 34 | engshadow §5, audit-gpu Q3 (중복) | `gate_deepseek41_step.rs:1062` `shadow_sets` / overlap=0 | 예 (레시피·게이트 0건, 거부만) | overlap=0 모드를 도는 레시피 | XS–S | measurement-tooling | TRACKER 게이트 추가 판단 |
| 35 | engshadow §5 | `chain/ffn.rs:885` `enqueue_shadowed` 8인자 | 예 (allow에 reason 있음) | `FfnIo`로 | S | lint-quality | TRACKER |
| 36 | fusion §7 | `model.rs:2119` `rope(q)` 48열 전부 | 옮김 `arch/deepseek2/dispatch.rs:256` | V2-Lite rope 3배 작업 | S | perf | TRACKER V2-Lite, 커널·A/B |
| 37 | fusion §7 | `model.rs:1216` `refresh_params` | 옮김 `model.rs:1026` → 이미지 H2D 한 번(`deepseek2/mod.rs:308`) | 호스트 YaRN 수학 | M | perf | DROP V2-Lite, 복사 넷이 하나로 |
| 38 | fusion §7 | `lib.rs:1188` prepare 매 호출 | 예 `lib.rs:1588` | eager 경로 host 검증 캐시 | S | perf | TRACKER |
| 39 | fusion §7 | `model.rs:1153` `GpuModel::step` 스텁 | 없음 — `model.rs:738` 구현 | 스텝 미구현 | L | correctness | DROP |
| 40 | fusion §7 | `model.rs:1462` `replay_layer` 호스트 경계 | 옮김 `deepseek2/taps.rs:310` 게이트 하니스 전용 | | S | lint-quality | DROP 조립 체인에 안 남음 |
| 41 | fusion §7 | `model.rs:2204` kqvc lo/hi 두 런치 | 없음 — `FlashLatentQ8Args{y,lo,hi}` 한 런치(`dispatch.rs:430`) | | XS | perf | DROP |
| 42 | hotset §5, plan `:21` | `experts.rs:298` "id order" 문서 | 옮김 `gpu-deepseek41/src/experts.rs:243` | "slot order of the card's list"로 | XS | docs | DO-NOW(C) |
| 43 | hotset §5, plan `:21` | `gpu/src/weights.rs` 바이트+워드 2배 | 예 `:481` `words_of` | KQuant 워드로 바로 모으기 | S | perf | TRACKER weights.rs는 load 소유 |
| 44 | hotset §5, plan `:21` | `gate_load_v41.rs` check 3 출력 길이 | 예 `:581` | 구간 수만 찍기 | XS | measurement-tooling | DO-NOW(H-load) |
| 45 | hotset §5 | `hybrid.rs:3,98,156`, `model.rs:127,144,420,455` "[0, n_l)" | 예 `hybrid.rs:3,105,163`, `model.rs:127,516` | V2-Lite 전용임을 문서에 | XS | docs | DO-NOW(B) hybrid.rs. model.rs 몫은 H-ngdraft |
| 46 | hotset §5 | `generate_ds41.rs:334` plan 줄 | 예 `:356` (목록 여부 없음) | plan 줄에 hot list 사용 표시 | XS | measurement-tooling | DO-NOW(H-ngdraft) |
| 47 | hotset §5 | 목록 실행 첫 토큰 분기 교차 확인 | 없음 — E16 답함(`64946ab`, KLD 0.00601) | | S | measurement-tooling | DROP |
| 48 | hybrid-engines §8 | `roofline.md:96-98` MLA | 없음 — `:97` 정정 2026-09-21 | | XS | docs | DROP |
| 49 | hybrid-engines §8 | `roofline.md:161` engram 상주 | 없음 — `:165` 정정 | | XS | docs | DROP |
| 50 | hybrid-engines §8 | `docs/orchestration.md:92` B2 게이트 축 | 없음 — plan.md로 병합, `plan.md:217`에 GPU 유휴 비율 | | XS | docs | DROP |
| 51 | hybrid-engines §8, hybrid-lit §9 (중복) | `plan.md:63` B2 게이트 | 없음 — `plan.md:217`, `:726`이 Qwen3.6 대 0.684 명시 | | XS | docs | DROP |
| 52 | hybrid-engines §8 | `roofline.md:50-56` 서빙 배치 표 | 예 `:48-55` (4⅓/2⅔/33 블록) | 캐시 설계 이전 배치 | XS | docs | DO-NOW(L) |
| 53 | hybrid-engines §8 | 업스트림 `ggml-backend.cpp:2398` printf | 레포 밖, 미확인 | 업스트림: 핫 루프 디버그 printf | XS | correctness | TRACKER MUL-7, FAIL-first 먼저 |
| 54 | hybrid-engines §8 | 업스트림 `ggml-backend.cpp:2019` `k_set_sync` | 레포 밖, 미확인 | 업스트림: 죽은 저장인지 질문 | XS | correctness | TRACKER MUL-7 |
| 55 | hybrid-lit §9 | `roofline.md:224` 1.53배 | 없음 — `:228` 2.91배로 정정 | | XS | docs | DROP |
| 56 | hybrid-lit §9 | `roofline.md:220` 0.09개 | 없음 — `:224` "유도" 표시 | | XS | docs | DROP |
| 57 | hybrid-lit §9 | `roofline.md:216-217` 대 `:83` | 없음 — `:91` 어긋남 명시, E2가 135–137 재현(`ef2f43f`) | | XS | docs | DROP |
| 58 | hybrid-lit §9 | `roofline.md:59-64` 500/700 행 | 부분 — `:78` 주의는 있고 선 그음 없음 | 낙관 행에 선 긋기 | XS | docs | DO-NOW(L) |
| 59 | hybrid-lit §9, jev §7 (중복) | `roofline.md:93` 147.7 대 215.6 / `hybrid.md:49` 주인 없음 | 부분 — E2 답함, `hybrid.md:49`는 아직 "WKS-35 전까지" | 막힘 해제 문장 갱신 | XS | docs | DO-NOW(L) hybrid.md |
| 60 | hybrid-lit §9 | `v41-ports.md:12-13` fault당 ~380 µs | 예 (hybrid-lit-report에만 있음) | B3 텔레메트리 계약에 상수 | XS | docs | DROP E1b가 콜드 행을 직접 잼 |
| 61 | hybrid-lit §9 | `fusion.md:41` | 없음 — 2026-09-21 덧붙임 | | XS | docs | DROP |
| 62 | hybrid-lit §9 | `plan.md:80` C3 "1.67배" | 예 `plan.md:234`, `:405` | 처리량·지연 최적 배치 두 행 | XS | docs | TRACKER 결정 |
| 63 | jev §7 | `plan.md:25` 빠른/느린 라우팅 기준 | 없음 — `plan.md:188`에 기준 | | XS | docs | DROP |
| 64 | jev §7 | `plan.md:81` k*≈2.2 대 MUL-43 τ | 부분 — `:235` k*≈2.2, τ 줄은 없어짐 | | XS | docs | DROP |
| 65 | jev §7 | `plan.md:121` CPU 표가 HANDOFF로 | 없음 — HANDOFF 병합, 「지금」에 스코어보드 | | XS | docs | DROP |
| 66 | jev §7 | `plan.md:246` THP 음성 결과가 코드 스팬 안 | 예 `plan.md:725` | 발견을 산문으로 꺼내기 | XS | docs | DO-NOW(L) |
| 67 | jev §7 | `hybrid-lit-report.md:424,576` ReMoE·CoX-MoE | 예 | 미획득 수치·미독 논문 | XS | docs | DROP 조사 기록 |
| 68 | jev §7 | `hybrid-engines.md:47` model.py 한 서브층 늦은 HC 독해 | 예 `:45-47` 주인 없음 | 층간 겹침 가능성 독해 | S | docs | TRACKER 조사 라운드 |
| 69 | ktok §5, ㊶ (중복) | `bench_v41_host.rs:425` `blocks()` 할당 | 옮김 `:464`, 호출 `:484,501,536,546` | 시간 안 할당 | XS | measurement-tooling | DO-NOW(H-load) |
| 70 | ktok §5, ㊶ (중복) | `ops.rs:1599` `MAX_DEFER_SLOTS` 16 | 예 `:909`, `:1599` | 16 초과 그룹이 호출자 결합 | XS–S | perf | TRACKER 디스패치 경로 |
| 71 | ktok §5, ㊶ (중복) | `cores.rs:817` q3_K 1col 비트 핀 | 없음 — `gate_mcol.rs:13,340`이 핀(`e0919a3`) | 게이트 | XS | measurement-tooling | DROP |
| 72 | ktok §5 | `plan.md` 「모델」 `BW_host` 128–130 | 없음 — E2 재핀(`ef2f43f`) | | XS | docs | DROP |
| 73 | markov-e6 §5, plan `:21` | `gate_dspark_read.rs:35` + 레시피 경로 | 예 `:34-35` 기본값, `justfile:770` 리터럴 | `BLOOMERY_DSPARK_MODEL`을 프로필에서 export | XS | measurement-tooling | DO-NOW(H-just) |
| 74 | markov-e6 §5 | `spec-decoding-report.md:57` "모른다" | 예 | E6 결과 0.06–0.08로 갱신 | XS | docs | DO-NOW(L) |
| 75 | nvme §6-1 | 헬퍼 스레드 팔 | engpre가 구현 중 | | S | perf | DROP 비행 중 |
| 76 | nvme §6-2 | 소비 쪽 minor fault, 사본 버퍼 | engpre·E1b 범위 | | S | perf | DROP 비행 중 |
| 77 | nvme §6-3 | `engram/src/lib.rs` "caching is not the design" | 예 (`cache.rs`·`reuse.rs` 있음, `:21` "nothing is copied") | 재사용 계측 먼저 | M | perf | TRACKER B3 |
| 78 | nvme §6-4 | 버킷별 정책 | 설계 논점 | | M | perf | TRACKER B3 |
| 79 | nvme §6-5 | 확대 16× 논점 | 설계 논점 | | — | docs | DROP |
| 80 | nvme §6-6 | O_DIRECT면 `/proc/self/io`·majflt 계측이 죽음 | 설계 논점 | 대체 증인을 같은 라운드에 | S | measurement-tooling | TRACKER B3c 스펙 조항 |
| 81 | nvme §6-7 | 페이지 캐시 예산 | 설계 논점. load의 fadvise와 겹침 | | S | perf | TRACKER |
| 82 | probes §5, plan `:21` | `host-rate.sh:47-52` env 팔 루프 | 예 | 한 임대 안 env 팔 교대 | S | measurement-tooling | TRACKER |
| 83 | probes §5, plan `:21` | `host-rate.sh:22-26` `--threads` 첫 인자만 | 예 `:22` | 위치 무관 파싱 | XS | measurement-tooling | DO-NOW(A) |
| 84 | probes §5 | `common.md`·`spec-probes.md` `cat` 임대 확인 | 없음 — common.md 고침(plan `:21`) | | XS | docs | DROP |
| 85 | probes §5 | `plan.md:620` E3 `HYBRID_OVERLAP=0` bench 팔 | 없음 — `:621` "GPU 팔로 남음" | | XS | docs | DROP |
| 86 | probes §5 | `Bench::round` `--warmup 1` | 사용 주의 | | — | measurement-tooling | DROP 버그 아님 |
| 87 | probes §5, telem(plan `:21`) (중복) | 3090 스텝 majflt 500–1100 | load 라운드 대상 | | S | measurement-tooling | DROP 비행 중(load) |
| 88 | skew §5, plan `:21` | `gate_deepseek41_skew.rs` step 게이트 ~350줄 복제 | 예 | `src/bin/common/` `#[path]` 모듈 | S | lint-quality | TRACKER 하니스 라운드 |
| 89 | skew §5, plan `:21` | `hybrid.rs` `Boundary.hsum` 이중 창 | 예 `:576`·`:598`, 읽는 곳 `deepseek2/dispatch.rs:999` | 필드 제거, `pages[0].hsum`으로 | XS | lint-quality | DO-NOW(B) |
| 90 | skew §5, plan `:21` | `HybridStats.go_early` pair 모드 의미 | 예 `hybrid.rs:966,1206` | 의미 재정의 | XS | measurement-tooling | TRACKER 결정 |
| 91 | skew §5 | `GpuModel::logits()` pair 뒤 행 A | 예 `model.rs:938-940` | 문서 한 줄 | XS | docs | DO-NOW(H-ngdraft) |
| 92 | b4-plan §6 | ik `build_deepseek4.cpp:719-727` rope 전 dup | 업스트림 #2455 | 업스트림 | XS | correctness | TRACKER MUL-7 |
| 93 | b4-plan §6 | `gpu-gates/lib.rs:277-280` `file_name` | 없음 — `:417` | | XS | measurement-tooling | DROP |
| 94 | b4-plan §6 | `gpu-gates/lib.rs:736` `.i32/.i64` 리더 | 없음 — `FileElem::I32/I64` `:339-340,1022` | | S | measurement-tooling | DROP |
| 95 | b4-plan §6, ⑩ (중복) | `dump_ref.cpp:272` src 열 occurrence·src2–5 | 예 — 머리 `:667` src0/src1만 | | S | measurement-tooling | TRACKER dsref가 덤퍼 쪽 |
| 96 | b4-plan §6 | `tools/ref/dump.sh:98` `--defer-experts`, REF_CTX·세트 이름 | 예 (옵션 없음, `:63` `CTX=$REF_CTX`) | | XS | measurement-tooling | TRACKER dsref 소유 가능 |
| 97 | b4-plan §6, ⑩ (중복) | 제자리 연산 별칭 검사기 | 예 (`tools/`에 없음) | | S | measurement-tooling | TRACKER 95가 선행 |
| 98 | b4-plan §6 | `tensor.rs:106` Q8Act 상한 20480 | 없음 — `Q8ACT_MAX_K = 20_480` `:92` | | XS | correctness | DROP |
| 99 | b4-plan §6 | `bench_v41.rs` heads 사이트 | 없음 — `Kernel::Q8Heads` `:149` | | XS | measurement-tooling | DROP |
| 100 | b4-plan §6 | `docs/oracle.md` 파일 이름·FA f16 문장 | 없음 — `oracle.md:62` | | XS | docs | DROP |
| 101 | b4-plan §6 | ik `llama-cparams.h:47` "off by default" | 없음 — ik#2508 머지 | 업스트림 | XS | correctness | DROP |
| 102 | b4-plan §6 | ik DEEPSEEK41 1e-20, skelectric 같은 패턴 | 알림 | 업스트림 | XS | correctness | TRACKER MUL-7 알림 |
| 103 | b5-plan §8 | V2-Lite 잔차 memcpy 핑퐁, argmax 여러 블록, router_topk 융합, norm_quant ×54 | 예 (plan ⑤ V2-Lite 레버에 그대로) | 네 레버 | S–M | perf | TRACKER V2-Lite |
| 104 | b5-plan §8 | `ik-ppl.sh:159` KLD base 경로 | 없음 — `--kld-base` `:17,260` | | XS | measurement-tooling | DROP |
| 105 | b5-plan §8 | `workstation.rs:106` 게이트 배치 | 없음 — `plan_gate` `:133` | | S | measurement-tooling | DROP |
| 106 | b5-plan §8 | `measured.py:16-17` 가정 교체 | 없음 — plan ⑨ 취소선 | | XS | docs | DROP |
| 107 | b5-plan §8 | `bench_v41.rs:142` 라우터 bf16 → f32 | 예 `:158-163` | 바이트 2배(K5) | S | perf | TRACKER B8 카드 |
| 108 | b5-plan §8, plan ⑨ (중복) | nsys-d6 3090 추적 A6000 재측정 | 예 (plan ⑨ `XS`로 남음) | | XS | measurement-tooling | TRACKER 리드 측정 |
| 109 | b5-plan §8 | `models/deepseek41.sh` d1n 변형 | 없음 — `:97` | | XS | measurement-tooling | DROP |
| 110 | v41-placement §4 | `roofline.md:21-23` + `gguf-inventory.rs:18,110` dense에서 engram 밀집 누락 | 문서는 정정(`:21`), 도구는 예 (`:18` "total − routed − engram") | 도구가 engram_wkv/k/q를 dense로 | S | measurement-tooling | TRACKER 도구 출력·inventory 문서 재생성 |
| 111 | v41-placement §4 | `plan.md:406` rows % 4 | 없음 (plan 재작성, 해당 문장 없음) | | XS | docs | DROP |
| 112 | v41-placement §4 | `host-surplus.md:185-187` MLA 산수 | 부분 — `:186-187` 정정, `:31` 표에 599 MB 남음 | | XS | docs | DO-NOW(L) `:31` |
| 113 | v41-placement §4 | `roofline.md:98` 576차원 | 없음 — `:98-99` 정정 | | XS | docs | DROP |
| 114 | v41-placement §4 | `gpu-design.md:37` 8 KB | 없음 — `:37` 81,936 B 명시 | | XS | docs | DROP |
| 115 | v41-placement §4 | Q8_0 scale 평면 f16 | 없음 — ㊲이 B11c로 닫음 | | M | perf | DROP |
| 116 | v41-placement §4 | `q8f32.rs:467` dp4a q8_1 판 | 예 (미측정) | int8 활성 | M | perf | TRACKER 정확도 클래스(plan ⑤) |
| 117 | v41-placement §4 | `lib.rs:1567` `q4k_gemv_sel` 없음 | 없음 — `q4k_sel.rs:66` | | S | perf | DROP |
| 118 | v41-placement §4 | `gguf/lib.rs:164-240` 단일 파일 리더 | 없음 — `Split` `:431` | | S | correctness | DROP |
| 119 | v41-placement §4 | `weights.rs:143` 전체 텐서 업로드만 | 없음 — `load_rows` `:198`, gather `:379` | | S | perf | DROP |
| 120 | v41-placement §4 | `weights.rs:243` Q5 k%32 | 없음 — `placement.rs:206` `k.is_multiple_of(blck)` | | XS | correctness | DROP |
| 121 | b5deep ㉗ | `body.rs:680` `history.capacity()` | 옮김 `gpu-deepseek41/src/body.rs:516`, `:1096` | 저장된 ctx_max와 비교 | XS | correctness | DO-NOW(H-load) |
| 122 | b5deep ㉘ | `generate_ds41::run` `fed > a.ctx` | 예 `generate_ds41.rs:303` | 계획 ctx_max와 비교 | XS | correctness | DO-NOW(H-ngdraft) |
| 123 | ikload ㉙ | llama-bench `--n-cpu-moe` 대 로더 `ncmoe` 방향 | 업스트림 후보 | 업스트림 | S | correctness | TRACKER MUL-7 FAIL-first |
| 124 | b5gen ⑩, ikload ㉚ (중복) | `depth-ds41.sh`·`depth-gpu.sh` CPU 경합 | 예 (`[cpu-busy]` 0건) | busiest 증인 → `[cpu-busy]`·STRICT | S | measurement-tooling | TRACKER 러너 라운드 |
| 125 | ikload ㉛ | ik 팔만 돌 때도 generate_ds41 빌드 | 예 `justfile:698` | | XS | measurement-tooling | DO-NOW(H-just) |
| 126 | ikload ㉜ | `IK_GPU_ENV` 빈 값 불가 | 예 `models/deepseek41.sh:66` `:=` | `=`로 | XS | measurement-tooling | DO-NOW(H-just) |
| 127 | b5gen ⑪ | generate_ds41 스텝별 fault | 없음 — `c1e733e`(telem) `:465,490` | | S | measurement-tooling | DROP |
| 128 | b5gen ⑫ | ik `add_cpu_buft_overrides` 로그 | ikload가 원인 확정(`e473e0e`) | 업스트림 | S | correctness | DROP |
| 129 | b5index ⑬ | `Seam::Attn.list` 따로 고름 | 예 `gpu-deepseek41/src/body.rs:1007` | 게이트: 소유자 하나로 | S | measurement-tooling | TRACKER |
| 130 | b5index ⑭ | `check_selection` 퇴화 점수 헛통과 | 예 `gate_deepseek41_step.rs:1916` | 게이트: 비퇴화 조건 | S | measurement-tooling | TRACKER |
| 131 | b5index ⑮ | `score_band` 동률 밴드가 행 거의 전부 | 예 `gate_deepseek41_step.rs:1744` | 게이트: 밴드 모델 재검토 | M | measurement-tooling | TRACKER |
| 132 | b5index ⑯ | `place` 조각 scratch 67 MB 대 2.7 MB | 미확인 | 추정 교정 | S | measurement-tooling | TRACKER |
| 133 | b5index ⑰ | cuda-oxide PTX 루트 쓰기, 캐시 `.so` 경쟁 | 없음 — 원장 #18·#19 | 업스트림 | XS | correctness | DROP 원장에 있음 |
| 134 | cpu3 ⑱ | `measure-profile` 레버 env 못 넘김 | 예 `justfile:355-356` | | XS | measurement-tooling | DO-NOW(H-just) |
| 135 | cpu3 ⑲ | 프리페치 첫 d행 | cpu5 뒤 미확인 | | XS | perf | TRACKER 측정 필요 |
| 136 | cpu3 ⑳ | 기본 spin 20000 | 예 `threads/src/lib.rs:31` | SPIN=160000 6바퀴 | S | perf | TRACKER 측정 |
| 137 | cpu4 ㉑ | 16블록 이하 단일 세그먼트 플랜 | 미확인 | 깊이 200–500 −1.3 % | S | perf | TRACKER 디스패치·A/B |
| 138 | cpu4 ㉒ | prefetch가 head 2..16에도 | cpu5 번들 타일 뒤 미확인(`attn.rs:1872,2175`) | | XS | perf | TRACKER A/B |
| 139 | cpu4 ㉓ | `tests/kv.rs:77,115`·`tests/mt.rs` 6토큰 | 예 (오라클 토큰 그대로) | 게이트: 깊은 불변 시험 | M | measurement-tooling | TRACKER |
| 140 | cpu4 ㉔ | `attn.rs` `#[doc(hidden)] pub` R27 | 예 `deepseek2/attn.rs:1128,1138,1152,1160,1595,1636` | | S | lint-quality | TRACKER 가시성 |
| 141 | cpu4 ㉕ | 프로파일 `attn_heads` 한 줄에 디스패치 셋 | 미확인 | 디스패치별 span | S | measurement-tooling | TRACKER |
| 142 | cpu2(⑥ 잔여) | `attn.rs` allow R16 reason | 예 `deepseek2/attn.rs:84`, `:2450` | reason 달기 | XS | lint-quality | DO-NOW(E) |
| 143 | cpu5 ㉝ | `flash_segment` 세그먼트 0 불균형 | 예 `attn.rs:1249` | 키 단위 경계 | M | perf | TRACKER 밴드 게이트 재판정 |
| 144 | cpu5 ㉞ | `SPLITK_R` 세그먼트 우선 배치 | 예 `attn.rs:2900` | 헤드 우선(패치는 스크래치) | S | perf | TRACKER 측정 1회 |
| 145 | cpu5 ㉟ | KQ0 DRAM 스트림 교차 | 미측정 | | M | perf | TRACKER |
| 146 | b5prof ㊱ | `hybrid.rs:948` 합류 0 +1 ms | 측정 항목 | NVTX를 generate_ds41에 | S | measurement-tooling | TRACKER |
| 147 | b5prof ㊱ | 층당 겹침 70–150 대 다리 690 µs | 조사 | | M | perf | TRACKER |
| 148 | b5prof ㊱ | `nsys-gpu.sh` CPU 샘플링, ctx d+128 고정 | 예 `:55`, `:60`(`-s` 없음) | | XS | measurement-tooling | DO-NOW(A) 샘플링 끄기만. ctx는 그대로 |
| 149 | auditbudget ㊲①, auditcpu ㊴ A (중복) | 호스트 다리 3.39–3.47 ms | E2·E4 부분 답(`ef2f43f`) | 원인 축소 | S | measurement-tooling | TRACKER |
| 150 | auditbudget ㊲②, auditload ㊳④, auditgpu ㊵ P1·Q1·Q4·Q7 (중복) | `body.rs:480` engram_kv 그늘, 그늘 게이트, 문서, 죽은 커널 | 없음 — `4f230f5`, `f56e770` | | S | perf | DROP |
| 151 | auditbudget ㊲ | G2 예측표 1,068 대 1,110 | 예 `plan.md:376` | +42 갱신 | XS | docs | DO-NOW(L) |
| 152 | auditload ㊳①②③ | engram 헬퍼, 소유형 HostLock, fadvise | 비행 중(load·engpre) | | M | perf | DROP 비행 중 |
| 153 | auditload ㊳ 품질 | `deepseek2/mod.rs:566-612` `hybrid_row` 가짜 Plan | 없음 — hotset "가짜 Plan 삭제"(`e32664a`) | | S | lint-quality | DROP |
| 154 | auditload ㊳ 품질 | `quant.rs:448,490` unwrap | 옮김 `gguf/src/quant.rs:464`, `:506` | `first_chunk`로 | XS | lint-quality | DO-NOW(E) |
| 155 | auditload ㊳ 품질 | `engram/src/lib.rs:77` `Io` 문맥 없음 | 예 `:79` | 경로·사이트 문맥 | XS | lint-quality | DO-NOW(H-engpre) |
| 156 | auditload ㊳ 품질, b4plan2(⑧) (중복) | `engram/src/lib.rs:35` "strict reader refuses" | 예 `:35` | 낡은 문서 | XS | docs | DO-NOW(H-engpre) 155와 같이 |
| 157 | auditload ㊳ 품질 | `gpu/src/lib.rs:1378` "device 0" | 예 | `for_card` 반영 | XS | docs | DO-NOW(B) |
| 158 | auditload ㊳ 품질 | Split을 세 번 연다, plan 두 번 | `generate_ds41.rs:295-302` 등 미재확인 | | S | perf | TRACKER load 뒤 |
| 159 | auditload ㊳ 품질 | 트리아지 ① `one_tensor_file` 경로 | 예 `plan.md` ①이 `model/tests/ops.rs`라 적음 | `ops.rs:2441`로 정정 | XS | docs | DO-NOW(L) |
| 160 | auditload ㊳ | 3090 둘째 모델 걸림돌 여섯 | 설계 | | M | perf | TRACKER DSpark 카드 |
| 161 | auditcpu ㊴ P1·P3 | 층 l+1 라우터 선적용 L3 프리페치, 풀 디스패치 셋 → 둘 | 설계 | | L/S–M | perf | TRACKER |
| 162 | auditcpu ㊴ 품질 Q3 | `qdot/src/lib.rs:483` `hmax_ps` Safety "SSE3" | 예 `:483` | "AVX and SSE3"로 | XS | docs | DO-NOW(E) |
| 163 | auditcpu ㊴ 품질 Q4 | `ops.rs` `Sink::Into` 원시 포인터 | 옮김 `Dest::Into` `ops.rs:736,758,836` | par 헤더만 넘기기 | S | lint-quality | TRACKER 디스패치 경로 |
| 164 | auditcpu ㊴ 품질 Q5 | `moe.rs:285` unwrap, `:313` `logits.into_data()` | 예 `moe.rs:285` | `route_inner` 안 맨 unwrap. `:313`은 라우터 로짓 블록이 free list 밖으로 나가 층마다 malloc 해제 | XS | lint-quality | TRACKER 디스패치 경로 — cpu1의 `route_inner` 할당 항목과 한 라운드, 같은 임대 A/B |
| 165 | auditcpu ㊴ 품질 Q6 | `(u32, f32)` 튜플 세 크레이트 | 예 `moe.rs:764,810,908,1024` | 이름 있는 타입 | XS–S | lint-quality | TRACKER 세 크레이트 동시 수정 |
| 166 | auditgpu ㊵ Q2 | `gate_deepseek41_step.rs:774` replay=eager D2 핀 없음 | 예 `:777` `[STEP4, D1]` | 게이트: D2를 루프에 | XS | measurement-tooling | DO-NOW(D) 판정 줄이 D2만큼 늘어난다 — 회귀 아님 |
| 167 | auditgpu ㊵ Q5 | SAFETY "as above" 14곳 | 예, 14곳 전부 남음 — `indexer.rs` 5, `compress.rs` 4, `hc.rs` 2, `chain/ffn.rs` 2, `engram_gate.rs` 1 | 경계 적기 | XS×14 | lint-quality | DO-NOW(C) 주석만, ptx-scan 동일로 증명 |
| 168 | auditgpu ㊵ Q6 | 층→조각 인덱스 불변식 7곳 | 예 `body.rs:333,348,1364` 등 | 헬퍼·newtype | S | lint-quality | TRACKER |
| 169 | auditgpu ㊵ Q8 | 런타임 assert·expect·`as usize` | 부분 — head.rs 없음. `chain/attn.rs:417` expect, `:686` `as usize` 남음 | const·`try_from` | XS | lint-quality | DO-NOW(C) |
| 170 | auditgpu ㊵ P2·P3·P4·P5 | L2 프리페치, wait_go 접기, handoff 런치 제거, q_b 두 런치 | 설계 | | S–M | perf | TRACKER |
| 171 | ktok ㊶ | `attn.rs:701` gm≠1, `head.rs` m=1, `hc.rs:83` `HC_MAX_TOKENS` | 예 `chain/attn.rs:744`, `hc.rs:76` | k토큰 벽 | S | correctness | TRACKER ktok 설계 |
| 172 | dspark ㊷ | `router.rs:41,44` `N_EXPERT`/`N_USED` 상수 | 예 | const generic | S | lint-quality | TRACKER DSpark |
| 173 | dspark ㊷ | `experts.rs:49` combine 슬롯 리터럴 6 | main에서 못 찾음(grep 두 번) | | S | lint-quality | TRACKER dspark 카드를 열 때 위치 확인 |
| 174 | dspark ㊷ | `gguf/quant.rs:33` MXFP4 없음 | 없음 — `ca0e2cf` | | XS | correctness | DROP |
| 175 | dspark ㊷ | `experts.rs:23` SwiGLU clamp 규칙 문서 | 부분 — ik 규칙은 적힘, `model.py`의 `silu(min(g,10))` 차이 없음 | 차이 한 줄 | XS | docs | DO-NOW(C) |
| 176 | telem(plan `:21`) | `HybridStats.straggle_max_ns` 누적 최댓값 | 예 `hybrid.rs:985,1210` | 창 max | S | measurement-tooling | TRACKER generate_ds41 쪽도 바뀜 |
| 177 | telem(plan `:21`) | `depth-ds41.sh:155` distinct_tokens가 step 0 포함 | 예 `:155` | | XS | measurement-tooling | DO-NOW(H-ngdraft) |
| 178 | dspark ㊷ | 3090 대역폭 250 W 캡 아래 | 예 (미확인, D 유도의 전제) | 캡 아래 3090 스트림 읽기 실측 | XS | measurement-tooling | TRACKER 리드 측정 |

행 160 번호는 없고 176에서 178로 이어집니다. 전체 행 수는 177개입니다. 같은 표를 저장해 둔 파일은 다음입니다.

`/private/tmp/claude-501/-Users-hckim-repo-bloomery/e5714b41-3850-42be-a722-79e1f225c3b5/scratchpad/improvement-spots.md`

git 상태와 박스는 건드리지 않았습니다.


## 처분 기록 (리드)

- 2026-09-24 06:50 **fixae 머지 `3a69ddb`**: 행 11·27·83·148(A), 25·142·154·162(E) 닫힘. 리드 재실행 check·lint 168·fmt·recipes·gate-dspark-read·gate-ds41-meta·gate-qdot·gate-1-1 전부 rc=0. 스펙 밖 하나 더 고침: `deepseek41/mod.rs`의 헬퍼 둘(스펙의 `hparams.rs:8-38`이 오기). fixae가 새로 본 것(트래커에 추가, XS 넷·S 하나): `dequant_ref.cpp` manifest.txt 비원자적 쓰기; `build-dequant.sh:10`과 형제가 공유 `$REF_BIN` 바이너리를 제자리에서 덮어씀(두 트랙 동시 gate-1-1이면 반쯤 쓰인 실행 파일, S); `IK_SQRT_SOFTPLUS` 상수 두 곳 중복; `fail()` 경로가 `.tmp.<pid>`를 안 지움; `justfile:369` `build-ref`가 이름과 달리 dequant_ref를 안 지음.
- 리드 문서 10건: `387501b`(행 2·9·52·58·59·66·74·112·151). 행 159(`one_tensor_file` 경로)는 plan.md에 그 문자열이 없어 확인 불가 — DROP.
- 2026-09-24 07:15 **fixcd 머지 `48c672e`**: 행 42·167·169·175(C), 20·33·166(D) 닫힘. 리드 재실행 check·lint 168·fmt·recipes·gate-gpu-ds41-hc·ds41-step·chain-attn 전부 rc=0, ptx-scan 표 md5 동일. 스펙과 다른 결과 둘: 175는 "차이 없음"이 아니라 `model.py`가 `silu(min(g,L))`, 우리(ik)는 `min(silu(g),L)` — 한 줄 문서; 33은 hc 게이트가 마지막 층(39)의 헤드 fold를 단독 fold로 세게 됨(`ds41_hc_post` 78→77, `fold` 2→3, total 160 불변 — 엔진 런치와 일치, 게이트 분류 정정). fixcd가 새로 본 것(트래커 XS 넷): `experts.rs:252` `sel` 문서("expert ids"→슬롯 자리), `params.rs:81` `Table::index`와 `chain/attn.rs::table_index` 중복(한 주인으로), plan ㊷의 clamp 차이 ~5e-5는 상대값(절대 상한 L=10에서 ~4.5e-4[유도]), `model.py:845`는 `swiglu_limit > 0`일 때만 clamp(우리는 L ≤ 1e-6 건너뜀).
- 2026-09-24 07:40 **fixbh 머지 `386ea8c`**: 행 15·19·22·157(B), 32·155·156(H-engpre), 73·125·126·134(H-just), 177, 69 닫힘(13). 리드 재실행 check·lint 168·fmt·recipes·chain-glue·engram·dspark-read·p6·mcol·bench-cpu-v41-host-check 전부 rc=0, ptx-scan 표 md5 동일. **자 변경**: `bench_v41_host`의 블록 할당이 토큰 타이머 밖으로 나가 `BW_host` 표는 전후를 같은 행에 두지 않는다(머지 뒤 재핀 자리 필요); `distinct_tokens`는 step 0 제외(N−1). fixbh가 새로 본 것(트래커 XS 5·S 1): `prefetch.rs:203` spawn `?`의 문맥 없는 Io; `build-qdot-ref.sh:41` DSpark 경로 리터럴 중복(deepseek2 프로필 아래라 못 읽음); `bench_v41_host` `token()`/`engine_rows_layer`의 층별 Vec 할당 5종(S); `engine_layer` 한 줄 래퍼; `glue.rs:490` 오류 라벨 이름 틀림; `box.sh:66` `BLOOMERY_BOX_ENV` 공백 값 불가(헤더에 명시). 남은 H: 45·89(hybrid.rs — load2 뒤), 44·121(load2 소유 파일), 46·91·122(generate_ds41.rs·model.rs — load2 뒤).

## 2026-09-25 추가 — fixup3(`9e72c80`)가 닫은 DO-NOW 16건

표의 DO-NOW(A·B·C·D·E) 가운데 리드가 09-25 새벽 스펙 `spec-fixup3.md`로 묶어 보낸 16건이다. 15건 닫힘, 1건 절반.

| # | 자리 | 처분 |
|---|---|---|
| 1 | `crates/model/tests/ds41_host.rs` 라우팅 id `as u32` | 닫음 — `u32::try_from` + 이름 붙은 panic |
| 2 | `gpu/src/model/launcher.rs` `ARM_SPIN`·sibling 테스트 | 닫음 — `ARM_YIELD`, 테스트는 마스크 안 SMT 짝 cpu를 고르고 없으면 SKIP |
| 3 | `tools/ref/depth-ds41.sh:149` `head=?` | 닫음 — 러너가 선 트리는 `head=<BLOOMERY_GIT_COMMIT>(box.sh) dirty_files=?`; 남의 트리는 `head=? dirty_files=?`(빌린 커밋을 찍지 않는다) |
| 4 | `depth-ds41.sh` CPU 경합 | 닫음 — `lease.sh` `guard_cpu`(단일 소유자), 행 끝 ` [cpu-busy]`, `BLOOMERY_OTHER_STRICT=1` rc 75; 문턱 50 %는 고른 값 |
| 5 | `MERGE_BATCH` 세 벌 | 닫음 — `gpu::flash::MERGE_BATCH` 하나, ptx-scan 116행 동일 |
| 6 | `qdot/tests/qdot.rs:1-5` 머리말 | 닫음 |
| 7 | `tools/ref/*_rate.cpp` 채움 바이트 | 닫음 — 일곱 하네스가 `qdot-rate`와 같은 바이트 |
| 8 | `oracle/deepseek41.rs` 세트 이름 `_plain` | 닫음 — 실제로 연 세트 이름을 찍는다 |
| 9 | `glue.rs:1095`·`prefetch.rs:466` float helper의 한 cpu 마스크 | 닫음 — `prefetch.rs` `place_helper`가 마스크를 넓히거나 이름 붙여 거부(FAIL-first rc 101 → 0); `glue.rs`는 고칠 것이 없었다 |
| 10 | serve·chat load 줄 `host_shadow=` | 닫음 |
| 11 | `generate_ds41.rs` 이중 open | 절반 — `Split::open`은 한 번; `PlanInputs::read` 둘째는 `body.rs:105`(ds41hcbranch 소유) → gpuq1 |
| 12 | `GREEDY_MARGIN` 중복 | 닫음 — gpu-gates lib 상수 하나 |
| 13 | `dump.sh` `.staging` 잔류 | 닫음 — 트랩이 트레일러 없는 `.staging`을 지우고 알린다 |
| 14 | `engram-corpus.sh`·`ik-greedy.sh` `~/` 가드 | 닫음 — `[71520]` 탐침, rc 65; 리드가 기본 tokenizer를 oracle.sh처럼 `ik-tilde`로 |
| 15 | `gguf-inventory.rs` 열 이름 | 닫음 — "roofline.md hand count, V4.1 mixed file" |
| 16 | `qdot/src/lib.rs` q8_K `v_wrap_i8` 탐침 | 닫음(초록) — `hw_q8k_codes_at_subnormal_scales`: f32 iscale이 \|y\| < 127.5를 지키므로 wrap 도달 불가, 테스트가 증명 |

라운드가 새로 보고한 지점 여덟은 `plan-triage.md` 「도구」·「받을 라운드별」에 올렸다.
