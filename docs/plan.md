# bloomery — 계획

이 문서는 계획이고, 숫자는 [rig-log](https://github.com/midagedev/rig-log)에 측정된 뒤에만 여기 옮겨 적는다. 통과 기준은 전부 측정으로 쓴다. 2026-09-25 새벽에 한 화면으로 다시 썼다 — 그 전 판은 [`plan-ledger.md`](plan-ledger.md) 「plan.md 2026-09-25 이전 판」에 원문 그대로 있다. 열린 항목은 [`plan-triage.md`](plan-triage.md), 끝난 것은 장부, 기계·파일·툴체인 사실은 [`facts.md`](facts.md)다.

## 지금 (2026-09-25 새벽)

**목표는 "커뮤니티 공개"(사용자 09-24 09:35)**: 3090 한두 장과 AVX2 Threadripper를 가진 사람이 받아서 돌리고 우리 숫자를 재현하는 공개 엔진. 순서를 정하는 축은 ① 타겟 카드의 헤드라인 숫자 ② 받아서 돌리는 경로 ③ 차별점(어긋난 드래프트 + DSpark, 뜨거운 목록) ④ 사람들이 쓰는 모델(Qwen3·GLM-4.7-Flash·V4-Flash)이다.

- **파일은 공개 `DeepSeek-V4.1-Flash-Q3_K_M` 하나다**(사용자 09-24 14:30 결정, `d771085`로 기본값). 혼합 파일 `attnQ8`은 시팅 10(참조 재생성) 뒤 지운다.
- **헤드라인(측정, A6000, plan (a) 카드 expert 2,668, 뜨거운 목록, prose/code 512 프롬프트 뒤 n 96)**: 우리 **42.9 / 42.4 tok/s**, ik plain 20.0(같은 임대, 혼합 파일 플래그 — 공개 파일 최적 플래그는 N5 미탐색), 예산 38G 40.3, 24G(3090 흉내) 35.8(rig-log 09-24#public-q3km-prose-code-and-budget). lcg 깊이 6·목록 없음은 29.7(09-25#launch-thread-lever-ab), llama.cpp PR 브랜치 21.5. Qwen3-30B-A3B 전 카드: mainline llama.cpp 대비 깊이 6 +6 %, 4096 −7 %(09-24#qwen3-30b-a3b-e28).
- **DSpark를 붙이면 [유도]** 48 GB급 ~45–50 tok/s = ik + DSpark의 1.4–1.5배; 3090 한 장은 +7 %(8.5 GB 드래프트가 expert 자리를 먹는다 — E27 레버 셋).
- **공개를 막는 것**: ① E21 3090 실측(공개 파일, 승인됨 — 시팅 큐 1) ② 우리 DSpark tok/s(`dsloop` `c00274c` 착륙 — 무손실 루프 동작, 측정은 시팅 DS) ③ 사용자 결정(공개 시점 — LICENSE는 MIT로 결정, 09-25) ④ **프리필 숫자**(사용자 09-25: 디코드만큼 중요, V4.1·Qwen3 둘 다) — ~~V4.1은 배치 프리필이 없어 토큰마다 디코드 스텝(≈ 43 tok/s, 512토큰 ≈ 12 s[유도]), Qwen3는 eager 8위치 패스; 우리·ik·llama.cpp 어느 쪽도 pp를 잰 적이 없다.~~ 09-25 15:30 정정: Qwen3는 `qwen3prefill` `dfa8b5d`로 pp512 **5,304**(llama.cpp 4,256의 1.25배)·pp4096 2,615(0.63배, rig-log 09-25#qwen3prefill-ab); V4.1은 `ds41batch`(B0, 비트 정확 512 배치)가 착륙 중이고 pp는 착륙 직후 시팅이 잰다(측정 대기).
- **비행 중(파동 21)**: `dspark-q3k` ‖ `ds41hcbranch` ‖ `unionhost` → `ds41splitk` ‖ `qwen3route` ‖ `fixup3`. 09-24 밤 ~ 09-25 새벽 착륙: `ds41router` `dd5422b` · `ds41join` `075ceef` · `ds41hcfin` `ecaacdd` · `ds41dense` `4953fdb` · `pubflip` `d771085` · `loudnan` `5fe50f6`(카드의 조용한 NaN → 폴트 워드) · `tokfix` `3ef2c8a` · `v4host` `55b5249` · `qwen3deep` `5712f25` · `gate-ds41-load` 닫음 `8dd878f`. 리드 재실행 lint **167**(main `43d5ba9`).
- **업스트림 PR**: ik 비드래프트 0(#2520·#2528 09-24 머지), 드래프트 [#2522](https://github.com/ikawrakow/ik_llama.cpp/pull/2522)(증거 준비 끝, #2512 착륙 뒤 ready)·[#2507](https://github.com/ikawrakow/ik_llama.cpp/pull/2507); cuda-oxide [#1329](https://github.com/NVlabs/cuda-oxide/pull/1329) 열림(#1314·#1321 머지); cutile-rs #309 열림. 한 레포에 비드래프트 1–2개까지. **ik DSpark는 탐욕 출력을 바꾼다**(rig-log 09-25#ik-dspark-not-lossless) — 공개 표의 ik + DSpark 행에 한 줄.

**진척·일정(리드 유도, 2026-09-25 15:30).** 분모는 09-24 공개 결정 이후의 공개 범위(M1 네 축 숫자 + DSpark 숫자 + 3090 실측 + M2 실행 경로 + 글·영상·스크럽)이고 단위는 카드 크기(XS ¼·S ½·M 1·L 2)다. 09-24 밤부터 오늘 15:30까지 착륙한 것이 약 27단위(라운드 19개 + 시팅 13회), 최소 공개까지 남은 것이 약 4단위(ds41batch 착륙·pp 시팅, E21, E29, 시팅 10·혼합 파일 삭제, BUILD.md·soak, 글·영상, 스크럽, README 표) → **최소 공개 기준 약 87 %**; 경쟁력 있는 pp(ds41ced·q3pflash·h1fold·ds41overlap·q3ubatch·fixup4·M2b, 약 8단위)까지 넣으면 **약 69 %**, hoststream까지면 약 65 %[유도]. 처리량은 오늘 실측으로 ≤ 4 병렬에서 L 1.5–2.5 h, M 1–1.5 h, 착륙 직렬화 20–30분, 하루 12–15단위. 임계 경로 셋: V4.1 pp4096 = ds41batch → ds41ced(+55 %[유도]) → hoststream(M2b가 정한다); Qwen3 pp4096 = q3pflash → q3ubatch; 공개 = E21 + pp 숫자 + 글·스크럽. 정상 상태의 파동 모양은 V4.1 체인 하나 ‖ Qwen3 하나 ‖ 호스트 티어 하나 ‖ 픽스업·도구 하나 — 규칙 1(`gpu/model.rs`·`chain/*.rs`·`hybrid.rs`는 한 시점에 하나)과 `ubatch.rs` 직렬이 병렬 폭을 정한다. 예측: 파동 24는 09-25 밤 착륙, 25·26은 09-26, 최종 숫자표와 글은 09-27 → **공개 후보 09-27(숫자만)·09-28(M2 포함)**[유도]. 위험: 3090 Xid 79(E21), q3pflash의 밴드 재유도, ds41ced의 정확 조건 둘, hoststream의 메모리 예산(214 + 84.6 GB > 264 GB).

### 마일스톤

| M | 된 것 | 다음 | 막는 것 |
|---|---|---|---|
| **M1 숫자 공개** | A6000 헤드라인(위 — 디코드 축과 ~~**pp512·pp4096 축이 비어 있다**~~ Qwen3 pp 축(09-25); V4.1 pp는 측정 대기), 카드 크기 곡선 24/38/41 GB, 카드 쪽 분해(E26), Qwen3 E28 | **E21** 3090 실측 + N5 ik 플래그 + ik + DSpark 같은 창, 온도 > 0 수락률(E29); rig-log 글(영어 요약 + 한국어) + 30초 영상(승인 뒤); 공개 전 스크럽; "우리가 아는 한 유일한 Rust 엔진" 주장은 글 쓰는 날 재확인 | 빌더 틈의 임대 자리 |
| **M2 돌려 볼 수 있게** | `bloomery-serve-ds41`·`bloomery-chat`·README·BUILD.md·THIRD_PARTY_NOTICES·로고·토크나이저 게이트 | BUILD.md에 prefix 재사용 두 문장, 30분 soak | 공개 시점(사용자; LICENSE는 MIT로 결정) |
| **DFlash(DSpark)** | 커널 `dshc`·`dsmx`, 적재·KV `4b963af`, 블록 패스 B `5dd34cb`, 폭별 그래프 C `11ce715`, 합집합 설계 | ~~`dspark-q3k`~~ `f1d1168` → ~~`dsloop`~~ `c00274c`(w = 1 수락 루프, 시팅 DS가 tok/s) → `uniongroup`(k > 1) | `ref-draft` 재덤프(시팅 10, 승인) |
| **M3 서버** | prefix 재사용(`47e36b5`·`5fdfe11`), reasoning/DSML(`7f68981`), `/props` 엔진 객체(`631cedc`) | ~~슬롯 save/restore(`slots`)~~ `75d2bad`(엔진 스냅샷은 `slotsnap`), 템플릿 정리 라운드 | — |
| **M4 모델** | Qwen3-30B-A3B 전 카드(체인·e2e·PPL·E28), IQ 커널, 선형 조사, V4 호스트 티어 커널 | ~~`qwen3route`~~ `be05ad7` → ~~`qwen3fuse`~~ `ae046aa`(653 → 508노드, 시팅 Q3F) → `qwen3bw`; V4 `v4meta` → 카드·압축기·hc; `linear` → `glm53` | 시팅 D(IQ3_XXS 속도), 사용자 결정(다운로드·DFlash 대상) |
| **M5 비전** | V0 오라클 `01bbe07`, V2 인코더 `4abc188` | V3 텍스트 쪽 주입(`visinj`) | M1·M2 뒤 |

### 파동 (빌더 다섯 이상 — 사용자 09-24 10:16; 측정은 증인으로 거른다)

측정 자리는 임대(flock)로 빌더를 세우고 돈다. 모든 스펙이 빌드 전에 `flock -n /root/bloomery-cpu.lock`를 폴링하므로 자리 하나(≤ 30분)는 빌더를 기다리게 한다. 측정 행은 증인 블록으로 걸러 cargo/rustc가 없던 행만 쓰고, 자리 둘을 연달아 붙이지 않는다. 빌드는 시차를 두어 띄운다(09-23의 다섯 동시 빌드 포화). 지난 파동 표(16–20)는 장부 「파동 이력」과 「plan.md 2026-09-25 이전 판」에 있다.

| 파동 | 병렬 | 리드가 그 사이 | 닫는 조건 |
|---|---|---|---|
| **21 (09-25 04:10~)** | `dspark-q3k` ‖ `ds41hcbranch` ‖ `unionhost` → `ds41splitk` ‖ `qwen3route` ‖ `fixup3` | plan 정리(이 판), E21, `spec-dsloop`, spots-triage 표 | dspark-q3k·hcbranch 머지 → dsloop 자리 |
| ~~22~~ 닫힘 09-25 10:05 | `dsloop` `c00274c` ‖ `qwen3fuse` `ae046aa` ‖ `load3` `5fef4b4` ‖ `slots` `75d2bad` (07:20 발사, base `ac250bb`; uniongroup·gpuq1은 23으로) | 시팅 B·C·D, A3 리베이스 A/B, 시팅 10(승인 뒤) | dsloop 머지 → DSpark tok/s |
| 23 (선발 10:00~) | ~~`prefillbench`~~ `ed3d6f3` ‖ ~~`ds41prefill`~~ 보고(백본 B 배치 호스트 티어, P ≳ 2,000에 D 스트리밍) ‖ `prefilllit`(사용자 09-25: 프리필 우선) → 시팅 L3·DS·Q3F·P(a·b·c)·M1·E21과 겹쳐 프리필 첫 라운드 `ringstage`(+M2 probe) ‖ `q3graph` ‖ `a5gemm`(문헌 보고 뒤), DS 뒤 `uniongroup`, 이어 `ds41batch`·`hosttile`(M1이 정한다)·N4 설계 (≤ 4씩; `gpuq1`은 이미 닫힌 일이라 삭제) | M1 글·영상, E28 재측정 | — |
| **24 (09-25 16:30~, 리드 유도 09-25 15:30)** | `q3pflash`(Qwen3 전용 프리필 flash, L) ‖ `ds41ced`(CED 삼각형 + prefweb 셋 — 균등 분할·prefix+suffix 게이트 케이스·append 계약, L) ‖ `h1fold`(qdot Q3_K −4 오프셋 접기 + 전치 에필로그, 정수 경로 비트 동일, M) ‖ `fixup4`(`_sel` 조용한 건너뜀·`attn.rs:188`·`elem.rs:384`·qwen3 router NaN·`depth-qwen3moe.sh`·hcbranch·dsloop XS 묶음 + M2b 프로브 bin, M) — 넷의 파일이 서로소(`arch/qwen3moe`+`flash_gqa.rs` / `gpu-deepseek41/{body,chain}` / `qdot`+`model/moe.rs` / `crates/gpu/src/{q4k_sel,q6k_sel,elem}.rs`·`gpu-deepseek41/src/attn.rs`·`tools/ref`) | ds41batch 착륙 직후 V4.1 pp 시팅(15분), 파동 경계에서 E21(30분)·Q3X(5분) — 시팅은 빌더가 없는 파동 경계에만(러너와 스펙 둘 다 임대를 폴링한다) | ds41ced·q3pflash 착륙 → V4.1·Qwen3 pp4096 재측정 |
| 25 (09-26 오전) | 아침 빌더 없는 틈에 리드가 oxidefork ①②(핀 이동, 전 게이트) → `ds41overlap`(두 ubatch 겹침, M) ‖ `q3ubatch`(M) ‖ `uniongroup`(M) ‖ `gates①`(하니스 통합, M) | M2b 시팅(10분, 등록 통과 여부 + 메모리 예산 산수), CED 뒤 V4.1 pp | M2b 결과 → hoststream 스펙 확정 또는 폐기 |
| 26 (09-26 오후~) | `hoststream`(M2b 통과·예산 해결 시, L) ‖ `h3tile`(4행 레지스터 타일, M–L) ‖ `slotsnap`(M–L) ‖ `fixup5`(트래커 10–15건) | 시팅 10(밤, 승인), 최종 숫자표 시팅(네 축 + 참조, 한 표) | 09-27 글·영상·스크럽 → 공개 후보(숫자만 09-27, M2 포함 09-28) |

### 리드 직렬 지점과 규칙

- 박스 임대는 하나. 측정 자리는 파동 사이 빌더 틈에, 자리 하나는 30분 안. 타이밍 숫자는 A6000, 3090은 E21 예외(사용자 승인 09-24).
- 머지는 보고 → diff → 리드 재실행 → 순서대로 ff → 푸시. 파동마다 푸시. 비트 불변 트랙 먼저, 산술 순서를 바꾸는 트랙(재핀) 마지막 — 재핀 귀속이 흐려지지 않게.
- 장기 라운드는 자기 워크트리·자기 `arch/<이름>/`에서 파동을 넘겨 살고, 게이트 단위로 30분 상한을 지킨다.
- 서브에이전트의 배경 대기(Monitor·배경 `until`)는 깨우지 못한다 — 스펙에 포그라운드 `sleep`, 리드가 rc 파일을 보고 깨운다.
- 공개 전 스크럽: `192.168`·`100.`·`.ts.net`·`admin`·BMC·호스트명·bug-report 아카이브.
- 파동마다 빌더 하나는 트래커 소화용(사용자 09-24). 「스펙 밖 개선 여지」는 `research/spots-triage-report.md`로 모으고 카드 없는 S 이하는 픽스업 라운드 하나로 10–15건씩.

### 사용자 결정 대기

목록은 [`plan-triage.md`](plan-triage.md) 「사용자 결정 대기」. LICENSE는 MIT 유지로 결정됐다(09-25). 리드 추천: 공개 시점은 레포와 숫자를 같이(E21 뒤).

## 목표

DeepSeek-V4.1-Flash를 이 워크스테이션(A6000 48 GB + 3090 24 GB, sm_86, 5975WX 32코어, 256 GB)에서 우리 엔진으로 서빙하고 공개한다. 호스트는 Rust, GPU 커널은 CUDA Rust(cuda-oxide), CPU expert 티어는 Rust AVX2. 기준선은 같은 자리에서 잰 것이고 공개 비교의 기준은 mainline llama.cpp(와 mistral.rs)다 — ik는 수치 기준(오라클)이자 내부 기준선으로 남는다. 왜 직접 만드는가와 ik 대 mainline 3 % 이야기(#2455)는 장부.

## 모델 — 먼저 유도하고, 측정은 유도가 빗나갈 때만 (2026-09-22)

라운드는 예측을 들고 연다. 측정은 예측이 밴드를 벗어났는지 보는 두 번째 행위이고, 벗어나면 모델의 어느 항이 틀렸는지가 그 라운드의 발견이다. 변경 클래스와 증명 방식, 「성능이 먼저」는 `AGENTS.md`가 정본이다.

### 비용 모델 (디코드 스텝)

`step_ms ≈ Σ_k max(bytes_k / BW, instr_k / issue, t_floor_k) + N_node · c_node + N_barrier · c_barrier + depth · c_key`

| 상수 | 값 | 교정한 측정 | 카드 |
|---|---|---|---|
| `c_node` | 0.852 µs(784노드), 0.871(504노드) — 빈 `touch` 그래프 | B9 `time-gpu-v41` | A6000 |
| `c_barrier` | 0.72 µs + 협동 런치 0.20 | A3g fmerge | 3090 |
| `BW`(큰 런치) | 567–701 GB/s(q8_0 615–680, `q4k_gemv_sel` 579, `q3k_gemv_sel` 619–639, 헤드 q6_K 701; `engram_wkv` 674 = 피크의 88 %). 이 근처의 커널은 발행 바운드가 아니다 | B9, B11, kr-dense | A6000 |
| `t_floor` | 작은 격자의 런치 하한: K=5120 q8_0 8.0 µs(64)·14.0(160), f32 13.5(48), q3_K 5.5–5.8(4–64); K=4096 q8_0 10.0; K=512 q3_K 1.45(16). kr-dense: T ≈ 1.16 + 0.47·iters µs(warp 하나가 행을 끝까지 걷는 사슬) | B11b, kr-dense | A6000 |
| `c_key`(MMA 기본) | 우리 0.160 µs(구간 0.141·0.166), ik 0.176 | 09-22-o 깊이 표 | A6000 |
| `BW_host` | 135–137 GB/s(16스레드 포화, 순수 읽기 상한 140–145, STREAM 147.7 best-of-5·같은 임대 140–144); 디스패치 고정비 6.9 µs[유도] | ktok·자리 1(09-24) | 호스트 |
| 브리지 | 36.0 + 119.5·k_host µs(k_host 2..6 직선; 프로토콜 바닥 21–36 µs × 40층 = 0.8–1.4 ms) | kr-moeattn nsys 행 | A6000 |
| 잡음 | 같은 바이너리 SD 0.6 %; 두 팔 평균 차의 95 % 구간 ±1.0 %(4바퀴)·±0.8 %(6) | 09-21 18회 | A6000 |

정정 이력(3090 시절 값, 재교정 경위)은 장부 「plan.md 2026-09-25 이전 판」. 점유율은 측정할 값이 아니라 계산할 값이다(regs·smem·스레드·SM 수).

### 오차 모델 (진단)

e2e 핀은 이산 개수(마진 ≥ 0.5 불일치 ≤ 6)라 경계에서 동전 던지기다. `exact-forced-32.tsv`가 1023위치 전부의 참 마진을 가지므로 σ(우리 마진 − 참 마진의 RMS)를 진단으로 함께 찍는다 — `gate-gpu-e2e`의 `forced_sigma`, `--margins PATH`. 첫 실측 σ 0.35(스칼라)·0.36(MMA), 시뮬 0.378. 위치별 차의 꼬리는 가우시안보다 훨씬 두껍다(RMS 4배 초과 위치 11개 대 기대 0.1) — 라우터 뒤집힘 의심, 직접 본 것은 아니다. 팔 비교는 σ 둘의 비교다. 모든 숫자는 `tok/s @ n=N, 깊이 D, 카드`로 적는다 — 조건이 없으면 숫자가 아니다.

## 라운드 운영

라운드 하나 = 워크트리 하나 = 파일 경계 하나 = 게이트 하나 = 완료 보고 하나. 위임은 opus 서브에이전트(Agent 도구, `model:"opus"`). 리드는 스펙·diff 독해·게이트 재실행·임대 측정·머지만 한다. 원칙(원문과 사고 경위는 장부):

0. 파동마다 빌더 하나는 트래커 소화용.
1. 같은 파일을 두 라운드가 동시에 만지지 않는다 — `gpu/model.rs`·`chain/*.rs`를 만지는 라운드는 한 시점에 하나.
2. 오라클(ik 덤프)이 직렬을 팬아웃으로 바꾼다 — 오라클 덤프 라운드가 먼저.
3. 상한은 리드의 검수 대역과 박스 임대 하나다. 재는 라운드는 한 시점에 하나, 코드·문서 라운드는 몇이든.
4. 조사는 코드보다 먼저 병렬로.
5. 예측이 없는 라운드는 열지 않는다(「모델」의 값·밴드·증명 방식).
6. 원인이 측정되지 않은 느림에는 구현 라운드를 열지 않는다 — 배가 프로브·참조 독해·커널 카운터 셋이 같은 단계를 가리킨 뒤에.
7. 계기를 한 번 의심한다 — 새 계기의 첫 표는 계기와 독립인 산술과 함께 읽는다(ncu `--launch-count`가 프롬프트 스텝을 잡은 사고; 어느 커널에 시간이 가는지는 `nsys-gpu.sh`가 답한다).
8. 박스를 쓰는 스펙에는 "카드가 떨어지면 즉시 멈춤"(Xid 79 뒤 재부팅은 사람·BMC).
9. 단계 경계마다 감사 파동(수학자 시선 = 비용 모델 잔차, 커널 엔지니어 시선 = 융합·비동기·배치·알고리즘). 구현 스펙은 넷을 적는다: 융합 후보·겹칠 비동기 구간·묶을 배치·대안 알고리즘.

라운드가 끝나면 카드에 실제 소요를 적고 파동 표에 머지 커밋을 적는다. 순서를 바꾸면 이유를 날짜와 함께 남기고 옛 줄은 선을 긋는다.

## 공개

- **참고한 엔진에 대한 예의(사용자 09-24)**: README 크레딧(ik_llama.cpp는 오라클이자 설계 참조, exllamav3·mistral.rs도 설계 참조; ik에서 옮긴 코드는 저작권 표기), 공정한 비교(ik는 가장 빠른 플래그 + DSpark로 같은 창, 재현 스크립트 동봉), 숫자가 나가기 전에 ikawrakow에게 두세 줄. PR 본문에는 bloomery 이야기를 섞지 않는다.
- **M4 모델 순서(조사 `models`, `research/models-survey.md`)**: Qwen3-30B-A3B 전 카드(됨) → Qwen3 dense → GLM-4.7-Flash(`deepseek2` 파일, V2-Lite 커널; `scratch.rs` `n_used != 6` 거부부터) → V4-Flash(`v4port` 결정: `arch/deepseek41` 안의 변형) → GLM-5.3-Flash(`glm5next`, KDA 선형 34층 + MLA 11층 + mHC — 새 커널 계열 L). seam 프로그램(`seamc` `aff69e7` 등, 이동 클래스 리팩터)은 끝났다.
- **비교 대상(사용자 09-24, local-ai-registry PR #83)**: Qwen3.6-35B-A3B EXL3 3.0 bpw + MTP로 3090에서 252/352 tok/s. 우리 첫 목표는 35B-A3B Q3_K급 파일, 드래프트 없이 3090에서 252 초과(`research/linear-attn.md` §6).

## 타겟 프로필 (사용자 09-24 "3090 하나 혹은 두 개에 스레드리퍼 AVX2로 V4.1 Flash를 돌려볼 사람")

| 항 | T1 기본 | T2 확장 | 우리 측정(참고) |
|---|---|---|---|
| GPU | RTX 3090 24 GB ×1 — 배치 = `--place gate`(dense + expert 14.9 GB = 888슬롯; 슬롯 = 상주 routed expert 하나 16,773,120 B) | 3090 ×2(두 카드 합류 경로 필요 — 걸림돌 여섯, 트리아지) | A6000 48 GB plan (a) 2,668슬롯(공개 파일) |
| CPU | AVX2, 8채널 DDR4(5975WX STREAM 147.7 GB/s) — 4채널이면 호스트 다리 ×2 | 같음 | 같음 |
| RAM | 256 GB 하한(호스트 expert 집합 196–244 GB) | 같음 | 264 GB |
| 저장 | NVMe(engram 테이블 mmap) | 같음 | 같음 |

T1의 실측은 E21(24G 흉내 35.8 tok/s, 예측 33–38). T2는 DSpark 드래프트의 3090 상주(8.65 GB)와 자리를 다툰다 — 드래프트가 먼저.

## 실험 큐 (열린 것만; 답한 것은 장부·rig-log)

| # | 실험 | 무엇을 정하나 | 비용 | 막는 것 |
|---|---|---|---|---|
| E21 | **3090 단일 배치 tok/s**(공개 파일, `--place gate` + 뜨거운 목록, 깊이 6/4096, prose·code 512, n 96, 2바퀴; ik·ik + DSpark·llama.cpp·N5 플래그 세트) | 공개 글 헤드라인 = 타겟 카드의 값 | 임대 ~30분 | 빌더 틈 |
| E29 | DSpark 수락률의 온도(`--temp 0/0.7/1.0`) | 드래프트 이득이 실사용 샘플링에서도 서는가 | ~15분 | `ik-draft.sh` 온도 인자 |
| E5b → E19 | 우리 greedy 출력에 `draft-accept.py`; lookup 게이트 정책 넷 채점 | 오라클 − 항상쌍 < 2 %면 게이트 버림 | 오프라인 | — |
| E17 → E18 | 뜨거운 목록의 프롬프트 의존(코퍼스 × 목록 적중 행렬 + 접두 N 열) | 정적 목록 하나로 충분한지(격차 > 10 %p면 E18 프롬프트 적응 배치) | ~10분 | — |
| E7 · E9 · E14 · E10 · E27 · E28 재측정 | 3090 250 W 실효 BW · engram 콜드 팔 · 0층 브리지 2.1–2.7 ms · 층 l+1 라우터 적중률 · 3090 DSpark 자리 · qwen3route 뒤 | 각각 dspark D 유도·C3·스텝 최대 노출 항·P1 여부·T1 DSpark 배치·두 번째 헤드라인 | 소 | — |

## 측정 프로토콜 치트시트

```
just depth-gpu-ds41 6 6@K=V bin:<path>:6 lcpp:6   # 같은 임대 팔 회전, 증인·비율 표
just time-gpu-ds41 --tokens "<ids>" -n 96         # prose/code 헤드라인 프롬프트
just ab-decode bloomery-<track>                    # CPU 디코드 같은 임대 A/B (디스패치 경로 라운드의 완료 조건)
just time-cpu-v41-host --arms …                    # 호스트 티어 팔 벤치
just measure-qdot-rate                             # 커널률(단일 스레드, ik 대조)
just gate-alloc                                    # 스텝 할당 래칫
```
비교는 같은 임대 안에서만. 기록 순서: bloomery 커밋 → rig-log 기록 → 트래커 코멘트 → 메모리.

## 규칙

- 시간 숫자를 재는 카드, 조용한 기계, 게이트 완화 금지, 종료 코드, 언어, 병렬 트랙은 `AGENTS.md`가 정본이다.
- 첨부 프로토콜(`/props` + 청크별 `timings`)은 1단계부터 — toktape가 어느 단계든 녹화할 수 있어야 한다.
- 이 엔진의 가치는 배치·스케줄러·커널이다. GGUF 파서·토크나이저는 우리 crate(gate-tokenizer가 `llama-tokenize`와 비트 동일).
- 머지 순서 규율: 재핀 없는 트랙 먼저, 산술 순서를 바꾸는 트랙 마지막.

## 옮긴 절

- [`facts.md`](facts.md): 모델 파일·기계·툴체인·설계 규칙·법칙과 교훈.
- [`plan-ledger.md`](plan-ledger.md): 디딤돌·단계·로드맵·모델 되짚기·의존 그래프·변경 클래스 원문·파동 이력 10–20·「지금」 이력·GPU 선 트리아지 원문·**plan.md·plan-triage.md의 2026-09-25 이전 판**.
- [`plan-triage.md`](plan-triage.md): 열린 라운드 카드·시팅 큐·사용자 결정·받을 라운드별 열린 항목.
