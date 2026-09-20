# HANDOFF — 2026-09-20 세션 종료 시점 인계

새 세션/새 환경에서 이 파일을 먼저 읽는다. 규칙의 정본은 [AGENTS.md](AGENTS.md), 작업 대기열의 정본은 이 파일과 [docs/plan.md](docs/plan.md), 측정 기록의 정본은 [rig-log](https://github.com/midagedev/rig-log)의 log/ + README 로그 표다.

## 1. 무엇을 하는 프로젝트인가

**bloomery** — Rust로 짜는 CPU LLM 추론 엔진. 목표는 같은 기계·같은 ISA에서 ik_llama.cpp를 이기는 것(사용자 지시: "커널을 단순 이식하기보다 더 낫게"). 현재 모델 DeepSeek-V2-Lite-Chat Q3_K_M, 최종 목표 DeepSeek V4.1-Flash. 하드웨어: 박스의 ThreadRipper Pro 5975WX(Zen 3, 32C, AVX2+FMA+F16C+BMI2+VAES+VPCLMULQDQ까지 — **GFNI/AVX-512/VNNI/AMX 없음**), STREAM triad 147.7 GB/s.

**스코어보드(2026-09-20, 같은 임대 측정)**: N=96 디코드 **39.71 tok/s**(ik 83.27 → 잔여 **2.10배**), N=8 37.02(ik 82.43 → 2.23배), 프리필 63.65, 스프레드 6.8%. 세션 누적 4.16 → 39.71(9.55배). 플랫폼 이론 천장 ≈ 113 tok/s(1.31 GB/step ÷ 147.7 GB/s).

## 2. 어디에 무엇이 있는가

| 위치 | 내용 |
|---|---|
| `~/repo/bloomery` (이 리포) | 엔진 본체. AGENTS.md = 규칙 정본. docs/research/ = 조사 문서, docs/RESULTS-*.md = 라운드 산출, docs/plan.md = 단계 계획 |
| `~/repo/rig-log` | 측정 기록(한국어 산문). 임대·증인 있는 숫자만. README 로그 표가 전체 인덱스. 최근: 2026-09-20-a…-h |
| gadak 트래커 | `GADAK_HOME=$HOME/.gadak gadak --workspace gdk`, 프로젝트 MUL. Done: MUL-1…38 전부 종결(코멘트에 판정). 미완: MUL-30(GPU, cuda-oxide 확정됨·착수 대기), MUL-6(CUDA 13.3 툴킷) |
| 박스 | 접근은 오직 `./tools/box.sh '<cmd>'`(rsync 단방향 → 박스 편집 금지). 원격: /root/repo/bloomery, 데이터: /root/bloomery-data, 임대 락: /root/bloomery-cpu.lock. GPU: 3090(idx 0, 개발용)만 — **A6000(idx 1) 금지** |
| 세션 메모리 | `~/.zcode/cli/memories/projects/rig-log-*/memory/` — 이 워크스페이스 세션에만 유효. 새 곳에서는 이 파일이 대체 |

**새 환경 체크리스트**: ① 두 리포 클론(bloomery, rig-log — 둘 다 GitHub에 푸시돼 있음) ② 박스 SSH 설정 + `tools/box.sh` 동작 확인 ③ gadak 설치/설정(또는 이슈 상태는 본 파일 §6으로 대체 가능) ④ 맥에서 게이트 금지(arm64) — 모든 게이트는 box.sh 경유.

## 3. 규칙 다이제스트 (전체는 AGENTS.md)

- 게이트/빌드는 박스에서만(`just` 레시피 = box.sh 래핑, ~~전 게이트 `timeout --kill-after=10 900` 감싸짐~~ 전 게이트가 `tools/gate.sh`를 거친다: 900초 상한 + cargo의 종료 코드 그대로 — 걸리는 게이트는 빨간 게이트). 정정(같은 날 저녁 리뷰): ef9e579의 첫 형태는 `… || echo`라 빨간 게이트도 0으로 끝났다. e3bb924가 닫았고, 13게이트 재실행에서 가려진 빨강은 없었다. 게이트 판정은 출력 문구가 아니라 종료 코드로 읽는다.
- 벤치마크 숫자는 임대(기계 전역 flock)+증인을 러너가 소유한 채 나온 것만. 손 측정 금지. 프로파일 표도 측정이다(옆에서 빌드 하나만 돌아도 비율이 흔들림).
- 게이트 완화 금지 — 재핀은 A/B 입증 + 날짜 주석과 함께만. 측정 안 된 수를 기록에 쓰지 않는다. 유도/추정은 명시.
- 한국어: 산문·커밋. 영어: 코드 주석. 커밋·푸시는 라운드 종결 시.
- 병렬 트랙: git worktree + box.sh 원격 디렉터리 자동 유도. 시작·끝 `just box-gc`. 에이전트 프롬프트는 자립적(AGENTS.md 먼저 읽기, push/main/임대 금지, 설명 없는 게이트 실패 시 정지 보고). 임대 창은 메인 단독(페이즈 분리: 0=메인 임대 덤프 → 1=병렬 트랙(박스 CPU 사용자 1개 + Mac전용/읽기전용) → 2=메인 머지·측정·기록).
- **머지 순서 규율**: 재핀 없는 트랙(비트 불변 주장)을 먼저 머지, 산술 순서를 바꾸는 트랙(재핀 발생)을 마지막에 — 재핀 귀속이 흐려지지 않게.

## 4. 현재 엔진 상태 (main = 이 문서 커밋 시점)

- **커널**: Q3_K×q8_K, Q4_K/Q5_0/Q5_1/Q6_K×q8_2_x4 전부 융합(crates/qdot). MUL-38 q_nope2 Q8_0×Q8_0(vpsignb 부호접기 maddubs, 비트 동일). MUL-36 flash SIMD(kq_dot 8레인 합순서 + V j축; 폴백 레버 `BLOOMERY_FLASH_SIMD=0`). MUL-37 직렬 activation quant 풀 이양 + moe gate/up 이중양자화 제거(ptr::eq). 커널률은 ik의 95–99%(Q3_K만 ik 대조 미측정).
- **디스패치 경로가 느리면 첫 질문은 스레드 스윕이다**(`SWEEP="8 16 24 32" just measure-decode`): 곡선이 평평하면 일이 아니라 기다림. 프로파일이 꺼진 청크는 락을 잡지 않는다(AGENTS.md).
- **게이트**: 14개 just 레시피(gate-ops/attn/ffn/moe/head/forward/kv/derived/mt/profile/threads/qdot/prompts/1-1). 머지 후에는 영향 게이트 + mt/forward/prompts 재실행이 관례.
- **핀 상태**: prompts KNOWN_DIVERGENCE = **{24}**(MUL-36 재핀, A/B 입증됨 — 이전 {14}). forward L_OUT_BANDS = (0,3e-3)(3,2e-3)(24,7e-2) + 날짜 주석. gate-mt는 재핀 불허 항목(스레드 무관 비트 동일).
- **스코어보드(2026-09-20 저녁, 같은 임대)**: N=96 **61.48 tok/s** 대 ik 82.88 → **잔여 1.35배**(아침 39.71/2.10배). 경위는 rig-log -j: 청크 끝의 무조건 수집 Mutex 제거(+44%) → 메인 스레드 고정 → mmap 프리폴트(+3.5%; 프리필 +18%는 6토큰 프롬프트 기준). → rintf libm 호출 제거(+8.3%, 2b2fdb6).
- **새벽(rig-log 09-21-a), 스코어보드**: 같은 임대 **N=96 76.97 tok/s(13.0 ms) 대 ik tg96 82.74 → 잔여 1.075배**, 프리필 116.7(6토큰). -k 뒤에 넣은 것: AVX2 활성값 양자화기(+0.9%), 풀의 notify 생략·워커별 완료 표식(풀 벤치 4.56 → 1.61 µs, 디코드 무차이), 행 디스패치의 꼬리 훔치기(`BLOOMERY_STEAL=0` 레버; 프리필 +25%, 디코드 무차이), 프로파일 열 `span ms`·`slowest ms`와 청크별 분포. 산수(레벨1, 스텝당): 평균 청크 9.3 ms(ik의 커널 시간 9.1과 같은 바닥) + 스큐 1.06 + 장벽 0.58 + 나머지 3.0. 남은 0.9 ms는 얇다 — 다음은 이종 묶음 디스패치(`docs/cpu-dispatch-plan.md` 3단계)와 matmul 앞뒤 1.09 ms.
- **밤(rig-log -k), 스코어보드**: 같은 임대 **N=96 72.30 tok/s(13.8 ms) 대 ik tg32 81.79 → 잔여 1.13배**, 프리필 97.83(6토큰). 사슬(A/B 상대값): 61.5 → 64.8(SwiGLU 8레인 `qdot::swiglu`, 발산 집합 {24} → {}) → 67.8(`.cargo/config.toml` `target-cpu=znver3`) → 72.5(스텝 플랜: 토큰과 무관한 조회 전부를 `Derived::new`로 — `Derived::plan()`, `*_with` 스텝 경로, 옛 시그니처는 래퍼). 산수: ik와 커널 스레드시간은 같은 선이고 차이는 워커 이용률, 즉 메인 스레드의 직렬 구간이다(`docs/cpu-dispatch-plan.md`). 래칫 게이트 `just gate-alloc`(14454 → 1291, LIMIT 1400). 다음: matmul 앞뒤 비용·배치 부기, 그 뒤 이종 묶음 디스패치. 기각: 내용 기반 입력 중복 제거, fat LTO.
- **남은 산수**: 레벨2 dot 합/32 = 8.5 ms/step(완전 병렬 내적), ik 스텝 전체 12.1 ms, 우리 16.3 ms → 비내적 7.8 ms를 3.6 아래로. perf상 임계 경로는 메인 스레드 하나(워커는 표본의 51%를 스핀으로 대기): rintf 10% · memset 6.6% · expf 3.2% · 자기 청크+장벽 18%.
- ~~**스테이지 표(N=96, 레벨1, 24.18 ms/step)**~~ (락 아래서 잰 표 — 새 표는 rig-log -j): batch Q3_K 24.1% · Q3_K 단일 18.4% · batch Q5_0 14.9% · Q4_K 13.5% · Q6_K(lm_head) 5.3% · q_nope2 5.1% · swiglu+F32+접착부 ~10% · **flash 0.40ms(1.6%)**. 전체 54.8 GB/s = STREAM의 37%.
- 주의: 레벨2 quant 열은 MUL-37 이후 워커 CPU합(벽시간 아님).

## 5. 축적된 지식 — 법칙과 교훈 (근거는 rig-log -g와 lib.rs 주석)

- **디스패치당 바이트 법칙**(MUL-35): 사이트 달성 GB/s는 디스패치당 바이트의 단조 포화 함수(172MB→119, 16.8→75, 2–10MB→47–51, 0.5–1.1MB→13–23 GB/s). 커널 MT 상한 122.6–136.3 GB/s(qdot-rate-mt), 풀 디스패치 세금 4.65µs×322회=1.5ms/step(pool-rate) — 커널·풀 무죄, 범인은 얇은 패킹. 활용도 낮은 사이트(예: q_nope2 25%)에서 커널 가속은 ×활용도만 벽시간에 나온다(MUL-38 실증).
- **`#[target_feature]` 교훈**(MUL-26/27): 누락되면 에러 없이 수십 배 느려짐(0.6 GB/s 사례). 헬퍼 분리 자체도 10–13% 손해 — 단일 함수 선호. 분리 시 헬퍼에도 속성.
- **게이트 3중 패턴**(MUL-27+): 인코더/커널 vs 에뮬레이터(인트린식 그래프 흉내 — 손 레인 유도는 틀림), vs ik 자체 커널(하네스에서 bx=행 스트라이드). 정수 경로는 결합법칙으로 비트 동일 — 에뮬레이터 불필요(MUL-38).
- **재핀 규율**: 집합 변화는 되돌림 레버 한 실행으로 A/B 입증(flash는 BLOOMERY_FLASH_SIMD=0).
- **리뷰 판정(2026-09-20 저녁)**: qdot의 `_mm*` 헬퍼(`hsum_i32`·`field_dot`·`q5x_codes`·`hsum_float_8`)는 속성 없이 `#[inline(always)]`로 호출자의 기능을 물려받는 형태이고, 릴리스 바이너리에 독립 심볼이 없음을 `nm`으로 확인했다 — 위 교훈의 "헬퍼에도 속성"은 인라인이 보장되지 않는 헬퍼에 한한다. 전역 RUSTFLAGS가 없으므로 속성은 전부 하중을 받는다(AGENTS.md Known state).
- **사고 보강**(ef9e579): 게이트 타임아웃·box-gc·트랙 체크리스트. "조용한 에이전트는 상태가 아니라 증상" — 박스 `pgrep -fa '<원격 경로>'` 부터.
- **하드웨어 판단 기록**: 3995WX(Zen2 64C) 교체는 무이득~역행(같은 DDR4 평면, 코어당 0.65배). 플랫폼을 바꾼다면 대역폭(8채널 DDR5).

## 6. 다음 라운드: 패킹 (조사 종합표 1순위 — docs/research/cpu-llm-ideas.md)

트래커가 정본이다: **MUL-39**(패킹, 설계 라운드부터) · MUL-40(gate·up 결합 GEMM) · MUL-41(전문가 블록 배치 GEMM) · MUL-42(워커 수 재스윕). 각 이슈에 지금까지 잰 값과 "무엇을 재면 끝나는가"가 있다. 아래는 그 요약.

**전제 변경(2026-09-20 저녁)**: 아래의 '디스패치당 바이트' 근거는 수집 Mutex가 있던 엔진에서 잰 것이다. 패킹의 새 근거는 디스패치 횟수(401/step)와 디스패치당 앞뒤 비용 11–60 µs, 그리고 메인 스레드의 직렬 구간이다. 순서: ① swiglu를 병렬 디스패치 안으로(MUL-40 에필로그) ② quant를 행 디스패치에 접기·shexp를 라우팅 배치에·같은 입력 q_a/kv_a 한 디스패치·스텝 내 할당 제거. 기각된 것: 폭 제한 디스패치, 동적 행 분배, malloc trim, 거버너, n-gram 추측(MUL-43).

~~ik 잔여 2.10배의 마지막 기제 = 스케줄링 구조~~(ggml식 텐서당 1노드·행-방향 연속 청크·노드별 장벽만으로 ik는 STREAM의 73%를 뽑음). 구체 후보(우선순위):
1. **gate·up 결합 GEMM + SiLU·mul 에필로그**(A×[B1,B2] 단일 GEMM) — swiglu 직렬 0.87ms + 디스패치 52→26 + batch Q3_K quant 잔여를 한 방에(Intel 실측 +12% 계열).
2. **MoE topk_ids 정렬 → 전문가별 블록 배치 GEMM**(게더의 GEMM 흡수, ZenDNN group_matmul 계열).
3. **텐서-1노드 연속 청크/디스패치 굵히기** — batch Q3_K 24.1%가 첫 표적. 잔여 상한 ~10ms/step(레벨2 유도).
설계 라운드로 시작할 것(청크 밸런스·같은-k 사이트 결합 — Q4_K의 53호출은 k 혼재 2,048/2,816). 게이트: 비트 불변이면 재핀 0(quant·패킹 이동), 산술 변화면 재핀+A/B.
**공짜 실험**: 워커 수 스윕(BLOOMERY_THREADS 8/16/24/32 — ZenDNN이 "128코어 단일 인스턴스 < 2×64 인스턴스"임을 공식 인정한 것과 같은 현상, 문헌상 20–30% 차이 사례).

## 7. 그 이후 대기열

이슈: MUL-43(스펙 디코딩 타당성 — n-gram 수용률 오프라인 계수가 먼저) · MUL-44(KV q8_0, 조건부) · MUL-30(GPU) · MUL-46(정리 라운드 — Done. 뼈대 분할이 들여온 swiglu 회귀는 d9626c4에서 닫음; 미결 잔여는 MUL-39에서 재측정: 디코드 swiglu 42.6 대 28.3 ms/32스텝, moe_trace·wv_b_heads·moe_expert_io +13 ms, 워커 파킹 64→590) · MUL-45(게이트 종료 코드 사고, 기록).

- **스펙 디코딩**(PARD식 k토큰 검증, 우리 추정 1.5–2.5×) — lm_head 대역폭 벽+고정비를 통째로 상각. 초안 모델 필요(후보: ~~자기 자신 Q3_K or~~ n-gram — 자기 자신은 드래프트 비용이 본체와 같아 성립하지 않는다; 수용률부터 센다, MUL-43).
- **GPU 단계**(MUL-30, cuda-oxide 확정): 박스 완비(nvcc 13.0·cargo-oxide 0.2.1·핀=업스트림 HEAD), Q4_K·Q6_K CUDA 커널 이미 존재(q4k_gemv 809.8 GB/s, MUL-9). ~~첫 일 = 엔진 배선 + 3090 ik 대비 tok/s.~~ 개정(2026-09-20 저녁): 첫 목표는 V2-Lite **전체**를 3090에 올려 ik CUDA와 대조(하이브리드는 그다음). 배선 전에 알아야 할 것 둘 — 스테이지 0 커널은 K=2048 전용이라 K=1408·2816·10944와 Q5_0·Q5_1·비-matmul 연산이 전부 남았고, GPU 커널은 활성값 q8_1(오차 바닥 3–5e-3)이라 CPU 오라클이 아닌 **ik `-ngl 99` CUDA 오라클**로 게이트해야 한다. 진행 중: 라이브러리 패키징 스파이크 + 연산·형상 인벤토리. cutile-rs는 MUL-6(13.3 툴킷) 뒤 별도 스파이크.
- 구조 부채(패킹 라운드 전에 볼 것): `ops.rs`의 `matmul_q_multi` 397줄, `moe.rs`의 `moe_ffn` 326줄 — 패킹은 바로 이 두 함수에 얹힌다. 양자화 사전 패스·PairWork 조립·행 워커를 나눠 두면 설계 라운드의 diff가 읽힌다. 비트 불변 리팩터이므로 mt/forward/prompts가 판정한다.
- 디코드 경로의 호출당 할당(리뷰 발견, 미측정): `wv_b_heads`가 스텝마다 헤드별 `Tensor2`·`TensorInfo`·`format!` 이름을 다시 짓고(호출당 ~35회), `MlaParams::read`가 스텝당 27번 메타데이터를 다시 읽는다. 레벨2 표에서 접착부 전체가 0.8ms/step(3%대)이라 상한은 작다 — 로드 시점으로 올리는 일은 패킹 라운드에 끼워서.
- Q5_0/Q5_1 커널 쌍과 flash 스칼라/AVX2 쌍은 의도된 쌍둥이다(헬퍼 분리 10–13% 손해 실측). 합치지 말고 TWIN 주석을 따라 양쪽을 같이 고친다. `tools/ref/*_ref.cpp`·`*_rate.cpp` 다섯 쌍의 복제는 하네스라 우선순위 낮음.
- 소형: KV q8_0(flash 이후 ctx 기울기), kq 부분합 4→8, 발산 {24} 원인(마진 0.151의 근타이 — 우선순위 낮음), THP 1회 A/B 노벨.
- ik 발전 감시: ik가 달라지면 오라클/참조 재생성(just build-ref-dump / argmax-ref).

## 8. 측정 프로토콜 치트시트

```
just measure-decode                    # N=8 창 + 같은 임대 ik (헤드라인)
just gate-alloc                        # 정상 상태 스텝의 할당자 호출 수 (내려가기만 하는 래칫)
just ab-decode bloomery-<track>        # 같은 임대 A/B — 디스패치 경로를 만진 라운드의 완료 조건 (절대값은 창마다 ~5% 움직인다)
./tools/box.sh 'BLOOMERY_DECODE_N=96 bash tools/ref/decode-measure.sh'   # N=96 창
./tools/box.sh 'BLOOMERY_DECODE_N=96 bash tools/ref/profile-measure.sh' # 스테이지 표 L1+L2
cargo build --release -p bloomery-qdot --bin qdot-rate-mt && .../qdot-rate-mt   # 커널 MT 상한(참고)
```
비교는 같은 임대 안에서만. 스텝 표의 첫 구간 편향(MUL-28) 주의. 기록 순서: bloomery 커밋 → rig-log 기록 + README 행 → gadak 코멘트+Done → 메모리.

## 9. 최근 세션 요약 (상세는 rig-log README 표)

- 2026-09-20 세션(이 날 전부): 4.16 → 39.71 tok/s. MUL-23~29(스레딩·KV·융합·Q4_K·flash 병렬) → MUL-31/32(Q6_K·Q5_0 병렬) → MUL-33/34(q_nope2 병렬·Q5_1) → MUL-35(포화도 진단) → MUL-36/37/38(flash SIMD·quant 풀·q_nope2 커널). 사고 1건(q_nope2 무한루크, 게이트 매달림)은 3중 보강으로 폐쇄(ef9e579).
- 조사 3트랙(Intel·AMD·광역) + GPU 정찰 완료 — docs/research/ 4편 + RESULTS 2편.
